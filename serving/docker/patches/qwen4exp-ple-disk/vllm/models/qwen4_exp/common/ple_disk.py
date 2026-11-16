# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp PLE embedding served from the checkpoint files (``VLLM_PLE_DISK=1``).

The table weights stay on disk: the layer registers zero-row placeholders, the
weight loader drops the checkpoint shards, and every lookup reads the rows it
needs through ``np.memmap`` views of the safetensors files (page-cache backed).
Only the small tensors (``weight_scale_2``, hash buffers) load normally.

A lookup syncs the ids to the host, gathers the unique rows into a persistent
pinned staging buffer (``ple_disk_table``), copies them to the GPU in one
async H2D copy and decodes them with the 0.33 lookup kernels, so the output
equals the device and pinned-host paths. The host work cannot be captured in
a CUDA graph: the lookup is the custom op ``vllm::qwen4_exp_ple_disk_lookup``
(opaque to torch.compile) and an eager break of the breakable CUDA graph, so
it needs ``cudagraph_mode=PIECEWISE`` (or ``--enforce-eager``), not FULL.

Prefetch (default; ``VLLM_PLE_DISK_PREFETCH=0`` restores the synchronous
lookup above): the model calls ``start_prefetch`` before the decoder layer
that precedes the PLE layer (0.33 ``nvidia/model.py`` ``_start_layer_ple_prefetch``).
It enqueues a non-blocking D2H copy of the ids and an event, and hands the
host staging to a background thread, so the main thread goes on launching
that layer. ``finalize_prefetch`` in the PLE layer waits for the staging,
then enqueues the H2D copy and the decode kernel. Both are eager breaks.

Early row fetch (``VLLM_PLE_DISK_EARLY=1``, needs prefetch and MTP): the n-gram
ids of each token of the next verify batch are hashed on the GPU as soon as
that token exists (the bonus token after sampling, each draft after its draft
pass; the runner and speculator hooks are in the ``qwen4exp-spec-step``
overlay), copied to the host, and their rows read on the prefetch thread
(``PLEEarlyFetch``, ``lookup.warm``). The batch's own staging then finds them
in the page cache. Which rows the batch stages and how is unchanged.

Slot-pool cache (``VLLM_PLE_DISK_CACHE_GB``, ``ple_disk_cache.py``): the GPU
keeps a CLOCK-managed pool of rows in their stored form. The host stages only
the missed rows (read with parallel ``pread``) plus a per-id int32 slot map;
the GPU scatters the missed rows into their slots and decodes through the slot
map with the same 0.33 kernel. Same staging and prefetch flow as above.
"""

import os
import time
from concurrent.futures import Future

import torch

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    eager_break_during_capture,
)
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.triton_utils import triton
from vllm.utils.torch_utils import direct_register_custom_op

from .ngram_embedding import (
    Qwen4ExpPLEEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLENvFp4EmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
    _lookup_nvfp4_ple_embedding_kernel,
    _lookup_ple_embedding_from_pinned_kernel,
)
from .ple_disk_table import (
    STATS,
    STATS_ZERO,
    env_int,
    open_ple_disk_lookup,
    ple_cache_requested,
)

logger = init_logger(__name__)

_LAYERS: dict[str, "Qwen4ExpPLEDiskEmbedding"] = {}
_STATS_SEC = env_int("VLLM_PLE_DISK_STATS_SEC", 30)
_PREFETCH = env_int("VLLM_PLE_DISK_PREFETCH", 1) != 0
_EARLY = env_int("VLLM_PLE_DISK_EARLY", 0) != 0
_stats_last = [0.0]
# Early row fetch of each PLE layer (VLLM_PLE_DISK_EARLY); see PLEEarlyFetch.
_EARLY_FETCHES: list["PLEEarlyFetch"] = []


class Qwen4ExpPLEDiskEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table read from the checkpoint files through the page cache."""

    supports_prefetch = _PREFETCH

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
        reduce_results: bool = True,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
            reduce_results=reduce_results,
        )
        self.num_ngram_heads = num_ngram_heads
        self.is_nvfp4 = isinstance(embedding_method, Qwen4ExpPLENvFp4EmbeddingMethod)
        # Staging order: packed rows, then (NVFP4) their block scales.
        table_row_bytes = {"weight": self.weight.shape[1] * self.weight.element_size()}
        if self.is_nvfp4:
            table_row_bytes["weight_scale"] = self.weight_scale.shape[1]
        model_path = (
            os.environ.get("VLLM_PLE_DISK_DIR")
            or get_current_vllm_config().model_config.model
        )
        if not os.path.isdir(model_path):
            raise RuntimeError(
                f"PLE disk: {model_path!r} is not a local directory; point "
                "--model at the checkpoint directory or set VLLM_PLE_DISK_DIR"
            )
        lookup_args = dict(
            num_rows=self.org_vocab_size,
            max_rows=max_total_tokens * self.etp_data_parallel_size * num_ngram_heads,
            vocab_start=self.shard_indices.org_vocab_start_index,
            vocab_end=self.shard_indices.org_vocab_end_index,
        )
        cache_slots = 0
        if ple_cache_requested():
            # Imported only with the cache switch set, so the cache-off setup
            # needs no extra mount.
            from .ple_disk_cache import open_ple_cached_lookup, ple_cache_slots

            cache_slots = ple_cache_slots(sum(table_row_bytes.values()))
        # GPU slot pools, one per table (packed rows, block scales), slot 0 zero.
        self._pools: list[torch.Tensor] | None = None
        if cache_slots:
            if self.tp_size != 1:
                raise RuntimeError(
                    "VLLM_PLE_DISK_CACHE_GB supports one PLE table rank "
                    "(ETP=TP=1) only"
                )
            self.lookup = open_ple_cached_lookup(
                model_path,
                extract_layer_index(prefix),
                table_row_bytes,
                num_slots=cache_slots,
                **lookup_args,
            )
            # Allocated here, inside load_model, so vLLM counts the pool as
            # model memory before it sizes the KV cache.
            device = torch.device("cuda", torch.cuda.current_device())
            self._pools = []
            for row_bytes in table_row_bytes.values():
                pool = torch.empty(
                    cache_slots, row_bytes, dtype=torch.uint8, device=device
                )
                pool[0].zero_()
                self._pools.append(pool)
        else:
            self.lookup = open_ple_disk_lookup(
                model_path, extract_layer_index(prefix), table_row_bytes, **lookup_args
            )
        self._staging: torch.Tensor | None = None
        self._device_staging: torch.Tensor | None = None
        # Prefetch state: pinned ids, the event after their D2H copy, the
        # running staging job and its row count.
        self._prefetch_ids: torch.Tensor | None = None
        self._prefetch_event: torch.cuda.Event | None = None
        self._prefetch_stream: torch.cuda.Stream | None = None
        self._prefetch_job = None
        self._prefetch_rows = 0
        self._early: PLEEarlyFetch | None = None
        _LAYERS[prefix] = self
        logger.info(
            "PLE disk: %s reads %d rows x %d B (%.1f GiB) from %s, "
            "%d B pinned staging, %d gather workers, prefetch %s, %s",
            prefix,
            self.org_vocab_size,
            self.lookup.row_bytes,
            self.org_vocab_size * self.lookup.row_bytes / 2**30,
            model_path,
            self.lookup.staging_bytes,
            self.lookup.tables[0].pool._max_workers,
            "on" if self.supports_prefetch else "off",
            (
                f"slot-pool cache {cache_slots} slots "
                f"({cache_slots * self.lookup.row_bytes / 2**30:.2f} GiB GPU, "
                f"{self.org_vocab_size * 4 / 2**30:.2f} GiB host index), "
                f"{self.lookup.reader.readers} readers, "
                f"{'packed ' + self.lookup.packed_path if self.lookup.packed_path else 'checkpoint rows'}"
                if cache_slots
                else "no slot-pool cache"
            ),
        )

    def attach_early_fetch(self, ngram: torch.nn.Module) -> None:
        """Set up the early row fetch with ``VLLM_PLE_DISK_EARLY=1``.

        ``ngram`` is the owning ``Qwen4ExpNGramEmbedding``, whose hash the
        fetch reuses.
        """
        if not _EARLY:
            return
        speculative_config = get_current_vllm_config().speculative_config
        num_tokens = (
            speculative_config.num_speculative_tokens if speculative_config else 0
        )
        if not self.supports_prefetch or not num_tokens:
            logger.warning(
                "PLE disk: VLLM_PLE_DISK_EARLY needs the prefetched lookup and "
                "speculative decoding; early row fetch off"
            )
            return
        self._early = PLEEarlyFetch(
            ngram,
            self.lookup,
            num_tokens,
            torch.device("cuda", torch.cuda.current_device()),
        )
        _EARLY_FETCHES.append(self._early)
        logger.info(
            "PLE disk: early row fetch on for %s: the bonus token and drafts "
            "1..%d of each verify batch",
            self.prefix,
            num_tokens - 1,
        )

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate a zero-row placeholder; the rows stay in the checkpoint."""
        del num_embeddings
        return torch.empty(0, embedding_dim, dtype=dtype, device="cpu")

    def weight_loader(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        checkpoint_start: int | None = None,
    ) -> None:
        if checkpoint_start is not None:
            # Table shards are read from disk at lookup time.
            return
        super().weight_loader(param, loaded_weight)

    # ETP reduction is identical to the pinned-host table's.
    _reduce_etp_embeddings = Qwen4ExpPLEPinnedHostEmbedding._reduce_etp_embeddings

    def _allocate_staging(self, device: torch.device) -> None:
        # First use is the eager profile run, before any graph capture.
        self._staging = torch.empty(
            self.lookup.staging_bytes, dtype=torch.uint8, pin_memory=True
        )
        self._device_staging = torch.empty(
            self.lookup.staging_bytes, dtype=torch.uint8, device=device
        )
        if self.supports_prefetch:
            self._prefetch_ids = torch.empty(
                self.lookup.max_rows, dtype=torch.int64, pin_memory=True
            )
            self._prefetch_event = torch.cuda.Event()
            # The id copy and its event live on a side stream: the prefetch
            # thread waits on that event while the main stream is already
            # capturing the next graph segment, and synchronizing on work of a
            # capturing stream invalidates the capture.
            self._prefetch_stream = torch.cuda.Stream(device=device)

    @eager_break_during_capture
    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Copy the ids to the host and stage their rows on the prefetch thread."""
        start_time = time.perf_counter()
        _refuse_capture()
        if self._early is not None:
            self._early.count = -1  # the batch the early fetch was for has begun
        if self._staging is None:
            self._allocate_staging(ngram_ids.device)
        ids_host = self._prefetch_ids
        event = self._prefetch_event
        assert ids_host is not None and event is not None and self._staging is not None
        slot_size, _ = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        num_ids = gathered_ids.numel()
        # The previous lookup's finalize has joined its job and enqueued its
        # H2D copy before this copy and event, so both host buffers are free
        # once the event fires.
        if BreakableCUDAGraphCapture.is_active():
            # While a graph is being captured, stage on this thread: a host
            # wait from the prefetch thread during the next segment's capture
            # invalidates it. Replays and eager steps take the async path.
            job: Future = Future()
            job.set_result(
                self.lookup.stage(
                    gathered_ids.reshape(-1).cpu().numpy(), self._staging.numpy()
                )
            )
            self._prefetch_rows = gathered_ids.shape[0]
            self._prefetch_job = job
            STATS["op_ms"] += (time.perf_counter() - start_time) * 1e3
            return
        prefetch_stream = self._prefetch_stream
        prefetch_stream.wait_stream(torch.cuda.current_stream())
        gathered_ids.record_stream(prefetch_stream)
        with torch.cuda.stream(prefetch_stream):
            ids_host[:num_ids].copy_(gathered_ids.reshape(-1), non_blocking=True)
            event.record(prefetch_stream)
        self._prefetch_rows = gathered_ids.shape[0]
        self._prefetch_job = self.lookup.stage_async(
            ids_host[:num_ids].numpy(), self._staging.numpy(), event.synchronize
        )
        STATS["op_ms"] += (time.perf_counter() - start_time) * 1e3

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        """Decode the prefetched rows for this DP rank's ``total_tokens``."""
        assert self._device_staging is not None
        # Allocated in the graph segment so the eager finalize writes a static
        # buffer that later segments read.
        output = torch.empty(
            (total_tokens, self.num_ngram_heads * self.embedding_dim),
            dtype=self.embedding_method.lookup_dtype(self),
            device=self._device_staging.device,
        )
        self._finalize_prefetch(output)
        return output

    @eager_break_during_capture
    def _finalize_prefetch(self, output: torch.Tensor) -> None:
        start_time = time.perf_counter()
        _refuse_capture()
        job = self._prefetch_job
        assert job is not None, "PLE disk finalize requires a prior start_prefetch"
        self._prefetch_job = None
        STATS["hit" if job.done() else "miss"] += 1
        staged = job.result()
        STATS["wait_ms"] += (time.perf_counter() - start_time) * 1e3
        rows = self._prefetch_rows
        if self.tp_size == 1:
            # No ETP reduction or DP selection: decode straight into output.
            self._upload_and_decode(
                staged, output.view(rows, self.num_ngram_heads, self.embedding_dim)
            )
        else:
            embeddings = output.new_empty(
                (rows, self.num_ngram_heads, self.embedding_dim)
            )
            self._upload_and_decode(staged, embeddings)
            _, slot_offset = self._get_dp_gather_slot(output.shape[0])
            if self.reduce_results:
                embeddings = self._reduce_etp_embeddings(embeddings)
            else:
                # The outer TP reduce-scatter requires ETP=TP.
                assert self.etp_data_parallel_size == 1
            embeddings = self._select_embeddings(
                embeddings, output.shape[0], slot_offset
            )
            output.copy_(embeddings.flatten(-2))
        STATS["calls"] += 1
        STATS["op_ms"] += (time.perf_counter() - start_time) * 1e3
        _log_stats()

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Gather ETP IDs, look up rows from disk, reduce, and select local rows."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        # Allocated in the graph segment so the eager lookup writes a static
        # buffer that later segments read.
        embeddings = torch.empty(
            (*gathered_ids.shape, self.embedding_dim),
            dtype=self.embedding_method.lookup_dtype(self),
            device=gathered_ids.device,
        )
        torch.ops.vllm.qwen4_exp_ple_disk_lookup(gathered_ids, embeddings, self.prefix)
        if self.reduce_results:
            embeddings = self._reduce_etp_embeddings(embeddings)
        else:
            # The outer TP reduce-scatter requires ETP=TP.
            assert self.etp_data_parallel_size == 1
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )

    def lookup_into(self, ngram_ids: torch.Tensor, output: torch.Tensor) -> None:
        """Read the rows for ``ngram_ids`` from disk and decode them to ``output``."""
        _refuse_capture()
        ids = ngram_ids.reshape(-1)
        if ids.numel() == 0:
            return
        if self._staging is None:
            self._allocate_staging(output.device)
        staging = self._staging
        assert staging is not None
        # The blocking D2H copy also waits for the previous lookup's H2D copy
        # on this stream, so the pinned staging buffer is free to reuse.
        start_time = time.perf_counter()
        ids_host = ids.cpu().numpy()
        STATS["id_sync_ms"] += (time.perf_counter() - start_time) * 1e3
        self._upload_and_decode(self.lookup.stage(ids_host, staging.numpy()), output)

    def _upload_and_decode(
        self, staged: tuple[int, int, list[int]], output: torch.Tensor
    ) -> None:
        """Copy staged rows to the GPU and decode them into ``output``."""
        if self._pools is not None:
            self._upload_and_decode_cached(staged, output)
            return
        num_ids, num_unique, (weight_offset, *scale_offsets) = staged
        if num_ids == 0:
            return
        staging = self._staging
        device_staging = self._device_staging
        assert staging is not None and device_staging is not None
        start_time = time.perf_counter()
        num_bytes = weight_offset + num_unique * self.lookup.row_bytes
        device_staging[:num_bytes].copy_(staging[:num_bytes], non_blocking=True)
        launch_time = time.perf_counter()
        STATS["h2d_ms"] += (launch_time - start_time) * 1e3
        inverse = device_staging[:weight_offset].view(torch.int64)
        weight_row_bytes = self.lookup.tables[0].row_bytes
        weight = device_staging[
            weight_offset : weight_offset + num_unique * weight_row_bytes
        ].view(num_unique, weight_row_bytes)
        if self.is_nvfp4:
            (scale_offset,) = scale_offsets
            scale_row_bytes = self.lookup.tables[1].row_bytes
            weight_scale = device_staging[
                scale_offset : scale_offset + num_unique * scale_row_bytes
            ].view(num_unique, scale_row_bytes)
            # Unique rows are local rows [0, num_unique); -1 marks other ranks'.
            _lookup_nvfp4_ple_embedding_kernel[(num_ids,)](
                weight,
                weight_scale,
                self.weight_scale_2,
                inverse,
                output,
                self.embedding_dim,
                0,
                num_unique,
                BLOCK_D=triton.next_power_of_2(self.embedding_dim),
            )
        else:
            _lookup_ple_embedding_from_pinned_kernel[(num_ids,)](
                weight,
                inverse,
                output,
                weight_row_bytes,
                0,
                num_unique,
                BLOCK_D=triton.next_power_of_2(weight_row_bytes),
            )
        STATS["launch_ms"] += (time.perf_counter() - launch_time) * 1e3

    def _upload_and_decode_cached(self, staged, output: torch.Tensor) -> None:
        """Upload the slot map and missed rows, fill their slots, decode.

        One H2D copy; ``index_copy_`` per table writes the missed rows into
        their slots; the 0.33 kernel then reads the pool through the slot map
        (slot 0 is the zero row). The step's pins are released once this is
        enqueued: a later step's scatter into an evicted slot runs after this
        decode on the same stream.
        """
        lookup = self.lookup
        pools = self._pools
        staging = self._staging
        device_staging = self._device_staging
        assert pools is not None
        assert staging is not None and device_staging is not None
        num_ids, num_misses, slot_offset, table_offsets, _ = staged
        try:
            if num_ids == 0:
                return
            start_time = time.perf_counter()
            last = lookup.tables[-1].row_bytes
            num_bytes = table_offsets[-1] + num_misses * last
            device_staging[:num_bytes].copy_(staging[:num_bytes], non_blocking=True)
            if num_misses:
                slots = device_staging[slot_offset : slot_offset + 8 * num_misses].view(
                    torch.int64
                )
                for pool, offset in zip(pools, table_offsets):
                    row_bytes = pool.shape[1]
                    pool.index_copy_(
                        0,
                        slots,
                        device_staging[offset : offset + num_misses * row_bytes].view(
                            num_misses, row_bytes
                        ),
                    )
            launch_time = time.perf_counter()
            STATS["h2d_ms"] += (launch_time - start_time) * 1e3
            slot_map = device_staging[: 4 * num_ids].view(torch.int32)
            num_slots = pools[0].shape[0]
            if self.is_nvfp4:
                _lookup_nvfp4_ple_embedding_kernel[(num_ids,)](
                    pools[0],
                    pools[1],
                    self.weight_scale_2,
                    slot_map,
                    output,
                    self.embedding_dim,
                    0,
                    num_slots,
                    BLOCK_D=triton.next_power_of_2(self.embedding_dim),
                )
            else:
                row_bytes = pools[0].shape[1]
                _lookup_ple_embedding_from_pinned_kernel[(num_ids,)](
                    pools[0],
                    slot_map,
                    output,
                    row_bytes,
                    0,
                    num_slots,
                    BLOCK_D=triton.next_power_of_2(row_bytes),
                )
            STATS["launch_ms"] += (time.perf_counter() - launch_time) * 1e3
        finally:
            lookup.release(staged)


class PLEEarlyFetch:
    """Early row fetch for the tokens of the next MTP verify batch.

    A verify batch is the bonus token ``t0`` (the last sampled token) and the
    drafts ``d1..dK``. A token's 16 n-gram ids depend only on itself and its
    ``ngram_size - 1`` predecessors, so they are final as soon as the token
    exists on the GPU: ``t0`` after sampling, ``d<j>`` after draft pass
    ``j - 1``. ``begin`` (runner, after sampling) and ``token`` (speculator,
    after each draft pass but the last) hash that token with the stock
    ``compute_ngram_ids`` into a persistent buffer, copy its ids to pinned
    memory on a side stream and record an event; ``lookup.warm_async`` reads
    the rows on the prefetch thread once the event fires. Only ``dK``'s rows
    are left for the batch's own staging to read from the disk.

    Only the host-side reads move earlier. The batch's ids, staging, upload
    and decode are the ones without this class, so the output is the same.
    Any difference in the guessed batch (another request, fewer drafts) only
    costs reads that nobody uses.

    No host sync: the main thread only enqueues small kernels, a D2H copy and
    an event. The buffers are reused one batch later; by then the batch's
    staging job, which the prefetch thread runs after these reads, has been
    joined in ``_finalize_prefetch``.
    """

    def __init__(
        self,
        ngram: torch.nn.Module,
        lookup,
        max_tokens: int,
        device: torch.device,
    ) -> None:
        self.ngram = ngram
        self.lookup = lookup
        self.max_tokens = max_tokens
        self.context_len = ngram.ngram_size - 1
        self.eos_token_id = ngram.eos_token_id
        # Predecessors of t0, then t0, d1, ..., d<max_tokens - 1>.
        self.tokens = torch.full(
            (self.context_len + max_tokens,),
            self.eos_token_id,
            dtype=torch.int32,
            device=device,
        )
        self.context_offsets = torch.arange(
            -self.context_len, 0, dtype=torch.int64, device=device
        )
        self.query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
        self.ids = torch.empty(
            max_tokens, ngram.ngram_heads, dtype=torch.int64, device=device
        )
        # Tokens fetched for the current batch; -1 = no batch armed.
        self.count = -1
        if device.type == "cuda":
            self.ids_host = torch.empty(
                max_tokens, ngram.ngram_heads, dtype=torch.int64, device="cpu",
                pin_memory=True,
            )  # explicit CPU: built at model init, under vLLM's default-device context
            self.events = [torch.cuda.Event() for _ in range(max_tokens)]
            # As for the prefetch ids: the prefetch thread waits on these
            # events, never on the main stream.
            self.stream = torch.cuda.Stream(device=device)

    def begin(
        self,
        all_token_ids: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        idx_mapping: torch.Tensor,
    ) -> None:
        """Arm for the next verify batch of request ``idx_mapping[0]`` and
        fetch t0's rows. Call after ``postprocess_sampled``.

        The context mirrors ``Qwen4ExpModelState._prepare_ngram_context`` of
        the next step, and t0 the first token that
        ``combine_sampled_and_draft_tokens`` writes.
        """
        context_len = self.context_len
        request = idx_mapping[:1].long()
        positions = (
            num_computed_tokens[request].long().unsqueeze(1) + self.context_offsets
        )
        context = all_token_ids[request.unsqueeze(1), positions.clamp_min(0)]
        self.tokens[:context_len].copy_(
            torch.where(
                positions >= 0, context, context.new_full((), self.eos_token_id)
            ).view(-1)
        )
        self.tokens[context_len : context_len + 1].copy_(
            last_sampled_tokens[request].view(-1)
        )
        self.count = 0
        self._fetch()

    def token(self, draft: torch.Tensor) -> None:
        """Fetch the rows of the next draft, ``draft`` (one token, on the GPU)."""
        index = self.count
        if not 0 < index < self.max_tokens:
            return
        start = self.context_len + index
        self.tokens[start : start + 1].copy_(draft.view(-1))
        self._fetch()

    def _fetch(self) -> None:
        index = self.count
        context_len = self.context_len
        start = context_len + index
        ids = self.ngram.compute_ngram_ids(
            self.tokens[start : start + 1],
            self.query_start_loc,
            self.tokens[index:start].view(1, context_len),
            output=self.ids[index : index + 1],
        )
        self.count = index + 1
        self._ship(index, ids)

    def _ship(self, index: int, ids: torch.Tensor) -> None:
        """Copy one token's ids to the host and read their rows when they land."""
        stream = self.stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.ids_host[index].copy_(ids.view(-1), non_blocking=True)
            self.events[index].record(stream)
        self.lookup.warm_async(
            self.ids_host[index].numpy(), self.events[index].synchronize
        )


def ple_early_begin(
    all_token_ids: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
) -> None:
    """Runner hook after sampling (``qwen4exp-spec-step``); see ``PLEEarlyFetch``."""
    for fetch in _EARLY_FETCHES:
        fetch.begin(all_token_ids, num_computed_tokens, last_sampled_tokens, idx_mapping)


def ple_early_token(draft: torch.Tensor) -> None:
    """Speculator hook after a draft pass (``qwen4exp-spec-step``)."""
    for fetch in _EARLY_FETCHES:
        fetch.token(draft)


def _refuse_capture() -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "PLE disk lookups run on the host and cannot be captured in a "
            "FULL CUDA graph; use -cc.cudagraph_mode=PIECEWISE or "
            "--enforce-eager"
        )


def _log_stats() -> None:
    now = time.monotonic()
    if _STATS_SEC <= 0 or now - _stats_last[0] < _STATS_SEC:
        return
    elapsed = now - _stats_last[0] if _stats_last[0] else float(_STATS_SEC)
    _stats_last[0] = now
    calls = STATS["calls"]
    if not calls:
        return
    per_call = {key: STATS[key] / calls for key in STATS if key.endswith("_ms")}
    cache_text = ""
    caches = [
        layer.lookup.cache
        for layer in _LAYERS.values()
        if layer._pools is not None
    ]
    if caches:
        hits, misses = STATS["cache_hits"], STATS["cache_misses"]
        cache_text = (
            f"; cache: {hits} hit rows, {misses} miss rows "
            f"({100.0 * hits / max(1, hits + misses):.1f}% hit), "
            f"{STATS['evictions']} evictions, {STATS['read_calls']} preads "
            f"({STATS['disk_reads']} waited for the disk), "
            f"{STATS['read_bytes'] / 2**20:.1f} MiB read, "
            f"{STATS['mmap_rows']} rows via mmap, "
            f"{sum(cache.used for cache in caches)}/"
            f"{sum(cache.num_slots - 1 for cache in caches)} slots used"
        )
    if _EARLY_FETCHES:
        tokens = STATS["early_tokens"]
        cache_text += (  # early row fetch, not cache-specific
            f"; early: {tokens} tokens, {STATS['early_rows']} rows read "
            f"({STATS['early_disk_reads']} reads waited for the disk), "
            f"{STATS['early_ms'] / max(1, tokens):.2f} ms/token"
        )
    logger.info(
        "PLE disk stats (last %.0fs): %d lookups (%d sync, %d prefetch hit, "
        "%d miss), %d rows (%d unique); ms/lookup: %.2f host op, %.2f prefetch "
        "wait, %.2f id sync, %.2f dedup, %.2f gather, %.3f H2D, %.3f decode "
        "launch%s",
        elapsed,
        calls,
        STATS["sync"],
        STATS["hit"],
        STATS["miss"],
        STATS["rows"],
        STATS["unique"],
        per_call["op_ms"],
        per_call["wait_ms"],
        per_call["id_sync_ms"],
        per_call["dedup_ms"],
        per_call["gather_ms"],
        per_call["h2d_ms"],
        per_call["launch_ms"],
        cache_text,
    )
    STATS.update(STATS_ZERO)


def _ple_disk_lookup(
    ngram_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    start_time = time.perf_counter()
    _LAYERS[layer_name].lookup_into(ngram_ids, output)
    STATS["calls"] += 1
    STATS["sync"] += 1
    STATS["op_ms"] += (time.perf_counter() - start_time) * 1e3
    _log_stats()


def _ple_disk_lookup_fake(
    ngram_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen4_exp_ple_disk_lookup",
    op_func=eager_break_during_capture(_ple_disk_lookup),
    mutates_args=["output"],
    fake_impl=_ple_disk_lookup_fake,
)
