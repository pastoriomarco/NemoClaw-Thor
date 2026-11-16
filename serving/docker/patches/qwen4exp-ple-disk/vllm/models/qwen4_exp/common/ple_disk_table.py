# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk-backed Qwen4Exp PLE row tables (host half of the disk PLE lookup).

The PLE n-gram table is far larger than what one token touches (16 rows per
token), so with ``VLLM_PLE_DISK=1`` it stays in the checkpoint files and is
read through ``np.memmap`` views of the tensors' byte ranges (page-cache
backed, like a GGUF mmap). This module only depends on NumPy so it can be
tested without vLLM; ``ple_disk.py`` holds the vLLM layer and GPU half.

Lookup (``PLEDiskLookup.stage``): dedup the row ids on the CPU, gather each
unique row of every table (packed NVFP4 rows, block scales) straight into one
staging byte buffer, and write the inverse indices in front of them, so the
GPU side needs one H2D copy and expands the rows itself. ``stage_async``
runs the same staging on a background thread once the ids are on the host,
so the GPU keeps working while the rows are gathered (prefetch).
"""

import json
import mmap
import os
import re
import struct
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

ENV_ENABLE = "VLLM_PLE_DISK"

# Per-row element sizes of the table dtypes the lookup supports.
_ITEMSIZE = {"U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "BF16": 2, "F16": 2}
_INDEX_BYTES = 8  # int64 inverse index per looked-up row


def ple_disk_enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "0").lower() in ("1", "true", "yes")


def ple_cache_requested() -> bool:
    """Whether a slot-pool cache switch is set (``ple_disk_cache.py``)."""
    return any(
        os.environ.get(name, "0") not in ("", "0")
        for name in ("VLLM_PLE_DISK_CACHE_GB", "VLLM_PLE_DISK_CACHE_SLOTS")
    )


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# Aggregate host-side timing, logged by ple_disk.py every
# VLLM_PLE_DISK_STATS_SEC seconds (0 = off). Counts: lookups, synchronous
# lookups, prefetches finished (hit) or not (miss) when the forward needed
# them. Times in ms: main-thread time in the lookup ops, main-thread wait for
# a prefetch, id D2H wait, dedup, row gather, H2D enqueue, decode launch.
STATS_ZERO = {
    "calls": 0,
    "sync": 0,
    "hit": 0,
    "miss": 0,
    "op_ms": 0.0,
    "wait_ms": 0.0,
    "id_sync_ms": 0.0,
    "dedup_ms": 0.0,
    "gather_ms": 0.0,
    "h2d_ms": 0.0,
    "launch_ms": 0.0,
    "rows": 0,
    "unique": 0,
    # Slot-pool cache (VLLM_PLE_DISK_CACHE_GB, ple_disk_cache.py): unique rows
    # served from the pool or read from disk, evicted rows, pread calls and
    # bytes (page-aligned, so more than the useful row bytes).
    "cache_hits": 0,
    "cache_misses": 0,
    "evictions": 0,
    "read_calls": 0,
    "read_bytes": 0,
    "disk_reads": 0,  # planned reads not in the page cache (RWF_NOWAIT)
    "mmap_rows": 0,  # missed rows gathered through the mmap (large steps)
    # Early row fetch (VLLM_PLE_DISK_EARLY): tokens fetched ahead of their
    # verify batch, rows read for them, reads of those that waited for the
    # disk, prefetch-thread time after the token's ids arrived.
    "early_tokens": 0,
    "early_rows": 0,
    "early_disk_reads": 0,
    "early_ms": 0.0,
}
STATS = dict(STATS_ZERO)


def parse_safetensors_header(path: str) -> tuple[dict, int]:
    """Return (tensor header, absolute data start) of a safetensors file."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


class PLEDiskRowTable:
    """Rows of one checkpoint-sharded table, e.g. ``ngram_embedding.weight``.

    ``shards`` maps the checkpoint shard index to ``(path, byte_offset, rows)``.
    Shard ``i`` holds global rows ``[i * shard_size, i * shard_size + rows)``,
    the layout ``Qwen4ExpNGramEmbedding.load_weights`` copies from.
    """

    def __init__(
        self,
        name: str,
        shards: dict[int, tuple[str, int, int]],
        row_bytes: int,
        num_rows: int,
        pool: ThreadPoolExecutor,
    ) -> None:
        if 0 not in shards:
            raise ValueError(f"PLE disk table {name}: shard 0 is missing")
        # Every shard but the last holds ceil(num_rows / split_ngram_parts) rows.
        shard_size = shards[0][2]
        num_shards = (num_rows + shard_size - 1) // shard_size
        self.name = name
        self.row_bytes = row_bytes
        self.num_rows = num_rows
        self.shard_size = shard_size
        self.pool = pool
        self.fast_rows = env_int("VLLM_PLE_DISK_FAST_ROWS", 512)
        self.chunk = max(1, env_int("VLLM_PLE_DISK_CHUNK", 2048))
        madv_random = env_int("VLLM_PLE_DISK_MADV_RANDOM", 1)
        self.mm: list[np.memmap] = []
        # Flat byte views of each checkpoint file, for the vectorized
        # decode-sized gather: one fancy-index read per file instead of one
        # np.take per shard.
        files: dict[str, int] = {}
        self.files: list[np.memmap] = []
        self.paths: list[str] = []  # checkpoint file of each self.files entry
        shard_file = []
        shard_base = []
        for index in range(num_shards):
            expected_rows = max(0, min(shard_size, num_rows - index * shard_size))
            if index not in shards:
                raise ValueError(f"PLE disk table {name}: shard {index} is missing")
            path, offset, rows = shards[index]
            if rows != expected_rows:
                raise ValueError(
                    f"PLE disk table {name}: shard {index} has {rows} rows, "
                    f"expected {expected_rows}"
                )
            table = np.memmap(
                path, dtype=np.uint8, mode="r", offset=offset, shape=(rows, row_bytes)
            )
            if madv_random:
                # Lookups touch scattered rows; readahead only evicts cache.
                table._mmap.madvise(mmap.MADV_RANDOM)
            self.mm.append(table)
            if path not in files:
                files[path] = len(self.files)
                flat = np.memmap(path, dtype=np.uint8, mode="r")
                if madv_random:
                    flat._mmap.madvise(mmap.MADV_RANDOM)
                self.files.append(flat)
                self.paths.append(path)
            shard_file.append(files[path])
            shard_base.append(offset)
        self.shard_file = np.asarray(shard_file, dtype=np.int64)
        self.shard_base = np.asarray(shard_base, dtype=np.int64)
        self.row_cols = np.arange(row_bytes, dtype=np.int64)

    def gather_into(self, rows: np.ndarray, out: np.ndarray) -> None:
        """Copy the sorted, unique, in-range ``rows`` into ``out[len(rows), B]``."""
        shard = rows // self.shard_size
        local = rows - shard * self.shard_size
        bounds = np.flatnonzero(np.diff(shard)) + 1
        starts = [0, *bounds.tolist()]
        ends = [*bounds.tolist(), rows.size]
        if rows.size <= self.fast_rows:
            # Decode-sized batches: thread-pool dispatch costs more than the
            # page-cache reads themselves, so gather inline, with one
            # vectorized byte gather per checkpoint file.
            file_ids = self.shard_file[shard]
            row_start = self.shard_base[shard] + local * self.row_bytes
            for file_id in np.unique(file_ids):
                sel = np.flatnonzero(file_ids == file_id)
                byte_idx = row_start[sel, None] + self.row_cols
                out[sel] = self.files[int(file_id)][byte_idx]
            return

        def run(task: tuple[int, int, int]) -> None:
            index, start, end = task
            # Page faults do the I/O; NumPy drops the GIL for the copy, so
            # tasks overlap their disk reads across threads.
            np.take(
                self.mm[index],
                local[start:end],
                axis=0,
                out=out[start:end],
                mode="clip",
            )

        tasks = [
            (int(shard[start]), chunk, min(chunk + self.chunk, end))
            for start, end in zip(starts, ends)
            for chunk in range(start, end, self.chunk)
        ]
        for _ in self.pool.map(run, tasks):
            pass


def find_ple_tables(
    model_path: str,
    layer_idx: int,
    suffixes: tuple[str, ...],
) -> dict[str, tuple[dict[int, tuple[str, int, int]], int, str]]:
    """Locate ``layers.<idx>.ple.ple_embedding.ngram_embedding.shard_<i>.<suffix>``.

    Returns ``{suffix: (shards, row_width, dtype)}`` with absolute byte offsets.
    """
    pattern = re.compile(
        rf"^(?!mtp\.)(?:.*\.)?layers\.{layer_idx}\.ple\.ple_embedding\."
        rf"ngram_embedding\.shard_(\d+)\.({'|'.join(map(re.escape, suffixes))})$"
    )
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        files = sorted(
            {
                os.path.join(model_path, file_name)
                for name, file_name in weight_map.items()
                if pattern.match(name)
            }
        )
    else:
        files = sorted(
            os.path.join(model_path, file_name)
            for file_name in os.listdir(model_path)
            if file_name.endswith(".safetensors")
        )

    tables: dict[str, tuple[dict[int, tuple[str, int, int]], int, str]] = {}
    for path in files:
        header, data_start = parse_safetensors_header(path)
        for name, meta in header.items():
            match = pattern.match(name)
            if match is None:
                continue
            shard_index, suffix = int(match.group(1)), match.group(2)
            rows, width = meta["shape"]
            dtype = meta["dtype"]
            start, end = meta["data_offsets"]
            if dtype not in _ITEMSIZE or end - start != rows * width * _ITEMSIZE[dtype]:
                raise ValueError(f"PLE disk tensor {name}: unsupported dtype/shape")
            shards, table_width, table_dtype = tables.setdefault(
                suffix, ({}, width, dtype)
            )
            if (width, dtype) != (table_width, table_dtype):
                raise ValueError(f"PLE disk tensor {name}: shards differ in layout")
            shards[shard_index] = (path, data_start + start, rows)
    return tables


class PLEDiskLookup:
    """Host half of a disk PLE lookup over one or more row-aligned tables."""

    def __init__(
        self,
        tables: list[PLEDiskRowTable],
        max_rows: int,
        vocab_start: int,
        vocab_end: int,
    ) -> None:
        self.tables = tables
        self.max_rows = max_rows
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.row_bytes = sum(table.row_bytes for table in tables)
        # Staging bytes needed for the largest batch: inverse + all tables.
        self.staging_bytes = max_rows * (_INDEX_BYTES + self.row_bytes)
        # One prefetch at a time; the thread starts on first use.
        self.prefetcher = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ple_disk_prefetch"
        )

    def stage(
        self, ids: np.ndarray, staging: np.ndarray
    ) -> tuple[int, int, list[int]]:
        """Write the inverse indices, then each table's unique rows, to ``staging``.

        Rows outside this rank's ``[vocab_start, vocab_end)`` get inverse ``-1``
        so the GPU side writes zeros for them, like the 0.33 lookup kernels.
        Returns ``(num_ids, num_unique, table_byte_offsets)``.
        """
        start_time = time.perf_counter()
        ids = ids.reshape(-1)
        num_ids = ids.size
        if num_ids > self.max_rows:
            raise ValueError(
                f"PLE disk staging holds {self.max_rows} rows, "
                f"but the batch needs {num_ids}"
            )
        owned = (ids >= self.vocab_start) & (ids < self.vocab_end)
        inverse = staging[: num_ids * _INDEX_BYTES].view(np.int64)
        if owned.all():
            # Sorted unique rows read with better locality, and repeated
            # n-grams cost one disk read and one H2D row.
            unique, inverse[:] = np.unique(ids, return_inverse=True)
        else:
            unique, owned_inverse = np.unique(ids[owned], return_inverse=True)
            inverse.fill(-1)
            inverse[owned] = owned_inverse
        gather_start = time.perf_counter()
        num_unique = unique.size
        offset = num_ids * _INDEX_BYTES
        offsets = []
        for table in self.tables:
            size = num_unique * table.row_bytes
            out = staging[offset : offset + size].reshape(num_unique, table.row_bytes)
            if num_unique:
                table.gather_into(unique, out)
            offsets.append(offset)
            offset += size
        end_time = time.perf_counter()
        STATS["dedup_ms"] += (gather_start - start_time) * 1e3
        STATS["gather_ms"] += (end_time - gather_start) * 1e3
        STATS["rows"] += num_ids
        STATS["unique"] += num_unique
        return num_ids, num_unique, offsets

    def stage_async(
        self,
        ids: np.ndarray,
        staging: np.ndarray,
        wait: Callable[[], None],
    ) -> "Future[tuple[int, int, list[int]]]":
        """``stage`` on the prefetch thread, after ``wait()`` makes ``ids`` valid.

        ``wait`` is the CUDA event that follows the ids' D2H copy. The caller
        keeps ``ids`` and ``staging`` untouched until the future is done.
        """

        def job() -> tuple[int, int, list[int]]:
            start_time = time.perf_counter()
            wait()
            STATS["id_sync_ms"] += (time.perf_counter() - start_time) * 1e3
            return self.stage(ids, staging)

        return self.prefetcher.submit(job)

    def warm(self, ids: np.ndarray) -> None:
        """Read the rows of ``ids`` and drop them; no lookup state changes.

        Early row fetch (``VLLM_PLE_DISK_EARLY``): the ids of one token of the
        next verify batch, known while the later tokens are still drafted. A
        later ``stage`` of the batch then finds these rows' pages in the page
        cache instead of waiting for the disk. Staging itself is unchanged, so
        its bytes are the same with or without this call.
        """
        start_time = time.perf_counter()
        ids = ids.reshape(-1)
        rows = np.unique(ids[(ids >= self.vocab_start) & (ids < self.vocab_end)])
        STATS["early_rows"] += self._read_early(rows)
        STATS["early_tokens"] += 1
        STATS["early_ms"] += (time.perf_counter() - start_time) * 1e3

    def _read_early(self, rows: np.ndarray) -> int:
        """Read ``rows`` (sorted, unique, owned) into scratch; return the count."""
        for table in self.tables:
            table.gather_into(rows, np.empty((rows.size, table.row_bytes), np.uint8))
        return rows.size

    def warm_async(self, ids: np.ndarray, wait: Callable[[], None]) -> Future:
        """``warm`` on the prefetch thread, after ``wait()`` makes ``ids`` valid.

        The one prefetch thread runs it in order with the staging jobs, so it
        never overlaps a ``stage`` and the next batch's staging job runs after
        it. The caller keeps ``ids`` untouched until that staging job is done.
        """

        def job() -> None:
            wait()
            self.warm(ids)

        return self.prefetcher.submit(job)


def open_ple_disk_lookup(
    model_path: str,
    layer_idx: int,
    table_row_bytes: dict[str, int],
    num_rows: int,
    max_rows: int,
    vocab_start: int,
    vocab_end: int,
) -> PLEDiskLookup:
    """Map the PLE tables of one layer and check them against the model layout.

    ``table_row_bytes`` lists the table suffixes in staging order with the row
    size the model expects; ``num_rows`` is the (padded) vocabulary size the
    checkpoint shards together cover.
    """
    found = find_ple_tables(model_path, layer_idx, tuple(table_row_bytes))
    pool = ThreadPoolExecutor(
        max_workers=max(1, env_int("VLLM_PLE_DISK_WORKERS", 14)),
        thread_name_prefix="ple_disk",
    )
    tables = []
    for suffix, row_bytes in table_row_bytes.items():
        if suffix not in found:
            raise ValueError(
                f"no PLE shard tensors for layers.{layer_idx}...shard_<i>.{suffix} "
                f"under {model_path}"
            )
        shards, width, dtype = found[suffix]
        if width * _ITEMSIZE[dtype] != row_bytes:
            raise ValueError(
                f"PLE disk table {suffix}: {width} x {dtype} rows, "
                f"expected {row_bytes} bytes"
            )
        tables.append(PLEDiskRowTable(suffix, shards, row_bytes, num_rows, pool))
    return PLEDiskLookup(tables, max_rows, vocab_start, vocab_end)
