# SPDX-License-Identifier: Apache-2.0
"""CPU self-test for the early PLE row fetch (``VLLM_PLE_DISK_EARLY``).

Host half (NumPy only):
- ``warm`` changes no lookup state. A run of MTP-shaped verify batches and
  prefill chunks, where every token but the last of a verify batch is warmed
  first (through ``warm_async``, as the GPU hooks do), stages the same bytes,
  the same slots and leaves the same CLOCK state as a run without warming,
  for the uncached and the slot-pool cached lookup (also a pool that evicts
  every step and a TP vocabulary range).
- Warming reads exactly the rows the pool lacks (all owned rows uncached);
  its reads count as ``early_*``, not in the staging jobs' read counters.
- ``warm_async`` waits for its ids, and a staging job submitted after it runs
  after it (one prefetch thread, in order).

GPU-half id logic (needs the vLLM 0.33 image with this overlay mounted; skipped
otherwise): ``PLEEarlyFetch`` on CPU tensors, hooks driven as the runner and
speculator drive them, must hash each token to the ids the target computes
for the whole verify batch (stock ``Qwen4ExpModelState._prepare_ngram_context``
and ``compute_ngram_ids``), for K = 1..5, short histories, EOS inside the
context and the drafts, and several n-gram layouts. The D2H copy, events and
the Triton hash kernel run only on the GPU.

Run: python3 -I tests/test_ple_disk_early.py   (needs numpy; the id check needs
the image's vllm with the overlay files mounted)
"""

import importlib.util
import os
import sys
import tempfile
import threading
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "vllm", "models", "qwen4_exp", "common")

package = types.ModuleType("pledisk")
package.__path__ = [COMMON]
sys.modules["pledisk"] = package
import pledisk.ple_disk_cache as pdc  # noqa: E402
import pledisk.ple_disk_table as pdt  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "test_ple_disk_table", os.path.join(HERE, "test_ple_disk_table.py")
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.NUM_ROWS = 20000

HEADS = 16
READ_KEYS = ("read_calls", "disk_reads", "read_bytes", "mmap_rows")


def make_batches(rng, num_rows, k):
    """Verify batches of k + 1 tokens x 16 ids (mostly hot rows), prefill
    chunks in between, and a few repeats."""
    hot = rng.integers(0, num_rows, 120)
    batches = []
    for index in range(36):
        if index % 9 == 0:
            batches.append((rng.integers(0, num_rows, 150 * HEADS), False))
            continue
        size = (k + 1) * HEADS
        ids = np.where(rng.random(size) < 0.7, rng.choice(hot, size),
                       rng.integers(0, num_rows, size))
        batches.append((ids.astype(np.int64), True))
    return batches


def snapshot(lookup):
    cache = getattr(lookup, "cache", None)
    if cache is None:
        return ()
    return (cache.row_slot.copy(), cache.slot_row.copy(), cache.slot_ref.copy(),
            cache.slot_pins.copy(), cache.hand, cache.used, cache.evictions)


def same_state(a, b):
    return len(a) == len(b) and all(
        np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
        for x, y in zip(a, b)
    )


def stage_bytes(lookup, staged):
    if isinstance(staged, pdc.CacheStaged):
        last = lookup.tables[-1].row_bytes
        return staged.table_offsets[-1] + staged.num_misses * last
    num_ids, num_unique, (weight_offset, *_) = staged
    return weight_offset + num_unique * lookup.row_bytes


def run(label, make_lookup, batches, k):
    """Stage ``batches`` with and without warming; everything must match."""
    plain, early = make_lookup(), make_lookup()
    plain_staging = np.zeros(plain.staging_bytes, np.uint8)
    early_staging = np.zeros(early.staging_bytes, np.uint8)
    warmed_tokens = warmed_rows = 0
    for step, (ids, verify) in enumerate(batches):
        if verify:
            # What warm should read: per token, its owned rows not cached.
            want = 0
            for token in range(k):
                rows = ids[token * HEADS : (token + 1) * HEADS]
                rows = np.unique(rows[(rows >= early.vocab_start)
                                      & (rows < early.vocab_end)])
                if hasattr(early, "cache"):
                    rows = rows[early.cache.row_slot[rows] == 0]
                want += rows.size
            before = dict(pdt.STATS)
            jobs = []
            for token in range(k):
                ready = threading.Event()
                host = np.zeros(HEADS, np.int64)
                jobs.append(early.warm_async(host, ready.wait))
                host[:] = ids[token * HEADS : (token + 1) * HEADS]
                ready.set()
            for job in jobs:
                job.result(timeout=30)
            assert pdt.STATS["early_tokens"] - before["early_tokens"] == k
            got_rows = pdt.STATS["early_rows"] - before["early_rows"]
            assert got_rows == want, (label, step, got_rows, want)
            for key in READ_KEYS:
                assert pdt.STATS[key] == before[key], (label, step, key)
            warmed_tokens += k
            warmed_rows += got_rows
        before = dict(pdt.STATS)
        staged_plain = plain.stage(ids, plain_staging)
        plain_reads = {key: pdt.STATS[key] - before[key] for key in READ_KEYS}
        before = dict(pdt.STATS)
        staged_early = early.stage_async(ids.copy(), early_staging, lambda: None).result(
            timeout=30
        )
        early_reads = {key: pdt.STATS[key] - before[key] for key in READ_KEYS}
        if isinstance(staged_plain, pdc.CacheStaged):
            assert staged_plain[:4] == staged_early[:4], (label, step)
            assert np.array_equal(staged_plain.slots, staged_early.slots), (label, step)
        else:
            assert staged_plain == staged_early, (label, step)
        size = stage_bytes(plain, staged_plain)
        assert np.array_equal(plain_staging[:size], early_staging[:size]), (label, step)
        # The batch's own reads are counted as before (same misses, same plan;
        # the RWF_NOWAIT split may differ, so only calls without NOWAIT).
        if not getattr(getattr(plain, "reader", None), "nowait", False):
            assert plain_reads == early_reads, (label, step, plain_reads, early_reads)
        if hasattr(plain, "release"):
            plain.release(staged_plain)
            early.release(staged_early)
        assert same_state(snapshot(plain), snapshot(early)), (label, step)
    print(f"ok  {label}: {len(batches)} batches stage the same bytes and state "
          f"with {warmed_tokens} tokens warmed ({warmed_rows} rows read early)")


def test_host(rng):
    row_bytes = {"weight": base.EMBED_DIM // 2, "weight_scale": base.EMBED_DIM // 16}
    max_rows = 150 * HEADS
    with tempfile.TemporaryDirectory() as directory:
        base.build_checkpoint(directory, rng)
        tables = pdt.open_ple_disk_lookup(
            directory, base.LAYER, row_bytes, num_rows=base.NUM_ROWS,
            max_rows=max_rows, vocab_start=0, vocab_end=base.NUM_ROWS,
        ).tables
        half = base.NUM_ROWS // 2
        for k in (1, 3, 5):
            batches = make_batches(rng, base.NUM_ROWS, k)
            run(f"uncached, K={k}",
                lambda: pdt.PLEDiskLookup(tables, max_rows, 0, base.NUM_ROWS),
                batches, k)
            run(f"uncached, TP range, K={k}",
                lambda: pdt.PLEDiskLookup(tables, max_rows, half, base.NUM_ROWS),
                batches, k)
            for slots, nowait in ((base.NUM_ROWS + 1, True), (max_rows + 1, True),
                                  (max_rows + 1, False)):

                def make(slots=slots, nowait=nowait):
                    lookup = pdc.PLECachedLookup(tables, max_rows, 0, base.NUM_ROWS,
                                                 slots, 2)
                    lookup.reader.nowait = nowait and lookup.reader.nowait
                    return lookup

                run(f"cached, {slots} slots, RWF_NOWAIT {nowait}, K={k}", make,
                    batches, k)

        # Warm reads exactly the rows the pool lacks.
        lookup = pdc.PLECachedLookup(tables, max_rows, 0, base.NUM_ROWS,
                                     max_rows + 1, 2)
        staging = np.zeros(lookup.staging_bytes, np.uint8)
        staged = lookup.stage(np.arange(0, 64, dtype=np.int64), staging)
        lookup.release(staged)
        before = dict(pdt.STATS)
        lookup.warm(np.asarray([3, 3, 70, 10, 900] + [70] * 11, np.int64))
        assert pdt.STATS["early_rows"] - before["early_rows"] == 2  # 70, 900
        before = dict(pdt.STATS)
        lookup.warm(np.arange(0, 16, dtype=np.int64))  # all cached
        assert pdt.STATS["early_rows"] == before["early_rows"]
        assert pdt.STATS["early_tokens"] == before["early_tokens"] + 1
        print("ok  warm reads only the rows the pool lacks")

        # One prefetch thread, in order: staging submitted after a warm that
        # waits for its ids runs only after it.
        plain = pdt.PLEDiskLookup(tables, max_rows, 0, base.NUM_ROWS)
        warm_ready = threading.Event()
        warm_job = plain.warm_async(np.arange(16, dtype=np.int64), warm_ready.wait)
        stage_job = plain.stage_async(np.arange(16, dtype=np.int64),
                                      np.zeros(plain.staging_bytes, np.uint8),
                                      lambda: None)
        time.sleep(0.2)
        assert not warm_job.done() and not stage_job.done()
        warm_ready.set()
        stage_job.result(timeout=30)
        assert warm_job.done()
        print("ok  warm_async waits for its ids; later staging runs after it")
    assert set(pdt.STATS) == set(pdt.STATS_ZERO)


# ---------------------------------------------------------- GPU-half id logic


def test_ids(rng):
    try:
        import torch

        from vllm.models.qwen4_exp.common import ple_disk
        from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
        from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
            Qwen4ExpNGramEmbedding,
        )
    except Exception as error:  # noqa: BLE001 - outside the image
        print(f"skip early ids (needs the vLLM image with the overlay): {error!r}")
        return
    print(f"    ple_disk from {ple_disk.__file__}")
    assert hasattr(ple_disk, "PLEEarlyFetch"), "overlay ple_disk.py not mounted"
    cls = Qwen4ExpNGramEmbedding
    eos, vocab, max_reqs, max_len = 7, 40, 4, 64
    checked = 0
    for ngram_size, heads_per_ngram in ((2, 16), (3, 8), (5, 4)):
        ngram = cls.__new__(cls)
        torch.nn.Module.__init__(ngram)
        ngram.ngram_size = ngram_size
        ngram.heads_per_ngram = heads_per_ngram
        ngram.ngram_heads = (ngram_size - 1) * heads_per_ngram
        ngram.eos_token_id = eos
        ngram.layer_multipliers = torch.tensor(
            cls._make_layer_multipliers(ngram_size=ngram_size, unigram_vocab_size=248320,
                                        seed=1234, ple_dense_layer_id=1),
            dtype=torch.long,
        )
        sizes, offsets, _ = cls._make_vocab_layout(
            ngram_vocab_size_base=1_000_003, ngram_heads=ngram.ngram_heads,
            ple_dense_layer_id=1,
        )
        ngram.ngram_heads_vocab_sizes = torch.tensor(sizes, dtype=torch.long)
        ngram.ngram_heads_offsets = torch.tensor(offsets, dtype=torch.long)
        context_len = ngram_size - 1
        state = types.SimpleNamespace(
            ngram_context=torch.full((max_reqs, context_len), eos, dtype=torch.int32),
            ngram_eos_token_id=eos,
            ngram_context_offsets=torch.arange(-context_len, 0, dtype=torch.int64),
        )
        for k in range(1, 6):
            for trial in range(40):
                # Request state after postprocess_sampled: t0 sits at
                # num_computed in all_token_ids and in last_sampled_tokens.
                all_token_ids = torch.from_numpy(
                    rng.integers(0, vocab, (max_reqs, max_len)).astype(np.int32)
                )
                request = int(rng.integers(0, max_reqs))
                num_computed = int(rng.integers(0, 12)) if trial % 3 == 0 else int(
                    rng.integers(0, max_len - 8)
                )
                computed = torch.zeros(max_reqs, dtype=torch.int32)
                computed[request] = num_computed
                last_sampled = torch.zeros(max_reqs, 1, dtype=torch.int64)
                last_sampled[request, 0] = all_token_ids[request, num_computed]
                drafts = torch.from_numpy(
                    rng.integers(0, vocab, (max_reqs, k)).astype(np.int64)
                )
                idx_mapping = torch.tensor([request], dtype=torch.int32)

                # Target: next step's context and ids of [t0, d1..dK].
                batch = types.SimpleNamespace(num_reqs=1, idx_mapping=idx_mapping)
                req_states = types.SimpleNamespace(
                    num_computed_tokens=types.SimpleNamespace(gpu=computed),
                    all_token_ids=types.SimpleNamespace(gpu=all_token_ids),
                )
                context = Qwen4ExpModelState._prepare_ngram_context(
                    state, batch, req_states
                )
                input_ids = torch.cat(
                    [last_sampled[request].int(), drafts[request].int()]
                )
                want = ngram.compute_ngram_ids(
                    input_ids, torch.tensor([0, k + 1], dtype=torch.int32), context
                )

                # Early fetch, driven like the hooks.
                fetch = ple_disk.PLEEarlyFetch(ngram, None, k, torch.device("cpu"))
                shipped = []
                fetch._ship = lambda index, ids: shipped.append((index, ids.clone()))
                fetch.begin(all_token_ids, computed, last_sampled, idx_mapping)
                if k > 1:
                    fetch.token(drafts[request : request + 1, 0])
                for step in range(1, k):  # speculator: steps 1..K-1, not last
                    if step < k - 1:
                        fetch.token(drafts[request : request + 1, step])
                fetch.token(drafts[request : request + 1, k - 1])  # beyond: no-op
                assert [index for index, _ in shipped] == list(range(k)), shipped
                for index, ids in shipped:
                    assert torch.equal(ids.view(-1), want[index]), (
                        ngram_size, k, trial, index)
                fetch.count = -1  # start_prefetch disarms
                fetch.token(drafts[request : request + 1, 0])
                assert len(shipped) == k
                checked += k
    print(f"ok  early ids equal the target's verify-batch ids ({checked} tokens, "
          "K=1..5, n-gram sizes 2/3/5)")


def main():
    rng = np.random.default_rng(2)
    test_host(rng)
    test_ids(rng)
    return 0


if __name__ == "__main__":
    sys.exit(main())
