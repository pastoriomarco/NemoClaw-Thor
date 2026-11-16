# SPDX-License-Identifier: Apache-2.0
"""CPU self-test for the PLE slot-pool cache (``ple_disk_cache.py``).

- CLOCK: the vectorized ``ClockSlotCache`` against a sequential port of
  Edge-LLM 0.12 ``PleCache`` (same slots, same evictions, same hand), plus
  pinning (rows of an unreleased step survive), slot 0, capacity refusal and
  rollback.
- Read planner: page alignment, sorted non-overlapping merged reads, no gap
  pages, every span's bytes at its planned position, against brute force.
- Slot-map gather: a synthetic NVFP4-shaped checkpoint; a sequence of decode
  and prefill steps through the cached lookup, with the GPU half emulated in
  NumPy (one upload, scatter into the pool, gather through the slot map),
  must give the same rows as today's uncached staging; also through the
  prefetch path, a repacked table (``tools/ple_repack.py``), a small pool that
  evicts every step, and (with torch) the NVFP4 decode.
- Env plumbing: switches off by default, GB / slots / readers / packed path.

Run: python3 -I tests/test_ple_disk_cache.py   (needs numpy; torch optional)
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import threading
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
COMMON = os.path.join(ROOT, "vllm", "models", "qwen4_exp", "common")

# Load the two modules as a package so the relative import works.
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
# A table larger than one step, so small pools must evict.
base.NUM_ROWS = 20000


# --------------------------------------------------------------- CLOCK


class ReferenceClock:
    """Sequential port of Edge-LLM 0.12 PleCache (reserveBatch / claimSlot /
    commit / RequestLease release) for one request per batch."""

    def __init__(self, num_rows, num_slots):
        self.num_rows = num_rows
        self.row = [-1] * num_slots
        self.refs = [0] * num_slots
        self.state = ["free"] * num_slots
        self.referenced = [False] * num_slots
        self.state[0] = "ready"
        self.rows = {}
        self.clock = 1
        self.evictions = 0

    def claim(self):
        usable = len(self.row) - 1
        for _ in range(2 * usable):
            slot = self.clock
            self.clock = 1 if self.clock == usable else self.clock + 1
            if self.refs[slot]:
                continue
            if self.state[slot] == "free":
                return slot
            if self.referenced[slot]:
                self.referenced[slot] = False
                continue
            del self.rows[self.row[slot]]
            self.evictions += 1
            self.row[slot], self.refs[slot] = -1, 0
            self.state[slot], self.referenced[slot] = "free", False
            return slot
        raise RuntimeError("no unpinned slot")

    def reserve(self, request):
        rows = sorted(set(r for r in request if r != -1))
        if len(rows) > len(self.row) - 1:
            raise ValueError("capacity")
        protected = []
        for r in rows:
            if r in self.rows:
                slot = self.rows[r]
                self.refs[slot] += 1
                self.referenced[slot] = True
                protected.append(slot)
        for r in rows:
            if r not in self.rows:
                slot = self.claim()
                self.row[slot], self.refs[slot] = r, 1
                self.state[slot], self.referenced[slot] = "ready", True
                protected.append(slot)
                self.rows[r] = slot
        mapping = [0 if r == -1 else self.rows[r] for r in request]
        pinned = sorted(set(s for s in mapping if s))
        for slot in pinned:  # the request lease
            self.refs[slot] += 1
        for slot in protected:  # commit
            self.refs[slot] -= 1
        return mapping, pinned

    def release(self, pinned):
        for slot in pinned:
            self.refs[slot] -= 1


def run_cached(cache, request):
    request = np.asarray(request, np.int64)
    owned = request != -1
    unique, inverse = np.unique(request[owned], return_inverse=True)
    slots, miss = cache.reserve(unique)
    mapping = np.zeros(request.size, np.int64)
    mapping[owned] = slots[inverse]
    return mapping, slots, miss, unique


def test_clock(rng):
    for num_slots, num_rows, max_batch, steps in (
        (17, 60, 16, 400),
        (64, 500, 40, 400),
        (2000, 20000, 1500, 60),  # several CLOCK windows per claim
        (3, 10, 2, 200),
    ):
        ours = pdc.ClockSlotCache(num_rows, num_slots)
        ref = ReferenceClock(num_rows, num_slots)
        hot = rng.integers(0, num_rows, max(2, num_rows // 8))
        for step in range(steps):
            size = int(rng.integers(1, max_batch + 1))
            # Mix of hot rows (hits), cold rows (misses), repeats, padding.
            request = np.where(
                rng.random(size) < 0.6,
                rng.choice(hot, size),
                rng.integers(0, num_rows, size),
            )
            request[rng.random(size) < 0.05] = -1
            if np.unique(request[request != -1]).size > num_slots - 1:
                continue
            want_map, want_pinned = ref.reserve(request.tolist())
            got_map, slots, miss, unique = run_cached(ours, request)
            assert got_map.tolist() == want_map, (num_slots, step)
            assert ours.hand == ref.clock, (num_slots, step)
            assert ours.evictions == ref.evictions, (num_slots, step)
            assert (slots > 0).all(), "slot 0 handed out"
            # Pinned during the step: the step's rows are its pins.
            assert sorted(slots.tolist()) == want_pinned
            assert (ours.slot_pins[slots] == 1).all()
            ours.release(slots)
            ref.release(want_pinned)
            assert not ours.slot_pins.any()
            # Index consistency both ways.
            cached = np.flatnonzero(ours.row_slot)
            assert np.array_equal(ours.slot_row[ours.row_slot[cached]], cached)
            assert ours.used == cached.size == len(ref.rows)
        print(f"ok  CLOCK equals Edge-LLM PleCache: {num_slots} slots, {steps} steps")

    # Pins: rows of a step not yet released are never evicted.
    cache = pdc.ClockSlotCache(1000, 33)
    held_rows = np.arange(0, 16)
    held, _ = cache.reserve(held_rows)
    for start in range(100, 900, 16):
        slots, _ = cache.reserve(np.arange(start, start + 16))
        assert not np.intersect1d(slots, held).size
        assert np.array_equal(cache.row_slot[held_rows], held)
        cache.release(slots)
    try:
        cache.reserve(np.arange(500, 517))  # 16 pinned + 17 > 32 usable
    except RuntimeError:
        pass
    else:
        raise AssertionError("claim beyond unpinned slots accepted")
    print("ok  pinned rows survive later steps; no slot when all pinned refuses")

    # Capacity: a step larger than the usable slots refuses before mutating.
    cache = pdc.ClockSlotCache(100, 9)
    try:
        cache.reserve(np.arange(9))
    except ValueError:
        pass
    else:
        raise AssertionError("oversize step accepted")
    assert not cache.slot_pins.any() and cache.used == 0
    # Rollback: new rows unmapped, hits stay, pins dropped.
    slots, _ = cache.reserve(np.arange(4))
    cache.release(slots)
    rows = np.arange(2, 8)
    slots, miss = cache.reserve(rows)
    cache.rollback(rows, slots, miss)
    assert (cache.row_slot[[2, 3]] > 0).all() and not cache.row_slot[4:8].any()
    assert not cache.slot_pins.any() and cache.used == 4
    print("ok  capacity refusal, rollback")


# --------------------------------------------------------- read planner


def test_planner(rng):
    page = 64
    for trial in range(300):
        num_files = int(rng.integers(1, 4))
        sizes = rng.integers(3 * page, 40 * page, num_files)
        data = [rng.integers(0, 256, int(size), dtype=np.uint8) for size in sizes]
        count = int(rng.integers(1, 40))
        length = int(rng.integers(1, 2 * page))
        file = rng.integers(0, num_files, count)
        offset = np.asarray(
            [rng.integers(0, sizes[f] - length + 1) for f in file], np.int64
        )
        if trial % 3 == 0:  # duplicates and neighbours
            offset[: count // 2] = offset[count // 2 : 2 * (count // 2)] + rng.integers(
                0, 2, count // 2
            ) * length
            file[: count // 2] = file[count // 2 : 2 * (count // 2)]
            offset = np.minimum(offset, sizes[file] - length)
        plan = pdc.plan_reads(file, offset, length, np.asarray(sizes), page)
        buffer = np.concatenate(
            [data[f][o : o + n] for f, o, n in zip(plan.file, plan.offset, plan.length)]
        ) if plan.file.size else np.empty(0, np.uint8)
        # Every span's bytes are at its planned position.
        for f, o, s in zip(file, offset, plan.source):
            assert np.array_equal(buffer[s : s + length], data[f][o : o + length])
        # Aligned, sorted, disjoint and not adjacent (adjacent ones merge).
        assert (plan.offset % page == 0).all()
        end = plan.offset + plan.length
        assert ((end % page == 0) | (end == sizes[plan.file])).all()
        for a in range(plan.file.size - 1):
            if plan.file[a] == plan.file[a + 1]:
                assert end[a] < plan.offset[a + 1]
            else:
                assert plan.file[a] < plan.file[a + 1]
        assert np.array_equal(plan.position[1:], np.cumsum(plan.length)[:-1])
        # No gap pages: each page read holds a requested byte, and each
        # requested page is read exactly once.
        want = set()
        for f, o in zip(file, offset):
            want |= {(int(f), p) for p in range(o // page, (o + length - 1) // page + 1)}
        got = [
            (int(f), p)
            for f, o, n in zip(plan.file, plan.offset, plan.length)
            for p in range(o // page, (o + n - 1) // page + 1)
        ]
        assert len(got) == len(set(got)) and set(got) == want
    print("ok  read planner: 300 random plans (alignment, merge, no gap pages)")


# ------------------------------------------------- slot-map vs today's path


class GpuEmulation:
    """The GPU half of ``_upload_and_decode_cached`` in NumPy."""

    def __init__(self, lookup):
        self.lookup = lookup
        slots = lookup.cache.num_slots
        self.pools = [np.zeros((slots, t.row_bytes), np.uint8) for t in lookup.tables]
        self.device = np.zeros(lookup.staging_bytes, np.uint8)

    def run(self, staged, staging):
        num_ids, num_misses, slot_offset, offsets, _ = staged
        last = self.lookup.tables[-1].row_bytes
        upload = offsets[-1] + num_misses * last
        self.device[:upload] = staging[:upload]  # the one H2D copy
        self.device[upload:] = 0xAB  # nothing beyond the upload may be read
        slots = self.device[slot_offset : slot_offset + 8 * num_misses].view(np.int64)
        for pool, offset in zip(self.pools, offsets):
            rows = self.device[offset : offset + num_misses * pool.shape[1]]
            pool[slots] = rows.reshape(num_misses, pool.shape[1])  # index_copy_
        slot_map = self.device[: 4 * num_ids].view(np.int32)
        out = [pool[slot_map] for pool in self.pools]  # kernel loads via slots
        self.lookup.release(staged)
        return out


def make_steps(rng, num_rows):
    hot = rng.integers(0, num_rows, 60)
    steps = []
    for index in range(40):
        if index % 10 == 0:  # prefill chunk
            ids = rng.integers(0, num_rows, 200 * 16)
        else:  # decode / verify: 1-4 tokens x 16 heads, mostly hot rows
            size = int(rng.integers(1, 5)) * 16
            ids = np.where(rng.random(size) < 0.7, rng.choice(hot, size),
                           rng.integers(0, num_rows, size))
        steps.append(ids.astype(np.int64))
    steps.append(np.asarray([0, base.NUM_ROWS - 1] * 8, np.int64))
    return steps


def check_lookup(label, cached, plain, steps, torch=None):
    staging = np.zeros(cached.staging_bytes, np.uint8)
    plain_staging = np.zeros(plain.staging_bytes, np.uint8)
    gpu = GpuEmulation(cached)
    hits = misses = 0
    for step, ids in enumerate(steps):
        before = dict(pdt.STATS)
        if step % 2:  # prefetch path: waits for the ids, then stages
            ready = threading.Event()
            host = np.zeros_like(ids)
            job = cached.stage_async(host, staging, ready.wait)
            host[:] = ids
            ready.set()
            staged = job.result(timeout=30)
        else:
            staged = cached.stage(ids, staging)
        got = gpu.run(staged, staging)
        want = base.expand(plain, plain_staging, ids)
        for g, w in zip(got, want):
            assert np.array_equal(g, w), (label, step)
        hits += pdt.STATS["cache_hits"] - before["cache_hits"]
        misses += pdt.STATS["cache_misses"] - before["cache_misses"]
        if torch is not None:
            a = base.nvfp4_reference(torch, got[0], got[1], 0.5)
            b = base.nvfp4_reference(torch, want[0], want[1], 0.5)
            assert torch.equal(a, b), (label, step)
    assert not cached.cache.slot_pins.any(), "pins leaked"
    print(
        f"ok  {label}: {len(steps)} steps equal today's rows "
        f"({hits} hit / {misses} miss rows, {cached.cache.evictions} evictions"
        f"{', NVFP4 decode equal' if torch is not None else ''})"
    )


def test_lookup(rng, torch):
    row_bytes = {"weight": base.EMBED_DIM // 2, "weight_scale": base.EMBED_DIM // 16}
    max_rows = 200 * 16
    with tempfile.TemporaryDirectory() as directory:
        base.build_checkpoint(directory, rng)
        steps = make_steps(rng, base.NUM_ROWS)
        plain = pdt.open_ple_disk_lookup(
            directory, base.LAYER, row_bytes, num_rows=base.NUM_ROWS,
            max_rows=max_rows, vocab_start=0, vocab_end=base.NUM_ROWS,
        )
        packed = os.path.join(directory, "packed.safetensors")
        subprocess.run(
            [sys.executable, "-I", os.path.join(ROOT, "tools", "ple_repack.py"),
             "--model", directory, "--layer", str(base.LAYER), "--out", packed,
             "--rows-per-chunk", "4500"],
            check=True, stdout=subprocess.DEVNULL,
        )
        for label, slots, readers, path, pread_rows, nowait in (
            ("cached, pool holds the table", base.NUM_ROWS + 1, 4, None, 4096, True),
            ("cached, pool = one step (evicts every step)", max_rows + 1, 3, None,
             4096, True),
            ("cached, 1 reader, no RWF_NOWAIT", max_rows + 1, 1, None, 4096, False),
            ("cached, 8 readers, no RWF_NOWAIT", max_rows + 1, 8, None, 4096, False),
            ("cached, repacked table", max_rows + 1, 8, packed, 4096, True),
            ("cached, mmap gather above 16 misses", max_rows + 1, 2, None, 16, True),
            ("cached, repacked, mmap gather above 16 misses", max_rows + 1, 2, packed,
             16, True),
        ):
            cached = pdc.PLECachedLookup(
                plain.tables, max_rows, 0, base.NUM_ROWS, slots, readers,
                packed_path=path, pread_rows=pread_rows,
            )
            cached.reader.nowait = nowait and cached.reader.nowait
            mmap_rows = pdt.STATS["mmap_rows"]
            check_lookup(label, cached, plain, steps, torch)
            assert (pdt.STATS["mmap_rows"] > mmap_rows) == (pread_rows == 16), label

        # Small read waves: many waves per step must give the same rows.
        wave = pdc.READ_WAVE_BYTES
        pdc.READ_WAVE_BYTES = 3 * 4096
        try:
            cached = pdc.PLECachedLookup(plain.tables, max_rows, 0, base.NUM_ROWS,
                                         max_rows + 1, 4)
            check_lookup("cached, 12 KiB read waves", cached, plain, steps[:12])
        finally:
            pdc.READ_WAVE_BYTES = wave

        # Repacked reads: one pread per missed row (no scale page).
        cached = pdc.PLECachedLookup(plain.tables, max_rows, 0, base.NUM_ROWS,
                                     max_rows + 1, 1, packed_path=packed)
        cached.reader.nowait = False  # count blocking preads only
        staging = np.zeros(cached.staging_bytes, np.uint8)
        calls = pdt.STATS["read_calls"]
        ids = np.asarray([3, 4000, 19999], np.int64)
        staged = cached.stage(ids, staging)
        assert staged.num_misses == 3 and pdt.STATS["read_calls"] - calls == 3
        calls = pdt.STATS["read_calls"]
        unpacked = pdc.PLECachedLookup(plain.tables, max_rows, 0, base.NUM_ROWS,
                                       max_rows + 1, 1)
        unpacked.reader.nowait = False
        unpacked.stage(ids, np.zeros(unpacked.staging_bytes, np.uint8))
        assert pdt.STATS["read_calls"] - calls == 6
        print("ok  repacked table reads one page per missed row (3 vs 6 preads)")

        # A packed file for another layout is refused.
        try:
            pdc.PLECachedLookup(plain.tables[:1], max_rows, 0, base.NUM_ROWS,
                                max_rows + 1, 1, packed_path=packed)
        except ValueError:
            print("ok  mismatched packed table refused")
        else:
            raise AssertionError("mismatched packed table accepted")

        # TP rank range: unowned ids map to slot 0 (zero rows).
        half = base.NUM_ROWS // 2
        cached = pdc.PLECachedLookup(plain.tables, max_rows, half, base.NUM_ROWS,
                                     max_rows + 1, 2)
        ranked = pdt.open_ple_disk_lookup(
            directory, base.LAYER, row_bytes, num_rows=base.NUM_ROWS,
            max_rows=max_rows, vocab_start=half, vocab_end=base.NUM_ROWS,
        )
        check_lookup("cached, rows outside the rank range -> slot 0", cached, ranked,
                     steps[:6])

        # Too few slots for a step refuses at construction.
        try:
            pdc.PLECachedLookup(plain.tables, max_rows, 0, base.NUM_ROWS, max_rows, 1)
        except ValueError:
            print("ok  pool smaller than one step refused")
        else:
            raise AssertionError("undersized pool accepted")
        return directory


def test_env(rng):
    for name in (pdc.ENV_CACHE_GB, pdc.ENV_CACHE_SLOTS, pdc.ENV_READERS,
                 pdc.ENV_PACKED, pdc.ENV_PREAD_ROWS):
        os.environ.pop(name, None)
    assert pdc.ple_cache_slots(90) == 0  # default off
    assert not pdt.ple_cache_requested()  # ple_disk.py skips the import
    os.environ[pdc.ENV_CACHE_GB] = "0.5"
    assert pdt.ple_cache_requested()
    assert pdc.ple_cache_slots(90) == 2**29 // 90
    os.environ[pdc.ENV_CACHE_SLOTS] = "12345"  # slots win over GB
    assert pdc.ple_cache_slots(90) == 12345
    os.environ.pop(pdc.ENV_CACHE_SLOTS)
    os.environ[pdc.ENV_CACHE_GB] = "junk"
    assert pdc.ple_cache_slots(90) == 0
    os.environ.pop(pdc.ENV_CACHE_GB)
    row_bytes = {"weight": base.EMBED_DIM // 2, "weight_scale": base.EMBED_DIM // 16}
    with tempfile.TemporaryDirectory() as directory:
        base.build_checkpoint(directory, rng)
        os.environ[pdc.ENV_READERS] = "5"
        os.environ[pdc.ENV_PREAD_ROWS] = "77"
        os.environ["VLLM_PLE_DISK_MADV_RANDOM"] = "0"
        lookup = pdc.open_ple_cached_lookup(
            directory, base.LAYER, row_bytes, num_rows=base.NUM_ROWS, max_rows=64,
            vocab_start=0, vocab_end=base.NUM_ROWS, num_slots=65,
        )
        assert lookup.reader.readers == 5 and lookup.packed_path is None
        assert lookup.pread_rows == 77
        assert len(lookup.reader.fds) == 2  # both checkpoint files, once each
        os.environ[pdc.ENV_READERS] = "33"
        try:
            pdc.open_ple_cached_lookup(
                directory, base.LAYER, row_bytes, num_rows=base.NUM_ROWS,
                max_rows=64, vocab_start=0, vocab_end=base.NUM_ROWS, num_slots=65,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("33 readers accepted")
        for name in (pdc.ENV_READERS, pdc.ENV_PREAD_ROWS, "VLLM_PLE_DISK_MADV_RANDOM"):
            os.environ.pop(name)
    assert set(pdt.STATS) == set(pdt.STATS_ZERO)
    print("ok  env: off by default, GB/slots/readers/MADV plumbing, 1..32 readers")


def main():
    rng = np.random.default_rng(1)
    try:
        import torch
    except ImportError:
        torch = None
    test_clock(rng)
    test_planner(rng)
    test_lookup(rng, torch)
    test_env(rng)
    return 0


if __name__ == "__main__":
    sys.exit(main())
