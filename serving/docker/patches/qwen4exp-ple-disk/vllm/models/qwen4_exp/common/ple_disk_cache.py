# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU slot-pool cache for the disk PLE table (host half, NumPy only).

``VLLM_PLE_DISK_CACHE_GB`` (or ``VLLM_PLE_DISK_CACHE_SLOTS``) turns it on.
Modelled on TensorRT Edge-LLM 0.12 ``--ple-residency cached``
(``cpp/runtime/embedding/pleCache``, ``pleReadPlan``, ``pleReadExecutor``):

- The GPU holds a pool of ``slots`` table rows in their stored form (packed
  NVFP4 values and block scales, one pool tensor per table). Slot 0 is
  reserved and zero: it is never a table row and serves rows this rank does
  not own.
- The host keeps the index (``ClockSlotCache``): row -> slot in a direct
  int32 array (0 = not cached), slot -> row, a CLOCK reference bit and a pin
  count per slot.
- A lookup dedups the ids, pins the step's whole union of rows (hits first,
  so a miss of the same step never evicts a hit), claims CLOCK slots for the
  misses, reads only the missed rows and stages: the per-id slot map
  (int32), the missed rows' slots and their bytes. The GPU side uploads
  that, scatters the missed rows into their slots and decodes through the
  slot map. Hits cost no disk read and no row upload.
- Miss reads (up to ``VLLM_PLE_DISK_PREAD_ROWS`` rows, i.e. decode and
  verify steps): page-aligned, sorted, merged ``pread`` ranges. One
  non-blocking pass (``RWF_NOWAIT``) on the calling thread takes what the
  page cache holds; the reads that must wait for the disk run on up to
  ``VLLM_PLE_DISK_READERS`` threads, so their NVMe latencies overlap. Larger
  miss sets (prefill chunks) use the vectorized mmap gather of the uncached
  path for the missed rows only, since one Python-level syscall per page
  costs more than the page faults there.
- The pins are released once the GPU work of the step is enqueued. Every
  later write to a slot (the scatter of a later step) is ordered after this
  step's decode on the same stream, so no fence is needed.

Every miss is read from the table on disk; nothing is dropped or
approximated, so the decoded rows equal the uncached path's.
"""

import json
import os
import struct
import time
from typing import NamedTuple

import numpy as np

from .ple_disk_table import (
    STATS,
    PLEDiskLookup,
    PLEDiskRowTable,
    env_int,
    open_ple_disk_lookup,
)

ENV_CACHE_GB = "VLLM_PLE_DISK_CACHE_GB"
ENV_CACHE_SLOTS = "VLLM_PLE_DISK_CACHE_SLOTS"
ENV_READERS = "VLLM_PLE_DISK_READERS"
ENV_PACKED = "VLLM_PLE_DISK_PACKED"
ENV_PREAD_ROWS = "VLLM_PLE_DISK_PREAD_ROWS"

MAX_READERS = 32  # as Edge-LLM's PleReadExecutor
# Host buffer for one wave of merged reads; larger miss sets read in waves.
READ_WAVE_BYTES = 64 << 20
# Repacked table: whole rows per 4 KiB block, so a row never spans two pages.
PACKED_BLOCK_BYTES = 4096
PACKED_FORMAT = "qwen4exp-ple-packed-v1"
PACKED_TENSOR = "ple_packed"


def ple_cache_slots(row_bytes: int) -> int:
    """Slot count from the environment (0 = cache off), slot 0 included."""
    slots = env_int(ENV_CACHE_SLOTS, 0)
    if slots > 0:
        return slots
    try:
        gib = float(os.environ.get(ENV_CACHE_GB, "0") or 0)
    except ValueError:
        gib = 0.0
    return max(0, int(gib * 2**30) // row_bytes)


class ClockSlotCache:
    """CLOCK slot index (Edge-LLM ``PleCache``), vectorized over a batch.

    ``reserve`` pins the batch's rows and returns their slots; ``release``
    drops those pins. The slot choice equals Edge-LLM's sequential CLOCK: the
    hand visits slots in order, skips pinned ones, takes a free or
    unreferenced slot, and clears the reference bit of referenced ones.
    """

    def __init__(self, num_rows: int, num_slots: int) -> None:
        if num_rows <= 0 or num_slots < 2 or num_slots > 2**31 - 1:
            raise ValueError(
                f"PLE cache needs 2 .. 2^31-1 slots and a nonempty table, got "
                f"{num_slots} slots for {num_rows} rows"
            )
        self.num_rows = num_rows
        self.num_slots = num_slots
        # Direct row -> slot map, 0 = not cached (slot 0 is never a row).
        # Touch every page now, so the 4 B/row are resident from the start
        # instead of growing during serving.
        self.row_slot = np.zeros(num_rows, np.int32)
        self.row_slot[::1024] = 0
        self.slot_row = np.full(num_slots, -1, np.int64)
        self.slot_ref = np.zeros(num_slots, bool)
        self.slot_pins = np.zeros(num_slots, np.int32)
        self.hand = 1
        self.used = 0
        self.evictions = 0

    def reserve(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pin sorted, unique, in-range ``rows``; return (slots int32, miss mask).

        Missed rows get CLOCK slots in row order; the caller fills them before
        any GPU work reads them, or calls ``rollback``.
        """
        if rows.size > self.num_slots - 1:
            raise ValueError(
                f"PLE cache step needs {rows.size} distinct rows, but the cache "
                f"has {self.num_slots - 1} usable slots (slot 0 is reserved); "
                f"raise {ENV_CACHE_GB} or lower --max-num-batched-tokens"
            )
        slots = self.row_slot[rows]
        miss = slots == 0
        hit_slots = slots[~miss]
        # Protect the whole union before CLOCK runs.
        self.slot_pins[hit_slots] += 1
        self.slot_ref[hit_slots] = True
        count = rows.size - hit_slots.size
        if count:
            claimed = self._claim(count)
            old = self.slot_row[claimed]
            evicted = old[old >= 0]
            self.row_slot[evicted] = 0
            self.evictions += evicted.size
            self.used += count - evicted.size
            miss_rows = rows[miss]
            self.slot_row[claimed] = miss_rows
            self.row_slot[miss_rows] = claimed
            slots[miss] = claimed
        return slots, miss

    def release(self, slots: np.ndarray) -> None:
        """Drop the pins ``reserve`` took for ``slots``."""
        self.slot_pins[slots] -= 1

    def rollback(self, rows: np.ndarray, slots: np.ndarray, miss: np.ndarray) -> None:
        """Undo a ``reserve`` whose misses were not filled (rows evicted for
        them stay evicted, as in Edge-LLM)."""
        miss_slots = slots[miss]
        self.row_slot[rows[miss]] = 0
        self.slot_row[miss_slots] = -1
        self.slot_ref[miss_slots] = False
        self.used -= miss_slots.size
        self.release(slots)

    def _claim(self, count: int) -> np.ndarray:
        usable = self.num_slots - 1
        claimed = []
        visited = 0
        while count:
            if visited >= 2 * usable:
                raise RuntimeError("PLE cache: CLOCK found no unpinned slot")
            window = min(usable, max(1024, 4 * count))
            index = (self.hand - 1 + np.arange(window)) % usable + 1
            unpinned = self.slot_pins[index] == 0
            referenced = self.slot_ref[index]
            take = np.flatnonzero(unpinned & ~referenced)[:count]
            # The hand stops right after the last slot it takes.
            stop = int(take[-1]) + 1 if take.size == count else window
            self.slot_ref[index[:stop][unpinned[:stop] & referenced[:stop]]] = False
            got = index[take]
            self.slot_pins[got] += 1
            self.slot_ref[got] = True
            claimed.append(got)
            count -= got.size
            visited += stop
            self.hand = int(index[stop - 1]) % usable + 1
        return np.concatenate(claimed).astype(np.int32)


class ReadPlan(NamedTuple):
    """Merged reads: ``file``, ``offset``, ``length``; ``position`` of each
    read in the concatenated read buffer; ``source`` of each span there."""

    file: np.ndarray
    offset: np.ndarray
    length: np.ndarray
    position: np.ndarray
    source: np.ndarray


def plan_reads(
    file: np.ndarray,
    offset: np.ndarray,
    length: int | np.ndarray,
    file_end: np.ndarray,
    page: int,
) -> ReadPlan:
    """Plan page-aligned reads for byte spans (Edge-LLM ``planPleReads``).

    Each span widens to whole pages (not past its file's end); the widened
    spans are sorted by (file, offset) and overlapping or adjacent ones are
    merged, so no page is read twice and no gap page is read. Unlike
    Edge-LLM, merged reads are not split at a maximum size: every page in a
    read holds a requested byte.
    """
    length = np.broadcast_to(np.asarray(length, np.int64), offset.shape)
    order = np.lexsort((offset, file))
    span_file = file[order]
    span_offset = offset[order]
    start = span_offset - span_offset % page
    end = span_offset + length[order]
    end = np.minimum(end + (-end) % page, file_end[span_file])
    # Running end of the current read within a file: (file, end) as one key.
    key = span_file.astype(np.int64) << 44 | end
    running = np.maximum.accumulate(key)
    new = np.ones(order.size, bool)
    new[1:] = (span_file[1:] != span_file[:-1]) | (start[1:] > (running[:-1] & ((1 << 44) - 1)))
    first = np.flatnonzero(new)
    read_offset = start[first]
    read_end = np.maximum.reduceat(end, first) if first.size else end[:0]
    read_length = read_end - read_offset
    position = np.zeros(first.size, np.int64)
    np.cumsum(read_length[:-1], out=position[1:])
    read_id = np.cumsum(new) - 1
    source = np.empty(order.size, np.int64)
    source[order] = position[read_id] + span_offset - read_offset[read_id]
    return ReadPlan(span_file[first], read_offset, read_length, position, source)


class PreadReader:
    """Parallel ``pread`` of planned reads (Edge-LLM ``PleReadExecutor``)."""

    def __init__(self, paths: list[str], readers: int, fadvise_random: bool) -> None:
        from concurrent.futures import ThreadPoolExecutor

        if not 1 <= readers <= MAX_READERS:
            raise ValueError(f"{ENV_READERS} must be 1..{MAX_READERS}, got {readers}")
        self.fds = [os.open(path, os.O_RDONLY | os.O_CLOEXEC) for path in paths]
        self.file_end = np.asarray([os.fstat(fd).st_size for fd in self.fds], np.int64)
        if fadvise_random:
            # Scattered rows: kernel readahead would only read unused pages.
            for fd in self.fds:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        self.readers = readers
        self.nowait = hasattr(os, "RWF_NOWAIT")
        self.pool = ThreadPoolExecutor(max_workers=readers, thread_name_prefix="ple_read")
        self.buffer = np.empty(0, np.uint8)

    def _read(self, plan: ReadPlan, items: list[tuple[int, int]], base: int) -> int:
        """Blocking reads of ``(read index, bytes already done)`` items."""
        view = memoryview(self.buffer)
        calls = 0
        for index, done in items:
            fd = self.fds[plan.file[index]]
            position = int(plan.position[index]) - base
            length = int(plan.length[index])
            offset = int(plan.offset[index])
            while done < length:
                count = os.preadv(
                    fd, [view[position + done : position + length]], offset + done
                )
                calls += 1
                if count <= 0:
                    raise OSError(f"PLE table file ended before byte {offset + length}")
                done += count
        return calls

    def _read_cached(
        self, plan: ReadPlan, first: int, last: int, base: int
    ) -> tuple[list[tuple[int, int]], int]:
        """Read what the page cache already holds, without blocking.

        ``RWF_NOWAIT`` returns at once (``EAGAIN``) when a read would wait for
        the disk; those reads, and the rest of short ones, go to the reader
        threads. Rows in the page cache thus cost one syscall on this thread
        and no thread hand-off.
        """
        if not self.nowait:
            return [(index, 0) for index in range(first, last)], 0
        view = memoryview(self.buffer)
        pending = []
        calls = 0
        for index in range(first, last):
            position = int(plan.position[index]) - base
            length = int(plan.length[index])
            calls += 1
            try:
                count = os.preadv(
                    self.fds[plan.file[index]],
                    [view[position : position + length]],
                    int(plan.offset[index]),
                    os.RWF_NOWAIT,
                )
            except BlockingIOError:
                count = 0
            except OSError:  # RWF_NOWAIT unsupported here: always block
                self.nowait = False
                return [(i, 0) for i in range(index, last)], calls
            if count < length:
                pending.append((index, max(count, 0)))
        return pending, calls

    def read_waves(self, plan: ReadPlan):
        """Yield ``(base, end)`` per wave once its reads are in ``self.buffer``;
        bytes ``[base, end)`` of the concatenated read space are then at
        ``self.buffer[: end - base]``."""
        total = int(plan.length.sum())
        cuts = np.searchsorted(
            plan.position, np.arange(0, total, READ_WAVE_BYTES), side="left"
        ).tolist()
        cuts = sorted(set(cuts) | {plan.length.size})
        for first, last in zip(cuts[:-1], cuts[1:]):
            base = int(plan.position[first])
            end = int(plan.position[last - 1] + plan.length[last - 1])
            if self.buffer.size < end - base:
                self.buffer = np.empty(max(end - base, READ_WAVE_BYTES), np.uint8)
            pending, calls = self._read_cached(plan, first, last, base)
            STATS["read_calls"] += calls
            STATS["disk_reads"] += len(pending)
            if len(pending) <= 1:
                STATS["read_calls"] += self._read(plan, pending, base)
            else:
                # The reads that wait for the disk, spread over the readers.
                tasks = min(self.readers, len(pending))
                for calls in self.pool.map(
                    lambda part: self._read(plan, part, base),
                    [pending[i::tasks] for i in range(tasks)],
                ):
                    STATS["read_calls"] += calls
            STATS["read_bytes"] += end - base
            yield base, end


class _Source(NamedTuple):
    """Where a row's bytes live: one span per row of ``length`` bytes;
    ``pieces`` are (column in the span, bytes, staging table index)."""

    file: np.ndarray  # per checkpoint shard (or one entry for a packed file)
    base: np.ndarray
    length: int
    pieces: list[tuple[int, int, int]]
    packed: bool  # one block-aligned file, PACKED_BLOCK_BYTES // length rows per block


class CacheStaged(NamedTuple):
    num_ids: int
    num_misses: int
    slot_offset: int  # int64 slots of the missed rows
    table_offsets: list[int]  # missed rows of each table
    slots: np.ndarray  # pinned slots of this step, for release


class PLECachedLookup(PLEDiskLookup):
    """Host half of a slot-pool cached lookup (``stage`` / ``release``).

    Staging layout: per-id slot map (int32), then at the next 8-byte boundary
    the missed rows' slots (int64), then each table's missed rows; the bytes
    up to the end of the last table are the step's whole upload.
    """

    def __init__(
        self,
        tables: list[PLEDiskRowTable],
        max_rows: int,
        vocab_start: int,
        vocab_end: int,
        num_slots: int,
        readers: int,
        packed_path: str | None = None,
        fadvise_random: bool = True,
        pread_rows: int = 4096,
    ) -> None:
        super().__init__(tables, max_rows, vocab_start, vocab_end)
        num_rows = tables[0].num_rows
        # One step's distinct rows must fit (Edge-LLM: min(rows, slots - 1)).
        if num_slots < min(max_rows, num_rows) + 1:
            raise ValueError(
                f"PLE cache has {num_slots} slots, but one step can need "
                f"{min(max_rows, num_rows)} rows plus slot 0; raise {ENV_CACHE_GB}"
            )
        self.cache = ClockSlotCache(num_rows, num_slots)
        self.page = os.sysconf("SC_PAGESIZE")
        self.staging_bytes = 4 * max_rows + 8 + max_rows * (8 + self.row_bytes)
        if packed_path:
            paths, self.sources = [packed_path], [
                _packed_source(packed_path, tables, num_rows)
            ]
        else:
            paths, self.sources = [], []
            for index, table in enumerate(tables):
                # Map this table's file indices into the shared file list.
                remap = []
                for path in table.paths:
                    if path not in paths:
                        paths.append(path)
                    remap.append(paths.index(path))
                file = np.asarray(remap, np.int64)[table.shard_file]
                self.sources.append(
                    _Source(
                        file,
                        table.shard_base,
                        table.row_bytes,
                        [(0, table.row_bytes, index)],
                        packed=False,
                    )
                )
        self.packed_path = packed_path
        self.reader = PreadReader(paths, readers, fadvise_random)
        # Steps with more misses (prefill chunks) gather them through the
        # mmap views instead: one vectorized gather per thread-pool task beats
        # one Python-level pread per page when most pages are cached.
        self.pread_rows = pread_rows
        self.packed_mm = None
        if packed_path:
            import mmap

            self.packed_mm = np.memmap(packed_path, dtype=np.uint8, mode="r")
            if fadvise_random:
                self.packed_mm._mmap.madvise(mmap.MADV_RANDOM)

    def _locate(self, source: _Source, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if source.packed:
            per_block = PACKED_BLOCK_BYTES // source.length
            offset = (
                source.base[0]
                + rows // per_block * PACKED_BLOCK_BYTES
                + rows % per_block * source.length
            )
            return np.zeros(rows.size, np.int64), offset
        shard_size = self.tables[0].shard_size
        shard = rows // shard_size
        offset = source.base[shard] + (rows - shard * shard_size) * source.length
        return source.file[shard], offset

    def stage(self, ids: np.ndarray, staging: np.ndarray) -> CacheStaged:
        start_time = time.perf_counter()
        ids = ids.reshape(-1)
        num_ids = ids.size
        if num_ids > self.max_rows:
            raise ValueError(
                f"PLE disk staging holds {self.max_rows} rows, "
                f"but the batch needs {num_ids}"
            )
        owned = (ids >= self.vocab_start) & (ids < self.vocab_end)
        if owned.all():
            unique, inverse = np.unique(ids, return_inverse=True)
        else:
            unique, inverse = np.unique(ids[owned], return_inverse=True)
        gather_start = time.perf_counter()
        evictions = self.cache.evictions
        slots, miss = self.cache.reserve(unique)
        slot_map = staging[: 4 * num_ids].view(np.int32)
        if owned.all():
            slot_map[:] = slots[inverse]
        else:
            slot_map.fill(0)  # slot 0: zero row, like the unowned rows today
            slot_map[owned] = slots[inverse]
        miss_rows = unique[miss]
        num_misses = miss_rows.size
        # Packed right after the slot map, so one H2D copy covers the step.
        slot_offset = 4 * num_ids + (-4 * num_ids) % 8
        staging[slot_offset : slot_offset + 8 * num_misses].view(np.int64)[:] = slots[
            miss
        ]
        offsets = []
        offset = slot_offset + 8 * num_misses
        outs = []
        for table in self.tables:
            size = num_misses * table.row_bytes
            offsets.append(offset)
            outs.append(staging[offset : offset + size].reshape(num_misses, table.row_bytes))
            offset += size
        try:
            if num_misses:
                self._read_rows(miss_rows, outs)
        except BaseException:
            self.cache.rollback(unique, slots, miss)
            raise
        end_time = time.perf_counter()
        STATS["dedup_ms"] += (gather_start - start_time) * 1e3
        STATS["gather_ms"] += (end_time - gather_start) * 1e3
        STATS["rows"] += num_ids
        STATS["unique"] += unique.size
        STATS["cache_hits"] += unique.size - num_misses
        STATS["cache_misses"] += num_misses
        STATS["evictions"] += self.cache.evictions - evictions
        return CacheStaged(num_ids, num_misses, slot_offset, offsets, slots)

    def _read_rows(self, rows: np.ndarray, outs: list[np.ndarray]) -> None:
        if rows.size > self.pread_rows:
            self._gather_mmap(rows, outs)
            return
        located = [self._locate(source, rows) for source in self.sources]
        plan = plan_reads(
            np.concatenate([file for file, _ in located]),
            np.concatenate([offset for _, offset in located]),
            np.concatenate(
                [np.full(rows.size, source.length, np.int64) for source in self.sources]
            ),
            self.reader.file_end,
            self.page,
        )
        sources = plan.source.reshape(len(self.sources), rows.size)
        for base, end in self.reader.read_waves(plan):
            for source, source_pos in zip(self.sources, sources):
                in_wave = np.flatnonzero((source_pos >= base) & (source_pos < end))
                if not in_wave.size:
                    continue
                local = source_pos[in_wave] - base
                for column, length, table_index in source.pieces:
                    outs[table_index][in_wave] = self.reader.buffer[
                        local[:, None] + column + np.arange(length)
                    ]

    def _gather_mmap(self, rows: np.ndarray, outs: list[np.ndarray]) -> None:
        """Gather sorted ``rows`` through the mmap views (page faults overlap
        across the gather pool's threads)."""
        STATS["mmap_rows"] += rows.size
        if self.packed_mm is None:
            for table, out in zip(self.tables, outs):
                table.gather_into(rows, out)
            return
        source = self.sources[0]
        _, offset = self._locate(source, rows)
        columns = np.arange(source.length)
        chunk = self.tables[0].chunk

        def run(start: int) -> None:
            part = self.packed_mm[offset[start : start + chunk, None] + columns]
            for column, length, table_index in source.pieces:
                outs[table_index][start : start + chunk] = part[:, column : column + length]

        for _ in self.tables[0].pool.map(run, range(0, rows.size, chunk)):
            pass

    def release(self, staged: CacheStaged) -> None:
        self.cache.release(staged.slots)

    def _read_early(self, rows: np.ndarray) -> int:
        """Early row fetch: read the rows the pool lacks, as ``stage`` would.

        Only reads ``row_slot``. It runs in order with the staging jobs on the
        prefetch thread, so the next ``stage`` claims and reads the same misses
        (plus the last token's), now from the page cache. Its read counters go to the ``early_*`` stats, so the ``preads``
        and ``waited for the disk`` counts stay those of the staging jobs.
        """
        rows = rows[self.cache.row_slot[rows] == 0]
        if not rows.size:
            return 0
        outs = [np.empty((rows.size, table.row_bytes), np.uint8) for table in self.tables]
        counted = ("read_calls", "disk_reads", "read_bytes", "mmap_rows")
        before = [STATS[key] for key in counted]
        self._read_rows(rows, outs)
        STATS["early_disk_reads"] += STATS["disk_reads"] - before[1]
        for key, value in zip(counted, before):
            STATS[key] = value
        return rows.size


def _packed_source(path: str, tables: list[PLEDiskRowTable], num_rows: int) -> _Source:
    header, data_start = parse_safetensors_header_with_metadata(path)
    meta = header.get("__metadata__", {})
    row_bytes = [table.row_bytes for table in tables]
    want = {
        "format": PACKED_FORMAT,
        "num_rows": str(num_rows),
        "row_bytes": json.dumps(row_bytes),
        "block_bytes": str(PACKED_BLOCK_BYTES),
    }
    for key, value in want.items():
        if meta.get(key) != value:
            raise ValueError(
                f"PLE packed table {path}: {key}={meta.get(key)!r}, expected {value!r}"
            )
    tensor = header.get(PACKED_TENSOR)
    if tensor is None or (data_start + tensor["data_offsets"][0]) % PACKED_BLOCK_BYTES:
        raise ValueError(f"PLE packed table {path}: no block-aligned {PACKED_TENSOR}")
    length = sum(row_bytes)
    pieces, column = [], 0
    for index, size in enumerate(row_bytes):
        pieces.append((column, size, index))
        column += size
    return _Source(
        np.zeros(1, np.int64),
        np.asarray([data_start + tensor["data_offsets"][0]], np.int64),
        length,
        pieces,
        packed=True,
    )


def parse_safetensors_header_with_metadata(path: str) -> tuple[dict, int]:
    """Like ``parse_safetensors_header``, keeping ``__metadata__``."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def open_ple_cached_lookup(
    model_path: str,
    layer_idx: int,
    table_row_bytes: dict[str, int],
    num_rows: int,
    max_rows: int,
    vocab_start: int,
    vocab_end: int,
    num_slots: int,
) -> PLECachedLookup:
    """Like ``open_ple_disk_lookup``, with the slot-pool cache in front."""
    plain = open_ple_disk_lookup(
        model_path, layer_idx, table_row_bytes, num_rows, max_rows, vocab_start, vocab_end
    )
    return PLECachedLookup(
        plain.tables,
        max_rows,
        vocab_start,
        vocab_end,
        num_slots,
        readers=env_int(ENV_READERS, 4),
        packed_path=os.environ.get(ENV_PACKED) or None,
        fadvise_random=env_int("VLLM_PLE_DISK_MADV_RANDOM", 1) != 0,
        pread_rows=env_int(ENV_PREAD_ROWS, 4096),
    )


__all__ = [
    "ClockSlotCache",
    "PLECachedLookup",
    "PreadReader",
    "ReadPlan",
    "plan_reads",
    "ple_cache_slots",
    "open_ple_cached_lookup",
]
