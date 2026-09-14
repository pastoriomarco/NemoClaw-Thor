#!/usr/bin/env python3
"""Add best-effort batched page prefetch to the installed mmap PLE reader.

The advice is enabled only for prefill-sized gathers by default.  This avoids
adding a syscall loop to the already-fast warm decode path while allowing NVMe
to queue random pages before the existing parallel NumPy gather faults them in.
"""

import ast
from pathlib import Path


path = Path("/usr/local/lib/python3.12/dist-packages/vllm_ple_mmap.py")
source = path.read_text()
marker = "NemoClaw-Thor: prefetch PLE pages"
if marker in source:
    print("PLE prefetch patch already present")
    raise SystemExit(0)

old = '''        self.paths: list[str | None] = [None] * (max(shards) + 1)
        self.mm: list[np.memmap | None] = [None] * (max(shards) + 1)
        self.rows_total = 0
'''
new = '''        self.paths: list[str | None] = [None] * (max(shards) + 1)
        self.mm: list[np.memmap | None] = [None] * (max(shards) + 1)
        # NemoClaw-Thor: prefetch PLE pages. Keep one read-only fd and the
        # safetensors data offset for every mapped shard.
        self.fds: list[int | None] = [None] * (max(shards) + 1)
        self.offsets: list[int] = [0] * (max(shards) + 1)
        self.rows_total = 0
'''
if source.count(old) != 1:
    raise RuntimeError("PLE table-layout anchor changed")
source = source.replace(old, new)

old = '''            self.paths[idx] = path
            self.mm[idx] = np.memmap(
                path, dtype=np.uint8, mode="r", offset=offset, shape=(rows, row_bytes)
            )
'''
new = '''            self.paths[idx] = path
            self.fds[idx] = os.open(path, os.O_RDONLY)
            self.offsets[idx] = int(offset)
            self.mm[idx] = np.memmap(
                path, dtype=np.uint8, mode="r", offset=offset, shape=(rows, row_bytes)
            )
'''
if source.count(old) != 1:
    raise RuntimeError("PLE mmap-open anchor changed")
source = source.replace(old, new)

old = '''        self.pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self.fast_rows = _env_int("VLLM_PLE_MMAP_FAST_ROWS", 512)

    def gather(self, ids: np.ndarray) -> np.ndarray:
'''
new = '''        self.pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self.fast_rows = _env_int("VLLM_PLE_MMAP_FAST_ROWS", 512)
        self.prefetch_min_rows = _env_int("VLLM_PLE_MMAP_PREFETCH_MIN_ROWS", 512)

    def _prefetch(self, ids: np.ndarray) -> None:
        """Best-effort POSIX_FADV_WILLNEED for unique pages in a large gather."""
        if (
            not _env_int("VLLM_PLE_MMAP_PREFETCH", 1)
            or ids.size < self.prefetch_min_rows
            or not hasattr(os, "posix_fadvise")
        ):
            return
        try:
            ids = np.unique(np.ascontiguousarray(ids, dtype=np.int64).reshape(-1))
            shard = ids // self.shard_size
            local = ids - shard * self.shard_size
            page_size = 4096
            for shard_id in np.unique(shard):
                fd = self.fds[int(shard_id)]
                if fd is None:
                    continue
                local_rows = local[shard == shard_id]
                byte_offsets = self.offsets[int(shard_id)] + local_rows * self.row_bytes
                first_pages = byte_offsets // page_size
                last_pages = (byte_offsets + self.row_bytes - 1) // page_size
                pages = np.unique(np.concatenate((first_pages, last_pages)))
                if pages.size == 0:
                    continue
                # Coalesce adjacent pages so sequential regions need one advice
                # call while random pages remain independent queue entries.
                breaks = np.flatnonzero(np.diff(pages) != 1) + 1
                for run in np.split(pages, breaks):
                    start = int(run[0]) * page_size
                    length = (int(run[-1]) - int(run[0]) + 1) * page_size
                    os.posix_fadvise(fd, start, length, os.POSIX_FADV_WILLNEED)
        except (OSError, ValueError, IndexError):
            # Advisory only: the existing mmap page-fault path remains correct.
            return

    def gather(self, ids: np.ndarray) -> np.ndarray:
'''
if source.count(old) != 1:
    raise RuntimeError("PLE gather-method anchor changed")
source = source.replace(old, new)

old = '''        t0 = _time.perf_counter()
        try:
            return self._gather(ids)
'''
new = '''        t0 = _time.perf_counter()
        try:
            self._prefetch(np.asarray(ids))
            return self._gather(ids)
'''
if source.count(old) != 1:
    raise RuntimeError("PLE gather-call anchor changed")
source = source.replace(old, new)

ast.parse(source)
path.write_text(source)
print(f"installed PLE page-prefetch patch in {path}")
