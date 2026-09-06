#!/usr/bin/env python3
"""Read-only PLE gather probe. Run with caches cleared BETWEEN variants.

Measures cold random-row disk latency, not end-to-end inference throughput.
Outputs hashes so variants must return identical bytes. Never alters weights.
"""
import argparse
import hashlib
import json
import mmap
import statistics
import time
import numpy as np
import vllm_ple_mmap as ple

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('model_path')
p.add_argument('--random-advice', action='store_true')
p.add_argument('--fast-rows', type=int, default=512)
a = p.parse_args()
shards, dtype, _ = ple._find_shards(a.model_path, 1)
cols = shards.pop('__cols__')
table = ple.MmapPleTable(shards, max(s[2] for s in shards.values()), cols,
                         ple._TABLE_DTYPES[dtype], workers=14)
table.fast_rows = a.fast_rows
for mapping in table.mm:
    if mapping is not None:
        mapping._mmap.madvise(mmap.MADV_RANDOM if a.random_advice else mmap.MADV_NORMAL)
rng = np.random.default_rng(42)
digest = hashlib.sha256()
times = []
for _ in range(100):
    ids = rng.integers(0, table.rows_total, size=64, dtype=np.int64)
    started = time.perf_counter()
    rows = table.gather(ids)
    times.append((time.perf_counter()-started)*1000)
    digest.update(rows.tobytes())
ids = rng.integers(0, table.rows_total, size=32768, dtype=np.int64)
started = time.perf_counter()
rows = table.gather(ids)
prefill_ms = (time.perf_counter()-started)*1000
digest.update(rows.tobytes())
print(json.dumps(dict(random_advice=a.random_advice, fast_rows=a.fast_rows,
                      mean_ms=statistics.mean(times), median_ms=statistics.median(times),
                      prefill_32768_rows_ms=prefill_ms, sha256=digest.hexdigest())), flush=True)
table.pool.shutdown()
