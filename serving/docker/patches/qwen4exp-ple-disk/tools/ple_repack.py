# SPDX-License-Identifier: Apache-2.0
"""Write the Qwen4Exp PLE table as one repacked file (offline, NumPy only).

The FP4-Local checkpoint stores each 90-byte row as 80 B of packed values and
10 B of block scales in separate tensors, so a cache miss reads two pages.
This tool writes one file where each row's values and scales are contiguous
and whole rows fill 4 KiB blocks (45 rows of 90 B, 46 B padding), so a miss
reads exactly one page. Use it with ``VLLM_PLE_DISK_PACKED=<file>`` together
with ``VLLM_PLE_DISK_CACHE_GB`` (the slot-pool path; the plain disk path keeps
reading the checkpoint).

The file is a safetensors file with one U8 tensor ``ple_packed`` of shape
``[blocks, 4096]``; its data starts on a 4 KiB boundary (the header is padded),
and ``__metadata__`` records format, layer, row count and row layout, which
the loader checks against the model.

Usage (in any container with NumPy, model directory read-only):
    python3 -I tools/ple_repack.py --model /model --layer 1 \\
        --out /out/ple_packed_l1.safetensors [--rows-per-chunk 4500000]
It reads the whole table once (~29 GB) and writes ~29 GB; ``--verify`` (on
by default) re-reads random rows of the output and compares them with the
checkpoint.
"""

import argparse
import importlib.util
import json
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "vllm", "models", "qwen4_exp", "common")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(COMMON, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pdt = _load("ple_disk_table")
BLOCK = 4096
FORMAT = "qwen4exp-ple-packed-v1"
TENSOR = "ple_packed"
SUFFIXES = ("weight", "weight_scale")


def open_tables(model, layer):
    found = pdt.find_ple_tables(model, layer, SUFFIXES)
    missing = [suffix for suffix in SUFFIXES if suffix not in found]
    if missing:
        raise SystemExit(f"no PLE tensors layers.{layer}...shard_<i>.{missing} in {model}")
    pool = ThreadPoolExecutor(max_workers=8)
    tables = []
    num_rows = None
    for suffix in SUFFIXES:
        shards, width, dtype = found[suffix]
        rows = sum(entry[2] for entry in shards.values())
        num_rows = rows if num_rows is None else num_rows
        if rows != num_rows:
            raise SystemExit(f"{suffix}: {rows} rows, weight has {num_rows}")
        row_bytes = width * pdt._ITEMSIZE[dtype]
        tables.append(pdt.PLEDiskRowTable(suffix, shards, row_bytes, num_rows, pool))
    return tables, num_rows


def header_bytes(num_rows, layer, row_bytes, blocks):
    meta = {
        "format": FORMAT,
        "layer": str(layer),
        "num_rows": str(num_rows),
        "row_bytes": json.dumps(row_bytes),
        "block_bytes": str(BLOCK),
    }
    header = {
        TENSOR: {"dtype": "U8", "shape": [blocks, BLOCK], "data_offsets": [0, blocks * BLOCK]},
        "__metadata__": meta,
    }
    raw = json.dumps(header).encode()
    raw += b" " * (-(8 + len(raw)) % BLOCK)  # data starts on a block boundary
    return struct.pack("<Q", len(raw)) + raw


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="checkpoint directory")
    parser.add_argument("--layer", type=int, default=1, help="PLE layer index")
    parser.add_argument("--out", required=True, help="output file")
    parser.add_argument("--rows-per-chunk", type=int, default=45 * 100_000)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    tables, num_rows = open_tables(args.model, args.layer)
    for table in tables:
        table.fast_rows = 0  # threaded sequential reads
    row_bytes = [table.row_bytes for table in tables]
    length = sum(row_bytes)
    per_block = BLOCK // length
    blocks = -(-num_rows // per_block)
    chunk = max(per_block, args.rows_per_chunk // per_block * per_block)
    print(
        f"{num_rows} rows x {row_bytes} B -> {per_block} rows per {BLOCK} B block, "
        f"{blocks} blocks, {blocks * BLOCK / 2**30:.2f} GiB",
        flush=True,
    )
    tmp = args.out + ".partial"
    start = time.monotonic()
    with open(tmp, "wb") as out:
        out.write(header_bytes(num_rows, args.layer, row_bytes, blocks))
        for first in range(0, num_rows, chunk):
            rows = np.arange(first, min(num_rows, first + chunk), dtype=np.int64)
            count = -(-rows.size // per_block)
            packed = np.zeros((count * per_block, length), np.uint8)
            column = 0
            for table in tables:
                part = np.empty((rows.size, table.row_bytes), np.uint8)
                table.gather_into(rows, part)
                packed[: rows.size, column : column + table.row_bytes] = part
                column += table.row_bytes
            data = np.zeros((count, BLOCK), np.uint8)
            data[:, : per_block * length] = packed.reshape(count, per_block * length)
            out.write(data.tobytes())
            done = rows[-1] + 1
            rate = done / max(time.monotonic() - start, 1e-9)
            print(f"  {done}/{num_rows} rows, {(num_rows - done) / rate:.0f} s left", flush=True)
    os.replace(tmp, args.out)

    if not args.no_verify:
        header, data_start = pdt.parse_safetensors_header(args.out)
        check = np.random.default_rng(0).integers(0, num_rows, 4096)
        check = np.unique(np.concatenate((check, [0, num_rows - 1])))
        flat = np.memmap(args.out, dtype=np.uint8, mode="r")
        offset = data_start + check // per_block * BLOCK + check % per_block * length
        got = flat[offset[:, None] + np.arange(length)]
        column = 0
        for table in tables:
            want = np.empty((check.size, table.row_bytes), np.uint8)
            table.gather_into(check, want)
            if not np.array_equal(got[:, column : column + table.row_bytes], want):
                raise SystemExit(f"verify failed for {table.name}")
            column += table.row_bytes
        print(f"verify ok: {check.size} rows equal the checkpoint")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
