# SPDX-License-Identifier: Apache-2.0
"""CPU self-test for the disk-backed Qwen4Exp PLE lookup (host half).

Builds a tiny checkpoint with an NVFP4-shaped PLE table split into shards over
two safetensors files (plus an index and distractor tensors), maps it with
``ple_disk_table`` and checks that staged rows equal direct indexing of the
full table: for decode-sized (inline) and prefill-sized (threaded) batches,
with repeated ids, and with a TP vocabulary range that leaves rows unowned.
The prefetch staging (``stage_async``) must wait for its ids and write the
same staging bytes as the synchronous ``stage``.
With torch installed it also decodes both through a reference NVFP4 decoder
written like the 0.33 Triton kernel and compares the BF16 outputs.

Run: python3 -I tests/test_ple_disk_table.py   (needs numpy; torch optional)
"""

import importlib.util
import json
import os
import struct
import sys
import tempfile
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.join(
    HERE, "..", "vllm", "models", "qwen4_exp", "common", "ple_disk_table.py"
)
spec = importlib.util.spec_from_file_location("ple_disk_table", MODULE)
pdt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdt)

EMBED_DIM = 160  # Qwen3.8-Flash-Next head_dim: 2560 / 16 n-gram heads
NUM_ROWS = 1000  # padded n-gram vocabulary (org_vocab_size)
SPLIT_PARTS = 7  # split_ngram_parts
LAYER = 1
PREFIX = f"model.language_model.layers.{LAYER}.ple.ple_embedding.ngram_embedding"


def write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        data = np.ascontiguousarray(array).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for data in blobs:
            f.write(data)


def build_checkpoint(directory, rng):
    weight = rng.integers(0, 256, (NUM_ROWS, EMBED_DIM // 2), dtype=np.uint8)
    # E4M3 codes 0x00..0x7e (positive finite scales).
    scale = rng.integers(0, 0x7F, (NUM_ROWS, EMBED_DIM // 16), dtype=np.uint8)
    shard_size = (NUM_ROWS + SPLIT_PARTS - 1) // SPLIT_PARTS
    files = ({}, {})
    for shard in range(SPLIT_PARTS):
        rows = slice(shard * shard_size, min(NUM_ROWS, (shard + 1) * shard_size))
        # Interleave shards over two files, weight and scale apart.
        files[shard % 2][f"{PREFIX}.shard_{shard}.weight"] = ("U8", weight[rows])
        files[(shard + 1) % 2][f"{PREFIX}.shard_{shard}.weight_scale"] = (
            "F8_E4M3",
            scale[rows],
        )
    files[0][f"{PREFIX}.weight_scale_2"] = ("F32", np.array([0.5], np.float32))
    # Distractors: other layer and MTP tensors with the same suffixes.
    files[1][f"mtp.layers.{LAYER}.ple.ple_embedding.ngram_embedding.shard_0.weight"] = (
        "U8",
        weight[:3],
    )
    files[1]["model.language_model.layers.11.ple.ple_embedding.ngram_embedding.shard_0.weight"] = (
        "U8",
        weight[:3],
    )
    weight_map = {}
    for index, tensors in enumerate(files):
        name = f"model-0000{index + 1}-of-00002.safetensors"
        write_safetensors(os.path.join(directory, name), tensors)
        weight_map.update(dict.fromkeys(tensors, name))
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    return weight, scale


def expand(lookup, staging, ids):
    """Rebuild per-id rows from the staging buffer like the GPU side does."""
    num_ids, num_unique, offsets = lookup.stage(ids, staging)
    inverse = staging[: num_ids * 8].view(np.int64)
    rows = []
    for table, offset in zip(lookup.tables, offsets):
        unique_rows = staging[offset : offset + num_unique * table.row_bytes].reshape(
            num_unique, table.row_bytes
        )
        out = np.zeros((num_ids, table.row_bytes), np.uint8)
        owned = inverse >= 0
        out[owned] = unique_rows[inverse[owned]]
        rows.append(out)
    return rows


def direct(weight, scale, ids, vocab_start, vocab_end):
    owned = (ids >= vocab_start) & (ids < vocab_end)
    safe = np.where(owned, ids, 0)
    return (
        np.where(owned[:, None], weight[safe], 0).astype(np.uint8),
        np.where(owned[:, None], scale[safe], 0).astype(np.uint8),
    )


def nvfp4_reference(torch, packed, scale, global_scale):
    """Decode like _lookup_nvfp4_ple_embedding_kernel (low nibble = even)."""
    e2m1 = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    packed = torch.from_numpy(packed).long()
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
    scales = torch.from_numpy(scale).view(torch.float8_e4m3fn).float()
    values = e2m1[codes] * scales.repeat_interleave(16, dim=-1) * global_scale
    return values.to(torch.bfloat16)


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as directory:
        weight, scale = build_checkpoint(directory, rng)
        row_bytes = {"weight": EMBED_DIM // 2, "weight_scale": EMBED_DIM // 16}
        cases = [
            ("decode, TP=1", 4 * 16, 0, NUM_ROWS),
            ("prefill, TP=1", 300 * 16, 0, NUM_ROWS),
            ("prefill, TP rank 1 of 2", 300 * 16, NUM_ROWS // 2, NUM_ROWS),
        ]
        for label, num_ids, vocab_start, vocab_end in cases:
            lookup = pdt.open_ple_disk_lookup(
                directory,
                LAYER,
                row_bytes,
                num_rows=NUM_ROWS,
                max_rows=num_ids,
                vocab_start=vocab_start,
                vocab_end=vocab_end,
            )
            for table in lookup.tables:
                if num_ids > 64:
                    table.fast_rows = 16  # force the threaded path
                table.chunk = 64  # several tasks per shard on the threaded path
            staging = np.empty(lookup.staging_bytes, np.uint8)
            # Repeats are common; include the first and last row explicitly.
            ids = rng.integers(0, NUM_ROWS, num_ids // 2)
            ids = np.concatenate((ids, ids[: num_ids // 2 - 2], [0, NUM_ROWS - 1]))
            rng.shuffle(ids)
            got_weight, got_scale = expand(lookup, staging, ids)
            want_weight, want_scale = direct(weight, scale, ids, vocab_start, vocab_end)
            assert np.array_equal(got_weight, want_weight), label
            assert np.array_equal(got_scale, want_scale), label
            print(f"ok  {label}: {num_ids} ids, rows equal direct indexing")

            # Prefetch: the job waits for the ids (the D2H event), then stages
            # exactly the bytes the synchronous path stages.
            want_staged = lookup.stage(ids, staging)
            want_bytes = staging.copy()
            ids_host = np.zeros(num_ids, np.int64)  # not yet "copied"
            async_staging = np.zeros_like(staging)
            ready = threading.Event()
            job = lookup.stage_async(ids_host, async_staging, ready.wait)
            ids_host[:] = ids  # the D2H copy lands before the event fires
            assert not job.done(), label
            ready.set()
            assert job.result(timeout=30) == want_staged, label
            num_bytes = want_staged[2][-1] + want_staged[1] * lookup.tables[-1].row_bytes
            assert np.array_equal(async_staging[:num_bytes], want_bytes[:num_bytes]), label
            print(f"ok  {label}: prefetch staging equals synchronous staging")

            try:
                import torch
            except ImportError:
                continue
            got = nvfp4_reference(torch, got_weight, got_scale, 0.5)
            want = nvfp4_reference(torch, want_weight, want_scale, 0.5)
            assert torch.equal(got, want), label
            print(f"ok  {label}: NVFP4 decode equal ({got.shape[1]} values/row)")

        # Too many ids for the staging buffer must refuse, not overrun.
        try:
            lookup.stage(np.zeros(lookup.max_rows + 1, np.int64), staging)
        except ValueError:
            print("ok  oversize batch refused")
        else:
            raise AssertionError("oversize batch accepted")
        job = lookup.stage_async(
            np.zeros(lookup.max_rows + 1, np.int64), staging, lambda: None
        )
        try:
            job.result(timeout=30)
        except ValueError:
            print("ok  oversize prefetch refused (raised to the caller)")
        else:
            raise AssertionError("oversize prefetch accepted")
        timed = ("id_sync_ms", "dedup_ms", "gather_ms")
        assert all(pdt.STATS[key] > 0 for key in timed), pdt.STATS
        assert set(pdt.STATS) == set(pdt.STATS_ZERO)
        print("ok  stats: " + ", ".join(f"{k}={pdt.STATS[k]:.2f}" for k in timed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
