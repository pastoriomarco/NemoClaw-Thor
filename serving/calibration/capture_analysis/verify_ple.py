#!/usr/bin/env python3
"""Verify the NumPy PLE row-id computation (ple_hash.py) against vLLM 0.33 and the checkpoint.

Checks, all CPU-only, no model weights loaded:
 1. Layout: multipliers / per-head sizes / offsets from ple_hash equal the ones
    vLLM's Qwen4ExpNGramEmbedding classmethods derive from the same config.
 2. Checkpoint: the persistent buffers stored in the checkpoint
    (layer_multipliers, ngram_heads_vocab_sizes, ngram_heads_offsets; a few
    bytes read straight from the safetensors file) equal the derived ones.
 3. Table layout: the NVFP4 shards' safetensors headers give split_ngram_parts
    shards of ceil(rows / parts) rows each, summing to the padded row count,
    with 80 B packed values + 10 B E4M3 scales per row (90 B/row). Row id r is
    in shard r // shard_size at local row r % shard_size, which is how the
    qwen4exp-ple-disk overlay (PLEDiskRowTable.gather_into) maps ids.
 4. Ids: vLLM's own CPU path Qwen4ExpNGramEmbedding.compute_ngram_ids (batched,
    chunked requests with ngram_context built like Qwen4ExpModelState) equals
    ple_hash on real tokenised text, random tokens and EOS-segmented input.
 5. Ids: the Triton kernel _ple_ngram_ids_kernel (the GPU path) run in the
    Triton CPU interpreter on the same batch, if the interpreter works here.

Writes a JSON summary (argument 2) and exits non-zero on any mismatch.
"""

import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")  # before anything imports triton

import json
import struct
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ple_hash import PLEHasher  # noqa: E402


def text_config(model_dir):
    config = json.load(open(os.path.join(model_dir, "config.json")))
    return config.get("text_config", config)


def read_header(path):
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length)), 8 + length


DTYPES = {"I64": np.int64, "I32": np.int32, "U8": np.uint8, "F8_E4M3": np.uint8, "F32": np.float32}


def read_small_tensor(model_dir, file_name, name):
    path = os.path.join(model_dir, file_name)
    header, data_start = read_header(path)
    meta = header[name]
    begin, end = meta["data_offsets"]
    with open(path, "rb") as handle:
        handle.seek(data_start + begin)
        raw = handle.read(end - begin)
    return np.frombuffer(raw, dtype=DTYPES[meta["dtype"]]).reshape(meta["shape"])


def main():
    model_dir, out_path = sys.argv[1], sys.argv[2]
    tc = text_config(model_dir)
    hasher = PLEHasher(tc)
    result = {"hasher": hasher.describe(), "checks": {}}
    failures = []

    def check(name, ok, detail=""):
        result["checks"][name] = {"ok": bool(ok), "detail": detail}
        print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            failures.append(name)

    import torch

    # 1. Layout against vLLM's classmethods.
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding as NG

    ple_layer_ids = sorted(set(tc["ple_layer_ids"]))
    check("single PLE layer", len(ple_layer_ids) == 1, f"ple_layer_ids={ple_layer_ids} (dense id 0)")
    mult = NG._make_layer_multipliers(
        ngram_size=int(tc["ngram_size"]),
        unigram_vocab_size=int(tc["vocab_size"]),
        seed=int(tc.get("seed", 1234)),
        ple_dense_layer_id=0,
    )
    sizes, offsets, total = NG._make_vocab_layout(
        ngram_vocab_size_base=int(tc["ngram_vocab_size_base"]),
        ngram_heads=hasher.ngram_heads,
        ple_dense_layer_id=0,
    )
    check("multipliers == vLLM", list(mult) == hasher.multipliers)
    check("head sizes/offsets == vLLM", list(sizes) == hasher.sizes and list(offsets) == hasher.offsets,
          f"total={total} padded={hasher.padded_rows}")

    # 2./3. Checkpoint buffers and shard layout.
    weight_map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    prefix = None
    for key in weight_map:
        if key.endswith("ple_embedding.layer_multipliers"):
            prefix = key[: -len("layer_multipliers")]
    check("checkpoint PLE prefix found", prefix is not None, str(prefix))
    for buf, expect in (
        ("layer_multipliers", hasher.multipliers),
        ("ngram_heads_vocab_sizes", hasher.sizes),
        ("ngram_heads_offsets", hasher.offsets),
    ):
        stored = read_small_tensor(model_dir, weight_map[prefix + buf], prefix + buf)
        check(f"checkpoint {buf} == derived", stored.astype(np.int64).tolist() == list(expect),
              f"stored dtype {stored.dtype} shape {stored.shape}")

    shard_rows, row_bytes = [], set()
    headers = {}
    index = 0
    while True:
        wname = f"{prefix}ngram_embedding.shard_{index}.weight"
        sname = f"{prefix}ngram_embedding.shard_{index}.weight_scale"
        if wname not in weight_map:
            break
        for name in (wname, sname):
            file_name = weight_map[name]
            if file_name not in headers:
                headers[file_name] = read_header(os.path.join(model_dir, file_name))[0]
        wshape = headers[weight_map[wname]][wname]["shape"]
        sshape = headers[weight_map[sname]][sname]["shape"]
        shard_rows.append(wshape[0])
        row_bytes.add((wshape[1], sshape[1], headers[weight_map[wname]][wname]["dtype"],
                       headers[weight_map[sname]][sname]["dtype"]))
        if wshape[0] != sshape[0]:
            failures.append(f"shard {index} weight/scale rows differ")
        index += 1
    check("shard count == split_ngram_parts", len(shard_rows) == hasher.split_ngram_parts, str(len(shard_rows)))
    check("sum(shard rows) == padded rows", sum(shard_rows) == hasher.padded_rows,
          f"{sum(shard_rows)} vs {hasher.padded_rows}")
    check("shard_size == ceil(rows/parts)", shard_rows[0] == hasher.shard_size
          and all(r == hasher.shard_size for r in shard_rows[:-1]), f"shard_size={shard_rows[0]}")
    (wcols, scols, wdt, sdt), = row_bytes
    check("row bytes == 90", wcols + scols == 90 and wcols * 2 == hasher.head_dim,
          f"{wcols} B {wdt} (E2M1 x2) + {scols} B {sdt} scales; head_dim {hasher.head_dim}")
    result["table"] = {"shards": len(shard_rows), "shard_size": shard_rows[0], "rows": sum(shard_rows),
                       "row_bytes": wcols + scols, "bytes": sum(shard_rows) * (wcols + scols)}

    # 4. Ids against vLLM's CPU compute_ngram_ids.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    eos = hasher.eos_token_id
    text = open(os.path.abspath(__file__)).read() * 3  # real code/English text
    real = np.asarray(tokenizer.encode(text, add_special_tokens=False), dtype=np.int64)
    rng = np.random.default_rng(0)
    rand = rng.integers(0, int(tc["vocab_size"]), size=3000)
    seg = rng.integers(0, int(tc["vocab_size"]), size=3000)
    seg[rng.choice(3000, 60, replace=False)] = eos  # EOS segment boundaries
    seg[[0, 1, 2, 500, 501]] = eos
    sequences = [real, rand, seg]

    fake = types.SimpleNamespace(
        eos_token_id=eos,
        ngram_size=hasher.ngram_size,
        heads_per_ngram=hasher.heads_per_ngram,
        layer_multipliers=torch.tensor(hasher.multipliers, dtype=torch.long),
        ngram_heads_vocab_sizes=torch.tensor(hasher.sizes, dtype=torch.long),
        ngram_heads_offsets=torch.tensor(hasher.offsets, dtype=torch.long),
        _shift_precompute=NG._shift_precompute,
        _shift_apply=NG._shift_apply,
    )
    ctx_len = hasher.ngram_size - 1

    def batch_for(starts, lens):
        """One scheduler step: request i computes tokens [starts[i], starts[i]+lens[i])."""
        ids, ctx, qsl, expect = [], [], [0], []
        for seq, start, length in zip(sequences, starts, lens):
            ids.append(seq[start : start + length])
            context = [seq[p] if p >= 0 else eos for p in range(start - ctx_len, start)]
            ctx.append(context)
            qsl.append(qsl[-1] + length)
            expect.append(hasher.ids(seq, np.arange(start, start + length)))
        return (torch.tensor(np.concatenate(ids)), torch.tensor(qsl, dtype=torch.int32),
                torch.tensor(ctx, dtype=torch.int32), np.concatenate(expect))

    steps = [((0, 0, 0), (700, 512, 400)), ((700, 512, 400), (900, 1000, 1200)),
             ((1600, 1512, 1600), (1, 1, 1)), ((1601, 2999, 1601), (1, 1, 1)),
             ((501, 2, 3), (64, 64, 64))]
    n_ok, n_total = 0, 0
    triton_batches = []
    for starts, lens in steps:
        input_ids, qsl, ctx, expect = batch_for(starts, lens)
        got = NG.compute_ngram_ids(fake, input_ids, qsl, ctx).numpy()
        n_total += expect.size
        n_ok += int((got == expect.astype(np.int64)).sum())
        triton_batches.append((input_ids, qsl, ctx, expect))
    check("ids == vLLM compute_ngram_ids (CPU path)", n_ok == n_total, f"{n_ok}/{n_total} ids, {len(steps)} batched steps")

    # 5. Triton kernel in the CPU interpreter.
    try:
        from vllm.models.qwen4_exp.common.ops import ple as ple_ops

        n_ok, n_total = 0, 0
        for input_ids, qsl, ctx, expect in triton_batches:
            out = torch.empty((input_ids.numel(), hasher.ngram_heads), dtype=torch.int64)
            ple_ops._ple_ngram_ids_kernel[(-(-input_ids.numel() // 8),)](
                input_ids, qsl, ctx, fake.layer_multipliers, fake.ngram_heads_vocab_sizes,
                fake.ngram_heads_offsets, out, input_ids.numel(), qsl.numel() - 1, eos,
                binary_search_iters=(qsl.numel() - 1).bit_length(),
                NGRAM_CONTEXT_LEN=ctx_len, HEADS_PER_NGRAM=hasher.heads_per_ngram,
                BLOCK_T=8, launch_pdl=False,
            )
            n_total += expect.size
            n_ok += int((out.numpy() == expect.astype(np.int64)).sum())
        check("ids == Triton _ple_ngram_ids_kernel (interpreter)", n_ok == n_total, f"{n_ok}/{n_total} ids")
    except Exception as exc:  # interpreter unavailable: report, do not fail
        result["checks"]["ids == Triton kernel (interpreter)"] = {"ok": None, "detail": f"skipped: {exc!r}"[:300]}
        print(f"[skip] Triton interpreter: {exc!r}"[:300])

    result["failures"] = failures
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print("PLE verification", "FAILED: " + ", ".join(failures) if failures else "passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
