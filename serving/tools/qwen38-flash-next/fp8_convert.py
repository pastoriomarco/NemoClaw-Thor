#!/usr/bin/env python3
# Derived from Saren-Arterius/qwen3.8-Flash-DGX-AutoRound
# tools/fp8_convert.py (Apache-2.0), by @Saren-Arterius.
"""Convert selected Qwen3.8 Flash-Next BF16 side weights to block FP8.

The output uses the DeepSeek-compatible layout consumed by vLLM:
FP8-E4M3 ``weight`` plus FP32 ``weight_scale_inv`` in 128x128 blocks.
Only the listed dense GDN, QSA, and shared-expert matrices are changed.
"""

import json
import re
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file


ROOT = sys.argv[1] if len(sys.argv) > 1 else sys.exit(
    "usage: fp8_convert.py <checkpoint_dir>"
)
BLOCK = 128
FP8_MAX = 448.0

TARGETS = re.compile(
    r"model\.language_model\.layers\.\d+\.("
    r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.shared_expert\.(gate_proj|up_proj|down_proj)"
    r")\.weight$"
)


def block_quant(weight: torch.Tensor):
    out_features, in_features = weight.shape
    assert out_features % BLOCK == 0 and in_features % BLOCK == 0, weight.shape
    blocks = weight.float().reshape(
        out_features // BLOCK, BLOCK, in_features // BLOCK, BLOCK
    )
    absmax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    scale = absmax / FP8_MAX
    quantized = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    dequantized = quantized.float() * scale
    relative_error = (
        (dequantized - blocks).abs().amax() / blocks.abs().amax()
    ).item()
    return (
        quantized.reshape(out_features, in_features),
        scale.squeeze(1).squeeze(-1).contiguous(),
        relative_error,
    )


def main():
    index_path = f"{ROOT}/model.safetensors.index.json"
    with open(index_path, encoding="utf-8") as index_file:
        index = json.load(index_file)
    weight_map = index["weight_map"]
    targets = {name: filename for name, filename in weight_map.items() if TARGETS.search(name)}
    by_file = {}
    for name, filename in targets.items():
        by_file.setdefault(filename, []).append(name)

    print(f"{len(targets)} tensors across {len(by_file)} shards")
    if len(targets) != 300 or len(by_file) != 4:
        raise RuntimeError(
            "expected the pinned RadixArk layout: 300 tensors across 4 shards"
        )

    worst = 0.0
    for position, (filename, names) in enumerate(sorted(by_file.items()), 1):
        path = f"{ROOT}/{filename}"
        tensors = load_file(path)
        for name in names:
            weight = tensors.pop(name)
            quantized, scale, relative_error = block_quant(weight)
            worst = max(worst, relative_error)
            tensors[name] = quantized
            scale_name = name.replace(".weight", ".weight_scale_inv")
            tensors[scale_name] = scale
            weight_map[scale_name] = filename
        shutil.move(path, path + ".bf16.bak")
        save_file(tensors, path)
        print(f"[{position}/{len(by_file)}] {filename}: {len(names)} tensors")

    with open(index_path, "w", encoding="utf-8") as index_file:
        json.dump(index, index_file)
    print(f"done. worst per-tensor max rel err: {worst:.4f}")
    if worst >= 0.10:
        raise RuntimeError("FP8 roundtrip error unexpectedly large")


if __name__ == "__main__":
    sys.exit(main())
