#!/bin/bash
# Merge per-half w1/w3 NVFP4 global scales at load time.
#
# Background:
# vLLM's W4A4 NVFP4 MoE kernel accepts one global scale per expert for the
# fused [w3, w1] GEMM (shape `[E]`), but some quantization producers (older
# llm-compressor runs, some NVIDIA/ModelOpt exports, the `dervig` MiniMax
# NVFP4 checkpoints) ship per-half scales with shape `[E, 2]` — separate
# values for w1 (gate_proj) and w3 (up_proj).
#
# The stock vLLM code in compressed_tensors_moe_w4a4_nvfp4.py:196 silently
# discards w3's scale and keeps only w1's:
#
#   w13_weight_global_scale = layer.w13_weight_global_scale[:, 0].contiguous()
#
# On checkpoints where w3's scale differs from w1's, this produces numerical
# garbage — FP4 has ~6-bit dynamic range, so applying w1's scale to w3's
# tiles overflows/underflows → NaN/Inf → null bytes in the output.
#
# Fix:
# llm-compressor itself (update_fused_layer_weight_global_scales in
# helpers.py) uses `min()` of the two scales when fusing. Replace the
# [:, 0] hack with the same torch.minimum() convention. 99.6% of expert
# pairs already match (per HF discussions on lukealonso/MiniMax-M2.5-NVFP4);
# for the remaining 0.4% the accuracy delta is sub-1% on GSM8K/MMLU.
#
# Affected models (non-exhaustive):
#   dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B-NVFP4-GB10
#   nvidia/Qwen3-30B-A3B-NVFP4 (older export)
#   any checkpoint where config_groups.*.weights specifies per-half scales
#
# Upstream status: this will NOT be fixed in vLLM — they consider split
# [E, 2] scales a producer bug. llm-compressor PR #1508 (2025-06-03) and
# later PRs fixed the producer side; older checkpoints are not being
# re-quantized.
#
# References:
# - https://github.com/vllm-project/vllm/issues/36331 (identical warning,
#   user fixed by switching to a correctly-quantized checkpoint)
# - https://github.com/vllm-project/llm-compressor src/llmcompressor/
#   modifiers/utils/helpers.py:update_fused_layer_weight_global_scales
# - https://huggingface.co/lukealonso/MiniMax-M2.5-NVFP4/discussions/2
#   (author's analysis: re-quantize with max(), 99.6% pairs already match)

set -euo pipefail

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
TARGET="${SITE_PACKAGES}/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py"

echo "Merging per-half w1/w3 NVFP4 global scales with torch.minimum()..."

if [ ! -f "$TARGET" ]; then
    echo "  WARNING: $TARGET not found (skipping)"
    exit 0
fi

python3 - "$TARGET" <<'PYEOF'
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text(encoding="utf-8")

OLD = "        w13_weight_global_scale = layer.w13_weight_global_scale[:, 0].contiguous()\n"
NEW = (
    "        # nemoclaw-thor mod (fix-nvfp4-moe-scale-merge): use min(w1, w3) "
    "instead of\n"
    "        # just w1's scale — matches llm-compressor's "
    "update_fused_layer_weight_global_scales\n"
    "        # convention and prevents FP4 overflow when w1 and w3 scales "
    "differ.\n"
    "        w13_weight_global_scale = torch.minimum(\n"
    "            layer.w13_weight_global_scale[:, 0],\n"
    "            layer.w13_weight_global_scale[:, 1],\n"
    "        ).contiguous()\n"
)

if "nemoclaw-thor mod (fix-nvfp4-moe-scale-merge)" in src:
    print("  already patched — skipping")
    sys.exit(0)

if OLD not in src:
    print(f"  WARNING: pattern not found in {path}")
    print("  vLLM version may have drifted — check the target line manually")
    sys.exit(1)

if src.count(OLD) > 1:
    print("  WARNING: pattern matched more than once — refusing ambiguous patch")
    sys.exit(1)

new_src = src.replace(OLD, NEW, 1)
path.write_text(new_src, encoding="utf-8")
print(f"  patched {path}")
print(f"  w13_weight_global_scale now uses torch.minimum(w1, w3)")
PYEOF

echo "Done."
