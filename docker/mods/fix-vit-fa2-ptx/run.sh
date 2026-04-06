#!/bin/bash

# Fix: vLLM's ViT (vision encoder) attention uses FA2 C kernels compiled with
# CUDA 13.2 PTX. On hosts with CUDA 13.0 driver, these kernels crash with
# "unsupported PTX toolchain". This mod patches the ViT attention wrapper to
# use PyTorch SDPA instead of the FA2 C extension.

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
VIT_ATTN="$SITE_PACKAGES/vllm/v1/attention/ops/vit_attn_wrappers.py"

if [ ! -f "$VIT_ATTN" ]; then
    echo "  WARNING: $VIT_ATTN not found (vLLM version mismatch?), skipping"
    exit 0
fi

if grep -q 'SDPA fallback for SM110' "$VIT_ATTN"; then
    echo "  Already patched (skipping)"
    exit 0
fi

cp "$VIT_ATTN" "${VIT_ATTN}.bak"
python3 /workspace/mods/fix-vit-fa2-ptx/patch_vit_attn.py "$VIT_ATTN"
echo "  Patched $VIT_ATTN (FA2 → SDPA for ViT attention)"
