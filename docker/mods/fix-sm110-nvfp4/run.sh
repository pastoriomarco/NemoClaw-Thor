#!/bin/bash

# SM110 (Thor): Runtime patches for NVFP4 on Jetson AGX Thor.
#
# Problem: vLLM's _C.scaled_fp4_quant kernel was not compiled for SM110
# (Dockerfile strips 11.0f from all CUTLASS arch lists). This breaks the
# NVFP4 activation quantization path used by cutlass/cudnn GEMM backends.
#
# Solution: Use FlashInfer's nvfp4_quantize (which HAS SM110a JIT kernels)
# as a drop-in replacement for the missing _C kernel. FlashInfer can produce
# both 8x4 and 128x4 scale factor layouts, matching what each GEMM backend
# expects.
#
# Also patches is_device_capability_family(100) to match SM110, enabling
# FlashInfer MoE/CUTLASS NVFP4 kernel selection.

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
VLLM_PKG="$SITE_PACKAGES/vllm"

echo "Applying SM110 NVFP4 patches..."

# Patch 1: is_device_capability_family — treat SM110 as SM100 family.
# Without this, NVFP4 MoE kernels fall back to Marlin. vllm-project/vllm#33333.
IFACE="$VLLM_PKG/platforms/interface.py"
if grep -q 'return (current_capability.to_int() // 10) == (capability // 10)' "$IFACE"; then
    sed -i 's|return (current_capability.to_int() // 10) == (capability // 10)|cc = current_capability.to_int(); return (cc // 10) == (capability // 10) or (capability == 100 and cc // 10 == 11)|' "$IFACE"
    echo "  Patched $IFACE (SM110 -> SM100 family)"
else
    echo "  Already patched or method signature changed (skipping)"
fi

# Patch 2: scaled_fp4_quant — use FlashInfer 128x4 as fallback for missing _C kernel.
# The _C.scaled_fp4_quant kernel wasn't compiled for SM110. FlashInfer's
# nvfp4_quantize works on SM110a and supports both scale factor layouts.
# We wrap the _C call in try/except and fall back to FlashInfer on failure.
CUSTOM_OPS="$VLLM_PKG/_custom_ops.py"
if grep -q 'torch.ops._C.scaled_fp4_quant.out(' "$CUSTOM_OPS" && ! grep -q 'SM110: _C kernel not compiled' "$CUSTOM_OPS"; then
    python3 "$PWD/patch_scaled_fp4_quant.py" "$CUSTOM_OPS"
else
    echo "  scaled_fp4_quant already patched or not found (skipping)"
fi

# Patch 3: FlashInfer JIT compilation context — include SM110 for major 10 targets.
# FlashInfer's fused MoE JIT kernels request supported_major_versions=[10, 12].
# SM110 is major=11, which doesn't match. Patch get_nvcc_flags_list to emit SM110a
# when SM100 family (major 10) is requested and the device is SM110.
FLASHINFER_CTX="$SITE_PACKAGES/flashinfer/compilation_context.py"
if grep -q 'if major_minor_tuple\[0\] in supported_major_versions' "$FLASHINFER_CTX" && ! grep -q 'SM110 enterprise Blackwell' "$FLASHINFER_CTX"; then
    python3 "$PWD/patch_flashinfer_jit.py" "$FLASHINFER_CTX"
else
    echo "  FlashInfer JIT context already patched or not found (skipping)"
fi
