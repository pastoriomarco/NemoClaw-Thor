#!/bin/bash

# SM110 (Thor): Patch cutlass_scaled_mm torch op to use PyTorch native FP8.
# The CUTLASS sm100 kernels are incompatible with SM110 and cause illegal
# instruction errors. This mod overrides the cutlass_scaled_mm CUDA dispatch
# with a Python fallback using torch._scaled_mm.

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
VLLM_PKG="$SITE_PACKAGES/vllm"

echo "Installing SM110 CUTLASS fallback..."
cp sm110_cutlass_fallback.py "$VLLM_PKG/_sm110_cutlass_fallback.py"
echo "  Copied to $VLLM_PKG/_sm110_cutlass_fallback.py"

# Append patch invocation to _custom_ops.py (runs after vllm._C is loaded)
if ! grep -q '_sm110_cutlass_fallback' "$VLLM_PKG/_custom_ops.py"; then
    cat >> "$VLLM_PKG/_custom_ops.py" <<'PATCH'

# SM110 (Thor): Override cutlass_scaled_mm with PyTorch native fallback
try:
    from vllm._sm110_cutlass_fallback import patch as _sm110_patch
    _sm110_patch()
    del _sm110_patch
except Exception as _e:
    import logging as _logging
    _logging.getLogger(__name__).warning("SM110 fallback patch failed: %s", _e)
PATCH
    echo "  Patched $VLLM_PKG/_custom_ops.py"
else
    echo "  Already patched (skipping)"
fi
