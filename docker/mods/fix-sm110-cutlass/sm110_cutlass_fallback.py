"""
SM110 (Thor) CUTLASS fallback: Replace cutlass_scaled_mm with PyTorch native.

CUTLASS sm100 kernels use instructions incompatible with SM110. This module
overrides the torch C++ op implementation to use torch._scaled_mm instead,
with a dequantize-multiply fallback for unsupported configurations (e.g.
block-wise scales, INT8).

Must be imported AFTER vllm._C is loaded.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_FP8_DTYPES = frozenset({
    torch.float8_e4m3fn, torch.float8_e4m3fnuz,
    torch.float8_e5m2, torch.float8_e5m2fnuz,
})


def _group_broadcast(scale, shape):
    """Broadcast scale using vLLM's group broadcast convention.

    For block-wise scales like (M, K//128) applied to (M, K), each scale
    element covers a group of consecutive elements along the last dim.
    """
    if scale.numel() == 1:
        return scale
    if scale.shape == shape:
        return scale
    # Per-row (M,1) or per-col (1,N) — only if it broadcasts to target shape
    if scale.dim() == 2 and len(shape) == 2:
        if scale.shape[0] in (1, shape[0]) and scale.shape[1] in (1, shape[1]):
            return scale
    if scale.dim() == 2 and len(shape) == 2:
        s0, s1 = scale.shape
        t0, t1 = shape
        result = scale
        if s0 < t0 and t0 % s0 == 0:
            result = result.repeat_interleave(t0 // s0, dim=0)
        if s1 < t1 and t1 % s1 == 0:
            result = result.repeat_interleave(t1 // s1, dim=1)
        return result[:t0, :t1]
    return scale


def _dequant_mm(out, a, b, a_scales, b_scales, bias):
    """Dequantize-then-multiply fallback for any dtype/scale config."""
    a_f = a.to(torch.float32)
    b_f = b.to(torch.float32)
    a_f = a_f * _group_broadcast(a_scales, a.shape)
    b_f = b_f * _group_broadcast(b_scales, b.shape)
    result = torch.mm(a_f, b_f)
    if bias is not None:
        result = result + bias.to(torch.float32)
    out.copy_(result.to(out.dtype))


def _native_scaled_mm(out, a, b, a_scales, b_scales, bias=None):
    """Drop-in replacement for cutlass_scaled_mm on SM110."""
    # Try torch._scaled_mm for FP8 inputs — it handles per-tensor, per-row,
    # and per-channel (RowWise) scales natively via cuBLAS.
    if a.dtype in _FP8_DTYPES:
        if b.stride(0) != 1:
            b = b.t().contiguous().t()
        try:
            result = torch._scaled_mm(
                a, b,
                scale_a=a_scales,
                scale_b=b_scales,
                bias=bias,
                out_dtype=out.dtype,
            )
            if isinstance(result, tuple):
                result = result[0]
            out.copy_(result)
            return
        except RuntimeError:
            pass  # Fall through to dequantize path

    # Fallback: dequantize and do regular matmul (slow but correct)
    _dequant_mm(out, a, b, a_scales, b_scales, bias)


def patch():
    """Override cutlass_scaled_mm with PyTorch native fallback for SM110."""
    try:
        _ = torch.ops._C.cutlass_scaled_mm.default
    except AttributeError:
        logger.warning("cutlass_scaled_mm op not found, skipping SM110 patch")
        return

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Overriding a previously registered kernel.*")
        try:
            torch.library.impl(
                "_C::cutlass_scaled_mm", "CUDA", _native_scaled_mm
            )
            logger.info(
                "SM110: Overrode cutlass_scaled_mm CUDA impl with torch._scaled_mm"
            )
        except Exception as e:
            logger.error("SM110: Failed to override cutlass_scaled_mm: %s", e)
