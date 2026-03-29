#!/usr/bin/env python3
"""Patch vllm/_custom_ops.py: wrap _C.scaled_fp4_quant.out with FlashInfer fallback."""
import sys

path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

old = """\
    else:
        # Pre-allocate and call .out variant (same behavior as old in-place API)
        output, output_scale = create_fp4_output_tensors(
            m, n, input.device, is_sf_swizzled_layout
        )
        torch.ops._C.scaled_fp4_quant.out(
            input,
            input_global_scale,
            is_sf_swizzled_layout,
            output=output,
            output_scale=output_scale,
        )"""

new = """\
    else:
        # Pre-allocate and call .out variant (same behavior as old in-place API)
        output, output_scale = create_fp4_output_tensors(
            m, n, input.device, is_sf_swizzled_layout
        )
        try:
            torch.ops._C.scaled_fp4_quant.out(
                input,
                input_global_scale,
                is_sf_swizzled_layout,
                output=output,
                output_scale=output_scale,
            )
        except (NotImplementedError, RuntimeError):
            # SM110: _C kernel not compiled. Fall back to FlashInfer 128x4 quant.
            from flashinfer import SfLayout
            from flashinfer import nvfp4_quantize as _fi_nvfp4_quantize
            output, output_scale = _fi_nvfp4_quantize(
                input, input_global_scale,
                sfLayout=SfLayout.layout_128x4, do_shuffle=False,
            )"""

if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print(f"  Patched {path} (FlashInfer 128x4 fallback for scaled_fp4_quant)")
else:
    if "torch.ops._C.scaled_fp4_quant.out(" in content and "SM110: _C kernel not compiled" not in content:
        print(f"  WARNING: Could not match exact code in {path} — manual fix needed")
        sys.exit(1)
    else:
        print(f"  Already patched (skipping)")
