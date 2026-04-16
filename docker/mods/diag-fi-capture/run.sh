#!/bin/bash
# Capture the exact FlashInfer wrapper .run() inputs for the drafter
# and replay with our standalone test to find the discrepancy.
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
FI="$SITE_PACKAGES/vllm/v1/attention/backends/flashinfer.py"

echo "Applying FlashInfer .run() capture diagnostic..."

python3 - "$FI" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f: c = f.read()

# Capture the prefill wrapper .run() call for non-causal (DFlash drafter)
# The non-causal prefill path is:
#   prefill_wrapper.run(prefill_query, kv_cache_permute, k_scale=..., v_scale=..., out=...)
# We want to capture inputs and output, then compute SDPA reference and compare.

old = """                    prefill_wrapper.run(
                        prefill_query,
                        kv_cache_permute,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=output[num_decode_tokens:],"""

new = """                    # CAPTURE: log FlashInfer prefill .run() inputs for non-causal
                    import sys as _csys
                    _cap_n = getattr(type(self), '_fi_cap_n', 0)
                    if _cap_n < 3 and not getattr(attn_metadata, 'causal', True) == False:
                        pass  # causal, skip
                    if _cap_n < 3 and getattr(attn_metadata, 'causal', True) == False:
                        type(self)._fi_cap_n = _cap_n + 1
                        p = lambda *a: print(*a, file=_csys.stderr, flush=True)
                        p(f"\\nFI-CAPTURE step={_cap_n}")
                        p(f"  prefill_query: {prefill_query.shape} {prefill_query.dtype}")
                        p(f"  kv_cache_permute: {kv_cache_permute.shape} strides={kv_cache_permute.stride()}")
                        p(f"  k_scale={layer._k_scale_float} v_scale={layer._v_scale_float}")
                        p(f"  output slice: [{num_decode_tokens}:]")
                        p(f"  wrapper type: {type(prefill_wrapper).__name__}")
                        p(f"  wrapper _causal: {getattr(prefill_wrapper, '_causal', 'N/A')}")
                        # Check wrapper context attributes
                        ctx = getattr(prefill_wrapper, '_context', None)
                        if ctx:
                            p(f"  wrapper._context._causal: {getattr(ctx, '_causal', 'N/A')}")
                            p(f"  wrapper._context._sm_scale: {getattr(ctx, '_sm_scale', 'N/A')}")
                        # Run FlashInfer
                        _out_before = output[num_decode_tokens:].clone()
                    prefill_wrapper.run(
                        prefill_query,
                        kv_cache_permute,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=output[num_decode_tokens:],"""

if old in c and "FI-CAPTURE" not in c:
    c = c.replace(old, new, 1)

    # Add post-run comparison
    old2 = """                    prefill_wrapper.run(
                        prefill_query,
                        kv_cache_permute,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=output[num_decode_tokens:],
                    )"""
    # Find the SECOND occurrence (the one we just added)
    # Actually, after our replacement, the .run() call is the only one
    # Add logging after the run
    new2 = """                    prefill_wrapper.run(
                        prefill_query,
                        kv_cache_permute,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=output[num_decode_tokens:],
                    )
                    if _cap_n < 3 and getattr(attn_metadata, 'causal', True) == False:
                        _out_after = output[num_decode_tokens:]
                        p(f"  FI output norm: {_out_after.float().norm():.6f}")
                        p(f"  FI output zero?: {(_out_after.float().abs().max() < 1e-6)}")
                        p(f"  FI output[:4]: {_out_after[0, :4].tolist()}")"""
    c = c.replace(old2, new2, 1)

    with open(path, "w") as f: f.write(c)
    print("  [1/1] Added FlashInfer .run() capture diagnostic")
else:
    print("  [1/1] SKIP (pattern not found or already patched)")

PYEOF
echo "  FlashInfer capture diagnostic complete."
