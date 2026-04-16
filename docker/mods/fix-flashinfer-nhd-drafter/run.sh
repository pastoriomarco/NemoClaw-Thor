#!/bin/bash
# Fix FlashInfer paged KV layout for DFlash drafter on SM110.
#
# Root cause: vLLM permutes kv_cache to HND layout for SM110, but FlashInfer's
# BatchPrefillWithPagedKVCacheWrapper produces wrong results with HND-permuted
# views. NHD (no permute) works correctly for non-causal prefill.
#
# Fix: Skip the HND permutation for drafter layers that use non-causal prefill
# (DFlash). The kv_cache is passed in its original NHD layout.
#
# This is safe because:
# - FlashInfer non-causal prefill works correctly with NHD (verified standalone)
# - The cache write (reshape_and_cache_flash) uses NHD
# - Only affects the drafter's attention, not the target's

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
FI="$SITE_PACKAGES/vllm/v1/attention/backends/flashinfer.py"

echo "Applying FlashInfer NHD layout fix for DFlash drafter..."

python3 - "$FI" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f:
    c = f.read()

# In FlashInferImpl.forward, the kv_cache is permuted to HND:
#   stride_order = FlashInferBackend.get_kv_cache_stride_order()
#   kv_cache_permute = kv_cache.permute(*stride_order)
#
# For non-causal prefill (DFlash), skip this permutation — use NHD directly.
# We detect non-causal by checking attn_metadata.prefill causal flag.

old = """        stride_order = FlashInferBackend.get_kv_cache_stride_order()
        kv_cache_permute = kv_cache.permute(*stride_order)"""

new = """        stride_order = FlashInferBackend.get_kv_cache_stride_order()
        # Fix: Skip HND permutation for non-causal prefill (DFlash drafter).
        # FlashInfer's prefill kernel produces wrong results with HND-permuted
        # views on SM110, but works correctly with NHD (original layout).
        _has_prefill = hasattr(attn_metadata, 'prefill') and attn_metadata.prefill is not None
        _has_causal = hasattr(attn_metadata, 'causal')
        _is_noncausal = _has_prefill and _has_causal and not attn_metadata.causal
        import sys as _sys
        _nhd_count = getattr(FlashInferImpl, '_nhd_log_count', 0)
        if _nhd_count < 20:
            FlashInferImpl._nhd_log_count = _nhd_count + 1
            print(f"NHD-FIX: prefill={_has_prefill} causal_attr={_has_causal} "
                  f"causal_val={getattr(attn_metadata, 'causal', 'N/A')} "
                  f"noncausal={_is_noncausal} stride={stride_order} "
                  f"kv_shape={kv_cache.shape} "
                  f"num_prefill={getattr(attn_metadata, 'num_prefill_tokens', 'N/A')} "
                  f"num_decode={getattr(attn_metadata, 'num_decode_tokens', 'N/A')}",
                  file=_sys.stderr, flush=True)
        if not _is_noncausal and stride_order != (0, 1, 2, 3, 4):
            kv_cache_permute = kv_cache.permute(*stride_order)
        else:
            kv_cache_permute = kv_cache"""

if old in c:
    c = c.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(c)
    print("  [1/1] Patched FlashInfer to use NHD for non-causal prefill")
else:
    print("  [1/1] WARNING: pattern not found")

PYEOF

echo "  FlashInfer NHD layout fix complete."
