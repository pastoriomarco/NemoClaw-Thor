#!/bin/bash
# Force FlashInfer prefill backend to "fa2" instead of "auto".
#
# flashinfer#1709: BatchPrefillWithPagedKVCacheWrapper.plan() produces wrong
# output for causal=False when trtllm-gen is internally selected by "auto".
# DFlash uses causal=False. Forcing "fa2" avoids the trtllm-gen code path.

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
FI="$SITE_PACKAGES/vllm/v1/attention/backends/flashinfer.py"

echo "Applying FlashInfer backend=fa2 fix (flashinfer#1709 workaround)..."

python3 - "$FI" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f:
    c = f.read()

# Change the wrapper creation in _get_prefill_wrapper to use backend="fa2"
old = """                self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self._get_workspace_buffer(), get_kv_cache_layout()
                )"""

new = """                self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self._get_workspace_buffer(), get_kv_cache_layout(),
                    backend="fa2",  # flashinfer#1709: auto may select trtllm-gen which breaks causal=False
                )"""

if old in c and 'backend="fa2"' not in c:
    c = c.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(c)
    print("  [1/1] Forced backend=fa2 in _get_prefill_wrapper")
else:
    print("  [1/1] SKIP (already patched or pattern not found)")

PYEOF
echo "  FlashInfer backend=fa2 fix complete."
