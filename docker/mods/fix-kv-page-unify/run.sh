#!/bin/bash

# Fix KV cache page size unification for hybrid models with DFlash drafter.
#
# Problem: Qwen3.5 is a hybrid model (full_attention + linear_attention/mamba).
# The mamba specs get page_size_padded set to match attention page size.
# When a DFlash drafter adds attention layers with different per-token sizes,
# unify_kv_cache_spec_page_size() tries to scale block_size to match the max
# page size. But for mamba specs, page_size_padded is a frozen dataclass field
# that doesn't change when block_size is replaced — causing an AssertionError.
#
# Fix: After adjusting block_size, also update page_size_padded if present.

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
KV_CACHE_UTILS="$SITE_PACKAGES/vllm/v1/core/kv_cache_utils.py"

echo "Applying KV cache page unification fix for DFlash + hybrid models..."

if [ ! -f "$KV_CACHE_UTILS" ]; then
    echo "  WARNING: kv_cache_utils.py not found at $KV_CACHE_UTILS (skipping)"
    exit 0
fi

python3 - "$KV_CACHE_UTILS" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

# The original code:
#   new_spec = replace(layer_spec, block_size=new_block_size)
#   assert new_spec.page_size_bytes == max_page_size
#
# Replace with code that also updates page_size_padded when present:
old = """            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size"""

new = """            new_spec = replace(layer_spec, block_size=new_block_size)
            # DFlash+hybrid fix: update page_size_padded if present
            if hasattr(new_spec, 'page_size_padded') and new_spec.page_size_padded is not None:
                if new_spec.page_size_bytes != max_page_size:
                    new_spec = replace(new_spec, page_size_padded=max_page_size)
            assert new_spec.page_size_bytes == max_page_size"""

if old in content:
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print("  Patched unify_kv_cache_spec_page_size for page_size_padded handling")
else:
    print("  WARNING: Pattern not found — code may have changed")

PYEOF

echo "  KV cache page unification fix complete."
