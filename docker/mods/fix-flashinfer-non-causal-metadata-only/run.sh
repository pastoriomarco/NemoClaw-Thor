#!/bin/bash
# Minimal non-causal metadata plumbing WITHOUT changing the actual FlashInfer kernel.
# This lets DFlash's causal=False assertion pass, but FlashInfer still runs causal kernels.
# Purpose: isolate whether the Xid 43 crash is from the non-causal kernel JIT.

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
FLASHINFER_PY="$SITE_PACKAGES/vllm/v1/attention/backends/flashinfer.py"

echo "Applying minimal FlashInfer non-causal metadata patches..."

python3 - "$FLASHINFER_PY" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

patched = False

# Patch 1: supports_non_causal
marker = "    forward_includes_kv_cache_update: bool = False"
if marker in content and "supports_non_causal" not in content:
    content = content.replace(marker, """    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    forward_includes_kv_cache_update: bool = False""")
    print("  [1] supports_non_causal")
    patched = True

# Patch 2: causal field on metadata
marker2 = "    cascade_wrapper: MultiLevelCascadeAttentionWrapper | None"
if marker2 in content and "\n    causal: bool" not in content:
    content = content.replace(marker2, marker2 + "\n\n    causal: bool = True")
    print("  [2] causal field")
    patched = True

# Patch 3: propagate causal to metadata object
marker3 = "            cascade_wrapper=None,\n        )"
if marker3 in content and "attn_metadata.causal" not in content:
    content = content.replace(marker3, marker3 + "\n        attn_metadata.causal = common_attn_metadata.causal")
    print("  [3] causal propagation")
    patched = True

# Patch 6a: remove prefill _causal assertion
old_assert = "                    assert prefill_wrapper._causal\n                    prefill_wrapper.run("
new_assert = "                    prefill_wrapper.run("
if old_assert in content:
    content = content.replace(old_assert, new_assert, 1)
    print("  [6a] removed assertion")
    patched = True

# Patch 6b: remove DCP _causal assertion
old_dcp = "                    assert prefill_wrapper._new_tokens._causal"
if old_dcp in content:
    content = content.replace(old_dcp, "                    pass  # assertion removed", 1)
    print("  [6b] removed DCP assertion")
    patched = True

# NOTE: patches 4, 4b, 5, 7, 8 are INTENTIONALLY OMITTED.
# plan() still called with causal=True -- FlashInfer runs causal kernels.

if patched:
    with open(path, "w") as f:
        f.write(content)
    print("  Metadata-only patches applied (causal kernels preserved)")
else:
    print("  WARNING: no patches applied")
PYEOF

echo "  Done."
