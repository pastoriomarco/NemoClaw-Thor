#!/bin/bash

# FlashInfer non-causal attention support for DFlash speculative decoding.
#
# Problem: DFlash requires causal=False for its verification pass, but only
# FlashAttentionBackend advertises supports_non_causal(). FlashInfer's kernels
# natively support non-causal attention (BatchPrefillWithPagedKVCacheWrapper
# accepts causal=True/False), but the vLLM FlashInferBackend:
#   1. Does not override supports_non_causal() (returns False)
#   2. Hardcodes causal=True in prefill .plan() calls
#   3. FlashInferMetadata has no causal field (DFlash assertion fails)
#
# Why not use flash_attn (FA4) instead?  FA4 can't handle head_dim=256 on
# SM110 due to TMEM hardware limits and falls back to FA2, which crashes
# (no PTX compiled for SM110). All Qwen3.5 DeltaNet models have full_attention
# layers with head_dim=256. FlashInfer handles head_dim=256 on SM110 fine.
#
# Solution: Patch FlashInferBackend to support non-causal attention by:
#   1. Adding supports_non_causal() classmethod returning True
#   2. Adding causal field to FlashInferMetadata dataclass
#   3. Propagating common_attn_metadata.causal through the build/plan chain
#   4-5. Patching prefill/cascade/DCP .plan() calls from causal=True to dynamic
#   6. Removing hardcoded _causal assertions in forward path
#   7. Bypassing TRTLLM prefill when non-causal (TRTLLM has no causal param)
#   8. Forcing prefill path for non-causal by overriding reorder_batch_threshold
#      (without this, SM110's elevated threshold routes DFlash through decode path)

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
FLASHINFER_PY="$SITE_PACKAGES/vllm/v1/attention/backends/flashinfer.py"

echo "Applying FlashInfer non-causal attention patch for DFlash..."

if [ ! -f "$FLASHINFER_PY" ]; then
    echo "  WARNING: flashinfer.py not found at $FLASHINFER_PY (skipping)"
    exit 0
fi

python3 - "$FLASHINFER_PY" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

patched = False

# --- Patch 1: FlashInferBackend.supports_non_causal() ---
# Add classmethod after supports_sink() or supports_compute_capability()
marker = "    forward_includes_kv_cache_update: bool = False"
if marker in content and "supports_non_causal" not in content:
    replacement = """    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    forward_includes_kv_cache_update: bool = False"""
    content = content.replace(marker, replacement)
    print("  [1/5] Added FlashInferBackend.supports_non_causal()")
    patched = True
else:
    print("  [1/5] supports_non_causal already present or marker not found (skipping)")

# --- Patch 2: FlashInferMetadata.causal field ---
# Add causal field to the dataclass
marker2 = "    cascade_wrapper: MultiLevelCascadeAttentionWrapper | None"
if marker2 in content and "\n    causal: bool" not in content:
    content = content.replace(
        marker2,
        marker2 + "\n\n    causal: bool = True"
    )
    print("  [2/5] Added causal field to FlashInferMetadata")
    patched = True
else:
    print("  [2/5] causal field already present or marker not found (skipping)")

# --- Patch 3: FlashInferMetadataBuilder.build() — set causal on metadata ---
# After FlashInferMetadata(...) construction, set causal from common_attn_metadata
marker3 = "            cascade_wrapper=None,\n        )"
if marker3 in content and "attn_metadata.causal" not in content:
    content = content.replace(
        marker3,
        marker3 + "\n        attn_metadata.causal = common_attn_metadata.causal"
    )
    print("  [3/5] Set attn_metadata.causal from common_attn_metadata")
    patched = True
else:
    print("  [3/5] causal propagation already present or marker not found (skipping)")

# --- Patch 4: Prefill pathway .plan() — use causal from metadata ---
# Change hardcoded causal=True to use common_attn_metadata.causal in the
# BatchPrefillWithPagedKVCacheWrapper.plan() call (non-DCP path)
# There are two occurrences of causal=True in prefill plan calls:
#   1. New tokens in BatchDCPPrefillWrapper.plan (line ~265)
#   2. Regular prefill BatchPrefillWithPagedKVCacheWrapper.plan (line ~1117)
# We also need cascade (line ~1032)
#
# Strategy: Replace specific hardcoded causal=True patterns in .plan() calls.
# The tricky part is being precise enough to only change the right ones.

# 4a: Regular prefill wrapper — the plan call inside the `else` branch of
# `if self.use_dcp:`. This is the most common path for DFlash.
# The pattern is unique: it's the only .plan() call with head_dim_qk + causal=True
old_prefill_plan = "                    head_dim_qk=self.head_dim,\n                        page_size=self.page_size,\n                        causal=True,"
new_prefill_plan = "                    head_dim_qk=self.head_dim,\n                        page_size=self.page_size,\n                        causal=common_attn_metadata.causal,"
if old_prefill_plan in content:
    content = content.replace(old_prefill_plan, new_prefill_plan, 1)
    print("  [4/5] Patched prefill .plan() causal=True -> common_attn_metadata.causal")
    patched = True
else:
    # Try a simpler pattern match — the indentation might differ
    # Look for the prefill_wrapper.plan block with causal=True after head_dim_qk
    import re
    pattern = r"(head_dim_qk=self\.head_dim,\s+page_size=self\.page_size,\s+)causal=True,"
    match = re.search(pattern, content)
    if match:
        content = content[:match.start()] + match.group(1) + "causal=common_attn_metadata.causal," + content[match.end():]
        print("  [4/5] Patched prefill .plan() via regex")
        patched = True
    else:
        print("  [4/5] WARNING: Could not find prefill plan causal=True pattern")

# 4b: Cascade attention — causal=True in cascade_wrapper.plan
old_cascade = "                causal=True,\n                sm_scale=self.sm_scale,"
if old_cascade in content:
    content = content.replace(old_cascade, "                causal=common_attn_metadata.causal,\n                sm_scale=self.sm_scale,", 1)
    print("  [4b]  Patched cascade .plan() causal")
    patched = True

# --- Patch 5: BatchDCPPrefillWrapper._new_tokens.plan() — causal for DCP path ---
# The new_tokens sub-wrapper hardcodes causal=True. For DFlash, this should
# also respect the common metadata. We add a causal parameter to the plan method.
old_dcp_new_tokens = "            causal=True,  # This is newtokens run"
if old_dcp_new_tokens in content:
    # First, add causal parameter to BatchDCPPrefillWrapper.plan signature
    old_dcp_sig = "        disable_split_kv: bool,\n    ):"
    new_dcp_sig = "        disable_split_kv: bool,\n        causal: bool = True,\n    ):"
    if old_dcp_sig in content:
        content = content.replace(old_dcp_sig, new_dcp_sig, 1)
    content = content.replace(old_dcp_new_tokens, "            causal=causal,  # This is newtokens run (DFlash: False)", 1)
    # Also update the caller in build() to pass causal
    old_dcp_call_end = "                        disable_split_kv=self.disable_split_kv,\n                    )\n                else:"
    new_dcp_call_end = "                        disable_split_kv=self.disable_split_kv,\n                        causal=common_attn_metadata.causal,\n                    )\n                else:"
    if old_dcp_call_end in content:
        content = content.replace(old_dcp_call_end, new_dcp_call_end, 1)
    print("  [5/5] Patched DCP new_tokens causal")
    patched = True
else:
    print("  [5/5] DCP new_tokens pattern not found (skipping)")

# --- Patch 6: Forward-path causal assertions ---
# The forward() method in FlashInferImpl has hardcoded assertions that check
# prefill_wrapper._causal is True. With DFlash, _causal is False after .plan().
# These are consistency checks — the kernel uses whatever was planned. Remove
# the hard assertions and replace with ones that match the metadata's causal value.

# 6a: Regular prefill forward assertion: "assert prefill_wrapper._causal"
old_fwd_assert = "                    assert prefill_wrapper._causal\n                    prefill_wrapper.run("
new_fwd_assert = "                    # Non-causal is valid for DFlash verification\n                    prefill_wrapper.run("
if old_fwd_assert in content:
    content = content.replace(old_fwd_assert, new_fwd_assert, 1)
    print("  [6a]  Removed prefill forward _causal assertion")
    patched = True
else:
    print("  [6a]  Prefill forward assertion not found (skipping)")

# 6b: DCP new_tokens forward assertion: "assert prefill_wrapper._new_tokens._causal"
old_dcp_fwd_assert = "                    assert prefill_wrapper._new_tokens._causal"
if old_dcp_fwd_assert in content:
    content = content.replace(old_dcp_fwd_assert, "                    # Non-causal is valid for DFlash verification", 1)
    print("  [6b]  Removed DCP new_tokens forward _causal assertion")
    patched = True
else:
    print("  [6b]  DCP new_tokens forward assertion not found (skipping)")

# --- Patch 7: Force FlashInfer native prefill when non-causal (bypass TRTLLM) ---
# TRTLLM attention (trtllm_batch_context_with_kv_cache) has NO causal parameter —
# it always runs causal attention. On SM110/Blackwell, TRTLLM is auto-detected for
# prefill when kv_cache_dtype=="auto" (BF16). DFlash needs causal=False for its
# verification pass. If TRTLLM handles DFlash's prefill, non-causal is silently
# ignored → 0% acceptance. This patch forces FlashInfer native path when causal=False.
old_trtllm_gate = "            has_spec=uses_spec_reorder,\n        )\n        decode_use_trtllm = ("
new_trtllm_gate = """            has_spec=uses_spec_reorder,
        )
        # DFlash non-causal: force FlashInfer native (TRTLLM lacks causal param)
        if not common_attn_metadata.causal:
            prefill_use_trtllm = False
        decode_use_trtllm = ("""
if old_trtllm_gate in content:
    content = content.replace(old_trtllm_gate, new_trtllm_gate, 1)
    print("  [7/7] Forced FlashInfer native prefill for non-causal attention")
    patched = True
else:
    print("  [7/7] WARNING: TRTLLM gate pattern not found")

# --- Patch 8: Force prefill path for non-causal (reorder_batch_threshold) ---
# On SM110/Blackwell with TRTLLM, reorder_batch_threshold is raised to
# 1 + 2*num_speculative_tokens (e.g. 17 for 8 spec tokens). This causes
# split_decodes_and_prefills to classify DFlash drafting requests (query_len=9)
# as DECODES. The decode path never applies causal=False — it hardcodes causal
# attention. Fix: when causal=False, override the threshold so multi-token
# queries are classified as prefills (where our non-causal patches apply).
old_split = "        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (\n            split_decodes_and_prefills(\n                common_attn_metadata,\n                decode_threshold=self.reorder_batch_threshold,"
new_split = """        # DFlash: force multi-token queries through prefill when non-causal
        _effective_threshold = self.reorder_batch_threshold
        if not common_attn_metadata.causal:
            _effective_threshold = 1
            import logging as _lg8
            _nc8 = _lg8.getLogger('dflash.noncausal')
            _nc8.setLevel(_lg8.INFO)
            if not _nc8.handlers:
                _nc8.addHandler(_lg8.StreamHandler())
            if not hasattr(self, '_nc8_count'):
                self._nc8_count = 0
            self._nc8_count += 1
            if self._nc8_count <= 3:
                _nc8.info(f'PATCH8: causal=False, threshold {self.reorder_batch_threshold}->1, max_qlen={common_attn_metadata.max_query_len}, reqs={common_attn_metadata.num_reqs}')
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=_effective_threshold,"""
if old_split in content:
    content = content.replace(old_split, new_split, 1)
    print("  [8/8] Forced prefill path for non-causal (override decode threshold)")
    patched = True
else:
    print("  [8/8] WARNING: split_decodes_and_prefills pattern not found")

if patched:
    with open(path, "w") as f:
        f.write(content)
    print("  FlashInfer non-causal patches applied successfully.")
else:
    print("  WARNING: No patches applied — patterns may have changed in this vLLM version.")

PYEOF

echo "  FlashInfer non-causal mod complete."
