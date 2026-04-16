#!/bin/bash
# Fix stale FlashInfer wrapper state for DFlash drafter.
#
# Bug: FlashInfer's BatchPrefillWithPagedKVCacheWrapper produces identical
# output across different drafter steps. The wrapper's internal GPU buffers
# don't refresh when .plan() is called with slightly different parameters.
#
# Fix: Create a FRESH wrapper instance for each build_for_drafting call.
# This guarantees clean internal state. Then pass it through to the metadata.

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
FI="$SITE_PACKAGES/vllm/v1/attention/backends/flashinfer.py"

echo "Applying fresh-wrapper fix for DFlash drafter..."

python3 - "$FI" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f:
    c = f.read()

if "_patched_build_for_drafting" in c:
    old_patch = """def _patched_build_for_drafting(self, common_attn_metadata, draft_index):
    \"\"\"Force prefill path + fresh KV metadata for drafter.\"\"\"
    original_threshold = self.reorder_batch_threshold
    # PR #36060: force prefill path
    if not getattr(self, 'use_trtllm_decode_attention', True):
        self.reorder_batch_threshold = 0
    # PR #39546: force fresh KV metadata (prevent stale paged_kv_indptr)
    original_trtllm_decode = getattr(self, 'use_trtllm_decode_attention', False)
    self.use_trtllm_decode_attention = False
    try:
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=False,  # Force full recomputation to prevent stale buffers
        )
    finally:
        self.reorder_batch_threshold = original_threshold
        self.use_trtllm_decode_attention = original_trtllm_decode"""

    new_patch = """def _patched_build_for_drafting(self, common_attn_metadata, draft_index):
    \"\"\"Force prefill + fresh wrapper for drafter (fix stale state).\"\"\"
    import sys as _s
    original_threshold = self.reorder_batch_threshold
    if not getattr(self, 'use_trtllm_decode_attention', True):
        self.reorder_batch_threshold = 0
    original_trtllm_decode = getattr(self, 'use_trtllm_decode_attention', False)
    self.use_trtllm_decode_attention = False
    # Create fresh prefill wrapper to avoid stale internal buffers
    old_wrapper = self._prefill_wrapper
    import torch
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
    self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, backend="auto")
    _n = getattr(type(self), '_fresh_wrapper_n', 0)
    if _n < 5:
        type(self)._fresh_wrapper_n = _n + 1
        print(f"FRESH-WRAPPER: step={_n} creating new BatchPrefillWithPagedKVCacheWrapper",
              file=_s.stderr, flush=True)
    try:
        result = self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=False,
        )
        return result
    finally:
        self.reorder_batch_threshold = original_threshold
        self.use_trtllm_decode_attention = original_trtllm_decode
        # Don't restore old wrapper — let the fresh one be used for this step
        # self._prefill_wrapper = old_wrapper"""

    if old_patch in c:
        c = c.replace(old_patch, new_patch, 1)
        with open(path, "w") as f:
            f.write(c)
        print("  [1/1] Patched: fresh wrapper per drafter step")
    else:
        print("  [1/1] WARNING: existing patch not found")
else:
    print("  [1/1] SKIP: base patch not found")

PYEOF

echo "  Fresh-wrapper fix complete."
