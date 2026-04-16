#!/bin/bash
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
FI="$SITE_PACKAGES/vllm/v1/attention/backends/flashinfer.py"

echo "Applying combined drafter-prefill + fresh-wrapper + logging fix..."

python3 - "$FI" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f:
    c = f.read()

override_code = '''

# Combined PR #36060 + fresh wrapper + verbose logging
_orig_build_for_drafting = FlashInferMetadataBuilder.build_for_drafting

def _patched_build_for_drafting(self, common_attn_metadata, draft_index):
    """Force prefill + logging for drafter."""
    import torch, sys as _s

    original_threshold = self.reorder_batch_threshold
    original_trtllm = getattr(self, 'use_trtllm_decode_attention', False)

    if not original_trtllm:
        self.reorder_batch_threshold = 0
    self.use_trtllm_decode_attention = False

    _n = getattr(type(self), '_fw_n', 0)
    if _n < 10:
        type(self)._fw_n = _n + 1
        print(f"BFD step={_n}: threshold={self.reorder_batch_threshold} "
              f"max_qlen={common_attn_metadata.max_query_len} "
              f"num_reqs={common_attn_metadata.num_reqs} "
              f"num_tokens={common_attn_metadata.num_actual_tokens} "
              f"qsl={common_attn_metadata.query_start_loc_cpu.tolist()[:5]}",
              file=_s.stderr, flush=True)

    try:
        result = self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=False,
        )
        if _n < 10:
            print(f"BFD result: num_prefills={result.num_prefills} "
                  f"num_decodes={result.num_decodes} "
                  f"has_prefill_wrapper={result.prefill is not None}",
                  file=_s.stderr, flush=True)
        return result
    finally:
        self.reorder_batch_threshold = original_threshold
        self.use_trtllm_decode_attention = original_trtllm

FlashInferMetadataBuilder.build_for_drafting = _patched_build_for_drafting
'''

marker = "def fast_plan_decode("
if marker in c and "_patched_build_for_drafting" not in c:
    c = c.replace(marker, override_code + "\n" + marker, 1)
    with open(path, "w") as f:
        f.write(c)
    print("  [1/1] Added patched build_for_drafting with logging")
elif "_patched_build_for_drafting" in c:
    print("  [1/1] SKIP (already patched)")
else:
    print("  [1/1] WARNING: marker not found")

PYEOF
echo "  Combined fix complete."
