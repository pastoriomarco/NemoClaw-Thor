#!/bin/bash
# Fix: Explicitly write query K/V to drafter's paged cache BEFORE self.attn().
#
# Root cause: unified_kv_cache_update inside Attention.forward may not write
# the drafter's query K/V to the cache (slot_mapping lookup miss or ordering issue).
# FlashInfer then reads stale target data from those pages → wrong output.
#
# Fix: In DFlashQwen3Attention.forward, after computing Q/K/V and before
# calling self.attn(), explicitly write K/V to the cache using do_kv_cache_update.
# This guarantees FlashInfer sees the correct query K/V.

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
QWEN3_DFLASH_PY="$SITE_PACKAGES/vllm/model_executor/models/qwen3_dflash.py"

echo "Applying explicit query KV write fix..."

python3 - "$QWEN3_DFLASH_PY" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

old = """        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)"""

new = """        q, k = self.rotary_emb(positions, q, k)

        # Explicit query KV write: ensure drafter's cache has query K/V
        # before FlashInfer reads it. Fixes stale target data on query pages.
        kv_cache = self.attn.kv_cache
        if kv_cache is not None and kv_cache.numel() > 0:
            from vllm.forward_context import get_forward_context
            fwd_ctx = get_forward_context()
            slot_mapping = fwd_ctx.slot_mapping
            if isinstance(slot_mapping, dict):
                slot_mapping = slot_mapping.get(self.attn.layer_name)
            if slot_mapping is not None:
                num_tokens = q.shape[0]
                k_for_cache = k.view(num_tokens, self.num_kv_heads, self.head_dim)
                v_for_cache = v.view(num_tokens, self.num_kv_heads, self.head_dim)
                self.attn.impl.do_kv_cache_update(
                    self.attn, k_for_cache, v_for_cache,
                    kv_cache, slot_mapping[:num_tokens],
                )

        attn_output = self.attn(q, k, v)"""

if old in content:
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print("  [1/1] Added explicit query KV write before self.attn()")
else:
    print("  [1/1] WARNING: pattern not found")

PYEOF

echo "  Explicit query KV write fix complete."
