#!/bin/bash
# Direct FlashInfer: bypass unified_attention_with_output, call wrapper.run() directly.
#
# Bug #5: FlashInfer produces wrong output through the unified_attention custom op.
# But standalone test calling wrapper.run() directly WORKS.
#
# This mod: in the DFlash forward, instead of self.attn(q, k, v):
# 1. Write query K/V to paged cache (via do_kv_cache_update)
# 2. Create a fresh BatchPrefillWithPagedKVCacheWrapper
# 3. Plan with correct parameters (from the layer's metadata)
# 4. Call wrapper.run() directly on the paged cache
#
# This is both the bug #5 workaround AND the efficient integration —
# one kernel pass, no gather/concat/GQA-expand copies.

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
QWEN3_DFLASH_PY="$SITE_PACKAGES/vllm/model_executor/models/qwen3_dflash.py"

echo "Applying DFlash direct FlashInfer fix..."

python3 - "$QWEN3_DFLASH_PY" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

old_forward = """    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        \"\"\"DFlash attention assumes that the KV cache is already populated
        with the context K/V from the target model's hidden states. This forward op
        computes attention for the query tokens only.
        See also: precompute_and_store_context_kv\"\"\"
        qkv = F.linear(hidden_states, self.qkv_proj.weight, self.qkv_proj.bias)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Per-head RMSNorm
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output"""

new_forward = '''    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """DFlash direct FlashInfer: bypass unified_attention, call wrapper.run() directly."""
        qkv = F.linear(hidden_states, self.qkv_proj.weight, self.qkv_proj.bias)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)

        q, k = self.rotary_emb(positions, q, k)

        num_tokens = q.shape[0]
        q_heads = q.view(num_tokens, self.num_heads, self.head_dim)
        k_heads = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v_heads = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        kv_cache = self.attn.kv_cache
        has_cache = kv_cache is not None and kv_cache.numel() > 0

        if has_cache:
            from vllm.forward_context import get_forward_context
            from flashinfer import BatchPrefillWithPagedKVCacheWrapper

            fwd_ctx = get_forward_context()

            # Get slot mapping for writing query K/V
            slot_mapping = fwd_ctx.slot_mapping
            if isinstance(slot_mapping, dict):
                slot_mapping = slot_mapping.get(self.attn.layer_name, None)

            # Write query K/V to paged cache
            if slot_mapping is not None:
                self.attn.impl.do_kv_cache_update(
                    self.attn, k_heads, v_heads, kv_cache, slot_mapping[:num_tokens],
                )

            # Get attention metadata for page indices
            attn_meta = None
            if fwd_ctx.attn_metadata:
                attn_meta = fwd_ctx.attn_metadata.get(self.attn.layer_name)

            if attn_meta is not None and hasattr(attn_meta, 'prefill') and attn_meta.prefill:
                # Use the ALREADY-PLANNED wrapper from metadata (it has correct page indices)
                wrapper = attn_meta.prefill.wrapper
                # Call .run() directly — bypassing unified_attention_with_output
                output = torch.empty(num_tokens, self.num_heads, self.head_dim,
                                     dtype=q_heads.dtype, device=q_heads.device)
                wrapper.run(
                    q_heads,
                    kv_cache,
                    k_scale=self.attn._k_scale_float if hasattr(self.attn, '_k_scale_float') else 1.0,
                    v_scale=self.attn._v_scale_float if hasattr(self.attn, '_v_scale_float') else 1.0,
                    out=output,
                )
                attn_output = output.view(num_tokens, -1)
            else:
                # Fallback: no FlashInfer metadata, use SDPA
                ctx_kv = getattr(self, '_sdpa_ctx_kv', None)
                if ctx_kv is not None:
                    ctx_k, ctx_v = ctx_kv
                    full_k = torch.cat([ctx_k, k_heads], dim=0)
                    full_v = torch.cat([ctx_v, v_heads], dim=0)
                else:
                    full_k = k_heads
                    full_v = v_heads
                q_sdpa = q_heads.unsqueeze(0).transpose(1, 2)
                groups = self.num_heads // self.num_kv_heads
                k_s = full_k.unsqueeze(0).transpose(1, 2)
                v_s = full_v.unsqueeze(0).transpose(1, 2)
                if groups > 1:
                    k_s = k_s.repeat_interleave(groups, dim=1)
                    v_s = v_s.repeat_interleave(groups, dim=1)
                scale = self.head_dim ** -0.5
                out = torch.nn.functional.scaled_dot_product_attention(
                    q_sdpa, k_s, v_s, attn_mask=None, scale=scale)
                attn_output = out.squeeze(0).transpose(0, 1).contiguous().view(num_tokens, -1)
        else:
            # No cache (warmup): compute with SDPA on query only
            q_sdpa = q_heads.unsqueeze(0).transpose(1, 2)
            groups = self.num_heads // self.num_kv_heads
            k_s = k_heads.unsqueeze(0).transpose(1, 2)
            v_s = v_heads.unsqueeze(0).transpose(1, 2)
            if groups > 1:
                k_s = k_s.repeat_interleave(groups, dim=1)
                v_s = v_s.repeat_interleave(groups, dim=1)
            scale = self.head_dim ** -0.5
            out = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_s, v_s, attn_mask=None, scale=scale)
            attn_output = out.squeeze(0).transpose(0, 1).contiguous().view(num_tokens, -1)

        output, _ = self.o_proj(attn_output)
        return output'''

if old_forward in content:
    content = content.replace(old_forward, new_forward, 1)
    with open(path, "w") as f:
        f.write(content)
    print("  [1/1] Patched DFlash forward with direct FlashInfer .run()")
else:
    print("  [1/1] WARNING: forward pattern not found")

PYEOF

echo "  DFlash direct FlashInfer fix complete."
