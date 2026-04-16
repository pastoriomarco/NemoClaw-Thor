#!/bin/bash
# Capture FlashInfer plan parameters, replay standalone, compare with SDPA.
# This definitively tests whether the plan parameters are correct.
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
QWEN3_DFLASH_PY="$SITE_PACKAGES/vllm/model_executor/models/qwen3_dflash.py"

echo "Applying FlashInfer replay diagnostic..."

python3 - "$QWEN3_DFLASH_PY" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

# Replace the forward to: use FlashInfer wrapper from metadata, then also compute SDPA, compare
old = """        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output"""

new = '''        # REPLAY DIAGNOSTIC: FlashInfer wrapper.run() vs standalone vs SDPA
        import sys as _sys
        from vllm.forward_context import get_forward_context
        _replay_n = getattr(DFlashQwen3Attention, '_replay_n', 0)

        num_tokens = q.shape[0]
        q_h = q.view(num_tokens, self.num_heads, self.head_dim)
        k_h = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v_h = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        kv_cache = self.attn.kv_cache
        has_cache = kv_cache is not None and kv_cache.numel() > 0

        if has_cache and _replay_n < 3 and num_tokens > 1:
            DFlashQwen3Attention._replay_n = _replay_n + 1
            p = lambda *a: print(*a, file=_sys.stderr, flush=True)

            fwd_ctx = get_forward_context()
            slot_mapping = fwd_ctx.slot_mapping
            if isinstance(slot_mapping, dict):
                slot_mapping = slot_mapping.get(self.attn.layer_name, None)

            # Write query K/V to cache
            if slot_mapping is not None:
                self.attn.impl.do_kv_cache_update(
                    self.attn, k_h, v_h, kv_cache, slot_mapping[:num_tokens])

            attn_meta = fwd_ctx.attn_metadata.get(self.attn.layer_name) if fwd_ctx.attn_metadata else None

            p(f"\\nREPLAY step={_replay_n} layer={self.attn.layer_name}")
            p(f"  q: {q_h.shape} kv_cache: {kv_cache.shape}")

            # Method 1: Pipeline FlashInfer (via self.attn)
            fi_out = self.attn(q, k, v)
            p(f"  FI-pipeline: norm={fi_out.float().norm():.2f} zero={fi_out.float().abs().max() < 1e-6}")

            # Method 2: Read from cache + SDPA
            ctx_kv = getattr(self, '_sdpa_ctx_kv', None)
            if ctx_kv is not None:
                ctx_k, ctx_v = ctx_kv
                # Read from paged cache
                ctx_slots = getattr(self, '_paged_ctx_slots_for_layer', None)
                if ctx_slots is not None and ctx_slots.numel() > 0 and kv_cache.numel() > 0:
                    ps = kv_cache.shape[2]
                    bi = ctx_slots // ps
                    pi = ctx_slots % ps
                    ctx_k = kv_cache[bi, 0, pi]
                    ctx_v = kv_cache[bi, 1, pi]
                full_k = torch.cat([ctx_k, k_h], dim=0)
                full_v = torch.cat([ctx_v, v_h], dim=0)
            else:
                full_k = k_h
                full_v = v_h

            q_s = q_h.unsqueeze(0).transpose(1, 2)
            g = self.num_heads // self.num_kv_heads
            k_s = full_k.unsqueeze(0).transpose(1, 2)
            v_s = full_v.unsqueeze(0).transpose(1, 2)
            if g > 1:
                k_s = k_s.repeat_interleave(g, dim=1)
                v_s = v_s.repeat_interleave(g, dim=1)
            scale = self.head_dim ** -0.5
            sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                q_s, k_s, v_s, attn_mask=None, scale=scale)
            sdpa_out = sdpa_out.squeeze(0).transpose(0, 1).contiguous().view(num_tokens, -1)
            p(f"  SDPA: norm={sdpa_out.float().norm():.2f}")

            # Method 3: Direct FlashInfer .run() with wrapper from metadata
            if attn_meta and hasattr(attn_meta, 'prefill') and attn_meta.prefill:
                wrapper = attn_meta.prefill.wrapper
                fi_direct = torch.empty_like(q_h)
                try:
                    wrapper.run(q_h, kv_cache,
                                k_scale=getattr(self.attn, '_k_scale_float', 1.0),
                                v_scale=getattr(self.attn, '_v_scale_float', 1.0),
                                out=fi_direct)
                    fi_direct_flat = fi_direct.view(num_tokens, -1)
                    p(f"  FI-direct: norm={fi_direct_flat.float().norm():.2f}")

                    diff_fi_sdpa = (fi_direct_flat.float() - sdpa_out.float()).abs().max().item()
                    p(f"  FI-direct vs SDPA: max_diff={diff_fi_sdpa:.4f}")
                except Exception as e:
                    p(f"  FI-direct FAILED: {e}")
            else:
                p(f"  No prefill wrapper available")

            # Use SDPA output (known correct)
            attn_output = sdpa_out
        else:
            # Normal path: just use self.attn
            attn_output = self.attn(q, k, v)

        output, _ = self.o_proj(attn_output)
        return output'''

if old in content:
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print("  [1/1] Added FlashInfer replay diagnostic")
else:
    print("  [1/1] WARNING: forward pattern not found")

PYEOF
echo "  Replay diagnostic complete."
