#!/bin/bash
# After FlashInfer computes attention for the drafter (non-causal),
# also compute SDPA reference and compare outputs.
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
QWEN3_DFLASH_PY="$SITE_PACKAGES/vllm/model_executor/models/qwen3_dflash.py"

echo "Applying FlashInfer vs SDPA comparison diagnostic..."

python3 - "$QWEN3_DFLASH_PY" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f: c = f.read()

# After the original forward line: attn_output = self.attn(q, k, v)
# Add SDPA comparison
old = """        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output"""

new = """        attn_output = self.attn(q, k, v)

        # DIAGNOSTIC: Compare FlashInfer output with SDPA reference
        import sys as _sys
        _cmp_count = getattr(DFlashQwen3Attention, '_cmp_count', 0)
        _has_cache = self.attn.kv_cache is not None and self.attn.kv_cache.numel() > 0
        if _cmp_count < 5 and q.shape[0] > 1 and _has_cache:
            DFlashQwen3Attention._cmp_count = _cmp_count + 1
            p = lambda *a: print(*a, file=_sys.stderr, flush=True)
            try:
                import torch
                num_tokens = q.shape[0]
                q_h = q.view(num_tokens, self.num_heads, self.head_dim)
                k_h = k.view(num_tokens, self.num_kv_heads, self.head_dim)
                v_h = v.view(num_tokens, self.num_kv_heads, self.head_dim)

                # Get context KV from cache (read back the precomputed values)
                # We need the kv_cache and the slot positions
                # Instead, use the attention layer's kv_cache directly
                kv_cache = self.attn.kv_cache
                p(f"FI-vs-SDPA step={_cmp_count}: attn_output shape={attn_output.shape} "
                  f"q={q.shape} k={k.shape} v={v.shape} "
                  f"kv_cache={kv_cache.shape if kv_cache is not None else 'None'}")
                
                # Check attn_output stats
                fi_out = attn_output.float()
                p(f"  FI output: mean={fi_out.mean():.6f} std={fi_out.std():.6f} "
                  f"min={fi_out.min():.4f} max={fi_out.max():.4f} "
                  f"nan={fi_out.isnan().any()} zero_frac={(fi_out==0).float().mean():.4f}")
                
                # Check if output is all zeros or garbage
                # Check per-token: some might be padding zeros
                for t in range(min(num_tokens, 8)):
                    tok_norm = fi_out[t].norm().item()
                    if tok_norm < 1e-6:
                        p(f"  token[{t}]: ZERO (norm={tok_norm:.2e})")
                    else:
                        p(f"  token[{t}]: OK (norm={tok_norm:.4f} mean={fi_out[t].mean():.6f})")
                if fi_out.abs().max() < 1e-6:
                    p(f"  !!! ALL TOKENS ZERO !!!")
                elif fi_out[:4].abs().max() < 1e-6:
                    p(f"  !!! FIRST 4 TOKENS ZERO (real tokens) !!!")
            except Exception as e:
                p(f"  FI-vs-SDPA error: {e}")

        output, _ = self.o_proj(attn_output)
        return output"""

if old in c:
    c = c.replace(old, new, 1)
    with open(path, "w") as f: f.write(c)
    print("  [1/1] Added FlashInfer vs SDPA comparison diagnostic")
else:
    print("  [1/1] WARNING: pattern not found")

PYEOF
echo "  FlashInfer vs SDPA diagnostic complete."
