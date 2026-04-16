#!/bin/bash
# DFlash SDPA attention with comprehensive diagnostics.
#
# Same as fix-dflash-sdpa but adds diagnostic logging to understand
# the 84% (1 token) → 8% (8 tokens) acceptance rate degradation.
#
# Diagnostics:
#   1. Compare SDPA output vs manual Q@K^T/sqrt(d)→softmax→@V
#   2. Attention weight analysis per query position
#   3. Context KV shape/stats validation
#   4. GQA expansion verification
#   5. Save tensor dumps to /tmp/dflash-diag/

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
QWEN3_DFLASH_PY="$SITE_PACKAGES/vllm/model_executor/models/qwen3_dflash.py"

echo "Applying DFlash SDPA + diagnostic attention patch..."

python3 - "$QWEN3_DFLASH_PY" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

patched = False

# --- Patch 1: Replace precompute_and_store_context_kv cache insert with buffer storage ---
old_cache_insert = """        if context_slot_mapping is None:
            return

        # --- Per-layer cache insert ---
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        for i in range(L):
            attn = self._attn_layers[i]
            kv_cache = attn.kv_cache
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                kv_cache,
                context_slot_mapping,
            )"""

new_cache_insert = """        if context_slot_mapping is None:
            return

        # --- Accumulate context K/V in plain buffers for SDPA (bypass paged KV) ---
        # The flash_attn paged KV cache persists across DFlash steps. Each
        # precompute call writes K/V for NEW tokens only. We must accumulate
        # to replicate that behavior.  Reset only on a brand-new request.
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        if not hasattr(self, '_sdpa_context_kv'):
            self._sdpa_context_kv = [None] * L
        first_pos = context_positions[0].item()
        is_new_request = (first_pos == 0)
        import sys as _sys
        for i in range(L):
            new_k = all_k_final[i].clone()
            new_v = all_v[i].clone()
            if self._sdpa_context_kv[i] is not None and not is_new_request:
                old_k, old_v = self._sdpa_context_kv[i]
                self._sdpa_context_kv[i] = (
                    torch.cat([old_k, new_k], dim=0),
                    torch.cat([old_v, new_v], dim=0),
                )
            else:
                self._sdpa_context_kv[i] = (new_k, new_v)
        if i == 0:
            buf_len = self._sdpa_context_kv[0][0].shape[0]
            print(f"PRECOMPUTE: first_pos={first_pos} new_request={is_new_request} "
                  f"added={num_ctx} total_buf={buf_len}", file=_sys.stderr, flush=True)"""

if old_cache_insert in content:
    content = content.replace(old_cache_insert, new_cache_insert, 1)
    print("  [1/3] Patched precompute: stores K/V in plain buffers")
    patched = True
else:
    print("  [1/3] WARNING: precompute cache insert pattern not found")

# --- Patch 2: Replace DFlashQwen3Attention.forward with SDPA + diagnostics ---
old_attn_forward = """    def forward(
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

new_attn_forward = '''    # --- Diagnostic state (class-level) ---
    _diag_counter = 0
    _diag_limit = 3
    _diag_first_layer = None  # Auto-detect first drafter layer

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """DFlash SDPA attention with diagnostics."""
        import os, sys
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

        # --- SDPA path ---
        num_tokens = q.shape[0]
        q_heads = q.view(num_tokens, self.num_heads, self.head_dim)
        k_heads = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v_heads = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        ctx_kv = getattr(self, '_sdpa_ctx_kv', None)
        if ctx_kv is not None:
            ctx_k, ctx_v = ctx_kv
            full_k = torch.cat([ctx_k, k_heads], dim=0)
            full_v = torch.cat([ctx_v, v_heads], dim=0)
            num_ctx = ctx_k.shape[0]
        else:
            full_k = k_heads
            full_v = v_heads
            num_ctx = 0

        # SDPA format: [1, heads, seq, dim]
        q_sdpa = q_heads.unsqueeze(0).transpose(1, 2)
        num_groups = self.num_heads // self.num_kv_heads
        k_sdpa = full_k.unsqueeze(0).transpose(1, 2)
        v_sdpa = full_v.unsqueeze(0).transpose(1, 2)
        if num_groups > 1:
            k_expanded = k_sdpa.repeat_interleave(num_groups, dim=1)
            v_expanded = v_sdpa.repeat_interleave(num_groups, dim=1)
        else:
            k_expanded = k_sdpa
            v_expanded = v_sdpa

        scale = self.head_dim ** -0.5

        # Main SDPA computation
        attn_out_4d = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_expanded, v_expanded,
            attn_mask=None, scale=scale,
        )

        # --- DIAGNOSTICS ---
        layer_name = self.layer_name
        # Auto-detect first drafter layer
        if DFlashQwen3Attention._diag_first_layer is None and ctx_kv is not None and num_tokens > 1:
            DFlashQwen3Attention._diag_first_layer = layer_name
        is_first_layer = (layer_name == DFlashQwen3Attention._diag_first_layer)
        do_diag = (
            DFlashQwen3Attention._diag_counter < DFlashQwen3Attention._diag_limit
            and num_tokens > 1  # skip single-token (warmup)
            and is_first_layer
            and ctx_kv is not None
        )

        if do_diag:
            DFlashQwen3Attention._diag_counter += 1
            step = DFlashQwen3Attention._diag_counter
            p = lambda *a: print(*a, file=sys.stderr, flush=True)

            p(f"\\n{'='*80}")
            p(f"DIAG step={step} layer={layer_name} num_query={num_tokens} num_ctx={num_ctx}")
            p(f"DIAG shapes: Q={q_sdpa.shape} K={k_expanded.shape} V={v_expanded.shape}")
            p(f"DIAG positions: {positions.tolist()}")
            p(f"DIAG num_heads={self.num_heads} num_kv_heads={self.num_kv_heads} head_dim={self.head_dim} groups={num_groups}")

            # Context KV stats
            p(f"DIAG ctx_k: shape={ctx_k.shape} mean={ctx_k.float().mean():.6f} std={ctx_k.float().std():.6f} "
              f"min={ctx_k.float().min():.6f} max={ctx_k.float().max():.6f} "
              f"nan={ctx_k.isnan().any()} inf={ctx_k.isinf().any()}")
            p(f"DIAG ctx_v: shape={ctx_v.shape} mean={ctx_v.float().mean():.6f} std={ctx_v.float().std():.6f}")

            # Query KV stats
            p(f"DIAG query_k: mean={k_heads.float().mean():.6f} std={k_heads.float().std():.6f}")
            p(f"DIAG query_v: mean={v_heads.float().mean():.6f} std={v_heads.float().std():.6f}")
            p(f"DIAG query_q: mean={q_heads.float().mean():.6f} std={q_heads.float().std():.6f}")

            # --- Manual attention computation ---
            # Q @ K^T / sqrt(d) -> softmax -> @ V
            with torch.no_grad():
                q_f = q_sdpa.float()
                k_f = k_expanded.float()
                v_f = v_expanded.float()

                logits = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
                weights = torch.nn.functional.softmax(logits, dim=-1)
                manual_out = torch.matmul(weights, v_f)

                # Compare SDPA vs manual
                diff = (attn_out_4d.float() - manual_out).abs()
                p(f"DIAG SDPA vs manual: max_diff={diff.max():.6e} mean_diff={diff.mean():.6e}")

                # --- Per-query-position attention analysis ---
                # weights shape: [1, num_heads, num_query, num_kv]
                total_kv = num_ctx + num_tokens
                for qi in range(min(num_tokens, 9)):
                    w = weights[0, :, qi, :]  # [num_heads, total_kv]
                    w_mean = w.mean(dim=0)  # [total_kv] averaged over heads

                    # Split into context vs query attention
                    ctx_attn = w_mean[:num_ctx].sum().item()
                    qry_attn = w_mean[num_ctx:].sum().item()
                    self_attn = w_mean[num_ctx + qi].item() if (num_ctx + qi) < total_kv else 0

                    # Top-5 attended positions
                    topk_vals, topk_idx = w_mean.topk(min(5, total_kv))
                    topk_str = ", ".join(f"pos{idx.item()}={val:.4f}" for idx, val in zip(topk_idx, topk_vals))

                    # Attention entropy (higher = more uniform)
                    entropy = -(w * w.clamp(min=1e-10).log()).sum(dim=-1).mean().item()

                    # Max logit (pre-softmax)
                    max_logit = logits[0, :, qi, :].max().item()
                    min_logit = logits[0, :, qi, :].min().item()

                    label = "bonus" if qi == 0 else f"mask_{qi-1}"
                    p(f"DIAG q[{qi}]({label}): ctx_attn={ctx_attn:.4f} qry_attn={qry_attn:.4f} "
                      f"self_attn={self_attn:.6f} entropy={entropy:.2f} "
                      f"logit_range=[{min_logit:.2f},{max_logit:.2f}]")
                    p(f"DIAG q[{qi}] top5: {topk_str}")

                # --- GQA verification: per-group attention ---
                # Check if different Q heads within a group agree
                if num_groups > 1:
                    group0_weights = weights[0, :num_groups, :, :]  # First KV head's Q group
                    group_std = group0_weights.std(dim=0).mean().item()
                    p(f"DIAG GQA group0 intra-group std: {group_std:.6f} (should be >0, heads specialize)")

                # --- Alternative: native GQA (enable_gqa) ---
                try:
                    native_gqa_out = torch.nn.functional.scaled_dot_product_attention(
                        q_sdpa, k_sdpa, v_sdpa,
                        attn_mask=None, scale=scale, enable_gqa=True,
                    )
                    gqa_diff = (attn_out_4d.float() - native_gqa_out.float()).abs()
                    p(f"DIAG native_gqa vs repeat_interleave: max_diff={gqa_diff.max():.6e} mean_diff={gqa_diff.mean():.6e}")
                    if gqa_diff.max() > 0.01:
                        p(f"DIAG WARNING: native GQA differs significantly!")
                except Exception as e:
                    p(f"DIAG native_gqa failed: {e}")

                # --- Save tensors for offline analysis ---
                try:
                    os.makedirs('/tmp/dflash-diag', exist_ok=True)
                    torch.save({
                        'q': q_sdpa.cpu(),
                        'k_expanded': k_expanded.cpu(),
                        'v_expanded': v_expanded.cpu(),
                        'k_raw': k_sdpa.cpu(),
                        'v_raw': v_sdpa.cpu(),
                        'ctx_k': ctx_k.cpu(),
                        'ctx_v': ctx_v.cpu(),
                        'query_k': k_heads.cpu(),
                        'query_v': v_heads.cpu(),
                        'positions': positions.cpu(),
                        'attn_output': attn_out_4d.cpu(),
                        'manual_output': manual_out.cpu(),
                        'manual_weights': weights.cpu(),
                        'manual_logits': logits.cpu(),
                        'num_ctx': num_ctx,
                        'num_query': num_tokens,
                        'num_heads': self.num_heads,
                        'num_kv_heads': self.num_kv_heads,
                        'head_dim': self.head_dim,
                        'scale': scale,
                    }, f'/tmp/dflash-diag/step{step}.pt')
                    p(f"DIAG tensors saved to /tmp/dflash-diag/step{step}.pt")
                except Exception as e:
                    p(f"DIAG save failed: {e}")

            p(f"{'='*80}\\n")

        # Reshape output
        attn_output = attn_out_4d.squeeze(0).transpose(0, 1).contiguous().view(num_tokens, -1)
        output, _ = self.o_proj(attn_output)
        return output'''

if old_attn_forward in content:
    content = content.replace(old_attn_forward, new_attn_forward, 1)
    print("  [2/3] Patched DFlashQwen3Attention.forward with SDPA + diagnostics")
    patched = True
else:
    print("  [2/3] WARNING: attention forward pattern not found")

if patched:
    with open(path, "w") as f:
        f.write(content)
    print("  Patches applied.")
else:
    print("  WARNING: No patches applied.")

PYEOF

# --- Patch 3: Wire up context K/V buffers from model to attention layers ---
python3 - "$QWEN3_DFLASH_PY" <<'PYEOF2'
import sys

path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

old_forward = """        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )"""

new_forward = """        # Propagate SDPA context K/V buffers to each attention layer
        if hasattr(self, '_sdpa_context_kv') and self._sdpa_context_kv[0] is not None:
            for i, layer in enumerate(self.layers):
                layer.self_attn._sdpa_ctx_kv = self._sdpa_context_kv[i]
        else:
            for layer in self.layers:
                layer.self_attn._sdpa_ctx_kv = None

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )"""

if old_forward in content:
    content = content.replace(old_forward, new_forward, 1)
    with open(path, "w") as f:
        f.write(content)
    print("  [3/3] Wired context K/V buffers to attention layers")
else:
    print("  [3/3] WARNING: forward loop pattern not found")

PYEOF2

echo "  DFlash SDPA diagnostic mod complete."
