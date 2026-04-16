#!/bin/bash
# Replace DFlash drafter's FlashInfer paged attention with PyTorch SDPA.
#
# Problem: FlashInfer's paged KV cache is incompatible with DFlash's
# precompute_and_store_context_kv pattern on SM110. The precompute writes
# K/V via do_kv_cache_update, but FlashInfer's .run() reads from wrong
# offsets, producing garbage draft tokens (0% acceptance).
#
# Fix: Store precomputed context K/V in plain tensors (per-layer buffers),
# then run PyTorch SDPA instead of FlashInfer during the drafter forward.
# SDPA supports non-causal natively and doesn't use paged KV.

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
QWEN3_DFLASH_PY="$SITE_PACKAGES/vllm/model_executor/models/qwen3_dflash.py"

echo "Applying DFlash SDPA attention patch..."

python3 - "$QWEN3_DFLASH_PY" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

patched = False

# --- Patch 1: Replace precompute_and_store_context_kv cache insert with buffer storage ---
# Instead of writing to paged KV cache via do_kv_cache_update, store K/V in
# per-layer buffers that the SDPA forward can access.

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
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        if not hasattr(self, '_sdpa_context_kv'):
            self._sdpa_context_kv = [None] * L
        # Reset on new request (first position is 0)
        first_pos = context_positions[0].item()
        is_new_request = (first_pos == 0)
        for i in range(L):
            new_k = all_k_final[i].clone()
            new_v = all_v[i].clone()
            if self._sdpa_context_kv[i] is not None and not is_new_request:
                # Append new context to existing (accumulate across iterations)
                old_k, old_v = self._sdpa_context_kv[i]
                self._sdpa_context_kv[i] = (
                    torch.cat([old_k, new_k], dim=0),
                    torch.cat([old_v, new_v], dim=0),
                )
            else:
                self._sdpa_context_kv[i] = (new_k, new_v)"""

if old_cache_insert in content:
    content = content.replace(old_cache_insert, new_cache_insert, 1)
    print("  [1/2] Patched precompute: stores K/V in plain buffers")
    patched = True
else:
    print("  [1/2] WARNING: precompute cache insert pattern not found")

# --- Patch 2: Replace DFlashQwen3Attention.forward to use SDPA ---
# Instead of calling self.attn(q, k, v) which goes through FlashInfer paged
# attention, concatenate context K/V with query K/V and run SDPA.

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

new_attn_forward = """    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        \"\"\"DFlash attention using PyTorch SDPA instead of FlashInfer paged KV.
        Context K/V are stored in plain buffers by precompute_and_store_context_kv.
        Query K/V are computed here and concatenated with context K/V for SDPA.\"\"\"
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

        # --- SDPA path: concatenate context K/V with query K/V ---
        # Context K/V buffer is set on this layer by the model's forward via _sdpa_ctx_kv

        # Reshape Q, K, V to [num_tokens, num_heads, head_dim]
        num_tokens = q.shape[0]
        q_heads = q.view(num_tokens, self.num_heads, self.head_dim)
        k_heads = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v_heads = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        # Get context K/V from the buffer
        ctx_kv = getattr(self, '_sdpa_ctx_kv', None)
        if ctx_kv is not None:
            ctx_k, ctx_v = ctx_kv
            # Concatenate context + query K/V: [ctx+query, num_kv_heads, head_dim]
            full_k = torch.cat([ctx_k, k_heads], dim=0)
            full_v = torch.cat([ctx_v, v_heads], dim=0)
        else:
            full_k = k_heads
            full_v = v_heads

        # Reshape for SDPA: [batch=1, num_heads, seq_len, head_dim]
        q_sdpa = q_heads.unsqueeze(0).transpose(1, 2)  # [1, num_heads, num_query, head_dim]

        # GQA: repeat K/V heads to match Q heads
        num_groups = self.num_heads // self.num_kv_heads
        full_k_expanded = full_k.unsqueeze(0).transpose(1, 2)  # [1, num_kv_heads, seq_len, head_dim]
        full_v_expanded = full_v.unsqueeze(0).transpose(1, 2)
        if num_groups > 1:
            full_k_expanded = full_k_expanded.repeat_interleave(num_groups, dim=1)
            full_v_expanded = full_v_expanded.repeat_interleave(num_groups, dim=1)

        # Run SDPA (non-causal by default — no attn_mask)
        scale = self.head_dim ** -0.5
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, full_k_expanded, full_v_expanded,
            attn_mask=None,  # non-causal
            scale=scale,
        )

        # Reshape back: [1, num_heads, num_query, head_dim] -> [num_query, num_heads * head_dim]
        attn_output = attn_output.squeeze(0).transpose(0, 1).contiguous().view(num_tokens, -1)

        output, _ = self.o_proj(attn_output)
        return output"""

if old_attn_forward in content:
    content = content.replace(old_attn_forward, new_attn_forward, 1)
    print("  [2/2] Patched DFlashQwen3Attention.forward to use SDPA")
    patched = True
else:
    print("  [2/2] WARNING: attention forward pattern not found")

if patched:
    with open(path, "w") as f:
        f.write(content)
    print("  DFlash SDPA patches applied.")
else:
    print("  WARNING: No patches applied.")

PYEOF

# --- Patch 3: Wire up context K/V buffers from model to attention layers ---
# The precompute stores _sdpa_context_kv on the DFlashQwen3Model.
# Each DFlashQwen3Attention needs access to its layer's buffer.
# We inject a hook in the DFlashQwen3Model.forward to propagate buffers.
python3 - "$QWEN3_DFLASH_PY" <<'PYEOF2'
import sys

path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

# Find the model's forward method and add buffer propagation before the layer loop
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

echo "  DFlash SDPA mod complete."
