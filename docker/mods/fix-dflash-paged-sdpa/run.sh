#!/bin/bash
# Hybrid fix: paged KV cache + manual SDPA reads for DFlash drafter.
#
# Precompute writes context K/V to paged cache (verified correct).
# Forward writes query K/V to paged cache, then reads ALL K/V back
# using the slot mappings, and runs SDPA.

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
QWEN3_DFLASH_PY="$SITE_PACKAGES/vllm/model_executor/models/qwen3_dflash.py"

echo "Applying DFlash paged-cache SDPA hybrid fix..."

# Patch 1: In precompute, ALSO store context slots + K/V on the model
python3 - "$QWEN3_DFLASH_PY" <<'PYEOF1'
import sys
path = sys.argv[1]
with open(path, "r") as f: c = f.read()

old = """        if context_slot_mapping is None:
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

new = """        if context_slot_mapping is None:
            return

        # --- Write to paged cache AND store context info for forward ---
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        if not hasattr(self, '_paged_ctx_slots'):
            self._paged_ctx_slots = None
            self._paged_ctx_kv = [None] * L
        # Store context slot mapping and K/V for the forward to read
        first_pos = context_positions[0].item()
        is_new = (first_pos == 0)
        if is_new:
            self._paged_ctx_slots = context_slot_mapping.clone()
        else:
            self._paged_ctx_slots = torch.cat([self._paged_ctx_slots, context_slot_mapping])
        for i in range(L):
            attn = self._attn_layers[i]
            kv_cache = attn.kv_cache
            # Write to paged cache
            if kv_cache is not None and kv_cache.numel() > 0:
                attn.impl.do_kv_cache_update(
                    attn, all_k_final[i], all_v[i], kv_cache, context_slot_mapping,
                )
            # Also accumulate plain K/V buffers (for reading back)
            new_k = all_k_final[i].clone()
            new_v = all_v[i].clone()
            if self._paged_ctx_kv[i] is not None and not is_new:
                old_k, old_v = self._paged_ctx_kv[i]
                self._paged_ctx_kv[i] = (torch.cat([old_k, new_k], dim=0),
                                          torch.cat([old_v, new_v], dim=0))
            else:
                self._paged_ctx_kv[i] = (new_k, new_v)"""

if old in c:
    c = c.replace(old, new, 1)
    with open(path, "w") as f: f.write(c)
    print("  [1/2] Patched precompute: paged cache write + context accumulation")
else:
    print("  [1/2] WARNING: precompute pattern not found")
PYEOF1

# Patch 2: Forward reads K/V from paged cache using slot mappings, runs SDPA
python3 - "$QWEN3_DFLASH_PY" <<'PYEOF2'
import sys
path = sys.argv[1]
with open(path, "r") as f: c = f.read()

old = """    def forward(
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

new = '''    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """DFlash paged-cache SDPA: read context from paged cache, concat with query, run SDPA."""
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

        # Read context K/V from paged cache via slot mapping
        kv_cache = self.attn.kv_cache
        ctx_kv = getattr(self, '_sdpa_ctx_kv', None)

        if ctx_kv is not None:
            ctx_k, ctx_v = ctx_kv
            if kv_cache is not None and kv_cache.numel() > 0:
                # Read context K/V back from paged cache (using stored slots)
                ctx_slots = getattr(self, '_paged_ctx_slots_for_layer', None)
                if ctx_slots is not None and ctx_slots.numel() > 0:
                    page_size = kv_cache.shape[2]
                    block_indices = ctx_slots // page_size
                    pos_in_block = ctx_slots % page_size
                    # Gather from cache: [N, kv_heads, head_dim]
                    ctx_k = kv_cache[block_indices, 0, pos_in_block]
                    ctx_v = kv_cache[block_indices, 1, pos_in_block]
            full_k = torch.cat([ctx_k, k_heads], dim=0)
            full_v = torch.cat([ctx_v, v_heads], dim=0)
        else:
            full_k = k_heads
            full_v = v_heads

        # SDPA
        q_sdpa = q_heads.unsqueeze(0).transpose(1, 2)
        num_groups = self.num_heads // self.num_kv_heads
        k_sdpa = full_k.unsqueeze(0).transpose(1, 2)
        v_sdpa = full_v.unsqueeze(0).transpose(1, 2)
        if num_groups > 1:
            k_sdpa = k_sdpa.repeat_interleave(num_groups, dim=1)
            v_sdpa = v_sdpa.repeat_interleave(num_groups, dim=1)

        scale = self.head_dim ** -0.5
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, attn_mask=None, scale=scale,
        )
        attn_output = attn_output.squeeze(0).transpose(0, 1).contiguous().view(num_tokens, -1)

        output, _ = self.o_proj(attn_output)
        return output'''

if old in c:
    c = c.replace(old, new, 1)
    with open(path, "w") as f: f.write(c)
    print("  [2/2] Patched forward: paged cache read + SDPA")
else:
    print("  [2/2] WARNING: forward pattern not found")
PYEOF2

# Patch 3: Wire context info from model to attention layers
python3 - "$QWEN3_DFLASH_PY" <<'PYEOF3'
import sys
path = sys.argv[1]
with open(path, "r") as f: c = f.read()

old = """        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )"""

new = """        # Wire context K/V and slot mapping to each attention layer
        if hasattr(self, '_paged_ctx_kv') and self._paged_ctx_kv[0] is not None:
            for i, layer in enumerate(self.layers):
                layer.self_attn._sdpa_ctx_kv = self._paged_ctx_kv[i]
                layer.self_attn._paged_ctx_slots_for_layer = self._paged_ctx_slots
        else:
            for layer in self.layers:
                layer.self_attn._sdpa_ctx_kv = None
                layer.self_attn._paged_ctx_slots_for_layer = None

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )"""

if old in c:
    c = c.replace(old, new, 1)
    with open(path, "w") as f: f.write(c)
    print("  [3/3] Wired context K/V and slots to attention layers")
else:
    print("  [3/3] WARNING: forward loop pattern not found")
PYEOF3

echo "  DFlash paged-cache SDPA hybrid complete."
