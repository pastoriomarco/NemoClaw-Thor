#!/bin/bash
# Dump actual KV cache content at the pages FlashInfer reads vs what precompute wrote.
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
QWEN3_DFLASH_PY="$SITE_PACKAGES/vllm/model_executor/models/qwen3_dflash.py"

echo "Applying cache dump diagnostic..."

# Patch precompute to save what was written
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

        # --- Per-layer cache insert + save what we wrote ---
        import sys as _s2
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        if not hasattr(self, '_dump_written'):
            self._dump_written = {}
        for i in range(L):
            attn = self._attn_layers[i]
            kv_cache = attn.kv_cache
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                kv_cache,
                context_slot_mapping,
            )
            # Save: what we wrote, where, and what the cache looks like after
            self._dump_written[i] = {
                'k': all_k_final[i].clone(),
                'v': all_v[i].clone(),
                'slots': context_slot_mapping.clone(),
                'cache_shape': kv_cache.shape,
            }"""

if old in c:
    c = c.replace(old, new, 1)
    with open(path, "w") as f: f.write(c)
    print("  [1/2] Patched precompute to save written data")
else:
    print("  [1/2] WARNING: precompute pattern not found")
PYEOF1

# Patch forward to dump what FlashInfer reads vs what was written
python3 - "$QWEN3_DFLASH_PY" <<'PYEOF2'
import sys
path = sys.argv[1]
with open(path, "r") as f: c = f.read()

old = """        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output"""

new = '''        attn_output = self.attn(q, k, v)

        import sys as _s3
        _dn = getattr(DFlashQwen3Attention, '_dn', 0)
        kv_cache = self.attn.kv_cache
        has_cache = kv_cache is not None and kv_cache.numel() > 0
        num_tokens = q.shape[0]

        from vllm.forward_context import get_forward_context as _gfc3
        _fctx3 = _gfc3()
        _has_meta3 = (_fctx3.attn_metadata is not None and
                      self.attn.layer_name in (_fctx3.attn_metadata or {}))

        if _dn < 1 and has_cache and num_tokens > 1 and _has_meta3:
            DFlashQwen3Attention._dn = _dn + 1
            p = lambda *a: print(*a, file=_s3.stderr, flush=True)
            p(f"\\n{'#'*60}")
            p(f"CACHE-DUMP layer={self.attn.layer_name}")

            # Get the slot_mapping used for query KV write
            sm = _fctx3.slot_mapping
            if isinstance(sm, dict):
                sm = sm.get(self.attn.layer_name)
            if sm is not None:
                p(f"  query_slot_mapping: {sm[:num_tokens].tolist()}")

            # Get what precompute wrote (from the model)
            model = None
            for parent in [self]:
                if hasattr(parent, '_dump_written'):
                    model = parent
                    break
            # Try getting from the model (DFlashQwen3Model)
            # The layer index is in the layer_name
            layer_idx = int(self.attn.layer_name.split('.')[-3]) - 32  # relative to drafter
            p(f"  drafter_layer_idx={layer_idx}")

            # Read what's in the cache at the precompute slots
            # We need to get _dump_written from the parent model
            # Access via the module hierarchy
            dump = None
            for name, mod in self.named_modules():
                pass  # just iterate
            # Access from class-level storage instead
            _dump_store = getattr(DFlashQwen3Attention, '_dump_store', None)
            if _dump_store and layer_idx in _dump_store:
                d = _dump_store[layer_idx]
                ctx_slots = d['slots']
                written_k = d['k']
                p(f"  precompute wrote {written_k.shape[0]} tokens to slots {ctx_slots[:5].tolist()}...")

                # Read back from cache at those slots
                page_size = kv_cache.shape[2]
                bi = ctx_slots // page_size
                pi = ctx_slots % page_size
                cache_k = kv_cache[bi, 0, pi]
                diff = (cache_k.float() - written_k.float()).abs().max().item()
                p(f"  PRECOMPUTE READBACK diff: {diff:.6e}")

                # Now check: what does FlashInfer's plan reference?
                attn_meta = _fctx3.attn_metadata.get(self.attn.layer_name)
                if attn_meta and attn_meta.prefill:
                    wrapper = attn_meta.prefill.wrapper
                    # The wrapper has internal paged_kv_indices after .plan()
                    for attr in ['_paged_kv_indices_buf', '_paged_kv_indptr_buf',
                                 '_paged_kv_last_page_len_buf']:
                        buf = getattr(wrapper, attr, None)
                        if buf is not None and buf.numel() < 20:
                            p(f"  wrapper.{attr}: {buf.tolist()}")
                        elif buf is not None:
                            p(f"  wrapper.{attr}: shape={buf.shape}")

                    # Read what's at the FlashInfer referenced pages
                    idx_buf = getattr(wrapper, '_paged_kv_indices_buf', None)
                    if idx_buf is not None and idx_buf.numel() > 0:
                        fi_pages = idx_buf[:10].tolist()
                        p(f"  FI reads from pages: {fi_pages}")
                        for pg in fi_pages[:3]:
                            if pg < kv_cache.shape[0]:
                                pg_k = kv_cache[pg, 0, :, :, :]
                                p(f"    page {pg} K[:2,0,:4]: {pg_k[:2, 0, :4].tolist()}")
                                # Compare with what precompute wrote
                                # precompute wrote to bi[0..N] which should include pg
                                if pg in bi.tolist():
                                    p(f"    page {pg} IS a precompute page")
                                else:
                                    p(f"    page {pg} is NOT a precompute page! precompute pages: {bi.unique().tolist()}")
            else:
                p(f"  No precompute dump available for layer {layer_idx}")
                p(f"  Available dumps: {list(getattr(DFlashQwen3Attention, '_dump_store', {}).keys())}")

            p(f"{'#'*60}")

        output, _ = self.o_proj(attn_output)
        return output'''

if old in c:
    c = c.replace(old, new, 1)
    with open(path, "w") as f: f.write(c)
    print("  [2/2] Patched forward with cache dump diagnostic")
else:
    print("  [2/2] WARNING: forward pattern not found")
PYEOF2

# Patch model forward to propagate dump data to attention layers
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

new = """        # Store precompute dump on the attention class for layer access
        if hasattr(self, '_dump_written'):
            from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Attention
            DFlashQwen3Attention._dump_store = self._dump_written

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
    print("  [3/3] Wired precompute dump to attention layers")
else:
    print("  [3/3] WARNING: forward loop not found")
PYEOF3

echo "  Cache dump diagnostic complete."
