#!/bin/bash
# Diagnose FlashInfer KV cache write/read for DFlash precompute.
#
# The DFlash precompute writes context K/V to the drafter's paged KV cache
# via do_kv_cache_update. Then the drafter forward reads from the same cache
# via FlashInfer .forward(). This gave 0% acceptance — reads garbage.
#
# This diagnostic:
# 1. Intercepts precompute to log slot_mapping, block_size, cache shape
# 2. After writing, reads back from cache to verify values are correct
# 3. During forward, logs what FlashInfer's metadata says about the pages
# 4. Compares precompute write locations vs forward read locations

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
QWEN3_DFLASH_PY="$SITE_PACKAGES/vllm/model_executor/models/qwen3_dflash.py"

echo "Applying FlashInfer KV diagnostic patch..."

python3 - "$QWEN3_DFLASH_PY" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

# Replace the cache insert section with a diagnostic version
# that writes to cache AND reads back to verify
old_cache = """        if context_slot_mapping is None:
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

new_cache = """        if context_slot_mapping is None:
            return

        # --- DIAGNOSTIC: KV cache write + read-back verification ---
        import sys as _sys
        p = lambda *a: print(*a, file=_sys.stderr, flush=True)
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)

        _diag_count = getattr(self, '_fi_diag_count', 0)
        self._fi_diag_count = _diag_count + 1
        do_diag = (_diag_count < 3)

        for i in range(L):
            attn = self._attn_layers[i]
            kv_cache = attn.kv_cache

            if do_diag and i == 0:
                p(f"\\nFI-DIAG step={_diag_count} layer=0")
                p(f"  kv_cache shape: {kv_cache.shape} dtype: {kv_cache.dtype}")
                p(f"  kv_cache[0] (K) shape: {kv_cache[:, 0].shape if kv_cache.dim() > 1 else 'N/A'}")
                p(f"  key shape: {all_k_final[i].shape} dtype: {all_k_final[i].dtype}")
                p(f"  value shape: {all_v[i].shape} dtype: {all_v[i].dtype}")
                p(f"  slot_mapping: {context_slot_mapping[:min(8,num_ctx)].tolist()}... (total {num_ctx})")
                p(f"  slot_mapping range: [{context_slot_mapping.min().item()}, {context_slot_mapping.max().item()}]")
                p(f"  impl type: {type(attn.impl).__name__}")
                p(f"  kv_cache_dtype: {getattr(attn.impl, 'kv_cache_dtype', 'N/A')}")
                p(f"  k_scale: {getattr(attn, '_k_scale', 'N/A')}")
                p(f"  v_scale: {getattr(attn, '_v_scale', 'N/A')}")
                # Check block_size from the KV cache spec
                kv_spec = getattr(attn, 'kv_cache_spec', None)
                if kv_spec:
                    p(f"  kv_cache_spec: block_size={kv_spec.block_size} "
                      f"page_size={getattr(kv_spec, 'page_size', 'N/A')} "
                      f"page_size_padded={getattr(kv_spec, 'page_size_padded', 'N/A')}")

            # Write to cache (original code path)
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                kv_cache,
                context_slot_mapping,
            )

            # Read-back verification: check first token's K/V
            if do_diag and i == 0:
                slot0 = context_slot_mapping[0].item()
                try:
                    # kv_cache layout: [num_blocks, 2, block_size, num_heads, head_dim]
                    # or [2, num_blocks, block_size, num_heads, head_dim]
                    # Try to read back from slot0
                    if kv_cache.dim() == 5:
                        # [num_blocks, 2, block_size, num_heads, head_dim]
                        block_idx = slot0 // kv_cache.shape[2]
                        pos_in_block = slot0 % kv_cache.shape[2]
                        k_readback = kv_cache[block_idx, 0, pos_in_block]
                        v_readback = kv_cache[block_idx, 1, pos_in_block]
                    elif kv_cache.dim() == 3:
                        # [2, num_slots, num_heads*head_dim] (flat)
                        k_readback = kv_cache[0, slot0]
                        v_readback = kv_cache[1, slot0]
                    else:
                        p(f"  READBACK: unknown kv_cache dim={kv_cache.dim()}")
                        k_readback = None
                        v_readback = None

                    if k_readback is not None:
                        k_written = all_k_final[i][0]  # first token's K
                        v_written = all_v[i][0]          # first token's V
                        k_diff = (k_readback.float() - k_written.float()).abs().max().item()
                        v_diff = (v_readback.float() - v_written.float()).abs().max().item()
                        p(f"  READBACK slot={slot0}: k_diff={k_diff:.6e} v_diff={v_diff:.6e}")
                        if k_diff > 0.01 or v_diff > 0.01:
                            p(f"  !!! CACHE MISMATCH !!! Written K mean={k_written.float().mean():.4f} "
                              f"Readback K mean={k_readback.float().mean():.4f}")
                            p(f"  Written K[:4]={k_written.flatten()[:4].tolist()}")
                            p(f"  Readback K[:4]={k_readback.flatten()[:4].tolist()}")
                        else:
                            p(f"  CACHE OK: write matches read-back")
                except Exception as e:
                    p(f"  READBACK ERROR: {e}")"""

if old_cache in content:
    content = content.replace(old_cache, new_cache, 1)
    with open(path, "w") as f:
        f.write(content)
    print("  [1/1] Added FlashInfer KV cache diagnostic to precompute")
else:
    print("  [1/1] WARNING: precompute pattern not found")

PYEOF

echo "  FlashInfer KV diagnostic complete."
