#!/bin/bash
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
QWEN3_DFLASH_PY="$SITE_PACKAGES/vllm/model_executor/models/qwen3_dflash.py"

echo "Applying direct-call + data_ptr diagnostic..."

python3 - "$QWEN3_DFLASH_PY" << 'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f: c = f.read()

# In precompute: log kv_cache data_ptr for layer 0
old_insert = """            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                kv_cache,
                context_slot_mapping,
            )"""

new_insert = """            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                kv_cache,
                context_slot_mapping,
            )
            import sys as _dp
            _dpc = getattr(type(self), '_dpc', 0)
            if i == 0 and _dpc < 3:
                type(self)._dpc = _dpc + 1
                print(f"PRECOMPUTE-PTR layer={attn.layer_name} cache_ptr={kv_cache.data_ptr()} shape={kv_cache.shape}",
                      file=_dp.stderr, flush=True)"""

if old_insert in c:
    c = c.replace(old_insert, new_insert, 1)
    print("  [1/2] Added precompute data_ptr logging")

# In forward: force use_direct_call and log data_ptr
old_fwd = """        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output"""

new_fwd = '''        # Force use_direct_call=True to bypass custom op boundary
        _orig_dc = self.attn.use_direct_call
        self.attn.use_direct_call = True

        import sys as _df
        _dfn = getattr(DFlashQwen3Attention, '_dfn', 0)
        kv_cache = self.attn.kv_cache
        if _dfn < 3 and kv_cache is not None and kv_cache.numel() > 0:
            from vllm.forward_context import get_forward_context
            fwd_ctx = get_forward_context()
            _has = self.attn.layer_name in (fwd_ctx.no_compile_layers or {})
            if _has:
                resolved_layer = fwd_ctx.no_compile_layers[self.attn.layer_name]
                resolved_cache = resolved_layer.kv_cache
                print(f"FORWARD-PTR layer={self.attn.layer_name} "
                      f"self.cache_ptr={kv_cache.data_ptr()} "
                      f"resolved_ptr={resolved_cache.data_ptr()} "
                      f"MATCH={kv_cache.data_ptr() == resolved_cache.data_ptr()} "
                      f"use_direct_call=True",
                      file=_df.stderr, flush=True)
                DFlashQwen3Attention._dfn = _dfn + 1

        attn_output = self.attn(q, k, v)
        self.attn.use_direct_call = _orig_dc

        output, _ = self.o_proj(attn_output)
        return output'''

if old_fwd in c:
    c = c.replace(old_fwd, new_fwd, 1)
    print("  [2/2] Added direct-call + data_ptr diagnostic")

with open(path, 'w') as f: f.write(c)
PYEOF

echo "  Direct-call diagnostic complete."
