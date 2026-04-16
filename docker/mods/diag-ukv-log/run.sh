#!/bin/bash
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "Adding UKV debug logging..."
python3 /tmp/test_kv_update.py "$SITE_PACKAGES" 2>/dev/null || python3 - "$SITE_PACKAGES" <<'PYEOF'
import sys
SITE_PACKAGES = sys.argv[1]
path = f"{SITE_PACKAGES}/vllm/model_executor/layers/attention/attention.py"
with open(path) as f: c = f.read()
old = """    if layer_slot_mapping is not None:
        assert hasattr(attn_layer.impl, "do_kv_cache_update"), ("""
new = """    import sys as _u; _un=globals().get('_uc',0)
    if _un<20 and 'layers.3' in str(layer_name):
        globals()['_uc']=_un+1
        print(f"UKV {layer_name}: sm={'None' if layer_slot_mapping is None else layer_slot_mapping.shape} cache={kv_cache.shape if kv_cache.numel()>0 else 'empty'}",file=_u.stderr,flush=True)
    if layer_slot_mapping is not None:
        assert hasattr(attn_layer.impl, "do_kv_cache_update"), ("""
if old in c:
    c = c.replace(old, new, 1)
    with open(path, 'w') as f: f.write(c)
    print("  Added UKV logging")
else:
    print("  WARNING: pattern not found")
PYEOF
echo "  UKV logging complete."
