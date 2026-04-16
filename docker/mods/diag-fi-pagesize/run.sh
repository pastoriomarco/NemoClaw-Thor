#!/bin/bash
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
FI="$SITE_PACKAGES/vllm/v1/attention/backends/flashinfer.py"
python3 - "$FI" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f: c = f.read()
old = "        self.page_size = self.kv_cache_spec.block_size"
new = """        self.page_size = self.kv_cache_spec.block_size
        import sys as _s
        print(f"FI-INIT: page_size={self.page_size} "
              f"kv_spec={self.kv_cache_spec} "
              f"num_kv_heads={self.kv_cache_spec.num_kv_heads} "
              f"head_size={self.kv_cache_spec.head_size} "
              f"block_size={self.kv_cache_spec.block_size}",
              file=_s.stderr, flush=True)"""
if old in c and "FI-INIT" not in c:
    c = c.replace(old, new, 1)
    with open(path, "w") as f: f.write(c)
    print("  Patched FlashInfer page_size logging")
PYEOF
echo "  FlashInfer page_size diagnostic complete."
