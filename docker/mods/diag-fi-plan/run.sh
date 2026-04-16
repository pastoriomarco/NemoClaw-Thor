#!/bin/bash
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
FI="$SITE_PACKAGES/vllm/v1/attention/backends/flashinfer.py"
python3 - "$FI" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f: c = f.read()
old = """                    prefill_wrapper.plan(
                        qo_indptr=qo_indptr_prefill_cpu,
                        paged_kv_indptr=paged_kv_indptr_prefill_cpu,
                        paged_kv_indices=paged_kv_indices,
                        paged_kv_last_page_len=paged_kv_last_page_len_prefill_cpu,"""
new = """                    import sys as _s
                    print(f"FI-PLAN: qo_indptr={qo_indptr_prefill_cpu.tolist()} "
                          f"kv_indptr={paged_kv_indptr_prefill_cpu.tolist()} "
                          f"kv_indices={paged_kv_indices.tolist()[:10]} "
                          f"last_page_len={paged_kv_last_page_len_prefill_cpu.tolist()} "
                          f"page_size={self.page_size} causal={common_attn_metadata.causal} "
                          f"num_qo={self.num_qo_heads} num_kv={self.num_kv_heads} hd={self.head_dim}",
                          file=_s.stderr, flush=True)
                    prefill_wrapper.plan(
                        qo_indptr=qo_indptr_prefill_cpu,
                        paged_kv_indptr=paged_kv_indptr_prefill_cpu,
                        paged_kv_indices=paged_kv_indices,
                        paged_kv_last_page_len=paged_kv_last_page_len_prefill_cpu,"""
if old in c and "FI-PLAN" not in c:
    c = c.replace(old, new, 1)
    with open(path, "w") as f: f.write(c)
    print("  Patched FlashInfer plan logging")
else:
    print("  SKIP plan patch")
PYEOF
echo "  FlashInfer plan diagnostic complete."
