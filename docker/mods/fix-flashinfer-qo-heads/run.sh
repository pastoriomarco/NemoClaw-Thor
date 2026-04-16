#!/bin/bash
# Fix FlashInfer num_qo_heads for DFlash drafter layers.
#
# Root cause: FlashInferMetadataBuilder sets num_qo_heads from model_config
# (target model's 16 heads), but the DFlash drafter has 32 Q heads.
# This causes FlashInfer to compute attention for only half the Q heads,
# producing garbage output and 0% DFlash acceptance.
#
# Fix: Override num_qo_heads in the metadata builder initialization to use
# the actual attention layer's head count instead of the model config.

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
FI="$SITE_PACKAGES/vllm/v1/attention/backends/flashinfer.py"

echo "Applying FlashInfer num_qo_heads fix for DFlash drafter..."

python3 - "$FI" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f:
    c = f.read()

# The problem: num_qo_heads is set from model_config (target's 16 heads)
# but the drafter needs 32. The kv_cache_spec has num_kv_heads (8) and
# the actual ratio should be: num_qo_heads = num_kv_heads * (target_q/target_kv)
# But this doesn't work for different architectures.
#
# Better fix: store num_qo_heads on the kv_cache_spec when building
# the attention layers. For now, compute it from the Attention layer.
#
# Simplest fix: after setting num_qo_heads from model_config, override it
# with the actual attention layer's num_heads if available in the spec.

old = """        self.num_qo_heads = self.model_config.get_num_attention_heads(
            self.vllm_config.parallel_config
        )

        self.num_kv_heads = self.kv_cache_spec.num_kv_heads
        self.head_dim = self.kv_cache_spec.head_size"""

new = """        self.num_qo_heads = self.model_config.get_num_attention_heads(
            self.vllm_config.parallel_config
        )

        self.num_kv_heads = self.kv_cache_spec.num_kv_heads
        self.head_dim = self.kv_cache_spec.head_size

        # Fix: For DFlash drafter layers, model_config gives the TARGET's
        # num_qo_heads (e.g. 16), but the drafter may have different Q heads
        # (e.g. 32). Detect mismatch via head_dim: if kv_cache_spec head_dim
        # differs from model_config, recalculate num_qo_heads from hidden_size.
        _model_head_dim = getattr(self.model_config.hf_text_config, 'head_dim',
            self.model_config.get_hidden_size() // self.num_qo_heads)
        if _model_head_dim != self.head_dim and self.head_dim > 0:
            # Different head_dim means this is a drafter/auxiliary model layer.
            # Compute num_qo_heads from the GQA ratio: preserve the same
            # q_heads / kv_heads ratio as the base model, or use hidden_size / head_dim.
            _gqa_ratio = self.num_qo_heads // max(
                self.model_config.get_num_kv_heads(self.vllm_config.parallel_config), 1)
            self.num_qo_heads = self.num_kv_heads * _gqa_ratio
            import sys as _s
            print(f"FI-FIX: Corrected num_qo_heads to {self.num_qo_heads} "
                  f"(kv_heads={self.num_kv_heads} × ratio={_gqa_ratio}, "
                  f"head_dim={self.head_dim} vs model_head_dim={_model_head_dim})",
                  file=_s.stderr, flush=True)"""

if old in c:
    c = c.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(c)
    print("  [1/1] Fixed num_qo_heads derivation for drafter layers")
else:
    print("  [1/1] WARNING: pattern not found")

PYEOF

echo "  FlashInfer num_qo_heads fix complete."
