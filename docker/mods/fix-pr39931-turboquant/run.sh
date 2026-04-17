#!/bin/bash
# Port PR #39931 (TurboQuant for hybrid models) into the installed vLLM.
#
# Background:
# The v6 container image tried to apply PR #39931 at build time, but the
# Dockerfile's git apply fell back to --reject on context conflicts and
# silently dropped hunks. Only TQFullAttentionSpec (a new class with no
# context to conflict) landed. The 4 in-place edits to existing files
# were all rejected:
#   - vllm/engine/arg_utils.py (gate + call-site signature)
#   - vllm/model_executor/layers/quantization/turboquant/config.py (hybrid-aware boundary)
#   - vllm/platforms/interface.py (TQ-aware page-size alignment)
#   - vllm/v1/attention/backends/turboquant_attn.py (flash_attn `out=` compat shim)
#
# Root cause: our image's vLLM main commit (9965f501a) is 22 commits BEHIND
# the PR's base commit (bf9a5ddb24). The diff's context lines don't match
# our older vLLM state, so git apply rejects the hunks.
#
# This mod replays the exact PR changes via Python string replacements,
# verifying each replacement happened and erroring out if the vLLM version
# drifts and any pattern doesn't match.

set -euo pipefail

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
VLLM="${SITE_PACKAGES}/vllm"

ARG_UTILS="${VLLM}/engine/arg_utils.py"
TQ_CONFIG="${VLLM}/model_executor/layers/quantization/turboquant/config.py"
PLATFORMS="${VLLM}/platforms/interface.py"
TQ_ATTN="${VLLM}/v1/attention/backends/turboquant_attn.py"

echo "Applying PR #39931 (TurboQuant hybrid support) — replaying rejected hunks..."

python3 - "${ARG_UTILS}" "${TQ_CONFIG}" "${PLATFORMS}" "${TQ_ATTN}" <<'PYEOF'
import sys
from pathlib import Path

arg_utils_path, tq_config_path, platforms_path, tq_attn_path = [Path(p) for p in sys.argv[1:5]]


def apply_patch(path: Path, old: str, new: str, name: str, *, idempotent_marker: str = "") -> bool:
    """Apply a verbatim old→new replacement. Returns True if something changed."""
    src = path.read_text(encoding="utf-8")
    if idempotent_marker and idempotent_marker in src:
        print(f"  [{name}] already patched — skipping")
        return False
    if old not in src:
        raise RuntimeError(f"[{name}] pattern not found in {path} — vLLM version may have drifted")
    new_src = src.replace(old, new, 1)
    if old in new_src:
        raise RuntimeError(f"[{name}] pattern matched more than once — refusing ambiguous patch")
    path.write_text(new_src, encoding="utf-8")
    print(f"  [{name}] patched {path}")
    return True


# HUNK 1: arg_utils.py — remove gate, change get_boundary_skip_layers call
arg_old = (
    '        # TurboQuant: auto-skip first/last 2 layers (boundary protection).\n'
    '        # These layers are most sensitive to quantization error.\n'
    '        # Users can add extra layers via --kv-cache-dtype-skip-layers.\n'
    '        if resolved_cache_dtype.startswith("turboquant_"):\n'
    '            if model_config.is_hybrid:\n'
    '                raise NotImplementedError(\n'
    '                    "TurboQuant KV cache is not supported for hybrid "\n'
    '                    "(attention + Mamba) models. Boundary layer protection "\n'
    '                    "requires uniform attention layers."\n'
    '                )\n'
    '            from vllm.model_executor.layers.quantization.turboquant.config import (\n'
    '                TurboQuantConfig,\n'
    '            )\n'
    '\n'
    '            num_layers = model_config.hf_text_config.num_hidden_layers\n'
    '            boundary = TurboQuantConfig.get_boundary_skip_layers(num_layers)\n'
    '            existing = set(cache_config.kv_cache_dtype_skip_layers)\n'
    '            merged = sorted(existing | set(boundary), key=lambda x: int(x))\n'
    '            cache_config.kv_cache_dtype_skip_layers = merged\n'
    '            logger.info(\n'
    '                "TQ: skipping layers %s for boundary protection (num_layers=%d)",\n'
    '                merged,\n'
    '                num_layers,\n'
    '            )\n'
)
arg_new = (
    '        if resolved_cache_dtype.startswith("turboquant_"):\n'
    '            from vllm.model_executor.layers.quantization.turboquant.config import (\n'
    '                TurboQuantConfig,\n'
    '            )\n'
    '\n'
    '            boundary = TurboQuantConfig.get_boundary_skip_layers(model_config)\n'
    '            existing = set(cache_config.kv_cache_dtype_skip_layers)\n'
    '            cache_config.kv_cache_dtype_skip_layers = sorted(\n'
    '                existing | set(boundary), key=int\n'
    '            )\n'
)
apply_patch(
    arg_utils_path, arg_old, arg_new, "arg_utils.py",
    idempotent_marker="get_boundary_skip_layers(model_config)",
)


# HUNK 2a: turboquant/config.py — imports + logger
cfg_old_1 = (
    '# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n'
    '"""TurboQuant configuration."""\n'
    '\n'
    'import math\n'
    'from dataclasses import dataclass\n'
)
cfg_new_1 = (
    '# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n'
    '"""TurboQuant configuration."""\n'
    '\n'
    'from __future__ import annotations\n'
    '\n'
    'import logging\n'
    'import math\n'
    'from dataclasses import dataclass\n'
    'from typing import TYPE_CHECKING\n'
    '\n'
    'if TYPE_CHECKING:\n'
    '    from vllm.config import ModelConfig\n'
    '\n'
    'logger = logging.getLogger(__name__)\n'
)
apply_patch(
    tq_config_path, cfg_old_1, cfg_new_1, "config.py imports",
    idempotent_marker="logger = logging.getLogger(__name__)",
)


# HUNK 2b: turboquant/config.py — rewrite get_boundary_skip_layers
cfg_old_2 = (
    '    @staticmethod\n'
    '    def get_boundary_skip_layers(num_layers: int, n: int = 2) -> list[str]:\n'
    '        """Get layer indices to skip TQ compression (boundary protection).\n'
    '\n'
    '        Returns first N and last N layer indices as strings, suitable for\n'
    '        kv_cache_dtype_skip_layers.\n'
    '        """\n'
)
cfg_new_2 = (
    '    @staticmethod\n'
    '    def get_boundary_skip_layers(\n'
    '        model_config: ModelConfig,\n'
    '        n: int = 2,\n'
    '    ) -> list[str]:\n'
    '        """Layer indices to skip TQ compression (boundary protection).\n'
    '\n'
    '        For hybrid models (attention + Mamba/linear-attention), boundary\n'
    '        protection is disabled — hybrids typically have only 8-12\n'
    '        full-attention layers and a hard n=2 on each side would cover\n'
    '        ~40 % of them.  The dense GSM8K baselines that motivate n=2\n'
    "        don't apply to hybrids.\n"
    '\n'
    '        For dense models, skips first N and last N attention layers.\n'
    '        Empirically required for aggressive presets (k3v4_nc, 3bit_nc)\n'
    '        — without it GSM8K drops ~30 points on Qwen3-4B.\n'
    '        """\n'
    '        if model_config.is_hybrid:\n'
    '            attn_indices = _get_full_attention_layer_indices(model_config)\n'
    '            if not attn_indices:\n'
    '                raise NotImplementedError(\n'
    '                    "TurboQuant KV cache requires identifiable "\n'
    '                    "full-attention layers, but none were found in "\n'
    '                    "the hybrid model config."\n'
    '                )\n'
    '            logger.info("TQ hybrid: full-attention layers %s", attn_indices)\n'
    '            return []\n'
    '\n'
    '        num_layers = model_config.hf_text_config.num_hidden_layers\n'
)
apply_patch(
    tq_config_path, cfg_old_2, cfg_new_2, "config.py get_boundary_skip_layers",
    idempotent_marker="TQ hybrid: full-attention layers",
)


# HUNK 2c: turboquant/config.py — from_cache_dtype annotation
cfg_old_3 = '    def from_cache_dtype(cache_dtype: str, head_dim: int) -> "TurboQuantConfig":\n'
cfg_new_3 = '    def from_cache_dtype(cache_dtype: str, head_dim: int) -> TurboQuantConfig:\n'
apply_patch(
    tq_config_path, cfg_old_3, cfg_new_3, "config.py from_cache_dtype",
    idempotent_marker='def from_cache_dtype(cache_dtype: str, head_dim: int) -> TurboQuantConfig:',
)


# HUNK 2d: turboquant/config.py — append _get_full_attention_layer_indices helper
helper_marker = "def _get_full_attention_layer_indices("
cfg_src = tq_config_path.read_text(encoding="utf-8")
if helper_marker not in cfg_src:
    helper_code = (
        '\n\n'
        'def _get_full_attention_layer_indices(model_config: ModelConfig) -> list[int]:\n'
        '    """Global indices of full-attention layers in a hybrid model.\n'
        '\n'
        '    Covers the conventions used across vLLM: ``layer_types`` (Qwen3.5/Next),\n'
        '    ``layers_block_type`` (Jamba/Zamba2), ``attn_type_list`` (Minimax).\n'
        '    """\n'
        '    text_cfg = model_config.hf_text_config\n'
        '    hf_cfg = model_config.hf_config\n'
        '\n'
        '    layer_types = getattr(text_cfg, "layer_types", None)\n'
        '    if layer_types is not None:\n'
        '        return [\n'
        '            i for i, t in enumerate(layer_types) if t in ("full_attention", "attention")\n'
        '        ]\n'
        '\n'
        '    layers_block_type = getattr(text_cfg, "layers_block_type", None)\n'
        '    if layers_block_type is not None:\n'
        '        return [\n'
        '            i for i, t in enumerate(layers_block_type) if t in ("attention", "hybrid")\n'
        '        ]\n'
        '\n'
        '    attn_type_list = getattr(hf_cfg, "attn_type_list", None)\n'
        '    if attn_type_list is not None:\n'
        '        return [i for i, t in enumerate(attn_type_list) if t == 1]\n'
        '\n'
        '    return []\n'
    )
    tq_config_path.write_text(cfg_src.rstrip() + helper_code, encoding="utf-8")
    print(f"  [config.py helper] appended _get_full_attention_layer_indices")
else:
    print(f"  [config.py helper] already present")


# HUNK 3: platforms/interface.py — TQ-aware page-size alignment
plat_old = (
    '                dtype=kv_cache_dtype,\n'
    '                kv_quant_mode=kv_quant_mode,\n'
    '            ).page_size_bytes\n'
    '        else:\n'
    '            attn_page_size_1_token = FullAttentionSpec(\n'
)
plat_new = (
    '                dtype=kv_cache_dtype,\n'
    '                kv_quant_mode=kv_quant_mode,\n'
    '            ).page_size_bytes\n'
    '        elif cache_config.cache_dtype.startswith("turboquant_"):\n'
    '            # TQ has a packed K|V layout; the standard FullAttentionSpec\n'
    '            # formula over-sizes it and trips unify_kv_cache_spec_page_size\n'
    '            # when all attention layers are TQ. With mixed skip+TQ the skip\n'
    '            # layers still use the standard layout — take max so mamba\n'
    '            # padding covers the largest actual page.\n'
    '            from vllm.model_executor.layers.quantization.turboquant.config import (\n'
    '                TurboQuantConfig,\n'
    '            )\n'
    '            from vllm.v1.kv_cache_interface import TQFullAttentionSpec\n'
    '\n'
    '            tq_cfg = TurboQuantConfig.from_cache_dtype(\n'
    '                cache_config.cache_dtype, model_config.get_head_size()\n'
    '            )\n'
    '            tq_page = TQFullAttentionSpec(\n'
    '                block_size=1,\n'
    '                num_kv_heads=model_config.get_num_kv_heads(parallel_config),\n'
    '                head_size=model_config.get_head_size(),\n'
    '                head_size_v=model_config.get_head_size(),\n'
    '                dtype=kv_cache_dtype,\n'
    '                kv_quant_mode=kv_quant_mode,\n'
    '                tq_slot_size=tq_cfg.slot_size_aligned,\n'
    '            ).page_size_bytes\n'
    '            if cache_config.kv_cache_dtype_skip_layers:\n'
    '                skip_page = FullAttentionSpec(\n'
    '                    block_size=1,\n'
    '                    num_kv_heads=model_config.get_num_kv_heads(parallel_config),\n'
    '                    head_size=model_config.get_head_size(),\n'
    '                    dtype=model_config.dtype,\n'
    '                ).page_size_bytes\n'
    '                attn_page_size_1_token = max(tq_page, skip_page)\n'
    '            else:\n'
    '                attn_page_size_1_token = tq_page\n'
    '        else:\n'
    '            attn_page_size_1_token = FullAttentionSpec(\n'
)
apply_patch(
    platforms_path, plat_old, plat_new, "platforms/interface.py",
    idempotent_marker="TQ has a packed K|V layout",
)


# HUNK 4: turboquant_attn.py — flash_attn `out=` kwarg compat shim
attn_old = (
    '_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()\n'
    'if _HAS_FLASH_ATTN:\n'
    '    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func\n'
)
attn_new = (
    '_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()\n'
    'if _HAS_FLASH_ATTN:\n'
    '    import inspect as _inspect\n'
    '\n'
    '    from vllm.v1.attention.backends.fa_utils import (\n'
    '        flash_attn_varlen_func as _flash_attn_varlen_func,\n'
    '    )\n'
    '\n'
    '    # Upstream flash-attn on ROCm lacks the `out=` kwarg; detect once.\n'
    '    try:\n'
    '        _FA_SUPPORTS_OUT = (\n'
    '            "out" in _inspect.signature(_flash_attn_varlen_func).parameters\n'
    '        )\n'
    '    except (TypeError, ValueError):\n'
    '        _FA_SUPPORTS_OUT = False\n'
    '\n'
    '    def flash_attn_varlen_func(*args, out=None, **kwargs):\n'
    '        kwargs.pop("out", None)\n'
    '        if _FA_SUPPORTS_OUT and out is not None:\n'
    '            kwargs["out"] = out\n'
    '            return _flash_attn_varlen_func(*args, **kwargs)\n'
    '        result = _flash_attn_varlen_func(*args, **kwargs)\n'
    '        if out is not None:\n'
    '            out.copy_(result)\n'
    '            return out\n'
    '        return result\n'
    '\n'
)
apply_patch(
    tq_attn_path, attn_old, attn_new, "turboquant_attn.py",
    idempotent_marker="_FA_SUPPORTS_OUT",
)


print("All PR #39931 hunks applied successfully.")
PYEOF
