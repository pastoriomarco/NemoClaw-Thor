# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[hc-fp8] QWEN4EXP_HC_FP8: online 8-bit for the target's hyperconnection GEMMs.

Off by default. With ``QWEN4EXP_HC_FP8=1`` the target model's hyperconnection
projections (per layer ``attn_hyper_connection`` and ``mlp_hyper_connection``:
the merged ``input_mix_weight_down_block_inject`` and ``input_mix_weight_up``;
the final ``hyper_connection_mixer``: ``input_mix_weight_down`` and
``input_mix_weight_up``) get the stock online methods of
``QWEN4EXP_HC_FP8_FORMAT="<down>[,<up>]"`` (default ``fp8_per_block,mxfp8``)
instead of BF16. Both checkpoints store them in BF16
and the stock model builds them with ``quant_config=None``, so this works the
same on FP4-Local and on NVIDIA NVFP4. The drafter's mixers (``mtp.*``) stay
BF16. The norms, combine and gate mix are elementwise and unchanged.

The converted merged down+inject layer runs through its Linear module (FP8
GEMM) plus the stock ``hc_silu``, not through the fused BF16 CuTeDSL
down+SiLU kernel, which needs a BF16 weight.
"""

import functools
import os
import re

import torch

from vllm.config.quantization import QuantizationConfigArgs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)

from .draft_head import _FP8_METHODS, _fp8_bytes

logger = init_logger(__name__)

# Runtime names of the target's HC projections, e.g.
# language_model.model.layers.7.mlp_hyper_connection.input_mix_weight_up and
# language_model.model.hyper_connection_mixer.input_mix_weight_down. The drafter
# ("mtp.layers.48.…", "mtp.hyper_connection_mixer.…") never matches.
HC_FP8_TARGET = (
    r"re:(.*\.)?model\.(layers\.\d+\.(attn|mlp)_hyper_connection"
    r"|hyper_connection_mixer)\."
    r"(input_mix_weight_down_block_inject|input_mix_weight_down|input_mix_weight_up)$"
)
_HC_FP8_RE = re.compile(HC_FP8_TARGET[len("re:") :])
# Down (+inject) and up projections, quantized per kind: fp8_per_block needs
# K % 128 == 0 (up has K=320), mxfp8 needs N % 32 == 0 (down has N=336).
HC_DOWN_TARGET = (
    r"re:(.*\.)?model\.(layers\.\d+\.(attn|mlp)_hyper_connection"
    r"|hyper_connection_mixer)\.(input_mix_weight_down_block_inject|input_mix_weight_down)$"
)
HC_UP_TARGET = (
    r"re:(.*\.)?model\.(layers\.\d+\.(attn|mlp)_hyper_connection"
    r"|hyper_connection_mixer)\.input_mix_weight_up$"
)


def hc_fp8_enabled() -> bool:
    return os.environ.get("QWEN4EXP_HC_FP8", "0") not in ("", "0")


@functools.cache
def _hc_quant_config(down: str, up: str) -> OnlineQuantizationConfig:
    # One stock config shared by all HC layers: it records the matched
    # prefixes and the online quantization time.
    return OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={HC_DOWN_TARGET: down, HC_UP_TARGET: up})
    )


def _hc_formats() -> tuple[str, str]:
    """(down, up) formats from QWEN4EXP_HC_FP8_FORMAT="<down>,<up>" (one value =
    both). Default fp8_per_block,mxfp8: the only pair whose kernels accept both
    shapes (down 336 x 10240, up 10240 x 320)."""
    raw = os.environ.get("QWEN4EXP_HC_FP8_FORMAT") or "fp8_per_block,mxfp8"
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) == 1:
        parts *= 2
    for fmt in parts:
        if len(parts) != 2 or fmt not in _FP8_METHODS:
            raise ValueError(
                "QWEN4EXP_HC_FP8_FORMAT must be <down>[,<up>] from "
                f"{sorted(_FP8_METHODS)}, got {raw!r}"
            )
    return parts[0], parts[1]


def hc_fp8_quant_config(prefix: str) -> OnlineQuantizationConfig | None:
    """The quant_config for the HC Linear at ``prefix``: None (stock BF16) unless
    QWEN4EXP_HC_FP8 is set and the prefix is a target HC projection."""
    if not hc_fp8_enabled() or not _HC_FP8_RE.match(prefix):
        return None
    return _hc_quant_config(*_hc_formats())


def load_hc_pad(linear: torch.nn.Module, rows: int) -> None:
    """Load zeros into the merged down+inject layer's alignment pad rows (its last
    logical shard, never in the checkpoint). The online loader then sees every
    row loaded and quantizes at load, and the 128-row scale block that holds the
    pad never takes its amax from uninitialized memory."""
    weight = linear.weight
    zeros = torch.zeros(rows, weight.shape[1], dtype=weight.dtype, device="cpu")
    weight.weight_loader(weight, zeros, len(linear.output_sizes) - 1)
    linear.hc_fp8_pad_rows = rows


def _hc_kind(name: str) -> str:
    """e.g. mlp_hyper_connection.input_mix_weight_up, hyper_connection_mixer.…"""
    parts = name.split(".")
    return ".".join(parts[-2:])


def log_hc_fp8(model: torch.nn.Module) -> None:
    """One line: the HC matrices converted per kind, the weight bytes one forward
    step reads before (BF16, pad rows excluded as the fused kernel skips them)
    and after, and the GEMM kernel."""
    if not hc_fp8_enabled():
        return
    down_fmt, up_fmt = _hc_formats()
    methods = {_FP8_METHODS[down_fmt]: down_fmt, _FP8_METHODS[up_fmt]: up_fmt}
    counts: dict[str, int] = {}
    kernels: set[str] = set()
    bf16 = fp8 = 0
    for name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if type(quant_method) not in methods or "hyper_connection" not in name:
            continue
        fmt = down_fmt if "input_mix_weight_down" in name else up_fmt
        n = module.output_size_per_partition
        k = module.input_size_per_partition
        kind = _hc_kind(name)
        counts[kind] = counts.get(kind, 0) + 1
        bf16 += 2 * (n - getattr(module, "hc_fp8_pad_rows", 0)) * k
        fp8 += _fp8_bytes(fmt, n, k)
        kernel = getattr(quant_method, "kernel", None) or getattr(
            quant_method, "fp8_linear", None
        )
        kernels.add(type(kernel).__name__)
    logger.info(
        "[hc-fp8] Hyperconnection GEMMs: %d matrices to online %s (%s): "
        "%.3f GB BF16 -> %.3f GB per step via %s",
        sum(counts.values()),
        f"{down_fmt} (down) / {up_fmt} (up)",
        ", ".join(f"{kind} {num}" for kind, num in sorted(counts.items())),
        bf16 / 1e9,
        fp8 / 1e9,
        "/".join(sorted(kernels)) or "-",
    )
