# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[draft-head] Opt-in byte reductions for the Qwen4Exp LM head and MTP drafter.

All switches default to off, which leaves the stock behaviour unchanged.

``QWEN4EXP_DRAFT_VOCAB=/path/ids.npy``
    The MTP draft step scores only these token ids (a reduced copy of the
    lm_head rows). The target still verifies with the full vocabulary, so the
    output is unchanged and only the acceptance length can drop.
``QWEN4EXP_LM_HEAD_MXFP8=1``
    Online MXFP8 (e4m3 values, one e8m0 scale per 32 inputs, dynamic MXFP8
    activations) for the target lm_head, which the drafter shares, and for the
    reduced draft head. Changes target logits slightly.
``QWEN4EXP_MTP_DENSE_MXFP8=1``
    Online MXFP8 for the BF16 dense projections of the MTP drafter
    (fc_embedding, fc_hidden, QSA qkv/o, shared expert). Drafter only.
"""

import os

import numpy as np
import torch

from vllm.config.quantization import QuantizationConfigArgs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.online.mxfp8 import (
    Mxfp8OnlineLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

logger = init_logger(__name__)

# The reduced head gets a multiple of this many rows: no ParallelLMHead vocab
# padding and aligned GEMM output rows.
DRAFT_VOCAB_ALIGN = 128

# Drafter layers that the checkpoint leaves in BF16 and that run through the
# generic linear path (the HC mixers, router and shared_expert_gate do not).
MTP_DENSE_TARGET = (
    r"re:(.*\.)?mtp\.(fc_embedding|fc_hidden"
    r"|layers\.\d+\.(self_attn\.(qkv_proj|o_proj)"
    r"|mlp\.shared_expert\.(gate_up_proj|down_proj)))$"
)


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0") not in ("", "0")


# QWEN4EXP_FP8_FORMAT picks the 8-bit format for the head and the MTP dense
# projections: "fp8_per_block" (128x128 weight blocks, per-token 1x128 dynamic
# activations; the format the earlier Flash-Next recipe used) or "mxfp8".
_FP8_METHODS = {
    "fp8_per_block": Fp8PerBlockOnlineLinearMethod,
    "mxfp8": Mxfp8OnlineLinearMethod,
}


def _fp8_format() -> str:
    fmt = os.environ.get("QWEN4EXP_FP8_FORMAT", "fp8_per_block")
    if fmt not in _FP8_METHODS:
        raise ValueError(
            f"QWEN4EXP_FP8_FORMAT must be one of {sorted(_FP8_METHODS)}, got {fmt!r}"
        )
    return fmt


def make_lm_head(
    num_embeddings: int,
    hidden_size: int,
    prefix: str,
    quantize_envs: tuple[str, ...] = ("QWEN4EXP_LM_HEAD_MXFP8",),
) -> ParallelLMHead:
    """A ParallelLMHead, online-quantized to QWEN4EXP_FP8_FORMAT when
    QWEN4EXP_LM_HEAD_MXFP8 is set. The online method loads the BF16 rows onto a
    meta weight and quantizes them as soon as they arrive, so no BF16 copy stays
    resident."""
    quant_method = (
        _FP8_METHODS[_fp8_format()]()
        if any(_enabled(name) for name in quantize_envs)
        else None
    )
    lm_head = ParallelLMHead(
        num_embeddings, hidden_size, prefix=prefix, quant_method=quant_method
    )
    if quant_method is not None:
        # vLLM's online loader times the quantization through the layer's
        # quant_config and asserts it is set (model_loader/utils.py,
        # online_quantization_timing_context); a bare quant_method leaves it None.
        lm_head.quant_config = OnlineQuantizationConfig(
            QuantizationConfigArgs(targets={prefix: _fp8_format()})
        )
        logger.info(
            "[draft-head] %s: online %s (%d x %d) via %s",
            prefix,
            _fp8_format(),
            num_embeddings,
            hidden_size,
            type(
                getattr(quant_method, "kernel", None)
                or getattr(quant_method, "fp8_linear", None)
            ).__name__,
        )
    return lm_head


def pad_draft_vocab(ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """Sorted unique ids, topped up with the lowest unused ids (the most frequent
    BPE merges) to a multiple of DRAFT_VOCAB_ALIGN."""
    ids = np.unique(ids.astype(np.int64))
    num_pad = -len(ids) % DRAFT_VOCAB_ALIGN
    if num_pad:
        unused = np.setdiff1d(np.arange(vocab_size), ids, assume_unique=True)
        ids = np.union1d(ids, unused[:num_pad])
    return ids


def load_draft_vocab(vocab_size: int) -> torch.Tensor | None:
    """The reduced draft vocabulary from QWEN4EXP_DRAFT_VOCAB, or None."""
    path = os.environ.get("QWEN4EXP_DRAFT_VOCAB")
    if not path:
        return None
    ids = pad_draft_vocab(np.load(path), vocab_size)
    logger.info(
        "[draft-head] MTP draft vocabulary: %d of %d ids from %s",
        len(ids),
        vocab_size,
        path,
    )
    return torch.as_tensor(ids, device=torch.get_default_device())


def maybe_quantize_mtp_dense(quant_config: QuantizationConfig) -> None:
    """Compose stock online MXFP8 onto the drafter's checkpoint quant config for
    the BF16 dense MTP projections (QWEN4EXP_MTP_DENSE_MXFP8)."""
    if not _enabled("QWEN4EXP_MTP_DENSE_MXFP8"):
        return
    quant_config.online_quantization_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={MTP_DENSE_TARGET: _fp8_format()})
    )
    logger.info(
        "[draft-head] MTP dense projections: online %s (%s)",
        _fp8_format(),
        MTP_DENSE_TARGET,
    )


def log_mtp_dense(quant_config: QuantizationConfig | None) -> None:
    """Log the drafter layers that maybe_quantize_mtp_dense matched."""
    online = getattr(quant_config, "online_quantization_config", None)
    if online is not None:
        logger.info(
            "[draft-head] MTP online %s layers: %s",
            _fp8_format(),
            "; ".join(online.quantized_layer_summaries),
        )
