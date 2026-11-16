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
``QWEN4EXP_MTP_DENSE_FP8=1``
    Online 8-bit (QWEN4EXP_FP8_FORMAT, default blockwise FP8) for the BF16
    dense projections of the MTP drafter (fc_embedding, fc_hidden, QSA qkv/o,
    shared expert). Drafter only. Named QWEN4EXP_MTP_DENSE_MXFP8 before
    2026-11-16.
``QWEN4EXP_SIDE_FP8=1`` ([side-fp8])
    Online FP8 for the target model's dense side layers that a ModelOpt
    MIXED_PRECISION checkpoint leaves in BF16 (GDN in/out, QSA qkv/o, shared
    expert; side_fp8_target()). A no-op when the checkpoint already quantizes
    them (FP4-Local ships them as MXFP8). Changes target outputs slightly.
``QWEN4EXP_QSA_UNFUSE_INDEXER=1`` ([qsa-unfuse])
    The target's QSA indexer projection stays its own linear instead of
    sharing the qkv_proj GEMM, so QWEN4EXP_SIDE_FP8_SKIP can keep only it in
    BF16 (self_attn.indexer.index_qk_proj). Exact by itself.

``QWEN4EXP_FP8_FORMAT`` picks the 8-bit format of all of the above.
"""

import math
import os
import re

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


# [side-fp8] Target dense side layers that the NVIDIA ModelOpt checkpoint leaves
# in BF16: the class the earlier RadixArk FP8 hybrid converted (GDN in_proj_qkv,
# in_proj_z, out_proj; QSA q/k/v/o; shared-expert gate/up/down), named as the
# runtime (fused) modules. The QSA qkv_proj GEMM also carries the indexer's
# index_qk_proj unless QWEN4EXP_QSA_UNFUSE_INDEXER. Not matched: the GDN in_proj_ba (N=96), router mlp.gate,
# shared_expert_gate, HC mixers, PLE kv_proj, experts (already NVFP4),
# embed_tokens, lm_head, and the drafter ("mtp.layers.N", see MTP_DENSE_TARGET).
SIDE_FP8_KINDS = (
    "linear_attn.in_proj_qkvz",
    "linear_attn.out_proj",
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "mlp.shared_expert.gate_up_proj",
    "mlp.shared_expert.down_proj",
)
# [qsa-unfuse] With QWEN4EXP_QSA_UNFUSE_INDEXER the indexer projection is its own
# runtime module and side-layer kind (it must not be a target while it is fused:
# the online matcher would then see an unmatched shard of qkv_proj).
QSA_INDEXER_KIND = "self_attn.indexer.index_qk_proj"


def side_fp8_target() -> str:
    """[side-fp8] The online-FP8 target regex: SIDE_FP8_KINDS (plus the unfused
    QSA indexer) minus QWEN4EXP_SIDE_FP8_SKIP, a comma-separated list of kinds
    to keep in BF16 (e.g. "self_attn.qkv_proj": the QSA projections, with the
    fused indexer; with the indexer unfused, "self_attn.indexer.index_qk_proj"
    keeps only the indexer)."""
    kinds = SIDE_FP8_KINDS + ((QSA_INDEXER_KIND,) if qsa_unfuse_indexer() else ())
    skip = {
        k.strip()
        for k in os.environ.get("QWEN4EXP_SIDE_FP8_SKIP", "").split(",")
        if k.strip()
    }
    if QSA_INDEXER_KIND in skip and QSA_INDEXER_KIND not in kinds:
        raise ValueError(
            f"QWEN4EXP_SIDE_FP8_SKIP={QSA_INDEXER_KIND} needs "
            "QWEN4EXP_QSA_UNFUSE_INDEXER=1 (the indexer is fused into "
            "self_attn.qkv_proj otherwise)."
        )
    if skip - set(kinds):
        raise ValueError(
            f"QWEN4EXP_SIDE_FP8_SKIP has unknown kinds {sorted(skip - set(kinds))}; "
            f"valid: {', '.join(kinds)}"
        )
    return (
        r"re:(.*\.)?model\.layers\.\d+\.("
        + "|".join(re.escape(k) for k in kinds if k not in skip)
        + r")$"
    )


# The same class as checkpoint (or fused) names, outside the drafter. If the
# checkpoint already quantizes any of them, QWEN4EXP_SIDE_FP8 is a no-op:
# an online target on a quantized layer would requantize it.
_SIDE_CHECKPOINT_NAME = re.compile(
    r"(?:^|(?<!mtp)\.)layers\.\d+\."
    r"(linear_attn\.(in_proj_qkvz?|in_proj_z|out_proj)"
    r"|self_attn\.(qkv_proj|q_proj|k_proj|v_proj|o_proj|indexer\.index_qk_proj)"
    r"|mlp\.shared_expert\.(gate_up_proj|gate_proj|up_proj|down_proj))$"
)


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0") not in ("", "0")


# QWEN4EXP_FP8_FORMAT picks the 8-bit format for the head, the MTP dense
# projections and the target side layers: "fp8_per_block" (128x128 weight
# blocks, per-token 1x128 dynamic activations; the format the earlier
# Flash-Next recipe used) or "mxfp8".
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
    """Compose stock online 8-bit (QWEN4EXP_FP8_FORMAT) onto the drafter's
    checkpoint quant config for the BF16 dense MTP projections
    (QWEN4EXP_MTP_DENSE_FP8)."""
    if not _enabled("QWEN4EXP_MTP_DENSE_FP8"):
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


def maybe_quantize_side_layers(quant_config: QuantizationConfig | None) -> bool:
    """[side-fp8] Compose stock online FP8 onto the target's checkpoint quant
    config for the BF16 dense side layers (QWEN4EXP_SIDE_FP8). Returns whether
    it did; call before the decoder layers are built."""
    if not _enabled("QWEN4EXP_SIDE_FP8"):
        return False
    name = None if quant_config is None else quant_config.get_name()
    if name != "modelopt_mixed":
        logger.info(
            "[side-fp8] QWEN4EXP_SIDE_FP8 is a no-op: it needs a ModelOpt "
            "MIXED_PRECISION checkpoint, this one is %s.",
            name,
        )
        return False
    quantized_layers = getattr(quant_config, "quantized_layers", None) or {}
    already = sorted(k for k in quantized_layers if _SIDE_CHECKPOINT_NAME.search(k))
    if already:
        algos = sorted({str(quantized_layers[k].get("quant_algo")) for k in already})
        logger.info(
            "[side-fp8] QWEN4EXP_SIDE_FP8 is a no-op: the checkpoint already "
            "quantizes %d side-layer matrices (%s, e.g. %s).",
            len(already),
            "/".join(algos),
            already[0],
        )
        return False
    if quant_config.online_quantization_config is not None:
        raise ValueError(
            "QWEN4EXP_SIDE_FP8 cannot be combined with --quantization-config: "
            "the target already has an online quantization config."
        )
    target = side_fp8_target()
    quant_config.online_quantization_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={target: _fp8_format()})
    )
    logger.info(
        "[side-fp8] Target side layers: online %s (%s)",
        _fp8_format(),
        target,
    )
    return True


def qsa_unfuse_indexer() -> bool:
    """[qsa-unfuse] QWEN4EXP_QSA_UNFUSE_INDEXER: the target's QSA layers keep
    indexer.index_qk_proj as its own linear (qsa.py's unfused LoRA layout)."""
    return _enabled("QWEN4EXP_QSA_UNFUSE_INDEXER")


def maybe_unfuse_qsa_indexer(quant_config: QuantizationConfig | None) -> bool:
    """[qsa-unfuse] Returns whether the target's QSA indexer projection stays
    unfused (Qwen4ExpDecoderLayer builds the layers). The quant configs match a
    fused module by its shards, so their qkv_proj entry drops the indexer, as
    the module now holds q/gate, k and v only. Call before the decoder layers
    are built."""
    if not qsa_unfuse_indexer():
        return False
    if quant_config is not None:
        quant_config.packed_modules_mapping = {
            **quant_config.packed_modules_mapping,
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        }
    logger.info(
        "[side-fp8] QSA indexer unfused: target self_attn.qkv_proj holds q/gate, "
        "k, v; self_attn.indexer.index_qk_proj is its own linear."
    )
    return True


def _layer_kind(name: str) -> str:
    """The module name after its layer index, e.g. linear_attn.out_proj."""
    parts = name.split(".")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            return ".".join(parts[i + 1 :])
    return name


def _fp8_bytes(fmt: str, n: int, k: int) -> int:
    """Weight bytes read per GEMM: e4m3 values plus the format's scales."""
    if fmt == "fp8_per_block":
        return n * k + math.ceil(n / 128) * math.ceil(k / 128) * 4  # fp32 128x128
    return n * k + n * math.ceil(k / 32)  # mxfp8: one e8m0 per 32 inputs


def log_side_layers(model: torch.nn.Module) -> None:
    """[side-fp8] One line: the converted matrices per kind, the weight bytes
    one forward step reads before and after, and the GEMM kernel."""
    fmt = _fp8_format()
    method_cls = _FP8_METHODS[fmt]
    counts: dict[str, int] = {}
    kernels: set[str] = set()
    bf16 = fp8 = 0
    for name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if type(quant_method) is not method_cls:
            continue
        if "hyper_connection" in name:  # [hc-fp8] counted by log_hc_fp8
            continue
        n = module.output_size_per_partition
        k = module.input_size_per_partition
        kind = _layer_kind(name)
        counts[kind] = counts.get(kind, 0) + 1
        bf16 += 2 * n * k
        fp8 += _fp8_bytes(fmt, n, k)
        kernel = getattr(quant_method, "kernel", None) or getattr(
            quant_method, "fp8_linear", None
        )
        kernels.add(type(kernel).__name__)
    logger.info(
        "[side-fp8] Target side layers: %d matrices to online %s (%s): "
        "%.3f GB BF16 -> %.3f GB per step via %s",
        sum(counts.values()),
        fmt,
        ", ".join(f"{kind} {num}" for kind, num in sorted(counts.items())),
        bf16 / 1e9,
        fp8 / 1e9,
        "/".join(sorted(kernels)) or "-",
    )
