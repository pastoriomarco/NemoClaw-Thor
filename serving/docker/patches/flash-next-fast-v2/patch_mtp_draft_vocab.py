#!/usr/bin/env python3
"""Add an opt-in reduced-vocabulary projection to the Flash-Next MTP head."""

import ast
import re
import sys


target = sys.argv[1]
marker = "NemoClaw-Thor: reduced MTP draft vocabulary"
hook = r'''

# NemoClaw-Thor: reduced MTP draft vocabulary
import os as _draft_vocab_os
from vllm.logger import init_logger as _draft_vocab_init_logger

_draft_vocab_logger = _draft_vocab_init_logger(__name__)


def _draft_vocab_compute_logits(
    self, hidden_states: torch.Tensor, spec_step_idx: int = 0
):
    state = getattr(self, "_draft_vocab_state", None)
    if state is None:
        import numpy as _np

        weight = self.lm_head.weight
        ids = torch.from_numpy(
            _np.load(_draft_vocab_os.environ["VLLM_MTP_DRAFT_VOCAB"]).astype(
                _np.int64
            )
        ).to(weight.device)
        vocabulary_size = int(
            getattr(self.logits_processor, "org_vocab_size", weight.shape[0])
        )
        ids = torch.unique(ids[(ids >= 0) & (ids < vocabulary_size)], sorted=True)
        if ids.numel() < 257 or ids.numel() >= vocabulary_size:
            raise RuntimeError(
                f"invalid reduced MTP vocabulary: {ids.numel()} of {vocabulary_size} ids"
            )
        reduced_weight = weight.index_select(0, ids).contiguous()
        state = self._draft_vocab_state = (ids, reduced_weight, vocabulary_size)
        _draft_vocab_logger.info(
            "MTP reduced draft vocabulary: %d of %d rows (%.0f -> %.0f MiB per step)",
            ids.numel(),
            vocabulary_size,
            weight.shape[0] * weight.shape[1] * weight.element_size() / 2**20,
            reduced_weight.numel() * reduced_weight.element_size() / 2**20,
        )

    ids, reduced_weight, vocabulary_size = state
    reduced_logits = torch.nn.functional.linear(
        hidden_states.to(reduced_weight.dtype), reduced_weight
    )
    soft_cap = getattr(self.logits_processor, "soft_cap", None)
    if soft_cap is not None:
        reduced_logits = torch.tanh(reduced_logits / soft_cap) * soft_cap
    scale = getattr(self.logits_processor, "scale", 1.0)
    if scale != 1.0:
        reduced_logits *= scale
    logits = reduced_logits.new_full(
        (reduced_logits.shape[0], vocabulary_size), float("-inf")
    )
    logits.index_copy_(1, ids, reduced_logits)
    return logits


if _draft_vocab_os.environ.get("VLLM_MTP_DRAFT_VOCAB"):
    __MTP_CLASS__.compute_logits = _draft_vocab_compute_logits
    _draft_vocab_logger.info(
        "Reduced MTP draft vocabulary enabled: %s",
        _draft_vocab_os.environ["VLLM_MTP_DRAFT_VOCAB"],
    )
'''

source = open(target).read()
if marker in source:
    print("reduced MTP vocabulary patch already present")
    raise SystemExit(0)

match = re.search(r"^class (\w+MTP)\(", source, re.MULTILINE)
if not match:
    raise RuntimeError("could not locate the Flash-Next MTP class")
source = source.rstrip("\n") + hook.replace("__MTP_CLASS__", match.group(1))
ast.parse(source)
open(target, "w").write(source)
print(f"installed reduced MTP vocabulary patch in {target}")
