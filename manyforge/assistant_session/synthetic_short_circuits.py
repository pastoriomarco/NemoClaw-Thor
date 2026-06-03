"""Synthetic-clarification bypass + retry-loop detector.

These are Cosmos-specific patches that fire when the model is what's
broken, not the transport. The relevant code lives at
``openclaw_assistant_bridge/service.py:355-508`` today; this module
provides the policy DTO that lane adapters consume to opt in/out.

The two synthetic short-circuits:

1. **Bypass clarification** — when the model returns text resembling a
   stalled "please clarify" with no tool calls AND the corpus precondition
   shows the prompt is action-shaped, the bridge re-issues with a stronger
   nudge instead of bubbling up an empty response.

2. **Retry-loop detector** — when the model emits the same tool with the
   same arguments more than threshold times, the bridge inserts an
   anti-perseveration system message and stops the loop.

Both patches apply to the MODEL output, not the transport, so they apply
equally to any lane that uses cosmos-reason2-8b. Per principle #1 of the
three-lane plan, OpenClaw is where they were proven; Direct and Hermes
get them opt-in until we benchmark with/without.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticPolicy:
    """Per-lane synthetic-short-circuit policy.

    Fields:
        bypass_clarification: when True, the bridge detects no-tool-call
            responses that resemble clarification questions for
            action-shaped prompts and re-issues with a stronger nudge.
        retry_loop_detector: when True, the bridge detects same-tool-
            same-args repetition over ``retry_loop_threshold`` and
            inserts an anti-perseveration nudge.
        retry_loop_threshold: how many identical tool calls before the
            detector fires (default 3 matches the OpenClaw bridge today).
    """

    bypass_clarification: bool = True
    retry_loop_detector: bool = True
    retry_loop_threshold: int = 3


# Per-lane defaults. OpenClaw uses the proven values. Direct and Hermes
# default to OFF until Phase 2 / Phase 4 benchmarks with-vs-without.
OPENCLAW_DEFAULT = SyntheticPolicy(
    bypass_clarification=True,
    retry_loop_detector=True,
    retry_loop_threshold=3,
)
DIRECT_DEFAULT = SyntheticPolicy(
    bypass_clarification=False,
    retry_loop_detector=False,
    retry_loop_threshold=3,
)
HERMES_DEFAULT = SyntheticPolicy(
    bypass_clarification=False,
    retry_loop_detector=False,
    retry_loop_threshold=3,
)


__all__ = [
    "DIRECT_DEFAULT",
    "HERMES_DEFAULT",
    "OPENCLAW_DEFAULT",
    "SyntheticPolicy",
]
