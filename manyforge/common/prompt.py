"""Agent prompt assembly — lane-aware preamble + RULES + tail checklist.

The composed prompt has three sections:

1. **Preamble**: instructions explaining the agent's role and bounds.
   This is identical across all three lanes.
2. **State block**: program/scene/tree projection (from
   :mod:`projection`). Identical across lanes.
3. **Tail checklist**: the recency-position reminder of the most-violated
   rules, appended after the user's request. Identical across lanes.

Lane-specific addenda (e.g. the OpenClaw discovery-protocol primer
introduced in Phase 3) get injected by passing ``discovery_mode`` to
:func:`build_agent_prompt`.

Phase 1 status: this module re-exports from
``openclaw_assistant_bridge.adapter`` with the
``discovery_mode`` parameter accepting only ``"direct"`` (Phase 2's
default) for now. Phase 3 expands the accepted values to include
``"openclaw_discovery"`` (the tool_search/tool_describe/tool_call
primer) and Phase 4 ``"hermes_direct"`` for the Hermes lane.
"""
from __future__ import annotations

from typing import Literal

from openclaw_assistant_bridge.adapter import build_agent_prompt as _build_agent_prompt

# Accepted ``discovery_mode`` values per lane. Phase 1 keeps only
# ``direct`` (the default behavior that adapter.py implements). Phase 3
# extends with ``openclaw_discovery``; Phase 4 with ``hermes_direct``.
DiscoveryMode = Literal["direct", "openclaw_discovery", "hermes_direct"]


def build_agent_prompt(
    payload,
    *,
    discovery_mode: DiscoveryMode = "direct",
    **kwargs,
):
    """Build the full agent prompt with lane-aware discovery framing.

    ``discovery_mode`` selects which addendum (if any) describes the tool
    discovery protocol to the model.

    - ``"direct"`` (default): the model sees the full tool catalog
      directly in ``tools[]``. No additional protocol explanation.
    - ``"openclaw_discovery"`` (Phase 3): adds the discovery-primer
      paragraph teaching ``tool_search`` → ``tool_describe`` → ``tool_call``.
    - ``"hermes_direct"`` (Phase 4): the Hermes lane. Selects the ``mcp``
      dispatch surface — the Hermes gateway exposes the ManyForge tools as
      native MCP functions (``mcp_manyforge_<id>``) directly in ``tools[]``,
      so the model is told to call them natively rather than through
      OpenClaw's code/tools discovery surface.
    """
    if discovery_mode not in ("direct", "openclaw_discovery", "hermes_direct"):
        raise ValueError(
            f"discovery_mode must be one of direct/openclaw_discovery/hermes_direct, "
            f"got {discovery_mode!r}"
        )
    # Phase 3/4: map lane-level discovery names to the adapter's concrete
    # dispatch-surface primer. Runtime OpenClaw still passes tool_surface from
    # OPENCLAW_ASSISTANT_TOOL_SURFACE directly; this mapping keeps direct calls
    # to the shared helper honest.
    if discovery_mode == "openclaw_discovery":
        kwargs.setdefault("tool_surface", "tools")
    elif discovery_mode == "hermes_direct":
        kwargs.setdefault("tool_surface", "mcp")
    return _build_agent_prompt(payload, **kwargs)


__all__ = ["build_agent_prompt", "DiscoveryMode"]
