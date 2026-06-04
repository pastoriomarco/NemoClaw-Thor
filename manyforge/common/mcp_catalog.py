"""Mode-scoped MCP tool catalog helpers.

The mode-scoped MCP wrapper is the only sanctioned mutation path on
every lane. These helpers expose mode-catalog inspection only. Prompt-
keyword tool-window inference is intentionally not part of the common
surface.

Phase 1 status: re-exports from ``openclaw_assistant_bridge.adapter``.
"""
from __future__ import annotations

from openclaw_assistant_bridge.adapter import (  # noqa: F401
    _allowed_tool_map as allowed_tool_map,
    _tool_ids as tool_ids,
)

__all__ = [
    "allowed_tool_map",
    "tool_ids",
]
