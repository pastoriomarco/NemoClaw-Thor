"""Mode-scoped MCP tool allowlist resolution.

The mode-scoped MCP wrapper is the only sanctioned mutation path on
every lane (per [HERMES-MIGRATION-ANALYSIS preserved evidence in plan
§3]). The bridges all consume this allowlist before dispatching a tool
call, ensuring the model can never call a tool that isn't in the active
``assistant_modes[<mode>].tools`` whitelist.

Phase 1 status: re-exports from ``openclaw_assistant_bridge.adapter``.
"""
from __future__ import annotations

from openclaw_assistant_bridge.adapter import (  # noqa: F401
    _allowed_tool_map as allowed_tool_map,
    _helper_tool_ids as helper_tool_ids,
    _ordered_known_tools as ordered_known_tools,
    _tool_ids as tool_ids,
    _tools_matching as tools_matching,
    mcp_allowed_tools_from_payload,
)

__all__ = [
    "allowed_tool_map",
    "helper_tool_ids",
    "mcp_allowed_tools_from_payload",
    "ordered_known_tools",
    "tool_ids",
    "tools_matching",
]
