"""Tool-call extraction, canonicalization, and lane envelope unwrap.

All three lanes consume the same model output shape (OpenAI structured
``tool_calls[]``) thanks to the vLLM ``--tool-call-parser`` translation.
But each lane wraps the result differently before the bridge sees it:

- **Direct**: bare OpenAI ``role: tool, tool_call_id, content`` messages.
  No unwrap needed.
- **OpenClaw**: ``tool_call`` (one of the three discovery control tools)
  returns ``{tool: <compact entry>, result: <real result>}``. The
  :func:`unwrap_openclaw_envelope` helper strips the wrapper.
- **Hermes**: Hermes prefixes MCP tools as ``mcp_<server>_<tool>``, so
  the bridge strips the ``mcp_manyforge_`` prefix on the way back into
  audit logs. Result content itself is unwrapped (no extra envelope).

Phase 1 status: re-exports from ``openclaw_assistant_bridge.adapter``.
The OpenClaw envelope unwrap helper is added here as a Phase 3
placeholder (returns the value unchanged in Phase 1 since today's
OpenClaw lane gets direct tools via the plugin attempt, not via
``tool_search``).
"""
from __future__ import annotations

from typing import Any

from openclaw_assistant_bridge.adapter import (  # noqa: F401
    _canonical_tool_name as canonical_tool_name,
    _dedupe_known as dedupe_known,
    _extract_tool_calls as extract_tool_calls,
)


def unwrap_openclaw_envelope(value: Any) -> Any:
    """Strip the ``{tool, result}`` wrap that OpenClaw's ``tool_call``
    control tool puts around real tool results.

    The OpenClaw native discovery surface (``tool_search`` /
    ``tool_describe`` / ``tool_call``) wraps every real tool result in
    ``{tool: <compact entry>, result: <real result>}`` per
    ``dist/pi-tools-iVT6BGHc.js:728-743`` (verified against the 2026.5.22
    npm wheel). The model must be taught to read ``.result``; this
    helper does the same on the bridge audit side so cross-lane reports
    show comparable shapes.

    Phase 1: no-op (today's OpenClaw lane has the plugin attempt active,
    not the discovery surface, so the wrap doesn't appear). Phase 3
    enables the wrap detection once the discovery surface is in use.

    A wrap is detected when ``value`` is a dict with both ``tool`` and
    ``result`` keys, where ``tool`` is itself a dict and ``result`` is
    not None.
    """
    if isinstance(value, dict):
        tool = value.get("tool")
        result = value.get("result")
        if isinstance(tool, dict) and result is not None and set(value.keys()) == {"tool", "result"}:
            return result
    return value


def strip_mcp_prefix(tool_name: str, server: str = "manyforge") -> str:
    """Strip the Hermes ``mcp_<server>_`` tool-name prefix.

    Hermes registers MCP tools as ``mcp_<server>_<tool>`` per upstream
    docs. For cross-lane audit parity (so the smoke runner's per-tool
    counters work without knowing which lane emitted the row), we strip
    the prefix on the audit side.

    Examples:
        >>> strip_mcp_prefix("mcp_manyforge_tree_draft_wrap_node")
        'tree_draft_wrap_node'
        >>> strip_mcp_prefix("tree_draft_wrap_node")  # already bare
        'tree_draft_wrap_node'
    """
    prefix = f"mcp_{server}_"
    if isinstance(tool_name, str) and tool_name.startswith(prefix):
        return tool_name[len(prefix):]
    return tool_name


__all__ = [
    "canonical_tool_name",
    "dedupe_known",
    "extract_tool_calls",
    "strip_mcp_prefix",
    "unwrap_openclaw_envelope",
]
