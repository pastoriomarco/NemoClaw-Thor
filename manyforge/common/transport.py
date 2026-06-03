"""``AssistantTransport`` interface — the shared contract every lane implements.

Per THREE-LANE-MIGRATION-PLAN.md §5.4, all three lanes implement the
same typed transport interface so the bridge supervisor and any
cross-lane harness (smoke runner, compare_lanes.py) can wire any
lane in without lane-specific branches.

The interface has four methods:

- ``build_request(prompt, ctx)`` — construct the wire-level request
  body for the lane's upstream (OpenClaw gateway, vLLM direct, Hermes
  sessions API).
- ``dispatch(request, timeout_s)`` — send it (typically async).
- ``parse_response(response)`` — convert the wire-level response into a
  ``ParsedResponse`` DTO.
- ``normalize_tool_calls(parsed)`` — extract OpenAI structured
  ``tool_calls[]`` from the parsed response, normalizing for lane
  quirks (OpenClaw ``{tool, result}`` unwrap; Hermes ``mcp_manyforge_``
  prefix strip; Direct: no normalization).

Phase 1 status: the interface is defined here as a ``Protocol`` so
existing lane code can declare conformance without inheriting. The
existing OpenClaw bridge does NOT yet implement the protocol — Phase 3
will refactor it to. The Direct lane wrapper lives at
``manyforge.lanes.direct.transport`` (Phase 2 ships a thin adapter
over the existing dev_ws/manyforge_assistant_bridge code).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# Re-export the discovery-mode literal from prompt.py for transports
# that need to pass it through.
from .prompt import DiscoveryMode  # noqa: F401


@dataclass
class SessionCtx:
    """Per-request session context passed into ``build_request``.

    Fields:
        conversation_id: the Composer-issued conversation identifier
            (becomes the OpenClaw conversationId and the Hermes session
            id).
        principal: the principal name recorded in audit (lane-derived
            in the MCP bridge; same value here for consistency).
        assistant_mode: the assistant-mode identifier (typically
            ``composer-assistant``).
        request_id: a per-request id; rotates every dispatch.
        session_key: the stable session key derived via
            :func:`manyforge.assistant_session.session_key.derive_session_key`.
            Used for compaction counter scoping and circuit-breaker state.
        catalog_hash: the catalog hash for the current mode manifest;
            included in the request so the upstream can detect catalog drift.
        discovery_mode: which discovery framing to apply to the prompt
            (``direct`` / ``openclaw_discovery`` / ``hermes_direct``).
        extra: lane-specific extras (e.g. Hermes X-Hermes-Session-Key).
    """

    conversation_id: str
    principal: str
    assistant_mode: str
    request_id: str
    session_key: str
    catalog_hash: str
    discovery_mode: DiscoveryMode = "direct"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class WireRequest:
    """The lane-specific wire request that the transport sends upstream."""

    method: str  # "POST" / "GET"
    url: str  # absolute URL of the lane's endpoint
    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class WireResponse:
    """The lane-specific wire response returned from ``dispatch``."""

    status: int
    headers: dict[str, str]
    body: dict[str, Any] | str
    duration_ms: float


@dataclass
class ParsedResponse:
    """Lane-agnostic parsed response shape consumed by the bridge loop."""

    message_content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    raw: WireResponse | None = None


@runtime_checkable
class AssistantTransport(Protocol):
    """The shared transport contract every lane implements.

    Subclasses (or any class that structurally matches) work with the
    bridge supervisor and any cross-lane harness without lane-specific
    branching.
    """

    lane: Literal["direct", "openclaw", "hermes"]

    def build_request(self, prompt: str, ctx: SessionCtx) -> WireRequest:
        """Construct the wire-level request body.

        ``prompt`` is the assembled agent prompt from
        :func:`manyforge.common.prompt.build_agent_prompt`. The
        transport may put it in ``messages[-1].content`` (Direct,
        Hermes) or pass through a different envelope (OpenClaw uses the
        ``openclaw/<agent>`` model id and the gateway routes it).
        """
        ...

    async def dispatch(
        self, request: WireRequest, *, timeout_s: float
    ) -> WireResponse:
        """Send the wire request, return the wire response."""
        ...

    def parse_response(self, response: WireResponse) -> ParsedResponse:
        """Convert the wire response into the lane-agnostic DTO."""
        ...

    def normalize_tool_calls(
        self, parsed: ParsedResponse
    ) -> list[dict[str, Any]]:
        """Extract OpenAI structured ``tool_calls[]`` from the parsed
        response, normalizing for lane quirks (OpenClaw unwrap, Hermes
        prefix strip, Direct: no-op)."""
        ...


__all__ = [
    "AssistantTransport",
    "ParsedResponse",
    "SessionCtx",
    "WireRequest",
    "WireResponse",
]
