"""ManyForge assistant envelope DTOs and helpers.

The Composer sends/receives the ``manyforge.assistant.provider_request.v0``
envelope to/from every lane. The fields are spec-driven (see
``manyforge_specs/`` per principle #4 of the three-lane plan); this
module exposes the parsing and DTO surface that bridges share.

Phase 1 status: re-exports from
``openclaw_assistant_bridge.adapter``. Phase 2 moves the
implementations.
"""
from __future__ import annotations

from openclaw_assistant_bridge.adapter import (  # noqa: F401
    AdapterConfig,
    AgentRunResult,
    error_envelope,
    is_action_shaped_prompt,
    request_id_from_payload,
)
from openclaw_assistant_bridge.adapter import (  # noqa: F401
    derive_gateway_session_key as derive_session_key,
)

__all__ = [
    "AdapterConfig",
    "AgentRunResult",
    "derive_session_key",
    "error_envelope",
    "is_action_shaped_prompt",
    "request_id_from_payload",
]
