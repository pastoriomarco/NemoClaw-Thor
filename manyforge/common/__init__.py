"""ManyForge assistant common — universal core shared across all three lanes.

This package extracts the lane-agnostic parts of the OpenClaw bridge's
``adapter.py`` into reusable modules. All three lanes (Direct, OpenClaw,
Hermes) consume these modules; lane adapters contribute only the
transport-specific glue.

Modules:
    projection      — program/scene/tree/catalog projection helpers
    prompt          — agent prompt assembly (preamble + RULES + tail checklist)
    envelope        — DTOs for the ``manyforge.assistant.provider_request.v0``
                      envelope (config, results, error responses)
    tool_calls      — tool-call extraction, canonicalization, OpenClaw envelope
                      unwrap helper
    mcp_allowlist   — mode-scoped MCP tool allowlist resolution

Status: Phase 1 of THREE-LANE-MIGRATION-PLAN.md. Initially a re-export
shim over ``openclaw_assistant_bridge.adapter`` so the OpenClaw lane
keeps working bit-identically; Phase 2 and Phase 4 will move the
implementations here as the Direct and Hermes lanes are wired up.

Behavior-preserving constraint (per plan §8 Phase 1): the public API of
the OpenClaw bridge is unchanged. Any divergence from the original
adapter.py functions is a bug, not a feature.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Surface the most-used exports at package root for ergonomic imports.
# Individual modules are also accessible directly (preferred for clarity).
from . import envelope  # noqa: F401
from . import mcp_allowlist  # noqa: F401
from . import projection  # noqa: F401
from . import prompt  # noqa: F401
from . import tool_calls  # noqa: F401
