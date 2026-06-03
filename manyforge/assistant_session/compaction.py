"""Compaction policy — bridge-fired /compact every N prompts.

The iter-32 production recipe (51/66 on cosmos-reason2-8b) for the
OpenClaw lane works because the bridge fires ``/compact`` to the
OpenClaw gateway every N user prompts (default 2). Without this,
OpenClaw's auto-compaction hits an ``already_compacted_recently``
cooldown after the first overflow and stops working.

This module exposes the policy as a small ``CompactionPolicy`` dataclass
that lane adapters consume. Each lane picks its own defaults:

- **OpenClaw**: ``enabled=True, every=2`` (the iter-32 recipe).
- **Hermes**: ``enabled=False`` — Hermes owns its own session lifecycle
  via the runs API; we don't override.
- **Direct**: ``enabled=True, every=N`` with the bridge doing token
  truncation rather than a ``/compact`` call. The trigger logic is the
  same; only the action differs.

Phase 1 status: re-exports the bookkeeping helpers from
``openclaw_assistant_bridge.service``. The dataclass is new (it's pure
config) so lane adapters in Phase 2+ can consume it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Lazy import: openclaw_assistant_bridge.service imports fastapi+httpx
# which are runtime dependencies that live in the bridge's venv. The
# package smoke tests run outside that venv, so we defer the underlying
# import to first call rather than module load. This keeps the package
# importable for testing the dataclass surface without the full bridge
# runtime stack.

def bump_session_request_counter(session_key: str) -> int:
    """Per-session-key request counter for compaction policy. Lazy import."""
    from openclaw_assistant_bridge.service import _bump_session_request_counter as _impl
    return _impl(session_key)


def should_fire_compact(session_request_count: int) -> bool:
    """Whether to fire `/compact` at this request count. Lazy import."""
    from openclaw_assistant_bridge.service import _should_fire_compact as _impl
    return _impl(session_request_count)


@dataclass(frozen=True)
class CompactionPolicy:
    """Per-lane compaction policy. Default values match OpenClaw iter-32.

    Fields:
        enabled: when False, the policy is a no-op (Hermes default).
        every: fire the compact action before every Nth request
               (skipping the first). Default 2 matches iter-32.
        action: what "compact" means for this lane.
                - ``"openclaw_slash_compact"`` — POST /compact text to
                  the gateway as a chat message (OpenClaw's slash-cmd).
                - ``"direct_truncate"`` — drop the oldest N turns from
                  the assembled prompt.
                - ``"hermes_session_summary"`` — call Hermes' session
                  summary endpoint (Phase 4 decides the exact verb).
                - ``"none"`` — disabled.
    """

    enabled: bool = True
    every: int = 2
    action: Literal[
        "openclaw_slash_compact",
        "direct_truncate",
        "hermes_session_summary",
        "none",
    ] = "openclaw_slash_compact"


# Per-lane defaults. Lane adapters import the matching constant.
OPENCLAW_DEFAULT = CompactionPolicy(
    enabled=True, every=2, action="openclaw_slash_compact"
)
DIRECT_DEFAULT = CompactionPolicy(
    enabled=True, every=4, action="direct_truncate"
)
HERMES_DEFAULT = CompactionPolicy(
    enabled=False, every=0, action="none"
)


__all__ = [
    "CompactionPolicy",
    "DIRECT_DEFAULT",
    "HERMES_DEFAULT",
    "OPENCLAW_DEFAULT",
    "bump_session_request_counter",
    "should_fire_compact",
]
