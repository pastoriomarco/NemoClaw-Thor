"""Session-key derivation — lane-agnostic.

The Composer envelope's ``conversationId`` + assistant mode + catalog
hash + revision metadata are hashed into a stable session key per
``adapter.derive_gateway_session_key``. This key:

- Is used by the bridge to scope per-session bookkeeping (compaction
  counter, circuit-breaker state).
- Rotates when any of its inputs change, so a deployment hot-reload or
  a corpus revision starts a fresh session.

The function lives in ``openclaw_assistant_bridge.adapter``; this
module re-exports it under the assistant_session namespace where lane
adapters naturally look for it (and under the same name the Hermes
runs API uses: ``session_key``).
"""
from __future__ import annotations

# Lazy import: openclaw_assistant_bridge.adapter imports httpx +
# composer-bridge helpers from the bridge's venv. Defer to first
# call so the package is importable for smoke-testing without those
# runtime deps installed.

def derive_session_key(payload):  # type: ignore[no-untyped-def]
    """Stable session key from the composer envelope. Lazy import."""
    from openclaw_assistant_bridge.adapter import derive_gateway_session_key as _impl
    return _impl(payload)


__all__ = ["derive_session_key"]
