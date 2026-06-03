"""Per-transport circuit breaker — re-export of the OpenClaw bridge module.

The circuit breaker is lane-agnostic by construction (it counts failed
dispatches and opens after a threshold). The existing implementation in
``openclaw_assistant_bridge/circuit_breaker.py`` (~124 lines) needs no
modification; this re-export makes it available under the new
``manyforge.assistant_session`` namespace for lane adapters.
"""
from __future__ import annotations

from openclaw_assistant_bridge.circuit_breaker import *  # noqa: F401, F403
from openclaw_assistant_bridge import circuit_breaker as _cb

# Re-export the module's public surface explicitly so static analysis
# can see it.
__all__ = getattr(_cb, "__all__", [name for name in dir(_cb) if not name.startswith("_")])
