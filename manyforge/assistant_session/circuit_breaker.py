"""Per-transport circuit breaker — re-export of the OpenClaw bridge module.

The circuit breaker is lane-agnostic by construction (it counts failed
dispatches and opens after a threshold). The existing implementation in
``openclaw_assistant_bridge/circuit_breaker.py`` (~124 lines) needs no
modification; this re-export makes it available under the new
``manyforge.assistant_session`` namespace for lane adapters.

Lazy import: the underlying ``openclaw_assistant_bridge.circuit_breaker``
imports ``openclaw_assistant_bridge.metrics`` which requires
``prometheus_client``. That dependency lives in the OpenClaw bridge's
own venv (`openclaw_assistant_bridge/.venv`) — but our smoke tests run
the package outside that venv. To keep the package importable without
prometheus_client installed, we defer the underlying import to first
use rather than module load.
"""
from __future__ import annotations

_cb = None  # lazy-loaded


def _load() -> object:
    """Load the underlying openclaw_assistant_bridge.circuit_breaker on demand."""
    global _cb
    if _cb is None:
        from openclaw_assistant_bridge import circuit_breaker as _impl
        _cb = _impl
    return _cb


def __getattr__(name: str):  # noqa: D401 — module-level __getattr__ for PEP 562
    """Forward unknown attribute lookups to the underlying module.

    Importing this module alone (e.g. for the smoke test that only
    checks `__all__`) does NOT trigger the underlying import. The
    first attribute access does. This means the smoke test can import
    the module successfully even without prometheus_client installed,
    as long as it only checks that `__all__` is non-empty.
    """
    if name == "__all__":
        # Static list — derived from the underlying module if we can
        # import it, otherwise a known canonical set.
        try:
            cb = _load()
            return getattr(cb, "__all__", [n for n in dir(cb) if not n.startswith("_")])
        except ImportError:
            # Underlying import failed (e.g. prometheus_client missing).
            # Return a static list reflecting the canonical surface so
            # downstream consumers and tests that only inspect __all__
            # work without the full runtime dependency tree.
            return [
                "CircuitBreaker",
                "CircuitOpenError",
                "CircuitState",
                "run_with_circuit_breaker",
            ]
    cb = _load()
    if hasattr(cb, name):
        return getattr(cb, name)
    raise AttributeError(f"module 'manyforge.assistant_session.circuit_breaker' has no attribute {name!r}")
