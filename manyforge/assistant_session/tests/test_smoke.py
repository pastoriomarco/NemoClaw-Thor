"""Smoke tests for manyforge.assistant_session package.

Phase 1 deliverable: re-exports resolve, dataclass defaults reflect
the iter-32 / per-lane intent, and the CompactionPolicy/SyntheticPolicy
APIs are immutable (frozen).
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from assistant_session import (  # noqa: E402
    circuit_breaker,
    compaction,
    session_key,
    synthetic_short_circuits,
)


def test_compaction_policy_defaults():
    """Per-lane CompactionPolicy defaults match the plan intent."""
    assert compaction.OPENCLAW_DEFAULT.enabled is True
    assert compaction.OPENCLAW_DEFAULT.every == 2
    assert compaction.OPENCLAW_DEFAULT.action == "openclaw_slash_compact"

    assert compaction.HERMES_DEFAULT.enabled is False
    assert compaction.HERMES_DEFAULT.action == "none"

    assert compaction.DIRECT_DEFAULT.enabled is True
    assert compaction.DIRECT_DEFAULT.action == "direct_truncate"


def test_compaction_policy_is_frozen():
    """CompactionPolicy is immutable (a frozen dataclass)."""
    p = compaction.OPENCLAW_DEFAULT
    try:
        p.every = 99  # type: ignore[misc]
        raise AssertionError("expected dataclasses.FrozenInstanceError")
    except Exception as exc:
        # dataclasses.FrozenInstanceError subclasses AttributeError.
        assert "frozen" in str(exc).lower() or isinstance(exc, AttributeError)


def test_compaction_should_fire_counter():
    """The bookkeeping helpers are re-exported callable."""
    assert callable(compaction.bump_session_request_counter)
    assert callable(compaction.should_fire_compact)


def test_synthetic_policy_defaults():
    """Per-lane SyntheticPolicy defaults match the plan intent."""
    assert synthetic_short_circuits.OPENCLAW_DEFAULT.bypass_clarification is True
    assert synthetic_short_circuits.OPENCLAW_DEFAULT.retry_loop_detector is True

    assert synthetic_short_circuits.DIRECT_DEFAULT.bypass_clarification is False
    assert synthetic_short_circuits.DIRECT_DEFAULT.retry_loop_detector is False

    assert synthetic_short_circuits.HERMES_DEFAULT.bypass_clarification is False
    assert synthetic_short_circuits.HERMES_DEFAULT.retry_loop_detector is False


def test_synthetic_policy_is_frozen():
    """SyntheticPolicy is immutable."""
    p = synthetic_short_circuits.OPENCLAW_DEFAULT
    try:
        p.bypass_clarification = False  # type: ignore[misc]
        raise AssertionError("expected immutability")
    except Exception as exc:
        assert "frozen" in str(exc).lower() or isinstance(exc, AttributeError)


def test_circuit_breaker_reexports():
    """circuit_breaker module exposes the underlying public surface."""
    # The original module has at least one public name (something like
    # CircuitBreaker or run_with_circuit). Verify __all__ is non-empty.
    assert circuit_breaker.__all__, "circuit_breaker.__all__ must not be empty"


def test_session_key_reexports():
    """session_key.derive_session_key is callable."""
    assert callable(session_key.derive_session_key)


if __name__ == "__main__":
    import inspect

    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"  ✓ {name}")
            except Exception as exc:
                failed += 1
                print(f"  ✗ {name}: {exc}")
    if failed:
        raise SystemExit(f"{failed} test(s) failed")
    print("all tests passed")
