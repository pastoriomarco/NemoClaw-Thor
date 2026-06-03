"""ManyForge assistant_session — per-session orchestration policy.

This package holds the orchestration logic that lives ABOVE the
transport layer but BELOW the lane-specific glue. Each policy below is
per-lane opt-in via a ``SessionPolicy`` config — defaults preserve the
iter-32 OpenClaw behavior, but Hermes and Direct lanes can disable
individual policies if their nature already handles the concern.

Modules:
    compaction              — bridge-fired /compact every N prompts
                              (the iter-32 production recipe). Default
                              on for OpenClaw; OFF for Hermes (Hermes
                              owns its session lifecycle via the runs
                              API); on with different threshold for
                              Direct (truncation strategy).
    synthetic_short_circuits — synthetic-clarification bypass + retry-loop
                              detector. Cosmos-specific patches. Default
                              on for OpenClaw (proven there); opt-in for
                              Direct and Hermes — flag them per-config
                              and benchmark with/without before defaulting on.
    circuit_breaker         — per-transport circuit breaker (timeout +
                              consecutive failure counter). Lane-agnostic;
                              always on.
    session_key             — derive the OpenClaw / Hermes session key
                              from the Composer envelope. Lane-agnostic.

Phase 1 status: re-export shim over the OpenClaw bridge's existing
service.py + circuit_breaker.py + adapter.py. Phase 2 + Phase 4 will
move implementations here as the new lanes consume them.
"""
from __future__ import annotations

from . import circuit_breaker  # noqa: F401
from . import compaction  # noqa: F401
from . import session_key  # noqa: F401
from . import synthetic_short_circuits  # noqa: F401

__version__ = "0.1.0"
