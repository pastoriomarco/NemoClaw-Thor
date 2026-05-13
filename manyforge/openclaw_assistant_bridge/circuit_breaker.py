"""Simple per-transport circuit breaker for the bridge → gateway dispatch path.

Three states: closed (healthy), open (failing fast), half-open (probe).

  closed → (N consecutive failures) → open
  open → (cooldown elapsed) → half-open
  half-open → success → closed
  half-open → failure → open (new cooldown window)

Per-transport state means a gateway_http meltdown does not also gate the
cli_shell_out path (or vice versa). State is process-local; restarts reset.

Opt-in via ``OPENCLAW_ASSISTANT_CIRCUIT_BREAKER_ENABLED``. When disabled,
``before_dispatch`` always returns True and ``record_*`` are no-ops, so
the breaker has zero behavioral impact.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .metrics import CIRCUIT_BREAKER_STATE, CIRCUIT_BREAKER_TRIPS_TOTAL


STATE_CLOSED = 0
STATE_HALF_OPEN = 1
STATE_OPEN = 2


@dataclass
class _TransportState:
    state: int = STATE_CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class CircuitBreaker:
    """Tiny async-safe circuit breaker.

    All public methods are coroutines because state transitions take a
    per-transport lock; the lock prevents two concurrent requests from
    both observing the same half-open state and double-probing.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        failure_threshold: int,
        cooldown_seconds: float,
    ) -> None:
        self._enabled = enabled
        self._failure_threshold = max(1, failure_threshold)
        self._cooldown_seconds = max(0.0, cooldown_seconds)
        self._states: dict[str, _TransportState] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _state_for(self, transport: str) -> _TransportState:
        st = self._states.get(transport)
        if st is None:
            st = _TransportState()
            self._states[transport] = st
            CIRCUIT_BREAKER_STATE.labels(transport=transport).set(STATE_CLOSED)
        return st

    async def before_dispatch(self, transport: str) -> tuple[bool, str | None]:
        """Return ``(allowed, reason)``. Reason is set when allowed is False.

        Side-effect: transitions open → half-open if the cooldown has
        elapsed, claiming the half-open probe slot for this caller.
        """
        if not self._enabled:
            return True, None
        st = self._state_for(transport)
        async with st.lock:
            if st.state == STATE_CLOSED:
                return True, None
            if st.state == STATE_OPEN:
                if time.monotonic() - st.opened_at < self._cooldown_seconds:
                    return False, "circuit_open_in_cooldown"
                # Cooldown elapsed: promote to half-open and let this caller probe.
                st.state = STATE_HALF_OPEN
                CIRCUIT_BREAKER_STATE.labels(transport=transport).set(STATE_HALF_OPEN)
                return True, None
            # STATE_HALF_OPEN: another probe is in flight; reject this caller
            # to preserve the single-probe invariant.
            return False, "circuit_half_open_probe_in_flight"

    async def record_success(self, transport: str) -> None:
        if not self._enabled:
            return
        st = self._state_for(transport)
        async with st.lock:
            st.consecutive_failures = 0
            if st.state != STATE_CLOSED:
                st.state = STATE_CLOSED
                CIRCUIT_BREAKER_STATE.labels(transport=transport).set(STATE_CLOSED)

    async def record_failure(self, transport: str) -> None:
        if not self._enabled:
            return
        st = self._state_for(transport)
        async with st.lock:
            st.consecutive_failures += 1
            if st.state == STATE_HALF_OPEN:
                # Half-open probe failed: reopen with fresh cooldown.
                st.state = STATE_OPEN
                st.opened_at = time.monotonic()
                CIRCUIT_BREAKER_STATE.labels(transport=transport).set(STATE_OPEN)
                CIRCUIT_BREAKER_TRIPS_TOTAL.labels(transport=transport).inc()
                return
            if (
                st.state == STATE_CLOSED
                and st.consecutive_failures >= self._failure_threshold
            ):
                st.state = STATE_OPEN
                st.opened_at = time.monotonic()
                CIRCUIT_BREAKER_STATE.labels(transport=transport).set(STATE_OPEN)
                CIRCUIT_BREAKER_TRIPS_TOTAL.labels(transport=transport).inc()
