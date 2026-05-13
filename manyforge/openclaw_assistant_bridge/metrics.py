"""Prometheus metrics for the OpenClaw assistant-provider bridge.

Opt-in via ``OPENCLAW_ASSISTANT_METRICS_ENABLED=true`` — the ``/metrics``
HTTP endpoint is only mounted when this is set. Metric **objects** are
always defined so call sites don't need conditional imports; with the
endpoint disabled they just accumulate counts in process memory and are
never scraped.

Label cardinality is deliberately low: ``status``, ``transport``, and
``stage`` are bounded enums. Request-level correlation (``requestId``,
``sessionId``) lives in the structured logs emitted by ``_log_event``;
never put high-cardinality fields on Prometheus labels.

Metric taxonomy modeled on SMG (Shepherd Model Gateway) — see
``serving/docs/COSMOS-REASON2-FINETUNE-PLAN.md`` lineage notes if you
need the provenance.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


# Bucket boundaries (seconds). Tuned for the OpenClaw shell-out path
# where warm gateway calls land at 5-10s and CLI bootstrap is ~40s.
# Adjust if observed p99 routinely lands above the top bucket.
_DURATION_BUCKETS = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0,
)


REQUESTS_TOTAL = Counter(
    "manyforge_bridge_requests_total",
    "Total assistant requests received by the bridge, partitioned by outcome.",
    labelnames=("status", "transport"),
)


REQUEST_DURATION = Histogram(
    "manyforge_bridge_request_duration_seconds",
    "Wall-clock duration of bridge request phases.",
    labelnames=("stage", "transport"),
    buckets=_DURATION_BUCKETS,
)


ACTIVE_REQUESTS = Gauge(
    "manyforge_bridge_active_requests",
    "Current count of in-flight assistant requests across all transports.",
)


TOOL_CALLS_TOTAL = Counter(
    "manyforge_bridge_tool_calls_total",
    "Tool calls observed in normalized assistant responses, by outcome.",
    labelnames=("outcome",),  # emitted | filtered_unknown | warning
)


COMPACT_FIRES_TOTAL = Counter(
    "manyforge_bridge_compact_fires_total",
    "Bridge-fired /compact attempts, by outcome.",
    labelnames=("outcome",),  # started | succeeded | failed
)


CIRCUIT_BREAKER_STATE = Gauge(
    "manyforge_bridge_circuit_breaker_state",
    "Circuit breaker state for the gateway/CLI dispatch path. "
    "0 = closed (healthy), 1 = half-open, 2 = open (failing fast).",
    labelnames=("transport",),
)


CIRCUIT_BREAKER_TRIPS_TOTAL = Counter(
    "manyforge_bridge_circuit_breaker_trips_total",
    "Number of times the circuit breaker has tripped open.",
    labelnames=("transport",),
)
