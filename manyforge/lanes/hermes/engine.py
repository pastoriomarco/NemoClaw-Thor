"""Hermes lane turn engine — transport-free, dependency-light core logic.

Separated from :mod:`lanes.hermes.service` so the per-turn logic (envelope in
→ prompt → run → observe → envelope out) is unit-testable without FastAPI or
httpx installed. The only imports here are the universal core (``common.*``,
which is stdlib-backed) and the lane's own transport/observer. The FastAPI
service constructs the httpx-backed :class:`HermesSessionDispatcher`, passes it
in, and persists the returned audit/session records.

Per THREE-LANE-MIGRATION-PLAN.md §5.3/§6: the engine does NOT loop tool calls
— Hermes owns its agent loop and dispatches its MCP tools to Composer through
the lane-neutral bridge. The engine submits the run, observes the event stream
(best-effort), and returns the final assistant message in the Composer
envelope. Mutations are applied + tracked Composer-side via the bridge-tools
callback log, so the returned envelope carries no ``toolCalls``/``proposals``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from common.envelope import derive_session_key, error_envelope, request_id_from_payload
from common.prompt import build_agent_prompt
from common.transport import SessionCtx

from .progress_observer import HermesProgressObserver
from .transport import HermesTransport

logger = logging.getLogger(__name__)

LANE = "hermes"

# Envelope constants — same source of truth common.envelope.error_envelope uses.
from openclaw_assistant_bridge.adapter import (  # noqa: E402
    DEFAULT_SCHEMA_VERSION,
    SUPPORTED_VERSION_FAMILY,
)

# Cross-request state for cancellation. request_id -> {"run_id", "dispatcher"}.
_ACTIVE: dict[str, dict[str, Any]] = {}
_CANCELLED: set[str] = set()


@dataclass
class TurnResult:
    """Outcome of one assistant turn, ready for the service to serialise."""

    status_code: int
    envelope: dict[str, Any]
    audit_entry: dict[str, Any] = field(default_factory=dict)
    session_events: list[dict[str, Any]] = field(default_factory=list)


def success_envelope(
    *, request_id: str, message: str, warnings: list[str], requires_review: bool = True
) -> dict[str, Any]:
    """Composer success envelope — mirrors ``error_envelope`` minus ``error``."""
    return {
        "version": SUPPORTED_VERSION_FAMILY,
        "schemaVersion": DEFAULT_SCHEMA_VERSION,
        "requestId": request_id,
        "message": message,
        "toolCalls": [],
        "proposals": [],
        "warnings": list(warnings),
        "mutated": False,
        "draftMutated": False,
        "requiresReview": requires_review,
    }


def tool_ids(payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for tool in payload.get("tools") or []:
        if isinstance(tool, dict):
            tid = str(tool.get("id") or tool.get("name") or "").strip()
            if tid:
                out.append(tid)
    return out


def mark_cancelled(request_id: str) -> dict[str, Any]:
    """Record a cancel request and return the active run handle (if any) so the
    service can call ``dispatcher.stop_run``. Pure state bookkeeping."""
    request_id = (request_id or "").strip()
    if not request_id:
        return {}
    _CANCELLED.add(request_id)
    return dict(_ACTIVE.get(request_id) or {})


async def run_assistant_turn(
    payload: Any,
    *,
    dispatcher: Any,
    model: str = "",
    principal: str = "hermes-sandbox",
    run_timeout_s: float = 600.0,
    breaker: Any = None,
    now_ms: int = 0,
    elapsed_ms_fn: Any = None,
) -> TurnResult:
    """Process one Composer provider-request envelope through the Hermes lane.

    ``dispatcher`` is a :class:`HermesSessionDispatcher` (real httpx-backed in
    the service; a fake in tests). ``breaker`` is an optional circuit breaker
    with the async ``before_dispatch``/``record_success``/``record_failure``
    surface. ``elapsed_ms_fn`` returns the handler's elapsed ms for the audit
    (injected so tests are deterministic).
    """
    if not isinstance(payload, dict):
        rid = f"hermes-adapter-{now_ms}"
        return TurnResult(
            400,
            error_envelope(request_id=rid, code="invalid_envelope", detail="request body must be a JSON object"),
        )

    request_id = request_id_from_payload(payload)
    if request_id in _CANCELLED:
        _CANCELLED.discard(request_id)
        return TurnResult(
            200,
            error_envelope(
                request_id=request_id,
                code="cancelled",
                detail=f"request {request_id} was cancelled before dispatch",
            ),
        )

    conversation_id = str(payload.get("conversationId") or request_id).strip()
    assistant_mode = str(payload.get("assistantMode") or "composer-assistant").strip()
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    catalog_hash = str(context.get("catalogHash") or "")

    ctx = SessionCtx(
        conversation_id=conversation_id,
        principal=principal,
        assistant_mode=assistant_mode,
        request_id=request_id,
        session_key=derive_session_key(payload),
        catalog_hash=catalog_hash,
        discovery_mode="hermes_direct",
    )

    try:
        prompt = build_agent_prompt(payload, discovery_mode="hermes_direct")
    except Exception as exc:  # noqa: BLE001
        return TurnResult(
            200,
            error_envelope(request_id=request_id, code="prompt_build_failed", detail=f"could not assemble agent prompt: {exc}"),
        )

    if breaker is not None and getattr(breaker, "enabled", False):
        allowed, reason = await breaker.before_dispatch(LANE)
        if not allowed:
            return TurnResult(
                200,
                error_envelope(request_id=request_id, code="circuit_open", detail=f"hermes lane circuit breaker open: {reason}"),
            )

    transport = HermesTransport(dispatcher=dispatcher, base_url=getattr(dispatcher, "_base", ""), model=model)
    _ACTIVE[request_id] = {"run_id": "", "dispatcher": dispatcher}
    dispatcher.on_run_started = lambda rid, _rid=request_id: _ACTIVE.get(_rid, {}).update(run_id=rid)
    try:
        wire = transport.build_request(prompt, ctx)
        response = await transport.dispatch(wire, timeout_s=run_timeout_s)
        parsed = transport.parse_response(response)
    except Exception as exc:  # noqa: BLE001 — never leak a 500 to Composer
        if breaker is not None:
            await breaker.record_failure(LANE)
        _ACTIVE.pop(request_id, None)
        return TurnResult(
            200,
            error_envelope(request_id=request_id, code="hermes_dispatch_error", detail=f"hermes dispatch failed: {exc}"),
        )
    finally:
        run_meta = _ACTIVE.pop(request_id, {})
        # Orphan guard. ``run_to_completion`` returns as soon as it observes a
        # terminal-type event, but breaking the SSE observation does NOT stop
        # the run: the Hermes gateway's agent loop can keep executing further
        # turns server-side after the scored tool call was already dispatched.
        # On the single-slot local-inference path those continuations pile up
        # as concurrent "orphan" runs and collapse the slot (observed: 4
        # concurrent runs → 300s stale-kills → cascade). Explicitly stop the
        # run so it cannot outlive this turn. Best-effort + idempotent (no-op
        # if the run already finished); never allowed to break the turn.
        _orphan_run_id = str(run_meta.get("run_id") or "").strip()
        if _orphan_run_id:
            try:
                await dispatcher.stop_run(_orphan_run_id, timeout_s=10.0)
            except Exception as exc:  # noqa: BLE001 — guard must never break the turn
                logger.debug("orphan-guard stop_run(%s) raised (ignored): %s", _orphan_run_id, exc)

    body = response.body if isinstance(response.body, dict) else {}
    events = body.get("events", []) if isinstance(body.get("events"), list) else []
    observer = HermesProgressObserver()
    observation = observer.observe(events, conversation_id=conversation_id)

    warnings: list[str] = []
    if body.get("error"):
        warnings.append(str(body.get("error")))

    ok = response.status == 200 and parsed.finish_reason != "failed"
    if breaker is not None:
        await (breaker.record_success(LANE) if ok else breaker.record_failure(LANE))

    latency_ms = float(elapsed_ms_fn()) if callable(elapsed_ms_fn) else 0.0
    audit_entry = {
        "ts": now_ms,
        "lane": LANE,
        "requestId": request_id,
        "conversationId": conversation_id,
        "principal": principal,
        "model": model,
        "transport": "hermes_runs",
        "runId": run_meta.get("run_id") or body.get("runId") or "",
        "toolsObserved": observation.tools_observed,
        "toolsExpected": tool_ids(payload),
        "compactionFires": 0,  # compaction OFF on this lane (policy.yaml)
        "memoryWrites": observation.memory_writes,
        "skillCreations": observation.skill_creations,
        "cronFires": observation.cron_fires,
        "delegations": observation.delegations,
        "latencyMs": round(latency_ms, 1),
        "exitReason": parsed.finish_reason,
        "errorChain": warnings,
    }

    if not ok:
        detail = body.get("error") or "hermes run did not complete successfully"
        return TurnResult(
            200,
            error_envelope(request_id=request_id, code="hermes_run_failed", detail=str(detail)),
            audit_entry=audit_entry,
            session_events=observation.session_events,
        )

    return TurnResult(
        200,
        success_envelope(
            request_id=request_id,
            message=parsed.message_content or "(hermes run completed with no final message)",
            warnings=warnings,
        ),
        audit_entry=audit_entry,
        session_events=observation.session_events,
    )


__all__ = [
    "TurnResult",
    "run_assistant_turn",
    "success_envelope",
    "tool_ids",
    "mark_cancelled",
    "LANE",
]
