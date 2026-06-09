"""Hermes lane bridge — FastAPI service on ``:8300``.

Per THREE-LANE-MIGRATION-PLAN.md §5.3/§6, this is the thin transport-wiring
shell around :mod:`lanes.hermes.engine`. It speaks the same Composer envelope
(``manyforge.assistant.provider_request.v0``) as the other lane bridges, so
Composer's HTTP provider needs no Hermes-specific parsing.

This module owns only the parts that need FastAPI + httpx: the route handlers,
the httpx-backed :class:`HermesSessionDispatcher` construction, audit/session
persistence, and the circuit-breaker wiring. All per-turn logic lives in
:mod:`lanes.hermes.engine` (dependency-light, unit-tested without httpx).

Launch: ``uvicorn lanes.hermes.service:app`` with cwd = the ``manyforge/`` root.
``scripts/lib/assistant.sh`` starts it on :8300 when ``ASSISTANT_PROVIDER=hermes``
and ``HERMES_LANE_PHASE4_ENABLED=true``.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import time
from typing import Any

# Make manyforge/ resolvable without an install step (parents[2] holds common/,
# assistant_session/, lanes/, openclaw_assistant_bridge/).
_MANYFORGE_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _MANYFORGE_ROOT not in sys.path:
    sys.path.insert(0, _MANYFORGE_ROOT)

import httpx  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from common.envelope import error_envelope  # noqa: E402

from . import engine  # noqa: E402
from .session_dispatcher import HermesSessionDispatcher  # noqa: E402

logger = logging.getLogger(__name__)


# ---- config ------------------------------------------------------------------


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _hermes_base_url() -> str:
    return _env("HERMES_BASE_URL", "http://127.0.0.1:8642")


def _api_server_key() -> str:
    return _env("API_SERVER_KEY")  # spike Q6: gateway reads API_SERVER_KEY


def _model_name() -> str:
    return _env("MANYFORGE_MODEL") or _env("THOR_MODEL_ID") or _env("MODEL_PROFILE")


def _principal() -> str:
    return _env("MANYFORGE_PRINCIPAL") or f"hermes-{_env('SANDBOX_NAME', 'sandbox')}"


def _run_timeout_s() -> float:
    try:
        return float(_env("MANYFORGE_HERMES_RUN_TIMEOUT_S", "600") or "600")
    except ValueError:
        return 600.0


def _audit_path() -> str:
    return _env("MANYFORGE_HERMES_AUDIT_PATH", "/tmp/manyforge-assistant-e2e/hermes-bridge-audit.jsonl")


def _build_circuit_breaker() -> Any:
    try:
        from assistant_session import circuit_breaker as cb  # lazy (needs prometheus)

        return cb.CircuitBreaker(
            enabled=_env("MANYFORGE_HERMES_CIRCUIT_BREAKER", "true").lower() in {"1", "true", "yes", "on"},
            failure_threshold=int(_env("MANYFORGE_HERMES_CB_THRESHOLD", "5") or "5"),
            cooldown_seconds=float(_env("MANYFORGE_HERMES_CB_COOLDOWN_S", "30") or "30"),
        )
    except Exception as exc:  # noqa: BLE001 — breaker is best-effort infra
        logger.warning("circuit breaker unavailable (%s); proceeding without it", exc)
        return None


_BREAKER = _build_circuit_breaker()


def _append_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except OSError as exc:
        logger.warning("could not append to %s: %s", path, exc)


# ---- app ---------------------------------------------------------------------

app = FastAPI(title="ManyForge Hermes assistant-provider adapter", version="0.1.0")


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "provider": engine.LANE,
        "hermesBaseUrl": _hermes_base_url(),
        "model": _model_name(),
        "principal": _principal(),
        "apiKeyConfigured": bool(_api_server_key()),
        "activeRequestIds": sorted(engine._ACTIVE),  # noqa: SLF001 — diagnostic only
    }


@app.post("/v1/manyforge/assistant/{request_id}/cancel")
async def cancel_request(request_id: str) -> dict[str, Any]:
    handle = engine.mark_cancelled(request_id)
    run_id = handle.get("run_id")
    dispatcher = handle.get("dispatcher")
    if run_id and dispatcher is not None:
        await dispatcher.stop_run(run_id)  # best-effort; never raises
        return {"requestId": request_id.strip(), "cancelled": True, "runId": run_id}
    return {"requestId": request_id.strip(), "cancelled": True, "runId": None}


@app.post("/v1/manyforge/assistant")
async def assistant(request: Request) -> JSONResponse:
    handler_started = time.perf_counter()
    request_ts_ms = int(time.time() * 1000)
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=400,
            content=error_envelope(
                request_id=f"hermes-adapter-{request_ts_ms}",
                code="invalid_json",
                detail=f"request body is not valid JSON: {exc}",
            ),
        )

    async with httpx.AsyncClient() as client:
        dispatcher = HermesSessionDispatcher(
            base_url=_hermes_base_url(), api_key=_api_server_key(), client=client
        )
        result = await engine.run_assistant_turn(
            payload,
            dispatcher=dispatcher,
            model=_model_name(),
            principal=_principal(),
            run_timeout_s=_run_timeout_s(),
            breaker=_BREAKER,
            now_ms=request_ts_ms,
            elapsed_ms_fn=lambda: (time.perf_counter() - handler_started) * 1000.0,
        )

    # Persist audit + session events (best-effort; outside the request path's
    # correctness — the Composer bridge-tools log is the hard source of truth).
    if result.audit_entry:
        _append_jsonl(_audit_path(), [result.audit_entry])
    if result.session_events:
        _append_jsonl(
            _env(
                "MANYFORGE_HERMES_SESSION_EVENTS_PATH",
                "/tmp/manyforge-assistant-e2e/hermes-session-events.jsonl",
            ),
            result.session_events,
        )

    return JSONResponse(status_code=result.status_code, content=result.envelope)


__all__ = ["app"]
