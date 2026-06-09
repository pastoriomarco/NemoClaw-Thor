"""Hermes ``/v1/runs`` session dispatcher — the lane's client to Hermes Agent.

Per THREE-LANE-MIGRATION-PLAN.md §5.3 and the Phase 0.5 contract spike
([docs/archive/PHASE-0.5-HERMES-SPIKE.md]), the Hermes lane talks to Hermes' native
**runs** API rather than ``/v1/chat/completions``. The spike (probe 5)
inspected ``gateway/platforms/api_server.py`` in the Hermes 0.14.0 wheel and
established the surface:

    POST   /v1/runs                 start a run; returns run_id (202)
    GET    /v1/runs/{run_id}        run status
    GET    /v1/runs/{run_id}/events SSE stream of structured lifecycle events
    POST   /v1/runs/{run_id}/approval  resolve a pending approval
    POST   /v1/runs/{run_id}/stop   interrupt a running agent

``/v1/runs`` was chosen over ``/v1/chat/completions`` + ``/v1/responses``
because only the runs API emits the structured lifecycle-event SSE stream
that the universal audit format (§4.7) and the progress observer consume,
and because ``approval``/``stop`` map cleanly onto ManyForge's review/cancel
semantics.

**Contract honesty.** The spike confirmed the *endpoint list* offline but its
live probe (probe 4) was blocked by a gateway-restart hurdle, so the exact
request-body schema and SSE event-type names were NOT live-verified. Every
such assumption is centralised in :class:`HermesRunsContract` below so a
single edit reconciles this client with whatever ``GET /v1/capabilities``
reports on the live gateway — no logic changes needed. The progress observer
treats unknown events as best-effort augmentation (drop-and-continue), so a
field-name mismatch degrades audit richness without breaking the lane.

Auth: ``Authorization: Bearer $API_SERVER_KEY`` (spike: ``gateway/config.py``
reads ``API_SERVER_KEY``). Session continuity: ``X-Hermes-Session-Id`` (the
Composer ``conversationId``) and ``X-Hermes-Session-Key`` headers.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HermesRunsContract:
    """Centralised wire-shape assumptions for the Hermes runs API.

    Everything the spike could not live-verify lives here so the lane can be
    reconciled with the live ``/v1/capabilities`` in one place. The defaults
    follow the documented Hermes Agent runs-API shape; adjust field names
    here (not in the dispatch logic) if the live gateway differs.
    """

    # POST /v1/runs request body. ``input_field`` carries the assembled agent
    # prompt; ``stream_field``/``stream_value`` request event streaming.
    input_field: str = "input"
    stream_field: str = "stream"
    stream_value: bool = True
    # Where the run id appears in the 202 response body (checked in order).
    run_id_keys: tuple[str, ...] = ("run_id", "runId", "id")
    # Run-level terminal SSE event types (case-insensitive substring match against
    # the event's type). VERIFIED LIVE 2026-06-08 against the Hermes runs API: a run
    # emits per-turn events `tool.started` / `tool.completed` / `message.delta` /
    # `reasoning.available`, and the ONLY run-terminal event is `run.completed`
    # (with `run.failed`/`run.cancelled`/etc. as the failure terminals). The earlier
    # bare-substring list ("completed", "error", …) matched `tool.completed` and so
    # ended observation after the run's FIRST tool call — truncating multi-tool turns
    # and the recover-from-4xx retry path (e.g. P3: tree_draft_insert_node 400 → must
    # read catalog + retry). We pin the `run.` prefix so only run-level events
    # terminate; per-turn tool/message/reasoning events never do.
    terminal_event_types: tuple[str, ...] = (
        "run.completed",
        "run.succeeded",
        "run.done",
        "run.failed",
        "run.error",
        "run.errored",
        "run.cancelled",
        "run.canceled",
        "run.stopped",
        "run.timeout",
        "run.timed_out",
        "run.aborted",
        "run.interrupted",
    )
    failure_event_types: tuple[str, ...] = (
        "run.failed",
        "run.error",
        "run.errored",
        "run.cancelled",
        "run.canceled",
        "run.stopped",
        "run.timeout",
        "run.timed_out",
        "run.aborted",
        "run.interrupted",
    )
    # Keys under an event's ``data`` that may carry the final assistant text,
    # checked in order. The first non-empty string wins.
    final_message_keys: tuple[str, ...] = (
        "output_text",
        "outputText",
        "output",
        "message",
        "content",
        "text",
        "response",
    )


DEFAULT_CONTRACT = HermesRunsContract()


@dataclass
class RunResult:
    """Terminal outcome of a Hermes run, assembled from the event stream."""

    run_id: str
    final_message: str
    status: str  # "completed" | "failed" | "cancelled" | "timeout" | "unknown"
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "completed"


class HermesDispatchError(RuntimeError):
    """Raised when the Hermes runs API returns a non-success status or is
    unreachable. The service layer maps this to a ManyForge ``error_envelope``."""

    def __init__(self, message: str, *, code: str = "hermes_dispatch_error") -> None:
        super().__init__(message)
        self.code = code


class HermesSessionDispatcher:
    """Async client for the Hermes ``/v1/runs`` API.

    Constructed per-request (cheap) or reused across requests. ``client`` is
    an ``httpx.AsyncClient``-compatible object injected for testability; the
    service factory passes a real one, tests pass a fake that yields canned
    SSE lines.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        client: Any,
        contract: HermesRunsContract = DEFAULT_CONTRACT,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._api_key = (api_key or "").strip()
        self._client = client
        self._contract = contract
        # Optional hook fired with the run id the instant ``start_run`` returns,
        # before the (possibly long) event stream begins. The service sets this
        # so its cancel route can map an in-flight request_id to a run_id and
        # call ``stop_run`` mid-flight. ``None`` = no-op.
        self.on_run_started: Any = None

    # ---- header / url helpers ------------------------------------------------

    def _headers(self, *, session_id: str, session_key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if session_id:
            headers["X-Hermes-Session-Id"] = session_id
        # The gateway 403s X-Hermes-Session-Key unless API_SERVER_KEY is configured
        # ("requires API key authentication"). The session continuity / long-term
        # memory scoping it enables is only available authenticated, so only send it
        # when we have a bearer key; otherwise it would fail every unauthenticated run.
        if session_key and self._api_key:
            headers["X-Hermes-Session-Key"] = session_key
        return headers

    def _url(self, path: str) -> str:
        return f"{self._base}{path}"

    # ---- lifecycle calls -----------------------------------------------------

    async def start_run(
        self,
        *,
        prompt: str,
        session_id: str,
        session_key: str,
        extra_body: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
    ) -> str:
        """POST /v1/runs; return the run id. Raises HermesDispatchError on
        a non-2xx status or a missing run id."""
        body: dict[str, Any] = {
            self._contract.input_field: prompt,
            self._contract.stream_field: self._contract.stream_value,
        }
        if extra_body:
            body.update(extra_body)
        resp = await self._client.post(
            self._url("/v1/runs"),
            headers=self._headers(session_id=session_id, session_key=session_key),
            json=body,
            timeout=timeout_s,
        )
        status = getattr(resp, "status_code", 0)
        if status not in (200, 201, 202):
            detail = _safe_text(resp)
            raise HermesDispatchError(
                f"POST /v1/runs returned {status}: {detail[:300]}",
                code="hermes_run_start_failed",
            )
        data = _safe_json(resp)
        run_id = ""
        if isinstance(data, dict):
            for key in self._contract.run_id_keys:
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    run_id = value.strip()
                    break
        if not run_id:
            raise HermesDispatchError(
                f"POST /v1/runs succeeded ({status}) but no run id in {self._contract.run_id_keys}; "
                f"body={str(data)[:200]}",
                code="hermes_run_id_missing",
            )
        return run_id

    async def stream_events(
        self, run_id: str, *, session_id: str = "", session_key: str = "", timeout_s: float = 600.0
    ) -> AsyncIterator[dict[str, Any]]:
        """GET /v1/runs/{run_id}/events; yield parsed SSE events as
        ``{"event": <type>, "data": <parsed-json-or-str>}``.

        Tolerant of both ``event:``/``data:`` framed SSE and bare ``data:``
        JSON lines. Never raises mid-stream for a malformed frame — it skips
        it (the observer is best-effort per the plan)."""
        url = self._url(f"/v1/runs/{run_id}/events")
        headers = self._headers(session_id=session_id, session_key=session_key)
        async with self._client.stream("GET", url, headers=headers, timeout=timeout_s) as resp:
            status = getattr(resp, "status_code", 0)
            if status != 200:
                detail = await _safe_aread(resp)
                raise HermesDispatchError(
                    f"GET /v1/runs/{run_id}/events returned {status}: {detail[:200]}",
                    code="hermes_events_failed",
                )
            event_type = ""
            data_lines: list[str] = []
            async for raw in resp.aiter_lines():
                line = raw.rstrip("\r")
                if line == "":  # frame boundary — dispatch what we have
                    if data_lines:
                        yield _assemble_event(event_type, data_lines)
                    event_type = ""
                    data_lines = []
                    continue
                if line.startswith(":"):  # SSE comment / heartbeat
                    continue
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:"):].lstrip())
            if data_lines:  # flush a trailing frame with no terminal blank line
                yield _assemble_event(event_type, data_lines)

    async def get_status(self, run_id: str, *, timeout_s: float = 15.0) -> dict[str, Any]:
        resp = await self._client.get(
            self._url(f"/v1/runs/{run_id}"),
            headers=self._headers(session_id="", session_key=""),
            timeout=timeout_s,
        )
        data = _safe_json(resp)
        return data if isinstance(data, dict) else {}

    async def stop_run(self, run_id: str, *, timeout_s: float = 15.0) -> dict[str, Any]:
        """POST /v1/runs/{run_id}/stop — interrupt a running agent. Best-effort:
        returns the response body, or ``{}`` and logs on failure (cancellation
        must not itself raise into the cancel route)."""
        try:
            resp = await self._client.post(
                self._url(f"/v1/runs/{run_id}/stop"),
                headers=self._headers(session_id="", session_key=""),
                json={},
                timeout=timeout_s,
            )
            return _safe_json(resp) if isinstance(_safe_json(resp), dict) else {}
        except Exception as exc:  # noqa: BLE001 — cancellation is best-effort
            logger.warning("stop_run(%s) failed: %s", run_id, exc)
            return {}

    async def resolve_approval(
        self, run_id: str, *, approval_id: str, approved: bool, timeout_s: float = 15.0
    ) -> dict[str, Any]:
        """POST /v1/runs/{run_id}/approval — resolve a pending approval gate."""
        resp = await self._client.post(
            self._url(f"/v1/runs/{run_id}/approval"),
            headers=self._headers(session_id="", session_key=""),
            json={"approval_id": approval_id, "approved": approved},
            timeout=timeout_s,
        )
        data = _safe_json(resp)
        return data if isinstance(data, dict) else {}

    # ---- orchestration -------------------------------------------------------

    async def run_to_completion(
        self,
        *,
        prompt: str,
        session_id: str,
        session_key: str,
        extra_body: dict[str, Any] | None = None,
        start_timeout_s: float = 30.0,
        stream_timeout_s: float = 600.0,
    ) -> RunResult:
        """Submit a run and consume its event stream to the terminal event.

        Returns a :class:`RunResult` carrying the final assistant message, the
        terminal status, and every observed event (for the progress observer
        and audit). Does NOT loop over tool calls — Hermes owns its agent loop
        and fires its MCP tools (which dispatch to Composer via the lane-neutral
        bridge); this method only submits and observes (plan §5.3).
        """
        run_id = await self.start_run(
            prompt=prompt,
            session_id=session_id,
            session_key=session_key,
            extra_body=extra_body,
            timeout_s=start_timeout_s,
        )
        if callable(self.on_run_started):
            try:
                self.on_run_started(run_id)
            except Exception as exc:  # noqa: BLE001 — the hook must never break a run
                logger.debug("on_run_started hook raised (ignored): %s", exc)
        events: list[dict[str, Any]] = []
        final_message = ""
        status = "unknown"
        error: str | None = None
        try:
            async for event in self.stream_events(
                run_id, session_id=session_id, session_key=session_key, timeout_s=stream_timeout_s
            ):
                events.append(event)
                candidate = self._extract_final_message(event)
                if candidate:
                    final_message = candidate
                etype = str(event.get("event") or "").lower()
                if any(term in etype for term in self._contract.terminal_event_types):
                    if any(fail in etype for fail in self._contract.failure_event_types):
                        status = "failed"
                        error = self._extract_error(event) or f"run terminated: {etype}"
                    else:
                        status = "completed"
                    break
        except Exception:
            # Once a run id exists, a broken event stream can otherwise leave an
            # orphaned Hermes run consuming the single local model slot after the
            # Composer request has already failed. /stop is best-effort and
            # intentionally swallowed by stop_run; preserve the original error.
            await self.stop_run(run_id)
            raise
        if status == "unknown" and events:
            # Stream ended without an explicit terminal event — accept the last
            # message we saw, but flag the ambiguity.
            status = "completed" if final_message else "unknown"
        return RunResult(
            run_id=run_id,
            final_message=final_message,
            status=status,
            events=events,
            error=error,
        )

    def _extract_final_message(self, event: dict[str, Any]) -> str:
        data = event.get("data")
        if isinstance(data, str) and data.strip():
            return data.strip()
        if not isinstance(data, dict):
            return ""
        for key in self._contract.final_message_keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            # Hermes sometimes nests the text under a {type,text} content list.
            if isinstance(value, list):
                parts = [
                    part.get("text")
                    for part in value
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                ]
                joined = "".join(p for p in parts if p)
                if joined.strip():
                    return joined.strip()
        return ""

    def _extract_error(self, event: dict[str, Any]) -> str | None:
        data = event.get("data")
        if isinstance(data, dict):
            for key in ("error", "detail", "message", "reason"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, dict):
                    nested = value.get("detail") or value.get("message")
                    if isinstance(nested, str) and nested.strip():
                        return nested.strip()
        return None


# ---- module helpers ----------------------------------------------------------


def _assemble_event(event_type: str, data_lines: list[str]) -> dict[str, Any]:
    raw = "\n".join(data_lines)
    parsed: Any = raw
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        parsed = raw
    # When the event type isn't on the SSE ``event:`` line, Hermes may carry it
    # inside the JSON payload (``type``/``event``); surface it for the observer.
    if not event_type and isinstance(parsed, dict):
        event_type = str(parsed.get("type") or parsed.get("event") or "").strip()
    return {"event": event_type, "data": parsed}


def _safe_json(resp: Any) -> Any:
    try:
        return resp.json()
    except Exception:  # noqa: BLE001
        return None


def _safe_text(resp: Any) -> str:
    try:
        return resp.text
    except Exception:  # noqa: BLE001
        return ""


async def _safe_aread(resp: Any) -> str:
    try:
        return (await resp.aread()).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""


__all__ = [
    "HermesSessionDispatcher",
    "HermesRunsContract",
    "DEFAULT_CONTRACT",
    "RunResult",
    "HermesDispatchError",
]
