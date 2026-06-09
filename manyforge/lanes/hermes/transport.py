"""``HermesTransport`` — the Hermes lane's :class:`AssistantTransport`.

Per THREE-LANE-MIGRATION-PLAN.md §5.4, every lane implements the same typed
transport so the bridge service and cross-lane harnesses wire any lane in
without branches. The Hermes implementation is distinctive in one way that the
interface comment in :mod:`common.transport` anticipates: **Hermes owns its own
agent loop.** So:

- ``build_request`` packages the assembled agent prompt + session identity into
  a ``POST /v1/runs`` :class:`WireRequest`.
- ``dispatch`` does NOT do a single round-trip; it submits the run and consumes
  the event stream to completion via :class:`HermesSessionDispatcher`. The
  whole Hermes agent loop (including its MCP tool calls, which dispatch to
  Composer through the lane-neutral ``manyforge-mcp-bridge.py``) happens inside
  this one ``dispatch``.
- ``parse_response`` surfaces the run's final assistant message.
- ``normalize_tool_calls`` returns ``[]`` — the bridge never sees individual
  tool calls on this lane (plan §6: "bridge never sees individual tool results,
  only the run's final response + progress events"). Tools that fired are
  observed out-of-band by :class:`HermesProgressObserver` from the same event
  stream and recorded in the universal audit; the hard source of truth for
  applied mutations is Composer's bridge-tools callback log.
"""
from __future__ import annotations

import time
from typing import Any, Literal

from common.transport import ParsedResponse, SessionCtx, WireRequest, WireResponse

from .session_dispatcher import HermesDispatchError, HermesSessionDispatcher, RunResult

# Private body keys build_request stashes for dispatch to read. Underscore-
# prefixed so they never collide with a real Hermes runs-API field.
_PROMPT_KEY = "input"
_SESSION_ID_KEY = "_hermes_session_id"
_SESSION_KEY_KEY = "_hermes_session_key"


class HermesTransport:
    """:class:`AssistantTransport` implementation for the Hermes lane.

    ``dispatcher`` is injected so the service factory supplies a real
    :class:`HermesSessionDispatcher` (httpx-backed) and tests supply a fake.
    """

    lane: Literal["direct", "openclaw", "hermes"] = "hermes"

    def __init__(
        self,
        *,
        dispatcher: HermesSessionDispatcher,
        base_url: str,
        model: str = "",
    ) -> None:
        self._dispatcher = dispatcher
        self._base = base_url.rstrip("/")
        self.model = model

    def build_request(self, prompt: str, ctx: SessionCtx) -> WireRequest:
        session_key = str(ctx.extra.get("hermes_session_key") or ctx.session_key or "")
        body: dict[str, Any] = {
            _PROMPT_KEY: prompt,
            "stream": True,
            _SESSION_ID_KEY: ctx.conversation_id,
            _SESSION_KEY_KEY: session_key,
        }
        # Headers here are for audit/inspection; the dispatcher rebuilds the
        # authoritative header set (incl. Bearer auth) at send time.
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Hermes-Session-Id": ctx.conversation_id,
            "X-Hermes-Session-Key": session_key,
        }
        return WireRequest(
            method="POST",
            url=f"{self._base}/v1/runs",
            headers=headers,
            body=body,
        )

    async def dispatch(self, request: WireRequest, *, timeout_s: float) -> WireResponse:
        body = request.body or {}
        prompt = str(body.get(_PROMPT_KEY) or "")
        session_id = str(body.get(_SESSION_ID_KEY) or "")
        session_key = str(body.get(_SESSION_KEY_KEY) or "")
        # Pass through any non-private extras (e.g. a future stream flag) to the
        # runs API, stripping our private bookkeeping keys.
        extra_body = {
            k: v
            for k, v in body.items()
            if k not in (_PROMPT_KEY, _SESSION_ID_KEY, _SESSION_KEY_KEY, "stream")
        }
        started = time.perf_counter()
        try:
            result: RunResult = await self._dispatcher.run_to_completion(
                prompt=prompt,
                session_id=session_id,
                session_key=session_key,
                extra_body=extra_body or None,
                stream_timeout_s=timeout_s,
            )
        except HermesDispatchError as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            return WireResponse(
                status=502,
                headers={},
                body={
                    "runId": "",
                    "finalMessage": "",
                    "status": "failed",
                    "error": str(exc),
                    "errorCode": exc.code,
                    "events": [],
                },
                duration_ms=duration_ms,
            )
        duration_ms = (time.perf_counter() - started) * 1000.0
        return WireResponse(
            status=200 if result.ok else 502,
            headers={},
            body={
                "runId": result.run_id,
                "finalMessage": result.final_message,
                "status": result.status,
                "error": result.error,
                "events": result.events,
            },
            duration_ms=duration_ms,
        )

    def parse_response(self, response: WireResponse) -> ParsedResponse:
        body = response.body if isinstance(response.body, dict) else {}
        return ParsedResponse(
            message_content=str(body.get("finalMessage") or ""),
            tool_calls=[],  # see normalize_tool_calls / class docstring
            finish_reason=str(body.get("status") or "unknown"),
            raw=response,
        )

    def normalize_tool_calls(self, parsed: ParsedResponse) -> list[dict[str, Any]]:
        # Hermes runs its own agent loop and fires tools through its MCP runtime;
        # the bridge does not receive structured tool_calls per turn. Tools that
        # fired are observed via HermesProgressObserver and recorded in the
        # universal audit. Returning [] keeps the interface honest rather than
        # fabricating tool_calls the bridge never saw.
        return []


__all__ = ["HermesTransport"]
