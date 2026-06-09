"""Unit tests for HermesTransport (AssistantTransport conformance + behavior)."""
from __future__ import annotations

from _fakes import FakeDispatcher, run
from common.transport import AssistantTransport, SessionCtx, WireResponse
from lanes.hermes.session_dispatcher import HermesDispatchError, RunResult
from lanes.hermes.transport import HermesTransport


def _ctx(**over):
    base = dict(
        conversation_id="conv-1",
        principal="hermes-sandbox",
        assistant_mode="composer-assistant",
        request_id="req-1",
        session_key="sk-1",
        catalog_hash="hash-1",
        discovery_mode="hermes_direct",
    )
    base.update(over)
    return SessionCtx(**base)


def test_conforms_to_assistant_transport_protocol():
    t = HermesTransport(dispatcher=FakeDispatcher(), base_url="http://hermes.test:8642")
    assert isinstance(t, AssistantTransport)
    assert t.lane == "hermes"


def test_build_request_targets_runs_with_session_headers():
    t = HermesTransport(dispatcher=FakeDispatcher(), base_url="http://hermes.test:8642")
    wire = t.build_request("the prompt", _ctx())
    assert wire.method == "POST"
    assert wire.url == "http://hermes.test:8642/v1/runs"
    assert wire.body["input"] == "the prompt"
    assert wire.body["_hermes_session_id"] == "conv-1"
    assert wire.headers["X-Hermes-Session-Id"] == "conv-1"
    assert wire.headers["X-Hermes-Session-Key"] == "sk-1"


def test_build_request_prefers_extra_session_key():
    t = HermesTransport(dispatcher=FakeDispatcher(), base_url="http://h:8642")
    ctx = _ctx()
    ctx.extra["hermes_session_key"] = "override-key"
    wire = t.build_request("p", ctx)
    assert wire.body["_hermes_session_key"] == "override-key"


def test_dispatch_success_packs_run_result():
    disp = FakeDispatcher(
        result=RunResult(run_id="rA", final_message="done", status="completed", events=[{"event": "x"}])
    )
    t = HermesTransport(dispatcher=disp, base_url="http://h:8642")
    wire = t.build_request("p", _ctx())
    resp = run(t.dispatch(wire, timeout_s=30))
    assert isinstance(resp, WireResponse)
    assert resp.status == 200
    assert resp.body["finalMessage"] == "done"
    assert resp.body["runId"] == "rA"
    # the private bookkeeping keys are stripped before hitting the runs API
    assert disp.run_calls[0]["prompt"] == "p"
    assert disp.run_calls[0]["session_id"] == "conv-1"


def test_dispatch_failed_run_maps_to_502():
    disp = FakeDispatcher(result=RunResult(run_id="rB", final_message="", status="failed", error="bad"))
    t = HermesTransport(dispatcher=disp, base_url="http://h:8642")
    resp = run(t.dispatch(t.build_request("p", _ctx()), timeout_s=30))
    assert resp.status == 502
    assert resp.body["status"] == "failed"
    assert resp.body["error"] == "bad"


def test_dispatch_catches_dispatch_error():
    disp = FakeDispatcher(raise_exc=HermesDispatchError("unreachable", code="hermes_events_failed"))
    t = HermesTransport(dispatcher=disp, base_url="http://h:8642")
    resp = run(t.dispatch(t.build_request("p", _ctx()), timeout_s=30))
    assert resp.status == 502
    assert resp.body["errorCode"] == "hermes_events_failed"


def test_parse_response_and_normalize_tool_calls():
    t = HermesTransport(dispatcher=FakeDispatcher(), base_url="http://h:8642")
    resp = WireResponse(
        status=200, headers={}, body={"finalMessage": "hi there", "status": "completed", "events": []}, duration_ms=12.0
    )
    parsed = t.parse_response(resp)
    assert parsed.message_content == "hi there"
    assert parsed.finish_reason == "completed"
    # Hermes owns its loop; the bridge never sees structured tool_calls per turn.
    assert t.normalize_tool_calls(parsed) == []
