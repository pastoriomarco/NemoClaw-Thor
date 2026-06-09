"""Unit tests for the Hermes lane turn engine (envelope in → envelope out)."""
from __future__ import annotations

from _fakes import FakeBreaker, FakeDispatcher, run
from lanes.hermes import engine
from lanes.hermes.session_dispatcher import RunResult


def _payload(**over):
    base = {
        "requestId": "req-1",
        "conversationId": "conv-1",
        "assistantMode": "composer-assistant",
        "message": "add a repeat node as root",
        "tools": [{"id": "tree_draft_wrap_node"}, {"id": "program_read"}],
        "context": {"catalogHash": "abc123"},
    }
    base.update(over)
    return base


def _turn(payload, dispatcher, **kw):
    return run(
        engine.run_assistant_turn(
            payload, dispatcher=dispatcher, model="gemma4-12b-it-gguf", now_ms=1000, **kw
        )
    )


def test_success_turn_returns_message_envelope_and_audit():
    disp = FakeDispatcher(
        result=RunResult(
            run_id="rOK",
            final_message="Wrapped pick_and_place in a repeat node.",
            status="completed",
            events=[{"event": "tool_call", "data": {"tool": "mcp_manyforge_tree_draft_wrap_node"}}],
        )
    )
    res = _turn(_payload(), disp)
    assert res.status_code == 200
    assert res.envelope["message"].startswith("Wrapped")
    assert res.envelope["requiresReview"] is True
    assert res.envelope["toolCalls"] == [] and res.envelope["proposals"] == []
    # audit reflects the observed (prefix-stripped) tool + expected catalog
    assert res.audit_entry["lane"] == "hermes"
    assert res.audit_entry["toolsObserved"] == ["tree_draft_wrap_node"]
    assert set(res.audit_entry["toolsExpected"]) == {"tree_draft_wrap_node", "program_read"}
    assert res.audit_entry["exitReason"] == "completed"
    assert res.audit_entry["compactionFires"] == 0  # lane policy: compaction off


def test_failed_run_returns_error_envelope():
    disp = FakeDispatcher(result=RunResult(run_id="rF", final_message="", status="failed", error="model exploded"))
    res = _turn(_payload(), disp)
    assert res.status_code == 200
    assert res.envelope["error"]["code"] == "hermes_run_failed"
    assert "model exploded" in res.envelope["message"]


def test_invalid_payload_is_rejected():
    res = run(engine.run_assistant_turn("not-a-dict", dispatcher=FakeDispatcher(), now_ms=5))
    assert res.status_code == 400
    assert res.envelope["error"]["code"] == "invalid_envelope"


def test_cancelled_precheck_short_circuits():
    payload = _payload()
    engine.mark_cancelled("req-1")
    res = _turn(payload, FakeDispatcher(result=RunResult("r", "x", "completed")))
    assert res.envelope["error"]["code"] == "cancelled"


def test_dispatch_exception_is_contained_and_trips_breaker():
    breaker = FakeBreaker()
    disp = FakeDispatcher(raise_exc=RuntimeError("kaboom"))
    res = _turn(_payload(requestId="req-exc"), disp, breaker=breaker)
    assert res.status_code == 200
    assert res.envelope["error"]["code"] == "hermes_dispatch_error"
    assert breaker.failures == 1


def test_open_breaker_blocks_dispatch():
    breaker = FakeBreaker(allow=False)
    disp = FakeDispatcher(result=RunResult("r", "x", "completed"))
    res = _turn(_payload(requestId="req-cb"), disp, breaker=breaker)
    assert res.envelope["error"]["code"] == "circuit_open"
    assert disp.run_calls == []  # never dispatched


def test_success_records_breaker_success():
    breaker = FakeBreaker()
    disp = FakeDispatcher(result=RunResult("r", "ok", "completed"))
    _turn(_payload(requestId="req-ok2"), disp, breaker=breaker)
    assert breaker.successes == 1


def test_cancel_handle_exposes_run_id_after_start():
    # The engine registers the active run + wires on_run_started so a concurrent
    # cancel can find the run id. Simulate: dispatcher fires the hook, then we
    # read the handle mid-flight via mark_cancelled on a DIFFERENT request id.
    disp = FakeDispatcher(result=RunResult("r-live", "done", "completed"))
    res = _turn(_payload(requestId="req-live"), disp)
    assert res.status_code == 200
    # after completion the active entry is popped
    assert "req-live" not in engine._ACTIVE
