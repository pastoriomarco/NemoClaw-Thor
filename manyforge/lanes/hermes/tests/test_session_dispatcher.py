"""Unit tests for the Hermes /v1/runs session dispatcher."""
from __future__ import annotations

import pytest

from _fakes import FakeAsyncClient, FakeResponse, run
from lanes.hermes.session_dispatcher import (
    HermesDispatchError,
    HermesSessionDispatcher,
)


def _dispatcher(client: FakeAsyncClient) -> HermesSessionDispatcher:
    return HermesSessionDispatcher(base_url="http://hermes.test:8642", api_key="secret", client=client)


def test_start_run_parses_run_id_and_sets_auth_headers():
    client = FakeAsyncClient(post_response=FakeResponse(status_code=202, json_body={"run_id": "r1"}))
    disp = _dispatcher(client)
    run_id = run(disp.start_run(prompt="hi", session_id="c1", session_key="k1"))
    assert run_id == "r1"
    # auth + session headers were sent
    _, url, kw = client.calls[0]
    assert url.endswith("/v1/runs")
    headers = kw["headers"]
    assert headers["Authorization"] == "Bearer secret"
    assert headers["X-Hermes-Session-Id"] == "c1"
    assert headers["X-Hermes-Session-Key"] == "k1"
    assert kw["json"]["input"] == "hi"


def test_start_run_accepts_alternate_run_id_keys():
    client = FakeAsyncClient(post_response=FakeResponse(status_code=200, json_body={"runId": "r2"}))
    assert run(_dispatcher(client).start_run(prompt="x", session_id="c", session_key="")) == "r2"


def test_start_run_raises_on_error_status():
    client = FakeAsyncClient(post_response=FakeResponse(status_code=500, text="boom"))
    with pytest.raises(HermesDispatchError) as ei:
        run(_dispatcher(client).start_run(prompt="x", session_id="c", session_key=""))
    assert ei.value.code == "hermes_run_start_failed"


def test_start_run_raises_when_run_id_missing():
    client = FakeAsyncClient(post_response=FakeResponse(status_code=202, json_body={"unexpected": 1}))
    with pytest.raises(HermesDispatchError) as ei:
        run(_dispatcher(client).start_run(prompt="x", session_id="c", session_key=""))
    assert ei.value.code == "hermes_run_id_missing"


def test_stream_events_parses_sse_frames():
    lines = [
        "event: tool_call",
        'data: {"tool": "mcp_manyforge_program_read"}',
        "",
        ": heartbeat",  # comment, ignored
        "event: run.completed",
        'data: {"output_text": "done"}',
        "",
    ]
    client = FakeAsyncClient(stream_response=FakeResponse(status_code=200, sse_lines=lines))
    events = list(_collect(_dispatcher(client).stream_events("r1")))
    assert [e["event"] for e in events] == ["tool_call", "run.completed"]
    assert events[0]["data"]["tool"] == "mcp_manyforge_program_read"
    assert events[1]["data"]["output_text"] == "done"


def test_stream_events_infers_type_from_payload_when_no_event_line():
    lines = ['data: {"type": "run.failed", "error": "nope"}', ""]
    client = FakeAsyncClient(stream_response=FakeResponse(status_code=200, sse_lines=lines))
    events = list(_collect(_dispatcher(client).stream_events("r1")))
    assert events[0]["event"] == "run.failed"


def test_run_to_completion_collects_final_message_and_fires_hook():
    post = FakeResponse(status_code=202, json_body={"run_id": "rX"})
    stream = FakeResponse(
        status_code=200,
        sse_lines=[
            "event: tool_call",
            'data: {"tool": "mcp_manyforge_scene_inspect"}',
            "",
            "event: run.completed",
            'data: {"output_text": "all done"}',
            "",
        ],
    )
    client = FakeAsyncClient(post_response=post, stream_response=stream)
    disp = _dispatcher(client)
    started: list[str] = []
    disp.on_run_started = started.append
    result = run(disp.run_to_completion(prompt="go", session_id="c1", session_key="k1"))
    assert result.ok and result.status == "completed"
    assert result.final_message == "all done"
    assert result.run_id == "rX"
    assert started == ["rX"]  # hook fired with the run id before streaming finished
    assert len(result.events) == 2


def test_run_to_completion_marks_failure_and_captures_error():
    post = FakeResponse(status_code=202, json_body={"run_id": "rF"})
    stream = FakeResponse(
        status_code=200,
        sse_lines=['event: run.failed', 'data: {"error": "model exploded"}', ""],
    )
    result = run(
        _dispatcher(FakeAsyncClient(post_response=post, stream_response=stream)).run_to_completion(
            prompt="go", session_id="c", session_key=""
        )
    )
    assert result.status == "failed"
    assert result.error == "model exploded"
    assert not result.ok


def test_run_to_completion_stops_run_when_event_stream_fails():
    post = FakeResponse(status_code=202, json_body={"run_id": "rOrphan"})
    stream = FakeResponse(
        status_code=200,
        sse_lines=["event: tool_call", 'data: {"tool": "mcp_manyforge_program_read"}', ""],
        stream_exc=RuntimeError("stream broke"),
    )
    client = FakeAsyncClient(post_response=post, stream_response=stream)
    with pytest.raises(RuntimeError, match="stream broke"):
        run(_dispatcher(client).run_to_completion(prompt="go", session_id="c", session_key=""))

    assert [call[1] for call in client.calls] == [
        "http://hermes.test:8642/v1/runs",
        "http://hermes.test:8642/v1/runs/rOrphan/events",
        "http://hermes.test:8642/v1/runs/rOrphan/stop",
    ]


def test_stop_run_is_best_effort_on_exception():
    client = FakeAsyncClient(post_exc=RuntimeError("network down"))
    assert run(_dispatcher(client).stop_run("rZ")) == {}  # swallowed, returns {}


# --- helper: collect an async generator synchronously --------------------------


def _collect(agen):
    async def _drain():
        return [item async for item in agen]

    return run(_drain())
