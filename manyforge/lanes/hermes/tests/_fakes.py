"""Test doubles for the Hermes lane: a fake httpx async client (drives the
session dispatcher) and a fake dispatcher (drives the transport + engine).

No third-party deps — everything here is hand-rolled so the lane core can be
tested under base Python.
"""
from __future__ import annotations

import asyncio
from typing import Any


def run(coro: Any) -> Any:
    """Run a coroutine to completion (avoids a pytest-asyncio dependency)."""
    return asyncio.run(coro)


class FakeResponse:
    """Stands in for an httpx Response AND a streaming response context manager."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        json_body: Any = None,
        text: str = "",
        sse_lines: list[str] | None = None,
        stream_exc: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._json = json_body
        self.text = text
        self._sse_lines = sse_lines or []
        self._stream_exc = stream_exc

    def json(self) -> Any:
        if self._json is None:
            raise ValueError("no json body")
        return self._json

    # streaming-response async context manager surface
    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *_exc: Any) -> bool:
        return False

    async def aiter_lines(self):
        for line in self._sse_lines:
            yield line
        if self._stream_exc is not None:
            raise self._stream_exc

    async def aread(self) -> bytes:
        return self.text.encode("utf-8")


class FakeAsyncClient:
    """Minimal stand-in for httpx.AsyncClient used by HermesSessionDispatcher."""

    def __init__(
        self,
        *,
        post_response: FakeResponse | None = None,
        stream_response: FakeResponse | None = None,
        get_response: FakeResponse | None = None,
        post_exc: Exception | None = None,
    ) -> None:
        self._post_response = post_response
        self._stream_response = stream_response
        self._get_response = get_response
        self._post_exc = post_exc
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("POST", url, kwargs))
        if self._post_exc is not None:
            raise self._post_exc
        return self._post_response or FakeResponse(status_code=200, json_body={})

    async def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("GET", url, kwargs))
        return self._get_response or FakeResponse(status_code=200, json_body={})

    def stream(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        # httpx's .stream() returns an async context manager, NOT a coroutine.
        self.calls.append((method, url, kwargs))
        return self._stream_response or FakeResponse(status_code=200, sse_lines=[])


class FakeDispatcher:
    """Stand-in for HermesSessionDispatcher used by transport + engine tests."""

    def __init__(
        self,
        *,
        result: Any = None,
        raise_exc: Exception | None = None,
        base: str = "http://hermes.test:8642",
    ) -> None:
        self._base = base
        self._result = result
        self._raise = raise_exc
        self.on_run_started: Any = None
        self.stopped: list[str] = []
        self.run_calls: list[dict[str, Any]] = []

    async def run_to_completion(self, **kwargs: Any) -> Any:
        self.run_calls.append(kwargs)
        if callable(self.on_run_started):
            self.on_run_started("run-test-123")
        if self._raise is not None:
            raise self._raise
        return self._result

    async def stop_run(self, run_id: str, **_kwargs: Any) -> dict[str, Any]:
        self.stopped.append(run_id)
        return {"stopped": True, "runId": run_id}


class FakeBreaker:
    """Async circuit-breaker double matching the CircuitBreaker surface."""

    def __init__(self, *, enabled: bool = True, allow: bool = True) -> None:
        self.enabled = enabled
        self._allow = allow
        self.successes = 0
        self.failures = 0

    async def before_dispatch(self, _transport: str) -> tuple[bool, str | None]:
        return (True, None) if self._allow else (False, "circuit_open_in_cooldown")

    async def record_success(self, _transport: str) -> None:
        self.successes += 1

    async def record_failure(self, _transport: str) -> None:
        self.failures += 1
