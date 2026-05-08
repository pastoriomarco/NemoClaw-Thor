"""FastAPI service for routing ManyForge assistant requests through OpenClaw."""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .adapter import (
    AdapterConfig,
    AgentRunResult,
    build_agent_prompt,
    build_gateway_chat_completions_command,
    build_openclaw_command,
    error_envelope,
    mcp_allowed_tools_from_payload,
    normalize_agent_response,
    normalize_chat_completions_response,
    parse_chat_completions_response,
    parse_openclaw_json,
    request_id_from_payload,
)


HOST = os.environ.get("OPENCLAW_ASSISTANT_BRIDGE_HOST", "127.0.0.1")
PORT = int(os.environ.get("OPENCLAW_ASSISTANT_BRIDGE_PORT", "8200"))


# Sampling defaults (temperature, top_k, top_p, chat_template_kwargs)
# are owned by vLLM at the server level — see
# nemoclaw-thor/serving/launch.sh `--override-generation-config` and
# `--default-chat-template-kwargs`. The bridge no longer injects per-
# request sampling fields. If you need to override on a single request,
# AdapterConfig still has gateway_temperature / gateway_top_k /
# gateway_top_p / gateway_enable_thinking fields (default None — not
# added to the body) that callers can patch directly.

# Composer base used to register the principal-binding so that live
# tool-call streaming can correlate MCP bridge calls back to the
# active chat. See the matching endpoints in
# manyforge_composer.backend.routes_assistant.bind_principal.
_COMPOSER_BASE = os.environ.get("OPENCLAW_ASSISTANT_COMPOSER_BASE", "http://127.0.0.1:9000").rstrip("/")
_ACTIVE_PROCESSES: dict[str, asyncio.subprocess.Process] = {}
_ACTIVE_REQUESTS: dict[str, dict[str, Any]] = {}
_CANCELLED: set[str] = set()


def _principal_for(cfg: "AdapterConfig") -> str:
    """The principal the in-sandbox MCP bridge subprocess uses on
    /api/assistant/bridge/tools envelopes. Mirrors the value set by
    the provisioner in setup-manyforge-assistant.sh
    (`openclaw-${SANDBOX}`)."""
    return f"openclaw-{cfg.sandbox}"


async def _bind_principal_async(principal: str, request_id: str) -> None:
    """Register the active chat for `principal` so the bridge endpoint
    can correlate tool calls coming from the MCP subprocess.
    Best-effort; failures are non-fatal."""
    if not principal or not request_id:
        return
    try:
        body = json.dumps({"requestId": request_id}).encode()
        proc = await asyncio.create_subprocess_exec(
            "curl", "-sS", "-X", "PUT",
            "-H", "Content-Type: application/json",
            "--max-time", "3",
            f"{_COMPOSER_BASE}/api/assistant/bridge/principal-binding/{principal}",
            "-d", body.decode("utf-8"),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
    except Exception:
        pass


async def _unbind_principal_async(principal: str) -> None:
    if not principal:
        return
    try:
        proc = await asyncio.create_subprocess_exec(
            "curl", "-sS", "-X", "DELETE",
            "--max-time", "3",
            f"{_COMPOSER_BASE}/api/assistant/bridge/principal-binding/{principal}",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
    except Exception:
        pass


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_optional_bool(name: str, default: bool | None) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if raw.strip().lower() in {"0", "false", "no", "off"}:
        return False
    return default


def _config_from_env() -> AdapterConfig:
    # Sampling defaults (temperature/top_k/top_p/enable_thinking) are
    # not resolved here — vLLM owns them server-side via
    # --override-generation-config / --default-chat-template-kwargs in
    # nemoclaw-thor/serving/launch.sh. AdapterConfig defaults all those
    # fields to None so the adapter omits them from the request body
    # and lets vLLM apply its own defaults.
    return AdapterConfig(
        sandbox=os.environ.get("OPENCLAW_ASSISTANT_SANDBOX", "my-assistant"),
        namespace=os.environ.get("OPENCLAW_ASSISTANT_NAMESPACE", "openshell"),
        container=os.environ.get("OPENCLAW_ASSISTANT_CONTAINER", "agent"),
        cluster_container=os.environ.get(
            "OPENCLAW_ASSISTANT_CLUSTER_CONTAINER",
            "openshell-cluster-nemoclaw",
        ),
        sandbox_user=os.environ.get("OPENCLAW_ASSISTANT_SANDBOX_USER", "sandbox"),
        agent=os.environ.get("OPENCLAW_ASSISTANT_AGENT", "main"),
        timeout_s=float(os.environ.get("OPENCLAW_ASSISTANT_TIMEOUT_S", "120")),
        openclaw_bin=os.environ.get("OPENCLAW_ASSISTANT_BIN", "openclaw"),
        local=os.environ.get("OPENCLAW_ASSISTANT_LOCAL", "false").lower()
        in {"1", "true", "yes"},
        thinking=os.environ.get("OPENCLAW_ASSISTANT_THINKING", "off"),
        auto_tool_window=os.environ.get("OPENCLAW_ASSISTANT_AUTO_TOOL_WINDOW", "true").lower()
        in {"1", "true", "yes"},
        allowed_tools_file=os.environ.get(
            "OPENCLAW_ASSISTANT_ALLOWED_TOOLS_FILE",
            "/tmp/manyforge-openclaw-allowed-tools.txt",
        ),
        use_gateway=os.environ.get("OPENCLAW_ASSISTANT_USE_GATEWAY", "false").lower()
        in {"1", "true", "yes"},
        gateway_port=int(os.environ.get("OPENCLAW_ASSISTANT_GATEWAY_PORT", "18789")),
        gateway_max_tokens=int(
            os.environ.get("OPENCLAW_ASSISTANT_GATEWAY_MAX_TOKENS", "4096")
        ),
    )


app = FastAPI(
    title="ManyForge OpenClaw assistant-provider adapter",
    version="0.1.0",
)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    cfg = _config_from_env()
    now = time.perf_counter()
    active = []
    for request_id, meta in sorted(_ACTIVE_REQUESTS.items()):
        started = float(meta.get("startedPerf") or now)
        active.append(
            {
                "requestId": request_id,
                "stage": meta.get("stage") or "running",
                "elapsedMs": round((now - started) * 1000.0, 1),
                "allowedMcpTools": list(meta.get("allowedMcpTools") or []),
                "promptChars": meta.get("promptChars"),
            }
        )
    return {
        "status": "ok",
        "provider": "openclaw",
        "sandbox": cfg.sandbox,
        "namespace": cfg.namespace,
        "container": cfg.container,
        "agent": cfg.agent,
        "activeRequests": active,
        "activeRequestIds": sorted(_ACTIVE_PROCESSES),
    }


@app.post("/v1/manyforge/assistant/{request_id}/cancel")
async def cancel_request(request_id: str) -> dict[str, Any]:
    request_id = request_id.strip()
    if not request_id:
        return {"cancelled": False, "detail": "request_id is blank"}
    _CANCELLED.add(request_id)
    proc = _ACTIVE_PROCESSES.get(request_id)
    if proc is not None and proc.returncode is None:
        proc.kill()
    meta = _ACTIVE_REQUESTS.get(request_id) or {}
    cfg = _config_from_env()
    session_id = str(meta.get("sessionId") or request_id)
    await _kill_sandbox_agent(cfg, session_id=session_id)
    return {"requestId": request_id, "cancelled": True}


@app.post("/v1/manyforge/assistant")
async def assistant(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        body = error_envelope(
            request_id=f"openclaw-adapter-{int(time.time() * 1000)}",
            code="invalid_json",
            detail=f"request body is not valid JSON: {exc}",
        )
        return JSONResponse(status_code=400, content=body)
    if not isinstance(payload, dict):
        body = error_envelope(
            request_id=f"openclaw-adapter-{int(time.time() * 1000)}",
            code="invalid_envelope",
            detail="request body must be a JSON object",
        )
        return JSONResponse(status_code=400, content=body)

    request_id = request_id_from_payload(payload)
    if request_id in _CANCELLED:
        _CANCELLED.discard(request_id)
        body = error_envelope(
            request_id=request_id,
            code="cancelled",
            detail=f"request {request_id} was cancelled before dispatch",
        )
        return JSONResponse(status_code=200, content=body)

    cfg = _config_from_env()
    total_started = time.perf_counter()
    prompt_started = time.perf_counter()
    # Gateway path skips the prompt-augmentation work the CLI path needs.
    # The persistent gateway has the manyforge MCP server registered with
    # mode-scoped enforcement at provisioner time, so the model already sees
    # the right tool surface for tools/list. But we still need to inject the
    # request preamble (nodeCatalog with kind/childrenMin/Max/parameters,
    # programSnapshot, sceneSnapshot, instructional text) so the model knows
    # the structural rules and live state without burning turns on
    # program.read / scene.inspect.
    #
    # Earlier (pre 2026-05-06) the gateway path passed JUST the user's
    # message and the model had to discover everything via tool calls.
    # That worked for trivial prompts but bottlenecked on multi-step
    # mutations: the model could spend 60-120s reading state before doing
    # anything. Build the same enriched prompt as the CLI shell-out path
    # so both paths give the model identical context. The extra preamble
    # is small (~5-15 KB) and gets trimmed if context is tight.
    if cfg.use_gateway:
        # Gateway path: do NOT narrow `allowedTools` per-request. The
        # in-sandbox MCP wrapper is launched once at gateway startup
        # and exposes whatever set was active then to vLLM via
        # `tools/list` — the bridge has no in-band channel to update
        # that allowlist per request. Narrowing the prompt's
        # `allowedTools` JSON while leaving the wrapper's `tools/list`
        # wide creates a disagreement the model resolves intermittently
        # ("I can't use program_read because it isn't available", even
        # though program_read IS in the schema list). Sending the full
        # mode surface in both places keeps them in lockstep — the
        # mode catalog is the real authorization boundary, and the
        # narrow window was only ever a soft prompt-economy hint.
        allowed_mcp_tools = None
        prompt = build_agent_prompt(payload, mcp_allowed_tools=allowed_mcp_tools)
    else:
        # CLI shell-out path: the per-request allowlist is written to
        # /tmp/manyforge-openclaw-allowed-tools.txt before openclaw
        # launches and the wrapper reads it on startup, so prompt and
        # `tools/list` agree. Narrowing here is safe.
        inferred_mcp_tools = (
            mcp_allowed_tools_from_payload(payload) if cfg.auto_tool_window else []
        )
        allowed_mcp_tools = inferred_mcp_tools or None
        prompt = build_agent_prompt(payload, mcp_allowed_tools=allowed_mcp_tools)
    prompt_ms = (time.perf_counter() - prompt_started) * 1000.0
    timeout_s = _request_timeout(payload, cfg.timeout_s)
    session_id = str(payload.get("conversationId") or request_id)
    _ACTIVE_REQUESTS[request_id] = {
        "startedPerf": time.perf_counter(),
        "stage": "dispatching_openclaw_gateway" if cfg.use_gateway else "dispatching_openclaw",
        "allowedMcpTools": allowed_mcp_tools or [],
        "promptChars": len(prompt),
        "sessionId": session_id,
        "transport": "gateway_http" if cfg.use_gateway else "cli_shell_out",
    }
    _log_event(
        "openclaw_request_started",
        requestId=request_id,
        timeoutS=timeout_s,
        allowedMcpTools=allowed_mcp_tools or [],
        promptChars=len(prompt),
        transport="gateway_http" if cfg.use_gateway else "cli_shell_out",
    )
    # Register the principal-binding for live tool-call streaming.
    # The MCP bridge subprocess substitutes its own bridge-process
    # conversation id in tool-call envelopes; this binding lets the
    # bridge endpoint resolve `principal -> active chat request` and
    # surface tool calls on the right assistant request in the UI.
    # Single-tenant per sandbox, so each new chat overwrites the
    # binding and explicit unbind is just hygiene.
    principal = _principal_for(cfg)
    await _bind_principal_async(principal, request_id)
    try:
        if cfg.use_gateway:
            command = build_gateway_chat_completions_command(
                config=cfg,
                payload=payload,
                timeout_s=timeout_s,
                # Pass the enriched prompt (preamble + request context +
                # user request) explicitly. Without this, the gateway-
                # path builder silently dropped the preamble and only
                # forwarded the raw user message — so all the structural
                # hints, snapshots, and instruction text were invisible
                # to the model. See adapter.py for the full back-story.
                message=prompt,
            )
        else:
            command = build_openclaw_command(
                config=cfg,
                message=prompt,
                timeout_s=timeout_s,
                session_id=session_id,
                mcp_allowed_tools=allowed_mcp_tools,
            )
        result = await _run_agent(
            request_id=request_id,
            command=command,
            timeout_s=timeout_s,
        )
    except asyncio.TimeoutError:
        await _kill_sandbox_agent(cfg, session_id=session_id)
        _log_event(
            "openclaw_request_timeout",
            requestId=request_id,
            timeoutS=timeout_s,
            allowedMcpTools=allowed_mcp_tools or [],
        )
        body = error_envelope(
            request_id=request_id,
            code="timeout",
            detail=f"OpenClaw agent exceeded timeout {timeout_s:.1f}s",
        )
        return JSONResponse(status_code=504, content=body)
    except RuntimeError as exc:
        _log_event("openclaw_request_error", requestId=request_id, detail=str(exc))
        body = error_envelope(
            request_id=request_id,
            code="upstream_call_error",
            detail=str(exc),
        )
        return JSONResponse(status_code=502, content=body)

    if request_id in _CANCELLED:
        _CANCELLED.discard(request_id)
        body = error_envelope(
            request_id=request_id,
            code="cancelled",
            detail=f"request {request_id} was cancelled",
        )
        return JSONResponse(status_code=200, content=body)

    if result.returncode != 0:
        detail = (
            f"OpenClaw agent exited with code {result.returncode}: "
            f"{(result.stderr or result.stdout)[:1000]}"
        )
        body = error_envelope(
            request_id=request_id,
            code="upstream_call_error",
            detail=detail,
        )
        return JSONResponse(status_code=502, content=body)

    parse_started = time.perf_counter()
    if cfg.use_gateway:
        # Gateway returns OpenAI chat.completion shape on stdout.
        response_json, parse_warnings = parse_chat_completions_response(result.stdout)
        body = normalize_chat_completions_response(
            payload=payload,
            response_json=response_json,
            stdout=result.stdout,
            stderr=result.stderr,
            parse_warnings=parse_warnings,
        )
    else:
        output_stream = result.stdout if result.stdout.strip() else result.stderr
        diagnostic_stderr = result.stderr if result.stdout.strip() else ""
        agent_json, parse_warnings = parse_openclaw_json(output_stream)
        body = normalize_agent_response(
            payload=payload,
            agent_json=agent_json,
            stdout=output_stream,
            stderr=diagnostic_stderr,
            parse_warnings=parse_warnings,
        )
    parse_normalize_ms = (time.perf_counter() - parse_started) * 1000.0
    total_ms = (time.perf_counter() - total_started) * 1000.0
    body["openclaw"] = {
        "adapter": "openclaw_assistant_bridge",
        "transport": "gateway_http" if cfg.use_gateway else "cli_shell_out",
        "durationMs": round(result.duration_ms, 1),
        "timings": {
            "promptBuildMs": round(prompt_ms, 1),
            "agentRunMs": round(result.duration_ms, 1),
            "parseNormalizeMs": round(parse_normalize_ms, 1),
            "totalAdapterMs": round(total_ms, 1),
        },
        "promptChars": len(prompt),
        "stdoutBytes": len(result.stdout.encode("utf-8")),
        "stderrBytes": len(result.stderr.encode("utf-8")),
        "allowedMcpTools": allowed_mcp_tools or [],
        "sandbox": cfg.sandbox,
        "agent": cfg.agent,
    }
    _log_event(
        "openclaw_request_complete",
        requestId=request_id,
        totalAdapterMs=round(total_ms, 1),
        agentRunMs=round(result.duration_ms, 1),
        allowedMcpTools=allowed_mcp_tools or [],
        promptChars=len(prompt),
    )
    return JSONResponse(status_code=200, content=body)


async def _run_agent(
    *,
    request_id: str,
    command: list[str],
    timeout_s: float,
) -> AgentRunResult:
    started = time.perf_counter()
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        _ACTIVE_REQUESTS.pop(request_id, None)
        raise RuntimeError(f"required command not found: {command[0]}") from exc
    _ACTIVE_PROCESSES[request_id] = proc
    if request_id in _ACTIVE_REQUESTS:
        _ACTIVE_REQUESTS[request_id]["stage"] = "openclaw_agent_running"
    try:
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_s + 5.0,
            )
        except asyncio.TimeoutError:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise
    finally:
        _ACTIVE_PROCESSES.pop(request_id, None)
        _ACTIVE_REQUESTS.pop(request_id, None)
    return AgentRunResult(
        stdout=stdout_b.decode("utf-8", errors="replace"),
        stderr=stderr_b.decode("utf-8", errors="replace"),
        returncode=proc.returncode if proc.returncode is not None else -1,
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )


def _request_timeout(payload: dict[str, Any], default_s: float) -> float:
    for key in ("timeoutSeconds", "timeout_s", "default_timeout_s"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    runtime = payload.get("runtime")
    if isinstance(runtime, dict):
        value = runtime.get("defaultTimeoutS")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return default_s


def _log_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


async def _kill_sandbox_agent(config: AdapterConfig, *, session_id: str) -> None:
    if not session_id:
        return
    remote = "pkill -f " + shlex.quote(session_id) + " || true"
    command = [
        "docker",
        "exec",
        config.cluster_container,
        "kubectl",
        "exec",
        "-n",
        config.namespace,
        config.sandbox,
        "-c",
        config.container,
        "--",
        "sh",
        "-lc",
        remote,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        _log_event(
            "openclaw_sandbox_kill_failed",
            sessionId=session_id,
            detail=f"{type(exc).__name__}: {exc}",
        )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "openclaw_assistant_bridge.service:app",
        host=HOST,
        port=PORT,
        log_level=os.environ.get("OPENCLAW_ASSISTANT_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
