"""FastAPI service for routing ManyForge assistant requests through OpenClaw."""
from __future__ import annotations

import asyncio
import base64
import collections
import json
import os
import re
import shlex
import time
from collections import OrderedDict
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .adapter import (
    AdapterConfig,
    AgentRunResult,
    build_agent_prompt,
    build_gateway_chat_completions_command,
    build_openclaw_command,
    derive_gateway_session_key,
    error_envelope,
    normalize_agent_response,
    normalize_chat_completions_response,
    parse_chat_completions_response,
    parse_openclaw_json,
    request_id_from_payload,
)
from .circuit_breaker import CircuitBreaker
from .metrics import (
    ACTIVE_REQUESTS,
    COMPACT_FIRES_TOTAL,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    TOOL_CALLS_TOTAL,
)


HOST = os.environ.get("OPENCLAW_ASSISTANT_BRIDGE_HOST", "127.0.0.1")
PORT = int(os.environ.get("OPENCLAW_ASSISTANT_BRIDGE_PORT", "8200"))

# 2026-05-10 (iter 32): bridge-fired periodic /compact. When set to a
# positive integer N, the bridge tracks the per-session-key request
# count and POSTs `/compact` to the gateway before every Nth request
# (skipping the very first). The compaction uses the same gateway
# chat-completions path as a normal user message; OpenClaw resolves
# `/compact` text as the slash-command via its commands registry.
# Rationale: OpenClaw's auto-compaction (triggered by context-overflow
# precheck against agents.defaults.contextTokens) hits an
# already_compacted_recently cooldown and stops working after the
# first overflow. Spacing compactions ourselves at predictable
# intervals avoids the cooldown trap. Off by default (N=0) so other
# operators of the bridge see no behavioral change.
_COMPACT_EVERY_N = int(os.environ.get("OPENCLAW_ASSISTANT_COMPACT_EVERY_N", "0") or "0")
_COMPACT_TIMEOUT_S = float(os.environ.get("OPENCLAW_ASSISTANT_COMPACT_TIMEOUT_S", "120") or "120")
_SESSION_REQUEST_COUNTER: dict[str, int] = {}
_STATE_CONTEXT_MODES = {"off", "always", "every_n", "first_then_every_n"}
_STATE_CONTEXT_COUNTER: dict[str, int] = {}
_STATE_CONTEXT_LAST_INCLUDED: dict[str, int] = {}


def _bump_session_request_counter(session_key: str) -> int:
    """Increment + return the request counter for this session key.

    Counter is per-process; resets when the bridge restarts. That is
    fine for the compaction trigger because we only care about request
    spacing within a single bridge lifetime.
    """
    n = _SESSION_REQUEST_COUNTER.get(session_key, 0) + 1
    _SESSION_REQUEST_COUNTER[session_key] = n
    return n


def _should_fire_compact(session_request_count: int) -> bool:
    """True iff this is the Nth, 2Nth, 3Nth, ... request for the
    session AND the feature is enabled. Skips request #1 — there is
    nothing to compact yet at session start.
    """
    if _COMPACT_EVERY_N <= 0:
        return False
    return session_request_count > 1 and (session_request_count - 1) % _COMPACT_EVERY_N == 0


def _state_context_mode() -> str:
    raw = (
        os.environ.get("OPENCLAW_ASSISTANT_STATE_CONTEXT")
        or os.environ.get("MANYFORGE_STATE_CONTEXT")
        or "first_then_every_n"
    ).strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "1": "always",
        "true": "always",
        "yes": "always",
        "on": "always",
    }
    mode = aliases.get(raw, raw)
    return mode if mode in _STATE_CONTEXT_MODES else "first_then_every_n"


def _state_context_every_n() -> int:
    raw = (
        os.environ.get("OPENCLAW_ASSISTANT_STATE_CONTEXT_EVERY_N")
        or os.environ.get("MANYFORGE_STATE_CONTEXT_EVERY_N")
        or "3"
    )
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


def _state_context_control(session_key: str) -> dict[str, Any]:
    sequence = _STATE_CONTEXT_COUNTER.get(session_key, 0) + 1
    _STATE_CONTEXT_COUNTER[session_key] = sequence
    mode = _state_context_mode()
    every_n = _state_context_every_n()
    include = False
    reason = "off"
    if mode == "always":
        include = True
        reason = "always"
    elif mode == "every_n":
        include = sequence % every_n == 0
        reason = "cadence" if include else "cadence_skip"
    elif mode == "first_then_every_n":
        include = sequence == 1 or (sequence - 1) % every_n == 0
        reason = "first" if sequence == 1 else ("cadence" if include else "cadence_skip")
    last_included = _STATE_CONTEXT_LAST_INCLUDED.get(session_key)
    if include:
        _STATE_CONTEXT_LAST_INCLUDED[session_key] = sequence
        last_included = sequence
    return {
        "schemaVersion": "manyforge.state_context_control.v0",
        "mode": mode,
        "everyN": every_n,
        "sequence": sequence,
        "include": include,
        "reason": reason,
        "lastIncludedSequence": last_included,
    }


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

# Per-transport circuit breaker around the bridge → gateway dispatch.
# Opt-in via OPENCLAW_ASSISTANT_CIRCUIT_BREAKER_ENABLED. When enabled,
# after N consecutive failures (timeout / RuntimeError / non-zero exit)
# the bridge fails fast with a 503 envelope for the cooldown window
# instead of repeatedly hammering a sick gateway. Half-open probe
# allows a single recovery attempt per cooldown cycle.
_CIRCUIT_BREAKER = CircuitBreaker(
    enabled=os.environ.get("OPENCLAW_ASSISTANT_CIRCUIT_BREAKER_ENABLED", "false").lower()
    in {"1", "true", "yes", "on"},
    failure_threshold=int(
        os.environ.get("OPENCLAW_ASSISTANT_CIRCUIT_BREAKER_THRESHOLD", "5") or "5"
    ),
    cooldown_seconds=float(
        os.environ.get("OPENCLAW_ASSISTANT_CIRCUIT_BREAKER_COOLDOWN_S", "30") or "30"
    ),
)
_ACTIVE_PROCESSES: dict[str, asyncio.subprocess.Process] = {}
_ACTIVE_REQUESTS: dict[str, dict[str, Any]] = {}
_CANCELLED: set[str] = set()

# Session health quarantine for OpenClaw lock/timeout failures.
#
# OpenClaw sessions are stateful. If an agent turn times out while holding the
# session write lock, later turns with the same session id can fail immediately
# with opaque 502s. Keep the containment explicit: mark the session poisoned,
# attempt guarded cleanup, and refuse to reuse it until recovery succeeds.
_POISONED_SESSIONS_MAX = int(
    os.environ.get("OPENCLAW_ASSISTANT_POISONED_SESSIONS_MAX", "128") or "128"
)
_SESSION_LOCK_CLEANUP_ENABLED = (
    os.environ.get("OPENCLAW_ASSISTANT_LOCK_CLEANUP_ENABLED", "true").lower()
    in {"1", "true", "yes", "on"}
)
_SESSION_LOCK_CLEANUP_TIMEOUT_S = float(
    os.environ.get("OPENCLAW_ASSISTANT_LOCK_CLEANUP_TIMEOUT_S", "8") or "8"
)
_SESSION_LOCK_CLEANUP_SIGKILL = (
    os.environ.get("OPENCLAW_ASSISTANT_LOCK_CLEANUP_SIGKILL", "true").lower()
    in {"1", "true", "yes", "on"}
)
_poisoned_sessions: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_poisoned_sessions_lock = asyncio.Lock()


def _principal_for(cfg: "AdapterConfig") -> str:
    """The principal the in-sandbox MCP bridge subprocess uses on
    /api/assistant/bridge/tools envelopes. Mirrors the value set by
    the provisioner in setup-manyforge-assistant.sh
    (`openclaw-${SANDBOX}`)."""
    return f"openclaw-{cfg.sandbox}"


def _session_lock_path(config: AdapterConfig, session_id: str) -> str:
    """Best-known OpenClaw session lock path inside the sandbox.

    OpenClaw stores shell-out agent session logs under
    ``/sandbox/.openclaw/agents/<agent>/sessions``. The lock file is
    advisory state for one session JSONL. The path is used only for guarded
    cleanup/diagnostics; if OpenClaw changes it, cleanup will fail closed by
    leaving the session quarantined.
    """
    safe_session = session_id.replace("/", "_")
    return (
        f"/sandbox/.openclaw/agents/{config.agent}/sessions/"
        f"{safe_session}.jsonl.lock"
    )


_LOCK_PID_RE = re.compile(r"(?:pid\s*[=:]\s*|pid=)(\d+)", re.IGNORECASE)


def _extract_lock_holder_pid(text: str) -> int | None:
    """Extract ``pid=123`` style lock-owner hints from OpenClaw errors."""
    if not text:
        return None
    match = _LOCK_PID_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _is_session_lock_error(text: str) -> bool:
    lowered = (text or "").lower()
    return "sessionwritelocktimeouterror" in lowered or "session file locked" in lowered


def _session_poison_code(reason: str) -> str:
    if reason in {"timeout", "compact_timeout"}:
        return reason
    if reason in {"session_lock_timeout", "compact_session_lock_timeout"}:
        return "session_lock_timeout"
    return "session_lock_poisoned"


async def _session_poison_mark(
    *,
    session_id: str,
    request_id: str,
    reason: str,
    detail: str,
    lock_path: str | None = None,
    lock_holder_pid: int | None = None,
) -> dict[str, Any]:
    """Quarantine an OpenClaw session after timeout/lock corruption.

    The quarantine prevents a single stuck OpenClaw session from cascading into
    many later Composer 502s. Recovery may clear this entry later, but until
    then the bridge returns an explicit session-health error before dispatch.
    """
    if not session_id:
        return {}
    entry = {
        "sessionId": session_id,
        "requestId": request_id,
        "reason": reason,
        "detail": detail[:1500],
        "lockPath": lock_path,
        "lockHolderPid": lock_holder_pid,
        "poisonedAt": time.time(),
        "lastRecoveryAttemptAt": None,
        "lastRecovery": None,
    }
    async with _poisoned_sessions_lock:
        existing = _poisoned_sessions.get(session_id)
        if existing:
            existing.update({k: v for k, v in entry.items() if v is not None})
            entry = dict(existing)
            _poisoned_sessions.move_to_end(session_id)
        else:
            _poisoned_sessions[session_id] = dict(entry)
        while len(_poisoned_sessions) > _POISONED_SESSIONS_MAX:
            _poisoned_sessions.popitem(last=False)
    _log_event(
        "openclaw_session_poisoned",
        requestId=request_id,
        sessionId=session_id,
        reason=reason,
        lockPath=lock_path,
        lockHolderPid=lock_holder_pid,
    )
    return entry


async def _session_poison_clear(session_id: str, *, request_id: str, reason: str) -> bool:
    async with _poisoned_sessions_lock:
        removed = _poisoned_sessions.pop(session_id, None)
    if removed is not None:
        _log_event(
            "openclaw_session_poison_cleared",
            requestId=request_id,
            sessionId=session_id,
            reason=reason,
        )
        return True
    return False


def _session_poison_snapshot(session_id: str) -> dict[str, Any] | None:
    entry = _poisoned_sessions.get(session_id)
    if entry is None:
        return None
    return dict(entry)


def _reset_session_poison_for_tests() -> None:
    """Drop all quarantined-session state. Test-only entry point."""
    _poisoned_sessions.clear()


def _session_health_error_body(
    *,
    request_id: str,
    session_id: str,
    code: str,
    detail: str,
    poison: dict[str, Any] | None = None,
    recovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = error_envelope(request_id=request_id, code=code, detail=detail)
    body["sessionHealth"] = {
        "state": "poisoned",
        "sessionId": session_id,
        "poison": poison or _session_poison_snapshot(session_id),
        "recovery": recovery,
    }
    body["warnings"].append(
        f"openclaw_session_poisoned: session {session_id!r} is quarantined; "
        "retry with a fresh session or after recovery clears the lock"
    )
    return body


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
    # Sampling defaults (temperature/top_k/top_p/enable_thinking) default
    # to None and the adapter omits them from the request body — vLLM
    # owns the server-side defaults via --override-generation-config /
    # --default-chat-template-kwargs in nemoclaw-thor/serving/launch.sh.
    # The env vars below exist as per-process escape hatches: a deployment
    # can flip enable_thinking on/off without rebuilding vLLM, useful for
    # agentic-turn vs chat-turn A/B testing (see Dynamo-style per-request
    # thinking control). Template-level "keep reasoning in history"
    # behavior (Dynamo's truncate_history_thinking=false) remains owned by
    # the chat template + --default-chat-template-kwargs at vLLM startup.
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
        use_gateway=os.environ.get("OPENCLAW_ASSISTANT_USE_GATEWAY", "false").lower()
        in {"1", "true", "yes"},
        gateway_port=int(os.environ.get("OPENCLAW_ASSISTANT_GATEWAY_PORT", "18789")),
        gateway_max_tokens=int(
            os.environ.get("OPENCLAW_ASSISTANT_GATEWAY_MAX_TOKENS", "4096")
        ),
        gateway_temperature=_env_float("OPENCLAW_ASSISTANT_GATEWAY_TEMPERATURE", None),
        gateway_top_k=_env_int("OPENCLAW_ASSISTANT_GATEWAY_TOP_K", None),
        gateway_top_p=_env_float("OPENCLAW_ASSISTANT_GATEWAY_TOP_P", None),
        gateway_enable_thinking=_env_optional_bool(
            "OPENCLAW_ASSISTANT_GATEWAY_ENABLE_THINKING", None
        ),
        tool_surface=_resolve_tool_surface(),
    )


def _resolve_tool_surface() -> str:
    """Resolve ``OPENCLAW_ASSISTANT_TOOL_SURFACE`` env to a primer key.

    Accepted values: ``code`` (default), ``tools``. Unknown values
    fall back to ``code`` and emit a one-shot INFO log so the
    operator can spot the misconfig in the bridge stdout. The
    proxy's drift check is the load-bearing surface-mismatch detector
    (see scripts/proxy/vllm-proxy.py); this resolver is just the
    bridge-prompt side of the contract.

    The default is INFO rather than WARN because the well-configured
    deployment path simply leaves OPENCLAW_ASSISTANT_TOOL_SURFACE
    unset and gets ``code`` — making that produce a WARN every
    request would be alarm-fatigue. Operators who want a hard
    requirement on explicit setting can monitor the
    ``bridge_tool_surface_default`` event.
    """
    raw = (os.environ.get("OPENCLAW_ASSISTANT_TOOL_SURFACE") or "").strip().lower()
    if raw in {"code", "tools"}:
        return raw
    if raw == "":
        _log_event(
            "bridge_tool_surface_default",
            configured=None,
            defaultedTo="code",
            note="OPENCLAW_ASSISTANT_TOOL_SURFACE unset; default 'code'",
        )
        return "code"
    _log_event(
        "bridge_tool_surface_unknown",
        configured=raw,
        defaultedTo="code",
        accepted=["code", "tools"],
    )
    return "code"


# =========================================================================
# Per-conversation tool-call history (cross-turn loop detector).
# =========================================================================
# FIX 5 originally read ``payload.get("messages")`` to find prior
# assistant turns' tool_calls. That worked for the bridge's synthetic
# probe path but is dead in production: Composer sends ONE prompt per
# turn and never populates ``messages[]`` as an OpenAI-style history.
# External review finding 9 (2026-06-03) made this concrete — the
# real-traffic detector never fired.
#
# This module-level history is the production-correct path. We record
# every observed tool call AFTER dispatch, keyed by
# ``(conversationId, assistantMode)``. On each incoming request the
# detector queries this history and short-circuits with a synthetic
# loop break if the same fingerprint repeats above threshold.
#
# Bounds:
# - ``_LOOP_HISTORY_MAX_CONVERSATIONS``: total conversations cached.
#   LRU eviction (move_to_end on touch, popitem(last=False) on
#   overflow). Conservative default 64 — covers concurrent demo
#   sessions without unbounded growth.
# - ``_LOOP_HISTORY_MAX_CALLS_PER_CONV``: ring-buffer depth per
#   conversation. A deque(maxlen=...) drops oldest. Default 50
#   covers many turns while keeping memory trivial.
#
# Concurrency: protected by ``_loop_history_lock`` because FastAPI
# may serve multiple requests concurrently and a stale read between
# update + check would let a true loop slip through.
_LOOP_HISTORY_MAX_CONVERSATIONS = int(
    os.environ.get("OPENCLAW_ASSISTANT_LOOP_HISTORY_MAX_CONVERSATIONS", "64") or "64"
)
_LOOP_HISTORY_MAX_CALLS_PER_CONV = int(
    os.environ.get("OPENCLAW_ASSISTANT_LOOP_HISTORY_MAX_CALLS_PER_CONV", "50") or "50"
)

# (conversation_id, assistant_mode) → deque[fingerprint_str]
# fingerprint = "<tool_name>::<canonical_args_json>"
_loop_history: "OrderedDict[tuple[str, str], collections.deque[str]]" = OrderedDict()
_loop_history_lock = asyncio.Lock()


def _loop_conversation_key(payload: dict[str, Any], request_id: str) -> tuple[str, str]:
    """Stable key for the per-conversation history.

    ``conversationId`` is the load-bearing dimension; ``assistantMode``
    is included because two concurrent lanes (e.g. composer-assistant
    vs. scene-authoring) on the same Composer instance can share a
    conversation id but have totally different tool surfaces, and we
    must not cross-pollinate their histories.
    """
    conv_id = str(payload.get("conversationId") or request_id)
    mode = str(payload.get("assistantMode") or "default")
    return (conv_id, mode)


def _canonicalize_tool_call_args(args: Any) -> str:
    """JSON-encoded, sort-keyed, whitespace-stripped args string.

    Equivalent fingerprints regardless of key order; falls back to
    repr-equivalent for non-JSON-able payloads so we never crash on
    history recording.
    """
    try:
        if isinstance(args, str):
            args_parsed = json.loads(args)
        else:
            args_parsed = args
        return json.dumps(args_parsed, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(args if args is not None else "").strip()


def _fingerprint_tool_call(call: dict[str, Any]) -> str | None:
    """Build a ``name::args`` fingerprint from a tool-call dict.

    Accepts both the OpenAI-style ``{function: {name, arguments}}``
    nesting and the bridge's normalized ``{name, arguments}`` /
    ``{toolId, arguments}`` shapes — Composer's bridge response uses
    the latter (toolId), older direct-OpenAI history items use the
    former. Returns None if no usable name is present.
    """
    if not isinstance(call, dict):
        return None
    fn = call.get("function") if isinstance(call.get("function"), dict) else None
    name = (fn.get("name") if fn else None) or call.get("name") or call.get("toolId")
    if not name:
        return None
    raw_args = (fn.get("arguments") if fn else None)
    if raw_args is None:
        raw_args = call.get("arguments")
    if raw_args is None:
        raw_args = call.get("args")
    return f"{name}::{_canonicalize_tool_call_args(raw_args)}"


async def _loop_history_record(
    conv_key: tuple[str, str], fingerprints: list[str]
) -> None:
    """Append observed fingerprints to the per-conversation history.

    LRU-touches the conversation key, evicts cold conversations when
    over the cap. Atomic under the module lock.
    """
    if not fingerprints:
        return
    async with _loop_history_lock:
        bucket = _loop_history.get(conv_key)
        if bucket is None:
            bucket = collections.deque(maxlen=_LOOP_HISTORY_MAX_CALLS_PER_CONV)
            _loop_history[conv_key] = bucket
        else:
            _loop_history.move_to_end(conv_key)
        for fp in fingerprints:
            bucket.append(fp)
        while len(_loop_history) > _LOOP_HISTORY_MAX_CONVERSATIONS:
            _loop_history.popitem(last=False)


def _loop_history_snapshot(conv_key: tuple[str, str]) -> list[str]:
    """Read-only snapshot of a conversation's fingerprint history.

    Returned as a plain list so caller iteration cannot mutate the
    underlying deque. Safe to call without the lock — Python list
    iteration over a deque snapshot is atomic enough for our
    read-then-decide use; the lock is reserved for write paths.
    """
    bucket = _loop_history.get(conv_key)
    if bucket is None:
        return []
    return list(bucket)


def _reset_loop_history_for_tests() -> None:
    """Drop all per-conversation history. Test-only entry point."""
    _loop_history.clear()


async def _loop_history_clear(conv_key: tuple[str, str]) -> bool:
    """Drop ONE conversation's loop-history. Returns True if anything was
    dropped.

    2026-06-05 (FIX: chained-conversation cascade). The per-conversation
    history is keyed only by ``(conversationId, assistantMode)`` and persists
    across turns. A benchmark/operator flow that reuses one conversationId
    across many DISTINCT user prompts (e.g. the PnP build chain) legitimately
    calls the same tool family across turns; those fingerprints accumulate and
    trip ``bridge_loop_detected_stop`` on a later turn, after which EVERY
    subsequent turn on that conversationId short-circuits (409 -> composer 502)
    without ever invoking the model — a cascade.

    Fix: clear the conversation's history after a turn that reaches a
    SUCCESSFUL final answer (200). Genuinely-stuck turns FAIL (nonzero exit /
    loop-stop) and do NOT hit the success path, so their fingerprints persist
    and cross-turn detection of repeatedly-FAILING loops is preserved. This
    does not weaken within-turn detection (that is the gateway's job + the
    mid-turn reflection injector). Endorsed by external review as the durable
    fix vs. the ``--no-chain-session`` measurement workaround (which loses
    conversational memory).
    """
    async with _loop_history_lock:
        return _loop_history.pop(conv_key, None) is not None


async def _record_history_from_result(
    *,
    conv_key: tuple[str, str],
    result_stdout: str,
    use_gateway: bool,
    payload: dict[str, Any],
) -> int:
    """Parse stdout for tool calls and append fingerprints to history.

    Used on EVERY exit path that has captured stdout — success, nonzero
    exit, and (eventually) partial-buffer paths — so the per-conversation
    loop detector sees every observation, not just clean successes
    (external review finding 2, 2026-06-03).

    Defensively parses: if the stdout is malformed or empty, returns 0
    and emits nothing. Never raises — the caller's error path proceeds
    as before, just with the bonus of having captured what tool calls
    the model attempted before the failure.

    Note on timeout-path limitation: ``asyncio.wait_for`` cancels
    ``proc.communicate()`` on timeout, which closes the pipes before we
    can read the buffer. Genuinely-timed-out invocations therefore lose
    their partial stdout and we cannot record what the model tried.
    Switching to a streaming reader would unblock this; deferred to a
    future refactor.

    Returns the number of fingerprints appended (for tests + log
    visibility).
    """
    if not result_stdout:
        return 0
    try:
        if use_gateway:
            response_json, _ = parse_chat_completions_response(result_stdout)
            if not isinstance(response_json, dict):
                return 0
            tmp_body = normalize_chat_completions_response(
                payload=payload,
                response_json=response_json,
                stdout=result_stdout,
                stderr="",
                parse_warnings=[],
            )
        else:
            agent_json, _ = parse_openclaw_json(result_stdout)
            if agent_json is None:
                return 0
            tmp_body = normalize_agent_response(
                payload=payload,
                agent_json=agent_json,
                stdout=result_stdout,
                stderr="",
                parse_warnings=[],
            )
    except Exception:
        # Best-effort: parse failures on the error path are not fatal.
        return 0
    if not isinstance(tmp_body, dict):
        return 0
    tool_calls = tmp_body.get("toolCalls")
    if not isinstance(tool_calls, list):
        return 0
    fingerprints = [
        fp for fp in (_fingerprint_tool_call(tc) for tc in tool_calls if isinstance(tc, dict))
        if fp
    ]
    if not fingerprints:
        return 0
    await _loop_history_record(conv_key, fingerprints)
    return len(fingerprints)


app = FastAPI(
    title="ManyForge OpenClaw assistant-provider adapter",
    version="0.1.0",
)


# Opt-in Prometheus metrics endpoint. When OPENCLAW_ASSISTANT_METRICS_ENABLED
# is unset/false the /metrics route is not mounted; metric objects in
# `.metrics` still accumulate cheaply in process memory. Enable via env to
# expose a standard Prometheus scrape target on the same FastAPI port.
_METRICS_ENABLED = os.environ.get("OPENCLAW_ASSISTANT_METRICS_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
if _METRICS_ENABLED:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
                "toolSurfaceScope": meta.get("toolSurfaceScope") or "assistant_mode",
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
    # Single timer for end-to-end metrics. Set before any early-return path
    # so the duration histogram covers parse errors too. ACTIVE_REQUESTS is
    # decremented at every return statement via explicit calls (the function
    # is too large to wrap cleanly in a single try/finally).
    handler_started = time.perf_counter()
    transport = "unknown"  # tightened to gateway_http / cli_shell_out once cfg loads
    ACTIVE_REQUESTS.inc()
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        body = error_envelope(
            request_id=f"openclaw-adapter-{int(time.time() * 1000)}",
            code="invalid_json",
            detail=f"request body is not valid JSON: {exc}",
        )
        _record_outcome("invalid_json", transport, time.perf_counter() - handler_started)
        ACTIVE_REQUESTS.dec()
        return JSONResponse(status_code=400, content=body)
    if not isinstance(payload, dict):
        body = error_envelope(
            request_id=f"openclaw-adapter-{int(time.time() * 1000)}",
            code="invalid_envelope",
            detail="request body must be a JSON object",
        )
        _record_outcome("invalid_envelope", transport, time.perf_counter() - handler_started)
        ACTIVE_REQUESTS.dec()
        return JSONResponse(status_code=400, content=body)

    request_id = request_id_from_payload(payload)
    if request_id in _CANCELLED:
        _CANCELLED.discard(request_id)
        body = error_envelope(
            request_id=request_id,
            code="cancelled",
            detail=f"request {request_id} was cancelled before dispatch",
        )
        _record_outcome("cancelled", transport, time.perf_counter() - handler_started)
        ACTIVE_REQUESTS.dec()
        return JSONResponse(status_code=200, content=body)

    cfg = _config_from_env()
    transport = "gateway_http" if cfg.use_gateway else "cli_shell_out"
    total_started = time.perf_counter()
    prompt_started = time.perf_counter()
    # Compute the per-conversation key once; used both by the loop
    # detector below AND by the post-dispatch history recorder near
    # the end of the handler. Cheap (a couple of dict lookups), and
    # hoisting it out of the conditional avoids a NameError when the
    # thresholds are zeroed for diagnostic runs.
    loop_conv_key = _loop_conversation_key(payload, request_id)
    loop_reflection_injected = False
    loop_guard_info: dict[str, Any] | None = None
    # 2026-05-31 (round 8 — fail-fast retry-loop detector): we observed
    # cosmos-reason2-8b spending 25-28 turns retrying the same tool with
    # the same args after the same validator error. OpenClaw has a
    # per-turn budget (15 attempts) but the composer/smoke harness keeps
    # making new bridge requests, each starting OpenClaw's budget fresh.
    # The detector now has two explicit phases:
    #   1. one visible loop-guard reflection appended to the next model prompt;
    #   2. a failure-shaped loop_detected envelope if repetition continues.
    # It never returns a synthetic assistant success.
    #
    # 2026-06-03 (external review finding 9): the prior implementation
    # read ``payload.get("messages")`` for history, which is dead in
    # production — Composer sends one prompt per turn with no OpenAI-
    # style messages[]. The production-correct path is the per-
    # conversation history maintained by this bridge module (recorded
    # after every dispatch). We consult BOTH sources here so the
    # synthetic probe path still works in tests, and the production
    # path actually fires across real Composer turns.
    LOOP_TOOL_THRESHOLD = int(os.environ.get("OPENCLAW_ASSISTANT_LOOP_TOOL_THRESHOLD", "5") or "5")
    # FIX 5: also detect same-tool-SAME-ARGS repetition (tighter than
    # plain same-name). When the model emits identical args twice in a
    # row across composer turns, we know the model isn't reading the
    # error envelope (FIX 4) and just retrying. Fire EARLIER than the
    # name-only detector since the signal is stronger.
    LOOP_ARGS_THRESHOLD = int(os.environ.get("OPENCLAW_ASSISTANT_LOOP_ARGS_THRESHOLD", "2") or "2")
    if LOOP_TOOL_THRESHOLD > 0 or LOOP_ARGS_THRESHOLD > 0:
        tool_call_names: list[str] = []
        tool_call_fingerprints: list[str] = []
        # PRODUCTION PATH: per-conversation history (loop_conv_key
        # is computed once at the top of this handler — see above).
        bridge_history = _loop_history_snapshot(loop_conv_key)
        for fp in bridge_history:
            # Fingerprints are formatted "name::args_json".
            # Defensively split — accept any well-formed entry.
            head, sep, _tail = fp.partition("::")
            if head and sep:
                tool_call_names.append(head)
                tool_call_fingerprints.append(fp)
        # FALLBACK PATH (probes, legacy direct-OpenAI callers): the
        # OpenAI-format messages[] in the payload itself. Production
        # Composer never populates this, so in production this loop is
        # a no-op; tests that build messages[] manually continue to
        # work.
        msgs_in = payload.get("messages") or []
        if isinstance(msgs_in, list):
            for m in msgs_in:
                if not isinstance(m, dict): continue
                if m.get("role") != "assistant": continue
                for tc in (m.get("tool_calls") or []):
                    if not isinstance(tc, dict): continue
                    fp = _fingerprint_tool_call(tc)
                    if not fp: continue
                    head, _sep, _tail = fp.partition("::")
                    if head:
                        tool_call_names.append(head)
                        tool_call_fingerprints.append(fp)
        if tool_call_names:
            from collections import Counter
            # Same-tool-same-args check first (tighter).
            fp_top_fp, fp_top_count = Counter(tool_call_fingerprints).most_common(1)[0]
            # Same-tool-name check (existing behavior).
            top_name, top_count = Counter(tool_call_names).most_common(1)[0]
            trigger = None
            phase = None
            if LOOP_ARGS_THRESHOLD > 0 and fp_top_count >= LOOP_ARGS_THRESHOLD + 1:
                # +1 because the same call repeats are counted relative
                # to the first appearance, not from zero. e.g.
                # threshold=2 means "fire after we've seen 3 identical
                # calls (1 original + 2 retries)".
                trigger = "same_args"
                phase = "fail"
                repeated_tool = fp_top_fp.split("::", 1)[0]
                repeated_count = fp_top_count
            elif LOOP_ARGS_THRESHOLD > 0 and fp_top_count == max(2, LOOP_ARGS_THRESHOLD):
                trigger = "same_args"
                phase = "reflect"
                repeated_tool = fp_top_fp.split("::", 1)[0]
                repeated_count = fp_top_count
            elif LOOP_TOOL_THRESHOLD > 0 and top_count >= LOOP_TOOL_THRESHOLD:
                trigger = "same_name"
                phase = "fail"
                repeated_tool = top_name
                repeated_count = top_count
            elif LOOP_TOOL_THRESHOLD > 0 and top_count == max(2, LOOP_TOOL_THRESHOLD - 1):
                trigger = "same_name"
                phase = "reflect"
                repeated_tool = top_name
                repeated_count = top_count
            if trigger:
                threshold = LOOP_TOOL_THRESHOLD if trigger == "same_name" else LOOP_ARGS_THRESHOLD
                loop_guard_info = {
                    "trigger": trigger,
                    "phase": phase,
                    "repeatedTool": repeated_tool,
                    "repeatCount": repeated_count,
                    "threshold": threshold,
                }
                if phase == "fail":
                    detail = (
                        f"loop detected: tool '{repeated_tool}' repeated "
                        f"{repeated_count} times ({trigger}) without recovery"
                    )
                    _log_event(
                        "bridge_loop_detected_stop",
                        requestId=request_id,
                        repeatedTool=repeated_tool,
                        repeatedCount=repeated_count,
                        threshold=threshold,
                        trigger=trigger,
                        conversationKey=list(loop_conv_key),
                        historyDepth=len(bridge_history),
                    )
                    body = error_envelope(
                        request_id=request_id,
                        code="loop_detected",
                        detail=detail,
                    )
                    body["error"].update(loop_guard_info)
                    body["warnings"].append(
                        f"loop_detected: tool={repeated_tool!r} repeated {repeated_count}x ({trigger})"
                    )
                    body["loopGuard"] = loop_guard_info
                    _record_outcome("loop_detected", transport, time.perf_counter() - handler_started)
                    ACTIVE_REQUESTS.dec()
                    return JSONResponse(status_code=409, content=body)
                loop_reflection_injected = True
                _log_event(
                    "bridge_loop_reflection_injected",
                    requestId=request_id,
                    repeatedTool=repeated_tool,
                    repeatedCount=repeated_count,
                    threshold=threshold,
                    trigger=trigger,
                    conversationKey=list(loop_conv_key),
                    historyDepth=len(bridge_history),
                )
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
    # Do not derive a request-scoped MCP allowlist from the user's prompt.
    # The lane/mode catalog is the enforcement boundary; the model must decide
    # whether to inspect, ask for clarification, or call one of the visible
    # tools. This keeps benchmarks from inheriting bridge-side intent guesses.
    prompt_payload = payload
    if loop_reflection_injected and loop_guard_info is not None:
        prompt_payload = dict(payload)
        original_message = prompt_payload.get("message")
        if not isinstance(original_message, str):
            original_message = json.dumps(original_message, sort_keys=True)
        repeated_tool = str(loop_guard_info.get("repeatedTool") or "")
        repeated_count = int(loop_guard_info.get("repeatCount") or 0)
        trigger = str(loop_guard_info.get("trigger") or "repeat")
        prompt_payload["message"] = (
            original_message.rstrip()
            + "\n\n## loop_guard_reflection\n"
            + f"You have already repeated `{repeated_tool}` {repeated_count} "
            + f"times in this conversation ({trigger}). Do not retry the same "
            + "tool call. Read the last validation error, change the arguments "
            + "or tool choice, or ask one concise clarification question if the "
            + "request is missing required information."
        )
    session_key = derive_gateway_session_key(payload)
    state_context_control = _state_context_control(session_key)
    prompt = build_agent_prompt(
        prompt_payload,
        tool_surface=cfg.tool_surface,
        state_context_control=state_context_control,
    )
    prompt_ms = (time.perf_counter() - prompt_started) * 1000.0
    REQUEST_DURATION.labels(stage="prompt_build", transport=transport).observe(prompt_ms / 1000.0)
    timeout_s = _request_timeout(payload, cfg.timeout_s)
    session_id = str(payload.get("conversationId") or request_id)
    poison = _session_poison_snapshot(session_id)
    if poison is not None:
        recovery = await _recover_poisoned_session(
            cfg,
            session_id=session_id,
            request_id=request_id,
            reason=str(poison.get("reason") or "previous_failure"),
            detail=str(poison.get("detail") or ""),
        )
        if recovery.get("recovered") is not True:
            _log_event(
                "openclaw_session_poisoned_rejected",
                requestId=request_id,
                sessionId=session_id,
                recoveryStatus=recovery.get("status"),
                reason=poison.get("reason"),
            )
            body = _session_health_error_body(
                request_id=request_id,
                session_id=session_id,
                code=_session_poison_code(str(poison.get("reason") or "")),
                detail=(
                    "OpenClaw session is quarantined after a previous "
                    f"{poison.get('reason') or 'failure'} and recovery did not "
                    "clear it. Start a fresh conversation/session or restart "
                    "the OpenClaw sandbox gateway before retrying."
                ),
                poison=poison,
                recovery=recovery,
            )
            _record_outcome("session_lock_poisoned", transport, time.perf_counter() - handler_started)
            ACTIVE_REQUESTS.dec()
            return JSONResponse(status_code=423, content=body)
    _ACTIVE_REQUESTS[request_id] = {
        "startedPerf": time.perf_counter(),
        "stage": "dispatching_openclaw_gateway" if cfg.use_gateway else "dispatching_openclaw",
        "toolSurfaceScope": "assistant_mode",
        "loopReflectionInjected": loop_reflection_injected,
        "stateContext": state_context_control,
        "promptChars": len(prompt),
        "sessionId": session_id,
        "transport": "gateway_http" if cfg.use_gateway else "cli_shell_out",
    }
    _log_event(
        "openclaw_request_started",
        requestId=request_id,
        timeoutS=timeout_s,
        toolSurfaceScope="assistant_mode",
        loopReflectionInjected=loop_reflection_injected,
        stateContext=state_context_control,
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

    # Circuit-breaker gate (opt-in via OPENCLAW_ASSISTANT_CIRCUIT_BREAKER_*).
    # When disabled this is a fast no-op. When enabled and OPEN, the bridge
    # short-circuits with a 503 envelope rather than dispatching to a sick
    # gateway. The compact firing (above) is intentionally NOT gated by the
    # breaker: it is best-effort already, and gating it would create a
    # confusing partial-state outcome where the user request runs but the
    # session is not compacted as expected.
    cb_allowed, cb_reason = await _CIRCUIT_BREAKER.before_dispatch(transport)
    if not cb_allowed:
        _log_event(
            "openclaw_circuit_breaker_open",
            requestId=request_id,
            transport=transport,
            reason=cb_reason or "",
        )
        body = error_envelope(
            request_id=request_id,
            code="circuit_breaker_open",
            detail=(
                "bridge circuit breaker is open after consecutive upstream "
                f"failures ({cb_reason or 'open'}); try again after the cooldown"
            ),
        )
        _record_outcome("circuit_breaker_open", transport, time.perf_counter() - handler_started)
        ACTIVE_REQUESTS.dec()
        return JSONResponse(status_code=503, content=body)

    try:
        # 2026-05-10 (iter 32): periodic bridge-fired compaction. When
        # OPENCLAW_ASSISTANT_COMPACT_EVERY_N is set, fire `/compact` to
        # the gateway before every Nth request on this session-key.
        # Sequential — wait for the compact to return before forwarding
        # the actual user prompt so the model sees the post-compaction
        # session state.
        #
        # REVISED 2026-06-03: compaction MUST work in cli_shell_out
        # mode too, not just gateway_http. The original code gated
        # compaction inside the `cfg.use_gateway` branch, which silently
        # disabled the iter-32 production recipe when the route fix
        # forced cli_shell_out. The trigger logic + session counter is
        # now lifted out; only the actual /compact dispatch differs by
        # transport (gateway: chat-completion; cli: openclaw agent).
        session_count = _bump_session_request_counter(session_key)
        if _should_fire_compact(session_count):
            COMPACT_FIRES_TOTAL.labels(outcome="started").inc()
            compact_started_perf = time.perf_counter()
            _log_event(
                "openclaw_compact_fire_started",
                requestId=request_id,
                sessionKey=session_key,
                sessionCount=session_count,
                every_n=_COMPACT_EVERY_N,
                timeoutS=_COMPACT_TIMEOUT_S,
                transport=transport,
            )
            if cfg.use_gateway:
                compact_command = build_gateway_chat_completions_command(
                    config=cfg,
                    payload=payload,
                    timeout_s=_COMPACT_TIMEOUT_S,
                    message="/compact",
                )
            else:
                # cli_shell_out path: send /compact as the user message
                # to `openclaw agent` and wait. OpenClaw's command
                # registry resolves `/compact` regardless of whether
                # it's received as a slash-command on the chat-completion
                # path or as a CLI message argument.
                #
                # CRITICAL — pass session_id so /compact targets the SAME
                # OpenClaw session the real chat request uses below
                # (see the build_openclaw_command call further down in
                # this function — same session_id). Without this,
                # /compact runs as a FRESH session and compacts nothing,
                # rendering FIX 1 functionally moot in CLI mode.
                # Reference: external review finding 8 (2026-06-03).
                compact_command = build_openclaw_command(
                    config=cfg,
                    message="/compact",
                    timeout_s=_COMPACT_TIMEOUT_S,
                    session_id=session_id,
                )
            try:
                compact_result = await _run_agent(
                    request_id=f"{request_id}-compact",
                    command=compact_command,
                    timeout_s=_COMPACT_TIMEOUT_S,
                )
                if compact_result.returncode != 0:
                    compact_detail = (
                        f"OpenClaw compact exited with code {compact_result.returncode}: "
                        f"{(compact_result.stderr or compact_result.stdout)[:1000]}"
                    )
                    raise RuntimeError(compact_detail)
                COMPACT_FIRES_TOTAL.labels(outcome="succeeded").inc()
                REQUEST_DURATION.labels(stage="compact", transport=transport).observe(
                    time.perf_counter() - compact_started_perf
                )
                _log_event(
                    "openclaw_compact_fire_succeeded",
                    requestId=request_id,
                    sessionKey=session_key,
                    sessionCount=session_count,
                    transport=transport,
                )
            except Exception as compact_exc:  # noqa: BLE001
                # Most compaction failures remain best-effort. Timeout and
                # session-lock shaped failures are different: they can leave
                # OpenClaw holding the session write lock, and continuing with
                # the real request would poison the rest of a chained
                # conversation. Quarantine those sessions immediately.
                COMPACT_FIRES_TOTAL.labels(outcome="failed").inc()
                compact_detail = f"{type(compact_exc).__name__}: {compact_exc}"
                _log_event(
                    "openclaw_compact_fire_failed",
                    requestId=request_id,
                    sessionKey=session_key,
                    sessionCount=session_count,
                    error=compact_detail,
                    transport=transport,
                )
                compact_reason = None
                if isinstance(compact_exc, asyncio.TimeoutError):
                    compact_reason = "compact_timeout"
                elif _is_session_lock_error(compact_detail):
                    compact_reason = "compact_session_lock_timeout"
                if compact_reason is not None:
                    poison = await _session_poison_mark(
                        session_id=session_id,
                        request_id=request_id,
                        reason=compact_reason,
                        detail=compact_detail,
                        lock_path=_session_lock_path(cfg, session_id),
                        lock_holder_pid=_extract_lock_holder_pid(compact_detail),
                    )
                    recovery = await _recover_poisoned_session(
                        cfg,
                        session_id=session_id,
                        request_id=request_id,
                        reason=compact_reason,
                        detail=compact_detail,
                    )
                    await _CIRCUIT_BREAKER.record_failure(transport)
                    body = _session_health_error_body(
                        request_id=request_id,
                        session_id=session_id,
                        code=_session_poison_code(compact_reason),
                        detail=(
                            "OpenClaw compaction failed in a way that can leave "
                            "the session lock held; the session was quarantined "
                            "before dispatching the user request."
                        ),
                        poison=poison,
                        recovery=recovery,
                    )
                    _record_outcome(
                        _session_poison_code(compact_reason),
                        transport,
                        time.perf_counter() - handler_started,
                    )
                    ACTIVE_REQUESTS.dec()
                    return JSONResponse(status_code=423, content=body)

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
            )
        result = await _run_agent(
            request_id=request_id,
            command=command,
            timeout_s=timeout_s,
        )
        REQUEST_DURATION.labels(stage="agent_run", transport=transport).observe(
            result.duration_ms / 1000.0
        )
    except asyncio.TimeoutError:
        await _kill_sandbox_agent(cfg, session_id=session_id)
        timeout_detail = f"OpenClaw agent exceeded timeout {timeout_s:.1f}s"
        poison = await _session_poison_mark(
            session_id=session_id,
            request_id=request_id,
            reason="timeout",
            detail=timeout_detail,
            lock_path=_session_lock_path(cfg, session_id),
        )
        recovery = await _recover_poisoned_session(
            cfg,
            session_id=session_id,
            request_id=request_id,
            reason="timeout",
            detail=timeout_detail,
        )
        await _CIRCUIT_BREAKER.record_failure(transport)
        _log_event(
            "openclaw_request_timeout",
            requestId=request_id,
            timeoutS=timeout_s,
            toolSurfaceScope="assistant_mode",
            sessionId=session_id,
            recoveryStatus=recovery.get("status"),
            recovered=bool(recovery.get("recovered")),
        )
        body = _session_health_error_body(
            request_id=request_id,
            session_id=session_id,
            code="timeout",
            detail=timeout_detail,
            poison=poison,
            recovery=recovery,
        )
        _record_outcome("timeout", transport, time.perf_counter() - handler_started)
        ACTIVE_REQUESTS.dec()
        return JSONResponse(status_code=504, content=body)
    except RuntimeError as exc:
        runtime_detail = str(exc)
        if _is_session_lock_error(runtime_detail):
            poison = await _session_poison_mark(
                session_id=session_id,
                request_id=request_id,
                reason="session_lock_timeout",
                detail=runtime_detail,
                lock_path=_session_lock_path(cfg, session_id),
                lock_holder_pid=_extract_lock_holder_pid(runtime_detail),
            )
            recovery = await _recover_poisoned_session(
                cfg,
                session_id=session_id,
                request_id=request_id,
                reason="session_lock_timeout",
                detail=runtime_detail,
            )
            await _CIRCUIT_BREAKER.record_failure(transport)
            _log_event(
                "openclaw_request_session_lock_timeout",
                requestId=request_id,
                sessionId=session_id,
                recoveryStatus=recovery.get("status"),
                recovered=bool(recovery.get("recovered")),
            )
            body = _session_health_error_body(
                request_id=request_id,
                session_id=session_id,
                code="session_lock_timeout",
                detail=runtime_detail,
                poison=poison,
                recovery=recovery,
            )
            _record_outcome(
                "session_lock_timeout",
                transport,
                time.perf_counter() - handler_started,
            )
            ACTIVE_REQUESTS.dec()
            return JSONResponse(status_code=423, content=body)
        await _CIRCUIT_BREAKER.record_failure(transport)
        _log_event("openclaw_request_error", requestId=request_id, detail=runtime_detail)
        body = error_envelope(
            request_id=request_id,
            code="upstream_call_error",
            detail=runtime_detail,
        )
        _record_outcome("upstream_call_error", transport, time.perf_counter() - handler_started)
        ACTIVE_REQUESTS.dec()
        return JSONResponse(status_code=502, content=body)

    if request_id in _CANCELLED:
        _CANCELLED.discard(request_id)
        body = error_envelope(
            request_id=request_id,
            code="cancelled",
            detail=f"request {request_id} was cancelled",
        )
        _record_outcome("cancelled", transport, time.perf_counter() - handler_started)
        ACTIVE_REQUESTS.dec()
        return JSONResponse(status_code=200, content=body)

    if result.returncode != 0:
        await _CIRCUIT_BREAKER.record_failure(transport)
        detail = (
            f"OpenClaw agent exited with code {result.returncode}: "
            f"{(result.stderr or result.stdout)[:1000]}"
        )
        # FIX (external review finding 2, 2026-06-03): record any tool
        # calls the model produced before the nonzero exit into the
        # per-conversation loop history. The model may have looped on a
        # bad tool 3 times and THEN crashed — we want the next request to
        # see those 3 attempts and fire same_args. Best-effort; parse
        # failures don't block the error envelope return.
        try:
            recorded = await _record_history_from_result(
                conv_key=loop_conv_key,
                result_stdout=result.stdout or "",
                use_gateway=cfg.use_gateway,
                payload=payload,
            )
            if recorded:
                _log_event(
                    "openclaw_error_path_history_recorded",
                    requestId=request_id,
                    recordedFingerprints=recorded,
                )
        except Exception:
            pass
        # Log the actual error so we can diagnose 502s without re-running.
        _log_event(
            "openclaw_request_exit_nonzero",
            requestId=request_id,
            returncode=result.returncode,
            stderr_excerpt=(result.stderr or "")[:1500],
            stdout_excerpt=(result.stdout or "")[:1500],
        )
        if _is_session_lock_error(detail):
            poison = await _session_poison_mark(
                session_id=session_id,
                request_id=request_id,
                reason="session_lock_timeout",
                detail=detail,
                lock_path=_session_lock_path(cfg, session_id),
                lock_holder_pid=_extract_lock_holder_pid(detail),
            )
            recovery = await _recover_poisoned_session(
                cfg,
                session_id=session_id,
                request_id=request_id,
                reason="session_lock_timeout",
                detail=detail,
            )
            body = _session_health_error_body(
                request_id=request_id,
                session_id=session_id,
                code="session_lock_timeout",
                detail=detail,
                poison=poison,
                recovery=recovery,
            )
            _record_outcome("session_lock_timeout", transport, time.perf_counter() - handler_started)
            ACTIVE_REQUESTS.dec()
            return JSONResponse(status_code=423, content=body)
        body = error_envelope(
            request_id=request_id,
            code="upstream_call_error",
            detail=detail,
        )
        _record_outcome("upstream_call_error", transport, time.perf_counter() - handler_started)
        ACTIVE_REQUESTS.dec()
        return JSONResponse(status_code=502, content=body)

    # Gateway round-trip succeeded with a clean exit code. Even if downstream
    # parse / normalize surfaces a content problem, the transport itself is
    # healthy — record success on the breaker so a partial-content blip does
    # not count against the failure threshold.
    await _CIRCUIT_BREAKER.record_success(transport)

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
    REQUEST_DURATION.labels(stage="parse", transport=transport).observe(
        parse_normalize_ms / 1000.0
    )
    total_ms = (time.perf_counter() - total_started) * 1000.0
    # Tool calls in the normalized envelope — `body["toolCalls"]` is set by
    # the adapter's normalize_* functions. Count emitted calls; if there
    # were tool-call warnings (unknown tool names, schema validation), count
    # those separately for visibility into model/schema drift.
    tool_calls = body.get("toolCalls") if isinstance(body, dict) else None
    if isinstance(tool_calls, list):
        TOOL_CALLS_TOTAL.labels(outcome="emitted").inc(len(tool_calls))
        # Record every observed tool call into the per-conversation
        # history so the loop-break detector at the head of the
        # handler can see what THIS conversation has called before.
        # This is the production path for FIX 5 — replaces the dead
        # ``payload.messages[]`` read with a live observation stream
        # (external review finding 9, 2026-06-03). Same recording is
        # also invoked from the nonzero-exit error path above
        # (external review finding 2) so a loop that crashes the
        # subprocess still populates history for the next request.
        fingerprints = [
            fp for fp in (_fingerprint_tool_call(tc) for tc in tool_calls if isinstance(tc, dict))
            if fp
        ]
        if fingerprints:
            await _loop_history_record(loop_conv_key, fingerprints)
    tool_warnings = body.get("toolCallWarnings") if isinstance(body, dict) else None
    if isinstance(tool_warnings, list) and tool_warnings:
        TOOL_CALLS_TOTAL.labels(outcome="warning").inc(len(tool_warnings))
    if loop_reflection_injected and loop_guard_info is not None:
        warnings = body.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
            body["warnings"] = warnings
        warnings.append("loop_guard_reflection_injected")
        body["assisted"] = True
        body["loopReflectionInjected"] = True
        body["loopGuard"] = loop_guard_info
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
        "toolSurfaceScope": "assistant_mode",
        "loopReflectionInjected": loop_reflection_injected,
        "stateContext": state_context_control,
        "sandbox": cfg.sandbox,
        "agent": cfg.agent,
    }
    _log_event(
        "openclaw_request_complete",
        requestId=request_id,
        totalAdapterMs=round(total_ms, 1),
        agentRunMs=round(result.duration_ms, 1),
        toolSurfaceScope="assistant_mode",
        loopReflectionInjected=loop_reflection_injected,
        stateContext=state_context_control,
        promptChars=len(prompt),
        toolSurface=cfg.tool_surface,
    )
    # FIX (2026-06-05): this turn reached a successful final answer, so it is
    # not a stuck loop. Clear the conversation's loop-history so legitimately-
    # repeated tool use across DISTINCT later user prompts on the same
    # conversationId does not accumulate into a false bridge_loop_detected_stop
    # cascade. Failing turns never reach here, so real stuck-loop detection
    # (across repeatedly-FAILING turns) is preserved.
    if await _loop_history_clear(loop_conv_key):
        _log_event(
            "bridge_loop_history_cleared_on_success",
            requestId=request_id,
            conversationKey=list(loop_conv_key),
        )
    _record_outcome("success", transport, time.perf_counter() - handler_started)
    ACTIVE_REQUESTS.dec()
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


def _record_outcome(status: str, transport: str, total_seconds: float) -> None:
    """Single point that increments the outcome counter + total-duration histogram.

    Status and transport are bounded enums; never pass requestId/sessionId
    or other high-cardinality fields here.
    """
    REQUESTS_TOTAL.labels(status=status, transport=transport).inc()
    REQUEST_DURATION.labels(stage="total", transport=transport).observe(total_seconds)


async def _kill_sandbox_agent(config: AdapterConfig, *, session_id: str) -> None:
    if not session_id:
        return
    remote = "pkill -f -- " + shlex.quote(session_id) + " || true"
    try:
        await _run_sandbox_exec(config, remote, timeout_s=5.0)
    except Exception as exc:  # noqa: BLE001
        _log_event(
            "openclaw_sandbox_kill_failed",
            sessionId=session_id,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _sandbox_exec_command(config: AdapterConfig, remote: str) -> list[str]:
    return [
        "nemoclaw",
        config.sandbox,
        "exec",
        "--no-tty",
        "--",
        "bash",
        "-lc",
        remote,
    ]


async def _run_sandbox_exec(
    config: AdapterConfig,
    remote: str,
    *,
    timeout_s: float,
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *_sandbox_exec_command(config, remote),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    return (
        proc.returncode if proc.returncode is not None else -1,
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )


def _session_lock_cleanup_script(
    *,
    agent: str,
    session_id: str,
    pid_hint: int | None,
    cleanup_enabled: bool,
    sigkill_enabled: bool,
    wait_s: float,
) -> str:
    """Return a single-line sandbox command for guarded lock cleanup."""
    script = f"""
import fcntl, json, os, re, signal, time
agent = {agent!r}
session_id = {session_id!r}
pid_hint = {pid_hint!r}
cleanup_enabled = {cleanup_enabled!r}
sigkill_enabled = {sigkill_enabled!r}
wait_s = {wait_s!r}
safe_session = session_id.replace("/", "_")
lock_path = f"/sandbox/.openclaw/agents/{{agent}}/sessions/{{safe_session}}.jsonl.lock"
result = {{
    "lockPath": lock_path,
    "lockExists": os.path.exists(lock_path),
    "cleanupEnabled": cleanup_enabled,
    "sigkillEnabled": sigkill_enabled,
    "status": "unknown",
}}
def read_text(path, limit=1024):
    try:
        with open(path, "rb") as fh:
            return fh.read(limit).decode("utf-8", errors="replace")
    except Exception:
        return ""
def remove_if_unlocked(path):
    try:
        with open(path, "a+") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False, "lock_busy"
            try:
                os.remove(path)
                return True, "unlocked_lock_removed"
            except FileNotFoundError:
                return True, "unlocked_lock_already_gone"
            except Exception as exc:
                return False, f"unlocked_lock_remove_failed: {{type(exc).__name__}}: {{exc}}"
    except Exception as exc:
        return False, f"lock_probe_failed: {{type(exc).__name__}}: {{exc}}"
lock_text = read_text(lock_path)
result["lockTextExcerpt"] = lock_text[:300]
pid = pid_hint
if not pid and lock_text:
    m = re.search(r"(?:pid\\s*[=:]\\s*|pid=)(\\d+)", lock_text, re.I)
    if m:
        try:
            pid = int(m.group(1))
        except Exception:
            pid = None
result["lockHolderPid"] = pid
if not os.path.exists(lock_path):
    result["status"] = "no_lock"
    result["recovered"] = True
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0)
removed_unlocked, unlocked_status = remove_if_unlocked(lock_path)
result["lockProbeStatus"] = unlocked_status
if removed_unlocked:
    result["status"] = unlocked_status
    result["recovered"] = True
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0)
if not pid:
    result["status"] = "lock_present_no_pid"
    result["recovered"] = False
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0)
proc_dir = f"/proc/{{pid}}"
alive = os.path.exists(proc_dir)
result["pidAlive"] = alive
if not alive:
    try:
        os.remove(lock_path)
        result["status"] = "stale_lock_removed"
        result["recovered"] = True
    except Exception as exc:
        result["status"] = "stale_lock_remove_failed"
        result["recovered"] = False
        result["error"] = f"{{type(exc).__name__}}: {{exc}}"
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0)
comm = read_text(f"{{proc_dir}}/comm", 200).strip()
cmdline = read_text(f"{{proc_dir}}/cmdline", 4096).replace("\\x00", " ").strip()
result["processComm"] = comm
result["processCmdlineExcerpt"] = cmdline[:500]
owner_ok = "openclaw" in comm.lower() or "openclaw" in cmdline.lower()
result["ownerVerified"] = owner_ok
if not owner_ok:
    result["status"] = "refused_owner_mismatch"
    result["recovered"] = False
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0)
if not cleanup_enabled:
    result["status"] = "owner_alive_cleanup_disabled"
    result["recovered"] = False
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0)
try:
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + max(0.1, wait_s)
    while time.monotonic() < deadline and os.path.exists(proc_dir):
        time.sleep(0.1)
    if os.path.exists(proc_dir) and sigkill_enabled:
        os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and os.path.exists(proc_dir):
            time.sleep(0.1)
    if os.path.exists(proc_dir):
        result["status"] = "owner_kill_failed"
        result["recovered"] = False
    else:
        try:
            os.remove(lock_path)
            result["status"] = "owner_killed_lock_removed"
            result["recovered"] = True
        except FileNotFoundError:
            result["status"] = "owner_killed_lock_already_gone"
            result["recovered"] = True
        except Exception as exc:
            result["status"] = "owner_killed_lock_remove_failed"
            result["recovered"] = False
            result["error"] = f"{{type(exc).__name__}}: {{exc}}"
except ProcessLookupError:
    try:
        os.remove(lock_path)
        result["status"] = "owner_disappeared_lock_removed"
        result["recovered"] = True
    except Exception as exc:
        result["status"] = "owner_disappeared_lock_remove_failed"
        result["recovered"] = False
        result["error"] = f"{{type(exc).__name__}}: {{exc}}"
except Exception as exc:
    result["status"] = "cleanup_exception"
    result["recovered"] = False
    result["error"] = f"{{type(exc).__name__}}: {{exc}}"
print(json.dumps(result, sort_keys=True))
"""
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    python_code = f"import base64; exec(base64.b64decode({encoded!r}).decode())"
    return "python3 -c " + shlex.quote(python_code)


async def _recover_poisoned_session(
    config: AdapterConfig,
    *,
    session_id: str,
    request_id: str,
    reason: str,
    detail: str,
) -> dict[str, Any]:
    """Attempt guarded recovery for a quarantined OpenClaw session."""
    pid_hint = _extract_lock_holder_pid(detail)
    lock_path = _session_lock_path(config, session_id)
    recovery: dict[str, Any] = {
        "reason": reason,
        "lockPath": lock_path,
        "pidHint": pid_hint,
        "recovered": False,
    }
    _log_event(
        "openclaw_session_recovery_started",
        requestId=request_id,
        sessionId=session_id,
        reason=reason,
        lockPath=lock_path,
        pidHint=pid_hint,
        cleanupEnabled=_SESSION_LOCK_CLEANUP_ENABLED,
    )
    try:
        returncode, stdout, stderr = await _run_sandbox_exec(
            config,
            _session_lock_cleanup_script(
                agent=config.agent,
                session_id=session_id,
                pid_hint=pid_hint,
                cleanup_enabled=_SESSION_LOCK_CLEANUP_ENABLED,
                sigkill_enabled=_SESSION_LOCK_CLEANUP_SIGKILL,
                wait_s=_SESSION_LOCK_CLEANUP_TIMEOUT_S,
            ),
            timeout_s=max(5.0, _SESSION_LOCK_CLEANUP_TIMEOUT_S + 5.0),
        )
        recovery["returncode"] = returncode
        recovery["stderrExcerpt"] = stderr[:500]
        try:
            parsed = json.loads(stdout.strip().splitlines()[-1])
            if isinstance(parsed, dict):
                recovery.update(parsed)
        except Exception:
            recovery["stdoutExcerpt"] = stdout[:1000]
        if recovery.get("recovered") is True:
            await _session_poison_clear(
                session_id,
                request_id=request_id,
                reason=str(recovery.get("status") or "recovered"),
            )
        else:
            async with _poisoned_sessions_lock:
                entry = _poisoned_sessions.get(session_id)
                if entry is not None:
                    entry["lastRecoveryAttemptAt"] = time.time()
                    entry["lastRecovery"] = dict(recovery)
                    _poisoned_sessions.move_to_end(session_id)
        _log_event(
            "openclaw_session_recovery_finished",
            requestId=request_id,
            sessionId=session_id,
            recovered=bool(recovery.get("recovered")),
            status=recovery.get("status"),
            lockHolderPid=recovery.get("lockHolderPid"),
            ownerVerified=recovery.get("ownerVerified"),
        )
        return recovery
    except Exception as exc:  # noqa: BLE001
        recovery.update(
            {
                "status": "recovery_command_failed",
                "recovered": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        async with _poisoned_sessions_lock:
            entry = _poisoned_sessions.get(session_id)
            if entry is not None:
                entry["lastRecoveryAttemptAt"] = time.time()
                entry["lastRecovery"] = dict(recovery)
        _log_event(
            "openclaw_session_recovery_failed",
            requestId=request_id,
            sessionId=session_id,
            detail=recovery["error"],
        )
        return recovery


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
