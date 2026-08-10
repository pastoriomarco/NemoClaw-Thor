#!/usr/bin/env python3
"""HTTP reverse proxy that logs and (optionally) mutates per-turn LLM
agent-loop traffic between any chat-completions client and vLLM.

Two roles in one process:

1. **Logger (always on).** Every chat-completion request and response is
   appended to a JSONL file with full request/response bodies, mutation
   diffs, and per-call latency.

2. **Mutator (opt-in via env vars).** When configured, the proxy rewrites
   outbound request bodies before forwarding to vLLM. Used in the iter-20
   production recipe to inject `max_tokens=2048` and
   `chat_template_kwargs.thinking_token_budget=512` into traffic that
   OpenClaw → vLLM otherwise sends without bounds.

Typical placement (iter-20 production setup): listen on `:8000`, forward
to vLLM on `:8050`. The OpenClaw gateway emits chat-completions to
`:8000`; this proxy is what they hit. Composer and the bridge are
upstream of OpenClaw and don't see this layer.

Usage:
  python3 vllm-proxy.py [--listen-port 8000]
                        [--upstream http://127.0.0.1:8050]
                        [--log-path /tmp/iter20_proxy.jsonl]

Environment variables (override flags):

  All ``OPENCLAW_PROXY_*`` env vars are accepted as deprecated aliases
  for the canonical ``MANYFORGE_PROXY_*`` names introduced in Phase 1
  of THREE-LANE-MIGRATION-PLAN.md. The proxy is lane-agnostic and no
  longer specific to OpenClaw; the rename reflects that. Both names
  work simultaneously; if both are set for the same suffix, the
  MANYFORGE_PROXY_* form wins. A one-time stderr deprecation notice is
  emitted at module import when any OPENCLAW_PROXY_* alias is read.

  MANYFORGE_PROXY_LISTEN_PORT
  MANYFORGE_PROXY_UPSTREAM
  MANYFORGE_PROXY_LOG_PATH
  MANYFORGE_PROXY_PROFILE        native | compat | prod  (Phase 1)
                                  Recorded in the proxy log banner and
                                  every audit entry so cross-lane
                                  reports can correlate behavior to
                                  the active mutation profile.

  (legacy names still accepted)
  OPENCLAW_PROXY_LISTEN_PORT
  OPENCLAW_PROXY_UPSTREAM
  OPENCLAW_PROXY_LOG_PATH
  OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS       cap output tokens (injected when caller omits)
  OPENCLAW_PROXY_OVERRIDE_REASONING_EFFORT override the standard reasoning_effort field
  OPENCLAW_PROXY_THINKING_TOKEN_BUDGET     soft cap on internal CoT
  OPENCLAW_PROXY_FORCE_ENABLE_THINKING     on | off | alternating-…
                                            Load-bearing control for the chat
                                            template's `enable_thinking` (the
                                            top-level field on the wire body is
                                            dead — template reads only
                                            `chat_template_kwargs.enable_thinking`).
                                            Per-profile default lives in
                                            NemoClaw-Thor/serving/config.sh as
                                            THOR_TARGET_PROXY_FORCE_ENABLE_THINKING.
  OPENCLAW_PROXY_FORCE_TOOL_CHOICE         required | required-first | alternating[-on-even]
  OPENCLAW_PROXY_USER_MESSAGE_SUFFIX       append text to last user message
  OPENCLAW_PROXY_OVERRIDE_TEMPERATURE / _TOP_P
  OPENCLAW_PROXY_USER_SUFFIX_FIRST_TURN_ONLY

Each upstream call is appended to the JSONL log as one record:
  {
    "ts": <unix_ms>,
    "request": {
      "method": "POST",
      "path": "/v1/chat/completions",
      "headers": {...},
      "body": <parsed json or raw string>,
      "mutation": {"mutations": {<key>: {"before": …, "after": …, …}}}
    },
    "response": {
      "status": 200,
      "headers": {...},
      "body": <parsed json or raw string>,
      "duration_ms": 123.4
    }
  }

The smoke harness reads this JSONL by byte-offset diff per request, so
the per-request view stays scoped without coordination with this process.

History: started life as a logging-only debug aid (`vllm-logging-proxy.py`
under `scripts/debug/`). The mutator path landed in iter 12; the
load-bearing `max_tokens` injection landed in iter 18b; renamed and
relocated to `scripts/proxy/` after iter 21 once the proxy became part of
the production-shaped recipe rather than a pure debug tool.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import http.client
import json
import os
import socketserver
import sys
import threading
import time


# =========================================================================
# Phase 1 of THREE-LANE-MIGRATION-PLAN.md — env-var alias adapter.
# =========================================================================
# The proxy is lane-agnostic and no longer specific to OpenClaw, so the
# canonical env-var prefix is now ``MANYFORGE_PROXY_*``. The historical
# ``OPENCLAW_PROXY_*`` names continue to work as deprecated aliases for
# one release cycle.
#
# Resolution order (highest priority first):
#   1. MANYFORGE_PROXY_<suffix>   (canonical)
#   2. OPENCLAW_PROXY_<suffix>    (deprecated alias)
#   3. argv flag default          (the original behavior — unchanged)
#
# Implementation: the actual ``os.environ.get(...)`` calls in the proxy
# source still read the OPENCLAW_PROXY_* names (renaming all 30+ call
# sites would be churn-heavy and increase merge conflicts; the rename
# is deferred to a future cycle). To make MANYFORGE_PROXY_* work as
# the canonical name TODAY, we materialize a BIDIRECTIONAL alias at
# import:
#
#   - For each MANYFORGE_PROXY_<suffix> set, copy to the OPENCLAW_PROXY_
#     equivalent if not already set. This makes the canonical name
#     actually drive the read sites.
#   - For each OPENCLAW_PROXY_<suffix> set (without an existing
#     MANYFORGE_ counterpart), copy to MANYFORGE_PROXY_ so the
#     canonical name is visible if anything probes that surface.
#
# If both names are set for the same suffix, **MANYFORGE_PROXY_ wins**:
# we overwrite the OPENCLAW_PROXY_ value with the MANYFORGE one before
# any read happens.
#
# A single stderr notice is emitted at import if any OPENCLAW_PROXY_*
# alias is observed without its MANYFORGE_ counterpart already set;
# this surfaces drift in caller scripts without breaking them. Set
# MANYFORGE_PROXY_NO_DEPRECATION=1 to silence.
def _apply_manyforge_proxy_aliases() -> None:
    legacy_prefix = "OPENCLAW_PROXY_"
    new_prefix = "MANYFORGE_PROXY_"
    aliased_legacy: list[str] = []  # legacy names observed without canonical
    # First pass: canonical MANYFORGE_PROXY_ → OPENCLAW_PROXY_ (overwrite).
    # This is the critical fix — without this, setting only the canonical
    # name has no effect because the read sites use OPENCLAW_PROXY_.
    for key, value in list(os.environ.items()):
        if not key.startswith(new_prefix):
            continue
        suffix = key[len(new_prefix):]
        if suffix in {"NO_DEPRECATION", "PROFILE"}:
            # Adapter-internal env names; do not project onto the
            # legacy prefix.
            continue
        os.environ[f"{legacy_prefix}{suffix}"] = value
    # Second pass: legacy OPENCLAW_PROXY_ → MANYFORGE_PROXY_ (only if
    # canonical not already set). Records observed legacy names so we
    # can emit the deprecation notice.
    for key, value in list(os.environ.items()):
        if not key.startswith(legacy_prefix):
            continue
        suffix = key[len(legacy_prefix):]
        new_key = f"{new_prefix}{suffix}"
        if new_key in os.environ:
            continue
        os.environ[new_key] = value
        aliased_legacy.append(key)
    if aliased_legacy and os.environ.get("MANYFORGE_PROXY_NO_DEPRECATION", "0") != "1":
        sys.stderr.write(
            "[vllm-proxy] DEPRECATION: env vars OPENCLAW_PROXY_* are renamed to "
            f"MANYFORGE_PROXY_*. {len(aliased_legacy)} legacy alias(es) active "
            "(set MANYFORGE_PROXY_NO_DEPRECATION=1 to silence): "
            f"{sorted(aliased_legacy)[:5]}{'…' if len(aliased_legacy) > 5 else ''}\n"
        )


_apply_manyforge_proxy_aliases()

# Active mutation profile name (Phase 1). Recorded in the proxy log
# banner and every audit entry so cross-lane reports can correlate
# behavior to the active profile. Lane adapters set this at startup;
# defaults to "compat" for OpenClaw and "native" for Direct/Hermes.
MANYFORGE_PROXY_PROFILE = (
    os.environ.get("MANYFORGE_PROXY_PROFILE", "").strip().lower() or "compat"
)
if MANYFORGE_PROXY_PROFILE not in ("native", "compat", "prod"):
    sys.stderr.write(
        f"[vllm-proxy] WARN: unknown MANYFORGE_PROXY_PROFILE={MANYFORGE_PROXY_PROFILE!r}; "
        "expected one of native|compat|prod. Continuing with the literal value.\n"
    )
import urllib.parse
from http.server import BaseHTTPRequestHandler


_LOG_LOCK = threading.Lock()
_LOG_PATH = "/tmp/openclaw_proxy.jsonl"
_UPSTREAM_HOST = "127.0.0.1"
_UPSTREAM_PORT = 18789
_UPSTREAM_SCHEME = "http"
_LISTEN_PORT = 18790

# 2026-06-05: upstream model-call socket timeout — configurable + coordinated.
# Previously hardcoded to 200.0s, which sat well BELOW the per-case assistant
# budget (ASSISTANT_TIMEOUT_S, default 300s). On the gemma/llama.cpp path a
# single prefill-heavy turn deep in a chained build (35k–43k-token prompts with
# cache invalidation; observed model-side generation up to ~446s) routinely
# exceeded 200s, so the proxy aborted the call (status=-1) ~200000ms in and
# surfaced `chat HTTP -1`/`502` to the runner before the case budget was spent.
#
# Canonical env: MANYFORGE_PROXY_UPSTREAM_TIMEOUT_S (legacy OPENCLAW_PROXY_
# accepted as a fallback; see _apply_manyforge_proxy_aliases). Standalone
# default 300s. The launcher COORDINATES this with the case budget (sets it
# slightly BELOW ASSISTANT_TIMEOUT_S) so the proxy — guardian of the single
# llama.cpp slot — fails first and closes the socket to release the slot before
# the case-level timeout, rather than racing it. Parsed defensively: a
# non-numeric value (e.g. "300s") warns and falls back instead of crashing the
# proxy at startup.
def _resolve_upstream_timeout_s(default: float = 300.0) -> float:
    raw = (
        os.environ.get("MANYFORGE_PROXY_UPSTREAM_TIMEOUT_S")
        or os.environ.get("OPENCLAW_PROXY_UPSTREAM_TIMEOUT_S")
    )
    if raw is None or not str(raw).strip():
        return default
    try:
        val = float(str(raw).strip())
        if val <= 0:
            raise ValueError("must be positive")
        return val
    except (ValueError, TypeError):
        sys.stderr.write(
            f"[vllm-proxy] WARNING: invalid upstream timeout {raw!r}; "
            f"falling back to {default}s\n"
        )
        return default


_UPSTREAM_TIMEOUT_S = _resolve_upstream_timeout_s()


def _resolve_max_request_chars(default: int = 0) -> int:
    raw = (
        os.environ.get("MANYFORGE_PROXY_MAX_REQUEST_CHARS")
        or os.environ.get("OPENCLAW_PROXY_MAX_REQUEST_CHARS")
    )
    if raw is None or not str(raw).strip():
        return default
    try:
        val = int(str(raw).strip())
        if val < 0:
            raise ValueError("must be non-negative")
        return val
    except (ValueError, TypeError):
        sys.stderr.write(
            f"[vllm-proxy] WARNING: invalid max request chars {raw!r}; "
            f"falling back to {default}\n"
        )
        return default


# 0 means disabled. This is an operator-set guardrail, not a default mutator:
# it protects constrained backends from pathological retained-history prompts
# once a deployment has chosen its real context/latency budget.
_MAX_REQUEST_CHARS = _resolve_max_request_chars()

# =========================================================================
# Tool-surface drift check (Commit C, 2026-06-03).
# =========================================================================
#
# The bridge prompt teaches the model to dispatch via either
# ``tool_search_code`` (code mode) or ``tool_search``/``tool_describe``/
# ``tool_call`` (tools mode). The active surface is determined by
# sandbox-side OpenClaw config, NOT by the bridge — so if the operator
# sets ``OPENCLAW_ASSISTANT_TOOL_SURFACE=tools`` at the bridge but
# OpenClaw still ships only ``tool_search_code``, the model gets a
# tools-mode primer for a code-mode surface and refuses to dispatch.
#
# The bridge cannot directly observe ``tools[]`` (in CLI shell-out mode
# it does not sit on the chat-completions path). The PROXY can —
# every chat-completions request flows through here. We compare the
# observed ``tools[]`` to the operator-declared surface and emit one
# warning per session if they disagree.
#
# Implementation:
# - ``OPENCLAW_PROXY_TOOL_SURFACE`` env (default empty = drift check
#   disabled, preserving backwards compatibility for proxy deployments
#   that do not run the lane-aware bridge).
# - Session key: the bridge passes ``X-OpenClaw-Conversation`` /
#   ``X-OpenClaw-Request`` headers in production; fall back to a
#   process-lifetime "default" key when absent so probes still warn.
# - Dedup: ``_TOOL_SURFACE_WARNED_PAIRS`` set (per (expected, observed)
#   pair, globally for the proxy lifetime). The previous per-session
#   dedup was changed to per-pair after empirical evidence (2026-06-03)
#   that real OpenClaw → vLLM traffic carries no session header — the
#   session-key would degenerate to a process sentinel and dedup would
#   collapse to one warning total. See _check_tool_surface_drift. We warn
#   the first time we see a session, then stay quiet — operators
#   want signal, not a per-request flood.
_TOOL_SURFACE_EXPECTED: str = ""  # "code" | "tools" | "" (disabled)
# Dedup model (revised 2026-06-03): real OpenClaw traffic carries no
# session header, so per-session dedup degenerates. We instead memo
# the (expected, observed) pairs we have already warned on — see
# _TOOL_SURFACE_WARNED_PAIRS defined below the helper functions.
_TOOL_SURFACE_LOCK = threading.Lock()
# When True, classifier="unknown" ALSO fires the drift warning. Off
# by default so the proxy stays quiet on direct-lane traffic (where
# tools[] is the real manyforge catalog, classified "unknown" from
# the surface-detection perspective). Operators of OpenClaw-only
# proxies can set OPENCLAW_PROXY_WARN_ON_UNKNOWN=true to catch
# direct-lane drift or malformed-tool requests (external review
# finding 4, 2026-06-03).
_TOOL_SURFACE_WARN_ON_UNKNOWN: bool = False


_CODE_MODE_TOOL_NAMES = frozenset({"tool_search_code"})
_TOOLS_MODE_TOOL_NAMES = frozenset({"tool_search", "tool_describe", "tool_call"})


def _classify_tools_array(tools: object) -> str:
    """Return ``"code"`` / ``"tools"`` / ``"unknown"`` based on names.

    Returns "code" if the observed names match the code-mode surface
    exactly (single ``tool_search_code``), "tools" if they include
    the three tools-mode verbs, "unknown" for anything else (an
    OpenClaw build we don't yet know about, or a probe with real
    tools[]).
    """
    if not isinstance(tools, list):
        return "unknown"
    names: list[str] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        n = fn.get("name") or t.get("name")
        if isinstance(n, str) and n:
            names.append(n)
    name_set = set(names)
    if name_set == _CODE_MODE_TOOL_NAMES:
        return "code"
    if _TOOLS_MODE_TOOL_NAMES.issubset(name_set):
        return "tools"
    return "unknown"


def _tool_surface_session_key(headers: object, body: object) -> str:
    """Best-effort session key for the audit log payload.

    Reality check (verified 2026-06-03 by inspecting real OpenClaw →
    vLLM traffic): the OpenAI JS SDK that OpenClaw uses does NOT
    forward a conversation/session header to vLLM. Headers present
    are ``User-Agent: OpenAI/JS``, ``X-Stainless-*``, ``traceparent``,
    ``Authorization`` — none stable per conversation. Body is the
    plain ``/v1/chat/completions`` envelope with no session field.
    So this function will return ``__no_session__`` for almost all
    real traffic; the dedup logic below is now (expected, observed)-
    pair based, NOT session-keyed, precisely because of this. The
    session key is still recorded in the warning payload for any
    test/probe context that DOES set a header.
    """
    if hasattr(headers, "get"):
        for h in ("X-OpenClaw-Conversation", "x-openclaw-conversation",
                  "X-Request-ID", "x-request-id", "traceparent"):
            v = headers.get(h)  # type: ignore[attr-defined]
            if v:
                return str(v)
    if isinstance(body, dict):
        for k in ("conversationId", "conversation_id", "session_id"):
            v = body.get(k)
            if v:
                return str(v)
    return "__no_session__"


# Module-level dedup memo: ``frozenset[(expected, observed)]``.
# Operator misconfig (e.g. configured=code, observed=tools) is a
# global condition — one warning is what the operator needs to
# diagnose, not per-session repetition. This memo is reset on
# proxy restart (a config change requires a restart, so the
# operator sees the warning again if the misconfig persists).
_TOOL_SURFACE_WARNED_PAIRS: set = set()


def _check_tool_surface_drift(
    *, path: str, headers: object, body: object, log_extra: dict
) -> None:
    """Emit a ``tool_surface_mismatch`` warning ONCE per unique
    (configured, observed) pair if the observed tools[] surface
    does not match the operator's configured
    ``OPENCLAW_PROXY_TOOL_SURFACE``.

    Dedup model (revised 2026-06-03 after finding that real OpenClaw
    traffic carries no session header — see _tool_surface_session_key
    docstring): we warn ONCE PER UNIQUE (expected, observed) PAIR
    for the proxy's lifetime. Operator misconfig is global. Tools
    mode and code mode produce a single warning each if both fire.
    A future build introducing a third surface produces one more.
    No per-session repetition, no per-session flood.

    Cheap (one set lookup per chat-completions).
    """
    if not _TOOL_SURFACE_EXPECTED:
        return  # drift check disabled
    if not path.endswith("/v1/chat/completions"):
        return
    if not isinstance(body, dict):
        return
    observed = _classify_tools_array(body.get("tools"))
    if observed == "unknown" and not _TOOL_SURFACE_WARN_ON_UNKNOWN:
        # Probe with no recognizable surface — nothing to enforce
        # unless the operator opted into strict-unknown mode.
        return
    expected = _TOOL_SURFACE_EXPECTED
    if observed == expected:
        return
    pair = (expected, observed)
    # Dedup: warn ONCE per (expected, observed) pair globally.
    with _TOOL_SURFACE_LOCK:
        if pair in _TOOL_SURFACE_WARNED_PAIRS:
            return
        _TOOL_SURFACE_WARNED_PAIRS.add(pair)
    session = _tool_surface_session_key(headers, body)
    # Per the external-review suggestion: warning carries configured,
    # observed, request/session id, and exact observed tool names.
    observed_names = []
    tools = body.get("tools") or []
    if isinstance(tools, list):
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            n = fn.get("name") or t.get("name")
            if isinstance(n, str) and n:
                observed_names.append(n)
    payload = {
        "event": "tool_surface_mismatch",
        "configured_surface": expected,
        "observed_surface": observed,
        "observed_tool_names": observed_names,
        "session_key": session,
        "request_id": log_extra.get("request_id"),
    }
    try:
        with _LOG_LOCK:
            with open(_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
    except Exception:
        pass
    # Also surface to stderr so operators watching logs in real time
    # see it immediately (the JSONL is for replay/audit).
    try:
        import sys as _sys
        print(
            f"openclaw-proxy: tool_surface_mismatch "
            f"configured={expected!r} observed={observed!r} "
            f"session={session!r} tools={observed_names!r}",
            file=_sys.stderr, flush=True,
        )
    except Exception:
        pass

# 2026-05-09: optional outbound mutation. Activated only when env vars
# below are set, so the proxy stays a pure logger by default. The
# motivation is OpenClaw's gateway dropping `tool_choice: "required"`
# on the gateway → vLLM hop — this layer can re-inject it (and other
# inference params) without touching OpenClaw or the bridge code.
_FORCE_TOOL_CHOICE: str | None = None        # "required" | "auto" | "required-first" |
                                             # "alternating" | "alternating-on-even" | None
                                             # "required-first" = inject "required" only on the FIRST
                                             # request per conversation (subsequent requests pass through).
                                             # "alternating" = inject on ODD-numbered turns (1, 3, 5, ...).
                                             # "alternating-on-even" = inject on EVEN-numbered turns (2, 4, 6, ...);
                                             # leaves odd turns free. Pairs with
                                             # _FORCE_ENABLE_THINKING="alternating-off-on-even" so turn 1
                                             # uses model's reasoning, turn 2 forces tool emission with
                                             # thinking-off (prevents the model from sliding into prose
                                             # narration on the second/fourth/etc. agentic round).
_SEEN_CONVERSATIONS: set[str] = set()        # tracks first-seen conversationIds for "required-first" mode
_OVERRIDE_TEMPERATURE: float | None = None   # e.g. 0.0 to make recovery deterministic
_OVERRIDE_MAX_TOKENS: int | None = None      # cap output tokens
_OVERRIDE_TOP_P: float | None = None
_OVERRIDE_REASONING_EFFORT: str | None = None  # e.g. "off" for tool-only lanes
_THINKING_TOKEN_BUDGET: int | None = None    # if set, injects chat_template_kwargs.thinking_token_budget
                                             # = N on every chat-completions request. Caps the
                                             # `<think>...</think>` envelope so it doesn't eat the
                                             # max_completion_tokens budget and starve the visible
                                             # assistant content. Per Qwen3-VL technical report
                                             # (arXiv 2509.17765), 512 is the sweet spot for 8B-class
                                             # robotics tool calls (~95% accuracy, half the latency
                                             # of unbounded; 256 drops accuracy ~6 pts; 1024 is overkill).
_FORCE_ENABLE_THINKING: str | None = None    # "on" | "off" | "alternating-off-on-even" | None
                                             # "alternating-off-on-even" = on odd turns (1, 3, 5),
                                             # don't mutate thinking (vLLM default applies); on even
                                             # turns (2, 4, 6), inject chat_template_kwargs.enable_thinking=false.
                                             # Pairs with alternating tool_choice mode for a "first-turn
                                             # reasons, second-turn forces tool" pattern within the
                                             # OpenClaw agent loop.
_USER_MESSAGE_SUFFIX: str | None = None      # appended to the LAST user message;
                                             # used to inject a generic
                                             # plan-then-execute hint without
                                             # touching corpus prompts.
_USER_SUFFIX_FIRST_TURN_ONLY: bool = False   # if True, only inject the suffix
                                             # when no `assistant` messages
                                             # appear yet (first turn of the
                                             # conversation). Mirrors the
                                             # required-first tool_choice
                                             # pattern. Avoids per-turn read
                                             # overhead on long agent loops /
                                             # corpus chains.
_MUTATE_PATHS = ("/v1/chat/completions",)    # only mutate chat-completions; leave /v1/models etc. alone


def _maybe_mutate_request(path: str, body: bytes) -> tuple[bytes, dict | None]:
    """Optionally rewrite the upstream request body to enforce
    tool_choice / temperature / max_tokens. Returns (new_body, mutation_record).

    The mutation_record is a small dict listing what was changed (or None
    if no mutation applied) — appended to the log so it's auditable.
    """
    # Only mutate POST chat-completions; everything else is pass-through.
    if not any(path.endswith(p) for p in _MUTATE_PATHS):
        return body, None
    # Skip if no mutation is configured at all.
    if (_FORCE_TOOL_CHOICE is None and _OVERRIDE_TEMPERATURE is None
            and _OVERRIDE_MAX_TOKENS is None and _OVERRIDE_TOP_P is None
            and _OVERRIDE_REASONING_EFFORT is None
            and _USER_MESSAGE_SUFFIX is None and _FORCE_ENABLE_THINKING is None
            and _THINKING_TOKEN_BUDGET is None
            and not _GUIDED_TOOL_CALLS
            and not _TOOL_ERROR_REWRITE
            and not _UNWRAP_TOOL_CALL_ARGS
            and not _TOOL_PARSER):
        return body, None
    try:
        parsed = json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError):
        return body, None  # non-JSON: leave alone
    if not isinstance(parsed, dict):
        return body, None

    changes: dict[str, dict] = {}

    # tool_choice — only inject when tools[] is non-empty AND user didn't
    # set tool_choice="none" explicitly (which means "I want no tool").
    # "required-first" mode: only inject on the FIRST request per
    # conversation. Subsequent requests pass through, allowing the
    # agent loop to exit naturally via a text-only response.
    if _FORCE_TOOL_CHOICE is not None:
        tools = parsed.get("tools")
        existing_tc = parsed.get("tool_choice")
        if isinstance(tools, list) and tools and existing_tc != "none":
            effective_tc = _FORCE_TOOL_CHOICE
            should_apply = True
            if _FORCE_TOOL_CHOICE in ("required-first", "alternating", "alternating-on-even"):
                msgs = parsed.get("messages") or []
                # Count assistant turns already in the conversation. The
                # bridge appends one assistant message per agent-loop turn,
                # so this is the model-turn index (0 = first request).
                assistant_count = sum(
                    1 for m in msgs
                    if isinstance(m, dict) and m.get("role") == "assistant"
                )
                if _FORCE_TOOL_CHOICE == "required-first":
                    should_apply = assistant_count == 0
                elif _FORCE_TOOL_CHOICE == "alternating-on-even":
                    # EVEN turns (2, 4, 6, ...) get tool_choice=required.
                    # turn_number = assistant_count + 1, so apply when
                    # turn is even → assistant_count is odd.
                    should_apply = (assistant_count % 2) == 1
                else:  # alternating (legacy, on odd turns)
                    should_apply = (assistant_count % 2) == 0
                effective_tc = "required"
            if should_apply and existing_tc != effective_tc:
                changes["tool_choice"] = {
                    "before": existing_tc,
                    "after": effective_tc,
                    "mode": _FORCE_TOOL_CHOICE,
                }
                parsed["tool_choice"] = effective_tc

    if _OVERRIDE_TEMPERATURE is not None:
        before = parsed.get("temperature")
        if before != _OVERRIDE_TEMPERATURE:
            changes["temperature"] = {"before": before, "after": _OVERRIDE_TEMPERATURE}
            parsed["temperature"] = _OVERRIDE_TEMPERATURE

    if _OVERRIDE_MAX_TOKENS is not None:
        # Common keys vary across upstream APIs; respect both. Inject
        # max_tokens even when the caller omitted it — OpenClaw → vLLM
        # leaves the field unset, which lets vLLM default to the model's
        # full context window and produce 30+ minute generations under
        # thinking-on.
        present_keys = [k for k in ("max_tokens", "max_completion_tokens") if k in parsed]
        if present_keys:
            for key in present_keys:
                if parsed[key] != _OVERRIDE_MAX_TOKENS:
                    changes[key] = {"before": parsed[key], "after": _OVERRIDE_MAX_TOKENS}
                    parsed[key] = _OVERRIDE_MAX_TOKENS
        else:
            changes["max_tokens"] = {"before": None, "after": _OVERRIDE_MAX_TOKENS, "injected": True}
            parsed["max_tokens"] = _OVERRIDE_MAX_TOKENS

    if _OVERRIDE_TOP_P is not None:
        before = parsed.get("top_p")
        if before != _OVERRIDE_TOP_P:
            changes["top_p"] = {"before": before, "after": _OVERRIDE_TOP_P}
            parsed["top_p"] = _OVERRIDE_TOP_P

    if _OVERRIDE_REASONING_EFFORT is not None:
        before = parsed.get("reasoning_effort")
        if before != _OVERRIDE_REASONING_EFFORT:
            changes["reasoning_effort"] = {
                "before": before,
                "after": _OVERRIDE_REASONING_EFFORT,
            }
            parsed["reasoning_effort"] = _OVERRIDE_REASONING_EFFORT

    # Always-on thinking_token_budget cap (no per-turn alternation).
    if _THINKING_TOKEN_BUDGET is not None:
        ctk = parsed.get("chat_template_kwargs")
        if not isinstance(ctk, dict):
            ctk = {}
        before_val = ctk.get("thinking_token_budget")
        if before_val != _THINKING_TOKEN_BUDGET:
            ctk["thinking_token_budget"] = _THINKING_TOKEN_BUDGET
            parsed["chat_template_kwargs"] = ctk
            changes["thinking_token_budget"] = {
                "before": before_val,
                "after": _THINKING_TOKEN_BUDGET,
            }

    # Per-turn enable_thinking control. Inject `chat_template_kwargs.enable_thinking`
    # based on _FORCE_ENABLE_THINKING mode and the current turn parity (counted
    # from `messages[*].role == "assistant"`).
    if _FORCE_ENABLE_THINKING is not None:
        msgs_for_thinking = parsed.get("messages") or []
        asst_count_thinking = sum(
            1 for m in msgs_for_thinking
            if isinstance(m, dict) and m.get("role") == "assistant"
        )
        target_value: bool | None = None
        if _FORCE_ENABLE_THINKING == "on":
            target_value = True
        elif _FORCE_ENABLE_THINKING == "off":
            target_value = False
        elif _FORCE_ENABLE_THINKING == "alternating-off-on-even":
            # Odd turns (asst_count = 0, 2, 4, ...) → don't mutate (vLLM default applies)
            # Even turns (asst_count = 1, 3, 5, ...) → force enable_thinking=false
            if asst_count_thinking % 2 == 1:
                target_value = False
            else:
                target_value = None  # leave alone
        if target_value is not None:
            ctk = parsed.get("chat_template_kwargs")
            if not isinstance(ctk, dict):
                ctk = {}
            before_val = ctk.get("enable_thinking")
            if before_val != target_value:
                ctk["enable_thinking"] = target_value
                parsed["chat_template_kwargs"] = ctk
                changes["enable_thinking"] = {
                    "before": before_val,
                    "after": target_value,
                    "mode": _FORCE_ENABLE_THINKING,
                    "asst_count": asst_count_thinking,
                }
            # 2026-06-01: top-level `enable_thinking` mirror removed.
            # The chat template (`chat_template.jinja:12`) only reads
            # `chat_template_kwargs.enable_thinking`; the top-level field
            # is dead at the template level. Earlier 2026-05-31 note
            # claimed vLLM treated top-level as source of truth and
            # ignored ctk — re-testing 2026-06-01 with reasoning_content
            # inspection proved that claim was wrong: when ctk.enable_thinking
            # is set, the template honors it. Keeping the wire surface
            # narrow to ctk simplifies the propagation trace.

    # Append a generic suffix to the LAST user message in `messages`.
    # Cross-cutting, not per-prompt — same wording fires every request.
    # The motivating use case is plan-then-execute: "before answering,
    # call the appropriate read tool to refresh state". Idempotent: if
    # the suffix is already at the tail of the last user message, skip.
    # Optionally restrict to first-turn-only via _USER_SUFFIX_FIRST_TURN_ONLY.
    if _USER_MESSAGE_SUFFIX:
        msgs = parsed.get("messages")
        if isinstance(msgs, list) and msgs and _USER_SUFFIX_FIRST_TURN_ONLY:
            # Skip suffix injection when prior assistant turns exist —
            # the conversation is mid-flight and the model has already
            # been guided once.
            assistant_count = sum(
                1 for m in msgs
                if isinstance(m, dict) and m.get("role") == "assistant"
            )
            if assistant_count > 0:
                msgs = None  # disable suffix path below
        if isinstance(msgs, list) and msgs:
            # Find the LAST message with role="user".
            last_user_idx = None
            for i in range(len(msgs) - 1, -1, -1):
                m = msgs[i]
                if isinstance(m, dict) and m.get("role") == "user":
                    last_user_idx = i
                    break
            if last_user_idx is not None:
                m = msgs[last_user_idx]
                content = m.get("content")
                # OpenAI-style content is a string OR a list of parts.
                if isinstance(content, str):
                    if not content.rstrip().endswith(_USER_MESSAGE_SUFFIX.rstrip()):
                        m["content"] = content.rstrip() + "\n\n" + _USER_MESSAGE_SUFFIX
                        changes["user_suffix"] = {
                            "before_len": len(content),
                            "after_len": len(m["content"]),
                            "appended_chars": len(_USER_MESSAGE_SUFFIX) + 2,
                        }
                elif isinstance(content, list):
                    # Find the last text part and append; otherwise add a new text part.
                    last_text_idx = None
                    for j in range(len(content) - 1, -1, -1):
                        part = content[j]
                        if isinstance(part, dict) and part.get("type") in ("text", None) and isinstance(part.get("text"), str):
                            last_text_idx = j
                            break
                    if last_text_idx is not None:
                        part = content[last_text_idx]
                        old = part.get("text", "")
                        if not old.rstrip().endswith(_USER_MESSAGE_SUFFIX.rstrip()):
                            part["text"] = old.rstrip() + "\n\n" + _USER_MESSAGE_SUFFIX
                            changes["user_suffix"] = {
                                "before_len": len(old),
                                "after_len": len(part["text"]),
                                "appended_chars": len(_USER_MESSAGE_SUFFIX) + 2,
                            }
                    else:
                        content.append({"type": "text", "text": _USER_MESSAGE_SUFFIX})
                        changes["user_suffix"] = {
                            "appended_part_chars": len(_USER_MESSAGE_SUFFIX),
                        }

    # 2026-06-01: parser-specific tool-call constraint (Level 2 guided).
    # When _TOOL_PARSER is known, choose the right vLLM constraint kind:
    #   - hermes      → structural_tag (JSON-in-tags, per-tool JSON schema)
    #   - qwen3_coder → grammar (Lark EBNF, per-tool param-key whitelist)
    # Model retains free choice ask/act; the constraint only activates
    # when the model emits `<tool_call>`. Skip if caller already set
    # response_format or structured_outputs.
    if _TOOL_PARSER and isinstance(parsed.get("tools"), list) and parsed["tools"]:
        existing_rf = parsed.get("response_format")
        existing_so = parsed.get("structured_outputs")
        if not isinstance(existing_rf, dict) and not isinstance(existing_so, dict):
            if _TOOL_PARSER == "hermes":
                try:
                    stag = _build_structural_tag(parsed["tools"], _TOOL_PARSER)
                except Exception:
                    stag = None
                if stag is not None:
                    parsed["response_format"] = stag
                    changes["response_format_structural_tag"] = {
                        "parser": _TOOL_PARSER,
                        "n_structures": len(stag.get("structures", [])),
                    }
            elif _TOOL_PARSER == "qwen3_coder":
                try:
                    grammar = _build_qwen3_coder_grammar(parsed["tools"])
                except Exception:
                    grammar = None
                if grammar:
                    parsed["structured_outputs"] = {"grammar": grammar}
                    changes["structured_outputs_grammar"] = {
                        "parser": _TOOL_PARSER,
                        "grammar_chars": len(grammar),
                    }

    # 2026-06-01: prompt-aware guided tool-call decoding (Level 1).
    # When the last user message is action-shaped AND the request has
    # tools[], force `tool_choice="required"` so vLLM uses guided
    # decoding against the tool schemas. Skip if FORCE_TOOL_CHOICE
    # already injected something (proxy modes own that field).
    if (_GUIDED_TOOL_CALLS and _FORCE_TOOL_CHOICE is None
            and isinstance(parsed.get("tools"), list) and parsed["tools"]):
        existing_tc = parsed.get("tool_choice")
        if existing_tc not in ("required", "none") and not isinstance(existing_tc, dict):
            # Extract the most-recent user message text
            last_user_text = ""
            for m in reversed(parsed.get("messages") or []):
                if not isinstance(m, dict) or m.get("role") != "user":
                    continue
                c = m.get("content")
                if isinstance(c, str):
                    last_user_text = c
                elif isinstance(c, list):
                    parts: list[str] = []
                    for p in c:
                        if isinstance(p, dict) and isinstance(p.get("text"), str):
                            parts.append(p["text"])
                    last_user_text = " ".join(parts)
                break
            # Find the user_request section to focus heuristic on the
            # user's actual prompt (not the wrapping rules/preamble).
            ur_marker = "## user_request"
            idx = last_user_text.find(ur_marker)
            target = last_user_text[idx + len(ur_marker):] if idx >= 0 else last_user_text
            target_lc = target.lower()
            # Quick action-shape heuristic
            has_verb = any(v in (" " + target_lc) for v in _ACTION_VERBS)
            has_noun = any(n in target_lc for n in _DOMAIN_NOUNS)
            if has_verb and has_noun:
                parsed["tool_choice"] = "required"
                changes["tool_choice_guided"] = {
                    "before": existing_tc,
                    "after": "required",
                    "reason": "action-shaped prompt",
                }

    # 2026-06-01: 4xx tool-result directive rewrite.
    # When the last tool message contains a validation failure with
    # structured guidance fields, prepend a directive preamble that
    # names the valid values from the structured fields so the model
    # can't miss them. Surgical — touches at most one message.
    if _TOOL_ERROR_REWRITE:
        msgs_te = parsed.get("messages")
        if isinstance(msgs_te, list) and msgs_te:
            last_tool_idx = -1
            for i in range(len(msgs_te) - 1, -1, -1):
                m = msgs_te[i]
                if isinstance(m, dict) and m.get("role") == "tool":
                    last_tool_idx = i
                    break
            if last_tool_idx >= 0:
                last_tool = msgs_te[last_tool_idx]
                content = last_tool.get("content")
                if isinstance(content, str) and content:
                    is_error = any(marker.lower() in content.lower()
                                   for marker in _TOOL_ERROR_MARKERS)
                    if is_error:
                        # Pull any structured field values to surface
                        try:
                            parsed_content = json.loads(content)
                        except (ValueError, TypeError):
                            parsed_content = None
                        directive_lines: list[str] = []

                        def _walk_for_fields(obj):
                            if isinstance(obj, dict):
                                for k, v in obj.items():
                                    if k in _VALIDATOR_FIELDS and isinstance(v, list) and v:
                                        preview = v[:8]
                                        suffix = "" if len(v) <= 8 else f" (+{len(v)-8} more)"
                                        directive_lines.append(
                                            f"  {k}: {preview}{suffix}"
                                        )
                                    else:
                                        _walk_for_fields(v)
                            elif isinstance(obj, list):
                                for x in obj:
                                    _walk_for_fields(x)

                        if parsed_content is not None:
                            _walk_for_fields(parsed_content)

                        already = content.startswith("[VALIDATOR]")
                        # Extract a "detail" / "message" / "error" prose snippet
                        # if present, for cases where structured fields are absent.
                        detail_text = ""
                        if parsed_content is not None and isinstance(parsed_content, dict):
                            for k in ("detail", "message", "error", "errorMessage"):
                                v = parsed_content.get(k)
                                if isinstance(v, str) and v.strip():
                                    detail_text = v.strip()
                                    break
                        if (directive_lines or detail_text) and not already:
                            # Build preamble. If we have structured fields, use the
                            # directive list. If only prose detail, use the prose
                            # as the directive.
                            preamble_lines = [
                                "[VALIDATOR] The previous call failed validation. "
                                "Read the error BELOW and fix the specific failing "
                                "field — do NOT retry with the same args, do NOT "
                                "invent new values."
                            ]
                            if directive_lines:
                                preamble_lines.extend(directive_lines)
                            elif detail_text:
                                # Cap the detail length to keep preamble tight
                                preamble_lines.append(f"  detail: {detail_text[:400]}")
                            preamble_lines.append("\n--- original tool result ---\n")
                            preamble = "\n".join(preamble_lines)
                            last_tool["content"] = preamble + content
                            changes["tool_error_rewrite"] = {
                                "field_count": len(directive_lines),
                                "used_detail_fallback": bool(detail_text and not directive_lines),
                                "tool_idx": last_tool_idx,
                            }

    # 2026-06-02: hermes <tool_call> wrapper unwrap on history.
    # When cosmos (and other hermes-trained models) emit a tool call
    # while thinking-on is active, vLLM's hermes parser sometimes
    # leaves the outer `<tool_call>{...}</tool_call>` wrapper inside
    # `arguments` instead of extracting just the inner object. The
    # turn that produced the call returns 200 (the parser DID record
    # the call), but on the NEXT turn the chat template calls
    # `json.loads(arguments)` and 400s on a wrapper that starts with
    # `<`. This pass detects that exact pattern and replaces the
    # wrapped string with just the inner `arguments` value (as a JSON
    # string). Pass-through on cleanly-shaped args, on non-string
    # args, and on args that fail to extract.
    if _UNWRAP_TOOL_CALL_ARGS:
        msgs_uw = parsed.get("messages")
        if isinstance(msgs_uw, list):
            unwrapped_count = 0
            for m in msgs_uw:
                if not isinstance(m, dict) or m.get("role") != "assistant":
                    continue
                tcs = m.get("tool_calls")
                if not isinstance(tcs, list):
                    continue
                for tc in tcs:
                    fn = tc.get("function") if isinstance(tc, dict) else None
                    if not isinstance(fn, dict):
                        continue
                    args = fn.get("arguments")
                    if not isinstance(args, str):
                        continue
                    stripped = args.lstrip()
                    if not stripped.startswith("<tool_call>"):
                        continue
                    # Strip wrapper tags; tolerate optional closing tag.
                    inner = stripped[len("<tool_call>"):]
                    end = inner.rfind("</tool_call>")
                    if end != -1:
                        inner = inner[:end]
                    inner = inner.strip()
                    # The inner payload is hermes-format:
                    #   {"name": "<tool>", "arguments": {...}}
                    # We want JUST the inner `arguments` object as a
                    # JSON string so the next-turn chat template can
                    # parse it. Fall back to leaving the field alone
                    # if anything in this extraction fails.
                    try:
                        parsed_inner = json.loads(inner)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(parsed_inner, dict):
                        continue
                    real_args = parsed_inner.get("arguments")
                    if real_args is None:
                        continue
                    if isinstance(real_args, str):
                        # Already a JSON string — pass through as-is.
                        fn["arguments"] = real_args
                    else:
                        # Re-serialize the dict to a JSON string.
                        fn["arguments"] = json.dumps(
                            real_args, separators=(",", ":")
                        )
                    unwrapped_count += 1
            if unwrapped_count:
                changes["unwrap_tool_call_args"] = {
                    "wrappers_stripped": unwrapped_count,
                }

    if not changes:
        return body, None
    new_body = json.dumps(parsed, separators=(",", ":")).encode("utf-8")
    return new_body, {"mutations": changes}


# Round 11 (2026-06-01): multi-criteria loop-break
#   5 independent detectors fire ONE generic reflection message.
#   Each detector is env-gated so any can be flipped off mid-run by
#   restarting just the proxy (vLLM container undisturbed). Inner-loop
#   only: consecutive counters reset on a non-tool assistant turn.
#
#   Existing knobs preserved:
#     OPENCLAW_PROXY_LOOP_REFLECT_AT (default 4)  same-tool reflect threshold
#     OPENCLAW_PROXY_LOOP_STOP_AT    (default 8)  same-tool hard-stop threshold
#
#   New trigger toggles (default ON; set to "0" to disable):
#     OPENCLAW_PROXY_LOOP_TRIGGER_SAME_TOOL       same tool >= REFLECT_AT
#     OPENCLAW_PROXY_LOOP_TRIGGER_SAME_ARGS       last 2 calls identical name+args
#     OPENCLAW_PROXY_LOOP_TRIGGER_RESULT_REPEAT   last 2 tool results identical
#     OPENCLAW_PROXY_LOOP_TRIGGER_NAMESPACE       same namespace >= NAMESPACE_AT
#     OPENCLAW_PROXY_LOOP_TRIGGER_TURN_COUNTER    any-tool consecutive >= TURN_COUNTER_AT
#
#   New thresholds:
#     OPENCLAW_PROXY_LOOP_NAMESPACE_AT      (default 5)
#     OPENCLAW_PROXY_LOOP_TURN_COUNTER_AT   (default 5)
#
#   One generic reflection message is injected for ALL triggers. The
#   trigger that fired is logged in the JSONL line for operator visibility.
#   First-to-fire wins (priority order matches detector list above).
_REFLECT_AT = int(os.environ.get("OPENCLAW_PROXY_LOOP_REFLECT_AT", "4") or "4")
_STOP_AT = int(os.environ.get("OPENCLAW_PROXY_LOOP_STOP_AT", "8") or "8")
_LEGACY_THRESHOLD = int(os.environ.get("OPENCLAW_PROXY_LOOP_TOOL_THRESHOLD", "0") or "0")
if _LEGACY_THRESHOLD > 0:
    _STOP_AT = _LEGACY_THRESHOLD
    if "OPENCLAW_PROXY_LOOP_REFLECT_AT" not in os.environ:
        _REFLECT_AT = 0

_NAMESPACE_AT = int(os.environ.get("OPENCLAW_PROXY_LOOP_NAMESPACE_AT", "5") or "5")
_NAMESPACE_STOP_AT = int(os.environ.get("OPENCLAW_PROXY_LOOP_NAMESPACE_STOP_AT", "16") or "16")
_TURN_COUNTER_AT = int(os.environ.get("OPENCLAW_PROXY_LOOP_TURN_COUNTER_AT", "5") or "5")

# 2026-06-01: generic malformed-tool-call detection (response-side).
# Fires when the response content contains a tool name from the request's
# tools[] AND tool-call format markers, but no structured tool_calls came
# through. Strong signal the model tried to emit a tool but the parser
# dropped it. Model-agnostic — uses tool names from the request, markers
# common across all known formats (hermes JSON, qwen3_coder XML,
# nemotron, mistral, etc.). Diagnostic only: writes a JSONL event with
# matched_tool + markers; the response itself is forwarded unmodified.
_DETECT_MALFORMED_TOOL_CALL = (
    os.environ.get("OPENCLAW_PROXY_DETECT_MALFORMED_TOOL_CALL", "1") or "1"
).strip().lower() in ("1", "true", "yes", "on")

_MALFORMED_MARKERS: tuple[str, ...] = (
    "<parameter=", "<function=", "<tool_call>", "</tool_call>", "</function>",
    "\"arguments\":", "\"tool_calls\":", "function_call",
)

# 2026-06-01: generic tool-name normalization (response-side mutation).
# Models routinely emit MCP tool names without their namespace prefix
# (e.g. `tree_draft_wrap_node` instead of `manyforge__tree_draft_wrap_node`).
# OpenClaw's MCP dispatcher requires the exact registered name, so the
# call fails. Active fix: when the model's emitted name is missing from
# the request's `tools[]` but has EXACTLY ONE prefix-suffix match in the
# catalog, rewrite the name in the response stream before the agent loop
# sees it. Model-agnostic (any namespace prefix) and unambiguous-only
# (multiple matches → leave alone; let dispatch fail honestly).
_NORMALIZE_TOOL_NAMES = (
    os.environ.get("OPENCLAW_PROXY_NORMALIZE_TOOL_NAMES", "1") or "1"
).strip().lower() in ("1", "true", "yes", "on")

# 2026-06-04: nested MCP-id normalization for OpenClaw tools-mode.
#
# Why this mutation exists. OpenClaw bundle-mcp exposes ManyForge tools
# behind three discovery verbs (tool_search/tool_describe/tool_call). The
# real ManyForge id appears nested inside `tool_call.arguments.id` and
# `tool_describe.arguments.id` — NOT in the OpenAI `function.name` field
# that NORMALIZE_TOOL_NAMES already handles. Cosmos (and probably other
# models trained on bare canonical ManyForge names) emits one of three
# wrong shapes:
#
#   bare:    "id":"tree_draft_insert_node"
#   dashed:  "id":"manyforge__tree-draft-insert_node"
#   MCP loc: "id":"mcp:bundle-mcp:manyforge__tree_draft_insert_node"
#
# OpenClaw rejects all three with "Unknown tool id" (verified 2026-06-04
# from /sandbox/.openclaw/logs/gateway-persistent.log). The model then
# loops through wrong shapes until it finally emits the canonical
# `manyforge__<bare_underscored>` form — costing many wasted turns per
# case and frequently hitting the 5-min composer cap.
#
# The fix is structural: rewrite the nested id field to the canonical
# `manyforge__<bare_underscored>` form before OpenClaw's MCP dispatcher
# sees it. This makes ALL the loops above succeed on the first try,
# regardless of which shape the model chose. Each rewrite is logged so
# the audit shows the original-vs-rewritten pair for forensics.
#
# Why text-level substitution. The nested id sits inside an
# `arguments` STRING (per the OpenAI tool-call wire shape), which in
# SSE streaming responses is built up across many delta chunks.
# Doing structural JSON walk per-chunk would require accumulating
# arguments state across the chunk stream. Text-level regex over
# the full response body is shorter, idempotent on already-canonical
# emissions, and works identically on JSON-body and SSE chunk shapes
# because the JSON-escaped form is the same in both. The regex only
# matches our specific ManyForge id shapes so false-positive risk
# (rewriting an unrelated `"id":...` field that happens to look like
# a ManyForge id) is essentially zero.
_NORMALIZE_NESTED_MCP_IDS = (
    os.environ.get("OPENCLAW_PROXY_NORMALIZE_NESTED_MCP_IDS", "1") or "1"
).strip().lower() in ("1", "true", "yes", "on")

# ManyForge-id surface patterns the nested-MCP-id normalizer matches.
# These are stable: the spec 485 amendment ADR-0020 locked underscore
# as the canonical wire form. Adding a new tool would extend the
# `tree_draft_|scene_draft_|...` alternation. Anything not matching
# stays untouched.
import re as _re

# Matches a nested ManyForge id field that is bare (missing the
# `manyforge__` prefix) but otherwise correctly underscored. We
# prepend `manyforge__` to canonicalize.
_NESTED_BARE_ID_PATTERN = _re.compile(
    r'"id":"(?P<id>'
    r'(?:tree_draft_|scene_draft_|program_draft_|blackboard_draft_)'
    r'[a-z0-9_]+'
    r'|program_read|catalog_read|skills_read|status_read|program_validate'
    r'|scene_inspect|inspect_isaac_scene|deployment_capabilities_read'
    r')"'
)

# Matches dashed forms in the ManyForge surface area. Dashes appear
# in two places: the `<surface>-draft-` separator (e.g. `tree-draft-`)
# and the suffix (e.g. `wrap-node` instead of `wrap_node`). Both
# need conversion to underscores. The optional `manyforge__` prefix
# is preserved if present.
#
# 2026-06-04: extended to cover MIXED-separator forms observed in the
# wild on Qwen3.6-35B-A3B-NVFP4: `manyforge__tree-draft_wrap_node`
# (first separator is dash, second is underscore) and the symmetric
# `tree_draft-X` form. The pattern now requires at least one dash to
# appear in the `<surface>(_|-)draft(_|-)` slot, so the all-underscore
# canonical form is still handled by the bare pattern (no double-firing).
#
# 2026-06-05: extended to cover DASHED FLAT (non-draft) read/inspect tools
# observed on gemma-4-12b: `manyforge__program-read`, `scene-inspect`, etc.
# The bare pattern already handles their underscore form; this branch adds
# the dashed form (each contains a dash, so no double-firing with bare).
_NESTED_DASHED_ID_PATTERN = _re.compile(
    r'"id":"(?P<prefix>manyforge__)?(?P<id>'
    # both separators dash, or first dash + second underscore
    r'(?:tree|scene|program|blackboard)-draft[-_][a-z0-9_-]+'
    # first underscore + second dash
    r'|(?:tree|scene|program|blackboard)_draft-[a-z0-9_-]+'
    # dashed flat read/inspect tools (no `draft` segment)
    r'|(?:program|catalog|skills|status)-read'
    r'|program-validate|scene-inspect|inspect-isaac-scene'
    r'|deployment-capabilities-read'
    r')"'
)

# Matches an MCP locator prefix `mcp:<server>:` that some models
# emit verbatim from the `tool_search` result's `id` field. We strip
# it to leave the dispatchable `manyforge__<bare>` name.
_NESTED_MCP_LOCATOR_PATTERN = _re.compile(
    r'"id":"mcp:[a-zA-Z0-9_-]+:(?P<id>manyforge__[a-z0-9_]+)"'
)

# Escaped-form variants. In SSE streaming responses, the `arguments`
# field is itself a JSON-encoded string, so its inner `"id":"..."`
# field is wire-encoded as `\"id\":\"...\"` (backslash-escaped
# quotes). The three patterns above only match the plain form;
# these mirror them for the escaped form so the same normalizer
# fires regardless of whether the chunk landed in JSON-body or
# SSE-stream shape. Substitution emits the same escaped wire form
# back so the surrounding JSON stays valid.
_NESTED_BARE_ID_ESCAPED_PATTERN = _re.compile(
    r'\\"id\\":\\"(?P<id>'
    r'(?:tree_draft_|scene_draft_|program_draft_|blackboard_draft_)'
    r'[a-z0-9_]+'
    r'|program_read|catalog_read|skills_read|status_read|program_validate'
    r'|scene_inspect|inspect_isaac_scene|deployment_capabilities_read'
    r')\\"'
)

_NESTED_DASHED_ID_ESCAPED_PATTERN = _re.compile(
    r'\\"id\\":\\"(?P<prefix>manyforge__)?(?P<id>'
    # both separators dash, or first dash + second underscore
    r'(?:tree|scene|program|blackboard)-draft[-_][a-z0-9_-]+'
    # first underscore + second dash
    r'|(?:tree|scene|program|blackboard)_draft-[a-z0-9_-]+'
    # dashed flat read/inspect tools (no `draft` segment)
    r'|(?:program|catalog|skills|status)-read'
    r'|program-validate|scene-inspect|inspect-isaac-scene'
    r'|deployment-capabilities-read'
    r')\\"'
)

_NESTED_MCP_LOCATOR_ESCAPED_PATTERN = _re.compile(
    r'\\"id\\":\\"mcp:[a-zA-Z0-9_-]+:(?P<id>manyforge__[a-z0-9_]+)\\"'
)


def _normalize_nested_mcp_ids_in_text(txt: str) -> tuple[str, list[dict]]:
    """Apply nested-MCP-id rewrites to a response body string. Works on
    both JSON and SSE shapes because JSON-escaped id strings are
    identical in both. Returns (new_txt, rewrites_list). The
    rewrites_list is a list of {original, rewritten} pairs for the
    mutation-summary log; empty if no change.

    Idempotent: already-canonical `"id":"manyforge__<x>"` strings do
    not match any pattern and pass through untouched.
    """
    if not _NORMALIZE_NESTED_MCP_IDS or not txt:
        return txt, []
    rewrites: list[dict] = []

    def _sub_mcp_locator(m: _re.Match) -> str:
        canonical = m.group("id")
        original = m.group(0)
        new_form = f'"id":"{canonical}"'
        if new_form != original:
            rewrites.append({
                "rule": "strip_mcp_locator",
                "original": original,
                "rewritten": new_form,
            })
        return new_form

    def _sub_dashed(m: _re.Match) -> str:
        prefix = m.group("prefix") or ""
        # Convert all dashes in the id body to underscores.
        bare = m.group("id").replace("-", "_")
        # Always prepend `manyforge__` — if it was already there, the
        # regex capture groups preserve it; if not, add it.
        if not prefix:
            prefix = "manyforge__"
        new_form = f'"id":"{prefix}{bare}"'
        original = m.group(0)
        if new_form != original:
            rewrites.append({
                "rule": "dash_to_underscore" if "-" in m.group("id") else "add_prefix",
                "original": original,
                "rewritten": new_form,
            })
        return new_form

    def _sub_bare(m: _re.Match) -> str:
        bare = m.group("id")
        new_form = f'"id":"manyforge__{bare}"'
        original = m.group(0)
        rewrites.append({
            "rule": "add_manyforge_prefix",
            "original": original,
            "rewritten": new_form,
        })
        return new_form

    def _sub_mcp_locator_escaped(m: _re.Match) -> str:
        canonical = m.group("id")
        original = m.group(0)
        new_form = f'\\"id\\":\\"{canonical}\\"'
        if new_form != original:
            rewrites.append({
                "rule": "strip_mcp_locator_escaped",
                "original": original,
                "rewritten": new_form,
            })
        return new_form

    def _sub_dashed_escaped(m: _re.Match) -> str:
        prefix = m.group("prefix") or ""
        bare = m.group("id").replace("-", "_")
        if not prefix:
            prefix = "manyforge__"
        new_form = f'\\"id\\":\\"{prefix}{bare}\\"'
        original = m.group(0)
        if new_form != original:
            rewrites.append({
                "rule": "dash_to_underscore_escaped" if "-" in m.group("id") else "add_prefix_escaped",
                "original": original,
                "rewritten": new_form,
            })
        return new_form

    def _sub_bare_escaped(m: _re.Match) -> str:
        bare = m.group("id")
        new_form = f'\\"id\\":\\"manyforge__{bare}\\"'
        original = m.group(0)
        rewrites.append({
            "rule": "add_manyforge_prefix_escaped",
            "original": original,
            "rewritten": new_form,
        })
        return new_form

    # Order matters: process MCP-locator first (most specific), then
    # dashed (catches both prefixed and unprefixed dashed forms),
    # then bare (catches anything without prefix that survived).
    # Plain form first, then escaped form for SSE streaming chunks.
    new_txt = _NESTED_MCP_LOCATOR_PATTERN.sub(_sub_mcp_locator, txt)
    new_txt = _NESTED_DASHED_ID_PATTERN.sub(_sub_dashed, new_txt)
    new_txt = _NESTED_BARE_ID_PATTERN.sub(_sub_bare, new_txt)
    new_txt = _NESTED_MCP_LOCATOR_ESCAPED_PATTERN.sub(_sub_mcp_locator_escaped, new_txt)
    new_txt = _NESTED_DASHED_ID_ESCAPED_PATTERN.sub(_sub_dashed_escaped, new_txt)
    new_txt = _NESTED_BARE_ID_ESCAPED_PATTERN.sub(_sub_bare_escaped, new_txt)
    return new_txt, rewrites


# 2026-06-05: streaming-aware nested-id normalizer.
#
# Why the text-regex above is not enough. OpenClaw streams chat completions
# as SSE: `tool_calls[*].function.arguments` is emitted ONE TOKEN PER
# `data:` event (verified empirically — a 38-char id like
# `manyforge__scene_draft_add_object` arrives as ~10+ separate chunks,
# longest fragment ~33 chars, with full chunk boilerplate
# `"}}]}}],"created":…\n\ndata: {…"arguments":"` between every token). So
# the complete `"id":"…"` substring exists NOWHERE in the response body and
# `_normalize_nested_mcp_ids_in_text` matches nothing (0 rewrites across a
# whole PnP run on gemma-4-12b — see docs/smoke-evidence/.../
# sse-fragmentation-proof.md).
#
# Since the proxy already BUFFERS the full response (see the handler), we can
# do a structural pass: parse every `data:` chunk, reassemble each
# (choice_index, tool_call_index) `function.arguments` fragment stream into
# the complete arguments string, canonicalize the nested ManyForge id in that
# now-contiguous string by REUSING `_normalize_nested_mcp_ids_in_text`, and —
# if it changed — write the corrected full arguments into that tool call's
# FIRST arguments-bearing chunk and blank the rest. OpenClaw concatenates the
# fragments, so first=full + rest="" yields the canonical assembled id while
# keeping every other field and chunk boundary byte-identical.
def _normalize_nested_mcp_ids_streaming(txt: str) -> tuple[str, list[dict]]:
    """SSE-aware variant of the nested-MCP-id normalizer. Returns
    (new_txt, rewrites). For non-SSE bodies (no `data:` events) returns the
    input unchanged with no rewrites — callers use the text-regex variant for
    the plain-JSON shape. Any structural surprise falls back to no-op so a
    parse hiccup never corrupts the stream."""
    if not _NORMALIZE_NESTED_MCP_IDS or not txt or "data:" not in txt:
        return txt, []
    try:
        lines = txt.split("\n")
        # Collect parsed `data:` chunk objects with their line index.
        chunks: list[tuple[int, dict]] = []
        for i, line in enumerate(lines):
            s = line.strip()
            if not s.startswith("data:"):
                continue
            payload = s[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            chunks.append((i, obj))
        if not chunks:
            return txt, []

        # Accumulate arguments fragments per (choice_index, tool_call_index).
        # locs records (chunk_pos, choice_index, tc_list_pos) so we can write
        # the corrected value back into the exact chunk it came from.
        acc: dict[tuple[int, int], dict] = {}
        for cpos, (_li, obj) in enumerate(chunks):
            for choice in obj.get("choices", []) or []:
                ci = choice.get("index", 0)
                delta = choice.get("delta") or {}
                for tpos, tc in enumerate(delta.get("tool_calls", []) or []):
                    if not isinstance(tc, dict):
                        continue
                    ti = tc.get("index", tpos)
                    fn = tc.get("function") or {}
                    if "arguments" not in fn:
                        continue
                    rec = acc.setdefault((ci, ti), {"frags": [], "locs": []})
                    rec["frags"].append(fn.get("arguments") or "")
                    rec["locs"].append((cpos, ci, tpos))
        if not acc:
            return txt, []

        all_rewrites: list[dict] = []
        changed_chunks: set[int] = set()
        for (_ci, _ti), rec in acc.items():
            assembled = "".join(rec["frags"])
            if not assembled or '"id"' not in assembled:
                continue
            new_assembled, rw = _normalize_nested_mcp_ids_in_text(assembled)
            if not rw or new_assembled == assembled:
                continue
            all_rewrites.extend(rw)
            for n, (cpos, ci, tpos) in enumerate(rec["locs"]):
                _li, obj = chunks[cpos]
                for choice in obj.get("choices", []) or []:
                    if choice.get("index", 0) != ci:
                        continue
                    tcs = (choice.get("delta") or {}).get("tool_calls") or []
                    if 0 <= tpos < len(tcs) and isinstance(tcs[tpos], dict) \
                            and "function" in tcs[tpos]:
                        tcs[tpos]["function"]["arguments"] = (
                            new_assembled if n == 0 else ""
                        )
                changed_chunks.add(cpos)

        if not all_rewrites:
            return txt, []
        for cpos in changed_chunks:
            li, obj = chunks[cpos]
            lines[li] = "data: " + json.dumps(obj, separators=(",", ":"))
        return "\n".join(lines), all_rewrites
    except Exception:
        # Never let a normalization bug corrupt the response stream.
        return txt, []

# 2026-06-02: response-side reasoning→content promotion.
# When vLLM is launched with --reasoning-parser (e.g. qwen3 for cosmos),
# the parser routes everything inside <think>...</think> blocks to
# the `reasoning` field and leaves `content` empty. OpenClaw 2026.5.22
# treats this as an incomplete terminal response ("code=incomplete_result")
# and refuses to surface it back to the bridge.
#
# Older OpenClaw (2026.4.24, what was current during the iter-32 win
# of 77.3% on cosmos) accepted reasoning-only responses. The newer
# stricter contract breaks every model that emits to reasoning.
#
# When this flag is on, the proxy rewrites streaming SSE chunks and
# non-streaming JSON responses so that `reasoning` content is mirrored
# into `content`. Tool_calls and other fields pass through untouched.
# This preserves both bridge correctness (sees content) and any
# reasoning-only consumer downstream (the reasoning field stays
# populated alongside).
_PROMOTE_REASONING_TO_CONTENT = (
    os.environ.get("OPENCLAW_PROXY_PROMOTE_REASONING_TO_CONTENT", "1") or "1"
).strip().lower() in ("1", "true", "yes", "on")

# 2026-06-01: prompt-aware guided tool-call decoding (request-side).
# When the LAST user message looks action-shaped (action verb + domain
# noun), force `tool_choice="required"` so vLLM's tool-calling backend
# uses guided decoding against the provided tools[] schemas. Eliminates
# the "wrong-arg-shape" class of failure (missing required field, wrong
# type, unknown enum) AND forces tool emission instead of prose on
# unambiguous action prompts. Generic across models + parsers — vLLM
# handles schema enforcement internally.
# Set OPENCLAW_PROXY_GUIDED_TOOL_CALLS=0 to disable. The action-shape
# heuristic mirrors the bridge's `is_action_shaped_prompt()`.
#
# NOTE 2026-06-01 second pass: Level 1 (tool_choice=required) interacts
# poorly with cosmos's thinking-on hermes parser — model produces empty
# output. Default is now 0; use only on profiles known to be compatible
# (typically non-thinking or qwen3_coder parser).
_GUIDED_TOOL_CALLS = (
    os.environ.get("OPENCLAW_PROXY_GUIDED_TOOL_CALLS", "0") or "0"
).strip().lower() in ("1", "true", "yes", "on")

# 2026-06-01: structural_tag injection for tool schemas (Level 2 guided).
# When parser is known, build a per-tool `structural_tag` config so the
# model can decide ask-vs-act freely, but if it DOES emit `<tool_call>`
# vLLM constrains the JSON inside to match one of the tool schemas.
# Eliminates the "args wrong shape" class without forcing tool emission.
#
# Parser must be specified via OPENCLAW_PROXY_TOOL_PARSER env var. Set
# per-profile in config.sh — cosmos=hermes, 4B/omni/35B=qwen3_coder.
# Disabled (empty) by default — opt-in.
_TOOL_PARSER = (os.environ.get("OPENCLAW_PROXY_TOOL_PARSER", "") or "").strip()


def _lark_escape(s: str) -> str:
    """Escape a string for use as a Lark terminal literal.
    Lark literals are double-quoted; embedded backslashes and quotes
    need escaping. Newlines in qwen3_coder format are written as \\n."""
    return s.replace("\\", "\\\\").replace("\"", "\\\"")


def _build_qwen3_coder_grammar(tools_list: list) -> str | None:
    """Build a Lark EBNF grammar that constrains the model to emit a
    qwen3_coder-format tool call:

        <tool_call>\\n<function=NAME>\\n<parameter=K>\\nVAL\\n</parameter>\\n...</function>\\n</tool_call>

    Each tool gets its own rule with the function name baked in and
    the parameter-key alternation restricted to that tool's own
    parameters. Values are unconstrained (any non-empty text up to
    the next \\n</parameter> closing) — full JSON-type enforcement on
    values would require nested grammar; the structure-level
    enforcement here already eliminates the wrong-tool-name and
    invalid-param-key failure classes.
    """
    if not isinstance(tools_list, list) or not tools_list:
        return None
    tool_rules: list[str] = []   # named per-tool rules
    tool_alts: list[str] = []    # names to alternate at the top
    for i, t in enumerate(tools_list):
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        name = fn.get("name") or t.get("name")
        if not isinstance(name, str) or not name:
            continue
        params_schema = fn.get("parameters")
        props = {}
        if isinstance(params_schema, dict):
            p = params_schema.get("properties")
            if isinstance(p, dict):
                props = p
        rule = f"tool_{i}"
        tool_alts.append(rule)
        param_keys = [k for k in props.keys() if isinstance(k, str)]
        if param_keys:
            keys_alt = " | ".join(f'"{_lark_escape(k)}"' for k in param_keys)
            tool_rules.append(
                f'{rule}: "<function={_lark_escape(name)}>" NL '
                f'param_{i}* "</function>"'
            )
            tool_rules.append(
                f'param_{i}: "<parameter=" ({keys_alt}) ">" NL '
                f'PARAM_VALUE NL "</parameter>" NL'
            )
        else:
            # tool with no params: emit just the function open/close
            tool_rules.append(
                f'{rule}: "<function={_lark_escape(name)}>" NL "</function>"'
            )
    if not tool_alts:
        return None
    alt = " | ".join(tool_alts)
    # Use a named NL terminal driven by a regex literal — vLLM/Lark
    # interprets `\n` inside string literals as a real newline char
    # before the grammar parser sees it, which breaks multi-line
    # rules. Using `NL: /\n/` keeps the newline as a regex token.
    grammar = (
        f'start: "<tool_call>" NL ({alt}) NL "</tool_call>"\n'
        + "\n".join(tool_rules)
        + '\n'
        + 'NL: /\\n/\n'
        # PARAM_VALUE: any non-`<` character, one or more. vLLM/Lark
        # regex backend doesn't support lookahead, so we can't accept
        # `<` inside values. Trade-off: param values containing a
        # literal `<` (e.g. a description string with HTML-ish text)
        # won't match and the constraint will fail. Acceptable for
        # the manyforge tool surface where values are typically JSON
        # primitives, IDs, or short text. Counter-cases would retry.
        + 'PARAM_VALUE: /[^<]+/\n'
    )
    return grammar


def _build_structural_tag(tools_list: list, parser: str) -> dict | None:
    """Build a vLLM structural_tag config for hermes parsers (JSON-in-tags).

    Returns the dict with `type:"structural_tag", structures:[...], triggers:[...]`
    or None if parser is not hermes or tools_list is empty.

    For qwen3_coder, use `_build_qwen3_coder_grammar` instead (returns a
    Lark grammar string for the `grammar` field).
    """
    if parser != "hermes":
        return None
    if not isinstance(tools_list, list) or not tools_list:
        return None
    structures: list[dict] = []
    for t in tools_list:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        name = fn.get("name") or t.get("name")
        if not isinstance(name, str) or not name:
            continue
        params = fn.get("parameters")
        if not isinstance(params, dict):
            params = {"type": "object"}
        structures.append({
            "begin": "<tool_call>\n",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"const": name},
                    "arguments": params,
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            },
            "end": "\n</tool_call>",
        })
    if not structures:
        return None
    return {
        "type": "structural_tag",
        "structures": structures,
        "triggers": ["<tool_call>"],
    }

_ACTION_VERBS = (
    "add ", "remove ", "delete ", "insert ", "update ", "wrap ", "swap ",
    "move ", "create ", "make ", "place ", "attach ", "detach ", "set ",
    "replace ", "rename ", "reorder ", "drop ", "modify ", "resize ",
    "change ",
)
_DOMAIN_NOUNS = (
    "node", "tree", "root", "sequence", "fallback", "parallel", "repeat",
    "retry", "inverter", "object", "obstacle", "box", "sphere", "cylinder",
    "ground", "collider", "graspable", "scene", "program", "parameter",
    "param", "blackboard",
)

# 2026-06-01: 4xx tool-result directive rewrite (request-side mutation).
# When the model's most recent tool message contains a validation
# failure with structured guidance fields (validParentNames,
# allowedNodeKinds, validNodeNames, validTargetNames, etc.), prepend
# a directive preamble that names the failing field + valid values
# so the model can't miss it. Model-agnostic — depends only on
# composer's validator returning the standard field names.
_TOOL_ERROR_REWRITE = (
    os.environ.get("OPENCLAW_PROXY_TOOL_ERROR_REWRITE", "1") or "1"
).strip().lower() in ("1", "true", "yes", "on")

# 2026-06-02: hermes wrapper unwrap for assistant.tool_calls[*].arguments.
# vLLM's hermes tool-call parser can fail to strip the
# `<tool_call>{...}</tool_call>` outer wrapper when extracting structured
# tool calls from cosmos's reasoning-on output. The resulting
# `arguments` string LOOKS like
#   "<tool_call>\n{\"name\":\"tree_draft_wrap_node\",\"arguments\":{...}}"
# instead of just the inner arguments object. On the NEXT turn, when the
# bridge feeds the conversation back to vLLM, the chat template tries
# `json.loads(arguments)` and 400s with
#   "Expecting value: line 1 column 1 (char 0)".
# This mutator opportunistically detects that pattern and replaces
# the wrapper-wrapped string with just the inner `arguments` object as
# a JSON string. No-op on cleanly-parsed args (pass-through), no-op on
# args that aren't strings (vLLM also accepts dict args). Default on;
# set =0 to disable per-profile if it ever bites a model that emits
# legitimate `<tool_call>` content for some other reason.
_UNWRAP_TOOL_CALL_ARGS = (
    os.environ.get("OPENCLAW_PROXY_UNWRAP_TOOL_CALL_ARGS", "1") or "1"
).strip().lower() in ("1", "true", "yes", "on")

_VALIDATOR_FIELDS = (
    "validParentNames", "validNodeNames", "validTargetNames",
    "allowedNodeKinds", "rejectedNodeKinds", "wrapperIdSuggestions",
    "validKindIds", "allowedKindIds",
)
_TOOL_ERROR_MARKERS = (
    "validation_error", "validation failed", "validation error",
    "\"status\": 4", "\"status\":4",
    "\"http_status\": 4", "\"httpStatus\": 4",  # snake AND camel
    "\"http_status\":4", "\"httpStatus\":4",
    "\"success\": false", "\"success\":false",
    "missing required parameter", "missing required",
    "\"error\":", "BadRequest", "UnprocessableEntity",
)


def _env_truthy(name: str, default: str = "1") -> bool:
    return (os.environ.get(name, default) or default).strip().lower() in ("1", "true", "yes", "on")


_TRIGGER_SAME_TOOL = _env_truthy("OPENCLAW_PROXY_LOOP_TRIGGER_SAME_TOOL")
_TRIGGER_SAME_ARGS = _env_truthy("OPENCLAW_PROXY_LOOP_TRIGGER_SAME_ARGS")
_TRIGGER_RESULT_REPEAT = _env_truthy("OPENCLAW_PROXY_LOOP_TRIGGER_RESULT_REPEAT")
_TRIGGER_NAMESPACE = _env_truthy("OPENCLAW_PROXY_LOOP_TRIGGER_NAMESPACE")
_TRIGGER_TURN_COUNTER = _env_truthy("OPENCLAW_PROXY_LOOP_TRIGGER_TURN_COUNTER")

_REFLECTION_MARKER = "[loop-reflection]"  # marker so we never inject twice
_GENERIC_REFLECTION_MSG = (
    f"{_REFLECTION_MARKER} You may be repeating yourself or stuck on this approach. "
    f"Stop and consider: "
    f"(a) a DIFFERENT tool, "
    f"(b) DIFFERENT arguments to the same tool, or "
    f"(c) ask the user one specific clarifying question if you need missing information. "
    f"Do not repeat the same failing call."
)


def _tool_name_from_assistant(m: dict) -> str | None:
    tcs = m.get("tool_calls") or []
    for tc in tcs:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            nm = fn.get("name") or tc.get("name")
            if nm:
                return nm
    return None


def _tool_args_from_assistant(m: dict) -> str | None:
    """Canonical-JSON string form of the first tool call's arguments.
    Falls back to repr on non-JSON. False-negative-tolerant by design."""
    tcs = m.get("tool_calls") or []
    for tc in tcs:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if args is None:
                args = tc.get("arguments")
            if isinstance(args, str):
                try:
                    return json.dumps(json.loads(args), sort_keys=True, separators=(",", ":"))
                except (ValueError, TypeError):
                    return args
            elif args is not None:
                try:
                    return json.dumps(args, sort_keys=True, separators=(",", ":"))
                except (TypeError, ValueError):
                    return repr(args)
    return None


def _namespace_of(tool_name: str) -> str:
    """Coarse namespace = first two `_`-separated segments
    (tree_draft_insert_node → tree_draft, scene_draft_add_object → scene_draft)."""
    parts = tool_name.split("_", 2)
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]


def _tool_names_from_body(body: object) -> list[str]:
    """Extract every tool name advertised in a chat-completion request body.
    Handles OpenAI-style `[{type:"function", function:{name:...}}]` and
    the flat-name fallback some clients use.

    Returns the full names PLUS prefix-stripped variants. MCP-namespaced
    tools come through as `manyforge__tree_draft_wrap_node` but models
    routinely emit just the suffix `tree_draft_wrap_node` (a known
    observed failure mode 2026-06-01). Including both forms keeps the
    detector's substring check robust to that drift."""
    if not isinstance(body, dict):
        return []
    tools = body.get("tools")
    if not isinstance(tools, list):
        return []
    out: list[str] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        n = fn.get("name") or t.get("name")
        if isinstance(n, str) and n:
            out.append(n)
            if "__" in n:
                suffix = n.split("__", 1)[1]
                if suffix and suffix not in out:
                    out.append(suffix)
    return out


def _response_content_and_tool_calls(resp_body: bytes | str) -> tuple[str, bool]:
    """Pull (concatenated content text, did_emit_tool_calls) from a
    chat-completion response. Handles both JSON (non-streaming) and SSE
    (streaming) shapes. Returns ("", False) on any parse failure — the
    detector treats that as "nothing to inspect"."""
    if not resp_body:
        return "", False
    if isinstance(resp_body, bytes):
        try:
            txt = resp_body.decode("utf-8", errors="replace")
        except Exception:
            return "", False
    else:
        txt = resp_body
    # First try JSON (non-streaming)
    try:
        data = json.loads(txt)
        if isinstance(data, dict):
            content_parts: list[str] = []
            tool_calls_seen = False
            for ch in (data.get("choices") or []):
                if not isinstance(ch, dict):
                    continue
                msg = ch.get("message") or {}
                c = msg.get("content")
                if isinstance(c, str):
                    content_parts.append(c)
                if msg.get("tool_calls"):
                    tool_calls_seen = True
            return "".join(content_parts), tool_calls_seen
    except (ValueError, TypeError):
        pass
    # Fall back to SSE
    import re as _re
    content_parts: list[str] = []
    tool_calls_seen = False
    for m in _re.finditer(r'data: (\{.*?\})\n\n', txt + '\n\n'):
        try:
            c = json.loads(m.group(1))
        except (ValueError, TypeError):
            continue
        ch = (c.get("choices") or [{}])[0]
        if not isinstance(ch, dict):
            continue
        d = ch.get("delta") or {}
        if isinstance(d.get("content"), str):
            content_parts.append(d["content"])
        if d.get("tool_calls"):
            tool_calls_seen = True
    return "".join(content_parts), tool_calls_seen


def _build_canonical_tool_map(req_body: object) -> dict[str, str]:
    """Map bare suffix → unique canonical (full prefixed) tool name.
    Only includes entries with EXACTLY ONE catalog match; ambiguous
    suffixes are dropped (rewrite would be unsafe). Full names map to
    themselves so already-correct emissions pass through trivially."""
    if not isinstance(req_body, dict):
        return {}
    tools = req_body.get("tools")
    if not isinstance(tools, list):
        return {}
    full_names: list[str] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        n = fn.get("name") or t.get("name")
        if isinstance(n, str) and n:
            full_names.append(n)
    canonical: dict[str, str | None] = {n: n for n in full_names}
    for full in full_names:
        if "__" in full:
            suffix = full.split("__", 1)[1]
            if not suffix:
                continue
            if suffix in canonical and canonical[suffix] != full:
                canonical[suffix] = None  # ambiguous; mark for removal
            elif suffix not in canonical:
                canonical[suffix] = full
    return {k: v for k, v in canonical.items() if v is not None}


def _promote_reasoning_to_content_in_response(
    resp_body: bytes | str,
) -> tuple[bytes, int]:
    """Mirror `reasoning` field content into `content` for OpenClaw 2026.5.22
    compatibility (it rejects content==null as 'incomplete_result' even
    when reasoning is populated).

    Handles:
      - Non-streaming `chat.completion` JSON: copies choices[i].message.reasoning
        into choices[i].message.content if content is None / "".
      - Streaming `chat.completion.chunk` SSE: copies delta.reasoning into
        delta.content if content is None / absent.

    Returns (new_body_bytes, mutations_count). When count==0, the body is
    returned unchanged.
    """
    if not _PROMOTE_REASONING_TO_CONTENT:
        return (resp_body if isinstance(resp_body, bytes)
                else (resp_body or "").encode("utf-8")), 0
    txt = resp_body.decode("utf-8", "replace") if isinstance(resp_body, bytes) else (resp_body or "")
    if not txt:
        return b"", 0

    mutations = 0

    # Non-streaming JSON shape: a single top-level object with `choices`.
    if txt.lstrip().startswith("{") and "data: " not in txt:
        try:
            data = json.loads(txt)
        except (ValueError, TypeError):
            return txt.encode("utf-8"), 0
        if isinstance(data, dict) and isinstance(data.get("choices"), list):
            for ch in data["choices"]:
                if not isinstance(ch, dict):
                    continue
                msg = ch.get("message")
                if isinstance(msg, dict):
                    reasoning = msg.get("reasoning")
                    content = msg.get("content")
                    if reasoning and (content is None or content == ""):
                        msg["content"] = reasoning
                        mutations += 1
            if mutations:
                return json.dumps(data).encode("utf-8"), mutations
        return txt.encode("utf-8"), 0

    # SSE streaming shape: `data: {...}\n\n` chunks.
    new_lines: list[str] = []
    for line in txt.split("\n"):
        if not line.startswith("data: "):
            new_lines.append(line)
            continue
        payload = line[6:]
        if payload == "[DONE]" or not payload.strip().startswith("{"):
            new_lines.append(line)
            continue
        try:
            chunk = json.loads(payload)
        except (ValueError, TypeError):
            new_lines.append(line)
            continue
        modified = False
        for ch in (chunk.get("choices") or []):
            if not isinstance(ch, dict):
                continue
            d = ch.get("delta")
            if isinstance(d, dict):
                reasoning = d.get("reasoning")
                content = d.get("content")
                if reasoning and (content is None or content == ""):
                    d["content"] = reasoning
                    modified = True
                    mutations += 1
            # Non-streaming-mixed: some servers also place a full message
            # on a final chunk.
            msg = ch.get("message")
            if isinstance(msg, dict):
                reasoning = msg.get("reasoning")
                content = msg.get("content")
                if reasoning and (content is None or content == ""):
                    msg["content"] = reasoning
                    modified = True
                    mutations += 1
        if modified:
            new_lines.append("data: " + json.dumps(chunk, separators=(",", ":")))
        else:
            new_lines.append(line)
    return "\n".join(new_lines).encode("utf-8"), mutations


def _normalize_tool_names_in_response(
    req_body: object, resp_body: bytes | str,
) -> tuple[bytes, list[dict]]:
    """Rewrite tool_call.name in a chat-completion response when the
    emitted name maps to a unique catalog entry under the bare-suffix
    rule. Handles both JSON and SSE response shapes. Returns
    (new_body_bytes, rewrites_list). rewrites_list is empty if no
    change was made; in that case new_body_bytes equals the input."""
    if not _NORMALIZE_TOOL_NAMES:
        return (resp_body if isinstance(resp_body, bytes)
                else (resp_body or "").encode("utf-8")), []
    canonical_map = _build_canonical_tool_map(req_body)
    if not canonical_map:
        return (resp_body if isinstance(resp_body, bytes)
                else (resp_body or "").encode("utf-8")), []
    if isinstance(resp_body, bytes):
        try:
            txt = resp_body.decode("utf-8")
        except UnicodeDecodeError:
            return resp_body, []
    else:
        txt = resp_body or ""
    if not txt:
        return b"", []
    rewrites: list[dict] = []

    def _rewrite_tc(tc_obj: dict) -> bool:
        fn = tc_obj.get("function") if isinstance(tc_obj, dict) else None
        if not isinstance(fn, dict):
            return False
        original = fn.get("name")
        if not isinstance(original, str) or not original:
            return False
        canonical = canonical_map.get(original)
        if canonical is None or canonical == original:
            return False
        fn["name"] = canonical
        rewrites.append({"original": original, "rewritten": canonical})
        return True

    # JSON shape first (non-streaming responses)
    if txt.lstrip().startswith("{"):
        try:
            data = json.loads(txt)
        except (ValueError, TypeError):
            data = None
        if isinstance(data, dict):
            modified = False
            for ch in (data.get("choices") or []):
                if not isinstance(ch, dict):
                    continue
                msg = ch.get("message") or {}
                for tc in (msg.get("tool_calls") or []):
                    if _rewrite_tc(tc):
                        modified = True
            if modified:
                return json.dumps(data).encode("utf-8"), rewrites
            return txt.encode("utf-8"), []

    # SSE shape: rewrite one `data: {...}` chunk at a time.
    new_lines: list[str] = []
    for line in txt.split("\n"):
        if not line.startswith("data: "):
            new_lines.append(line)
            continue
        payload = line[6:]
        if payload == "[DONE]" or not payload.strip().startswith("{"):
            new_lines.append(line)
            continue
        try:
            chunk = json.loads(payload)
        except (ValueError, TypeError):
            new_lines.append(line)
            continue
        modified = False
        for ch in (chunk.get("choices") or []):
            if not isinstance(ch, dict):
                continue
            d = ch.get("delta") or {}
            for tc in (d.get("tool_calls") or []):
                if _rewrite_tc(tc):
                    modified = True
            # Some clients also send tool_calls on the non-delta message
            msg = ch.get("message") or {}
            for tc in (msg.get("tool_calls") or []):
                if _rewrite_tc(tc):
                    modified = True
        if modified:
            new_lines.append("data: " + json.dumps(chunk, separators=(",", ":")))
        else:
            new_lines.append(line)
    return "\n".join(new_lines).encode("utf-8"), rewrites


def _detect_malformed_tool_call(
    req_body: object, resp_body: bytes | str,
) -> dict | None:
    """Generic, model-agnostic detector: response content names a tool
    AND contains tool-call format markers, but NO structured tool_calls
    came through. Returns a dict on hit, None otherwise."""
    if not _DETECT_MALFORMED_TOOL_CALL:
        return None
    tool_names = _tool_names_from_body(req_body)
    if not tool_names:
        return None
    content, has_tcs = _response_content_and_tool_calls(resp_body)
    if has_tcs or not content:
        return None
    matched_tool = next((n for n in tool_names if n in content), None)
    if matched_tool is None:
        return None
    matched_markers = [m for m in _MALFORMED_MARKERS if m in content]
    if not matched_markers:
        return None
    return {
        "matched_tool": matched_tool,
        "markers": matched_markers,
        "content_preview": content[:300].replace("\n", "\\n"),
        "content_chars": len(content),
    }


def _check_loop_short_circuit(
    path: str, body: bytes,
) -> tuple[bytes | None, bytes | None, dict | None]:
    """Inspect /v1/chat/completions for runaway loops via 5 detectors.

    Returns (error_json, mutated_body, trigger_info):
      - all None: forward as-is
      - error_json set: hard-stop with an OpenAI-style error body
        (no further forward)
      - mutated_body set: forward the mutated body (reflection injected)
      - trigger_info {trigger, tool, count, ...}: appended to JSONL log

    Each detector is independently env-gated. First-to-fire wins.
    Inner-loop only: a non-tool assistant turn resets all consecutive counters.
    """
    any_trigger_enabled = (
        _TRIGGER_SAME_TOOL or _TRIGGER_SAME_ARGS or _TRIGGER_RESULT_REPEAT
        or _TRIGGER_NAMESPACE or _TRIGGER_TURN_COUNTER
    )
    if _STOP_AT <= 0 and _REFLECT_AT <= 0 and not any_trigger_enabled:
        return None, None, None
    if not path.endswith("/v1/chat/completions"):
        return None, None, None
    try:
        parsed = json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError):
        return None, None, None
    if not isinstance(parsed, dict):
        return None, None, None
    msgs = parsed.get("messages") or []
    if not isinstance(msgs, list):
        return None, None, None

    # Build TAIL run of consecutive assistant-with-tool-call turns.
    # A non-tool assistant turn resets the run. This is the "inner loop"
    # the detectors all reason over.
    consecutive_tail: list[dict] = []
    for m in msgs:
        if not isinstance(m, dict): continue
        if m.get("role") != "assistant": continue
        nm = _tool_name_from_assistant(m)
        if nm is None:
            consecutive_tail = []  # reset on no-tool assistant turn
            continue
        consecutive_tail.append({
            "name": nm,
            "args": _tool_args_from_assistant(m),
        })

    # Tool-result history (full order; we only need the last 2)
    tool_results: list[str] = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "tool":
            c = m.get("content")
            if isinstance(c, str):
                tool_results.append(c)

    # Marker check: never inject twice in the same conversation
    already_injected = any(
        isinstance(m, dict) and m.get("role") == "user"
        and isinstance(m.get("content"), str) and _REFLECTION_MARKER in m["content"]
        for m in msgs
    )

    if not consecutive_tail:
        return None, None, None

    # Same-tool run length at the tail (used by both hard-stop and detector 1)
    same_tool_run = 1
    for i in range(len(consecutive_tail) - 1, 0, -1):
        if consecutive_tail[i]["name"] == consecutive_tail[i - 1]["name"]:
            same_tool_run += 1
        else:
            break
    last_name = consecutive_tail[-1]["name"]

    def _loop_error_body(*, trigger: str, tool: str, count: int, detail: str,
                         namespace: str | None = None) -> bytes:
        error: dict = {
            "message": detail,
            "type": "manyforge_proxy_loop_detected",
            "param": None,
            "code": "loop_detected",
            "loopGuard": {
                "trigger": trigger,
                "tool": tool,
                "count": count,
            },
        }
        if namespace:
            error["loopGuard"]["namespace"] = namespace
        return json.dumps({"error": error}, separators=(",", ":")).encode("utf-8")

    # ----- HARD STOP at _STOP_AT (existing safety; not gated by trigger flags) -----
    if _STOP_AT > 0 and same_tool_run >= _STOP_AT:
        detail = (
            f"loop detected: tool '{last_name}' repeated {same_tool_run} "
            "times in this conversation without recovery"
        )
        return _loop_error_body(
            trigger="hard_stop",
            tool=last_name,
            count=same_tool_run,
            detail=detail,
        ), None, {
            "trigger": "hard_stop", "tool": last_name, "count": same_tool_run,
        }

    if already_injected:
        return None, None, None

    fired: dict | None = None

    # Detector 1: SAME TOOL consecutive >= _REFLECT_AT
    if (fired is None and _TRIGGER_SAME_TOOL and _REFLECT_AT > 0
            and same_tool_run >= _REFLECT_AT):
        fired = {"trigger": "same_tool", "tool": last_name, "count": same_tool_run}

    # Detector 2: SAME TOOL + SAME ARGS (last 2 calls identical)
    if (fired is None and _TRIGGER_SAME_ARGS and len(consecutive_tail) >= 2):
        last = consecutive_tail[-1]
        prev = consecutive_tail[-2]
        if (last["name"] == prev["name"]
                and last["args"] is not None
                and last["args"] == prev["args"]):
            fired = {
                "trigger": "same_args", "tool": last_name, "count": 2,
                "args_preview": (last["args"] or "")[:120],
            }

    # Detector 3: RESULT REPEAT (last 2 tool results identical 200-char prefix)
    # Inner-loop guard: require >=2 tool calls in the current run.
    # Otherwise across-case chains (where case A's final result + case B's
    # first result happen to match) could false-fire on case B turn 1.
    if (fired is None and _TRIGGER_RESULT_REPEAT
            and len(tool_results) >= 2 and len(consecutive_tail) >= 2):
        a = (tool_results[-1] or "")[:200]
        b = (tool_results[-2] or "")[:200]
        if a and a == b:
            fired = {
                "trigger": "result_repeat", "tool": last_name, "count": 2,
                "result_preview": a[:120],
            }

    # Detector 4: SAME NAMESPACE consecutive >= _NAMESPACE_AT
    # Also enforces NAMESPACE_STOP_AT (default 16, ~3x NAMESPACE_AT=5) as
    # hard stop — same_tool's STOP_AT doesn't cover the cosmos pattern of
    # alternating tools within the same namespace (e.g. tree_draft_*
    # ping-pong between replace_subtree and wrap_node). Without this,
    # namespace-alternation loops spin until smoke-runner's per-case
    # timeout fires, wasting GPU on a single case.
    if (_TRIGGER_NAMESPACE and _NAMESPACE_AT > 0):
        last_ns = _namespace_of(last_name)
        ns_run = 1
        for i in range(len(consecutive_tail) - 1, 0, -1):
            if (_namespace_of(consecutive_tail[i]["name"])
                    == _namespace_of(consecutive_tail[i - 1]["name"])):
                ns_run += 1
            else:
                break
        # Hard stop first (regardless of already_injected): the SSE response
        # exits the agent loop and prevents unbounded GPU spend, but returns
        # an error-shaped response so benchmark scorers do not count it as
        # a successful assistant answer.
        if _NAMESPACE_STOP_AT > 0 and ns_run >= _NAMESPACE_STOP_AT:
            detail = (
                f"loop detected: {ns_run} consecutive calls within the "
                f"'{last_ns}_*' tool family without convergence"
            )
            return _loop_error_body(
                trigger="namespace_hard_stop",
                tool=last_name,
                count=ns_run,
                detail=detail,
                namespace=last_ns,
            ), None, {
                "trigger": "namespace_hard_stop", "tool": last_name,
                "namespace": last_ns, "count": ns_run,
            }
        if fired is None and ns_run >= _NAMESPACE_AT:
            fired = {
                "trigger": "same_namespace", "tool": last_name,
                "namespace": last_ns, "count": ns_run,
            }

    # Detector 5: TURN COUNTER (any-tool consecutive run >= _TURN_COUNTER_AT)
    if (fired is None and _TRIGGER_TURN_COUNTER and _TURN_COUNTER_AT > 0
            and len(consecutive_tail) >= _TURN_COUNTER_AT):
        fired = {
            "trigger": "turn_counter", "tool": last_name,
            "count": len(consecutive_tail),
        }

    if fired is None:
        return None, None, None

    # Inject the generic reflection as a user message AFTER the last tool result
    new_msgs = list(msgs)
    last_tool_idx = -1
    for i, m in enumerate(new_msgs):
        if isinstance(m, dict) and m.get("role") == "tool":
            last_tool_idx = i
    insert_idx = (last_tool_idx + 1) if last_tool_idx >= 0 else len(new_msgs)
    new_msgs.insert(insert_idx, {"role": "user", "content": _GENERIC_REFLECTION_MSG})
    parsed["messages"] = new_msgs
    mutated = json.dumps(parsed).encode("utf-8")
    return None, mutated, fired


def _try_parse_json(blob: bytes) -> tuple[object | None, str]:
    """Return (parsed, raw_text). parsed is None when not valid JSON."""
    try:
        text = blob.decode("utf-8", errors="replace")
    except Exception:
        text = repr(blob)[:2048]
    try:
        return json.loads(text), text
    except Exception:
        return None, text


def _truncate(text: str, limit: int = 8192) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"…<truncated {len(text) - limit} chars>"


def _message_stats(body_json: object) -> dict:
    if not isinstance(body_json, dict):
        return {"message_count": None, "role_counts": {}}
    messages = body_json.get("messages")
    if not isinstance(messages, list):
        return {"message_count": None, "role_counts": {}}
    role_counts: dict[str, int] = {}
    largest_chars = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        content = message.get("content")
        if isinstance(content, str):
            largest_chars = max(largest_chars, len(content))
        elif content is not None:
            try:
                largest_chars = max(largest_chars, len(json.dumps(content, default=str)))
            except Exception:
                largest_chars = max(largest_chars, len(str(content)))
    return {
        "message_count": len(messages),
        "role_counts": role_counts,
        "largest_message_chars": largest_chars,
    }


def _history_budget_error_body(
    *,
    body_chars: int,
    max_chars: int,
    body_json: object,
) -> bytes:
    stats = _message_stats(body_json)
    return json.dumps(
        {
            "error": {
                "message": (
                    f"history budget exceeded: request body is {body_chars} chars, "
                    f"limit is {max_chars}. Compact, summarize, or start a fresh "
                    "session before retrying."
                ),
                "type": "manyforge_proxy_history_budget_exceeded",
                "param": "messages",
                "code": "history_budget_exceeded",
                "historyBudget": {
                    "bodyChars": body_chars,
                    "maxRequestChars": max_chars,
                    **stats,
                },
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")


# 2026-06-06: history-budget TRIM ladder (fail-open). Replaces the old hard-413
# when a chat request exceeds MANYFORGE_PROXY_MAX_REQUEST_CHARS. The openclaw
# agent loop accumulates large, RE-FETCHABLE tool results in history (a single
# `catalog_read` dump was ~55% of a 199k request) and re-ships them every turn,
# ballooning the prompt. Rejecting the turn kills completable cases (verified:
# guard@200k dropped the PnP block from pass->fail). Instead the proxy SHEDS
# content the model can simply request again, in escalating rungs, then forwards.
#
# Safety contract: only RE-FETCHABLE READ results are elided (the model can
# re-call the tool); mutation/state results are NEVER touched. Message
# envelopes are preserved (role + tool_call_id) — only `content` is replaced
# with a short stub — so OpenAI tool_call<->tool_result pairing stays valid.
#
# Rung 1 (keep-latest read elision) always runs; rungs 2-3 escalate only while
# still over budget. Rung 4 (oldest-turn truncation + composer warning) + the
# manual reset button are Phase B.
_REFETCHABLE_READ_TOOLS = frozenset({
    "catalog_read", "program_read", "status_read", "skills_read",
    "program_validate", "scene_inspect", "inspect_isaac_scene",
    "deployment_capabilities_read",
})
# Don't parse/elide tool results smaller than this — the bloat is the big dumps;
# small reads aren't worth the churn.
_MIN_ELIDABLE_READ_CHARS = 2000
_READ_ELIDED_STUB = (
    '{"elided":true,"reason":"history_budget","note":"re-fetchable read result '
    'removed to fit the history budget; call the tool again if you still need it"}'
)
_REASONING_ELIDED_STUB = "[reasoning elided to fit history budget]"


def _content_len(message: dict) -> int:
    c = message.get("content")
    if isinstance(c, str):
        return len(c)
    if c is None:
        return 0
    try:
        return len(json.dumps(c, default=str))
    except Exception:
        return len(str(c))


def _refetchable_read_kind(message: object) -> str | None:
    """If ``message`` is a tool result produced by a re-fetchable read tool and
    is big enough to be worth eliding, return the bare tool name; else None.
    The producing tool is read from the OpenClaw result envelope
    (``{"tool":{"id"/"name":"...manyforge__<tool>"}}``)."""
    if not isinstance(message, dict) or message.get("role") != "tool":
        return None
    if _content_len(message) < _MIN_ELIDABLE_READ_CHARS:
        return None
    content = message.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except Exception:
            return None
    if not isinstance(content, dict):
        return None
    tool = content.get("tool")
    name = (tool.get("name") or tool.get("id")) if isinstance(tool, dict) else None
    if not isinstance(name, str):
        return None
    bare = name.rsplit(":", 1)[-1]
    if bare.startswith("manyforge__"):
        bare = bare[len("manyforge__"):]
    return bare if bare in _REFETCHABLE_READ_TOOLS else None


def _trim_history_to_budget(body_json: dict, max_chars: int) -> tuple[list[str], int]:
    """Fail-open history trim ladder; mutates ``body_json['messages']`` in
    place. Returns ``(rungs_applied, chars_shed)``. Rung 1 always runs;
    rungs 2-3 escalate only while still over budget."""
    messages = body_json.get("messages")
    if not isinstance(messages, list) or not messages:
        return [], 0

    def _size() -> int:
        return len(json.dumps(body_json, separators=(",", ":")))

    before = _size()
    applied: list[str] = []

    by_kind: dict[str, list[int]] = {}
    for i, m in enumerate(messages):
        k = _refetchable_read_kind(m)
        if k:
            by_kind.setdefault(k, []).append(i)

    def _stub(idx: int, stub: str) -> bool:
        m = messages[idx]
        if not isinstance(m, dict) or m.get("content") == stub:
            return False
        m["content"] = stub
        return True

    # Rung 1 (always): stub OLD re-fetchable reads, keep the latest of each kind.
    n1 = sum(_stub(idx, _READ_ELIDED_STUB)
             for idxs in by_kind.values() for idx in idxs[:-1])
    if n1:
        applied.append(f"stub_old_reads={n1}")
    if _size() <= max_chars:
        return applied, before - _size()

    # Rung 2: stub ALL re-fetchable reads (including the latest of each kind).
    n2 = sum(_stub(idxs[-1], _READ_ELIDED_STUB) for idxs in by_kind.values())
    if n2:
        applied.append(f"stub_all_reads={n2}")
    if _size() <= max_chars:
        return applied, before - _size()

    # Rung 3: stub OLD assistant reasoning (keep tool_calls + latest reasoning).
    asst = [i for i, m in enumerate(messages)
            if isinstance(m, dict) and m.get("role") == "assistant"]
    n3 = 0
    for idx in asst[:-1]:
        m = messages[idx]
        c = m.get("content")
        if isinstance(c, str) and c and c != _REASONING_ELIDED_STUB:
            m["content"] = _REASONING_ELIDED_STUB
            n3 += 1
    if n3:
        applied.append(f"stub_old_reasoning={n3}")
    if _size() <= max_chars:
        return applied, before - _size()

    # Rung 4 (last resort): drop the OLDEST messages (after a leading system
    # prompt) until under budget, then drop any tool result whose assistant
    # tool_call no longer survives so the remaining sequence stays a valid
    # tool_call<->result sequence. This sheds real STATE the model can't
    # re-fetch, so it is last and the caller raises a user-facing warning.
    head = (1 if messages and isinstance(messages[0], dict)
            and messages[0].get("role") == "system" else 0)
    dropped = 0
    while _size() > max_chars and (len(messages) - head) > 2:
        del messages[head]
        dropped += 1
    orphans = 0
    if dropped:
        live_ids = {
            tc.get("id")
            for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or []) if isinstance(tc, dict)
        }
        kept = []
        for m in messages:
            if (isinstance(m, dict) and m.get("role") == "tool"
                    and m.get("tool_call_id")
                    and m.get("tool_call_id") not in live_ids):
                orphans += 1
                continue
            kept.append(m)
        if orphans:
            messages[:] = kept
        applied.append(
            f"truncate_oldest={dropped}" + (f"+orphans={orphans}" if orphans else "")
        )

    return applied, before - _size()


# The composer injects a per-request manifest (carried in a user message) that
# includes the manyforge conversationId. We read it so history trim/truncate
# events can be correlated to a conversation and surfaced to the user as a
# warning by the composer. Bounded id length keeps the regex cheap/safe.
_CONVERSATION_ID_RE = _re.compile(r'"conversationId"\s*:\s*"([^"]{1,128})"')


def _extract_conversation_id(body_json: object) -> str | None:
    """Best-effort conversationId from the request. Returns None if absent so a
    manifest-format change degrades gracefully (the event is still logged,
    just without the id) rather than breaking the request path."""
    if not isinstance(body_json, dict):
        return None
    messages = body_json.get("messages")
    if not isinstance(messages, list):
        return None
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        parts: list[str] = []
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    parts.append(p["text"])
        for text in parts:
            if "conversationId" in text:
                mm = _CONVERSATION_ID_RE.search(text)
                if mm:
                    return mm.group(1)
    return None


def _append_log(record: dict) -> None:
    line = json.dumps(record, sort_keys=False, default=str)
    with _LOG_LOCK:
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


class ProxyHandler(BaseHTTPRequestHandler):
    """One handler per request. Reads the full body, forwards
    upstream, captures the full response, logs both."""

    # We never serve our own pages — only forward.
    server_version = "OpenClawProxy/0.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        # Quiet the default access log; we have our own JSONL.
        return

    def _read_request_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _forward(self, method: str) -> None:
        path = self.path
        body = self._read_request_body()
        # Strip hop-by-hop headers per RFC 7230 §6.1.
        hop_headers = {
            "connection", "keep-alive", "proxy-authenticate",
            "proxy-authorization", "te", "trailer", "transfer-encoding",
            "upgrade", "host", "content-length",
        }
        forward_headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in hop_headers
        }

        # Optional mutation pass — must run BEFORE Content-Length is set.
        mutated_body, mutation_record = _maybe_mutate_request(path, body)
        body = mutated_body  # forwarded body == possibly mutated body

        # Tool-surface drift check. Read-only — emits a JSONL+stderr
        # warning if the observed tools[] does not match the operator-
        # declared OPENCLAW_PROXY_TOOL_SURFACE. Cheap (parse-once,
        # dedup per session). Inert when the env var is unset.
        if _TOOL_SURFACE_EXPECTED and path.endswith("/v1/chat/completions"):
            try:
                parsed_body = json.loads(body.decode("utf-8", errors="replace"))
            except (ValueError, UnicodeDecodeError):
                parsed_body = None
            if isinstance(parsed_body, dict):
                _check_tool_surface_drift(
                    path=path,
                    headers=self.headers,
                    body=parsed_body,
                    log_extra={"request_id": self.headers.get("X-Request-ID")},
                )

        # 2026-06-01 (round 10 — cascading loop break):
        # Two-stage defense against same-tool-same-error runaway:
        #   1. At _REFLECT_AT same-tool calls: INJECT a synthetic user
        #      message into messages[] urging the model to change
        #      tactics, then forward. Model gets a fresh chance.
        #   2. At _STOP_AT same-tool calls: HARD-STOP with an OpenAI-
        #      style error body, no further GPU spend. This is failure-
        #      shaped on purpose; benchmark scorers must not treat a
        #      loop guard as a successful assistant answer.
        # Both stages are bounded so the model can't escape indefinitely.
        # Configured via OPENCLAW_PROXY_LOOP_REFLECT_AT (default 4) and
        # OPENCLAW_PROXY_LOOP_STOP_AT (default 8). Legacy single-knob
        # OPENCLAW_PROXY_LOOP_TOOL_THRESHOLD still honored.
        loop_error, loop_mutated_body, loop_trigger_info = _check_loop_short_circuit(path, body)
        if loop_error is not None:
            # Hard stop — synthesize an error-shaped response and return.
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(loop_error)))
            self.end_headers()
            self.wfile.write(loop_error)
            self.wfile.flush()
            _append_log({
                "ts": int(time.time() * 1000),
                "event": "proxy_loop_hard_stop",
                "path": path,
                "status": 409,
                "trigger_info": loop_trigger_info,
            })
            return
        if loop_mutated_body is not None:
            # Reflection injected — forward the mutated body
            body = loop_mutated_body
            _append_log({
                "ts": int(time.time() * 1000),
                "event": "proxy_loop_reflection_injected",
                "path": path,
                "trigger_info": loop_trigger_info,
            })

        if _MAX_REQUEST_CHARS > 0 and path.endswith("/v1/chat/completions"):
            # FAIL-OPEN: shed re-fetchable content (rungs 1-3) then HARD-TRIM the
            # oldest content (rung 4) to fit the budget. The proxy NEVER rejects:
            # if a single recent message still exceeds budget after the hard
            # trim, we forward it anyway and warn -- the model handles its own
            # context window; we do not pre-fail the turn. (The budget is
            # auto-sized to ~90% of the model context by the launcher, so a
            # residual over-budget is rare and still within the actual window.)
            body_json, body_raw = _try_parse_json(body)
            if isinstance(body_json, dict):
                # Read the conversationId BEFORE trimming (rung 4 may drop the
                # manifest message), so the composer can correlate the warning.
                conv_id = _extract_conversation_id(body_json)
                rungs, shed = _trim_history_to_budget(body_json, _MAX_REQUEST_CHARS)
                if rungs:
                    body = json.dumps(body_json, separators=(",", ":")).encode("utf-8")
                still_over = len(body) > _MAX_REQUEST_CHARS
                # Rung 4 (or a residual over-budget) forgets real state -> a
                # user-facing warning the composer surfaces in its AI section.
                truncated = still_over or any(
                    r.startswith("truncate_oldest") for r in rungs)
                if rungs or still_over:
                    _append_log({
                        "ts": int(time.time() * 1000),
                        "event": "proxy_history_truncated" if truncated
                        else "proxy_history_trimmed",
                        "path": path,
                        "conversationId": conv_id,
                        "rungs": rungs,
                        "chars_shed": shed,
                        "body_chars_after": len(body),
                        "max_request_chars": _MAX_REQUEST_CHARS,
                        "stillOverAfterHardTrim": still_over,
                        "userWarning": (
                            "Conversation history was trimmed/truncated to fit "
                            "the model context window; older steps may be "
                            "forgotten. Reset the conversation if results degrade."
                        ) if truncated else None,
                    })

        forward_headers["Content-Length"] = str(len(body))

        ts_in = time.time()
        request_record = {
            "method": method,
            "path": path,
            "headers": dict(forward_headers),
            "body_chars": len(body),
        }
        if mutation_record is not None:
            request_record["mutation"] = mutation_record
        body_json, body_raw = _try_parse_json(body)
        if body_json is not None:
            request_record["body"] = body_json
        else:
            request_record["body_raw_excerpt"] = _truncate(body_raw, 1048576)

        # Forward to upstream. Buffered (we read entire response before
        # returning to caller). For SSE/streaming responses this serializes
        # what would otherwise be incremental — but our internal HTTP server
        # (BaseHTTPRequestHandler) doesn't support chunked output cleanly,
        # and a partial write without Content-Length leaves the client
        # waiting on connection close. Buffering preserves correctness;
        # the latency cost is acceptable for the smoke harness's purposes.
        conn = None
        try:
            # Per-request socket timeout (MANYFORGE_PROXY_UPSTREAM_TIMEOUT_S,
            # default 300s — coordinated just below the assistant/corpus budget
            # by the launcher). Bounds a single runaway generation so it cannot
            # hold the single llama.cpp slot indefinitely; see the
            # _UPSTREAM_TIMEOUT_S note for why the old hardcoded 200s sat below
            # the budget and cut off legitimate prefill-heavy turns on gemma.
            conn = http.client.HTTPConnection(
                _UPSTREAM_HOST, _UPSTREAM_PORT, timeout=_UPSTREAM_TIMEOUT_S,
            )
            conn.request(method, path, body=body, headers=forward_headers)
            up_resp = conn.getresponse()
            resp_body = up_resp.read()
            resp_status = up_resp.status
            resp_headers = dict(up_resp.getheaders())
            conn.close()
        except Exception as exc:
            # Deterministic cleanup: close the upstream socket so llama.cpp
            # sees the client disconnect and can abandon the now-orphaned
            # generation instead of holding its single slot to completion.
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
            timed_out = type(exc).__name__ in ("timeout", "TimeoutError")
            elapsed_s = round(time.time() - ts_in, 1)
            # Surface the proxy failure as a 502 to the bridge so it's not
            # confused with a gateway error; log the failure too. A timeout is
            # its own explicit event (with the budget) so operators can tell
            # "model too slow / budget too low" from a genuine upstream crash.
            err_text = (
                f"proxy upstream {'timeout' if timed_out else 'error'} after "
                f"{elapsed_s}s (budget {_UPSTREAM_TIMEOUT_S}s): "
                f"{type(exc).__name__}: {exc}"
            )
            _append_log({
                "ts": int(ts_in * 1000),
                "event": "proxy_upstream_timeout" if timed_out else "proxy_upstream_error",
                "timeout_s": _UPSTREAM_TIMEOUT_S,
                "request": request_record,
                "response": {
                    "status": -1,
                    "error": err_text,
                    "duration_ms": round((time.time() - ts_in) * 1000.0, 1),
                },
            })
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(err_text)))
            self.end_headers()
            self.wfile.write(err_text.encode())
            return

        duration_ms = (time.time() - ts_in) * 1000.0

        # 2026-06-02: reasoning→content promotion (response-side mutation).
        # OpenClaw 2026.5.22 treats responses with `content==null` as
        # incomplete_result errors. Models with --reasoning-parser (e.g.
        # cosmos with qwen3) put all output in `reasoning`. Mirror it
        # into `content` so OpenClaw is satisfied. Pass-through when
        # content is already populated. See OPENCLAW_PROXY_PROMOTE_REASONING_TO_CONTENT.
        promote_mutations = 0
        if _PROMOTE_REASONING_TO_CONTENT and path.endswith("/v1/chat/completions"):
            try:
                new_body, promote_mutations = _promote_reasoning_to_content_in_response(resp_body)
                if promote_mutations:
                    resp_body = new_body
            except Exception:
                promote_mutations = 0

        # 2026-06-01: tool-name normalization (response-side mutation).
        # Rewrite emitted tool_call names that drop the MCP namespace
        # prefix to their unique canonical form. Diagnostic event logged
        # per rewrite. Path-gated to /v1/chat/completions and only fires
        # when there's a UNIQUE match (ambiguous cases pass through).
        normalize_rewrites: list[dict] = []
        if _NORMALIZE_TOOL_NAMES and path.endswith("/v1/chat/completions"):
            try:
                new_body, normalize_rewrites = _normalize_tool_names_in_response(
                    body_json, resp_body,
                )
                if normalize_rewrites:
                    resp_body = new_body
            except Exception:
                normalize_rewrites = []
        if normalize_rewrites:
            _append_log({
                "ts": int(time.time() * 1000),
                "event": "proxy_tool_name_normalized",
                "path": path,
                "rewrites": normalize_rewrites,
            })

        # 2026-06-04: nested-MCP-id normalization for tools-mode dispatch.
        # Rewrites broken model emissions of the ManyForge id field
        # inside `tool_call`/`tool_describe` arguments (bare ids,
        # dashed forms, MCP-locator prefixes) to the canonical
        # `manyforge__<bare>` shape OpenClaw accepts. Path-gated to
        # /v1/chat/completions because that's where the streamed tool
        # calls land. Operates on the text body (works for both JSON
        # and SSE shapes since the JSON-escaped id strings are
        # identical). Idempotent on already-canonical responses.
        nested_id_rewrites: list[dict] = []
        if _NORMALIZE_NESTED_MCP_IDS and path.endswith("/v1/chat/completions"):
            try:
                resp_text = (resp_body.decode("utf-8") if isinstance(resp_body, bytes)
                             else (resp_body or ""))
                # SSE responses fragment `function.arguments` one token per
                # `data:` event, so the contiguous `"id":"…"` the text-regex
                # needs never appears in the body. Use the structural,
                # reassemble-then-normalize variant for the streaming shape;
                # fall back to the text-regex for plain-JSON bodies.
                if "data:" in resp_text:
                    new_text, nested_id_rewrites = _normalize_nested_mcp_ids_streaming(resp_text)
                else:
                    new_text, nested_id_rewrites = _normalize_nested_mcp_ids_in_text(resp_text)
                if nested_id_rewrites:
                    resp_body = new_text.encode("utf-8")
            except Exception:
                nested_id_rewrites = []
        if nested_id_rewrites:
            _append_log({
                "ts": int(time.time() * 1000),
                "event": "proxy_nested_mcp_id_normalized",
                "path": path,
                "rewrites": nested_id_rewrites,
            })

        resp_body_json, resp_body_raw = _try_parse_json(resp_body)
        response_record: dict = {
            "status": resp_status,
            "headers": resp_headers,
            "duration_ms": round(duration_ms, 1),
            "body_chars": len(resp_body),
        }
        if normalize_rewrites:
            response_record["normalize_rewrites"] = normalize_rewrites
        if nested_id_rewrites:
            response_record["nested_id_rewrites"] = nested_id_rewrites
        if resp_body_json is not None:
            response_record["body"] = resp_body_json
        else:
            response_record["body_raw_excerpt"] = _truncate(resp_body_raw, 1048576)

        _append_log({
            "ts": int(ts_in * 1000),
            "request": request_record,
            "response": response_record,
        })

        # 2026-06-01: response-side malformed-tool-call detection (generic).
        # Diagnostic-only — does not modify the response. Writes a
        # separate JSONL event for easy grep / per-model metric.
        if _DETECT_MALFORMED_TOOL_CALL and path.endswith("/v1/chat/completions"):
            try:
                detection = _detect_malformed_tool_call(body_json, resp_body)
            except Exception:
                detection = None
            if detection is not None:
                _append_log({
                    "ts": int(time.time() * 1000),
                    "event": "proxy_malformed_tool_call_detected",
                    "path": path,
                    "matched_tool": detection["matched_tool"],
                    "markers": detection["markers"],
                    "content_chars": detection["content_chars"],
                    "content_preview": detection["content_preview"],
                })

        # Forward response to the bridge. Strip hop-by-hop again on
        # the way back; preserve Content-Type and any custom headers.
        self.send_response(resp_status)
        for hk, hv in resp_headers.items():
            if hk.lower() in hop_headers:
                continue
            self.send_header(hk, hv)
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def do_POST(self) -> None:  # noqa: N802
        self._forward("POST")

    def do_GET(self) -> None:  # noqa: N802
        self._forward("GET")

    def do_PUT(self) -> None:  # noqa: N802
        self._forward("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self._forward("DELETE")


class ThreadedProxyServer(socketserver.ThreadingMixIn,
                          socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def _resolve_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-port", type=int, default=int(
        os.environ.get("OPENCLAW_PROXY_LISTEN_PORT", "18790")))
    parser.add_argument("--bind", type=str, default=os.environ.get(
        "OPENCLAW_PROXY_BIND", "127.0.0.1"),
        help="Bind address (use 0.0.0.0 to be reachable from "
             "containerised callers via the docker bridge IP).")
    parser.add_argument("--upstream", type=str, default=os.environ.get(
        "OPENCLAW_PROXY_UPSTREAM", "http://127.0.0.1:18789"))
    parser.add_argument("--log-path", type=str, default=os.environ.get(
        "OPENCLAW_PROXY_LOG_PATH", "/tmp/openclaw_proxy.jsonl"))
    # Mutation flags (off by default — proxy stays a pure logger):
    parser.add_argument("--force-tool-choice", type=str, default=os.environ.get(
        "OPENCLAW_PROXY_FORCE_TOOL_CHOICE", ""),
        help='If set to "required" / "auto": override `tool_choice` on '
             "outbound chat-completions. Mitigates the OpenClaw gateway "
             "dropping the parameter on the gateway → vLLM hop.")
    parser.add_argument("--override-temperature", type=str, default=os.environ.get(
        "OPENCLAW_PROXY_OVERRIDE_TEMPERATURE", ""),
        help="If set (float): overwrite `temperature` on every chat-completions request.")
    parser.add_argument("--override-max-tokens", type=str, default=os.environ.get(
        "OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS", ""),
        help="If set (int): overwrite `max_tokens` / `max_completion_tokens`.")
    parser.add_argument("--override-top-p", type=str, default=os.environ.get(
        "OPENCLAW_PROXY_OVERRIDE_TOP_P", ""),
        help="If set (float): overwrite `top_p`.")
    parser.add_argument("--override-reasoning-effort", type=str, default=os.environ.get(
        "OPENCLAW_PROXY_OVERRIDE_REASONING_EFFORT", ""),
        help="If set: overwrite the standard `reasoning_effort` request field. "
             "Useful for endpoints such as DS4 that support low/high/max/off.")
    parser.add_argument("--user-message-suffix", type=str, default=os.environ.get(
        "OPENCLAW_PROXY_USER_MESSAGE_SUFFIX", ""),
        help="If set: appended to the LAST user message on every "
             "chat-completions request. Use for cross-cutting "
             "plan-then-execute hints (e.g. 'Before answering, call "
             "the appropriate read tool to refresh state.'). "
             "Idempotent — won't double-append on retries.")
    parser.add_argument("--user-suffix-first-turn-only",
        type=str, default=os.environ.get(
            "OPENCLAW_PROXY_USER_SUFFIX_FIRST_TURN_ONLY", ""),
        help="If '1' / 'true': only inject the user suffix on the FIRST "
             "turn of each conversation (when no `assistant` messages "
             "are present in the request). Avoids per-turn overhead on "
             "long chains where a single read-first hint is enough.")
    parser.add_argument("--thinking-token-budget", type=str, default=os.environ.get(
        "OPENCLAW_PROXY_THINKING_TOKEN_BUDGET", ""),
        help="If set (int): inject chat_template_kwargs.thinking_token_budget=N "
             "on every chat-completions request. Caps the <think> envelope. "
             "Recommended 512 for 8B-class robotics tool calls.")
    parser.add_argument("--force-enable-thinking", type=str, default=os.environ.get(
        "OPENCLAW_PROXY_FORCE_ENABLE_THINKING", ""),
        help="Per-turn enable_thinking control. Modes: 'on' (always inject "
             "chat_template_kwargs.enable_thinking=true), 'off' (always "
             "false), 'alternating-off-on-even' (don't mutate odd turns / "
             "leave vLLM default; force false on even turns — turn parity "
             "by assistant message count). Empty = no mutation.")
    return parser.parse_args()


def _parse_optional_float(raw: str) -> float | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_optional_int(raw: str) -> int | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def main() -> None:
    global _UPSTREAM_HOST, _UPSTREAM_PORT, _UPSTREAM_SCHEME
    global _LISTEN_PORT, _LOG_PATH
    global _FORCE_TOOL_CHOICE, _OVERRIDE_TEMPERATURE
    global _OVERRIDE_MAX_TOKENS, _OVERRIDE_TOP_P, _OVERRIDE_REASONING_EFFORT
    global _USER_MESSAGE_SUFFIX, _USER_SUFFIX_FIRST_TURN_ONLY
    global _FORCE_ENABLE_THINKING, _THINKING_TOKEN_BUDGET
    global _TOOL_SURFACE_EXPECTED, _TOOL_SURFACE_WARN_ON_UNKNOWN

    cfg = _resolve_config()
    parsed = urllib.parse.urlparse(cfg.upstream)
    _UPSTREAM_SCHEME = parsed.scheme or "http"
    _UPSTREAM_HOST = parsed.hostname or "127.0.0.1"
    _UPSTREAM_PORT = parsed.port or (443 if _UPSTREAM_SCHEME == "https" else 18789)
    _LISTEN_PORT = cfg.listen_port
    _LOG_PATH = cfg.log_path

    # Mutation config (None = no mutation for that field)
    ftc = (cfg.force_tool_choice or "").strip()
    _FORCE_TOOL_CHOICE = ftc if ftc else None
    _OVERRIDE_TEMPERATURE = _parse_optional_float(cfg.override_temperature)
    _OVERRIDE_MAX_TOKENS = _parse_optional_int(cfg.override_max_tokens)
    _OVERRIDE_TOP_P = _parse_optional_float(cfg.override_top_p)
    reasoning_effort = (cfg.override_reasoning_effort or "").strip()
    _OVERRIDE_REASONING_EFFORT = reasoning_effort if reasoning_effort else None
    sfx = (cfg.user_message_suffix or "").strip()
    _USER_MESSAGE_SUFFIX = sfx if sfx else None
    sfx_first = (cfg.user_suffix_first_turn_only or "").strip().lower()
    _USER_SUFFIX_FIRST_TURN_ONLY = sfx_first in ("1", "true", "yes", "on")
    fet = (cfg.force_enable_thinking or "").strip()
    _FORCE_ENABLE_THINKING = fet if fet else None
    _THINKING_TOKEN_BUDGET = _parse_optional_int(cfg.thinking_token_budget)

    # Tool-surface drift check (Commit C, 2026-06-03). Inert when
    # the env var is unset, preserving back-compat for proxy
    # deployments that don't run the lane-aware bridge.
    tool_surface_expected = (
        os.environ.get("OPENCLAW_PROXY_TOOL_SURFACE")
        or os.environ.get("OPENCLAW_ASSISTANT_TOOL_SURFACE")
        or ""
    ).strip().lower()
    if tool_surface_expected and tool_surface_expected not in ("code", "tools"):
        sys.stderr.write(
            f"openclaw-proxy: OPENCLAW_PROXY_TOOL_SURFACE="
            f"{tool_surface_expected!r} not in {{'code','tools'}}; "
            "drift check disabled.\n"
        )
        tool_surface_expected = ""
    _TOOL_SURFACE_EXPECTED = tool_surface_expected
    _TOOL_SURFACE_WARN_ON_UNKNOWN = (
        os.environ.get("OPENCLAW_PROXY_WARN_ON_UNKNOWN", "")
        .strip()
        .lower()
        in ("1", "true", "yes", "on")
    )

    # Truncate prior log so each session starts fresh; harness handles
    # offset-from-baseline anyway, but a clean file on launch is
    # convenient for ad-hoc cat'ing.
    open(_LOG_PATH, "w").close()

    server = ThreadedProxyServer((cfg.bind, _LISTEN_PORT), ProxyHandler)
    mutation_summary = []
    if _FORCE_TOOL_CHOICE is not None:
        mutation_summary.append(f"tool_choice→{_FORCE_TOOL_CHOICE}")
    if _OVERRIDE_TEMPERATURE is not None:
        mutation_summary.append(f"temperature={_OVERRIDE_TEMPERATURE}")
    if _OVERRIDE_MAX_TOKENS is not None:
        mutation_summary.append(f"max_tokens={_OVERRIDE_MAX_TOKENS}")
    if _OVERRIDE_TOP_P is not None:
        mutation_summary.append(f"top_p={_OVERRIDE_TOP_P}")
    if _OVERRIDE_REASONING_EFFORT is not None:
        mutation_summary.append(f"reasoning_effort={_OVERRIDE_REASONING_EFFORT}")
    if _USER_MESSAGE_SUFFIX is not None:
        preview = _USER_MESSAGE_SUFFIX[:40].replace("\n", " ")
        if len(_USER_MESSAGE_SUFFIX) > 40:
            preview += "…"
        mutation_summary.append(f'user_suffix="{preview}"')
    if _FORCE_ENABLE_THINKING is not None:
        mutation_summary.append(f"enable_thinking={_FORCE_ENABLE_THINKING}")
    if _THINKING_TOKEN_BUDGET is not None:
        mutation_summary.append(f"thinking_token_budget={_THINKING_TOKEN_BUDGET}")
    # Multi-criteria loop-reflection summary
    triggers_on = []
    if _TRIGGER_SAME_TOOL: triggers_on.append(f"same_tool>={_REFLECT_AT}")
    if _TRIGGER_SAME_ARGS: triggers_on.append("same_args>=2")
    if _TRIGGER_RESULT_REPEAT: triggers_on.append("result_repeat>=2")
    if _TRIGGER_NAMESPACE: triggers_on.append(f"namespace>={_NAMESPACE_AT}(stop@{_NAMESPACE_STOP_AT})")
    if _TRIGGER_TURN_COUNTER: triggers_on.append(f"turn_counter>={_TURN_COUNTER_AT}")
    if triggers_on or _STOP_AT > 0:
        loop_str = (",".join(triggers_on) if triggers_on else "off") + f" stop@{_STOP_AT}"
        mutation_summary.append(f"loop_reflect=[{loop_str}]")
    if _DETECT_MALFORMED_TOOL_CALL:
        mutation_summary.append("malformed_tool_detect=on")
    if _NORMALIZE_TOOL_NAMES:
        mutation_summary.append("normalize_tool_names=on")
    if _NORMALIZE_NESTED_MCP_IDS:
        mutation_summary.append("normalize_nested_mcp_ids=on")
    if _GUIDED_TOOL_CALLS:
        mutation_summary.append("guided_tool_calls=on(action-shaped)")
    if _TOOL_PARSER:
        constraint_kind = "structural_tag" if _TOOL_PARSER == "hermes" else "grammar"
        mutation_summary.append(f"tool_constraint={_TOOL_PARSER}({constraint_kind})")
    if _TOOL_ERROR_REWRITE:
        mutation_summary.append("tool_error_rewrite=on")
    if _PROMOTE_REASONING_TO_CONTENT:
        mutation_summary.append("promote_reasoning_to_content=on")
    if _UNWRAP_TOOL_CALL_ARGS:
        mutation_summary.append("unwrap_tool_call_args=on")
    mutation_str = ", ".join(mutation_summary) if mutation_summary else "logging-only (no mutations)"
    # Banner identifies the active mutation profile (Phase 1 of
    # THREE-LANE-MIGRATION-PLAN.md) so cross-lane reports can correlate
    # behavior to profile. Banner name kept as ``openclaw-logging-proxy``
    # for log-grep backward compatibility; the profile field is the new
    # source of truth.
    print(
        f"openclaw-logging-proxy listening on {cfg.bind}:{_LISTEN_PORT} "
        f"-> {_UPSTREAM_SCHEME}://{_UPSTREAM_HOST}:{_UPSTREAM_PORT} "
        f"(profile: {MANYFORGE_PROXY_PROFILE}; log: {_LOG_PATH}; mode: {mutation_str})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutdown", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
