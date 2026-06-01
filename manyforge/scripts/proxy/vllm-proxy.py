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
  OPENCLAW_PROXY_LISTEN_PORT
  OPENCLAW_PROXY_UPSTREAM
  OPENCLAW_PROXY_LOG_PATH
  OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS       cap output tokens (injected when caller omits)
  OPENCLAW_PROXY_THINKING_TOKEN_BUDGET     soft cap on internal CoT
  OPENCLAW_PROXY_FORCE_ENABLE_THINKING     on | off | alternating-…
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
import hashlib
import http.client
import json
import os
import socketserver
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler


_LOG_LOCK = threading.Lock()
_LOG_PATH = "/tmp/openclaw_proxy.jsonl"
_UPSTREAM_HOST = "127.0.0.1"
_UPSTREAM_PORT = 18789
_UPSTREAM_SCHEME = "http"
_LISTEN_PORT = 18790

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
            and _USER_MESSAGE_SUFFIX is None and _FORCE_ENABLE_THINKING is None
            and _THINKING_TOKEN_BUDGET is None):
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
            # 2026-05-31: composers like manyforge-composer also send a
            # top-level `enable_thinking` field. vLLM treats that as the
            # source of truth and IGNORES chat_template_kwargs when both
            # are present (observed empirically on cosmos-reason2-8b
            # 2026-05-31: chat_template_kwargs.enable_thinking=True +
            # top-level enable_thinking=False → reasoning field stays
            # empty). Mirror the chat_template_kwargs decision to the
            # top-level field so the model actually engages thinking.
            top_before = parsed.get("enable_thinking")
            if top_before != target_value:
                parsed["enable_thinking"] = target_value
                changes["enable_thinking_top"] = {
                    "before": top_before,
                    "after": target_value,
                }

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

    if not changes:
        return body, None
    new_body = json.dumps(parsed, separators=(",", ":")).encode("utf-8")
    return new_body, {"mutations": changes}


# Round 10 (2026-06-01): cascading loop-break
#   reflect_at  = inject reflection user message into messages[] when
#                 same tool reaches this count (default 4)
#   stop_at     = hard-stop with synthetic SSE when count reaches this
#                 (default 8 — gives the model one shot after reflection)
# Old single-threshold env (LOOP_TOOL_THRESHOLD) maps to stop_at if set.
_REFLECT_AT = int(os.environ.get("OPENCLAW_PROXY_LOOP_REFLECT_AT", "4") or "4")
_STOP_AT = int(os.environ.get("OPENCLAW_PROXY_LOOP_STOP_AT", "8") or "8")
_LEGACY_THRESHOLD = int(os.environ.get("OPENCLAW_PROXY_LOOP_TOOL_THRESHOLD", "0") or "0")
if _LEGACY_THRESHOLD > 0:
    _STOP_AT = _LEGACY_THRESHOLD
    # If only legacy is set, disable reflection (preserve old behavior)
    if "OPENCLAW_PROXY_LOOP_REFLECT_AT" not in os.environ:
        _REFLECT_AT = 0

_REFLECTION_MARKER = "[loop-reflection]"  # marker so we never inject twice


def _check_loop_short_circuit(path: str, body: bytes) -> tuple[bytes | None, bytes | None]:
    """Inspect /v1/chat/completions request for runaway same-tool loops.
    Returns (synthetic_sse_response, mutated_body):
      - both None: forward as-is
      - synthetic_sse_response set: short-circuit (do not forward)
      - mutated_body set: forward the mutated body (reflection injected)
    Designed to break the openclaw per-turn retry-loop where the model
    keeps calling the same tool with the same args after the validator
    fails repeatedly with the same error.
    """
    if _STOP_AT <= 0 and _REFLECT_AT <= 0:
        return None, None
    if not path.endswith("/v1/chat/completions"):
        return None, None
    try:
        parsed = json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError):
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    msgs = parsed.get("messages") or []
    if not isinstance(msgs, list):
        return None, None
    # Count CONSECUTIVE same-tool calls at the TAIL of the assistant
    # turn history. Lifetime counting (the prior implementation) broke
    # chained smoke cases: when composer shares a conversationId across
    # cases (e.g. PnP_01..PnP_20), the lifetime same-tool count crosses
    # _STOP_AT by the 3rd or 4th case, then hard-stops every subsequent
    # case on turn 1. Consecutive counting only fires when the model is
    # ACTUALLY stuck in a same-tool loop within the current case.
    # A turn that calls a DIFFERENT tool resets the run length to 1
    # (or 0 if that turn has no tool call).
    from collections import Counter as _Counter
    consecutive_count = 0
    top_name = None
    for m in msgs:
        if not isinstance(m, dict): continue
        if m.get("role") != "assistant": continue
        tcs = m.get("tool_calls") or []
        if not tcs:
            # Assistant turn with no tool call → reset
            consecutive_count = 0
            top_name = None
            continue
        # Take the first tool name from this assistant turn
        nm = None
        for tc in tcs:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                nm = fn.get("name") or tc.get("name")
                if nm: break
        if not nm:
            consecutive_count = 0
            top_name = None
            continue
        if nm == top_name:
            consecutive_count += 1
        else:
            top_name = nm
            consecutive_count = 1
    if top_name is None or consecutive_count == 0:
        return None, None
    top_count = consecutive_count

    # ----- HARD STOP at _STOP_AT -----
    if _STOP_AT > 0 and top_count >= _STOP_AT:
        model = parsed.get("model") or "unknown"
        stop_msg = (
            f"[loop-break] I have called `{top_name}` {top_count} times in this "
            f"conversation. Repeated retries hit the same validator error. "
            f"Stopping to avoid runaway. Please refine the request with the "
            f"specific missing field, pick a different tool, or restate the goal."
        )
        chatcmpl_id = "chatcmpl-loopbreak"
        chunk1 = {
            "id": chatcmpl_id, "object": "chat.completion.chunk", "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": stop_msg}, "finish_reason": None}],
        }
        chunk2 = {
            "id": chatcmpl_id, "object": "chat.completion.chunk", "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        sse = (
            f"data: {json.dumps(chunk1)}\n\n"
            f"data: {json.dumps(chunk2)}\n\n"
            f"data: [DONE]\n\n"
        )
        return sse.encode("utf-8"), None

    # ----- REFLECTION INJECTION at _REFLECT_AT -----
    if _REFLECT_AT > 0 and top_count >= _REFLECT_AT:
        # Skip if reflection already injected for this convo (any user
        # message containing the marker)
        already = any(
            isinstance(m, dict) and m.get("role") == "user"
            and isinstance(m.get("content"), str) and _REFLECTION_MARKER in m["content"]
            for m in msgs
        )
        if already:
            return None, None
        # Extract the most recent tool error to feed back to the model
        last_err = ""
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "tool":
                c = m.get("content") or ""
                if isinstance(c, str):
                    last_err = c[:400]
                break
        reflect_msg = (
            f"{_REFLECTION_MARKER} STOP. You have called `{top_name}` "
            f"{top_count} times with the same arguments and got the same "
            f"error each time. Last error snippet: {last_err!r}. "
            f"Do NOT retry the same call. Choose ONE of: "
            f"(a) call a DIFFERENT tool, "
            f"(b) change the specific failing argument named in the error "
            f"(not other arguments), or "
            f"(c) emit a short clarifying question for the user. "
            f"Reply now with your next action — but do not repeat the "
            f"failing call."
        )
        # Inject as a user message AFTER the last tool result
        new_msgs = list(msgs)
        # Find the index of the last tool message; insert after it
        last_tool_idx = -1
        for i, m in enumerate(new_msgs):
            if isinstance(m, dict) and m.get("role") == "tool":
                last_tool_idx = i
        insert_idx = (last_tool_idx + 1) if last_tool_idx >= 0 else len(new_msgs)
        new_msgs.insert(insert_idx, {"role": "user", "content": reflect_msg})
        parsed["messages"] = new_msgs
        mutated = json.dumps(parsed).encode("utf-8")
        return None, mutated

    return None, None
    return sse.encode("utf-8")


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

        # 2026-06-01 (round 10 — cascading loop break):
        # Two-stage defense against same-tool-same-error runaway:
        #   1. At _REFLECT_AT same-tool calls: INJECT a synthetic user
        #      message into messages[] urging the model to change
        #      tactics, then forward. Model gets a fresh chance.
        #   2. At _STOP_AT same-tool calls: HARD-STOP with synthetic
        #      SSE assistant response saying "stuck", no further GPU
        #      spend. OpenClaw exits its agent loop cleanly.
        # Both stages are bounded so the model can't escape indefinitely.
        # Configured via OPENCLAW_PROXY_LOOP_REFLECT_AT (default 4) and
        # OPENCLAW_PROXY_LOOP_STOP_AT (default 8). Legacy single-knob
        # OPENCLAW_PROXY_LOOP_TOOL_THRESHOLD still honored.
        loop_sse, loop_mutated_body = _check_loop_short_circuit(path, body)
        if loop_sse is not None:
            # Hard stop — synthesize SSE and return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(loop_sse)))
            self.end_headers()
            self.wfile.write(loop_sse)
            self.wfile.flush()
            _append_log({
                "ts": int(time.time() * 1000),
                "event": "proxy_loop_hard_stop",
                "path": path,
            })
            return
        if loop_mutated_body is not None:
            # Reflection injected — forward the mutated body
            body = loop_mutated_body
            _append_log({
                "ts": int(time.time() * 1000),
                "event": "proxy_loop_reflection_injected",
                "path": path,
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
        try:
            # Per-request safety net: 200s socket timeout prevents a
            # single runaway generation (e.g. thinking-on with no
            # max_tokens cap) from holding a thread + KV slot for 10+
            # minutes. The smoke runner's own case timeout is 244s, so
            # 200s here ensures the proxy fails first and releases the
            # upstream slot before the runner gives up.
            conn = http.client.HTTPConnection(
                _UPSTREAM_HOST, _UPSTREAM_PORT, timeout=200.0,
            )
            conn.request(method, path, body=body, headers=forward_headers)
            up_resp = conn.getresponse()
            resp_body = up_resp.read()
            resp_status = up_resp.status
            resp_headers = dict(up_resp.getheaders())
            conn.close()
        except Exception as exc:
            # Surface the proxy failure as a 502 to the bridge so it's
            # not confused with a gateway error; log the failure too.
            err_text = f"proxy upstream error: {type(exc).__name__}: {exc}"
            _append_log({
                "ts": int(ts_in * 1000),
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
        resp_body_json, resp_body_raw = _try_parse_json(resp_body)
        response_record: dict = {
            "status": resp_status,
            "headers": resp_headers,
            "duration_ms": round(duration_ms, 1),
            "body_chars": len(resp_body),
        }
        if resp_body_json is not None:
            response_record["body"] = resp_body_json
        else:
            response_record["body_raw_excerpt"] = _truncate(resp_body_raw, 1048576)

        _append_log({
            "ts": int(ts_in * 1000),
            "request": request_record,
            "response": response_record,
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
    global _OVERRIDE_MAX_TOKENS, _OVERRIDE_TOP_P
    global _USER_MESSAGE_SUFFIX, _USER_SUFFIX_FIRST_TURN_ONLY
    global _FORCE_ENABLE_THINKING, _THINKING_TOKEN_BUDGET

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
    sfx = (cfg.user_message_suffix or "").strip()
    _USER_MESSAGE_SUFFIX = sfx if sfx else None
    sfx_first = (cfg.user_suffix_first_turn_only or "").strip().lower()
    _USER_SUFFIX_FIRST_TURN_ONLY = sfx_first in ("1", "true", "yes", "on")
    fet = (cfg.force_enable_thinking or "").strip()
    _FORCE_ENABLE_THINKING = fet if fet else None
    _THINKING_TOKEN_BUDGET = _parse_optional_int(cfg.thinking_token_budget)

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
    if _USER_MESSAGE_SUFFIX is not None:
        preview = _USER_MESSAGE_SUFFIX[:40].replace("\n", " ")
        if len(_USER_MESSAGE_SUFFIX) > 40:
            preview += "…"
        mutation_summary.append(f'user_suffix="{preview}"')
    if _FORCE_ENABLE_THINKING is not None:
        mutation_summary.append(f"enable_thinking={_FORCE_ENABLE_THINKING}")
    if _THINKING_TOKEN_BUDGET is not None:
        mutation_summary.append(f"thinking_token_budget={_THINKING_TOKEN_BUDGET}")
    mutation_str = ", ".join(mutation_summary) if mutation_summary else "logging-only (no mutations)"
    print(
        f"openclaw-logging-proxy listening on {cfg.bind}:{_LISTEN_PORT} "
        f"-> {_UPSTREAM_SCHEME}://{_UPSTREAM_HOST}:{_UPSTREAM_PORT} "
        f"(log: {_LOG_PATH}; mode: {mutation_str})",
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
