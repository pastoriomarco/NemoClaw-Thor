#!/usr/bin/env python3
"""Dump full pipeline trace for one case.

Usage: diagnose_case.py <case_id> <proxy_jsonl_path> <smoke_report_json>

Output sections:
  1. Smoke verdict + failure reasons
  2. Expected tool calls / state_after / soft assertions
  3. Per-chat-completion trace: prompt fragment, model output, mutations,
     detector events, response.
  4. Tool result dispatches (from proxy log if visible)
  5. Root-cause hypothesis (heuristic)
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path


def case_in_msgs(msgs, case_id):
    for m in msgs:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        txt = c if isinstance(c, str) else " ".join(
            p.get("text", "") for p in (c or []) if isinstance(p, dict)
        )
        if case_id in txt:
            return True
    return False


def parse_sse_response(raw):
    """Extract (content, tool_calls, reasoning, finish) from an SSE body."""
    content = []
    tool_calls = []  # list of {name, arguments}
    reasoning = []
    finish = None
    if not raw:
        return "", [], "", None
    tc_pending = {}  # index -> {name, args}
    for m in re.finditer(r'data: (\{.*?\})\n\n', raw + '\n\n'):
        try:
            c = json.loads(m.group(1))
        except (ValueError, TypeError):
            continue
        ch = (c.get("choices") or [{}])[0]
        if not isinstance(ch, dict):
            continue
        d = ch.get("delta") or {}
        if isinstance(d.get("content"), str):
            content.append(d["content"])
        if isinstance(d.get("reasoning_content"), str):
            reasoning.append(d["reasoning_content"])
        for tc in (d.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            idx = tc.get("index", 0)
            slot = tc_pending.setdefault(idx, {"name": "", "args": ""})
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            if fn.get("arguments"):
                slot["args"] += fn["arguments"]
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
    for idx in sorted(tc_pending.keys()):
        tool_calls.append(tc_pending[idx])
    return "".join(content), tool_calls, "".join(reasoning), finish


def parse_json_response(raw):
    """Non-streaming JSON response."""
    if not raw:
        return "", [], "", None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    content = ""
    tool_calls = []
    reasoning = ""
    finish = None
    for ch in (data.get("choices") or []):
        msg = ch.get("message") or {}
        if isinstance(msg.get("content"), str):
            content += msg["content"]
        if isinstance(msg.get("reasoning_content"), str):
            reasoning += msg["reasoning_content"]
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            tool_calls.append({"name": fn.get("name", ""), "args": fn.get("arguments", "")})
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
    return content, tool_calls, reasoning, finish


def extract_user_prompt(msgs):
    """Find the bridge's full user prompt for diagnostic preview."""
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    return p["text"]
    return ""


def short_args(args_str, limit=160):
    if not args_str:
        return ""
    if isinstance(args_str, dict):
        args_str = json.dumps(args_str)
    args_str = str(args_str).replace("\n", " ")
    return (args_str[:limit] + "…") if len(args_str) > limit else args_str


def main(case_id, proxy_log_path, report_path):
    proxy_lines = []
    with open(proxy_log_path) as f:
        for line in f:
            try:
                proxy_lines.append(json.loads(line))
            except (ValueError, TypeError):
                pass

    # Get smoke report verdict
    report = None
    if Path(report_path).exists():
        try:
            report = json.loads(Path(report_path).read_text())
        except (ValueError, OSError):
            pass
    case_report = None
    if report:
        for c in report.get("cases", []) or []:
            if c.get("id") == case_id:
                case_report = c
                break

    # Get all chat completions for this case + events between them
    turns = []
    events = []
    cur_turn_idx = -1
    for r in proxy_lines:
        ev = r.get("event")
        if ev:
            events.append({"after_turn": cur_turn_idx, "rec": r})
            continue
        req = r.get("request") or {}
        if req.get("path") != "/v1/chat/completions":
            continue
        body = req.get("body") or {}
        if not isinstance(body, dict):
            continue
        msgs = body.get("messages") or []
        if not case_in_msgs(msgs, case_id):
            continue
        cur_turn_idx = len(turns)
        # Parse the request
        user_prompt = extract_user_prompt(msgs)
        # Find the actual user_request line in the prompt
        ur_match = re.search(r'## user_request\s*\n(.+?)(?:\n## |\Z)', user_prompt, re.S)
        user_request = ur_match.group(1).strip()[:300] if ur_match else "(not found)"
        # Count prior assistant tool turns visible to model
        prior_asst_tool_turns = sum(
            1 for m in msgs
            if isinstance(m, dict) and m.get("role") == "assistant" and (m.get("tool_calls") or [])
        )
        prior_tool_results = sum(
            1 for m in msgs if isinstance(m, dict) and m.get("role") == "tool"
        )
        # Mutations
        mut = req.get("mutation") or {}
        mut_keys = list((mut.get("mutations") or {}).keys())
        # Parse response
        resp = r.get("response") or {}
        raw = resp.get("body_raw_excerpt") or ""
        out = parse_sse_response(raw)
        if not (out[0] or out[1] or out[2]):
            jr = parse_json_response(raw)
            if jr is not None:
                out = jr
        content, tool_calls, reasoning, finish = out
        turns.append({
            "user_request": user_request,
            "prior_asst_tool_turns": prior_asst_tool_turns,
            "prior_tool_results": prior_tool_results,
            "mutations": mut_keys,
            "normalize_rewrites": resp.get("normalize_rewrites") or [],
            "duration_ms": resp.get("duration_ms"),
            "status": resp.get("status"),
            "content": content,
            "tool_calls": tool_calls,
            "reasoning": reasoning,
            "finish": finish,
        })

    # Heuristic root-cause
    hypothesis = None
    if case_report:
        fails = case_report.get("fail_reasons") or case_report.get("failures") or []
        if not isinstance(fails, list):
            fails = [str(fails)]
        # Pattern matching
        fail_text = " ".join(str(f) for f in fails)
        if "expected NO tool calls" in fail_text:
            hypothesis = "Model acted on a clarification-required prompt (should have asked instead)."
        elif "not observed (or never reached 2xx)" in fail_text:
            if any(t["tool_calls"] for t in turns):
                hypothesis = "Tool call emitted but dispatch failed (validation error or wrong args/name)."
            else:
                hypothesis = "Model did not emit a tool call at all (clarification, prose, or malformed)."
        elif "chat HTTP -1" in fail_text or "chat HTTP 502" in fail_text:
            hypothesis = "Pipeline error (vLLM crash, proxy 502, or composer timeout)."
        elif "state_after" in fail_text:
            hypothesis = "Tool ran but final program/scene state diverged from expected."

    # Render report
    print(f"## Case: {case_id}")
    print()
    print(f"**Status**: {case_report.get('status') if case_report else 'unknown'}")
    print(f"**Duration**: {case_report.get('duration_s') if case_report else '?'}s")
    if case_report:
        fails = case_report.get("fail_reasons") or case_report.get("failures") or []
        soft = case_report.get("soft_failures") or []
        if fails:
            print(f"**Fail reasons**:")
            for fr in fails:
                print(f"  - {fr}")
        if soft:
            print(f"**Soft failures**:")
            for sf in soft:
                print(f"  - {sf}")
    if hypothesis:
        print()
        print(f"**Root-cause hypothesis**: {hypothesis}")
    print()
    print(f"### Pipeline trace ({len(turns)} chat-completion turn(s), "
          f"{len(events)} detector event(s))")
    print()
    for i, t in enumerate(turns, 1):
        print(f"#### Turn {i} ({t['duration_ms']}ms, status {t['status']})")
        if i == 1:
            print(f"  user_request: {t['user_request']!r}")
        print(f"  history visible to model: {t['prior_asst_tool_turns']} asst-tool turn(s), "
              f"{t['prior_tool_results']} tool-result(s)")
        print(f"  proxy mutations applied: {t['mutations']}")
        if t["normalize_rewrites"]:
            print(f"  proxy tool-name rewrites: {t['normalize_rewrites']}")
        if t["reasoning"]:
            print(f"  reasoning ({len(t['reasoning'])}c): {t['reasoning'][:200]!r}")
        if t["content"]:
            print(f"  content ({len(t['content'])}c): {t['content'][:300]!r}")
        if t["tool_calls"]:
            print(f"  tool_calls:")
            for tc in t["tool_calls"]:
                print(f"    - {tc['name']}({short_args(tc['args'])})")
        print(f"  finish_reason: {t['finish']}")
        # Events that occurred after this turn (before the next)
        between = [e["rec"] for e in events
                   if e["after_turn"] == i - 1 or e["after_turn"] == i]
        for ev in between:
            evn = ev.get("event")
            if evn == "proxy_tool_name_normalized":
                rws = ev.get("rewrites") or []
                for rw in rws:
                    print(f"  >>> NORMALIZE: {rw.get('original')} -> {rw.get('rewritten')}")
            elif evn == "proxy_malformed_tool_call_detected":
                print(f"  >>> MALFORMED: tool={ev.get('matched_tool')} markers={ev.get('markers')}")
            elif evn == "proxy_loop_reflection_injected":
                ti = ev.get("trigger_info") or {}
                print(f"  >>> LOOP-REFLECT: {ti.get('trigger')} count={ti.get('count')}")
            elif evn == "proxy_loop_hard_stop":
                ti = ev.get("trigger_info") or {}
                print(f"  >>> LOOP-STOP: {ti.get('trigger')} count={ti.get('count')}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <case_id> <proxy_jsonl> [smoke_report.json]")
        sys.exit(1)
    case_id = sys.argv[1]
    proxy_log = sys.argv[2]
    report = sys.argv[3] if len(sys.argv) > 3 else ""
    main(case_id, proxy_log, report)
