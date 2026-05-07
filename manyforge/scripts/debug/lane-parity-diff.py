#!/usr/bin/env python3
"""Lane-parity diff harness — FULL-FIDELITY edition.

Runs the SAME assistant prompt on both lanes (direct + OpenClaw)
sequentially, captures the *exact* chat-completion request that
arrives at vLLM on each lane, and diffs every field.

This version captures EVERYTHING so we can pinpoint why the OpenClaw
lane decodes differently from the direct lane against the same model:

  - the FULL system + user prompt content (no truncation)
  - the FULL tools[] array including every schema field
  - tool_choice, response_format, chat_template_kwargs verbatim
  - all sampling params (temperature, top_p, top_k, max_tokens,
    max_output_tokens, stop, n, seed, presence/frequency_penalty,
    logprobs, top_logprobs, logit_bias, user, stream, stream_options,
    metadata, instructions, reasoning, …)
  - any non-standard top-level fields
  - the FULL response body (incl. choices[].message + tool_calls)

Per-lane outputs (written to /tmp):
  lane_parity_<ts>_direct_request_<turn>.json
  lane_parity_<ts>_direct_response_<turn>.json
  lane_parity_<ts>_openclaw_request_<turn>.json
  lane_parity_<ts>_openclaw_response_<turn>.json
  lane_parity_<ts>_summary.json     ← combined summary + diff
  lane_parity_<ts>_diff.txt         ← human-readable side-by-side

Required setup:
  - Direct-lane vLLM proxy on :8001 logging to /tmp/vllm_direct_proxy.jsonl
  - OpenClaw-lane vLLM proxy on :8002 logging to /tmp/vllm_openclaw_proxy.jsonl
  - manyforge_assistant_bridge: BRIDGE_UPSTREAM_BASE_URL=http://127.0.0.1:8001/v1
  - OpenClaw config: models.providers.inference.baseUrl=http://host.openshell.internal:8002/v1

Usage:
  python3 lane_parity_diff.py "add a repeat node as root"
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


COMPOSER_BASE = "http://127.0.0.1:9000"
DIRECT_PROXY_LOG = Path("/tmp/vllm_direct_proxy.jsonl")
OPENCLAW_PROXY_LOG = Path("/tmp/vllm_openclaw_proxy.jsonl")

# Top-level fields we explicitly recognise on a chat-completion body.
# Anything outside this set is reported as "extra_fields".
CHAT_KNOWN_FIELDS = {
    "model", "messages", "input", "tools", "tool_choice",
    "stream", "stream_options",
    "max_tokens", "max_output_tokens",
    "temperature", "top_p", "top_k",
    "stop", "n", "seed",
    "presence_penalty", "frequency_penalty",
    "logprobs", "top_logprobs", "logit_bias",
    "user", "response_format",
    "chat_template_kwargs",
    "metadata", "instructions", "reasoning",
    "parallel_tool_calls", "tool_resources",
    "extra_body", "service_tier",
}

SAMPLING_FIELDS = [
    "temperature", "top_p", "top_k",
    "max_tokens", "max_output_tokens",
    "stop", "n", "seed",
    "presence_penalty", "frequency_penalty",
    "logprobs", "top_logprobs", "logit_bias",
    "stream", "stream_options",
    "response_format",
    "parallel_tool_calls",
]


def _file_offset(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _read_since(path: Path, offset: int) -> list[dict]:
    if offset < 0:
        offset = 0
    try:
        with path.open("rb") as fh:
            fh.seek(offset)
            raw = fh.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return []
    out: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _switch_composer(provider: str) -> None:
    """Restart the Composer container with the given assistant provider.

    nemoclaw -> direct lane (port 8100)
    openclaw -> OpenClaw lane (port 8200)
    """
    if provider == "nemoclaw":
        endpoint = "http://127.0.0.1:8100/v1/manyforge/assistant"
    else:
        endpoint = "http://127.0.0.1:8200/v1/manyforge/assistant"

    subprocess.run(["docker", "rm", "-f", "manyforge-e2e-composer"],
                   capture_output=True, timeout=20)
    subprocess.run([
        "docker", "run", "-d", "--rm",
        "--name", "manyforge-e2e-composer",
        "--network", "host",
        "-v", "/home/tndlux/workspaces/dev_ws/src/manyforge:/workspace",
        "-v", "manyforge_build-cache:/tmp/manyforge-build",
        "-w", "/workspace",
        "-e", f"MANYFORGE_ASSISTANT_PROVIDER={provider}",
        "-e", f"MANYFORGE_ASSISTANT_ENDPOINT_URL={endpoint}",
        "-e", "MANYFORGE_ASSISTANT_TIMEOUT_S=60",
        "manyforge-dev:latest",
        "bash", "-lc",
        f"python -m manyforge_composer "
        f"--catalog-path /workspace/manyforge_behavior/resources/node_catalog.yaml "
        f"--host 0.0.0.0 --port 9000 --hmi-port 8081 --mcp-http "
        f"--assistant-provider {provider} "
        f"--assistant-endpoint {endpoint} "
        f"--assistant-timeout-s 60",
    ], capture_output=True, timeout=30, check=True)
    for _ in range(15):
        try:
            urllib.request.urlopen(f"{COMPOSER_BASE}/api/infra/status", timeout=3)
            return
        except Exception:
            time.sleep(1)


def _reset_program() -> None:
    body = json.dumps({
        "path": "/workspace/examples/pick_and_place_ur10e_robotiq.program.yaml",
        "deploymentPath": "/workspace/examples/assistant_modes_scene_authoring.deployment.yaml",
        "forceDiscardOverrides": True,
    }).encode()
    req = urllib.request.Request(
        f"{COMPOSER_BASE}/api/program/load",
        data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=30).read()


def _send_chat(message: str, request_id: str, conversation_id: str,
               timeout_s: float = 90.0) -> dict:
    body = json.dumps({
        "message": message,
        "mode": "provider",
        "conversationId": conversation_id,
        "requestId": request_id,
        "assistantMode": "composer-assistant",
    }).encode()
    req = urllib.request.Request(
        f"{COMPOSER_BASE}/api/assistant/chat",
        data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return {
                "status": resp.status,
                "body": resp.read().decode("utf-8", errors="replace"),
                "elapsed_s": time.time() - started,
            }
    except Exception as exc:
        return {
            "status": -1,
            "body": f"<error {type(exc).__name__}: {exc}>",
            "elapsed_s": time.time() - started,
        }


def run_lane(lane: str, message: str) -> dict:
    """Run one prompt on one lane; return a dict with the captured
    vLLM-bound requests + responses + the assistant final answer."""
    assert lane in ("direct", "openclaw"), lane
    provider = "nemoclaw" if lane == "direct" else "openclaw"
    log_path = DIRECT_PROXY_LOG if lane == "direct" else OPENCLAW_PROXY_LOG

    print(f"\n[parity] switching composer to {lane} lane…", flush=True)
    _switch_composer(provider)
    print(f"[parity] composer ready on {lane}", flush=True)

    _reset_program()
    time.sleep(1)
    bl = _file_offset(log_path)
    rid = f"parity-{lane}-{int(time.time() * 1000)}"
    print(f"[parity] sending: {message!r} on {lane} ({rid})", flush=True)
    chat = _send_chat(message, rid, rid, timeout_s=90.0)
    print(f"[parity] {lane}: {chat['elapsed_s']:.1f}s status={chat['status']}",
          flush=True)
    time.sleep(1.5)
    records = _read_since(log_path, bl)
    print(f"[parity] {lane}: {len(records)} vLLM-bound chat-completion(s) captured",
          flush=True)
    return {
        "lane": lane,
        "request_id": rid,
        "chat": chat,
        "vllm_records": records,
    }


def _full_summary(req_record: dict) -> dict:
    """Full-fidelity summary of one vLLM-bound chat-completion record.

    Preserves complete prompt content, complete tool schemas, every
    sampling parameter. Nothing is truncated."""
    request = req_record.get("request", {}) or {}
    response = req_record.get("response", {}) or {}
    body = request.get("body") or {}
    if not isinstance(body, dict):
        return {"error": "request body not a dict",
                "raw_excerpt": request.get("body_raw_excerpt", "")[:500]}

    out: dict[str, Any] = {
        "path": request.get("path"),
        "method": request.get("method"),
        "model": body.get("model"),
        "tool_choice": body.get("tool_choice"),
        "chat_template_kwargs": body.get("chat_template_kwargs"),
    }

    # Sampling params (preserved verbatim, including absent → None).
    sampling: dict[str, Any] = {}
    for k in SAMPLING_FIELDS:
        if k in body:
            sampling[k] = body.get(k)
    out["sampling"] = sampling

    # Messages — FULL content, no truncation.
    msgs = body.get("messages") or []
    out["messages_count"] = len(msgs)
    out["messages_roles"] = [m.get("role") for m in msgs if isinstance(m, dict)]
    out["messages_full"] = []
    for m in msgs:
        if not isinstance(m, dict):
            out["messages_full"].append({"raw": str(m)[:1000]})
            continue
        c = m.get("content")
        rec: dict[str, Any] = {"role": m.get("role")}
        if isinstance(c, str):
            rec["content_chars"] = len(c)
            rec["content"] = c  # FULL content
        elif c is None:
            rec["content"] = None
        else:
            rec["content_type"] = type(c).__name__
            rec["content"] = c  # full structured content (list/dict/etc)
        # Tool-call related fields (assistant turn) and tool-result fields.
        for k in ("name", "tool_call_id", "tool_calls", "function_call",
                  "refusal", "audio", "annotations"):
            if k in m:
                rec[k] = m[k]
        out["messages_full"].append(rec)

    # Tools — FULL schema, no truncation.
    tools = body.get("tools") or []
    out["tools_count"] = len(tools)
    tools_summary: list[dict[str, Any]] = []
    tools_full: list[Any] = []
    for t in tools:
        if not isinstance(t, dict):
            tools_summary.append({"raw_type": type(t).__name__})
            tools_full.append(t)
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else {}
        name = t.get("name") or fn.get("name")
        desc = t.get("description") or fn.get("description") or ""
        params = t.get("parameters") or fn.get("parameters") or {}
        prop_keys: list[str] = []
        if isinstance(params, dict):
            props = params.get("properties")
            if isinstance(props, dict):
                prop_keys = list(props.keys())
        tools_summary.append({
            "type": t.get("type"),
            "name": name,
            "desc_chars": len(desc) if isinstance(desc, str) else 0,
            "param_property_count": len(prop_keys),
            "param_property_keys": prop_keys,
            "required": (params or {}).get("required") if isinstance(params, dict) else None,
        })
        tools_full.append(t)
    out["tools_summary"] = tools_summary
    out["tools_full"] = tools_full

    # Any non-standard fields.
    out["extra_fields"] = sorted(set(body.keys()) - CHAT_KNOWN_FIELDS - {"messages", "tools"})
    out["extra_field_values"] = {
        k: body.get(k) for k in out["extra_fields"]
    }

    # Response (status + content).
    resp_body = response.get("body")
    out["response_status"] = response.get("status")
    out["response_duration_ms"] = response.get("duration_ms")
    out["response_body"] = resp_body  # full
    # Convenience: extracted assistant message + tool calls.
    if isinstance(resp_body, dict):
        choices = resp_body.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            out["response_assistant_message"] = msg
            out["response_finish_reason"] = choices[0].get("finish_reason")
        out["response_usage"] = resp_body.get("usage")

    return out


def _diff_keys(a: set, b: set) -> tuple[list, list, list]:
    return sorted(a - b), sorted(b - a), sorted(a & b)


def _diff_value(name: str, dv: Any, ov: Any, lines: list[str]) -> None:
    same = dv == ov
    marker = "  " if same else "❗"
    ds = json.dumps(dv, default=str) if not isinstance(dv, str) else repr(dv)
    os_ = json.dumps(ov, default=str) if not isinstance(ov, str) else repr(ov)
    lines.append(f"{marker} {name:30} | direct={ds[:80]}")
    lines.append(f"   {'':30} | openclaw={os_[:80]}")


def render_diff(direct: dict, openclaw: dict, ts: int) -> str:
    """Return a human-readable diff string. Also writes per-lane
    request/response bodies to disk for byte-level inspection."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  LANE PARITY DIFF (full-fidelity)")
    lines.append("=" * 78)

    d_records = direct["vllm_records"]
    o_records = openclaw["vllm_records"]
    lines.append(f"\nvLLM chat-completions per lane: "
                 f"direct={len(d_records)} openclaw={len(o_records)}")

    if not d_records or not o_records:
        lines.append("  (need >=1 chat-completion captured per lane to diff)")
        return "\n".join(lines)

    # Summarise each turn separately.
    d_summaries = [_full_summary(r) for r in d_records]
    o_summaries = [_full_summary(r) for r in o_records]

    # Write FULL bodies to disk for offline inspection.
    out_dir = Path("/tmp")
    for i, r in enumerate(d_records):
        (out_dir / f"lane_parity_{ts}_direct_request_{i}.json").write_text(
            json.dumps(r.get("request"), indent=2, default=str))
        (out_dir / f"lane_parity_{ts}_direct_response_{i}.json").write_text(
            json.dumps(r.get("response"), indent=2, default=str))
    for i, r in enumerate(o_records):
        (out_dir / f"lane_parity_{ts}_openclaw_request_{i}.json").write_text(
            json.dumps(r.get("request"), indent=2, default=str))
        (out_dir / f"lane_parity_{ts}_openclaw_response_{i}.json").write_text(
            json.dumps(r.get("response"), indent=2, default=str))

    # First-turn diff (load-bearing — this is the call that decides
    # whether the model emits a tool call vs prose).
    d0 = d_summaries[0]
    o0 = o_summaries[0]

    # ---- Top-level scalars ----
    lines.append("\nTOP-LEVEL FIELDS (turn 0)")
    for f in ("path", "model", "tool_choice", "chat_template_kwargs"):
        _diff_value(f, d0.get(f), o0.get(f), lines)

    # ---- Sampling params ----
    lines.append("\nSAMPLING PARAMS (turn 0)")
    sd = d0.get("sampling") or {}
    so = o0.get("sampling") or {}
    sampling_keys = sorted(set(sd) | set(so))
    if not sampling_keys:
        lines.append("   (both lanes sent zero sampling params — vLLM defaults)")
    for k in sampling_keys:
        _diff_value(k, sd.get(k, "<absent>"), so.get(k, "<absent>"), lines)

    # ---- Tools ----
    lines.append("\nTOOLS (turn 0)")
    lines.append(f"  count: direct={d0.get('tools_count')} openclaw={o0.get('tools_count')}")
    d_names = [t.get("name") for t in (d0.get("tools_summary") or [])]
    o_names = [t.get("name") for t in (o0.get("tools_summary") or [])]
    only_d, only_o, common = _diff_keys(set(d_names), set(o_names))
    if only_d:
        lines.append(f"  ONLY in direct ({len(only_d)}): {only_d}")
    if only_o:
        lines.append(f"  ONLY in openclaw ({len(only_o)}): {only_o}")
    if not only_d and not only_o:
        lines.append(f"  ✓ same {len(common)} tool names")

    # Per-tool field-by-field diff (using NAME as key).
    d_by_name = {t.get("name"): t for t in (d0.get("tools_summary") or [])}
    o_by_name = {t.get("name"): t for t in (o0.get("tools_summary") or [])}
    for name in sorted(set(d_by_name) | set(o_by_name)):
        d_t = d_by_name.get(name)
        o_t = o_by_name.get(name)
        if d_t == o_t:
            continue
        lines.append(f"  TOOL DIFF: {name!r}")
        for f in ("type", "desc_chars", "param_property_count",
                  "param_property_keys", "required"):
            dv = (d_t or {}).get(f)
            ov = (o_t or {}).get(f)
            if dv != ov:
                lines.append(f"    {f}: direct={json.dumps(dv, default=str)[:100]} "
                             f"vs openclaw={json.dumps(ov, default=str)[:100]}")

    # Full tool schema byte-equality check (fast path).
    d_full = d0.get("tools_full") or []
    o_full = o0.get("tools_full") or []
    same_full_tools = json.dumps(d_full, sort_keys=True, default=str) == \
                      json.dumps(o_full, sort_keys=True, default=str)
    lines.append(f"  full tools[] byte-equal? {'YES' if same_full_tools else 'NO'}")
    if not same_full_tools:
        lines.append(f"  (see lane_parity_{ts}_*_request_0.json for full schemas)")

    # ---- Messages ----
    lines.append("\nMESSAGES (turn 0)")
    d_msgs = d0.get("messages_full") or []
    o_msgs = o0.get("messages_full") or []
    lines.append(f"  count: direct={len(d_msgs)} openclaw={len(o_msgs)}")
    lines.append(f"  roles: direct={d0.get('messages_roles')}")
    lines.append(f"         openclaw={o0.get('messages_roles')}")

    for i in range(max(len(d_msgs), len(o_msgs))):
        d_m = d_msgs[i] if i < len(d_msgs) else {}
        o_m = o_msgs[i] if i < len(o_msgs) else {}
        d_role = d_m.get("role")
        o_role = o_m.get("role")
        d_content = d_m.get("content") if d_m else None
        o_content = o_m.get("content") if o_m else None
        d_chars = (len(d_content) if isinstance(d_content, str)
                   else len(json.dumps(d_content, default=str)) if d_content else 0)
        o_chars = (len(o_content) if isinstance(o_content, str)
                   else len(json.dumps(o_content, default=str)) if o_content else 0)
        lines.append(f"  message[{i}] direct={d_role!r}/{d_chars}c "
                     f"openclaw={o_role!r}/{o_chars}c")
        if isinstance(d_content, str) and isinstance(o_content, str):
            if d_content == o_content:
                lines.append("    ✓ identical content")
            else:
                lines.append("    ❗ content DIFFERS — first divergence:")
                # Find first byte that differs.
                for j in range(min(len(d_content), len(o_content))):
                    if d_content[j] != o_content[j]:
                        s = max(0, j - 60)
                        lines.append(f"    @char {j}: ...{d_content[s:j+60]!r}")
                        lines.append(f"           vs ...{o_content[s:j+60]!r}")
                        break
                else:
                    lines.append(f"    one is prefix of other (lengths "
                                 f"{len(d_content)} vs {len(o_content)})")
        elif d_content != o_content:
            lines.append(f"    structural diff (non-string content)")

    # ---- Extras ----
    if d0.get("extra_fields") or o0.get("extra_fields"):
        lines.append("\nEXTRA NON-STANDARD FIELDS (turn 0)")
        lines.append(f"  direct: {d0.get('extra_field_values')}")
        lines.append(f"  openclaw: {o0.get('extra_field_values')}")

    # ---- Response ----
    lines.append("\nRESPONSE (turn 0)")
    lines.append(f"  status: direct={d0.get('response_status')} "
                 f"openclaw={o0.get('response_status')}")
    lines.append(f"  finish_reason: direct={d0.get('response_finish_reason')!r} "
                 f"openclaw={o0.get('response_finish_reason')!r}")
    lines.append(f"  duration_ms: direct={d0.get('response_duration_ms')} "
                 f"openclaw={o0.get('response_duration_ms')}")
    d_msg = d0.get("response_assistant_message") or {}
    o_msg = o0.get("response_assistant_message") or {}
    lines.append(f"  assistant.content chars: direct="
                 f"{len(d_msg.get('content') or '')} openclaw="
                 f"{len(o_msg.get('content') or '')}")
    d_tcs = d_msg.get("tool_calls") or []
    o_tcs = o_msg.get("tool_calls") or []
    lines.append(f"  tool_calls count: direct={len(d_tcs)} openclaw={len(o_tcs)}")
    for j, tc in enumerate(d_tcs):
        fn = (tc.get("function") or {})
        lines.append(f"    direct[{j}].function.name = {fn.get('name')!r}, "
                     f"args_chars={len(fn.get('arguments') or '')}")
    for j, tc in enumerate(o_tcs):
        fn = (tc.get("function") or {})
        lines.append(f"    openclaw[{j}].function.name = {fn.get('name')!r}, "
                     f"args_chars={len(fn.get('arguments') or '')}")
    if d_msg.get("content"):
        lines.append(f"  direct content (first 400): {d_msg.get('content')[:400]!r}")
    if o_msg.get("content"):
        lines.append(f"  openclaw content (first 400): {o_msg.get('content')[:400]!r}")

    # ---- Subsequent turns ----
    if len(d_summaries) > 1 or len(o_summaries) > 1:
        lines.append(f"\nSUBSEQUENT TURNS")
        lines.append(f"  direct turns: {len(d_summaries)}")
        for i, s in enumerate(d_summaries):
            am = s.get("response_assistant_message") or {}
            lines.append(f"    direct[{i}] tool_choice={s.get('tool_choice')} "
                         f"finish={s.get('response_finish_reason')} "
                         f"tool_calls={len(am.get('tool_calls') or [])}")
        lines.append(f"  openclaw turns: {len(o_summaries)}")
        for i, s in enumerate(o_summaries):
            am = s.get("response_assistant_message") or {}
            lines.append(f"    openclaw[{i}] tool_choice={s.get('tool_choice')} "
                         f"finish={s.get('response_finish_reason')} "
                         f"tool_calls={len(am.get('tool_calls') or [])}")

    # ---- Composer-level final answers ----
    lines.append("\nASSISTANT FINAL ANSWERS (Composer-side response)")
    try:
        d_resp = json.loads(direct["chat"]["body"])
        d_ans = (d_resp.get("message") or "(empty)")
        lines.append(f"  direct ({len(d_ans)}c): {d_ans[:300]!r}")
    except Exception as e:
        lines.append(f"  direct parse err: {e}")
    try:
        o_resp = json.loads(openclaw["chat"]["body"])
        o_ans = (o_resp.get("message") or "(empty)")
        lines.append(f"  openclaw ({len(o_ans)}c): {o_ans[:300]!r}")
    except Exception as e:
        lines.append(f"  openclaw parse err: {e}")

    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} '<prompt>'", file=sys.stderr)
        sys.exit(2)
    prompt = sys.argv[1]
    ts = int(time.time() * 1000)

    direct = run_lane("direct", prompt)
    openclaw = run_lane("openclaw", prompt)

    # Combined summary JSON for offline tooling.
    summary_path = Path(f"/tmp/lane_parity_{ts}_summary.json")
    summary_path.write_text(json.dumps({
        "prompt": prompt,
        "direct": direct,
        "openclaw": openclaw,
    }, indent=2, default=str))
    print(f"\nfull capture: {summary_path}", flush=True)

    diff_text = render_diff(direct, openclaw, ts)
    diff_path = Path(f"/tmp/lane_parity_{ts}_diff.txt")
    diff_path.write_text(diff_text)
    print(diff_text)
    print(f"\ndiff written: {diff_path}", flush=True)


if __name__ == "__main__":
    main()
