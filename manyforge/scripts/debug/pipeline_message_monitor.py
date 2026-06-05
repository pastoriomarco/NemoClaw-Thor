#!/usr/bin/env python3
"""Live inter-component message monitor for the ManyForge assistant pipeline.

Prints the ACTUAL message content on every hop (not pass/fail counts):
  runner -> composer (HTTP: /api/assistant/chat, /api/program/*, MCP tools)
  composer <-> bridge/agent (bridge log: openclaw_request_started/complete/timeout)
  agent/bridge -> vllm-proxy(:8000) -> llama.cpp(:8050)
        (proxy JSONL: model, prompt/compl tokens, finish, tool_calls, MUTATIONS)
  model (llama.cpp log: prompt eval / eval timings, tool stops)

IMPORTANT (field paths — verified 2026-06-05; getting these wrong produces false
findings, e.g. "proxy bypassed" or a "510" parsed out of a 51004.9ms duration):
  - vllm-proxy.jsonl entry = {ts, request{method,path,mutation{mutations},body{model,messages,tools,...}}, response{status,usage,choices}}
    The request body (model/messages) is under request.BODY.*, NOT request.*.
    Mutations are under request.mutation.mutations. HTTP status is response.status.
  - composer status comes from the uvicorn access line  "POST <path> HTTP/1.1" <status>
    NOT from the structured "... <status> <duration>ms" line (duration digits != status).

Cross-check any surprising conclusion against the raw log by requestId before reporting.

Usage:
  pipeline_message_monitor.py [--proxy PATH] [--bridge PATH] [--composer NAME]
      [--model NAME] [--done-file PATH] [--seconds N] [--poll N]
"""
import argparse, json, os, re, subprocess, time

UVICORN = re.compile(r'"(POST|GET|PUT|DELETE) (\S+) HTTP/[\d.]+" (\d{3})')


def tailer(path):
    off = [0]
    def read_new():
        try:
            sz = os.path.getsize(path)
        except OSError:
            return []
        if sz < off[0]:
            off[0] = 0          # file truncated/rotated → re-read from start
        if sz == off[0]:
            return []
        with open(path, "r", errors="replace") as f:
            f.seek(off[0]); data = f.read(); off[0] = f.tell()
        return data.splitlines()
    return read_new


def proxy_line(ln):
    try:
        d = json.loads(ln)
    except Exception:
        return None
    req = d.get("request", {}) or {}
    if req.get("path") != "/v1/chat/completions":
        return None                                  # skip /v1/models probes etc.
    body = req.get("body", {}) or {}
    resp = d.get("response", {}) or {}
    u = resp.get("usage", {}) or {}
    ch = (resp.get("choices") or [{}])[0]
    tcs = [c.get("function", {}).get("name") for c in (ch.get("message", {}).get("tool_calls") or [])]
    muts = (req.get("mutation") or {}).get("mutations") or {}
    mut_s = [f"{k} {v.get('before')}->{v.get('after')}" for k, v in muts.items()] if isinstance(muts, dict) else []
    parts = [f"model={body.get('model')}", f"status={resp.get('status')}",
             f"prompt_tok={u.get('prompt_tokens')}", f"compl_tok={u.get('completion_tokens')}",
             f"finish={ch.get('finish_reason')}"]
    if tcs:
        parts.append(f"tool_calls={tcs}")
    if mut_s:
        parts.append(f"MUT=[{', '.join(mut_s)}]")
    return "  [agent->proxy:8000->model:8050] " + " ".join(str(p) for p in parts)


def docker_since(name, since):
    try:
        o = subprocess.run(["docker", "logs", "--since", since, name],
                           capture_output=True, text=True, timeout=8)
        return (o.stdout + o.stderr).splitlines()
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default="/tmp/manyforge-assistant-e2e/vllm-proxy.jsonl")
    ap.add_argument("--bridge", default="/tmp/manyforge-assistant-e2e/known-good-bridge.log")
    ap.add_argument("--composer", default="manyforge-e2e-composer")
    ap.add_argument("--model", default="manyforge-e2e-vllm")
    ap.add_argument("--done-file", default="")
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("--poll", type=int, default=8)
    a = ap.parse_args()

    print("=== LIVE inter-component message monitor ===", flush=True)
    rp, rb = tailer(a.proxy), tailer(a.bridge)
    deadline = time.time() + a.seconds
    while time.time() < deadline:
        for ln in rp():
            s = proxy_line(ln)
            if s:
                print(s, flush=True)
        for ln in docker_since(a.composer, f"{a.poll + 2}s"):
            m = UVICORN.search(ln)
            if m and m.group(2).startswith("/api/") and "/api/runtime/external" not in m.group(2):
                print(f"  [runner/agent->composer] {m.group(1)} {m.group(2)} -> {m.group(3)}", flush=True)
        for ln in rb():
            low = ln.lower()
            if "healthz" in low:
                continue
            if any(k in low for k in ("openclaw_request", "error", "forbidden", "403", "timeout", "loopreflection")):
                print(f"  [bridge<->agent] {ln[:170]}", flush=True)
        for ln in docker_since(a.model, f"{a.poll + 2}s"):
            if any(k in ln for k in ("prompt eval time", "eval time", "stop processing", "error")) and "slot update_slots" not in ln:
                print(f"  [model:8050] {ln.strip()[:150]}", flush=True)
        if a.done_file and os.path.exists(a.done_file):
            print(">>> done-file present — stopping", flush=True)
            break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
