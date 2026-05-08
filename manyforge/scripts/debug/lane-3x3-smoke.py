#!/usr/bin/env python3
"""3-round x 3-prompt smoke on both lanes — outcome aggregation.

For each lane:
  switch Composer container once
  for round in 1..3:
    for prompt in (P1, P2, P3):
      forceDiscardOverrides reset
      send prompt
      record:
        - bridge response status + elapsed
        - actual tool calls fired (Composer audit log delta-window)
        - whether draft was mutated (check_state_changed)
End: render a pass-rate table per lane × prompt, plus median timings.
"""
from __future__ import annotations

import os
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

COMPOSER_BASE = "http://127.0.0.1:9000"
PROMPTS = [
    ("P1", "add a repeat node as root"),
    ("P2", "add a box of size 1.0, 0.02, 0.25 in position 0.0, -0.15, 0.125"),
    ("P3", "add an upsert node at the end of pick_and_place sequence that places "
           "'graspable' object in the same position and orientation as the scene "
           "start, with the same original size"),
]
ROUNDS = 3
TIMEOUT_S = 140.0  # cascade: OpenClaw 120 → bridge effective 125 → Composer 130 → harness 140 (10s settle/debug buffer)


def _switch_composer(provider: str) -> None:
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
        "-v", f"{os.path.expanduser('~/workspaces/dev_ws/src/manyforge')}:/workspace",
        "-v", "manyforge_build-cache:/tmp/manyforge-build",
        "-w", "/workspace",
        "-e", f"MANYFORGE_ASSISTANT_PROVIDER={provider}",
        "-e", f"MANYFORGE_ASSISTANT_ENDPOINT_URL={endpoint}",
        "-e", "MANYFORGE_ASSISTANT_TIMEOUT_S=130",
        "manyforge-dev:latest",
        "bash", "-lc",
        f"python -m manyforge_composer "
        f"--catalog-path /workspace/manyforge_behavior/resources/node_catalog.yaml "
        f"--host 0.0.0.0 --port 9000 --hmi-port 8081 --mcp-http "
        f"--assistant-provider {provider} "
        f"--assistant-endpoint {endpoint} "
        f"--assistant-timeout-s 130",
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


def _send_chat(message: str, request_id: str) -> dict:
    body = json.dumps({
        "message": message,
        "mode": "provider",
        "conversationId": request_id,
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
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
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


def _fetch_composer_tool_logs_since(t0: float) -> list[dict]:
    """Fetch Composer container access log entries newer than t0 (epoch s)."""
    proc = subprocess.run(
        ["docker", "logs", "--since", str(int(t0) - 1), "manyforge-e2e-composer"],
        capture_output=True, timeout=10,
    )
    raw = proc.stdout.decode("utf-8", errors="replace")
    out: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"): continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        path = d.get("path") or ""
        if not path.startswith("/api/assistant/bridge/tools/"): continue
        # extract tool name and status
        out.append({
            "tool": path.split("/")[-1],
            "status": d.get("status_code"),
            "duration_ms": d.get("duration_ms"),
            "client": d.get("client"),
            "ts": d.get("timestamp"),
        })
    return out


def _draft_state() -> dict:
    """Return current Composer draft summary (program tree + scene)."""
    try:
        with urllib.request.urlopen(f"{COMPOSER_BASE}/api/program/state",
                                     timeout=10) as resp:
            d = json.load(resp)
        return d
    except Exception as exc:
        return {"error": str(exc)}


def _count_nodes(tree: dict) -> int:
    if not isinstance(tree, dict): return 0
    n = 1
    for child in tree.get("children") or []:
        n += _count_nodes(child)
    return n


def run_one(lane: str, prompt_id: str, prompt: str, round_idx: int) -> dict:
    _reset_program()
    time.sleep(0.5)
    pre_state = _draft_state()
    pre_tree_nodes = _count_nodes((pre_state.get("program") or {}).get("tree") or {})
    pre_scene_count = len(((pre_state.get("program") or {}).get("scene") or {}).get("resources") or [])
    t0 = time.time()
    rid = f"3x3-{lane}-{prompt_id}-r{round_idx}-{int(t0*1000)}"
    chat = _send_chat(prompt, rid)
    time.sleep(1.0)  # let any in-flight draft mutation settle
    post_state = _draft_state()
    post_tree_nodes = _count_nodes((post_state.get("program") or {}).get("tree") or {})
    post_scene_count = len(((post_state.get("program") or {}).get("scene") or {}).get("resources") or [])
    tool_logs = _fetch_composer_tool_logs_since(t0)
    succ_tools = [t for t in tool_logs if t.get("status") == 200]
    fail_tools = [t for t in tool_logs if t.get("status") and t.get("status") != 200]
    return {
        "lane": lane,
        "prompt_id": prompt_id,
        "round": round_idx,
        "request_id": rid,
        "chat_status": chat.get("status"),
        "elapsed_s": round(chat.get("elapsed_s", 0.0), 1),
        "tree_node_delta": post_tree_nodes - pre_tree_nodes,
        "scene_count_delta": post_scene_count - pre_scene_count,
        "tool_calls_ok": [t["tool"] for t in succ_tools],
        "tool_calls_fail": [(t["tool"], t["status"]) for t in fail_tools],
        "n_tools_ok": len(succ_tools),
        "n_tools_fail": len(fail_tools),
        "answer": (json.loads(chat["body"]).get("message", "") if chat.get("status") == 200 else chat["body"])[:160],
    }


def lane_pass(r: dict, prompt_id: str) -> bool:
    """Decide pass/fail for one run. Pass = (tools fired) AND (draft state changed)."""
    if r["n_tools_ok"] < 1:
        return False
    if prompt_id == "P2":
        return r["scene_count_delta"] >= 1
    # P1 + P3: tree change
    return r["tree_node_delta"] >= 1


def main() -> None:
    ts = int(time.time() * 1000)
    results: list[dict] = []
    for lane in ("direct", "openclaw"):
        provider = "nemoclaw" if lane == "direct" else "openclaw"
        print(f"\n{'='*78}\n[{lane.upper()}] switching Composer container ({provider})…",
              flush=True)
        _switch_composer(provider)
        time.sleep(1.5)
        for round_idx in range(1, ROUNDS + 1):
            for pid, prompt in PROMPTS:
                print(f"\n[{lane} round {round_idx}/{ROUNDS}] {pid}: {prompt[:60]}…",
                      flush=True)
                r = run_one(lane, pid, prompt, round_idx)
                p = lane_pass(r, pid)
                results.append({**r, "pass": p})
                marker = "✅" if p else "❌"
                print(f"    {marker} status={r['chat_status']} "
                      f"elapsed={r['elapsed_s']}s "
                      f"tools_ok={r['n_tools_ok']} fail={r['n_tools_fail']} "
                      f"tree_d={r['tree_node_delta']} scene_d={r['scene_count_delta']}",
                      flush=True)
                if not p:
                    print(f"      answer: {r['answer']!r}", flush=True)

    # Save full data
    out = Path(f"/tmp/lane_3x3_smoke_{ts}.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull data: {out}", flush=True)

    # Summary table
    print(f"\n{'='*78}\n  PASS-RATE TABLE\n{'='*78}")
    print(f"{'Lane':10}  {'Prompt':6}  {'Pass':6}  {'Median elapsed':14}  {'Tools fired (median)':22}")
    for lane in ("direct", "openclaw"):
        for pid, _ in PROMPTS:
            recs = [r for r in results if r["lane"] == lane and r["prompt_id"] == pid]
            n_pass = sum(1 for r in recs if r["pass"])
            elapsed = sorted(r["elapsed_s"] for r in recs)
            med_elapsed = elapsed[len(elapsed) // 2]
            tools_counts = sorted(r["n_tools_ok"] for r in recs)
            med_tools = tools_counts[len(tools_counts) // 2]
            print(f"{lane:10}  {pid:6}  {n_pass}/{len(recs):<4}  "
                  f"{med_elapsed:>10.1f} s    {med_tools:>4} tool(s)")
    print()


if __name__ == "__main__":
    main()
