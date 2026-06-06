#!/usr/bin/env python3
"""self_heal.py — TEST-ONLY (smoke harness).

Self-healing for chained corpus families (e.g. pnp_build). When a chained step
fails, restore the canonical state and rewrite the openclaw session transcript so
the model continues the chain as if the step had succeeded ("the model doesn't
notice"). Two halves, both driven by the SAME golden change (single source of truth):

  STATE  — replay-from-base: reset the live program+scene to base, then re-apply the
           ordered golden changes 1..N via the real bridge tools (canonical post-N).
  HISTORY— splice: rewrite the failed step's turn in the openclaw session `.jsonl`
           (in the sandbox) to a golden assistant tool-call + tool-result + "done".

NOTHING here runs in production: it only POSTs to a test composer's bridge tool
endpoints and edits sandbox session files, between steps, during a smoke run. The
"fabricate success" capability lives only in this test module.

See docs/self-healing-chain-harness.md.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time

from smoke_corpus_runner import _post_json, reset_program, fetch_catalog_hash


def resolve_container(substr: str) -> str | None:
    """Resolve the running sandbox container name from a name substring."""
    try:
        out = subprocess.run(
            ["docker", "ps", "--filter", f"name={substr}", "--format", "{{.Names}}"],
            capture_output=True, timeout=10,
        ).stdout.decode().strip().splitlines()
        return out[0] if out else None
    except Exception:
        return None


def apply_change(composer: str, tool: str, args: dict, catalog_hash: str, tag: str):
    envelope = {
        "requestId": f"selfheal-{tag}-{int(time.time()*1000)}",
        "assistantMode": "composer-assistant",
        "catalogHash": catalog_hash,
        "arguments": args,
    }
    return _post_json(f"{composer}/api/assistant/bridge/tools/{tool}", envelope, timeout=20.0)


def replay_to_canonical(composer: str, chain_spec: dict, upto_step: int) -> tuple[bool, str]:
    """Reset to base, then apply golden changes for steps 1..upto_step (the failed
    step inclusive) → canonical post-step state. Returns (ok, detail)."""
    base = chain_spec.get("base") or {}
    code, _ = reset_program(composer, base["deployment_path"], base["empty_program_path"])
    if code != 200:
        return False, f"reset-to-base HTTP {code}"
    catalog_hash = fetch_catalog_hash(composer)
    for step in (chain_spec.get("steps") or [])[:upto_step]:
        for ch in step.get("changes") or []:
            code, resp = apply_change(composer, ch["tool"], ch.get("args") or {}, catalog_hash, step["id"])
            ok = code == 200 and (not isinstance(resp, dict) or resp.get("success") is not False)
            if not ok:
                msg = ((resp.get("result") or {}).get("message") if isinstance(resp, dict) else None) or str(resp)
                return False, f"{step['id']}:{ch['tool']} -> {str(msg)[:200]}"
    return True, ""


# Runs INSIDE the sandbox (docker exec). Reads a golden-step payload on stdin,
# finds the last user message in the session .jsonl (the failed step's prompt),
# drops the model's wrong response after it, and appends a golden
# assistant(toolCall) -> toolResult -> assistant(text) chain, parent-linked.
_IN_SANDBOX_SPLICE = r'''
import sys, json, os
from datetime import datetime, timezone
p = json.load(sys.stdin)
path, calls, text = p["session_path"], p["calls"], p["text"]
try:
    lines = [json.loads(l) for l in open(path) if l.strip()]
except FileNotFoundError:
    print("NO_SESSION_FILE"); sys.exit(2)
u_idx = None
for i, d in enumerate(lines):
    m = d.get("message")
    if isinstance(m, dict) and m.get("role") == "user":
        u_idx = i
if u_idx is None:
    print("NO_USER_MSG"); sys.exit(3)
keep = lines[: u_idx + 1]
parent = keep[-1]["id"]
h8 = lambda: os.urandom(4).hex()
cid = lambda: os.urandom(16).hex()
ts = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
out = list(keep)
for c in calls:
    inner = {"id": "manyforge__" + c["tool"], "args": c["args"]}
    aid = h8()
    out.append({"type": "message", "id": aid, "parentId": parent, "timestamp": ts(),
                "message": {"role": "assistant", "content": [
                    {"type": "toolCall", "id": cid(), "name": "tool_call",
                     "arguments": inner, "partialArgs": json.dumps(inner)}]}})
    rid = h8()
    res = json.dumps({"toolId": c["tool"], "success": True, "result": {"kind": "ok"}})
    out.append({"type": "message", "id": rid, "parentId": aid, "timestamp": ts(),
                "message": {"role": "toolResult", "content": [{"type": "text", "text": res}]}})
    parent = rid
out.append({"type": "message", "id": h8(), "parentId": parent, "timestamp": ts(),
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}})
with open(path, "w") as f:
    for d in out:
        f.write(json.dumps(d) + "\n")
print("SPLICED dropped=%d added=%d total=%d" % (len(lines) - len(keep), len(out) - len(keep), len(out)))
'''


def splice_golden_turn(container: str, agent: str, conversation_id: str, golden_step: dict,
                       text: str = "Done.") -> tuple[bool, str]:
    """Rewrite the failed step's turn in the openclaw session transcript (in the
    sandbox) to the golden tool-call(s). Returns (ok, detail)."""
    session_path = f"/sandbox/.openclaw/agents/{agent}/sessions/{conversation_id}.jsonl"
    payload = {
        "session_path": session_path,
        "calls": [{"tool": c["tool"], "args": c.get("args") or {}}
                  for c in (golden_step.get("changes") or [])],
        "text": text,
    }
    if not payload["calls"]:
        return True, "no golden calls (no-op step)"
    # `python3 -c <script>` runs the splice; the JSON payload is piped on stdin and
    # the script reads it via sys.stdin. (-c program and stdin are independent.)
    try:
        proc = subprocess.run(
            ["docker", "exec", "-i", container, "python3", "-c", _IN_SANDBOX_SPLICE],
            input=json.dumps(payload).encode(),
            capture_output=True, timeout=30,
        )
    except Exception as e:
        return False, f"docker exec failed: {e}"
    out = (proc.stdout or b"").decode(errors="replace").strip()
    err = (proc.stderr or b"").decode(errors="replace").strip()
    if proc.returncode != 0 or not out.startswith("SPLICED"):
        return False, f"splice rc={proc.returncode} out={out!r} err={err[:200]!r}"
    return True, out


def self_heal(composer: str, container: str, agent: str, conversation_id: str,
              chain_spec: dict, failed_step: int) -> tuple[bool, str]:
    """Full self-heal after a chained step failure: canonical state replay + golden
    transcript splice. `failed_step` is 1-based (the corpus chain_step)."""
    ok, detail = replay_to_canonical(composer, chain_spec, failed_step)
    if not ok:
        return False, f"state replay failed: {detail}"
    steps = chain_spec.get("steps") or []
    if failed_step - 1 >= len(steps):
        return False, f"no golden step for index {failed_step}"
    ok, detail = splice_golden_turn(container, agent, conversation_id, steps[failed_step - 1])
    if not ok:
        return False, f"transcript splice failed: {detail}"
    return True, detail
