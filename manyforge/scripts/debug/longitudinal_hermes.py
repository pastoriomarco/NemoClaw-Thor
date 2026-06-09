#!/usr/bin/env python3
"""longitudinal_hermes.py — the Hermes-native longitudinal bake-off harness.

Per THREE-LANE-MIGRATION-PLAN.md §9.3 and Phase 4. Unlike the per-turn smoke
corpus (which judges every lane on stateless per-case pass rate), this harness
judges the Hermes lane on the metric it is *allowed* to win on: session-over-
session compounding. It drives N sessions × M turns of a repeated-pattern
corpus through the Composer assistant API (which must be pointed at the Hermes
lane, ``ASSISTANT_PROVIDER=hermes``) and measures whether later sessions:

  - complete the repeated task in FEWER turns (turns-to-completion trend),
  - emit skill creations (skill emergence), and
  - hit memory (a later session completing in 1 turn / via an emerged skill).

The distinctive-Hermes signals (skill creations, memory writes, cron fires,
delegations) are read from the bridge's ``hermes-session-events.jsonl``
(written by lanes/hermes/progress_observer.py + service.py), grouped by the
conversationId the harness assigns per session.

Memory is a determinism hazard (plan §3): pass ``--reset-hermes-state`` to clear
``/sandbox/.hermes/{memories,sessions,runtime/state.db}`` before the run so the
longitudinal sequence starts from a clean baseline. Without it the run
accumulates on top of whatever state is already there.

Usage:
    longitudinal_hermes.py [--sessions 10] [--turns-per-session 8]
        [--corpus longitudinal_corpus.yaml] [--composer http://127.0.0.1:9000]
        [--sandbox hermes-assistant] [--reset-hermes-state] [--report out.json]

Stdlib-only. The corpus may be JSON (parsed with stdlib) or YAML (needs
PyYAML); if neither a --corpus nor PyYAML is available, the embedded default
corpus is used so the harness always runs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

DEFAULT_COMPOSER = "http://127.0.0.1:9000"
DEFAULT_SESSION_EVENTS = "/tmp/manyforge-assistant-e2e/hermes-session-events.jsonl"

# Embedded default corpus — a repeated pattern designed for compounding. Each
# session attempts the SAME task (rephrased per turn) until the expected tool
# fires; later sessions should need fewer turns if memory/skills compound.
DEFAULT_CORPUS: dict[str, Any] = {
    "api_version": "manyforge.longitudinal_corpus.v0",
    "pattern": {
        "name": "wrap_root_in_repeat",
        "expect_tool": "tree_draft_wrap_node",
        "turns": [
            "Wrap the current root node in a new repeat node named loop_root, configured for infinite cycles.",
            "Make loop_root (a repeat node, infinite cycles) the new root, with the existing tree as its child.",
            "Insert a repeat node loop_root above the root so the whole behavior tree repeats forever.",
        ],
    },
    # Stated once on session 1, turn 1, then never repeated — later sessions
    # test whether Hermes remembers the preference (memory hit).
    "preference": {
        "statement": (
            "Remember this for future requests: I always want the root behavior "
            "wrapped in a repeat node named loop_root with infinite cycles."
        ),
    },
}


# ---- HTTP (stdlib; mirrors smoke_corpus_runner) ------------------------------


def _post_json(url: str, body: dict, timeout: float = 200.0) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(raw)
            except ValueError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8", "replace"))
        except Exception:
            return exc.code, str(exc)
    except Exception as exc:  # noqa: BLE001
        return -1, f"<error {type(exc).__name__}: {exc}>"


def send_chat(composer: str, prompt: str, conversation_id: str, timeout_s: float) -> tuple[int, Any]:
    body = {
        "message": prompt,
        "mode": "provider",
        "conversationId": conversation_id,
        "requestId": f"{conversation_id}-{uuid.uuid4().hex[:8]}",
        "assistantMode": "composer-assistant",
        "timeoutSeconds": int(timeout_s),
    }
    return _post_json(f"{composer}/api/assistant/chat", body, timeout=timeout_s + 5.0)


# ---- corpus + state ----------------------------------------------------------


def load_corpus(path: str | None) -> dict[str, Any]:
    if not path:
        return DEFAULT_CORPUS
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml  # optional
    except ImportError:
        print(f"  ! PyYAML unavailable; ignoring {path}, using embedded default corpus")
        return DEFAULT_CORPUS
    return yaml.safe_load(text)


def resolve_container(sandbox: str) -> str | None:
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:  # noqa: BLE001
        return None
    for name in out.splitlines():
        if re.match(rf"openshell-{re.escape(sandbox)}(-|$)", name.strip()):
            return name.strip()
    return None


def reset_hermes_state(sandbox: str) -> bool:
    container = resolve_container(sandbox)
    if not container:
        print(f"  ! could not resolve container for sandbox '{sandbox}'; state NOT reset")
        return False
    subprocess.run(
        ["docker", "exec", container, "bash", "-lc",
         "rm -rf /sandbox/.hermes/memories /sandbox/.hermes/sessions /sandbox/.hermes/runtime/state.db || true"],
        capture_output=True, text=True, timeout=30,
    )
    print(f"  ✓ reset /sandbox/.hermes state in {container}")
    return True


def read_session_events(path: str, since_offset: int) -> tuple[list[dict[str, Any]], int]:
    """Read session-event records appended after ``since_offset`` bytes. Returns
    (records, new_offset). Best-effort — missing file yields ([], since_offset)."""
    if not os.path.exists(path):
        return [], since_offset
    records: list[dict[str, Any]] = []
    with open(path, "rb") as fh:
        fh.seek(since_offset)
        chunk = fh.read()
        new_offset = fh.tell()
    for line in chunk.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records, new_offset


# ---- run ---------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    corpus = load_corpus(args.corpus)
    pattern = corpus.get("pattern") or {}
    expect_tool = str(pattern.get("expect_tool") or "")
    turn_prompts: list[str] = list(pattern.get("turns") or [])
    if not turn_prompts:
        raise SystemExit("corpus pattern has no turns")
    preference = (corpus.get("preference") or {}).get("statement")

    run_tag = f"long-{uuid.uuid4().hex[:8]}"
    events_path = args.session_events

    if args.reset_hermes_state:
        print("Resetting Hermes state for a clean longitudinal baseline:")
        reset_hermes_state(args.sandbox)

    # Start reading the session-events log from its current end so we only
    # attribute events produced by this run.
    offset = os.path.getsize(events_path) if os.path.exists(events_path) else 0

    sessions: list[dict[str, Any]] = []
    print(f"\nLongitudinal run {run_tag}: {args.sessions} sessions × up to "
          f"{args.turns_per_session} turns  (pattern={pattern.get('name')}, expect_tool={expect_tool})\n")

    for s in range(1, args.sessions + 1):
        conv = f"{run_tag}-s{s}"
        completed_turn = None
        turns_log: list[dict[str, Any]] = []
        for t in range(1, args.turns_per_session + 1):
            # Session 1, turn 1 also states the preference once (memory seed).
            prompt = turn_prompts[(t - 1) % len(turn_prompts)]
            if s == 1 and t == 1 and preference:
                prompt = f"{preference}\n\n{prompt}"
            t0 = time.time()
            code, resp = send_chat(args.composer, prompt, conv, args.timeout_s)
            dt = time.time() - t0
            # Drain this turn's session events; detect the expected tool.
            recs, offset = read_session_events(events_path, offset)
            conv_recs = [r for r in recs if r.get("conversationId") == conv]
            tools = [r.get("tool") for r in conv_recs if r.get("kind") == "tool_call"]
            turns_log.append({"turn": t, "status": code, "elapsedS": round(dt, 1),
                              "toolsObserved": tools, "events": len(conv_recs)})
            print(f"  s{s} t{t}: http={code} {dt:5.1f}s tools={tools}")
            if expect_tool and expect_tool in tools:
                completed_turn = t
                break
        # Aggregate this session's distinctive-Hermes signals from its events.
        all_recs, _ = read_session_events(events_path, 0)
        sess_recs = [r for r in all_recs if r.get("conversationId") == conv]
        kinds = [r.get("kind") for r in sess_recs]
        sessions.append({
            "session": s,
            "conversationId": conv,
            "completedTurn": completed_turn,
            "turnsUsed": len(turns_log),
            "skillCreations": kinds.count("skill_created"),
            "memoryWrites": kinds.count("memory_write"),
            "cronFires": kinds.count("cron_fire"),
            "delegations": kinds.count("delegation"),
            "turns": turns_log,
        })

    return summarize(run_tag, expect_tool, sessions)


def summarize(run_tag: str, expect_tool: str, sessions: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [s for s in sessions if s["completedTurn"] is not None]
    ttc = [s["completedTurn"] for s in completed]
    first_half = ttc[: len(ttc) // 2] or ttc
    second_half = ttc[len(ttc) // 2:] or ttc
    avg = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
    # Memory hit-rate: sessions that completed on turn 1 (no re-derivation),
    # i.e. the pattern was recalled rather than re-discovered.
    one_turn = [s for s in completed if s["completedTurn"] == 1]
    report = {
        "runTag": run_tag,
        "expectTool": expect_tool,
        "sessions": len(sessions),
        "completed": len(completed),
        "totalSkillCreations": sum(s["skillCreations"] for s in sessions),
        "totalMemoryWrites": sum(s["memoryWrites"] for s in sessions),
        "totalCronFires": sum(s["cronFires"] for s in sessions),
        "totalDelegations": sum(s["delegations"] for s in sessions),
        "avgTurnsToCompletion": avg(ttc),
        "avgTurnsFirstHalf": avg(first_half),
        "avgTurnsSecondHalf": avg(second_half),
        "memoryHitRate": round(len(one_turn) / len(sessions), 2) if sessions else 0.0,
        "perSession": sessions,
    }
    improved = (
        report["avgTurnsFirstHalf"] is not None
        and report["avgTurnsSecondHalf"] is not None
        and report["avgTurnsSecondHalf"] < report["avgTurnsFirstHalf"]
    )
    report["compoundingObserved"] = bool(improved)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Hermes longitudinal bake-off harness (plan §9.3)")
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--turns-per-session", type=int, default=8)
    ap.add_argument("--corpus", default=None, help="JSON or YAML corpus; default is embedded")
    ap.add_argument("--composer", default=DEFAULT_COMPOSER)
    ap.add_argument("--sandbox", default=os.environ.get("OPENCLAW_ASSISTANT_SANDBOX", "hermes-assistant"))
    ap.add_argument("--session-events", default=DEFAULT_SESSION_EVENTS)
    ap.add_argument("--timeout-s", type=float, default=200.0)
    ap.add_argument("--reset-hermes-state", action="store_true",
                    help="clear /sandbox/.hermes/{memories,sessions,runtime/state.db} before the run")
    ap.add_argument("--report", default=None, help="write the JSON report to this path")
    args = ap.parse_args()

    report = run(args)

    print("\n" + "=" * 64)
    print(f"LONGITUDINAL REPORT  {report['runTag']}")
    print("=" * 64)
    print(f"  sessions completed : {report['completed']}/{report['sessions']}")
    print(f"  avg turns-to-done  : {report['avgTurnsToCompletion']}  "
          f"(first half {report['avgTurnsFirstHalf']} → second half {report['avgTurnsSecondHalf']})")
    print(f"  memory hit-rate    : {report['memoryHitRate']}  (sessions completing in 1 turn)")
    print(f"  skill emergences   : {report['totalSkillCreations']}   "
          f"memory writes: {report['totalMemoryWrites']}   "
          f"cron: {report['totalCronFires']}   delegations: {report['totalDelegations']}")
    print(f"  compounding seen   : {report['compoundingObserved']}")
    print("=" * 64)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"  report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
