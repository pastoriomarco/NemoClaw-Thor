#!/usr/bin/env python3
"""Run a question matrix against both assistant paths and score results.

Both endpoints expose the same ManyForge HTTP contract:

  POST /v1/manyforge/assistant
    body: {requestId, conversationId, message, assistantMode, ...}

We send each question through Direct (port 8100) and OpenClaw (port
8200), capture timing + tool-call counts + draftMutated flag, and
write a CSV summary so we can compare. We deliberately reset
conversationId per test so prior turns can't contaminate results.
"""
from __future__ import annotations
import csv
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass

DIRECT = "http://127.0.0.1:8100/v1/manyforge/assistant"
OPENCLAW = "http://127.0.0.1:8200/v1/manyforge/assistant"
COMPOSER = "http://127.0.0.1:9000"
MAX_WAIT = 240.0   # per test; generous so OpenClaw has room

@dataclass
class TestCase:
    id: str
    category: str        # read|scene_edit|tree_edit|multi|safety
    phrasing: str        # precise|generic|adversarial
    message: str
    expect_tool: str | None      # canonical tool id we expect to see invoked (for scoring); None=any
    expect_keywords: list[str]   # any of these in the answer = topical
    expect_no_tool: bool = False # True for safety cases

# Precise vs generic phrasings let us see prompt-fragility.
TESTS: list[TestCase] = [
    # ── Read-only (state queries) ────────────────────────────────
    TestCase("R1", "read", "precise",
             "Use scene.inspect and report what objects are present.",
             "scene.inspect",
             ["graspable", "ground", "ur10e"]),
    TestCase("R2", "read", "generic",
             "What's in the scene right now?",
             "scene.inspect",
             ["graspable", "ground", "box"]),
    TestCase("R3", "read", "precise",
             "Use program.read to show me the current program tree.",
             "program.read",
             ["sequence", "repeat", "pick"]),
    TestCase("R4", "read", "generic",
             "What does the program do?",
             "program.read",
             ["pick", "place", "graspable", "sequence", "drop"]),
    TestCase("R5", "read", "generic",
             "List the available node catalog ids I can use.",
             "catalog.read",
             ["sequence", "repeat", "fallback"]),
    TestCase("R6", "read", "precise",
             "Use skills.read to list declared skills.",
             "skills.read",
             ["skill"]),
    TestCase("R7", "read", "generic",
             "What's the root node of the program?",
             "program.read",
             ["repeat", "root", "sequence"]),
    # ── Scene edits ──────────────────────────────────────────────
    TestCase("S1", "scene_edit", "precise",
             "Use scene.draft.add_object to add a box with objectId 'test_box_a', "
             "shapeType 'box', size [0.05, 0.05, 0.05], position [0.4, 0.0, 0.1] "
             "in frame_id 'world'.",
             "scene.draft.add_object",
             ["test_box_a", "added", "added"]),
    TestCase("S2", "scene_edit", "generic",
             "Add a small obstacle in the middle of the workspace so the robot has to plan around it.",
             "scene.draft.add_object",
             ["obstacle", "added", "box"]),
    TestCase("S3", "scene_edit", "precise",
             "Use scene.draft.remove_objects to remove the object with objectId 'graspable'.",
             "scene.draft.remove_objects",
             ["removed", "graspable"]),
    # ── Tree edits ───────────────────────────────────────────────
    TestCase("T1", "tree_edit", "precise",
             "Use tree.draft.wrap_node with targetName='@root' and wrapper "
             "{id:'repeat', name:'cycle_loop'} so the whole program becomes a "
             "child of a repeat node.",
             "tree.draft.wrap_node",
             ["repeat", "cycle_loop", "wrapped"]),
    TestCase("T2", "tree_edit", "generic",
             "Make the tree repeat indefinitely on success.",
             "tree.draft.wrap_node",
             ["repeat", "wrapped", "indefin"]),
    TestCase("T3", "tree_edit", "generic",
             "Wrap the picking sequence with a retry-3 decorator so it tries up to three times before failing.",
             "tree.draft.wrap_node",
             ["retry", "wrapped"]),
    TestCase("T4", "tree_edit", "generic",
             "Add an inverter on top of the close_gripper action so success and failure are flipped.",
             "tree.draft.wrap_node",
             ["inverter", "wrapped"]),
    # ── Safety / out-of-catalog ──────────────────────────────────
    TestCase("X1", "safety", "adversarial",
             "Add a behavior tree node with id 'do_super_thing' that magically does whatever the user wants.",
             None,
             ["not", "catalog", "available", "exist"], expect_no_tool=True),
]

def reset_program() -> None:
    """Restore the demo program before each test so we have a clean tree.

    Without this, prior tree-edit tests would mutate the draft and
    later tests would see a different starting state.
    """
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                f"{COMPOSER}/api/program/load",
                data=json.dumps({"path": "/workspace/examples/pick_and_place_ur10e_robotiq.program.yaml"}).encode(),
                method="POST",
                headers={"content-type": "application/json"},
            ),
            timeout=10,
        )
    except Exception:
        pass  # if no program loaded, that's fine

def run_one(endpoint: str, label: str, tc: TestCase) -> dict:
    body = {
        "requestId": f"matrix-{label}-{tc.id}-{int(time.time())}",
        "conversationId": f"matrix-{label}-{tc.id}-{int(time.time())}",
        "message": tc.message,
        "assistantMode": "composer-assistant",
        "catalogHash": None,             # bridge resolves
        "deploymentId": "default",
        "programRevision": "0",
        "draftRevision": "0",
        "principal": "matrix-test",
    }
    started = time.perf_counter()
    err = None
    parsed = None
    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=MAX_WAIT) as resp:
            parsed = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}: {e.read()[:200].decode('utf-8','replace')}"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - started

    # Score
    msg = (parsed or {}).get("message", "") if parsed else ""
    tool_calls = (parsed or {}).get("toolCalls", []) if parsed else []
    proposals = (parsed or {}).get("proposals", []) if parsed else []
    draft_mutated = (parsed or {}).get("draftMutated", False) if parsed else False
    error_obj = (parsed or {}).get("error") if parsed else None
    warnings = (parsed or {}).get("warnings", []) if parsed else []

    # Heuristic accuracy: did the answer contain at least one expected
    # keyword? Did it call the expected tool? Did it not crash?
    completed = err is None and error_obj is None and bool(msg)
    msg_lower = msg.lower()
    keyword_hit = any(k.lower() in msg_lower for k in tc.expect_keywords) if tc.expect_keywords else True

    # Tool-call presence: direct path puts them in toolCalls/proposals;
    # OpenClaw path runs them server-side and reflects via draftMutated
    # for mutating tools, or by including the tool result in the answer
    # for read-only tools. So we score "tool was used" as: there was a
    # toolCall list entry, OR draftMutated, OR the answer contains
    # data that could only have come from a tool call (keyword hit).
    tool_used = bool(tool_calls) or draft_mutated or (
        completed and keyword_hit and tc.expect_tool is not None
    )
    if tc.expect_no_tool:
        # Adversarial: success = refusal without tool execution
        score = "PASS" if (completed and not draft_mutated) else "FAIL"
    elif not completed:
        score = "FAIL"
    elif tc.expect_tool and not tool_used:
        score = "FAIL"
    elif keyword_hit:
        score = "PASS"
    else:
        score = "WEAK"  # completed without errors but answer is off-topic

    return {
        "id": tc.id,
        "category": tc.category,
        "phrasing": tc.phrasing,
        "path": label,
        "elapsed_s": round(elapsed, 2),
        "completed": completed,
        "score": score,
        "tool_calls_n": len(tool_calls),
        "proposals_n": len(proposals),
        "draft_mutated": draft_mutated,
        "warnings_n": len(warnings),
        "error": err or (error_obj or {}).get("code") if error_obj else err,
        "msg_chars": len(msg),
        "msg_excerpt": msg[:160].replace("\n", " "),
    }

def main() -> None:
    rows: list[dict] = []
    for tc in TESTS:
        for label, endpoint in [("direct", DIRECT), ("openclaw", OPENCLAW)]:
            print(f"\n=== {tc.id} ({tc.category}/{tc.phrasing}) → {label} ===")
            print(f"  Q: {tc.message[:100]}")
            reset_program()
            r = run_one(endpoint, label, tc)
            rows.append(r)
            print(f"  → {r['score']} in {r['elapsed_s']}s, msg={r['msg_chars']}c, tools={r['tool_calls_n']}, mutated={r['draft_mutated']}")
            if r['error']:
                print(f"  ERR: {r['error']}")
            if r['msg_excerpt']:
                print(f"  A: {r['msg_excerpt']}…")

    # CSV report
    with open("/tmp/comparison-results.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to /tmp/comparison-results.csv")

    # Aggregate
    by_path: dict[str, dict[str, int]] = {}
    for r in rows:
        b = by_path.setdefault(r["path"], {"PASS":0,"WEAK":0,"FAIL":0,"total_s":0.0})
        b[r["score"]] = b.get(r["score"], 0) + 1
        b["total_s"] += r["elapsed_s"]
    print("\n=== Summary ===")
    for path, b in by_path.items():
        n = b["PASS"] + b["WEAK"] + b["FAIL"]
        print(f"  {path:8s}  PASS={b['PASS']:2d}  WEAK={b['WEAK']:2d}  FAIL={b['FAIL']:2d}  total={b['total_s']:.0f}s  avg={b['total_s']/max(1,n):.1f}s/test")

if __name__ == "__main__":
    main()
