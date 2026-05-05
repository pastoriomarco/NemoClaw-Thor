#!/usr/bin/env python3
"""Comparison matrix v2: build the full v0 envelope both bridges accept.

Fetches the live mode manifest from Composer and wraps each test
message in a proper provider_request.v0 envelope (with tools,
catalog, runtime, etc.). The OpenClaw bridge ignores most of it
but the direct bridge requires it; this version is symmetric and
runs identically against both endpoints.
"""
from __future__ import annotations
import csv
import json
import time
import uuid
import urllib.request
import urllib.error
import sys
sys.path.insert(0, "/tmp")
from run_comparison_tests import TESTS, reset_program, COMPOSER  # type: ignore  # noqa: E402

DIRECT = "http://127.0.0.1:8100/v1/manyforge/assistant"
OPENCLAW = "http://127.0.0.1:8200/v1/manyforge/assistant"
MAX_WAIT = 240.0
ASSISTANT_MODE = "composer-assistant"

def fetch_manifest() -> dict:
    with urllib.request.urlopen(
        f"{COMPOSER}/api/assistant/modes/{ASSISTANT_MODE}", timeout=10
    ) as resp:
        return json.loads(resp.read())

def build_envelope(manifest: dict, message: str, request_id: str, conv_id: str) -> dict:
    tool_ids = [t["id"] for t in manifest.get("tools", [])]
    return {
        "version": "manyforge.assistant.provider_request.v0",
        "schemaVersion": "0.1.0",
        "requestId": request_id,
        "providerId": "matrix-test",
        "conversationId": conv_id,
        "message": message,
        "requestedTools": [],
        "context": {},
        "runtime": {"programLoaded": True, "cycleState": "idle"},
        "tools": manifest.get("tools", []),
        "skills": manifest.get("skills", []),
        "nodes": manifest.get("nodes", []),
        "catalog": {
            "skills": manifest.get("skills", []),
            "tools": tool_ids,
            "nodes": manifest.get("nodes", []),
        },
        "assistantMode": ASSISTANT_MODE,
        "constraints": {
            "mutatesState": False,
            "requiresReview": True,
            "proposalStatus": "draft",
            "allowedToolCallStatuses": ["proposed","skipped","completed","failed"],
        },
    }

def run_one(endpoint: str, label: str, tc, manifest: dict) -> dict:
    rid = f"matrix2-{label}-{tc.id}-{uuid.uuid4().hex[:8]}"
    body = build_envelope(manifest, tc.message, rid, rid)
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
        err = f"HTTP {e.code}: {e.read()[:300].decode('utf-8','replace')}"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - started

    msg = (parsed or {}).get("message") or "" if parsed else ""
    tool_calls = (parsed or {}).get("toolCalls", []) if parsed else []
    proposals = (parsed or {}).get("proposals", []) if parsed else []
    draft_mutated = (parsed or {}).get("draftMutated", False) if parsed else False
    error_obj = (parsed or {}).get("error") if parsed else None
    warnings = (parsed or {}).get("warnings", []) if parsed else []

    completed = err is None and error_obj is None and bool(msg)
    msg_lower = msg.lower()
    keyword_hit = any(k.lower() in msg_lower for k in tc.expect_keywords) if tc.expect_keywords else True
    tool_used = bool(tool_calls) or draft_mutated or (
        completed and keyword_hit and tc.expect_tool is not None
    )
    if tc.expect_no_tool:
        score = "PASS" if (completed and not draft_mutated) else "FAIL"
    elif not completed:
        score = "FAIL"
    elif tc.expect_tool and not tool_used:
        score = "FAIL"
    elif keyword_hit:
        score = "PASS"
    else:
        score = "WEAK"

    return {
        "id": tc.id, "category": tc.category, "phrasing": tc.phrasing,
        "path": label, "elapsed_s": round(elapsed, 2),
        "completed": completed, "score": score,
        "tool_calls_n": len(tool_calls), "proposals_n": len(proposals),
        "draft_mutated": draft_mutated, "warnings_n": len(warnings),
        "error": err or ((error_obj or {}).get("code") if error_obj else None),
        "msg_chars": len(msg),
        "msg_excerpt": msg[:200].replace("\n", " "),
    }

def main() -> None:
    print("Fetching manifest...")
    manifest = fetch_manifest()
    print(f"  catalogHash={manifest['catalogHash'][:16]}, {len(manifest['tools'])} tools, {len(manifest['nodes'])} nodes")

    rows = []
    for tc in TESTS:
        for label, endpoint in [("direct", DIRECT), ("openclaw", OPENCLAW)]:
            print(f"\n=== {tc.id} ({tc.category}/{tc.phrasing}) → {label} ===")
            print(f"  Q: {tc.message[:100]}")
            reset_program()
            time.sleep(1)  # let any reset settle
            r = run_one(endpoint, label, tc, manifest)
            rows.append(r)
            print(f"  → {r['score']} in {r['elapsed_s']}s, msg={r['msg_chars']}c, tools={r['tool_calls_n']}, proposals={r['proposals_n']}, mutated={r['draft_mutated']}")
            if r['error']:
                print(f"  ERR: {r['error'][:200]}")
            if r['msg_excerpt']:
                print(f"  A: {r['msg_excerpt']}…")

    with open("/tmp/comparison-results-v2.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to /tmp/comparison-results-v2.csv")

    by_path: dict = {}
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
