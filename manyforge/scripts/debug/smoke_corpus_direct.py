#!/usr/bin/env python3
"""Run the ported smoke corpus against ManyForge 0.2.1's direct assistant path.

Transport: Composer ``POST /api/assistant/requests`` + status polling (the AI
tab path; the bridge calls the model server named by the active ``ai_model``
component). Tool calls and their arguments come from the completed response's
``toolCalls``; state comes from ``GET /api/program/draft`` (Program v3).

Per case: the Program Draft is discarded back to Applied (a chained case only
at step 1), fixtures are seeded through ``POST /api/program/draft/mutations``,
the prompt is submitted once, then the assertion engine of
``smoke_corpus_runner.py`` scores tools, state and answer.

Run on the ManyForge host against an isolated Mock Draft (it discards the
Draft). It never Saves, Applies, runs a Program or moves a robot.

    python3 smoke_corpus_direct.py --corpus smoke_corpus_v021.yaml \
        --enable-recovery-turn --report /tmp/corpus.json
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_corpus_runner as legacy  # noqa: E402  (assertion engine only)

HEADLESS = "http://127.0.0.1:9010"
# The capture proxy tags each model call with the case and corpus turn here.
CASE_FILE: Path | None = None


def mark_case(case_id: str | None, turn: int | None) -> None:
    if CASE_FILE is not None:
        CASE_FILE.write_text(json.dumps({"case_id": case_id, "turn": turn}))
MODE = "composer-assistant"
EPOCH = f"smoke-corpus.{uuid.uuid4().hex[:12]}"

# --- expected-value extensions --------------------------------------------
# "one_of a|b": the actual value equals any listed alternative.
_base_value_matches = legacy._value_matches


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str) and expected.startswith("one_of "):
        return any(_base_value_matches(actual, alt) for alt in expected[7:].split("|"))
    return _base_value_matches(actual, expected)


legacy._value_matches = _value_matches


# --- HTTP -------------------------------------------------------------------
def call(method: str, url: str, body: Any = None, timeout: float = 30.0) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw.decode(errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return -1, f"{type(e).__name__}: {e}"


class Cell:
    def __init__(self, composer: str, headless: str) -> None:
        self.composer = composer.rstrip("/")
        self.headless = headless.rstrip("/")

    def get(self, path: str) -> Any:
        st, js = call("GET", self.composer + path)
        if st != 200:
            raise RuntimeError(f"GET {path}: HTTP {st} {str(js)[:300]}")
        return js

    def fence(self) -> dict[str, Any]:
        ws = self.get("/api/program/workspace")
        aw = ws["authoringWorkspace"]
        _, adm = call("GET", self.headless + "/api/cell/authoring-admission")
        draft = self.get("/api/program/draft")
        return {"draftId": draft["draft_id"], "expectedDraftRevision": draft["draft_revision"],
                "expectedAdmissionGeneration": adm["admission_generation"],
                "requestOwnerEpoch": EPOCH, "requestId": f"smoke-fixture.{uuid.uuid4().hex}",
                "expectedWorkspaceEpoch": aw["workspaceEpoch"],
                "expectedWorkspaceRevision": aw["workspaceRevision"]}

    def mutate(self, tool: str, arguments: dict[str, Any]) -> None:
        st, js = call("POST", self.composer + "/api/program/draft/mutations",
                      {**self.fence(), "mutation": tool, "arguments": arguments})
        if st != 200:
            raise RuntimeError(f"fixture {tool}: HTTP {st} {str(js)[:400]}")

    def discard(self) -> None:
        st, js = call("POST", self.composer + "/api/program/discard", self.fence())
        if st != 200:
            raise RuntimeError(f"discard: HTTP {st} {str(js)[:400]}")

    def state(self) -> dict[str, Any]:
        authoring = self.get("/api/program/draft")["authoring"]
        forest = authoring.get("forest") or []
        return {
            "program": {"tree": forest[0] if forest else None, "forest": forest,
                        "parameters": authoring.get("parameters") or [],
                        "blackboard": authoring.get("blackboard") or {"keys": [], "namespaces": []}},
            "scene": {"objects": authoring.get("scene") or []},
        }

    def submit(self, message: str, conversation_id: str, timeout_s: float) -> tuple[str, dict[str, Any], float]:
        request = {"requestId": f"smoke-corpus.{uuid.uuid4().hex}", "requestOwnerEpoch": EPOCH,
                   "sourceLabel": "assistant_smoke_corpus", "correlationId": None,
                   "conversationId": conversation_id, "assistantMode": MODE,
                   # Composer requires canonical text (no surrounding whitespace).
                   "message": message.strip(),
                   "requestedTools": [], "fieldPolicy": {}}
        t0 = time.monotonic()
        st, status = call("POST", self.composer + "/api/assistant/requests", request)
        if st not in (200, 202) or not isinstance(status, dict):
            return "submit_failed", {"http": st, "body": status}, time.monotonic() - t0
        path = "/api/assistant/requests/" + urllib.parse.quote(request["requestId"], safe="")
        deadline = t0 + timeout_s
        while status.get("state") == "running" and time.monotonic() < deadline:
            time.sleep(0.5)
            st, polled = call("GET", self.composer + path)
            if st == 200 and isinstance(polled, dict):
                status = polled
        elapsed = time.monotonic() - t0
        if status.get("state") == "running":
            call("POST", self.composer + path + "/cancel", {})
            return "timeout", status, elapsed
        return status.get("state", "unknown"), status, elapsed


# --- observation normalization ----------------------------------------------
def _add_orientation_aliases(value: Any) -> Any:
    """Expose v3 ``orientation: {xyzw|rpy_deg}`` as the leaf names the
    legacy rotation matcher understands (orientation_xyzw / _rpy_deg)."""
    if isinstance(value, list):
        return [_add_orientation_aliases(v) for v in value]
    if not isinstance(value, dict):
        return value
    out = {k: _add_orientation_aliases(v) for k, v in value.items()}
    o = value.get("orientation")
    if isinstance(o, dict):
        if "xyzw" in o:
            out["orientation_xyzw"] = o["xyzw"]
        if "rpy_deg" in o:
            out["orientation_rpy_deg"] = o["rpy_deg"]
    return out


def normalize_call(call_row: dict[str, Any], pre: dict[str, Any]) -> dict[str, Any]:
    args = copy.deepcopy(call_row.get("arguments") or {})
    tree = pre["program"]["tree"] or {}
    root = tree.get("instance_id")
    for key in ("target_instance_id", "parent_instance_id", "new_parent_instance_id", "instance_id"):
        if args.get(key) == "@root" and root:
            args[key] = root
    if call_row.get("toolId") == "tree_draft_insert_node":
        # End insertion: no position given, index == child count, or after the last child.
        parent = args.get("parent_instance_id")
        siblings = None
        stack = [tree] if tree else []
        while stack:
            n = stack.pop()
            if n.get("instance_id") == parent:
                siblings = [c["instance_id"] for c in n.get("children", [])]
                break
            stack.extend(n.get("children", []))
        if not any(k in args for k in ("index", "after_instance_id", "before_instance_id")):
            args["index"] = "end"
        elif siblings is not None and args.get("index") == len(siblings):
            args["index"] = "end"
        elif siblings and args.get("after_instance_id") == siblings[-1]:
            args["index"] = "end"
            args.setdefault("parent_instance_id", parent)
    args = _add_orientation_aliases(args)
    ok = call_row.get("status") == "completed"
    result = call_row.get("result") if isinstance(call_row.get("result"), dict) else {}
    error = None
    if not ok:
        # The v2 bridge's failure: message plus typed path/expected fields.
        error = result.get("message") or json.dumps(result.get("detail"))[:600]
    return {"tool": call_row.get("toolId"), "status": 200 if ok else 422, "arguments": args,
            "bridge_status": call_row.get("status"), "bridge_error": error}


def recovery_message(observed: list[dict[str, Any]], expected_tools_called: Any) -> str | None:
    """The harness's one follow-up turn after a failed case (v2 bridge fields)."""
    failed = [c for c in observed if c["status"] >= 400]
    if failed:
        last = failed[-1]
        return (
            f"The previous `{last['tool']}` call failed: {last['bridge_error'] or 'see its tool result'} "
            "Use the failure's `path` and `expected` fields (and `recovery.action` when present) "
            "to correct the arguments, then retry the original request. "
            "Do not repeat the failed call verbatim."
        )
    return legacy._build_recovery_message(observed, expected_tools_called=expected_tools_called)


# --- placement judged by the result ----------------------------------------------
PLACEMENT_KEYS = ("parent_instance_id", "index", "after_instance_id", "before_instance_id")


def _locate(tree: dict[str, Any] | None, instance_id: str) -> tuple[dict | None, int | None]:
    """The parent node and child index of one instance in the post-state tree."""
    stack = [tree] if tree else []
    while stack:
        node = stack.pop()
        for index, child in enumerate(node.get("children", [])):
            if child.get("instance_id") == instance_id:
                return node, index
            stack.append(child)
    return None, None


def split_placement(case: dict) -> tuple[dict, list[dict[str, Any]]]:
    """Move insert placement expectations out of args_contain.

    Any valid placement form that yields the same tree passes (parent+index,
    parent with index omitted, after the previous sibling, before the next).
    """
    case = copy.deepcopy(case)
    specs = []
    for expectation in case.get("expected", {}).get("tools_called") or []:
        if expectation.get("name") != "tree_draft_insert_node":
            continue
        args = expectation.get("args_contain") or {}
        spec = {key: args.pop(key) for key in PLACEMENT_KEYS if key in args}
        if spec:
            specs.append(spec)
    return case, specs


def assert_placement(specs: list[dict[str, Any]], observed: list[dict[str, Any]],
                     post: dict[str, Any], failures: list[str]) -> None:
    inserted = [o["arguments"].get("node", {}).get("instance_id") for o in observed
                if o["tool"] == "tree_draft_insert_node" and o["status"] < 400
                and isinstance(o["arguments"].get("node"), dict)]
    tree = post["program"]["tree"]
    for spec in specs:
        landed = [(iid, *_locate(tree, iid)) for iid in inserted if iid]
        landed = [row for row in landed if row[1] is not None]
        if not landed:
            failures.append(f"placement {spec}: no inserted node found in the Draft")
            continue

        def ok(row) -> bool:
            _iid, parent, index = row
            siblings = [c.get("instance_id") for c in parent.get("children", [])]
            if "parent_instance_id" in spec and parent.get("instance_id") != spec["parent_instance_id"]:
                return False
            want = spec.get("index")
            if want == "end" and index != len(siblings) - 1:
                return False
            if isinstance(want, int) and index != want:
                return False
            if "after_instance_id" in spec and (index == 0 or siblings[index - 1] != spec["after_instance_id"]):
                return False
            if "before_instance_id" in spec and (index + 1 >= len(siblings) or siblings[index + 1] != spec["before_instance_id"]):
                return False
            return True

        if not any(ok(row) for row in landed):
            where = [(iid, parent.get("instance_id"), index) for iid, parent, index in landed]
            failures.append(f"placement {spec} not met by the Draft; inserted at {where}")


# --- one case ------------------------------------------------------------------
def run_case(cell: Cell, case: dict, header: dict, chains: dict[str, str],
             timeout_s: float, recovery: bool) -> dict[str, Any]:
    cid = case["id"]
    pre = case.get("precondition") or {}
    chain_id, step = pre.get("chain_id"), pre.get("chain_step")
    result: dict[str, Any] = {"case_id": cid, "category": case.get("category"),
                              "detail_level": case.get("detail_level"), "prompt": case["prompt"],
                              "turns": []}
    harness_calls: list[dict[str, Any]] = []
    result["harness_calls"] = harness_calls
    mark_case(cid, 0)
    try:
        if not chain_id or step == 1:
            cell.discard()
            harness_calls.append({"call": "discard", "t": time.time()})
            if pre.get("fresh_program"):
                fx = header["fresh_program_fixture"]
                cell.mutate("tree_draft_replace_subtree",
                            {"instance_id": "@root", "subtree": fx["tree"]})
                harness_calls.append({"call": "fresh_program", "t": time.time(),
                                      "mutation": "tree_draft_replace_subtree", "subtree": fx["tree"]})
                ids = [o["scene_resource_id"] for o in cell.state()["scene"]["objects"]]
                if ids:
                    cell.mutate("scene_draft_remove_objects", {"scene_resource_ids": ids})
                    harness_calls.append({"call": "fresh_program_scene", "t": time.time(),
                                          "mutation": "scene_draft_remove_objects", "ids": ids})
        for fx in pre.get("fixtures") or []:
            cell.mutate(fx["tool"], fx["arguments"])
            harness_calls.append({"call": "fixture", "t": time.time(), "mutation": fx["tool"],
                                  "arguments": fx["arguments"]})
        pre_state = cell.state()
    except RuntimeError as error:
        result.update(status="harness_error", failures=[str(error)])
        return result

    if chain_id:
        conversation = chains.setdefault(chain_id, f"smoke-chain-{chain_id}.{uuid.uuid4().hex[:8]}")
    else:
        conversation = f"smoke-case-{cid}.{uuid.uuid4().hex[:8]}"
    budget = float(pre.get("chain_timeout_s", timeout_s))

    mark_case(cid, 1)
    state, status, elapsed = cell.submit(case["prompt"], conversation, budget)
    observed, answer, failures, soft = [], "", [], []

    def absorb(state_: str, status_: dict[str, Any], elapsed_: float) -> None:
        nonlocal answer
        response = status_.get("response") if isinstance(status_, dict) else None
        error = status_.get("error") if isinstance(status_, dict) else None
        # Full evidence: the complete Composer status response of this turn
        # (toolCalls with arguments, status and result; message; exitReason,
        # turnCount, needsClarification, warnings; error).
        turn = {"state": state_, "elapsed_s": round(elapsed_, 2), "status": status_}
        if isinstance(response, dict):
            answer = response.get("message") or ""
            observed.extend(normalize_call(c, pre_state) for c in response.get("toolCalls") or [])
            turn.update(exit_reason=response.get("exitReason"), model_turns=response.get("turnCount"),
                        tool_calls=[(c.get("toolId"), c.get("status")) for c in response.get("toolCalls") or []])
        if state_ == "submit_failed":
            turn["error_code"] = "submit_http_" + str(status_.get("http"))
            turn["error_message"] = str(status_.get("body"))[:300]
        if isinstance(error, dict):
            turn["error_code"] = error.get("code")
            turn["error_message"] = str(error.get("message"))[:300]
        result["turns"].append(turn)

    absorb(state, status, elapsed)
    if state != "completed":
        failures.append(f"request {state}: {result['turns'][-1].get('error_code')}")
    post = cell.state()
    scored_case, placement = split_placement(case)
    legacy.assert_tools(scored_case, observed, failures, soft, pre_state=pre_state)
    assert_placement(placement, observed, post, failures)
    legacy.assert_state(case, post, failures)
    legacy.assert_answer(case, answer, soft)
    legacy.demote_args_when_state_proven(case, failures, soft)

    recovered = False
    if recovery and failures and state == "completed":
        message = recovery_message(observed, case.get("expected", {}).get("tools_called"))
        if message is not None:
            # The harness's own recovery turn: its exact text is evidence.
            result["recovery_turn"] = {"message": message, "t": time.time()}
            mark_case(cid, 2)
            state2, status2, elapsed2 = cell.submit(message, conversation, budget)
            absorb(state2, status2, elapsed2)
            retry: list[str] = []
            post = cell.state()
            legacy.assert_tools(scored_case, observed, retry, pre_state=pre_state)
            assert_placement(placement, observed, post, retry)
            legacy.assert_state(case, post, retry)
            if not retry:
                recovered, failures = True, []

    result.update(
        status="fail" if failures else "recovered-pass" if recovered else "soft-pass" if soft else "pass",
        failures=failures, soft_failures=soft, answer=answer,
        tools=[(o["tool"], o["bridge_status"]) for o in observed],
        elapsed_s=round(sum(t["elapsed_s"] for t in result["turns"]), 2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parent / "smoke_corpus_v021.yaml"))
    ap.add_argument("--composer", default="http://127.0.0.1:3000")
    ap.add_argument("--headless", default=HEADLESS)
    ap.add_argument("--filter", default=None, help="regex on case id")
    ap.add_argument("--timeout-s", type=float, default=180.0, help="per-turn budget (bridge cap is 180 s)")
    ap.add_argument("--enable-recovery-turn", action="store_true")
    ap.add_argument("--report", required=True)
    ap.add_argument("--case-file", default=None,
                    help="file the capture proxy reads to tag model calls with case/turn")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    corpus = yaml.safe_load(open(args.corpus))
    cases = corpus["cases"]
    if args.filter:
        rx = re.compile(args.filter)
        cases = [c for c in cases if rx.search(c["id"])]
    global CASE_FILE
    CASE_FILE = Path(args.case_file) if args.case_file else None
    cell = Cell(args.composer, args.headless)
    model = cell.get(f"/api/assistant/modes/{MODE}")["model_endpoint"]["model_id"]
    print(f"{len(cases)} cases; model={model}; composer={args.composer}")
    chains: dict[str, str] = {}
    results = []
    marks = {"pass": "PASS ", "recovered-pass": "RECOV", "soft-pass": "SOFT ", "fail": "FAIL ",
             "harness_error": "HARN ", "excluded": "EXCL "}
    for case in cases:
        if case.get("status") == "excluded":
            results.append({"case_id": case["id"], "status": "excluded",
                            "reason": case.get("exclusion_reason")})
            continue
        r = run_case(cell, case, corpus, chains, args.timeout_s, args.enable_recovery_turn)
        results.append(r)
        print(f"{marks[r['status']]} {r['case_id']:48s} {r.get('elapsed_s', 0):6.1f}s "
              f"{r.get('tools', '')} {r.get('failures') or ''}")
    cell.discard()

    scored = [r for r in results if r["status"] not in ("excluded", "harness_error")]
    count = lambda s: sum(r["status"] == s for r in scored)
    effective = count("pass") + count("recovered-pass") + count("soft-pass")
    turn_times = [t["elapsed_s"] for r in scored for t in r.get("turns", [])]
    summary = {"model": model, "corpus": args.corpus, "scored": len(scored),
               "pass": count("pass"), "recovered": count("recovered-pass"), "soft": count("soft-pass"),
               "fail": count("fail"),
               "harness_errors": sum(r["status"] == "harness_error" for r in results),
               "excluded": sum(r["status"] == "excluded" for r in results),
               "effective_rate": round(100.0 * effective / len(scored), 1) if scored else None,
               "first_try_rate": round(100.0 * count("pass") / len(scored), 1) if scored else None,
               "turn_latency_s": {
                   "n": len(turn_times),
                   "mean": round(sum(turn_times) / len(turn_times), 2) if turn_times else None,
                   "p50": sorted(turn_times)[len(turn_times) // 2] if turn_times else None,
                   "max": max(turn_times) if turn_times else None}}
    print(json.dumps(summary, indent=2))
    Path(args.report).write_text(json.dumps({"summary": summary, "results": results}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
