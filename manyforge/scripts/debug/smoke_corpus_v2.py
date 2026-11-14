#!/usr/bin/env python3
"""Run the outcome smoke corpus (v2) against ManyForge's direct assistant path.

A case passes when the operator's request is fulfilled in the Program Draft
(outcome_scorer.py; schema: ASSISTANT_CONTEXT_AND_OUTCOME_TESTS_PLAN.md appendix).
Tool choice, argument shapes, call counts, time and tokens are recorded as
efficiency metrics, not scored. There is no runner recovery turn.

Transport: Composer ``POST /api/assistant/requests`` + status polling. Draft
reads: ``GET /api/program/draft``. Setup/fixtures/chain repair:
``POST /api/program/draft/mutations`` and ``POST /api/program/discard``.

Every run starts model_capture_proxy.py (2 GB cap) on --proxy-port; the ai_model
component's base_url must already point at it (assist_setup_noload.py <model>
http://127.0.0.1:<port>/v1). The runner never edits the Deployment.

Run on the ManyForge host against an isolated Mock Draft (it discards the
Draft). It never Saves, Applies, runs a Program or moves a robot.

    python3 smoke_corpus_v2.py --corpus smoke_corpus_v3.yaml --reasoning off \
        --report ~/assist-phaseA/runs/v2-off.json
    python3 smoke_corpus_v2.py --validate --corpus smoke_corpus_v3.yaml
    python3 smoke_corpus_v2.py --rescore OLD.json --captures OLD.model-requests.jsonl \
        --corpus smoke_corpus_v3.yaml --report rescored.json
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
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
import outcome_scorer as score  # noqa: E402

HERE = Path(__file__).resolve().parent
MODE = "composer-assistant"
EPOCH = f"smoke-corpus.{uuid.uuid4().hex[:12]}"
SCENARIO_ALIASES = {"mock": "mock_ur10e_assistant_scene_authoring",
                    "sim": "sim_ur10e_assistant_scene_authoring"}


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
    def __init__(self, composer: str, headless: str, case_file: Path | None = None) -> None:
        self.composer = composer.rstrip("/")
        self.headless = headless.rstrip("/")
        self.case_file = case_file

    def mark(self, case_id: str | None, turn: int | None) -> None:
        """Tag the capture proxy's next model calls with the case and corpus turn."""
        if self.case_file is not None:
            self.case_file.write_text(json.dumps({"case_id": case_id, "turn": turn}))

    def get(self, path: str) -> Any:
        st, js = call("GET", self.composer + path)
        if st != 200:
            raise RuntimeError(f"GET {path}: HTTP {st} {str(js)[:300]}")
        return js

    def fence(self) -> dict[str, Any]:
        aw = self.get("/api/program/workspace")["authoringWorkspace"]
        _, adm = call("GET", self.headless + "/api/cell/authoring-admission")
        draft = self.get("/api/program/draft")
        return {"draftId": draft["draft_id"], "expectedDraftRevision": draft["draft_revision"],
                "expectedAdmissionGeneration": adm["admission_generation"],
                "requestOwnerEpoch": EPOCH, "requestId": f"smoke-fixture.{uuid.uuid4().hex}",
                "expectedWorkspaceEpoch": aw["workspaceEpoch"],
                "expectedWorkspaceRevision": aw["workspaceRevision"]}

    def mutate(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        st, js = call("POST", self.composer + "/api/program/draft/mutations",
                      {**self.fence(), "mutation": tool, "arguments": arguments})
        if st != 200:
            raise RuntimeError(f"mutation {tool}: HTTP {st} {str(js)[:400]}")
        return js if isinstance(js, dict) else {}

    def discard(self) -> None:
        st, js = call("POST", self.composer + "/api/program/discard", self.fence())
        if st != 200:
            raise RuntimeError(f"discard: HTTP {st} {str(js)[:400]}")

    def draft(self) -> dict[str, Any]:
        d = self.get("/api/program/draft")
        return {"authoring": d.get("authoring") or {}, "diagnostics": d.get("diagnostics") or [],
                "program_revision": d.get("program_revision"), "draft_revision": d.get("draft_revision")}

    def submit(self, message: str, conversation_id: str, timeout_s: float) -> tuple[str, dict[str, Any], float]:
        request = {"requestId": f"smoke-corpus.{uuid.uuid4().hex}", "requestOwnerEpoch": EPOCH,
                   "sourceLabel": "assistant_smoke_corpus", "correlationId": None,
                   "conversationId": conversation_id, "assistantMode": MODE,
                   "message": message.strip(), "requestedTools": [], "fieldPolicy": {}}
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


# --- setup and repair -------------------------------------------------------------
def _fresh_program(cell: Cell, header: dict, log: list[dict]) -> None:
    tree = header["fresh_program_fixture"]["tree"]
    cell.mutate("tree_draft_replace_subtree", {"instance_id": "@root", "subtree": tree})
    log.append({"call": "fresh_program", "t": time.time(), "subtree": tree})
    ids = list(score.Draft(cell.draft()).scene)
    if ids:
        cell.mutate("scene_draft_remove_objects", {"scene_resource_ids": ids})
        log.append({"call": "fresh_program_scene", "t": time.time(), "ids": ids})


def _fixtures(cell: Cell, case: dict, log: list[dict]) -> None:
    for fx in (case.get("preconditions") or {}).get("fixtures") or []:
        cell.mutate(fx["tool"], fx["arguments"])
        log.append({"call": "fixture", "t": time.time(), "mutation": fx["tool"], "arguments": fx["arguments"]})


def setup_case(cell: Cell, case: dict, header: dict, log: list[dict], chains: dict[str, dict]) -> dict | None:
    """Discard to the Applied Program (non-chain cases) or bring a chain to this step.

    A chain step never runs on the Applied Program: when the steps before it did not run
    in this conversation (filtered out), the chain is seeded with the fresh Program plus
    the expected results of the skipped steps, applied like a chain repair. Returns the
    seed record, or None when no step was seeded."""
    pre = case.get("preconditions") or {}
    chain_id, step = pre.get("chain_id"), pre.get("chain_step")
    if not chain_id:
        cell.discard()
        log.append({"call": "discard", "t": time.time()})
        _fixtures(cell, case, log)
        return None
    last = chains.get(chain_id, {}).get("last_step", 0)
    seed = None
    if last == 0 or step == 1:
        cell.discard()
        log.append({"call": "discard", "t": time.time()})
        _fresh_program(cell, header, log)
        last = 0
    skipped = sorted((c for c in header.get("cases") or []
                      if (c.get("preconditions") or {}).get("chain_id") == chain_id
                      and last < (c.get("preconditions") or {}).get("chain_step", 0) < step),
                     key=lambda c: c["preconditions"]["chain_step"])
    if skipped:
        seed = {"steps": [c["id"] for c in skipped], "mutations": [], "ok": True, "failure": None}
        for c in skipped:
            _fixtures(cell, c, log)
            start = cell.draft()
            try:
                for m in score.repair_mutations(c, start):
                    cell.mutate(m["tool"], m["arguments"])
                    seed["mutations"].append(m)
            except score.RepairUnsupported as error:
                seed.update(ok=False, failure=f"{c['id']}: {error}")
                break
            fail, _ = score.evaluate_all((c.get("outcome") or {}).get("draft") or [],
                                         score.Draft(start), score.Draft(cell.draft()))
            if fail:
                seed.update(ok=False, failure=f"{c['id']}: {fail}")
                break
        log.append({"call": "chain_seed", "t": time.time(), **seed})
    _fixtures(cell, case, log)
    chains.setdefault(chain_id, {"conversation": f"smoke-chain-{chain_id}.{uuid.uuid4().hex[:8]}"})["last_step"] = step
    return seed


def repair_step(cell: Cell, case: dict, before: dict) -> dict[str, Any]:
    """Bring a failed chain step's Draft to its expected result: restore the Draft the
    step started from, then apply the step's expected result. Recorded, never scored."""
    rec: dict[str, Any] = {"restore": [], "apply": []}
    try:
        last = None
        for _ in range(50):
            muts = score.restore_mutations(before, cell.draft())
            if not muts:
                break
            if muts == last:
                raise RuntimeError(f"restore does not converge: {str(muts)[:200]}")
            last = muts
            for m in muts:
                cell.mutate(m["tool"], m["arguments"])
                rec["restore"].append(m)
        start = cell.draft()
        for m in score.repair_mutations(case, start):
            cell.mutate(m["tool"], m["arguments"])
            rec["apply"].append(m)
        fail, _ = score.evaluate_all((case.get("outcome") or {}).get("draft") or [],
                                     score.Draft(start), score.Draft(cell.draft()))
        rec.update(ok=fail is None, failure=fail)
    except (RuntimeError, score.RepairUnsupported) as error:
        rec.update(ok=False, failure=f"{type(error).__name__}: {error}")
    return rec


# --- capture metrics --------------------------------------------------------------
class Captures:
    """Reads the proxy JSONL incrementally; efficiency metrics per case."""

    def __init__(self, path: Path | None) -> None:
        self.path, self.offset = path, 0

    def take(self) -> list[dict]:
        if self.path is None or not self.path.exists():
            return []
        with self.path.open("rb") as f:
            f.seek(self.offset)
            data = f.read()
        end = data.rfind(b"\n") + 1
        self.offset += end
        rows = []
        for line in data[:end].splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        return rows


def model_metrics(rows: list[dict]) -> dict[str, Any]:
    """Per-case model-call metrics from the capture proxy rows (not scored)."""
    rows = [r for r in rows if "response" in r]
    total = 0
    reasoning_total: int | None = None
    truncated = 0
    first_reasoning_tokens = first_reasoning_chars = None
    for i, r in enumerate(rows):
        res = r.get("response") if isinstance(r.get("response"), dict) else {}
        usage = res.get("usage") or {}
        total += usage.get("total_tokens") or 0
        rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if rt is not None:
            reasoning_total = (reasoning_total or 0) + rt
        choice = (res.get("choices") or [{}])[0] if isinstance(res.get("choices"), list) else {}
        if isinstance(choice, dict) and choice.get("finish_reason") == "length":
            truncated += 1
        if i == 0:
            first_reasoning_tokens = rt
            try:
                msg = res["choices"][0]["message"]
                first_reasoning_chars = len(msg.get("reasoning_content") or msg.get("reasoning") or "")
            except (KeyError, IndexError, TypeError):
                pass
    return {"model_calls": len(rows), "total_tokens": total,
            "reasoning_tokens": reasoning_total, "length_truncated_calls": truncated,
            "first_call_reasoning_tokens": first_reasoning_tokens,
            "first_call_reasoning_chars": first_reasoning_chars}


# --- one case ------------------------------------------------------------------
def run_case(cell: Cell, case: dict, header: dict, chains: dict[str, dict], timeout_s: float) -> dict[str, Any]:
    cid = case["id"]
    pre = case.get("preconditions") or {}
    chain_id = pre.get("chain_id")
    result: dict[str, Any] = {"case_id": cid, "category": case.get("category"),
                              "detail_level": case.get("detail_level"), "prompt": case["prompt"],
                              "turns": [], "harness_calls": []}
    cell.mark(cid, 0)
    if chain_id and chains.get(chain_id, {}).get("broken"):
        # the chain's Draft is no longer the expected cumulative state: not the model's failure
        return {**result, "status": "HARNESS_ERROR", "failures": [f"chain {chain_id}: {chains[chain_id]['broken']}"]}
    try:
        seed = setup_case(cell, case, header, result["harness_calls"], chains)
        before = cell.draft()
    except RuntimeError as error:
        if chain_id:
            chains.setdefault(chain_id, {})["broken"] = f"setup of {cid} failed: {error}"
        return {**result, "status": "HARNESS_ERROR", "failures": [str(error)]}
    if seed is not None:
        result["chain_seed"] = seed
        if not seed["ok"]:
            chains[chain_id]["broken"] = f"chain seed failed at {seed['failure']}"
            return {**result, "status": "HARNESS_ERROR", "failures": [f"chain seed: {seed['failure']}"]}
    if chain_id:
        chain = chains[chain_id]
        if chain.pop("fresh_next", False) or seed is not None:
            # The Draft was changed behind the model's back (repair or seed): its chained history
            # no longer describes the Draft, so this step starts a new conversation.
            chain["conversation"] = f"smoke-chain-{chain_id}.{uuid.uuid4().hex[:8]}"
            if seed is None:
                result["fresh_after_repair"] = True
        conversation = chain["conversation"]
    else:
        conversation = f"smoke-case-{cid}.{uuid.uuid4().hex[:8]}"
    budget = float(pre.get("chain_timeout_s", timeout_s))
    calls: list[dict] = []

    def submit(message: str, turn: int) -> dict[str, Any]:
        cell.mark(cid, turn)
        state, status, elapsed = cell.submit(message, conversation, budget)
        response = status.get("response") if isinstance(status, dict) else None
        response = response if isinstance(response, dict) else {}
        tool_calls = response.get("toolCalls") or []
        calls.extend(tool_calls)
        error = status.get("error") if isinstance(status, dict) else None
        result["turns"].append({"turn": turn, "message": message, "state": state, "elapsed_s": round(elapsed, 2),
                                "exit_reason": response.get("exitReason"), "model_turns": response.get("turnCount"),
                                "needs_clarification": response.get("needsClarification"),
                                "tool_calls": [(c.get("toolId"), c.get("status")) for c in tool_calls],
                                "error": error, "status": status})
        return {"state": state, "answer": response.get("message") or "",
                "needs_clarification": bool(response.get("needsClarification"))}

    def read() -> dict[str, Any]:
        d = cell.draft()
        result["turns"][-1]["draft_after"] = d  # per-turn evidence for the offline re-score
        return d

    try:
        verdict = score.run_conversation(case, before, submit, read)
    except RuntimeError as error:
        return {**result, "status": "HARNESS_ERROR", "failures": [str(error)]}
    after = verdict.pop("after")
    result.update(verdict)
    result["draft_before"] = before
    result["draft_after"] = after
    result["changes"] = score.changes(score.Draft(before), score.Draft(after))
    result["tools"] = [(c.get("toolId"), c.get("status")) for c in calls]
    result["metrics"] = {"calls": len(calls), "failed_calls": sum(c.get("status") != "completed" for c in calls),
                         "wall_s": round(sum(t["elapsed_s"] for t in result["turns"]), 2)}
    if chain_id and result["status"] == "FAIL":
        cell.mark(cid, -1)
        result["repair"] = repair_step(cell, case, before)
        chains[chain_id]["fresh_next"] = True
        if not result["repair"].get("ok"):
            chains[chain_id]["broken"] = f"repair of {cid} failed: {result['repair'].get('failure')}"
    return result


# --- report -------------------------------------------------------------------------
def _pct(values: list[float]) -> dict[str, Any]:
    v = sorted(x for x in values if x is not None)
    return {"n": len(v), "p50": v[len(v) // 2] if v else None, "max": v[-1] if v else None}


def summarize(results: list[dict], header: dict[str, Any]) -> dict[str, Any]:
    scored = [r for r in results if r["status"] in ("PASS", "PASS_CLARIFIED", "FAIL")]
    clar_none = [r for r in scored if r["status"] == "PASS_CLARIFIED" and r.get("clarify") == "none"]
    passed = [r for r in scored if r["status"] == "PASS" or (r["status"] == "PASS_CLARIFIED" and r.get("clarify") != "none")]
    rate = lambda n: round(100.0 * n / len(scored), 1) if scored else None
    cats: dict[str, dict[str, int]] = {}
    for r in scored:
        c = cats.setdefault(r.get("category") or "?", {"n": 0, "pass": 0, "pass_after_clarification": 0, "fail": 0})
        c["n"] += 1
        key = "fail" if r["status"] == "FAIL" else "pass_after_clarification" if r in clar_none else "pass"
        c[key] += 1
    m = lambda k: [r.get("metrics", {}).get(k) for r in scored]
    return {**header, "scored": len(scored), "pass": len(passed), "pass_after_clarification": len(clar_none),
            "fail": sum(r["status"] == "FAIL" for r in scored),
            "not_run": sum(r["status"] == "NOT_RUN" for r in results),
            "harness_errors": sum(r["status"] == "HARNESS_ERROR" for r in results),
            "pass_rate": rate(len(passed)), "pass_rate_with_clarification": rate(len(passed) + len(clar_none)),
            "chain_repairs": [{"case_id": r["case_id"], "ok": r["repair"].get("ok")} for r in results if r.get("repair")],
            "efficiency": {k: _pct(m(k)) for k in ("calls", "failed_calls", "wall_s", "model_calls",
                                                    "total_tokens", "reasoning_tokens", "length_truncated_calls",
                                                    "first_call_reasoning_tokens", "first_call_reasoning_chars")},
            "length_truncated_calls_total": sum(r.get("metrics", {}).get("length_truncated_calls") or 0 for r in scored),
            "categories": cats}


def log_line(r: dict) -> str:
    met = r.get("metrics") or {}
    first = (r.get("failures") or [""])[0]
    tag = {"PASS": "PASS", "PASS_CLARIFIED": "PASS_CLARIFIED", "FAIL": "FAIL"}.get(r["status"], r["status"])
    line = f"{tag} {r['case_id']} {met.get('wall_s', 0):.1f} {met.get('calls', 0)}/{met.get('failed_calls', 0)} {first}".rstrip()
    if r.get("fresh_after_repair"):
        line += "  [fresh conversation after repair]"
    if r.get("chain_seed"):
        line += f"  [seeded from {r['chain_seed']['steps'][0]}..{r['chain_seed']['steps'][-1]}]"
    if r.get("repair"):
        line += f"  [repair {'ok' if r['repair'].get('ok') else 'FAILED: ' + str(r['repair'].get('failure'))}]"
    return line.rstrip()


def category_table(summary: dict) -> str:
    rows = [f"{'category':40s} {'n':>3s} {'pass':>5s} {'clar':>5s} {'fail':>5s}"]
    for c, v in sorted(summary["categories"].items()):
        rows.append(f"{c:40s} {v['n']:3d} {v['pass']:5d} {v['pass_after_clarification']:5d} {v['fail']:5d}")
    return "\n".join(rows)


# --- run environment -------------------------------------------------------------------
def endpoint_refusal(endpoint: dict, proxy_port: int, reasoning: str | None,
                     max_output_tokens: int | None) -> str | None:
    """Why the live mode's model endpoint does not match the run's expectations, or None."""
    want_base = f"http://127.0.0.1:{proxy_port}/v1"
    if endpoint.get("base_url") != want_base:
        return (f"the live model endpoint is {endpoint.get('base_url')!r}, not the capture proxy {want_base!r}.\n"
                f"Point it with: assist_setup_noload.py {endpoint.get('model_id')} {want_base}")
    contract = endpoint.get("request_contract") or {}
    if reasoning and contract.get("reasoning") != reasoning:
        return f"the live request_contract.reasoning is {contract.get('reasoning')!r}; the run expects {reasoning!r}"
    if max_output_tokens is not None and contract.get("max_output_tokens") != max_output_tokens:
        return (f"the live request_contract.max_output_tokens is {contract.get('max_output_tokens')!r}; "
                f"the run expects {max_output_tokens}")
    return None



def scenario_of(case: dict, header: dict) -> str:
    s = case.get("scenario") or Path(str(header.get("scenario", ""))).name.removesuffix(".scenario")
    return SCENARIO_ALIASES.get(s, s)


def start_proxy(port: int, upstream: str, out: Path, case_file: Path, max_bytes: int,
                sampling: str | None = None) -> subprocess.Popen:
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(f"port {port} is in use (an old capture proxy?); stop it first")
    log = open(out.with_suffix(".proxy.log"), "a")
    proc = subprocess.Popen([sys.executable, str(HERE / "model_capture_proxy.py"), "--listen", str(port),
                             "--upstream", upstream, "--out", str(out), "--case-file", str(case_file),
                             "--max-bytes", str(max_bytes)] + (["--sampling", sampling] if sampling else []),
                            stdout=log, stderr=log)
    for _ in range(50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return proc
        if proc.poll() is not None:
            raise SystemExit(f"capture proxy exited ({proc.returncode}); see {log.name}")
        time.sleep(0.1)
    proc.terminate()
    raise SystemExit("capture proxy did not start listening")


# --- offline re-score from a v2 report --------------------------------------------------------
def rejudge(case: dict, old: dict) -> dict[str, Any]:
    """Judge one finished v2 case again from its own report entry (recorded turns, Draft
    before and after) with the current scorer. No Composer, no model."""
    turns = old.get("turns") or []
    before = old.get("draft_before")
    if before is None or old.get("draft_after") is None or not turns:
        return {"status": "NOT_RECONSTRUCTABLE", "reason": "no Draft before/after or turns in the report"}
    drafts = []
    for k, t in enumerate(turns):
        if t.get("draft_after") is not None:
            drafts.append(t["draft_after"])
        elif k == len(turns) - 1:
            drafts.append(old["draft_after"])
        elif old.get("asked") and not t.get("needs_clarification"):
            drafts.append(before)  # the old flow asked only on an unchanged Draft
        else:
            return {"status": "NOT_RECONSTRUCTABLE", "reason": f"Draft after turn {k + 1} not recorded"}
    state = {"i": 0}

    def submit(message: str, turn: int) -> dict[str, Any]:
        if state["i"] >= len(turns):
            raise LookupError(f"turn {turn} was not recorded")
        t = turns[state["i"]]
        state["i"] += 1
        response = ((t.get("status") or {}).get("response") or {})
        return {"state": t.get("state"), "answer": response.get("message") or "",
                "needs_clarification": bool(t.get("needs_clarification"))}

    try:
        verdict = score.run_conversation(case, before, submit, lambda: drafts[state["i"] - 1])
    except LookupError as error:
        return {"status": "NOT_RECONSTRUCTABLE", "reason": str(error)}
    verdict.pop("after")
    return verdict


def rescore_report(corpus: dict, old: dict) -> list[dict]:
    cases = {c["id"]: c for c in corpus["cases"]}
    out = []
    for r in old.get("results", []):
        if r.get("status") not in ("PASS", "PASS_CLARIFIED", "FAIL") or r["case_id"] not in cases:
            out.append(r)
            continue
        new = {**r, **rejudge(cases[r["case_id"]], r), "old_status": r["status"],
               "old_failures": r.get("failures")}
        out.append(new)
    return out


# --- re-score from saved evidence ----------------------------------------------------------
CONTEXT_PREFIX = "ManyForge bounded assistant context:"


def captured_before(rows: list[dict], case_id: str) -> dict | None:
    """The Draft the model saw at the first call of a case (pre-A1 prompts embed it as JSON)."""
    for r in rows:
        if r.get("case_id") != case_id or r.get("corpus_turn") != 1 or r.get("model_call") != 1:
            continue
        try:
            text = r["request"]["messages"][0]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        if not isinstance(text, str) or not text.startswith(CONTEXT_PREFIX):
            return None
        try:
            ctx, _ = json.JSONDecoder().raw_decode(text[len(CONTEXT_PREFIX):].lstrip())
        except ValueError:
            return None
        draft = ctx.get("draft") or {}
        return {"authoring": draft.get("authoring") or {}, "diagnostics": draft.get("diagnostics") or []}
    return None


def rescore(cell: Cell, corpus: dict, old: dict, rows: list[dict]) -> list[dict]:
    """Rebuild each old case's Draft before/after by replaying its recorded turn-1 Draft
    mutations through Composer (same mutation path the bridge used), verify the replay
    against the recorded results, then judge with the v2 outcome. Writes the Draft."""
    cases = {c["id"]: c for c in corpus["cases"]}
    rescore_chains: dict[str, dict] = {}
    out = []
    for r in old.get("results", []):
        cid = r.get("case_id")
        case = cases.get(cid)
        res: dict[str, Any] = {"case_id": cid, "category": r.get("category")}
        if case is None or not r.get("turns"):
            out.append({**res, "status": "NOT_RECONSTRUCTABLE", "reason": "no v2 case or no recorded turns"})
            continue
        turn = r["turns"][0]
        response = (turn.get("status") or {}).get("response") or {}
        if r.get("recovery_turn"):
            res["note"] = "harness recovery turn excluded; judged on turn 1"
        try:
            setup_case(cell, case, corpus, [], rescore_chains)
            before = cell.draft()
        except RuntimeError as error:
            out.append({**res, "status": "NOT_RECONSTRUCTABLE", "reason": str(error)})
            continue
        seen = captured_before(rows, cid)
        if seen is not None and not score.values_equal(
                {k: v for k, v in seen["authoring"].items() if k not in ("schema_version", "positions")},
                {k: v for k, v in before["authoring"].items() if k not in ("schema_version", "positions")}):
            out.append({**res, "status": "NOT_RECONSTRUCTABLE",
                        "reason": "replayed Draft before differs from the captured prompt Draft"})
            continue
        mismatch = None
        last_rev = None
        for c in response.get("toolCalls") or []:
            if c.get("effect") != "composer_draft_mutating" or c.get("status") != "completed":
                continue
            recorded = c.get("result") or {}
            try:
                got = cell.mutate(c["toolId"], c.get("arguments") or {})
            except RuntimeError as error:
                mismatch = f"replay of {c['toolId']} refused: {error}"
                break
            got_details = got.get("details", got.get("result", {}).get("details") if isinstance(got.get("result"), dict) else None)
            if recorded.get("details") is not None and got_details is not None and got_details != recorded["details"]:
                mismatch = f"replay of {c['toolId']} gave {got_details}, recorded {recorded['details']}"
                break
            last_rev = recorded.get("programRevision", last_rev)
        after = cell.draft()
        if mismatch is None and last_rev is not None and after.get("program_revision") not in (None, last_rev):
            mismatch = "replayed program revision differs from the recorded one"
        if mismatch:
            out.append({**res, "status": "NOT_RECONSTRUCTABLE", "reason": mismatch})
            continue
        answer = response.get("message") or ""
        outcome = case.get("outcome") or {}
        changed = score.draft_changed(score.Draft(before), score.Draft(after))
        asked = bool(response.get("needsClarification")) or (not changed and score.is_question(answer))
        mode = outcome.get("clarify", "none")
        if asked and outcome.get("clarification_answer") and mode != "required":
            out.append({**res, "status": "NOT_RECONSTRUCTABLE",
                        "reason": "the model asked; the clarification turn is not in the evidence"})
            continue
        if mode == "required" or (asked and mode == "allowed"):
            ok = asked and not changed
            fails = [] if ok else ["clarify: expected a question with no Draft change"]
            if ok and outcome.get("clarification_answer"):
                out.append({**res, "status": "NOT_RECONSTRUCTABLE",
                            "reason": "asked correctly; the clarification turn is not in the evidence"})
                continue
        elif asked:
            fails = ["clarify: the model asked instead of acting"]
        else:
            fails = score.judge(outcome, before, after, answer)
        out.append({**res, "status": "FAIL" if fails else "PASS", "failures": fails,
                    "old_status": r.get("status"), "answer": answer,
                    "changes": score.changes(score.Draft(before), score.Draft(after))})
    return out


# --- main ------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--corpus", default=str(HERE / "smoke_corpus_v3.yaml"))
    ap.add_argument("--composer", default="http://127.0.0.1:3000")
    ap.add_argument("--headless", default="http://127.0.0.1:9010")
    ap.add_argument("--filter", default=None, help="regex on case id")
    ap.add_argument("--timeout-s", type=float, default=180.0, help="per-turn budget (bridge cap is 180 s)")
    ap.add_argument("--report", help="JSON report; the text log goes to stdout")
    ap.add_argument("--reasoning", choices=("on", "off"),
                    help="expected live request_contract.reasoning of the mode's model endpoint; the run refuses on mismatch")
    ap.add_argument("--max-output-tokens", type=int,
                    help="expected live request_contract.max_output_tokens; the run refuses on mismatch")
    ap.add_argument("--proxy-port", type=int, default=8010)
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--capture-max-bytes", type=int, default=2 * 1024 ** 3)
    ap.add_argument("--sampling", help='experiment: JSON the proxy merges into every model request, e.g. {"temperature":null}')
    ap.add_argument("--validate", action="store_true", help="validate the corpus and exit")
    ap.add_argument("--rescore", help="old run report to re-score (replays its Draft mutations)")
    ap.add_argument("--captures", help="the old run's model-requests.jsonl (with --rescore)")
    ap.add_argument("--rescore-report", help="v2 run report to re-judge offline with the current scorer")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    corpus = yaml.safe_load(open(args.corpus))
    problems = score.validate_corpus(corpus)
    if args.validate:
        print("\n".join(problems) or f"{len(corpus.get('cases') or [])} cases valid")
        return 1 if problems else 0
    if problems:
        print("corpus invalid:\n  " + "\n  ".join(problems))
        return 2
    if not args.report:
        ap.error("--report is required")
    cases = corpus["cases"]
    if args.filter:
        rx = re.compile(args.filter)
        cases = [c for c in cases if rx.search(c["id"])]
    report = Path(args.report).expanduser()

    if args.rescore_report:
        old = json.load(open(args.rescore_report))
        results = rescore_report(corpus, old)
        for r in results:
            if r.get("old_status") and r["old_status"] != r["status"]:
                print(f"CHANGED {r['old_status']} -> {log_line(r)}")
        header = {k: v for k, v in (old.get("summary") or {}).items()
                  if k in ("model", "reasoning", "max_output_tokens", "scenario", "corpus", "captures", "started")}
        summary = {**summarize(results, header), "rescored_from": args.rescore_report}
        print(category_table(summary))
        print(json.dumps({k: v for k, v in summary.items() if k != "categories"}, indent=1, default=str))
        report.write_text(json.dumps({"summary": summary, "results": results}, indent=1, default=str))
        return 0
    if args.rescore:
        cell = Cell(args.composer, args.headless)
        old = json.load(open(args.rescore))
        rows = [json.loads(l) for l in open(args.captures)] if args.captures else []
        results = rescore(cell, corpus, old, rows)
        cell.discard()
        for r in results:
            print(f"{r['status']} {r['case_id']} {(r.get('failures') or [r.get('reason', '')])[0] if r.get('failures') or r.get('reason') else ''}")
        report.write_text(json.dumps({"rescored_from": args.rescore, "results": results}, indent=1, default=str))
        return 0

    case_file = report.with_suffix(".case.json")
    captures_path = report.with_suffix(".model-requests.jsonl")
    cell = Cell(args.composer, args.headless, case_file)
    endpoint = cell.get(f"/api/assistant/modes/{MODE}")["model_endpoint"]
    refusal = endpoint_refusal(endpoint, args.proxy_port, args.reasoning, args.max_output_tokens)
    if refusal:
        print(refusal)
        return 2
    contract = endpoint.get("request_contract") or {}
    reasoning, max_output_tokens = contract.get("reasoning"), contract.get("max_output_tokens")
    current = (cell.get("/api/scenarios/current").get("current_scenario") or {}).get("scenario_id")
    model = endpoint["model_id"]
    case_file.write_text(json.dumps({"case_id": None, "turn": None}))
    sampling = json.dumps(json.loads(args.sampling)) if args.sampling else None
    proxy = start_proxy(args.proxy_port, args.upstream, captures_path, case_file, args.capture_max_bytes, sampling)
    header = {"model": model, "reasoning": reasoning, "max_output_tokens": max_output_tokens, "scenario": current, "corpus": args.corpus,
              "sampling_override": sampling,
              "captures": str(captures_path), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(f"{len(cases)} cases; model={model}; reasoning={reasoning}; max_output_tokens={max_output_tokens}; sampling={sampling or 'bridge'}; scenario={current}; captures={captures_path}")
    captures = Captures(captures_path)
    chains: dict[str, dict] = {}
    results = []
    try:
        for case in cases:
            if case.get("status") == "excluded":
                results.append({"case_id": case["id"], "category": case.get("category"), "status": "EXCLUDED",
                                "reason": case.get("exclusion_reason")})
                continue
            if scenario_of(case, corpus) != current:
                results.append({"case_id": case["id"], "category": case.get("category"), "status": "NOT_RUN",
                                "reason": f"scenario {scenario_of(case, corpus)} not loaded"})
                print(f"NOT_RUN {case['id']} scenario {scenario_of(case, corpus)}")
                continue
            r = run_case(cell, case, corpus, chains, args.timeout_s)
            time.sleep(0.3)  # let the proxy flush the last call
            r.setdefault("metrics", {}).update(model_metrics(
                [row for row in captures.take() if row.get("case_id") == case["id"] and (row.get("corpus_turn") or 0) > 0]))
            results.append(r)
            print(log_line(r))
            report.write_text(json.dumps({"summary": summarize(results, header), "results": results},
                                         indent=1, default=str))
        cell.mark(None, None)
        cell.discard()
    finally:
        proxy.terminate()
    summary = summarize(results, header)
    print(category_table(summary))
    print(json.dumps({k: v for k, v in summary.items() if k != "categories"}, indent=1, default=str))
    report.write_text(json.dumps({"summary": summary, "results": results}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
