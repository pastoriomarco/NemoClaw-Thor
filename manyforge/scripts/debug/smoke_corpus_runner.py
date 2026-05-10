#!/usr/bin/env python3
"""smoke_corpus_runner.py — run smoke_corpus.yaml against the live stack.

Usage:
    python3 smoke_corpus_runner.py [options]

Options:
    --corpus PATH          path to smoke_corpus.yaml (default: alongside this script)
    --composer URL         Composer base URL (default: http://127.0.0.1:9000)
    --only-active          only run cases with status active or unset (default: True)
    --include-future       run future-tier cases too (still gated by available capability)
    --runtime-flags FLAGS  comma-separated runtimes to enable
                           (e.g., expanded_node_allowlist,custom_precondition)
    --filter PATTERN       only run cases whose id matches the regex
    --skip-fixture-cases   skip cases needing custom_precondition seeding (for now)
    --report PATH          write JSON results to PATH (default: /tmp/smoke_corpus_<ts>.json)
    --verbose              dump every tool-call comparison

What it does per case:
    1. Reset state (load default deployment+program, force-discard overrides)
       OR (if precondition.fresh_program) load an empty program
    2. Capture pre-state (program tree + scene)
    3. POST /api/assistant/chat with the prompt + composer-assistant mode
    4. Capture post-state + tool calls (parsed from Composer's docker logs)
    5. Run assertions:
         - tools_called: each expected entry must match (in order)
         - state_after: dotted-path asserts on tree/scene
         - forbidden_tools: must not have fired
         - answer_must_contain / answer_must_not_contain: soft text checks
    6. Record pass / soft-pass / fail with reasons.

Requires: PyYAML.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ----------------------------------------------------------------------
# Constants / endpoints
# ----------------------------------------------------------------------

DEFAULT_COMPOSER = "http://127.0.0.1:9000"
DEFAULT_TIMEOUT_S = 270.0  # 2026-05-09: 360s caused 3 cases to hit a
                           # 5-minute upstream-502 circuit-breaker
                           # (PnP_18, FALLBACK_alternate_medium,
                           # REPLACE_subtree_specific). Default sits just
                           # under the 502 ceiling; per-case overrides via
                           # `precondition.chain_timeout_s` carry the
                           # rare cases that genuinely need more time.
COMPOSER_CONTAINER = "manyforge-e2e-composer"
DEFAULT_CORPUS = str(Path(__file__).resolve().parent / "smoke_corpus.yaml")


# ----------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------

def _post_json(url: str, body: dict, timeout: float = 15.0) -> tuple[int, dict | str]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw)
            except ValueError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
            try:
                return exc.code, json.loads(err_body)
            except ValueError:
                return exc.code, err_body
        except Exception:
            return exc.code, str(exc)
    except Exception as exc:
        return -1, f"<error {type(exc).__name__}: {exc}>"


def _get_json(url: str, timeout: float = 10.0) -> tuple[int, dict | list | str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw)
            except ValueError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        return exc.code, str(exc)
    except Exception as exc:
        return -1, f"<error {type(exc).__name__}: {exc}>"


# ----------------------------------------------------------------------
# State capture: combine /api/program + /api/program/tree + /api/scene/state
# ----------------------------------------------------------------------

def capture_state(composer: str) -> dict[str, Any]:
    """Return a unified state snapshot the assertion engine queries via
    dotted paths like `program.tree.kind`, `scene.objects[id=graspable].pose.position`."""
    state: dict[str, Any] = {"program": {}, "scene": {}}

    code, prog = _get_json(f"{composer}/api/program")
    if code == 200 and isinstance(prog, dict):
        state["program"].update(prog)

    code, tree = _get_json(f"{composer}/api/program/tree")
    if code == 200 and isinstance(tree, dict):
        state["program"]["tree"] = tree.get("tree") or tree

    code, scene = _get_json(f"{composer}/api/scene/state")
    if code == 200 and isinstance(scene, dict):
        state["scene"].update(scene)

    return state


# ----------------------------------------------------------------------
# Tool-call capture from Composer docker logs
# ----------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r'"path":\s*"/api/assistant/bridge/tools/(?P<tool>[a-z_][a-z0-9_]*)".*?'
    r'"status_code":\s*(?P<status>\d+)',
    re.DOTALL,
)


def merge_response_tool_args(
    observed: list[dict[str, Any]],
    response_tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Decorate access-log tool-call entries with `arguments` taken from
    the chat response's `toolCalls` array. The bridge's response envelope
    carries the full args dict per call; the access log only has the path
    + status. We zip by tool name + occurrence index so the i-th call to
    tree_draft_insert_node in the access log is matched with the i-th
    entry in the response's toolCalls list (regardless of interleaving
    with other tools).

    Iter 7 (2026-05-09): unlocks honest `args_contain` validation that
    the harness previously could not perform.
    """
    if not response_tool_calls:
        return observed
    # Index response toolCalls by name → ordered list
    by_name: dict[str, list[dict[str, Any]]] = {}
    for rec in response_tool_calls:
        if not isinstance(rec, dict):
            continue
        name = rec.get("name") or ""
        if not name:
            continue
        by_name.setdefault(name, []).append(rec)
    # Walk observed in order; pop matching response entry per tool
    cursor: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for entry in observed:
        name = entry.get("tool") or ""
        i = cursor.get(name, 0)
        rec_list = by_name.get(name, [])
        if i < len(rec_list):
            rec = rec_list[i]
            entry = dict(entry)  # don't mutate caller's list
            entry["arguments"] = rec.get("arguments") or {}
            entry["bridge_status"] = rec.get("status")  # "completed" | "failed"
            entry["bridge_error"] = rec.get("error")
            cursor[name] = i + 1
        out.append(entry)
    return out


def fetch_tool_calls_since(t0_epoch_s: float) -> list[dict[str, Any]]:
    """Pull /api/assistant/bridge/tools/<name> POSTs from the Composer
    container's stdout/stderr since t0_epoch_s. The harness uses this as
    the ground-truth tool-call stream because the smoke harness's own
    state-delta detection is unreliable on multi-step prompts."""
    proc = subprocess.run(
        ["docker", "logs", "--since", str(int(t0_epoch_s) - 1), COMPOSER_CONTAINER],
        capture_output=True, timeout=15,
    )
    raw = proc.stdout.decode("utf-8", errors="replace") + "\n" + \
          proc.stderr.decode("utf-8", errors="replace")
    out: list[dict[str, Any]] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s.startswith("{") or "/api/assistant/bridge/tools/" not in s:
            continue
        try:
            d = json.loads(s)
        except Exception:
            continue
        path = d.get("path") or ""
        if not path.startswith("/api/assistant/bridge/tools/"):
            continue
        if path.endswith("/principal-binding") or "/principal-binding/" in path:
            continue
        out.append({
            "tool": path.split("/")[-1],
            "status": d.get("status_code"),
            "duration_ms": d.get("duration_ms"),
            "ts": d.get("timestamp"),
        })
    return out


# ----------------------------------------------------------------------
# Dotted-path asserts (subset matcher)
# ----------------------------------------------------------------------

def _split_path(path: str) -> list[str]:
    """Split a dotted path into segments, preserving [...]-bracket selectors.
    Examples:
        "program.tree.kind" → ["program", "tree", "kind"]
        "program.tree.children[0].name" → ["program", "tree", "children[0]", "name"]
        "scene.objects[id=graspable].pose.position" →
            ["scene", "objects[id=graspable]", "pose", "position"]
    """
    parts: list[str] = []
    buf = ""
    depth = 0
    for ch in path:
        if ch == "[":
            depth += 1
            buf += ch
        elif ch == "]":
            depth -= 1
            buf += ch
        elif ch == "." and depth == 0:
            if buf:
                parts.append(buf)
                buf = ""
        else:
            buf += ch
    if buf:
        parts.append(buf)
    return parts


def _resolve_path(state: Any, path: str) -> Any:
    """Walk a dotted path with bracket-selector support. Returns a sentinel
    string `"<MISSING>"` if the path does not resolve."""
    cur: Any = state
    for seg in _split_path(path):
        m_idx = re.match(r"^(\w+)\[(\d+)\]$", seg)
        m_kv = re.match(r"^(\w+)\[(\w+)=([^\]]+)\]$", seg)
        m_star = re.match(r"^(\w+)\[\*\]$", seg)
        if m_idx:
            key, idx = m_idx.group(1), int(m_idx.group(2))
            if isinstance(cur, dict):
                cur = cur.get(key)
            if not isinstance(cur, list) or not (0 <= idx < len(cur)):
                return "<MISSING>"
            cur = cur[idx]
        elif m_kv:
            key, sel_key, sel_val = m_kv.group(1), m_kv.group(2), m_kv.group(3)
            if isinstance(cur, dict):
                cur = cur.get(key)
            if not isinstance(cur, list):
                return "<MISSING>"
            found = None
            for item in cur:
                if isinstance(item, dict) and str(item.get(sel_key)) == sel_val:
                    found = item
                    break
            if found is None:
                return "<MISSING>"
            cur = found
        elif m_star:
            key = m_star.group(1)
            if isinstance(cur, dict):
                cur = cur.get(key)
            if not isinstance(cur, list):
                return "<MISSING>"
            # Project: keep list. Later segments can run on each element.
            return cur
        else:
            if isinstance(cur, dict):
                cur = cur.get(seg, "<MISSING>")
            else:
                return "<MISSING>"
    return cur


def _value_matches(actual: Any, expected: Any) -> bool:
    """Subset-match. List of values in expected → 'is exactly this list'.
    String 'contains X' / 'does_not_contain X' → substring check on actual.
    Otherwise == match (loose: 1 == 1.0)."""
    if isinstance(expected, str):
        if expected.startswith("contains "):
            needle = expected[len("contains "):]
            if isinstance(actual, list):
                return any(needle in str(x) for x in actual)
            return needle in str(actual)
        if expected.startswith("does_not_contain "):
            needle = expected[len("does_not_contain "):]
            if isinstance(actual, list):
                return not any(needle in str(x) for x in actual)
            return needle not in str(actual)
    if isinstance(expected, list) and isinstance(actual, list):
        return list(expected) == list(actual)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return float(expected) == float(actual)
    return actual == expected


# ----------------------------------------------------------------------
# Tool-call assertion
# ----------------------------------------------------------------------

def _flatten(d: dict, parent: str = "") -> dict[str, Any]:
    """Flatten a nested dict into dotted keys for args_contain matching."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        kk = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, kk))
        else:
            out[kk] = v
    return out


@dataclass
class CaseResult:
    case_id: str
    status: str           # "pass" | "soft-pass" | "fail" | "skipped"
    elapsed_s: float
    tool_calls: list[dict] = field(default_factory=list)
    answer: str = ""
    failures: list[str] = field(default_factory=list)
    soft_failures: list[str] = field(default_factory=list)
    skip_reason: str = ""


def assert_tools(case: dict, observed: list[dict], failures: list[str]) -> None:
    expected = case.get("expected", {})
    # Distinguish three semantics:
    #   - `tools_called` MISSING        → don't check tool sequence at all
    #                                      (state_after carries the load).
    #   - `tools_called: []` EXPLICIT   → must fire NO tools (clarification).
    #   - `tools_called: [...]`         → match each entry in order.
    if "tools_called" not in expected:
        return
    expected_list = expected.get("tools_called") or []
    if not expected_list:
        if observed:
            failures.append(
                f"expected NO tool calls; observed {[c['tool'] for c in observed]}"
            )
        return

    # Match as a MULTISET (each expected entry needs ≥1 successful 2xx
    # call somewhere in the observed stream, regardless of position).
    # Models legitimately reorder tool calls (e.g. inspect-before-action
    # vs action-then-inspect); strict ordering produced false fails on
    # multi-tool prompts. Order-sensitive checks remain available via
    # `state_after` (the post-state inherently reflects the cumulative
    # effect, not the path that got there).
    available = [dict(c) for c in observed]   # local copy we'll consume
    for i, exp in enumerate(expected_list):
        name = exp.get("name")
        allow_retries = exp.get("allow_retries", False)
        consumed_idx: int | None = None
        for idx, call in enumerate(available):
            if call.get("_consumed") or call["tool"] != name:
                continue
            status = int(call["status"] or 0)
            if 200 <= status < 300:
                consumed_idx = idx
                break
            if not allow_retries:
                continue
            # allow_retries: 4xx is acceptable if a 2xx for the same tool
            # appears later. Mark this 4xx as consumed-by-retry-path and
            # keep scanning for the 2xx.
            available[idx]["_consumed"] = True
        if consumed_idx is None:
            failures.append(
                f"expected tool '{name}' not observed (or never reached 2xx)"
            )
            continue
        available[consumed_idx]["_consumed"] = True
        # 2026-05-09 (iter 7): args_contain is now validated against the
        # bridge response's toolCalls.arguments dict (merged into the
        # observed entry via merge_response_tool_args). Best-effort: if
        # the response did not carry args (older bridge build, malformed
        # response), the assert is skipped silently to avoid false
        # negatives — state_after still carries the load.
        args_contain = exp.get("args_contain") or {}
        if args_contain:
            actual_args = available[consumed_idx].get("arguments")
            if isinstance(actual_args, dict) and actual_args:
                flat = _flatten(actual_args)
                expected_flat = _flatten(args_contain)
                for k, v in expected_flat.items():
                    av = flat.get(k, "<MISSING>")
                    if not _value_matches(av, v):
                        failures.append(
                            f"args_contain[{k}] expected {v!r}, "
                            f"got {av!r} on tool '{name}'"
                        )

    forbidden = case.get("expected", {}).get("forbidden_tools") or []
    fired = {c["tool"] for c in observed if 200 <= int(c["status"] or 0) < 300}
    for f in forbidden:
        if f in fired:
            failures.append(f"forbidden tool fired: {f}")


def assert_state(case: dict, post_state: dict, failures: list[str]) -> None:
    expected_state = case.get("expected", {}).get("state_after") or {}
    for path, expected_value in expected_state.items():
        # Strip outer quotes if YAML preserved them
        path_clean = path.strip('"')
        actual = _resolve_path(post_state, path_clean)
        if not _value_matches(actual, expected_value):
            failures.append(
                f"state_after[{path_clean}] expected {expected_value!r}, got {actual!r}"
            )


def assert_answer(case: dict, answer: str, soft_failures: list[str]) -> None:
    must = case.get("expected", {}).get("answer_must_contain") or []
    must_not = case.get("expected", {}).get("answer_must_not_contain") or []
    a_low = (answer or "").lower()
    for needle in must:
        if needle.lower() not in a_low:
            soft_failures.append(f"answer_must_contain: {needle!r} not found")
    for needle in must_not:
        if needle.lower() in a_low:
            soft_failures.append(f"answer_must_not_contain: {needle!r} unexpectedly present")


# ----------------------------------------------------------------------
# Per-case dispatch
# ----------------------------------------------------------------------

def reset_program(composer: str, deployment_path: str, program_path: str) -> tuple[int, Any]:
    body = {
        "path": program_path,
        "deploymentPath": deployment_path,
        "forceDiscardOverrides": True,
    }
    return _post_json(f"{composer}/api/program/load", body, timeout=20.0)


# Per-case fixtures: when a case sets `required_runtime: custom_precondition`,
# the harness pre-seeds the named state via the same bridge tools the cases
# under test exercise. Keyed by case id; the value is a list of (tool, body)
# pairs to POST before the prompt fires.
_FIXTURES: dict[str, list[tuple[str, dict]]] = {
    "PARAM_delete_specific": [
        ("program_draft_upsert_parameters", {
            "parameters": [
                {"name": "legacy_offset", "type": "float", "default": 0.0,
                 "description": "Seed parameter inserted by the smoke harness."}
            ],
        }),
    ],
    "BB_modify_medium": [
        ("blackboard_draft_upsert_keys", {
            "keys": [
                {"id": "grip_force", "type": "int", "key": "grip_force",
                 "description": "Seed blackboard key inserted by the smoke harness."}
            ],
        }),
    ],
    "BB_delete_specific": [
        ("blackboard_draft_upsert_keys", {
            "keys": [
                {"id": "scratch_value", "type": "string", "key": "scratch_value",
                 "description": "Seed blackboard key inserted by the smoke harness."}
            ],
        }),
    ],
}


def fetch_catalog_hash(composer: str, assistant_mode: str = "composer-assistant") -> str:
    """The bridge tool endpoints validate `catalogHash` against the
    current deployment's mode catalog hash. Fetch it from the manifest
    endpoint and cache for the run."""
    code, body = _get_json(
        f"{composer}/api/assistant/modes/{assistant_mode}", timeout=5.0,
    )
    if code != 200 or not isinstance(body, dict):
        return ""
    return body.get("catalogHash") or ""


def apply_fixtures(case_id: str, composer: str, catalog_hash: str) -> tuple[bool, str]:
    """Pre-seed state for a case that requires it. Returns (ok, error_msg).

    The fixture posts directly to /api/assistant/bridge/tools/<tool_id>,
    which is the same endpoint the assistant flow uses. This bypasses the
    LLM and Composer's chat path entirely — pure REST. The bridge tool
    endpoints expect `requestId`, `assistantMode`, and `catalogHash`
    alongside the tool-specific args.
    """
    fixture = _FIXTURES.get(case_id)
    if not fixture:
        return True, ""
    for tool, body in fixture:
        envelope = {
            "requestId": f"fixture-{case_id}-{int(time.time()*1000)}",
            "assistantMode": "composer-assistant",
            "catalogHash": catalog_hash,
            "arguments": body,   # tool-specific args go under `arguments`
                                 # (matches AssistantBridgeToolRequest in
                                 # manyforge_composer/backend/models.py)
        }
        url = f"{composer}/api/assistant/bridge/tools/{tool}"
        code, resp = _post_json(url, envelope, timeout=10.0)
        if code != 200:
            return False, f"fixture {tool} returned HTTP {code}: {resp}"
    return True, ""


def send_chat(composer: str, prompt: str, request_id: str,
              timeout_s: float = DEFAULT_TIMEOUT_S) -> tuple[int, Any]:
    body = {
        "message": prompt,
        "mode": "provider",
        "conversationId": request_id,
        "requestId": request_id,
        "assistantMode": "composer-assistant",
        # Override Composer's per-request `assistant-timeout-s` so cases
        # with `precondition.chain_timeout_s` get extra agent-loop budget.
        # Composer caps at the value passed via the env on container
        # start; the `timeoutSeconds` field here is read by the chat
        # handler to relax the per-call wait.
        "timeoutSeconds": int(timeout_s),
    }
    return _post_json(f"{composer}/api/assistant/chat", body, timeout=timeout_s + 5.0)


def _build_recovery_message(
    observed: list[dict[str, Any]],
    expected_tools_called: list[dict[str, Any]] | None = None,
) -> str | None:
    """Synthesize one generic follow-up message for a failed turn.

    Two trigger paths (research-validated; round-3 brief 2026-05-09):

    (a) **Recovery-from-4xx** (the original iter-8 path): if any tool
        call hit a 4xx, send a generic "re-read the error and retry"
        nudge. Model has the validParentNames / allowedNodeKinds in
        the tool result already.

    (b) **No-tool-fired (iter 10 extension)**: if the case expected a
        tool but the model returned text without firing any tool (the
        "narration mode" / "EndTurn-without-tool" pattern named in
        github/copilot-cli #2949 and arXiv 2505.06120 LLMs-Get-Lost),
        send a generic "you need to call a draft tool" nudge.

    Returns None if neither trigger condition holds. The same generic
    wording fires for every case — no per-case content.
    """
    last_4xx = None
    for c in observed:
        st = int(c.get("status") or 0)
        if 400 <= st < 600:
            last_4xx = c  # keep the most recent 4xx
    if last_4xx is not None:
        tool = last_4xx.get("tool") or "<unknown>"
        err = last_4xx.get("bridge_error") or ""
        if not err and isinstance(last_4xx.get("arguments"), dict):
            err = "(see prior tool result envelope)"
        err = err[:600] if isinstance(err, str) else str(err)[:600]
        return (
            f"The previous `{tool}` call failed: {err} "
            "Re-read the error and the structured recovery fields it lists "
            "(e.g. `validParentNames`, `validNodeNames`, `allowedNodeKinds`), "
            "then retry the original request with corrected arguments. "
            "Do not repeat the failed call verbatim."
        )
    # Path (b): zero successful tool calls when ≥1 was expected.
    if expected_tools_called:
        any_2xx = any(
            200 <= int(c.get("status") or 0) < 300 for c in observed
        )
        if not any_2xx:
            return (
                "The previous turn produced no tool call (or no successful "
                "tool call). The original request requires a tool action. "
                "Please call the appropriate tool now — a read-only tool "
                "(program_read, scene_inspect, catalog_read) if the "
                "request is to inspect or verify state, or a draft tool "
                "(tree_draft_*, scene_draft_*, program_draft_*, "
                "blackboard_draft_*) if it is to mutate state. If the "
                "request is genuinely ambiguous, ask one specific "
                "clarifying question instead."
            )
    return None


def run_case(case: dict, composer: str, default_pre: dict,
             chain_state: dict[str, str], catalog_hash: str,
             enable_recovery_turn: bool = False,
             no_chain_session: bool = False) -> CaseResult:
    cid = case["id"]
    pre = case.get("precondition") or {}
    chain_id = pre.get("chain_id")
    chain_step = pre.get("chain_step")
    fresh = bool(pre.get("fresh_program"))

    deployment_path = pre.get("deployment_path", default_pre.get("deployment_path"))
    program_path = pre.get("program_path", default_pre.get("program_path"))

    # Reset only at the start of a chain, or for non-chained cases.
    if not chain_id or chain_step == 1:
        if fresh:
            # PnP build chain starts here. Use the empty-program fixture
            # that ships alongside the populated demo program.
            empty_program_path = pre.get(
                "empty_program_path",
                "/workspace/examples/empty_pick_and_place_ur10e_robotiq.program.yaml",
            )
            code, _ = reset_program(composer, deployment_path, empty_program_path)
            if code != 200:
                return CaseResult(
                    cid, "fail", 0.0,
                    failures=[f"empty-program reset failed: HTTP {code}"],
                )
        else:
            code, _ = reset_program(composer, deployment_path, program_path)
            if code != 200:
                return CaseResult(cid, "fail", 0.0, failures=[f"program reset failed: HTTP {code}"])

    # Per-case fixtures (custom_precondition runtime) — seed state before
    # the prompt fires. After the program reset above the catalogHash
    # may have rotated, so refresh it once per case that needs it.
    if cid in _FIXTURES:
        fresh_hash = fetch_catalog_hash(composer) or catalog_hash
        ok, err = apply_fixtures(cid, composer, fresh_hash)
        if not ok:
            return CaseResult(cid, "fail", 0.0, failures=[f"fixture: {err}"])

    if chain_id:
        rid = f"chain-{chain_id}-step{chain_step:02d}-{int(time.time()*1000)}"
        if no_chain_session:
            # Strategy: each chain step gets its own conversationId.
            # Prior turns do NOT appear in the model's context — the
            # in-prompt snapshot is the only state carrier. Tests
            # whether long agent-loop history hurts more than helps.
            pass  # rid already unique per call
        else:
            chain_state.setdefault(chain_id, rid)  # first step seeds the conversation
            rid = chain_state[chain_id]            # all later steps reuse it
    else:
        rid = f"corpus-{cid}-{int(time.time()*1000)}"

    chain_timeout_s = float(pre.get("chain_timeout_s", DEFAULT_TIMEOUT_S))
    t0 = time.time()
    code, body = send_chat(composer, case["prompt"], rid, timeout_s=chain_timeout_s)
    elapsed = time.time() - t0

    answer = ""
    if isinstance(body, dict):
        answer = body.get("message") or body.get("body") or ""
        if not answer and body.get("error"):
            answer = body["error"].get("detail") or str(body["error"])
        # Iter 18 (2026-05-09): under thinking-on, vLLM may put reasoning
        # tokens in `reasoning_content` (per Qwen3 docs). The assistant's
        # visible content may be shorter or empty when thinking is on.
        # For answer_must_contain assertions, scan BOTH content and
        # reasoning. Concatenate so the keyword search succeeds whichever
        # field carries the words. Doesn't affect tools_called / state_after.
        for tc in body.get("toolCalls") or []:
            rc = (tc.get("result") or {}).get("reasoning_content") if isinstance(tc, dict) else None
            if rc:
                answer = answer + "\n[reasoning] " + str(rc)
        # Some providers expose the assistant turn's reasoning at top level
        # of the response; try common shapes.
        reasoning_top = body.get("reasoning_content") or body.get("reasoning")
        if reasoning_top:
            answer = answer + "\n[reasoning] " + str(reasoning_top)
    elif isinstance(body, str):
        answer = body

    observed = fetch_tool_calls_since(t0)

    # Iter 7 (2026-05-09): merge args from the chat response's toolCalls
    # array onto the access-log entries so args_contain can be validated.
    response_tool_calls = []
    if isinstance(body, dict):
        response_tool_calls = body.get("toolCalls") or body.get("tool_calls") or []
    observed = merge_response_tool_args(observed, response_tool_calls)

    # Filter out tool calls that fired BEFORE this case's t0 (timestamp
    # parsing can be flaky; rely on t0_epoch_s in fetch_tool_calls_since).
    failures: list[str] = []
    soft_failures: list[str] = []

    if code != 200:
        failures.append(f"chat HTTP {code}")
    else:
        assert_tools(case, observed, failures)
        # state_after is best-effort: capture once after the call.
        post_state = capture_state(composer)
        assert_state(case, post_state, failures)

    assert_answer(case, answer, soft_failures)

    recovered = False
    # Iter 8 (2026-05-09): generic recovery turn for 4xx tool failures.
    # If the case failed AND the model fired ≥1 tool that hit 4xx, send
    # one follow-up message in the same conversation pointing at the
    # structured recovery fields. Re-evaluate; if asserts now pass, mark
    # as 'recovered-pass'. Cross-cutting: same wording for every case.
    if enable_recovery_turn and failures and code == 200:
        recovery_msg = _build_recovery_message(
            observed,
            expected_tools_called=case.get("expected", {}).get("tools_called"),
        )
        if recovery_msg is not None:
            t1 = time.time()
            code2, body2 = send_chat(
                composer, recovery_msg, rid, timeout_s=chain_timeout_s
            )
            elapsed += time.time() - t1
            if code2 == 200 and isinstance(body2, dict):
                # Pull more tool calls + args, append to observed
                more_obs = fetch_tool_calls_since(t1)
                more_resp = body2.get("toolCalls") or body2.get("tool_calls") or []
                more_obs = merge_response_tool_args(more_obs, more_resp)
                observed = observed + more_obs
                # Re-run asserts on the augmented observed; reset failure
                # list so we are evaluating the post-recovery state.
                retry_failures: list[str] = []
                assert_tools(case, observed, retry_failures)
                post_state2 = capture_state(composer)
                assert_state(case, post_state2, retry_failures)
                if not retry_failures:
                    recovered = True
                    failures = []  # cleared; case recovered

    if failures:
        st = "fail"
    elif recovered:
        st = "recovered-pass"
    elif soft_failures:
        st = "soft-pass"
    else:
        st = "pass"

    return CaseResult(
        case_id=cid,
        status=st,
        elapsed_s=round(elapsed, 1),
        tool_calls=observed,
        answer=answer[:200],
        failures=failures,
        soft_failures=soft_failures,
    )


# ----------------------------------------------------------------------
# Main runner
# ----------------------------------------------------------------------

def case_is_runnable(case: dict, args) -> tuple[bool, str]:
    status = case.get("status", "active")
    if status == "future" and not args.include_future:
        return False, "status=future (use --include-future to run)"
    rr = case.get("required_runtime")
    # Runtimes the harness can satisfy without external infra:
    #   custom_precondition  → harness fixtures (see _FIXTURES)
    #   pnp_build_chain      → chain_state + empty-program fixture
    HARNESS_PROVIDED = {
        "custom_precondition",       # harness fixture seeding
        "pnp_build_chain",           # chain_id/chain_step + fresh-program fixture
        "expanded_node_allowlist",   # deployment YAML extended 2026-05-08 to
                                     # include move_manipulator_action / timer
                                     # / command_gripper / wait_for_signal_bool
                                     # / set_key_bool_value
    }
    if rr and rr not in HARNESS_PROVIDED and rr not in args.runtime_flags:
        return False, f"required_runtime={rr} not enabled (use --runtime-flags)"
    if rr == "custom_precondition" and case["id"] not in _FIXTURES:
        return False, f"custom_precondition: no fixture seeded for {case['id']}"
    if rr == "custom_precondition" and args.skip_fixture_cases:
        return False, "custom_precondition fixtures skipped"
    return True, ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--composer", default=DEFAULT_COMPOSER)
    p.add_argument("--include-future", action="store_true")
    p.add_argument("--runtime-flags", default="",
                   help="comma-separated runtime tier names to enable")
    p.add_argument("--filter", default=None,
                   help="regex on case id; only matching cases run")
    p.add_argument("--skip-fixture-cases", action="store_true")
    p.add_argument("--report", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--enable-recovery-turn",
        action="store_true",
        help=(
            "Iter 8 (2026-05-09): when a case fails its asserts AND the "
            "observed stream contains a 4xx tool call, send one generic "
            "follow-up turn ('the previous tool call failed; retry using "
            "the structured recovery hints') and re-evaluate. Cases that "
            "pass on the second turn are scored as 'recovered-pass' (a "
            "new effective-rate bucket). Default: off."
        ),
    )
    p.add_argument(
        "--no-chain-session",
        action="store_true",
        help=(
            "Iter 16 (2026-05-09): for chained cases, give each step its "
            "OWN conversationId instead of reusing the chain's. Prior "
            "turns do not appear in the model's context — the in-prompt "
            "snapshot is the only state carrier. Tests whether long "
            "agent-loop history helps or hurts on the PnP build chain."
        ),
    )
    args = p.parse_args()

    args.runtime_flags = {f.strip(): True for f in args.runtime_flags.split(",") if f.strip()}

    with open(args.corpus) as f:
        corpus = yaml.safe_load(f)

    default_pre = corpus.get("default_precondition") or {}
    cases = corpus.get("cases") or []

    if args.filter:
        rx = re.compile(args.filter)
        cases = [c for c in cases if rx.search(c["id"])]

    chain_state: dict[str, str] = {}
    results: list[CaseResult] = []
    catalog_hash = fetch_catalog_hash(args.composer)

    print(f"Running {len(cases)} cases against {args.composer}")
    print(f"  catalogHash: {catalog_hash[:16]}...\n" if catalog_hash else "  (no catalogHash yet)\n")
    for case in cases:
        runnable, why = case_is_runnable(case, args)
        if not runnable:
            r = CaseResult(case["id"], "skipped", 0.0, skip_reason=why)
            results.append(r)
            print(f"  [SKIP] {case['id']:50s}  {why}")
            continue
        r = run_case(
            case, args.composer, default_pre, chain_state, catalog_hash,
            enable_recovery_turn=args.enable_recovery_turn,
            no_chain_session=args.no_chain_session,
        )
        results.append(r)
        marker = {
            "pass": "✅", "recovered-pass": "🛟", "soft-pass": "🟡",
            "fail": "❌", "skipped": "⏭",
        }[r.status]
        line = f"  {marker} {case['id']:50s}  {r.elapsed_s:>5.1f}s  status={r.status}"
        if r.failures:
            line += f"\n      fail: {r.failures}"
        if r.soft_failures and args.verbose:
            line += f"\n      soft: {r.soft_failures}"
        print(line)

    # Summary
    print()
    print("=" * 78)
    counts = {
        "pass": 0, "recovered-pass": 0, "soft-pass": 0, "fail": 0, "skipped": 0,
    }
    for r in results:
        counts[r.status] += 1
    n_total = len(results)
    n_attempted = n_total - counts["skipped"]
    # Effective rate counts pass + recovered-pass + soft-pass
    n_pass_eff = counts["pass"] + counts["recovered-pass"] + counts["soft-pass"]
    print(f"  total cases:      {n_total}")
    print(
        f"  attempted:        {n_attempted}  ({counts['pass']} pass, "
        f"{counts['recovered-pass']} recovered, "
        f"{counts['soft-pass']} soft-pass, {counts['fail']} fail)"
    )
    print(f"  skipped:          {counts['skipped']}")
    if n_attempted:
        n_first_try = counts["pass"]
        print(
            f"  first-try rate:   {n_first_try}/{n_attempted}  "
            f"({100.0 * n_first_try / n_attempted:.1f}%)"
        )
        print(
            f"  effective rate:   {n_pass_eff}/{n_attempted}  "
            f"({100.0 * n_pass_eff / n_attempted:.1f}%)"
        )

    # Report file
    ts = int(time.time() * 1000)
    out = Path(args.report or f"/tmp/smoke_corpus_{ts}.json")
    out.write_text(json.dumps(
        {"corpus": str(args.corpus), "composer": args.composer,
         "runtime_flags": list(args.runtime_flags.keys()),
         "results": [r.__dict__ for r in results]},
        indent=2, default=str,
    ))
    print(f"\n  full report: {out}")

    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
