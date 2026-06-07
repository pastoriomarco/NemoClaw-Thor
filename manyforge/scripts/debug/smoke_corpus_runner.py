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

# Per-case status lines must stream to disk realtime when stdout is
# redirected to a file (the typical `nohup ... > /tmp/iterN_runner.log`
# invocation). Default Python stdio block-buffers at 4KB on a non-tty,
# which deferred all 66 cases' verdicts to end-of-run on iter 33 and
# blocked early-halt decisions. Reconfigure here so callers don't need
# to remember `python3 -u`.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:  # Python <3.7 (defensive)
    pass

# ----------------------------------------------------------------------
# Constants / endpoints
# ----------------------------------------------------------------------

DEFAULT_COMPOSER = "http://127.0.0.1:9000"

# Per-request timeout the runner sends as `timeoutSeconds` in the chat body.
# The bridge honors this REQUEST value over its env-side default
# (`OPENCLAW_ASSISTANT_TIMEOUT_S`), so the runner-side constant is the real
# ceiling for any case that doesn't supply `precondition.chain_timeout_s`.
#
# History:
#   2026-05-09: lowered to 270 because 360 caused 3 cases to hit the
#               composer's 5-minute (300s) upstream-502 circuit-breaker
#               (PnP_18, FALLBACK_alternate_medium, REPLACE_subtree_specific).
#               Default sat at 270 to leave a 30s margin under the 300s wall.
#   2026-06-03: raised to 295 after the Phase 0 iter-32 reproduction showed
#               three cases (WRAP_root_generic, TREE_insert_runtime_medium,
#               MOVE_generic) hitting exactly 275.1s — the 270 + 5s HTTP
#               wrapper timeout. The bridge-side env was already set to 300s
#               by the launcher (start-openclaw-assistant-bridge.sh
#               OPENCLAW_ASSISTANT_TIMEOUT_S default) but the request-body
#               value the runner sent was overriding it. 295 keeps a 5s
#               safety margin under the composer's 300s wall while giving
#               the 270+ cases a fair shot at completing.
#
# Env override available via MANYFORGE_SMOKE_DEFAULT_TIMEOUT_S so future
# operators can re-tune for a different composer cap without a code change.
DEFAULT_TIMEOUT_S = float(
    os.environ.get("MANYFORGE_SMOKE_DEFAULT_TIMEOUT_S", "295") or "295"
)
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

    # Draft scene resources (created by scene_draft_* tools) live in
    # program.sceneResources and are NOT mirrored into the runtime
    # /api/scene/state until the draft is materialized/cycled. Standalone
    # scene-edit cases (e.g. P2_scene_add_specific) assert `scene.objects[...]`
    # immediately after a draft op, so the runtime layer is empty and the
    # assertion fails regardless of what the model did — a false negative.
    # Fall back to the draft resources when the runtime scene carries no
    # objects, so state_after measures the draft the tool actually mutated.
    # (sceneResources uses shape.box_dimensions_m / pose.position_m, which the
    # assertion alias map already maps from shape.box_dims / pose.position.)
    # Draft scene resources (created by scene_draft_* tools) live in
    # program.sceneResources and are NOT mirrored into the runtime
    # /api/scene/state until the draft is materialized/cycled — the runtime
    # layer holds only base geometry. state_after asserts `scene.objects[...]`
    # right after a draft op, so without this the model's draft-added/updated
    # object is invisible and the assertion is a FALSE negative (it never
    # matches regardless of what the model did). Merge the draft resources into
    # scene.objects (draft overrides runtime by id; new draft objects appended),
    # normalizing to the legacy assertion keys (shape.box_dims / pose.position /
    # id) because assert_state's path resolver is NOT alias-aware (only the
    # args_contain matcher is).
    draft_objs = state["program"].get("sceneResources")
    if isinstance(draft_objs, list) and draft_objs:
        def _norm(o: dict) -> dict:
            o = dict(o)
            sh = dict(o.get("shape") or {})
            if "box_dimensions_m" in sh:
                sh.setdefault("box_dims", sh["box_dimensions_m"])
            o["shape"] = sh
            po = dict(o.get("pose") or {})
            if "position_m" in po:
                po.setdefault("position", po["position_m"])
            o["pose"] = po
            if "objectId" in o:
                o.setdefault("id", o["objectId"])
            return o

        merged = list(state["scene"].get("objects") or [])
        idx = {(o.get("id") or o.get("objectId")): i for i, o in enumerate(merged)}
        for o in draft_objs:
            o = _norm(o)
            oid = o.get("id") or o.get("objectId")
            if oid in idx:
                merged[idx[oid]] = o
            else:
                merged.append(o)
        state["scene"]["objects"] = merged

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
            # 2026-06-03: actually PROJECT remaining segments per element.
            # The legacy behavior was `return cur` immediately, which left
            # the whole list of dicts in the result; downstream `contains X`
            # matches then ran as substring checks against repr(list),
            # producing false-positives when X appeared anywhere inside any
            # element (e.g. `[1.0, 0.02, 0.25]` could match against
            # `pose.position_m=[1.0, 0.02, 0.25]` even though the corpus
            # intended `shape.box_dimensions_m`).
            #
            # Walk the rest of the path against each element, collect the
            # per-element resolutions, and return them as a list of values.
            # Downstream `_value_matches`'s `contains X` rule already
            # handles list inputs (iterates and substring-matches), so the
            # net effect is a properly-projected per-element check.
            remaining_idx = _split_path(path).index(seg) + 1
            remaining_segs = _split_path(path)[remaining_idx:]
            if not remaining_segs:
                return cur
            sub_path = ".".join(remaining_segs)
            projected: list[Any] = []
            for item in cur:
                projected.append(_resolve_path(item, sub_path))
            return projected
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


# 2026-06-03: schema-drift-tolerant args matching.
#
# The smoke corpus was authored against an older manifest where:
#   shape.box_dims      (now: shape.box_dimensions_m)
#   pose.position       (now: pose.position_m)
#   shape.type, shape.size remain accepted as alternative emissions
#
# In addition the bridge accepts the call payload under either the flat
# top-level shape (the legacy form some models emit — objectId,
# shapeType, size, position at the root) or the canonical nested form
# (sceneResource.shape.*, sceneResource.pose.*) — both are equivalent
# in effect and both reach the same Composer state.
#
# Without alias-awareness, `args_contain[shape.box_dims]` produced
# false-negative failures even when the model called the tool correctly
# with `sceneResource.shape.box_dimensions_m`. This map declares the
# accepted alternative paths for each legacy key. Matching is "any of":
# if the expected value is found at the legacy path OR at any alias,
# args_contain passes for that key.
#
# Adding a new alias is safe — the matcher tries the legacy key first
# so existing corpora keep working. The reverse (rewriting the corpus
# to canonical names) is not done here because some smokes still target
# older deployments via fixtures.
_ARG_PATH_ALIASES: dict[str, list[str]] = {
    # Scene resource paths — legacy short → canonical / SI-unit-suffixed
    "shape.box_dims":   ["shape.box_dimensions_m",
                         "sceneResource.shape.box_dimensions_m",
                         "sceneResource.shape.box_dims",
                         "size",
                         "sceneResource.shape.size"],
    "shape.box_dimensions_m": ["sceneResource.shape.box_dimensions_m",
                               "shape.box_dims",
                               "size"],
    "shape.type":       ["sceneResource.shape.type",
                         "shapeType"],
    "shape.size":       ["shape.box_dimensions_m",
                         "sceneResource.shape.box_dimensions_m",
                         "sceneResource.shape.size",
                         "size"],
    # Cylinder/sphere primitives — spec 485 line 420 lists `diameter` as
    # a model-friendly flat alias for the canonical SI-suffixed nested
    # form `sceneResource.shape.diameter_m`. Same shape for `height`.
    "shape.diameter":   ["shape.diameter_m",
                         "sceneResource.shape.diameter_m",
                         "sceneResource.shape.diameter",
                         "diameter"],
    "shape.height":     ["shape.height_m",
                         "sceneResource.shape.height_m",
                         "sceneResource.shape.height",
                         "height"],
    "shape.radius":     ["shape.radius_m",
                         "sceneResource.shape.radius_m",
                         "sceneResource.shape.radius",
                         "radius"],
    "pose.position":    ["pose.position_m",
                         "sceneResource.pose.position_m",
                         "sceneResource.pose.position",
                         "position"],
    "pose.position_m":  ["sceneResource.pose.position_m",
                         "pose.position",
                         "position"],
    "pose.orientation": ["pose.orientation_rpy_deg",
                         "pose.orientation_xyzw",
                         "sceneResource.pose.orientation_rpy_deg",
                         "sceneResource.pose.orientation_xyzw",
                         "orientation"],
    "pose.frame_id":    ["sceneResource.pose.frame_id",
                         "frameId",
                         "frame_id"],
    # Tree-insert positional sugar — models commonly emit either
    # `parentName + position=-1` (legacy) or `afterName=<last-child>`
    # (semantically equivalent end-insertion). Without semantic state
    # we can't fully prove equivalence here, but we can at least let
    # the matcher accept `afterName` as a name surrogate for parentName
    # when the corpus is checking the former.
    "parentName":       ["afterName", "beforeName"],
    # Tree-insert positional index — corpus was authored against an
    # older schema where the field was `position`. Current
    # tree_draft_insert_node exposes `index` (integer). Verified
    # 2026-06-03 against the live manifest `inputSchema.properties`.
    # Without this alias a model that correctly emits `index: 0`
    # would fail args_contain[position: 0]. The `position=-1` special
    # case in `_match_args_contain` handles the end-insertion variant.
    "position":         ["index"],
    # tree_draft_replace_subtree was renamed: the wrapper field used
    # to be `subtree_root` (with a nested `kind` discriminator), now
    # it is `subtree` (with `id` as the discriminator). Live manifest
    # for tree_draft_replace_subtree exposes only `subtree`. Active
    # corpus cases at smoke_corpus.yaml lines 353, 577, 1027 still
    # use `subtree_root.kind`.
    "subtree_root.kind": ["subtree.id"],
    "subtree_root":     ["subtree"],
}


def _match_args_contain(
    expected_flat: dict[str, Any],
    actual_args: dict[str, Any],
    pre_state: dict[str, Any] | None = None,
) -> list[str]:
    """Validate `args_contain` against the model's emitted arguments,
    tolerating the field-name drift between corpus and live manifest.

    Returns the list of mismatch messages (empty if everything matched).
    For each expected `(legacy_path, expected_value)`:
      1. Try the legacy path verbatim against the flattened actual args.
      2. If missing or value mismatch, walk `_ARG_PATH_ALIASES[legacy_path]`
         and accept the first alias that yields a matching value.
      3. Also try the LEAF name only (e.g. `box_dimensions_m`) anywhere in
         the flattened tree — handles unanticipated nesting like
         `node.params.box_dimensions_m`.
      4. If no alias matches, emit a mismatch message that lists what was
         actually present at the legacy path so the reader can tell
         whether it's a real model error or a schema-drift false negative.
    """
    flat = _flatten(actual_args)
    failures: list[str] = []
    for legacy_path, expected_value in expected_flat.items():
        # Direct hit
        av = flat.get(legacy_path, "<MISSING>")
        if _value_matches(av, expected_value):
            continue
        # Alias paths
        aliases = _ARG_PATH_ALIASES.get(legacy_path, [])
        matched_alias: str | None = None
        for alias in aliases:
            cand = flat.get(alias, "<MISSING>")
            if _value_matches(cand, expected_value):
                matched_alias = alias
                break
        if matched_alias is not None:
            continue
        # Leaf-name fallback — accept the expected value if found anywhere
        # under a key whose final dotted segment equals the legacy leaf.
        leaf = legacy_path.rsplit(".", 1)[-1]
        leaf_hit = False
        for k, v in flat.items():
            if k.rsplit(".", 1)[-1] != leaf:
                continue
            if _value_matches(v, expected_value):
                leaf_hit = True
                break
        if leaf_hit:
            continue
        # Tree-insert end-position equivalence: when the corpus expects
        # `position: -1` (end insertion by index) the model may
        # legitimately express end-insertion in three other ways:
        #   a) `afterName: <last_child>`     — end by named anchor
        #   b) `parentName: <parent>` only   — composer default IS end
        #   c) `position: -1` literally      — direct match (handled above)
        # Without post-state we accept (a) and (b) as semantic matches.
        # 2026-06-03 tightening (per reviewer): reject the lenient
        # parentName-only fallback when the model also emitted an
        # explicit `index` value other than -1 — that's a conflicting
        # ordering choice, not an end-insertion. Combined with the
        # other args_contain entries (node.id, node.params.*),
        # false-positive risk stays low.
        if legacy_path == "position" and expected_value == -1:
            explicit_idx = flat.get("index")
            has_conflicting_index = (
                isinstance(explicit_idx, int) and explicit_idx != -1
            )
            if has_conflicting_index:
                # Model emitted `index: 5` (or similar). That is NOT an
                # end insertion regardless of parentName/afterName.
                pass
            elif flat.get("afterName") or flat.get("parentName"):
                continue
        # @root literal resolution for tree edits — spec 485 §tree-edit
        # documents `targetName: "@root"` (and `parentName: "@root"`,
        # `newParentName: "@root"`) as a literal alias for the current
        # root, accepted on wrap_node / replace_subtree / update_node_params
        # / insert_node's parentName / move_node's newParentName. When the
        # corpus expects e.g. `targetName: pick_and_place` and the model
        # emitted `targetName: "@root"`, accept iff pre_state proves
        # `pick_and_place` IS the current root.
        if legacy_path in ("targetName", "parentName", "newParentName"):
            actual = flat.get(legacy_path)
            if actual == "@root" and isinstance(pre_state, dict):
                root_id = (
                    (pre_state.get("program") or {}).get("tree", {}).get("id")
                )
                if root_id and root_id == expected_value:
                    continue
        failures.append(
            f"args_contain[{legacy_path}] expected {expected_value!r}, "
            f"got {av!r} (also tried aliases: {aliases or '[]'})"
        )
    return failures


@dataclass
class CaseResult:
    case_id: str
    status: str           # "pass" | "soft-pass" | "recovered-pass" | "fail" | "skipped" | "contaminated"
    elapsed_s: float
    tool_calls: list[dict] = field(default_factory=list)
    answer: str = ""
    failures: list[str] = field(default_factory=list)
    soft_failures: list[str] = field(default_factory=list)
    skip_reason: str = ""
    healed: bool = False        # --self-heal fired after this chained step's failure
    heal_detail: str = ""       # splice/replay detail, or the heal-failure reason


# (Removed 2026-06-07) The opt-in chain-state contamination detector
# (`MANYFORGE_SMOKE_DETECT_CHAIN_CONTAMINATION`) only handled the narrow
# loop-break-sentinel poisoning case. It is superseded by the self-healing chain
# mechanism (restore canonical state + splice a golden turn on ANY chained-step
# failure) — see self_heal.py + docs/self-healing-chain-harness.md. The
# `"contaminated"` status is retained in the marker/summary tables as an inert
# vestige (nothing emits it now).


def assert_tools(case: dict, observed: list[dict], failures: list[str],
                 soft_failures: list[str] | None = None,
                 pre_state: dict[str, Any] | None = None) -> None:
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

    def _pick_best(matchers: list[str], allow_retries: bool) -> int | None:
        """Pick the best call matching any name in `matchers`. Tiered
        preference:
          T1 — bridge_status=completed AND structured args (non-_raw)
          T2 — bridge_status=completed (even if args is just {"_raw": ...})
          T3 — any 2xx (legacy behavior — last resort)

        Rationale: smoke audit captures BOTH the bridge HTTP status (always
        200 if the call reached the bridge) AND a bridge_status of
        "completed" | "failed". On direct lane the model often emits a
        first-turn call with malformed args (bridge_status=failed,
        arguments={"_raw": "<tool_call>...</tool_call>"}) and then a
        recovery-turn call with structured args (bridge_status=completed,
        arguments={"sceneResource": {...}}). The legacy first-2xx picker
        consumed the failed call and ran args_contain against `_raw` only —
        producing false-negative MISSING failures even when the model's
        SECOND attempt was the canonical schema. This picker chooses the
        actually-executed structured call instead.
        """
        # Tier 1
        for idx, call in enumerate(available):
            if call.get("_consumed") or call["tool"] not in matchers:
                continue
            status = int(call["status"] or 0)
            if 200 <= status < 300 and call.get("bridge_status") == "completed":
                args = call.get("arguments") or {}
                if args and not (len(args) == 1 and "_raw" in args):
                    return idx
        # Tier 2
        for idx, call in enumerate(available):
            if call.get("_consumed") or call["tool"] not in matchers:
                continue
            status = int(call["status"] or 0)
            if 200 <= status < 300 and call.get("bridge_status") == "completed":
                return idx
        # Tier 3 (legacy)
        for idx, call in enumerate(available):
            if call.get("_consumed") or call["tool"] not in matchers:
                continue
            status = int(call["status"] or 0)
            if 200 <= status < 300:
                return idx
            if allow_retries:
                available[idx]["_consumed"] = True
        return None

    for i, exp in enumerate(expected_list):
        name = exp.get("name")
        # 2026-06-01: support `alt_names` (list) for equivalent-effect tools.
        # E.g. scene_draft_add_object accepts scene_draft_upsert_objects as
        # an alt — both reach the same effect (scene resource created/updated).
        # If the primary `name` is missing but an alt fired with 2xx, the
        # entry is consumed and the FAIL is recorded as a SOFT_FAIL (the
        # outer runner promotes this to a soft-pass for the case).
        alt_names = exp.get("alt_names") or []
        allow_retries = exp.get("allow_retries", False)
        matched_alt: str | None = None
        consumed_idx = _pick_best([name], allow_retries)
        if consumed_idx is None and alt_names:
            consumed_idx = _pick_best(alt_names, allow_retries)
            if consumed_idx is not None:
                matched_alt = available[consumed_idx]["tool"]
        if consumed_idx is None:
            failures.append(
                f"expected tool '{name}' not observed (or never reached 2xx)"
            )
            continue
        available[consumed_idx]["_consumed"] = True
        if matched_alt is not None:
            # Soft-pass: alt tool with equivalent effect. Route to
            # soft_failures so the case is recorded as soft-pass rather
            # than fail. Falls back to fail-list if caller didn't pass
            # soft_failures (older callers — shouldn't happen in this
            # runner but defensively safe).
            target = soft_failures if soft_failures is not None else failures
            target.append(
                f"alt-tool '{matched_alt}' used instead of '{name}' (equivalent effect)"
            )
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
                # 2026-06-03: schema-drift-tolerant matching. The corpus
                # uses pre-rename field names (shape.box_dims, pose.position,
                # parentName + position=-1) while live manifests now expose
                # SI-suffixed canonical names (shape.box_dimensions_m,
                # pose.position_m) AND the bridge accepts either the flat
                # top-level form or the nested sceneResource.* form. The
                # legacy strict `flat.get(k)` matcher generated false
                # negatives even when the model called the tool correctly.
                expected_flat = _flatten(args_contain)
                for msg in _match_args_contain(expected_flat, actual_args, pre_state):
                    failures.append(f"{msg} on tool '{name}'")

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


def _answer_match_variants(text: str) -> tuple[str, str, str]:
    """Return raw, whitespace-normalized, and compact answer variants.

    OpenClaw's streamed token collector can persist visible answers with
    newlines between subword tokens (`up\nsert`, `pick\n_and\n_place`).
    Corpus prose checks should not false-negative on that transport artifact.
    """

    raw = (text or "").lower()
    spaced = re.sub(r"\s+", " ", raw).strip()
    compact = re.sub(r"\s+", "", raw)
    return raw, spaced, compact


def _answer_contains(answer_variants: tuple[str, str, str], needle: str) -> bool:
    raw, spaced, compact = answer_variants
    needle_raw = needle.lower()
    needle_spaced = re.sub(r"\s+", " ", needle_raw).strip()
    needle_compact = re.sub(r"\s+", "", needle_raw)
    return (
        needle_raw in raw
        or needle_spaced in spaced
        or needle_compact in compact
    )


def assert_answer(case: dict, answer: str, soft_failures: list[str]) -> None:
    must = case.get("expected", {}).get("answer_must_contain") or []
    must_not = case.get("expected", {}).get("answer_must_not_contain") or []
    variants = _answer_match_variants(answer or "")
    for needle in must:
        if not _answer_contains(variants, needle):
            soft_failures.append(f"answer_must_contain: {needle!r} not found")
    for needle in must_not:
        if _answer_contains(variants, needle):
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
        # Capture state ONCE: serves both as pre_state for `@root`-like
        # alias resolution AND as the post_state for state_after asserts.
        # Note this is "state observed after the chat returns" — the
        # composer has already applied any mutations the model invoked.
        # For `@root` resolution we trust that the program tree root id
        # was stable across the brief chat window (program load happens
        # before the chat fires; mutations don't rename the root).
        observed_state = capture_state(composer)
        assert_tools(case, observed, failures, soft_failures,
                     pre_state=observed_state)
        assert_state(case, observed_state, failures)

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
                post_state2 = capture_state(composer)
                assert_tools(case, observed, retry_failures,
                             pre_state=post_state2)
                assert_state(case, post_state2, retry_failures)
                if not retry_failures:
                    recovered = True
                    failures = []  # cleared; case recovered

    # (The opt-in chain-contamination detector was removed 2026-06-07 — superseded
    # by the self-healing chain mechanism, which restores canonical state + splices a
    # golden turn on ANY chained-step failure, not just the narrow loop-break case.
    # See self_heal.py + docs/self-healing-chain-harness.md.)
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
    p.add_argument(
        "--self-heal",
        action="store_true",
        help=(
            "Self-healing chains (TEST-ONLY): on a chained-step FAIL, restore the "
            "canonical state (replay-from-base via golden changes) and splice a "
            "golden turn into the openclaw session transcript, so the NEXT step "
            "runs from a correct state with the model unaware of the failure. Does "
            "not change the failed step's own score. See self_heal.py + "
            "docs/self-healing-chain-harness.md."
        ),
    )
    p.add_argument("--self-heal-golden",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "golden_trajectories.yaml"))
    p.add_argument("--self-heal-container", default="openshell-my-assistant",
                   help="name substring of the openshell sandbox container holding session files")
    p.add_argument("--self-heal-agent", default="manyforge-composer")
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

    # Self-healing chains (TEST-ONLY, opt-in via --self-heal). Loaded lazily so the
    # default run path is untouched and never imports the heal machinery.
    self_heal_specs: dict = {}
    self_heal_container = None
    _sh = None
    if args.self_heal:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import self_heal as _sh  # noqa: E402
        with open(args.self_heal_golden) as gf:
            self_heal_specs = yaml.safe_load(gf) or {}
        self_heal_container = _sh.resolve_container(args.self_heal_container)
        print(f"  self-heal: ENABLED  chains={list(self_heal_specs)}  container={self_heal_container}")

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
            # 2026-06-03: contamination marker. Mid-blue circle is
            # distinct from the ❌ failure marker so a glance at the
            # log makes the harness-issue vs. model-miss split
            # immediately obvious. Only emitted when the opt-in
            # MANYFORGE_SMOKE_DETECT_CHAIN_CONTAMINATION=1 is set.
            "contaminated": "🔵",
        }.get(r.status, "?")
        line = f"  {marker} {case['id']:50s}  {r.elapsed_s:>5.1f}s  status={r.status}"
        if r.failures:
            line += f"\n      fail: {r.failures}"
        if r.soft_failures and args.verbose:
            line += f"\n      soft: {r.soft_failures}"
        print(line)

        # Self-healing chains (TEST-ONLY): on a chained-step hard FAIL, restore the
        # canonical state + splice a golden turn so the NEXT step runs from a correct
        # state with the model unaware. The failed step keeps its own (fail) score —
        # we only prevent the cascade to step N+1.
        if _sh is not None and not args.no_chain_session and r.status == "fail":
            pre = case.get("precondition") or {}
            ch_id, ch_step = pre.get("chain_id"), pre.get("chain_step")
            spec = self_heal_specs.get(ch_id) if ch_id else None
            conv = chain_state.get(ch_id) if ch_id else None
            if spec and conv and ch_step and self_heal_container:
                ok, detail = _sh.self_heal(args.composer, self_heal_container,
                                           args.self_heal_agent, conv, spec, case["id"])
                r.healed = ok          # r is already in `results` by reference → lands in the report
                r.heal_detail = detail
                print(f"      🩹 self-heal {'OK' if ok else 'FAILED'} (step {ch_step}): {detail}")

    # Summary
    print()
    print("=" * 78)
    counts = {
        "pass": 0, "recovered-pass": 0, "soft-pass": 0, "fail": 0,
        "skipped": 0, "contaminated": 0,
    }
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    n_total = len(results)
    # Contaminated cases are runner-side issues (the model was never
    # invoked); they don't count toward "attempted" the same way fails
    # do. The denominator therefore excludes them, which avoids
    # depressing the effective rate by harness state-leakage. When the
    # detector is OFF (the default), `counts['contaminated']` is 0 and
    # this clause is a no-op.
    n_attempted = n_total - counts["skipped"] - counts["contaminated"]
    # Effective rate counts pass + recovered-pass + soft-pass
    n_pass_eff = counts["pass"] + counts["recovered-pass"] + counts["soft-pass"]
    print(f"  total cases:      {n_total}")
    print(
        f"  attempted:        {n_attempted}  ({counts['pass']} pass, "
        f"{counts['recovered-pass']} recovered, "
        f"{counts['soft-pass']} soft-pass, {counts['fail']} fail)"
    )
    print(f"  skipped:          {counts['skipped']}")
    if counts["contaminated"]:
        print(
            f"  contaminated:     {counts['contaminated']}  "
            f"(chain echoed bridge loop-break sentinel; model not invoked; "
            f"excluded from attempted/effective)"
        )
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

    healed = [r.case_id for r in results if r.healed]
    heal_failed = [r.case_id for r in results if r.heal_detail and not r.healed]
    if healed:
        print(f"  self-healed:      {len(healed)}  ({', '.join(healed)})")
    if heal_failed:
        print(f"  self-heal FAILED: {len(heal_failed)}  ({', '.join(heal_failed)})")

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
