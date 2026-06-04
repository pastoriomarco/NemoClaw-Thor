"""Contract translation for the OpenClaw-routed ManyForge assistant adapter.

This module is deliberately free of FastAPI and subprocess imports so the
translation layer can be tested without a live NemoClaw/OpenShell sandbox.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import time
from dataclasses import dataclass
from json import JSONDecoder
from typing import Any


SUPPORTED_VERSION_FAMILY = "manyforge.assistant.provider_request.v0"
DEFAULT_SCHEMA_VERSION = "0.1.0"


# 2026-06-01 — recency-position tail checklist
# Compact restatement of the most-violated rules, appended AFTER the
# user_request so the model sees them in maximum-attention position.
# The full rules_block higher up provides the per-rule rationale; this
# tail block is the short-form reminder right before action. Env-gated:
# set OPENCLAW_BRIDGE_TAIL_CHECKLIST=0 to disable.
_TAIL_CHECKLIST_ENABLED_RAW = (
    os.environ.get("OPENCLAW_BRIDGE_TAIL_CHECKLIST", "1") or "1"
).strip().lower() in ("1", "true", "yes", "on")


# 2026-06-03: A/B probe lever for the Phase 0 baseline comparison.
#
# The post-iter-32 rules block has gained Rule 8a (camelCase + concrete
# scene_draft schemas) and the tail checklist. Earlier revisions also
# carried corpus-specific Rule 11a/11b clauses; those were removed so the
# bridge prompt gives generic clarification guidance rather than encoding
# benchmark-pattern decisions.
#
# Set OPENCLAW_BRIDGE_ITER32_COMPAT_MODE=1 to STRIP rule 8a and
# force-disable the tail checklist. Rule 0 (the dispatch primer) is
# kept because tools-mode dispatch is structurally different from
# iter-32's surface (manyforge_* in tools[] directly) and the primer is
# load-bearing for the model to engage the tool_search/describe/call
# discovery verbs at all. Stripping rules 1-10 too is overshoot — those
# pre-date the cosmos regression iterations and are not under the
# A/B-vs-iter-32 question.
#
# Use case: a small 8-12 case A/B against full-prompt mode to test
# whether the post-iter-32 rule additions are net-helpful on cosmos
# today (the reviewer's hypothesis is that they are over-prescriptive
# and slightly hurt rather than help). Default OFF so the full rules
# stay live for production smoke runs.
_ITER32_COMPAT_MODE = (
    os.environ.get("OPENCLAW_BRIDGE_ITER32_COMPAT_MODE", "0") or "0"
).strip().lower() in ("1", "true", "yes", "on")

# Effective tail-checklist switch: OFF when either the explicit env-flag is
# false OR the iter-32 compat mode strips it.
_TAIL_CHECKLIST_ENABLED = _TAIL_CHECKLIST_ENABLED_RAW and not _ITER32_COMPAT_MODE

_TAIL_CHECKLIST_BODY = (
    "\n\n"
    "## tail_checklist (apply BEFORE emitting your next action)\n"
    "- Re-read the user prompt: does it name WHERE (parent / position / target sibling)? "
    "If not, output ONE clarification question — do NOT default to root or last sibling.\n"
    "- Action verb + node/object? Emit a tool call. Never reply with `NO_REPLY`.\n"
    "- On a 4xx tool result: read `allowedNodeKinds` / `validParentNames` from the response "
    "and pick from there. Do NOT retry with identical args.\n"
    "- If `result.delta.rootKind` already matches the user's requested kind, the goal is "
    "satisfied — emit the final assistant message and stop."
)


# =========================================================================
# Discovery-protocol primers (one per OpenClaw tool surface).
# =========================================================================
#
# OpenClaw exposes one of two surfaces to the model, determined by
# sandbox-side OpenClaw config. The bridge prompt MUST match the active
# surface or the model can't dispatch — wrong primer = the well-known
# "I can't use the tool because it isn't available" refusal pattern.
#
# The bridge does NOT detect which surface is active. The operator sets
# ``OPENCLAW_ASSISTANT_TOOL_SURFACE`` (env → AdapterConfig.tool_surface),
# the prompt builder picks the matching primer, and the vLLM proxy emits
# a ``tool_surface_mismatch`` warning if the observed ``tools[]`` does
# not match the configured surface (see scripts/proxy/vllm-proxy.py).
#
# Both primers leave Rule 8a (argument-shape correctness) intact —
# camelCase + nested-object rules are surface-agnostic.

_PRIMER_CODE_MODE = (
    "0. **Dispatch surface — code mode.** Your tools[] contains a "
    "single verb `tool_search_code`. Its description says: \"Run "
    "JavaScript in an isolated Node subprocess with openclaw.tools."
    "search, openclaw.tools.describe, and openclaw.tools.call for "
    "large tool catalogs.\" That sentence is LITERAL — the only "
    "bridge name available inside the code body is "
    "`openclaw.tools`, NOT `tools`, NOT `window.openclaw.tools`, "
    "NOT `globalThis.tools`. Use EXACTLY: "
    "(a) `await openclaw.tools.call(\"<tool_name>\", {...args})` — "
    "dispatches the named ManyForge tool with the given args. THIS "
    "IS THE ONLY DISPATCH FORM. Both the named tool id (e.g. "
    "\"tree_draft_wrap_node\") and the args object are required. "
    "(b) `await openclaw.tools.describe(\"<tool_name>\")` — fetches "
    "the parameter schema for a tool. Use when you do not remember "
    "the exact argument keys. Returns `{tool, parameters, ...}`. "
    "(c) `await openclaw.tools.search(\"<query>\", {...opts})` — "
    "discovery by substring tokenizer. Use only when the tool name "
    "is unknown. "
    "Most efficient pattern: `tool_search_code({code: \"const r = "
    "await openclaw.tools.call('tree_draft_wrap_node', {targetName: "
    "'pick_and_place', wrapper: {id: 'repeat', name: 'outer_repeat', "
    "params: {max_iterations: 3}}}); return r;\"})`. Always `await`; "
    "always `return` the result so OpenClaw forwards it. "
    "DO NOT write `tools.call(...)` — `tools` is undefined and your "
    "call will fail with `ReferenceError: tools is not defined`. "
    "DO NOT write `tools.<tool_name>(...)` — same failure. "
    "DO NOT write `window.openclaw.tools.call(...)` — there is no "
    "`window` in Node, you will get `ReferenceError: window is not "
    "defined`. "
    "DO NOT define your own `const tools = {...}` mock at the top "
    "of the body — your mock has no manyforge methods and every "
    "`tools.<x>` call will throw `TypeError: tools.<x> is not a "
    "function`. "
    "Argument shapes inside `call()`'s second arg are camelCase + "
    "nested objects (Rule 8a)."
)

_PRIMER_TOOLS_MODE = (
    "0. **Dispatch surface — tools mode.** Your tools[] contains "
    "three discovery verbs: `tool_search`, `tool_describe`, "
    "`tool_call`. Real ManyForge tools are NOT in tools[] directly; "
    "they are discovered through `tool_search` and invoked via "
    "`tool_call`. The real tool result is in the `.result` field of "
    "`tool_call`'s return; the `.tool` wrapper is informational. "
    "Argument shapes inside `args` are camelCase + nested objects "
    "(see Rule 8a below)."
)

# 2026-06-04: Rule 0b — OpenClaw MCP tool-id contract. The post-Phase-3
# OpenClaw bundle-mcp surface exposes ManyForge tools under three
# fields per `tool_search` result row:
#
#   - id    : "mcp:bundle-mcp:manyforge__<bare_id>"   (fully qualified MCP id)
#   - name  : "manyforge__<bare_id>"                  (DISPATCH name — use this)
#   - label : "<bare_id>"                              (display only)
#
# Earlier primer text said to use the bare canonical form (e.g.
# `tree_draft_insert_node`) for `tool_describe`/`tool_call`. That is
# the `label`, NOT the callable name. OpenClaw rejects it with
# "Unknown tool id". Verified empirically 2026-06-04 from the
# gateway log:
#   tool_describe failed: Unknown tool id: tree_draft_insert_node
#   tool_describe failed: Unknown tool id: tree_draft_wrap_node
# The model then often invents dashed variants
# (`manyforge__tree-draft-insert_node`), which OpenClaw also rejects.
# The contract this rule pins down is: copy the `name` field from a
# prior `tool_search` result VERBATIM, preserving the `manyforge__`
# prefix and the underscore separators.
#
# The proxy's NORMALIZE_TOOL_NAMES mutation operates on the OPENAI
# function name (`tool_calls[*].function.name`), which is always one
# of the three discovery verbs and never broken. It does NOT touch
# the nested ManyForge id inside `tool_call`'s JSON arguments. So
# the prompt has to carry the contract.
_PRIMER_TOOLS_MODE_RULE_0B = (
    "0b. **OpenClaw tool-id contract.** Every `tool_search` result "
    "row carries three fields: `id` (a fully-qualified MCP locator "
    "like `mcp:bundle-mcp:manyforge__<x>`), `name` (the dispatch id "
    "like `manyforge__<x>`), and `label` (a display string like "
    "`<x>`). When you call a ManyForge MCP tool, use the `name` "
    "field exactly — preserve the `manyforge__` prefix, the "
    "underscores, and the casing; do not use `label` for dispatch. "
    "Example: a search row "
    "`{name: \"manyforge__tree_draft_insert_node\", "
    "label: \"tree_draft_insert_node\", ...}` is called as "
    "`tool_call({id: \"manyforge__tree_draft_insert_node\", "
    "args: {...}})`. Bare ids "
    "(`tree_draft_insert_node`) and dashed variants "
    "(`manyforge__tree-draft-insert_node`) return "
    "`Unknown tool id` from OpenClaw."
)

# Maps tool_surface value → primer string. Module-level so callers
# (build_agent_prompt + tests) can introspect.
_TOOL_SURFACE_PRIMERS: dict[str, str] = {
    "code": _PRIMER_CODE_MODE,
    "tools": _PRIMER_TOOLS_MODE,
}


def _primer_for_tool_surface(tool_surface: str) -> str:
    """Return the dispatch-surface primer for ``tool_surface``.

    Falls back to the code-mode primer on an unknown value, with a
    structured log line so operators can find it. The bridge does
    not crash on misconfig — it prompts as if for the default and
    relies on the proxy's drift check to surface the operator error.
    """
    primer = _TOOL_SURFACE_PRIMERS.get(tool_surface)
    if primer is not None:
        return primer
    # Defer the log import to avoid a circular dep at module load.
    try:
        from .service import _log_event  # type: ignore[attr-defined]
        _log_event(
            "bridge_tool_surface_unknown",
            configured=tool_surface,
            fallback="code",
            accepted=sorted(_TOOL_SURFACE_PRIMERS.keys()),
        )
    except Exception:
        pass
    return _PRIMER_CODE_MODE


@dataclass(frozen=True)
class AdapterConfig:
    """Runtime configuration for invoking OpenClaw inside a sandbox."""

    sandbox: str = "my-assistant"
    namespace: str = "openshell"
    container: str = "agent"
    cluster_container: str = "openshell-cluster-nemoclaw"
    sandbox_user: str = "sandbox"
    agent: str = "main"
    timeout_s: float = 120.0
    openclaw_bin: str = "openclaw"
    local: bool = False
    # `thinking`: passed as `--thinking off|on` to OpenClaw CLI. OpenClaw
    # translates this into a top-level `enable_thinking` field on the
    # chat-completion request body sent to the proxy. Per the trace
    # validated 2026-06-01, the chat template does NOT read this top-
    # level field — it reads `chat_template_kwargs.enable_thinking`
    # instead. So this flag is **DEPRECATED for thinking control**: it
    # produces only the inert top-level field. The load-bearing control
    # is `THOR_TARGET_PROXY_FORCE_ENABLE_THINKING` set per-profile in
    # NemoClaw-Thor/serving/config.sh, which the proxy enforces by
    # injecting `chat_template_kwargs.enable_thinking` on every request.
    # Kept here for back-compat with older OpenClaw CLI flag plumbing;
    # do not rely on it to control reasoning.
    thinking: str = "off"
    # Gateway HTTP path: when use_gateway is true, the adapter calls the
    # in-sandbox OpenClaw gateway's /v1/chat/completions endpoint via
    # `kubectl exec curl` instead of shelling out a fresh `openclaw agent`
    # process per request. The persistent gateway eliminates the per-call
    # ~40s CLI bootstrap; warm calls drop from minute-scale to single-digit
    # seconds. The gateway internally runs the tool-call loop, so this
    # adapter only sees the final assistant message in the response.
    # Mode-scoped enforcement is preserved by the manyforge MCP wrapper
    # (registered with the gateway at provisioner time).
    use_gateway: bool = False
    gateway_port: int = 18789
    gateway_max_tokens: int = 4096
    # Sampling parameters for the gateway chat-completion request.
    # **Defaults are None** so the bridge sends a clean envelope.
    # vLLM is the single source of truth for sampling defaults via
    # `--override-generation-config` and `--default-chat-template-kwargs`
    # (see nemoclaw-thor/serving/launch.sh, the matching profile block).
    # If you need a per-call override (typically only for debugging),
    # patch AdapterConfig directly — the helpers and YAML/env-var
    # resolution were removed 2026-05-06 once vLLM-side defaults landed.
    gateway_temperature: float | None = None
    gateway_top_k: int | None = None
    gateway_top_p: float | None = None
    # `gateway_enable_thinking`: when set, the bridge writes
    # `chat_template_kwargs.enable_thinking` into the request body sent
    # to the OpenClaw gateway. **DEPRECATED 2026-06-01**: this hook
    # was never wired up to any default and the proxy now enforces
    # `enable_thinking` per-profile via `THOR_TARGET_PROXY_FORCE_ENABLE_THINKING`
    # in NemoClaw-Thor/serving/config.sh (single source of truth).
    # Setting `OPENCLAW_ASSISTANT_GATEWAY_ENABLE_THINKING=true|false`
    # at the bridge layer still works for callers that bypass the
    # proxy, but the proxy's mirror will override it on the standard
    # path. Kept for back-compat; do not rely on it for the standard
    # OpenClaw lane.
    gateway_enable_thinking: bool | None = None

    # Tool surface OpenClaw exposes to the model. Set by the
    # operator via OPENCLAW_ASSISTANT_TOOL_SURFACE (see
    # service.AdapterConfig.from_env). Two values currently used:
    #
    #   "code"  — OpenClaw 2026.5.6+ default: tools[] contains the
    #             single ``tool_search_code`` verb. Model must wrap
    #             real tool calls in ``tool_search_code({code:
    #             "...await tools.<name>(...)..."})``. Requires the
    #             code-mode primer in the bridge prompt.
    #   "tools" — Discrete control verbs: tools[] contains
    #             ``tool_search``, ``tool_describe``, ``tool_call``.
    #             Model dispatches via tool_call({id, args}). Matches
    #             what iter-32 (51/66) ran against. Requires the
    #             tools-mode primer.
    #
    # The bridge does NOT control which surface OpenClaw exposes —
    # that's a sandbox-side OpenClaw config — but the prompt MUST
    # match the active surface or the model can't dispatch. Drift is
    # detected by the vLLM proxy (see scripts/proxy/vllm-proxy.py),
    # not by the bridge, because the bridge does not sit on the
    # chat-completions path in CLI shell-out mode.
    tool_surface: str = "code"


@dataclass(frozen=True)
class AgentRunResult:
    """Captured output from one OpenClaw agent invocation."""

    stdout: str
    stderr: str
    returncode: int
    duration_ms: float


def request_id_from_payload(payload: dict[str, Any]) -> str:
    value = payload.get("requestId") or payload.get("providerRequestId") or ""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return f"openclaw-adapter-{int(time.time() * 1000)}"


_DETAIL_NODE_LIMIT = 64
_DETAIL_RESOURCE_LIMIT = 64


def _collect_tree_node_names(node: dict[str, Any], out: list[str]) -> None:
    if not isinstance(node, dict):
        return
    nm = node.get("name")
    if isinstance(nm, str) and nm:
        out.append(nm)
    for c in (node.get("children") or []):
        _collect_tree_node_names(c, out)


def _collect_ancestor_paths(node: dict[str, Any],
                              path: list[str],
                              ancestors_by_name: dict[str, list[str]]) -> None:
    if not isinstance(node, dict):
        return
    nm = node.get("name")
    if isinstance(nm, str) and nm:
        ancestors_by_name[nm] = list(path)
    next_path = path + ([nm] if isinstance(nm, str) and nm else path)
    for c in (node.get("children") or []):
        _collect_ancestor_paths(c, next_path, ancestors_by_name)


def _project_tree_node(node: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    nm = node.get("name") if isinstance(node.get("name"), str) else None
    must_detail = nm is not None and nm in ctx["referenced"]
    ctx["n"] += 1
    if ctx["n"] > ctx["limit"] and not must_detail:
        ctx["omitted"] += 1
        out: dict[str, Any] = {"__detail_omitted__": True}
        if nm:
            out["name"] = nm
        return out
    out = {"id": node.get("id"), "name": nm}
    p = node.get("params")
    if isinstance(p, dict) and p:
        out["params_keys"] = sorted(p.keys())
    children = node.get("children") or []
    if isinstance(children, list) and children:
        out["children"] = [
            _project_tree_node(c, ctx)
            for c in children
            if isinstance(c, dict)
        ]
    return out


def _build_program_summary(snap: dict[str, Any]) -> dict[str, Any]:
    """Compact projection of programSnapshot.

    Mirror of manyforge_assistant_bridge.bridge._build_program_summary
    (same algorithm: always-complete nodes_index + bounded first-N
    detail). Drops full raw JSON dump that the openclaw bridge
    previously sent verbatim (~15 KB) — projection is ~2-3 KB. The
    LLM still has every node name in `nodes_index` so it can reference
    any node in the program; the structure of the bounded detail slice
    gives it enough context for common tree-edit args without a
    prompt-keyword promotion path."""
    if not isinstance(snap, dict):
        return {"__error__": "programSnapshot is not a dict"}
    program = snap.get("program") if isinstance(snap.get("program"), dict) else {}
    summary: dict[str, Any] = {
        "programTreeHash": snap.get("programTreeHash"),
        "program": {
            "name": program.get("name"),
            "description": program.get("description"),
        },
    }
    tree = program.get("tree") if isinstance(program.get("tree"), dict) else None
    if tree is not None:
        all_names: list[str] = []
        _collect_tree_node_names(tree, all_names)
        summary["program"]["nodes_index"] = all_names
        ctx = {"n": 0, "limit": _DETAIL_NODE_LIMIT,
               "referenced": set(), "omitted": 0}
        summary["program"]["tree"] = _project_tree_node(tree, ctx)
        if ctx["omitted"]:
            summary["program"]["__nodes_detail_omitted_count__"] = ctx["omitted"]
    params = program.get("parameters")
    if isinstance(params, list):
        summary["program"]["parameters"] = [
            p.get("name") for p in params
            if isinstance(p, dict) and isinstance(p.get("name"), str)
        ]
    bb = program.get("blackboard")
    if isinstance(bb, dict):
        summary["program"]["blackboard_keys"] = sorted(
            k for k in bb.keys() if isinstance(k, str)
        )
    return summary


def _project_scene_object(o: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"object_id": o.get("object_id")}
    shape = o.get("shape")
    if isinstance(shape, dict):
        s_out: dict[str, Any] = {"type": shape.get("type")}
        for k in ("box_dimensions_m", "sphere_radius_m",
                  "cylinder_radius_m", "cylinder_height_m",
                  "mesh_resource_uri", "mesh_scale"):
            v = shape.get(k)
            if v is not None:
                s_out[k] = v
        out["shape"] = s_out
    pose = o.get("pose_in_universe")
    if isinstance(pose, dict):
        out["pose_in_universe"] = pose
    if "attached" in o:
        out["attached"] = o.get("attached")
    if o.get("attached_to") is not None:
        out["attached_to"] = o.get("attached_to")
    return out


def _build_scene_summary(snap: dict[str, Any]) -> dict[str, Any]:
    """Compact projection of sceneSnapshot.

    Mirror of the manyforge_assistant_bridge equivalent. Drops always-
    null sibling shape fields (sphere_radius_m, cylinder_*, mesh_*
    when type=box, etc.) and items past _DETAIL_RESOURCE_LIMIT. Keeps
    the always-complete object_ids index for traversal. The projection
    is prompt-independent by design."""
    if not isinstance(snap, dict):
        return {"__error__": "sceneSnapshot is not a dict"}
    out: dict[str, Any] = {}
    robot = snap.get("robot")
    if isinstance(robot, dict):
        out["robot"] = {
            k: robot.get(k)
            for k in (
                "robot_id", "group_id", "base_frame", "tcp_frame",
                "joint_names", "joint_positions", "tcp_pose_in_base",
            )
            if k in robot
        }
    objects = snap.get("objects")
    if isinstance(objects, list):
        all_ids: list[str] = [
            o.get("object_id") for o in objects
            if isinstance(o, dict) and isinstance(o.get("object_id"), str)
        ]
        out["object_ids"] = all_ids
        kept_idxs: set[int] = set()
        for i, _id in enumerate(all_ids[:_DETAIL_RESOURCE_LIMIT]):
            kept_idxs.add(i)
        detail_records = []
        for i, o in enumerate(objects):
            if i not in kept_idxs:
                continue
            if not isinstance(o, dict):
                continue
            detail_records.append(_project_scene_object(o))
        out["objects"] = detail_records
        if len(detail_records) < len(all_ids):
            out["__objects_detail_omitted_count__"] = (
                len(all_ids) - len(detail_records)
            )
    return out


def _project_node_catalog(catalog: list[Any]) -> list[dict[str, Any]]:
    """Project nodeCatalog entries to id + kind + 1-line description.

    Drops the full per-node `parameters[]` schemas that previously
    bloated the envelope (~25 KB raw → ~3 KB projected). The full
    per-tool parameter schemas are already advertised in the
    OpenAI-style `tools[]` array sent on every chat-completion, so
    duplicating them in the user message is pure waste."""
    out = []
    if not isinstance(catalog, list):
        return out
    for entry in catalog:
        if not isinstance(entry, dict):
            continue
        nid = entry.get("id")
        if not isinstance(nid, str):
            continue
        desc = entry.get("description") or ""
        if isinstance(desc, str):
            # First non-empty line, capped at 160 chars — enough to
            # disambiguate kinds without dumping the full body.
            first_line = next((line.strip() for line in desc.split("\n")
                              if line.strip()), "")
            short_desc = first_line[:160]
        else:
            short_desc = ""
        out.append({
            "id": nid,
            "kind": entry.get("kind") or nid,
            "description": short_desc,
        })
    return out


def _snapshot_state_hashes(
    program_snapshot: dict[str, Any] | None,
    scene_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(program_snapshot, dict):
        for key in ("programTreeHash", "programRevision", "draftRevision"):
            value = program_snapshot.get(key)
            if value is not None:
                out[key] = value
    if isinstance(scene_snapshot, dict):
        for key in ("sceneHash", "sceneRevision", "revision"):
            value = scene_snapshot.get(key)
            if value is not None:
                out[key] = value
    return out


def _build_state_context(
    *,
    program_snapshot: dict[str, Any] | None,
    scene_snapshot: dict[str, Any] | None,
    control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the neutral state capsule included in the request context.

    Inclusion is controlled by the service layer. This function only
    packages read-only state in a fixed, prompt-independent shape.
    """
    control = control if isinstance(control, dict) else {}
    include = bool(control.get("include", True))
    has_snapshots = program_snapshot is not None or scene_snapshot is not None
    context: dict[str, Any] = {
        "schemaVersion": "manyforge.state_context.v0",
        "included": include and has_snapshots,
        "policy": control.get("mode") or "always",
        "everyN": control.get("everyN"),
        "sequence": control.get("sequence"),
        "reason": control.get("reason") or ("included" if include else "cadence_skip"),
        "lastIncludedSequence": control.get("lastIncludedSequence"),
        "hashes": _snapshot_state_hashes(program_snapshot, scene_snapshot),
    }
    if not has_snapshots:
        context["reason"] = "no_snapshots"
        return context
    if not include:
        return context
    if program_snapshot is not None:
        context["program"] = _build_program_summary(program_snapshot)
    if scene_snapshot is not None:
        context["scene"] = _build_scene_summary(scene_snapshot)
    return context


def _project_skill_catalog(catalog: list[Any]) -> list[dict[str, Any]]:
    """Project skillCatalog entries to id + 1-line description."""
    out = []
    if not isinstance(catalog, list):
        return out
    for entry in catalog:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("id")
        if not isinstance(sid, str):
            continue
        desc = entry.get("description") or ""
        if isinstance(desc, str):
            first_line = next((line.strip() for line in desc.split("\n")
                              if line.strip()), "")
            short_desc = first_line[:160]
        else:
            short_desc = ""
        out.append({"id": sid, "description": short_desc})
    return out


def build_agent_prompt(
    payload: dict[str, Any],
    *,
    tool_surface: str = "code",
    state_context_control: dict[str, Any] | None = None,
) -> str:
    """Build the OpenClaw prompt for one ManyForge assistant request.

    Envelope-size optimization (2026-05-08):
    Per-turn user message was ~60 KB before, contributing to OpenClaw's
    preemptive context-overflow guard firing after ~3 turns at 64K
    context. This rewrite drops the per-turn cost to ~10-15 KB by:

    - Replacing raw `programSnapshot` / `sceneSnapshot` JSON dumps with
      a `stateContext` capsule containing structured projections that
      keep complete name/id indexes + bounded first-N detail. The
      service layer controls inclusion cadence; projection is never
      prompt-keyword based.
    - Replacing `nodeCatalog` (full per-node parameters[]) with a
      kind+1-line-description list. Full param schemas already live in
      the OpenAI `tools[]` array on the same request — duplicating in
      the user message is pure waste.
    - Same for `skillCatalog`.
    - Keeping `allowedTools` as a compact mode-catalog id list while
      dropping prompt-derived request windows.
    - Dropping `visibleMcpTools` (string list redundant with tools[]).
    - Dropping the per-turn 12-rule RULES block. The same rules live in
      the workspace AGENTS.md (``${MANYFORGE_ROOT}/agent-skills/
      manyforge-composer/workspace-AGENTS.md``) which OpenClaw injects
      into the system prompt on every turn — the per-turn re-injection
      was duplicate, ~3 KB per turn × every turn.
    """

    request_id = request_id_from_payload(payload)
    assistant_mode = str(payload.get("assistantMode") or "").strip() or "(none)"
    message = payload.get("message")
    if not isinstance(message, str):
        message = json.dumps(message, sort_keys=True)

    # `allowedTools` kept (id-only list, ~500 B). Although the canonical
    # tool surface lives in the OpenAI `tools[]` array on every
    # chat-completion, today's smoke (2026-05-08) showed that removing
    # this id list caused P1 wrap-root to regress 3/3 → 0/3: the model
    # gives up faster on validation errors when it isn't reminded by
    # name, in close proximity to RULES + user_request, that the tools
    # are callable. This is the full assistant-mode catalog, not a
    # request-scoped prompt-derived window.
    #
    # Dropped 2026-05-08 and NOT restored:
    #   - `allowedNodes`: strict subset projection of `nodeCatalog[*].id`
    #     — derivable from the catalog the model already sees. No
    #     regression observed.
    #   - `allowedSkills`: same reasoning.

    tools_raw = payload.get("tools")
    tools = tools_raw if isinstance(tools_raw, list) else []
    visible_tool_ids = [
        t.get("id") for t in tools
        if isinstance(t, dict) and isinstance(t.get("id"), str) and t.get("id")
    ]
    allowed_tool_ids = visible_tool_ids

    node_catalog_raw = payload.get("nodeCatalog")
    node_catalog = node_catalog_raw if isinstance(node_catalog_raw, list) else []
    skill_catalog_raw = payload.get("skillCatalog")
    skill_catalog = skill_catalog_raw if isinstance(skill_catalog_raw, list) else []
    program_snapshot_raw = payload.get("programSnapshot")
    program_snapshot = program_snapshot_raw if isinstance(program_snapshot_raw, dict) else None
    scene_snapshot_raw = payload.get("sceneSnapshot")
    scene_snapshot = scene_snapshot_raw if isinstance(scene_snapshot_raw, dict) else None
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}

    preamble: dict[str, Any] = {
        "requestId": request_id,
        "assistantMode": assistant_mode,
        "conversationId": payload.get("conversationId"),
        "runtime": runtime,
        "allowedTools": allowed_tool_ids,
        "nodeCatalog": _project_node_catalog(node_catalog),
        "skillCatalog": _project_skill_catalog(skill_catalog),
    }
    preamble["stateContext"] = _build_state_context(
        program_snapshot=program_snapshot,
        scene_snapshot=scene_snapshot,
        control=state_context_control,
    )

    # Full RULES block restored 2026-05-08: smoke after the condense
    # showed P1 wrap-root regress 3/3 → 0/3. The 4 rules I had moved
    # to "rely on workspace AGENTS.md" weren't enough on their own;
    # the proximity of the verbose rules to RULES + user_request in
    # the same per-turn message is load-bearing. ~1 KB extra cost is
    # acceptable at 256K context.
    # 2026-06-03: filter empty entries so OPENCLAW_BRIDGE_ITER32_COMPAT_MODE
    # can drop individual rules by yielding "" without leaving blank lines.
    rules_block = "\n".join(
        s for s in [
            "RULES (do not violate):",
            # Rule 0 is the dispatch-surface primer. It teaches the
            # model HOW to invoke real tools given the active
            # OpenClaw surface (code mode via tool_search_code, or
            # tools mode via tool_search/tool_describe/tool_call).
            # The argument-shape rules below (8a) are surface-
            # agnostic and apply identically in both modes.
            _primer_for_tool_surface(tool_surface),
            # Rule 0b is tools-mode specific: pins the MCP-exposed
            # name/id/label shape the model must use when the
            # `tool_search` result is the discovery surface. Not
            # emitted in code mode (the tool_search_code primer
            # already documents `openclaw.tools.call`'s direct
            # access pattern). Gated by the compat mode like the
            # other post-iter-32 rules so the A/B can isolate the
            # contract's contribution to the recovery rate.
            (_PRIMER_TOOLS_MODE_RULE_0B
             if tool_surface == "tools" and not _ITER32_COMPAT_MODE
             else ""),
            "1. Catalog ids are immutable. Use only the literal id from "
            "the nodeCatalog or skillCatalog below. Never invent ids by "
            "composing a kind id with an object name (e.g. NEVER "
            "`attach_graspable` — the kind is `attach_object_to_link` "
            "and `graspable` goes in `params.object_id`). If the id you "
            "want is not present below, it does not exist.",
            "2. The `id` field of a node is the catalog identifier "
            "(immutable). The `name` field is a caller-chosen instance "
            "label and may include any descriptive text. Object "
            "identifiers belong in `params.object_id`, never in `id`.",
            "3. `scene_draft_*` tools edit STATIC scene resources "
            "(compile-time, present for every cycle). `tree_draft_*` "
            "tools edit RUNTIME behavior (executes during a cycle at a "
            "specific point). They are distinct namespaces. Crucial "
            "carve-out: the nodeCatalog also exposes RUNTIME node "
            "kinds that act on collision objects during execution — "
            "`add_collision_object`, `upsert_collision_object`, "
            "`remove_collision_object`, `update_collision_object_pose`, "
            "`attach_object_to_link`, `detach_object_from_link`. These "
            "are NODES inserted with `tree_draft_insert_node` (kind = "
            "one of the above), NOT scene_draft tools. Decision: if the "
            "user says \"a node that places/removes/updates X at the "
            "end of <sequence>\" or \"during the cycle\" or \"at "
            "runtime\" or anywhere phrasing implies behavior tree "
            "ordering, use `tree_draft_insert_node` with the matching "
            "runtime kind. Use `scene_draft_*` only when the user is "
            "authoring the static scene (\"add an obstacle to the "
            "scene\", \"place a box at <pose>\", \"resize the "
            "ground\") with no reference to the tree, sequence, "
            "cycle, or runtime ordering.",
            "4. `stateContext` in the context below is read-only live "
            "request state. When `stateContext.included=true`, read "
            "`stateContext.program` and `stateContext.scene` directly "
            "for node names, object ids, dimensions, and poses. When "
            "the capsule is skipped by cadence, call `program_read` or "
            "`scene_inspect` if the request requires state you cannot "
            "infer safely.",
            "5. The `nodeCatalog` and `skillCatalog` in the context "
            "below carry the SAME payload `catalog_read` and "
            "`skills_read` would return — full descriptions, "
            "parameters, capabilities. Read them directly; do not "
            "round-trip through the read tools for first-pass "
            "discovery.",
            "6. On a 4xx tool-call response, read the structured fields "
            "(`allowedNodeKinds`, `validParentNames`, "
            "`validTargetNames`, `validNodeNames`, "
            "`wrapperIdSuggestions`, `rejectedNodeKinds`). Pick a value "
            "from those fields. Do NOT retry with identical args.",
            "7. If the same call has just failed twice with the same "
            "error class, stop and emit a final assistant message "
            "asking the user for direction. Do not try a third time.",
            "8. To make a node the new ROOT of the tree (\"add X as "
            "root\", \"as the tree root\", \"wrap the root with X\", "
            "\"X around the root\"), use `tree_draft_wrap_node` with "
            "the CURRENT root's name as `targetName`. Never use "
            "`tree_draft_insert_node` for root-replacement intents — "
            "insert places a node as a child of an existing parent, so "
            "calling it repeatedly nests deeper instead of reaching the "
            "root, producing a spiral. If you cannot identify the "
            "current root from `stateContext.program`, stop and ask.",
            "" if _ITER32_COMPAT_MODE else
            "8a. **Tool argument names are camelCase, NOT snake_case.** "
            "Even when emitting JavaScript-like code via "
            "`tool_search_code`, the JSON object keys passed to real "
            "tools must use the EXACT camelCase field names from the "
            "schema. Common pitfalls and correct forms: "
            "`tree_draft_wrap_node` → `{targetName, wrapper: {id, "
            "name, params}}` (NOT `target_name`/`wrapper_id`); "
            "`tree_draft_insert_node` → `{nodeName, parentName, "
            "node: {id, params}, index | afterName | beforeName}` "
            "(all top-level — there is NO `position` wrapper; the "
            "`node` payload does NOT carry `name`, that is the "
            "top-level `nodeName`); "
            "`tree_draft_update_node_params` → `{nodeName, params, "
            "merge}` (NOT `node_name`); "
            "`tree_draft_delete_node` → `{nodeName}`; "
            "`tree_draft_move_node` → `{nodeName, newParentName, "
            "position}`; "
            "`tree_draft_change_node_kind` → `{nodeName, newId, "
            "params}` (NOT `newKind` — the field is `newId` because it "
            "is the new catalog identifier); "
            "`tree_draft_replace_subtree` → `{nodeName, subtree: {id, "
            "name, params, children}}` (NOT `targetName`/`replacement`; "
            "the schema uses `nodeName` + `subtree`); "
            "`scene_draft_add_object` → `{objectId, shape: {type, "
            "box_dimensions_m}, pose: {frame_id, position_m, "
            "orientation_xyzw}}` (top-level is `objectId`, NOT `name`; "
            "shape uses `box_dimensions_m`, NOT `box_dims`; pose uses "
            "`orientation_xyzw` (quaternion [x,y,z,w]) — NOT "
            "`orientation_quat`; `orientation_rpy_deg` is also accepted); "
            "`scene_draft_update_object` → `{objectId, pose, shape}`; "
            "`scene_draft_remove_object` → `{objectId}`. "
            "Top-level keys are camelCase. Compound payloads are "
            "nested objects, never flattened (use `wrapper: {id: ..., "
            "name: ...}`, NOT `wrapper_id` + `wrapper_name` separately). "
            "Argument-key validation failures are the #1 source of "
            "loops; the validator's error envelope includes "
            "`expectedSchema`, `providedArgs`, and `diff` fields with a "
            "`hint` line naming the EXACT rename (e.g. \"Rename keys: "
            "'target_name' → 'targetName'\"). Read those fields before "
            "retrying — identical-args retries are auto-detected and "
            "terminated as runaway loops.",
            "9. Every tool result carries a `result.delta` block with "
            "the live state diff: `result.delta.rootName` and "
            "`result.delta.rootKind` are the LIVE tree root after "
            "this call; `result.delta.sinceCallStart` lists what "
            "*this* call just changed (added/removed/moved nodes, "
            "added/removed/changed scene objects, with per-key "
            "before/after for params/pose/dimensions); "
            "`result.delta.sinceConversationStart` lists cumulative "
            "changes since this conversation began. Read it before "
            "calling another tool. Goal-completion check: if the user "
            "asked for \"a repeat as root\" and "
            "`result.delta.rootKind == 'repeat'`, the goal is "
            "satisfied — emit the final assistant message and stop. "
            "Never wrap a node whose kind already matches the request; "
            "that just nests deeper. The delta replaces guessing "
            "\"did my mutation land?\" — never call `program_read` "
            "to confirm what the delta already shows.",
            "11. **Action-shaped requests require a decision.** If the "
            "user asks to add, remove, delete, insert, update, wrap, swap, "
            "move, create, make, place, attach, detach, set, replace, or "
            "modify a node, object, scene resource, or program element, "
            "either emit the appropriate tool call when the required target "
            "and parameters are known, or ask one concise clarification "
            "question when required information is missing. Do not invent a "
            "parent, target, wrapper, position, object id, or parameter value "
            "that the user did not provide and that is not available in the "
            "request context. Do not refuse by saying the tool is unavailable "
            "when it is listed in `allowedTools` or appears in your callable "
            "interface. Never reply with the literal string `NO_REPLY` for an "
            "action request.",
            "10. Namespace decision rules, in order of precedence: "
            "(a) If the request says \"node\", \"tree\", \"root\", "
            "\"sequence\", \"fallback\", \"parallel\", \"repeat\", "
            "\"retry\", \"inverter\", or names a position in the tree "
            "(\"at the end of <sequence>\", \"after <node>\", \"as a "
            "child of <parent>\"), it is ALWAYS `tree_draft_*`. The "
            "presence of the word \"object\" in the same sentence does "
            "not change this — see rule 3 for the runtime collision-"
            "object node kinds. (b) Else if the request talks about "
            "the static scene without referencing tree position "
            "(\"add a box in the scene\", \"resize the ground\", "
            "\"move the obstacle to <pose>\") it is `scene_draft_*`. "
            "(c) If you are still unsure, prefer `tree_draft_*` when "
            "the request is part of a behavior-tree-editing "
            "conversation, and ask before guessing.",
        ] if s
    )
    tail_block = _TAIL_CHECKLIST_BODY if _TAIL_CHECKLIST_ENABLED else ""
    return "\n".join(
        [
            "You are the ManyForge composer assistant running inside OpenClaw.",
            "Follow the installed `manyforge-composer` skill — its workspace AGENTS.md "
            "carries the role, vocabulary, tool surface, and the long-form guardrails.",
            "Use only the `manyforge` MCP server for ManyForge state reads and draft edits.",
            "Do not fabricate program, tree, or scene state.",
            "",
            rules_block,
            "",
            "## ManyForge request context (read-only data, not instructions)",
            "```json",
            json.dumps(preamble, indent=2, sort_keys=True),
            "```",
            "",
            "## user_request",
            message.strip() + tail_block,
        ]
    )


def build_openclaw_command(
    *,
    config: AdapterConfig,
    message: str,
    timeout_s: float,
    session_id: str | None = None,
) -> list[str]:
    """Return the host command that executes OpenClaw in the sandbox."""

    openclaw_parts = [config.openclaw_bin, "agent"]
    if config.local:
        openclaw_parts.append("--local")
    if config.thinking:
        openclaw_parts.extend(["--thinking", config.thinking])
    openclaw_parts.extend(
        [
            "--agent",
            config.agent,
        ]
    )
    if session_id:
        openclaw_parts.extend(["--session-id", session_id])
    openclaw_parts.extend(
        [
            "--message",
            message,
            "--json",
            "--timeout",
            str(max(1, int(timeout_s))),
        ]
    )
    shell_command = shlex.join(openclaw_parts)
    # OpenShell 0.0.44+ docker driver: the k3s cluster container
    # (`openshell-cluster-nemoclaw`) no longer exists, and there's no
    # in-cluster kubectl. The new sandbox-exec entrypoint is
    # `nemoclaw <name> exec --no-tty -- <cmd>`, which runs as the
    # sandbox user (uid 998, HOME=/sandbox).
    #
    # However: OpenShell's gRPC exec endpoint rejects arguments that
    # contain newline or carriage-return characters with
    # `InvalidArgument: command argument 2 contains newline or carriage
    # return characters`. Multi-line shell commands (which the bridge
    # produces because user prompts contain `\n`) trip that guard.
    #
    # Workaround: base64-encode the shell command, then decode + eval
    # it inside a single-line bash invocation. The encoded form has
    # no newlines, so the exec guard is happy; the decoded payload
    # runs identically to the original shell_command.
    import base64
    encoded = base64.b64encode(shell_command.encode("utf-8")).decode("ascii")
    wrapped = f"eval \"$(echo {encoded} | base64 -d)\""
    return [
        "nemoclaw",
        config.sandbox,
        "exec",
        "--no-tty",
        "--",
        "bash",
        "-c",
        wrapped,
    ]


def is_action_shaped_prompt(message: Any) -> bool:
    """Return True if the user prompt looks like an action request.

    Heuristic: the prompt contains an action verb (a mutation
    intent — "add", "wrap", "update", etc.) AND a domain noun
    (a tree/scene/program object). When True, the bridge passes
    ``tool_choice: "required"`` to vLLM, structurally forcing the
    model to emit a tool call rather than refusing in prose.
    Returning False for ambiguous prompts ("what does this do?",
    "explain X") preserves auto-mode for legitimate read-only
    inquiries.
    """
    if not isinstance(message, str):
        return False
    text = message.lower()
    action_verbs = (
        "add", "remove", "delete", "insert", "update", "wrap",
        "swap", "move", "create", "make", "place", "attach",
        "detach", "set", "replace", "rename", "reorder", "drop",
        "modify", "resize", "change",
    )
    domain_nouns = (
        "node", "tree", "root", "sequence", "fallback", "parallel",
        "repeat", "retry", "inverter", "object", "obstacle", "box",
        "sphere", "cylinder", "ground", "collider", "graspable",
        "scene", "program", "parameter", "param", "blackboard",
    )
    has_verb = any(v in text for v in action_verbs)
    has_noun = any(n in text for n in domain_nouns)
    return has_verb and has_noun


def derive_gateway_session_key(payload: dict[str, Any]) -> str:
    """Compute the OpenClaw session key for one Composer chat request.

    The key isolates one Composer conversation from another at the
    OpenClaw gateway scheduler so stuck-session diagnostics, lifecycle
    events, and queue state from a prior conversation do not leak into
    the next one's prompt context. The key is also rotated when any of
    catalogHash / programRevision / deploymentId changes within the
    same conversation, so a deployment reload or draft-apply mid-stream
    invalidates the agent's stale context the next time the user sends
    a message.

    Inputs (all optional; the function is permissive):

    - ``conversationId`` (top-level; assigned by Composer per chat
      thread): the primary scope for the session.
    - ``context.catalogHash``: changes when the deployment YAML's tool
      catalog rotates.
    - ``context.programRevision`` / ``context.draftRevision``: change
      when the operator adopts a draft, loads a new program, or
      hot-edits the tree.
    - ``context.deploymentId``: changes when a different deployment is
      loaded in Composer.

    The returned key is sent verbatim as the ``x-openclaw-session-key``
    HTTP header on ``/v1/chat/completions`` (per
    docs.openclaw.ai/gateway/openai-http-api). Value shape is opaque to
    the gateway; we use a prefixed compact hash so it is recognizable in
    logs and short enough to fit header length limits.
    """

    conversation_id = str(payload.get("conversationId") or "").strip()
    if not conversation_id:
        # Fall back to requestId when Composer doesn't pass a
        # conversationId, so we still get one-session-per-request
        # isolation rather than no isolation at all.
        conversation_id = request_id_from_payload(payload)

    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    parts = [
        conversation_id,
        str(context.get("catalogHash") or ""),
        str(context.get("programRevision") or ""),
        str(context.get("draftRevision") or ""),
        str(context.get("deploymentId") or ""),
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    safe_conv = re.sub(r"[^A-Za-z0-9_-]+", "_", conversation_id)[:40] or "conv"
    return f"manyforge-{safe_conv}-{digest}"


def build_gateway_chat_completions_command(
    *,
    config: AdapterConfig,
    payload: dict[str, Any],
    timeout_s: float,
    message: str | None = None,
) -> list[str]:
    """Return a host-side curl command that calls the OpenClaw gateway.

    The canonical NemoClaw setup (``configure-local-provider.sh`` ->
    ``ensure_sandbox_gateway_running``) spawns the gateway inside the
    openshell SSH-session network namespace and exposes it on the host
    as ``127.0.0.1:<gateway_port>`` via the openshell port-forward
    tunnel. We call directly from the host through that tunnel; no
    kubectl-exec wrapper is needed.

    The provisioner uses ``--auth none`` (the documented default) so the
    SSH-tunnel hop already enforces auth. If the gateway is configured
    for token auth instead, set ``OPENCLAW_GATEWAY_TOKEN`` to inject a
    Bearer header.

    Required policy precondition: the sandbox network policy must include
    an endpoint for ``host.openshell.internal:8000`` (vLLM) with the
    canonical ``allowed_ips`` field opting in to the resolved private IP.
    Without that, OpenShell's SSRF guard rejects the gateway's outbound
    fetch to vLLM and chat-completions return ``internal error``. The
    ``manyforge-composer`` preset shipped with this repo provides this;
    apply it via ``setup-manyforge-assistant.sh``.

    Session isolation: every request is stamped with an
    ``x-openclaw-session-key`` header derived from the Composer
    conversationId plus the catalog and revision metadata in the
    request context. This isolates conversations at the gateway
    scheduler and rotates the session whenever a relevant identifier
    changes, preventing the stuck-session / observability-leak class of
    bug where one conversation inherits a prior conversation's stale
    state.

    The persistent gateway eliminates the per-call ~40s CLI bootstrap of
    the openclaw-agent shell-out path; warm calls drop to 5-10s.
    """

    # If a fully-built prompt is passed via `message`, use it (this is
    # what `build_agent_prompt(payload, ...)` returns — preamble +
    # request context + user request, all in one string). Falls back
    # to the raw payload message for back-compat with any caller that
    # hasn't been updated. Without the explicit `message` parameter
    # this builder used to *silently discard* the enriched prompt and
    # only forward the user's raw message — so nodeCatalog,
    # programSnapshot, sceneSnapshot, and instructional text were never
    # reaching the model. Verified 2026-05-06 against a 25KB built
    # prompt that landed as a 196-char user message in the OpenClaw
    # session log; fixed here.
    user_message = message if isinstance(message, str) else payload.get("message")
    if not isinstance(user_message, str):
        user_message = json.dumps(user_message, sort_keys=True)
    request_body: dict[str, Any] = {
        "model": f"openclaw/{config.agent}",
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": config.gateway_max_tokens,
        "stream": False,
    }
    # 2026-05-09: tool_choice injection disabled here — the new
    # vllm-proxy on :8000 sits between OpenClaw and vLLM and
    # owns per-turn tool_choice control (alternating mode etc.). Having
    # the bridge ALSO inject tool_choice="required" at the top of the
    # stack would conflict with the proxy's per-turn logic, since
    # OpenClaw forwards the bridge's value into every internal vLLM
    # call and the proxy would have to STRIP it on odd turns rather
    # than just inject on even turns. Cleaner to leave the field unset
    # here and let the proxy be the single source of truth. Re-enable
    # for the direct lane when we revisit it.
    # raw_user_prompt = payload.get("message")
    # if is_action_shaped_prompt(raw_user_prompt):
    #     request_body["tool_choice"] = "required"
    # Sampling-parameter parity with the direct-vLLM bridge. The gateway
    # accepts standard OpenAI fields and forwards them; without these
    # the model rambles 2-15x longer per turn at OpenClaw's defaults.
    if config.gateway_temperature is not None:
        request_body["temperature"] = config.gateway_temperature
    if config.gateway_top_k is not None:
        request_body["top_k"] = config.gateway_top_k
    if config.gateway_top_p is not None:
        request_body["top_p"] = config.gateway_top_p
    chat_template_kwargs: dict[str, Any] = {}
    if config.gateway_enable_thinking is not None:
        chat_template_kwargs["enable_thinking"] = config.gateway_enable_thinking
    if chat_template_kwargs:
        request_body["chat_template_kwargs"] = chat_template_kwargs
    body_json = json.dumps(request_body)
    curl_timeout = max(5, int(timeout_s) - 1)
    session_key = derive_gateway_session_key(payload)
    args = [
        "curl",
        "-sS",
        "--max-time",
        str(curl_timeout),
        "-H",
        "Content-Type: application/json",
        "-H",
        f"x-openclaw-session-key: {session_key}",
        "-X",
        "POST",
        f"http://127.0.0.1:{config.gateway_port}/v1/chat/completions",
        "-d",
        body_json,
    ]
    token = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "").strip()
    if token:
        # Insert Authorization header just before the URL.
        args = args[:8] + ["-H", f"Authorization: Bearer {token}"] + args[8:]
    return args


def parse_chat_completions_response(
    stdout: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Extract the OpenAI-shaped chat-completions response.

    Curl emits a single JSON document. We accept either that or a JSON
    object embedded in noisier output (e.g., warning lines from tooling).
    """

    text = stdout.strip()
    if not text:
        return None, ["gateway returned empty body"]
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None, []
    except ValueError:
        pass

    decoder = JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(candidate, dict):
            return candidate, []
    return None, ["could not parse JSON from gateway response"]


def normalize_chat_completions_response(
    *,
    payload: dict[str, Any],
    response_json: dict[str, Any] | None,
    stdout: str,
    stderr: str = "",
    parse_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Convert an OpenAI chat-completions response into the ManyForge envelope.

    The OpenClaw gateway runs the tool-call loop internally, so the response
    we see has only the final assistant message. Tool-call audit lives on
    the ManyForge side (bridge audit records carry requestId + assistantMode
    + catalogHash + principal). The envelope we return therefore has empty
    toolCalls/proposals; draftMutated is conservatively false here and gets
    its real value from Composer's own state delta after the request lands.
    """

    request_id = request_id_from_payload(payload)
    warnings = list(parse_warnings or [])
    if stderr.strip():
        warnings.append(f"gateway stderr: {_truncate(stderr.strip(), 500)}")

    if response_json is None:
        message = _truncate(stdout.strip(), 2000) or (
            "OpenClaw gateway returned no parseable response."
        )
        return {
            "version": SUPPORTED_VERSION_FAMILY,
            "schemaVersion": DEFAULT_SCHEMA_VERSION,
            "requestId": request_id,
            "message": message,
            "toolCalls": [],
            "proposals": [],
            "warnings": warnings,
            "mutated": False,
            "draftMutated": False,
            "requiresReview": True,
        }

    err = response_json.get("error")
    if isinstance(err, dict):
        detail = err.get("message") or err.get("type") or "gateway returned an error"
        return {
            "version": SUPPORTED_VERSION_FAMILY,
            "schemaVersion": DEFAULT_SCHEMA_VERSION,
            "requestId": request_id,
            "message": str(detail),
            "toolCalls": [],
            "proposals": [],
            "warnings": warnings + [f"gateway error: {detail}"],
            "mutated": False,
            "draftMutated": False,
            "requiresReview": True,
            "error": {"code": err.get("type") or "gateway_error", "detail": str(detail)},
        }

    message = ""
    choices = response_json.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                message = _flatten_chat_content(content).strip()
    if not message:
        message = "OpenClaw gateway returned an empty assistant message."
        warnings.append("gateway response had no choices[0].message.content")

    # Filter OpenClaw scheduler/lifecycle telemetry that the gateway can
    # inject into the agent's input context (stuck-session warnings,
    # failover decisions, queue-depth notices). When the model echoes any
    # of these patterns in its visible answer, replace the message with a
    # generic clarification and surface the leakage as a structured warning
    # so reviewers can see what happened. Without this, stuck-session
    # diagnostics get mistaken for real assistant questions ("which session
    # do you mean?") that the user has no way to answer. Pure-prefix /
    # exact-substring detection is conservative — false positives in normal
    # answers are unlikely because these phrases are OpenClaw-internal.
    leakage_match = _detect_observability_leakage(message)
    if leakage_match is not None:
        warnings.append(
            "openclaw_observability_leakage: model echoed an OpenClaw "
            f"scheduler diagnostic ({leakage_match!r}); message replaced "
            "with a neutral fallback. The gateway scheduler likely had a "
            "stuck/dirty session for this conversationId; restart the "
            "gateway via configure-local-provider.sh and consider whether "
            "session isolation is wired through correctly."
        )
        message = (
            "I couldn't form a useful answer. The OpenClaw gateway's "
            "internal scheduler emitted a diagnostic that bled into my "
            "context. Please retry your request; if the problem persists, "
            "restart the gateway."
        )

    return {
        "version": SUPPORTED_VERSION_FAMILY,
        "schemaVersion": DEFAULT_SCHEMA_VERSION,
        "requestId": request_id,
        "message": message,
        "toolCalls": [],
        "proposals": [],
        "warnings": warnings,
        "mutated": False,
        "draftMutated": False,
        "requiresReview": True,
    }


# OpenClaw scheduler/lifecycle phrases that have been observed leaking
# into the agent's input context and then echoed back into model output.
# Match conservatively (case-insensitive substring) so a user genuinely
# discussing "stuck sessions" or "lane errors" in conversation isn't
# false-positive'd. The patterns here come from real gateway log
# observation, not speculation; add new ones as they're seen in the
# wild.
#
# Two categories of pattern:
#   1) Direct echoes of gateway diagnostic strings (`stuck session`,
#      `lane task error`, etc.). Low risk of false positive — these are
#      OpenClaw-internal phrasings unlikely to appear in normal
#      ManyForge conversation.
#   2) Model "asking for a session" responses that arise when the agent
#      has only `session_status` in its tool list and conflates the
#      user's ManyForge question with a session lookup. Patterns here
#      are broader because the model varies wording; ManyForge's domain
#      (scenes, behavior trees, programs) doesn't naturally include
#      "session" terminology, so substring matches are safe.
_OPENCLAW_LEAKAGE_PATTERNS: tuple[str, ...] = (
    # Category 1: direct gateway diagnostic echoes
    "stuck session",
    "lane task error",
    "embedded_run_failover_decision",
    "embedded_run_agent_end",
    "queue depth",
    # Category 2: model asking for a "session" identifier in response
    # to a ManyForge question (the actual symptom we observed)
    "session key",
    "session id",
    "session identifier",
    "which session",
    "the session you",
    "the session for",
    "clarify which session",
)


def _detect_observability_leakage(message: str) -> str | None:
    """Return the matched leakage phrase, or None if the message is clean.

    Matches case-insensitive substrings against
    ``_OPENCLAW_LEAKAGE_PATTERNS``. The first match wins; we surface that
    one in the warning so a reviewer can correlate it with gateway logs.
    """

    if not isinstance(message, str) or not message.strip():
        return None
    lowered = message.lower()
    for pattern in _OPENCLAW_LEAKAGE_PATTERNS:
        if pattern.lower() in lowered:
            return pattern
    return None


def parse_openclaw_json(stdout: str) -> tuple[Any | None, list[str]]:
    """Extract a JSON object from OpenClaw stdout.

    OpenClaw normally emits one JSON document with ``--json``. In practice,
    wrappers can prepend logs, so this accepts full-output JSON, JSON-lines, or
    the last decodable object embedded in the stream.
    """

    text = stdout.strip()
    if not text:
        return None, ["OpenClaw produced no stdout"]
    try:
        return json.loads(text), []
    except ValueError:
        pass

    warnings: list[str] = []
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped or stripped[0] not in "[{":
            continue
        try:
            return json.loads(stripped), warnings
        except ValueError:
            continue

    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped not in {"{", "["} and not stripped.startswith(("{", "[")):
            continue
        candidate = "\n".join(lines[index:]).strip()
        try:
            return json.loads(candidate), warnings
        except ValueError:
            continue

    decoder = JSONDecoder()
    parsed: Any | None = None
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        parsed = candidate
    if parsed is None:
        warnings.append("Could not parse JSON from OpenClaw stdout; using text fallback")
    return parsed, warnings


def normalize_agent_response(
    *,
    payload: dict[str, Any],
    agent_json: Any | None,
    stdout: str,
    stderr: str = "",
    parse_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Convert OpenClaw output into the ManyForge assistant-provider envelope."""

    request_id = request_id_from_payload(payload)
    allowed_tools = _allowed_tool_map(payload)
    allowed_tool_ids = set(allowed_tools)
    warnings = list(parse_warnings or [])
    if stderr.strip():
        warnings.append(f"OpenClaw stderr: {_truncate(stderr.strip(), 500)}")

    message = _extract_message(agent_json)
    if not message:
        message = _truncate(stdout.strip(), 2000)
    if not message:
        message = "OpenClaw completed without a visible assistant message."

    tool_calls, tool_warnings = _extract_tool_calls(agent_json, allowed_tools)
    warnings.extend(tool_warnings)

    proposals = []
    if isinstance(agent_json, dict) and isinstance(agent_json.get("proposals"), list):
        proposals = [item for item in agent_json["proposals"] if isinstance(item, dict)]

    draft_mutated = _contains_truthy_key(agent_json, {"draftMutated", "draft_mutated"})
    if not draft_mutated:
        draft_mutated = any(
            call.get("status") == "completed"
            and allowed_tools.get(str(call.get("name")), "") == "composer_draft_mutating"
            for call in tool_calls
        )

    out_of_catalog = sorted(
        {
            str(call.get("name"))
            for call in tool_calls
            if str(call.get("name")) not in allowed_tool_ids
        }
    )
    if out_of_catalog:
        warnings.append(
            "Dropped out-of-catalog OpenClaw tool references: "
            + ", ".join(out_of_catalog)
        )
        tool_calls = [
            call for call in tool_calls if str(call.get("name")) in allowed_tool_ids
        ]

    return {
        "version": SUPPORTED_VERSION_FAMILY,
        "schemaVersion": DEFAULT_SCHEMA_VERSION,
        "requestId": request_id,
        "message": message,
        "toolCalls": tool_calls,
        "proposals": proposals,
        "warnings": warnings,
        "mutated": False,
        "draftMutated": draft_mutated,
        "requiresReview": True,
    }


def error_envelope(
    *,
    request_id: str,
    code: str,
    detail: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "version": SUPPORTED_VERSION_FAMILY,
        "schemaVersion": DEFAULT_SCHEMA_VERSION,
        "requestId": request_id,
        "message": detail,
        "toolCalls": [],
        "proposals": [],
        "warnings": list(warnings or [detail]),
        "mutated": False,
        "draftMutated": False,
        "requiresReview": True,
        "error": {"code": code, "detail": detail},
    }


def _allowed_tool_map(payload: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        tool_id = str(tool.get("id") or tool.get("name") or "").strip()
        if tool_id:
            result[tool_id] = str(tool.get("effect") or "")
    return result


def _tool_ids(payload: dict[str, Any]) -> list[str]:
    return list(_allowed_tool_map(payload).keys())


def _dedupe_known(known_order: list[str], candidates: list[str]) -> list[str]:
    wanted = {candidate for candidate in candidates if candidate in known_order}
    return [tool_id for tool_id in known_order if tool_id in wanted]


def _flatten_chat_content(content: Any) -> str:
    """Flatten an OpenAI chat-completions ``message.content`` into plain text.

    Modern chat models (and chat templates that emit Anthropic-style content
    blocks) can return ``content`` as either:

    * a flat string ``"hello"``,
    * a list of typed blocks ``[{"type": "text", "text": "hello"}, ...]``, or
    * (less commonly) a single block ``{"type": "text", "text": "hello"}``.

    Without this helper the gateway path used to accept only the flat-string
    form and fall through to an "empty assistant message" warning when the
    model emitted blocks. Worse, downstream consumers that did ``str(content)``
    surfaced the Python ``repr`` of the list in the assistant chat — visible
    to the user as ``[{'type': 'text', 'text': '...'}]`` literal text. This
    helper joins the ``text`` field of every text-typed block (skipping
    other block kinds such as ``image_url`` that the chat surface cannot
    render textually) so the caller always sees a clean string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            text = block.get("text")
            # Accept text-typed blocks and untyped blocks that nonetheless
            # carry a `text` field — some adapters drop the `type` field.
            if isinstance(text, str) and (block_type in (None, "text") or block_type is None):
                parts.append(text)
        return "".join(parts)
    return ""


def _extract_message(data: Any) -> str:
    if isinstance(data, dict):
        for key in (
            "finalAssistantVisibleText",
            "finalAssistantText",
            "assistantVisibleText",
            "finalResponse",
            "message",
            "text",
            "response",
            "output",
        ):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("result", "data"):
            nested = _extract_message(data.get(key))
            if nested:
                return nested
        payloads = data.get("payloads")
        if isinstance(payloads, list):
            for payload in payloads:
                if isinstance(payload, dict):
                    text = payload.get("text")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
                nested = _extract_message(payload)
                if nested:
                    return nested
    if isinstance(data, list):
        for item in data:
            nested = _extract_message(item)
            if nested:
                return nested
    return ""


def _extract_tool_calls(
    data: Any,
    allowed_tools: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    calls: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    def add_call(raw_name: Any, raw: dict[str, Any] | None = None) -> None:
        if not isinstance(raw_name, str) or not raw_name.strip():
            return
        canonical = _canonical_tool_name(raw_name, set(allowed_tools))
        if not canonical:
            if "manyforge" in raw_name.lower():
                warnings.append(f"Could not map OpenClaw tool name to ManyForge id: {raw_name}")
            return
        raw = raw or {}
        status = str(raw.get("status") or raw.get("state") or "completed").strip() or "completed"
        if status.lower() in {"success", "succeeded", "ok"}:
            status = "completed"
        elif status.lower() in {"error", "errored"}:
            status = "failed"
        arguments = raw.get("arguments") if isinstance(raw.get("arguments"), dict) else {}
        error = raw.get("error") if isinstance(raw.get("error"), str) else None
        result = raw.get("result") if isinstance(raw.get("result"), dict) else {}
        key = (canonical, status, json.dumps(arguments, sort_keys=True))
        if key in seen:
            return
        seen.add(key)
        record = {
            "name": canonical,
            "status": status if status in {"proposed", "skipped", "completed", "failed"} else "completed",
            "arguments": arguments,
            "result": result,
        }
        if error:
            record["error"] = error
        calls.append(record)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            tool_summary = value.get("toolSummary")
            if isinstance(tool_summary, dict):
                for item in _list_value(tool_summary.get("tools")):
                    if isinstance(item, str):
                        add_call(item)
                    elif isinstance(item, dict):
                        add_call(
                            item.get("name")
                            or item.get("toolName")
                            or item.get("tool")
                            or item.get("id"),
                            item,
                        )
                for key in ("toolCalls", "tool_calls", "calls"):
                    for item in _list_value(tool_summary.get(key)):
                        if isinstance(item, dict):
                            add_call(
                                item.get("name")
                                or item.get("toolName")
                                or item.get("tool")
                                or item.get("id"),
                                item,
                            )
            for key in ("toolCalls", "tool_calls", "mcpToolCalls", "mcp_tool_calls"):
                for item in _list_value(value.get(key)):
                    if isinstance(item, dict):
                        add_call(
                            item.get("name")
                            or item.get("toolName")
                            or item.get("tool")
                            or item.get("id"),
                            item,
                        )
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    walk(nested)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)
    return calls, warnings


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _canonical_tool_name(raw: str, allowed: set[str]) -> str | None:
    candidates = [raw.strip()]
    for candidate in list(candidates):
        if "__" in candidate:
            candidates.append(candidate.split("__", 1)[1])
            candidates.append(candidate.rsplit("__", 1)[1])
        if candidate.startswith("manyforge."):
            candidates.append(candidate[len("manyforge.") :])
        if candidate.startswith("manyforge__"):
            candidates.append(candidate[len("manyforge__") :])
    candidates.extend(candidate.replace("__", ".") for candidate in list(candidates))

    for candidate in candidates:
        if candidate in allowed:
            return candidate

    normalized_allowed = {_safe_name(tool_id): tool_id for tool_id in allowed}
    for candidate in candidates:
        safe = _safe_name(candidate)
        if safe in normalized_allowed:
            return normalized_allowed[safe]
    return None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _contains_truthy_key(data: Any, keys: set[str]) -> bool:
    if isinstance(data, dict):
        for key, value in data.items():
            if key in keys and value is True:
                return True
            if isinstance(value, (dict, list)) and _contains_truthy_key(value, keys):
                return True
    elif isinstance(data, list):
        return any(_contains_truthy_key(item, keys) for item in data)
    return False


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"
