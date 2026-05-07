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


@dataclass(frozen=True)
class AdapterConfig:
    """Runtime configuration for invoking OpenClaw inside a sandbox."""

    sandbox: str = "my-assistant"
    namespace: str = "openshell"
    container: str = "agent"
    cluster_container: str = "openshell-cluster-nemoclaw"
    sandbox_user: str = "sandbox"
    agent: str = "main"
    timeout_s: float = 180.0
    openclaw_bin: str = "openclaw"
    local: bool = False
    thinking: str = "off"
    auto_tool_window: bool = True
    allowed_tools_file: str = "/tmp/manyforge-openclaw-allowed-tools.txt"
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
    gateway_enable_thinking: bool | None = None


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


def build_agent_prompt(
    payload: dict[str, Any],
    *,
    mcp_allowed_tools: list[str] | None = None,
) -> str:
    """Build the OpenClaw prompt for one ManyForge assistant request."""

    request_id = request_id_from_payload(payload)
    assistant_mode = str(payload.get("assistantMode") or "").strip() or "(none)"
    message = payload.get("message")
    if not isinstance(message, str):
        message = json.dumps(message, sort_keys=True)

    visible_tool_ids = set(mcp_allowed_tools) if mcp_allowed_tools is not None else None
    tools = [
        {
            "id": str(tool.get("id") or tool.get("name") or ""),
            "effect": str(tool.get("effect") or ""),
            "description": str(tool.get("description") or "")[:600],
        }
        for tool in payload.get("tools") or []
        if isinstance(tool, dict)
        and (tool.get("id") or tool.get("name"))
        and (
            visible_tool_ids is None
            or str(tool.get("id") or tool.get("name") or "") in visible_tool_ids
        )
    ]
    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    # nodeCatalog (added 2026-05-06 in manyforge.assistant.provider_request.v0)
    # is the rich sibling of `nodes`: each entry has {id, kind,
    # childrenMin, childrenMax, description, parameters[], ...}.
    # Including it in the preamble is the load-bearing route for the
    # LLM to know e.g. that `repeat` is a decorator with
    # childrenMax=1 — which we found the model needs upfront to avoid
    # retrying invalid decorator inserts. Falls back to [] for older
    # clients that haven't been updated.
    node_catalog_raw = payload.get("nodeCatalog")
    node_catalog = node_catalog_raw if isinstance(node_catalog_raw, list) else []
    # programSnapshot + sceneSnapshot (J, 2026-05-06): current state
    # at request-time. The LLM uses these instead of round-tripping
    # through program.read / scene.inspect on every request — saves
    # ~30s/turn under thinking-on. Both None-able when no program /
    # scene is loaded; the openclaw bridge accepts that gracefully.
    program_snapshot_raw = payload.get("programSnapshot")
    program_snapshot = program_snapshot_raw if isinstance(program_snapshot_raw, dict) else None
    scene_snapshot_raw = payload.get("sceneSnapshot")
    scene_snapshot = scene_snapshot_raw if isinstance(scene_snapshot_raw, dict) else None
    skills = payload.get("skills") if isinstance(payload.get("skills"), list) else []
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}

    preamble = {
        "requestId": request_id,
        "assistantMode": assistant_mode,
        "conversationId": payload.get("conversationId"),
        "runtime": runtime,
        "skills": skills,
        "allowedNodes": nodes,
        # Structural hints per node — see comment above. Empty when
        # the assistant-provider envelope predates the field.
        "nodeCatalog": node_catalog,
        # Current-state snapshots (J). None when not provided. The
        # LLM should treat these as the live state at request-time
        # and only call program.read / scene.inspect to refresh
        # after applying mutations within this same request.
        "programSnapshot": program_snapshot,
        "sceneSnapshot": scene_snapshot,
        "allowedTools": tools,
        "visibleMcpTools": (
            list(mcp_allowed_tools) if mcp_allowed_tools is not None else "full mode surface"
        ),
        "contextKeys": sorted(context.keys()),
    }

    return "\n".join(
        [
            "You are the ManyForge composer assistant running inside OpenClaw.",
            "Follow the installed `manyforge-composer` skill.",
            "Use only the `manyforge` MCP server for ManyForge state reads and draft edits.",
            "Do not fabricate program, tree, or scene state.",
            "If the visible ManyForge MCP tools are insufficient, say what is missing instead of inventing a tool.",
            "When finished, briefly report the tool result for the Composer chat transcript.",
            # The preamble below carries the live state of the program "
            # tree (`programSnapshot`) and scene (`sceneSnapshot`) at "
            # request-time — read those directly when you need to find "
            # node names, object ids, positions, etc. You only need to "
            # call `manyforge__program-read` or `manyforge__scene-inspect` "
            # to *refresh* state AFTER you have applied a mutation in "
            # this same request.
            "The `programSnapshot` and `sceneSnapshot` keys in `ManyForge request context` below carry the LIVE current state — read them directly for node names, object ids, dimensions, and poses. Only call `manyforge__program-read` or `manyforge__scene-inspect` to refresh state AFTER you have applied a mutation in this same request.",
            "",
            "ManyForge request context:",
            json.dumps(preamble, indent=2, sort_keys=True),
            "",
            "User request:",
            message.strip(),
        ]
    )


def build_openclaw_command(
    *,
    config: AdapterConfig,
    message: str,
    timeout_s: float,
    session_id: str | None = None,
    mcp_allowed_tools: list[str] | None = None,
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
    if mcp_allowed_tools is not None:
        tool_csv = ",".join(mcp_allowed_tools)
        tool_file = shlex.quote(config.allowed_tools_file)
        shell_command = (
            f"printf %s {shlex.quote(tool_csv)} > {tool_file}; "
            f"trap 'rm -f {tool_file}' EXIT; "
            f"{shell_command}"
        )
    return [
        "docker",
        "exec",
        config.cluster_container,
        "kubectl",
        "exec",
        "-n",
        config.namespace,
        config.sandbox,
        "-c",
        config.container,
        "--",
        "su",
        config.sandbox_user,
        "-c",
        shell_command,
    ]


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
                if isinstance(content, str) and content.strip():
                    message = content.strip()
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


def mcp_allowed_tools_from_payload(payload: dict[str, Any]) -> list[str]:
    """Return the narrowed ManyForge MCP tool surface for one request.

    Composer's UI currently sends a mode but not a request-specific tool list.
    When ``requestedTools`` is present we honor it directly. Otherwise this
    applies a deliberately conservative keyword window for obvious scene/tree
    edits and fails open for broad program-building prompts.
    """

    requested = [
        item.strip()
        for item in payload.get("requestedTools") or []
        if isinstance(item, str) and item.strip()
    ]
    if requested:
        return _ordered_known_tools(payload, requested)

    message = payload.get("message")
    if not isinstance(message, str):
        return []
    text = message.lower()
    all_tools = _tool_ids(payload)
    if not all_tools:
        return []

    broad_terms = (
        "build a program",
        "create a program",
        "make a program",
        "pick and place",
        "isaac",
        "collider",
        "colliders",
        "whole program",
        "full program",
    )
    if any(term in text for term in broad_terms):
        selected = _tools_matching(all_tools, ("catalog.", "program.", "tree.", "scene."))
        return selected if len(selected) < len(all_tools) else []

    runtime_terms = (
        "cycle",
        "runtime",
        "during execution",
        "when running",
        "per cycle",
        "each iteration",
        "after each iteration",
        "at the end",
        "end-of-cycle",
        "behavior tree",
        "bt ",
    )
    add_terms = ("add", "create", "insert", "append", "put")
    remove_terms = ("remove", "delete", "drop")
    update_terms = ("update", "move", "change", "set", "modify", "resize")
    control_flow_terms = ("repeat", "sequence", "fallback", "parallel", "retry", "inverter")
    tree_terms = (
        "tree",
        "node",
        "root",
        "repeat",
        "sequence",
        "fallback",
        "parallel",
        "wrap",
        "child",
        "children",
        "insert",
        "replace",
    )
    scene_terms = (
        "scene",
        "object",
        "collision object",
        "box",
        "sphere",
        "graspable",
        "ground",
        "pose",
        "position",
        "diameter",
        "dimension",
        "dimensions",
    )

    selected: list[str] = []
    is_runtime_intent = any(term in text for term in runtime_terms)
    is_tree_intent = is_runtime_intent or any(term in text for term in tree_terms)
    is_scene_intent = any(term in text for term in scene_terms)

    # Prefer one-purpose BT tools when the wording is specific enough. This is
    # intentionally request-scoped routing, not authorization: Composer still
    # enforces the mode catalog and bounded-autonomy envelope on every call.
    if is_tree_intent:
        if _looks_like_root_wrap(text, control_flow_terms):
            selected.append("tree.draft.wrap_node")
        elif any(term in text for term in ("replace", "subtree")):
            selected.append("tree.draft.replace_subtree")
        elif any(term in text for term in ("reorder", "move node", "move the node")):
            selected.append("tree.draft.move_node")
        elif any(term in text for term in ("parameter", "parameters", "param", "params")):
            selected.append("tree.draft.update_node_params")
        elif (
            "node" in text
            and any(term in text for term in remove_terms)
            and not is_runtime_intent
        ):
            selected.append("tree.draft.delete_node")
        elif is_runtime_intent or any(term in text for term in add_terms):
            selected.append("tree.draft.insert_node")
        else:
            selected.extend(_tools_matching(all_tools, ("tree.", "program.")))

    if not selected and is_scene_intent:
        if any(term in text for term in remove_terms):
            selected.append("scene.draft.remove_objects")
        elif any(term in text for term in update_terms):
            selected.append("scene.draft.update_object")
        elif any(term in text for term in add_terms):
            selected.append("scene.draft.add_object")
        else:
            selected.extend(_tools_matching(all_tools, ("scene.",)))

    if not selected:
        return []
    selected.extend(_helper_tool_ids(all_tools))
    return _dedupe_known(all_tools, selected)


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


def _ordered_known_tools(payload: dict[str, Any], requested: list[str]) -> list[str]:
    return _dedupe_known(_tool_ids(payload), requested)


def _tools_matching(tool_ids: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return [tool_id for tool_id in tool_ids if tool_id.startswith(prefixes)]


def _looks_like_root_wrap(text: str, control_flow_terms: tuple[str, ...]) -> bool:
    """Return true for requests that make a wrapper the new root.

    Small models frequently say "add a repeat node as root and make the
    current root its child" rather than using the exact verb "wrap". Treat
    that as the same operation so the visible MCP surface is the single
    high-level helper instead of the whole tree-edit family.
    """

    if "root" not in text:
        return False
    has_wrapper_kind = any(term in text for term in control_flow_terms)
    if not has_wrapper_kind:
        return False
    return (
        "wrap" in text
        or "current root" in text
        or "new root" in text
        or "as root" in text
        or "its child" in text
        or "as its child" in text
    )


def _helper_tool_ids(tool_ids: list[str]) -> list[str]:
    helpers = (
        "catalog.read",
        "program.read",
        "scene.inspect",
        "deployment.capabilities.read",
        "skills.read",
        "status.read",
        "program.validate",
    )
    known = set(tool_ids)
    return [tool_id for tool_id in helpers if tool_id in known]


def _dedupe_known(known_order: list[str], candidates: list[str]) -> list[str]:
    wanted = {candidate for candidate in candidates if candidate in known_order}
    return [tool_id for tool_id in known_order if tool_id in wanted]


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
