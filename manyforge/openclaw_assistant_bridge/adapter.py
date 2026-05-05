"""Contract translation for the OpenClaw-routed ManyForge assistant adapter.

This module is deliberately free of FastAPI and subprocess imports so the
translation layer can be tested without a live NemoClaw/OpenShell sandbox.
"""
from __future__ import annotations

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


def build_gateway_chat_completions_command(
    *,
    config: AdapterConfig,
    payload: dict[str, Any],
    timeout_s: float,
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

    The persistent gateway eliminates the per-call ~40s CLI bootstrap of
    the openclaw-agent shell-out path; warm calls drop to 5-10s.
    """

    user_message = payload.get("message")
    if not isinstance(user_message, str):
        user_message = json.dumps(user_message, sort_keys=True)
    request_body = {
        "model": f"openclaw/{config.agent}",
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": config.gateway_max_tokens,
        "stream": False,
    }
    body_json = json.dumps(request_body)
    curl_timeout = max(5, int(timeout_s) - 1)
    args = [
        "curl",
        "-sS",
        "--max-time",
        str(curl_timeout),
        "-H",
        "Content-Type: application/json",
        "-X",
        "POST",
        f"http://127.0.0.1:{config.gateway_port}/v1/chat/completions",
        "-d",
        body_json,
    ]
    token = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "").strip()
    if token:
        args = args[:6] + ["-H", f"Authorization: Bearer {token}"] + args[6:]
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
