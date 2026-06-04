"""Adapter unit tests.

The bridge has two transports, selected by the `OPENCLAW_ASSISTANT_USE_GATEWAY`
env var (`AdapterConfig.use_gateway`):

  HOT PATH — gateway HTTP (production default):
    build_agent_prompt(payload)
      → build_gateway_chat_completions_command()
      → curl POST to OpenClaw's chat-completions endpoint
      → parse_chat_completions_response()
      → normalize_chat_completions_response()
    Plus the structured projections that build_agent_prompt calls into:
      _build_program_summary, _build_scene_summary,
      _project_node_catalog, _project_skill_catalog.

  COLD PATH — CLI shell-out (fallback transport):
    build_agent_prompt(payload)
      → build_openclaw_command() (docker exec … kubectl exec … openclaw agent)
      → parse_openclaw_json() (extract structured JSON from stdout)
      → normalize_agent_response()

Tests are grouped below. The hot-path block is the one that has to stay
correct for production OpenClaw + Cosmos-Reason2-8B traffic; the cold-path
block guards the alternate transport so it remains usable.
"""

from __future__ import annotations

import json

from openclaw_assistant_bridge.adapter import (
    AdapterConfig,
    build_agent_prompt,
    build_gateway_chat_completions_command,
    build_openclaw_command,
    derive_gateway_session_key,
    normalize_agent_response,
    normalize_chat_completions_response,
    parse_chat_completions_response,
    parse_openclaw_json,
)


def _payload() -> dict:
    return {
        "requestId": "req-1",
        "conversationId": "conv-1",
        "assistantMode": "composer-assistant",
        "message": "Read the program.",
        "tools": [
            {"id": "program_read", "effect": "read_only", "description": "Read program"},
            {
                "id": "tree_draft_wrap_node",
                "effect": "composer_draft_mutating",
                "description": "Wrap a BT node",
            },
        ],
        "nodes": ["sequence", "repeat"],
        "skills": ["manyforge-composer"],
        "runtime": {"programLoaded": True, "cycleState": "idle"},
        "context": {"source": "test"},
    }


def _rich_payload(message: str) -> dict:
    return {
        **_payload(),
        "message": message,
        "tools": [
            {"id": "catalog_read", "effect": "read_only", "description": "Read catalog"},
            {"id": "program_read", "effect": "read_only", "description": "Read program"},
            {"id": "scene_inspect", "effect": "read_only", "description": "Inspect scene"},
            {
                "id": "tree_draft_wrap_node",
                "effect": "composer_draft_mutating",
                "description": "Wrap a BT node",
            },
            {
                "id": "tree_draft_insert_node",
                "effect": "composer_draft_mutating",
                "description": "Insert a BT node",
            },
            {
                "id": "tree_draft_update_node_params",
                "effect": "composer_draft_mutating",
                "description": "Update BT node params",
            },
            {
                "id": "tree_draft_move_node",
                "effect": "composer_draft_mutating",
                "description": "Move a BT node",
            },
            {
                "id": "tree_draft_replace_subtree",
                "effect": "composer_draft_mutating",
                "description": "Replace a BT subtree",
            },
            {
                "id": "tree_draft_delete_node",
                "effect": "composer_draft_mutating",
                "description": "Delete a BT node",
            },
            {
                "id": "scene_draft_add_object",
                "effect": "composer_draft_mutating",
                "description": "Add scene object",
            },
            {
                "id": "scene_draft_update_object",
                "effect": "composer_draft_mutating",
                "description": "Update scene object",
            },
            {
                "id": "scene_draft_remove_objects",
                "effect": "composer_draft_mutating",
                "description": "Remove scene objects",
            },
        ],
    }


# ============================================================================
# HOT PATH — gateway transport (production)
# ============================================================================


def test_build_prompt_includes_mode_tools_and_user_message() -> None:
    prompt = build_agent_prompt(_payload())
    assert "manyforge-composer" in prompt
    assert "composer-assistant" in prompt
    # `allowedTools` id list is restored (load-bearing per smoke
    # 2026-05-08); the test_payload's tool ids should appear.
    assert "program_read" in prompt
    # User's raw message is forwarded verbatim under "## user_request".
    assert "Read the program." in prompt


def test_build_prompt_exposes_full_mode_tool_catalog() -> None:
    # Prompt-derived request windows were removed. The id list remains useful
    # model context, but it is now the full assistant-mode catalog supplied by
    # Composer, not a per-prompt subset.
    prompt = build_agent_prompt(_rich_payload("add a box"))
    assert "scene_draft_add_object" in prompt
    assert "catalog_read" in prompt
    assert "scene_draft_remove_objects" in prompt
    # `allowedNodes` and `allowedSkills` were dropped (no regression
    # observed in smoke; ids are derivable from the catalogs).
    assert "allowedNodes" not in prompt
    assert "allowedSkills" not in prompt


def _rich_payload_with_snapshots(message: str = "add a box of size 1 0.02 0.5") -> dict:
    """Payload that includes programSnapshot + sceneSnapshot, like production.

    Composer always sends both snapshots on assistant requests once a
    program is loaded. The structured projections in `build_agent_prompt`
    operate on this shape; tests that omit snapshots miss those code paths.
    """
    payload = _rich_payload(message)
    payload["programSnapshot"] = {
        "programTreeHash": "abc123",
        "program": {
            "name": "pick_and_place_demo",
            "description": "A demo program",
            "tree": {
                "name": "pick_and_place",
                "kind": "sequence",
                "params": {},
                "children": [
                    {"name": "approach", "kind": "move_to_pose", "params": {}, "children": []},
                    {"name": "graspable_pickup", "kind": "pick_object", "params": {}, "children": []},
                ],
            },
            "parameters": [{"name": "speed", "type": "float"}],
            "blackboard_keys": ["target_pose"],
        },
    }
    payload["sceneSnapshot"] = {
        "robot": {"id": "ur10e", "links": ["base_link", "tool0"]},
        "objects": [
            {
                "id": "graspable",
                "shape": {"type": "box", "box_dims": [0.05, 0.05, 0.05]},
                "pose": {"position": [0.0, 0.0, 0.0]},
            },
            {
                "id": "ground",
                "shape": {"type": "box", "box_dims": [2.0, 2.0, 0.01]},
                "pose": {"position": [0.0, 0.0, 0.0]},
            },
        ],
    }
    payload["nodeCatalog"] = [
        {"id": "sequence", "kind": "sequence",
         "description": "Run children in order",
         "parameters": [{"name": "memory", "type": "bool"}]},
        {"id": "upsert_collision_object", "kind": "upsert_collision_object",
         "description": "Add or replace a collision object at runtime",
         "parameters": [
             {"name": "object_id", "type": "string"},
             {"name": "pose", "type": "Pose"},
         ]},
    ]
    payload["skillCatalog"] = [
        {"id": "manyforge-composer", "description": "Composer authoring skill"},
    ]
    return payload


def test_build_prompt_keeps_complete_node_index_and_object_ids_index() -> None:
    """Structured projections must keep ALL names reachable as indexes.

    The detail window is bounded first-N, but the index lists are
    unconditional — the model must always be able to reference any
    node-name or object-id by string in a later tool call.
    """
    prompt = build_agent_prompt(_rich_payload_with_snapshots())
    assert "stateContext" in prompt
    assert '"programSnapshot"' not in prompt
    assert '"sceneSnapshot"' not in prompt
    # All program tree node names appear at least somewhere in the index.
    assert "pick_and_place" in prompt
    assert "approach" in prompt
    assert "graspable_pickup" in prompt
    # All scene object ids appear.
    assert "graspable" in prompt
    assert "ground" in prompt


def test_build_prompt_can_skip_state_context_by_cadence() -> None:
    """The service can skip the compact state capsule by cadence without
    removing the request metadata or the raw user request."""
    prompt = build_agent_prompt(
        _rich_payload_with_snapshots("update graspable pose to (1.0, 0.0, 0.0)"),
        state_context_control={
            "mode": "every_n",
            "everyN": 3,
            "sequence": 2,
            "include": False,
            "reason": "cadence_skip",
            "lastIncludedSequence": 1,
        },
    )
    assert '"stateContext"' in prompt
    assert '"included": false' in prompt
    assert '"reason": "cadence_skip"' in prompt
    assert '"program": {' not in prompt
    assert '"scene": {' not in prompt
    assert "update graspable pose" in prompt


def test_build_prompt_node_catalog_uses_projected_shape() -> None:
    """The nodeCatalog projection drops per-node JSON-Schema parameters[]
    in favor of id+kind+description (saves ~28 KB envelope). Param schemas
    live in the OpenAI tools[] array on the same chat-completion request."""
    prompt = build_agent_prompt(_rich_payload_with_snapshots())
    # The `kind` and `description` fields are kept.
    assert "upsert_collision_object" in prompt
    assert "Add or replace a collision object" in prompt
    # The full per-node parameters[] schemas are NOT emitted into the
    # nodeCatalog block. (`object_id` may still appear in RULES rule 1
    # which mentions params.object_id literally — so check for the
    # JSON-Schema-shaped key that only appears in raw param schemas.)
    assert '"type": "Pose"' not in prompt


def test_parse_chat_completions_response_handles_pure_json() -> None:
    """Curl returns a single JSON document on success."""
    body = json.dumps({
        "id": "chatcmpl-abc",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    })
    parsed, warnings = parse_chat_completions_response(body)
    assert parsed is not None
    assert parsed["choices"][0]["message"]["content"] == "ok"
    assert warnings == []


def test_parse_chat_completions_response_extracts_from_noisy_output() -> None:
    """Curl can prepend warning lines (e.g. SSL deprecation) before JSON.
    The parser must locate the first balanced JSON object."""
    body = (
        "Warning: NSS deprecation\n"
        "* using HTTP/1.1\n"
        + json.dumps({"choices": [{"message": {"content": "ok"}}]})
        + "\n"
    )
    parsed, warnings = parse_chat_completions_response(body)
    assert parsed is not None
    assert parsed["choices"][0]["message"]["content"] == "ok"


def test_parse_chat_completions_response_returns_none_for_empty() -> None:
    parsed, warnings = parse_chat_completions_response("")
    assert parsed is None
    assert any("empty" in w for w in warnings)


def test_parse_chat_completions_response_returns_none_for_garbage() -> None:
    parsed, warnings = parse_chat_completions_response("not json at all")
    assert parsed is None
    assert any("could not parse" in w for w in warnings)


def test_normalize_chat_completions_response_extracts_assistant_message() -> None:
    """Happy path: pull `choices[0].message.content` into the envelope's
    `message` field. toolCalls/proposals are empty because the gateway
    runs the tool loop internally — Composer's state delta is the
    authoritative mutation signal."""
    response = normalize_chat_completions_response(
        payload=_payload(),
        response_json={
            "choices": [{"message": {"role": "assistant",
                                     "content": "Program root is pick_and_place."}}],
        },
        stdout="ignored",
    )
    assert response["message"] == "Program root is pick_and_place."
    assert response["toolCalls"] == []
    assert response["proposals"] == []
    assert response["draftMutated"] is False
    assert response["requiresReview"] is True


def test_normalize_chat_completions_response_flattens_content_blocks() -> None:
    """Some chat templates emit `content` as an array of typed blocks
    (e.g. `[{"type": "text", "text": "..."}]`) rather than a string.
    `_flatten_chat_content` must collapse that to plain text."""
    response = normalize_chat_completions_response(
        payload=_payload(),
        response_json={
            "choices": [{"message": {"role": "assistant", "content": [
                {"type": "text", "text": "Wrapped the root."},
            ]}}],
        },
        stdout="ignored",
    )
    assert response["message"] == "Wrapped the root."


def test_normalize_chat_completions_response_handles_error_object() -> None:
    """vLLM/OpenClaw can return `{"error": {"type": "...", "message": "..."}}`
    instead of choices on failure (e.g. context_length_exceeded). The
    normalizer must surface the error as a structured warning + message."""
    response = normalize_chat_completions_response(
        payload=_payload(),
        response_json={"error": {"type": "context_length_exceeded",
                                 "message": "prompt too large"}},
        stdout="ignored",
    )
    assert "prompt too large" in response["message"]
    assert response["error"]["code"] == "context_length_exceeded"
    assert any("gateway error" in w for w in response["warnings"])


def test_normalize_chat_completions_response_handles_empty_response() -> None:
    """When parse returns None (empty/garbage stdout), normalize must
    still produce a valid envelope with a fallback message + warning."""
    response = normalize_chat_completions_response(
        payload=_payload(),
        response_json=None,
        stdout="",
        parse_warnings=["gateway returned empty body"],
    )
    assert response["requiresReview"] is True
    assert "no parseable response" in response["message"]
    assert any("empty body" in w for w in response["warnings"])


def test_normalize_chat_completions_response_filters_observability_leakage() -> None:
    """If the model echoes an OpenClaw scheduler diagnostic (e.g.
    'stuck session', 'lane task error'), the message is replaced with a
    neutral fallback and a structured warning is surfaced."""
    response = normalize_chat_completions_response(
        payload=_payload(),
        response_json={
            "choices": [{"message": {"role": "assistant",
                                     "content": "I detected a stuck session in your request."}}],
        },
        stdout="ignored",
    )
    assert "stuck session" not in response["message"]
    assert any("openclaw_observability_leakage" in w for w in response["warnings"])


# ============================================================================
# COLD PATH — CLI shell-out transport (alternate, only when use_gateway=false)
# ============================================================================


# build_openclaw_command wraps the inner ``openclaw agent ...``
# invocation in ``bash -c 'eval "$(echo <base64> | base64 -d)"'`` so
# literal flags do NOT appear in command[-1] directly. The base64
# wrap was added to make the encoded shell invocation safe against
# arg-list reshaping by intermediate transports (kubectl/nemoclaw exec).
# These tests decode the inner invocation before asserting (reviewer
# finding 7 — tests pre-dated the wrap).
import base64 as _b64
import re as _re


def _decoded_inner_invocation(command: list[str]) -> str:
    """Return the base64-decoded inner ``openclaw agent ...`` line.

    Raises if the command does not contain the expected wrap, so the
    caller's assertion error message includes the bad command shape.
    """
    flat = command[-1] if isinstance(command, list) else str(command)
    match = _re.search(r"echo ([A-Za-z0-9+/=]+) \| base64 -d", flat)
    assert match, (
        f"build_openclaw_command output does not contain the expected "
        f"base64-encoded openclaw invocation in command[-1]:\n{flat}"
    )
    return _b64.b64decode(match.group(1)).decode("utf-8", "replace")


def test_build_command_targets_sandbox_openclaw_agent() -> None:
    command = build_openclaw_command(
        config=AdapterConfig(sandbox="my-assistant", agent="main"),
        message="hello there",
        timeout_s=42,
    )
    # The OUTER command is now the ``nemoclaw <sandbox> exec`` wrapper
    # (replaced ``docker exec openshell-cluster-nemoclaw`` for parity
    # with the OpenShell CLI). Sandbox is the second token; ``exec``
    # is the third.
    assert command[0] == "nemoclaw"
    assert "my-assistant" in command
    assert "exec" in command
    inner = _decoded_inner_invocation(command)
    assert "openclaw agent --thinking off --agent main" in inner
    assert "--message 'hello there'" in inner or "--message hello" in inner
    assert "--timeout 42" in inner


def test_build_command_places_local_flag_before_agent_selector() -> None:
    command = build_openclaw_command(
        config=AdapterConfig(local=True),
        message="hello",
        timeout_s=10,
    )
    inner = _decoded_inner_invocation(command)
    assert "openclaw agent --local --thinking off --agent main" in inner


def test_build_command_can_override_thinking_level() -> None:
    command = build_openclaw_command(
        config=AdapterConfig(thinking="minimal"),
        message="hello",
        timeout_s=10,
    )
    inner = _decoded_inner_invocation(command)
    assert "openclaw agent --thinking minimal --agent main" in inner


def test_build_command_can_set_session_id() -> None:
    command = build_openclaw_command(
        config=AdapterConfig(),
        message="hello",
        timeout_s=10,
        session_id="conversation-1",
    )
    inner = _decoded_inner_invocation(command)
    assert "--session-id conversation-1" in inner


def test_build_command_does_not_write_request_scoped_tool_window() -> None:
    command = build_openclaw_command(
        config=AdapterConfig(),
        message="hello",
        timeout_s=10,
    )
    inner = _decoded_inner_invocation(command)
    assert "manyforge-openclaw-allowed-tools.txt" not in inner
    assert "trap 'rm -f" not in inner
    assert "openclaw agent" in inner


def test_parse_openclaw_json_accepts_logged_json_line() -> None:
    data, warnings = parse_openclaw_json(
        "starting\n"
        + json.dumps({"finalAssistantVisibleText": "done", "toolSummary": {"tools": []}})
        + "\n"
    )
    assert warnings == []
    assert data["finalAssistantVisibleText"] == "done"


def test_parse_openclaw_json_accepts_logged_pretty_json() -> None:
    data, warnings = parse_openclaw_json(
        "Config warnings:\n"
        "[bundle-mcp] registered tool\n"
        + json.dumps(
            {
                "payloads": [{"text": "done"}],
                "meta": {"completion": {"stopReason": "stop"}},
            },
            indent=2,
        )
        + "\n"
    )
    assert warnings == []
    assert data["payloads"][0]["text"] == "done"


def test_normalize_maps_openclaw_namespaced_tool_calls() -> None:
    response = normalize_agent_response(
        payload=_payload(),
        agent_json={
            "finalAssistantVisibleText": "Program root is pick_and_place.",
            "toolSummary": {"tools": ["manyforge__program_read"]},
        },
        stdout="{}",
    )
    assert response["message"] == "Program root is pick_and_place."
    assert response["toolCalls"] == [
        {
            "name": "program_read",
            "status": "completed",
            "arguments": {},
            "result": {},
        }
    ]
    assert response["draftMutated"] is False


def test_normalize_extracts_openclaw_payload_text() -> None:
    response = normalize_agent_response(
        payload=_payload(),
        agent_json={
            "payloads": [{"text": "Program root is pick_and_place."}],
            "meta": {"durationMs": 123},
        },
        stdout="{}",
    )
    assert response["message"] == "Program root is pick_and_place."


def test_normalize_ignores_non_call_tool_metadata() -> None:
    response = normalize_agent_response(
        payload=_payload(),
        agent_json={
            "payloads": [{"text": "Wrapped."}],
            "meta": {
                "systemPromptReport": {
                    "tools": {
                        "listChars": 0,
                        "schemaChars": 123,
                        "entries": [{"name": "manyforge__tree-draft-wrap_node"}],
                        "calls": 0,
                    }
                }
            },
        },
        stdout="{}",
    )
    assert response["message"] == "Wrapped."
    assert response["toolCalls"] == []


def test_normalize_marks_draft_mutated_for_successful_draft_tool() -> None:
    response = normalize_agent_response(
        payload=_payload(),
        agent_json={
            "message": "Wrapped the root.",
            "toolCalls": [
                {
                    "name": "manyforge__tree_draft_wrap_node",
                    "status": "success",
                    "arguments": {"targetName": "@root"},
                }
            ],
        },
        stdout="{}",
    )
    assert response["toolCalls"][0]["name"] == "tree_draft_wrap_node"
    assert response["toolCalls"][0]["status"] == "completed"
    assert response["draftMutated"] is True


def test_text_fallback_when_openclaw_json_missing() -> None:
    response = normalize_agent_response(
        payload=_payload(),
        agent_json=None,
        stdout="plain assistant answer",
        parse_warnings=["Could not parse JSON"],
    )
    assert response["message"] == "plain assistant answer"
    assert "Could not parse JSON" in response["warnings"]


# ============================================================================
# HOT PATH (continued) — session keys, gateway command, observability filter
# ============================================================================
# Session-key derivation, gateway HTTP command construction, and the
# observability leakage filter all run on the gateway transport. They live
# at the bottom of the file because they post-date the other HOT PATH
# tests; logically they belong together with them.


def test_session_key_includes_conversation_id() -> None:
    """Two requests with the same conversationId should hash to the same key."""
    a = derive_gateway_session_key({"conversationId": "conv-A", "context": {}})
    b = derive_gateway_session_key({"conversationId": "conv-A", "context": {}})
    assert a == b
    assert a.startswith("manyforge-conv-A-")


def test_session_key_isolates_distinct_conversations() -> None:
    """Different conversationIds must produce different keys (the
    primary session-isolation property)."""
    a = derive_gateway_session_key({"conversationId": "conv-A", "context": {}})
    b = derive_gateway_session_key({"conversationId": "conv-B", "context": {}})
    assert a != b


def test_session_key_rotates_on_catalog_hash_change() -> None:
    """Same conversation, different catalogHash -> different session key.
    A deployment reload mid-stream must invalidate the cached session
    so OpenClaw doesn't reuse stale tool-list context."""
    base = {"conversationId": "conv-A", "context": {"catalogHash": "abc"}}
    drift = {"conversationId": "conv-A", "context": {"catalogHash": "xyz"}}
    assert derive_gateway_session_key(base) != derive_gateway_session_key(drift)


def test_session_key_rotates_on_program_revision_change() -> None:
    """Same conversation, different programRevision -> different session.
    A draft adopt mid-conversation invalidates 'the tree's root is X'
    style cached context."""
    base = {"conversationId": "conv-A", "context": {"programRevision": "r1"}}
    drift = {"conversationId": "conv-A", "context": {"programRevision": "r2"}}
    assert derive_gateway_session_key(base) != derive_gateway_session_key(drift)


def test_session_key_rotates_on_deployment_id_change() -> None:
    base = {"conversationId": "conv-A", "context": {"deploymentId": "d1"}}
    drift = {"conversationId": "conv-A", "context": {"deploymentId": "d2"}}
    assert derive_gateway_session_key(base) != derive_gateway_session_key(drift)


def test_session_key_falls_back_to_request_id_when_no_conversation() -> None:
    """If Composer doesn't pass a conversationId, we still get
    one-session-per-request isolation rather than zero isolation."""
    a = derive_gateway_session_key({"requestId": "req-1"})
    b = derive_gateway_session_key({"requestId": "req-2"})
    assert a != b


def test_gateway_command_stamps_session_key_header() -> None:
    """The host-side curl must carry the x-openclaw-session-key header
    so the gateway scheduler isolates this conversation's session."""
    cfg = AdapterConfig(use_gateway=True, agent="manyforge-composer")
    cmd = build_gateway_chat_completions_command(
        config=cfg,
        payload={
            "conversationId": "conv-test",
            "context": {"catalogHash": "cafebabe"},
            "message": "hello",
        },
        timeout_s=60,
    )
    # Curl args contain the header twice-flag-pair: -H "x-openclaw-session-key: ..."
    joined = " ".join(cmd)
    assert "x-openclaw-session-key:" in joined
    # And the value is the derived session key (deterministic for the input)
    expected_key = derive_gateway_session_key(
        {"conversationId": "conv-test", "context": {"catalogHash": "cafebabe"}}
    )
    assert expected_key in joined


def test_gateway_command_session_key_changes_when_context_drifts() -> None:
    """End-to-end: a drift in catalogHash must change the header value."""
    cfg = AdapterConfig(use_gateway=True)
    cmd_a = build_gateway_chat_completions_command(
        config=cfg,
        payload={"conversationId": "conv-X", "context": {"catalogHash": "h1"}, "message": "x"},
        timeout_s=60,
    )
    cmd_b = build_gateway_chat_completions_command(
        config=cfg,
        payload={"conversationId": "conv-X", "context": {"catalogHash": "h2"}, "message": "x"},
        timeout_s=60,
    )
    assert " ".join(cmd_a) != " ".join(cmd_b)


def test_observability_leakage_filter_replaces_stuck_session_phrase() -> None:
    """The 'which session you're referring to' response we observed live
    must be rewritten to a neutral fallback and surfaced as a warning."""
    response = normalize_chat_completions_response(
        payload={"requestId": "req-leak", "conversationId": "conv-leak"},
        response_json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "I'm not sure which session you're referring to. "
                            "Could you provide the session key (or name) "
                            "you'd like me to check?"
                        ),
                    },
                    "finish_reason": "stop",
                }
            ]
        },
        stdout="",
    )
    # Original leaky message must NOT be passed through to the user.
    assert "session key" not in response["message"].lower()
    # A structured warning MUST be present so reviewers see what happened.
    leakage_warnings = [w for w in response["warnings"] if "openclaw_observability_leakage" in w]
    assert leakage_warnings, "expected an openclaw_observability_leakage warning"


def test_observability_leakage_filter_lets_normal_answers_through() -> None:
    """A real assistant answer must NOT be filtered (no false positives)."""
    response = normalize_chat_completions_response(
        payload={"requestId": "req-ok"},
        response_json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The current scene contains one box and one ground plane.",
                    },
                    "finish_reason": "stop",
                }
            ]
        },
        stdout="",
    )
    assert "current scene contains" in response["message"]
    assert not any("openclaw_observability_leakage" in w for w in response["warnings"])


def test_observability_leakage_filter_catches_session_or_scene_clarification() -> None:
    """The 2026-05-05 observed second-attempt response variant — model
    asking 'Could you clarify which session or scene you're asking
    about?' — must also be caught. The first version of the filter was
    too narrow and missed this wording."""
    response = normalize_chat_completions_response(
        payload={"requestId": "req-leak-2", "conversationId": "conv-leak"},
        response_json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Could you clarify which session or scene you're asking about?",
                    },
                    "finish_reason": "stop",
                }
            ]
        },
        stdout="",
    )
    assert "session" not in response["message"].lower() or response["message"].startswith("I couldn't")
    assert any("openclaw_observability_leakage" in w for w in response["warnings"])


def test_observability_leakage_filter_catches_session_key_for_session() -> None:
    """The first-attempt wording from the same screenshot —
    'I'm not sure which scene you're referring to. Could you provide
    more details or the session key for the session you'd like
    information about?' — was also missed by the narrow patterns."""
    response = normalize_chat_completions_response(
        payload={"requestId": "req-leak-3"},
        response_json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "I'm not sure which scene you're referring to. Could you "
                            "provide more details or the session key for the session "
                            "you'd like information about?"
                        ),
                    },
                    "finish_reason": "stop",
                }
            ]
        },
        stdout="",
    )
    assert any("openclaw_observability_leakage" in w for w in response["warnings"])


def test_observability_leakage_filter_catches_lane_task_error() -> None:
    """Defense-in-depth: if a 'lane task error' string from the gateway
    log ever ends up in model output, filter that too."""
    response = normalize_chat_completions_response(
        payload={"requestId": "req-lte"},
        response_json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "lane task error: lane=main durationMs=49593",
                    },
                    "finish_reason": "stop",
                }
            ]
        },
        stdout="",
    )
    assert "lane task error" not in response["message"].lower()
    assert any("openclaw_observability_leakage" in w for w in response["warnings"])
