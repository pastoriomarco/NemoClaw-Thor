from __future__ import annotations

import json

from openclaw_assistant_bridge.adapter import (
    AdapterConfig,
    build_agent_prompt,
    build_gateway_chat_completions_command,
    build_openclaw_command,
    derive_gateway_session_key,
    mcp_allowed_tools_from_payload,
    normalize_agent_response,
    normalize_chat_completions_response,
    parse_openclaw_json,
)


def _payload() -> dict:
    return {
        "requestId": "req-1",
        "conversationId": "conv-1",
        "assistantMode": "composer-assistant",
        "message": "Read the program.",
        "tools": [
            {"id": "program.read", "effect": "read_only", "description": "Read program"},
            {
                "id": "tree.draft.wrap_node",
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
            {"id": "catalog.read", "effect": "read_only", "description": "Read catalog"},
            {"id": "program.read", "effect": "read_only", "description": "Read program"},
            {"id": "scene.inspect", "effect": "read_only", "description": "Inspect scene"},
            {
                "id": "tree.draft.wrap_node",
                "effect": "composer_draft_mutating",
                "description": "Wrap a BT node",
            },
            {
                "id": "tree.draft.insert_node",
                "effect": "composer_draft_mutating",
                "description": "Insert a BT node",
            },
            {
                "id": "tree.draft.update_node_params",
                "effect": "composer_draft_mutating",
                "description": "Update BT node params",
            },
            {
                "id": "tree.draft.move_node",
                "effect": "composer_draft_mutating",
                "description": "Move a BT node",
            },
            {
                "id": "tree.draft.replace_subtree",
                "effect": "composer_draft_mutating",
                "description": "Replace a BT subtree",
            },
            {
                "id": "tree.draft.delete_node",
                "effect": "composer_draft_mutating",
                "description": "Delete a BT node",
            },
            {
                "id": "scene.draft.add_object",
                "effect": "composer_draft_mutating",
                "description": "Add scene object",
            },
            {
                "id": "scene.draft.update_object",
                "effect": "composer_draft_mutating",
                "description": "Update scene object",
            },
            {
                "id": "scene.draft.remove_objects",
                "effect": "composer_draft_mutating",
                "description": "Remove scene objects",
            },
        ],
    }


def test_build_prompt_includes_mode_tools_and_user_message() -> None:
    prompt = build_agent_prompt(_payload())
    assert "manyforge-composer" in prompt
    assert "composer-assistant" in prompt
    assert "program.read" in prompt
    assert "Read the program." in prompt


def test_build_prompt_filters_tool_descriptions_to_visible_window() -> None:
    prompt = build_agent_prompt(
        _rich_payload("add a box"),
        mcp_allowed_tools=["scene.draft.add_object", "catalog.read"],
    )
    assert "scene.draft.add_object" in prompt
    assert "catalog.read" in prompt
    assert "tree.draft.wrap_node" not in prompt
    assert "scene.draft.remove_objects" not in prompt


def test_build_command_targets_sandbox_openclaw_agent() -> None:
    command = build_openclaw_command(
        config=AdapterConfig(sandbox="my-assistant", agent="main"),
        message="hello there",
        timeout_s=42,
    )
    assert command[:3] == ["docker", "exec", "openshell-cluster-nemoclaw"]
    assert "my-assistant" in command
    assert "openclaw agent --thinking off --agent main" in command[-1]
    assert "--message 'hello there'" in command[-1]
    assert "--timeout 42" in command[-1]


def test_build_command_places_local_flag_before_agent_selector() -> None:
    command = build_openclaw_command(
        config=AdapterConfig(local=True),
        message="hello",
        timeout_s=10,
    )
    assert "openclaw agent --local --thinking off --agent main" in command[-1]


def test_build_command_can_override_thinking_level() -> None:
    command = build_openclaw_command(
        config=AdapterConfig(thinking="minimal"),
        message="hello",
        timeout_s=10,
    )
    assert "openclaw agent --thinking minimal --agent main" in command[-1]


def test_build_command_can_set_session_id() -> None:
    command = build_openclaw_command(
        config=AdapterConfig(),
        message="hello",
        timeout_s=10,
        session_id="conversation-1",
    )
    assert "--session-id conversation-1" in command[-1]


def test_build_command_can_scope_manyforge_mcp_tools() -> None:
    command = build_openclaw_command(
        config=AdapterConfig(),
        message="hello",
        timeout_s=10,
        mcp_allowed_tools=["tree.draft.wrap_node", "catalog.read"],
    )
    assert (
        "tree.draft.wrap_node,catalog.read > "
        "/tmp/manyforge-openclaw-allowed-tools.txt"
    ) in command[-1]
    assert "trap 'rm -f /tmp/manyforge-openclaw-allowed-tools.txt' EXIT" in command[-1]
    assert "openclaw agent" in command[-1]


def test_mcp_allowed_tools_uses_requested_tools_when_present() -> None:
    payload = {
        **_payload(),
        "requestedTools": ["tree.draft.wrap_node", "not.in.catalog"],
    }
    assert mcp_allowed_tools_from_payload(payload) == ["tree.draft.wrap_node"]


def test_mcp_allowed_tools_infers_tree_window_for_node_prompt() -> None:
    payload = {**_payload(), "message": "add a repeat node as the new root"}
    assert mcp_allowed_tools_from_payload(payload) == [
        "program.read",
        "tree.draft.wrap_node",
    ]


def test_mcp_allowed_tools_keeps_root_wrap_window_narrow() -> None:
    payload = _rich_payload("add a repeat node as root node, and make the current root node its child")
    assert mcp_allowed_tools_from_payload(payload) == [
        "catalog.read",
        "program.read",
        "scene.inspect",
        "tree.draft.wrap_node",
    ]


def test_mcp_allowed_tools_routes_runtime_object_reset_to_tree_insert() -> None:
    payload = _rich_payload("at the end of the cycle, remove graspable and re-add it")
    assert mcp_allowed_tools_from_payload(payload) == [
        "catalog.read",
        "program.read",
        "scene.inspect",
        "tree.draft.insert_node",
    ]


def test_mcp_allowed_tools_routes_compile_time_scene_add_to_scene_add() -> None:
    payload = _rich_payload("add a box of size 1.0, 0.02, 0.2 at position 0, -0.2, 0.1")
    assert mcp_allowed_tools_from_payload(payload) == [
        "catalog.read",
        "program.read",
        "scene.inspect",
        "scene.draft.add_object",
    ]


def test_mcp_allowed_tools_fails_open_for_plain_query() -> None:
    assert mcp_allowed_tools_from_payload(_payload()) == []


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
            "toolSummary": {"tools": ["manyforge__program.read"]},
        },
        stdout="{}",
    )
    assert response["message"] == "Program root is pick_and_place."
    assert response["toolCalls"] == [
        {
            "name": "program.read",
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
                    "name": "manyforge__tree.draft.wrap_node",
                    "status": "success",
                    "arguments": {"targetName": "@root"},
                }
            ],
        },
        stdout="{}",
    )
    assert response["toolCalls"][0]["name"] == "tree.draft.wrap_node"
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


# -----------------------------------------------------------------------
# Gateway-lane: session isolation, revision guards, observability filter
# -----------------------------------------------------------------------


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
