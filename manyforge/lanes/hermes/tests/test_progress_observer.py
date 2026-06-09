"""Unit tests for the Hermes progress observer (events → universal audit)."""
from __future__ import annotations

from lanes.hermes.progress_observer import (
    KIND_MEMORY_WRITE,
    KIND_SKILL_CREATED,
    KIND_TOOL_CALL,
    HermesProgressObserver,
)


def _obs(events):
    return HermesProgressObserver().observe(events, conversation_id="conv-1")


def test_tool_call_prefix_is_stripped_for_audit_parity():
    out = _obs([{"event": "tool_call", "data": {"tool": "mcp_manyforge_tree_draft_wrap_node"}}])
    assert out.tools_observed == ["tree_draft_wrap_node"]
    assert out.session_events[0]["kind"] == KIND_TOOL_CALL
    assert out.session_events[0]["tool"] == "tree_draft_wrap_node"
    assert out.session_events[0]["raw_name"] == "mcp_manyforge_tree_draft_wrap_node"


def test_tool_name_from_nested_dict():
    out = _obs([{"event": "tool.invoked", "data": {"tool": {"name": "mcp_manyforge_scene_inspect"}}}])
    assert out.tools_observed == ["scene_inspect"]


def test_distinctive_hermes_counters():
    out = _obs(
        [
            {"event": "memory.write", "data": {"summary": "user prefers retry-3"}},
            {"event": "skill.created", "data": {"name": "wrap_in_retry"}},
            {"event": "cron.fired", "data": {}},
            {"event": "delegation.spawn", "data": {"child": "subagent-1"}},
        ]
    )
    assert (out.memory_writes, out.skill_creations, out.cron_fires, out.delegations) == (1, 1, 1, 1)
    kinds = [e["kind"] for e in out.session_events]
    assert KIND_MEMORY_WRITE in kinds and KIND_SKILL_CREATED in kinds
    skill_rec = next(e for e in out.session_events if e["kind"] == KIND_SKILL_CREATED)
    assert skill_rec["skill"] == "wrap_in_retry"


def test_unknown_event_types_are_dropped_best_effort():
    out = _obs([{"event": "some.unmapped.event", "data": {"x": 1}}, {"event": "", "data": {}}])
    assert out.tools_observed == []
    assert out.session_events == []
    assert out.raw_event_count == 2  # counted, but nothing classified


def test_malformed_events_do_not_crash():
    out = _obs([{"event": "tool_call", "data": "not-a-dict"}, "totally-bogus", {"event": "memory"}])
    # tool_call with non-dict data yields no tool name → no observed tool;
    # bare string is skipped; memory event with no data still counts.
    assert out.tools_observed == []
    assert out.memory_writes == 1


def test_multiple_tool_calls_preserve_order():
    out = _obs(
        [
            {"event": "tool_call", "data": {"tool": "mcp_manyforge_program_read"}},
            {"event": "tool_call", "data": {"tool": "mcp_manyforge_scene_draft_add_object"}},
        ]
    )
    assert out.tools_observed == ["program_read", "scene_draft_add_object"]
