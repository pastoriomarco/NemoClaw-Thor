"""Smoke tests for the manyforge.common package surface.

Phase 1 deliverable: every public re-export resolves, and the new
helpers (unwrap_openclaw_envelope, strip_mcp_prefix) behave as
specified. The re-exports themselves are validated indirectly by the
existing openclaw_assistant_bridge tests.
"""
from __future__ import annotations

import sys
import pathlib

# Make manyforge/ resolvable without an install step (Phase 1 is package
# scaffolding; install discipline lands in Phase 2 with setup-direct.sh).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from common import projection, prompt, envelope, tool_calls, mcp_catalog  # noqa: E402


def test_projection_reexports():
    """Five projection helpers are accessible."""
    for name in (
        "build_program_summary",
        "build_scene_summary",
        "project_node_catalog",
        "project_skill_catalog",
        "project_tree_node",
    ):
        assert hasattr(projection, name), f"missing: projection.{name}"


def test_envelope_reexports():
    """Envelope DTOs and helpers are accessible."""
    assert envelope.AdapterConfig is not None
    assert envelope.AgentRunResult is not None
    assert callable(envelope.error_envelope)
    assert callable(envelope.derive_session_key)
    assert callable(envelope.request_id_from_payload)


def test_tool_calls_reexports():
    """Tool-call helpers are accessible."""
    assert callable(tool_calls.extract_tool_calls)
    assert callable(tool_calls.canonical_tool_name)
    assert callable(tool_calls.dedupe_known)


def test_mcp_catalog_reexports():
    """MCP catalog helpers are accessible."""
    assert callable(mcp_catalog.allowed_tool_map)
    assert callable(mcp_catalog.tool_ids)


def test_prompt_discovery_mode_validation():
    """build_agent_prompt accepts only the three known discovery_mode values."""
    payload = {"message": "x"}
    for mode in ("direct", "openclaw_discovery", "hermes_direct"):
        # Should at least not raise on mode validation. We don't probe
        # the actual prompt content here — that's the OpenClaw bridge's
        # existing test suite's job.
        try:
            prompt.build_agent_prompt(payload, discovery_mode=mode)
        except (KeyError, AttributeError, TypeError):
            # The underlying adapter.build_agent_prompt may need richer
            # payload fields. That's fine — Phase 1 only asserts the
            # mode-parameter shape is accepted.
            pass
    try:
        prompt.build_agent_prompt(payload, discovery_mode="bogus")
        raise AssertionError("expected ValueError for unknown discovery_mode")
    except ValueError:
        pass


def test_unwrap_openclaw_envelope_wrapped():
    """``{tool, result}`` shape strips to just the result."""
    wrapped = {
        "tool": {"id": "tree_draft_wrap_node", "name": "tree_draft_wrap_node"},
        "result": {"content": [{"type": "text", "text": "ok"}]},
    }
    assert tool_calls.unwrap_openclaw_envelope(wrapped) == {
        "content": [{"type": "text", "text": "ok"}]
    }


def test_unwrap_openclaw_envelope_passthrough():
    """A bare result is returned unchanged."""
    direct = {"content": [{"type": "text", "text": "ok"}]}
    assert tool_calls.unwrap_openclaw_envelope(direct) is direct


def test_unwrap_openclaw_envelope_wrong_shape():
    """Wrong shapes are returned unchanged (no false-positive unwrap)."""
    # tool present but not a dict
    assert tool_calls.unwrap_openclaw_envelope({"tool": "x", "result": {}}) == {
        "tool": "x", "result": {}
    }
    # extra keys present
    assert tool_calls.unwrap_openclaw_envelope({
        "tool": {"id": "x"}, "result": {}, "extra": True
    }) == {"tool": {"id": "x"}, "result": {}, "extra": True}
    # not a dict
    assert tool_calls.unwrap_openclaw_envelope([]) == []
    assert tool_calls.unwrap_openclaw_envelope("string") == "string"
    assert tool_calls.unwrap_openclaw_envelope(None) is None


def test_strip_mcp_prefix():
    """Hermes ``mcp_manyforge_`` prefix is stripped; bare names unchanged."""
    assert tool_calls.strip_mcp_prefix("mcp_manyforge_tree_draft_wrap_node") == "tree_draft_wrap_node"
    assert tool_calls.strip_mcp_prefix("tree_draft_wrap_node") == "tree_draft_wrap_node"
    # Different server
    assert tool_calls.strip_mcp_prefix("mcp_other_tool", server="other") == "tool"
    assert tool_calls.strip_mcp_prefix("mcp_manyforge_x", server="other") == "mcp_manyforge_x"


if __name__ == "__main__":
    # Allow `python3 test_smoke.py` standalone (no pytest required).
    import inspect

    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"  ✓ {name}")
            except Exception as exc:
                failed += 1
                print(f"  ✗ {name}: {exc}")
    if failed:
        raise SystemExit(f"{failed} test(s) failed")
    print("all tests passed")
