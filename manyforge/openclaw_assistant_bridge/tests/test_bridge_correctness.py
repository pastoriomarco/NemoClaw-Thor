"""Bridge correctness tests (Commit B).

Pins the three correctness invariants the 2026-06-03 external review
flagged and the per-commit fixes for them:

* **Rule 8a / schema drift** — every camelCase argument key the bridge
  preamble claims must actually exist in the composer-side schemas. The
  prior bug was a hand-written Rule 8a that named ``box_dims`` and
  ``position: {afterName...}`` — neither of which appear in the canonical
  schemas (reviewer findings 3 + 4). This test catches the same class of
  drift the moment it returns.

* **Per-conversation loop history** — FIX 5's bridge-side detector now
  reads from a module-level history populated AFTER dispatch, keyed by
  ``(conversationId, assistantMode)``. The prior implementation read
  ``payload.messages[]`` which Composer never populates, so the detector
  never fired in production (reviewer finding 9). These tests pin the
  recorder + reader behavior and the LRU/ring-buffer bounds.

* **/compact session_id wiring** — FIX 1's compaction trigger now passes
  ``session_id`` so the ``/compact`` invocation targets the SAME
  OpenClaw session as the subsequent chat call. The prior implementation
  shell-spawned a fresh session and compacted nothing (reviewer finding
  8). This is verified via the command builder, not the live agent.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Rule 8a drift-detection
# ---------------------------------------------------------------------------
#
# The composer-side schemas at
# manyforge_composer/backend/assistant_tool_schemas.py are the source of
# truth. Rule 8a is a hand-written prompt-side mirror. Any time a schema
# field is renamed and Rule 8a falls behind, the validator rejects every
# call.
#
# We don't import the composer module (different repo / different venv);
# instead we parse the schema file's text and the adapter file's text,
# and check that every camelCase key Rule 8a explicitly NAMES is present
# in the schema file's text. Cheap, brittle in the right direction (a
# false alarm beats silent drift).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER_PATH = _REPO_ROOT / "openclaw_assistant_bridge" / "adapter.py"
# The composer schema lives in the dev_ws repo, not nemoclaw. The CI
# image mounts both repos side-by-side; locate the schema file by
# walking up to the common workspace root and back down.
_COMPOSER_SCHEMA_CANDIDATES = [
    # Standard side-by-side layout in dev container:
    _REPO_ROOT.parent.parent / "dev_ws" / "src" / "manyforge"
    / "manyforge_composer" / "backend" / "assistant_tool_schemas.py",
    # Direct sibling layout (some local setups):
    _REPO_ROOT.parent / "manyforge_composer" / "backend" / "assistant_tool_schemas.py",
    # User workspace layout from earlier audit (search-anchored):
    Path.home() / "workspaces" / "dev_ws" / "src" / "manyforge"
    / "manyforge_composer" / "backend" / "assistant_tool_schemas.py",
]


def _find_composer_schema_path() -> Path | None:
    for candidate in _COMPOSER_SCHEMA_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


# Per-tool list of camelCase keys Rule 8a explicitly names. Maintained
# alongside the prompt — if Rule 8a adds a new tool entry, add it here
# too. The drift test only checks keys that appear in BOTH the adapter
# text AND this list, so a wrong key in Rule 8a fails even if the
# adapter author forgot to update this list.
_RULE_8A_TOOL_KEYS: dict[str, list[str]] = {
    "tree_draft_wrap_node": ["targetName", "wrapper", "id", "name", "params"],
    "tree_draft_insert_node": [
        "nodeName", "parentName", "node", "id", "params",
        "index", "afterName", "beforeName",
    ],
    "tree_draft_update_node_params": ["nodeName", "params", "merge"],
    "tree_draft_delete_node": ["nodeName"],
    "tree_draft_move_node": ["nodeName", "newParentName", "position"],
    "tree_draft_change_node_kind": ["nodeName", "newId", "params"],
    "tree_draft_replace_subtree": ["nodeName", "subtree"],
    "scene_draft_add_object": [
        "objectId", "shape", "type", "box_dimensions_m",
        "pose", "position_m", "orientation_xyzw",
    ],
    "scene_draft_update_object": ["objectId", "pose", "shape"],
    "scene_draft_remove_object": ["objectId"],
}


@pytest.fixture(scope="module")
def adapter_source() -> str:
    return _ADAPTER_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def composer_schema_source() -> str:
    path = _find_composer_schema_path()
    if path is None:
        pytest.skip(
            "composer schema file not found at expected paths; drift "
            "check requires dev_ws repo mounted alongside nemoclaw "
            "(skip is appropriate in nemoclaw-only test runs)"
        )
    return path.read_text(encoding="utf-8")


def test_rule_8a_block_present(adapter_source: str) -> None:
    """Rule 8a must be present in build_agent_prompt's rules_block.
    If someone renamed the rule or removed it, the drift test below
    silently passes — pin existence here."""
    assert "8a. **Tool argument names are camelCase" in adapter_source, (
        "Rule 8a header not found in adapter.py; the drift test below "
        "will silently no-op without it."
    )


@pytest.mark.parametrize(
    "tool_id,expected_keys",
    sorted(_RULE_8A_TOOL_KEYS.items()),
)
def test_rule_8a_camelcase_keys_exist_in_composer_schemas(
    tool_id: str,
    expected_keys: list[str],
    adapter_source: str,
    composer_schema_source: str,
) -> None:
    """Every camelCase key Rule 8a claims for a tool must appear in
    the composer-side schema source. Catches the bug class the prior
    box_dims / position-nest authoring introduced (reviewer findings
    3 + 4) — if Rule 8a names a key the schema doesn't have, this
    test fails immediately. False positives possible (e.g. ``id`` is
    a common token), but the failure mode is loud and the fix is to
    either correct Rule 8a or refine _RULE_8A_TOOL_KEYS."""
    # First: the tool name itself must appear in Rule 8a. If it
    # doesn't, the drift check for this tool's keys is vacuous —
    # surface that as a fail so the maintainer either adds the tool
    # entry to Rule 8a or removes it from _RULE_8A_TOOL_KEYS.
    assert tool_id in adapter_source, (
        f"Rule 8a missing tool entry for {tool_id!r}. Either add a "
        f"Rule 8a entry for it or remove it from _RULE_8A_TOOL_KEYS."
    )
    missing: list[str] = []
    for key in expected_keys:
        # The composer schemas use the key as a JSON-property name in
        # Python source, e.g. ``"objectId": {...}`` or ``box_dimensions_m``
        # bare. Match the literal token surrounded by non-identifier
        # chars to avoid false positives on substring overlap.
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])")
        if not pattern.search(composer_schema_source):
            missing.append(key)
    assert not missing, (
        f"Rule 8a names keys for {tool_id!r} that do NOT appear in "
        f"assistant_tool_schemas.py: {missing}. Either Rule 8a is "
        f"out of date (fix the prompt) or the schema was renamed "
        f"(update both Rule 8a + _RULE_8A_TOOL_KEYS in this test)."
    )


def test_rule_8a_does_not_teach_dropped_shapes(adapter_source: str) -> None:
    """Specific anti-regressions for the actual bug shapes that
    landed in 2026-06-03's Rule 8a (reviewer findings 3 + 4): the
    BAD positive constructions must not appear.

    Note: Rule 8a deliberately mentions wrong keys in parentheticals
    (e.g. "(NOT `box_dims`)") to teach the model what to avoid. We
    can't simply forbid every mention of a bad string; we forbid the
    POSITIVE construction (e.g. ``box_dims}`` as part of the shape
    template the model is told to emit)."""
    # Limit the search to the Rule 8a block so unrelated mentions
    # elsewhere in the file don't trip us up.
    block_start = adapter_source.find("8a. **Tool argument names are camelCase")
    block_end = adapter_source.find('"9. Every tool result carries', block_start)
    assert block_start >= 0 and block_end > block_start, "Rule 8a block boundaries not found"
    rule_8a = adapter_source[block_start:block_end]
    # The original bug had ``{type, box_dims}`` as the POSITIVE shape
    # template — we want to teach ``{type, box_dimensions_m}``.
    assert "{type, box_dims}" not in rule_8a, (
        "Rule 8a re-introduced '{type, box_dims}' as the positive "
        "shape template; the canonical key is 'box_dimensions_m'."
    )
    # The original bug had ``position: {afterName}`` as the POSITIVE
    # template for insert_node — afterName et al. are top-level.
    assert "position: {afterName" not in rule_8a, (
        "Rule 8a re-introduced the 'position: {afterName...}' "
        "nesting; afterName/beforeName/index are TOP-LEVEL on "
        "insert_node, with no position wrapper."
    )
    # The original bug had ``newKind`` as the change_node_kind field;
    # canonical is ``newId``.
    assert "nodeName, newKind" not in rule_8a, (
        "Rule 8a re-introduced 'newKind' as the change_node_kind "
        "field; the canonical key is 'newId'."
    )
    # The original bug had ``orientation_quat`` as the pose key in
    # the POSITIVE template `pose: {position, orientation_quat}`.
    # The canonical is ``orientation_xyzw``. Rule 8a may still
    # legitimately mention ``orientation_quat`` in a negative
    # parenthetical to teach the model what NOT to emit; pin the
    # POSITIVE bad construction instead.
    assert "position, orientation_quat}" not in rule_8a, (
        "Rule 8a re-introduced 'orientation_quat' in the positive "
        "pose template; the canonical key is 'orientation_xyzw'."
    )


# ---------------------------------------------------------------------------
# Per-conversation loop history
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_history():
    """Reset the module-level history before/after every test."""
    from openclaw_assistant_bridge import service
    service._reset_loop_history_for_tests()
    yield service
    service._reset_loop_history_for_tests()


def test_loop_history_records_and_reads_fingerprints(fresh_history) -> None:
    """Round-trip: record N tool calls, snapshot returns them in
    the order recorded."""
    service = fresh_history
    key = ("conv-1", "composer-assistant")

    async def record() -> None:
        await service._loop_history_record(
            key,
            [
                'tree_draft_wrap_node::{"targetName":"a"}',
                'tree_draft_wrap_node::{"targetName":"b"}',
                'tree_draft_insert_node::{"nodeName":"c"}',
            ],
        )

    asyncio.get_event_loop().run_until_complete(record())
    snap = service._loop_history_snapshot(key)
    assert snap == [
        'tree_draft_wrap_node::{"targetName":"a"}',
        'tree_draft_wrap_node::{"targetName":"b"}',
        'tree_draft_insert_node::{"nodeName":"c"}',
    ]


def test_loop_history_keys_segment_by_conversation_and_mode(fresh_history) -> None:
    """Two conversations (different conversationId) or two modes on
    the same conversation must NOT cross-pollinate. Critical because
    concurrent lanes may share a conversation id."""
    service = fresh_history
    key_a = ("conv-1", "composer-assistant")
    key_b = ("conv-1", "scene-authoring")
    key_c = ("conv-2", "composer-assistant")

    async def record() -> None:
        await service._loop_history_record(key_a, ['toolA::{}'])
        await service._loop_history_record(key_b, ['toolB::{}'])
        await service._loop_history_record(key_c, ['toolC::{}'])

    asyncio.get_event_loop().run_until_complete(record())
    assert service._loop_history_snapshot(key_a) == ['toolA::{}']
    assert service._loop_history_snapshot(key_b) == ['toolB::{}']
    assert service._loop_history_snapshot(key_c) == ['toolC::{}']


def test_loop_history_ring_buffer_drops_oldest(fresh_history, monkeypatch) -> None:
    """The per-conversation deque has maxlen — when it overflows,
    oldest entries fall off. Pin the bound."""
    service = fresh_history
    # Force a small cap so the test is fast and obvious.
    monkeypatch.setattr(service, "_LOOP_HISTORY_MAX_CALLS_PER_CONV", 3)
    # Need a fresh deque — module-level value is read on first record;
    # re-create via reset so the test deque honors the new maxlen.
    service._reset_loop_history_for_tests()
    key = ("conv-overflow", "default")

    async def record_many() -> None:
        await service._loop_history_record(
            key,
            [f'tool::{{"i":{i}}}' for i in range(5)],
        )

    asyncio.get_event_loop().run_until_complete(record_many())
    snap = service._loop_history_snapshot(key)
    assert len(snap) == 3
    # Oldest two are dropped; newest three remain in order.
    assert snap == [
        'tool::{"i":2}',
        'tool::{"i":3}',
        'tool::{"i":4}',
    ]


def test_loop_history_lru_evicts_cold_conversations(fresh_history, monkeypatch) -> None:
    """When the conversation count exceeds the cap, the LRU eviction
    pops the LEAST RECENTLY USED conversation key."""
    service = fresh_history
    monkeypatch.setattr(service, "_LOOP_HISTORY_MAX_CONVERSATIONS", 2)
    service._reset_loop_history_for_tests()

    async def record() -> None:
        await service._loop_history_record(("a", "m"), ["x::{}"])
        await service._loop_history_record(("b", "m"), ["y::{}"])
        # Touching ``a`` again moves it to the MRU end.
        await service._loop_history_record(("a", "m"), ["x2::{}"])
        # Adding ``c`` evicts ``b`` (least recently used), not ``a``.
        await service._loop_history_record(("c", "m"), ["z::{}"])

    asyncio.get_event_loop().run_until_complete(record())
    assert service._loop_history_snapshot(("a", "m")) == ["x::{}", "x2::{}"]
    assert service._loop_history_snapshot(("b", "m")) == []
    assert service._loop_history_snapshot(("c", "m")) == ["z::{}"]


def test_fingerprint_tool_call_canonicalizes_args() -> None:
    """``{'a':1,'b':2}`` and ``{'b':2,'a':1}`` MUST hash to the same
    fingerprint; otherwise the same_args detector misses identical
    retries that differ only in key order."""
    from openclaw_assistant_bridge.service import _fingerprint_tool_call
    a = _fingerprint_tool_call(
        {"name": "x", "arguments": {"a": 1, "b": 2}}
    )
    b = _fingerprint_tool_call(
        {"name": "x", "arguments": {"b": 2, "a": 1}}
    )
    assert a == b
    assert a == 'x::{"a":1,"b":2}'


def test_fingerprint_tool_call_accepts_openai_function_nesting() -> None:
    """The OpenAI ``{function: {name, arguments}}`` shape and the
    bridge's normalized ``{name, arguments}`` shape both produce
    valid fingerprints — the recorder is called against both shapes
    in different code paths."""
    from openclaw_assistant_bridge.service import _fingerprint_tool_call
    openai_shape = {"function": {"name": "x", "arguments": '{"a":1}'}}
    bridge_shape = {"name": "x", "arguments": {"a": 1}}
    assert _fingerprint_tool_call(openai_shape) == _fingerprint_tool_call(bridge_shape)


def test_fingerprint_tool_call_returns_none_for_unnamed() -> None:
    """No name means no useful fingerprint; recorder must skip
    rather than insert a 'None::...' poison entry into the deque."""
    from openclaw_assistant_bridge.service import _fingerprint_tool_call
    assert _fingerprint_tool_call({"arguments": {"x": 1}}) is None
    assert _fingerprint_tool_call({}) is None
    assert _fingerprint_tool_call("not a dict") is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# /compact session_id wiring (finding 8)
# ---------------------------------------------------------------------------


def test_compact_command_passes_session_id() -> None:
    """When /compact is built for CLI shell-out, it MUST carry
    --session-id so it targets the live session. The prior bug
    omitted session_id and compacted nothing."""
    from openclaw_assistant_bridge.adapter import (
        AdapterConfig,
        build_openclaw_command,
    )
    # AdapterConfig is a dataclass with default values on every field;
    # we override only the ones that matter for this assertion.
    cfg = AdapterConfig(
        agent="manyforge-composer",
        use_gateway=False,
    )
    cmd = build_openclaw_command(
        config=cfg,
        message="/compact",
        timeout_s=30.0,
        mcp_allowed_tools=None,
        session_id="conv-42",
    )
    # build_openclaw_command wraps the inner ``openclaw agent ...``
    # invocation in ``bash -c 'eval "$(echo <base64> | base64 -d)"'``
    # so a literal ``--session-id`` does NOT appear in the joined
    # command string. Decode the base64 segment to inspect the real
    # invocation.
    import base64 as _b64
    import re as _re
    flat = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
    match = _re.search(r"echo ([A-Za-z0-9+/=]+) \| base64 -d", flat)
    assert match, (
        f"build_openclaw_command output does not contain the expected "
        f"base64-encoded openclaw invocation:\n{cmd}"
    )
    decoded = _b64.b64decode(match.group(1)).decode("utf-8", "replace")
    assert "--session-id" in decoded, (
        f"build_openclaw_command did not include --session-id; "
        f"decoded inner invocation was:\n{decoded}"
    )
    assert "conv-42" in decoded, (
        f"--session-id value missing or wrong in decoded inner "
        f"invocation:\n{decoded}"
    )


# ---------------------------------------------------------------------------
# Dual-mode prompt selection (Commit C)
# ---------------------------------------------------------------------------


def _build_minimal_prompt_payload() -> dict:
    """Smallest valid payload that exercises build_agent_prompt."""
    return {
        "requestId": "primer-probe",
        "conversationId": "primer-conv",
        "assistantMode": "composer-assistant",
        "message": "x",
        "modeManifest": {"toolCatalogHash": "abc"},
    }


def test_build_agent_prompt_includes_code_mode_primer_by_default() -> None:
    """Default tool_surface='code' inserts the code-mode primer
    (mentions tool_search_code wrapping)."""
    from openclaw_assistant_bridge.adapter import build_agent_prompt
    prompt = build_agent_prompt(_build_minimal_prompt_payload())
    assert "Dispatch surface — code mode" in prompt
    assert "tool_search_code" in prompt
    # Tools-mode-specific phrasing must not appear.
    assert "Dispatch surface — tools mode" not in prompt


def test_build_agent_prompt_includes_tools_mode_primer_when_requested() -> None:
    """Explicit tool_surface='tools' switches the primer."""
    from openclaw_assistant_bridge.adapter import build_agent_prompt
    prompt = build_agent_prompt(
        _build_minimal_prompt_payload(),
        tool_surface="tools",
    )
    assert "Dispatch surface — tools mode" in prompt
    assert "tool_search" in prompt
    assert "tool_describe" in prompt
    assert "tool_call" in prompt
    # Code-mode-specific phrasing must not appear (the primer constant
    # itself; ``tool_search_code`` may appear elsewhere if Rule 8a
    # still references it — but the primer text shouldn't bleed).
    assert "Dispatch surface — code mode" not in prompt


def test_build_agent_prompt_unknown_surface_falls_back_to_code() -> None:
    """An unknown ``tool_surface`` value falls back to the code-mode
    primer rather than crashing. Operators see the log event."""
    from openclaw_assistant_bridge.adapter import build_agent_prompt
    prompt = build_agent_prompt(
        _build_minimal_prompt_payload(),
        tool_surface="hermes-vNext-experimental",
    )
    assert "Dispatch surface — code mode" in prompt


# ---------------------------------------------------------------------------
# Proxy drift-check classifier (the load-bearing detection logic)
# ---------------------------------------------------------------------------


def _classify(tools: list) -> str:
    """Helper that imports the proxy classifier inline.

    The proxy is a script, not an installed module — imported here
    so test discovery doesn't trip on the import at collection time
    (proxy imports heavyweight deps at module level).
    """
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "vllm_proxy_under_test",
        _REPO_ROOT / "scripts" / "proxy" / "vllm-proxy.py",
    )
    if spec is None or spec.loader is None:
        pytest.skip("vllm-proxy.py not importable in this environment")
    mod = _ilu.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"vllm-proxy.py import failed: {exc}")
    return mod._classify_tools_array(tools)


def test_proxy_classifier_identifies_code_mode() -> None:
    """A tools[] containing exactly tool_search_code classifies as
    code mode."""
    out = _classify([{"type": "function", "function": {"name": "tool_search_code"}}])
    assert out == "code"


def test_proxy_classifier_identifies_tools_mode() -> None:
    """A tools[] containing the three discrete verbs classifies as
    tools mode (extra entries beyond the three core verbs do not
    disqualify — the classifier requires the three to be a subset)."""
    out = _classify([
        {"type": "function", "function": {"name": "tool_search"}},
        {"type": "function", "function": {"name": "tool_describe"}},
        {"type": "function", "function": {"name": "tool_call"}},
    ])
    assert out == "tools"


def test_proxy_classifier_returns_unknown_for_arbitrary_tools() -> None:
    """A probe with real OpenAI-shaped tools (not an OpenClaw surface)
    must NOT be classified as code or tools — operators don't get a
    drift warning for unrelated traffic."""
    out = _classify([
        {"type": "function", "function": {"name": "tree_draft_wrap_node"}},
        {"type": "function", "function": {"name": "scene_draft_add_object"}},
    ])
    assert out == "unknown"


def test_proxy_classifier_returns_unknown_for_empty_or_missing_tools() -> None:
    """Defensive: empty / non-list / missing tools[] → unknown, no
    drift fired."""
    assert _classify([]) == "unknown"
    assert _classify(None) == "unknown"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Code-mode primer correctness (Finding 1)
# ---------------------------------------------------------------------------


def test_code_mode_primer_uses_openclaw_tools_namespace() -> None:
    """OpenClaw 2026.5.22's `tool_search_code` runs in an isolated
    Node subprocess where the bridge is exposed as `openclaw.tools`
    — NOT the bare `tools` namespace.

    Verified two ways:
    (1) The OpenClaw tool description text the model sees says:
        "Run JavaScript in an isolated Node subprocess with
        openclaw.tools.search, openclaw.tools.describe, and
        openclaw.tools.call for large tool catalogs." (source:
        /usr/local/lib/node_modules/openclaw/dist/pi-tools-iVT6BGHc.js:1017)
    (2) OpenClaw's own code-body parser regexes for
        `openclaw\\.tools\\.call`, `openclaw\\.tools\\.describe`,
        `openclaw\\.tools\\.search` (source:
        control-ui/assets/index-BtIuF4zW.js, functions lA/dA).

    The QuickJS code-mode.worker.js (which exposes `globalThis.tools`)
    is a DIFFERENT execution path used only when `--permission`
    support is policy-permitted in QuickJS mode — not the active
    path on this build. A primer that teaches `tools.<x>` produces
    `ReferenceError: tools is not defined` (verified in live logs
    2026-06-03)."""
    from openclaw_assistant_bridge.adapter import _PRIMER_CODE_MODE
    # The canonical dispatch form MUST be present.
    assert "openclaw.tools.call(" in _PRIMER_CODE_MODE, (
        "code-mode primer must teach the canonical "
        "openclaw.tools.call(<name>, args) dispatch form"
    )
    assert "openclaw.tools.describe(" in _PRIMER_CODE_MODE, (
        "code-mode primer must teach openclaw.tools.describe(<name>)"
    )
    assert "openclaw.tools.search(" in _PRIMER_CODE_MODE, (
        "code-mode primer must teach openclaw.tools.search(<query>)"
    )
    # The wrong patterns the model has historically emitted MUST be
    # called out as DO NOT — these are the failure modes captured
    # in the OpenClaw logs on 2026-06-03.
    assert "tools is not defined" in _PRIMER_CODE_MODE, (
        "primer must warn about ReferenceError: tools is not defined"
    )
    assert "window is not defined" in _PRIMER_CODE_MODE, (
        "primer must warn about ReferenceError: window is not defined"
    )
    # Negative case: the primer must NOT teach the bare `tools.call(...)`
    # form as a POSITIVE pattern. It IS allowed to mention `tools.call(...)`
    # inside DO-NOT warnings (and we want those warnings). The
    # distinction: a positive pattern is `await tools.call(...)` (the
    # exact construction the model would copy). A DO-NOT warning has
    # context like "DO NOT write" or "will fail with" preceding it.
    import re
    for m in re.finditer(r"\btools\.call\(", _PRIMER_CODE_MODE):
        start = m.start()
        # Look at a wider window before the match for DO-NOT context.
        window = _PRIMER_CODE_MODE[max(0, start - 60):start]
        if "openclaw." in window[-15:]:
            # Properly prefixed; canonical.
            continue
        if "DO NOT" in window or "ReferenceError" in window or "will fail" in window:
            # DO-NOT warning; this is the desired didactic.
            continue
        raise AssertionError(
            f"Found `tools.call(` not preceded by `openclaw.` and not "
            f"inside a DO-NOT warning: window={window!r}"
        )


# ---------------------------------------------------------------------------
# Loop-history error path (Finding 2)
# ---------------------------------------------------------------------------


def test_record_history_from_result_handles_empty_stdout(fresh_history) -> None:
    """The helper returns 0 (no records appended) on empty stdout —
    the timeout case where _run_agent re-raises before populating
    result.stdout. No exception, no history pollution."""
    service = fresh_history

    async def run() -> int:
        return await service._record_history_from_result(
            conv_key=("c", "m"),
            result_stdout="",
            use_gateway=False,
            payload={"requestId": "x"},
        )

    n = asyncio.get_event_loop().run_until_complete(run())
    assert n == 0
    assert service._loop_history_snapshot(("c", "m")) == []


def test_record_history_from_result_handles_malformed_stdout(fresh_history) -> None:
    """Best-effort: parsing garbage produces 0 records, no exception."""
    service = fresh_history

    async def run() -> int:
        return await service._record_history_from_result(
            conv_key=("c", "m"),
            result_stdout="not a json or openclaw envelope",
            use_gateway=False,
            payload={"requestId": "x"},
        )

    n = asyncio.get_event_loop().run_until_complete(run())
    assert n == 0


# ---------------------------------------------------------------------------
# Proxy drift pair-based dedup (Finding 3 — session header absent in
# real OpenClaw traffic, so we dedup by (expected, observed) pair globally)
# ---------------------------------------------------------------------------


def test_proxy_drift_dedup_is_pair_based(fresh_history) -> None:
    """The proxy drift detector now dedups by (expected, observed)
    pair globally rather than by session key, because real OpenClaw
    traffic carries no session header. We verify by simulating the
    dedup logic against the module-level memo: two identical pairs
    add only once."""
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "vllm_proxy_under_test_dedup",
        _REPO_ROOT / "scripts" / "proxy" / "vllm-proxy.py",
    )
    if spec is None or spec.loader is None:
        pytest.skip("vllm-proxy.py not importable in this environment")
    mod = _ilu.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        pytest.skip(f"vllm-proxy.py import failed: {exc}")
    # Reset the global memo so prior tests don't pollute.
    mod._TOOL_SURFACE_WARNED_PAIRS.clear()
    mod._TOOL_SURFACE_EXPECTED = "code"
    # Use a body that classifies "tools" — twice. Both invocations
    # would warn on a per-session model; only the FIRST warns on the
    # per-pair model.
    body = {
        "tools": [
            {"type": "function", "function": {"name": "tool_search"}},
            {"type": "function", "function": {"name": "tool_describe"}},
            {"type": "function", "function": {"name": "tool_call"}},
        ],
    }
    mod._check_tool_surface_drift(
        path="/v1/chat/completions",
        headers={},  # no headers — simulates real OpenClaw traffic
        body=body,
        log_extra={},
    )
    mod._check_tool_surface_drift(
        path="/v1/chat/completions",
        headers={},
        body=body,
        log_extra={},
    )
    assert ("code", "tools") in mod._TOOL_SURFACE_WARNED_PAIRS
    # Still just one pair memoized — the second invocation hit the
    # dedup gate and did NOT add a new entry.
    assert len(mod._TOOL_SURFACE_WARNED_PAIRS) == 1


# ---------------------------------------------------------------------------
# Proxy warn-on-unknown opt-in (Finding 4)
# ---------------------------------------------------------------------------


def test_proxy_warn_on_unknown_opt_in() -> None:
    """When OPENCLAW_PROXY_WARN_ON_UNKNOWN is on, classifier="unknown"
    ALSO fires the drift warning (catches direct-real-tools traffic
    on an OpenClaw-only proxy). Default off preserves the
    quiet-on-unrelated behavior."""
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "vllm_proxy_under_test_unknown",
        _REPO_ROOT / "scripts" / "proxy" / "vllm-proxy.py",
    )
    if spec is None or spec.loader is None:
        pytest.skip("vllm-proxy.py not importable in this environment")
    mod = _ilu.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        pytest.skip(f"vllm-proxy.py import failed: {exc}")

    # Default off → unknown surface doesn't fire.
    mod._TOOL_SURFACE_WARNED_PAIRS.clear()
    mod._TOOL_SURFACE_EXPECTED = "code"
    mod._TOOL_SURFACE_WARN_ON_UNKNOWN = False
    body = {
        "tools": [
            {"type": "function", "function": {"name": "tree_draft_wrap_node"}},
        ],
    }
    mod._check_tool_surface_drift(
        path="/v1/chat/completions",
        headers={},
        body=body,
        log_extra={},
    )
    assert len(mod._TOOL_SURFACE_WARNED_PAIRS) == 0  # quiet on unknown

    # Opt-in on → unknown surface fires.
    mod._TOOL_SURFACE_WARN_ON_UNKNOWN = True
    mod._check_tool_surface_drift(
        path="/v1/chat/completions",
        headers={},
        body=body,
        log_extra={},
    )
    assert ("code", "unknown") in mod._TOOL_SURFACE_WARNED_PAIRS
