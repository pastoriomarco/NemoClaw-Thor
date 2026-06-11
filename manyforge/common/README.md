# `manyforge.common` — universal core for the three-lane bridges

This package holds the lane-agnostic code shared by all three assistant
lanes (Direct model, OpenClaw, Hermes Agents).

## What lives here

| Module | Responsibility |
|---|---|
| [`projection.py`](./projection.py) | Program / scene / tree / catalog projection helpers. Translate Composer state snapshots into compact summaries embedded in agent prompts. |
| [`prompt.py`](./prompt.py) | Agent prompt assembly. Preamble + RULES + state block + tail checklist, with a per-lane `discovery_mode` parameter that swaps in the OpenClaw discovery primer (Phase 3) or the Hermes direct-catalog header (Phase 4). |
| [`envelope.py`](./envelope.py) | DTOs and helpers for the `manyforge.assistant.provider_request.v0` envelope. `AdapterConfig`, `AgentRunResult`, `request_id_from_payload`, `error_envelope`, `derive_session_key`, `is_action_shaped_prompt`. |
| [`tool_calls.py`](./tool_calls.py) | Tool-call extraction (`extract_tool_calls`), canonicalization (`canonical_tool_name`), OpenClaw envelope unwrap (`unwrap_openclaw_envelope`, Phase 3), Hermes `mcp_manyforge_` prefix strip (`strip_mcp_prefix`, Phase 4). |
| [`mcp_catalog.py`](./mcp_catalog.py) | Mode-scoped MCP tool catalog helpers. Prompt-keyword tool-window inference is intentionally not part of the common surface. |

## What does NOT live here

- **Transport-specific command builders** (OpenClaw CLI args, Hermes
  session API requests, Direct lane chat-completions) — those belong in
  the per-lane `AssistantTransport` implementation under
  `manyforge/lanes/<lane>/`.
- **Response parsing for a specific provider** — same reasoning.
- **Orchestration policy** (compaction, circuit breaker) — those live in
  [`manyforge.assistant_session`](../assistant_session/).

## Intended consumers

- `openclaw_assistant_bridge/` (NemoClaw-Thor) — imports as `from manyforge.common.projection import build_program_summary`.
- `manyforge_assistant_bridge/` (dev_ws/manyforge) — the direct-lane
  bridge. Phase 2 lands the import path; until then the projection mirror
  there is documented technical debt (adapter.py:183 docstring).
- Future `hermes_assistant_bridge/` (Phase 4).
- `manyforge-mcp-bridge.py` (dev_ws/manyforge/scripts/) — does not
  import this package (it's a standalone JSON-RPC bridge with its own
  shape) but follows the same lane-neutralization discipline.

## Phase 1 status (behavior-preserving)

Every module in this package currently re-exports from
[`openclaw_assistant_bridge/adapter.py`](../openclaw_assistant_bridge/adapter.py).
Functions are not copied; they are imported and re-aliased. This keeps
Phase 1 a zero-runtime-behavior-change refactor: existing call sites in
adapter.py / service.py keep working unchanged; new call sites can
import via this package surface.

Phase 2 and Phase 4 will MOVE implementations here as the Direct and
Hermes lanes are wired up. At that point adapter.py becomes a thin
OpenClaw-specific shim that imports from this package.

## Compatibility note

While the lane-neutralization of `manyforge-mcp-bridge.py` was
behavior-preserving for tool-call semantics, the audit-log row gains 4
columns and the `CONVERSATION_ID`/`request_id` formats become *variable*
(though they resolve to the same string `"openclaw-..."` for OpenClaw
deployments by content, the format is now a variable). Log-parsing
tools that match by *column position* need to know about the new
columns (lane / principal / conversationId / assistantMode).
