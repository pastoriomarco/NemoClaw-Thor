# `manyforge/lanes/openclaw/`

Lane-specific artifacts for the OpenClaw assistant lane.

## Contents

| File | Purpose |
|---|---|
| [`skill_addendum.md`](./skill_addendum.md) | Discovery-protocol primer appended to the lane-agnostic skill body. Teaches the model how to use OpenClaw 2026.5.6+'s native `tool_search` / `tool_describe` / `tool_call` compaction surface efficiently. |
| [`policy.yaml`](./policy.yaml) | `SessionPolicy` config for this lane (compaction, discovery_mode). Defaults reflect the current native OpenClaw route. |

## Phase 3 status

Phase 3 of the THREE-LANE-MIGRATION-PLAN is the empirical validation
that this skill addendum closes the gap between the discovery-surface
overhead and the iter-32 baseline. Gate is **≥46/66 (≈70%)** on a clean
discovery-surface run on cosmos-reason2-8b.

If the gate passes, the archived plugin artifacts (in
`manyforge/archive/openclaw-plugin-attempt-2026-06-02/`) can be deleted
in Phase 5. If it doesn't, the archived artifacts remain as a
feature-flagged rollback path (`OPENCLAW_LANE_MODE=plugin|native`)
until a future cycle revisits the plugin path.

## Architecture

The OpenClaw lane runs the model agent loop inside the OpenClaw gateway
container (in-sandbox), not in the bridge. The bridge's job is:

1. Receive the `manyforge.assistant.provider_request.v0` envelope from
   Composer.
2. Build the agent prompt via `manyforge.common.prompt.build_agent_prompt(
   payload, discovery_mode="openclaw_discovery")` — appends THIS lane's
   `skill_addendum.md` to the lane-agnostic skill body.
3. POST it to the OpenClaw gateway at `:18789/v1/chat/completions` with
   model id `openclaw/manyforge-composer`.
4. Apply the lane's `SessionPolicy` (compaction every 2) and leave
   clarification/intent decisions to the model.
5. Parse the response and audit the discovery turns vs the real-tool
   turns separately (so the bake-off can show the round-trip overhead
   honestly).

## What's NOT here

- Provider registration: the `LANE_REGISTRY` in
  [`assistant_provider.py`](/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_composer/backend/assistant_provider.py)
  carries the `openclaw` entry.
- Sandbox onboarding: the `setup-manyforge-assistant.sh` script
  bootstraps the OpenClaw sandbox (skill, MCP server, agent profile,
  policy presets).
- The actual transport implementation: lives in
  [`openclaw_assistant_bridge/`](../../openclaw_assistant_bridge/) and
  will become an `AssistantTransport` Protocol implementation in a
  future cycle.
- The retired plugin attempt: archived at
  [`manyforge/archive/openclaw-plugin-attempt-2026-06-02/`](../../archive/openclaw-plugin-attempt-2026-06-02/).
  Why we pivoted from the plugin to the discovery primer (the bundled
  `nemoclaw` plugin owns `registerProvider({id:"inference"})` and OpenClaw
  exposes no `extendProvider`): see
  [THREE-LANE-MIGRATION-PLAN.md](../../docs/THREE-LANE-MIGRATION-PLAN.md)
  §3 (line 46) and §10 (line 500).
