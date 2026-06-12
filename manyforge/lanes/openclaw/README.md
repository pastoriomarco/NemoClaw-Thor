# `manyforge/lanes/openclaw/`

Lane-specific artifacts for the OpenClaw assistant lane.

## Contents

| File | Purpose |
|---|---|
| [`skill_addendum.md`](./skill_addendum.md) | Discovery-protocol primer appended to the lane-agnostic skill body. Teaches the model how to use OpenClaw 2026.5.6+'s native `tool_search` / `tool_describe` / `tool_call` compaction surface efficiently. |
| [`policy.yaml`](./policy.yaml) | `SessionPolicy` config for this lane (compaction, discovery_mode). Defaults reflect the current native OpenClaw route. |

## Phase 3 status — GATE PASSED (on the strong models)

Phase 3's gate (**≥46/66 ≈70%** on a clean discovery-surface run) was
cleared on the stronger models: gemma-QAT 52/66 (2026-06-07 sweep) and
51/66 in the 2026-06-09 three-lane head-to-head; qwen3.6-35b 51/66. The
historical anchor cosmos-reason2-8b scored below gate (39/66, caveated —
see [PHASE-3-OPENCLAW-NATIVE-RESULT.md](../../docs/PHASE-3-OPENCLAW-NATIVE-RESULT.md)
and [PHASE-5-PRODUCTION-DECISION.md](../../docs/PHASE-5-PRODUCTION-DECISION.md)),
and the clean-start model default moved to gemma-QAT.

The archived plugin artifacts (in
`manyforge/archive/openclaw-plugin-attempt-2026-06-02/`) are retained as a
rollback path until the (ad-interim) Phase 5 decision is finalized.

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
  [`assistant_provider.py`](https://github.com/pastoriomarco/manyforge/blob/main/manyforge_composer/backend/assistant_provider.py)
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

## Operational & related docs

- **Bring-up + live-monitoring (operational):**
  [`manyforge/docs/operations/LANE_BRINGUP.md`](https://github.com/pastoriomarco/manyforge/blob/main/docs/operations/LANE_BRINGUP.md)
  — the `openclaw` section (bridge on `:8200`; revives the gateway in the
  `my-assistant` sandbox; health `curl http://127.0.0.1:8200/healthz`).
- **Phase 3 result:** [`../../docs/PHASE-3-OPENCLAW-NATIVE-RESULT.md`](../../docs/PHASE-3-OPENCLAW-NATIVE-RESULT.md).
- **Benchmarks + scorer analysis:** [`../../docs/LANE-COMPARISON.md`](../../docs/LANE-COMPARISON.md).
