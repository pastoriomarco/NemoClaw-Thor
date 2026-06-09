# Phase 1 — Specs Audit

Per [THREE-LANE-MIGRATION-PLAN.md §8 Phase 1 Specs Check](../THREE-LANE-MIGRATION-PLAN.md), every modification to manyforge code must consult `manyforge_specs/` before landing. This document records every spec section consulted during the Phase 1 refactor and whether it was respected, amended, or extended.

## Files modified in Phase 1

### Inside `dev_ws/manyforge/`

1. [`scripts/manyforge-mcp-bridge.py`](/home/tndlux/workspaces/dev_ws/src/manyforge/scripts/manyforge-mcp-bridge.py) — lane-neutralized.
2. [`manyforge_composer/backend/assistant_provider.py`](/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_composer/backend/assistant_provider.py) — added `LANE_REGISTRY` + reject-unknown-id behavior.
3. [`scripts/lib/assistant.sh`](/home/tndlux/workspaces/dev_ws/src/manyforge/scripts/lib/assistant.sh) — added launcher-side `LANE_REGISTRY` case statement.

### Inside `NemoClaw-Thor/manyforge/`

4. [`scripts/proxy/vllm-proxy.py`](../scripts/proxy/vllm-proxy.py) — added `MANYFORGE_PROXY_*` env alias adapter + `MANYFORGE_PROXY_PROFILE`.
5. New package: [`common/`](../common/) with 5 modules + tests.
6. New package: [`assistant_session/`](../assistant_session/) with 4 modules + tests.
7. New policy split: [`policies/manyforge-egress-shared.yaml`](../policies/manyforge-egress-shared.yaml), [`policies/manyforge-openclaw.overlay.yaml`](../policies/manyforge-openclaw.overlay.yaml), [`policies/manyforge-hermes.overlay.yaml`](../policies/manyforge-hermes.overlay.yaml).

## Specs consulted

### [`manyforge_specs/docs/cross-workspace-conventions.md`](/home/tndlux/workspaces/dev_ws/src/manyforge_specs/docs/cross-workspace-conventions.md)

The load-bearing spec for Phase 1. It establishes:

| Spec clause | Phase 1 treatment |
|---|---|
| The wire-family identifier `manyforge.assistant.provider_request.v0` is stable; family is the major, `schemaVersion` is the minor. | **Respected.** The new `LANE_REGISTRY` entries do NOT mutate the envelope shape. `assistant_provider.py`'s `NemoClawAssistantProvider` still emits the same field set; the lane-id resolution is internal to the provider-id dispatch, not a wire-format change. |
| `manyforge`, `manyforge_assistant_bridge`, and `openclaw_assistant_bridge` all accept both old and new envelope versions. | **Respected.** No envelope versioning changes in Phase 1. Both bridges still parse the v0 envelope; LANE_REGISTRY is a dispatch layer above. |
| Mode-scoped MCP wrapper at `/api/assistant/bridge/tools/{toolId}` is the only sanctioned mutation surface. | **Respected.** `manyforge-mcp-bridge.py` continues to POST there exclusively. Lane-neutralization changes the `principal` and `conversationId` *prefixes* (now lane-derived) but not the bridge endpoint or its envelope fields. |
| Direct-lane bridge on `127.0.0.1:8100` runs from `manyforge/manyforge_assistant_bridge/`. | **Respected.** The launcher's `LANE_REGISTRY` records 8100 as the direct lane's default port. The bridge code in `dev_ws/manyforge_assistant_bridge/` is unchanged in Phase 1. Phase 2 will revisit whether to move it (Open question Q1). |

### [`manyforge_specs/docs/INDEX.md`](/home/tndlux/workspaces/dev_ws/src/manyforge_specs/docs/INDEX.md)

Scanned for spec entries touching the bridge envelope, MCP wrapper, or skill prompt. No additional specs flagged for Phase 1 work.

### [`manyforge_specs/docs/agent-playbook.md`](/home/tndlux/workspaces/dev_ws/src/manyforge_specs/docs/agent-playbook.md)

Scanned for prompt/skill conventions that the Phase 3 OpenClaw skill rewrite would touch. Phase 1 does NOT modify the agent skill prompt (the rewrite is Phase 3) — `discovery_mode` parameter is plumbed through `common/prompt.py` but defaults to `"direct"` (the existing behavior). No agent-playbook entries violated.

### `manyforge_specs/docs/spec/*` (the spec/ subdirectory)

Scanned 17 spec files for Phase 1 impact:

| Spec | Phase 1 touches? |
|---|---|
| `100-component-model-and-interactions.md` | No — Phase 1 is implementation-side. |
| `420-program-yaml-schema.md` | No. |
| `430-deployment-artifact-schema.md` | No. |
| `440-node-catalog-format-and-versioning.md` | No. |
| `460-skill-catalog-and-intervention.md` | Phase 3 will touch (skill rewrite). Phase 1: not yet. |
| `500-observability-tracing-and-performance.md` | **Respected — extended audit row.** The MCP bridge's audit row now includes `lane / principal / conversationId / assistantMode` columns (was just `ts / tool / requestId / success / args`). The four new columns extend, not break, the existing observability schema. Consumers reading the old fields continue to work; new consumers (cross-lane reports) get the extra context. **Spec amendment recommended**: the spec section on bridge-audit JSONL should be amended to document the four new columns. Filed as a Phase 5 follow-up since the consumer surface (compare_lanes.py) doesn't exist until then. |
| `510-runtime-diagnostics-and-scene-tracing.md` | No. |

## Spec amendments recommended (not landed yet)

The following spec amendments are flagged for follow-up. None block Phase 1 — they describe improvements that codify Phase 1's reality after the fact.

1. **`cross-workspace-conventions.md`** — add a section describing the `LANE_REGISTRY` discipline (composer side + launcher side) and the convention that unknown provider ids must raise rather than fall back silently. This codifies the foot-gun fix.
2. **`spec/500-observability-tracing-and-performance.md`** — document the four extended audit-row columns (`lane / principal / conversationId / assistantMode`) so future tooling can rely on them.
3. **`cross-workspace-conventions.md`** — record that `MANYFORGE_PROXY_*` is the canonical env-var prefix for the vLLM proxy and `OPENCLAW_PROXY_*` is a deprecated alias for one release cycle.

## Conflicts encountered

**None.** Phase 1's behavior-preserving constraint (no concurrent OpenClaw native/plugin behavior changes) meant every change was either:

- A pure addition (new modules in `common/` and `assistant_session/`; new policy overlay files; new env-var canonical names alongside the legacy aliases), OR
- A re-export of an existing surface, OR
- A schema extension (audit row), OR
- A type-safe rejection of previously-silent unknown inputs (`LANE_REGISTRY`).

No spec section's described behavior was violated. The `LANE_REGISTRY` adoption of the `hermes` provider id is forward-looking (marked `inert=True` in the registry; the launcher rejects activation without a Phase 4 feature flag), so no spec describing Hermes behavior is yet authoritative.

## Sign-off

- [x] Phase 1 modifications consulted the relevant `manyforge_specs/` sections.
- [x] No spec was silently violated.
- [x] Spec amendments recommended but not yet landed are listed above.
- [x] Behavior-preserving constraint (THREE-LANE plan §8 Phase 1) is satisfied — `git revert` of any Phase 1 commit cleanly restores prior behavior.
