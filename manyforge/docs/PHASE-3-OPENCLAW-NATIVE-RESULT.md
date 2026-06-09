# Phase 3 — OpenClaw Native Discovery-Surface Result

Phase 3 gate of [THREE-LANE-MIGRATION-PLAN.md](./THREE-LANE-MIGRATION-PLAN.md): does the discovery-protocol skill addendum (in [`manyforge/lanes/openclaw/skill_addendum.md`](../lanes/openclaw/skill_addendum.md)) close the gap between the OpenClaw 2026.5.6+ native shim and the iter-32 51/66 baseline on cosmos-reason2-8b?

## Status — pending

The empirical validation requires a known-good OpenClaw stack with the proxy in path. Phase 0 D-1 and O-1 both confirmed the proxy mutations are necessary for cosmos-reason2-8b accuracy on this corpus regardless of lane:

| Run | Lane | Proxy in path | Effective rate |
|---|---|---|---|
| Iter-32 baseline (memory, 2026-05-10) | OpenClaw (plugin attempt) | YES | 51/66 (77.3%) |
| Phase 0 D-1 (this branch) | Direct | YES (via launcher) | 28/66 (42.4%) |
| Phase 0 O-1 (this branch) | OpenClaw (native discovery) | **NO** (vLLM direct) | 14/66 (21.2%) |

The 14/66 O-1 result is NOT a measure of the discovery surface — it's a measure of the OpenClaw stack with the proxy mutations missing. Failure patterns mirror Direct D-1 (`<MISSING>` args, expected tool not observed), confirming the issue is upstream of the lane choice.

## Pre-requisites for the real Phase 3 measurement

Before re-running Phase 3, the proxy must be in path. The setup is documented in [COMPOSER-ASSISTANT-RUNBOOK.md](./COMPOSER-ASSISTANT-RUNBOOK.md) but the foundation triage is open:

1. **vLLM must be on `:8050`** (not `:8000`).
2. **vllm-proxy on `:8000`** with the four mutations enabled (`UNWRAP_TOOL_CALL_ARGS`, `PROMOTE_REASONING_TO_CONTENT`, `NORMALIZE_TOOL_NAMES`, `TOOL_ERROR_REWRITE`), forwarding to `:8050`.
3. **OpenClaw gateway** with `models.providers.inference.baseUrl` pointed at the proxy (`http://host.openshell.internal:8000/v1`).
4. **Active mutation profile recorded** in proxy banner: `compat` (Phase 0 baseline), `native` (Phase 3 target after skill addendum is fully effective), or `prod` (post-Phase 5 production default).

The current `./scripts/launch.sh` flow has a race condition where `start-model.sh`'s implicit proxy start and the launcher's explicit proxy step collide on the PID file. Workarounds tried:

- `THOR_RESTART_PROXY=0` skips `start-model.sh`'s proxy → vLLM lands on `:8050`, launcher's proxy step runs → works on first start, fails on subsequent if any stale process survives.
- `START_VLLM_PROXY=false` skips the launcher's proxy step → vLLM lands on `:8000` (the lane we got in O-1), no proxy in path → this is the failed configuration.

**Recommended fix path** (deferred to a future cycle):

1. Make `start-model.sh` honor `MANYFORGE_PROXY_LISTEN_PORT` so vLLM and the proxy port are coordinated from a single source of truth.
2. Have the launcher's proxy step detect an already-managed proxy from `start-model.sh` (matching PID file) and reuse it instead of erroring.
3. Add a `setup-openclaw.sh` (Phase 3 task) that takes a clean machine to a working OpenClaw lane with proxy in path, idempotent.

## What Phase 3 has landed (code)

Phase 3 keyboard work is complete:

- **`manyforge/lanes/openclaw/skill_addendum.md`** — discovery-protocol primer (5 efficiency rules + pre-named tool vocabulary + worked example).
- **`manyforge/lanes/openclaw/policy.yaml`** — `SessionPolicy` for the lane (compaction every 2, synthetic short-circuits ON, discovery_mode=openclaw_discovery).
- **`manyforge/lanes/openclaw/README.md`** — lane contents + Phase 3 gate criteria.

When the foundation triage lands, re-run the smoke corpus through the OpenClaw lane with the skill addendum appended to the system prompt and the proxy in path. Pass criteria: **≥46/66 (≈70%)**.

## Decision per Phase 3 gate

If the rerun passes (≥46/66): the archived plugin artifacts can be deleted in Phase 5.

If it doesn't: document the gap, keep the archived artifacts available as a feature-flagged rollback (`OPENCLAW_LANE_MODE=plugin|native`), and proceed to Phase 4 either way — the architectural shape doesn't change, only the production routing default does.

## Decision (2026-06-03)

**Tools mode is the production OpenClaw default.** Code mode is functional but model-quality-limited on cosmos-reason2-8b. Full numbers, methodology, and verdict in [LANE-COMPARISON.md](./LANE-COMPARISON.md). Brief:

| Mode | Real cases | First-try | Effective | Gate (46/66 ≈ 70%) |
|---|---|---|---|---|
| Tools (`tool_search`/`tool_describe`/`tool_call`) | 12 | 50.0% | 58.3% | ~65% extrapolated — within striking distance |
| Code (`tool_search_code`, corrected primer) | 31 | 12.9% | 29.0% | ~21/74 ≈ 29% — far below |

The Phase 3 gate is not yet definitively cleared (the tools-mode smoke crashed mid-run after the gateway-restart-to-flip; recovery cost a clean 74-case run). The 12-case sample is decisive on direction but not on the precise pass-rate.

**Defaults landed under this decision:**

- `start-openclaw-assistant-bridge.sh`: `OPENCLAW_ASSISTANT_TOOL_SURFACE=tools` (was `code`).
- `scripts/lib/assistant.sh`: same default propagated to the bridge launcher AND the vllm-proxy launcher.
- `setup-manyforge-assistant.sh` (Sandbox provisioner): writes `tools.toolSearch = {enabled: true, mode: "tools"}` to `/sandbox/.openclaw/openclaw.json` on provision.
- Archived plugin artifacts at `archive/openclaw-plugin-attempt-2026-06-02/` stay available as a rollback path until Phase 5.

**Deferred to multi-model bake-off**: re-run both modes on qwen3.6-35B-NVFP4 (the 93/100 tool-eval-bench winner). If the 35B model closes the gap in code mode, the lane default should be model-dependent.

## Version tuple

| Component | Version |
|---|---|
| cosmos-reason2-8b | `nvidia/Cosmos-Reason2-8B` |
| OpenClaw | `2026.5.22` (NemoClaw `lkg`) |
| Hermes | `2026.5.16` / `0.14.0` (not yet active) |
| NemoClaw | `lkg` (`v0.0.55` + 4 commits) |
| vLLM | per cosmos profile |
| Proxy mutation profile | `compat` (when in path) / NONE (Phase 0 O-1) |
