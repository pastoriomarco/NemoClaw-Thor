# NemoClaw-Thor Documentation Index

This repo is the **support layer** for running ManyForge's AI assistant on
Jetson AGX Thor (SM110a / Blackwell): local model **serving**, the assistant
**composer lanes** (direct / openclaw / hermes), and the **sandbox + agent
framework** wiring that connects a served model to the Composer.

It is not the deployment repo and not the specs repo:

| Repo | Owns | Where |
|---|---|---|
| **`manyforge`** | Deployment / orchestration: launch, setup, runtime operation, operational diagnostics. The **Direct-lane bridge** (`manyforge_assistant_bridge/`) lives here. | `dev_ws/src/manyforge` |
| **`manyforge_specs`** | Normative specs, ADRs, and core development plans (kernel / collision / planner / scene). Private. | `dev_ws/src/manyforge_specs` |
| **`NemoClaw-Thor`** (this repo) | Model serving + assistant-lane dev/analysis + operator setup for Thor/Orin. | here |

Cross-repo ownership detail: [`manyforge_specs/docs/cross-workspace-conventions.md`](/home/tndlux/workspaces/dev_ws/src/manyforge_specs/docs/cross-workspace-conventions.md).

## Start here

- [`AGENTS.md`](AGENTS.md) — scope, cross-repo authority map, boundary rules,
  branch/commit workflow. **Read first before changing anything.**
- [`README.md`](README.md) — landing page / quickstart.
- [`USER_QUICKSTART_MANUAL.md`](USER_QUICKSTART_MANUAL.md) — full operator
  procedure: swap setup, image rebuild, sandbox workflows, cleanup,
  troubleshooting.
- [`PLANS_INDEX.md`](PLANS_INDEX.md) — **status board** for every plan and
  report in this repo (live + archived), with stage/phase and pickup state.

## Operator setup & session workflow

- [`ORIN-SETUP.md`](ORIN-SETUP.md) — Jetson Orin AGX bring-up specifics.
- [`setup/NEMOCLAW-OPENCLAW-WORKFLOW.md`](setup/NEMOCLAW-OPENCLAW-WORKFLOW.md)
  — canonical end-to-end recipe (start model → wire OpenShell → dispatch
  agent), including scripted / non-interactive use.
- [`setup/`](setup/) — `configure-local-provider.sh`, `status.sh`,
  `checks.sh`, `sandbox-runtime.sh`, `policies/`.
- [`serving/start-model.sh`](serving/start-model.sh) /
  [`serving/start-duo.sh`](serving/start-duo.sh) — start a served model
  (profiles in [`serving/config.sh`](serving/config.sh)).

## The three assistant lanes

The Composer assistant has three first-class lanes, but the live stack starts
one lane per Composer process via `ASSISTANT_PROVIDER`. The
[`manyforge/lanes/lane_routing.yaml`](manyforge/lanes/lane_routing.yaml)
file is design-only today, not a runtime router.
**Operational bring-up + live-monitoring for all three lanes lives in the
deployment repo** at
[`manyforge/docs/operations/LANE_BRINGUP.md`](/home/tndlux/workspaces/dev_ws/src/manyforge/docs/operations/LANE_BRINGUP.md);
the dev/analysis deep-dives live here.

- **Hub:** [`manyforge/docs/THREE-LANE-MIGRATION-PLAN.md`](manyforge/docs/THREE-LANE-MIGRATION-PLAN.md)
  — architecture, the three load-bearing principles, per-phase plan.
- **Comparison / benchmark:** [`manyforge/docs/LANE-COMPARISON.md`](manyforge/docs/LANE-COMPARISON.md)
  — consolidated cross-lane numbers, speed, and scorer analysis.

| Lane | Wire path | Dev/analysis docs | Implementation |
|---|---|---|---|
| **Direct model** | in-process bridge → local model endpoint (no gateway hops; fastest) | [`lanes/direct/README.md`](manyforge/lanes/direct/README.md) | bridge in **`manyforge` repo** (`manyforge_assistant_bridge/`) |
| **OpenClaw** | Composer → OpenClaw gateway agent (native `tool_search`/`describe`/`call` discovery) | [`lanes/openclaw/README.md`](manyforge/lanes/openclaw/README.md), [`PHASE-3-OPENCLAW-NATIVE-RESULT.md`](manyforge/docs/PHASE-3-OPENCLAW-NATIVE-RESULT.md) | `manyforge/openclaw_assistant_bridge/` (here) |
| **Hermes** | Hermes Agents native MCP + session/runs APIs (memory + skills; opt-in `HERMES_LANE_PHASE4_ENABLED`) | [`lanes/hermes/README.md`](manyforge/lanes/hermes/README.md), [`PHASE-4-HERMES-LONGITUDINAL.md`](manyforge/docs/PHASE-4-HERMES-LONGITUDINAL.md) | `manyforge/lanes/hermes/` (here) |

Production-decision status (interim): [`manyforge/docs/PHASE-5-PRODUCTION-DECISION.md`](manyforge/docs/PHASE-5-PRODUCTION-DECISION.md).

## Assistant pipeline reference (read in this order)

- [`manyforge/docs/COMPOSER-ASSISTANT-ARCHITECTURE.md`](manyforge/docs/COMPOSER-ASSISTANT-ARCHITECTURE.md)
  — canonical architecture: request flow, env-var knobs, thinking subsystem,
  loop-defense, profile catalog. Read first for "how does it work?"
- [`manyforge/docs/COMPOSER-ASSISTANT-RUNBOOK.md`](manyforge/docs/COMPOSER-ASSISTANT-RUNBOOK.md)
  — per-symptom debugging across the gates. Read first for "it's broken, why?"
- [`manyforge/docs/MANYFORGE-MCP-INTEGRATION.md`](manyforge/docs/MANYFORGE-MCP-INTEGRATION.md)
  — MCP wire details, tool-name mangling, principal binding.
- [`manyforge/docs/MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md`](manyforge/docs/MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md)
  — model selection plan for Thor + Orin budgets.
- [`manyforge/docs/MANYFORGE-PROFILE-CALIBRATION.md`](manyforge/docs/MANYFORGE-PROFILE-CALIBRATION.md)
  — sizing methodology before adding a profile or changing memory knobs.
- [`manyforge/docs/SMOKE-CORPUS.md`](manyforge/docs/SMOKE-CORPUS.md) +
  [`manyforge/docs/SMOKE-ITER-RUNBOOK.md`](manyforge/docs/SMOKE-ITER-RUNBOOK.md)
  — the smoke corpus and the cold-start order for running an iteration.
- [`manyforge/docs/self-healing-chain-harness.md`](manyforge/docs/self-healing-chain-harness.md)
  — self-healing chain harness design.
- [`manyforge/docs/UPSTREAM-ISSUE-local-inference-allowed-ips.md`](manyforge/docs/UPSTREAM-ISSUE-local-inference-allowed-ips.md)
  — upstream `local-inference` preset issue draft.

## Serving & model tuning (`serving/`)

- [`serving/config.sh`](serving/config.sh) — model profiles.
- [`serving/docs/KV-CACHE-BUDGET.md`](serving/docs/KV-CACHE-BUDGET.md) — Thor
  128 GB unified-memory KV budget reference.
- [`serving/docs/`](serving/docs/) — serving plans, recipes, performance
  reports, and architectural investigations. Status board for these is in
  [`PLANS_INDEX.md`](PLANS_INDEX.md). Highlights:
  - **V9.1 serving execution** — [`V9.1-EXECUTION.md`](serving/docs/V9.1-EXECUTION.md)
    (plan + results; companions `-FOLLOWUP-TASKS`, `-IMAGE-NOTES`, `-TASK4-FP4-UNLOCK`).
  - **Serving recipe** — [`V9-35B-A3B-NVFP4-NVIDIA-RECIPE.md`](serving/docs/V9-35B-A3B-NVFP4-NVIDIA-RECIPE.md),
    baseline [`V9-SMOKE-CORPUS-BASELINE.md`](serving/docs/V9-SMOKE-CORPUS-BASELINE.md).
  - **Fine-tune** — [`COSMOS-REASON2-FINETUNE-PLAN.md`](serving/docs/COSMOS-REASON2-FINETUNE-PLAN.md).
  - **Investigations (durable references)** — [`DFLASH-INVESTIGATION.md`](serving/docs/DFLASH-INVESTIGATION.md),
    [`DS4-DEEPSEEK-V4-FLASH-INVESTIGATION.md`](serving/docs/DS4-DEEPSEEK-V4-FLASH-INVESTIGATION.md),
    [`MINIMAX-M27-INVESTIGATION.md`](serving/docs/MINIMAX-M27-INVESTIGATION.md),
    [`COSMOS-REASON2-32B-QUANTIZATION.md`](serving/docs/COSMOS-REASON2-32B-QUANTIZATION.md).
  - **Bench / perf** — [`TOOL-EVAL-BENCH-THOR.md`](serving/docs/TOOL-EVAL-BENCH-THOR.md),
    [`PERFORMANCE-V7.md`](serving/docs/PERFORMANCE-V7.md).
- [`serving/agentic-bench/README.md`](serving/agentic-bench/README.md) — bench
  harness and candidate plan.
- [`serving/docker/`](serving/docker/) — Thor-specific build notes and patch
  rationale.

## Archived docs (retained, not deleted)

Completed/superseded narrative docs live under
[`manyforge/docs/archive/`](manyforge/docs/archive/) and are catalogued with
their archive path in [`PLANS_INDEX.md`](PLANS_INDEX.md). Archived code/build
artifacts from the OpenClaw plugin attempt live under
[`manyforge/archive/openclaw-plugin-attempt-2026-06-02/`](manyforge/archive/openclaw-plugin-attempt-2026-06-02/).

## Raw smoke evidence — kept in place

Dated per-run evidence directories under
[`manyforge/docs/smoke-evidence/`](manyforge/docs/smoke-evidence/) are
**append-only and never archived/relocated** — they are the reproducibility
record the analysis docs link back to. Current sets:

- `2026-06-01-3model-bakeoff/`
- `2026-06-05-gemma4-12b-it-gguf/`
- `2026-06-05-gemma4-12b-128k-fix1-pnp/`
- `2026-06-06-gemma4-history-budget-failopen/`
- `2026-06-07-thor-orin-smoke-sweep-qat/`
- `2026-06-09-thor-three-lane-parity-qat/`
