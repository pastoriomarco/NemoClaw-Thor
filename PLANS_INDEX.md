> Docs entrypoint: [`INDEX.md`](INDEX.md)

# NemoClaw-Thor Plans & Reports Status Board

This is the single recap of every plan-like and report-like document in this
repo — the assistant-lane plans under [`manyforge/docs/`](manyforge/docs/) and
the serving plans / investigations under [`serving/docs/`](serving/docs/). It
exists so any contributor (human or LLM) can answer, without re-reading every
file: what each plan covers, what stage it's at, and where the current
authoritative status is.

Pure reference/runbook docs (architecture, MCP integration, profile
calibration, smoke corpus/runbook, lane comparison) are catalogued in
[`INDEX.md`](INDEX.md), not here. When a plan's lived state changes, update
this board first, then reconcile the plan body.

Scope note: ManyForge **core** plans (kernel / collision / planner / scene)
live in `manyforge_specs` —
[`manyforge_specs/docs/plans/PLANS_INDEX.md`](/home/tndlux/workspaces/dev_ws/src/manyforge_specs/docs/plans/PLANS_INDEX.md).
This board covers only the assistant-lane and serving work owned here.

## Status legend

- **In progress** — active work this cycle.
- **Landed (opt-in)** — implemented and verified, gated behind a flag pending a
  live bake-off before it becomes a default.
- **Interim** — a decision/result doc that records the current call but is
  explicitly not final.
- **Draft** — written; awaiting approval before execution.
- **Partially shipped** — some phases landed; residual scope tracked in the doc.
- **Shipped (result)** — the work this doc covered is delivered; kept for the
  result + rationale.
- **Investigation (reference)** — a durable deep-dive kept as dev reference even
  after the immediate work is done.
- **Archived** — completed/superseded; moved under an `archive/` folder,
  retained not deleted, cited below with its archive path.

## Active work focus

- **Three-lane assistant migration** — [`manyforge/docs/THREE-LANE-MIGRATION-PLAN.md`](manyforge/docs/THREE-LANE-MIGRATION-PLAN.md)
  is the hub (rev. 6, 2026-06-08). Phases 0 / 0.5 / 1 are complete (archived);
  Phase 3 (OpenClaw native) landed; Phase 4 (Hermes) is landed opt-in; Phase 5
  production decision is interim. Latest evidence: same-day three-lane parity
  run on gemma4-12b QAT — see [`manyforge/docs/LANE-COMPARISON.md`](manyforge/docs/LANE-COMPARISON.md)
  and `smoke-evidence/2026-06-09-thor-three-lane-parity-qat/`.
- **V9.1 serving execution** — [`serving/docs/V9.1-EXECUTION.md`](serving/docs/V9.1-EXECUTION.md)
  (consolidated plan + results; executed 2026-05-30 → 31): Phases 0-4 ran, Phase 5
  deferred. Status markers in the doc are point-in-time as of 2026-05-31. Forward
  FP4 work in [`V9.1-TASK4-FP4-UNLOCK.md`](serving/docs/V9.1-TASK4-FP4-UNLOCK.md) +
  [`V9.1-FOLLOWUP-TASKS.md`](serving/docs/V9.1-FOLLOWUP-TASKS.md).

## Assistant-lane plans (`manyforge/docs/`)

| Plan | Status | Stage / phase | Notes |
|---|---|---|---|
| [`THREE-LANE-MIGRATION-PLAN.md`](manyforge/docs/THREE-LANE-MIGRATION-PLAN.md) | In progress | Hub; rev. 6 (2026-06-08) | Architecture + the three load-bearing principles + per-phase plan/gates. Phases 0/0.5/1 archived complete. |
| [`PHASE-3-OPENCLAW-NATIVE-RESULT.md`](manyforge/docs/PHASE-3-OPENCLAW-NATIVE-RESULT.md) | Shipped (result) | Phase 3 | OpenClaw native `tool_search`/`describe`/`call` discovery-surface result. |
| [`PHASE-4-HERMES-LONGITUDINAL.md`](manyforge/docs/PHASE-4-HERMES-LONGITUDINAL.md) | Landed (opt-in) | Phase 4 | Hermes lane implemented + unit-verified, gated `HERMES_LANE_PHASE4_ENABLED`; live longitudinal numbers operator-driven, TBD. Impl: `manyforge/lanes/hermes/`. |
| [`PHASE-5-PRODUCTION-DECISION.md`](manyforge/docs/PHASE-5-PRODUCTION-DECISION.md) | Interim | Phase 5 | Records the current production-default call; not final until full bake-off lands. |
| [`MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md`](manyforge/docs/MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md) | Partially shipped | — | LLM-stack deployment / model-selection plan for Thor + Orin budgets; delivered phases + open follow-ups. |
| [`self-healing-chain-harness.md`](manyforge/docs/self-healing-chain-harness.md) | In progress | design + plan | Self-healing chain harness design, plan & documentation. |
| [`UPSTREAM-ISSUE-local-inference-allowed-ips.md`](manyforge/docs/UPSTREAM-ISSUE-local-inference-allowed-ips.md) | Open (upstream) | issue draft | `local-inference` preset blocks gateway-embedded inference on Docker-bridge deployments. |

## Serving plans, recipes & investigations (`serving/docs/`)

| Document | Status | Notes |
|---|---|---|
| [`V9.1-EXECUTION.md`](serving/docs/V9.1-EXECUTION.md) | Completed (2026-05-30 → 31) | Consolidated V9.1 execution plan + results (Phases 0-4 ran; 5 deferred). Status markers point-in-time. |
| [`V9.1-FOLLOWUP-TASKS.md`](serving/docs/V9.1-FOLLOWUP-TASKS.md) | Open | V9.1 follow-up tasks / findings. |
| [`V9.1-TASK4-FP4-UNLOCK.md`](serving/docs/V9.1-TASK4-FP4-UNLOCK.md) | Done (2026-05-31) | Thor sm_110a FP4 kernel unlock. |
| [`V9.1-IMAGE-NOTES.md`](serving/docs/V9.1-IMAGE-NOTES.md) | Reference | vLLM nightly with PR #42124 (LM-head ModelOpt). |
| [`V9-35B-A3B-NVFP4-NVIDIA-RECIPE.md`](serving/docs/V9-35B-A3B-NVFP4-NVIDIA-RECIPE.md) | Reference (recipe) | Qwen3.6-35B-A3B-NVFP4 NVIDIA serving recipe; not in v0.22.0. |
| [`V9-SMOKE-CORPUS-BASELINE.md`](serving/docs/V9-SMOKE-CORPUS-BASELINE.md) | Baseline (2026-05-30) | v9 image smoke-corpus baseline. |
| [`COSMOS-REASON2-FINETUNE-PLAN.md`](serving/docs/COSMOS-REASON2-FINETUNE-PLAN.md) | Plan | Cosmos-Reason2 fine-tune + NVFP4 quantize on Thor. |
| [`COSMOS-REASON2-32B-QUANTIZATION.md`](serving/docs/COSMOS-REASON2-32B-QUANTIZATION.md) | Investigation (reference) | 2026-04-30 — 32B quantization on Thor. |
| [`DFLASH-INVESTIGATION.md`](serving/docs/DFLASH-INVESTIGATION.md) | Investigation (reference) | 2026-04-15…17 — DFlash speculative decoding on SM110. |
| [`DS4-DEEPSEEK-V4-FLASH-INVESTIGATION.md`](serving/docs/DS4-DEEPSEEK-V4-FLASH-INVESTIGATION.md) | Investigation (reference) | 2026-05-12 (upd 05-19) — DeepSeek-V4-Flash on Thor. |
| [`MINIMAX-M27-INVESTIGATION.md`](serving/docs/MINIMAX-M27-INVESTIGATION.md) | Investigation (reference) | 2026-04-22 — MiniMax-M2.7 REAP on Thor. |
| [`TOOL-EVAL-BENCH-THOR.md`](serving/docs/TOOL-EVAL-BENCH-THOR.md) | Bench report | Consolidated Thor tool-eval-bench report. |
| [`PERFORMANCE-V7.md`](serving/docs/PERFORMANCE-V7.md) | Perf report (historical) | v7 image coverage report; point-in-time. |
| [`KV-CACHE-BUDGET.md`](serving/docs/KV-CACHE-BUDGET.md) | Reference (live) | Thor 128 GB unified-memory KV budget. |

> The V9.1 cluster's completed parts and the closed investigations are
> candidates for a future compaction/archive pass; that pass is intentionally
> deferred and gated on confirmation (statuses above are nuanced). Nothing
> here is archived yet.

## Archived plans & reports (`manyforge/docs/archive/`)

Retained, not deleted. Each was completed or superseded; cited here with its
archive path.

| Document (archive path) | Why archived |
|---|---|
| [`archive/LANE-COMPARISON-direct-vs-openclaw.md`](manyforge/docs/archive/LANE-COMPARISON-direct-vs-openclaw.md) | Superseded — folded into [`LANE-COMPARISON.md`](manyforge/docs/LANE-COMPARISON.md). |
| [`archive/PHASE-0-LANE-BASELINE.md`](manyforge/docs/archive/PHASE-0-LANE-BASELINE.md) | Completed phase 0 (pre-refactor lane baselines). |
| [`archive/PHASE-0.5-HERMES-SPIKE.md`](manyforge/docs/archive/PHASE-0.5-HERMES-SPIKE.md) | Completed phase 0.5 (Hermes contract spike). |
| [`archive/PHASE-1-SPECS-AUDIT.md`](manyforge/docs/archive/PHASE-1-SPECS-AUDIT.md) | Completed phase 1 (specs audit). |
| [`archive/REBUILD-2026-06-02.md`](manyforge/docs/archive/REBUILD-2026-06-02.md) | Completed 2026-06-02 rebuild — consolidates the rebuild record (OpenShell 0.0.44 + OpenClaw 2026.5.22), the findings, and the final bake-off report. |
| [`archive/SMOKE-BAKEOFF-2026-06-01-3model.md`](manyforge/docs/archive/SMOKE-BAKEOFF-2026-06-01-3model.md) | Completed 3-model pre-rebuild bake-off. |
| [`archive/WORKSPACE-PROMPT-OPTIMIZATION.md`](manyforge/docs/archive/WORKSPACE-PROMPT-OPTIMIZATION.md) | Completed 2026-05-06 prompt-tuning session. |
| [`archive/PIPELINE-TRACE-2026-06-03.md`](manyforge/docs/archive/PIPELINE-TRACE-2026-06-03.md) | Closed OpenClaw-lane failure diagnosis. |
| [`archive/THREE-LANE-SCORER-NOTE.md`](manyforge/docs/archive/THREE-LANE-SCORER-NOTE.md) | Moved from the `manyforge` deployment repo (which keeps no analysis); 2026-06-09 scorer-strictness analysis, summarized in [`LANE-COMPARISON.md`](manyforge/docs/LANE-COMPARISON.md). |
| [`manyforge/archive/openclaw-plugin-attempt-2026-06-02/`](manyforge/archive/openclaw-plugin-attempt-2026-06-02/) | Archived OpenClaw plugin-attempt code/build artifacts (not docs). |

## The three lanes

Per-lane dev/analysis docs and implementations (operational bring-up +
live-monitoring is in the deployment repo's
[`LANE_BRINGUP.md`](/home/tndlux/workspaces/dev_ws/src/manyforge/docs/operations/LANE_BRINGUP.md)):

| Lane | Dev doc | Implementation | Routing default today |
|---|---|---|---|
| Direct vLLM | [`lanes/direct/README.md`](manyforge/lanes/direct/README.md) | bridge in **`manyforge` repo** (`manyforge_assistant_bridge/`) | latency-sensitive override only |
| OpenClaw | [`lanes/openclaw/README.md`](manyforge/lanes/openclaw/README.md) | `manyforge/openclaw_assistant_bridge/` | **`default_lane`** in [`lane_routing.yaml`](manyforge/lanes/lane_routing.yaml) (starting default, not final) |
| Hermes | [`lanes/hermes/README.md`](manyforge/lanes/hermes/README.md) | `manyforge/lanes/hermes/` | opt-in (`HERMES_LANE_PHASE4_ENABLED`) for long-running |

## Raw smoke evidence — kept in place

`manyforge/docs/smoke-evidence/` holds dated per-run evidence directories. They
are **append-only, never archived/relocated**, and are the reproducibility
record the analysis docs link to. Current sets are listed in
[`INDEX.md`](INDEX.md#raw-smoke-evidence--kept-in-place).

## How to use this board

- When you start work on a plan, set its row to **In progress** and note the
  stage. When a phase ships, update the row — do not delete it.
- When a plan/report finishes or is superseded, move the file under the matching
  `archive/` folder and move its row into "Archived" **with the archive path**,
  in the same commit. Never drop a row.
- Raw `smoke-evidence/` directories are never archived — they are evidence, not
  narrative.
