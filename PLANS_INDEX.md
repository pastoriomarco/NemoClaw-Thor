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
`docs/plans/PLANS_INDEX.md` in the private `manyforge_specs` repo (maintainers clone it as a sibling workspace).
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

- **DS4 Thor optimization** — [`serving/docs/DS4-THOR-OPTIMIZATION-PLAN.md`](serving/docs/DS4-THOR-OPTIMIZATION-PLAN.md)
  completed its tuning phases on 2026-08-05. The runtime is now pinned to
  v0.5.6.2 while retaining the fixed v0.5.5 performance/deep-context baseline.
- **Three-lane assistant migration** — [`manyforge/docs/THREE-LANE-MIGRATION-PLAN.md`](manyforge/docs/THREE-LANE-MIGRATION-PLAN.md)
  is the hub (rev. 6, 2026-06-08). **AD INTERIM since 2026-06-12** (operator
  decision): phases 0/0.5/1/2/3 complete, Phase 4 landed opt-in with the
  longitudinal gate unrun, Phase 5 interim. Two gating items remain (Hermes
  apples-to-apples rerun; Phase-4 longitudinal run); resumes when those are
  scheduled. Latest evidence: same-day three-lane parity
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
| [`THREE-LANE-MIGRATION-PLAN.md`](manyforge/docs/THREE-LANE-MIGRATION-PLAN.md) | AD INTERIM (2026-06-12) | Hub; rev. 6 (2026-06-08) | Architecture + the three load-bearing principles + per-phase plan/gates. Phases 0/0.5/1 archived complete; 2/3 complete; paused with two gating items (Hermes apples-to-apples rerun, Phase-4 longitudinal). |
| [`PHASE-3-OPENCLAW-NATIVE-RESULT.md`](manyforge/docs/PHASE-3-OPENCLAW-NATIVE-RESULT.md) | Shipped (result) | Phase 3 | OpenClaw native `tool_search`/`describe`/`call` discovery-surface result. |
| [`PHASE-4-HERMES-LONGITUDINAL.md`](manyforge/docs/PHASE-4-HERMES-LONGITUDINAL.md) | Landed (opt-in) | Phase 4 | Hermes lane implemented + unit-verified, gated `HERMES_LANE_PHASE4_ENABLED`; live longitudinal numbers operator-driven, TBD. Impl: `manyforge/lanes/hermes/`. |
| [`PHASE-5-PRODUCTION-DECISION.md`](manyforge/docs/PHASE-5-PRODUCTION-DECISION.md) | Interim | Phase 5 | Records the current production-default call; not final until full bake-off lands. |
| [`MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md`](manyforge/docs/MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md) | Partially shipped | — | LLM-stack deployment / model-selection plan for Thor + Orin budgets; delivered phases + open follow-ups. |
| [`self-healing-chain-harness.md`](manyforge/docs/self-healing-chain-harness.md) | In progress | design + plan | Self-healing chain harness design, plan & documentation. |
| [`UPSTREAM-ISSUE-local-inference-allowed-ips.md`](manyforge/docs/UPSTREAM-ISSUE-local-inference-allowed-ips.md) | Open (upstream) | issue draft | `local-inference` preset blocks gateway-embedded inference on Docker-bridge deployments. |

## Serving plans, recipes & investigations (`serving/docs/`)

| Document | Status | Notes |
|---|---|---|
| [`DS4-THOR-OPTIMIZATION-PLAN.md`](serving/docs/DS4-THOR-OPTIMIZATION-PLAN.md) | Completed (2026-08-05); v0.5.6.2 adoption smoke (2026-08-10) | Docker-only Entrpi DS4 optimization for Thor `sm_110`: repaired upstream top-512 selector, 512K/4K stable default, optional faster 256K/8K profile, and recorded rejected D2R/SM110a candidates. The v0.5.6.2 runtime passes Chat/Responses smoke; fixed performance and deep-context evidence remains explicitly labelled v0.5.5. |
| [`V9.1-EXECUTION.md`](serving/docs/V9.1-EXECUTION.md) | Completed (2026-05-30 → 31) | Consolidated V9.1 execution plan + results (Phases 0-4 ran; 5 deferred). Status markers point-in-time. |
| [`V9.1-FOLLOWUP-TASKS.md`](serving/docs/V9.1-FOLLOWUP-TASKS.md) | Open | V9.1 follow-up tasks / findings. |
| [`V9.1-TASK4-FP4-UNLOCK.md`](serving/docs/V9.1-TASK4-FP4-UNLOCK.md) | Done (2026-05-31) | Thor sm_110a FP4 kernel unlock. |
| [`V9.1-IMAGE-NOTES.md`](serving/docs/V9.1-IMAGE-NOTES.md) | Reference | vLLM nightly with PR #42124 (LM-head ModelOpt). |
| [`V9-35B-A3B-NVFP4-NVIDIA-RECIPE.md`](serving/docs/V9-35B-A3B-NVFP4-NVIDIA-RECIPE.md) | Reference (recipe) | Qwen3.6-35B-A3B-NVFP4 NVIDIA serving recipe; not in v0.22.0. |
| [`V9-SMOKE-CORPUS-BASELINE.md`](serving/docs/V9-SMOKE-CORPUS-BASELINE.md) | Baseline (2026-05-30) | v9 image smoke-corpus baseline. |
| [`COSMOS-REASON2-FINETUNE-PLAN.md`](serving/docs/COSMOS-REASON2-FINETUNE-PLAN.md) | Deferred (2026-06-12) | Cosmos-Reason2 fine-tune + NVFP4 quantize on Thor. Deferred since the clean-start model default moved to gemma-QAT (2026-06-07 sweep + 06-09 head-to-head); revisit only if cosmos is re-anchored. |
| [`COSMOS-REASON2-32B-QUANTIZATION.md`](serving/docs/COSMOS-REASON2-32B-QUANTIZATION.md) | Investigation (reference) | 2026-04-30 — 32B quantization on Thor. |
| [`DFLASH-INVESTIGATION.md`](serving/docs/DFLASH-INVESTIGATION.md) | Investigation (reference) | 2026-04-15…17 — DFlash speculative decoding on SM110. |
| [`DS4-DEEPSEEK-V4-FLASH-INVESTIGATION.md`](serving/docs/DS4-DEEPSEEK-V4-FLASH-INVESTIGATION.md) | Landed (opt-in; Thor profile) | Dockerized Entrpi DS4 v0.5.6.2 with the 0731 base + matching DSpark drafter persists weights at `~/thor-hf-cache/ds4/`. The `sm_110` image uses Entrpi's repaired streaming exact top-512 selector. The default is 524,288 context, two banks, 4K effective chunks, and an 8 GiB live-memory floor. The v0.5.5 baseline includes stable queued concurrency and exact 244,518-token retrieval; the earlier 479,817-token capacity proof is retained. Runbook: [`DS4-ON-THOR.md`](serving/docs/DS4-ON-THOR.md). |
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
[`LANE_BRINGUP.md`](https://github.com/pastoriomarco/manyforge/blob/main/docs/operations/LANE_BRINGUP.md)):

| Lane | Dev doc | Implementation | Routing default today |
|---|---|---|---|
| Direct model | [`lanes/direct/README.md`](manyforge/lanes/direct/README.md) | bridge in **`manyforge` repo** (`manyforge_assistant_bridge/`) | latency-sensitive override only |
| OpenClaw | [`lanes/openclaw/README.md`](manyforge/lanes/openclaw/README.md) | `manyforge/openclaw_assistant_bridge/` | Current startup default via `ASSISTANT_PROVIDER=openclaw`; `lane_routing.yaml` is design-only |
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
