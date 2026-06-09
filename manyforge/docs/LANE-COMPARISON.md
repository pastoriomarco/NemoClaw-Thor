# Lane comparison (direct / openclaw / hermes)

> Canonical lane-comparison reference (dev/analysis). Phase 3 three-lane result
> measured 2026-06-03; see [THREE-LANE-MIGRATION-PLAN.md](./THREE-LANE-MIGRATION-PLAN.md).
> Earlier direct-vs-openclaw A/B benchmark is archived at
> [archive/LANE-COMPARISON-direct-vs-openclaw.md](./archive/LANE-COMPARISON-direct-vs-openclaw.md).
> Operational per-lane bring-up / operate / live-monitor lives in the **`manyforge`**
> deployment repo: `docs/operations/LANE_BRINGUP.md`. The latest three-lane parity
> run (with the corrected scorer) folds in here from the manyforge scorer-note.

## TL;DR

For OpenClaw's `tool_search` shim, **tools mode (`tool_search` / `tool_describe` / `tool_call`) is the production default**. Code mode (`tool_search_code`) is functional but model-quality-limited on cosmos-reason2-8b — ~2× worse first-try rate, ~2× worse effective rate.

| Lane / mode | Cases | Pass | Soft-pass | Fail | First-try | Effective |
|---|---|---|---|---|---|---|
| OpenClaw **tools mode** (corrected primer) | 12 | **6** | 1 | 5 | **50.0%** | **58.3%** |
| OpenClaw **code mode** (corrected primer) | 31 | 4 | 5 | 22 | 12.9% | 29.0% |
| Direct lane | TBD next sprint | — | — | — | — | — |
| Hermes lane (per-turn, memory off) | Pipeline fixes validated — clean 75-case run pending | — | — | — | — | — |

Both partial samples — neither smoke completed all 74 cases (stack issues on tools mode at case 13, smoke killed at user request on code mode at case 26 of 31, both produced 0.0s-time tail-entries from cascaded composer-state-reset failures that are NOT model measurements). The first-try and effective rates above are computed only on real per-case measurements (cases that produced a non-zero elapsed time).

## 2026-06-08 Hermes readiness update

The Hermes lane is ready for a clean full-corpus measurement, but the current
numbers should not be compared as final baselines. The contaminated exploratory
runs were useful for pipeline diagnosis and closed four concrete issues:

1. **Native-MCP dispatch primer.** Hermes is now prompted against its native
   MCP surface rather than an OpenClaw-shaped discovery flow.
2. **Dispatcher termination.** The Hermes dispatcher treats only `run.*`
   lifecycle events as terminal. `tool.completed` no longer truncates recovery
   loops.
3. **Lean catalog Rule 5.** `nodeCatalog` is documented as the lean chooser
   surface. Parameterized node kinds must fetch full schema via `catalog_read`
   instead of relying on stripped inline params.
4. **Catalog-read loop breaker.** Repeated identical read-only `catalog_read`
   fetches are interrupted so the model acts instead of analyzing indefinitely.

Insert-family micro-probe after those fixes:

| Case | Verdict | Interpretation |
|---|---|---|
| `P3_tree_insert_runtime_obj_specific` | soft-pass | Correct family path, residual param / assertion weakness. |
| `TREE_insert_runtime_medium` | pass | Lean catalog + loop-breaker fixed the prior under-act. |
| `INSERT_position_first_specific` | fail | Wrong node-kind choice (`command_gripper` vs `wait_for_signal_bool`); model comprehension, not dispatch. |

Effective result: **2/3**, matching the inline-params variant without the
prompt-size penalty. The remaining insert fail is classified as a model ceiling
for gemma4 unless the same case passes in another lane/model with the same
catalog contract.

## Next clean comparison sequence

1. Run a clean Hermes full corpus with **self-heal on**, live monitoring, reset
   Hermes state, and the lean-catalog + loop-breaker pipeline. Do not use
   inline param bloat for the baseline.
2. Produce the Hermes taxonomy: pass / soft-pass / fail, first-try and
   effective rates, latency distribution, heal count, catalog-read loop-breaker
   events, MCP breaker events, and failure buckets.
3. Restore OpenClaw to its production tools-mode configuration after the Hermes
   run, then run same-day OpenClaw and Direct baselines on the same 75-case
   corpus, same host, same model/profile, same proxy profile, same self-heal
   policy.
4. Compare final lanes only from clean runs. Exploratory contaminated runs stay
   as diagnostic evidence, not score evidence.

Review follow-ups to close before publishing final lane claims:

- **`toolsObserved` telemetry parity.** Hermes audit must populate
  `toolsObserved[]` from the same correctness source as Direct/OpenClaw:
  Composer bridge-tool callbacks, with Hermes progress events used only as
  augmentation.
- **MCP breaker tuning.** Repeated validation 400s should back off or quarantine
  the offending run so one bad insert-family case cannot contaminate later
  cases via transient MCP circuit-breaker trips.
- **Contract probe.** Before each full corpus run, probe the live assistant mode
  and MCP surface: catalog hash, expected tool IDs, lean `nodeCatalog` shape,
  `catalog_read` availability, and required parameter schema for representative
  parameterized node kinds.

## Why tools mode wins

Three concrete reasons traced to OpenClaw 2026.5.22's runtime contract:

1. **The model dispatches against an explicit schema.** Tools-mode exposes `tool_search/describe/call` as discrete control verbs. The model emits `tool_call({id: "tree_draft_wrap_node", args: {...}})` directly — one OpenAI-style function call, one round trip to the catalog. Schema for `tool_call`'s args is on the tools[] list; the model can self-correct from the validator's error envelope.

2. **Code mode requires the model to write JavaScript that wraps the call.** The model must emit `tool_search_code({code: "const r = await openclaw.tools.call('tree_draft_wrap_node', {...}); return r;"})`. That's two indirections: (a) recognize that `openclaw.tools.call` is the active binding (NOT `tools.call`, NOT `window.openclaw.tools.call`, NOT a mock `const tools = {...}`), (b) author syntactically-valid JS body with correct `await`/`return`. Cosmos-reason2-8b reliably gets (a) wrong unless told explicitly (we measured this — see the unfixed-primer numbers below).

3. **Code mode fails open without runtime backpressure.** When the JS body throws (`ReferenceError`, `TypeError`, `Unknown tool id`), OpenClaw's per-turn budget consumes a retry but the per-conversation history we maintain doesn't see the failed tool call because it never made it to `body.toolCalls`. The model loops inside one OpenClaw invocation; the bridge can't break it across composer turns because there are no composer turns to break across. Tools mode produces normal function-call attempts whose failures land in the response envelope, where they're observable and recordable.

## What's in each sample

### Tools mode — 12 real cases (smoke killed by upstream stack crash after case 13)

| Case | Time (s) | Result |
|---|---|---|
| P1_wrap_root_specific | 27.8 | ❌ fail |
| P2_scene_add_specific | 24.4 | ❌ fail |
| P3_tree_insert_runtime_obj_specific | 99.5 | 🟡 soft-pass |
| WRAP_root_generic | 35.8 | ❌ fail |
| WRAP_root_medium | 213.4 | ✅ **pass** |
| SCENE_add_generic | 49.3 | ✅ **pass** |
| SCENE_add_medium | 41.8 | ✅ **pass** |
| PnP_02_scene_graspable | 23.5 | ✅ **pass** |
| PnP_03_scene_pick_pad | 27.1 | ✅ **pass** |
| PnP_04_scene_place_pad | 161.3 | ❌ fail |
| PnP_05_tree_root | 55.1 | ✅ **pass** |
| PnP_06_approach | 161.6 | ❌ fail |

Median latency ~50s; P95 ~213s; the 161s failures are model-loop-to-budget cases. Stack crashed at PnP_07 because killing the persistent OpenClaw gateway (PID 1957) to apply the `tools.toolSearch.mode = "tools"` config change had a cascade effect: the gateway didn't auto-respawn, the bridge's CLI shell-out fell back to embedded mode, and several invocations corrupted the composer's per-program-load state via repeated scene-object adds.

Recovery procedure (now codified in `/tmp/flip-to-tools-mode.sh` and validated): patch openclaw.json → restart gateway via `openclaw gateway --allow-unconfigured --bind loopback --auth token` → bounce bridge with `OPENCLAW_ASSISTANT_TOOL_SURFACE=tools`. Future runs should use the launcher path that ships this default.

### Code mode — 31 real cases (smoke killed per the 10-case rule at case 10; runner continued to case 31 before exit)

Failure breakdown by mechanism (from live OpenClaw logs in `/sandbox/.openclaw/logs/gateway-persistent.log`):

- **15 of 22 fails: model called a tool NOT in the MCP allowlist** for that prompt (e.g. P1 prompt asked for `wrap_node`, model emitted `tree_draft_insert_node` which was blocked).
- **4 of 22 fails: model emitted syntactically wrong JS body** even after the corrected primer (e.g. `const tools = {...mock...}; await tools.call(...)` recursive stack overflow; one case `window.openclaw.tools.call(...)` reference error).
- **3 of 22 fails: model picked the right tool but reached OpenClaw's per-invocation budget (15 turns) on retries.**

The four PASSES (`FALLBACK_retry_specific`, `PnP_02_scene_graspable`, `PnP_03_scene_pick_pad`, `PnP_05_tree_root`) prove code mode IS functional for cosmos-reason2-8b — when the prompt is simple enough that the model picks the right tool first try AND constructs the JS body without an editor-style fallback pattern, dispatch succeeds and state mutates correctly. The five SOFT-PASSES (state mutated but text-assertion missed) corroborate: the dispatch path works end-to-end.

The signal is not "code mode is broken." It's "code mode demands a model better-than-cosmos at constructing JS bodies on the first attempt, AND a model with better tool-selection over MCP-allowlist constraints." Both gaps shrink with a stronger model.

## The corrected-primer history (load-bearing for reproducibility)

The first 10-case code-mode smoke we ran (~09:13 UTC) used the WRONG primer that taught `await tools.<tool_name>(...)` and `tools.call("<name>", ...)` based on the QuickJS `code-mode.worker.js` file we initially inspected. That file is NOT the active execution path on OpenClaw 2026.5.22 with cosmos-reason2-8b. The active path is a Node subprocess where the only available bridge is `openclaw.tools.call / .search / .describe`.

Live OpenClaw logs from that run (saved as evidence in commit `2bcea45`'s message body):

```
[tools] tool_search_code failed: ReferenceError: tools is not defined
  raw_params={"code": "const result = await tools.scene_draft_add_object({...}); return result;"}
[tools] tool_search_code failed: ReferenceError: window is not defined
  raw_params={"code": "... window.openclaw.tools.call(tool, args) ..."}
[tools] tool_search_code failed: RangeError: Maximum call stack size exceeded
  raw_params={"code": "const tools = {call: async (n, a) => tools.call(n, a)}; ..."}
```

The model, given a primer naming a non-existent bridge, **invented mock implementations** to fill the gap. Those mocks failed in predictable ways: missing globals, mock methods calling themselves recursively, etc. Every smoke result under the wrong primer is contaminated and IS NOT a measure of model quality — the dispatch never reached the manyforge catalog.

The corrected primer (commit `2bcea45`) teaches:

```js
const r = await openclaw.tools.call("tree_draft_wrap_node", {
  targetName: "pick_and_place",
  wrapper: {id: "repeat", name: "outer_repeat", params: {max_iterations: 3}}
});
return r;
```

…explicitly DO-NOT-WRITE-tools.\*, DO-NOT-WRITE-window.\*, DO-NOT-define-mock-tools, with the OpenClaw-emitted error messages quoted so the model recognizes failure-mode fingerprints.

## Phase 3 gate verdict

Per [THREE-LANE-MIGRATION-PLAN.md](./THREE-LANE-MIGRATION-PLAN.md) Phase 3, the gate is: **OpenClaw native lane achieves ≥46/66 (~70%) on a clean discovery-surface run.**

- Code mode: 29% effective on 31 real cases — extrapolates to ~21/74 = far below gate.
- Tools mode: 58% effective on 12 real cases — extrapolates to ~43/74 ≈ 65% — within striking distance of the gate but not over it.

**Decision**: tools mode is the production OpenClaw default; the archived `archive/openclaw-plugin-attempt-2026-06-02/` artifacts remain available as a rollback path until Phase 5. The gap to the 70% gate is investigated in the multi-model bake-off (deferred): cosmos-4B, qwen3.6-35B-NVFP4, and any successor cosmos variants.

## What changes in the production stack

1. **OpenClaw lane provisioner** (`setup-manyforge-assistant.sh` + sandbox openclaw.json template): set `tools.toolSearch = {enabled: true, mode: "tools"}` by default.
2. **Bridge env default** (`start-openclaw-assistant-bridge.sh`, `scripts/lib/assistant.sh`): `OPENCLAW_ASSISTANT_TOOL_SURFACE=tools` is the new default. Operators can override to `code` for the multi-model bake-off.
3. **Proxy env default** (`scripts/lib/assistant.sh`): `OPENCLAW_PROXY_TOOL_SURFACE=tools` so drift check is sensitive on the production path.

The dual-mode infrastructure stays — we ship both primers, both detection paths, both env conventions. Only the default flips.

## Version tuple

| Component | Version |
|---|---|
| Model | `nvidia/Cosmos-Reason2-8B` |
| OpenClaw | `2026.5.22` (NemoClaw `lkg`) |
| NemoClaw | `lkg` (`v0.0.55` + commits) |
| vLLM | per cosmos profile |
| Proxy profile | `compat` |
| Bridge | three-lane-migration branch HEAD `2bcea45` |

## Next-step short smoke (Phase 3 follow-up, deferred to multi-model bake-off)

Run 5 cases × 4 scenarios on **qwen3.6-35B-NVFP4-MTP-2 FP8KV** (the 93/100 tool-eval-bench winner per memory):

1. OpenClaw lane, code mode, 5 cases
2. OpenClaw lane, tools mode, 5 cases
3. Direct lane, 5 cases
4. (optional) Hermes lane scaffold-only (no actual dispatch)

Compare first-try rate to the cosmos-reason2-8b numbers above. If 35B's first-try rate in code mode is ≥40% while tools mode is similar, the model-quality gap is the bottleneck and we should make the lane default model-dependent.

## 35B NVFP4 NVIDIA bake-off (2026-06-03) — deferred

Ran 5-case OpenClaw tools-mode probe against `qwen3.6-35b-a3b-nvfp4-nvidia` (NVIDIA's ModelOpt v0.44.0 NVFP4 quant + MTP K=3 per the Spark recipe, on the current Thor stack with the launcher's start-model.sh path). Result: **0/5** vs cosmos 4/5 on the identical case set (P1_wrap_root_specific, WRAP_root_medium, SCENE_add_generic, SCENE_add_medium, PnP_05_tree_root).

### What we observed

All five cases ran to OpenClaw's per-invocation budget and failed with the same root cause: the 35B model never reached a successful `tool_call` against the manyforge catalog. The proxy capture of one in-flight assistant turn quoted the model verbalizing:

> "I've been stuck trying to find the right tool name for manyforge tools. Every tool name I've tried returns 'Unknown tool id'. The only tool that seems to work is from the openclaw core namespace, like..."

The model is using tools mode correctly (`tool_search`, `tool_describe`, `tool_call` all appear in its emissions). It just can't bridge from the tool-search results to a valid `tool_call({id: "<manyforge-id>", args: ...})` invocation. Most likely cause: the `--reasoning-parser qwen3` setting routes the model's "I should call tree_draft_wrap_node" thinking into a separate channel that doesn't reach the bridge as a tool-call attempt, while the visible `content` channel carries the model's confusion narrative.

### Why this is NOT a tools-mode-vs-code-mode signal

The 0/5 vs cosmos 4/5 gap is **stack-level integration** (tool-call parser + reasoning parser + chat template for the qwen3.6-35B-A3B-NVFP4 model on the Thor build of vLLM), not a discovery-surface issue. The 56/66 (84.8%) baseline noted in `serving/config.sh` for this same profile was measured on an earlier stack configuration that we have not been able to reproduce in this branch. Resolving the gap requires:

1. Confirm `qwen3_coder` tool-call parser is the right pick for `qwen-fixed-froggeric.jinja` + this NVFP4 weight set (the qwen3.6-35B template tree has diverged across releases).
2. Probe whether `--reasoning-parser qwen3` is interfering with tool-call emission (try `--reasoning-parser none` for a control run).
3. Validate the served `tools[]` schema against what `tool_search` returns — Phase 3's "Unknown tool id" investigation surfaced that the OpenClaw catalog ID can differ from the bare manyforge name; the 35B may be picking the wrong shape.

### Decision

**Defer the 35B bake-off** to a follow-up cycle. Cosmos-reason2-8b stays the production model anchor with tools mode as the OpenClaw lane default. The `serving/config.sh` 56/66 number remains the aspirational target — we know it's achievable in principle, just not on this stack as currently wired.

Production decision is unchanged from the cosmos-only data:
- OpenClaw tools mode > OpenClaw code mode (58.3% vs 29.0% effective on cosmos)
- Tools mode is the production default
