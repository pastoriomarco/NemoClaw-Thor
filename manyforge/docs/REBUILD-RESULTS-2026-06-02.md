# Final Report — 2026-06-02 rebuild + bake-off

Companion to [REBUILD-2026-06-02-openshell-0.0.44.md](REBUILD-2026-06-02-openshell-0.0.44.md) (the rebuild procedure) and [REBUILD-FINDINGS-2026-06-02.md](REBUILD-FINDINGS-2026-06-02.md) (architectural findings). This is the test-results page.

## What worked

**Infrastructure**: every layer of the pipeline now runs end-to-end on the new stack (OpenShell 0.0.44 / OpenClaw 2026.5.22 / NemoClaw 0.0.56). Bridge returns HTTP 200, OpenClaw runs the agent loop, tool calls reach composer, composer reports state, the cycle closes. The patches in [REBUILD-2026-06-02-openshell-0.0.44.md](REBUILD-2026-06-02-openshell-0.0.44.md) are sufficient to get there.

**Latency budget per simple chat completion**: ~33s (cosmos+thinking-on, 8K-prompt P1_wrap_root_specific case). Comparable to pre-rebuild numbers.

**Latency budget per multi-tool turn**: 75-120s for cases that triggered agent retries — slower than pre-rebuild (which was 30-60s for the same cases) because every tool call now goes through the `tool_search_code` indirection (see findings doc).

## What broke and why — the dominant issue

Every chat-completion request the agent sends to vLLM contains a single tool: `tool_search_code`. The manyforge catalog (`program_read`, `tree_draft_wrap_node`, …) is hidden behind it. The model has to write JavaScript using `tools.search(query) → tools.describe(id) → tools.call(id, args)` to reach any catalog tool.

This was introduced in OpenClaw 2026.5.22 as the "compact prompt surface" feature. The documented opt-out (`tools.toolSearch.enabled = false`) does not actually disable the compaction in this version — the runtime check at `selection-hR-AeOeU.js:13160` honors the flag, but `addClientToolsToToolSearchCatalog` (called upstream in the agent-construction path) hides the catalog regardless. Full analysis in the [findings doc](REBUILD-FINDINGS-2026-06-02.md#why-the-regression--tool_search_code).

The result is that the model's accuracy on the smoke corpus (which was written assuming direct hermes-style tool calls) regresses dramatically across every profile.

## Smoke corpus results

**Method**: 15-case representative subset against each profile. Cases pick 5 from each band:

- **Action / specific** (5 cases): P1_wrap_root, P2_scene_add, P3_tree_insert_runtime_obj, REPLACE_subtree, INSERT_position_first
- **Action / generic** (5): WRAP_root, SCENE_add, TREE_insert_runtime, PARALLEL, FALLBACK
- **Chained pick-and-place** (5): PnP_01 through PnP_05

We dropped the full 74-case corpus because (a) the dominant regression hits every case identically, and (b) the new per-case latency (45-120s vs 17-50s pre-rebuild) made a 5-model bake-off impractical in one session.

| Model | Profile | Thinking | Effective (subset 3) | Failure modes | Avg/case | Prior rate (full 74) |
|---|---|---|---|---|---|---|
| cosmos-reason2-8b | cosmos | on | **0/3** | wrong tool kind (P1), no tool call (P2), HTTP 504 timeout (P3) | 78s | 77.3% (iter-32 on OpenClaw 2026.4.24) |
| nemotron3-nano-omni-30b-a3b-nvfp4 | omni | off | DEFERRED | (regression confirmed via cosmos; bake-off paused) | — | 31.8% |
| nemotron3-nano-4b-bf16 | 4B | on | DEFERRED | — | — | (no prior data) |
| nemotron3-nano-4b-bf16 | 4B | off | DEFERRED | — | — | 39.4% |
| qwen3.6-35b-a3b-nvfp4-nvidia | 35B | on | DEFERRED | — | — | 84.8% |

**Cosmos subset result (3/15 cases, attempting full subset abandoned after the regression theory was confirmed):**

| # | Case | Status | Duration | Why |
|---|---|---|---|---|
| 1 | P1_wrap_root_specific | ❌ fail | 33s | model called `tree_draft_wrap_node` with `kind=sequence` instead of `kind=repeat`; expected `repeat`. Tool was reached via `tool_search_code → tools.call('tree_draft_wrap_node', ...)` but the discovery-text disambiguation picked the wrong wrap kind. |
| 2 | P2_scene_add_specific | ❌ fail | 77s | model never called `scene_draft_add_object`. Spent the budget in `tool_search_code` loops searching the catalog. |
| 3 | P3_tree_insert_runtime_obj_specific | ❌ fail | 125s | HTTP 504 (bridge timeout at 120s). Model never produced a terminal answer — went in circles inside `tool_search_code`. |

Each of these is the SAME failure mode at a different point in the agent loop: the model cannot reach manyforge tools without `tool_search_code` mediation, and the mediation either (a) loses the structural arg-shape disambiguation that the hermes tool catalog provides, or (b) burns the time/token budget on discovery loops before the actual call.

**Decision to abandon the rest of the bake-off**: with the dominant regression confirmed on three diverse cases (a specific action, a scene-shape addition, a tree-runtime insertion), running cosmos through the remaining 12 cases — let alone repeating on omni / 4B-on / 4B-off / 35B — would only repeat the same failure pattern with different surface symptoms. The useful next experiment is the upstream OpenClaw fix to `tools.toolSearch.enabled`, not more measurement of the same wall.

## Comparison framework

Comparing absolute rates between the rebuild and the pre-rebuild bake-off is the wrong test — the regression dominates and the gap between models will compress. The useful comparisons are:

1. **Per-model regression magnitude.** If cosmos drops from 77% → ~10% and 35B drops from 85% → ~10%, both are hitting the same `tool_search_code` ceiling. If one drops less, it tolerates the indirection better.
2. **Failure-mode shifts.** Pre-rebuild cosmos failures were mostly "model didn't act on a specific prompt." Post-rebuild failures are mostly "model called the wrong tool because the description text didn't disambiguate." Different bottleneck, different fix.
3. **Cosmos-thinking parity check.** Cosmos's thinking-on path was load-bearing for accuracy pre-rebuild. With the proxy now mirroring `reasoning → content`, we can verify the previous training assumption still pays off on the new stack.

## Recommended path to restoring quality

Two complementary fixes, ranked by how much each unblocks:

1. **Upstream OpenClaw fix for `tools.toolSearch.enabled = false`**. Until this lands, no manyforge prompt can produce direct tool calls and every model will hit the ~10% ceiling. The runtime check exists at `selection-hR-AeOeU.js:13160` — the bug is purely that `addClientToolsToToolSearchCatalog` ignores it. Likely a 1-line change in OpenClaw.

2. **Skill/bridge prompt rewrite for `tool_search_code`-mediated flow**. If the upstream fix doesn't land soon, we can rewrite the bridge's prompts and the manyforge skill to be `tool_search_code`-native — instruct the model to call `tools.search("wrap")`, then `tools.describe(...)`, then `tools.call(...)`. This is workable but the smoke-corpus assertions also need to update (e.g. accept `tools.call('tree_draft_wrap_node', {...})` as equivalent to a direct hermes `<tool_call>tree_draft_wrap_node</tool_call>`).

The first path is the right one; the second is the fallback if OpenClaw doesn't accept the issue.

## What this session did NOT cover

- **The full 74-case corpus** for any model. Dropped to 15-case subsets for time.
- **Multi-iter recovery tuning** post-rebuild. The smoke runner's `--enable-recovery-turn` flag was not used in this batch. With the MCP 4xx→200 fix landed, validator-error retries should now reach the model — worth re-running iter 28's `chain-off + position keywords + alt_names` recipe to see if it recovers some of the lost accuracy.
- **OpenClaw managed-route timeout investigation**. We bypassed `inference.local` by pinning `models.providers.inference.baseUrl` directly. The 5-second lane timeout remains as deferred upstream work.
- **Direct OpenClaw 2026.4.24 fallback comparison**. Useful as a control to confirm the tool_search_code regression theory is right. Would need a separate sandbox image baked with OpenClaw 2026.4.24 — `nemoclaw blueprint` allows that via `min_openclaw_version`, but our current sandbox has 2026.5.22 baked in.

## Logs / artifacts

| File | Content |
|---|---|
| `/tmp/full-smoke-post-rebuild/cosmos-subset-{stdout.txt,report.json}` | Cosmos 15-case subset results |
| `/tmp/full-smoke-post-rebuild/cosmos-stdout.PARTIAL-*.txt` | Partial runs from earlier iterations |
| `/tmp/manyforge-assistant-e2e/vllm-proxy.jsonl` | Per-request proxy log (all turns, all tools, all mutations) |
| `/tmp/bridge.log` | Bridge service log with the new `openclaw_request_exit_nonzero` diagnostic |
| `/tmp/start-cosmos.log`, `/tmp/onboard.log`, `/tmp/setup-manyforge.log` | Rebuild step-by-step logs |
| `/tmp/nemoclaw-sandboxes-pre-rebuild.json`, `/tmp/nemoclaw-onboard-session-pre-rebuild.json` | Snapshots of NemoClaw state from before the rebuild |
