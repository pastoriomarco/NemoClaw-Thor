# Thor smoke-corpus sweep — 7 model runs incl. gemma-QAT (2026-06-07)

**Hardware:** Jetson AGX Thor @ **120W** (same power as all prior + Orin benchmarks → directly comparable).
**Method (identical across runs):** full 75-case `smoke_corpus.yaml` (9 future-tier skipped), `--self-heal` ON
(PnP chain de-confounded; heal harness exercised; heal-info captured in report JSON), recovery-turn OFF.
**Switching:** vLLM/QAT loads via `launch.sh restart` (drops page-cache — load-bearing for the Thor vLLM
memory leak); GGUF reloads model-only. Reports: `/tmp/smoke-sweep/<profile>.json`.

## Scoreboard (ranked by effective rate)

| # | Profile | Lane | first-try | effective | soft | fail | heals | PnP chain | decode | wall-clock |
|---|---------|------|-----------|-----------|------|------|-------|-----------|--------|------------|
| 🥇 | **gemma4-12b-it-gguf (QAT)** | GGUF | 71.2% | **78.8%** | 5 | 14 | 5 | **14/19** | 20.6 med / **38 max** | medium |
| 🥈 | **qwen3.6-35b-a3b-nvfp4** | vLLM | 72.7% | 77.3% | 3 | 15 | 3 | 16/19 | ~17 (MTP peaks 41) | slow (30–300s) |
| 🥉 | **gemma4-12b-it-gguf (plain)** | GGUF | 62.1% | 71.2% | 6 | 19 | 9 | 10/19 | 17.0 | medium |
| 4 | **cosmos-reason2-8b** | vLLM | 43.9% | 59.1% | 10 | 27 | 3 | 16/19 | ~12 aggr | medium |
| 5 | **nemotron3-nano-4b-gguf** | GGUF | 43.9% | 57.6% | 9 | 28 | 9 | 10/19 | **55.5** | fast |
| 6 | **cosmos-reason2-8b-gguf** | GGUF | 42.4% | 54.5% | 8 | 30 | 4 | 15/19 | 20.9 | slow |
| 7 | **nemotron-omni-30b-a3b-nvfp4 (think-OFF)** | vLLM | 25.8% | 42.4% | 11 | 38 | 7 | 12/19 | ~14 aggr | fastest (5–30s) |

## QAT verification (the headline of this run)

gemma-QAT replaces plain gemma as the main model (commit `7dcb689`). Verified against plain (Thor) + Orin:

| metric | plain (Thor) | **QAT (Thor)** | Δ | Orin QAT |
|--------|-------------|----------------|---|----------|
| first-try | 62.1% | **71.2%** | +9.1 | 65.2% |
| effective | 71.2% | **78.8%** | +7.6 | 74.2% |
| PnP chain | 10/19 | **14/19** | +4 | 14/19 ✓ exact |
| heals | 9 | 5 | −4 | ✓ |
| decode | 17.0 t/s | 20.6 med / 38 max | +21% med, +35% ceiling | 29 (probe) |

- **Chain 14/19 exactly matches Orin's QAT.** QAT held **PnP_01–13 first-try, zero heals**, then hit the deep-chain compaction wall ~step 16 (delayed ~7 steps vs plain). The speed→chain-survivability thesis is confirmed by intervention, not just correlation.
- **End-to-end gain is larger on Thor** (+7.6 effective vs Orin's +3.0). **QAT gemma is now the highest effective of the whole sweep (78.8%)** — edging qwen-35b — as a lightweight GGUF (no vLLM, ~⅓ the params). **Verdict: keep QAT.**
- Speed: median +21% is context-dragged by long chain decodes; the clean short-context ceiling (~38 t/s) is where the ~2× shows, matching Orin's probe regime.
- **Refines the Orin sink-law:** plain gemma & qwen both decode ~16–17 t/s yet plain sank ~step 11–15 and qwen rode to 20 → sink ≈ (session-size × per-turn-length) ÷ (decode × compaction-budget), not decode alone. Bounded thinking + spec-decode keep turns compact.

## Cross-cutting findings

1. **Reasoning mode is decisive and corpus-specific.** omni **think-OFF cratered (42.4%)** — fast wrong tool-calls + 18–54-entry tool-loops → 502s (an *inversion* of NVIDIA's think-off tool-regime claim). cosmos **think-ON** over-reasoned → timeouts. qwen's **bounded** think-on (512 tok) is the sweet spot.
2. **The PnP chain splits the field**, all consistent with the compaction-sink law: agentic specialists (qwen, cosmos-vLLM 16/19; cosmos-gguf 15/19; QAT 14/19) ride the heal-ballooned session; slow plain GGUF (gemma/nemotron 10/19) sink in the back-half to `chat HTTP 502/-1` **compaction-timeouts** (the heal splices the full golden transcript → session balloons to 100–165 turns → openclaw must compact in-budget → slow decode can't).
3. **Architecture × Thor bandwidth:** nemotron-4b (Mamba) decode +42% vs Orin (39→55) — bandwidth-bound; attention models barely move. plain gemma is platform-invariant (71.2% both boxes).
4. **Self-heal harness:** fired on every chained fail across all 7 runs, **healed every time invoked**; cascade-prevention held. `dropped=N` doubles as a diagnostic (0 = pure timeout; 18–54 = tool-call flooding, omni).

## Cases NO model solved — analysis (the actionable core)

Across all 6 base Thor models, **4 cases failed (hard) on every one**. Of those, QAT rescued 1, leaving **3 truly-universal failures** (every model incl. QAT, both platforms). Verdict on each, with evidence:

### 1. P2_scene_add_specific — 🐛 HARNESS BUG (not the model). **FIXED 2026-06-07.** Was unwinnable by any model.
**Fix shipped** in `smoke_corpus_runner.py` `capture_state`: merge `program.sceneResources` (the draft objects a `scene_draft_*` op actually mutates) into `scene.objects`, normalized to the legacy assertion keys (`box_dims`/`position`/`id`), since the runtime `/api/scene/state` layer the assertion read never reflects an un-materialized draft and `state_after` resolution is alias-unaware. P2 now passes end-to-end through QAT gemma; the 6 other scene-asserting cases showed no state_after regression (2 unrelated temp=1.0 tool-emission/answer flips). Root-cause detail below:
*Prompt:* "add a box of size 1.0, 0.02, 0.25 at position 0.0, -0.15, 0.125". *Assertion:* `scene.objects[*].shape.box_dims contains [1.0,0.02,0.25]`.
**Evidence (direct tool test, 3×, bypassing the model):** `scene_draft_add_object` returns HTTP 200 / success and the object + dims persist in the **draft** — but the assertion reads the **runtime** `scene.objects` snapshot (from the viz/cycle layer), where a draft-only add appears only eventually (0 objects in 2/3 probes, populated in 1). So the box never deterministically appears at the asserted path **regardless of what the model does**.
**Fix:** assert against the draft scene, or have the runner apply/cycle the draft before `state_after`. (Separately, P2b's cylinder case is blocked by a real composer `diameter→radius` normalization gap — already filed.)
**Could I solve it as the model? No** — no tool call populates the asserted runtime path for a draft op. It's a false failure; fix the harness.

### 2. INSERT_position_first_specific — 🎯 GENUINE CAPABILITY. Not a harness bug. I'd very likely solve it.
*Prompt:* "insert a wait_for_signal_bool node as child index 0 of pick_and_place that waits for start_pick=true". *Expected:* `tree_draft_insert_node(parentName=pick_and_place, position=0, node.id=wait_for_signal_bool, params{key:start_pick, expected_value:true})`; `state_after children[0].id == wait_for_signal_bool`.
**Why models fail:** they emit the wrong node at index 0 (got `command_gripper`) or fumble the schema (a *novel* node kind + a position arg triggers long catalog_read recovery loops on small models; there's also a `signal_id/expected` vs canonical `key/expected_value` schema-drift trap).
**Could I solve it? Yes, very likely** — the instruction is precise and unambiguous. Read the live catalog for `wait_for_signal_bool`'s param schema, emit one `tree_draft_insert_node` at `position:0` with `key/expected_value`. The schema-drift trap is exactly what checking the live manifest avoids. This is a real small-model weakness, not an unsolvable task.

### 3. CUR_runtime_remove_then_restore_graspable — 🎯 GENUINE CAPABILITY (hard, domain-specific). Not a harness bug. I'd probably solve it with care (~70%).
*Prompt:* "at the end of pick_and_place, remove graspable from the runtime scene and then upsert it again at its initial scene pose and size". *Expected:* `scene_inspect` + insert `remove_collision_object` node + insert `upsert_collision_object` node (box, `box_dimensions_m=[0.1,0.01,0.01]`, `pose.frame_id=universe`). *Forbidden:* `scene_draft_remove_objects`, `scene_draft_update_object`.
**Why models fail:** they fire the **forbidden draft-time tools** (`scene_draft_remove_objects`) instead of inserting **runtime behavior-tree nodes** (`remove_collision_object`/`upsert_collision_object`), and skip `scene_inspect` (needed to read graspable's initial pose+size to "restore" — those values aren't in the prompt). The trap is conflating "runtime scene mutation" (tree nodes that execute during the cycle) with "draft scene edit".
**Could I solve it? Probably, with care** — IF I correctly read the catalog to distinguish runtime nodes from draft tools, `scene_inspect` first to learn the size/pose, then insert both nodes and avoid the forbidden tools. It hinges on the runtime-vs-draft distinction being clear from the catalog; it's the genuinely hard one and a legitimate test of multi-step + domain reasoning.

### Bottom line
- **1 of 3 is a harness bug** (P2) — a *false* failure that no model (incl. me) can pass; fix the corpus/runner. **Fix it.**
- **2 of 3 are genuine model cases** (INSERT_position, CUR_runtime) that correctly expose small-model weakness (schema-fumbling; runtime-vs-draft confusion). They are **not** corpus bugs — "fixing" them means a more careful/capable model, not a corpus change. I'd likely solve INSERT_position cleanly and CUR_runtime with care.
- Thor's 6-model set already solved **7 of Orin's 11** universal failures (qwen/cosmos-vLLM rescued the chain-infra + move/restraint cases) — the "universally unsolved" set is **not fundamental**; it collapses to the corpus bug + a 2-case capability core once stronger/faster models are in the pool.

## Recommendations
- **Composer-assistant production:** **gemma-QAT** is now the best lightweight all-rounder (78.8%, no vLLM, ~2× faster than plain); **qwen-35b** if you want the top chain (16/19) and latency is tolerable; **cosmos-reason2-8b (vLLM)** as the balanced agentic default.
- **Fast control loops / long context:** nemotron-4b-gguf (55 t/s, Mamba-flat).
- **Don't ship:** cosmos-gguf for isolated work; omni-think-off as-is (re-test reasoning-ON).

## Follow-ups
1. ✅ **Fix P2 — DONE** (`capture_state` merges the draft scene into `scene.objects`). Broader latent gap worth a follow-up: `state_after` path resolution is **alias-unaware** (only `args_contain` is) — other legacy `box_dims`/`position` state asserts would mismatch the SI-suffixed data if not on the draft path; consider making the resolver alias-aware.
2. **DFlash spec-decode** for qwen-35b (saved to memory) — potential ~2× with a tool-calling validation pass.
3. **omni reasoning-ON re-test** — think-off is the wrong regime for this corpus.
4. **Mine bridge/proxy logs** for poison/compaction/proxy-reflection harness counts (Orin-depth).
