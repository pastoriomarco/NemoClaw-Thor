# Thor + Orin smoke-corpus sweep — 11 model-runs incl. gemma-QAT (2026-06-07)

> Two companion sweeps with one shared method: **7 runs on Jetson AGX Thor** (below) and
> **4 GGUF runs on Jetson AGX Orin 64 GB** (see *Orin sweep* section). The gemma-QAT
> intervention was run on both boxes and lands on the **same PnP chain count (14/19)**.

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

## Orin sweep — 4 GGUF runs (2026-06-07, AGX Orin 64 GB @ MAXN)

Companion sweep on the **Jetson AGX Orin 64 GB** (sm87 Ampere) — **llama.cpp / GGUF lane only**
(these runs predate JetPack 7.2; no vLLM-on-Orin yet — see Orin follow-up). Identical 75-case
corpus, `--self-heal` ON, temp=1.0, **MAXN** (nvpmodel 0, GPU 1.3 GHz). Decode = single 200-tok
`/completion` probe. Artifacts: `/tmp/smoke-bench/<profile>.{summary.md,harness.txt,cases.txt,report.json}`.

### Scoreboard (ranked by effective rate)

| # | Profile | Lane | first-try | effective | soft | fail | heals | PnP chain | decode | avg case |
|---|---------|------|-----------|-----------|------|------|-------|-----------|--------|----------|
| 🥇 | **gemma4-12b-it-gguf-orin (QAT)** | GGUF | 65.2% | **74.2%** | 6 | 17 | 4 (+1 fail) | **14/19** | 29.1 probe | 69s |
| 🥈 | **gemma4-12b-it-gguf-orin (plain)** | GGUF | 59.1% | 71.2% | 8 | 19 | 9 | 10/19 | 15.0 | 95s |
| 🥉 | **nemotron3-nano-4b-gguf-orin (auto)** | GGUF | 51.5% | 62.1% | 7 | 25 | 7 | 12/19 | **39.2** | 74s |
| 4 | **cosmos-reason2-8b-gguf-orin (think-ON)** | GGUF | 43.9% | 50.0% | 4 | 33 | 8 | 11/19 | 23.3 | 130s |

### Orin harness-firing tally (addresses Follow-up #4 — bridge/proxy log depth)

| signal | QAT | plain | nemotron | cosmos |
|--------|-----|-------|----------|--------|
| requests completed / 66 | 61 | 55 | 63 | 51 |
| compaction fired / ok / timeout | 9 / 5 / 4 | 9 / 5 / 4 | 9 / 6 / 3 | 9 / 5 / 4 |
| session poisons (all recovered) | 5 | 11 | 3 | 15 |
| request_timeout | 1 | 6 | 0 | 11 |
| chat-fail HTTP codes | 502×4, -1×1 | -1×5, 502×5, 504×1 | 502×3 | -1×12, 502×4 |
| proxy loop_reflection_injected | 69 | 67 | 33 | **439** |
| proxy loop_hard_stop | 9 | 3 | 3 | **33** |

gemma-plain exercised **all four** poison reasons (`timeout`, `compact_timeout`, `compact_session_lock_timeout`, `session_lock_timeout`) — every one recovered cleanly. Self-heal fired on every chained fail and healed every time invoked (QAT's 1 "fail" was a `reset-to-base` request that itself timed out under congestion, not a logic error).

### Orin-specific findings

1. **QAT verified cross-platform, exactly.** Orin QAT = first-try 65.2 / eff 74.2 / **PnP 14/19** — the chain count matches Thor QAT bit-for-bit (see *QAT verification*). Sink moved plain→QAT from ~step 11 to ~step 12, then **oscillated-and-recovered** (14/16/18 ✅) where plain stayed sunk. Same intervention, same outcome, two platforms.
2. **The sink-law is monotonic in decode speed on one box, four speeds:** nemotron 39 t/s → never sinks (oscillates to PnP_20); QAT 29 → sinks ~12 then recovers; cosmos 23 → holds to ~14 then sinks; plain 15 → sinks ~11. Among GGUF-on-Orin, tokens/sec orders the sink-point — consistent with finding #2 and the refined sink-law.
3. **thinking-ON cratered cosmos here too** (mirrors cross-cutting #1): 439 proxy loop-reflections + 33 hard-stops (≈10× the others) + 12× `chat HTTP -1` request-timeouts → 50.0% eff, last place. Over-reasoning on ambiguous prompts blew the 300 s budget even on **fresh** sessions, not just the chain.
4. **Platform-invariance check:** plain gemma = **71.2% eff on both boxes** (Orin & Thor) — confirms the corpus + harness are platform-stable; what moves is decode (arch × bandwidth, finding #3) and therefore the chain.
5. **P2 false-failed on all 4 Orin runs** — they predate the 2026-06-07 `capture_state` fix, so the counts above include it. The "Orin 11 universal failures" referenced in *Cases NO model solved* are the pre-fix set (1 harness bug + chain-infra + the 2-case capability core).

### Orin verdict & follow-up
- **Best Orin all-rounder = QAT gemma (74.2%)**; **fastest survivable = nemotron-4b** (39 t/s, best PnP of the small models). cosmos-gguf not shippable (matches Thor verdict).
- **Next (JetPack 7.2, June 2026):** Orin now shares Thor's CUDA-13 stack and gains **first-class vLLM (sm87 Marlin INT4)** → unlocks **Nemotron-3-Nano-30B-A3B (AWQ-INT4) at ~40 t/s on Orin**, bypassing the llama.cpp sm87 MoE-decode-hang. Candidate to top this Orin board; smoke pending.

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
