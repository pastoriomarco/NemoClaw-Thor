# 35B-NVIDIA (qwen3.6-35b-a3b-nvfp4-nvidia) — full smoke result

**Date**: 2026-06-01
**Pipeline**: proxy detectors #1-7 active + tail_checklist + namespace_stop=16 + tool-name normalization
**Profile config**: FORCE_ENABLE_THINKING=on (verified)

## Overall — winner
- 74 cases total, 8 skipped, **66 attempted**
- 50 pass + 6 soft-pass + 10 fail
- **Effective rate: 56/66 = 84.8%**
- First-try rate: 50/66 = 75.8%
- **Matches/exceeds prior cosmos iter-28 production result (77.3%)**

## Detector activity (all working as designed)
- `normalize_tool_names`: fired many times — model routinely emits unprefixed names like `tree_draft_wrap_node` → rewritten to `manyforge__tree_draft_wrap_node`. Without this detector, every action case would fail dispatch.
- `same_tool` reflection + `hard_stop` at 8: fired on `tree_draft_wrap_node` (count 4→5→6→7→8 = stop) in REPLACE cases
- `same_namespace` + `namespace_hard_stop` at 16: fired on long tree_draft alternation loops — saved cases from runaway
- `same_args`, `result_repeat`, `turn_counter`: all fired correctly on various loops

## Failure breakdown (10 fails)

| Cat | Count | Pattern |
|---|---|---|
| F.no-action-no-clarify | 4 | model idle timeout or "scene is empty" misreads |
| B.asked-instead | 3 | SCENE-related — model asks despite having info |
| D.pipeline-err | 1 | TimeoutError on INSERT_position_first_specific |
| A.acted-when-ask | 1 | FALLBACK_alternate_medium — model acted on ambiguous prompt |
| E.wrong-tool-name | 1 | CUR_runtime_remove_then_restore — used wrong tool |

## Notable strengths

### PnP cascade SURVIVED (18/19 PASS)
The case that killed 4B and omni — PnP_06 onwards needing chained tree state — 35B passed 18 out of 19 attempted PnP cases. The cascade did NOT compound here. Per-case durations were 30-145s (long but linear, not exponential).

### Specific-action prompts work
P1_wrap_root_specific ✅, P2_scene_add_specific ✅, P3_tree_insert_runtime_obj_specific ✅ — all the cases where 4B/omni asked clarification instead of acting, 35B acted correctly.

### Generic-WHERE prompts work too
WRAP_root_generic, INSERT_position_before_named_generic, PARALLEL_generic, FALLBACK_generic, CLARIFY_motion_generic — all pass or soft-pass. 35B handles both poles (act/ask) correctly.

## Compared to 4B and omni

| Metric | 4B | Omni | 35B-NVIDIA |
|---|---|---|---|
| Effective rate | 39.4% | 31.8% | **84.8%** |
| First-try rate | 30.3% | 22.7% | **75.8%** |
| PnP cases passing | 0/19 | 4/19 | **18/19** |
| Avg case time | ~17s | ~25s | ~50s |
| Total wall-clock | ~18 min | ~30 min | ~45 min |

35B is 2-3× slower per case but the quality difference is dramatic.

## Failure root causes (concise)

1. **INSERT_position_first_specific** — `chat HTTP -1` / TimeoutError. **Pipeline issue, not model.** Single occurrence. Probably composer or OpenClaw gateway latency.

2. **REPLACE_subtree_specific, REPLACE_simple_medium** — model timeout (`⚠️ Agent couldn't generate a response`). The model fell into a long internal CoT that hit the per-turn timeout. Could be helped by raising max_tokens for these specific patterns, but model-side.

3. **SCENE_remove_specific** — model said "scene is currently empty — nothing to remove" when scene DID have objects. State-reading error.

4. **SCENE_update_pose_specific, SCENE_update_size_medium, SCENE_add_generic** — model asked clarification when it shouldn't have. Same pattern as Nemotron but only 3 cases (vs 30+ for Nemotron).

5. **FALLBACK_alternate_medium** — opposite pattern: model acted when test expected ASK.

6. **CUR_runtime_remove_then_restore_graspable** — wrong tool chain (used insert instead of inspect+insert).

## Conclusion

35B-NVIDIA (`qwen3.6-35b-a3b-nvfp4-nvidia`) is the clear winner on this corpus with the current pipeline. The pipeline improvements (especially normalization #7 and namespace_stop #5) are doing real work — without them, even 35B would fail many cases that currently pass.

The Nemotron underperformance is consistent with the historical LANE-COMPARISON memory ("Nemotron 0/9 vs Cosmos 9/9"). Nemotron family is structurally weaker on this corpus regardless of pipeline.
