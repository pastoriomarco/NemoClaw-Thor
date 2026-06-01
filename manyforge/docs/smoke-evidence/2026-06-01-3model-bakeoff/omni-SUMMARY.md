# Omni (nemotron3-nano-omni-30b-a3b-nvfp4) — full smoke result

**Date**: 2026-06-01
**Pipeline**: proxy detectors #1-7 active + tail_checklist + namespace_stop=16 + tool-name normalization
**Profile config**: FORCE_ENABLE_THINKING=off (verified)

## Overall
- 74 cases total, 8 skipped, **66 attempted**
- 15 pass + 6 soft-pass + 45 fail
- **Effective rate: 21/66 = 31.8%**
- First-try rate: 15/66 = 22.7%
- **Worse than 4B (39.4%)** despite being a much larger model (30B-A3B vs 4B)

## Detector activity (all working)
- `normalize_tool_names`: fired many times — `tree_draft_*` / `scene_draft_*` / `program_read` / `scene_inspect` etc.
- `same_namespace` + `namespace_hard_stop` (cap=16): fired correctly, ended several runaway namespace loops cleanly
- `same_tool` + `hard_stop` (cap=8): fired correctly, broke runaway loops on `session_status` and `tree_draft_wrap_node`
- `result_repeat`: fired on repeated `catalog_read` calls (model fetching same catalog repeatedly)
- `same_args`, `turn_counter`: fired throughout, providing nudges
- `malformed_tool_call`: 0 fires (model emitted proper format)

## Failure breakdown (45 fails)

| Cat | Count | Pattern |
|---|---|---|
| **B. asked-instead-of-acting** | **30** | over-clarification on directive prompts (same as 4B) |
| E. dispatched-not-recognized | 11 | model picked wrong tool (e.g. `change_node_kind` instead of `replace_subtree`) |
| C. dispatched-but-state-wrong | 4 | tool ran but final state diverged from expected |

## Notable omni-specific behaviors

### Long PnP cascade (worst pattern)
The PnP series chains state across 20 cases. Once PnP_06 failed, subsequent cases inherited the broken state and all failed (PnP_06 through PnP_20 = 15 cases failed). 4B had the same issue but failures were 10-30s; omni's are 30-180s due to deeper loops.

### session_status loops
Omni frequently called OpenClaw's internal `session_status` tool 4-8 times before getting to manyforge tools. The `same_tool` hard_stop at count=8 fired multiple times to break these. This adds 30-80s per affected case.

### namespace_hard_stop save
Detector #5's hard_stop at namespace count=16 fired in cases where the model was alternating across the entire `tree_draft_*` family. Without it, those cases would have spun until the per-case timeout (244s) — saving 60-100s per fire. Verified working.

### Tool selection errors
Several REPLACE cases had omni emit `change_node_kind` when the expected tool was `replace_subtree`. The 4B had the same misclassification. Different failure-mode from 4B's "ask first" — here the model commits to a (wrong) tool confidently.

## Compared to 4B

| Metric | 4B | Omni | Diff |
|---|---|---|---|
| Effective rate | 39.4% | 31.8% | omni WORSE by 7.6pts |
| First-try rate | 30.3% | 22.7% | omni WORSE |
| Asked-instead-of-acted | 37 | 30 | omni slightly better |
| Pipeline errors | 0 | 0 | tied |
| Wall-clock | shorter | much longer (PnP cascade) | omni 2-3× slower |

## Diagnostic summary

The bigger model (omni, 30B vs 4B) didn't help. Both models share the over-clarification bias, and omni adds:
- Longer per-case latency (deeper internal CoT loops)
- More confidence in WRONG tool selection (E. category)
- Session_status confusion (calling OpenClaw's bootstrap tool repeatedly)

**This is conclusive evidence that the pipeline isn't the bottleneck.** Two different-sized models from the same family hit similar ~30-40% ceilings on this corpus. The bottleneck is one of:
- The corpus is genuinely hard for Nemotron-style training
- Our prompt framing (bridge tail_checklist + Rule 11a) pushes too far toward clarification
- The MCP tool namespace setup confuses the model

Next: try 35B-NVIDIA-NVFP4 (Qwen3.6 base, completely different model family). If it also caps at 30-40%, the corpus / prompt is the limit. If it jumps significantly higher, the issue is Nemotron-family training.
