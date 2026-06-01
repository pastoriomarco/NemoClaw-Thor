# 3-Model Smoke Bake-Off — 2026-06-01

3-model comparison on the 66-case smoke corpus with the full pipeline (detectors #1-7, bridge `tail_checklist`, namespace_stop=16, tool-name normalization, structural_tag/qwen3_coder grammar, tool_error_rewrite).

Evidence preserved under [smoke-evidence/2026-06-01-3model-bakeoff/](smoke-evidence/2026-06-01-3model-bakeoff/). Diagnostic tooling: [../scripts/debug/diagnose_smoke_case.py](../scripts/debug/diagnose_smoke_case.py).

## Headline

**`qwen3.6-35b-a3b-nvfp4-nvidia` wins 56/66 (84.8% effective), matching the prior cosmos production winner (77.3% iter 28) and beating it on first-try rate.**

Nemotron family (4B + omni) underperforms structurally (39.4% / 31.8%) — consistent with the historical LANE-COMPARISON ("Nemotron 0/9 vs Cosmos 9/9").

| Model | Family | Effective | First-try | PnP pass | Avg/case | Wall-clock | Evidence |
|---|---|---|---|---|---|---|---|
| `Nemotron-3-Nano-4B-BF16` | Nemotron | 39.4% | 30.3% | 0/19 | ~17s | ~18 min | [4b-SUMMARY.md](smoke-evidence/2026-06-01-3model-bakeoff/4b-SUMMARY.md) |
| `Nemotron-3-Nano-Omni-30B-A3B-NVFP4` | Nemotron | 31.8% | 22.7% | 4/19 | ~25s | ~30 min | [omni-SUMMARY.md](smoke-evidence/2026-06-01-3model-bakeoff/omni-SUMMARY.md) |
| **`Qwen3.6-35B-A3B-NVFP4-NVIDIA`** | **Qwen3.6** | **84.8%** | **75.8%** | **18/19** | ~50s | ~45 min | [35b-SUMMARY.md](smoke-evidence/2026-06-01-3model-bakeoff/35b-SUMMARY.md) |

Raw `report.json` + `stdout.txt` per model alongside the SUMMARY files (`.txt` not `.log` because `*.log` is gitignored).

## Pipeline validation

All 7 detectors + bridge `tail_checklist` + `namespace_stop=16` validated live:

| Detector | Status | Evidence |
|---|---|---|
| #1 same_tool consecutive ≥ 4 → reflect, ≥ 8 → hard_stop | working | Fired on omni's `tree_draft_wrap_node` runs; saved compute. |
| #2 same_args ≥ 2 | working | Fired when models retried with identical args. |
| #3 result_repeat ≥ 2 | working | Fired on `catalog_read` repeat-fetches. |
| #4 same_namespace ≥ 5 → reflect, ≥ 16 → hard_stop | **critical** | Saved several omni cases from runaway tree_draft alternation. |
| #5 turn_counter ≥ 5 | working | Fired throughout. |
| #6 malformed_tool_call detection | **partial** | Missed 4B's bare-inline `tool_name {json}` format — marker list needs expansion. |
| #7 normalize_tool_names | **load-bearing** | Fired hundreds of times; without it every model fails every action case (prefix drop). |

`enable_thinking` propagation trace cleanup also verified live: `chat_template_kwargs.enable_thinking` injected per-profile by proxy, no dead top-level mirror. Single source of truth in [`serving/config.sh`](../../serving/config.sh).

## What the pipeline did not catch

1. **4B's bare-inline tool format** — `tool_name {json}` instead of `<tool_call><function=NAME>...`. Detector #6's marker list (`"<parameter="`, `"<function="`, `"arguments":`) didn't include "tool name immediately followed by JSON object as plain text." Worth adding a `tool_name + json-object` regex marker.

2. **"I'm giving up" patterns** — when 4B fell back to repeatedly outputting `'program_read'` (literal string) across 10+ PnP cases, no detector fired because the model wasn't looping at the chat-completion level — it was producing one response per case with zero tools. The chat-completion succeeds, the case fails. Hard to detect generically.

3. **Cross-case state pollution** — omni's PnP_06 failure poisoned PnP_07-20. Detectors operate per-chat-completion; pollution accumulates at the conversation level (shared `conversationId`). Possible future detector: when N consecutive cases on the same conversationId fail, signal the runner to start a fresh conversation.

## Cascade analysis

**4B PnP cascade — pure format failure.** All 19 cases failed with `tools=[]` because the model emitted `scene_draft_add_object {"objectId": ...}` as text content, not as structured tool calls. The qwen3_coder parser couldn't extract them. Detector #6 missed this format. After PnP_05's failure, 4B fell into "let me just read state" mode → answered `'program_read'` literally for the next 12+ cases.

**Omni PnP cascade — state pollution + agent-loop breakdown.** Worked through PnP_01-05 (4/5 pass), then PnP_06 hit the runaway loop on `tree_draft_wrap_node`. Hard_stop fired correctly (saved compute) but the case still failed. PnP_07-12 emitted progressively fewer tool calls. PnP_13-20: model literally output `[loop-break] I have called session_status 8 times...` from a prior turn's hard_stop response that got cached in context. **The hard_stop response itself became a contagion.** Real interaction effect worth investigating.

**35B PnP cascade — unbroken.** 18/19 PASS. The 1 fail (`PnP_17_home`) was a single-case tool-not-observed, not a cascade. Chained state worked end-to-end.

## Why Nemotron looks like a regression but isn't

Per [project memory `[Production default = OpenClaw + Cosmos-8B (2026-05-07)]`](../../../../../.claude/projects/-home-tndlux-workspaces-nemoclaw/memory/project_lane_parity_cosmos8b.md): historical LANE-COMPARISON was **9/9 cosmos, 1/9 Qwen3.6, 0/9 Nemotron** on the production matrix.

So:

1. **Nemotron was always weak on this corpus.** 4B/omni hitting ~30-40% is their natural ceiling, not a regression.
2. **Qwen3.6 was historically 1/9.** Now 35B-NVIDIA reaches 84.8% with the new pipeline. **Genuine improvement, mostly from detector #7 (normalize_tool_names).**
3. **Cosmos was the production king at 77%.** Cosmos was NOT re-run with the new pipeline in this round. **Unknown whether cosmos still wins or 35B-NVIDIA is now best.**

## Next steps (deferred, not actionable yet)

1. **Re-run cosmos full smoke with the new pipeline.** Cheapest experiment. If cosmos ≥ 80%, current pipeline is non-regressive on its prior winner. If cosmos drops, identify which new detector/rule causes regression on cosmos.

2. **Run Nemotron variants without bridge `tail_checklist`** (env: `OPENCLAW_BRIDGE_TAIL_CHECKLIST=0`). The tail_checklist was added for cosmos's "act-bias" — for Nemotron's "ask-bias" it may reinforce the wrong direction.

3. **Run Nemotron with `enable_thinking=on`** to test whether CoT helps the PnP cascade.

4. **Expand detector #6's marker list** to catch 4B's `tool_name {json}` inline malformation.

5. **Add a cross-case state-pollution detector** — when N consecutive cases on the same conversationId fail, signal the runner that a context reset would help. Or simply revert to chain-session OFF for known-fragile chains.

## Why this corpus is informative

It measures both *act-bias* (does the model act on directive prompts?) and *ask-bias* (does the model clarify on ambiguous prompts?). PnP_01-20 specifically stresses chained state — one bad state-mutation early poisons every downstream case until a fresh conversation. Models that pass PnP at all are demonstrating something real about state-tracking, not just first-turn tool-call generation.

35B-NVIDIA is the first model in this lineage to pass PnP at scale (18/19). Cosmos's prior production wins were on shorter chains (66-case corpus without the full PnP cascade). The next cosmos re-run will tell us whether cosmos was the right pick because it generalizes well, or only because the corpus didn't stress chained state hard enough.
