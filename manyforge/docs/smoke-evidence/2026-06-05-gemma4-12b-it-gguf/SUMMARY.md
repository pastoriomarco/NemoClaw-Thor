# Smoke corpus — gemma-4-12b-it (GGUF) — 3-lane matrix

**Date:** 2026-06-05
**Model:** `unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL` + `unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL` speculative draft, served by llama.cpp (`ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-thor`) on Jetson Thor, `--reasoning off --jinja`, 64K ctx, served id `gemma-4-12b-it` on `:8000`.
**Corpus:** `manyforge/scripts/debug/smoke_corpus.yaml` — 66 active cases (9 future skipped).
**Harness:** `smoke_corpus_runner.py`. Per-lane timeouts: composer 180s, bridge wall 170s. Model served **without** the vLLM mutator proxy (direct llama.cpp); llama.cpp ignores the request `model` field so the sandbox's nominal `cosmos-reason2-8b` id is harmless.

## Results

| Lane / mode | Effective rate | First-try | Clean rate¹ | Infra failures |
|---|---|---|---|---|
| **direct** (nemoclaw bridge → vLLM-style, `tool_choice=required`) | **35/66 = 53.0%** | 29/66 (43.9%) | 35/62 = 56.5% | 4 |
| **openclaw-tools** (gateway, `toolSearch.mode=tools`) | **37/66 = 56.1%** | 31/66 (47.0%) | 37/47 = **78.7%** | 19 |
| **openclaw-code** (gateway, `toolSearch.mode=code`) | **20/66 = 30.3%** | 17/66 (25.8%) | 20/33 = 60.6% | 33 |

¹ *Clean rate* = passes ÷ (cases that were served without an infrastructure 502/504). The honest model-capability signal; the effective rate counts infra failures as failures.

### Failure breakdown (non-infra)

| category | direct | oc-tools | oc-code |
|---|---|---|---|
| pass | 29 | 31 | 17 |
| soft-pass | 6 | 6 | 3 |
| **arg shape/value mismatch** | **14** | 0 | 0 |
| bounded-autonomy (fired when should ask) | 6 | 5 | 2 |
| expected tool not fired | 6 | 1 | 9 |
| state mismatch | 1 | 4 | 2 |
| infra (502/504) | 4 | 19 | 33 |

## Key findings

1. **`tools` mode ≈ direct ≫ `code` mode.** Mirrors the documented cosmos-8b pattern (tools 58% / code 29%). Code mode forces the model to author JS (`openclaw.tools.call(...)`), producing long generations that time out: openclaw-code had **17× HTTP 504** vs 2 for tools.

2. **gemma's tool-arg quirk is lane-dependent.** In the **direct** lane gemma's raw args are asserted verbatim, and it consistently emits flat snake_case (`shape_type`, `size`) instead of the corpus's nested `shape.type` / `box_dims` → **14 arg-shape failures**. In **both openclaw lanes this is 0** — the OpenClaw tool_search/tool_call dispatch coerces args to the tool schema, masking the quirk. This is why openclaw-tools' clean rate (78.7%) is well above direct's (56.5%).

3. **Reproducible 16-case gateway-session cascade.** In both openclaw lanes, cases **41–56 (PnP_04 → PnP_20)** failed instantly (0.0–0.1s, HTTP 502). The chained pick-and-place build trips an OpenClaw gateway session-takeover crash mid-chain; every subsequent chained case 502s until the session recovers (case 57+ pass). The **direct lane is immune** (no gateway session). This single cascade accounts for 16 of the 19 (tools) / 16 of the 33 (code) infra failures.

4. **Tool-calling fundamentally works.** gemma-4-12b via llama.cpp `--jinja` emits correct OpenAI `tool_calls` (validated standalone and across all lanes). The model is genuinely usable for the composer-assistant task; on cleanly-served cases it reaches ~57–79%.

## Caveats / infra notes (problems encountered & solved)

- **Model crash on first direct run:** the model container (originally launched `-it --rm` from a terminal) died ~case 14, cascading 51 instant-502s and producing a spurious 4.5%. Restarted **detached with `--restart unless-stopped`** (no `--rm`); RestartCount stayed 0 across all three subsequent full runs.
- **502 cascade (finding 3)** is an OpenClaw 2026.5.22 gateway fragility on long chained sessions, not a model defect — re-running just the PnP subset after recovery would lift both openclaw numbers.
- Runs used the standalone llama.cpp model **without** the vLLM mutator proxy, so absolute numbers are not directly comparable to prior cosmos/vLLM evidence; the cross-lane *comparison here* is apples-to-apples (same model, same corpus, same timeouts).

## Reproduce

Cell runners are archived here (`run-cell-direct.sh`, `run-cell-openclaw.sh`). Each brings up the composer (`manyforge-e2e-composer`) for its lane, starts the bridge as a child, runs the full corpus, and tears down. Model must be serving on `:8000`.
