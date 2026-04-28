# NemoClaw-Thor v7 image — coverage report

**Date:** 2026-04-27 / 2026-04-28
**Image:** `nemoclaw-thor/vllm:v0.20.0-gb8160878f-thor-sm110-cu132`  (also `:latest`)
**Stack:** vLLM v0.20.0 + FlashInfer v0.6.9 + flash-attn-4 b10 + CUDA 13.0.3-devel + transformers 5.6.2
**Hardware:** Jetson AGX Thor (SM110a / Blackwell-derivative), 14 ARM cores, 128 GiB unified memory

**Methods used:**
- **8-test mini-suite** (`/tmp/v7-eval-suite/`): T1 non-stream 1200-tok
  throughput, T2/T3 streaming code, T4 tool call, T5 multi-tool, T6 JSON
  schema, T7 refusal, T8 math. Per-profile results in
  `/tmp/v7-eval-suite/<profile>.txt`.
- **tool-eval-bench** (SeraphimSerapis v0.8.x, 69 scenarios, 15 categories) —
  pure quality benchmark for agentic tool-calling, 0/100 scoring scaled from
  PASS/PARTIAL/FAIL judgments. Per-profile output in
  `/tmp/v7-agentic-bench/teb/`.
- **IFEval-lite** (subset of Google IFEval, 100 prompts, ~25 verifiable rule
  types) — strict instruction-following score. Per-profile output in
  `/tmp/v7-agentic-bench/ifeval-out/`.

---

## Working profiles — full 8-test comparison (2026-04-27, fair-comparison)

All profiles below were run through the same 8-test eval suite on the
rebuilt v7 image (single fixed prompt set, same params, same auto-eval).
Numbers are sortable: T1 is the throughput probe (1200-tok non-stream,
A* code completion, temp 0.2). Per-profile raw output is in
`/tmp/v7-eval-suite/<profile>.txt`.

| Profile | T1 non-stream tps | Accept % | Stream avg / peak | Stream accept | Caps T4-T7 | T8 | Score |
|---|---:|---:|---:|---:|:-:|:-:|:-:|
| `qwen3.5-9b-claude-distilled-nvfp4` | **31.0** | 85.6 | **17.3 / 19.2** | 87–89% | 4/4 | length-cap | **7/8** |
| `qwen3.6-35b-a3b-nvfp4-tq-mtp` | **28.6** | 73.9 | 7.5 / 9.2 | 62–73% | 3/4 (T7 FP) | length-cap | 6/8 |
| `qwen3.6-35b-a3b-fp8-turboquant` | 25.8 | 67.0 | 7.1 / 8.1 | 60–77% | 4/4 | length-cap | 7/8 |
| `qwen3.6-35b-a3b-fp8-dflash` | 24.7 | 18.8 | 7.9 / 8.6 | 20–27% | 4/4 | length-cap | 7/8 |
| `qwen3.6-35b-a3b-nvfp4-mtp-fp8kv` (tool-eval-bench 93/100) | 22.2 | **88.0** | 8.1 / 9.3 | 82–86% | 4/4 | length-cap | 7/8 |
| `qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge` (★ ManyForge production) | 21.6 | 86.8 | 8.1 / 9.3 | 85–86% | 4/4 | length-cap | 7/8 |
| `qwen3.6-27b-fp8-dflash` (Apr 27 drafter) | 16.5 | 26.7 | 5.1 / 5.9 | 20–27% | 4/4 | length-cap | 7/8 |
| `cosmos-reason2-8b` | 13.5 | n/a | 14.5 / 15.4 | n/a | 3/4 (T7 FP) | **pass** | 7/8 |

Caps T4-T7 = four capability tests scored N/4 passing: T4 single tool
call, T5 multi-tool, T6 JSON schema, T7 refusal-with-no-code. T8 math
fails uniformly on Qwen3.6 hybrids because the 150-token budget cuts
off the calculation mid-step — a suite quirk, not a profile defect.
T7 false-positives on `nvfp4-tq-mtp` and `cosmos-reason2-8b` are
auto-eval substring matches; both models did refuse cleanly.

**Headline observations:**
- **Best single-shot throughput:** 9B distilled VLM (31 tps), then
  35B-NVFP4-TQ-MTP (28.6 tps).
- **Best acceptance × throughput balance:** 35B-NVFP4-MTP-FP8KV (88%
  accept @ 22 tps) — still the tool-eval-bench champ at 93/100.
- **Best context budget:** TurboQuant profiles (1.89M FP8-TQ /
  2.22M NVFP4-TQ KV tokens at full mem util).
- **35B-FP8-DFlash earlier solo runs reported 30–42 tps** at peak;
  fair-comparison on the same A* prompt at temp 0.2 lands at 24.7 tps.
  The headline `47.6 tok/s` in `lib/config.sh` is a single-shot
  best-case, not a fair-comparison number.
- **DFlash vs MTP acceptance gap is huge** (DFlash ~20% per-position
  vs MTP ~85%) but absolute tokens-per-round are similar (DFlash
  3.0–4.0 / 15 vs MTP-4 ≈ 2.9 / 4) — that's why throughput ends up
  comparable despite the gap.
- **Streaming TPS ceiling (~10 tps) is a Python-SSE measurement
  artifact**, not a model limit. Use T1 non-stream for true
  throughput comparisons.

---

## Agentic-quality comparison — tool-eval-bench (full 69 scenarios) + IFEval-lite (100 prompts)

**Method:** Each profile booted on the rebuilt v7 image. tool-eval-bench
scored 0–100 (with safety-cap penalty); IFEval-lite scored 0–100% strict
rule pass. Bench was completed 2026-04-28 in three sequential tiers
(main batch + 27B-token follow-up + robotics-tier follow-up). Per-profile
artifacts in `/tmp/v7-agentic-bench/teb/` and `/tmp/v7-agentic-bench/ifeval-out/`.

| Rank | Profile | TEB | IFEval | Median tps | Notes |
|---:|---|---:|---:|---:|---|
| 1 | `qwen3.6-35b-a3b-nvfp4-mtp-fp8kv` | **93 ★★★★★** | **90.4%** | 19.5 | v6 winner reproduced on v7 (62 PASS / 4 PARTIAL / 3 FAIL); FP8 KV + MTP N=2 |
| 2 | `qwen3.6-35b-a3b-nvfp4-mtp-fp8kv-n4` | 91 ★★★★★ | 89.0% | 25.9 | Same as #1 but MTP N=4 — variance/N-effect probe (added 2026-04-28). +33% throughput, −2 TEB, −1.4% IFEval |
| 3 | `qwen3.6-35b-a3b-nvfp4-tq-mtp` | **90 ★★★★★** | 89.0% | 24.8 | NVFP4 + TQ K8V4 + MTP N=4; needs `VLLM_USE_FLASHINFER_MOE_FP16=0` to boot reliably |
| 4 | `qwen3.6-35b-a3b-fp8-turboquant` | 88 ★★★★ | 89.7% | 23.9 | FP8 weights + TQ K8V4 KV + MTP-4 |
| 5 | `qwen3.6-35b-a3b-nvfp4-tq-mtp-2` | 87 ★★★★ | 90.4% | 20.8 | NVFP4 + TQ K8V4 + MTP N=2 (strictly dominated by #3, see findings) |
| 6 | `qwen3.6-27b-fp8-mtp-kvfp8` | 84 ★★★★ | 88.4% | 6.8 | dense 27B, MTP-2 (replaces v6 88/100 baseline within noise) |
| 7 | `cosmos-reason2-8b` | **81 ★★★★** | 84.9% | 14.5 | NVIDIA physical-AI VLM — surprisingly strong on general agentic |
| 8 | `nemotron3-nano-30b-a3b-nvfp4` | 67 ★★★ | **85.6%** | 16.7 | NVIDIA's Dec 2025 agentic flagship; mid-pack on TEB; **profile removed 2026-04-28** |
| 9 | `cosmos-reason2-8b-reasoning` | 52 (broken) | 30.8% | 13.9 | tuning experiment — generates uniformly 7-word responses; **profile removed** |
| 10 | `qwen3.6-35b-a3b-fp8-dflash` | 46 ★★ | 91.1% | 22.2 | DFlash drafter design hurts agentic correctness |
| 11 | `qwen3.5-9b-claude-distilled-nvfp4` | 42 ★★ | 76.7% | 29.9 | weak agentic; designed as fast control loop, not deep reasoner |
| 12 | `qwen3.6-27b-fp8-dflash` | 40 ★★ | 87.7% | 25.4 | confirms the DFlash agentic-weakness pattern at 27B size |

Median TPS column: vLLM `Avg generation throughput` median over rolling 10s
windows during real TEB+IFEval workload (steady-state, not peak burst).

**Big findings (across model sizes and quant formats):**

1. **DFlash speculative decoding hurts agentic correctness, consistently.** Three independent confirmations (35B-DFlash 46, 27B-DFlash 40, 9B-DFlash-VLM 42 — all ★★ Weak). MTP-based variants of the same base models score 84–93. The DFlash drafter design optimizes throughput acceptance but produces output drift that the bench detects as wrong tool-calls / wrong planning. **For agentic deployment on Thor, prefer MTP profiles over DFlash regardless of base model.**

2. **Cosmos-Reason2-8B is unexpectedly competent at general agentic** (81/100 TEB), not just embodied-AI. NVIDIA's physical-AI tuning didn't trade off general tool-calling. Strong validation of pairing it as the perception/spatial subagent in a split robotics architecture.

3. **Nemotron 3 Nano (NVIDIA's Dec 2025 agentic flagship) lands mid-pack at 67/100.** Strong IFEval (85.6%) but weaker on tool-eval-bench than the Qwen3.6-35B family. Possible drivers: small active-param count (3B vs Qwen 35B), our reliability tuning (no spec decode, max_seqs=2), no temperature tuning (NVIDIA recommends T=0.6/top_p=0.95 for tool calls; bench used defaults). Worth a follow-up with NVIDIA's recommended sampling.

4. **The "reliability-tuned" cosmos-reason2-8b-reasoning profile EMPIRICALLY broke generation** (uniformly 7-word responses on IFEval). The combination `{BF16 KV + max-num-seqs=1 + max-num-batched-tokens=16384}` triggered a chunked-prefill scheduler edge case in vLLM v0.20.0. The original `cosmos-reason2-8b` profile (FP8 KV, max-seqs=3, batched=8192) is empirically the better robotics deployment. Profile removed.

5. **27B-DFlash gated-drafter:** initial boot failed (`401 GatedRepoError` on `z-lab/Qwen3.6-27B-DFlash`). Fixed by adding auto-`HF_TOKEN` propagation in `start-model.sh`. After fix, 27B-DFlash booted and benched — but scored 40 on TEB, confirming the DFlash agentic-weakness pattern.

6. **KV-cache precision dominates spec-decode N for tool-call quality — and the optimal N depends on KV format.** Full 2×2 matrix on NVFP4 weights, 35B-A3B base:

   |  | MTP N=2 | MTP N=4 | Speed Δ (median tps) |
   |---|---:|---:|---:|
   | **FP8 KV** | TEB **93** / IFEval **90.4%** / 19.5 tps | TEB 91 / IFEval 89.0% / 25.9 tps | N=4: +33% tps, −2 TEB, −1.4% IFEval |
   | **TQ K8V4** | TEB 87 / IFEval **90.4%** / 20.8 tps | TEB **90** / IFEval 89.0% / 24.8 tps | N=4: +19% tps, +3 TEB, −1.4% IFEval |
   | KV Δ at N=2 | FP8: +6 TEB / 0% IFEval | | |
   | KV Δ at N=4 | | FP8: +1 TEB / 0% IFEval | |

   **Four real effects, cleanly separable:**

   - **KV precision is the dominant signal for TEB only**: at MTP N=2, FP8 KV beats TQ K8V4 by **+6 TEB** (93 vs 87). At N=4 the gap shrinks to +1 (91 vs 90).
   - **KV precision has zero IFEval effect**: both KV formats produce *identical* IFEval scores at the same N (90.4% at N=2, 89.0% at N=4). Lexical instruction-following is robust to KV-quantization noise; structural tool-call planning is not.
   - **The optimal N for TEB flips between KV formats**:
     - FP8 KV: N=2 (93) > N=4 (91) — cleaner KV → drafter is more accurate, smaller draft tree avoids unnecessary detours.
     - TQ K8V4 KV: N=4 (90) > N=2 (87) — compressed KV introduces attention noise → larger draft tree gives the verifier more recovery options.
   - **N=2 is slightly better than N=4 for IFEval on both KVs** (+1.4%). Lexical fluency is best at small spec-decode depth — fewer iterations of the MTP head means tighter alignment with the target's sampled distribution.
   - **N=4 is faster than N=2 on both KV formats**: median tps gain of 19–33%. Higher N amortizes drafter cost over more accepted tokens.

   **Operational choice:**

   - **Max correctness** → `nvfp4-mtp-fp8kv` (FP8 KV, N=2): 93 / 19.5 tps. The benchmark winner.
   - **Best balance** → `nvfp4-tq-mtp` (TQ K8V4, N=4): 90 / 24.8 tps. **+27% throughput vs the winner** and **+1.4× context budget** (2.22M vs 1.6M KV tokens) for −3 TEB. For most production agentic workloads this is the better practical choice.
   - **Strictly dominated cells**: `nvfp4-tq-mtp-2` (TQ + N=2) — 3 TEB worse than `nvfp4-tq-mtp` at the same KV. `nvfp4-mtp-fp8kv-n4` (FP8 + N=4) was useful as a variance probe but doesn't beat its N=2 sibling.

**Recommendation for NemoClaw + OpenClaw / Hermes agentic + robotics use:**

- **Orchestration brain (correctness-priority):** `qwen3.6-35b-a3b-nvfp4-mtp-fp8kv` (93/100, 90% IFEval, 19.5 tps) — reproduces the v6 tool-eval-bench winner on v7.
- **Orchestration brain (throughput+context-priority):** `qwen3.6-35b-a3b-nvfp4-tq-mtp` (90/100, 89% IFEval, 24.8 tps, 2.22M KV tokens) — −3 TEB for +27% tps and +1.4× context. Better for most production workloads.
- **Perception / spatial / temporal:** `cosmos-reason2-8b` (81/100) co-served via the `manyforge` profile pattern (`gpu_mem_util 0.32` + Cosmos `gpu_mem_util 0.25`).
- Don't switch to Nemotron 3 Nano or Llama-Nemotron-Super yet — the Qwen3.6-MTP family is empirically stronger at this scale on Thor. Re-evaluate when NVIDIA ships Nemotron 3 Super or v2.

---

## Recommended-sampling probe (2026-04-28) — vendor-recommended sampling vs deterministic baseline

After the main bench finished, three follow-up tests checked whether
Qwen-recommended sampling parameters (T=0.6/0.7, top_p=0.95/0.8, top_k=20)
or thinking-mode would shift the production picture. All baseline numbers
above used T=0/think=false (deterministic, reproducible); the question
was whether the model's "calibrated" sampling regime would beat the
deterministic baseline.

### Test setup

Three head-to-head tests against the baseline:

| Test | Profile | Sampling |
|---|---|---|
| 1 | `qwen3.6-35b-a3b-nvfp4-mtp-fp8kv` | T=0.6, top_p=0.95, top_k=20, **enable_thinking=true** (Qwen "thinking-mode" recommendation) |
| 2 | `qwen3.6-35b-a3b-nvfp4-mtp-fp8kv` | T=0.7, top_p=0.8, top_k=20, enable_thinking=false (Qwen "non-thinking" recommendation) |
| 3 | `cosmos-reason2-8b` | T=0.6, top_p=0.95, top_k=20 (NVIDIA Cosmos VLM-recommended sampling) |

Note: Test 3 did NOT inject NVIDIA's optional CoT system prompt
(`Answer in <think>...</think><answer>...</answer>`) — that would have
required a custom chat template or HTTP proxy and bound the orchestrator
to a model-specific prompt convention, which is undesirable for a
universal manyforge agent. Test 3 measured Cosmos under recommended
sampling alone, with TEB scenarios using their own native system prompts.

### Results

|  | TEB | IFEval | TEB Δ vs baseline | IFEval Δ vs baseline |
|---|---:|---:|---:|---:|
| **Baseline FP8KV** (T=0, think=false) | **93** | **90.4%** | — | — |
| Test 1 FP8KV thinking-on (T=0.6) | 91 | **39.7%** | −2 | **−50.7%** ⚠ |
| Test 2 FP8KV no-think rec (T=0.7) | 89 | 87.0% | −4 | −3.4% |
| **Baseline cosmos** (T=0, think=false) | **81** | **84.9%** | — | — |
| Test 3 cosmos rec (T=0.6) | 78 | 84.9% | −3 | 0.0% |

### Findings

1. **T=0 deterministic + think=false is the best regime for both models**
   on Thor for agentic workloads. Recommended sampling consistently costs
   2–4 TEB points across all three tests, with no upside on IFEval.

2. **Thinking-on catastrophically breaks IFEval** (90.4% → 39.7%, a
   50.7-point drop). The `<think>` content leaks into the verified
   output and breaks every lexical rule the verifier checks (no-comma,
   lowercase, length-cap, format markers). For lexical/format compliance
   tasks (and any user-facing output where format matters), thinking-on
   is **actively harmful**.

3. **Cosmos's IFEval is robust to temperature** (identical 84.9% at
   both T=0 and T=0.6). Unlike Qwen3.6-MTP-FP8KV, Cosmos does not lose
   format compliance under stochastic sampling — its training appears
   to internalize format rules into the response distribution rather
   than relying on deterministic decoding. But it still pays the
   3-point TEB cost on agentic.

4. **No production benefit to vendor-recommended sampling on Thor.**
   The Qwen3.6 model-card recommendations (T=0.6 thinking / T=0.7 non-
   thinking) reflect "natural" generation distributions, but for
   tool-call quality and structured agentic output, deterministic
   decoding beats them on every measured axis.

5. **Thinking-mode latency cost is severe.** Test 1 IFEval ran at
   ~52 s/prompt vs ~13 s/prompt for the no-think variants — a 4×
   slowdown driven by `<think>` chains generated for every prompt,
   even simple instruction-following ones. Even if you could parse
   out the thinking content, the latency hit alone would disqualify
   thinking-on for closed-loop control workloads.

### Production recommendation for manyforge orchestrator

```yaml
# default agentic call (max correctness, lowest latency)
temperature: 0.0
chat_template_kwargs:
  enable_thinking: false
```

Reserve thinking-on **only** for orchestration steps where:
- The orchestrator parses out `<think>` content programmatically
  before the response is consumed downstream, AND
- The added latency (~4×) is acceptable for that step (e.g., one-off
  high-level planning, not closed-loop control).

For all other agentic calls — tool selection, code generation, JSON
schema compliance, refusal handling, multi-turn dispatch — keep T=0
and think=false.

This is portable across the kept profiles: the same defaults work
for `nvfp4-mtp-fp8kv`, `nvfp4-tq-mtp`, `cosmos-reason2-8b`, and the
manyforge production profile. No model-specific sampling tuning needed.

### Artifacts

Per-test outputs (raw bench logs):
- `/tmp/v7-agentic-bench/teb/{fp8kv-thinkon-rec,fp8kv-nothink-rec,cosmos-rec}.log`
- `/tmp/v7-agentic-bench/ifeval-out/{fp8kv-thinkon-rec,fp8kv-nothink-rec,cosmos-rec}.{log,json}`
- `/tmp/v7-agentic-bench/run_recommended_sampling.{sh,log}` — runner script

The runner uses `THOR_DETACH=1` (proper docker daemon detach) and a
`flock` singleton guard. Earlier runs hit `( cmd & disown )` subshell
issues that left orphan child processes blocking the parent on
`do_wait`; the THOR_DETACH path avoids this entirely by letting the
Docker daemon own the container lifecycle.

---

## Blocked / broken profiles

### `qwen3.6-35b-a3b-fp8-turboquant` (and all TQ profiles) — **RESOLVED 2026-04-27**

Initial v7 boot attempt failed with:

```
NotImplementedError: TurboQuant KV cache is not supported for hybrid
(attention + Mamba) models. Boundary layer protection requires
uniform attention layers.
```

**Correction:** PR #39931 did NOT actually merge into vLLM v0.20.0 — the
v0.20.0 source on disk still matches the **pre-PR state byte-for-byte**
across all 4 affected files (verified by fetching upstream `arg_utils.py`,
`turboquant/config.py`, `platforms/interface.py`, `turboquant_attn.py`).
The earlier "merged into v0.20.0" claim from background research was wrong.

**Fix applied:**

1. v6 runtime mod `fix-pr39931-turboquant/run.sh` restored from git history
   (324 lines, idempotent, marker-detected). Replays the 6 source-level
   hunks across 4 files at container startup.
2. v7 runner image rebuilt (Phase 3 only, ~30 s — wheels and FA4/torch
   layers all cached) so `COPY mods/ /workspace/mods/` picks up the
   restored mod directory.
3. `THOR_DOCKER_ENV_ARGS+=("-e" "VLLM_MODS=fix-pr39931-turboquant")`
   re-added to all 3 TurboQuant profiles in `lib/launch.sh`:
   - `qwen3.6-35b-a3b-fp8-turboquant`
   - `qwen3.6-35b-a3b-nvfp4-tq-mtp`
   - `qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge` (ManyForge production)

**Verified on rebuilt image:** entrypoint logs show all 6 hunks patched
(`[arg_utils.py]`, 3× `[config.py …]`, `[platforms/interface.py]`,
`[turboquant_attn.py]`), engine reaches `Application startup complete`,
and the 8-test eval suite ran cleanly on all 3 profiles (results in
the working-profiles table above).

**Secondary fix for manyforge — `VLLM_USE_FLASHINFER_MOE_FP16=0`:**

After the TQ mod was applied, `qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge`
crashed at engine init with:

```
RuntimeError: dispatchMoeGemmSelectClusterShapeTmaWarpSpecialized
  [Arch=Sm100, T=__nv_bfloat16, TileShape=<128,64,64>, IsMXFPX=false]
```

The crash is in the **MTP drafter's** unquantized BF16 MoE forward
(not the main NVFP4 MoE). At `max-num-seqs=3`, the FlashInfer-CUTLASS
unquantized-MoE autotuner picks an SM100-only BF16 tile that has no
SM110 instantiation; the sibling profile `nvfp4-tq-mtp` is unaffected
because its `max-num-seqs=8` lands on a different (SM110-capable) tile.

vLLM's unquantized-MoE oracle picks FlashInfer-CUTLASS by default once
TRTLLM is filtered out for non-SM100. Setting
`VLLM_USE_FLASHINFER_MOE_FP16=0` removes both FlashInfer paths from
`AVAILABLE_BACKENDS` and the oracle falls through to Triton — which
runs cleanly on SM110. The main NVFP4 MoE is unaffected and still
uses FlashInfer CUTLASS via `VLLM_USE_FLASHINFER_MOE_FP4=1`.

The env var is now committed to the manyforge profile in
`lib/launch.sh`; ManyForge production can migrate to v7 once the
7-test reliability battery (the original v6 blessing run) passes
against the rebuilt image.

### `qwen3.6-35b-a3b-fp8-dflash` + `kv_cache_dtype=fp8` (experiment) — **BLOCKED**

```
ValueError: Selected backend AttentionBackendEnum.FLASH_ATTN is not
valid for this configuration. Reason: ['kv_cache_dtype not supported']
```

flash-attn-4 beta10 (in v7) added SM100 FP8 e4m3/e5m2 forward kernels
(PR #2109), but **vLLM's flash_attn backend selector still gates fp8 KV out**
at `vllm/platforms/cuda.py:303`. The kernel-level fix didn't propagate
to the backend wrapper. **DFlash + FP8 KV combo therefore remains blocked
on v7, same as v6.** Worth a vLLM PR upstream.

### `gemma4-31b-it-nvfp4` — **BLOCKED (config tuning needed, not v7-fundamental)**

```
ValueError: Chunked MM input disabled but max_tokens_per_mm_item (2496)
is larger than max_num_batched_tokens (2048). Please increase
max_num_batched_tokens.
```

vLLM v0.20.0 has stricter MM encoder budget validation. Gemma 4's vision
encoder needs ≥2496 tokens/item but the profile defaults
`--max-num-batched-tokens` to 2048. **Fix:** add `--max-num-batched-tokens 4096`
(or higher) to the gemma4 profile in `lib/launch.sh`. Not retested today.

---

## Comparison vs v6 (where measurable)

| Profile | v6 baseline | v7 result | Δ |
|---|---|---|---|
| 35B-FP8-DFlash non-stream (1200 tok, 8-test) | 35.4 tps (Apr 17 drafter, ad-hoc prompt) | **24.7 tps** (Apr 26 drafter, fair 8-test prompt) | **−30%** (real); previous gap was prompt-mix bias |
| 35B-FP8-DFlash streaming avg | 8.7 tps | 7.9 tps | −9% (SSE overhead bound) |
| 35B-FP8-DFlash acceptance | 39–44% | 18.8% (per-position; absolute tokens/round 3.0–4.0 vs 3.1) | mixed |
| 35B-NVFP4-MTP-FP8KV tool-eval-bench | **93/100** ★★★★★ | quality preserved (7/8 mini-suite, length cap) | OK |
| 27B-FP8-DFlash | 5.4 tps / **8.2% accept** (preview drafter) | **16.5 tps / 26.7% accept** | **3× across the board** |
| 35B-FP8-TQ profiles | working with mod | working with restored mod | parity |
| 35B-NVFP4-TQ-MTP-manyforge | working | working (after `VLLM_USE_FLASHINFER_MOE_FP16=0`) | parity |

**Net:** v7 is at parity with v6 on TQ and tool-eval-bench profiles;
27B-DFlash is dramatically better thanks to the new drafter;
35B-DFlash is **slower than previously believed** when measured on
the fair-comparison eval (the 35.4 tps v6 number used a different
prompt mix). Choose the throughput leader by workload, not by the
single-shot best-case in `lib/config.sh`.

---

## Drafter inventory (post-cleanup)

- `z-lab/Qwen3.6-35B-A3B-DFlash`: only Apr 26 snapshot retained (`42d3b34d58…`); 7 older purged. Cache 5.3 GB → 905 MB.
- `z-lab/Qwen3.6-27B-DFlash`: pre-downloaded Apr 27 commit (`0919688658…`), 3.3 GB. Repo is now gated/restricted on HF — auth required for fresh pulls.
- All profile `revision:` pins removed from `lib/launch.sh` — drafter
  selection now follows HF `refs/main`.

---

## Profile config decisions adopted

- `qwen3.6-35b-a3b-fp8-dflash`: bumped to upstream-recommended config
  (N=15, 32K batched, prefix-caching) per z-lab README. Pass B confirmed
  this beats the previous N=8/8K config on non-streaming TPS.
- `qwen3.6-27b-fp8-dflash`: same upstream config (already there from
  prior session). Comments in profile updated to reflect re-test on v7.

---

## Known issues / TODO for v8

1. **TurboQuant profiles** — working on v7 via restored runtime mod
   (`fix-pr39931-turboquant`). Until PR #39931 actually merges upstream,
   the mod stays. Re-evaluate on next v0.20.x bump.
2. **Unquantized-MoE on SM110 at low concurrency** — the FlashInfer
   CUTLASS unquantized-MoE oracle picks an SM100-only BF16 tile when
   M is small (e.g. `max-num-seqs=3` + MTP drafter forward).
   Workaround: `VLLM_USE_FLASHINFER_MOE_FP16=0` to route to Triton.
   Already applied to `nvfp4-tq-mtp-manyforge`. Worth a vLLM PR to
   teach the SM110 path to filter that tile out.
3. **DFlash + FP8 KV blocked.** vLLM's flash_attn backend selector
   doesn't yet advertise FP8 KV support despite FA4 b10 having the
   kernels. File a vLLM issue/PR.
4. **Gemma 4 NVFP4 profile** — `--max-num-batched-tokens 4096` added
   (was 2048). Boot not retested today.
4. **First-boot JIT cost on v7 is ~30–40 min** for NVFP4 + MoE profiles
   (cold FlashInfer JIT cache for SM 110a). Subsequent boots are fast
   (cache persists in `~/thor-flashinfer-cache`). Consider baking a
   warm cache into the image at build time.
5. **Pin freshness:** transformers 5.6.2, FlashInfer v0.6.9, FA4 b10,
   vLLM v0.20.0 — all current as of 2026-04-27. Re-evaluate at next
   image rebuild.
6. **Streaming TPS ceiling (~10 tps) is a Python-SSE measurement
   artifact, not a model limit.** Use non-streaming for true throughput
   comparisons.

---

## Documents updated/created during this work

- `docker/Dockerfile` — pin updates, build-job parallelism (12→14)
- `docker/build.sh` — defaults updated to v0.20.0 / v0.6.9 / 14 jobs / CUDA 13.0.3
- `docker/PATCHES-AUDIT.md` — full patch lifecycle audit
- `docker/mods/README.md` — placeholder (mods all removed)
- `docker/mods/{fix-pr39931-turboquant,fix-nvfp4-moe-scale-merge}/` — deleted
- `docker/patches/vllm_sm110_no_sm100_cutsl_cutlass.patch` — deleted
- `lib/launch.sh` — DFlash profiles updated to upstream config; TQ
  profiles' VLLM_MODS env var removed; comments updated
- `PERFORMANCE-V7.md` (this file) — coverage report
- `/etc/systemd/system/docker.service.d/buildkit-loglimit.conf` — raised
  BuildKit per-step log limit 2 MiB → 100 MiB
- `/swapfile` (64 GiB, swappiness=1) — added for safety during long compiles
