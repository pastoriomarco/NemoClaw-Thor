# Thor accelerator ROI plan

**Started:** 2026-11-07
**Status:** Draft — analysis only, no code changes made
**Owner question:** Thor carries engines that serving never touches (PVA, NVENC,
NVDEC, MIG, 14 mostly-idle Neoverse cores) and one engine it touches only
partially (`tcgen05` tensor cores behind software gates). Which of those can be
turned into serving throughput, and in what order should they be attempted?

Facts and arithmetic behind every estimate here are in
[`THOR-ACCELERATORS.md`](THOR-ACCELERATORS.md). This doc is the work list.

---

## Recommended setup routes (summary)

Read this section for the decision; read §2–§3 for the evidence.

**The asymmetry that shapes both routes.** Flash-Next is an MoE (125 B total,
~6 B active), so **active bytes are still reducible** — routing, PLE, dense side
layers. The 27B is dense, so quantization is nearly exhausted (MLP already
4-bit; only attention and `lm_head` remain) and the only remaining lever is
**tokens per pass**. Flash-Next is a bytes-centric route; the 27B is a
drafter-centric route.

### Route A — Qwen3.8-Flash-Next

| Layer | Choice |
|---|---|
| Weights | `nvidia/Qwen3.8-Flash-Next-FP4-Local` (official, 70.3 GiB, MTP n=4, FP8-KV recipe) as base; audit which dense layers are still BF16 and re-apply the FP8/FP4 hybrid conversion |
| Engine | **stay on vLLM** with the SM110a overlay (fused GDN, QSA det top-k, PLE mmap, XGrammar fix). `CUTE_DSL_ARCH=sm_110a`, never `sm_121a` |
| Kernels | full decode CUDA graphs; targeted shape autotune (never global); confirm native FP4, no Marlin |
| PLE | hashed-resident table (~12 GiB, E9) first, then FP4 (~24 GiB), current FP8 (47.7 GiB, paged) last |
| Drafting | MTP 4–5 and restore/expand the 47 k coding draft vocabulary |
| KV | FP8 above ~32 K, BF16 for short agent turns |
| Optional big lever | top-k experts 10 → 5 + a ~37.75 M-param heal, −20 % active bytes (E13) |

**Target:** 46.7 → **58–65** (quantization + MTP tuning) → **70–80** with the
routing cut; stretch ~90.

**Guard rail:** the official thread's 48.9 tok/s is `llama-benchy tg32` on
Spark — not comparable to the repo's 46.7 `sgbench` C1. Compare with the repo's
own harness or not at all.

### Route A revision (2026-11-14): three weight alternatives on one stack

Status: design. Measured so far: only the FP4-Local bring-up
([`QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md`](QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md)) and the old 46.7 tok/s recipe.

**Engine change.** Official vLLM 0.33 plus read-only file overlays replaces the preview fork. 0.33 ships
Flash-Next natively (`models/qwen4_exp`) and selected tensor-core kernels on sm_110 without gate patches
(`FlashInferCutlassMxfp8LinearKernel`, `VLLM_CUTLASS` NVFP4 MoE with FP4-Local).

**Shared stack (identical for all three, so only the weights differ):**

| Item | Choice | Output changes? |
|---|---|---|
| PLE table | on NVMe through the page cache (`VLLM_PLE_DISK=1`), lookup **prefetched** instead of synchronous (today ~18 ms GPU idle per step) | no |
| CUDA graphs | full decode graphs once the lookup no longer syncs; measure, do not assume (Mjolnir saw a graphed draft 7–16 % slower than eager at one request on sm_110) | no |
| GDN | confirm the fused CUDA decode kernel is selected on sm_110 | no |
| QSA top-k | `persistent_topk` on sm_110 (`cooperative_topk` needs a 16-block cluster, Thor's maximum is 8; upstream #56084/#56102) | no |
| Drafting | MTP swept at 3/4/5; reduced draft vocabulary (ours 47 K, Mjolnir 64 K) | no (acceptance only) |
| `lm_head` | FP8 (BF16 in all three checkpoints, ~1.27 GB read per target and draft step); alternative: upstream screened head #60566 | low risk |
| MTP dense parts | FP8 (BF16 in all three) | no (drafter only) |
| KV | BF16; FP8 only as a measured option at long context (E11) | — |
| Thinking | 32 K budget (E14) | task-level |

**Alternatives:**

| | A. old recipe, updated | B. FP4-Local | C. nvidia NVFP4 |
|---|---|---|---|
| Base | `RadixArk/Qwen3.8-Flash-Next-NVFP4` (126 GiB) | `nvidia/Qwen3.8-Flash-Next-FP4-Local` (96 GiB, QAD) | `nvidia/Qwen3.8-Flash-Next-NVFP4` (133 GB, rounded only) |
| As shipped | experts NVFP4, PLE FP8, rest BF16 | experts NVFP4 (CSF scales), side layers MXFP8, PLE NVFP4 | experts NVFP4, PLE FP8, MTP experts block FP8, rest BF16 |
| Left to quantize | 300 side matrices (our blockwise-FP8 delta), MTP dense, `lm_head` | `lm_head`, MTP dense | as A |
| Extra overlays | FP8-hybrid linked snapshot | CSF (#64312) + mapper (#64869) | FP8-hybrid linked snapshot; MoE fallback (CUTLASS FP4 MoE faults on Thor with this checkpoint, upstream #62275) |
| Quality evidence | 94/100 Tool Eval, 100/100 hard (old stack) | NVIDIA: 100/100 short, 89/100 hard (harness may differ) | none |
| Further step, quality-gated | side layers FP8 → NVFP4 (E3) | — | as A |

A and C have the same structure, so A is built on C's checkpoint if the tensor names match the FP8-delta
script (to verify); that avoids the RadixArk re-download.

**Mjolnir (first raised by the maintainer 2026-10-19, reviewed again 2026-11-14) — what overlaps and where
each idea actually came from:**
- Reduced MTP draft vocabulary: ours since the RadixArk recipe `sm110-fast-v2` (2026-09-14/16, 47,149
  coding tokens, [`QWEN38-FLASH-NEXT-FAST-THOR.md`](QWEN38-FLASH-NEXT-FAST-THOR.md)); Mjolnir uses 64 K.
  Mjolnir's A/B discipline for it is worth keeping.
- FP8 heads and FP8 for other parts we quantize ourselves: the maintainer's proposal. FP8 heads were already
  in our recipe on 2026-10-12 ("keep our model and our FP8 heads"); quantizing selected parts for the tcgen05
  path and an FP8 MTP head were proposed on 2026-11-08/09; blockwise FP8 for those parts on 2026-11-14.
  Mjolnir also runs an FP8 `lm_head`, but it is not the origin.
- QSA `cooperative_topk` sm_110 failure: hit independently on 2026-11-14 (upstream #56084/#56102); Mjolnir
  carries the same guard.
- Draft passes in full CUDA graphs: motivated by our R3 step timing (draft passes host-bound) and Mjolnir's
  draft-decode capture, both 2026-11-14 — the one idea with a direct Mjolnir link.
- Mjolnir's MoE guard (#62275) is only relevant for alternative C. Not adopted: its tcgen05 QSA decode kernel
  (needs FP8 KV, eager only, no MTP verify batches). Excluded for quality: FP6 KV, the hashed 12 GiB PLE
  table, the routing cut (E13).

**Acceptance:** each alternative is scored on the 88-scenario Tool Eval and the smoke corpus with identical
sampling and thinking settings, against 94/100 overall and 100/100 hard.

**Order:** shared stack on B (loaded now) → A on C's weights when the download completes → score both.

### Route B — Qwen3.8-27B

| Step | Action | Why |
|---|---|---|
| 1 | Kernel audit (E10), read-only | 8.3 vs 22 tok/s spread = up to 2.3× |
| 2 | Target autotune widths 9/18/27/36 plus the `lm_head` shape | `tactic=-1` fallback is already documented |
| 3 | Promote `qwen3.8-27b-nvfp4-dspark` (E12) | Spark analogue: 18.5 → 36.6 tok/s from drafting alone |
| 4 | QKV/gating protection per the quantization plan | quality floor for anything further |
| 5 | Drafter FP8, then FP4 challenger (E12) | +12 % / +19 %, output safe by construction |
| 6 | If prose acceptance is weak, MTP-5 instead of deep DSpark | MTP-8: 63 tok/s on short answers, 20 on reasoning prose |
| 7 | SGLang only if the 2×2 attribution below proves the engine matters | porting cost: loses the entire SM110a kernel stack |

**Target:** 22.5 → **30–38** (kernel + DSpark) → **42–52** (drafter quantization);
stretch 55–60.

### Shared foundation

1. **Thinking budget capped at 32 K** (E14) — 27B 3.97 → 3.25 min per correct
   answer, Flash-Next 4.00 → 2.99, and 17–29 runaway truncation loops disappear.
2. Full decode CUDA graphs (E1).
3. Official NVIDIA weights at the same precision mix as the current community
   quant.
4. Targeted shape autotuning only.
5. KV dtype per workload, not globally.

### Expected end state

| | Today | Route target | Stretch | Binding limit |
|---|---:|---:|---:|---|
| Flash-Next | 46.7 | 65–80 | ~90 | remaining active bytes + acceptance |
| 27B | 22.5 | 42–52 | 55–60 | dense model: only drafting remains |

Sequencing is in §6. Aggregate at C4 should scale ~1.8–2× on top of the
per-stream numbers, per the near-linear scaling already measured in
[`KV-CACHE-BUDGET.md`](KV-CACHE-BUDGET.md).

### GB10 precedent (external evidence, **not** Thor results)

Every row below is a single-GB10 measurement or claim. GB10 is `sm_121` with a
different kernel ecosystem; treat these as evidence of what this silicon class
can do, not as Thor expectations.

| Source | Finding | Feeds |
|---|---|---|
| [azampatti, "up to 70 tok/s"](https://forums.developer.nvidia.com/t/up-to-70tok-s-qwen3-8-flash-next-int4-autoround/382733) | top-k experts 10→5 + a 37.75 M-param heal → 60–70 tok/s on coding | E13 |
| same thread, SVJuno's build | resident **12 GiB hashed PLE table**, full decode CUDA graphs, MTP-2, BF16 KV at 495 k tokens | E9, E1 |
| [NVIDIA official Flash-Next thread](https://forums.developer.nvidia.com/t/nvidia-official-qwen3-8-flash-next-checkpoint-for-single-dgx-spark/385994) | `nvidia/Qwen3.8-Flash-Next-FP4-Local`, MTP n=4, FP8 KV, 70.32 GiB model, 1.23 M KV tokens, pp ~2.1 K t/s, tg32 48.9, tool-eval 100/100 short + 89/100 hard, median turn 2.6 s | Route A base |
| [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | official ModelOpt checkpoints for both models | weight base |
| [27B 11-config study](https://forums.developer.nvidia.com/t/comprehensive-qwen3-8-27b-study-on-dgx-sparks-quantization-speculative-decoding-and-tp-dp-scaling/381102) | TP=1: NVFP4+MTP-3 18.5 tok/s, NVFP4+**DSpark/SGLang 36.6**; BF16 4.5 vs a 4.9 roofline (273/55.6) | E12, Route B |
| [27B Spanish spec-decode A/B](https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-one-spark-with-thinking-on-spanish-assistant-workload-a-data-point-on-speculative-decoding/387403) | MTP-3 acceptance 0.76/0.58/0.46; MTP-8 63.4 short vs 20.0 reasoning prose; a reduced draft head can never draft out-of-set tokens | E12, draft-vocab audit |
| [27B thinking-budget study](https://forums.developer.nvidia.com/t/qwen3-8-on-one-dgx-spark-400-task-follow-up-on-low-thinking-parallel-reasoning-and-thinking-budgets/388057) | 32 K thinking budget: 27B 3.97 → 3.25, Flash-Next 4.00 → 2.99 min per correct answer | E14 |

### The engine question is not settled (2×2 attribution)

The 36.6 tok/s result changes **engine, drafter and weights at once**; the
study's own "what changed" column reads *engine + spec decode switch*. Do not
attribute it to SGLang until this table is filled:

|  | MTP-3 | DSpark K=7 |
|---|---|---|
| **vLLM** | 18.5 (known, Spark) | **missing cell — measure first** |
| **SGLang** | ~18–20 (expected) | 36.6 (known, Spark) |

Same RadixArk weights, 262 K context, FP8 KV, same harness. If vLLM + DSpark
lands near 35 the engine was never the story and the SM110a kernel stack
survives; if it stalls near 25, the residual is scheduler and tree-attention
maturity.

Porting cost if SGLang wins: the fused GDN decode kernel (`sm_110a`), the
deterministic QSA top-k `.so`, the PLE `mmap` module, the XGrammar termination
patch and the FlashInfer CUTLASS MoE gate are all vLLM-shaped and would have to
be re-established from scratch.

---

## 1. The constraint everything is measured against

Decode is DRAM-bound on this module:

```
tok/s ≈ 245 GB/s ÷ (W / k)      (W = bytes read per verified pass,
                                  k = accepted tokens per pass)
```

- Measured read bandwidth is **241–245 GB/s** of 273 theoretical
  (`bench/bw_bench.cu` in [`DS4-THOR-OPTIMIZATION-PLAN.md`](DS4-THOR-OPTIMIZATION-PLAN.md)) —
  the DRAM pipe is already at 88–90 %, so bandwidth-side tuning has ≤ ~10 %
  headroom.
- The Flash-Next FP8-hybrid delta (34.8 → 46.7 tok/s for 5.42 GiB BF16 →
  2.71 GiB FP8) implies **k ≈ 1.5 output tokens per pass, W ≈ 10.6 GB per pass**
  (derived, see [`THOR-ACCELERATORS.md`](THOR-ACCELERATORS.md) §5). At MTP=3
  that is ~0.5 drafts accepted per pass — low enough that acceptance, not just
  quantization, deserves instrumentation first.

Therefore every candidate below is scored on exactly two effects: **does it
reduce W**, or **does it raise k** (or cut overhead that is neither). Candidates
that do neither are listed as non-goals in §5 so they are not re-litigated.

---

## 2. Work list

Ordered by (expected gain ÷ effort). Status is `not started` for all: nothing
in this plan has been executed.

| ID | Experiment | Expected effect | Effort | Confidence |
|---|---|---|---:|---|
| E1 | Re-enable CUDA graphs for `cosmos-reason2-2b` / `cosmos-reason2-8b` (LLM graphed; ViT path kept on its safe backend) | +10–20 % on those two profiles | 1 d | High |
| E2 | CPU isolation: pin frontend/sampler/tokenizer off the GPU-feed cores | +2–5 %, lower p99 | 1 d | Medium |
| E3 | Extend the Flash-Next hybrid from FP8 to **NVFP4** for the 300 dense GDN/QSA/shared-expert matrices | **+20 %** (46.7 → ~56 tok/s predicted) | 3–5 d | Medium |
| E4 | NVFP4 **KV cache** on `sm_110a` — upstream RFE first, own kernel only if it stalls | more seqs/context per GiB; lower KV read traffic | 1 d RFE / 2 wk kernel | Medium (RFE), Low (kernel) |
| E5 | CPU n-gram / prompt-lookup drafting alongside MTP=3 | variance is wide; cheap to measure | 2–3 d | Low–Medium |
| E6 | Own `sm_110a` FP4 grouped-GEMM+SwiGLU (fix the CUTEDSL allowlist at kernel level) | +15–30 % prefill/TTFT | 1–2 wk | Medium |
| E7 | TRT-Edge-LLM v0.8 bake-off against vLLM on the same model pair | unknown; potentially large on prefill | 0.5 d rebuild | Medium |
| E8 | PVA pre-processing for the VLM lane via `cuPVA` on a CUDA stream | few % of VLM TTFT; large only for multi-camera | 1 wk | Low |
| E9 | **PLE: hashed-resident table (first), then FP8 → FP4** (Flash-Next) | −24 to −36 GiB + big TTFT win on 32K+ prompts; +0–6 % decode (layout-dependent) | 3–4 d | Medium |
| E10 | **Q3.8-27B kernel audit** + targeted FP4 autotune (read-only first) | 0 % if already native+tuned; up to ~2.3× if Marlin is in the path; 5–10 % from shape tuning | 1 d audit, 2 d fix | High (audit) |
| E11 | **KV dtype matrix** (BF16 vs FP8 × 8K/32K/256K × both models) | 0–10 % at long context; up to 2× sequence capacity | 2 d | High |
| E12 | **Drafter quantization** (FP8 first, FP4 challenger) | **+12 % (FP8) / +19 % (FP4)** derived for 27B + DSpark | 3–5 d | Medium |
| E13 | **Top-k expert routing cut + heal** (Flash-Next, external precedent) | −20 % active bytes → ~+25 % at ~2 % quality cost | 1–2 wk (training) | Medium |
| E14 | **32 K thinking-budget cap** (both models) | 27B 3.97 → 3.25, Flash-Next 4.00 → 2.99 min per correct answer | hours (config + eval) | High (external, 400-task study) |

### 2.1 Model precision inventory (basis for E3, E9, E12)

Everything below is the *checkpoint as served today*; which kernel executes it
is a separate variable (E10).

**Qwen3.8-Flash-Next** — 126 GiB base, 76.75 GiB loaded, 46.7 tok/s C1,
MTP=3, BF16 KV, 3 slots:

| Component | Dtype | Where it costs |
|---|---|---|
| Routed experts | NVFP4 (native block-scaled) | DRAM weights |
| 300 dense GDN/QSA/shared-expert matrices | blockwise FP8 E4M3 (`CutlassFp8BlockScaledMMKernel`) | DRAM weights |
| **PLE table (47.7 GiB)** | FP8, `mmap` raw bytes from NVMe/page cache | disk I/O + page cache |
| KV cache | BF16 | DRAM, grows with context |
| GDN recurrent state | BF16 | per-step state |
| Embedded MTP head (47 149-token draft vocab) | BF16 | DRAM every pass |
| Vision tower | BF16 | image requests only |

**Qwen3.8-27B** — `unsloth` NVFP4, ~24 GiB weights, FP8 KV, MTP K=3, 7 slots,
~22.5 tok/s short-request decode:

| Component | Dtype |
|---|---|
| MLP | NVFP4 W4A4 |
| attention/GDN, `lm_head`, final MLP | FP8 |
| vision, MTP head | BF16 |
| KV | FP8 |
| (`RadixArk` 27B variant instead carries `lm_head` in FP4 — the source of the `(9, 2560) × (2560, 248320)` `fp4_gemm` shape) |

### 2.2 Explicitly not on this list

`mma.ws`/2-CTA clusters and 2:4 sparsity are **compute-side** levers. They are
folded into E6 rather than tracked separately, because at batch 1 they cannot
beat the DRAM roofline — they matter only for prefill/TTFT and high batch.

---

## 3. Per-experiment detail

### E1 — CUDA graphs on the two Cosmos VLM profiles

**Trigger.** `serving/launch.sh` still passes `--enforce-eager` for
`cosmos-reason2-2b` (line ~172) and `cosmos-reason2-8b` (line ~208), both with
`--mm-encoder-attn-backend TORCH_SDPA`. The documented cost of eager mode is
10–20 % steady-state throughput.

**Hypothesis.** The eager requirement is a ViT-path problem, not an LLM-path
problem. Graph the decoder only; keep the ViT backend it needs. Check whether
the v0.22 "Qwen3.5 ViT full CUDA graph" work in `VERSIONS.md` removes the
remaining constraint, in which case graph both.

**Bounded by.** First-token latency and torch.compile/capture time
(`USER_QUICKSTART_MANUAL.md` records 2–5 min capture at batch sizes 1/2/4).

**Rollback.** Re-add the flag; the profiles are case arms in `launch.sh`, no
image change.

### E2 — CPU isolation

**Trigger.** No `isolcpus`, `taskset` or `numactl` appears anywhere in the
repo. On a 14-core Neoverse-V3AE with a 40–130 W envelope, the Python
frontend, detokenizer and sampler compete with the threads that keep the GPU
fed.

**Hypothesis.** Dedicating ~4 cores to vLLM frontend/sampler and excluding them
from the polling/feed threads buys a few percent and cuts tail latency, at zero
model-quality risk.

**Deliverable.** A documented `isolcpus`/`taskset` recipe in the launch path
plus before/after p50/p95/p99 at C1 and C8. Do not change profile defaults as
part of this experiment.

### E3 — NVFP4 for the dense side layers (highest expected payoff)

**Trigger.** The current best recipe converts 5.42 GiB of BF16 dense GDN/QSA/
shared-expert weights to 2.71 GiB of blockwise FP8 and gained +34 %
([`QWEN38-FLASH-NEXT-FAST-THOR.md`](QWEN38-FLASH-NEXT-FAST-THOR.md)). The
tensor core supports FP4 at 2× the FP8 rate per SM, and the module's problem is
bytes read per pass, so the same conversion at FP4 should recover roughly
another 1.35 GB/pass.

**Predicted.** `245 × 1.5 ÷ (7.89 − 1.35) ≈ 56 tok/s`, i.e. **+20 % over 46.7**
(derived; assumes unchanged k).

**Mechanism.** Reuse the already-validated machinery: local Docker-only hybrid
preparation that leaves the pinned snapshot untouched, the
`sm110a-fp4-dsl-unlock` mod (7 patches, idempotent) and
`VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass` — the combination that booted
`FLASHINFER_CUTLASS` on Thor for the first time and reached ~22 tok/s on the
RedHat 35B NVFP4 path.

**Gates.** Must hold the 88-scenario Tool Eval at **94/100 overall / 100 Hard
Mode**, keep the 256K exact-needle result, produce no Xid/illegal access, and
be reversible at launch. FP8 stays the documented fallback; the unmodified
checkpoint stays the quality baseline.

**Main risk.** Quality. FP4 on GDN/QSA is a smaller-error change than FP4 on
attention, but it is not free — the 27B NVFP4 plan already needed
QKV protection.

### E4 — FP4 KV cache

**Trigger.** `ROADMAP.md` records this as upstream-blocked: PR #40177 landed in
vLLM 0.21 for SM100 only, and no `sm_110a` cubins exist in NVIDIA's artifactory
as of the v8.1/v8.2 audits.

**Per repo policy** (`AGENTS.md` rule 2, "don't re-implement upstream"), the
primary action is an upstream RFE plus a periodic artifact audit, not a
private fork. Build a local kernel only if (a) the RFE stalls **and** (b) the
C1/C8 budget is actually KV-limited — which the Flash-Next profile is not
today (842 439 BF16-KV tokens, 3.21× full 256K contexts).

**Value if it lands.** Lower KV read traffic per token at long context and more
sequences per GiB; it does not change single-stream short-prompt decode.

### E5 — CPU n-gram / prompt-lookup drafting

**Trigger.** MTP=3 already exploits the built-in draft heads. Agent traffic has
a second exploitable structure: prompt-echo and long repeated tool schemas.
N-gram drafting needs no model, costs ~nothing, and runs on cores that are
otherwise idle — it composes with, rather than replaces, MTP.

**Hypothesis.** Gains concentrate on Composer tool-loop turns, not on the
sgbench C1 number. Measure with the smoke corpus, not just C1.

**Risk.** vLLM's n-gram proposer interacts with the MTP path and with XGrammar
termination (cf. the PR #52805 stop-token fix). Treat acceptance-rate drift as
the primary metric, and cap the added verifier work so it cannot regress plain
decode.

### E6 — Own FP4 grouped-GEMM with SwiGLU for `sm_110a`

**Trigger.** The CUTEDSL wall is a kernel-level check:
`"Blockscaled contiguous gather grouped GEMM with SwiGLU requires SM100 family
(Blackwell: SM100, SM103). Got SM110."` The ISA audit says the instructions
exist. This is the difference between "patch 7 call sites" and "own the
kernel".

**Scope.** A CUTLASS/CUTE-DSL implementation compiled for `sm_110a`, using
`tcgen05` with optional `mma.ws` and 2-CTA pairs, validated against the
reference kernel bit-for-tolerance on the model's real expert shapes.

**Payoff target.** The DS4 characterization (31–36 % of Spark prefill on a
dequantizing path) shows how much prefill is left on the table when the tensor
core is bypassed. Expect TTFT and long-prompt turnaround to move more than C1.

### E7 — TRT-Edge-LLM bake-off

**Trigger.** The artifact already exists (`serving/docker/Dockerfile.trt`,
`./serving/docker/build-trt.sh`) but the lane is parked on the
NVIDIA-acknowledged v0.8 concurrency/tool-call blockers. No expected work
before v0.8 ships.

**Why it is on the list at all.** NVIDIA's own stack is the most likely to
carry working FP4/TMEM kernels for `sm_110a` before upstream vLLM does.
Half-day rebuild when v0.8 lands; compare on the same model pair and the same
88-scenario gate.

### E8 — PVA for the VLM front end

**Trigger.** VLM profiles pay image pre-processing on SMs that could be
decoding.

**Scope.** Move resize/normalize/undistort to PVA via `cuPVA`, submitted on a
CUDA stream so it overlaps decode. Bounded to the VLM lane; do not attempt any
part of the ViT.

**Kill criterion.** If preprocessing is <5 % of VLM TTFT on the real corpus,
stop — the offload cannot matter.

### E9 — PLE table FP8 → FP4 (Qwen3.8-Flash-Next)

**Trigger.** 47.7 GiB of the checkpoint is a per-layer embedding table that is
**file-backed and read during inference**, i.e. PLE gathers are a page-cache/disk
path, not a tensor-core path. Measured with the in-repo probe
(`serving/benchmarks/flash-next-ple-io.py`, 100 random gathers, caches cleared):

| Mapping / gather | 64-row | 32 768-row |
|---|---:|---:|
| original / serial | 36.22 ms | 3 561 ms |
| `MADV_RANDOM` / serial | 10.17 ms | 422 ms |
| `MADV_RANDOM` / parallel | **2.06 ms** | **414 ms** |

> **Cheaper alternative to try first (external precedent).** An external
> single-Spark build reports a **resident 12 GiB hashed PLE table** alongside
> full decode CUDA graphs and BF16 KV at 495 k tokens
> ([thread](https://forums.developer.nvidia.com/t/up-to-70tok-s-qwen3-8-flash-next-int4-autoround/382733)).
> That is a **4× reduction that stays in RAM**, versus 2× for FP4 that is still
> paged, and it needs no change to the gather path's dtype handling. Order the
> work as: hashed-resident first, FP4 second, current FP8 last. Verify the hash
> scheme's collision behaviour before trusting it for exact retrieval.

**Mechanism (why this is tractable).** `vllm_ple_mmap.py` maps each shard as
`np.memmap(..., dtype=np.uint8, shape=(rows, row_bytes))` and gathers **raw
bytes**; `row_bytes` comes from safetensors metadata, not from a hardcoded
dtype. The reader is therefore dtype-agnostic — a narrower table needs no
reader change. What FP4 adds is (a) an offline E2M1 conversion of the PLE tensor
plus its block scales, (b) a consumer-side unpack on the gathered rows, which is
a handful of rows per token and trivially cheaper than the 2 ms fault-in it
replaces, (c) a scale-tensor layout that survives the shard mapping.

**Expected.** Halves PLE bytes (47.7 → ~24 GiB) and frees ~24 GiB of page
cache that currently competes with 76.75 GiB of resident weights. The decode
gain depends on one unmeasured fact: whether a decode step gathers one row per
token or one row per token **per layer**. The probe shape (64 rows) suggests the
latter, in which case a step pays up to ~2 ms of a ~25 ms budget; if it is the
former, decode gain is ~0 and the win is almost entirely in **long-prompt TTFT**
(the 32 768-row gather is 414 ms) plus memory relief. Note the second-order
effect: because the table is file-backed, freed page cache can matter more than
the gather time itself.

**Why the risk is unusually low.** An embedding lookup has no accumulation and
no cross-token mixing — quantization error is per row, per token, and does not
compound the way it does in attention or a GEMM.

**Gates.** Byte equality is impossible at 4 bits, so the verification is
different from the FP8 hybrid: sampled-row max-abs-error against the FP8 table,
then the full 88-scenario Tool Eval against 94/100 + 100 Hard Mode, plus the
256K needle test. Keep the FP8 table as the documented fallback and the
unmodified snapshot as the quality baseline.

**Kill criterion.** If a warm run shows the PLE working set already
page-cache-resident and the step time unchanged when gathers are served from
cache, the win is only bytes-per-gather. Measure warm first; do not judge this
from a cold run.

**Artifact cost.** A second ~24 GiB local artifact to generate, verify, and
host. Mirror the existing linked-hybrid + four-shard delta pattern rather than
duplicating the checkpoint.

### E10 — Qwen3.8-27B kernel audit (read-only first)

**Trigger.** The 27B launch arm sets no `VLLM_NVFP4_GEMM_BACKEND`; the
documented SM110 oracle behaviour differs between MoE and dense paths, and the
repo already records a 2.3× spread between FP4 execution modes on this exact
hardware:

| Path | Kernel class | Measured |
|---|---|---:|
| Software FP4 (W4A16) | `MARLIN` (dequant → FP16 MMA) | **8.3 tok/s** |
| Native NVFP4 | `VLLM_CUTLASS` block-scaled | 18.9 tok/s |
| Native NVFP4 | `FLASHINFER_CUTLASS` (after the 7-patch unlock) | **22 tok/s** |

Corroborating detail from the SM110 MoE patch: FlashInfer 0.6.17 **maps SM110
onto its SM100 CUTLASS implementation** and enables CUDA major version 11 for
it — i.e. the native path really is the `tcgen05` one, so any fallback is pure
loss. The DSpark overlay exists for the same reason in miniature: the
`(9, 2560) × (2560, 248320)` `lm_head` FP4 shape hit `fp4_gemm tactic=-1`
(no tuned kernel) and had to be autotuned explicitly.

**Method (no writes to the image).** Boot the profile once, capture the engine
log, classify every quantized module, and record the numbers in this doc:

```bash
docker logs <container> 2>&1 | grep -iE \
  "moe_backend|fp4_gemm|tactic=-1|Cutlass[A-Za-z0-9]*Kernel|Marlin|scaled_fp4|quantization"
```

**Deliverable.** A table of `module → kernel → tuned/untuned` for the 27B
profile, plus the same for the DSpark variant.

**Expected.** 0 % if everything is already native and tuned; **up to ~2.3×** if
any hot module is on Marlin; 5–10 % from autotuning the FP4 shapes the decoder
actually produces (verification widths, `lm_head`, prefill widths).

**Caution.** Do not enable global FlashInfer autotune: the Flash-Next launcher
deliberately passes `--no-enable-flashinfer-autotune`, and the 27B DSpark
overlay shows the targeted-width approach instead. Autotune the shapes you will
actually run.

### E11 — KV dtype matrix (measurement, no new kernels)

**Trigger.** BF16 KV was chosen for Flash-Next on quality grounds and is free at
agent-turn context lengths, but nothing in the repo records where that stops
being true. FP8 KV is the 27B default; an older FP8-KV profile crashed with an
illegal memory access at 8-concurrency, so reliability is part of the question.

**Matrix.** 2 models (`qwen3.8-27b-nvfp4`, Flash-Next hybrid) × 2 dtypes
(BF16, FP8) × 3 context targets (8K, 32K, 256K) × C1 and C3/C4, all at fixed
`gpu_memory_utilization`.

**Per cell, record.** Decode tok/s, TTFT, "GPU KV cache size" tokens from the
boot log, allocation per sequence, max concurrent sequences achieved, and a
short quality probe (20-scenario Tool Eval subset or full 88 where affordable)
plus the needle test at 256K.

**Constraints to respect.** FP8 KV requires the `flashinfer` attention backend on
SM110 regardless of `head_dim`; `flash_attn` is valid only at `head_dim=128` with
BF16 KV; and every FP8-KV cell needs the Xid/no-illegal-access check, not just a
throughput number.

**Deliverable.** The decision rule that currently exists only as folklore:

| Workload | KV dtype |
|---|---|
| agent tool loops, ctx ≤ 32K, 3–7 slots | BF16 |
| 256K fan-out, many concurrent sequences | FP8 |
| retrieval/needle evaluation | BF16 |

**Expected.** Little to no C1 gain at short context; the payoff is knowing
exactly where the quality/throughput crossover sits, and possibly 2× sequence
capacity at long context.

### E12 — Drafter quantization (FP8 first, FP4 challenger)

**Trigger.** Every drafter in use is BF16 and is read on every verification
pass:

| Drafter | BF16 | FP8 | FP4 |
|---|---:|---:|---:|
| DSpark v2 (27B) | 3.7 GB | 1.85 GB | 0.93 GB |
| DFlash (35B) | ~0.9 GiB | 0.45 GiB | 0.22 GiB |
| Embedded MTP head (47 k draft vocab) | ~0.2 GiB | 0.1 GiB | 0.05 GiB |

**Derivation.** For 27B + DSpark, pass traffic ≈ 13.5 GB (target) + 3.7 GB
(drafter) = 17.2 GB. Quantizing only the drafter gives 15.35 GB (**+12 %**) or
14.4 GB (**+19 %**) at unchanged `k`.

**Evidence that the acceptance risk is small.** z-lab reports **46.52 % (FP8
drafter) vs 47.24 % (BF16 drafter)** acceptance. The repo's own DFlash notes
also record that an *FP8 target with a BF16 drafter* can diverge numerically —
the lesson is to quantize drafter and target **consistently**, not to keep the
drafter wide.

**Gate (from the roofline, so it is checkable rather than a vibe).** Since
`tok/s ∝ k / bytes`, a quantized drafter wins iff the mean accepted length does
not fall by more than the byte ratio:

- FP8 drafter must retain **> ~90 %** of the BF16 mean accepted length.
- FP4 drafter must retain **> ~81 %**.

Output correctness cannot regress — the target verifies every token — so the
only failure mode is throughput.

**Order of work.** MTP head first (no separate checkpoint, no compatibility
view), then a quantized DSpark drafter reusing the existing
`Qwen3DSparkModel` compatibility `config.json`, then an FP4 challenger if FP8
clears its gate. Do this **after** E3/E9/E10, because a lower `W` multiplies
with any `k` gained here.

> **Attribution warning.** The external 36.6 tok/s figure for
> "DSpark + SGLang" changes engine, drafter and weights simultaneously. Fill
> the 2×2 in the route summary (above) before spending any effort on an engine
> port; the vLLM + DSpark cell is missing and is the cheap experiment.

### E13 — Top-k expert routing cut + heal (Flash-Next)

**Trigger.** An external single-Spark build
([forum thread](https://forums.developer.nvidia.com/t/up-to-70tok-s-qwen3-8-flash-next-int4-autoround/382733))
reports **60–70 tok/s** on coding by routing each token to **5 of 10 experts
instead of 10**, dropping active parameters from 6 B to 4.8 B, then training a
**37.75 M-parameter always-on component** — everything else frozen — to imitate
the uncut model's output distribution. Nothing was removed: all 512 experts
remain, and the MTP draft head is deliberately kept BF16 at top-10. Quality is
claimed within ~2 % of the 4-bit baseline.

**Why it fits the roofline rather than fighting it.** This is a `W` reduction on
the dominant term. It is also larger than any quantization step left on this
model: the dense side layers that gave +34 % were 5.42 GiB of a ~79 GiB resident
model, whereas −20 % of active experts cuts the per-pass read by roughly a
fifth.

**Mechanism on Thor.** Adopt, do not reinvent: take the published routing config
and the heal delta, then either serve the INT4 AutoRound checkpoint directly or
transplant the ~38 M healed tensors into the NVFP4 base. The second path is what
the forum discussion proposes so NVFP4 users can test the heal without
retraining.

**Gates.** 88-scenario Tool Eval at 94/100 with 100 Hard Mode, the 256K needle
test, and — critically — a non-code prose check. A routing cut changes model
*capability*, not just precision, and agent tool-selection is where small
distribution drift surfaces first.

**Kill criterion.** If the published heal delta cannot be transplanted into the
serving checkpoint, park the item. Training a heal from scratch is a
fine-tuning project outside this repo's scope (see `AGENTS.md` ownership).

### E14 — Thinking-budget cap (both models)

**Trigger.** A 400-task coding/reasoning study on one Spark
([thread](https://forums.developer.nvidia.com/t/qwen3-8-on-one-dgx-spark-400-task-follow-up-on-low-thinking-parallel-reasoning-and-thinking-budgets/388057))
found deep thinking hitting the 65 536-token output cap **29 times** on
Flash-Next and **17 times** on the 27B, mostly as repeated reasoning loops. A
**32 768-token thinking budget** moved:

| Model | Correct / 400 | Minutes per correct answer |
|---|---|---|
| 27B, deep thinking | 372 | 3.97 |
| 27B, 32 K budget | **383** | **3.25** |
| Flash-Next, deep thinking | 364 | 4.00 |
| Flash-Next, 32 K budget | **387** | **2.99** |

**Why it belongs in a throughput plan.** It changes nothing about tok/s and
everything about *time to a correct answer*, which is the metric the assistant
is actually judged on: roughly two thirds of generated tokens in a thinking
answer are reasoning.

**Mechanism.** Cap the thinking phase (`thinking_token_budget` /
`default-chat-template-kwargs`), then let the model produce its final answer.
The repo already has thinking-precedence rules in
`manyforge/docs/COMPOSER-ASSISTANT-ARCHITECTURE.md`; this must not override
them silently.

**Gates.** Measure with the smoke corpus, not `sgbench` — this is a task-level
change. A/B the cap against uncapped on both models; only 32 K was tested
externally, so treat 16 K / 48 K as open.

**Risk.** Capping reasoning can lower quality on genuinely hard tasks. The
external study reports gains on its task set, not a universal rule.

---

## 4. Measurement protocol (applies to every experiment)

Use the same harness for before/after, and record it in the doc:

1. **C1 / C8** via `sgbench.sh` with coding prompts, `enable_thinking: false`,
   temperature ≈ 0.2, ~1200-token outputs. This is the only cross-source
   comparable number (`README.md` methodology note).
2. **Quick regression** via `serving/benchmarks/bench-throughput.py` and
   `serving/benchmarks/bench-concurrent.py` — 100-token outputs, so use them for
   direction, never as a headline.
3. **TTFT** at 2K and 21K prompt depth.
4. **Quality**: 88-scenario Tool Eval against the 94/100 baseline, plus the
   256K exact-needle test where the change touches KV or attention.
5. **Stability**: no Xid, no illegal access, no CUDA-graph failure, no server
   restart over the run.
6. **Provenance**: image digest, profile label, drafter SHA (unpinned drafters
   drift upstream).

Non-negotiable gates carried over from
[`DS4-THOR-OPTIMIZATION-PLAN.md`](DS4-THOR-OPTIMIZATION-PLAN.md): build only
for Thor (`sm_110` / `sm_110a`), never emit or run Spark `sm_121` code; preserve
exact/accepted output against the baseline; keep streaming and non-streaming
API behaviour; retain the deep-context needle result.

---

## 5. Non-goals (with the reason, so they stay closed)

| Idea | Why not |
|---|---|
| **MIG for throughput** | MIG partitions SMs, not LPDDR5X bandwidth. Slices contend on the same 273 GB/s, and carving 20 SMs is a net loss. QoS/co-tenancy only |
| **NVENC/NVDEC for serving** | Video ingest only. No path from codec engines to token generation |
| **PVA for matmuls** | 12-bit fixed point at ~32 px/cycle/VPU; orders of magnitude off |
| **DLA offload** | Thor has no DLA. Orin's 2× NVDLA v2 did not carry forward |
| **2:4 sparsity ahead of FP4** | Compute-side only, needs pruning-aware finetune, and dense weights give nothing. Revisit after E3/E6 |
| **Rewriting the serving stack from scratch** | At batch 1 the framework is not the bottleneck; the DRAM is. vLLM's value (XGrammar, paged KV, prefix cache, MoE routing) is not GPU-bound. Own kernels, not the framework |
| **New profile defaults during these experiments** | `serving/config.sh` profile changes require operator approval plus the full profile-calibration retest |

---

## 6. Sequencing

Aligned with the route summary at the top: audit → config → weights →
drafting → quantization → exotic.

1. **Phase 0 (1–3 d, zero model risk):** **E10** (read-only kernel audit; it
   can invalidate the premise of every other item) plus **E1** + **E2**
   (runtime/OS, reversible at launch) and **E14** (thinking-budget A/B, config
   only).
2. **Phase 1 (1–2 wk, quality-gated):** E3 (dense side layers → NVFP4) and
   **E9** (PLE — hashed-resident first, FP4 second). These are the two
   `W`-reducing changes with the largest bytes at stake; E9 is the lower-risk of
   the pair.
3. **Phase 2 (measurement + low commitment):** **E11** (KV matrix — no new
   kernels, resolves an open decision), **E12 step 1** (promote DSpark inside
   vLLM and fill the missing 2×2 cell), E4 RFE + artifact audit, E5 smoke-corpus
   trial, E7 rebuild when v0.8 ships.
4. **Phase 3 (after `W` is settled):** **E12** drafter quantization (MTP head
   first, then DSpark FP8, then FP4 challenger), because its payoff multiplies
   with the `W` reduction from Phase 1.
5. **Phase 4 (multi-week, highest ceiling):** E6, **E13** (routing cut, only via
   external artifacts), then E8 if the VLM lane becomes primary.

Stop rules:

- If E3 fails its quality gate, do not proceed to E6 on the same quantization
  assumption — re-derive from the measured `W` and `k` first.
- If E10 finds Marlin on a 27B hot path, fix that before any quantization work;
  a 2.3× regression dominates every other item on this list.
- If E9's warm-run measurement shows no step-time change, stop at the FP8 table
  and spend the effort on E3/E12 instead.
- If the vLLM + DSpark cell lands near 35 tok/s, drop the SGLang port from
  consideration entirely and keep the SM110a kernel stack.
- If E13's heal delta cannot be transplanted into a servable checkpoint, park
  it rather than starting a training effort in this repo.

---

## 7. Open questions

- **k ≈ 1.5 is derived, not measured.** Instrumenting accepted-tokens-per-pass
  on the Flash-Next profile would turn the whole roofline from arithmetic into a
  budget, and would tell us whether MTP tuning (draft vocabulary, N) is worth
  more than quantization. It is also the gate input for E12.
- **Which kernel actually wins on the 27B profile today?** E10 answers this; the
  8.3 / 18.9 / 22 tok/s spread means the answer changes the whole plan.
- **Is the PLE working set page-cache resident in a warm run?** Determines
  whether E9 buys step time or only bytes.
- **Is DRAM bandwidth actually shared across MIG slices on Thor?** The
  architecture says yes; measuring it would settle the argument permanently.
- **Does `mma.ws` help at moderate batch (4–16) on 20 SMs?** Unknown; the
  instruction is available and unused. Folded into E6.
- **Where does PVA help beyond the VLM front end** — radar/lidar preprocessing
  in the robotics lane, or perception on the same box during serving? Out of
  scope here, noted for the ROS/Isaac side.
