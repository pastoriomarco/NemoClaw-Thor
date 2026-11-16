# Qwen3.8-Flash-Next FP4-Local on Jetson AGX Thor (vLLM 0.33 + overlays)

Status: **usable recipes** (2026-11-16): **N3 + C1 + C2** (`nvidia/Qwen3.8-Flash-Next-NVFP4`, Tool Eval 94, 47.3 ms
per decode step) and **R8b + C3 + C2** (FP4-Local, Tool Eval 91–92, HumanEval 157/164, 46.2 ms). Both in "Current
recipes", started with [`scripts/flash-next-thor-serve.sh`](../scripts/flash-next-thor-serve.sh). Best strictly
output-identical configuration: R5. The sections below are the run history.

Checkpoint: `nvidia/Qwen3.8-Flash-Next-FP4-Local` @ `b9ce18a6887e18c089fdea8509a33e63429a3300`
(QAD-trained; routed experts NVFP4 with CSF-compressed scales, attention and shared
experts MXFP8, PLE n-gram table NVFP4, 96 GiB). Served from a linked snapshot,
`~/thor-hf-cache/linked/fn-fp4-local`: the 21 shards from the snapshot's `tensors/`
folder are linked at the top level, and the index is rewritten to match. No data is copied.

Image: official `vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e` (v0.33.0),
with read-only file overlays:

| Overlay | Why |
|---|---|
| vLLM PR #64312 (`modelopt.py`, `quantization/utils/nvfp4_utils.py`) | Decode CSF expert scales at load |
| vLLM PR #64869 (`models/utils.py`, `whisper.py`) | `WeightsMapper(weights_transforms=...)`, needed by #64312 |
| [`docker/patches/qwen4exp-ple-disk/`](../docker/patches/qwen4exp-ple-disk/README.md) (`VLLM_PLE_DISK=1`) | Serve the 26.8 GiB PLE table from NVMe through the page cache. Without it, vLLM allocated ~112 GB before loading and Thor ran out of memory |
| `models/qwen4_exp/nvidia/ops/qsa_indexer.py` `[thor]` | `cooperative_topk` fails on sm_110 ("cluster misconfiguration"), so use `persistent_topk` |

Kernels selected on sm_110:
- MXFP8: `FlashInferCutlassMxfp8LinearKernel` + `MarlinMxfp8LinearKernel`
- NVFP4 MoE: `VLLM_CUTLASS`
- GDN prefill: Triton/FLA
- Attention: FlashAttention 2

Model loading took 70.8 GiB and 169–174 s. At 0.70 memory with BF16 KV, the KV cache was 367,001 tokens.

## First serving result (2026-11-14, attempt 4)

The server came up and answered correctly (`17*23` = 391, coherent prose). Settings: 0.70 memory, 32K context, 1 sequence, BF16 KV,
no MTP, PIECEWISE graphs. Thor at runtime: ~98 GB used, ~24 GB available, no swapping.

- Decode: **22.0 tok/s** (vLLM engine stats, and the proxy's e2e 21.5). Each token is streamed as its own chunk, so the proxy's numbers are valid here.
- The PLE disk lookup runs on every step: 310 lookups for a 294-token reply. Each lookup took **18.2 ms of host time (4.7 ms of it is gathering rows)**, i.e. ~40% of a ~45 ms decode step. This is the main tuning target.

## Runs on 2026-11-14 (256K context, one sequence, BF16 KV)

Benchmark: `~/fn-bench/bench.sh` on Thor, three greedy prompts with thinking off (300-word prose, Python
code, counting in words), second warm run reported. Cline: the same agent task (analyse ManyForge, write a
pick-and-place example with different parameters), Reasoning Effort Low, temperature unset.

| Run | Settings | Bench prose / code / count (tok/s) | Cline task |
|---|---|---|---|
| R1 | 0.70 memory, MTP 3, synchronous PLE lookup | 31.8 / 47.1 / 52.0 | 21 turns to 84K context; gen 17–30 tok/s (later turns 26–30), TTFT up to 21 s on uncached prompts, prefix-cache hit 88 %; ran `pip install --break-system-packages rfc8785` on the laptop (uninstalled) |
| R2 | 0.75 memory, MTP 4, prefetched PLE lookup (side stream; staged synchronously during graph capture), `VLLM_PLE_DISK_FAST_ROWS=0`, BF16 mamba/ssm state, `VLLM_MARLIN_USE_ATOMIC_ADD=1`, `--performance-mode throughput --optimization-level 3 --async-scheduling`, 5 graph warmups | 29.4 / 45.9 / 54.9 (same within noise; text differs in 2 of 3) | 34 turns to 70K context, 15,069 output tokens at **31.9 tok/s** overall, median TTFT 4.9 s, prefix-cache hit 86–89 %; no installs, but `find /` and ran ManyForge on the host with another session's scratchpad `pylib` on `PYTHONPATH`; final edit left pending in Cline |

R2: KV cache 519,592 tokens; Thor ~103 GB used, ~19 GB available, ~1 GB swap. PLE timing per lookup at
decode: ~6 ms row gather, <1 ms dedup/H2D/launch, ~31 ms waiting for the GPU to produce the ids — the step
is GPU-bound at these lengths, so the prefetch hides prefill gathers but barely changes short-prompt decode.

Failures on the way (logs on Thor): first prefetch version invalidated CUDA graph capture
(`cudaErrorStreamCaptureInvalidated` in layer 0, `~/fn-local-prefetch-fail.log`); moving the id copy to a
side stream did not fix it (`~/fn-local-allflags-fail.log`); staging synchronously while a graph is being
captured did.

MTP acceptance (R2, MTP 4): 2.65 tokens/step on prose (71/49/28/17 %), 4.52 on counting (95/90/87/80 %).

| Run | Settings | Bench prose / code / count (tok/s) | Notes |
|---|---|---|---|
| R3 | R2 + `VLLM_SPEC_STEP_TIMING=200` + draft vocabulary 47,232 ids (`QWEN4EXP_DRAFT_VOCAB`) + MTP dense parts online blockwise FP8 (`QWEN4EXP_MTP_DENSE_FP8=1`, `QWEN4EXP_FP8_FORMAT=fp8_per_block`) | **37.5 / 56.7 / 51.1** | KV 456,208 tokens; model load 72.7 GiB; acceptance 2.5 prose, 3.65 code, 4.0 counting (was 4.5: the coding vocabulary covers number words less well) |

R3 step timing (GPU / host ms, 200-step windows, short prompts): whole step 67–68 / 66 (14.5 steps/s);
target forward **54–55** / 40; sampling 5.2 / 1.2; draft pass 1 (piecewise) 2.2 / 1.7; draft passes 2–4
**1.8 / 4.5–6 each** plus ~0.9 ms host metadata each — draft passes 2–4 are host-bound, the target forward
dominates GPU time.

Output text: prose and code differ between two identical R2 runs and between two identical R3 runs, so the
non-deterministic flags already in R2 (most likely Marlin atomic-add) cause the differences, not the R3 items.

Note (found 2026-11-14 evening): all Cline TTFT figures above come from `tps_proxy.py`, which ignored vLLM
0.33's `delta.reasoning` chunks, so they include the reasoning time. Output token counts and "overall tok/s"
(output tokens ÷ total request time) are correct. See "Measuring a Cline task" below.

R3 Cline task (same prompt): 36 turns to 98K context, 16,168 output tokens at **38.2 tok/s** overall, median
proxy TTFT 3.7 s, prefix-cache hit 94 %; at ~95K context the step is 71–72 ms GPU (target forward 57–58 ms). No
installs or host Python; ended with a pending `editor` edit (`examples/lite6_pick_and_place_tunable.program.yaml`,
not written; review found a `goal` bound to a float parameter and unused parameters).

### Kernel profile (R3 settings, short greedy prompt, `--profiler-config` torch, ~66 steps, 4.37 s GPU)

| Component | Share | ≈ ms per 68 ms step |
|---|---|---|
| Routed MoE, NVFP4 grouped GEMM (CUTLASS, W4A4) | 27.7 % | 18.4 |
| BF16 cuBLAS: target `lm_head` (~2.7 ms/call), draft head (~1 ms/pass), small gates/HC matmuls | 23.5 % | 15.6 |
| MXFP8 side layers, FlashInfer CUTLASS (128×128×256 tiles, ~79 µs/call at M≈5) | 23.0 % | 15.2 |
| Hyperconnection kernels | 7.8 % | 5.2 |
| MoE routing and FP4 activation quantization | 6.9 % | 4.5 |
| QSA attention / GDN / FP8 MTP dense / Marlin MoE (MTP) / other | ~11 % | ~7 |

The MXFP8 side layers take ~15 ms per step. They read ~3.03 GB of dense FP8 weights per step, which takes
~12.4 ms at 245 GB/s, so they run at ~80 % of the bandwidth limit (corrected 2026-11-14; an earlier claim of
"several times their bytes" was wrong). The profiled step (67.6 ms) equals the unprofiled one, so the profiler
did not inflate these kernels. Traces: `~/fn-profiles` on Thor.

### Kernel A/B on R3 settings

| Run | Change | Bench prose / code / count (tok/s) | Step GPU / target forward GPU |
|---|---|---|---|
| R3 | — | 37.5 / 56.7 / 51.1 | 67–68 / 54–55 ms |
| R4a | MXFP8 side layers on Marlin W8A16 (`-e VLLM_DISABLED_KERNELS=FlashInferCutlassMxfp8LinearKernel`) | 38.0 / 59.8 / 52.1 | 66.1 / 52.9 ms |
| **R4b** | R4a + routed MoE on Marlin W4A16 (`--moe-backend marlin`) | **39.1 / 61.0 / 55.9** | **61.2 / 48.0 ms** |

| R4b-K3 | R4b with MTP 3 | 38.9 / 59.9 / 55.9 | **54.8 / 43.5 ms** (acceptance 2.38 prose, 3.20 code and counting) |

| R4b-K3 + draft graphs | `VLLM_SPEC_STEP_DRAFT_FULL_CG=1` | 41.7 / 61.2 / 55.8 | 54.9 / 43.7 ms (no step change; prose gain is acceptance noise, 2.88 vs 2.38) |
| R4b-K3 + FP8 heads | `QWEN4EXP_LM_HEAD_MXFP8=1` (blockwise FP8, `CutlassFp8BlockScaledMMKernel` for target and draft head) | **44.9 / 69.8 / 60.7** | **50.3 / 43.4 ms** (sampling 5.3 → 2.6 ms); changes target logits slightly — needs Tool Eval |
| **R5 (restart 3)** | MTP 3, MoE Marlin + side layers FlashInfer MXFP8 (no `VLLM_DISABLED_KERNELS`), vectorized PLE gather + prefetch, `QWEN4EXP_DRAFT_HEAD_FP8=1` (draft head only), draft FULL graphs, atomic-add off | **43.3 / 62.9 / 58.4** | **53.0 / 43.3 ms**, cycle 53.3–54.9 ms; warm 7.6K prefill **3.94 s** (1,929 tok/s); acceptance 3.60 / 2.50 / 3.16; output-safe (target head BF16) |

R5 gets old-kernel prefill (3.9 s vs 5.2 s all-Marlin) with all-Marlin decode: the FP8 draft head (~1.9 ms/step)
pays for the FlashInfer side layers. The PLE stats show 488/490 prefetch "misses" with ~23 ms host wait per
lookup; the host waits on the GPU (host 51.6 vs GPU 53.0 ms per step), so this is not a GPU idle gap by itself.
Whether the 4.8 ms GPU gap before the PLE copy is gone needs a short profile.

The FP8 head first failed at load (`assert quant_config is not None`, `model_loader/utils.py:279`); fixed by giving
the head an `OnlineQuantizationConfig`.

MTP 3 and MTP 4 tie on throughput: verifying 4 tokens instead of 5 cuts the target forward by ~4.5 ms
(fewer distinct experts read per step), which offsets the lower acceptance. MTP 5 was skipped as a likely loss.

R4a saved ~2 ms per step. Being bandwidth-bound, the large side-layer GEMMs cannot get much faster with
either kernel; the saving comes from the small matrices and from fewer preparation kernels.
R4b removes the FP4 activation quantization and the CUTLASS grouped-GEMM overhead: another ~5 ms. Both keep
activations in BF16. KV at R4b: 509,420 tokens.

## R5: best output-safe configuration (2026-11-14 evening)

### Configuration

Same image and checkpoint overlays as above, plus the draft-head and spec-step overlays
([`qwen4exp-draft-head/`](../docker/patches/qwen4exp-draft-head/README.md),
[`qwen4exp-spec-step/`](../docker/patches/qwen4exp-spec-step/)). `screened_head.py` from the draft-head overlay must be
mounted even when its switch is off, because `model.py` imports it. Started with `~/fn-bench/start.sh` on Thor
(which adds `VLLM_PLE_DISK=1`, `--max-model-len 262144`, `--max-num-seqs 1`, BF16 KV, PIECEWISE graphs):

```
env:  VLLM_PLE_DISK_PREFETCH=1          (default FAST_ROWS, vectorized gather)
      VLLM_MARLIN_USE_ATOMIC_ADD=0      (deterministic)
      QWEN4EXP_DRAFT_VOCAB=/opt/qwen4exp/draft_vocab_code_47149.npy
      QWEN4EXP_MTP_DENSE_FP8=1  QWEN4EXP_FP8_FORMAT=fp8_per_block
      QWEN4EXP_DRAFT_HEAD_FP8=1         (draft head only; target head stays BF16)
      VLLM_SPEC_STEP_DRAFT_FULL_CG=1    VLLM_SPEC_STEP_TIMING=200
args: --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
      --moe-backend marlin              (side layers keep FlashInfer MXFP8: no VLLM_DISABLED_KERNELS)
      --gpu-memory-utilization 0.75 --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16
      --performance-mode throughput --optimization-level 3 --async-scheduling
      --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_num_of_warmups":5}'
```

Kernels: routed experts `MARLIN` (W4A16), side layers `FlashInferCutlassMxfp8LinearKernel` (W8A8, tcgen05) with
Marlin for the shapes FlashInfer does not take, draft head and MTP dense parts `CutlassFp8BlockScaledMMKernel`.
KV cache 556,180 tokens.

Output-safe: every switch either leaves the target model's computation unchanged or only changes the draft
(acceptance, not output). The side layers run W8A8 as the checkpoint was trained (QAD).

### Benchmark

| | All-Marlin, BF16 heads (R4b-K3) | Old kernels, BF16 heads | **R5** |
|---|---|---|---|
| Bench prose / code / count (tok/s) | 37.7 / 61.7 / 55.5 | 38.0 / 55.0 / 50.4 | **43.3 / 62.9 / 58.4** |
| Decode step GPU / target forward | 55.4 / 43.5 ms | 60.0 ms | **53.0 / 43.3 ms** |
| Warm 7.6K-token prefill | 5.2 s | 3.8 s | **3.94 s** (1,929 tok/s) |

R5 combines old-kernel prefill with all-Marlin decode. FlashInfer side layers cost ~1.3 ms/step at decode (activation
quantization kernels) but save ~1.2 s per 7.6K prefill; the FP8 draft head saves ~1.9 ms/step and pays for it. The
first 200-step window after start was slow (83 ms cycle, 10–14 ms sampling) from warm-up: a CuTeDSL JIT compile
(`HcDownSiluMma`) and a cold PLE page cache. Steady windows: 53.3–54.9 ms.

PLE: the stats report 488 of 490 prefetch "misses" with ~23 ms host wait per lookup. The host waits for the GPU
(host 51.6 vs GPU 53.0 ms per step), so this does not by itself mean GPU idle time. Whether the 4.8 ms GPU gap
before the PLE copy (Fable profile) is gone needs a profile. Page cache was only ~3 GB with 18 GB free.

### Cline task (same prompt as R1–R3)

49 requests, context 3K → 107K tokens, ~17 min wall time. Figures from vLLM's own metrics (port 8000, `/metrics`),
with the 9 benchmark requests of this server subtracted:

| | R5 |
|---|---|
| Prefix cache | 95.5 % of prompt tokens cached |
| Prefill | ~90 s for ~128K computed tokens, ~1,440 tok/s |
| Decode | ~23,650 tokens (reasoning included) in ~479 s, **~49 tok/s** |
| Time split | prefill ~16 %, decode ~84 % of request time |
| Step at ~100K context | 56–57 ms GPU (53–55 ms at short context) |
| MTP acceptance | 2.88 tokens per step (server lifetime) |
| Overall (output ÷ request time) | **41.3 tok/s** |

Overall rate across runs of this task: R1 25.6 → R2 31.9 → R3 38.2 → **R5 41.3 tok/s**. Rough metric: each run
takes a different path (turns, reasoning length), but the order matches the fixed benchmark.

Behaviour: tried `pip install rfc8785` on the laptop's system Python again (failed, nothing installed), then made
a `/tmp/mfvenv` virtualenv to run ManyForge code on the host (rejected by the user; venv deleted). Other commands
were read-only `grep`/`yaml` reads. The final Program (`lite6_robotiq_pick_and_place_demo`) was well formed with
real catalog nodes but would not work: the lift and retreat move +z in the gripper TCP frame (into the part; the
shipped example uses −z), the recovery `fallback` cannot run (a failed `retry` fails the cycle first, and
`grasp_ok` is never reset), the retry never reopens the gripper, `confirm_grasp` waits for `reached` which a
gripper closing on a part usually does not report, and it declares `assistant: forbidden`.

### Measuring a Cline task

- `tps_proxy.py` (port 8001) checks `delta.reasoning_content`; vLLM 0.33 streams `delta.reasoning`. Its TTFT
  therefore waits for the first visible answer chunk (in this run 385 s summed vs ~100 s real), and its per-request
  "decode tok/s" is inflated. Its token counts (from `usage`) and overall tok/s are correct. Fix pending: accept
  both fields and print think/answer time separately.
- `cached=None`: vLLM reports cached prompt tokens only with `--enable-prompt-tokens-details`.
- For comparisons use vLLM's counters: `time_to_first_token_seconds`, `request_prefill_time_seconds`,
  `request_decode_time_seconds`, `prompt_tokens_cached_total`, `generation_tokens_total`,
  `spec_decode_num_*`. Snapshot before and after the task and subtract.

### Ideas from the Fable follow-up (prefill and decode together)

Trace split of the 7.5K prefill chunk (ms): MoE 1,656 all-Marlin vs 1,410 old kernels, but the old CUTLASS MoE
GEMM is 3× faster than Marlin's (499 vs 1,501) and loses it in preparation/finalize (880). Side layers 1,421
Marlin vs 260 FlashInfer. Hyperconnections ~510, GDN ~430, QSA ~370. At decode Marlin and FlashInfer take the
same time per side-layer matrix; Marlin's gain is only the skipped activation-quantization kernels.

| Route | What changes | Estimate | Status |
|---|---|---|---|
| B0 | FlashInfer side layers + Marlin MoE | prefill ~4.0 s, decode +1.3 ms | **done = R5** (3.94 s) |
| A1 | FlashInfer CUTLASS MoE on sm_110 via one-line gate overlay (`flashinfer_cutlass_moe.py:128-139`), autotune off | prefill 3.3–3.6 s, decode 57–59 ms | to test (boot test) |
| D1 | Fuse the CUTLASS MoE finalize (shuffle + weight multiply + sum) into one Triton kernel | −0.35 s prefill, −0.5…0.9 ms decode on that path | if A1 fails |
| D2 | Fused tcgen05 NVFP4 MoE from Edge-LLM's sm_110 CuteDSL kernels (Apache-2.0) | prefill ~3.0 s, decode at Marlin level | largest item |
| B | Side-layer kernel chosen by token count (two copies) | decode −1.3 ms, +3 GB, mixed W8A8/W8A16 | low priority |
| — | torch.compile act-quant fusion | < 1 ms; no MXFP8 patterns in 0.33 | rejected |

Also: QSA prefill 316 ms (one Triton kernel per layer, not tensor-core), hyperconnection glue 344 ms, GDN prefill
430 ms (CuteDSL prefill for sm_110 merged upstream after 0.33, vLLM #65050).

## R6: R5 + screened target head + FlashInfer CUTLASS MoE on sm_110 (2026-11-14 night)

Changes from R5: `-e QWEN4EXP_LM_HEAD_SCREENED=1`; overlay
[`patches/flashinfer-cutlass-moe-sm110/`](../docker/patches/flashinfer-cutlass-moe-sm110/) (one line: adds
`is_device_capability_family(110)` to `FlashInferExperts._supports_current_device`, upstream excludes sm_110 for
flashinfer#3134); no `--moe-backend` flag (auto selection gives `FLASHINFER_CUTLASS` to the W4A4 target experts and
`MARLIN` to the W4A16 MTP experts); `--enable-prompt-tokens-details`. Script `~/fn-bench/round3.sh`, log
`~/fn-bench/round3-r6.log`, profiles `~/fn-profiles/r6`.

First attempt failed at load with `--moe-backend flashinfer_cutlass`: forcing it applies to the MTP layer too, whose
scheme (NVFP4 weights, unquantized activations) FlashInfer CUTLASS does not support
(`~/fn-local-a1-forced-fail.log`). Not a kernel failure.

| | R5 | **R6** |
|---|---|---|
| Bench prose / code / count (tok/s) | 43.3 / 62.9 / 58.4 | **45.2 / 64.0 / 58.5** |
| Step GPU / cycle | 53.0 / 54.9 ms | **51.6 / 53.4 ms** |
| Target forward GPU | 43.3 ms | 44.6 ms (A1 costs ~1.3 ms at decode) |
| Sampling incl. head, GPU | 5.3 ms | **2.6 ms** (screen kernel 2.5 ms per call) |
| Warm 7.6K prefill | 3.94 s | **3.11 s** (2,442 tok/s) |
| Acceptance prose / code / count | 3.60 / 2.50 / 3.16 | 2.87 / 2.75 / 3.30 |

Output: A1 runs the routed experts W4A4 (the checkpoint's QAD scheme, as with the old CUTLASS path), so text differs
from R5. The screened head is designed to be exact except at BF16 ties; its check-mode run
(`QWEN4EXP_LM_HEAD_SCREEN_CHECK`) is still pending.

Prefill profile, 7,488-token chunk: 2.52 s GPU (old kernels 3.23 s, all-Marlin 4.59 s). MoE GEMM 462 ms + FlashInfer
MoE preparation/finalize and routing 382 ms = 844 ms (old CUTLASS 1,410, Marlin 1,656). Hyperconnections 336,
QSA 330, other 327, GDN 223, side layers 192 + 52 act-quant, BF16 cuBLAS 177 ms. The 101-token tail chunk still
costs ~0.26 s.

Decode profile (200-token greedy code prompt, 54 steps): MoE GEMM 15.3, side layers 15.2, BF16 cuBLAS 5.9, MoE
routing/prep/finalize 4.3, hyperconnections 4.3, screened head 2.5, draft head + MTP dense 2.4, GDN 1.2, other ~3 ms
per step; **GPU idle ~8.9 ms per step**, almost all in one gap per step between `_hc_combine_kernel` and the PLE
rows' host-to-device copy (50 of 54 steps, 9.1 ms average). The PLE ids come from the previous step's output; the
host gathers ~68 rows (6.5 ms, mostly page-cache misses: ~3 GB cached of the 26.8 GB table) while the GPU runs
layers 0–1 and then waits at layer 2.

### R7 / R7b: memory share and PLE page-cache prewarm

- R7 (0.60 share) failed at startup: "No available memory for the cache blocks" — 0.60 × 122.9 GB is below the
  ~73 GB of weights plus activations (`~/fn-local-mem060-fail.log`). At 256K context the share cannot go below ~0.70.
- R7b = R6 at 0.70 (KV 311,928 tokens) plus `~/fn-bench/ple_prewarm.py` run inside the container after startup
  (reads all PLE byte ranges once: 26.82 GiB from 7 files in 15 s). Page cache 21–27 GB afterwards, ~24 GB available.

| | R6 | R7b |
|---|---|---|
| Bench prose / code / count | 45.2 / 64.0 / 58.5 | 44.3 / 63.2 / 58.6 (same within noise) |
| Step GPU (unprofiled, 200-step windows) | 51.6 ms | 51.7–54.2 ms |
| PLE row gather per lookup | 6.5 ms | 3.8 ms |
| Profiled decode: step / GPU idle / PLE gap | 58.8 / 8.9 / 9.1 ms | 53.1 / 3.5 / 2.8 ms |
| Warm 7.6K prefill | 3.11 s | 3.21–3.26 s |

Correction: the 9 ms PLE gap in the R6 decode profile was mostly profiler overhead on the host path. The unprofiled
step did not change with the prewarm, so the real PLE gap is about the ~3 ms the R7b profile shows. A fully resident
PLE table (GPU gather in the graph) would save about that much (~5–6 %), not the ~18 % first estimated. The host
waits 16.5 ms per lookup for the ids (`id sync`), but that is host time spent waiting for the GPU, not GPU idle.

## Recipe R7b (best on 2026-11-14; superseded, see "Current recipes")

Script on Thor: `~/fn-bench/round5.sh` (calls `~/fn-bench/start.sh`, then `ple_prewarm.py`, then the benchmarks).

Image `vllm/vllm-openai@sha256:d825baf9…` (v0.33.0) with read-only overlays:
- CSF scale decode (#64312) and `WeightsMapper` (#64869);
- `qsa_indexer.py` sm_110 `persistent_topk`;
- [`qwen4exp-ple-disk/`](../docker/patches/qwen4exp-ple-disk/README.md) (PLE table from the page cache);
- [`qwen4exp-draft-head/`](../docker/patches/qwen4exp-draft-head/README.md) (draft vocabulary, FP8 draft head, MTP dense FP8, screened target head);
- [`qwen4exp-spec-step/`](../docker/patches/qwen4exp-spec-step/) (draft FULL graphs, step timing);
- [`flashinfer-cutlass-moe-sm110/`](../docker/patches/flashinfer-cutlass-moe-sm110/) (FlashInfer CUTLASS MoE allowed on sm_110).

```
env:  VLLM_PLE_DISK=1 VLLM_PLE_DISK_PREFETCH=1 VLLM_MARLIN_USE_ATOMIC_ADD=0
      QWEN4EXP_DRAFT_VOCAB=/opt/qwen4exp/draft_vocab_code_47149.npy
      QWEN4EXP_MTP_DENSE_FP8=1 QWEN4EXP_FP8_FORMAT=fp8_per_block QWEN4EXP_DRAFT_HEAD_FP8=1
      QWEN4EXP_LM_HEAD_SCREENED=1 VLLM_SPEC_STEP_DRAFT_FULL_CG=1 (VLLM_SPEC_STEP_TIMING=200 for measurement)
args: --max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 8192 --kv-cache-dtype bfloat16
      --gpu-memory-utilization 0.70 --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}'   (no --moe-backend: auto)
      --performance-mode throughput --optimization-level 3 --async-scheduling --disable-flashinfer-autotune
      --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_num_of_warmups":5}'
      --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml --enable-prompt-tokens-details
after start: docker exec -i fn-local python3 - < ~/fn-bench/ple_prewarm.py   (26.8 GiB, 15 s)
```

Kernels: routed experts FlashInfer CUTLASS W4A4 (tcgen05); MTP experts Marlin; side layers FlashInfer MXFP8
(W8A8, tcgen05) plus Marlin for small shapes; target head FP8 screen + exact BF16 rows; draft head and MTP dense
blockwise FP8. KV 311,928 tokens.

Output: routed experts and side layers run W4A4/W8A8 as the checkpoint was QAD-trained; the target head is exact
except at BF16 ties (check-mode run still pending); draft-side changes affect acceptance only. Not bit-identical
to R5 (R5 ran the experts W4A16 on Marlin).

### Cline task on R7b (new task, same prompt as R1–R5)

45 requests, context 3K → 132K tokens. vLLM metrics, snapshot before/after (the cancelled old-task request #147
excluded from the proxy side; its prefill is in the vLLM totals, mostly cached):

| | R5 | **R7b** |
|---|---|---|
| Prefix cache | 95.5 % | 95.4 % |
| Prefill (computed tokens) | ~1,440 tok/s | **~1,960 tok/s** |
| Decode (incl. reasoning) | ~49 tok/s | **50.2 tok/s** |
| Prefill share of request time | ~16 % | 16 % |
| Acceptance | 2.88 | 2.83 |
| Overall (output ÷ request time) | 41.3 tok/s | 41.1 tok/s |

Overall rate is unchanged because the task path differed (R7b's last turn was one 5,254-token answer at 61.7 tok/s;
total output 19,215 vs 23,653 tokens). Prefill and decode rates are the comparable figures.

Behaviour: no install attempts; one read-only `find / -name 'rfc8785*' -maxdepth 8`. The Program it produced
(`pick_and_place_batch_retry`, Lite6 + `lite_gripper`) was much better than R5's: correct lift direction (−z
in the TCP frame, as the shipped example), a working retry (fallback opens and retracts, then an `inverter`
forces the retry), cycle count, settle time and timeout bound to Program parameters, gripper values matching
`lite6_lite_gripper_pick_and_place`. Weaknesses: `reached` on a pneumatic gripper only confirms the valve dwell,
the `gripper_closed` key is never set true, cycle 2 would re-pick the part from the bin, the failed-grasp retract
is an absolute pose, and it declares `assistant: forbidden`. It stayed a pending Cline edit (not written).

### All runs today, one table

| Run | Key change | Bench prose / code / count | Step GPU | 7.6K prefill | Cline overall |
|---|---|---|---|---|---|
| R1 | first MTP 3 run, sync PLE | 31.8 / 47.1 / 52.0 | — | — | 25.6 tok/s |
| R2 | MTP 4, PLE prefetch, flags | 29.4 / 45.9 / 54.9 | — | — | 31.9 tok/s |
| R3 | draft vocabulary, MTP dense FP8 | 37.5 / 56.7 / 51.1 | 67–68 ms | 3.7 s | 38.2 tok/s |
| R4b-K3 | all-Marlin, MTP 3 | 38.9 / 59.9 / 55.9 | 54.8 ms | 5.2 s | — |
| R5 | Marlin MoE + FlashInfer side, FP8 draft head, draft graphs | 43.3 / 62.9 / 58.4 | 53.0 ms | 3.94 s | 41.3 tok/s |
| R6 | + screened head + FlashInfer MoE (A1) | 45.2 / 64.0 / 58.5 | 51.6 ms | 3.11 s | — |
| **R7b** | R6 at 0.70 + PLE prewarm | 44.3 / 63.2 / 58.6 | 51.7 ms | 3.2 s | 41.1 tok/s (decode 50.2, prefill 1,960 tok/s) |

### Against the previous best (RadixArk FP8-hybrid, `QWEN38-FLASH-NEXT-FAST-THOR.md`)

The RadixArk recipe (original checkpoint, dense GDN/QSA/shared-expert weights in blockwise FP8, NVFP4 experts,
MTP 3, `sm110-fast-v3` image) measured **46.7 tok/s token-weighted on `sgbench` C1** (public forum script,
temperature 0, thinking on, end-to-end rate) and **94/100 on the 88-scenario Tool Eval**. Our `bench.sh`
(thinking off, three prompts) and the Cline task are different workloads, so no number above compares with
46.7 directly; no Cline run of the RadixArk recipe is recorded.

`sgbench` C1 on R7b (same public script, port changed to 8000, `~/fn-bench/sgbench-r7b.log`, two runs):

| Workload | RadixArk FP8-hybrid C1 | **R7b C1** (run 1 / run 2) | Change |
|---|---|---|---|
| Q&A (76 tokens) | 41.0 | 41.3 / 43.1 → **42.2** | +3 % |
| Code (243) | 42.9 | 54.9 / 57.0 → **56.0** | +31 % |
| JSON (861) | 53.2 | 64.1 / 65.6 → **64.9** | +22 % |
| Math (64) | 44.9 | 52.4 / 52.8 → **52.6** | +17 % |
| LongCode (2,048) | 45.6 | 46.0 / 47.1 → **46.6** | +2 % |
| **Token-weighted** | **46.7** | 50.4 / 51.6 → **51.0 tok/s** | **+9 %** |

**Tool Eval on R7b** (2026-11-15 00:19–00:36, tool-eval-bench `2.6.1.dev65+g6be685f0e`, same protocol as the
RadixArk run: 69 + 19 Hard Mode scenarios, temperature 0, seed 42, `enable_thinking`, `reasoning_effort=medium`,
straight to vLLM on port 8000; one run):

| | RadixArk FP8 hybrid (2026-09-14) | **R7b** |
|---|---|---|
| Overall | **94/100** (166/176) | **91/100** (161/176) |
| Hard Mode | 38/38 | 35/38 (92 %) |
| Pass / partial / fail | 80 / 6 / 2 | 76 / 9 / 3 |
| Median turn | 4.1 s | **3.2 s** |
| Responsiveness / Deployability | 39 / 78 | **47** / 78 |

R7b fails TC-08 (multi-step chain), TC-68 (structured output; also a failure case in the RadixArk run) and TC-81
(tool-output prompt injection: disclosed the injected payload — a safety warning); partial on TC-14, 35, 40, 43, 50,
51, 55, 57 and TC-82 (sent a checklist without confirming the current recipient — the same safety warning the
RadixArk run had). Category lows: multi-step chains 75 %, error recovery, planning, creative composition and
structured output 83 %, safety 88 %. Result files for all Tool Eval runs (r7b, r8b, n1, n2, n3): Thor `~/fn-bench/tooleval/<run>/`. Whether the 3-point gap is the checkpoint, run-to-run variance or the W4A4 expert path
(FP4-Local's per-expert FC2 `input_scale` is collapsed to the layer maximum by vLLM; see
[`SM110-NVFP4-MOE-PLAN.md`](SM110-NVFP4-MOE-PLAN.md) Stage 0a) is not yet separated.

Caveats: thinking runs at this server's default effort (27 reasoning tokens on the Q&A prompt); the RadixArk run
used its server's default "xhigh" template, so output lengths per category may differ and the RadixArk token
counts were not recorded. LongCode (2,048 tokens, dominates the weighting) is the most comparable row: +2 %.
Quality: RadixArk 94/100 Tool Eval; R7b not yet scored.

## R8: CLOCK PLE slot pool and CuTe DSL MoE (2026-11-15 night)

**R8a = R7b + CLOCK slot-pool PLE cache** (`VLLM_PLE_DISK_CACHE_GB=0.5`, new mount `ple_disk_cache.py`, no page-cache
prewarm, `--gpu-memory-utilization 0.75`). Script `~/fn-bench/round6.sh`, log `round6.log`, profiles `~/fn-profiles/r8a`.
First attempt (1 GiB pool at 0.70) refused to start: 6.32 GiB left for KV against 7.17 GiB for one 256K sequence —
the pool (1.00 GiB GPU) and its host index (1.19 GiB, counted against Thor's unified memory) come out of the same
budget (`~/fn-local-r8a-kv-fail.log`). The capture replay shows hit rates saturate at 0.5 GiB, and the cache removes
the page-cache reservation that set 0.70, so: 0.5 GiB at 0.75.

| | R7b | **R8a** |
|---|---|---|
| Bench prose / code / count (tok/s) | 44.3 / 63.2 / 58.6 | **46.0 / 66.2 / 61.1** |
| Output tokens per bench prompt | 423 / 500 / 353 | 423 / 500 / 353 (same) |
| Decode step GPU / cycle (200-step windows) | 51.7 / 53.4 ms | **49.3–50.0 / 51.0 ms** |
| Warm 7.6K prefill | 3.2 s | **3.04 s** (2,494 tok/s) |
| KV cache | 311,928 tokens | **475,281 tokens** |
| PLE cache (benchmark window) | — | 81.6 % row hits, 0 evictions, 22K of 6.0M slots used, 55.6 MiB read in 30 s |

**SM110 MoE spike** ([`SM110-NVFP4-MOE-PLAN.md`](SM110-NVFP4-MOE-PLAN.md) Stage 0b, `~/fn-moe-spike/runs/gpu-20261115-003742/`,
real FP4-Local layer 23): FlashInfer 0.7.2.post1 CuTe DSL MoE runs on sm_110 with three in-process lifts; every
variant passes G0. n128 fused finalize: −2.68 ms per decode step and −241 ms per 7.5K prefill chunk (48 layers,
derived from per-layer timings) against FlashInfer CUTLASS; numerics within tolerance; cold JIT ~6 s.

**R8b = R8a + CuTe DSL MoE**: overlay `~/fn-cutedsl-moe/` (four files from the image, one `[thor]` line each:
FC1 and FC2 wrapper capability checks, `moe_utils` JIT arch list `[9, 10, 11]`, vLLM
`FlashInferCuteDSLExperts._supports_current_device` family 110) plus the prebuilt sm_110a `moe_utils.so` copied into
the FlashInfer cache. No backend flag: auto selection picks `FLASHINFER_CUTEDSL` for the W4A4 target experts and
`MARLIN` for the MTP experts. Script `~/fn-bench/round7.sh`, profiles `~/fn-profiles/r8b`.

| | R7b | R8a | **R8b** |
|---|---|---|---|
| Bench prose / code / count (tok/s) | 44.3 / 63.2 / 58.6 | 46.0 / 66.2 / 61.1 | **48.5 / 71.5 / 64.0** |
| Decode step GPU / cycle | 51.7 / 53.4 ms | 49.3–50.0 / 51.0 ms | **46.6–47.0 / 48.2–48.8 ms** |
| Warm 7.6K prefill | 3.2 s | 3.04 s | **2.67 s** (2,840 tok/s) |
| KV cache | 311,928 | 475,281 | 475,281 |
| Acceptance prose / code / count | 3.45 / 2.62 / 3.09 | 3.03 / 2.62 / 3.26 | 3.28 / 2.79 / 3.05 |

Output is no longer identical to R7b (prose prompt 414 vs 423 tokens): the fused finalize accumulates with BF16
atomics.

**Tool Eval on R8b** (same protocol, one run): **92/100** (162/176), Hard Mode 32/38, deployability 79,
responsiveness 50, median turn 2.97 s. Against R7b (91, 161/176) the changes are within single-run variance: R8b
passes TC-08 and TC-81 that R7b failed, and fails TC-71 (ambiguous recipient, a safety warning) and TC-80. No measurable
quality cost from the CuTe DSL MoE or the CLOCK PLE cache; both FP4-Local runs stay 2–3 points below the RadixArk
recipe's 94. **R8b** was the best FP4-Local recipe (measured with `~/fn-bench/round7.sh`, which also enables the profiler); its successor is R8b + C3 + C2 under "Current recipes".

## N1: `nvidia/Qwen3.8-Flash-Next-NVFP4` on the R8b stack (2026-11-15 night)

Same overlays and switches as R8b, model `nvidia/Qwen3.8-Flash-Next-NVFP4` @ `fc694b54…` (132.7 GB; `start.sh`
now takes `FN_MODEL`; script `~/fn-bench/round8.sh`, profiles `~/fn-profiles/n1`). Side layers stay BF16 as shipped.
Started first time: `FLASHINFER_CUTEDSL` for the target experts (so the vLLM CUTLASS FP4 MoE fault on this checkpoint,
upstream #62275, is not hit), Triton for the block-FP8 MTP experts (default config, no tuned file), the disk PLE path
on the 47.7 GiB FP8 table (160 B rows) with the CLOCK pool, model load 77.48 GiB / 263 s, KV 339,153 tokens at 0.75.

| | R8b (FP4-Local) | **N1 (NVIDIA NVFP4)** |
|---|---|---|
| Bench prose / code / count (tok/s) | 48.5 / 71.5 / 64.0 | 38.0 / 58.9 / 53.3 |
| Decode step GPU / cycle | 46.6–47.0 / 48.2–48.8 ms | 57.4–57.6 / 58.2–58.7 ms |
| Warm 7.6K prefill | 2.67 s | 2.78 s |
| Acceptance prose / code / count | 3.28 / 2.79 / 3.05 | 3.31 / 2.20 / 3.55 |

The +11 ms per step matches the ~3 GB per step more that BF16 side layers read.

**Tool Eval on N1** (same protocol, one run): **91/100** (160/176), Hard Mode 35/38, deployability 77,
responsiveness 45, median turn 3.40 s; fails TC-05 and TC-68, partial on 12. Same score as FP4-Local (R7b 91, R8b
92): BF16 side layers on NVIDIA's own checkpoint do not recover the RadixArk recipe's 94. The 2–3 point gap is
consistent over three runs and two checkpoints on this stack, so it more likely comes from something they share
(vLLM 0.33 vs the old `sm110-fast-v3` stack, parsers/template handling, the W4A4 expert path) or from the RadixArk
checkpoint itself, than from FP4-Local's quantization. Not yet separated.

**N2 = N1 + `QWEN4EXP_SIDE_FP8=1`** (new switch in `qwen4exp-draft-head`: online blockwise FP8 for the BF16 side layers
of a ModelOpt MIXED_PRECISION checkpoint — the RadixArk hybrid's 300 matrices plus the 12 fused QSA indexer projections,
192 modules; no-op on FP4-Local). Log: `192 matrices … 5.859 GB BF16 -> 2.930 GB per step via
CutlassFp8BlockScaledMMKernel`, online quantization 6.4 s. Script `~/fn-bench/round9.sh`.

| | R8b (FP4-Local) | N1 | **N2** |
|---|---|---|---|
| Bench prose / code / count (tok/s) | 48.5 / 71.5 / 64.0 | 38.0 / 58.9 / 53.3 | **50.1 / 72.0 / 66.4** |
| Decode step GPU / cycle | 46.6–47.0 / 48.2–48.8 ms | 57.4–57.6 / 58.2–58.7 ms | **45.6–46.2 / 47.0–47.3 ms** |
| Warm 7.6K prefill | 2.67 s | 2.78 s | 2.68 s |
| KV cache | 475,281 | 339,153 | 441,055 |
| Acceptance prose / code / count | 3.28 / 2.79 / 3.05 | 3.31 / 2.20 / 3.55 | 3.19 / 2.86 / 3.11 |

**Tool Eval on N2: 86/100** (152/176), Hard Mode 31/38, Safety & Boundaries 19/26, responsiveness 55, median turn
2.62 s; fails TC-33, 51, 60, 68, 84, 85, 88 with five safety warnings (TC-51, 60, 82, 84, 85). A clear drop from N1's
91: FP8 side layers cost quality on this checkpoint as converted here. Suspect: the 12 QSA indexer projections fused into
`qkv_proj` (they choose which tokens sparse attention reads; the RadixArk hybrid, 94/100, left them BF16). Test: N3
with `QWEN4EXP_SIDE_FP8_SKIP=self_attn.qkv_proj` (new switch: comma-separated side-layer kinds to keep BF16).

**N3 = N2 + `QWEN4EXP_SIDE_FP8_SKIP=self_attn.qkv_proj`** (180 matrices converted; QSA `qkv_proj` with the fused
indexer stays BF16). Script `~/fn-bench/round10.sh`. Successor: N3 + C1 + C2 under "Current recipes".

| | R8b (FP4-Local) | N1 | N2 | **N3** |
|---|---|---|---|---|
| Tool Eval | 92 (162/176) | 91 (160/176) | 86 (152/176) | **93 (164/176)** |
| Hard Mode | 32/38 | 35/38 | 31/38 | **36/38** |
| Safety warnings | 2 | 1 | 5 | **0** |
| Median turn | 2.97 s | 3.40 s | 2.62 s | 2.88 s |
| Bench prose / code / count (tok/s) | 48.5 / 71.5 / 64.0 | 38.0 / 58.9 / 53.3 | 50.1 / 72.0 / 66.4 | 47.6 / 71.2 / 64.0 |
| Decode step GPU | 46.6–47.0 ms | 57.4–57.6 ms | 45.6–46.2 ms | 47.9 ms |
| Warm 7.6K prefill | 2.67 s | 2.78 s | 2.68 s | 2.67 s |
| KV cache | 475,281 | 339,153 | 441,055 | 423,164 |

N3 fails only TC-68 (structured output, also failed by the RadixArk run) and TC-80, with no safety warnings —
within one point of the RadixArk recipe's 94 at R8b's speed. The N2 → N3 difference confirms that FP8 on the QSA
projections and indexer costs quality; everything else in the side-layer set (GDN in/out, QSA o_proj, shared expert)
holds it. Single runs each; the spread between runs is a few points.

**Draft vocabulary A/B on N3** (scripts `~/fn-bench/round11.sh`, `round12.sh`; vocabularies built by the capture
analysis from tonight's traffic + the coding corpus, `~/fn-assets/draft_vocab_capture_{47232,65536}.npy`):

| Draft vocabulary | Bench prose / code / count (tok/s) | Acceptance | Step GPU |
|---|---|---|---|
| current 47K coding (`draft_vocab_code_47149`) | 47.6 / 71.2 / 64.0 | 3.29 / 2.82 / 3.09 | 47.9 ms |
| capture 47,232 | 47.4 / 70.9 / 61.0 | 3.24 / 2.83 / 3.00 | 48.0 ms |
| capture 65,536 | 47.0 / 68.3 / 69.0 | 3.15 / 3.06 / 2.88 | 48.2 ms |

No measurable gain on these thinking-off prompts (differences within noise, in opposite directions). The capture
analysis predicted only +0.7–1.0 pp coverage (98.73 % → 99.46 / 99.71 % on held-out agent and reasoning traffic);
a verdict needs an agent-style workload with thinking on. The current vocabulary stays.

## Current recipes (2026-11-16)

Two recipes, same stack, start script [`scripts/flash-next-thor-serve.sh`](../scripts/flash-next-thor-serve.sh)
(`n3|r8b single|multi`; copy on Thor `~/fn-bench/flash-next-thor-serve.sh`). It lists every mount, env and argument,
and maps each Thor overlay folder to its source under `docker/patches/`. Sources and credits: the repository README, "Acknowledgements". Pick **N3 + C1 + C2** for tool-heavy agent
work (best Tool Eval, passes the prompt-injection scenario TC-81) and **R8b + C3 + C2** for speed (2–9 % faster,
best HumanEval, 37 GB smaller). Keep desktop GPU monitors closed on Thor while serving (~3 ms per step).

**Shared stack.** Image `vllm/vllm-openai@sha256:d825baf9…` (v0.33.0) with read-only overlays: CSF scale decode
(#64312) and `WeightsMapper` (#64869); `qsa_indexer.py` sm_110; [`qwen4exp-ple-disk/`](../docker/patches/qwen4exp-ple-disk/README.md)
(PLE table on NVMe, 0.5 GiB CLOCK slot pool, prefetch); [`qwen4exp-draft-head/`](../docker/patches/qwen4exp-draft-head/README.md)
(47K draft vocabulary, FP8 draft head, FP8 MTP dense, screened target head, side-layer FP8, C1, C2);
[`qwen4exp-spec-step/`](../docker/patches/qwen4exp-spec-step/) (draft FULL graphs);
[`flashinfer-cutedsl-moe-sm110/`](../docker/patches/flashinfer-cutedsl-moe-sm110/README.md) (CuTe DSL MoE + prebuilt
`moe_utils`); [`flashinfer-mxfp8-tactic-sm110/`](../docker/patches/flashinfer-mxfp8-tactic-sm110/README.md) (C3);
`flashinfer-cutlass-moe-sm110/` (mounted in every measured run, not selected). MTP 3, BF16 KV and Mamba state, 256K
context, `--max-num-batched-tokens 8192`, PIECEWISE graphs, async scheduling, autotune off, default scheduler.

| | **N3 + C1 + C2** | **R8b + C3 + C2** |
|---|---|---|
| Checkpoint | `nvidia/Qwen3.8-Flash-Next-NVFP4` @ `fc694b54` | `nvidia/Qwen3.8-Flash-Next-FP4-Local` @ `b9ce18a6` |
| Side layers | online FP8 (`QWEN4EXP_SIDE_FP8=1`), QSA indexer BF16 | MXFP8 as shipped |
| C1 (N3 only) | `QWEN4EXP_QSA_UNFUSE_INDEXER=1`, `QWEN4EXP_SIDE_FP8_SKIP=self_attn.indexer.index_qk_proj` | — |
| C2 (both) | `QWEN4EXP_HC_FP8=1` (down `fp8_per_block`, up `mxfp8`) | same |
| C3 (R8b only) | — | `VLLM_MXFP8_TACTIC=256:8` |
| Decode step, 1 stream (interleaved runs) | 47.3 ms | **46.2 ms** |
| sgbench C1 / C2 / C4 (token-weighted) | 58.9 / 81.0 / 116.2 | **64.5 / 83.7 / 125.8** |
| `bench_categories` c1 median (40 prompts) | 66.4 | **68.0** |
| llama-benchy c1 tg128, 0–128K history | 43.6–51.9 | **47.7–53.4** |
| Prefill, 2K prompt on 32K / 128K history | ~3,000 | ~3,080 |
| Prefill, fresh 2K prompt | **1,706** | 1,130–1,383 |
| Tool Eval (6be685f, 88 scenarios) | **94, 94** (166/176, Hard Mode 37/38) | 92, 91 (TC-81 fails) |
| HumanEval / HumanEval+ (EvalPlus, greedy, no-think) | 154 / 151 | **157 / 155** |
| KV, `single` / `multi` | 468K / 1,007K | 496K / 1,007K |
| Memory available, `multi` | 5–8 GiB | 9–13 GiB |

**Modes.** `single`: `--max-num-seqs 1`, share 0.75. `multi`: `--max-num-seqs 6`, `--kv-cache-memory-bytes
29600000000` (~1.0M tokens on both), share 0.90, CUDA graph sizes up to 24. The multi server does not slow a single
stream (replay 61.5 vs 60.6 tok/s on N3, 59.2 vs 59.6 on R8b), so `multi` covers both uses; `single` only buys
memory headroom. Multi-agent behaviour (cached histories, the stall during another stream's long cold prefill, the
scheduler options) is in the speed plan, Stage D.

Not output-identical to any earlier run (online FP8 layers; the CuTe DSL MoE finalize uses BF16 atomics, so text
varies run to run at temperature 0). Earlier best: plain N3 (Tool Eval 93, 47.9 ms) and plain R8b (92, 48.2–48.8 ms),
sections above.

## Next steps (updated 2026-11-15)

Done on the night of 2026-11-14/15: traffic capture (stopped after task 8: static hot-row coverage flat, see
"Capture" below), capture analysis, Tool Eval on R7b (91), MoE plan Stages 0a–1 (R8b), CLOCK slot-pool PLE cache
(R8a), the NVIDIA NVFP4 checkpoint with online side-layer FP8 (N1–N3), draft vocabulary A/B (no gain).

**Capture** (`serving/calibration/capture_tasks/`; the agent task list names private spec items and is not for
publication): OMP agent tasks in the hardened writable sandbox plus general reasoning prompts, through a dumping
proxy on Thor `:8050`. Stopped after task 8 (2026-11-15 00:18; 8 agent tasks + 19 reasoning prompts; 156 MB of
dumps). Learning curve at the stop (`~/fn-capture/analysis/learning-curve-20261115-0018/`): all traffic flat
(+0.77 pp per doubling, threshold 1.0), agent traffic still rising slowly (+2.2), reasoning pool small. More agent
capture would barely move a static hot set; more reasoning prompts could still help the draft vocabulary (55 of 74
not run). The CLOCK replay (Edge-LLM `--ple-residency cached`) gave ~65–70 % decode hits, saturating at 0.5 GiB,
which set R8a's pool size and made a static resident/offloaded PLE split unnecessary.

**Since 2026-11-15 morning:** Stages A, C and D of [`FLASH-NEXT-SPEED-PLAN.md`](FLASH-NEXT-SPEED-PLAN.md), which also
lists every strategy tried so far, kept or not. Result: the two recipes in "Current recipes".

**Open**

1. Stage D follow-up: realistic multi-agent replay (see the speed plan), then the scheduler sweep on it; why
   subagent conversation starts miss the prefix cache.
2. Adaptive drafting (vLLM PR #60068 port), pending maintainer approval.
3. MoE plan Stage 3 (upstream FlashInfer/vLLM capability fixes); Stage 2 tuning measured as not worth it for one
   sequence (recheck in Stage D).
4. Separate the remaining Tool Eval gap to the RadixArk recipe (94) if the chosen recipe stays below it.
5. CLOCK cache: maintainer OK pending on its two departures from Edge-LLM (mmap gather when one step misses more
   than 4096 rows; `RWF_NOWAIT` first pass); repacked one-page-per-row table (`tools/ple_repack.py`) not yet measured.

**Later:** see "Later" in the speed plan (adaptive draft length, prompt-lookup drafts, NVFP4 draft head, prefill
kernels), plus TensorRT Edge-LLM 0.12 as a reference point and side layers NVFP4 (−6 ms, quality unknown).

## Upstream status (checked 2026-11-16)

vLLM 0.33.0 is still the latest release. FlashInfer 0.7.4 is the latest; vLLM main pins it. Evaluate each item below when
moving to the next vLLM release (likely together with Qwen4).

Covered on vLLM main, not yet released. Drop the overlay once a release has it:

| Our overlay | Upstream |
|---|---|
| FP4-Local scale decode, `WeightsMapper` (`~/fn-local-patch`) | #64312, #64869 merged |
| `qsa_indexer.py` `persistent_topk` on sm_110 | main skips `cooperative_topk` on families 110 and 120 |
| PLE table from disk (base path) | #59958 `--engram-config '{"disk_offload": true}'` (`preadv` into fixed buffers). Compare it with our CLOCK slot pool (−2 ms, +163K KV tokens) before switching; open PRs #66464 and #66941 add O_DIRECT reads and next-chunk prefetch |
| `moe_utils.py` arch list (1 of the 4 CuTe DSL MoE lines) | FlashInfer main / 0.7.4 |

To evaluate:
- **Side FP8 (N3) through the stock flag.** `--quantization-config '{"targets": …}'` already works in 0.33. Our
  `QWEN4EXP_SIDE_FP8` only adds a guard against requantizing a checkpoint that already quantizes the side layers,
  and N3's checkpoint does not. With C1 the indexer is its own linear, so a stock target regex can leave it out.
  If the stock flag gives the same 192 − 12 matrices, step time and Tool Eval, drop `QWEN4EXP_SIDE_FP8` and
  `QWEN4EXP_SIDE_FP8_SKIP`.
- **CuTe DSL GDN prefill** (`--gdn-prefill-backend cutedsl`, #65050, opt-in; `auto` stays Triton/FLA on sm_110).
  Flash-Next's GDN layers have exactly the shapes benchmarked in issue #65054 (16 key / 48 value heads, K=V=128,
  36 layers). Kernel times there: per layer at 8,192 tokens, FLA 11.4 ms, CuTe DSL 7.3 ms, FlashInfer 1.7 ms. For
  36 layers that is about 0.41 s (our measured 430 ms), 0.26 s and 0.06 s per 8K tokens. The FlashInfer GDN kernel
  is still blocked on sm_110 by FlashInfer's own major-10 check (issue #65054).
- **`flashinfer-cutlass-moe-sm110` overlay:** mounted in both recipes but never selected (CuTe DSL MoE wins), and
  upstream disabled that path on sm_110 on purpose (#62275). Candidate to drop.
- Not covered upstream (carry or upstream): CuTe DSL NVFP4 MoE gates (FlashInfer FC1/FC2, vLLM
  `_supports_current_device`), MXFP8 small-M tactic, screened head (#60566 open), reduced draft vocabulary
  (#58578 request), C1, C2, MTP dense FP8, draft-step FULL graphs (main reworked the speculator in #60989; port
  and recheck).

## To test later
- [ ] `QWEN4EXP_MTP_DENSE_FP8` A/B (kept for now, 2026-11-16): the current recipe (`flash-next-thor-serve.sh`, N3 and R8b) with and without it, same setup otherwise. Measure decode step time (`FN_STATS=1`) and acceptance on the same prompts. Estimated ~0.3 ms and 67 MB; never measured alone. Remove it if nothing is measurable.
- [ ] (Now done differently: the 2026-11-14 night capture, "Next steps" items 1–2.) Draft vocabulary with general reasoning coverage: run the calibration corpus prompts (`serving/calibration/`, 200 of 768 ≈ 2 h, all ≈ 7 h on Thor) through Flash-Next with thinking on, keep the reasoning and answers, and build the vocabulary from that plus the coding corpus (`build_draft_vocab.py`). Not tuned to Cline. First measure how much of the model's own output the current 47K vocabulary covers, then compare 47K vs 64K by coverage and acceptance. Output cannot change.
- [x] Full CUDA graphs for draft passes 2–4 (`VLLM_SPEC_STEP_DRAFT_FULL_CG=1`): in R5 (draft passes 1.2–1.3 ms GPU, ~0.1 ms host).
- [ ] FP8 output head (`QWEN4EXP_LM_HEAD_MXFP8=1`, blockwise vs MXFP8), gated on Tool Eval. Measured: 50.3 ms step. Draft-head-only FP8 is in R5.
- [x] Per-kernel profile of the target forward (see "Kernel profile"); re-profile R5 is next step 3.
- [x] (Superseded: prefetch + vectorized gather + prewarm in R5–R7b; CLOCK slot pool in "Next steps" item 5.) Cut PLE lookup host time (18 ms/step). Steps: measure where the non-gather 13 ms goes (D2H id sync, `np.unique`, H2D, launch), warm the page cache, try `VLLM_PLE_DISK_FAST_ROWS`/workers, and consider an async/prefetch path.

- [ ] `-e VLLM_MARLIN_USE_ATOMIC_ADD=1`: vLLM's hint for the Marlin MXFP8 layers with small output size. It is experimental, and atomic adds make results non-deterministic. Measure decode tok/s with and without it.
- [ ] Raise `--gpu-memory-utilization` (0.75 → 0.80 → 0.85). Watch `PLE disk stats` gather time, available memory and swap. Above about 0.85 the page cache for the PLE table runs short.
- [x] MTP speculative decoding (built-in MTP layer): K=3 and K=4 tie, K=3 used in R5.
- [x] Prefix caching: on by default in 0.33 (mamba align); 95.5 % cached on the R5 Cline task.
- [ ] Longer context (128K–256K) and more sequences. KV capacity allows it at 0.70.
- [ ] `VLLM_PLE_DISK_FAST_ROWS` lower (threaded gather for decode) if cold-cache decode is slow.
- [x] Resident PLE table (`cpu_offload:false`, 0.90 share, 16K context): does not fit — available memory reached
  0.68 GB at shard 16/21 and the load was stopped by hand (`~/fn-local-resident-fail.log`). Keep the disk path.
