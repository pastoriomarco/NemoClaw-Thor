# Qwen3.8-Flash-Next FP4-Local on Jetson AGX Thor (vLLM 0.33 + overlays)

Status: **speed search in progress** (2026-11-14). Best configuration so far: **R7b** (see "Current best
recipe: R7b"). Best strictly output-identical configuration: R5. Neither is a recommended recipe until Tool
Eval scores it.

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
| R3 | R2 + `VLLM_SPEC_STEP_TIMING=200` + draft vocabulary 47,232 ids (`QWEN4EXP_DRAFT_VOCAB`) + MTP dense parts online blockwise FP8 (`QWEN4EXP_MTP_DENSE_MXFP8=1`, `QWEN4EXP_FP8_FORMAT=fp8_per_block`) | **37.5 / 56.7 / 51.1** | KV 456,208 tokens; model load 72.7 GiB; acceptance 2.5 prose, 3.65 code, 4.0 counting (was 4.5: the coding vocabulary covers number words less well) |

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
      QWEN4EXP_MTP_DENSE_MXFP8=1  QWEN4EXP_FP8_FORMAT=fp8_per_block
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

## Current best recipe: R7b (2026-11-14 night)

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
      QWEN4EXP_MTP_DENSE_MXFP8=1 QWEN4EXP_FP8_FORMAT=fp8_per_block QWEN4EXP_DRAFT_HEAD_FP8=1
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

Caveats: thinking runs at this server's default effort (27 reasoning tokens on the Q&A prompt); the RadixArk run
used its server's default "xhigh" template, so output lengths per category may differ and the RadixArk token
counts were not recorded. LongCode (2,048 tokens, dominates the weighting) is the most comparable row: +2 %.
Quality: RadixArk 94/100 Tool Eval; R7b not yet scored.

## Next steps (proposed order)

1. **R5 + screened target head** (`QWEN4EXP_LM_HEAD_SCREENED=1`, vLLM PR #60566 backport): FP8 screen over the
   248K-row head, exact BF16 logits for the rows that can still be in the top k. Same results as the BF16 head
   except exact ties; est. −2.3…2.6 ms/step, +0.66 GB. One run with `QWEN4EXP_LM_HEAD_SCREEN_CHECK` first.
2. **A1 boot test** (FlashInfer CUTLASS MoE on sm_110).
3. **Short profile of R5**: is the PLE GPU gap gone; where the remaining 53 ms goes.
4. **Proxy fix** + `--enable-prompt-tokens-details`.
5. **Adaptive draft length + prompt-lookup drafts** (needs an agent; acceptance is the largest remaining decode pool
   on agent tasks, where tool output repeats prompt text).
6. Depending on 2: D1 or D2 — now planned in [`SM110-NVFP4-MOE-PLAN.md`](SM110-NVFP4-MOE-PLAN.md) (FlashInfer CuTe DSL MoE gate overlay first, Edge-LLM kernel port as fallback).
7. **Quality gate**: Tool Eval on R5 and on the FP8-head variant before either becomes a recipe.
8. Later: hyperconnections BF16→FP8 (−2.6 ms, quality unknown), memory share 0.70 vs 0.75 for the PLE page cache,
   reasoning draft vocabulary night run, sgbench vs the old 46.7 tok/s.

## To test later
- [ ] Draft vocabulary with general reasoning coverage: run the calibration corpus prompts (`serving/calibration/`, 200 of 768 ≈ 2 h, all ≈ 7 h on Thor) through Flash-Next with thinking on, keep the reasoning and answers, and build the vocabulary from that plus the coding corpus (`build_draft_vocab.py`). Not tuned to Cline. First measure how much of the model's own output the current 47K vocabulary covers, then compare 47K vs 64K by coverage and acceptance. Output cannot change.
- [x] Full CUDA graphs for draft passes 2–4 (`VLLM_SPEC_STEP_DRAFT_FULL_CG=1`): in R5 (draft passes 1.2–1.3 ms GPU, ~0.1 ms host).
- [ ] FP8 output head (`QWEN4EXP_LM_HEAD_MXFP8=1`, blockwise vs MXFP8), gated on Tool Eval. Measured: 50.3 ms step. Draft-head-only FP8 is in R5.
- [x] Per-kernel profile of the target forward (see "Kernel profile"); re-profile R5 is next step 3.
- [ ] Cut PLE lookup host time (18 ms/step). Steps: measure where the non-gather 13 ms goes (D2H id sync, `np.unique`, H2D, launch), warm the page cache, try `VLLM_PLE_DISK_FAST_ROWS`/workers, and consider an async/prefetch path.

- [ ] `-e VLLM_MARLIN_USE_ATOMIC_ADD=1`: vLLM's hint for the Marlin MXFP8 layers with small output size. It is experimental, and atomic adds make results non-deterministic. Measure decode tok/s with and without it.
- [ ] Raise `--gpu-memory-utilization` (0.75 → 0.80 → 0.85). Watch `PLE disk stats` gather time, available memory and swap. Above about 0.85 the page cache for the PLE table runs short.
- [x] MTP speculative decoding (built-in MTP layer): K=3 and K=4 tie, K=3 used in R5.
- [x] Prefix caching: on by default in 0.33 (mamba align); 95.5 % cached on the R5 Cline task.
- [ ] Longer context (128K–256K) and more sequences. KV capacity allows it at 0.70.
- [ ] `VLLM_PLE_DISK_FAST_ROWS` lower (threaded gather for decode) if cold-cache decode is slow.
- [x] Resident PLE table (`cpu_offload:false`, 0.90 share, 16K context): does not fit — available memory reached
  0.68 GB at shard 16/21 and the load was stopped by hand (`~/fn-local-resident-fail.log`). Keep the disk path.
