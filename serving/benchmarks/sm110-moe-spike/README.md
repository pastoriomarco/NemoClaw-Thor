# SM110 NVFP4 MoE spike (Stage 0b)

Stage 0b of [`serving/docs/SM110-NVFP4-MOE-PLAN.md`](../../docs/SM110-NVFP4-MOE-PLAN.md): one container run on
Thor, no server, that decides gate G0. It compares the routed-expert MoE of Qwen3.8-Flash-Next
(H 2560, I 640, 512 experts, top-10, SwiGLU) on identical tensors:

| Candidate | What it is |
|---|---|
| `fi_cutlass` | (i) FlashInfer CUTLASS MoE, called with exactly the arguments of vLLM 0.33 `FlashInferExperts.apply` (R7b) |
| `fi_cutedsl_n128_fused` | (ii) FlashInfer 0.7.2.post1 CuTe DSL NVFP4 MoE as vLLM 0.33 `FlashInferCuteDSLExperts.apply` calls it, tactic `(128, ((128,128),(1,1)), ((128,128),(1,1)))`, atomic fused finalize |
| `fi_cutedsl_n256_fused` | same, N tile 256 for FC1 and FC2 |
| `fi_cutedsl_n128_det`, `_n256_det` | deterministic two-stage finalize (`use_fused_finalize=False`) |
| `fi_cutedsl_n128_fused_vllmdefault` | no explicit tactic, i.e. what vLLM would pick with `--disable-flashinfer-autotune` |
| (iii) Edge-LLM K1/K2 | documented stub only (`edge_llm_fallback` in `spike.py`); plan section 6.2 |

The measured region of every candidate is "activation NVFP4 quantization + experts" (what differs between
the backends). Router/top-k is identical in both and excluded. Weight preparation uses vLLM's own functions
for each backend (`prepare_nvfp4_moe_layer_for_fi_or_cutlass`, `prepare_nvfp4_moe_layer_for_flashinfer_cutedsl`,
`nvfp4_swizzled_scale_to_cutedsl_mma_view`, alpha folding as in each experts class), activation-scale variant (A)
(one shared scale, max over experts) as in R7b.

The sm_110 gates are lifted **in the spike's process only** (`sm110_lift.py`, no file in the image changes):

| Lift | File:line in the image (FlashInfer 0.7.2.post1) | Overlay equivalent |
|---|---|---|
| L1 | `flashinfer/fused_moe/cute_dsl/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py:686-692` `if major != 10: raise` | accept `major in (10, 11)` |
| L2 | `flashinfer/fused_moe/cute_dsl/blockscaled_contiguous_grouped_gemm_finalize_fusion.py:511-517` same | same |
| L3 | `flashinfer/jit/moe_utils.py:86-88` `supported_major_versions=[9, 10]` | `[9, 10, 11]` |

vLLM's own gate (`flashinfer_cutedsl_moe.py:88-94`, family 100) is not on this path; it is the fourth line a
Stage 1 overlay needs.

## Files

| File | Purpose |
|---|---|
| `run.sh` | runner: `gpu`, `dry` (CPU only, `--runtime runc`), `prebuild` (CPU only) |
| `spike.py` | the spike |
| `sm110_lift.py` | in-process gate lifts L1-L3 |
| `prebuild_moe_utils.py` | builds FlashInfer's `moe_utils` JIT module (moe_sort, memset, unpermute) for sm_110a on CPU |
| `xcompile_sm110a.py` | CPU-only cross-compile of the CuTe DSL FC1/FC2 kernels for sm_110a (used by `dry`) |

On Thor: `~/fn-moe-spike/code/` (this directory), `~/fn-moe-spike/fiws/` (FlashInfer JIT workspace with the
prebuilt `moe_utils.so` for sm_110a, and the CuTe DSL file cache), `~/fn-moe-spike/runs/<mode>-<stamp>/`
(outputs).

## Commands

Prerequisite already done on 2026-11-15 (CPU only, while serving): `~/fn-moe-spike/code/run.sh prebuild`
built `moe_utils` for sm_110a in 543 s on 3 CPUs; the `.so` is in `~/fn-moe-spike/fiws`, so the GPU run
does not pay it.

GPU run (needs the GPU free: `run.sh` refuses to start while `fn-local` runs):

```bash
~/fn-moe-spike/code/run.sh gpu                                  # FP4-Local layer 23, all phases
~/fn-moe-spike/code/run.sh gpu --weights nvfp4 --phases jit numerics   # optional: NVIDIA checkpoint numerics
```

Useful options: `--layer N`, `--weights fp4local|nvfp4|random`, `--variants n128_fused n256_fused n128_det
n256_det vllmdefault`, `--routings uniform skewed`, `--decode-t ...`, `--prefill-t ...`, `--replays 200`,
`--calls-per-graph 8`, `--phases jit numerics timing determinism profile jitwarm`.

CPU dry run (safe while serving; validates imports, lifts, CSF decode + weight loading, vLLM weight prep for
both layouts, the torch reference, argument binding against the installed signatures, and cross-compiles the
CuTe DSL FC1/FC2 kernels for sm_110a):

```bash
~/fn-moe-spike/code/run.sh dry [--experts 512] [--weights nvfp4]
```

## Expected run time (GPU)

About 10 to 20 minutes, bounded by `timeout 3600`: weights of one layer (1.42 GB read + 1,536 CSF decodes,
under a minute), cold JIT (moe_utils load under 1 s; CuTe DSL compile about 1 s per kernel on CPU in the dry
run, so seconds per variant), numerics (reference loop over experts, 1 to 3 min), timing (2 routings x 10
token counts x 6 candidates; graph replays for T <= 512 at 200 replays x 8 calls, eager 20 iterations for
2,048 / 7,488 / 8,192), determinism (20 runs at T = 4 and 7,488), profile (5 replays each at T = 4 and
7,488), and a second process for the warm-cache start.

## Phases and outputs

All in `~/fn-moe-spike/runs/gpu-<stamp>/` (a few hundred KB; no traces are written):

| Output | Content |
|---|---|
| `log.txt` | full console log |
| `results.json` | everything below, machine-readable |
| `summary.md` | tables: timing (ms per MoE call = one layer), numerics, determinism, profile split, G0 verdict |
| `profile_T{4,7488}_<cand>.json` | kernel time per call by category (`quant_in`, `cutlass_*`, `dsl_sort_routing`, `dsl_fc1_gather_swiglu`, `dsl_memset`, `dsl_fc2_finalize`, `dsl_unpermute`) and the top kernels |

| Phase | What | Notes |
|---|---|---|
| `env` | versions, device (cc, SMs, smem, L2), lifts | |
| `jit` | moe_utils load/build time; first call of each CuTe DSL variant (= cold compile) with errors and tracebacks | a failure here is the main G0 risk |
| `numerics` | each candidate vs a torch fp32 reference of the checkpoint (dequantized weights, W4A4 with variant-A activation quantization, `ref_nvfp4_quant` from vLLM); also the W4A16 reference; T = 1, 4, 64, 7,488; edge cases: 127/128/129 tokens all routed to the same 10 experts, a zero router-weight row | rel L2, per-token cosine (min, median), max abs, non-finite count |
| `timing` | routings `uniform` and `skewed` (Zipf 1.1); T = 1, 4, 5, 8, 16, 64, 512 under CUDA-graph replay (8 different inputs/routings per graph, like successive layers, so weights are not reused from L2); T = 2,048, 7,488, 8,192 eager | ms per MoE call; distinct experts per call reported |
| `determinism` | 20 runs at T = 4 (graph replay) and 7,488 (eager), bit-equality against the first | atomic vs two-stage finalize |
| `profile` | torch profiler on replays at T = 4 and eager at 7,488 | split setup / FC1 / FC2 / finalize |
| `jitwarm` | a child process with the same CuTe DSL file cache: first call incl. moe_utils load | server-restart cost |

## Mapping to gate G0 (plan section 8)

`results.json["G0"]` holds a per-variant verdict computed as follows:

| G0 condition | Field | Rule used |
|---|---|---|
| compiles and runs on sm_110 with Python-level changes only, all token counts, no crash | `runs_all_token_counts` | first call OK and every timing/numerics case OK |
| numerics within tolerance of the reference | `numerics_within_1p5x_cutlass` | rel L2 vs reference <= 1.5 x the CUTLASS path's rel L2, no non-finite values, every numerics case |
| decode >= 10 % faster and prefill not slower, or prefill >= 20 % faster and decode not slower | `decode_T4_ratio_vs_cutlass`, `prefill_T7488_ratio_vs_cutlass`, `perf_gate` | uniform routing, T = 4 (MTP-3 verify at batch 1) and T = 7,488 (R7b chunk); savings x48 layers also given |
| cold JIT acceptable (proposed 5 min) | `cold_jit_total_s`, `cold_jit_ok` | moe_utils load (prebuilt) + all first calls; the one-time moe_utils build (543 s on 3 CPUs) is reported separately and is cached across restarts in the FlashInfer JIT cache |

Interpretation notes:
- In R7b's decode profile one CUTLASS MoE call at T = 4 costs about 0.40 ms (0.31 grouped GEMM + 0.09
  quant/prep/finalize). 10 % of that is about 0.04 ms per layer, about 1.9 ms per step: the plan's "about 2 ms".
  Compare `decode_saving_ms_per_step_48_layers` with the plan's 1.3 to 3.5 ms estimate.
- Determinism is reported, not gated (plan section 7 decides it later).
- If a CuTe DSL variant fails at first call, `results.json["jit"][<variant>]` carries the exception and
  traceback; that is the evidence for whether the failure is FlashInfer-specific (go to Stage 1b, Edge-LLM) or
  a DSL/hardware issue (stop).
