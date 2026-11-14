# SM110 NVFP4 MoE kernels for Qwen3.8-Flash-Next in vLLM: investigation plan

Status: **Proposed — to investigate** (2026-11-14). Nothing executed yet. Next step: Stage 0a (no GPU).

Goal: bring NVIDIA's tcgen05 NVFP4 MoE kernels (the fused gather + grouped GEMM + SwiGLU + requant FC1 and the
grouped GEMM + router-weight + scatter-reduce FC2, the "D2" route of
[`QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md`](QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md)) to vLLM on Jetson AGX Thor for both
`nvidia/Qwen3.8-Flash-Next-FP4-Local` and `nvidia/Qwen3.8-Flash-Next-NVFP4`, measured against the current best
recipe R7b, with go/no-go gates before any integration work. Baseline and measurements: R5–R7b sections of that
document.

Origin: design by a Fable 5.2 (high effort) analysis agent on 2026-11-14, from the TensorRT Edge-LLM 0.12.0
sources, the vLLM 0.33.0 sources, FlashInfer 0.7.2.post1 (upstream pages) and our Thor measurements. It is kept
as written; it has not been reviewed against a GPU run.

Labels: **[code]** read in source (file:line), **[web]** public upstream page, **[derived]** computed or inferred
from labelled facts, **[speculation]** plausible but unverified.

Path roots used below:
- `E/` = TensorRT Edge-LLM source at tag `v0.12.0` (`github.com/NVIDIA/TensorRT-Edge-LLM`)
- `V/` = the `vllm/` package of vLLM `v0.33.0` (as shipped in the image pinned in `QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md`)
- `DOC` = [`QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md`](QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md)
- `K1` = `E/kernelSrcs/nvfp4_moe_cutedsl/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py` (FC1)
- `K2` = `E/kernelSrcs/nvfp4_moe_cutedsl/blockscaled_contiguous_grouped_gemm_finalize_fusion.py` (FC2)

---

## 0. Summary

**Recommendation.** Do not start with a port of the Edge-LLM kernels. Start with a half-day spike that
measures three candidates on identical tensors, and make the main route the one that is already wired
into vLLM 0.33: FlashInfer's CuTe DSL MoE (`FLASHINFER_CUTEDSL`), which is the same kernel family as
Edge-LLM's SM110 group and is blocked on sm_110 only by capability gates. Keep the Edge-LLM port as the
fallback if FlashInfer's copy fails on sm_110 for reasons beyond the gates.

**Why the framing changes.**
1. FlashInfer 0.7.2.post1 (the version pinned by vLLM 0.33) ships kernels with the same class name,
   the same NVIDIA BSD-3 header, the same warp layout and the same default tile (128x128, 1 CTA,
   cluster 1x1) as Edge-LLM's `nvfp4_moe_cutedsl` [web]. vLLM 0.33 already has the experts class,
   the weight interleave, the scale layout view and the oracle entry for it [code]. Most of the
   "remaining glue" in the framing exists.
2. Nothing needs re-exporting for 512 experts / top-10. Expert count is a runtime argument, FC1 does
   not use top-k, FC2 takes it at run time, and the Edge-LLM runner accepts 512 experts and top-k up to 32
   [code]. The "128 / top-8" text is stale README prose.
3. There is no CuTe DSL version mismatch on paper: Edge-LLM 0.12 and vLLM 0.33 both pin
   `nvidia-cutlass-dsl 4.8.0` [code][web]. In-process JIT is the natural path; AOT objects are not needed.
4. Decode cannot gain much. The routed-expert GEMM already runs at the DRAM limit [derived from DOC and
   the earlier report]. Only the routing/prep/finalize share can shrink.

**Expected gain** (medium-low confidence, to be replaced by the spike's numbers):
decode step 51.7 ms to about 49 ms (range 48.2 to 50.4, i.e. 2.5 to 7 %); warm 7.6K prefill 3.2 s to about
2.9 s (range 2.85 to 3.05, i.e. 5 to 11 %); Cline-type workload about +5 % overall (range +2 to +9 %).

**First concrete step.** Stage 0a (no GPU): confirm the image facts and split the existing R7b profile
buckets by kernel name. Then Stage 0b: one microbenchmark container run on Thor when it is free.

---

## 1. Feasibility facts from the code

### 1.1 Expert count, top-k, tiles, cluster

| Item | Fact | Source |
|---|---|---|
| Expert count | Runtime argument `l`; "one compiled kernel serves any expert count" | [code] `K1:3857-3867`, `K2:2826-2833` |
| Export dummies | `--dummy-experts 128 --dummy-top-k 8` only drive tracing: "dummy value only drives AOT tracing, not the cubin" | [code] `E/kernelSrcs/nvfp4_moe_cutedsl/export_fc1_kernel.py:155-156`, `export_fc2_kernel.py:134-135`, `E/kernelSrcs/build_cutedsl.py:1584-1700` |
| Top-k in FC1 | `self.topk = topk` is stored and never read; A rows are indexed directly by the mapping tensor | [code] `K1:333` (only occurrence), `K1:1714-1717`, `K1:1736` |
| Top-k in FC2 | Runtime `top_k: cutlass.Int64`; `token_idx = expanded_idx // topK` uses the run-time shape | [code] `K2:565`, `K2:2131-2134`, `K2:2813-2814`, `K2:2921-2923` |
| Runner limits | `kSupportedNumExperts = {128, 256, 512}`, `kMaxTopK = 32` | [code] `E/cpp/kernels/moe/nvfp4_cutedsl/cuteDslNvfp4MoeSm110Runner.h:90-92`, `.cpp:143-172` |
| Stale docs | README says `num_experts = 128`, `top_k = 8`; plugin README says `E == 128`, `0 < top_k <= 8` | [code] `E/kernelSrcs/nvfp4_moe_cutedsl/README.md:63-65`, `E/cpp/plugins/nvfp4MoePlugin/README.md:28,42-43` |
| 512 / top-32 is a product case | "NVFP4 MoE ... Nemotron 3 Super 120B-A12B checkpoints with 512 routed experts and top-32 routing" | [code] `E/CHANGELOG.md:95`; test `E/unittests/cpp/kernels/moe/nvfp4MoeCuteDslSm110Tests.cu:1033-1061` |
| M tile | 128 only (1 CTA): `M_TILE_SIZE = 128`; `kRowTileAlign = 128` | [code] `export_common.py:25`, `Sm110Runner.h:84`, README `:220` |
| N tile | 128 or 256 exported; the runner launches n128 only | [code] `export_fc1_kernel.py:174`, `Sm110Runner.h:125-128`, README `:58-59` |
| Cluster | `(1, 1)` in both exports | [code] `export_fc1_kernel.py:50`, `export_fc2_kernel.py:48` |
| Kernel-class limits | M tile 128 or 256, N tile 64/128/192/256, cluster M,N <= 4 and power of two | [code] `K1:3643-3690`, `K2:2622-2664` |
| Shared-memory size | Taken from the `"sm_100"` table, also on sm_110 | [code] `K1:408`, `K2:293` |
| Active clusters | Runner passes the SM count (20 on Thor) as `max_active_clusters` | [code] `Sm110Runner.cpp:299,338` |

### 1.2 Alignment against Flash-Next (H = 2560, I = 640, FC1_N = 1280)

Rules: `H % 128 == 0`, `I % 64 == 0`, `FC1_N % 128 == 0` [code] `Sm110Runner.cpp:161-170`.

| Check | Value | Result |
|---|---|---|
| H % 128 | 2560 = 20 x 128 | pass |
| I % 64 | 640 = 10 x 64 | pass |
| FC1_N % 128 (and % 256) | 1280 = 10 x 128 = 5 x 256 | pass for n128 and n256 |
| FC2 N = H, % 256 | 2560 = 10 x 256 | pass for n256 |
| Scale columns, FC1: H/16 = 160, multiple of 4 | 40 K-tiles | pass, no padding |
| Scale columns, FC2: I/16 = 40, multiple of 4 | 10 K-tiles | pass, no padding |
| SwiGLU interleave granularity 64 rows | 640 = 10 chunks | pass |
| FlashInfer hidden alignment 256 | 2560 = 10 x 256 | pass, no padding ([code] `V/model_executor/layers/quantization/utils/flashinfer_utils.py:297-309`) |

Shapes confirmed from the checkpoint header: gate/up `[640, 1280]` packed bytes, down `[2560, 320]`,
MTP scales `[640, 160]` and `[2560, 40]` [code] FP4-Local safetensors header.

### 1.3 Dtypes

- Edge-LLM plugin I/O is FP16 only (`ioDtype != kIODT_FP16` is rejected) [code] `Sm110Runner.cpp:147`.
- The FC2 exporter accepts `--output_dtype bf16|fp16`; only fp16 is shipped [code] `export_fc2_kernel.py:154`,
  `export_common.py:28-37`, README `:124-126`. The kernel has BF16 branches for the scatter-reduce
  [code] `K2:2190-2240`, `K2:2583`.
- FC1 clears the output with a fused FP16 zero store (`zero_f16x8`) [code] `K1:1493-1519`. The bit pattern is
  also a BF16 zero, so a BF16 FC2 can use it [derived].
- FlashInfer's copy defaults to BF16 output and accepts FP16 or BF16 [web, `fused_moe.py` at v0.7.2.post1].
- FP16 overflows above 65,504; BF16 does not. With a BF16 output variant the question does not arise [derived].

### 1.4 Activation

- Flash-Next: `hidden_act = silu`, no clamp or limit key in the config [code] FP4-Local `config.json`.
- Edge-LLM builder uses SwiGLU and softmax top-k for this model [code]
  `E/tensorrt_edgellm/builder/models/qwen4_exp/modeling_qwen4_exp_sparse_moe.py:49-50`.
- Kernel SwiGLU: `(alpha*up) * silu(alpha*gate)`, computed with `exp2(fastmath)` and `rcp_approx`,
  no clamp [code] `K1:3049-3106`. FC1 weights must be 64-row interleaved `(up, gate)` [code] README `:210-212`.
- Approximate reciprocals are also used in the FP4 requantization [code] `K1:2850-2950`. Results will match a
  reference within tolerance, not bit for bit [derived].

### 1.5 Scales and alphas

- Weight scale storage `[E, ceil(N/128), ceil((K/16)/4), 32, 4, 4]`, viewed as
  `(32, 4, n//128, 4, scale_k//4, l)` with order `(2,1,4,0,3,5)` [code] `K1:3892-3897`.
- vLLM builds the same view for FlashInfer: swizzle to `(E, M/128, 4, 32, K/4, 4)` then
  `view(num_groups, m_tiles, k_tiles, 32, 4, 4).permute(3,4,1,5,2,0)` [code]
  `V/model_executor/layers/quantization/utils/nvfp4_utils.py:13-45`, `flashinfer_fp4_moe.py:193-215`; [web]
  `flashinfer/cute_dsl/utils.py`. A byte-compare test belongs in Stage 1 (I did not prove byte identity).
- FC1: `alpha_val = alpha[e] * input_global_scale[e]`; requant uses `norm_const = 1 / down_input_scale[e]`
  [code] `K1:2720-2725`.
- FC2: `alpha_val * token_scale * down_input_scale[e]` [code] `K2:2134`.
- Activation input to FC1 is a linear (row-major) scale layout `[rows, K/16]` [code]
  `E/cpp/kernels/moe/fp4SupportKernels/fp4Quantize.h:41-54`.

### 1.6 Routing metadata

- `tile_idx_to_group_idx[tiles]`, `tile_idx_to_mn_limit[tiles]`, `permuted_idx_to_expanded_idx[permuted_m]`
  (padding = -1), `num_non_exiting_tiles[1]`, all int32 on the device [code]
  `E/cpp/kernels/moe/fp4SupportKernels/nvfp4MoeTypes.h:54-60`, `buildLayout.cu:41-116`.
- Bound: `permuted_m = pad128(routed_rows + E * 127)` [code] `Sm110Runner.cpp:69-73`.
  For 8,192 tokens x 10: `pad128(81,920 + 65,024) = 146,944` rows, about 47 MB of FP4 intermediate [derived].
- The layout builder is a single-block kernel, no host sync [code] `buildLayout.cu:124-146`.

### 1.7 Launch order and CUDA graphs (Edge-LLM)

- General path: layout build, routed FP4 quantization, FC1 (clears output, gather, GEMM, SwiGLU, requant),
  FC2 (GEMM, alpha, router weight, scatter-reduce). Four launches per layer [code] `Sm110Runner.cpp:269-345`.
- A fused layout+quant kernel exists only for `numTokens == 1` [code] `Sm110Runner.cpp:131-139`,
  `fp4Quantize.cu:62,443`. Our MTP-3 verify step has 4 tokens, so it would not be used as is [derived].
- PDL: FC1 and FC2 carry `griddepcontrol_wait` / `launch_dependents`; one artifact serves both launch modes
  [code] `K1:1493,3025`, `K2:1322,2267`, `E/cpp/kernels/moe/nvfp4_cutedsl/nvfp4MoePdlConfig.h:28-33`.
- NVIDIA's test captures `runner.run` in a CUDA graph and requires bit-equal output over replays
  (small cases, T <= 8) [code] `nvfp4MoeCuteDslSm110Tests.cu:773-818`.
- NVIDIA's accuracy threshold against a CPU reference is loose: median cosine >= 0.94 [code] same file `:872-874`.

### 1.8 CuTe DSL version, JIT and AOT

| Fact | Source |
|---|---|
| Edge-LLM pins `nvidia-cutlass-dsl 4.8.0` | [code] `E/kernelSrcs/README.md:43,77,201` |
| vLLM v0.33.0 `requirements/cuda.txt` pins `nvidia-cutlass-dsl[cu13]==4.8.0`, `flashinfer-python==0.7.2.post1`, `quack-kernels==0.6.5`, `apache-tvm-ffi==0.1.11` | [web] |
| vLLM 0.33 already JIT-compiles CuTe DSL kernels in process with `cute.compile(..., options="--enable-tvm-ffi")` and a warm-up registry | [code] `V/models/qwen4_exp/nvidia/ops/cute_dsl/hc_down_silu.py:59-110`, `V/model_executor/warmup/jit_warmup_cutedsl_helper.py:12,33-46`, `V/model_executor/warmup/cutedsl_warmup.py:95-112` |
| We already see such a JIT on Thor (`HcDownSiluMma`, slow first window) | [code] `DOC:169-170` |
| Edge-LLM builds for sm_110 without patching the DSL (`CUTE_DSL_ARCH` only) | [code] `E/kernelSrcs/cutedsl_utils/cutedsl_compile_wrapper.py:161-171` |
| In May (cutlass-dsl 4.5.x) `BlockScaledMmaOp.admissible_archs` lacked `sm_110a` and we patched it | [code] `serving/docker/mods/sm110a-fp4-dsl-unlock/run.sh`, `serving/docs/V9.1-TASK4-FP4-UNLOCK.md:43` |
| The shipped AOT pack contains the sm_110 objects (`libcutedsl_aarch64.a`, 54 MB, with `nvfp4_moe_sm110_*` headers) | [code] `E/kernelSrcs/cuteDSLPrebuilt/cutedsl_aarch64_sm_110_cuda13.tar.gz` (listing only) |

Implications:
- **JIT in process** (recommended): no binary artifacts, fits read-only file overlays, same pattern as the
  in-tree kernels. Cost: compile time at start, before graph capture. Compile time on Thor is unknown; the
  warm-up runs before capture already trigger it [derived].
- **AOT objects**: need a C++ shim linked into the image or a Python object loader. That breaks the
  "no image rebuild" rule unless cutlass-dsl can load exported objects from Python
  [speculation: I did not verify such a loader in 4.8.0]. Only worth looking at if JIT time is a problem.
- CUDA graphs: both paths give static launches. Shapes that change per step (rows, tiles) are run-time
  arguments or device tensors; nothing needs a host read [code] 1.6, 1.7.

### 1.9 Licence

`K1` and `K2` carry `SPDX-License-Identifier: BSD-3-Clause` (NVIDIA, 2025/2026) [code] `K1:1-2`, `K2:1-2`.
The export scripts and helpers are Apache-2.0. The framing's "Apache-2.0" is true for the repository, not
for the two kernel files. BSD-3 is compatible with vLLM and FlashInfer but the headers must stay.

---

## 2. How Edge-LLM runs Flash-Next's MoE

- It uses this group. `Qwen4ExpSparseMoeBlock.forward` calls `F.nvfp4_moe(...)` with `cfg.num_experts` (512),
  `cfg.num_experts_per_tok` (10), SwiGLU, softmax top-k; `sm12x` selects the GeForce plugin instead
  [code] `modeling_qwen4_exp_sparse_moe.py:86-101`. On Thor that is `Nvfp4MoePlugin` and the shipped
  n128 objects. **The ready shape config is "the shipped pack at run-time shapes"; there is nothing to re-export.**
- The MTP draft's experts take a different path (FP16 MoE or W4A16) [code] same file `:65-69,104-156`.
- The shared expert and its gate are separate dense layers added afterwards [code] same file `:158-161`.

Three properties make Edge-LLM a tolerance-level cross-check, not a bit-level reference for vLLM:

1. **Weights are requantized.** The packer dequantizes ModelOpt NVFP4 to fp32, rounds to BF16 and quantizes
   again with fresh block scales so that alphas are 1.0: "Requantization absorbs those per-projection scales
   into the plugin block scales and emits unit alphas" [code]
   `E/tensorrt_edgellm/builder/weight_packing/nvfp4.py:336-347,388-409`; run-time equivalent
   `assembleNormalizedNvfp4Weights` at `E/cpp/runtime/weight/checkpointWeightAssemble.cpp:1541-1559`;
   bindings `fc1_alpha`, `fc2_alpha` = fill 1.0 at `E/tensorrt_edgellm/builder/models/qwen4_exp/weights.py:116-117,130-131`.
   This is in general not lossless [derived]. vLLM keeps the checkpoint codes and scales and passes
   `weight_scale_2` as alpha. **Do not copy Edge-LLM's packing.**
2. **Activation global scales are 1.0** for this model (`input_global_scale`, `down_input_scale` = fill 1.0)
   [code] `weights.py:132-135`, although the FP4-Local checkpoint has `input_scale` for all 73,728 expert
   projections [code] FP4-Local `model.safetensors.index.json`. Other Edge-LLM models bind the calibrated scales
   (`nvfp4_moe_input_scale_binding`) [code] `E/tensorrt_edgellm/builder/core/weights.py:441-473`.
3. **No CSF decoder** in Edge-LLM 0.12 (no match for "csf" in `tensorrt_edgellm/`, `cpp/`, `tests/`, `docs/`)
   [code]. It can read only the standard NVFP4 checkpoint.

Activation-scale semantics differ in three places:

| Variant | Who | Activation global scale |
|---|---|---|
| (A) one shared scale = max over experts, quantize once per token | vLLM FlashInfer paths, i.e. R7b today | [code] `V/.../quant_utils.py:77-96`, `flashinfer_fp4_moe.py:135-141` |
| (B) per-expert scale, quantize once per routed row | Edge-LLM generic path; supported natively by `K1` | [code] `fp4Quantize.h:48-54`, `K1:2720` |
| (C) scale 1.0 | Edge-LLM for qwen4_exp | [code] `weights.py:132-135` |

The global scale only changes how the FP8 block scales round (and where they saturate or go subnormal)
[derived]. I do not know which variant QAD training used [unknown]. Staying on (A) means no semantic change
against R7b. (B) at prefill quantizes 10x more rows (74,880 instead of 7,488 per layer) [derived].

---

## 3. What vLLM 0.33 and FlashInfer 0.7.2.post1 already have

### 3.1 vLLM

| Piece of "glue" in the framing | Already in vLLM 0.33 | Source |
|---|---|---|
| Experts class | `FlashInferCuteDSLExperts` (Standard activation format, no-op finalize, W4A4 only) | [code] `V/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py:34-246` |
| Device gate | `p.is_device_capability_family(100)` | [code] same file `:88-94` |
| Quant scheme | `(kNvfp4Static, kNvfp4Dynamic)` and per-token variant; no W4A16 | [code] same file `:101-109` |
| Weight repack | `[w1,w3]` to `[up; gate]`, pad to alignment, 64-row interleave, swizzle | [code] `V/.../quantization/utils/flashinfer_fp4_moe.py:109-190` |
| Scale layout | `nvfp4_swizzled_scale_to_cutedsl_mma_view` (a view, no copy) | [code] same file `:193-215`; `V/.../oracle/nvfp4.py:579-581` |
| Alpha folding | `w13_weight_scale_2 *= w13_input_scale`, same for w2 | [code] `flashinfer_cutedsl_moe.py:79-81` |
| Input NVFP4 quant | modular prepare step (`scaled_fp4_quant`, linear scales for this backend) | [code] `V/.../fused_moe/utils.py:318-323,464-491`; `oracle/nvfp4.py:597-603` |
| Oracle order | TRTLLM, CUTEDSL, CUTEDSL_BATCHED, FLASHINFER_CUTLASS, VLLM_CUTLASS, MARLIN, ... | [code] `oracle/nvfp4.py:203-212` |
| FlashInfer CUTLASS gate (our current overlay target) | "SM110 excluded: flashinfer-ai/flashinfer#3134" | [code] `V/.../experts/flashinfer_cutlass_moe.py:128-139` |

With the CuTe DSL gate opened, auto selection would give `FLASHINFER_CUTEDSL` to the W4A4 target experts
(TRTLLM stays unsupported on 110: `trtllm_nvfp4_moe.py:204-211`) and fall through to `MARLIN` for the
W4A16 MTP experts, as it does today with the CUTLASS overlay (`DOC:237-238`) [derived].

### 3.2 FlashInfer 0.7.2.post1 [web]

- `flashinfer/fused_moe/cute_dsl/blackwell/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py`:
  class `BlockScaledContiguousGatherGroupedGemmKernel`, NVIDIA BSD-3 header, warps 0-3 epilogue, 4-7 LDGSTS,
  8 MMA, 9 TMA, 10 scheduler, 11 sync-transform, `get_smem_capacity_in_bytes("sm_100")`. Same as `K1`
  ([code] `K1:1-2,124-136,342-351,408`).
- Default tactic: tile 128, GEMM1 `((128,128),(1,1))`, GEMM2 `((128,128),(1,1))`. Other tactics: N 256,
  GEMM2 cluster `(1,2)`, tile 256 with 2 CTAs (`tuner.py`).
- Routing by a JIT C++ module (`moe_sort`), no `.item()`; pre-allocated buffers recommended for graphs (`moe_utils.py`).
- `use_fused_finalize=True` (default): atomic scatter into a zeroed output. `False`: deterministic two-stage
  (write `num_tokens * top_k` rows, then `moe_unpermute`).
- Output dtype BF16 (default) or FP16. Modes `w4a4`, `w4a8`, `w4a16`.
- Gates found: `@supported_compute_capability([100, 103, 107])` on the API functions (`fused_moe.py`);
  `if major != 10: raise ... "requires SM10x family"` in the FC1 wrapper file;
  `require_cute_dsl_arch(device, native_only=True)`, which passes if the installed DSL's `Arch` enum has
  `sm_110a` (`availability.py`). The fetch tool showed these files only in part, so there may be more gates.

### 3.3 Differences between the two copies

| | Edge-LLM `nvfp4_moe_cutedsl` | FlashInfer 0.7.2.post1 `cute_dsl` |
|---|---|---|
| A row index | mapping tensor used directly (routed rows) [code] `K1:1714-1717` | `mapping // topk` (one row per token) [web] |
| Activation global scale | per-expert tensors [code] `K1:2720-2725` | scalar `norm_const`, optional per-token scale [web] |
| Output clear | fused into FC1 [code] `K1:1493-1519` | separate memset [web] |
| Setup launches per layer | 2 (1 at one token) [code] `Sm110Runner.cpp:269-280` | quant + sort + memset = 3 [derived] |
| SwiGLU clamp / alpha / beta, SiTU | no | yes [web] |
| Deterministic finalize option | no (atomic block-reduce only) [code] `K2:2218-2243` | yes [web] |
| BF16 output | exporter option, not shipped | default |
| sm_110 | shipped and tested by NVIDIA | gated off |

---

## 4. Where I disagree with the framing

1. **"Re-export for 512 experts / top-10"**: not needed (1.1). With in-process JIT there is no export at all.
2. **"Remaining glue: routing metadata, input quant, weight and scale repack, experts class"**: all four exist
   in vLLM 0.33 + FlashInfer for the same kernel family (3.1, 3.2). The missing part is enablement on sm_110,
   validation and tuning.
3. **"CuTe DSL version mismatch"**: both sides pin 4.8.0 (1.8). It remains to confirm the version inside our
   image digest.
4. **"FC2 FP16 output vs model BF16"**: a BF16 variant exists in both copies (1.3).
5. **"Edge-LLM's plugin shows launch order, metadata and tensor contract"**: true, but its weight packing and
   its activation scales must not be copied (section 2).
6. **"D2: decode at Marlin level, prefill ~3.0 s"** (`DOC:225`): the decode part is bounded by about 3.4 ms per
   step in the profile and 1.3 ms in the unprofiled step (5.1). The prefill estimate looks right or slightly
   conservative.
7. **Licence**: the two kernel files are BSD-3-Clause, not Apache-2.0 (1.9).
8. **Decode-specialised sm_110 kernel**: I recommend not writing one (5.4).

Confirmed parts of the framing: FC1 = gather + block-scaled grouped GEMM + SwiGLU + NVFP4 requant; FC2 =
grouped GEMM + alpha + router weight + scatter-reduce to `[T, H]` FP16 [code] README `:14-21`; the scale
layout; the metadata set; the need for static shapes and device-side metadata; the MTP layer staying on Marlin.

---

## 5. Performance analysis

### 5.1 Decode is bound by memory bandwidth [derived]

- One expert = 1,638,400 + 204,800 + 819,200 + 102,400 B = 2.76 MB (FC1 weights, FC1 scales, FC2 weights,
  FC2 scales; shapes from 1.2).
- Per step: n distinct experts x 2.76 MB x 48 layers. n = 28: 3.72 GB, 15.2 ms at 245 GB/s. n = 32: 4.25 GB, 17.3 ms.
- Measured routed-expert GEMM: 15.3 ms (FlashInfer CUTLASS), 15.1 ms (Marlin) `DOC:263-265`, task brief.
- So the GEMM runs at 88 to 100 % of the byte limit. A different kernel reads the same bytes. **No GEMM gain
  at decode should be expected from any tcgen05 kernel.**
- What can shrink: "MoE routing/prep/finalize 4.3 ms" per step (`DOC:263-264`). Marlin needs 0.9 ms for its share.
  That bucket also contains the router and top-k, which do not change. Upper bound about 3.4 ms.
- The unprofiled step shows a smaller difference: target forward 44.6 ms with FlashInfer CUTLASS against 43.3 ms
  with Marlin, i.e. 1.3 ms (`DOC:249`).
- Estimate: 1.3 to 3.5 ms per step, centre 2.5 ms, of 51.7 ms (2.5 to 7 %). Confidence medium-low.

### 5.2 Prefill, 7,488-token chunk [derived]

Today (`DOC:258-260`): MoE GEMM 462 ms + prep/finalize/routing 382 ms = 844 ms of 2.52 s GPU.

Byte floor of the fused pipeline at 245 GB/s, 48 layers:

| Traffic | Per layer | 48 layers | Time |
|---|---|---|---|
| All 512 experts' weights and scales | 1.42 GB | 67.9 GB | 277 ms |
| FC2 scatter-reduce into `[T, H]` 16-bit output, 10 contributions per token, read+write | 0.77 GB | 36.8 GB | 150 ms |
| FC1 gather of FP4 input (9.6 MB x 10) | 0.10 GB | 4.6 GB | 19 ms |
| FC1 output FP4 + scales, written then read | 0.07 GB | 3.4 GB | 14 ms |
| Sum | | | about 460 ms |

Plus input quantization and sort. Estimate for the fused path: 500 to 620 ms, i.e. 220 to 340 ms less per chunk
(9 to 13 % of the chunk's GPU time). Warm 7.6K prefill 3.2 s to about 2.85 to 3.0 s. Confidence medium-low: the
scatter-reduce cost on Thor is the least certain term (it may be cheaper if rows stay in cache, or dearer if
atomics contend).

Variant (B) of section 2 would add about 0.5 GB of quantization traffic per layer (10x the rows) and should be
avoided at prefill [derived].

### 5.3 End to end

Cline-type task: prefill about 16 % and decode about 84 % of request time (`DOC:186,334`).
0.84 x 5 % + 0.16 x 10 % = about 5.8 %. Range +2 to +9 %. Decode 50.2 to about 52.8 tok/s, prefill 1,960 to about
2,200 tok/s [derived].

### 5.4 Tile and variant choices

- **n128 vs n256**: both valid for 1280 and 2560 (1.2). n256 halves the number of tile work units. NVIDIA ships
  n128 as production and keeps n256 "for follow-up benchmarking" (README `:58-59`). Measure both in the spike.
- **M tile 256 / 2 CTAs**: FlashInfer has the tactic; Edge-LLM does not ship it for sm_110 (README `:197-203,220`).
  At 146 rows per expert on average (74,880 rows / 512), both tile sizes pad to 256 rows [derived]. Thor has
  failed with a cluster kernel before ("cluster misconfiguration", `DOC:21`). Treat as optional, after n128/n256.
- **Clusters other than (1,1)**: FlashInfer's GEMM2 `(1,2)` tactic exists; same caution.
- **Decode-specialised kernel**: the sm_120 group has a fused decode kernel because sm_120 has no tcgen05 path
  of this kind. On sm_110 a decode-specialised GEMM would read the same bytes (5.1). What it could save is one
  launch per layer and the FC1-to-FC2 round trip through memory, about 0.25 to 0.5 ms per step [speculation].
  That does not justify a new 2,500-line kernel. The only decode-specific item worth considering is a fused
  "sort + quantize" setup kernel for small token counts (Edge-LLM has it for one token only), and only if the
  spike shows the setup kernels cost 1 ms per step or more.
- **Launch overhead**: a FlashInfer reviewer quotes 20 to 50 us per CuTe DSL launch outside CUDA graphs
  [web, flashinfer#7827]. All decode measurements must therefore be taken under graph replay.

### 5.5 Comparison targets

| Path | Decode per step (GEMM + prep) | 7,488-token prefill (GEMM + prep) | Source |
|---|---|---|---|
| FlashInfer CUTLASS W4A4 (R7b) | 15.3 + 4.3 ms | 462 + 382 ms | `DOC:258-265` |
| Marlin W4A16 | 15.1 + 0.9 ms | 1,501 + 155 ms | task brief, `DOC:215-217` |
| vLLM CUTLASS (old) | about 18.4 + 4.5 ms at MTP 4 | 499 + 880 ms | `DOC:86-90,215-217` |
| CuTe DSL fused (estimate) | about 15 + 1.5 to 3 ms | 500 to 620 ms total | 5.1, 5.2 |

---

## 6. Integration architecture

### 6.1 Route A2 (main): open FlashInfer CuTe DSL MoE on sm_110

Overlay `serving/docker/patches/flashinfer-cutedsl-moe-sm110/`, same pattern as the existing overlays
(full copies of the image's files with marked hunks, mounted read-only; `flashinfer-cutlass-moe-sm110/` is the model):

| File (in the image) | Change |
|---|---|
| `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py` | `_supports_current_device`: accept family 110 when `VLLM_FLASHINFER_CUTEDSL_SM110=1`. With the switch off the file behaves as stock, so R7b stays reproducible. Optional second switch to pass `use_fused_finalize=False` for the determinism A/B. |
| `flashinfer/fused_moe/cute_dsl/fused_moe.py` | add 110 to the three `supported_compute_capability` lists |
| `flashinfer/fused_moe/cute_dsl/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py` and the finalize wrapper | accept `major == 11` where `major != 10` is rejected |
| any further gate found in Stage 0a (for example the `moe_utils` JIT module's architecture list) | same kind of change |

The exact lines must be taken from the files inside the image; I saw the FlashInfer files only through the
fetch tool. The existing CUTLASS overlay stays mounted, so that with the switch off the oracle still falls
back to `FLASHINFER_CUTLASS`.

Properties:
- Interfaces: unchanged. Modular kernel experts class, existing prepare/finalize, existing oracle, existing
  weight hooks (3.1).
- CUDA graphs: static shapes per captured token count; metadata on the device; the JIT happens in the eager
  warm-up runs before capture (`cudagraph_num_of_warmups: 5`). Prefill chunks are not captured.
- MTP layer: stays on Marlin through auto selection. Do not pass `--moe-backend`: forcing a backend also
  applies to the MTP layer and fails at load (`DOC:241-243`).
- Shared expert: MXFP8 dense MLP with a sigmoid gate, separate side-layer path, untouched.
- Semantics: activation scale variant (A), same as R7b. Differences against R7b are numerical only
  (approximate reciprocals, summation order, BF16 atomic accumulation).
- No second weight copy; the repack is in place.

### 6.2 Route D2-port (fallback): Edge-LLM kernels JIT-compiled in vLLM

Only if A2 fails in the spike for FlashInfer-specific reasons while the Edge-LLM sources compile and run.

- Kernels: vendor `K1`, `K2`, `custom_pipeline.py`, `cute_utils.py`, `moe_compat.py` unchanged (about 7,600
  lines, BSD-3 / Apache headers kept) under `vllm/model_executor/kernels/moe/cute_dsl_nvfp4_sm110/` in the overlay.
- Compile: `cute.compile(kernel.wrapper, <pointers>, <run-time ints>, options="--enable-tvm-ffi")`, as in
  `export_fc1_kernel.py:84-164` minus `export_to_c`, with vLLM's helper (`jit_warmup_cutedsl_helper.py:33-46`).
  FC2 compiled with BF16 output. Variants: SwiGLU FC1 n128 (and n256 if it wins), FC2 n128/n256.
- Experts class: keep the `FLASHINFER_CUTEDSL` backend slot so that the existing weight prep and quant config
  are reused; in the overlaid `flashinfer_cutedsl_moe.py`, on sm_110, `apply` calls the vendored kernels
  instead of `flashinfer_cute_dsl_fused_moe_nvfp4`. This keeps the overlay to one vLLM file plus new files.
  For an upstream PR it would become its own backend.
- Activation quantization: variant (A). Quantize once per token with the existing prepare step; give FC1 a
  second int32 map "permuted row to token row" (`permuted_to_expanded // top_k`, -1 kept) because `K1` indexes
  A directly; give FC2 the original `permuted_to_expanded` map.
- Routing metadata: reuse FlashInfer's `moe_sort` if it loads on sm_110, else a single-block kernel ported from
  `buildLayout.cu:41-116` (Triton or CuTe DSL). No torch `argsort`/`bincount` chains (many launches).
- Scales: FC1 `alpha` = folded `g1_alphas`, `input_global_scale` = ones; FC1 `down_input_scale` = `w2_input_scale`
  (shared value, forward direction); FC2 `alpha` = folded `g2_alphas` with `down_input_scale` = ones.
  These identities must be checked against the torch reference in Stage 1 before any speed work.
- Buffers: one persistent set owned by the class and shared by all 48 layers (they run one after another),
  sized for the largest token count seen, grown only outside graph capture.
- Launch chain: FC1 with `fuse_output_zero=1` and PDL, then FC2 with PDL.

### 6.3 Overlay first, upstream later

A2 is already in upstream shape: the vLLM change is a gate, the FlashInfer change is a capability list.
D2-port is not: vLLM maintainers have said in-tree CuTe DSL kernels may be removed for lack of maintenance
[web, vLLM #65050 review], so vendored kernels belong in FlashInfer, not in a vLLM PR.

---

## 7. Correctness strategy

References, in order of authority:
1. **Torch reference**, written for this work: dequantize checkpoint weights (codes x block scale x
   `weight_scale_2`), emulate W4A4 activation quantization of variant (A) in fp32, SwiGLU, FC2, router weights,
   fp32 sum. One layer at a time. This is the ground truth for "what the checkpoint means".
2. **FlashInfer CUTLASS path** (R7b): the path we serve today. Its deviation from reference 1 sets the tolerance.
3. **Edge-LLM plugin**: tolerance-level only and only for the standard checkpoint (section 2). Use it only
   if it is already runnable on Thor; do not build it for this purpose.

Inputs: real weights of single layers (for example 0, 23, 47) from both checkpoints, loaded from safetensors
by an offline harness (FP4-Local through the CSF decoder of PR #64312; 1.42 GB per layer), and real
activations captured once from a served run (`hidden_states`, `topk_ids`, `topk_weights` at a 4-token verify
step and at one prefill chunk).

Tests:

| Test | Pass criterion |
|---|---|
| Scale-layout byte compare (vLLM view vs Edge-LLM `swizzle_nvfp4_mma_scales`, `nvfp4.py:269-282`) on one expert | identical bytes |
| Layer output vs reference 1, decode and prefill captures, both checkpoints | per-token cosine and relative L2 no worse than 1.5x the CUTLASS path's own deviation; calibrate once, then freeze the numbers |
| Empty experts (decode: about 480 of 512 unused) | pass, no write outside `[T, H]` (guard bands as in `nvfp4MoeCuteDslSm110Tests.cu:971-978`) |
| Single token (M = 1) | pass |
| Group sizes 127, 128, 129 rows for one expert; all tokens to the same 10 experts | pass |
| Largest chunk: 8,192 tokens (81,920 routed rows, `permuted_m` 146,944) | pass, buffers large enough |
| Router ties | ties are resolved before the kernel; feed identical `topk_ids` to both paths. Add one synthetic case with equal router weights within a token |
| Zero router weight for a token | output row is zero |
| Determinism: 20 graph replays, same input | report bit-equality at M = 4 and at 7,488 tokens. Atomic accumulation order across CTAs is not fixed by the code [derived]; NVIDIA's own test is bit-equal only at T <= 8 |
| Output range | max absolute MoE output on the captures; only relevant if an FP16 variant is ever used |
| CUDA-graph capture at each captured token count | no capture error, no host sync in the MoE region of a profile |
| MTP layer | still `MARLIN` in the start-up log |
| End to end, greedy bench prompts | position of first token divergence against R7b and logprob drift reported; acceptance within run-to-run noise |
| Quality gate | Tool Eval, as for every recipe change in `DOC` |

If the output is not bit-stable and that is not acceptable, measure `use_fused_finalize=False`
(deterministic, slower) before deciding.

---

## 8. Staged workflow with gates

Effort is agent working time; Thor sessions are the bottleneck and need Thor free.

### Stage 0a: facts without GPU (about 1 h)

1. From the image (file reads only): installed versions of `nvidia-cutlass-dsl`, `flashinfer-python`;
   whether `Arch.sm_110a` exists and is in `BlockScaledMmaOp.admissible_archs`; every sm_110 gate in
   `flashinfer/fused_moe/cute_dsl/**` and in the `moe_utils` JIT spec, with file:line; whether FlashInfer keeps
   compiled CuTe DSL kernels across restarts and where.
2. Split the existing R7b traces by kernel name: the 4.3 ms decode bucket and the 382 ms prefill bucket
   (router/top-k, input quant, sort/permute, activation, finalize).
3. Standard NVFP4 checkpoint index: are `input_scale` tensors present, same shapes?

Gate G0a: stop if the image's DSL is not 4.8.x and has no `sm_110a` (then the old `admissible_archs` patch
would be needed first), or if the removable part of the 4.3 ms bucket is under 1.5 ms and the prefill prep
is mostly unavoidable work.

### Stage 0b: half-day spike on Thor (about 3 h agent, 1 to 2 h of Thor, no server)

One `docker run --rm` of the stock image with a script directory mounted. No vLLM integration, no file
changes in the image; gates are lifted in the script's own process.

Candidates on identical tensors (synthetic FP4 weights with Flash-Next shapes: E = 512, top-10, H = 2560,
I = 640; all 512 experts of one layer so that weights do not fit in cache):
- (i) `flashinfer.cutlass_fused_moe`, the R7b baseline;
- (ii) FlashInfer CuTe DSL MoE: default tactic, n256 tactic, fused finalize on and off;
- (iii) Edge-LLM `K1`/`K2` JIT-compiled through `kernel.wrapper`, only if (ii) does not compile or run;
- optional extra row: FlashInfer CuTe DSL `quant_mode="w4a16"` [web], to know whether a tcgen05 W4A16 path
  exists for an R5-style configuration. Information only.

Token counts 1, 4, 5, 8, 16, 64, 512, 2,048, 7,488, 8,192; routing uniform and skewed. Timing under CUDA-graph
replay for counts up to 512 (200 replays), eager for the larger ones; one profiler pass for the split into
setup, FC1, FC2. Numerics against the torch reference. Record cold JIT time.

Gate G0 (all must hold to continue):
- compiles and runs on sm_110 with Python-level changes only, across all token counts, without a crash;
- numerics within tolerance of the reference;
- decode: routed-MoE time per layer at 4 tokens at least 10 % below (i) in the same harness (about 2 ms per
  step) and prefill not slower, **or** prefill at 7,488 tokens at least 20 % below (i) (about 170 ms per chunk)
  and decode not slower;
- cold JIT time acceptable for a server start (proposed limit: 5 minutes in total).

If G0 fails for (ii) and passes for (iii): go to Stage 1b. If it fails for both: stop (section 10).

### Stage 1: A2 overlay, boot, correctness (about 3 to 4 h agent, one Thor session)

Build the overlay of 6.1, boot both checkpoints, run the tests of section 7, then the standard measurements:
`VLLM_SPEC_STEP_TIMING=200` windows, torch profiler for one decode run and one 7,488-token prefill chunk,
`bench.sh`, warm 7.6K prefill.

Gate G1: both checkpoints boot; the log shows `FLASHINFER_CUTEDSL` for target experts and `MARLIN` for MTP;
section 7 passes; step GPU at least 1.5 ms below R7b in 200-step windows **or** warm 7.6K prefill at least
0.2 s below R7b, with neither metric worse; no graph capture failure.

### Stage 1b (fallback, only after a G0 result that asks for it): D2-port (about 1 to 1.5 days agent, two Thor sessions)

Implement 6.2. Same tests and the same gate G1. This is the "largest item" of `DOC:225`; it needs the
maintainer's approval as a new mechanism before it starts.

### Stage 2: tuning (about 2 to 4 h agent, one Thor session)

- n128 vs n256 per regime (decode, prefill); optionally cluster `(1,2)` for FC2 and the 256-row tile.
- Pin the chosen tactic. vLLM does not expose the tactic of this backend; the choices are FlashInfer's
  autotune cache or a constant in the overlay. We run with `--disable-flashinfer-autotune` today, so a
  constant is simpler.
- PDL on/off A/B.
- Setup kernels: only if they cost 1 ms per step or more, fuse sort + quantize for small token counts.

Gate G2: final numbers at least as good as G1; Tool Eval not worse than R7b; recipe written to `DOC` as R8.

### Stage 3: upstream (about 2 to 3 h agent to prepare; calendar time is review)

See section 9.

---

## 9. Path to upstream adoption

Context [web]:
- **flashinfer#3134** ("Unsupported tile ... for arch 100", Thor): the CUTLASS MoE on sm_110 loaded the sm_100
  artifact and hit unregistered tile/cluster shapes. Closed as not planned on 2026-10-25; a fork has commits
  enabling FlashInfer CUTLASS NVFP4 MoE on Thor.
- **vLLM #40808** (merged 2026-05-01): removed sm_110 from `FlashInferExperts`, "once SM110 MoE cubins ship"
  the workaround should go. Our CUTLASS overlay reverses this line.
- **vLLM #65050** (merged 2026-11-03): enables the CuTe DSL GDN prefill on sm_110. Accepted with Thor test logs
  (13 tests), a GSM8K comparison and TTFT numbers; no sm_110 CI runner exists. A maintainer asked why not
  FlashInfer and noted in-tree CuTe DSL may be removed.
- **flashinfer#7827** (open): GDN prefill on sm_110 in FlashInfer. Reviewers asked for CUDA-graph based
  measurements and a clear benchmark protocol, and for a CPU test that pins sm_110 thresholds without hardware.
- **vLLM #64312** (merged 2026-11-03): CSF decode at load; kernels see standard E4M3 scales.

PR shape for route A2:
1. **FlashInfer PR**: add sm_110 to the CuTe DSL NVFP4 MoE (W4A4): capability lists, the `major` checks,
   tactics restricted to what was measured on Thor (1 CTA, cluster 1x1; n128/n256). Evidence: Thor logs of
   the MoE test files, graph-replay benchmarks at decode and prefill shapes against the CUTLASS path, and the
   fact that Edge-LLM 0.12 ships the same kernels for sm_110 with the same tile. Add a CPU-only test for any
   sm_110 tactic rule, as #7827 was asked to.
2. **vLLM PR**: `[Kernel] Enable FlashInfer CuTe DSL NVFP4 MoE on SM110`: the gate in
   `FlashInferCuteDSLExperts._supports_current_device`, conditional on a FlashInfer that supports it (a small
   `has_...` probe in `vllm/utils/flashinfer.py`), plus the kernel test parameterised for the device. State
   that no sm_110 CI exists and attach Thor results (model: #65050). Mention that the oracle then prefers
   CuTe DSL over CUTLASS on sm_110, which also makes the #40808 exclusion harmless for NVFP4.
3. **Optional FlashInfer follow-ups** from Edge-LLM's copy, each with numbers: output clear fused into FC1;
   fused sort + quantize for small token counts; per-expert activation scales.

If the fallback D2-port was needed: propose the sm_110 differences to FlashInfer first. A vLLM-tree home would
only have the model-specific precedent (`vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/moe/`,
`vllm/models/qwen4_exp/nvidia/ops/cute_dsl/`) and the maintenance objection above.

---

## 10. Risks, mitigations, and when not to do this

| Risk | Mitigation |
|---|---|
| Decode gain is small because the GEMM is bandwidth-bound | Spike first; gates G0/G1 are in ms, not in hopes |
| DSL 4.8.0 rejects some op for native sm_110a | Stage 0a check; `CUTE_DSL_ARCH`; the old `admissible_archs` overlay as last resort |
| More FlashInfer gates than the three found, or the `moe_utils` C++ JIT does not build for CUDA major 11 | Stage 0a inventory; fallback to a ported layout kernel; Stage 1b |
| First-time JIT is long (36 minutes were seen in May for a FlashInfer C++ module, `V9.1-TASK4-FP4-UNLOCK.md:36`) | measure in the spike; persistent FlashInfer cache directory as today |
| Atomic BF16 accumulation: outputs not bit-stable, slightly more rounding noise than an fp32 sum | determinism test; `use_fused_finalize=False` A/B; Tool Eval |
| Cluster or 2-CTA tactics fail on Thor (20 SMs, cluster limit 8) | stay on cluster 1x1, the only shape NVIDIA ships for sm_110 |
| PDL inside captured graphs misbehaves on sm_110 | A/B with PDL off; NVIDIA's test covers capture with PDL |
| Numerics differ from R7b (approximate reciprocals, fast-math SiLU) | tolerance tests against the torch reference; Tool Eval |
| Overlay drift | full-file copies pinned to the image digest and to FlashInfer 0.7.2.post1 |
| Copying Edge-LLM's packing or scale-1.0 choice by accident | section 2; reference 1 is built from checkpoint values |
| BSD-3 headers dropped in a port | keep headers; note in the PR |

**I would recommend not doing this (beyond the spike) if:**
1. the spike shows less than about 2 ms per decode step and less than about 170 ms per 7.5K prefill chunk;
2. the only working route is the D2-port and the measured gain is at the low end. One to one and a half days
   of agent work plus a maintained 7,600-line fork for about 3 % is a poor trade against the open items in
   `DOC:385-398` (adaptive draft length and prompt-lookup drafts; hyperconnections to FP8, estimated -2.6 ms);
3. the output cannot be made stable enough for the maintainer's output-safety requirement without the
   deterministic finalize, and that finalize removes the gain;
4. enabling it requires changing compiled artifacts in the image rather than Python files;
5. Tool Eval drops against R7b.

Note on route D1 (`DOC:224`, fuse the CUTLASS finalize in Triton): A2 removes the same finalize work and more.
D1 remains the cheap alternative if both CuTe DSL candidates fail the spike.

---

## 11. Assumptions and unknowns

| # | Unknown | Resolved by |
|---|---|---|
| U1 | The image digest really has `nvidia-cutlass-dsl 4.8.0` and FlashInfer 0.7.2.post1 (requirements say so; a local note says 0.7.2.post1) | Stage 0a |
| U2 | DSL 4.8.0 admits `sm_110a` for the block-scaled tcgen05 MMA when compiling natively (Edge-LLM proves it for cross-compiled AOT) | Stage 0a / 0b |
| U3 | Complete list of sm_110 gates in FlashInfer's CuTe DSL MoE (files seen only in part) | Stage 0a |
| U4 | Standard NVFP4 checkpoint: `input_scale` present? (the one shard header I have shows weights and weight scales only) | Stage 0a |
| U5 | Composition of the 4.3 ms and 382 ms buckets | Stage 0a |
| U6 | Distinct experts per layer at 4 tokens (24 to 28 assumed earlier, 31 to 33 in the brief) | existing traces or the spike |
| U7 | Which activation-scale variant QAD training used | not needed for A2 (no change against R7b); a torch-only comparison of (A), (B), (C) on captured activations if the maintainer wants it |
| U8 | CuTe DSL JIT time on Thor and whether it is cached across restarts | Stage 0b |
| U9 | All MoE kernels lie inside PIECEWISE graph segments | Stage 1 profile |
| U10 | Whether the atomic scatter is bit-stable in practice at 4 tokens and at 7.5K | Stage 1 |
| U11 | Scatter-reduce cost on Thor at prefill (largest term of the estimate after the weights) | Stage 0b |
| U12 | A Python loader for exported CuTe DSL objects in 4.8.0 (only relevant if JIT time is a problem) | not planned |

Assumptions: 245 GB/s as the usable DRAM rate (earlier report: 190 to 257 measured); the R7b profile is
representative; FlashInfer's default tactic applies on sm_110 once the gates are lifted; the web summaries of
upstream files and PRs are accurate (they came through a summarising fetch tool and were not read line by line).

---

## 12. Sources

Local: paths under `E/`, `V/` and `DOC` as cited; the earlier Fable decode analysis summarised in `DOC`;
FP4-Local `config.json`, `model.safetensors.index.json` and a safetensors header;
`serving/docker/patches/*`; `serving/docs/V9.1-TASK4-FP4-UNLOCK.md`;
`serving/docker/mods/sm110a-fp4-dsl-unlock/run.sh`.

Web (fetched 2026-11-14):
- https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/requirements/cuda.txt
- https://github.com/flashinfer-ai/flashinfer/issues/3134
- https://github.com/vllm-project/vllm/pull/40808
- https://github.com/vllm-project/vllm/pull/65050
- https://github.com/vllm-project/vllm/pull/64312
- https://github.com/flashinfer-ai/flashinfer/pull/7827
- https://github.com/flashinfer-ai/flashinfer/tree/v0.7.2.post1/flashinfer/fused_moe/cute_dsl (and `fused_moe.py`, `tuner.py`, `moe_utils.py`, `blackwell/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py`, the wrapper of the same name one level up)
- https://raw.githubusercontent.com/flashinfer-ai/flashinfer/v0.7.2.post1/flashinfer/cute_dsl/availability.py and `utils.py`
