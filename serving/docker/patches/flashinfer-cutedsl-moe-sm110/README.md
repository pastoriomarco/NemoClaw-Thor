# FlashInfer CuTe DSL NVFP4 MoE on sm_110 (Jetson AGX Thor)

Read-only file overlay for the official `vllm/vllm-openai@sha256:d825baf9…` image (vLLM 0.33.0, FlashInfer
0.7.2.post1, nvidia-cutlass-dsl 4.8.0). It admits sm_110 to FlashInfer's CuTe DSL NVFP4 MoE (fused gather +
grouped GEMM + SwiGLU + requant FC1; grouped GEMM + router weight + scatter-reduce FC2), which vLLM already
integrates as the `FLASHINFER_CUTEDSL` MoE backend. CuTe DSL 4.8.0 generates native `sm_110a` code for these
kernels; only capability checks blocked them. Investigation and measurements:
[`../../../docs/SM110-NVFP4-MOE-PLAN.md`](../../../docs/SM110-NVFP4-MOE-PLAN.md) and
[`../../../docs/QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md`](../../../docs/QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md) (R8b, N1–N3).

Each file is the image's own file with one line changed, marked `[thor]`:

| File | Change |
|---|---|
| `flashinfer/fused_moe/cute_dsl/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py:688` | FC1 wrapper: `major != 10` → `major not in (10, 11)` |
| `flashinfer/fused_moe/cute_dsl/blockscaled_contiguous_grouped_gemm_finalize_fusion.py:513` | FC2 wrapper: same |
| `flashinfer/jit/moe_utils.py:87` | `moe_utils` JIT arch list `[9, 10]` → `[9, 10, 11]` |
| `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py:92` | `_supports_current_device`: also family 110 |

## Use

Mount each file over its path under `/usr/local/lib/python3.12/dist-packages/`, read-only, e.g.

```
-v $O/flashinfer/jit/moe_utils.py:/usr/local/lib/python3.12/dist-packages/flashinfer/jit/moe_utils.py:ro
```

(Thor: `~/fn-bench/round7.sh` adds all four.) No `--moe-backend` flag: vLLM's automatic selection ranks
`FLASHINFER_CUTEDSL` above `FLASHINFER_CUTLASS`, so the W4A4 target experts get it and the W4A16 (FP4-Local) or
block-FP8 (NVIDIA NVFP4) MTP experts keep Marlin / Triton. Forcing `--moe-backend` would also apply to the MTP layer
and fail. The log shows `Using 'FLASHINFER_CUTEDSL' NvFp4 MoE backend`.

`moe_utils` (moe_sort, memset, unpermute) is a C++/CUDA JIT module with no prebuilt sm110a build in
`flashinfer_jit_cache`. With the arch list lifted it builds for sm_110a in ~9 min on 3 CPUs on first use; prebuild it
(`serving/benchmarks/sm110-moe-spike/run.sh prebuild`) and copy `cached_ops/moe_utils/` into the FlashInfer cache the
server mounts (`~/flashinfer-cache/0.7.2.post1/110a/`) to avoid that. The CuTe DSL kernels themselves JIT in ~1 s each.

## Notes

- Output is not bit-identical to FlashInfer CUTLASS: the fused finalize accumulates with BF16 atomics. Tool Eval
  showed no measurable quality cost (R8b 92/100 vs R7b 91/100).
- Measured on FP4-Local (R8a → R8b): decode step 49.3–50.0 → 46.6–47.0 ms, warm 7.6K prefill 3.04 → 2.67 s.
- Both checkpoints' FC2 activation scales are collapsed to one value per layer by vLLM; FP4-Local's are per expert
  in the checkpoint (see the plan, Stage 0a).
- Licences: the FlashInfer files are FlashInfer's (Apache-2.0, with NVIDIA BSD-3 kernel headers inside the package);
  the vLLM file is vLLM's (Apache-2.0). Only the marked lines differ from the image.
