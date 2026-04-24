# MiniMax-M2.7 REAP-139B-A10B-NVFP4 on Jetson Thor — investigation notes

**Date**: 2026-04-22
**Verdict on W4A4-GB10 variant**: not viable on Thor (SM110a). Profile kept in the source tree for reference, checkpoint removed from disk.
**Next attempt**: the W4A16 sibling (`dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B-NVFP4`, no `-GB10` suffix) — scheduled for tonight. W4A16 uses a different GEMM kernel path and does not carry the split per-half scale mismatch, so both the MARLIN fallback and the 45%-attenuation damage above should be avoided. The config.json ignore-list patch for its 62 MoE router gates is already staged at [scripts/patch-minimax-w4a16-config.sh](scripts/patch-minimax-w4a16-config.sh).

## What we tried

Target: `dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B-NVFP4-GB10`
(75 GB, 62 all-attention layers, head_dim=128, 154 experts × 10B active).

Profile added in [lib/config.sh](lib/config.sh) + [lib/launch.sh](lib/launch.sh) as
`minimax-m2.7-139b-a10b-nvfp4`. Runtime mod created:
[docker/mods/fix-nvfp4-moe-scale-merge/run.sh](docker/mods/fix-nvfp4-moe-scale-merge/run.sh)
(replaces `w13_weight_global_scale[:, 0]` with `torch.minimum(w1, w3)`).

Tested matrix of attention and MoE backends until one produced readable output.

## Findings

### 1. NVFP4 MoE backends on SM110 — all but one are unusable

| Backend | Result | Why |
|---------|--------|-----|
| `FLASHINFER_TRTLLM` | Gated off | vLLM's check: *"Flashinfer TRTLLM MOE backend is only supported on SM100 and later"* — SM110 not whitelisted. |
| `FLASHINFER_CUTEDSL` / `_BATCHED` | Gated off | Raises at init: *"kernel does not support current device cuda"*. |
| `FLASHINFER_CUTLASS` | Loads, **silently emits null bytes** | Autotuner skips 34 of 35 tactics for `trtllm::fused_moe::gemm1` and 68 of 69 for `gemm2`. No valid tile for 154-expert × `[3072, 3072]` shape on SM110 → dispatcher fell through to a path that returns zeros. All 50 decoded tokens = token id 0. |
| `VLLM_CUTLASS` | **Crashes at runtime** | Kernel symbol: `run_fp4_blockwise_scaled_group_mm_sm100` — hardcoded for SM100. Fails with `status=7 num_experts=154 M=8 N=3072 K=3072`. |
| `MARLIN` | **Runs**, produces readable output | Not SM-gated — dequantizes FP4 → BF16 in registers and uses a standard BF16 tensor-core GEMM. Works on SM70+. |

The working path was `VLLM_TEST_FORCE_FP8_MARLIN=1` + `VLLM_USE_FLASHINFER_MOE_FP4=0`.

### 2. TurboQuant K8V4 KV cache is incompatible with FlashInfer attention

TQ requires FlashAttention 2; FlashInfer backend rejects `kv_cache_dtype=turboquant_k8v4`.
Do not set `--attention-backend flashinfer` when using TQ — let vLLM auto-select (it picks
FA2 + `turboquant_attn` for TQ layers).

Also: even though this model is pure-attention (no hybrid layers), the `fix-pr39931-turboquant`
runtime mod is still required when `--kv-cache-dtype turboquant_k8v4` is set. My first-pass
comment "NO need for fix-pr39931-turboquant mod on this profile" was wrong.

### 3. The per-half w1/w3 global scale ratio is not 99.6% matched

The community analysis on sibling checkpoints (lukealonso/MiniMax-M2.5-NVFP4) claimed 99.6%
of expert pairs have w1_gs ≈ w3_gs, so `torch.minimum()` merging gives a sub-1% accuracy
delta. **This is not true for this checkpoint.** Sampling layer 10 experts 0–19:

```
expert   0: w1=2.2144e+04  w3=1.9200e+04  ratio=1.153
expert   2: w1=1.3248e+04  w3=2.1632e+04  ratio=1.633
expert   5: w1=1.2352e+04  w3=1.8944e+04  ratio=1.534
expert  10: w1=1.2928e+04  w3=2.2016e+04  ratio=1.703
expert  17: w1=1.0688e+04  w3=1.9456e+04  ratio=1.820
```

Ratios up to 1.82× → taking `min()` attenuates one half of the SwiGLU by up to 45%. The
resulting model produces legible English but with visible damage.

### 4. Performance — 12 tok/s, expected for MARLIN at 10B-active

| Run | Tokens | Decode time | **TPS** |
|-----|--------|-------------|---------|
| Single request, 256 out | 256 | 21.1 s | 12.14 |
| Single request, 512 out | 512 | 42.5 s | 12.06 |
| Single request, 1024 out | 796 | 65.6 s | 12.13 |

TTFT: 0.3–0.45 s (good — 10B-active MoE).

For context, the theoretical ceiling is `273 GB/s ÷ (10 GB active) = 27 tok/s` with FP4-native
compute — ~54 tok/s if we could treat the 10B active as 5 GB (FP4 @ 0.5 B/param). MARLIN loads
half the bytes but does BF16 matmul, so the only win is memory bandwidth on the load, not the
compute. 12 tok/s is on-profile for MARLIN at this active-param count.

With a working `FLASHINFER_TRTLLM` path we'd expect 20–25 tok/s. That path is not available on
SM110.

### 5. Capability — degraded

Three prompts at max_tokens=400–800:

- **Monty Hall (requested: 2 sentences)** — trails off in reasoning trace, gets probability
  wrong mid-stream, never produces the requested format.
- **Code gen (`longest_palindromic_substring`)** — output contains token-level glitches:
  `whileload>=0`, `s[left]==s[right]` jammed without spaces, `retstype`, misplaced
  backticks. Not runnable.
- **"5 Italian cities sorted by population, no other text"** — rambles, hallucinates
  populations ("Rome about 276 million"), never produces the clean list.

This is consistent with the scale-merge damage: the model still "knows" language but makes
more factual errors and formatting mistakes than a healthy checkpoint of the same size should.

## What would be needed to make this work

1. **A proper FP4-native MoE kernel for SM110** — either upstream waits for `FLASHINFER_TRTLLM`
   to whitelist SM110, or vLLM's `run_fp4_blockwise_scaled_group_mm_sm100` grows an SM110
   variant. Not a local fix.
2. **Per-half global scales in the kernel API** — the fused `[w3, w1]` NVFP4 GEMM accepts one
   scalar per expert. Fixing the 45% attenuation needs the kernel to take shape `[E, 2]` and
   apply w1's scale to the first half of the output, w3's to the second. That's a kernel
   modification, not a Python patch.
3. **Re-quantize the checkpoint** with `max(w1, w3)` (i.e. widen both halves to the larger
   range, losing a little precision on the narrower half but avoiding the 45% attenuation
   entirely). Would need to run llm-compressor's `update_fused_layer_weight_global_scales`
   over the existing NVFP4 tensors. Full day of work, needs a Blackwell or SM100 machine to
   validate — and still leaves the kernel-availability problem unsolved.

## Artifacts kept

- [docker/mods/fix-nvfp4-moe-scale-merge/run.sh](docker/mods/fix-nvfp4-moe-scale-merge/run.sh) —
  reusable for any future NVFP4 W4A4 checkpoint with split per-half scales. Not specific to
  MiniMax.
- [lib/config.sh](lib/config.sh) + [lib/launch.sh](lib/launch.sh) profile block — left in
  place with a header comment pointing to this document. The working launch config (MARLIN
  forced, `VLLM_USE_FLASHINFER_MOE_FP4=0`, fp8 KV, `max_num_seqs=1`, `max_model_len=16384`)
  is preserved so it can be re-activated without re-deriving the right combination.

## Artifacts removed

- `~/thor-hf-cache/hub/models--dervig--m51Lab-MiniMax-M2.7-REAP-139B-A10B-NVFP4-GB10/` (75 GB)
