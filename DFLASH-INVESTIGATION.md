# DFlash Speculative Decoding on Jetson AGX Thor (SM110)

**Date**: 2026-04-15 — 2026-04-16
**Target model**: Qwen3.5-9B-FP8 (`lovedheart/Qwen3.5-9B-FP8`)
**Drafter model**: z-lab/Qwen3.5-9B-DFlash (1B params, BF16)
**Platform**: Jetson AGX Thor, SM110 (Blackwell), 128 GB unified memory
**vLLM**: 0.19.1rc1.dev195 (commit e281cb721)
**FlashInfer**: 0.6.7 (commit 904fa8cbc)
**Goal**: ~45% DFlash acceptance rate (z-lab benchmark: 46.52% FP8, 47.24% BF16)

---

## Problems Found (ordered by difficulty)

### Problem 1: 0% DFlash acceptance rate — UNSOLVED

**Symptom**: DFlash launches and serves inference, but draft tokens are always
wrong. Acceptance rate is 0.26% (position 0 ~2%, positions 1-7 = 0%).

**What we know**:
- Draft tokens are coherent English words but don't match target predictions
- Example: for "Count from 1 to 20", drafter predicts `\n, I, 0, ,` instead of `1, ,, 2, ,`
- Both FP8 and BF16 target models show identical 0% acceptance
- z-lab benchmarks show FP8 works (46.52% vs 47.24% BF16 — only 0.7% drop)
- FlashInfer `causal=False` metadata is correctly set (DFlash assertion passes)
- FlashInfer kernel uses `MaskMode.NON_CAUSAL` when `_causal=False`
- Drafter receives correct input_ids, positions, non-NaN hidden states
- z-lab's working config uses `--attention-backend flash_attn` (FA4), not FlashInfer

**Ruled out**:
- Non-causal vs causal: tested with causal=True (metadata-only mod), acceptance
  identical at 0%. The attention masking is NOT the problem.
- FP8 vs BF16 target model: both show 0%.
- Slot mapping values: SLOTS debug captured, slots are internally consistent
  (contiguous, correct block_id math, correct drafter block_table used).

**SLOTS debug output (test #17)**:
```
SLOTS #1: block_size=16, num_context=18, num_query=9,
  cad.seq_lens=[18], new_seq_lens=[27],
  ctx_slots[:8]=[26880, 26881, 26882, 26883, 26884, 26885, 26886, 26887],
  query_slots[:9]=[26898, 26899, 26900, 26901, 26902, 26903, 26904, 26905, 26906],
  ctx_pos[:8]=[0, 1, 2, 3, 4, 5, 6, 7],
  query_pos[:9]=[18, 19, 20, 21, 22, 23, 24, 25, 26]
DRAFT #1: draft_tokens=[161284, 41979, 36272, 158357, 2822, 11930, 2804, 6860]  (CJK garbage)
DRAFT #2: draft_tokens=[321, 381, 25, 310, 310, 310, 328, 11]  (generic English)
```

**Current theory**: FlashInfer's paged KV mechanism is incompatible with DFlash's
`precompute_and_store_context_kv` + forward pattern. The precompute writes K/V
via `do_kv_cache_update(slot_mapping)`, but FlashInfer's `.run()` reads from
different offsets or the K/V tensor layout is incompatible. DRAFT #1 reads
garbage (uninitialized memory), DRAFT #2-3 read stale/wrong context.

This is supported by the fact that z-lab ONLY tested with `flash_attn` backend,
never FlashInfer. The DFlash paged KV write/read pattern may depend on flash_attn-
specific tensor layouts that FlashInfer doesn't match.

**Proposed solutions** (ordered by estimated effort):

| Option | What | Effort | Risk |
|--------|------|--------|------|
| 1. Patch FlashInfer paged KV | Debug exactly why `do_kv_cache_update` + `.run()` disagree on K/V offsets | 4-8 hours | High — requires FlashInfer C++/CUDA internals |
| 2. PyTorch SDPA for drafter | Monkey-patch drafter's 5 attention layers to use `F.scaled_dot_product_attention` with non-paged KV | 1-2 hours | Low — uses proven PyTorch code, drafter is tiny |
| 3. Bypass paged KV entirely | Rewrite precompute + forward to use plain K/V buffers, no KV cache | 2-4 hours | Medium — breaks vLLM KV lifecycle management |

**Recommended**: Option 2 (SDPA for drafter). Minimal code, proven attention
implementation, sidesteps the FlashInfer paged KV issue entirely.

---

### Problem 2: Xid 43 GPU crash on FP8 models — SOLVED

**Symptom**: GPU hang (Xid 43: "GPU stopped processing") when loading any FP8
model (with or without DFlash/MTP). Requires full system reboot to recover.

**Root cause**: `CutlassFp8BlockScaledMMKernel` — a CUTLASS FP8 GEMM kernel that
uses `enable_sm100f_only`, which crashes on SM110 at runtime. The FP8 model
auto-selects this kernel during model loading (weight dequantization).

**Why it was hard to find**: The CUTLASS crash corrupted the CUDA context silently.
Subsequent operations (KV cache setup, CUBLAS calls, FlashInfer JIT) failed with
different errors (CUBLAS_STATUS_EXECUTION_FAILED, cublasCreate failure), making
the root cause appear to be block_size, FlashInfer JIT, TRTLLM, or KV cache dtype.

**Fix**: Add `CutlassFp8BlockScaledMMKernel` to `VLLM_DISABLED_KERNELS` env var.
vLLM falls back to `TritonFp8BlockScaledMMKernel` which works on SM110.

```bash
VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel,CutlassFp8BlockScaledMMKernel
```

**Why NVFP4 profiles worked**: They use `--quantization modelopt` with
`VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`, which never selects the
block-scaled FP8 kernel.

---

### Problem 3: FA2/FA4 cannot run on SM110 for head_dim=256 — UNSOLVABLE

**Symptom**: Xid 43 GPU hang when using `--attention-backend flash_attn`.

**Root cause**: Two separate hardware limitations:
- **FA4** (CuTe DSL): Can't handle head_dim=256 due to TMEM hardware capacity
  limits on Blackwell. Falls back to FA2.
- **FA2**: Uses SM80-specific shared memory access patterns that compile for SM110
  but produce invalid execution at runtime (Xid 43). Even with native SM110 cubins
  (52 cubins compiled and verified via cuobjdump), the kernels crash.

**Impact**: `flash_attn` backend cannot be used for Qwen3.5 models on SM110.
FlashInfer is the only viable attention backend.

**Workaround**: Use `--attention-backend flashinfer`. FlashInfer handles both
head_dim=128 (drafter) and head_dim=256 (target) on SM110 correctly.

---

### Problem 4: FlashInfer non-causal attention not supported — SOLVED

**Symptom**: vLLM refuses to start DFlash with FlashInfer backend:
`non-causal attention not supported`

**Root cause**: FlashInfer's vLLM backend doesn't advertise `supports_non_causal()`,
hardcodes `causal=True` in `.plan()` calls, and has no `causal` field on metadata.
DFlash requires non-causal attention for parallel draft token generation.

**Fix**: `fix-flashinfer-non-causal` runtime mod — 8 patches:
1. `supports_non_causal()` → True
2. `causal` field on FlashInferMetadata
3. `build()` propagates `common_attn_metadata.causal`
4. Prefill `.plan()` uses propagated causal value (was hardcoded True)
4b. Cascade `.plan()` uses propagated causal
5. DCP new_tokens uses causal parameter
6a/6b. Removed `_causal` assertions in forward
7. Bypass TRTLLM prefill when `causal=False`
8. Override `reorder_batch_threshold` when `causal=False`

---

### Problem 5: KV cache page_size assertion on hybrid models — SOLVED

**Symptom**: `AssertionError` in `unify_kv_cache_spec_page_size()` when DFlash
drafter adds attention layers with different per-token KV sizes to a hybrid model.

**Root cause**: Qwen3.5 hybrid model has mamba specs with `page_size_padded` set
to match attention page size. When block_size is adjusted during unification,
`page_size_padded` is a frozen dataclass field that doesn't update.

**Fix**: `fix-kv-page-unify` runtime mod — after adjusting block_size, also
update `page_size_padded` if present.

---

### Problem 6: CUDA 13.2/13.0 PTX version mismatch — KNOWN ISSUE

**Symptom**: `cudaErrorUnsupportedPtxVersion` after system reboot.

**Root cause**: Docker image uses CUDA 13.2 toolkit, host has CUDA 13.0 driver.
PTX compiled with 13.2 can't be JIT'd by 13.0 driver. Works before reboot
because GPU-side cubin caches persist across container restarts.

**Status**: Known issue. Fix requires rebuilding image with CUDA 13.0 base.
The Xid 43 fix (Problem 2) was the actual blocker — this PTX issue only
manifests after reboots and can be worked around by not clearing caches.

---

## Current Working Configuration

```bash
# Environment
VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel,CutlassFp8BlockScaledMMKernel
VLLM_MODS=fix-flashinfer-non-causal,fix-kv-page-unify

# vLLM args
vllm serve lovedheart/Qwen3.5-9B-FP8 \
  --attention-backend flashinfer \
  --enforce-eager \
  --language-model-only \
  --kv-cache-dtype bfloat16 \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-9B-DFlash","num_speculative_tokens":8}'
```

**Status**: Launches, serves inference, no GPU crashes. Acceptance rate ~0%.

---

## Attention Backend Compatibility Matrix (SM110 / Qwen3.5)

| Backend | head_dim=128 (drafter) | head_dim=256 (target) | Non-causal | Status |
|---------|----------------------|----------------------|------------|--------|
| FA4 (CuTe DSL) | Works | TMEM limit → FA2 fallback | Yes | Broken |
| FA2 (any cubins) | Untested | Xid 43 GPU hang | Yes | Broken |
| FlashInfer | Works | Works | Patched | 0% acceptance |
| triton_attn | Works | Works | No support | Can't use |

---

## Original Image Pinned Versions

| Component | Version / Commit |
|-----------|-----------------|
| CUDA base | nvidia/cuda:13.2.0-devel-ubuntu24.04 |
| torch | 2.12.0.dev20260410+cu132 |
| vLLM | 0.19.1rc1.dev195 (e281cb721) |
| FlashInfer | 0.6.7 (904fa8cbc) |
| triton | 3.7.0 |
| transformers | 5.5.3 |
| FA2 cubins | SM80 only |

---

## Full Test History

### Xid 43 Crash Isolation (16 tests)

| # | Config | Result | GPU |
|---|--------|--------|-----|
| 1 | No mods, bare DFlash | Python error: "non-causal not supported" | Clean |
| 2 | fix-kv-page-unify only | Python error: "non-causal not supported" | Clean |
| 3 | metadata-only non-causal + kv-page-unify | **Xid 43** | Crashed |
| 4 | full non-causal + kv-page-unify | **Xid 43** | Crashed |
| 5 | metadata-only only (no kv-page-unify) | **Xid 43** | Crashed |
| 6 | Standalone FlashInfer: head_dim=128, 32/8 heads | **PASS** | Clean |
| 7 | Standalone FlashInfer: page_size=560 | **PASS** | Clean |
| 8 | Standalone FlashInfer: causal=False | **PASS** | Clean |
| 9 | DFlash + skip-dflash-warmup + BF16 KV | **Xid 43** | Crashed |
| 10 | DFlash + skip-dflash-warmup + FP8 KV | **Xid 43** (block_size=528) | Crashed |
| 11 | FP8 9B + MTP (no DFlash) + BF16 KV | **Xid 43** (block_size=528) | Crashed |
| 12 | DFlash + TRTLLM disabled | **Xid 43** (block_size=560) | Crashed |
| 13 | Pre-warm NVFP4 (block_size=1056), then DFlash | **Xid 43** | Crashed |
| 14 | NVFP4 profile + MTP (FP8 KV, block_size=1056) | **Works** | Clean |
| 15 | FP8 + FP8 KV + NO spec decode + **BlockScaled disabled** | **Works** | Clean |
| 16 | FP8 + FP8 KV + DFlash + **BlockScaled disabled** | **Works** | Clean |

**Root cause**: Tests 1-14 all had `CutlassFp8BlockScaledMMKernel` enabled
(not in disabled list). Test 14 worked because NVFP4 doesn't use that kernel.
Tests 15-16 proved the fix.

### DFlash Acceptance Rate Tests

| # | Config | Acceptance | Notes |
|---|--------|-----------|-------|
| A | FlashInfer + FP8 target + BF16 KV (pre-crash-fix, cached JIT) | 0.3% | pos0=2-5%, pos1-7=0% |
| B | FlashInfer + BF16 target + BF16 KV (pre-crash-fix, cached JIT) | 0.3% | Identical to FP8 |
| C | flash_attn backend | Xid 43 | FA2 crashes SM110 |
| D | num_speculative_tokens=1 | NaN | Edge case with 2 query tokens |
| E | FP8 + FP8 KV + DFlash + BlockScaled disabled | dtype mismatch | Drafter writes BF16 K/V into FP8 cache |
| F | FP8 + BF16 KV + DFlash + BlockScaled disabled + non-causal | 0.4% | SLOTS captured, garbage draft tokens |
| G | FP8 + BF16 KV + DFlash + BlockScaled disabled + **causal** | 0.0% | Identical — non-causal vs causal makes no difference |

### Infrastructure Issues

1. **Model re-download**: Wrong `--download-dir` path caused 11 GB re-download
2. **Container --rm**: Crashes lost all logs. Use named containers for debugging.
3. **Debug counter consumed by dummy runs**: Fixed with `input_ids.any()` guard
4. **GPU state corruption**: Multiple Xid 43 events require full reboot
5. **Wheel conflicts**: Multiple vLLM wheels in `wheels/` dir caused install failure
6. **Chat template mount**: Docker creates directory if container path doesn't exist.
   Mount to a path that exists in the container (e.g. `/tmp/chat.jinja`).

---

## Planned Improvements

Each change should be tested individually: non-DFlash profile first, then DFlash.

### Image rebuild

| Change | Why | Priority | Reference |
|--------|-----|----------|-----------|
| CUDA 13.0 base | Match host CUDA 13.0 driver, survive reboots without JIT cache | High | `nvidia/cuda:13.0.0-devel-ubuntu24.04` |
| Drop FA2 SM110 patch | Confirmed Xid 43, FA2 not viable on SM110 | Done | — |

### Packages to add

| Package | What it does | Reference | Notes |
|---------|-------------|-----------|-------|
| InstantTensor | Fast model loading (up to 32x on H200) | `pip install instanttensor`, vLLM flag `--load-format instanttensor` | Already in Dockerfile, not yet tested on Thor |
| TriAttention | KV cache compression via trigonometric frequency-domain scoring (up to 10.7x KV reduction, 2.5x throughput) | `pip install triattention @ git+https://github.com/WeianMao/triattention.git` | vLLM plugin, OpenAI-compatible API. Not on PyPI. |
| TurboQuant | Online KV cache compression (PolarQuant keys + uniform values, up to 4.9x compression, pure Triton) | Merged in vLLM main (PR [#38479](https://github.com/vllm-project/vllm/pull/38479), Apr 15 2026). Flag: `--kv-cache-dtype turboquant_k8v4` | Requires newer vLLM (post dev195). SM121 tested with Triton fallback. |

### vLLM PRs to cherry-pick

| PR | What it does | Reference | Notes |
|----|-------------|-----------|-------|
| #32165 | Separate KV cache dtype for draft model | [vllm-project/vllm#32165](https://github.com/vllm-project/vllm/pull/32165) | Allows `--speculative-config '{"kv_cache_dtype":"auto"}'` so drafter uses BF16 KV while target uses FP8. Not merged (needs-rebase). Would enable FP8 KV for target + BF16 for drafter. |

### Ideal new container design

Based on all findings, the target container should contain:

| Component | Target version | Source | Notes |
|-----------|---------------|--------|-------|
| CUDA base | `nvidia/cuda:13.0.0-devel-ubuntu24.04` | Docker Hub | Match host CUDA 13.0 driver |
| PyTorch | cu130 nightly (pin date) | `https://download.pytorch.org/whl/nightly/cu130` | Pin exact date for reproducibility |
| vLLM | Latest main + PR #32165 | `github.com/vllm-project/vllm` main + `--apply-vllm-pr 32165` | TurboQuant (merged), latest DFlash fixes, separate draft KV dtype |
| FlashInfer | 0.6.7 (commit 904fa8cbc) or latest | `github.com/flashinfer-ai/flashinfer` | Pin to tested commit, or test latest |
| InstantTensor | latest | `pip install instanttensor` | Fast model loading |
| TriAttention | git main | `pip install triattention @ git+https://github.com/WeianMao/triattention.git` | Not on PyPI |
| fastsafetensors | latest | `pip install fastsafetensors` | Already present |
| transformers | 5.5.x | `pip install transformers==5.5.0` | Qwen3.5 + Gemma 4 support |

**Build-time settings:**
- `TORCH_CUDA_ARCH_LIST=11.0a`
- `FLASHINFER_CUDA_ARCH_LIST=11.0a`
- FA2 SM110 patch: NOT included (confirmed Xid 43, FA2 kernels use SM80-specific patterns)
- flash_attn shim symlink: baked into image (`ln -s vllm/vllm_flash_attn flash_attn` for FA4 CuTe DSL imports)

**Runtime environment (entrypoint):**
- `VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel,CutlassFp8BlockScaledMMKernel`

**Runtime mods (volume-mounted, applied at container start):**
- `fix-flashinfer-non-causal` — 8 patches for DFlash non-causal attention on FlashInfer
- `fix-kv-page-unify` — hybrid model mamba page size assertion fix

**Key risk**: vLLM dev319 previously crashed with CUBLAS in GDN layers, but that
was a cascading failure from `CutlassFp8BlockScaledMMKernel` (now disabled).
With the kernel disabled, newer vLLM should work. Test incrementally:
1. Non-DFlash 9B NVFP4 profile first
2. Then DFlash profile
3. If GDN crash recurs, bisect between dev195 and latest main

### Future: DDTree

| Item | What it does | Reference | Notes |
|------|-------------|-----------|-------|
| DDTree | Diffusion Draft Tree — tree-structured DFlash candidates for up to 8.2x speedup | [liranringel/ddtree](https://github.com/liranringel/ddtree) on GitHub, [paper](https://arxiv.org/abs/2602.06036) | Standalone HuggingFace Transformers implementation. No vLLM integration — would require tree verification with custom 4D attention masks, KV cache compaction, SDPA-only target model. Major engineering effort. Track upstream for vLLM integration PRs. |
