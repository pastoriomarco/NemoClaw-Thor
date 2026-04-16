# DFlash Speculative Decoding Investigation — Jetson AGX Thor (SM110)

**Date**: 2026-04-15
**Target model**: Qwen3.5-9B (FP8 and BF16)
**Drafter model**: z-lab/Qwen3.5-9B-DFlash (1B params, BF16)
**Platform**: Jetson AGX Thor, SM110 (Blackwell), 128 GB unified memory
**vLLM**: 0.19.1rc1.dev302 (commit 68be0f853)
**Goal**: Achieve ~45% DFlash acceptance rate (z-lab benchmark: 46.52% FP8, 47.24% BF16)

---

## Summary

DFlash speculative decoding produces **0% acceptance rate** on Thor regardless of
attention backend or target model precision. The root cause is a combination of:

1. **FlashInfer non-causal attention** produces wrong draft tokens despite correct
   metadata (`causal=False` confirmed at all levels)
2. **FA2/FA4 cannot run on SM110** for head_dim=256 — FA4 hits TMEM hardware limits,
   FA2 causes Xid 43 GPU hangs even with native SM110 cubins
3. **CUDA 13.2 container on CUDA 13.0 host** causes PTX version mismatch after reboot
   (JIT cubins get cleared, PTX from 13.2 can't re-JIT on 13.0 driver)

---

## Test Results

### 1. FlashInfer + FP8 target (baseline)

- **Config**: `--attention-backend flashinfer`, `lovedheart/Qwen3.5-9B-FP8`, `--kv-cache-dtype bfloat16`
- **Mods**: fix-flashinfer-non-causal, fix-kv-page-unify, debug-dflash
- **Result**: 0.3% acceptance (position 0: ~2-5%, positions 1-7: 0%)
- **Observations**:
  - Draft tokens are coherent English words but don't match target predictions
  - Example: for "Count from 1 to 20", drafter predicts `\n, I, 0, ,, \n, \n, I, \n` instead of `1, ,, 2, ,, 3, ...`
  - Hidden states are non-NaN, reasonable statistics (std=2.6 after final norm)
  - `attn_metadata.causal = False` confirmed by DFlash assertion (line 267 of dflash.py)

### 2. FlashInfer + BF16 target (isolate FP8)

- **Config**: `--attention-backend flashinfer`, `Qwen/Qwen3.5-9B` (BF16), `--kv-cache-dtype bfloat16`
- **Result**: 0.3% acceptance — identical to FP8
- **Conclusion**: **FP8 is NOT the cause.** z-lab benchmarks confirm FP8 works (46.52% vs 47.24% BF16)

### 3. flash_attn backend (match z-lab config)

- **Config**: `--attention-backend flash_attn`, fix-sm110-fa4 mod
- **Result**: Xid 43 — GPU hang during warmup
- **Details**:
  - FA4 selects correctly for drafter (head_dim=128)
  - FA4 can't handle target's head_dim=256 → falls back to FA2
  - FA2 compiled with native SM110 cubins (52 cubins verified via cuobjdump)
  - FA2 kernels use SM80-specific shared memory patterns that compile but cause GPU hang
  - `dmesg` shows: `NVRM: Xid (PCI:0000:01:00): 43, pid=XXXX, name=VLLM::EngineCor`
  - Xid 43 = "GPU stopped processing" — application error, GPU remains healthy after reset

### 4. flash_attn + flash_attn shim (FA4 CuTe DSL fix)

- **Config**: Same as #3, plus fix-flash-attn-shim mod (symlink flash_attn → vllm.vllm_flash_attn)
- **Result**: Same Xid 43
- **Note**: The shim fixed `ModuleNotFoundError: No module named 'flash_attn'` from FA4 CuTe,
  but the underlying FA2 GPU hang remained

### 5. num_speculative_tokens=1 (simplify DFlash)

- **Config**: FlashInfer, 1 speculative token instead of 8
- **Result**: NaN in drafter hidden states, draft_tokens=[0]
- **Note**: Possible FlashInfer edge case with very small query sizes (2 tokens)

### 6. CUDA 13.2 image after reboot

- **Config**: Original working image (6f16ec812cd1), FlashInfer
- **Result**: `cudaErrorUnsupportedPtxVersion` — PTX compiled with unsupported toolchain
- **Root cause**: CUDA 13.2 container generates PTX that the host's CUDA 13.0 driver
  can't JIT-compile. Worked before reboot because cubins were cached GPU-side.

---

## Attention Backend Compatibility Matrix (SM110 / Qwen3.5)

| Backend | head_dim=128 (drafter) | head_dim=256 (target) | Non-causal | Status |
|---------|----------------------|----------------------|------------|--------|
| FA4 (CuTe DSL) | Works | TMEM limit → falls back to FA2 | Yes | Partial |
| FA2 (SM110 cubins) | Untested | Xid 43 GPU hang | Yes | Broken |
| FA2 (SM80 cubins) | N/A | PTX version crash | N/A | Broken |
| FlashInfer | Works | Works | Patched (causal=False) | 0% acceptance |
| triton_attn | Works | Works | No (hardcoded causal=True) | Can't use for DFlash |
| flex_attention | Unknown | Unknown | No | Not viable |

**Only FlashInfer supports all three requirements** (SM110 + head_dim=256 + non-causal).

---

## FlashInfer Non-Causal Patches Applied (fix-flashinfer-non-causal mod)

8 patches total:

1. `FlashInferBackend.supports_non_causal()` → True
2. `FlashInferMetadata.causal` field added
3. `build()` propagates `common_attn_metadata.causal` to metadata
4. Prefill `.plan()` uses `common_attn_metadata.causal` (was hardcoded `True`)
4b. Cascade `.plan()` uses propagated causal value
5. DCP new_tokens `.plan()` uses causal parameter
6a/6b. Removed `_causal` assertions in forward path
7. Bypass TRTLLM prefill when `causal=False` (TRTLLM has no causal parameter)
8. Override `reorder_batch_threshold` to 1 when `causal=False`
   (forces multi-token queries through prefill path where non-causal is applied)

**Verification**: Patch 8 diagnostic confirmed `causal=False` reaches `build()`.
The `reorder_batch_threshold` was already 1 for the drafter builder (Patch 8 is
a no-op for the drafter but protects against future threshold changes).

---

## Confirmed Working Components

- **Target model forward**: Produces correct output (verified via curl)
- **Aux hidden state capture**: Layers [1, 8, 15, 22, 29] captured correctly
- **fc projection**: shape=[4096, 20480], BF16, std~80-104 (expected)
- **hidden_norm (RMSNorm)**: Normalizes fc output from std~104 to std~0.92
- **precompute_and_store_context_kv**: Applies hidden_norm internally, then KV projection + K-norm + RoPE
- **Drafter forward**: Receives correct input_ids=[bonus, mask, mask, ...], positions sequential,
  no NaN, output std~2.6
- **lm_head/embed_tokens sharing**: Both BF16, correctly shared from target
- **FlashInfer kernel**: `BatchPrefillWithPagedKVCacheWrapper._causal` is set by `.plan()`,
  `run()` uses `MaskMode.NON_CAUSAL` when `_causal=False`

---

## Remaining Hypothesis: KV Cache Slot Mapping

The most likely cause of 0% acceptance is a **block_size mismatch** in the KV cache
slot mapping between the DFlash Triton kernel and the drafter's attention:

- **Target/drafter hybrid model**: block_size=560 (unified with mamba page size)
- **DFlash Triton kernel** (`copy_and_expand_dflash_inputs_kernel`): computes
  `slot = block_id * block_size + (position % block_size)` using `self.block_size`
- **`self.block_size`** is set from `draft_attn_groups[0].get_metadata_builder().kv_cache_spec.block_size`
- If this doesn't match the block_size used to allocate the KV cache pages,
  precomputed context KV goes to wrong slots → drafter reads garbage → wrong predictions

**Debug instrumentation added** (SLOTS #N logging in dflash.py `set_inputs_first_pass`):
logs `self.block_size`, context slot mapping values, query slot mapping values,
and seq_lens. This was ready to capture but the CUDA 13.2 PTX crash prevented testing.

**Next step**: After CUDA 13.0 rebuild, launch with FlashInfer + debug-dflash mod
and capture the SLOTS output to verify slot mapping correctness.

---

## Infrastructure Issues Encountered

1. **Model re-download**: Manual `docker run` with wrong `--download-dir` path caused
   11 GB re-download of FP8 model to `/data/models/huggingface/hub/` instead of using
   cached model at `/data/models/hub/`. Fixed by using `start-model.sh` which handles
   mount paths correctly.

2. **Container --rm**: Crash debugging was hampered by `--rm` flag in `start-model.sh`
   which removes containers on exit. Named containers without `--rm` are needed for
   post-mortem log inspection.

3. **Debug counter consumed by dummy runs**: Forward pass debug counters were exhausted
   by warmup/profiling dummy runs (all-zero inputs). Fixed with guard:
   `_is_real = input_ids.any().item()`

4. **Multiple Xid 43 events corrupted GPU state**: After FA2 Xid 43 crashes, the GPU
   remained in a degraded state until full system reboot.

5. **Old vLLM wheel conflicts**: Multiple `vllm-*.whl` files in `wheels/` directory
   caused `uv pip install` to fail with "conflicting URLs for package vllm".
   Fixed by cleaning old wheels before rebuild.

---

## Current Rebuild (in progress)

**CUDA 13.0 full rebuild** — FlashInfer + vLLM + runner:

| Change | Why |
|--------|-----|
| CUDA 13.0 base image | Match host driver, eliminate PTX version mismatch |
| PyTorch cu130 wheels | Consistent CUDA version throughout |
| Drop FA2 SM110 patch | Confirmed Xid 43, FA2 can't work on SM110 |
| vLLM latest main | Includes TurboQuant (merged Apr 15), latest DFlash fixes |
| TriAttention | KV cache compression plugin (up to 10.7x reduction) |
| InstantTensor | Already present, fast model loading |

After rebuild: resume FlashInfer DFlash slot mapping investigation.

---

## Updated Analysis (2026-04-16)

### Key discovery: DFlash crashes the ORIGINAL image too

After rebooting Thor and clearing JIT caches, we confirmed:
- **Non-DFlash 9B NVFP4 profile**: works perfectly on original image after reboot
- **DFlash profile**: Xid 43 crash on the SAME original image after reboot

This means DFlash **never survived a reboot**. It only worked in previous sessions
because FlashInfer JIT caches from prior non-DFlash runs happened to cover the
kernel configurations DFlash needed. When caches are cleared (reboot), the DFlash
warmup triggers FlashInfer JIT compilation for a kernel configuration that produces
a bad cubin on SM110, causing a GPU hang.

### What's different about DFlash warmup vs normal warmup

| Aspect | Non-DFlash | DFlash |
|--------|-----------|--------|
| Models profiled | Target only | Target + drafter |
| Attention modes | Causal only | Causal + non-causal |
| KV cache block_size | Normal (16) | Unified (560, mamba page match) |
| Speculative tokens | 0 or 1 (MTP) | 8 (parallel drafting) |
| FlashInfer kernels JIT'd | Standard configs | Additional non-causal + unusual block_size |

### Original image pinned versions

| Component | Version / Commit |
|-----------|-----------------|
| CUDA base | nvidia/cuda:13.2.0-devel-ubuntu24.04 |
| nvcc | 13.2, V13.2.51 |
| torch | 2.12.0.dev20260410+cu132 |
| vLLM | 0.19.1rc1.dev195 (e281cb721) |
| FlashInfer | 0.6.7 (904fa8cbc) |
| triton | 3.7.0 |
| transformers | 5.5.3 |
| FA2 cubins | SM80 only |
| fastsafetensors | 0.2.2 |
| instanttensor | NOT installed |

---

## Step-by-step plan to get DFlash working

### Phase A: Isolate the Xid 43 crash trigger (requires reboot between each test)

**Step A1**: DFlash WITHOUT `fix-flashinfer-non-causal` mod
- Keep: fix-kv-page-unify, debug-dflash
- Remove: fix-flashinfer-non-causal
- Expected: vLLM will refuse to start (DFlash asserts `causal=False` on attention
  metadata). If it gets past the assertion somehow, it means non-causal was never
  the issue.
- If it crashes with Xid 43 before the assertion: the crash is in the drafter
  model initialization, not in non-causal attention.

**Step A2**: DFlash WITHOUT `fix-kv-page-unify` mod
- Keep: fix-flashinfer-non-causal, debug-dflash
- Remove: fix-kv-page-unify
- Expected: may crash with `page_size_bytes` assertion. If it launches, the
  block_size will be different (not 560), which tests whether block_size=560
  is the JIT crash trigger.

**Step A3**: DFlash with ALL mods but `--max-num-batched-tokens 512`
- Smaller profiling batch → simpler FlashInfer kernels JIT'd
- If this works: the crash is in a specific kernel shape triggered by larger batches

**Step A4**: Pre-warm FlashInfer with non-DFlash profile first
- Launch 9B NVFP4 (works), let it fully warm up and populate JIT caches
- Stop it
- Launch DFlash profile (reuses cached kernels, only JITs new configs)
- If this works: the crash is a first-run JIT issue that goes away with partial cache

### Phase B: Fix the 0% acceptance rate

Once DFlash launches without crashing:

**Step B1**: Capture SLOTS debug output
- The `[0/4]` instrumentation logs block_size, slot mapping values, seq_lens
- Compare context slot mapping with query slot mapping
- Verify block_size matches between Triton kernel and FlashInfer attention

**Step B2**: Verify KV cache consistency
- Check if precomputed context KV is written to the same cache slots that
  the drafter's attention reads from
- Check if the block table used by the Triton kernel matches the drafter's
  KV cache group

**Step B3**: Test with `--additional-config '{"gdn_prefill_backend":"triton"}'`
- Force Triton for GDN (should already be default on SM110, but be explicit)

### Phase C: Incremental image rebuild

Each step = one change + full test (non-DFlash + DFlash):

**Step C1**: CUDA 13.0 base only
- Change: CUDA_BASE, PyTorch cu130, keep same vLLM/FlashInfer commits
- Risk: PyTorch cu130 might have different cuBLAS behavior
- Test: non-DFlash 9B NVFP4, then DFlash

**Step C2**: + InstantTensor
- `uv pip install instanttensor` in runner stage
- Test: `--load-format instanttensor` on any profile

**Step C3**: + TriAttention
- `uv pip install triattention @ git+https://github.com/WeianMao/triattention.git`
- Test: import works, no runtime errors

**Step C4**: + TurboQuant (newer vLLM)
- Update vLLM ref to latest main (includes TurboQuant PR #38479)
- Risk: GDN code changes, DFlash changes, attention backend changes
- Test: non-DFlash first, then DFlash
- If fails: bisect between dev195 and latest to find breaking commit
