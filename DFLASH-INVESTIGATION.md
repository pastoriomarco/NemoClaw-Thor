# DFlash Speculative Decoding on Jetson AGX Thor (SM110)

**Date**: 2026-04-15 — 2026-04-17
**Target model**: Qwen3.6-35B-A3B-FP8 (production), Qwen3.5-9B-FP8 (investigation)
**Drafter model**: z-lab/Qwen3.6-35B-A3B-DFlash (matched), z-lab/Qwen3.5-9B-DFlash (9B)

---

## NemoClaw-routed Benchmark Results (2026-04-17 evening)

Measured through the NemoClaw pipeline (sandbox → OpenShell proxy → vLLM),
complex coding task (1200-token async rate-limited HTTP client implementation),
temperature 0.2, `/no_think`.

| Profile | Single req | N-concurrent | Notes |
|---------|-----------|-------------|-------|
| `qwen3.6-35b-a3b-nvfp4-dflash` | **40.77 / 45.71 tok/s** (cold/warm) | **120.51 @ 4-seqs** / **192.45 @ 8-seqs** | Peak per-req 36, mean 25-32 |
| `qwen3.6-35b-a3b-nvfp4-tq-mtp` | **28.58 tok/s** (single, N=4) | **153.62 @ 8-seqs** | ~19-20 tok/s per req. Requires `fix-pr39931-turboquant` mod |
| ~~`qwen3.6-35b-a3b-nvfp4-mtp-fp8kv`~~ | 29.02 tok/s (single) | **CRASH** — CUDA illegal memory at 8-concurrent | Profile **removed** — superseded by tq-mtp on every axis |

**Key observations**:
- **DFlash wins throughput at every concurrency level** (45.7 → 192.5 tok/s scaling 1→8)
- TQ-MTP wins **max context capacity** (262K ctx with ~29x max concurrency = 7.6M token budget vs DFlash's ~1.3M)
- DFlash default now `max_num_seqs=5` + `max_model_len=262144` to expose full 256K context
- NVFP4+MTP+FP8KV removed: TQ-MTP beats it on single throughput (noise), concurrent (works vs crashes), KV capacity (2.22M vs 1.68M), context (262K vs 131K)

**Note on vLLM's max-concurrency reporting**: The log line
`Maximum concurrency for N tokens per request: Mx` is NOT simply
`available_kv / N`. vLLM factors in prefix caching, block reuse, and
scheduling heuristics, so the reported M can exceed the naive
`kv_tokens / ctx_tokens` ratio (e.g. 678K KV / 131K ctx = 5.17x naive
vs 10.09x reported).

### Known Problems (2026-04-17)

**1. PR #39931 partially failed to apply at image build** ✓ Fixed via mod

### Root cause

Our image's vLLM commit (`9965f501a`, Apr 16 morning) is **22 commits behind**
the PR's base commit (`bf9a5ddb24`, Apr 16 evening). The PR diff's context lines
around each hunk don't match our older vLLM file state, so `git apply` rejected
the in-place edits.

The `d25dbf1` commit changed the Dockerfile's PR apply logic to tolerate
partial failures:
```dockerfile
curl ... | git apply -v --exclude='tests/*' || \
(echo "WARNING: partial apply..." && curl ... | git apply --reject -v)
```

For PR #39931, the `--exclude='tests/*'` apply failed on context drift. The
fallback `git apply --reject` silently dropped all in-place edits. (Correction
to earlier analysis: `TQFullAttentionSpec` is not added by this PR at all —
it was already in vLLM main at our pinned commit. The PR only modifies 4
existing files, all of which failed to apply.)

**As of commit `fc33d58`+, `VLLM_PRS=""` at build time** — the PR is delivered
entirely via the runtime mod. See `docker/NOTES.md` for the full rationale.

**Historical image state (before the mod)**:

| Change | Applied? | File |
|--------|---------|------|
| `TQFullAttentionSpec` class | pre-existing | `vllm/v1/kv_cache_interface.py` — already in main @ 9965f501a, NOT from this PR |
| Gate removal | ✗ | `vllm/engine/arg_utils.py` |
| `get_boundary_skip_layers(model_config)` signature + hybrid logic | ✗ | `vllm/model_executor/layers/quantization/turboquant/config.py` |
| `_get_full_attention_layer_indices` helper | ✗ | same |
| TQ-aware page-size alignment | ✗ | `vllm/platforms/interface.py` |
| Flash_attn `out=` kwarg compat shim | ✗ | `vllm/v1/attention/backends/turboquant_attn.py` |

### Fix: `fix-pr39931-turboquant` runtime mod

`docker/mods/fix-pr39931-turboquant/run.sh` replays all 5 rejected edits as
verbatim Python `str.replace` operations against the installed vLLM at
container start. Key properties:

- **Not `git apply`** — uses exact-string matching, immune to context drift
  around the hunks
- **Each replacement verified**: must match exactly once (errors loudly
  otherwise), with `idempotent_marker` so re-applying is a no-op
- **All 5 hunks ported** (both arg_utils.py edits, 4 config.py changes incl.
  the new helper function, platforms/interface.py TQ branch,
  turboquant_attn.py FA shim)

### Verification

```bash
./start-model.sh qwen3.6-35b-a3b-nvfp4-tq-mtp
# [entrypoint] Applying mod: fix-pr39931-turboquant
# Applying PR #39931 (TurboQuant hybrid support) — replaying rejected hunks...
#   [arg_utils.py] patched ...
#   [config.py imports] patched ...
#   [config.py get_boundary_skip_layers] patched ...
#   [config.py from_cache_dtype] patched ...
#   [config.py helper] appended _get_full_attention_layer_indices
#   [platforms/interface.py] patched ...
#   [turboquant_attn.py] patched ...
# All PR #39931 hunks applied successfully.
#
# INFO [config.py:185] TQ hybrid: full-attention layers [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
# INFO Available KV cache memory: 70.78 GiB
# INFO GPU KV cache size: 2,225,328 tokens
# INFO Maximum concurrency for 262,144 tokens per request: 29.05x
```

The "TQ hybrid: full-attention layers" log line confirms the PR's hybrid-aware
logic is running — it correctly identified Qwen3.6-35B-A3B's 10 attention
layers (every 4th of 40 total, the DeltaNet layers are excluded from TQ
compression as intended).

### Wired into launch.sh

Both profiles that use TurboQuant get the mod via `VLLM_MODS`:
- `qwen3.6-35b-a3b-nvfp4-tq-mtp` — NVFP4 weights + TQ K8V4 + MTP N=4
- `qwen3.6-35b-a3b-fp8-turboquant` — FP8 weights + TQ K8V4 + MTP N=4

### Long-term upstream fix

Options for the next image rebuild:
1. **Pin `VLLM_REF=bf9a5ddb24`** (PR base commit) — PR applies cleanly but
   we lose 22 commits of unrelated upstream fixes
2. **Use `git fetch origin pull/39931/head` + `git merge`** — 3-way merge
   handles context drift; fails loudly on real conflicts
3. **Revert d25dbf1's `--reject` fallback** to make partial applies fail the
   build (the current fallback silently ships broken images)
4. **Keep the runtime mod** as the canonical path; decouple from upstream PR
   application entirely

**2. MTP-FP8KV crashes under 8-concurrent load** ⚠️

Single-request worked at 29 tok/s. Under 8 parallel requests, engine crashed:
```
CUDA error: an illegal memory access was encountered
Unsupported tile (128, 64, 64) and cluster (1, 2, 1) shape combination for arch 100
```

Root cause: MoE grouped GEMM autotuner picks different kernel tiles at larger
M dimension (batch 128 from 8 seqs × spec tokens). Some tile/cluster combos
valid on SM100 fail on SM110. The initial single-request autotune pass at
smaller M avoids the bad tile, so first run works.

**Potential workarounds to investigate**:
- Reduce `max_num_seqs` to 4 for MTP profiles (limits M dimension)
- Use `VLLM_USE_FLASHINFER_MOE_FP4=0` (falls back to alternative MoE backend)
- Pre-warm with concurrent batch before serving (forces autotuner to reject
  bad tiles early — might still crash)
- Upstream fix: restrict MoE autotuner tile candidates on SM110

**3. TriAttention plugin auto-activates and crashes** ✓ Fixed

TriAttention v0.1.0 is installed as a vLLM `general_plugin` entry point.
Auto-registers at vLLM startup, monkey-patches the worker, then crashes at
first inference call without a calibration stats file
(`TRIATTN_FATAL_TRITON_SCORING_REQUIRED:stats_path_not_set`).

**Fix applied**: `ENABLE_TRIATTENTION=0` in launch.sh (official off switch).
Upstream CUDA calibration script exists (`scripts/calibrate.py`) but Qwen3.6
is not in the supported-model matrix — porting would require DeltaNet-hybrid
layer support upstream.

---

## 🚀 BREAKTHROUGH: 47.6 tok/s on Qwen3.6-35B-A3B-FP8 (2026-04-17)

**Key discovery**: Qwen3.6-35B-A3B has `head_dim=128` (not 256 like the 9B model).
FA2 works natively on SM110 for head_dim≤128 → DFlash with `--attention-backend
flash_attn` works **without any runtime mods**, matching z-lab's tested configuration.

With the matched z-lab/Qwen3.6-35B-A3B-DFlash drafter:
- **47.6 tok/s average, 94 tok/s peak** (single sequence, DFlash-15)
- **34-48% acceptance**, mean length 5.3-8.3 tokens
- **4.1x speedup** over 35B baseline (11.6 tok/s)

### Full Results Table (Qwen3.6-35B-A3B-FP8 on v6 container)

| Config | Tok/s | Acceptance | Mean Len | Notes |
|--------|-------|-----------|----------|-------|
| Baseline (no spec) | 11.6 | — | 1.0 | |
| MTP N=4 | 29.9 | 77.7% | 4.1 | Built-in heads, no drafter |
| DFlash-3 (mismatched) | 22.1 | 66.3% | 3.0 | Qwen3.5 drafter on 3.6 target |
| DFlash-8 (mismatched) | 26.8 | 27.3% | 3.2 | |
| DFlash-15 (mismatched) | 30.7 | 19.0% | 3.9 | |
| DDTree k=3 (mismatched) | 26.6 | 29.3% | 3.3 | Tree overhead not worth it |
| **DFlash-15 (MATCHED)** | **47.6 avg (94 peak)** | **34-48%** | **5.3-8.3** | **z-lab/Qwen3.6 drafter** |
| DFlash-15 × 8 concurrent | 110.3 | — | — | 3.7x scaling (mismatched) |

### Production Config

```bash
VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel,CutlassFp8BlockScaledMMKernel
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --attention-backend flash_attn \
  --enforce-eager --language-model-only \
  --gpu-memory-utilization 0.8 --kv-cache-dtype bfloat16 \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":15}'
```

Image: `nemoclaw-thor/vllm-v6:latest` (vLLM dev356, clean, no mods)
HF token required for gated drafter model.

### Why head_dim Matters

| Model | hidden_size | Q heads | KV heads | head_dim | FA2 on SM110 |
|-------|-----------|---------|----------|----------|-------------|
| Qwen3.5-9B | 4096 | 16 | 4 | **256** | ❌ Crashes |
| Qwen3.6-35B-A3B | 2048 | 16 | 2 | **128** | ✅ Works |
| DFlash drafter (35B) | 2048 | 32 | 4 | **64** | ✅ Works |

---

## Previous Investigation (Qwen3.5-9B, FlashInfer)
**Platform**: Jetson AGX Thor, SM110 (Blackwell), 128 GB unified memory
**vLLM**: 0.19.1rc1.dev195 (commit e281cb721)
**FlashInfer**: 0.6.7 (commit 904fa8cbc)
**Goal**: ~45% DFlash acceptance rate (z-lab benchmark: 46.52% FP8, 47.24% BF16)

---

## Problems Found (ordered by difficulty)

### Problem 1: 0% DFlash acceptance rate — SOLVED (7.7%, improving)

**Symptom**: DFlash launches and serves inference, but draft tokens are always
wrong. Acceptance rate was 0% (position 0 ~2%, positions 1-7 = 0%).

**Root cause**: FlashInfer's paged KV cache mechanism is incompatible with
DFlash's `precompute_and_store_context_kv` + forward pattern on SM110.
The precompute writes K/V via `do_kv_cache_update(slot_mapping)`, but
FlashInfer's `.run()` reads from wrong offsets or the K/V tensor layout
is incompatible. Draft tokens read from uninitialized or stale memory.

**How we found it**:
1. SLOTS debug captured slot mapping values — internally consistent, correct
   drafter block_table used, block_size=16
2. Tested causal=True vs causal=False — identical 0% acceptance, ruling out
   non-causal masking as the cause
3. Standalone FlashInfer tests passed — the kernel itself works on SM110
4. z-lab only tested with `flash_attn` backend, never FlashInfer — suggesting
   the write/read pattern depends on flash_attn-specific tensor layouts

**SLOTS debug output** (confirmed correct slot mapping):
```
SLOTS #1: block_size=16, num_context=18, num_query=9,
  ctx_slots[:8]=[26880, 26881, 26882, 26883, 26884, 26885, 26886, 26887],
  query_slots[:9]=[26898, 26899, 26900, 26901, 26902, 26903, 26904, 26905, 26906]
DRAFT #1: [161284, 41979, 36272, ...]  (CJK garbage — reading uninitialized KV cache)
```

**Fix**: `fix-dflash-sdpa` runtime mod — replaces FlashInfer paged KV attention
with PyTorch SDPA for the drafter's 5 attention layers only:

1. `precompute_and_store_context_kv`: stores K/V in plain per-layer buffers
   instead of writing to paged KV cache via `do_kv_cache_update`
2. `DFlashQwen3Attention.forward`: uses `F.scaled_dot_product_attention` with
   concatenated context+query K/V instead of FlashInfer `.run()`
3. Model forward propagates per-layer context K/V buffers to attention layers

The target model continues using FlashInfer for its attention (head_dim=256,
paged KV works correctly for normal attention — only the DFlash precompute
pattern was broken).

**Results after fix**:
- Position 0: **28.5%** acceptance (was ~2%)
- Overall: **7.7%** (208/2696 tokens accepted)
- Mean acceptance length: **1.4-1.8 tokens**
- Per-position: 28.5%, 7.7%, 5.6%, 5.3%, 5.0%, 4.1%, 3.6%, 1.8%

**Gap vs z-lab benchmark (46.52%)**:
Still below target. Possible reasons:
- FP8 model + BF16 drafter may have numerical divergence in hidden states
- Model generates `Thinking Process:` reasoning tokens the drafter wasn't trained on
- The no-think chat template may not fully suppress reasoning
- SDPA scale factor or numerical precision differences vs flash_attn
- Need to test with more diverse prompts for stable metrics

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
VLLM_MODS=fix-flashinfer-non-causal,fix-kv-page-unify,fix-dflash-sdpa

# vLLM args
vllm serve lovedheart/Qwen3.5-9B-FP8 \
  --attention-backend flashinfer \
  --enforce-eager \
  --language-model-only \
  --kv-cache-dtype bfloat16 \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-9B-DFlash","num_speculative_tokens":8}'
```

**Status**: Launches, serves inference, no GPU crashes.
**Acceptance rate**: 7.7% overall (8 tokens), position 0 at 28.5%.
**1-token acceptance**: 85-92% (DFlash with num_speculative_tokens=1).
**Target**: 46.52% (z-lab benchmark, flash_attn backend on standard GPU).

**Root cause of 85% → 8% degradation**: The DFlash parallel denoising mechanism
(8 mask tokens denoising simultaneously with non-causal attention) does not work
correctly with SDPA. Single-token prediction (where only 1 mask token attends to
context + bonus) works at 85-92%. Adding 7 more mask tokens causes severe
interference, dropping all positions to ~8%.

z-lab achieved 47% with flash_attn paged KV attention. Our SDPA uses plain
concatenated K/V buffers instead of paged KV. The parallel denoising seems to
depend on flash_attn's specific numerical path or paged KV layout.

**What we ruled out** (via comprehensive diagnostics):
- FP8 quantization: BF16 target gives same ~8% acceptance
- SDPA computation: manual Q@K^T→softmax→@V matches SDPA output (max_diff=0.06, BF16 precision)
- GQA expansion: native GQA vs repeat_interleave gives 0 diff
- SDPA backend: MATH (FP32 softmax) gives same ~8% as default (flash/efficient)
- Context accumulation: verified context grows properly across DFlash steps (20→29→38)
- Attention patterns: bonus token correctly attends to recent context (RoPE decay visible)

**Update (2026-04-16 late)**: Paged-cache SDPA hybrid improved acceptance 3x:

### FlashInfer Bug Investigation (2026-04-16, comprehensive)

DFlash on SM110 requires FlashInfer for attention (FA2/FA4 crash on SM110 for
head_dim=256). The DFlash drafter (32 Q heads, 8 KV heads, head_dim=128) has
different architecture from the target (16 Q heads, 4 KV heads, head_dim=256).
This heterogeneous configuration exposes multiple FlashInfer integration bugs.

**Note**: DFlash was never officially tested with FlashInfer upstream.
PR #36847 (DFlash merge) states: *"FlashInfer (TRTLLM) attention does not
support non-causality. A follow-up would be to allow different attention
backends for the drafter and the target model."*

#### Bug #1: `num_qo_heads` mismatch — FIXED

**Symptom**: FlashInfer attention computed with 16 Q heads (target's) instead of 32 (drafter's).
**Root cause**: `FlashInferMetadataBuilder` reads `num_qo_heads` from `model_config`
(target model) instead of from the drafter's `kv_cache_spec`.
**Fix**: Mod `fix-flashinfer-qo-heads` — detect head_dim mismatch with model_config and
derive `num_qo_heads` from `num_kv_heads × GQA_ratio`.
**Upstream**: PR #39775 (open, 2026-04-14) adds `num_q_heads` to `AttentionSpec` — same fix.

**Verification**: `FI-FIX: Corrected num_qo_heads to 32 (kv_heads=8 × ratio=4)`

#### Bug #2: HND stride corruption — NOT APPLICABLE on SM110

**Symptom**: FlashInfer produces wrong attention with HND-permuted kv_cache view.
**Root cause**: Standalone test confirms NHD layout works for non-causal prefill,
HND (permuted) doesn't. FlashInfer's paged prefill kernel on SM110 doesn't handle
strided HND views correctly.
**Status**: SM110 already uses NHD (identity stride_order), so this bug is dormant.
Would affect platforms that use HND layout.

**Verification**: Standalone test — NHD non-causal max_diff=0.008 (OK), HND max_diff=2.2 (broken).

#### Bug #3: All-zero attention output — FIXED

**Symptom**: `unified_attention_with_output` returns all-zero tensor for drafter layers.
**Root cause**: On SM110/SM121, `use_trtllm_decode_attention=False`, causing drafter's
1-token steps to be classified as "decode" and routed to `BatchDecodeWithPagedKVCacheWrapper`
which returns zeros on SM110 (crashes on SM121).
**Fix**: Mod `fix-flashinfer-drafter-prefill` (port of PR #36060) — override
`build_for_drafting` to set `reorder_batch_threshold=0`, forcing prefill path.
**Upstream**: PR #36060 (open, 2026-03-04) — identical fix for SM121.

**Verification**: After fix, output norm=2204 (non-zero). Before: norm=0.

#### Bug #4: Stale wrapper internal state — NOT THE ROOT CAUSE

**Symptom**: FlashInfer output identical across different drafter steps (same norm=2204.661133).
**Hypothesis**: `BatchPrefillWithPagedKVCacheWrapper` internal GPU buffers don't refresh
between `.plan()` calls.
**Testing**: Created fresh wrapper instance per step — no improvement. Also tested
`fast_build=False` and `use_trtllm_decode_attention=False` overrides — no improvement.
**Conclusion**: The identical outputs may be because the first two DFlash steps have
very similar context (only a few tokens different), producing similar attention.
The wrapper IS refreshing correctly.
**Upstream**: PR #39546 (open, 2026-04-10) fixes a related stale buffer issue for
TRTLLM paths on Blackwell, but doesn't apply to our non-TRTLLM path.

#### Bug #5: FlashInfer pipeline produces wrong attention output — UNFIXED

**Symptom**: FlashInfer produces non-zero but WRONG attention output for the drafter
(max_diff=153.7 vs SDPA reference, 3.4x magnitude difference). 0.1-0.3% acceptance.
Output magnitude: FlashInfer norm=2204 vs SDPA norm=658 for layer 32 (first drafter).
Layer 33 closer: norm=852 vs 957, diff=34. Error originates in layer 32 and propagates.

**Comprehensive verification (all confirmed correct)**:

| Component | Method | Result |
|-----------|--------|--------|
| `sm_scale` | Log in `build_for_drafting` | 0.088388 (1/√128) ✓ |
| `num_qo_heads` | Log + FlashInfer API logging | 32 ✓ |
| `num_kv_heads` | FlashInfer API logging | 8 ✓ |
| `head_dim` | FlashInfer API logging | 128 ✓ |
| `page_size` | FlashInfer API + cache shape dim 2 | 16 ✓ |
| `causal` | FlashInfer API logging | False ✓ |
| NHD layout | stride_order = identity | ✓ |
| `num_prefills` | Log in `build_for_drafting` | 1 (prefill wrapper created) ✓ |
| Prefill wrapper | Metadata check | Not None ✓ |
| Cache write: precompute | Slot readback | 0.000000 diff ✓ |
| Cache write: query KV | `unified_kv_cache_update` logging | Fires, sm=torch.Size([18]) ✓ |
| Cache tensor: drafter vs target | Shape comparison | Separate: [72100,2,16,8,128] vs [72100,2,16,4,256] ✓ |
| Layer names in slot_mapping | Dict lookup logging | All 5 drafter layers found ✓ |
| Standalone FlashInfer kernel | Known-pattern test | Matches SDPA (max_diff=0.008) ✓ |
| Direct `.run()` bypass | Called wrapper.run() directly | Still wrong (same as pipeline) |
| Fresh wrapper per step | New instance each call | No improvement |
| `backend="fa2"` | Forced non-trtllm-gen | No improvement |
| Explicit pre-write of query KV | `do_kv_cache_update` before `self.attn()` | No improvement |

**What we ruled OUT**:
- Custom op dispatch (`unified_attention_with_output`): direct `.run()` same result
- Missing KV writes: both precompute and query writes verified firing
- Stale wrapper state: fresh wrapper per step, no improvement
- Wrong backend: fa2 forced, no improvement
- Slot mapping mismatch: correct layer names, correct slot tensors
- kv_sharing skip: `kv_sharing_target_layer_name = None` on all drafter layers

**Remaining hypothesis**: FlashInfer's JIT-compiled prefill kernel for SM110 produces
different results when operating on vLLM's large shared cache pool (72100 blocks) vs
a standalone small cache. The `.plan()` parameters are identical, but the kernel's
page-indexing behavior differs with the large cache. Possible JIT code generation
issue where the kernel's index arithmetic overflows or uses wrong addressing for
block indices > ~1000.

**Evidence**: Cache dump showed FlashInfer reads pages [1680, 1681]. Page 1680 is
correct (precompute wrote there). Page 1681 contained stale TARGET model data.
Even after explicit query KV write to page 1681, FlashInfer output didn't change —
suggesting FlashInfer's kernel reads from a DIFFERENT offset within the page than
where `reshape_and_cache_flash` wrote. This could be a cache stride/offset
calculation bug in the JIT kernel for SM110 with `num_kv_heads=8, head_dim=128`
(the drafter's configuration that differs from the target's `4×256`).

**Custom op path traced**:
```
self.attn(q, k, v)
  → Attention.forward()
    → torch.ops.vllm.unified_kv_cache_update(k, v, "model.layers.32.self_attn.attn")
        → get_attention_context(layer_name)
          → forward_context.no_compile_layers[layer_name]  # gets drafter's Attention
          → attn_layer.kv_cache                             # gets drafter's cache
          → slot_mapping.get(layer_name)                    # gets query slots
        → attn_layer.impl.do_kv_cache_update(...)          # writes to drafter cache ✓
    → torch.ops.vllm.unified_attention_with_output(q, k, v, out, "model.layers.32.self_attn.attn")
        → get_attention_context(layer_name)                 # same lookup
        → FlashInferImpl.forward(q, k, v, layer, kv_cache, attn_metadata, output)
          → prefill_wrapper.run(q, kv_cache, out=output)    # FlashInfer kernel → WRONG
```

**Upstream references**:
- flashinfer-ai/flashinfer#1709: `.plan()` wrong output for `causal=False` with trtllm-gen
- flashinfer-ai/flashinfer#2849: `BatchPrefillWithPagedKVCacheRun` crash on SM121
- vllm#37754: FlashInfer + MTP crash on SM121 with illegal memory access

**Additional verification (2026-04-16 late, from second-opinion analysis)**:

| Check | Method | Result |
|-------|--------|--------|
| `kv_cache.data_ptr()` precompute vs forward | Log in both paths | **MATCH** (281432027037696) |
| `use_direct_call=True` (bypass custom op) | Forced on drafter Attention | **Same 0.1%** — not a custom op issue |
| `maybe_transfer_kv_layer` decorator | Source review | **No-op** (no KV transfer group) |
| `window_left` / `logits_soft_cap` | Model config check | **Both None** (Qwen3 doesn't use them) |
| `no_compile_layers` key type | Source review | Raw strings, matches slot_mapping keys |
| `infer_global_hyperparameters` leakage | sm_scale logged + config check | **Correct** (drafter group has own params) |

**Exhaustive elimination summary**:
Every component of the `unified_attention_with_output` dispatch chain has been
independently verified correct. The cache is the same physical tensor in precompute
and forward. The plan parameters match the standalone test. The standalone kernel
produces correct output. Yet the pipeline kernel produces wrong output (max_diff=153.7,
3.4x magnitude). The only remaining difference is the **cache pool size** (72100+
blocks in pipeline vs ~100 in standalone).

**To reproduce upstream**: File issue at flashinfer-ai/flashinfer with:
1. SM110 (Jetson AGX Thor), FlashInfer 0.6.8, CUDA 13.0
2. `BatchPrefillWithPagedKVCacheWrapper` with `causal=False`
3. 32 Q heads, 8 KV heads, head_dim=128, page_size=16, BF16
4. Large cache pool (72100+ blocks) — works with small cache (~100 blocks), fails with large
5. `backend="auto"` and `backend="fa2"` both fail
6. All plan parameters verified identical between passing and failing cases
7. `kv_cache.data_ptr()` verified same between write and read paths
8. `use_direct_call=True` (bypassing torch custom op) produces same wrong output

### Paged-Cache SDPA Hybrid (fix-dflash-paged-sdpa) — BEST WORKING APPROACH

Workaround for bugs #3-5: Keep paged cache writes via `do_kv_cache_update`
(verified correct), but replace FlashInfer forward with manual cache reads + SDPA.

1. Precompute: writes context K/V to paged cache via `reshape_and_cache_flash`
2. Forward: writes query K/V to cache, reads ALL K/V back using slot indices
3. SDPA computes attention over paged-cache K/V (not plain buffers)

| Method | Acceptance (8 tok) | Tok/s | Notes |
|--------|-------------------|-------|-------|
| Plain SDPA buffers | 8% | 21.8 (DFlash-3) | Original workaround |
| **Paged-cache SDPA** | **18-22%** | **25.7 (DFlash-8)** | **3x acceptance gain** |
| z-lab (flash_attn) | 47% | N/A | Target benchmark |

Reading from paged cache gives better K/V than plain buffers because
`reshape_and_cache_flash` applies the same write path that flash_attn uses.
Remaining gap to z-lab's 47% is the SDPA vs flash_attn attention computation itself.

### Upstream References

| PR / Issue | Description | Relevance |
|-----------|-------------|-----------|
| [#39775](https://github.com/vllm-project/vllm/pull/39775) | Fix `num_qo_heads` for heterogeneous drafters | Our bug #1 (same fix) |
| [#36060](https://github.com/vllm-project/vllm/pull/36060) | Force prefill for drafter on SM121 | Our bug #3 (same fix) |
| [#39546](https://github.com/vllm-project/vllm/pull/39546) | Fix stale FlashInfer buffers on Blackwell | Related to bug #4 |
| [#39126](https://github.com/vllm-project/vllm/pull/39126) | Separate attention backend for drafter | Would solve bug #5 |
| [#13264](https://github.com/vllm-project/vllm/issues/13264) | Drafter inherits target's `static_forward_context` | Root cause family |
| [#36847](https://github.com/vllm-project/vllm/pull/36847) | DFlash merge — notes FI incompatibility | Official acknowledgment |

### Runtime Mods Summary

All DFlash mods in `docker/mods/`:

| Mod | Purpose | Required? |
|-----|---------|-----------|
| `fix-flashinfer-non-causal` | 8 patches for non-causal attention in FlashInfer | Yes (DFlash) |
| `fix-kv-page-unify` | Hybrid model mamba page size assertion | Yes (Qwen3.5) |
| `fix-flashinfer-qo-heads` | Correct Q head count for drafter metadata | Yes (bug #1) |
| `fix-flashinfer-drafter-prefill` | Force prefill path + fresh wrapper for drafter | Yes (bug #3) |
| `fix-dflash-sdpa` | SDPA workaround (plain buffers) — 8% acceptance | Superseded |
| `fix-dflash-paged-sdpa` | Paged-cache SDPA hybrid — 22% acceptance | **Recommended** |
| `fix-dflash-branchspec` | BranchSpec top-K verification | Optional |
| `diag-flashinfer-kv` | KV cache write/read-back verification | Debug only |
| `diag-fi-vs-sdpa` | FlashInfer vs SDPA output comparison | Debug only |
| `diag-fi-capture` | FlashInfer .run() parameter capture | Debug only |

---

## Attention Backend Compatibility Matrix (SM110 / Qwen3.5)

| Backend | head_dim=128 (drafter) | head_dim=256 (target) | Non-causal | DFlash Status |
|---------|----------------------|----------------------|------------|---------------|
| FA4 (CuTe DSL) | Works | TMEM limit → FA2 fallback | Yes | Broken |
| FA2 (any cubins) | Untested | Xid 43 GPU hang | Yes | Broken |
| FlashInfer native | 5 bugs found, 4 fixed | Works | Patched | 0.3% (bug #5) |
| FlashInfer + paged SDPA | Reads from cache + SDPA | Works | Yes | **22% (working)** |
| PyTorch SDPA (plain) | Works | N/A (target uses FI) | Yes | 8% (degraded) |
| triton_attn | Works | Works | No support | Can't use |
| MTP (built-in heads) | N/A (no drafter) | Uses target's backend | N/A | **76% / 28 tok/s** |

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

## DFlash-N Sweep: Optimal Speculative Token Count (2026-04-16)

Tested DFlash with varying `num_speculative_tokens` on FP8 9B target, SM110.
All tests use SDPA attention (fix-dflash-sdpa mod), FlashInfer for target.

| N | Tok/s | Acceptance | Mean Len | vs Baseline | Notes |
|---|-------|-----------|----------|-------------|-------|
| 0 | 18.2 | N/A | 1.00 | 1.00x | No speculative decode |
| 1 | 16.2 | 91.6% | 1.92 | 0.89x | High acceptance but drafter overhead cancels gain |
| **2** | **19.7** | **82.8%** | **2.66** | **1.08x** | Good sweet spot |
| **3** | **21.8** | **54.3%** | **2.63** | **1.20x** | **Optimal: +20% throughput** |
| 4 | 20.0 | 55.3% | 3.21 | 1.10x | Diminishing returns |
| 6 | 14.2 | 7.5% | 1.45 | 0.78x | Parallel denoising collapses |
| 8 | 13.1 | 3.9% | 1.31 | 0.72x | Worse than baseline |

**Key findings**:
1. Acceptance drops sharply between N=4 (55%) and N=6 (7.5%) — DFlash parallel
   denoising breaks down with SDPA around N=5
2. Drafter overhead is significant on Thor: 1B model reads ≈77% of 9B target time
3. N=3 is optimal: enough tokens accepted (2.63 mean) to overcome drafter overhead
4. Thor concurrent scaling: 8 parallel requests → 9.5x throughput (160 tok/s)

**Implication for BranchSpec tree**: Use DFlash-3 as the subtree drafter at each
tree level. k=3 branches per level × DFlash-3 per branch = 9 candidates per level.
With 3 tree levels: 1+3+9=13 drafter calls (batchable to ~3 effective on Thor).
Expected throughput: 3-5x single-sequence baseline = 55-90 tok/s for single user.

## BranchSpec: Multi-Candidate Verification (2026-04-16)

BranchSpec modifies the rejection sampler to accept if the target's token is
in the drafter's top-K candidates at each position (instead of exact top-1 match).
Zero extra drafter cost — top-K is computed alongside the greedy sample.

| Config | Tok/s | Acceptance | Mean Len | vs Baseline |
|--------|-------|-----------|----------|-------------|
| No spec decode | 18.2 | N/A | 1.00 | 1.00x |
| DFlash-3 (top-1) | 21.8 | 54.3% | 2.63 | 1.20x |
| DFlash-3 + BranchSpec K=3 | 22.3 | 56-66% | 2.93 | 1.22x |
| DFlash-3 + BranchSpec K=5 | 22.5 | 55-73% | 3.19 | 1.23x |

**Per-position acceptance improvement** (BranchSpec K=5 peak):
- pos 0: 83.3% (was 80% with top-1)
- pos 1: 71.8% (was 50%)
- pos 2: 64.1% (was 30%)

**Observations**:
1. BranchSpec improves acceptance significantly (54%→73%) but throughput only
   +2-3% over DFlash-3 baseline because the extra `compute_logits` call and
   Python-side top-K check add overhead.
2. A Triton kernel for top-K membership check would eliminate the Python overhead.
3. The real win from BranchSpec requires multi-level trees (Option C Hybrid) where
   each level expands k branches with DFlash-3 subtrees.

**Concurrent scaling benchmark** (no spec decode, same prompts):
| Concurrent seqs | Tok/s | Speedup |
|-----------------|-------|---------|
| 1 | 16.9 | 1.0x |
| 2 | 34.8 | 2.1x |
| 4 | 84.0 | 5.0x |
| 8 | 160.8 | 9.5x |

Thor scales nearly linearly with concurrency — spare compute can run
multiple speculative branches essentially for free.

## MTP: The Winner for Thor (2026-04-16)

**Key discovery**: MTP (Multi-Token Prediction) built into Qwen3.5 dramatically
outperforms DFlash on Thor because MTP heads share the target model's weights —
**zero additional weight reads** on Thor's bandwidth-limited architecture.

DFlash uses a separate 1B drafter model (2GB weight reads per step = 77% of target
cost). MTP uses tiny linear heads (~16MB) that ride free on the target forward.

### MTP N-Token Sweep

| N | Tok/s | Acceptance | Mean Len | vs Baseline |
|---|-------|-----------|----------|-------------|
| 0 | 18.2 | N/A | 1.0 | 1.00x |
| 1 | 24.8 | 89-97% | 1.95 | 1.36x |
| 2 | 26.4 | 81-90% | 2.78 | 1.45x |
| 3 | 27.0 | 65-84% | 3.49 | 1.48x |
| **4** | **28.0** | **69-76%** | **4.03** | **1.54x** |
| 5 | 26.9 | 66-73% | 4.44 | 1.48x |

**MTP N=4 is optimal: 28.0 tok/s (+54% over baseline)**

### MTP N=4 Concurrent Scaling

| Concurrent | Tok/s | Speedup (vs 1 MTP) |
|------------|-------|---------------------|
| 1 | 27.0 | 1.00x |
| 2 | 44.0 | 1.63x |
| 4 | 99.7 | 3.70x |
| 8 | 163.0 | 6.05x |

### Why MTP > DFlash on Thor

| Factor | DFlash | MTP |
|--------|--------|-----|
| Drafter model | 1B separate (2GB weights) | Built-in heads (~16MB) |
| Weight reads per step | +2GB (77% of target) | ~0 (shared backbone) |
| Best single-seq throughput | 21.8 tok/s (+20%) | **28.0 tok/s (+54%)** |
| Acceptance at N=4 | 55% (SDPA degradation) | **76%** |
| Concurrent 8-seq aggregate | N/A (crashes) | **163 tok/s** |

### BranchSpec on MTP: Negative Result

BranchSpec top-K verification (accepting if target token ∈ drafter top-K) was
tested on MTP N=4 with K=3. Result: **19.0 tok/s** — 32% SLOWER than plain MTP.

Root cause: BranchSpec forces a full `compute_logits` call to get top-K, which
doubles the logits computation cost. MTP's optimized `get_top_tokens` path avoids
full-vocab logits entirely. The overhead exceeds the acceptance gain.

Lesson: on bandwidth-limited platforms, minimizing compute per step matters more
than maximizing acceptance rate. MTP's lean design wins.

### All Approaches Ranked (Single-Sequence Throughput on Thor)

| Rank | Method | Tok/s | vs Baseline | Key Advantage |
|------|--------|-------|-------------|---------------|
| 1 | **MTP N=4** | **28.0** | **+54%** | Zero drafter overhead |
| 2 | MTP N=3 | 27.0 | +48% | Slightly fewer misses |
| 3 | MTP N=2 | 26.4 | +45% | Higher acceptance (90%) |
| 4 | MTP N=1 | 24.8 | +36% | 97% acceptance |
| 5 | DFlash-3 + BranchSpec K=5 | 22.5 | +23% | Top-K verification |
| 6 | DFlash-3 | 21.8 | +20% | Best DFlash config |
| 7 | N-gram N=4 | 19.8 | +9% | Zero drafter, pattern-based |
| 8 | N-gram N=8 | 19.5 | +7% | More speculative tokens |
| 9 | No speculation | 18.2 | baseline | — |
| 10 | DFlash-1 | 16.2 | -11% | Drafter overhead too high |
| 11 | DFlash-8 | 13.1 | -28% | Parallel denoising broken |

### Recommended Production Configuration

```bash
vllm serve lovedheart/Qwen3.5-9B-FP8 \
  --attention-backend flashinfer \
  --enforce-eager \
  --language-model-only \
  --kv-cache-dtype bfloat16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
```

For multi-user: add `--max-num-seqs 8` for up to 163 tok/s aggregate.

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
| H | FP8 + BF16 KV + DFlash + BlockScaled disabled + **SDPA drafter** | **7.7%** | Position 0: 28.5%, mean length 1.4-1.8. FlashInfer paged KV was the root cause. |
| I | SDPA + **1 spec token** (FP8 target, BF16 KV) | **85-92%** | Single mask token predicts correctly. DFlash as MTP works great. |
| J | SDPA + 8 tokens + **BF16 target** (Qwen/Qwen3.5-9B) | **4-10%** | Same as FP8. Rules out FP8 quantization as cause. |
| K | SDPA **MATH backend** + 8 tokens (FP8 target) | **3-13%** | FP32 softmax gives same results. Rules out SDPA kernel precision. |
| L | SDPA + diagnostic: **no accumulation** (always reset context buffer) | **6-9%** | Similar to accumulated — recent context dominates via RoPE decay. |
| M | SDPA + diagnostic: **with accumulation** (context grows 20→29→38) | **6%** | Accumulation works but doesn't improve acceptance. |

### SDPA Diagnostic Findings (Test L-M)

Comprehensive diagnostic added to SDPA forward (layer 0 only, first 3 steps):

| Metric | Finding |
|--------|---------|
| SDPA vs manual attention | max_diff=0.06-0.07 (normal BF16 precision) |
| Native GQA vs repeat_interleave | max_diff=0.000000 (identical) |
| Context KV stats | No NaN/Inf, mean≈0, std≈1.7 (K) / 4.7 (V) |
| Bonus token ctx_attn | 69-80% on context, 20-31% on query (mask) tokens |
| Attention entropy | 1.5-1.7 for bonus, 0.9-1.5 for mask tokens |
| Max logit range | [-10.9, 35.8] — attention is sharp, not uniform |

**Key observation from attention patterns**:
- Bonus token: 70% attention on context (mostly last few tokens via RoPE), 30% on self/masks
- Mask tokens: 59-88% on context, attending strongly to last context token + bonus token
- Later mask positions attend MORE to context (88%) and less to query tokens (12%)
- All mask V values are identical (same embedding), only K differs by RoPE position

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

### flash-attn-4 with SM110 support — POTENTIAL GAME CHANGER

| Item | Details | Reference |
|------|---------|-----------|
| flash-attn-4 v4.0.0.beta9 | **Explicit SM110a/SM110f support** (released April 15 2026) | `pip install flash-attn-4` |
| SM110 assertion | `assert self.arch >= Arch.sm_100 and self.arch <= Arch.sm_110f` | Kernel code |
| JIT compiled | CuTeDSL — no prebuilt arch-specific wheel, compiles at runtime | Requires `nvidia-cutlass-dsl >= 4.4.x` |
| Status | Beta, actively developed. SM120 fwd+bwd added in beta6. | Separate package from `flash-attn` |

**Why this matters**: If flash-attn-4 handles head_dim=256 on SM110 (the TMEM
limit that blocked FA4 before), we can use `--attention-backend flash_attn`
directly — the **exact config z-lab tested with**. This would likely give the
full 47% acceptance without the SDPA workaround, since the DFlash code was
designed and tested with flash_attn's paged KV implementation.

**Test plan**: Install `flash-attn-4` in the container, verify it handles
head_dim=256 on SM110, then test DFlash with `--attention-backend flash_attn`.

### Future: BranchSpec — Bandwidth-Aware Tree Speculation for Thor

**Context**: Thor's unified memory architecture has high compute capacity but
limited memory bandwidth (273 GB/s). Single-sequence decoding is bandwidth-bound,
leaving compute cores underutilized. Running multiple speculative branches as
parallel requests in the same batch exploits this spare compute.

**Key finding enabling this approach**: DFlash with 1 spec token achieves
85-92% acceptance on SM110 with SDPA. The parallel denoising (8 tokens)
degrades to 8%, but single-token prediction is excellent.

**BranchSpec Algorithm**:

```
Input: prefix P, target model T, drafter D, budget B, branching factor k
Output: extended sequence P' with n new verified tokens

1. DRAFT: Run DFlash-1 on P → logits L₀ for next token
   Sample top-k tokens: {c₁, c₂, ..., cₖ} from L₀

2. EXPAND: For each candidate cᵢ:
   Run DFlash-1 on P+cᵢ → logits L₁ᵢ
   Sample top-k: {cᵢ₁, cᵢ₂, ..., cᵢₖ} from L₁ᵢ
   Continue expanding until budget B is exhausted (best-first by cumulative log-prob)

3. VERIFY: Build tree attention mask M:
   - Each candidate sees its ancestors + the shared prefix
   - Siblings do NOT see each other (causal within branch, independent across)
   Flatten tree to B candidate sequences
   Submit as one batch to target model T (all share prefix KV via prefix caching)
   Target verifies all candidates in parallel (uses spare compute)

4. ACCEPT: Walk tree from root, accept longest path where target agrees
   All verified tokens become the new prefix.
   Rejected branches are discarded.
```

**Expected throughput improvement** (conservative estimates):

| Per-token accuracy | k=1 (baseline) | k=3 (tree) | k=5 (tree) |
|-------------------|----------------|------------|------------|
| 85% (our 1-token) | 1.85 tok/step | ~4.2 tok/step | ~5.8 tok/step |
| 47% (z-lab ref) | 1.89 tok/step | ~3.5 tok/step | ~4.9 tok/step |

*tok/step = mean verified tokens per DFlash+verify cycle*

Formula: For depth d with per-level acceptance p and branching k,
expected accepted length = Σᵢ₌₁ᵈ [1 - (1-p)ᵏ]ⁱ (geometric series with effective p' = 1-(1-p)ᵏ)

With p=0.85, k=3: p'=0.997, expected length ≈ min(d, 1/0.003) ≈ d (almost always full depth)
But drafter accuracy drops for later positions in the branch, so realistic estimate is 3-6 tokens.

**Thor-specific advantage**: Processing k=3 branches in parallel costs almost nothing
extra — Thor's compute is underutilized during single-sequence decode. The only cost
is KV cache memory for k branches (mitigated by prefix caching for shared prefix).

**Implementation path**:
1. Use DFlash with `num_speculative_tokens=1` as the base drafter (85% accuracy)
2. Run k parallel draft sequences by submitting k "virtual requests" per step
3. Each virtual request shares the same prefix (vLLM prefix caching)
4. Target model verifies all k branches in one batch
5. Accept the longest verified path

This avoids the 8% parallel denoising issue entirely by using sequential 1-token
DFlash in each branch, while exploiting Thor's parallel compute for verification.

### DDTree Reference

| Item | What it does | Reference | Notes |
|------|-------------|-----------|-------|
| DDTree | Diffusion Draft Tree — tree-structured DFlash candidates, verifies multiple continuations in single target forward | [paper](https://arxiv.org/html/2604.12989), [code](https://github.com/liranringel/ddtree), [project page](https://liranringel.github.io/ddtree/) | Published April 14 2026. Built on HuggingFace Transformers. Uses best-first heap to select promising continuations under node budget. Our SDPA approach is a good foundation — SDPA supports arbitrary attn_mask for tree attention. |
