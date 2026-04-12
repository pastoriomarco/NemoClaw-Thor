# NemoClaw-Thor vLLM Container Build Notes

Builds vLLM + FlashInfer from source for **Jetson AGX Thor (SM110a / Blackwell)**.
Adapted from [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (SM121) — stripped of Ray/cluster logic, retargeted for single-node Thor with CUDA 13.2 and JetPack 7.1.

---

## Current Working Configuration (v2, 2026-03-28)

**Image**: `nemoclaw-thor/vllm:main-g58a249bc6-thor-sm110-cu132`
**vLLM commit**: `58a249bc6` (main branch, includes PR #38126 — cross-suffix `cuda_archs_loose_intersection` fix)
**Performance**: **27.3 tok/s** on Qwen3.5-35B-A3B-NVFP4 (vs ~9.5 tok/s in v1, 2.87x improvement)
**Features**: Native SM110 NVFP4 kernels, CUDA graphs (FULL + PIECEWISE), tool calling, FP8 KV cache, prefix caching

### Runtime flags (qwen3.5-35b-a3b-nvfp4 profile)

| Flag | Value | Reason |
|------|-------|--------|
| `VLLM_NVFP4_GEMM_BACKEND` | `flashinfer-cutlass` | SM110 in FlashInfer CUTLASS CC list (128x4 layout) |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `1` | FlashInfer CUTLASS for MoE NVFP4 |
| `VLLM_FLASHINFER_MOE_BACKEND` | `throughput` | Use CUTLASS backend (not TRTLLM — hardcoded `major==10||12` rejects SM110) |
| `VLLM_DISABLED_KERNELS` | `CutlassFP8ScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel` | Prevent SM100 CUTLASS kernels from being selected on SM110 |
| `--attention-backend` | `triton_attn` | FlashInfer TRTLLM FMHA rejects SM110; Flash Attention v2 PTX not compiled for SM110 |
| `--max-num-batched-tokens` | `4096` | Mamba cache align mode: block_size=2096 > default max_num_batched_tokens=2048 |
| `--kv-cache-dtype` | `fp8` | FP8 KV cache (cuBLAS FP8 channel-wise rejects SM110, but KV cache FP8 is fine) |
| `--enable-prefix-caching` | set | Works with Mamba align mode at max_num_batched_tokens=4096 |
| `enforce_eager` | NOT set | CUDA graphs enabled — key to 27.3 tok/s |

### Build-time patches applied to CMakeLists.txt

The patch file `vllm_sm110_no_sm100_cutlass.patch` was written for an older vLLM commit and **fails to apply** on commit g58a249bc6 (all 4 hunks). The build script skips it gracefully.

Instead, a two-step approach is used:

**Step 1 — sed strips all `11.0f`** from arch lists (prevents SM100 CUTLASS kernels that use `enable_sm100f_only` from being built for SM110):
```bash
sed -i 's/;11\.0f//g; s/11\.0f;//g; s/"11\.0f"/""/g' CMakeLists.txt
```

**Step 2 — Python restores `11.0f` in specific variables** (symbols needed at `_C.abi3.so` load time, or for correct kernel dispatch):

| Variable | Occurrences | Kernels compiled | Why needed |
|----------|-------------|-----------------|------------|
| `FP4_ARCHS "10.0f"` → `"10.0f;11.0f"` | 1 | `nvfp4_quant_kernels.cu`, `nvfp4_scaled_mm_kernels.cu` | Native `scaled_fp4_quant_sm1xxa`, `cutlass_scaled_fp4_mm_sm100a` on SM110 |
| `SCALED_MM_ARCHS "10.0f"` → `"10.0f;11.0f"` | 3 | `scaled_mm_c3x_sm100.cu`, `grouped_mm_c3x_sm100.cu` | `cutlass_moe_mm_sm100` symbol referenced unconditionally at `_C.abi3.so` load |
| `CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;12.0f"` → `"9.0a;10.0f;11.0f;12.0f"` | 1 | `moe_data.cu` | `get_cutlass_moe_mm_data_caller`, `get_cutlass_batched_moe_mm_data_caller` referenced unconditionally |

> **Note on the backward-search heuristic bug**: An earlier attempt used a heuristic that searched backward from each `12.8 AND SCALED_MM_ARCHS` guard to find the arch variable. This failed because each section has the pattern:
> ```cmake
> if(>= 13.0)
>   cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f" ...)   # target
> else()
>   cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a;10.3a" ...)  # else-branch, found FIRST
> endif()
> if(>= 12.8 AND SCALED_MM_ARCHS)  # anchor
> ```
> The else-branch line is closer to the anchor than the if-branch, so the backward search hits it first and breaks. Fix: use `str.replace()` directly on the exact string.

**Step 3 — FlashInfer Python patches** (baked into build):
- `flashinfer_cutlass_moe.py`: add `is_device_capability_family(110)` and `(120)` checks (SM110 was rejected by `family(100)` only guard)
- `flashinfer_trtllm_moe.py`: same guard (unfixed upstream as of vLLM 0.18.x)

---

## SM110 Compatibility Issues — Complete Map

| Issue | Root cause | Fix |
|-------|-----------|-----|
| `scaled_fp4_quant` not compiled for SM110 | `FP4_ARCHS` only includes SM100f by default | Restore `11.0f` in `FP4_ARCHS` (PR #38126 cross-suffix makes `11.0f` → `11.0a`) |
| `cutlass_moe_mm_sm100` undefined symbol | `grouped_mm_c3x_sm100.cu` not compiled (SCALED_MM_ARCHS empty after sed strip) | Restore `11.0f` in `SCALED_MM_ARCHS` (3 occurrences) |
| `get_cutlass_moe_mm_data_caller` undefined symbol | `moe_data.cu` not compiled (CUTLASS_MOE_DATA_ARCHS empty after sed strip) | Restore `11.0f` in `CUTLASS_MOE_DATA_ARCHS` |
| MoE NVFP4 selects Marlin fallback | `is_device_capability_family(100)` returns False for SM110 | Python patch `flashinfer_cutlass_moe.py` and `flashinfer_trtllm_moe.py` to add `family(110)/(120)` |
| FlashInfer TRTLLM FMHA rejects SM110 | C++ check `major == 10 || major == 12` | Use `triton_attn` backend |
| Flash Attention v2 PTX not compiled for SM110 | PTX compiled only for SM90 and SM100 | Use `triton_attn` backend |
| CUTLASS scaled_mm crashes on SM110 | `enable_sm100f_only` flag rejects non-SM100f | `VLLM_DISABLED_KERNELS` prevents dispatch; NVFP4 uses FlashInfer path |
| TRTLLM MoE kernel rejects SM110 | C++ `major == 10 || major == 12` guard | `VLLM_FLASHINFER_MOE_BACKEND=throughput` uses CUTLASS instead |
| FP8 weight quantization fails on SM110 | cuBLAS returns `CUBLAS_STATUS_NOT_SUPPORTED` for channel-wise FP8 | Use NVFP4 model instead |
| Mamba cache align assertion fails | `block_size=2096 > max_num_batched_tokens=2048` (default) | `--max-num-batched-tokens 4096` |
| SM100 CUTLASS selected incorrectly | Kernel registry includes SM100 kernels without SM110 guard | `VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel` |

---

## Experiment Log

### Why we switched from FP8 to NVFP4

vLLM 0.16 with `qwen3_coder` streaming was crashing with `IndexError` during tool calls. Rather than patch the streaming parser, we upgraded to vLLM 0.18+ which had the fix. During vLLM 0.18 bringup, we discovered cuBLAS returns `CUBLAS_STATUS_NOT_SUPPORTED` for channel-wise FP8 on SM110 — the FP8 model path is a dead end on Thor. NVFP4 (`compressed-tensors` quantization, `tcgen05` tensor ops) is natively Blackwell and works across SM100/SM110/SM120.

### v1 (9.5 tok/s) — runtime mods approach

The first working build used runtime Python monkey-patches (applied at container start via `VLLM_MODS=fix-sm110-nvfp4`):
- `is_device_capability_family()` patch to return True for SM110 on `family(100)` checks
- `scaled_fp4_quant` fallback stub (symbol not compiled for SM110 in v1 build)
- FlashInfer JIT context patch to allow SM110 in JIT dispatch

This worked but was slow because:
1. Runtime patches intercepted critical dispatch paths
2. CUDA graphs could not be enabled (eager mode forced)
3. FlashInfer JIT cache needed ~50 min to build on first launch

Performance: ~9.5 tok/s decode.

### v2 build attempts — undefined symbol rabbit hole

When moving to native kernels (removing runtime mods), we hit a cascade of undefined symbols at `_C.abi3.so` import time:

**Attempt v2a–v2d**: Discovered `vllm_sm110_no_sm100_cutlass.patch` was stripping `11.0f` from `FP4_ARCHS`, preventing NVFP4 kernels from being compiled. Fixed by removing that hunk from the patch. PR #38126 (merged 2026-03-27) made this work by fixing `cuda_archs_loose_intersection` cross-suffix matching (`11.0f` → `11.0a`).

**v2e**: `cutlass_moe_mm_sm100` undefined. This symbol comes from `grouped_mm_c3x_sm100.cu`, compiled only when `SCALED_MM_ARCHS` includes `11.0f`. The patch stripped it. Tried backward-search heuristic to restore it — heuristic broke because it found the `else()` branch SCALED_MM_ARCHS first.

**v2f**: Diagnostic fallback showed the `12.8` anchor was found but restoration still failed — same heuristic bug. Switched to `str.replace()` — fixed `cutlass_moe_mm_sm100`. But `patch` failed entirely on new commit (all 4 hunks at wrong line numbers for g58a249bc6). Builds proceeded with sed-only stripping.

**v2g**: `get_cutlass_moe_mm_data_caller` undefined. `moe_data.cu` controlled by `CUTLASS_MOE_DATA_ARCHS` — also stripped by sed, never restored. Added CUTLASS_MOE_DATA_ARCHS restore.

**v2h (current)**: All symbols resolved. `_C` imports cleanly. All three symbol groups defined as `T` (text, defined):
- `scaled_fp4_quant_sm1xxa`, `cutlass_scaled_fp4_mm_sm100a` (native SM110 NVFP4)
- `cutlass_moe_mm_sm100` (MoE symbol, exists but dispatched via FlashInfer)
- `get_cutlass_moe_mm_data_caller` et al. (MoE data symbols)

**First launch**: Hit `AssertionError: block_size (2096) must be <= max_num_batched_tokens (2048)` — Mamba cache align mode (triggered by `--enable-prefix-caching` on Qwen3.5-MoE) requires this. Fixed with `--max-num-batched-tokens 4096`.

**Second launch**: Successful. 27.3 tok/s confirmed.

### First-launch JIT compilation (one-time per host)

| Cache | Location (host) | First launch time | Subsequent launches |
|-------|----------------|-------------------|---------------------|
| FlashInfer CUTLASS GEMM/MoE | `~/thor-flashinfer-cache` | ~50 min | instant (from cache) |
| Torch AOT compile (model graph + Triton attn kernels) | `~/thor-vllm-cache` | ~50 min (runs in parallel with FlashInfer) | 4.45 s |
| CUDA graph warmup (profiling run) | in-process (not cached) | 95 s | 95 s (every launch) |

Total first launch: ~50-60 min. Subsequent launches: ~4-6 min.

---

## Image Portability — Moving to Another Thor

### What needs to be transferred

| Artifact | Size | Where | Notes |
|----------|------|-------|-------|
| Docker image | ~16 GiB | Docker daemon | Contains vLLM + FlashInfer compiled for SM110a |
| FlashInfer JIT cache | ~889 MB | `~/thor-flashinfer-cache/` | GEMM/MoE kernel cubins; without this, first launch takes ~50 min |
| Torch AOT compile cache | ~2 GiB | `~/thor-vllm-cache/` | Compiled model graph; without this, first launch takes ~50 min |
| Model weights | ~22 GiB | `~/thor-hf-cache/hub/models--Kbenkhaled--Qwen3.5-35B-A3B-NVFP4/` | Can re-download from HuggingFace instead |

### Option A — Push image to a registry (recommended)

```bash
# On source Thor: push image
docker tag nemoclaw-thor/vllm:main-g58a249bc6-thor-sm110-cu132 \
    ghcr.io/YOUR_ORG/nemoclaw-thor-vllm:main-g58a249bc6-sm110-cu132
docker push ghcr.io/YOUR_ORG/nemoclaw-thor-vllm:main-g58a249bc6-sm110-cu132

# On target Thor: pull
docker pull ghcr.io/YOUR_ORG/nemoclaw-thor-vllm:main-g58a249bc6-sm110-cu132
docker tag ghcr.io/YOUR_ORG/nemoclaw-thor-vllm:main-g58a249bc6-sm110-cu132 \
    nemoclaw-thor/vllm:main-g58a249bc6-thor-sm110-cu132
```

### Option B — Save/load image as tar

```bash
# On source Thor: save (~8-10 GiB compressed)
docker save nemoclaw-thor/vllm:main-g58a249bc6-thor-sm110-cu132 | \
    gzip > nemoclaw-thor-vllm-sm110-cu132.tar.gz

# Transfer (rsync, scp, USB drive, etc.)
rsync -avP nemoclaw-thor-vllm-sm110-cu132.tar.gz user@target-thor:~/

# On target Thor: load
docker load < nemoclaw-thor-vllm-sm110-cu132.tar.gz
```

### Transferring the JIT caches (avoids first-launch 50-min wait)

```bash
# On source Thor: bundle caches
tar czf nemoclaw-thor-jit-caches.tar.gz \
    -C $HOME thor-flashinfer-cache thor-vllm-cache

# On target Thor: restore
tar xzf nemoclaw-thor-jit-caches.tar.gz -C $HOME
```

### Transferring model weights (optional — can re-download instead)

```bash
# On source Thor: bundle model (22 GiB)
tar czf qwen3.5-35b-a3b-nvfp4-weights.tar.gz \
    -C ~/thor-hf-cache/hub models--Kbenkhaled--Qwen3.5-35B-A3B-NVFP4

# On target Thor: restore
mkdir -p ~/thor-hf-cache/hub
tar xzf qwen3.5-35b-a3b-nvfp4-weights.tar.gz -C ~/thor-hf-cache/hub
```

### Summary: minimal transfer for instant startup on target

Image (16 GiB) + JIT caches (2.9 GiB) + model (22 GiB) = ~41 GiB total.
After transfer, startup time on target Thor: ~4-6 min (same as source).

Without JIT caches: startup time on first launch = ~50-60 min (JIT recompiles, then cached).

---

## Build Usage

```bash
# Full rebuild (latest vLLM main):
cd src/NemoClaw-Thor/docker
./build.sh

# Incremental rebuild (reuse cached FlashInfer wheels):
./build.sh --skip-flashinfer

# Pin to specific vLLM commit:
./build.sh --vllm-ref 58a249bc6

# Lower parallelism if build OOMs:
./build.sh --build-jobs 4
```

## Build phases

1. **FlashInfer** — clones and builds FlashInfer wheels for SM110a (ccache + cubin cache)
2. **vLLM** — clones at specified ref, applies patches, sed+Python CMakeLists fixups, builds wheel
3. **Runner** — installs wheels into clean CUDA runtime image

Wheels are cached in `./wheels/`. `--skip-flashinfer` or `--skip-vllm` reuse them.

## File layout

```
docker/
├── Dockerfile              # Multi-stage build (base → flashinfer-builder → vllm-builder → runner)
├── build.sh                # Build orchestration
├── NOTES.md                # This file
├── patches/
│   ├── flashinfer_cache.patch          # Skip re-downloading existing cubins
│   └── vllm_sm110_no_sm100_cutlass.patch  # Strips SM100 CUTLASS from arch lists
│                                           # NOTE: fails on vLLM g58a249bc6+, skipped gracefully
│                                           # Python patch in Dockerfile handles it instead
└── wheels/                 # Exported wheel cache (gitignored)
    ├── flashinfer_*.whl
    └── vllm-*.whl
```

## Related

- PR #38126: `cuda_archs_loose_intersection` cross-suffix fix (merged 2026-03-27) — enables `11.0f` to match `11.0a` in build system
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — original SM121 reference
- [vllm-turboquant](https://github.com/pastoriomarco/vllm-turboquant) — next: KV-cache compression (SM110 port planned, see memory notes)
