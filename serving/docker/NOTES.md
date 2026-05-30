# NemoClaw-Thor vLLM Container Build Notes

Builds vLLM + FlashInfer from source for **Jetson AGX Thor (SM110a / Blackwell)**.
Adapted from [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (SM121) — stripped of Ray/cluster logic, retargeted for single-node Thor with CUDA 13.2 and JetPack 7.1.

---

## Current Working Configuration (v6, pinned 2026-04-17)

**Image tags**:
- `nemoclaw-thor/vllm:latest` — primary tag used by launch.sh
- `nemoclaw-thor/vllm:main-g9965f501a-thor-sm110-cu132` — SHA-based, auto-generated
- `nemoclaw-thor/vllm:v6-pinned-2026-04-17` — explicit safekeeping tag

All three point to the same image layer (built 2026-04-16 11:40 UTC).

**Benchmark results** (through NemoClaw pipeline, 1200-token coding task):

| Profile | Single | 8-concurrent | Max context |
|---------|--------|-------------|-------------|
| Qwen3.6-35B-A3B-NVFP4 + DFlash-15 | 45.7 tok/s | **192.5 tok/s aggregate** | 256K (5 seqs) |
| Qwen3.6-35B-A3B-NVFP4 + TQ-MTP N=4 | 28.6 tok/s | 153.6 tok/s aggregate | 256K (29x budget) |

**Features**: Native SM110 NVFP4/FP8, DFlash speculative decoding, MTP N=4,
TurboQuant K8V4 KV cache, flash_attn (head_dim=128), tool calling (qwen3_xml).

### Pinned versions

The `build-vllm.sh` defaults and `Dockerfile.vllm` pip installs were pinned on
2026-04-17 (commit `fc33d58`) so a fresh rebuild produces the same image.
**Reset to `main`/unpinned when starting the next development stint.**

| Layer | Pinned value |
|-------|--------------|
| **CUDA base** | `nvidia/cuda:13.0.0-devel-ubuntu24.04` |
| **vLLM ref** | `9965f501a89204769a53c86cdee2528947373747` (main @ 2026-04-16) |
| **vLLM PRs** | (none — see "PR #39931 state" below) |
| **FlashInfer ref** | `25b324dbad53942a695a1f00cd7837800de25634` |
| **PyTorch nightly (cu130)** | `torch==2.12.0.dev20260415+cu130` |
| | `torchvision==0.27.0.dev20260415+cu130` |
| | `torchaudio==2.11.0.dev20260415+cu130` |
| **triton** | `3.7.0+gitb4e20bbe` |
| **transformers** | `5.5.4` |
| **fastsafetensors** | `0.2.2` |
| **instanttensor** | `0.1.8` |
| **triattention** | `git@91bb3c27e5000aa2e1f5abb3f247376597f2b5af` |
| **flash-attn-4** | `4.0.0b9` |
| **nvidia-cutlass-dsl** | `4.4.2` |
| **nvidia-nvshmem-cu13** | `3.4.5` |
| **apache-tvm-ffi** | `0.1.10` |

### PR #39931 state (fully delivered via runtime mod)

**Short answer**: PR #39931 is **not applied at build time**. The full PR is
replayed by a runtime mod (`fix-pr39931-turboquant`) that applies all 5
in-place edits at container start.

#### Why no build-time apply

Earlier builds tried `VLLM_PRS="39931"` which used `git apply -v
--exclude='tests/*' || git apply --reject -v`. The `--reject` fallback
silently dropped hunks that conflicted with our pinned vLLM commit
`9965f501a` (which is 22 commits behind the PR's base `bf9a5ddb24`). Net
result: all 5 in-place edits were rejected, so `VLLM_PRS="39931"` added
nothing to the image.

Additional concern: the PR is **unmerged and open**. Upstream can rebase or
force-push to change its contents. A `curl pull/39931.diff | git apply`
invocation would pull whatever state the PR is in on that day, making the
"pinned" build non-deterministic.

`VLLM_PRS=""` avoids both problems — the build only uses the pinned vLLM
commit, nothing fetched from a non-pinned source.

Note on `TQFullAttentionSpec`: this class is sometimes attributed to PR #39931
but actually already exists in vLLM main at commit `9965f501a`. It was merged
via a different code path and is present in our pinned vLLM regardless of any
PR application. The PR diff does not touch `kv_cache_interface.py`.

#### What the PR adds (all handled by the mod)

| PR hunk | File |
|---------|------|
| Gate removal (hybrid model allowed) | `vllm/engine/arg_utils.py` |
| `get_boundary_skip_layers(model_config)` signature | `vllm/engine/arg_utils.py` + `.../turboquant/config.py` |
| Hybrid-aware layer selection + `_get_full_attention_layer_indices` helper | `.../turboquant/config.py` |
| TQ-aware `_align_hybrid_block_size` | `vllm/platforms/interface.py` |
| Flash_attn `out=` kwarg shim | `vllm/v1/attention/backends/turboquant_attn.py` |

#### The runtime mod

`mods/fix-pr39931-turboquant/run.sh` applies all 5 hunks as
exact-string `str.replace` operations with verify + idempotent markers. It is
**not baked into the image** — it lives in the NemoClaw-Thor repo and is
delivered at container start via launch.sh's bind mount:

```bash
# From ../launch.sh
docker_mount_args+=(-v "${THOR_MODS_HOST_DIR}:/workspace/mods:ro")
```

The image's baked-in `/workspace/mods/` is overlaid by this bind mount at
runtime. Profiles that need the mod opt in via
`VLLM_MODS=fix-pr39931-turboquant` in their env args:

- `qwen3.6-35b-a3b-nvfp4-tq-mtp`
- `qwen3.6-35b-a3b-fp8-turboquant`

The entrypoint (`/workspace/entrypoint.sh`) reads `VLLM_MODS` and runs each
named mod's `run.sh` before `exec`-ing into vllm.

#### Why mod-only (no build-time apply)

- **Deterministic builds**: nothing fetched from an unmerged upstream PR
- **Hot-swappable**: mod can be updated without rebuilding the 30-60 min image
- **Context-resilient**: `str.replace` with explicit OLD/NEW blocks is less
  fragile than `git apply` when vLLM main drifts
- **Fail-loud**: the mod errors if any pattern doesn't match exactly once —
  no silent hunk drops

#### When the PR merges upstream

If vLLM merges PR #39931 and we bump `VLLM_REF` to a commit that includes
the changes: the mod's OLD strings won't match any more. The mod will fail
with "pattern not found" — a loud signal that it's no longer needed. At
that point, delete the mod, remove `VLLM_MODS=fix-pr39931-turboquant` from
launch.sh, and the build is cleanly on upstream.

See `../docs/DFLASH-INVESTIGATION.md` "Known Problems" section for the full
chronology of the investigation.

### Key runtime env vars

| Env var | Value | Why |
|---------|-------|-----|
| `VLLM_DISABLED_KERNELS` | `CutlassFP8ScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel,CutlassFp8BlockScaledMMKernel` | Prevents SM110 Xid 43 crash on FP8 models |
| `ENABLE_TRIATTENTION` | `0` | Disable TriAttention plugin (auto-crashes without calibration stats) |
| `HF_TOKEN` | mounted from host | Gated drafter download (z-lab/Qwen3.6-35B-A3B-DFlash) |
| `VLLM_MODS` | per-profile | Runtime mod opt-in (e.g. `fix-pr39931-turboquant` for TQ profiles) |

### Key changes from v2/v4/v5

| Change | Why |
|--------|-----|
| vLLM 0.19.1rc1.dev338 (commit 9965f501a, pinned) | Qwen3.6 + DFlash + TurboQuant support |
| PR #39931 replayed via runtime mod (no build-time apply) | TurboQuant on hybrid (DeltaNet) models |
| transformers pinned to 5.5.4 | Qwen3.6 model class support |
| All v4/v5 runtime mods removed from launch.sh | head_dim=128 on Qwen3.6 means flash_attn works natively; most fixes no longer needed |
| New runtime mod: `fix-pr39931-turboquant` | Completes the partial PR application |
| `CutlassFp8BlockScaledMMKernel` added to disabled list | Prevents Xid 43 GPU crash on FP8 models |
| HF token mount + env var | Required for gated drafter model download |
| `ENABLE_TRIATTENTION=0` | TriAttention plugin added to image but not usable (no Qwen3.6 calibration) |

---

## Previous Working Configuration (v2, 2026-03-28)

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
./build-vllm.sh

# Incremental rebuild (reuse cached FlashInfer wheels):
./build-vllm.sh --skip-flashinfer

# Pin to specific vLLM commit:
./build-vllm.sh --vllm-ref 58a249bc6

# Lower parallelism if build OOMs:
./build-vllm.sh --build-jobs 4
```

## Build phases

1. **FlashInfer** — clones and builds FlashInfer wheels for SM110a (ccache + cubin cache)
2. **vLLM** — clones at specified ref, applies patches, sed+Python CMakeLists fixups, builds wheel
3. **Runner** — installs wheels into clean CUDA runtime image

Wheels are cached in `./wheels/`. `--skip-flashinfer` or `--skip-vllm` reuse them.

## Build history (per-image-generation wall times)

Recorded so future bumps can predict cost and explain regressions. All numbers
are wall-clock at the documented `BUILD_JOBS` setting on Thor (14× ARM Cortex-A78AE,
122 GiB unified memory, 449 GiB free on `/var/lib/docker`).

### v9 — staged 2026-05-21 (not yet built)

`BUILD_JOBS=14` (planned). Pin set: vLLM v0.22.0 + FlashInfer v0.6.12 +
flash-attn-4 4.0.0b15 + nvidia-cutlass-dsl 4.5.2 + transformers 5.9.0 +
nvidia-cudnn-cu13 9.23.0.39 + fastsafetensors 0.3.2 + instanttensor 0.1.9 +
apache-tvm-ffi 0.1.11 (CUDA 13.0.3, torch nightly held at 2026-04-26+cu130).
See `Dockerfile.vllm` header for full per-pin rationale.

**Major-version vLLM bump** (0.20.1 → 0.22.0, skipping v0.21 as a waypoint):
the v9 build is the first release with vLLM's batch-invariant Cutlass FP8 path
on the FP8-KV decode lane — the production cosmos-reason2-8b profile (BF16
weights + FP8 KV cache + Qwen3-VL family) is exactly the target of the
upstream "+28.9% E2E latency" headline. Also lands FlashInfer 0.6.12's XQA
kernel fixes (connects to the FlashRT-on-Thor XQA FP8-KV transferable
optimization identified during audit). Carries forward the deferred v0.21
features: FP8-on-Thor formalization (PR #39712 removes runtime SM-guard
workarounds in `launch.sh`), streaming tool dispatch primitives (#40700,
#41110 — upstream basis for Bridge Tier 1.1), XGrammar 0.2.0 structural tags
for strict tool calling (#40894), Qwen3-VL deepstack heavy-load fix.

**Expected build cost.** Phase 1 (FlashInfer JIT cache) is the cost driver
and depends on the cubin manifest at the new pin. v8.1 stretched Phase 1 to
~3h 5min because v0.6.10 introduced the NVFP4 KV cache attention path
(PR #3097) requiring SM110a-targeted JIT compile of all FA paths. v0.6.12
adds further kernels (per-token NVFP4 quant, XQA fixes) on top, so Phase 1
budget should be **at least the v8.1 baseline (~3h 5min)** and possibly
20-40 min longer if the cubin manifest has grown again. Phases 2-3 should
be comparable to v8.1 (~1h 6min + 6m 34s). **Total expected wall time:
~4h 30min – 5h.**

**Pre-build checklist** (lessons from v8.1):

1. Prune stale wheels in `serving/docker/wheels/`. v8.1 hit a runner-stage
   failure because v0.6.9 + v0.6.10 wheels cohabited; `uv pip install`
   rejected duplicate package URLs. Same risk for v0.6.10 + v0.6.12 cohabitation.
2. Ensure `BUILDKIT_STEP_LOG_MAX_SIZE` is set to 100 MiB (default in build-vllm.sh).
3. Confirm 14 ARM cores are not contended by an active vLLM model on the host
   (peak compile RSS at MAX_JOBS=14 stays under 30 GiB but doesn't leave much
   headroom for a model).
4. **vLLM 0.22 → FlashInfer 0.6.12 override**: vLLM 0.22 ships pinned to
   FlashInfer 0.6.11.post2. We build FlashInfer 0.6.12 separately and the
   runner stage installs from the wheel cache (which contains 0.6.12). The
   override happens naturally as long as the wheels dir contains 0.6.12 and
   not 0.6.11.post2. Verify the wheels dir state before kicking the build.

### v8.1 — built 2026-05-06 18:44 (image `v0.20.1-g132765e35-thor-sm110-cu132-v8.1`)

`BUILD_JOBS=14`. Pin set: vLLM v0.20.1 + FlashInfer v0.6.10 + flash-attn-4 4.0.0b12 +
nvidia-cutlass-dsl 4.5.0 + transformers 5.8.0 + nvidia-cudnn-cu13 9.21.1.3 +
fastsafetensors 0.3.1 + apache-tvm-ffi 0.1.11 (CUDA 13.0.3, torch nightly held at
2026-04-26+cu130). See `Dockerfile.vllm` header for full per-pin rationale.

| Phase | Duration | Notes |
|---|---|---|
| **Phase 1 — FlashInfer** | **~3h 5min** (13:47 → 16:52) | dominant cost |
| └ cubin downloads (11,962 files) | ~14 min (14:02 → 14:16) | NVIDIA artifactory at ~16 files/sec; first-time cache populate |
| └ cubin wheel pack | (interleaved) | flashinfer_cubin-0.6.10 = 325M (vs 282M for v0.6.9; +43M ≈ +3,000 cubins for new NVFP4 KV path) |
| └ JIT cache compile (FLASHINFER_CUDA_ARCH_LIST=11.0a) | ~2h 36min | the long pole; SM110a-targeted JIT compile of all FA paths including the new NVFP4 KV cache attention kernels (PR #3097, all-arch SM80+) |
| **Phase 2 — vLLM CUDA compile** | **~1h 6min** (16:52 → 17:58) | step #22 elapsed = 3950.6s, 358 ninja jobs |
| └ FA2 SM80 hdim256/causal templates (heavy CUTLASS) | front-loaded slow stretch | ~1.2 jobs/min during this phase |
| └ FA3 SM90 hdim192/64 instantiations | tail-end fast stretch | ~9.2 jobs/min after the heavy templates clear |
| **Phase 3a — Runner (failed attempt)** | ~3 min (17:58 → 18:01) | died at step #14 |
| └ Failure mode | uv pip install URL conflict | `serving/docker/wheels/` had v0.6.9 + v0.6.10 wheels side by side; uv refused two file:// URLs claiming the same package name |
| (idle: diagnosis + cleanup + green light) | ~37 min (18:01 → 18:38) | non-build wall time |
| **Phase 3b — Runner (resume, `--skip-flashinfer --skip-vllm`)** | **6m 34s** (18:38:22 → 18:44:56) | BuildKit cache hit through step #13; re-executed step #14 onward against the cleaned wheels dir |
| **Pure build wall time** | **~4h 21min** | sum of Phases 1+2+3a+3b |
| **End-to-end including diagnosis idle** | ~4h 58min | from launch to image landed |

**Lessons:**

1. **FlashInfer phase scales with cubin manifest growth.** v0.6.10 added the
   NVFP4 KV cache attention path (PR #3097, all-arch SM80+); the SM110a-targeted
   JIT compile of the new kernels is what stretched Phase 1 from ~30–40 min on
   v8 to ~3h on v8.1. Future v0.6.x bumps may continue this trend; budget more
   than the v7→v8 transition needed.
2. **Wheel-dir cohabitation breaks the runner stage.** Stale wheels from prior
   builds remain in `serving/docker/wheels/` and the Dockerfile's
   `ls /workspace/wheels/*.whl | grep -v flashinfer_cubin` glob passes them all
   to `uv pip install`, which refuses duplicate package URLs. **Always prune
   stale wheels before a build with bumped FlashInfer/vLLM refs.** Future
   build-vllm.sh hardening: prune older versions of any package wheel before
   running the runner stage. Tracked as a v8.x cleanup item.
3. **Resume after runner-stage failure is cheap.** Buildkit caches all stages
   before the failing RUN. With `--skip-flashinfer --skip-vllm` the resume took
   ~7 min, not another 4h. The wheel directory is the durable artifact;
   recompilation is unnecessary if the wheels are still valid.
4. **The `sm_110a` placeholder in v0.6.10's cubin manifest is empty** (only
   `checksums.txt`, no actual cubins). NVIDIA appears to be preparing to ship
   SM110a CuTeDSL FMHA cubins in a near-future release — when populated, we
   could stop excluding `flashinfer_cubin` from the runner stage and skip the
   JIT compile for that path (potential ~30–60 min savings on Phase 1).

### v8 — built 2026-04-29 10:14 (image `v0.20.0-gb8160878f-thor-sm110-cu132-v8`)

`BUILD_JOBS=14`. Pin set: vLLM v0.20.0 + FlashInfer v0.6.9 + flash-attn-4 4.0.0b10
+ nvidia-cutlass-dsl 4.4.2 + transformers 5.7.0 + nvidia-cudnn-cu13 9.20.0.48.

Approximate wall time **~60–80 min**. Per-phase breakdown not preserved at the
time; the v7→v8 hop was largely image-hygiene (apt cuDNN drop + audio deps)
without bumped FlashInfer or vLLM refs, so cubin-cache and ccache hits absorbed
most of the work.

## File layout

```
docker/
├── Dockerfile.vllm                     # vLLM multi-stage build (base → flashinfer-builder → vllm-builder → runner)
├── Dockerfile.trt                      # TensorRT-Edge-LLM standalone build (single-stage)
├── Dockerfile.bundle                   # vLLM production bundle (baked-in JIT caches)
├── Dockerfile.overlay                  # Quick add-package overlay (dev convenience)
├── build-vllm.sh                       # vLLM build orchestration (multi-phase)
├── build-trt.sh                        # TRT-Edge-LLM build orchestration (single-stage)
├── bundle.sh                           # vLLM production bundle wrapper
├── NOTES.md                            # This file
├── patches/
│   ├── flashinfer_cache.patch          # Skip re-downloading existing cubins
│   └── trt_edge_llm_v0.7.0_thor.patch  # TRT-Edge-LLM Thor build fixes (CUTE_DSL forwarding + .so path)
└── wheels/                             # Exported wheel cache (gitignored)
    ├── flashinfer_*.whl
    └── vllm-*.whl
```

## Related

- PR #38126: `cuda_archs_loose_intersection` cross-suffix fix (merged 2026-03-27) — enables `11.0f` to match `11.0a` in build system
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — original SM121 reference
- [vllm-turboquant](https://github.com/pastoriomarco/vllm-turboquant) — next: KV-cache compression (SM110 port planned, see memory notes)
