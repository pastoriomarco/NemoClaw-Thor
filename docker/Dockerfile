# syntax=docker/dockerfile:1.6
#
# NemoClaw-Thor vLLM Container — v7 generation
# Builds vLLM + FlashInfer from source for Jetson AGX Thor (sm110 / Blackwell).
#
# Adapted from spark-vllm-docker (sm121) — stripped of Ray/cluster logic,
# retargeted for single-node Thor with CUDA 13.0 and JetPack 7.1.
#
# Recommended invocation for the v7 image:
#   ./build.sh --vllm-ref v0.20.0 --flashinfer-ref v0.6.9
#
# Why those pins (researched 2026-04-27):
#   - vLLM v0.20.0 (tagged 2026-04-27) is the first stable release containing all
#     the Thor-relevant fixes: PR #39233 (sm_110 build target), PR #39546 (SM100+
#     spec-decode test fix), PR #40092 (TurboQuant FA3/FA4 prefill enablement),
#     PR #40654 (avoid seq_lens_cpu GPU->CPU sync), MRv2 acceptance fixes.
#   - FlashInfer v0.6.9 (released 2026-04-24) — the version vLLM v0.20.0 expects.
#     Pure-Python wheel + CuTe-DSL JIT, SM110 detected at runtime, no special args.
#   - flash-attn-4==4.0.0b10 (pinned in runner stage below) — adds SM100 FP8 e4m3/
#     e5m2 forward (PR #2109), unblocks DFlash + FP8 KV combo on Blackwell.
#
# Usage:
#   ./build.sh --vllm-ref v0.20.0 --flashinfer-ref v0.6.9   # canonical v7 build
#   ./build.sh --vllm-ref main                              # bleeding edge, NOT recommended
#   ./build.sh --build-jobs 8                               # lower parallelism if constrained

# ── Build tunables ──────────────────────────────────────────────────
# Thor has 14 ARM cores (Cortex-A78AE). Default to full parallelism. With 128 GiB
# unified memory, peak CUDA compile RSS at 14 concurrent jobs stays well under
# 30 GiB. Drop to 8-10 only if rebuilding while serving a model.
ARG BUILD_JOBS=14
ARG CUDA_BASE=nvidia/cuda:13.0.3-devel-ubuntu24.04

# =========================================================
# STAGE 1 — Base build image
# =========================================================
FROM ${CUDA_BASE} AS base

ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy
ENV VLLM_BASE_DIR=/workspace/vllm

# Build dependencies
RUN apt update && \
    apt install -y --no-install-recommends --allow-change-held-packages \
    curl vim cmake build-essential ninja-build \
    libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
    python3-dev python3-pip git wget \
    libnccl-dev libnccl2 libibverbs1 libibverbs-dev rdma-core \
    ccache \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# PyTorch nightly (cu130 — matches host CUDA 13.0 driver, avoids PTX version mismatch).
# Pinned to 2026-04-26 nightly for v7 build. Note: vLLM v0.20.0 pins torch==2.11.0
# stable, but no 2.11.0 stable cu130/aarch64 wheel exists for Thor — we ride the
# nightly. The 2026-04-15 nightly used in v6 has been pruned from the index, so
# v6 is no longer rebuildable; v7 uses the 2.13-nightly that is currently live.
# Triton ships bundled inside the torch wheel — no explicit pin needed.
ENV UV_HTTP_TIMEOUT=300
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
     uv pip install \
         "torch==2.13.0.dev20260426+cu130" \
         --index-url https://download.pytorch.org/whl/nightly/cu130 && \
     uv pip install \
         "torchvision==0.27.0.dev20260426+cu130" \
         "torchaudio==2.11.0.dev20260426+cu130" \
         --index-url https://download.pytorch.org/whl/nightly/cu130 && \
     uv pip install "nvidia-nvshmem-cu13==3.6.5" "apache-tvm-ffi==0.1.10" filelock pynvml requests tqdm

# Ccache
ENV PATH=/usr/lib/ccache:$PATH
ENV CCACHE_DIR=/root/.ccache
ENV CCACHE_MAXSIZE=50G
ENV CCACHE_COMPRESS=1
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

WORKDIR $VLLM_BASE_DIR

# Thor = sm110, compute capability 11.0
ARG TORCH_CUDA_ARCH_LIST="11.0a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas


# =========================================================
# STAGE 2 — FlashInfer build
# =========================================================
FROM base AS flashinfer-builder

ARG FLASHINFER_CUDA_ARCH_LIST="11.0a"
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
WORKDIR $VLLM_BASE_DIR
ARG FLASHINFER_REF=main
ARG CACHEBUST_FLASHINFER=1

# Clone or update FlashInfer
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "flashinfer" ]; then \
        echo "Cache miss: Cloning FlashInfer..." && \
        git clone --recursive https://github.com/flashinfer-ai/flashinfer.git; \
        if [ "$FLASHINFER_REF" != "main" ]; then \
            cd flashinfer && git checkout ${FLASHINFER_REF}; \
        fi; \
    else \
        echo "Cache hit: Fetching FlashInfer updates..." && \
        cd flashinfer && \
        git fetch origin && \
        git fetch origin --tags --force && \
        (git checkout --detach origin/${FLASHINFER_REF} 2>/dev/null || git checkout ${FLASHINFER_REF}) && \
        git submodule update --init --recursive && \
        git clean -fdx && \
        git gc --auto; \
    fi && \
    cp -a /repo-cache/flashinfer /workspace/flashinfer

WORKDIR /workspace/flashinfer

# Avoid re-downloading existing cubins (pure build-speed optimization;
# tolerant if upstream artifacts.py changed and patch context no longer matches —
# absence only means we re-fetch cubins, not a build break).
COPY patches/flashinfer_cache.patch .
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=cubins-cache,target=/workspace/flashinfer/flashinfer-cubin/flashinfer_cubin/cubins \
    (patch -p1 --forward < flashinfer_cache.patch || \
     echo "flashinfer_cache.patch did not apply cleanly — proceeding without cubin-cache fast path") && \
    sed -i -e 's/license = "Apache-2.0"/license = { text = "Apache-2.0" }/' -e '/license-files/d' pyproject.toml && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    cd flashinfer-cubin && uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    cd ../flashinfer-jit-cache && uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    cd .. && git rev-parse HEAD > /workspace/wheels/.flashinfer-commit


# =========================================================
# STAGE 3 — FlashInfer wheel export
# =========================================================
FROM scratch AS flashinfer-export
COPY --from=flashinfer-builder /workspace/wheels /


# =========================================================
# STAGE 4 — vLLM build
# =========================================================
FROM base AS vllm-builder

ARG TORCH_CUDA_ARCH_LIST="11.0a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
WORKDIR $VLLM_BASE_DIR

ARG CACHEBUST_VLLM=1
ARG VLLM_REF=main

# Clone or update vLLM
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "vllm" ]; then \
        echo "Cache miss: Cloning vLLM..." && \
        git clone --recursive https://github.com/vllm-project/vllm.git; \
        if [ "$VLLM_REF" != "main" ]; then \
            cd vllm && git checkout ${VLLM_REF}; \
        fi; \
    else \
        echo "Cache hit: Fetching vLLM updates..." && \
        cd vllm && \
        git fetch origin && \
        git fetch origin --tags --force && \
        (git checkout --detach origin/${VLLM_REF} 2>/dev/null || git checkout ${VLLM_REF}) && \
        git submodule update --init --recursive && \
        git clean -fdx && \
        git gc --auto; \
    fi && \
    cp -a /repo-cache/vllm $VLLM_BASE_DIR/

WORKDIR $VLLM_BASE_DIR/vllm

# Optional: apply specific vLLM PRs at build time
ARG VLLM_PRS=""
RUN if [ -n "$VLLM_PRS" ]; then \
        echo "Applying PRs: $VLLM_PRS"; \
        for pr in $VLLM_PRS; do \
            echo "Fetching and applying PR #$pr..."; \
            curl -fL "https://github.com/vllm-project/vllm/pull/${pr}.diff" | git apply -v --exclude='tests/*' || \
            (echo "WARNING: PR #$pr partial apply (excluding tests)..." && \
             curl -fL "https://github.com/vllm-project/vllm/pull/${pr}.diff" | git apply --reject -v 2>&1; true); \
        done; \
    fi

# Prepare build requirements
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    for f in requirements/test.txt requirements/test/cuda.txt; do \
        [ -f "$f" ] && sed -i '/^triton\b/d; /^fastsafetensors\b/d' "$f" || true; \
    done && \
    if [ -f requirements/build.txt ]; then \
        uv pip install -r requirements/build.txt; \
    else \
        uv pip install -r requirements/build/cuda.txt; \
    fi

# Apply build-time patches (conditional — skip if already fixed upstream)
# Copy entire patches dir; vllm_*.patch files are optional
COPY patches/ /tmp/patches/
RUN for p in /tmp/patches/vllm_*.patch; do \
        [ -f "$p" ] || continue; \
        echo "Applying $(basename $p)..."; \
        patch -p1 < "$p" || echo "Patch $(basename $p) not applicable, skipping"; \
    done

# SM110 (Thor) CMakeLists.txt arch lists check — vLLM v0.20.0 PR #39233
# made sm_110 a first-class build target; FP4_ARCHS, SCALED_MM_ARCHS,
# MLA_ARCHS, and MoE arch lists all natively include "11.0f". We verify
# (informational) but do NOT modify — the prior strip-and-restore approach
# was for v0.19 and is no longer correct. Audit: docker/PATCHES-AUDIT.md.
RUN echo "Verifying SM110 arch inclusion in CMakeLists.txt..." && \
    echo "  FP4_ARCHS 11.0f count: $(grep -c 'FP4_ARCHS.*11\.0f' CMakeLists.txt || echo 0)" && \
    echo "  SCALED_MM_ARCHS 11.0f count: $(grep -c 'SCALED_MM_ARCHS.*11\.0f' CMakeLists.txt || echo 0)" && \
    echo "  MLA_ARCHS 11.0f count: $(grep -c 'MLA_ARCHS.*11\.0f' CMakeLists.txt || echo 0)" && \
    echo "  Total 11.0f: $(grep -c '11\.0f' CMakeLists.txt || echo 0)"

# MoE backend selection on SM110 — informational verification only (v7+).
#
# vLLM v0.20.0 has TWO FlashInfer MoE backend families:
#   1. FlashInfer CUTLASS MoE  (vllm/.../flashinfer_cutlass_moe.py)
#         — already includes SM110 in its family-capability gate upstream.
#   2. FlashInfer TRT-LLM Gen MoE (vllm/.../experts/trtllm_{bf16,fp8,mxfp4,nvfp4}_moe.py)
#         — gate is `is_device_capability_family(100)` only; rejects SM110.
#
# All our active profiles set:
#     VLLM_USE_FLASHINFER_MOE_FP4=1
#     VLLM_FLASHINFER_MOE_BACKEND=latency
# which select the CUTLASS MoE path, so the TRT-LLM Gen rejection does NOT
# affect them — vLLM falls back to CUTLASS MoE (SM110-supported) cleanly.
#
# We INTENTIONALLY do NOT patch the TRT-LLM Gen gate in v7 because:
#   (a) our profiles don't use that path,
#   (b) we have no evidence TRT-LLM Gen kernels are actually safe on SM110a
#       (the gate guards against this — opening it might unlock runtime crashes),
#   (c) the v6 patch silently no-op'd on v0.20.0's split files anyway, so
#       v6's measured perf was already with TRT-LLM Gen DISABLED on SM110.
#
# If a future profile needs TRT-LLM Gen MoE on SM110, add a targeted patch
# with offline pre-validation against the exact upstream source. See
# docker/PATCHES-AUDIT.md for the v6→v7 transition history.
RUN python3 - <<'PYEOF'
import pathlib
verify = "vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py"
trtllm_files = [
    "vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py",
    "vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py",
    "vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py",
    "vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py",
]
p = pathlib.Path(verify)
if p.exists() and "is_device_capability_family(110)" in p.read_text():
    print(f"VERIFIED {verify}: SM110 family check present (CUTLASS MoE path is SM110-ready)")
else:
    print(f"NOTE {verify}: SM110 check missing or file moved — CUTLASS MoE may need a patch")
for f in trtllm_files:
    p = pathlib.Path(f)
    status = "exists (SM110 NOT patched — falls back to CUTLASS path)" if p.exists() else "missing"
    print(f"INFO  {f}: {status}")
print()
print("MoE backend selection: CUTLASS (SM110-ready upstream) is the active path")
print("via VLLM_USE_FLASHINFER_MOE_FP4=1 + VLLM_FLASHINFER_MOE_BACKEND=latency.")
PYEOF

# NOTE: FA2 SM110 patch removed — confirmed Xid 43 (GPU hang) at runtime.
# FA2 kernels use SM80-specific shared memory patterns that cause hardware
# faults on SM110 even when compiled with native cubins.
#
# FA4 handles SM110 fine, including head_dim=256 (empirically validated
# 2026-04-26 on both Qwen3.6-27B-FP8 and Qwen3.6-35B-A3B-FP8 — the prior
# "head_dim=256 hits TMEM limit" claim in earlier docs was wrong; FA4 runs
# the full-attention layers cleanly through its SM_100 dispatch path on Thor).
# FA4 still does NOT support FP8 KV cache on Blackwell pre-beta10 — beta10
# (pinned below) ships FP8 e4m3/e5m2 forward kernels (PR #2109).
#
# FlashInfer remains the right choice for FP8 KV + non-DFlash workloads;
# flash_attn (FA4) is preferred for DFlash speculative decoding paths.

# Compile vLLM wheel
RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    git rev-parse HEAD > /workspace/wheels/.vllm-commit


# =========================================================
# STAGE 5 — vLLM wheel export
# =========================================================
FROM scratch AS vllm-export
COPY --from=vllm-builder /workspace/wheels /


# =========================================================
# STAGE 6 — Runner (final image)
# =========================================================
FROM ${CUDA_BASE} AS runner

ARG BUILD_JOBS=14
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV VLLM_BASE_DIR=/workspace/vllm

ENV PIP_CACHE_DIR=/root/.cache/pip
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy

# Runtime dependencies
RUN apt update && \
    apt install -y --no-install-recommends --allow-change-held-packages \
    python3 python3-pip python3-dev vim curl git wget \
    libcudnn9-cuda-13 \
    libnccl-dev libnccl2 libibverbs1 libibverbs-dev rdma-core \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

WORKDIR $VLLM_BASE_DIR

# Tiktoken encodings (used by many models' tokenizers)
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken \
         "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken \
         "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

ARG PRE_TRANSFORMERS=1
ENV UV_HTTP_TIMEOUT=300

# PyTorch (cu130 — matches host CUDA 13.0 driver) — pinned to 2026-04-26 nightly (v7)
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
     uv pip install \
         "torch==2.13.0.dev20260426+cu130" \
         --index-url https://download.pytorch.org/whl/nightly/cu130 && \
     uv pip install \
         "torchvision==0.27.0.dev20260426+cu130" \
         "torchaudio==2.11.0.dev20260426+cu130" \
         --index-url https://download.pytorch.org/whl/nightly/cu130 && \
     uv pip install "nvidia-nvshmem-cu13==3.6.5" "apache-tvm-ffi==0.1.10"

# Install FlashInfer + vLLM wheels (bind-mounted from host ./wheels/)
# Exclude flashinfer_cubin wheel: precompiled cubins are downloaded from upstream
# and target sm100f — they crash on sm110 (Thor).
# INCLUDE flashinfer_jit_cache: JIT-compiled .so files are built with
# FLASHINFER_CUDA_ARCH_LIST=11.0a and contain valid sm_110a code.
# This enables the FlashInfer attention backend on Thor.
# With --tf5 (PRE_TRANSFORMERS=1, the default): override vLLM's and compressed-tensors'
# transformers<5 constraint, then explicitly install transformers==5.6.2 (v7).
# transformers 5.2.0: first native Qwen3.5 support.
# transformers 5.3.0: NVFP4 Blackwell quantization integration, Qwen3.5 bug fixes.
# transformers 5.4.0: Qwen3.5 classification, FP8 grouped matmul speedup.
# transformers 5.5.0: Gemma 4 model class (Gemma4ForConditionalGeneration),
#   Gemma4Config, Gemma4VisionConfig, Gemma4AudioConfig, Gemma4Processor.
# transformers 5.6.x: post-5.5 Qwen3.6-MoE / Gemma 4 / NVFP4 fixes.
RUN --mount=type=bind,source=wheels,target=/workspace/wheels \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    if [ "$PRE_TRANSFORMERS" = "1" ]; then \
        echo "transformers>=5.0.0" > /tmp/tf-override.txt && \
        uv pip install $(ls /workspace/wheels/*.whl | grep -v flashinfer_cubin) --override /tmp/tf-override.txt && \
        uv pip install "transformers==5.6.2" --override /tmp/tf-override.txt; \
    else \
        uv pip install $(ls /workspace/wheels/*.whl | grep -v flashinfer_cubin); \
    fi

# Runtime environment
ARG TORCH_CUDA_ARCH_LIST="11.0a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG FLASHINFER_CUDA_ARCH_LIST="11.0a"
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
ENV TIKTOKEN_ENCODINGS_BASE=$VLLM_BASE_DIR/tiktoken_encodings
ENV PATH=$VLLM_BASE_DIR:$PATH

# Runtime mods directory — applied at container start via entrypoint
COPY mods/ /workspace/mods/

# No Ray — single-node only.
# - fastsafetensors/instanttensor: fast model loading (--load-format instanttensor)
# - triattention: KV cache compression via trigonometric frequency-domain scoring
#   (vLLM plugin — transparent, up to 10.7x KV memory reduction for long reasoning)
# - flash-attn-4 beta10: FA4 with SM110a/SM110f support (JIT via CuTeDSL).
#   Enables --attention-backend flash_attn for DFlash on SM110 with head_dim=256.
#   Beta10 (2026-04-22) adds SM100 FP8 e4m3/e5m2 forward (PR #2109), unblocking
#   DFlash + FP8 KV combo. Plus SM100 seqlen_k=0 crash fix (PR #2470).
#   Requires nvidia-cutlass-dsl >= 4.4.x
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install "fastsafetensors==0.3" "instanttensor==0.1.8" && \
    uv pip install "triattention @ git+https://github.com/WeianMao/triattention.git@325297218a" && \
    uv pip install "flash-attn-4==4.0.0b10" "nvidia-cutlass-dsl==4.4.2"

# flash_attn shim: merge vendored vllm_flash_attn modules into the flash-attn-4
# package. flash-attn-4 only provides cute/, but vLLM also imports flash_attn.ops
# (for rotary embeddings) and flash_attn_interface (for varlen_fwd). Copy missing
# modules from the vendored copy.
RUN SITE=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    FA="$SITE/flash_attn" && \
    VFA="$SITE/vllm/vllm_flash_attn" && \
    if [ -d "$FA" ] && [ -d "$VFA" ]; then \
        for subdir in ops; do \
            if [ ! -e "$FA/$subdir" ] && [ -d "$VFA/$subdir" ]; then \
                cp -r "$VFA/$subdir" "$FA/$subdir" && \
                echo "Copied $subdir from vllm_flash_attn to flash_attn"; \
            fi; \
        done; \
        for pyfile in flash_attn_interface.py; do \
            if [ ! -e "$FA/$pyfile" ] && [ -f "$VFA/$pyfile" ]; then \
                cp "$VFA/$pyfile" "$FA/$pyfile" && \
                echo "Copied $pyfile from vllm_flash_attn to flash_attn"; \
            fi; \
        done; \
    elif [ ! -e "$FA" ] && [ -d "$VFA" ]; then \
        ln -s "$VFA" "$FA" && \
        echo "Created flash_attn symlink (flash-attn-4 not installed)"; \
    else \
        echo "flash_attn shim: nothing to do"; \
    fi

# Build metadata (generated by build.sh)
COPY build-metadata.yaml /workspace/build-metadata.yaml

# Default entrypoint: apply mods then exec into vllm
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh
ENTRYPOINT ["/workspace/entrypoint.sh"]
