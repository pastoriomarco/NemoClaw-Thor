# syntax=docker/dockerfile:1.6
#
# NemoClaw-Thor vLLM Container
# Builds vLLM + FlashInfer from source for Jetson AGX Thor (sm110 / Blackwell).
#
# Adapted from spark-vllm-docker (sm121) — stripped of Ray/cluster logic,
# retargeted for single-node Thor with CUDA 13.0 and JetPack 7.1.
#
# Usage:
#   ./build.sh                       # default: latest vLLM main
#   ./build.sh --vllm-ref v0.8.5     # pin to a tag
#   ./build.sh --build-jobs 6        # lower parallelism on constrained host

# ── Build tunables ──────────────────────────────────────────────────
ARG BUILD_JOBS=8
ARG CUDA_BASE=nvidia/cuda:13.0.0-devel-ubuntu24.04

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

# PyTorch nightly (cu130 — matches host CUDA 13.0 driver, avoids PTX version mismatch)
ENV UV_HTTP_TIMEOUT=300
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
     uv pip install torch triton \
         --index-url https://download.pytorch.org/whl/nightly/cu130 && \
     uv pip install torchvision torchaudio \
         --index-url https://download.pytorch.org/whl/nightly/cu130 && \
     uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2" filelock pynvml requests tqdm

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

# Avoid re-downloading existing cubins
COPY patches/flashinfer_cache.patch .
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=cubins-cache,target=/workspace/flashinfer/flashinfer-cubin/flashinfer_cubin/cubins \
    patch -p1 < flashinfer_cache.patch && \
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
            curl -fL "https://github.com/vllm-project/vllm/pull/${pr}.diff" | git apply -v; \
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

# SM110 (Thor) CUTLASS arch fix for vLLM 0.19:
#
# vLLM 0.19 CMakeLists.txt natively includes "11.0f" in FP4_ARCHS, SCALED_MM_ARCHS,
# MLA_ARCHS, and MoE arch lists when CUDA >= 13.0. This is a major improvement over
# 0.18 where we had to strip-and-restore 11.0f manually.
#
# Remaining issue: CUTLASS sm100 scaled_mm kernels (non-MoE) still use
# enable_sm100f_only which rejects SM110 at runtime. The runtime fallback mod
# (fix-sm110-cutlass) handles this by routing to PyTorch native _scaled_mm.
#
# Verify the upstream arch lists are intact (no stripping needed):
RUN echo "Verifying SM110 arch inclusion in CMakeLists.txt..." && \
    echo "  FP4_ARCHS 11.0f count: $(grep -c 'FP4_ARCHS.*11\.0f' CMakeLists.txt || echo 0)" && \
    echo "  SCALED_MM_ARCHS 11.0f count: $(grep -c 'SCALED_MM_ARCHS.*11\.0f' CMakeLists.txt || echo 0)" && \
    echo "  MLA_ARCHS 11.0f count: $(grep -c 'MLA_ARCHS.*11\.0f' CMakeLists.txt || echo 0)" && \
    echo "  Total 11.0f: $(grep -c '11\.0f' CMakeLists.txt || echo 0)"

# SM110 MoE kernel selection:
# flashinfer_cutlass_moe.py: FIXED upstream in 0.19 — includes SM110 check.
# flashinfer_trtllm_moe.py: still only checks family(100) as of 0.19.0.
RUN python3 - <<'PYEOF'
import pathlib

PATCHES = [
    ("vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py", [
        # Verify upstream fix is present (no-op)
    ]),
    ("vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py", [
        (
            "p.is_cuda() and p.is_device_capability_family(100)",
            "p.is_cuda() and (p.is_device_capability_family(100) or p.is_device_capability_family(110) or p.is_device_capability_family(120))",
        ),
    ]),
]

for fname, replacements in PATCHES:
    p = pathlib.Path(fname)
    if not p.exists():
        print(f"SKIP {fname}: not found")
        continue
    text = p.read_text()
    if not replacements:
        if "is_device_capability_family(110)" in text:
            print(f"VERIFIED {fname}: SM110 support present upstream")
        else:
            print(f"WARNING {fname}: SM110 support NOT found upstream — manual patch may be needed")
        continue
    changed = False
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            changed = True
    if changed:
        p.write_text(text)
        print(f"PATCHED {fname}")
    elif "is_device_capability_family(110)" in text:
        print(f"ALREADY PATCHED {fname}")
    else:
        print(f"PATTERN NOT FOUND in {fname} (check vLLM version)")
PYEOF

# NOTE: FA2 SM110 patch removed — confirmed Xid 43 (GPU hang) at runtime.
# FA2 kernels use SM80-specific shared memory patterns that cause hardware
# faults on SM110 even when compiled with native cubins. FA4 handles SM110
# but can't do head_dim=256 (TMEM limit). FlashInfer is the only viable
# attention backend for Qwen3.5 on Thor.

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

ARG BUILD_JOBS=8
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

# PyTorch (cu130 — matches host CUDA 13.0 driver)
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
     uv pip install torch triton \
         --index-url https://download.pytorch.org/whl/nightly/cu130 && \
     uv pip install torchvision torchaudio \
         --index-url https://download.pytorch.org/whl/nightly/cu130 && \
     uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

# Install FlashInfer + vLLM wheels (bind-mounted from host ./wheels/)
# Exclude flashinfer_cubin wheel: precompiled cubins are downloaded from upstream
# and target sm100f — they crash on sm110 (Thor).
# INCLUDE flashinfer_jit_cache: JIT-compiled .so files are built with
# FLASHINFER_CUDA_ARCH_LIST=11.0a and contain valid sm_110a code.
# This enables the FlashInfer attention backend on Thor.
# With --tf5 (PRE_TRANSFORMERS=1, the default): override vLLM's and compressed-tensors'
# transformers<5 constraint, then explicitly install transformers==5.5.0.
# transformers 5.2.0: first native Qwen3.5 support.
# transformers 5.3.0: NVFP4 Blackwell quantization integration, Qwen3.5 bug fixes.
# transformers 5.4.0: Qwen3.5 classification, FP8 grouped matmul speedup.
# transformers 5.5.0: Gemma 4 model class (Gemma4ForConditionalGeneration),
#   Gemma4Config, Gemma4VisionConfig, Gemma4AudioConfig, Gemma4Processor.
RUN --mount=type=bind,source=wheels,target=/workspace/wheels \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    if [ "$PRE_TRANSFORMERS" = "1" ]; then \
        echo "transformers>=5.0.0" > /tmp/tf-override.txt && \
        uv pip install $(ls /workspace/wheels/*.whl | grep -v flashinfer_cubin) --override /tmp/tf-override.txt && \
        uv pip install "transformers==5.5.0" --override /tmp/tf-override.txt; \
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
# - flash-attn-4: FA4 with SM110a/SM110f support (JIT via CuTeDSL, beta9)
#   Enables --attention-backend flash_attn for DFlash on SM110 (head_dim=256).
#   Requires nvidia-cutlass-dsl >= 4.4.x
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install fastsafetensors instanttensor && \
    uv pip install "triattention @ git+https://github.com/WeianMao/triattention.git" && \
    uv pip install "flash-attn-4>=4.0.0.beta9" "nvidia-cutlass-dsl>=4.4"

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
