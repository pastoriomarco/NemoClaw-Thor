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
ARG CUDA_BASE=nvidia/cuda:13.2.0-devel-ubuntu24.04

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

# PyTorch nightly (cu132 — for cuBLAS FP8 SM110 fixes)
# torch+triton from cu132; torchvision+torchaudio from cu130 (no cu132 ARM64 wheels)
ENV UV_HTTP_TIMEOUT=300
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
     uv pip install torch triton \
         --index-url https://download.pytorch.org/whl/nightly/cu132 && \
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
    sed -i '/^triton\b/d' requirements/test.txt && \
    sed -i '/^fastsafetensors\b/d' requirements/test.txt && \
    uv pip install -r requirements/build.txt

# Apply build-time patches (conditional — skip if already fixed upstream)
# Copy entire patches dir; vllm_*.patch files are optional
COPY patches/ /tmp/patches/
RUN for p in /tmp/patches/vllm_*.patch; do \
        [ -f "$p" ] || continue; \
        echo "Applying $(basename $p)..."; \
        patch -p1 < "$p" || echo "Patch $(basename $p) not applicable, skipping"; \
    done

# SM110 (Thor) CUTLASS arch fix:
# Step 1: Strip 11.0f from all SM100 CUTLASS arch lists (SCALED_MM, MLA, MoE, etc.)
#         CUTLASS sm100 kernels use enable_sm100f_only which rejects SM110 at runtime
#         with illegal-instruction crashes. The targeted patch above handles the known
#         blocks; this sed catches any remaining occurrences in newer vLLM versions.
# Step 2: Restore 11.0f for two specific arch groups:
#   - FP4_ARCHS: NVFP4 quantization kernels — safe on SM110, no enable_sm100f_only.
#     PR #38126 (merged 2026-03-27) fixes cuda_archs_loose_intersection cross-suffix
#     matching so "11.0f" correctly produces SM110a kernels with CUDA_ARCH=11.0a.
#   - MoE SCALED_MM_ARCHS (controls grouped_mm_c3x_sm100.cu compilation):
#     vLLM reuses SCALED_MM_ARCHS for both non-MoE scaled_mm and MoE grouped_mm.
#     The non-MoE SCALED_MM_ARCHS stays stripped (enable_sm100f_only crashes SM110).
#     The MoE SCALED_MM_ARCHS must include 11.0f so grouped_mm_c3x_sm100.cu is
#     compiled and its symbol exists; vLLM dev209+ references it unconditionally
#     at _C.abi3.so load time. Kernel is never dispatched — FlashInfer MoE is used
#     (VLLM_USE_FLASHINFER_MOE_FP4=1 + VLLM_FLASHINFER_MOE_BACKEND=throughput).
RUN echo "Step 1: Strip 11.0f from all SM100 CUTLASS arch lists..." && \
    sed -i 's/;11\.0f//g; s/11\.0f;//g; s/"11\.0f"/""/g' CMakeLists.txt && \
    echo "  Stripped. Remaining 11.0f count: $(grep -c '11\.0f' CMakeLists.txt || true)"

RUN python3 - <<'CMAKEPYEOF'
import pathlib

text = pathlib.Path("CMakeLists.txt").read_text()

# Restore 11.0f in FP4_ARCHS (NVFP4 quant kernels — safe on SM110, no enable_sm100f_only)
n_fp4 = text.count('cuda_archs_loose_intersection(FP4_ARCHS "10.0f"')
text = text.replace(
    'cuda_archs_loose_intersection(FP4_ARCHS "10.0f"',
    'cuda_archs_loose_intersection(FP4_ARCHS "10.0f;11.0f"')
print(f"PATCHED {n_fp4} FP4_ARCHS occurrences with 11.0f")

# Restore 11.0f in ALL SCALED_MM_ARCHS "10.0f" occurrences:
#   - Non-MoE SM100 (scaled_mm_c3x_sm100.cu): safe — disabled via VLLM_DISABLED_KERNELS
#   - MoE SM100 (grouped_mm_c3x_sm100.cu): REQUIRED — symbol referenced unconditionally
#     at _C.abi3.so load time by vLLM dev209+; kernel never dispatched (FlashInfer MoE used)
# NOTE: the backward-search heuristic fails because the else-branch "10.0a;10.1a;10.3a"
# is closer to the 12.8 guard than the if-branch "10.0f", causing a premature break.
# Direct str.replace is simpler and correct.
n_scaled = text.count('cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f"')
text = text.replace(
    'cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f"',
    'cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f"')
print(f"PATCHED {n_scaled} SCALED_MM_ARCHS occurrences with 11.0f")

# Restore 11.0f in CUTLASS_MOE_DATA_ARCHS (moe_data.cu — symbols referenced unconditionally
# at _C.abi3.so load time: get_cutlass_moe_mm_data_caller, get_cutlass_batched_moe_mm_data_caller,
# get_cutlass_moe_mm_problem_sizes_from_expert_offsets_caller).
# Kernel never dispatched on SM110 — FlashInfer MoE used (VLLM_USE_FLASHINFER_MOE_FP4=1).
n_moe_data = text.count('cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;12.0f"')
text = text.replace(
    'cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;12.0f"',
    'cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;11.0f;12.0f"')
print(f"PATCHED {n_moe_data} CUTLASS_MOE_DATA_ARCHS occurrences with 11.0f")

pathlib.Path("CMakeLists.txt").write_text(text)
print(f"Final 11.0f count: {text.count('11.0f')}")
CMAKEPYEOF

# SM110 MoE kernel selection: add is_device_capability_family(110)/(120) checks so
# NVFP4 CUTLASS MoE kernels are preferred over Marlin on SM110.
# flashinfer_cutlass_moe.py: fixed upstream in PR #33417 (no-op if already present).
# flashinfer_trtllm_moe.py: still unfixed upstream as of vLLM 0.18.x.
RUN python3 - <<'PYEOF'
import pathlib

PATCHES = [
    ("vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py", [
        (
            "p.is_device_capability_family(100)\n        )\n        and has_flashinfer_cutlass_fused_moe",
            "p.is_device_capability_family(100)\n            or p.is_device_capability_family(110)\n            or p.is_device_capability_family(120)\n        )\n        and has_flashinfer_cutlass_fused_moe",
        ),
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

# PyTorch (cu132 — for cuBLAS FP8 SM110 fixes)
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
     uv pip install torch triton \
         --index-url https://download.pytorch.org/whl/nightly/cu132 && \
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
# transformers<5 constraint, then explicitly install transformers==5.4.0.
# transformers 5.2.0: first native Qwen3.5 support.
# transformers 5.3.0: NVFP4 Blackwell quantization integration, Qwen3.5 bug fixes.
# transformers 5.4.0: Qwen3.5 classification, FP8 grouped matmul speedup.
RUN --mount=type=bind,source=wheels,target=/workspace/wheels \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    if [ "$PRE_TRANSFORMERS" = "1" ]; then \
        echo "transformers>=5.0.0" > /tmp/tf-override.txt && \
        uv pip install $(ls /workspace/wheels/*.whl | grep -v flashinfer_cubin) --override /tmp/tf-override.txt && \
        uv pip install "transformers==5.4.0" --override /tmp/tf-override.txt; \
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

# No Ray — single-node only. Install fastsafetensors for fast model loading.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install fastsafetensors

# Build metadata (generated by build.sh)
COPY build-metadata.yaml /workspace/build-metadata.yaml

# Default entrypoint: apply mods then exec into vllm
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh
ENTRYPOINT ["/workspace/entrypoint.sh"]
