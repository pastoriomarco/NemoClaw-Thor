#!/bin/bash
# build-trt.sh — Build NemoClaw-Thor TensorRT-Edge-LLM container for Thor (sm110)
#
# Companion to build-vllm.sh (vLLM image) and bundle.sh (vLLM production bundle).
#
# Single-stage Docker build of docker/Dockerfile.trt. Produces a standalone
# image (~22 GB) for production edge deployment. Does NOT inherit from the
# vLLM image — TRT-Edge-LLM has its own Python venv at /opt/trt-venv with
# torch 2.10.0+cu130 and the prebuilt CuTe DSL kernels for SM110a.
#
# What this script does:
#   1. Validates the prebuilt CuTe DSL tarball exists in the upstream repo
#      (kernelSrcs/cuteDSLPrebuilt/cutedsl_aarch64_sm_110_cuda13.tar.gz —
#       baked into TRT-Edge-LLM v0.7.0; if missing, NVIDIA's release is broken).
#   2. Runs `docker build -f Dockerfile.trt` with the right tags.
#   3. Tags both `:latest` and a version-specific tag for rollback.
#
# Apt cache mounts (id=apt-cache-thor) and pip cache mounts (id=trt-pip)
# are shared with build-vllm.sh, so package downloads are reused across
# both image builds.
#
# Usage:
#   ./build-trt.sh                                   # defaults: TRT-Edge-LLM release/0.7.0
#   ./build-trt.sh --trt-ref release/0.8.0          # version bump
#   ./build-trt.sh --image-name my-trt:custom        # custom tag
#   ./build-trt.sh --no-cache                        # force fresh build (skip BuildKit cache)
#   ./build-trt.sh --rebuild-trt                     # force re-clone of TRT-Edge-LLM
#
# When upstream version-bumps:
#   - Patch file at patches/trt_edge_llm_v0.7.0_thor.patch must apply.
#     If `git apply --3way` succeeds, you're fine.
#     If it fails, .rej files in /workspace/trt-edge-llm/ show what
#     context shifted. Either fix the patch or upstream a PR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Raise BuildKit step log limit (libnvinfer 2.1 GB download + cmake/ninja
# floods can otherwise truncate with "[output clipped, log limit 2MiB]").
export BUILDKIT_STEP_LOG_MAX_SIZE=${BUILDKIT_STEP_LOG_MAX_SIZE:-104857600}    # 100 MiB
export BUILDKIT_STEP_LOG_MAX_SPEED=${BUILDKIT_STEP_LOG_MAX_SPEED:-10485760}   # 10 MiB/s

# ── Defaults ────────────────────────────────────────────────────────
# v0.7.0 is the first release with full Thor (sm110) support via the
# `EMBEDDED_TARGET=jetson-thor` toolchain. Lower versions don't ship the
# prebuilt CuTe DSL kernels for SM110.
TRT_REF="release/0.7.0"
IMAGE_NAME="nemoclaw-thor/trt-edge-llm"
IMAGE_TAG=""  # auto-generated from TRT_REF if empty
CUDA_BASE="nvidia/cuda:13.0.3-devel-ubuntu24.04"
NO_CACHE=0
REBUILD_TRT=0

# ── Parse arguments ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --trt-ref)     TRT_REF="$2";        shift 2 ;;
        --cuda-base)   CUDA_BASE="$2";      shift 2 ;;
        --image-name)  IMAGE_NAME="$2";     shift 2 ;;
        --image-tag)   IMAGE_TAG="$2";      shift 2 ;;
        --no-cache)    NO_CACHE=1;          shift ;;
        --rebuild-trt) REBUILD_TRT=1;       shift ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1 ;;
    esac
done

# Auto-generate image tag from the TRT ref. e.g. release/0.7.0 → 0.7.0-thor-sm110
if [ -z "$IMAGE_TAG" ]; then
    TAG_SLUG="${TRT_REF#release/}"
    TAG_SLUG="${TAG_SLUG//\//-}"
    IMAGE_TAG="${TAG_SLUG}-thor-sm110"
fi

# ── Build args ──────────────────────────────────────────────────────
BUILD_ARGS=(
    --build-arg "TRT_EDGE_LLM_REF=${TRT_REF}"
    --build-arg "CUDA_BASE=${CUDA_BASE}"
)

if [ "$REBUILD_TRT" = "1" ]; then
    BUILD_ARGS+=(--build-arg "CACHEBUST_TRT=$(date +%s)")
fi

DOCKER_BUILD_FLAGS=(--network host)
if [ "$NO_CACHE" = "1" ]; then
    DOCKER_BUILD_FLAGS+=(--no-cache)
fi

# ── Build ────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Building TensorRT-Edge-LLM image"
echo "  TRT ref:     ${TRT_REF}"
echo "  CUDA base:   ${CUDA_BASE}"
echo "  Image:       ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Also tagged: ${IMAGE_NAME}:latest"
echo "=========================================="
echo ""

docker build "${DOCKER_BUILD_FLAGS[@]}" \
    "${BUILD_ARGS[@]}" \
    -f Dockerfile.trt \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -t "${IMAGE_NAME}:latest" \
    .

echo ""
echo "=========================================="
echo "  Build complete!"
echo "=========================================="
echo "  Image:        ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Also tagged:  ${IMAGE_NAME}:latest"
echo "  TRT ref:      ${TRT_REF}"
echo ""
echo "Activate at runtime in a container started with --runtime nvidia."
echo "Example smoke test:"
echo "  docker run --rm --runtime nvidia --network host \\"
echo "      -v \${HOME}/thor-hf-cache:/root/.cache/huggingface \\"
echo "      --entrypoint bash ${IMAGE_NAME}:latest \\"
echo "      -c '/opt/trt-venv/bin/python -c \"from tensorrt_edgellm import _edgellm_runtime; print(\\\"OK\\\")\"'"
echo ""
