#!/bin/bash
# build-llamacpp.sh — Build NemoClaw-Thor llama.cpp container for Thor (sm_110a)
#
# Companion to build-vllm.sh (vLLM image) and build-trt.sh (TRT-Edge-LLM image).
#
# Two-stage Docker build of docker/Dockerfile.llamacpp. Produces a small
# (~1 GB) standalone image for the GGUF serving lane. Does NOT inherit from
# the vLLM image — llama.cpp has zero Python deps, just C++ + CUDA, so the
# runtime stage is just CUDA runtime + the llama-server binary.
#
# Why this exists:
#   The GGUF-only lane (unsloth *-MTP-GGUF, custom mixed quants, etc.) that
#   vLLM and TRT-Edge-LLM can't serve cleanly. See Dockerfile.llamacpp
#   header for the full rationale.
#
# What this script does:
#   1. Validates LLAMA_CPP_REF is recent enough for MTP support (merged
#      2026-05-16). Warns if the ref is older.
#   2. Runs `docker build -f Dockerfile.llamacpp` with the right tags.
#   3. Tags both `:latest` and a version-specific tag for rollback.
#
# Apt cache mounts (id=apt-cache-thor) are shared with build-vllm.sh and
# build-trt.sh, so package downloads are reused across all three builds.
#
# Usage:
#   ./build-llamacpp.sh                              # defaults: master (latest)
#   ./build-llamacpp.sh --llama-ref b6800            # pin to a specific build tag
#   ./build-llamacpp.sh --image-name my-llama:custom # custom tag
#   ./build-llamacpp.sh --no-cache                   # force fresh build
#   ./build-llamacpp.sh --rebuild-llama              # force re-clone of llama.cpp
#
# When upstream version-bumps:
#   No patch files yet — llama.cpp builds clean for sm_110a from upstream.
#   If a future build breaks, document the symptom + fix in NOTES.md and
#   consider adding a patches/llama_cpp_<ref>_thor.patch following the
#   TRT-Edge-LLM pattern.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# BuildKit step log limits — llama.cpp build is much smaller than vLLM/TRT
# but cmake + CUDA compile output can still hit the default 2 MiB cap.
export BUILDKIT_STEP_LOG_MAX_SIZE=${BUILDKIT_STEP_LOG_MAX_SIZE:-104857600}    # 100 MiB
export BUILDKIT_STEP_LOG_MAX_SPEED=${BUILDKIT_STEP_LOG_MAX_SPEED:-10485760}   # 10 MiB/s

# ── Defaults ────────────────────────────────────────────────────────
# MTP support merged 2026-05-16 (--spec-type draft-mtp). Pin to a known-good
# tag once available; default to master for now since llama.cpp uses rolling
# bNNNN release tags rather than semver.
LLAMA_CPP_REF="master"
IMAGE_NAME="nemoclaw-thor/llama-cpp"
IMAGE_TAG=""  # auto-generated from LLAMA_CPP_REF if empty
CUDA_BASE_BUILD="nvidia/cuda:13.0.3-devel-ubuntu24.04"
CUDA_BASE_RUNTIME="nvidia/cuda:13.0.3-runtime-ubuntu24.04"
CUDA_ARCH="110a"  # Jetson Thor
NO_CACHE=0
REBUILD_LLAMA=0

# Date threshold for MTP support warning
MTP_MERGE_DATE="2026-05-16"

# ── Parse arguments ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --llama-ref)        LLAMA_CPP_REF="$2";    shift 2 ;;
        --cuda-base-build)  CUDA_BASE_BUILD="$2";  shift 2 ;;
        --cuda-base-runtime) CUDA_BASE_RUNTIME="$2"; shift 2 ;;
        --cuda-arch)        CUDA_ARCH="$2";         shift 2 ;;
        --image-name)       IMAGE_NAME="$2";        shift 2 ;;
        --image-tag)        IMAGE_TAG="$2";         shift 2 ;;
        --no-cache)         NO_CACHE=1;             shift ;;
        --rebuild-llama)    REBUILD_LLAMA=1;        shift ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1 ;;
    esac
done

# Auto-generate image tag from the llama.cpp ref.
#   master           → master-thor-sm110a
#   b6800            → b6800-thor-sm110a
#   release/X.Y.Z    → X.Y.Z-thor-sm110a
if [ -z "$IMAGE_TAG" ]; then
    TAG_SLUG="${LLAMA_CPP_REF#release/}"
    TAG_SLUG="${TAG_SLUG//\//-}"
    IMAGE_TAG="${TAG_SLUG}-thor-sm${CUDA_ARCH}"
fi

# ── Sanity warning: ref might predate MTP ───────────────────────────
if [ "$LLAMA_CPP_REF" != "master" ]; then
    cat <<EOF
NOTE: You pinned LLAMA_CPP_REF=${LLAMA_CPP_REF}.
MTP speculative decoding (--spec-type draft-mtp) was merged ${MTP_MERGE_DATE}.
If your pinned ref predates that, MTP-fused GGUFs like unsloth/Qwen3.6-27B-MTP-GGUF
will load but MTP-side speculation won't engage.
Verify with: docker run --rm <image> --help | grep -i 'spec-type'

EOF
fi

# ── Build args ──────────────────────────────────────────────────────
BUILD_ARGS=(
    --build-arg "LLAMA_CPP_REF=${LLAMA_CPP_REF}"
    --build-arg "CUDA_BASE_BUILD=${CUDA_BASE_BUILD}"
    --build-arg "CUDA_BASE_RUNTIME=${CUDA_BASE_RUNTIME}"
    --build-arg "CUDA_ARCH=${CUDA_ARCH}"
)

if [ "$REBUILD_LLAMA" = "1" ]; then
    BUILD_ARGS+=(--build-arg "CACHEBUST_LLAMA=$(date +%s)")
fi

DOCKER_BUILD_FLAGS=(--network host)
if [ "$NO_CACHE" = "1" ]; then
    DOCKER_BUILD_FLAGS+=(--no-cache)
fi

# ── Build ────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Building llama.cpp image"
echo "=========================================="
echo "  llama.cpp ref:    ${LLAMA_CPP_REF}"
echo "  CUDA arch:        sm_${CUDA_ARCH}"
echo "  CUDA base (build):   ${CUDA_BASE_BUILD}"
echo "  CUDA base (runtime): ${CUDA_BASE_RUNTIME}"
echo "  Image name+tag:   ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Also tagging:     ${IMAGE_NAME}:latest"
echo ""

DOCKER_BUILDKIT=1 docker build \
    "${DOCKER_BUILD_FLAGS[@]}" \
    "${BUILD_ARGS[@]}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -t "${IMAGE_NAME}:latest" \
    -f Dockerfile.llamacpp \
    .

echo ""
echo "=========================================="
echo "  Build complete"
echo "=========================================="
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

# Show the committed git ref for reproducibility
COMMIT=$(docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" cat /etc/llama-cpp-commit 2>/dev/null || echo "unknown")
TAG=$(docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" cat /etc/llama-cpp-tag 2>/dev/null || echo "unknown")
echo "  Committed llama.cpp commit: ${COMMIT}"
echo "  Committed llama.cpp tag:    ${TAG}"
echo ""

# Show available size
docker image ls "${IMAGE_NAME}:${IMAGE_TAG}" --format "  Image size:   {{.Size}}"
echo ""

echo "Quick smoke test (downloads a tiny model + generates a few tokens):"
echo "  docker run --rm --runtime nvidia --network host \\"
echo "    -v ~/.cache/huggingface:/root/.cache/huggingface \\"
echo "    ${IMAGE_NAME}:${IMAGE_TAG} \\"
echo "    -hf ggml-org/gemma-3-270m-GGUF --port 8000 --n-gpu-layers 99"
echo ""
echo "Or test sm_110a kernel compilation succeeded:"
echo "  docker run --rm --runtime nvidia ${IMAGE_NAME}:${IMAGE_TAG} --version"
echo ""
