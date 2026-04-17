#!/bin/bash
# build.sh — Build NemoClaw-Thor vLLM container for Jetson AGX Thor (sm110)
#
# Orchestrates a multi-stage Docker build:
#   Phase 1: FlashInfer wheels
#   Phase 2: vLLM wheels
#   Phase 3: Final runner image
#
# All wheels are cached in ./wheels/ so incremental rebuilds skip
# already-built components.
#
# Usage:
#   ./build.sh                              # defaults: latest main, 8 jobs
#   ./build.sh --vllm-ref v0.8.5            # pin vLLM to a tag/branch/SHA
#   ./build.sh --flashinfer-ref v0.6.1      # pin FlashInfer
#   ./build.sh --build-jobs 6               # limit parallelism
#   ./build.sh --apply-vllm-pr 12345        # cherry-pick a PR onto vLLM
#   ./build.sh --tf5                        # use transformers >= 5
#   ./build.sh --skip-flashinfer            # reuse existing FlashInfer wheels
#   ./build.sh --skip-vllm                  # reuse existing vLLM wheels
#   ./build.sh --image-name my-vllm:latest  # custom image tag

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ────────────────────────────────────────────────────────
# Pinned to the validated v6 build (2026-04-17). Override with --vllm-ref / etc.
# Reset to "main" when starting a new development stint.
VLLM_REF="9965f501a89204769a53c86cdee2528947373747"
FLASHINFER_REF="25b324dbad53942a695a1f00cd7837800de25634"
BUILD_JOBS=8
CUDA_BASE="nvidia/cuda:13.0.0-devel-ubuntu24.04"
TORCH_CUDA_ARCH_LIST="11.0a"
FLASHINFER_CUDA_ARCH_LIST="11.0a"
IMAGE_NAME="nemoclaw-thor/vllm"
IMAGE_TAG=""  # auto-generated if empty
# PR #39931 adds TQFullAttentionSpec. Only new-file hunks apply cleanly; the
# in-place edits are replayed by docker/mods/fix-pr39931-turboquant at runtime.
VLLM_PRS="39931"
PRE_TRANSFORMERS=0
SKIP_FLASHINFER=0
SKIP_VLLM=0
REBUILD_FLASHINFER=0
REBUILD_VLLM=0

# ── Parse arguments ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vllm-ref)           VLLM_REF="$2";              shift 2 ;;
        --flashinfer-ref)     FLASHINFER_REF="$2";         shift 2 ;;
        --build-jobs)         BUILD_JOBS="$2";             shift 2 ;;
        --cuda-base)          CUDA_BASE="$2";              shift 2 ;;
        --gpu-arch)           TORCH_CUDA_ARCH_LIST="$2"
                              FLASHINFER_CUDA_ARCH_LIST="$2"; shift 2 ;;
        --image-name)         IMAGE_NAME="$2";             shift 2 ;;
        --image-tag)          IMAGE_TAG="$2";              shift 2 ;;
        --apply-vllm-pr)      VLLM_PRS="${VLLM_PRS:+$VLLM_PRS }$2"; shift 2 ;;
        --tf5)                PRE_TRANSFORMERS=1;          shift ;;
        --skip-flashinfer)    SKIP_FLASHINFER=1;           shift ;;
        --skip-vllm)          SKIP_VLLM=1;                 shift ;;
        --rebuild-flashinfer) REBUILD_FLASHINFER=1;        shift ;;
        --rebuild-vllm)       REBUILD_VLLM=1;              shift ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1 ;;
    esac
done

# ── Directories ─────────────────────────────────────────────────────
WHEELS_DIR="$SCRIPT_DIR/wheels"
mkdir -p "$WHEELS_DIR"

# ── Cache busters ───────────────────────────────────────────────────
# Change these to force a re-clone in the Docker build cache
CACHEBUST_FLASHINFER=$(date +%s)
CACHEBUST_VLLM=$(date +%s)

# ── Phase 1: FlashInfer ─────────────────────────────────────────────
if [ "$SKIP_FLASHINFER" = "0" ]; then
    echo ""
    echo "=========================================="
    echo "  Phase 1: Building FlashInfer wheels"
    echo "  Ref: ${FLASHINFER_REF}"
    echo "  Arch: ${FLASHINFER_CUDA_ARCH_LIST}"
    echo "=========================================="
    echo ""

    FLASHINFER_ARGS=(
        --build-arg "BUILD_JOBS=${BUILD_JOBS}"
        --build-arg "CUDA_BASE=${CUDA_BASE}"
        --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
        --build-arg "FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}"
        --build-arg "FLASHINFER_REF=${FLASHINFER_REF}"
    )

    if [ "$REBUILD_FLASHINFER" = "1" ]; then
        FLASHINFER_ARGS+=(--build-arg "CACHEBUST_FLASHINFER=${CACHEBUST_FLASHINFER}")
    fi

    docker build --network host \
        "${FLASHINFER_ARGS[@]}" \
        --target flashinfer-export \
        --output "type=local,dest=${WHEELS_DIR}" \
        -f Dockerfile .

    echo ""
    echo "FlashInfer wheels:"
    ls -lh "$WHEELS_DIR"/*.whl 2>/dev/null | grep -i flashinfer || echo "  (none found)"
    echo ""
else
    echo "Skipping FlashInfer build (--skip-flashinfer)"
    if ! ls "$WHEELS_DIR"/*flashinfer*.whl &>/dev/null; then
        echo "ERROR: No FlashInfer wheels found in ${WHEELS_DIR}/" >&2
        exit 1
    fi
fi

# ── Phase 2: vLLM ───────────────────────────────────────────────────
if [ "$SKIP_VLLM" = "0" ]; then
    echo ""
    echo "=========================================="
    echo "  Phase 2: Building vLLM wheel"
    echo "  Ref: ${VLLM_REF}"
    echo "  Arch: ${TORCH_CUDA_ARCH_LIST}"
    [ -n "$VLLM_PRS" ] && echo "  PRs: ${VLLM_PRS}"
    echo "=========================================="
    echo ""

    VLLM_ARGS=(
        --build-arg "BUILD_JOBS=${BUILD_JOBS}"
        --build-arg "CUDA_BASE=${CUDA_BASE}"
        --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
        --build-arg "VLLM_REF=${VLLM_REF}"
        --build-arg "VLLM_PRS=${VLLM_PRS}"
    )

    if [ "$REBUILD_VLLM" = "1" ]; then
        VLLM_ARGS+=(--build-arg "CACHEBUST_VLLM=${CACHEBUST_VLLM}")
    fi

    docker build --network host \
        "${VLLM_ARGS[@]}" \
        --target vllm-export \
        --output "type=local,dest=${WHEELS_DIR}" \
        -f Dockerfile .

    echo ""
    echo "vLLM wheel:"
    ls -lh "$WHEELS_DIR"/*.whl 2>/dev/null | grep -i vllm || echo "  (none found)"
    echo ""
else
    echo "Skipping vLLM build (--skip-vllm)"
    if ! ls "$WHEELS_DIR"/vllm*.whl &>/dev/null; then
        echo "ERROR: No vLLM wheel found in ${WHEELS_DIR}/" >&2
        exit 1
    fi
fi

# ── Build metadata ──────────────────────────────────────────────────
VLLM_COMMIT="unknown"
FLASHINFER_COMMIT="unknown"
[ -f "$WHEELS_DIR/.vllm-commit" ] && VLLM_COMMIT=$(cat "$WHEELS_DIR/.vllm-commit")
[ -f "$WHEELS_DIR/.flashinfer-commit" ] && FLASHINFER_COMMIT=$(cat "$WHEELS_DIR/.flashinfer-commit")

# Auto-generate image tag from vLLM commit
if [ -z "$IMAGE_TAG" ]; then
    VLLM_SHORT="${VLLM_COMMIT:0:9}"
    IMAGE_TAG="${VLLM_REF}-g${VLLM_SHORT}-thor-sm110-cu132"
    # Sanitize tag (replace / with -)
    IMAGE_TAG="${IMAGE_TAG//\//-}"
fi

cat > "$SCRIPT_DIR/build-metadata.yaml" <<EOF
build_date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
script_commit: $(cd "$SCRIPT_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")
vllm_ref: ${VLLM_REF}
vllm_commit: ${VLLM_COMMIT}
flashinfer_ref: ${FLASHINFER_REF}
flashinfer_commit: ${FLASHINFER_COMMIT}
gpu_arch: ${TORCH_CUDA_ARCH_LIST}
cuda_base: ${CUDA_BASE}
platform: jetson-thor-sm110
build_args:
  build_jobs: ${BUILD_JOBS}
  pre_transformers: ${PRE_TRANSFORMERS}
  vllm_prs: "${VLLM_PRS}"
EOF

# ── Phase 3: Runner ─────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Phase 3: Building runner image"
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "=========================================="
echo ""

docker build --network host \
    --build-arg "BUILD_JOBS=${BUILD_JOBS}" \
    --build-arg "CUDA_BASE=${CUDA_BASE}" \
    --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
    --build-arg "FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}" \
    --build-arg "PRE_TRANSFORMERS=${PRE_TRANSFORMERS}" \
    --target runner \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -t "${IMAGE_NAME}:latest" \
    -f Dockerfile .

echo ""
echo "=========================================="
echo "  Build complete!"
echo "=========================================="
echo "  Image:             ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Also tagged:       ${IMAGE_NAME}:latest"
echo "  vLLM ref:          ${VLLM_REF} (${VLLM_COMMIT:0:9})"
echo "  FlashInfer ref:    ${FLASHINFER_REF} (${FLASHINFER_COMMIT:0:9})"
echo "  GPU arch:          ${TORCH_CUDA_ARCH_LIST}"
echo "  CUDA base:         ${CUDA_BASE}"
echo "  Transformers >= 5: $([ "$PRE_TRANSFORMERS" = "1" ] && echo "yes" || echo "no")"
echo ""
echo "To use with NemoClaw-Thor:"
echo "  export THOR_VLLM_IMAGE=${IMAGE_NAME}:${IMAGE_TAG}"
echo "  ./start-model.sh qwen3.5-35b-a3b-fp8"
echo ""
