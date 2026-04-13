#!/usr/bin/env bash
# bundle.sh — Build NemoClaw-Thor bundled image (vLLM + baked JIT caches)
#
# Creates a self-contained image that starts serving in ~4-6 min instead of
# ~50-60 min, by baking in pre-built FlashInfer and Torch AOT kernel caches.
#
# Usage:
#   ./bundle.sh                              # build locally
#   ./bundle.sh --push ghcr.io/ORG/REPO     # build and push to registry
#   ./bundle.sh --base my-vllm:custom        # use a different base image
#   ./bundle.sh --tag my-org/my-image:v1     # custom output tag
#
# Prerequisites:
#   - nemoclaw-thor/vllm:main-g58a249bc6-thor-sm110-cu132 must exist locally
#   - ~/thor-flashinfer-cache must be populated (run the model at least once)
#   - ~/thor-vllm-cache must be populated (run the model at least once)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────
BASE_IMAGE="nemoclaw-thor/vllm:main-g58a249bc6-thor-sm110-cu132"
BUNDLE_TAG="nemoclaw-thor/vllm:main-g58a249bc6-thor-sm110-cu132-bundled"
PUSH_REGISTRY=""
FLASHINFER_CACHE="${HOME}/thor-flashinfer-cache"
VLLM_CACHE="${HOME}/thor-vllm-cache"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --push)            PUSH_REGISTRY="$2"; shift 2 ;;
        --base)            BASE_IMAGE="$2"; shift 2 ;;
        --tag)             BUNDLE_TAG="$2"; shift 2 ;;
        --flashinfer-cache) FLASHINFER_CACHE="$2"; shift 2 ;;
        --vllm-cache)      VLLM_CACHE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
if ! docker image inspect "$BASE_IMAGE" &>/dev/null; then
    echo "ERROR: Base image not found: $BASE_IMAGE" >&2
    echo "  Build it first with: ./build.sh" >&2
    exit 1
fi

if [ ! -d "$FLASHINFER_CACHE" ] || [ -z "$(ls -A "$FLASHINFER_CACHE" 2>/dev/null)" ]; then
    echo "ERROR: FlashInfer cache is empty or missing: $FLASHINFER_CACHE" >&2
    echo "  Start the model at least once to populate the JIT caches:" >&2
    echo "  ./start-model.sh qwen3.5-35b-a3b-nvfp4" >&2
    exit 1
fi

if [ ! -d "$VLLM_CACHE" ] || [ ! -d "$VLLM_CACHE/torch_compile_cache" ]; then
    echo "ERROR: vLLM torch compile cache is empty or missing: $VLLM_CACHE/torch_compile_cache" >&2
    echo "  Start the model at least once to populate the JIT caches." >&2
    exit 1
fi

# ── Create minimal build context ──────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Building NemoClaw-Thor bundled image"
echo "  Base:            ${BASE_IMAGE}"
echo "  Tag:             ${BUNDLE_TAG}"
echo "  FlashInfer cache: ${FLASHINFER_CACHE} ($(du -sh "$FLASHINFER_CACHE" | cut -f1))"
echo "  vLLM cache:      ${VLLM_CACHE} ($(du -sh "$VLLM_CACHE" | cut -f1))"
echo "=========================================="
echo ""

TMPCTX=$(mktemp -d)
trap "rm -rf $TMPCTX" EXIT

# Copy Dockerfile and entrypoint into context
cp "${SCRIPT_DIR}/Dockerfile.bundle" "${TMPCTX}/Dockerfile"
cp "${SCRIPT_DIR}/bundle-entrypoint.sh" "${TMPCTX}/bundle-entrypoint.sh"
cp -a "${SCRIPT_DIR}/../templates" "${TMPCTX}/templates"

# Copy caches into context (rsync for speed/progress; fall back to cp)
echo "Copying FlashInfer cache to build context..."
if command -v rsync &>/dev/null; then
    rsync -a --info=progress2 "${FLASHINFER_CACHE}/" "${TMPCTX}/flashinfer-cache/"
else
    cp -r "${FLASHINFER_CACHE}" "${TMPCTX}/flashinfer-cache"
fi

echo "Copying vLLM cache to build context..."
# vLLM cache files are written by the container running as root, so we need sudo.
if command -v rsync &>/dev/null; then
    sudo rsync -a --info=progress2 "${VLLM_CACHE}/" "${TMPCTX}/vllm-cache/"
else
    sudo cp -r "${VLLM_CACHE}/." "${TMPCTX}/vllm-cache/"
fi
# Make copied files readable by the current user for docker build
sudo chown -R "$(id -u):$(id -g)" "${TMPCTX}/vllm-cache/"

echo ""
echo "Build context size: $(du -sh "$TMPCTX" | cut -f1)"
echo ""

# ── Build ─────────────────────────────────────────────────────────────────────
docker build \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -t "${BUNDLE_TAG}" \
    "${TMPCTX}"

echo ""
echo "Built: ${BUNDLE_TAG}"
echo "Size:  $(docker image inspect "${BUNDLE_TAG}" --format '{{.Size}}' | numfmt --to=iec 2>/dev/null || docker image inspect "${BUNDLE_TAG}" --format '{{.Size}}')"

# ── Push ──────────────────────────────────────────────────────────────────────
if [ -n "$PUSH_REGISTRY" ]; then
    REMOTE_TAG="${PUSH_REGISTRY}:main-g58a249bc6-thor-sm110-cu132-bundled"
    echo ""
    echo "Tagging as: ${REMOTE_TAG}"
    docker tag "${BUNDLE_TAG}" "${REMOTE_TAG}"
    echo "Pushing to registry..."
    docker push "${REMOTE_TAG}"
    echo ""
    echo "Pushed: ${REMOTE_TAG}"
    echo ""
    echo "To run on another Thor:"
    echo "  docker run --rm --runtime nvidia --gpus all --ipc=host --network host \\"
    echo "    -v ~/my-hf-cache:/data/models/huggingface \\"
    echo "    ${REMOTE_TAG}"
else
    echo ""
    echo "To push to a registry:"
    echo "  docker login ghcr.io  # or your registry"
    echo "  ./bundle.sh --push ghcr.io/YOUR_ORG/nemoclaw-thor-vllm"
    echo ""
    echo "To run locally:"
    echo "  docker run --rm --runtime nvidia --gpus all --ipc=host --network host \\"
    echo "    -v ~/my-hf-cache:/data/models/huggingface \\"
    echo "    ${BUNDLE_TAG}"
fi
