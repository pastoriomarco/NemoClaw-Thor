#!/usr/bin/env bash
# Build the verified Qwen3.8 vLLM overlay for Jetson AGX Thor (SM110).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${QWEN38_THOR_IMAGE:-nemoclaw-thor/vllm-openai:qwen38-thor-sm110}"

echo "Building ${IMAGE_NAME}"
docker build \
    --network host \
    --progress plain \
    --file "${SCRIPT_DIR}/Dockerfile.qwen38-thor-sm110" \
    --tag "${IMAGE_NAME}" \
    "$@" \
    "${SCRIPT_DIR}"
