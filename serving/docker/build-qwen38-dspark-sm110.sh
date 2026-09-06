#!/usr/bin/env bash
# Build the Qwen3.8 DSpark tuning overlay for Jetson AGX Thor (SM110).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="${QWEN38_THOR_BASE_IMAGE:-nemoclaw-thor/vllm-openai:qwen38-thor-sm110}"
IMAGE_NAME="${QWEN38_DSPARK_THOR_IMAGE:-nemoclaw-thor/vllm-openai:qwen38-dspark-sm110}"

if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    echo "Required local base image is missing: ${BASE_IMAGE}" >&2
    echo "Build it first with ./serving/docker/build-qwen38-thor-sm110.sh" >&2
    exit 1
fi

echo "Building ${IMAGE_NAME} from local ${BASE_IMAGE}"
docker build \
    --pull=false \
    --network host \
    --progress plain \
    --build-arg "QWEN38_THOR_BASE_IMAGE=${BASE_IMAGE}" \
    --file "${SCRIPT_DIR}/Dockerfile.qwen38-dspark-sm110" \
    --tag "${IMAGE_NAME}" \
    "$@" \
    "${SCRIPT_DIR}"
