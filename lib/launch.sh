#!/usr/bin/env bash
# lib/launch.sh — Shared vLLM launcher logic for NemoClaw-Thor
#
# Source this file; do not execute it directly.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This file is meant to be sourced, not executed directly." >&2
    exit 1
fi

prepare_thor_launch_profile() {
    local profile="${1:-${THOR_MODEL_PROFILE:-}}"

    THOR_VLLM_IMAGE="${THOR_VLLM_IMAGE:-ghcr.io/nvidia-ai-iot/vllm:0.16.0-g15d76f74e-r38.2-arm64-sbsa-cu130-24.04}"
    THOR_VLLM_BIND_HOST="${THOR_VLLM_BIND_HOST:-0.0.0.0}"
    THOR_VLLM_PORT="${THOR_VLLM_PORT:-8000}"
    THOR_HF_CACHE_DIR="${THOR_HF_CACHE_DIR:-$HOME/thor-hf-cache}"
    THOR_VLLM_CACHE_DIR="${THOR_VLLM_CACHE_DIR:-$HOME/thor-vllm-cache}"
    THOR_TORCH_CACHE_DIR="${THOR_TORCH_CACHE_DIR:-$HOME/thor-torch-cache}"

    THOR_LAUNCH_HOST_MODEL_PATH=""
    THOR_LAUNCH_MODEL_SOURCE=""
    THOR_LAUNCH_GPU_MEMORY_UTILIZATION=""
    THOR_LAUNCH_MAX_NUM_BATCHED_TOKENS=""
    THOR_LAUNCH_SPECULATIVE_CONFIG=""
    THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH=""
    THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH=""
    THOR_CHAT_TEMPLATE_HOST_DIR="${THOR_CHAT_TEMPLATE_HOST_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/templates}"

    THOR_DOCKER_ENV_ARGS=()
    THOR_VLLM_ARGS=()

    case "${profile}" in
        qwen3.5-122b-a10b-nvfp4-resharded)
            THOR_LAUNCH_MODEL_SOURCE="/data/models/huggingface/hub/qwen-3.5-122b-a10b-nvfp4-resharded/resharded"
            THOR_LAUNCH_HOST_MODEL_PATH="${THOR_HF_CACHE_DIR}/hub/qwen-3.5-122b-a10b-nvfp4-resharded/resharded"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.90}"
            THOR_LAUNCH_MAX_NUM_BATCHED_TOKENS="${THOR_MAX_NUM_BATCHED_TOKENS:-8192}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen3-tool-call-compat.jinja"
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH="/opt/nemoclaw-thor/templates/qwen3-tool-call-compat.jinja"
            THOR_DOCKER_ENV_ARGS+=("-e" "VLLM_USE_FLASHINFER_MOE_FP4=0")
            THOR_VLLM_ARGS+=(
                "--quantization" "compressed-tensors"
                "--attention-backend" "FLASHINFER"
                "--language-model-only"
                "--reasoning-parser" "qwen3"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
            )
            ;;
        qwen3.5-27b-fp8)
            THOR_LAUNCH_MODEL_SOURCE="Qwen/Qwen3.5-27B-FP8"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.90}"
            THOR_LAUNCH_SPECULATIVE_CONFIG="${THOR_SPECULATIVE_CONFIG:-{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":2}}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen3-tool-call-compat.jinja"
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH="/opt/nemoclaw-thor/templates/qwen3-tool-call-compat.jinja"
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--tensor-parallel-size" "1"
                "--reasoning-parser" "qwen3"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
                "--speculative-config" "${THOR_LAUNCH_SPECULATIVE_CONFIG}"
            )
            ;;
        qwen3.5-35b-a3b-fp8)
            THOR_LAUNCH_MODEL_SOURCE="Qwen/Qwen3.5-35B-A3B-FP8"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.90}"
            THOR_LAUNCH_SPECULATIVE_CONFIG="${THOR_SPECULATIVE_CONFIG:-{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":2}}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen3-tool-call-compat.jinja"
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH="/opt/nemoclaw-thor/templates/qwen3-tool-call-compat.jinja"
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--tensor-parallel-size" "1"
                "--reasoning-parser" "qwen3"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
                "--speculative-config" "${THOR_LAUNCH_SPECULATIVE_CONFIG}"
            )
            ;;
        qwen3.5-35b-a3b-nvfp4)
            THOR_LAUNCH_MODEL_SOURCE="Kbenkhaled/Qwen3.5-35B-A3B-NVFP4"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.85}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen3-tool-call-compat.jinja"
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH="/opt/nemoclaw-thor/templates/qwen3-tool-call-compat.jinja"
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--reasoning-parser" "qwen3"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
                "--enable-prefix-caching"
            )
            ;;
        *)
            fail "Unsupported model profile: ${profile}"
            print_supported_model_profiles
            return 1
            ;;
    esac

    THOR_LAUNCH_MAX_MODEL_LEN="${THOR_MAX_MODEL_LEN:-${THOR_TARGET_MAX_MODEL_LEN}}"
    THOR_LAUNCH_KV_CACHE_DTYPE="${THOR_KV_CACHE_DTYPE:-${THOR_TARGET_KV_CACHE_DTYPE}}"
    THOR_LAUNCH_MAX_NUM_SEQS="${THOR_MAX_NUM_SEQS:-${THOR_TARGET_MAX_NUM_SEQS}}"

    THOR_VLLM_ARGS=(
        "${THOR_VLLM_ARGS[@]}"
        "--served-model-name" "${THOR_MODEL_ID}"
        "--host" "${THOR_VLLM_BIND_HOST}"
        "--port" "${THOR_VLLM_PORT}"
        "--gpu-memory-utilization" "${THOR_LAUNCH_GPU_MEMORY_UTILIZATION}"
        "--max-model-len" "${THOR_LAUNCH_MAX_MODEL_LEN}"
        "--kv-cache-dtype" "${THOR_LAUNCH_KV_CACHE_DTYPE}"
        "--max-num-seqs" "${THOR_LAUNCH_MAX_NUM_SEQS}"
    )

    if [[ -n "${THOR_LOCAL_VLLM_API_KEY}" && "${THOR_LOCAL_VLLM_API_KEY}" != "dummy" ]]; then
        THOR_VLLM_ARGS+=("--api-key" "${THOR_LOCAL_VLLM_API_KEY}")
    fi

    if [[ -n "${THOR_LAUNCH_MAX_NUM_BATCHED_TOKENS}" ]]; then
        THOR_VLLM_ARGS+=("--max-num-batched-tokens" "${THOR_LAUNCH_MAX_NUM_BATCHED_TOKENS}")
    fi

    if [[ -n "${THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH}" ]]; then
        THOR_VLLM_ARGS+=("--chat-template" "${THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH}")
    fi
}

check_thor_launch_prereqs() {
    if ! command -v docker &>/dev/null; then
        fail "docker command not found"
        fix "Install Docker and the NVIDIA container runtime first."
        return 1
    fi

    if ! docker info &>/dev/null; then
        fail "Docker daemon is not running"
        fix "Run: sudo systemctl start docker"
        return 1
    fi

    if [[ -n "${THOR_LAUNCH_HOST_MODEL_PATH}" && ! -d "${THOR_LAUNCH_HOST_MODEL_PATH}" ]]; then
        fail "Required model path not found: ${THOR_LAUNCH_HOST_MODEL_PATH}"
        info "This profile uses a pre-resharded local model path."
        fix "Follow the download instructions in:"
        fix "  $(cd "$(dirname "${BASH_SOURCE[0]}")/../../thor_llm/models/${THOR_MODEL_PROFILE}" 2>/dev/null && pwd)/README.md"
        return 1
    fi

    if [[ -n "${THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH}" && ! -f "${THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH}" ]]; then
        fail "Required chat template not found: ${THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH}"
        fix "Restore the NemoClaw-Thor templates directory or set THOR_CHAT_TEMPLATE_HOST_DIR."
        return 1
    fi

    mkdir -p "${THOR_HF_CACHE_DIR}" "${THOR_VLLM_CACHE_DIR}" "${THOR_TORCH_CACHE_DIR}"
    return 0
}

print_thor_launch_summary() {
    echo "  Image:              ${THOR_VLLM_IMAGE}"
    echo "  Profile:            ${THOR_MODEL_PROFILE}"
    echo "  Source:             ${THOR_LAUNCH_MODEL_SOURCE}"
    echo "  Served model id:    ${THOR_MODEL_ID}"
    echo "  Bind:               ${THOR_VLLM_BIND_HOST}:${THOR_VLLM_PORT}"
    echo "  Max context:        ${THOR_LAUNCH_MAX_MODEL_LEN}"
    echo "  KV cache dtype:     ${THOR_LAUNCH_KV_CACHE_DTYPE}"
    echo "  Max num seqs:       ${THOR_LAUNCH_MAX_NUM_SEQS}"
    echo "  GPU mem util:       ${THOR_LAUNCH_GPU_MEMORY_UTILIZATION}"
    if [[ -n "${THOR_LAUNCH_MAX_NUM_BATCHED_TOKENS}" ]]; then
        echo "  Max batched tokens: ${THOR_LAUNCH_MAX_NUM_BATCHED_TOKENS}"
    fi
    if [[ -n "${THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH}" ]]; then
        echo "  Chat template:      ${THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH}"
    fi
    echo "  HF cache:           ${THOR_HF_CACHE_DIR}"
    echo "  vLLM cache:         ${THOR_VLLM_CACHE_DIR}"
    echo "  Torch cache:        ${THOR_TORCH_CACHE_DIR}"
}

run_thor_vllm_container() {
    local docker_tty_args=()
    local docker_mount_args=()

    if [[ -t 0 && -t 1 ]]; then
        docker_tty_args=(-i -t)
    fi

    if [[ -n "${THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH}" ]]; then
        docker_mount_args=(-v "${THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH}:${THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH}:ro")
    fi

    docker run --rm \
        "${docker_tty_args[@]}" \
        --runtime nvidia --gpus all \
        --ipc=host --network host \
        -e HF_HOME=/data/models/huggingface \
        -e HF_HUB_CACHE=/data/models/huggingface/hub \
        -e TRANSFORMERS_CACHE=/data/models/huggingface/hub \
        -v "${THOR_HF_CACHE_DIR}:/data/models/huggingface" \
        -v "${THOR_VLLM_CACHE_DIR}:/root/.cache/vllm" \
        -v "${THOR_TORCH_CACHE_DIR}:/root/.cache/torch" \
        "${docker_mount_args[@]}" \
        "${THOR_DOCKER_ENV_ARGS[@]}" \
        "${THOR_VLLM_IMAGE}" \
        vllm serve "${THOR_LAUNCH_MODEL_SOURCE}" "${THOR_VLLM_ARGS[@]}"
}
