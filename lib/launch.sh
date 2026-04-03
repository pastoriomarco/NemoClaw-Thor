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

    THOR_VLLM_IMAGE="${THOR_VLLM_IMAGE:-nemoclaw-thor/vllm:latest}"
    THOR_VLLM_BIND_HOST="${THOR_VLLM_BIND_HOST:-0.0.0.0}"
    THOR_VLLM_PORT="${THOR_VLLM_PORT:-8000}"
    THOR_HF_CACHE_DIR="${THOR_HF_CACHE_DIR:-$HOME/thor-hf-cache}"
    THOR_VLLM_CACHE_DIR="${THOR_VLLM_CACHE_DIR:-$HOME/thor-vllm-cache}"
    THOR_TORCH_CACHE_DIR="${THOR_TORCH_CACHE_DIR:-$HOME/thor-torch-cache}"
    THOR_FLASHINFER_CACHE_DIR="${THOR_FLASHINFER_CACHE_DIR:-$HOME/thor-flashinfer-cache}"

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

    # SM110 (Thor): CUTLASS sm100 kernels are incompatible — disable them.
    # FlashInfer FP8 is re-enabled: JIT cache has sm_110a GEMM kernels.
    THOR_DOCKER_ENV_ARGS+=(
        -e "VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel"
    )

    case "${profile}" in
        qwen3.5-122b-a10b-nvfp4-resharded)
            THOR_LAUNCH_MODEL_SOURCE="/data/models/huggingface/hub/qwen-3.5-122b-a10b-nvfp4-resharded/resharded"
            THOR_LAUNCH_HOST_MODEL_PATH="${THOR_HF_CACHE_DIR}/hub/qwen-3.5-122b-a10b-nvfp4-resharded/resharded"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.85}"
            THOR_LAUNCH_MAX_NUM_BATCHED_TOKENS="${THOR_MAX_NUM_BATCHED_TOKENS:-8192}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen3-tool-call-compat.jinja"
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH="/opt/nemoclaw-thor/templates/qwen3-tool-call-compat.jinja"
            # SM110 NVFP4: FlashInfer CUTLASS for GEMM + MoE, FlashInfer for attention.
            # FlashInfer v0.6.7 FMHA works on SM110 (verified on 27B distilled, +38% vs triton_attn).
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass"
                "-e" "VLLM_USE_FLASHINFER_MOE_FP4=1"
                "-e" "VLLM_FLASHINFER_MOE_BACKEND=throughput"
            )
            THOR_VLLM_ARGS+=(
                "--attention-backend" "flashinfer"
                "--language-model-only"
                "--reasoning-parser" "qwen3"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
                "--enable-prefix-caching"
            )
            ;;
        qwen3.5-27b-fp8)
            THOR_LAUNCH_MODEL_SOURCE="Qwen/Qwen3.5-27B-FP8"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.8}"
            THOR_LAUNCH_SPECULATIVE_CONFIG="${THOR_SPECULATIVE_CONFIG:-{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":2}}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen3-tool-call-compat.jinja"
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH="/opt/nemoclaw-thor/templates/qwen3-tool-call-compat.jinja"
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--tensor-parallel-size" "1"
                "--attention-backend" "flashinfer"
                "--language-model-only"
                "--reasoning-parser" "qwen3"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
                "--speculative-config" "${THOR_LAUNCH_SPECULATIVE_CONFIG}"
            )
            ;;
        qwen3.5-35b-a3b-fp8)
            THOR_LAUNCH_MODEL_SOURCE="Qwen/Qwen3.5-35B-A3B-FP8"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.8}"
            THOR_LAUNCH_SPECULATIVE_CONFIG="${THOR_SPECULATIVE_CONFIG:-{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":2}}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen3-tool-call-compat.jinja"
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH="/opt/nemoclaw-thor/templates/qwen3-tool-call-compat.jinja"
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--tensor-parallel-size" "1"
                "--attention-backend" "flashinfer"
                "--language-model-only"
                "--reasoning-parser" "qwen3"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
                "--speculative-config" "${THOR_LAUNCH_SPECULATIVE_CONFIG}"
            )
            ;;
        qwen3.5-35b-a3b-nvfp4)
            THOR_LAUNCH_MODEL_SOURCE="Kbenkhaled/Qwen3.5-35B-A3B-NVFP4"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.8}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen3-tool-call-compat.jinja"
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH="/opt/nemoclaw-thor/templates/qwen3-tool-call-compat.jinja"
            # SM110 NVFP4: FlashInfer CUTLASS for GEMM + MoE, FlashInfer for attention.
            # FlashInfer v0.6.7 FMHA works on SM110 (verified on 27B distilled, +38% vs triton_attn).
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass"
                "-e" "VLLM_USE_FLASHINFER_MOE_FP4=1"
                "-e" "VLLM_FLASHINFER_MOE_BACKEND=throughput"
            )
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "flashinfer"
                "--language-model-only"
                "--reasoning-parser" "qwen3"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
                "--enable-prefix-caching"
                "--max-num-batched-tokens" "4096"
            )
            ;;
        qwen3.5-27b-claude-distilled-nvfp4)
            # Qwen3.5-27B DeltaNet hybrid: 48 linear_attention + 16 full_attention layers.
            # Mixed NVFP4 (W4A4 for gate/up/o_proj) + FP8 (down_proj, QKV) + BF16 (lm_head).
            # Only the 16 full_attention layers use KV cache — much smaller KV footprint than pure dense.
            # FlashInfer v0.6.7 FMHA works on SM110 (+38% vs triton_attn, verified).
            # No MoE — VLLM_USE_FLASHINFER_MOE_FP4 not applicable.
            # linear_attention layers handled by GatedDeltaNetAttention (vllm built-in).
            THOR_LAUNCH_MODEL_SOURCE="mconcat/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-NVFP4"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.8}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen3-tool-call-compat.jinja"
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH="/opt/nemoclaw-thor/templates/qwen3-tool-call-compat.jinja"
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass"
            )
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "flashinfer"
                "--language-model-only"
                "--reasoning-parser" "qwen3"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
                "--enable-prefix-caching"
                "--speculative-config" '{"method":"mtp","num_speculative_tokens":1}'
            )
            ;;
        qwen3.5-27b)
            THOR_LAUNCH_MODEL_SOURCE="Qwen/Qwen3.5-27B"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.7}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen3-tool-call-compat.jinja"
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH="/opt/nemoclaw-thor/templates/qwen3-tool-call-compat.jinja"
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--tensor-parallel-size" "1"
                "--attention-backend" "flashinfer"
                "--language-model-only"
                "--enforce-eager"
                "--reasoning-parser" "qwen3"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_coder"
            )
            ;;
        gemma4-31b-it-q4)
            # llama.cpp profile — launches llama-server in the NVIDIA Gemma 4 container.
            # Gemma 4 31B Q4_K_M with optional E4B draft model for speculative decoding.
            # Reasoning via --reasoning-format deepseek (emits reasoning_content).
            # Hybrid SWA/global attention with fused Gated Delta Net kernels.
            THOR_LAUNCH_BACKEND="llamacpp"
            THOR_LAUNCH_MODEL_SOURCE="llama-server (llama.cpp)"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="n/a"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH=""
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH=""
            THOR_DOCKER_ENV_ARGS=()
            THOR_VLLM_ARGS=()

            THOR_LLAMACPP_IMAGE="${THOR_LLAMACPP_IMAGE:-ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-thor}"
            THOR_LLAMACPP_MODEL_PATH="${THOR_LLAMACPP_MODEL_PATH:-${THOR_HF_CACHE_DIR}/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf}"
            THOR_LLAMACPP_DRAFT_PATH="${THOR_LLAMACPP_DRAFT_PATH:-}"
            THOR_LLAMACPP_DRAFT_N="${THOR_LLAMACPP_DRAFT_N:-4}"
            THOR_LLAMACPP_CTX="${THOR_LLAMACPP_CTX:-1048576}"
            THOR_LLAMACPP_PARALLEL="${THOR_LLAMACPP_PARALLEL:-${THOR_TARGET_MAX_NUM_SEQS}}"
            THOR_LLAMACPP_CACHE_TYPE_K="${THOR_LLAMACPP_CACHE_TYPE_K:-q8_0}"
            THOR_LLAMACPP_CACHE_TYPE_V="${THOR_LLAMACPP_CACHE_TYPE_V:-q8_0}"
            THOR_LLAMACPP_CACHE_RAM="${THOR_LLAMACPP_CACHE_RAM:-2048}"
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
        "--compilation-config" '{"custom_ops":["-quant_fp8","-quant_fp8","-quant_fp8"]}'
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

    if [[ "${THOR_LAUNCH_BACKEND:-vllm}" == "llamacpp" ]]; then
        if [[ ! -f "${THOR_LLAMACPP_MODEL_PATH:-}" ]]; then
            fail "Model file not found: ${THOR_LLAMACPP_MODEL_PATH:-<not set>}"
            fix "Download the GGUF model first."
            return 1
        fi
        if [[ -n "${THOR_LLAMACPP_DRAFT_PATH:-}" && ! -f "${THOR_LLAMACPP_DRAFT_PATH}" ]]; then
            warn "Draft model not found: ${THOR_LLAMACPP_DRAFT_PATH}"
            info "Speculative decoding will be disabled. Download the draft model to enable it."
        fi
    fi

    mkdir -p "${THOR_HF_CACHE_DIR}" "${THOR_VLLM_CACHE_DIR}" "${THOR_TORCH_CACHE_DIR}" "${THOR_FLASHINFER_CACHE_DIR}"
    return 0
}

print_thor_launch_summary() {
    echo "  Profile:            ${THOR_MODEL_PROFILE}"
    echo "  Source:             ${THOR_LAUNCH_MODEL_SOURCE}"
    echo "  Served model id:    ${THOR_MODEL_ID}"
    echo "  Bind:               ${THOR_VLLM_BIND_HOST}:${THOR_VLLM_PORT}"
    echo "  Max context:        ${THOR_LAUNCH_MAX_MODEL_LEN}"
    echo "  Max num seqs:       ${THOR_LAUNCH_MAX_NUM_SEQS}"

    if [[ "${THOR_LAUNCH_BACKEND:-vllm}" == "llamacpp" ]]; then
        echo "  Backend:            llama.cpp (llama-server)"
        echo "  Image:              ${THOR_LLAMACPP_IMAGE}"
        echo "  Model:              ${THOR_LLAMACPP_MODEL_PATH}"
        if [[ -f "${THOR_LLAMACPP_DRAFT_PATH:-}" ]]; then
            echo "  Draft model:        ${THOR_LLAMACPP_DRAFT_PATH}"
            echo "  Draft tokens:       ${THOR_LLAMACPP_DRAFT_N}"
        else
            echo "  Draft model:        (none)"
        fi
        echo "  Context (total):    ${THOR_LLAMACPP_CTX}"
        echo "  Parallel slots:     ${THOR_LLAMACPP_PARALLEL}"
        echo "  KV cache type:      K=${THOR_LLAMACPP_CACHE_TYPE_K} V=${THOR_LLAMACPP_CACHE_TYPE_V}"
    else
        echo "  Image:              ${THOR_VLLM_IMAGE}"
        echo "  KV cache dtype:     ${THOR_LAUNCH_KV_CACHE_DTYPE}"
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
        echo "  FlashInfer cache:   ${THOR_FLASHINFER_CACHE_DIR}"
    fi
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
        -e NVIDIA_DISABLE_REQUIRE=true \
        -e HF_HOME=/data/models/huggingface \
        -e HF_HUB_CACHE=/data/models/huggingface/hub \
        -e TRANSFORMERS_CACHE=/data/models/huggingface/hub \
        -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
        -v "${THOR_HF_CACHE_DIR}:/data/models/huggingface" \
        -v "${THOR_VLLM_CACHE_DIR}:/root/.cache/vllm" \
        -v "${THOR_TORCH_CACHE_DIR}:/root/.cache/torch" \
        -v "${THOR_FLASHINFER_CACHE_DIR}:/root/.cache/flashinfer" \
        "${docker_mount_args[@]}" \
        "${THOR_DOCKER_ENV_ARGS[@]}" \
        "${THOR_VLLM_IMAGE}" \
        vllm serve "${THOR_LAUNCH_MODEL_SOURCE}" "${THOR_VLLM_ARGS[@]}"
}

run_thor_llamacpp_container() {
    local docker_tty_args=()

    if [[ -t 0 && -t 1 ]]; then
        docker_tty_args=(-i -t)
    fi

    # Map host paths under THOR_HF_CACHE_DIR to /data/models inside the container.
    local model_container_path="/data/models/${THOR_LLAMACPP_MODEL_PATH#"${THOR_HF_CACHE_DIR}/"}"

    local draft_args=()
    if [[ -f "${THOR_LLAMACPP_DRAFT_PATH:-}" ]]; then
        local draft_container_path="/data/models/${THOR_LLAMACPP_DRAFT_PATH#"${THOR_HF_CACHE_DIR}/"}"
        draft_args=(-md "${draft_container_path}" --draft "${THOR_LLAMACPP_DRAFT_N}")
    fi

    docker run --rm \
        "${docker_tty_args[@]}" \
        --runtime nvidia --network host \
        -v "${THOR_HF_CACHE_DIR}:/data/models" \
        "${THOR_LLAMACPP_IMAGE}" \
        llama-server \
            -m "${model_container_path}" \
            "${draft_args[@]}" \
            --host "${THOR_VLLM_BIND_HOST}" \
            --port "${THOR_VLLM_PORT}" \
            -np "${THOR_LLAMACPP_PARALLEL}" \
            -c "${THOR_LLAMACPP_CTX}" \
            --cache-type-k "${THOR_LLAMACPP_CACHE_TYPE_K}" \
            --cache-type-v "${THOR_LLAMACPP_CACHE_TYPE_V}" \
            --cache-ram "${THOR_LLAMACPP_CACHE_RAM}" \
            --reasoning-format deepseek \
            --reasoning auto
}
