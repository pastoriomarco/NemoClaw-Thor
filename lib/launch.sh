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
    THOR_MODS_HOST_DIR="${THOR_MODS_HOST_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../docker/mods" && pwd)}"

    THOR_DOCKER_ENV_ARGS=()
    THOR_VLLM_ARGS=()

    # SM110 (Thor): CUTLASS sm100 kernels are incompatible — disable them.
    # FlashInfer FP8 is re-enabled: JIT cache has sm_110a GEMM kernels.
    THOR_DOCKER_ENV_ARGS+=(
        -e "VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel"
    )

    case "${profile}" in
        qwen3.5-122b-a10b-nvfp4)
            THOR_LAUNCH_MODEL_SOURCE="/data/models/huggingface/hub/models--Sehyo--Qwen3.5-122B-A10B-NVFP4"
            THOR_LAUNCH_HOST_MODEL_PATH="${THOR_HF_CACHE_DIR}/hub/models--Sehyo--Qwen3.5-122B-A10B-NVFP4"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.8}"
            THOR_LAUNCH_MAX_NUM_BATCHED_TOKENS="${THOR_MAX_NUM_BATCHED_TOKENS:-8192}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen3-tool-call-compat.jinja"
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH="/opt/nemoclaw-thor/templates/qwen3-tool-call-compat.jinja"
            # SM110 NVFP4: FlashInfer CUTLASS for GEMM + MoE, FlashInfer for attention.
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass"
                "-e" "VLLM_USE_FLASHINFER_MOE_FP4=1"
                "-e" "VLLM_FLASHINFER_MOE_BACKEND=throughput"
            )
            # MTP speculative decoding: model has mtp_num_hidden_layers=1.
            # HackMD Spark guide uses num_speculative_tokens=3; start with 1 (safe).
            THOR_VLLM_ARGS+=(
                "--attention-backend" "flashinfer"
                "--language-model-only"
                "--reasoning-parser" "qwen3"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_xml"
                "--enable-prefix-caching"
                "--speculative-config" '{"method":"mtp","num_speculative_tokens":1}'
            )
            ;;
        qwen3.5-122b-a10b-nvfp4-resharded)
            THOR_LAUNCH_MODEL_SOURCE="/data/models/huggingface/hub/qwen-3.5-122b-a10b-nvfp4-resharded/resharded"
            THOR_LAUNCH_HOST_MODEL_PATH="${THOR_HF_CACHE_DIR}/hub/qwen-3.5-122b-a10b-nvfp4-resharded/resharded"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.85}"
            THOR_LAUNCH_MAX_NUM_BATCHED_TOKENS="${THOR_MAX_NUM_BATCHED_TOKENS:-8192}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen3-tool-call-compat.jinja"
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH="/opt/nemoclaw-thor/templates/qwen3-tool-call-compat.jinja"
            # SM110 NVFP4: FlashInfer CUTLASS for GEMM + MoE, FlashInfer for attention.
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
                "--tool-call-parser" "qwen3_xml"
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
                "--tool-call-parser" "qwen3_xml"
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
                "--tool-call-parser" "qwen3_xml"
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
                "--tool-call-parser" "qwen3_xml"
                "--enable-prefix-caching"
                "--max-num-batched-tokens" "4096"
            )
            ;;
        qwopus3.5-27b-nvfp4)
            # Qwen3.5-27B DeltaNet hybrid: 48 linear_attention + 16 full_attention layers.
            # Mixed NVFP4 (W4A4 for gate/up/o_proj) + FP8 (down_proj, QKV) + BF16 (lm_head).
            # Only the 16 full_attention layers use KV cache — much smaller KV footprint than pure dense.
            # FlashInfer v0.6.7 FMHA works on SM110 (+38% vs triton_attn, verified).
            # No MoE — VLLM_USE_FLASHINFER_MOE_FP4 not applicable.
            # linear_attention layers handled by GatedDeltaNetAttention (vllm built-in).
            THOR_LAUNCH_MODEL_SOURCE="ShinePixelOrg/Qwopus3.5-27B-v3-NVFP4"
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
                "--tool-call-parser" "qwen3_xml"
                "--enable-prefix-caching"
                "--speculative-config" '{"method":"mtp","num_speculative_tokens":1}'
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
                "--tool-call-parser" "qwen3_xml"
                "--enable-prefix-caching"
                "--speculative-config" '{"method":"mtp","num_speculative_tokens":1}'
            )
            ;;
        qwen3.5-27b-claude-distilled-v2-nvfp4)
            # Qwen3.5-27B DeltaNet hybrid: same architecture as v1.
            # v2 distillation from mconcat/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-NVFP4.
            THOR_LAUNCH_MODEL_SOURCE="mconcat/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-NVFP4"
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
                "--tool-call-parser" "qwen3_xml"
                "--enable-prefix-caching"
                "--speculative-config" '{"method":"mtp","num_speculative_tokens":1}'
            )
            ;;
        gemma4-31b-it-nvfp4)
            # Gemma 4 31B IT NVFP4 — dense model, ~17 GB in VRAM.
            # Vision enabled (SigLIP2 ~550M params), tool calling via gemma4 parser.
            # Thinking mode via reasoning-parser deepseek_r1 (<|think|> tokens).
            # --attention-backend triton_attn: FlashInfer kernels crash on head_dim=512
            # (Gemma 4 global attention layers). FlashInfer JIT generates invalid MMA
            # tiling for dim>256. triton_attn handles arbitrary head sizes.
            # See vllm-project/vllm#38887. NVFP4 GEMM still uses flashinfer-cutlass.
            # --mm-encoder-attn-backend TORCH_SDPA: workaround for #38411 — ViT FA2
            # PTX crash on SM110 with CUDA 13.0 host driver.
            THOR_LAUNCH_MODEL_SOURCE="nvidia/Gemma-4-31B-IT-NVFP4"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.80}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH=""
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH=""
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass"
            )
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "triton_attn"
                "--quantization" "modelopt"
                "--reasoning-parser" "gemma4"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "gemma4"
                "--enable-prefix-caching"
                "--mm-encoder-attn-backend" "TORCH_SDPA"
            )
            ;;
        gemma4-26b-a4b-it)
            # Gemma 4 26B-A4B IT — MoE (128 total, 8 active, 1 shared), ~52 GB BF16.
            # 3.8B active params per token — inference speed comparable to 4B dense.
            # Vision enabled, tool calling via gemma4 parser, thinking via <|think|>.
            # KV cache is small: hybrid SWA (1024 window) + 5 global attention layers.
            # No NVFP4 quant available — runs at BF16, needs careful memory budgeting.
            # triton_attn: same head_dim=512 FlashInfer limitation as 31B.
            THOR_LAUNCH_MODEL_SOURCE="google/gemma-4-26B-A4B-it"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.80}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH=""
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH=""
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "triton_attn"
                "--reasoning-parser" "gemma4"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "gemma4"
                "--enable-prefix-caching"
                "--mm-encoder-attn-backend" "TORCH_SDPA"
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

    # Mount host mods directory so new/updated mods are available without rebuild
    if [[ -d "${THOR_MODS_HOST_DIR}" ]]; then
        docker_mount_args+=(-v "${THOR_MODS_HOST_DIR}:/workspace/mods:ro")
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
        -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor \
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
