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
    # CutlassFp8BlockScaledMMKernel: uses enable_sm100f_only, crashes SM110 (Xid 43).
    # FlashInfer FP8 is re-enabled: JIT cache has sm_110a GEMM kernels.
    #
    # ENABLE_TRIATTENTION=0: disable TriAttention plugin (official off switch).
    # TriAttention auto-registers and crashes at inference time without a
    # sparse_stats_path (TRIATTN_FATAL_TRITON_SCORING_REQUIRED:stats_path_not_set).
    # Qwen3.6-35B-A3B is not in TriAttention's supported-model matrix — it would
    # require porting the CUDA calibration script for DeltaNet hybrid layers.
    # Revisit only if upstream adds Qwen3.6 support.
    THOR_DOCKER_ENV_ARGS+=(
        -e "VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel,CutlassFp8BlockScaledMMKernel"
        -e "ENABLE_TRIATTENTION=0"
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
        # qwen3.5-122b-a10b-nvfp4-resharded removed
        qwen3.6-35b-a3b-fp8-dflash)
            # ★ BEST THROUGHPUT: 47.6 tok/s avg, 94 tok/s peak with matched z-lab drafter.
            # Qwen3.6-35B-A3B has head_dim=128 → FA2 works natively on SM110.
            # DFlash-15 with flash_attn backend. No runtime mods needed.
            # z-lab/Qwen3.6-35B-A3B-DFlash is gated — requires HF token.
            THOR_LAUNCH_MODEL_SOURCE="Qwen/Qwen3.6-35B-A3B-FP8"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.8}"
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "flash_attn"
                "--enforce-eager"
                "--language-model-only"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_xml"
                "--max-num-batched-tokens" "4096"
                "--speculative-config" '{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":15}'
            )
            ;;
        qwen3.6-35b-a3b-fp8-mtp-fp8kv)
            # MTP N=4 + FP8 KV: 25.7 tok/s, 80% acceptance, 1.44M KV tokens.
            # No PR needed. Simpler alternative to TurboQuant.
            THOR_LAUNCH_MODEL_SOURCE="Qwen/Qwen3.6-35B-A3B-FP8"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.8}"
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--enforce-eager"
                "--language-model-only"
                "--kv-cache-dtype" "fp8"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_xml"
                "--max-num-batched-tokens" "4096"
                "--speculative-config" '{"method":"mtp","num_speculative_tokens":4}'
            )
            ;;
        qwen3.6-35b-a3b-fp8-turboquant)
            # TurboQuant K8V4 + MTP N=4: 26.2 tok/s, 78% acceptance, 1.89M KV tokens at 0.8.
            # Best KV compression (~2.6x). Requires vLLM with PR #39931 (baked in v6 image)
            # + fix-pr39931-turboquant mod (full PR replay — gate removal +
            # TQFullAttentionSpec presence).
            THOR_LAUNCH_MODEL_SOURCE="Qwen/Qwen3.6-35B-A3B-FP8"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.5}"
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_MODS=fix-pr39931-turboquant"
            )
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--enforce-eager"
                "--language-model-only"
                "--kv-cache-dtype" "turboquant_k8v4"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_xml"
                "--max-num-batched-tokens" "4096"
                "--speculative-config" '{"method":"mtp","num_speculative_tokens":4}'
            )
            ;;
        qwen3.6-35b-a3b-nvfp4-dflash)
            # ★★ FASTEST: 54.7 tok/s, 66% acceptance. NVFP4 weights + DFlash-15.
            # head_dim=128 → flash_attn works natively on SM110.
            # z-lab/Qwen3.6-35B-A3B-DFlash is gated — requires HF token.
            THOR_LAUNCH_MODEL_SOURCE="RedHatAI/Qwen3.6-35B-A3B-NVFP4"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.8}"
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "flash_attn"
                "--enforce-eager"
                "--language-model-only"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_xml"
                "--max-num-batched-tokens" "4096"
                "--speculative-config" '{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":15}'
            )
            ;;
        qwen3.6-35b-a3b-nvfp4-tq-mtp)
            # ★ MAX CONTEXT: 28.0 tok/s, 79% acceptance, 2.22M KV tokens.
            # NVFP4 weights + TurboQuant K8V4 KV + MTP N=4.
            # Requires vLLM with PR #39931 (baked in v6 image) + fix-pr39931-turboquant (runtime mod replaying PR
            # mod (makes hybrid-model gate conditional on TQFullAttentionSpec presence).
            THOR_LAUNCH_MODEL_SOURCE="RedHatAI/Qwen3.6-35B-A3B-NVFP4"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.8}"
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass"
                "-e" "VLLM_USE_FLASHINFER_MOE_FP4=1"
                "-e" "VLLM_FLASHINFER_MOE_BACKEND=throughput"
                "-e" "VLLM_MODS=fix-pr39931-turboquant"
            )
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--enforce-eager"
                "--language-model-only"
                "--kv-cache-dtype" "turboquant_k8v4"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_xml"
                "--max-num-batched-tokens" "4096"
                "--speculative-config" '{"method":"mtp","num_speculative_tokens":4}'
            )
            ;;
        # qwen3.6-35b-a3b-nvfp4-mtp-fp8kv removed — crashes under 8-concurrent
        # (MoE autotuner picks invalid SM110 tile at M=128). Superseded by
        # qwen3.6-35b-a3b-nvfp4-tq-mtp which is strictly better on all axes.
        # qwen3.5-35b-a3b-nvfp4 removed — superseded by qwen3.6
        qwen3.5-9b-claude-distilled-nvfp4)
            # Qwen3.5-9B VLM: DeltaNet hybrid (linear_attention + full_attention) with visual encoder.
            # Claude 4.6 Opus reasoning-distilled, NVFP4 MLP-only + FP8 KV. Visual encoder kept bf16.
            # Multimodal: vision + text + tools. No --language-model-only.
            # --mm-encoder-attn-backend TORCH_SDPA: workaround for SM110 ViT PTX crash (#38411).
            # --max-num-batched-tokens 4096: MTP speculative decode defaults to 2048 which throttles
            #   throughput. 4096 gives the scheduler enough headroom for 8 seqs + draft tokens.
            # Use the dedicated no-think chat template variant for this fast-control profile.
            # The standard Qwen template opens <think> by default, which causes the 9B distilled
            # model to burn the full token budget reasoning before it emits content/tool calls.
            # Also do not advertise a Qwen reasoning parser for this profile: OpenClaw's current
            # OpenAI-completions path expects ordinary content/tool_calls, and vLLM's split
            # reasoning channel leaves the embedded agent with no final content to consume.
            # Source: Alexzander85/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-NVFP4-MLP-FP8KV
            THOR_LAUNCH_MODEL_SOURCE="Alexzander85/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-NVFP4-MLP-FP8KV"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.4}"
            THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen3-tool-call-compat-nothink.jinja"
            THOR_LAUNCH_CHAT_TEMPLATE_CONTAINER_PATH="/opt/nemoclaw-thor/templates/qwen3-tool-call-compat-nothink.jinja"
            THOR_DOCKER_ENV_ARGS+=(
                "-e" "VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass"
            )
            THOR_VLLM_ARGS+=(
                "--download-dir" "/data/models/huggingface/hub"
                "--attention-backend" "flashinfer"
                "--quantization" "modelopt"
                "--enable-auto-tool-choice"
                "--tool-call-parser" "qwen3_xml"
                "--enable-prefix-caching"
                "--mm-encoder-attn-backend" "TORCH_SDPA"
                "--max-num-batched-tokens" "4096"
                "--speculative-config" '{"method":"mtp","num_speculative_tokens":1}'
            )
            ;;
        # qwen3.5-9b-dflash removed
        # qwen3.5-9b-bf16-dflash removed
        # qwen3.5-27b-claude-distilled-nvfp4 removed
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
        gemma4-e4b-it)
            # Gemma 4 E4B IT — MoE (8B total, ~4B active per token), ~16 GB BF16.
            # Native function calling (gemma4 parser), vision, audio.
            # No NVFP4 quant available — runs at BF16, light enough at 0.4 GPU util.
            # triton_attn: same head_dim=512 FlashInfer limitation as all Gemma 4 models.
            # SWA (sliding window attention) + few global layers = small KV footprint.
            # --mm-encoder-attn-backend TORCH_SDPA: SM110 ViT PTX crash workaround.
            # 128K context (native for E-series, vs 256K for medium models).
            THOR_LAUNCH_MODEL_SOURCE="google/gemma-4-E4B-it"
            THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.4}"
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
        ${HF_TOKEN:+-e "HF_TOKEN=${HF_TOKEN}"} \
        -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
        -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor \
        -e "MAX_JOBS=${THOR_MAX_JOBS:-12}" \
        -e "NINJAFLAGS=-j${THOR_MAX_JOBS:-12}" \
        -e "MAKEFLAGS=-j${THOR_MAX_JOBS:-12}" \
        -e "CMAKE_BUILD_PARALLEL_LEVEL=${THOR_MAX_JOBS:-12}" \
        -v "${THOR_HF_CACHE_DIR}:/data/models/huggingface" \
        -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
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
