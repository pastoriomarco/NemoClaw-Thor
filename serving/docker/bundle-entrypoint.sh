#!/bin/bash
# bundle-entrypoint.sh — NemoClaw-Thor bundled image entrypoint
#
# If called with no args (or just "serve"), starts vLLM with defaults from env vars.
# If called with explicit args, passes them through (allows full override).
#
# Env vars (all optional, have defaults):
#   VLLM_MODEL              Model ID or local path (default: nvidia/Qwen3.6-35B-A3B-NVFP4)
#   VLLM_PORT               Port to bind (default: 8000)
#   VLLM_HOST               Host to bind (default: 0.0.0.0)
#   VLLM_MAX_MODEL_LEN      Max context length in tokens (default: 262144)
#   VLLM_KV_CACHE_DTYPE     KV cache dtype: fp8 or auto (default: fp8)
#   VLLM_GPU_MEM_UTIL       GPU memory utilization 0.0-1.0 (default: 0.55)
#   VLLM_MAX_NUM_SEQS       Max concurrent sequences (default: 4)
#   VLLM_API_KEY            API key (optional, enables auth)
#   VLLM_SERVED_MODEL_NAME  Served model name alias (default: same as VLLM_MODEL)
#
# Examples:
#   # Default: serve Qwen3.6-35B-A3B-NVFP4
#   docker run ... nemoclaw-thor/vllm:bundled
#
#   # Custom model or port
#   docker run ... -e VLLM_MODEL=myorg/MyModel -e VLLM_PORT=9000 nemoclaw-thor/vllm:bundled
#
#   # Full override (pass explicit vllm serve args)
#   docker run ... nemoclaw-thor/vllm:bundled vllm serve mymodel --port 9000 --attention-backend flashinfer
set -e

# Apply any VLLM_MODS if set (inherited from base entrypoint logic)
MODS_DIR="/workspace/mods"
if [ -n "$VLLM_MODS" ]; then
    IFS=',' read -ra MOD_LIST <<< "$VLLM_MODS"
    for mod in "${MOD_LIST[@]}"; do
        mod=$(echo "$mod" | xargs)
        mod_path="${MODS_DIR}/${mod}"
        if [ -d "$mod_path" ] && [ -x "$mod_path/run.sh" ]; then
            echo "[entrypoint] Applying mod: ${mod}"
            (cd "$mod_path" && ./run.sh)
        else
            echo "[entrypoint] WARNING: mod not found: ${mod}" >&2
        fi
    done
fi

# If explicit args given (and first arg isn't "serve"), pass through directly
if [ "$#" -gt 0 ] && [ "$1" != "serve" ]; then
    exec "$@"
fi

# Build vllm serve command from env vars
MODEL="${VLLM_MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}"
SERVED_NAME="${VLLM_SERVED_MODEL_NAME:-${MODEL}}"

# Serving flags mirror the qwen3.6-35b-a3b-nvfp4-nvidia profile (NVIDIA Spark
# recipe). --quantization modelopt + --moe-backend marlin are load-bearing:
# without them engine init crashes (no FlashInfer NVFP4 MoE backend matches
# W4A16's 16-bit activations). MTP K=3 speculative decode, the qwen3_coder tool
# parser, and the froggeric chat template complete the recipe. See
# serving/launch.sh (qwen3.6-35b-a3b-nvfp4-nvidia) for the full rationale.
ARGS=(
    vllm serve "${MODEL}"
    --served-model-name "${SERVED_NAME}"
    --host "${VLLM_HOST:-0.0.0.0}"
    --port "${VLLM_PORT:-8000}"
    --download-dir /data/models/huggingface/hub
    --quantization modelopt
    --moe-backend marlin
    --attention-backend flashinfer
    --enforce-eager
    --language-model-only
    --enable-prefix-caching
    --enable-chunked-prefill
    --async-scheduling
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --default-chat-template-kwargs '{"enable_thinking":true}'
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
    --trust-remote-code
    --max-model-len "${VLLM_MAX_MODEL_LEN:-262144}"
    --kv-cache-dtype "${VLLM_KV_CACHE_DTYPE:-fp8}"
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.55}"
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-4}"
    --max-num-batched-tokens 8192
    --compilation-config '{"custom_ops":["-quant_fp8","-quant_fp8","-quant_fp8"]}'
)

if [ -n "${VLLM_API_KEY}" ] && [ "${VLLM_API_KEY}" != "dummy" ]; then
    ARGS+=(--api-key "${VLLM_API_KEY}")
fi

# Chat template: bundled at /opt/nemoclaw-thor/templates/qwen-fixed-froggeric.jinja
TEMPLATE=/opt/nemoclaw-thor/templates/qwen-fixed-froggeric.jinja
if [ -f "$TEMPLATE" ]; then
    ARGS+=(--chat-template "$TEMPLATE")
fi

echo "[entrypoint] Starting vLLM for SM110 (Qwen3.6 NVFP4 profile)"
echo "[entrypoint] Model: ${MODEL}"
echo "[entrypoint] Bind: ${VLLM_HOST:-0.0.0.0}:${VLLM_PORT:-8000}"
exec "${ARGS[@]}"
