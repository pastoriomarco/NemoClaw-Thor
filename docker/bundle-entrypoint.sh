#!/bin/bash
# bundle-entrypoint.sh — NemoClaw-Thor bundled image entrypoint
#
# If called with no args (or just "serve"), starts vLLM with defaults from env vars.
# If called with explicit args, passes them through (allows full override).
#
# Env vars (all optional, have defaults):
#   VLLM_MODEL              Model ID or local path (default: Kbenkhaled/Qwen3.5-35B-A3B-NVFP4)
#   VLLM_PORT               Port to bind (default: 8000)
#   VLLM_HOST               Host to bind (default: 0.0.0.0)
#   VLLM_MAX_MODEL_LEN      Max context length in tokens (default: 65536)
#   VLLM_KV_CACHE_DTYPE     KV cache dtype: fp8 or auto (default: fp8)
#   VLLM_GPU_MEM_UTIL       GPU memory utilization 0.0-1.0 (default: 0.8)
#   VLLM_MAX_NUM_SEQS       Max concurrent sequences (default: 20)
#   VLLM_API_KEY            API key (optional, enables auth)
#   VLLM_SERVED_MODEL_NAME  Served model name alias (default: same as VLLM_MODEL)
#
# Examples:
#   # Default: serve Qwen3.5-35B-A3B-NVFP4
#   docker run ... nemoclaw-thor/vllm:bundled
#
#   # Custom model or port
#   docker run ... -e VLLM_MODEL=myorg/MyModel -e VLLM_PORT=9000 nemoclaw-thor/vllm:bundled
#
#   # Full override (pass explicit vllm serve args)
#   docker run ... nemoclaw-thor/vllm:bundled vllm serve mymodel --port 9000 --attention-backend triton_attn
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
MODEL="${VLLM_MODEL:-Kbenkhaled/Qwen3.5-35B-A3B-NVFP4}"
SERVED_NAME="${VLLM_SERVED_MODEL_NAME:-${MODEL}}"

ARGS=(
    vllm serve "${MODEL}"
    --served-model-name "${SERVED_NAME}"
    --host "${VLLM_HOST:-0.0.0.0}"
    --port "${VLLM_PORT:-8000}"
    --download-dir /data/models/huggingface/hub
    --attention-backend triton_attn
    --language-model-only
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_xml
    --enable-prefix-caching
    --max-model-len "${VLLM_MAX_MODEL_LEN:-65536}"
    --kv-cache-dtype "${VLLM_KV_CACHE_DTYPE:-fp8}"
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.8}"
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-20}"
    --max-num-batched-tokens 4096
    --compilation-config '{"custom_ops":["-quant_fp8","-quant_fp8","-quant_fp8"]}'
)

if [ -n "${VLLM_API_KEY}" ] && [ "${VLLM_API_KEY}" != "dummy" ]; then
    ARGS+=(--api-key "${VLLM_API_KEY}")
fi

# Chat template: bundled at /opt/nemoclaw-thor/templates/qwen3-tool-call-compat.jinja
TEMPLATE=/opt/nemoclaw-thor/templates/qwen3-tool-call-compat.jinja
if [ -f "$TEMPLATE" ]; then
    ARGS+=(--chat-template "$TEMPLATE")
fi

echo "[entrypoint] Starting vLLM for SM110 (Qwen3.5 NVFP4 profile)"
echo "[entrypoint] Model: ${MODEL}"
echo "[entrypoint] Bind: ${VLLM_HOST:-0.0.0.0}:${VLLM_PORT:-8000}"
exec "${ARGS[@]}"
