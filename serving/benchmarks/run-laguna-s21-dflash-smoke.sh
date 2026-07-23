#!/bin/bash
# Smoke-launch Laguna-S-2.1-NVFP4 + DFlash drafter on Thor (sm_110).
# Standalone on purpose: first Thor run of this model family anywhere; promote
# to a config.sh/launch.sh profile only after this proves out.
#
# Recipe basis: recipes.vllm.ai/poolside/Laguna-S-2.1 (validated on vLLM
# 0.25.1), with community corrections:
#   - num_speculative_tokens 7, not poolside's original 15: positions 6-15
#     accept at ~0%, so 15 wastes drafter compute (poolside has since adopted 7).
#   - poolside says "--moe-backend triton" but v0.25.1 rejects triton for
#     NvFP4 MoE (valid: cutlass, flashinfer_trtllm, flashinfer_cutlass,
#     flashinfer_cutedsl, flashinfer_b12x, marlin, humming, emulation).
#     The triton advice targets the FP8 variant's DeepGEMM conflict; for NVFP4
#     we use marlin, the backend already proven on Thor by the qwen 35B NVFP4
#     profile. Drafter is dense, so no drafter-side moe_backend needed.
#   - NVFP4 quantization is auto-detected from quantization_config; no
#     --quantization flag.
#   - NVFP4 KV cache is sm_100/sm_103-only; fp8 KV is the Thor path.
# Unknowns this smoke run answers: triton-MoE NVFP4 GEMM coverage on sm_110,
# poolside_v1 parser availability in 0.25.1, DFlash acceptance on our workloads.
#
# Fallbacks if boot fails, in order:
#   1. -e VLLM_USE_V2_MODEL_RUNNER=1   (hybrid SWA/full DFlash drafters, PR #47914)
#   2. --kv-cache-dtype auto           (if fp8 conflicts with checkpoint quant cfg)
#   3. --moe-backend flashinfer_cutlass / marlin  (if triton NVFP4 unsupported on sm_110;
#      marlin likely breaks DFlash -- test without spec config first)
#   4. --tool-call-parser glm47        (XS card's fallback for pre-#47311 builds)
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-laguna-s21-dflash-smoke}"
THOR_HF_CACHE_DIR="${THOR_HF_CACHE_DIR:-$HOME/thor-hf-cache}"
THOR_VLLM_CACHE_DIR="${THOR_VLLM_CACHE_DIR:-$HOME/thor-vllm-cache}"
THOR_TORCH_CACHE_DIR="${THOR_TORCH_CACHE_DIR:-$HOME/thor-torch-cache}"
THOR_FLASHINFER_CACHE_DIR="${THOR_FLASHINFER_CACHE_DIR:-$HOME/thor-flashinfer-cache}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.25.1}"
PORT="${PORT:-8000}"
# Tunables (override via env for sweep runs)
NUM_SPEC="${NUM_SPEC:-7}"
KV_DTYPE="${KV_DTYPE:-fp8}"
MAX_LEN="${MAX_LEN:-131072}"
BATCHED_TOKENS="${BATCHED_TOKENS:-8192}"
GPU_UTIL="${GPU_UTIL:-0.80}"
THINKING="${THINKING:-true}"

HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"

docker run -d --rm --name "$CONTAINER_NAME" --entrypoint "" \
    --runtime nvidia --gpus all \
    --ipc=host --network host \
    -e NVIDIA_DISABLE_REQUIRE=true \
    -e HF_HOME=/data/models/huggingface \
    -e HF_HUB_CACHE=/data/models/huggingface/hub \
    -e TRANSFORMERS_CACHE=/data/models/huggingface/hub \
    ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
    -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor \
    -e TORCHINDUCTOR_COMPILE_THREADS=4 \
    -e MAX_JOBS=8 \
    -e CUTE_DSL_ARCH=sm_110a \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -v "${THOR_HF_CACHE_DIR}:/data/models/huggingface" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -v "${THOR_VLLM_CACHE_DIR}:/root/.cache/vllm" \
    -v "${THOR_TORCH_CACHE_DIR}:/root/.cache/torch" \
    -v "${THOR_FLASHINFER_CACHE_DIR}:/root/.cache/flashinfer" \
    "$IMAGE" \
    vllm serve poolside/Laguna-S-2.1-NVFP4 \
        --download-dir /data/models/huggingface/hub \
        --host 0.0.0.0 --port "$PORT" \
        --trust-remote-code \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --kv-cache-dtype "$KV_DTYPE" \
        --max-num-batched-tokens "$BATCHED_TOKENS" \
        --moe-backend marlin \
        --enable-auto-tool-choice \
        --tool-call-parser poolside_v1 \
        --reasoning-parser poolside_v1 \
        --default-chat-template-kwargs '{"enable_thinking": '"$THINKING"'}' \
        --enable-prefix-caching \
        --speculative-config '{"method":"dflash","model":"poolside/Laguna-S-2.1-DFlash-NVFP4","num_speculative_tokens":'"$NUM_SPEC"'}'

echo "Launched container ${CONTAINER_NAME} (port ${PORT}). Follow with:"
echo "  docker logs -f ${CONTAINER_NAME}"
