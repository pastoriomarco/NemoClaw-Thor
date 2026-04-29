#!/bin/bash
# Run IFEval (HuggingFace Open LLM Leaderboard variant) against TRT-Edge-LLM Omni server.
# Standardized, citable, used by every frontier model release.
set -euo pipefail

BENCH_DIR="${BENCH_DIR:-/home/tndlux/agentic-bench}"
RESULTS_DIR="${BENCH_DIR}/results"
LOG_DIR="${BENCH_DIR}/logs"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL_ID="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
LIMIT="${LIMIT:-}"  # set LIMIT=50 for a quick pipeline test, empty for full run

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

LIMIT_ARGS=()
[ -n "${LIMIT}" ] && LIMIT_ARGS=(--limit "${LIMIT}")

echo "[$(date +%H:%M:%S)] Starting IFEval against ${ENDPOINT}"
echo "  Model: ${MODEL_ID}"
echo "  Limit: ${LIMIT:-full (541 prompts)}"

# Point HF cache at the populated dir where the model already lives. Avoids a
# fresh re-download into ~/.cache/huggingface (which was written-to by docker-
# as-root in earlier sessions and is now permission-blocked for this user).
export HF_HOME="${HF_HOME:-/home/tndlux/thor-hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/tndlux/thor-hf-cache/hub}"

"${BENCH_DIR}/.venv/bin/lm-eval" run \
    --model local-chat-completions \
    --model_args "model=${MODEL_ID},base_url=${ENDPOINT},num_concurrent=1,max_retries=3,tokenizer_backend=huggingface,tokenizer=${MODEL_ID}" \
    --tasks leaderboard_instruction_following \
    --batch_size 1 \
    --gen_kwargs "temperature=0.0,max_tokens=1280" \
    --apply_chat_template \
    --output_path "${RESULTS_DIR}/ifeval-trt-omni" \
    --log_samples \
    "${LIMIT_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/ifeval-trt-omni.log"

echo "[$(date +%H:%M:%S)] IFEval complete. Results in ${RESULTS_DIR}/ifeval-trt-omni/"
