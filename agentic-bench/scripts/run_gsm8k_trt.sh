#!/bin/bash
# GSM8K-CoT zero-shot against TRT-Edge-LLM Omni server.
# Standardized math-reasoning benchmark, exact-match scoring.
set -euo pipefail

BENCH_DIR="${BENCH_DIR:-/home/tndlux/agentic-bench}"
RESULTS_DIR="${BENCH_DIR}/results"
LOG_DIR="${BENCH_DIR}/logs"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL_ID="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
LIMIT="${LIMIT:-250}"  # default 250 (canonical "first N" subset for stable comparison)

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

LIMIT_ARGS=()
[ -n "${LIMIT}" ] && LIMIT_ARGS=(--limit "${LIMIT}")

export HF_HOME="${HF_HOME:-/home/tndlux/thor-hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/tndlux/thor-hf-cache/hub}"

echo "[$(date +%H:%M:%S)] Starting GSM8K-CoT zero-shot against ${ENDPOINT}"
echo "  Model: ${MODEL_ID}"
echo "  Limit: ${LIMIT:-full (1319 prompts)}"

"${BENCH_DIR}/.venv/bin/lm-eval" run \
    --model local-chat-completions \
    --model_args "model=${MODEL_ID},base_url=${ENDPOINT},num_concurrent=1,max_retries=3,tokenizer_backend=huggingface,tokenizer=${MODEL_ID}" \
    --tasks gsm8k_cot_zeroshot \
    --batch_size 1 \
    --gen_kwargs "temperature=0.0,max_tokens=1024" \
    --apply_chat_template \
    --output_path "${RESULTS_DIR}/gsm8k-trt-omni" \
    --log_samples \
    "${LIMIT_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/gsm8k-trt-omni.log"

echo "[$(date +%H:%M:%S)] GSM8K complete. Results in ${RESULTS_DIR}/gsm8k-trt-omni/"
