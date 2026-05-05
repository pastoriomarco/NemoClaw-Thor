#!/usr/bin/env bash
# start-model.sh — Start a local inference server for NemoClaw-Thor
#
# Usage:
#   ./serving/start-model.sh [model-profile]

set -euo pipefail

# Auto-read HF_TOKEN from the standard cache file so gated-repo drafters
# (e.g. z-lab/Qwen3.6-27B-DFlash) can be downloaded without manual export.
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/setup/checks.sh"
source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/launch.sh"

for arg in "$@"; do
    if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
        echo "Usage: ./serving/start-model.sh [model-profile]"
        echo ""
        print_supported_model_profiles
        echo ""
        echo "Environment overrides:"
        echo "  THOR_MAX_MODEL_LEN"
        echo "  THOR_MAX_NUM_SEQS"
        echo "  THOR_OPENCLAW_MAIN_MAX_CONCURRENT"
        echo "  THOR_KV_CACHE_DTYPE"
        echo "  THOR_GPU_MEMORY_UTILIZATION"
        echo "  THOR_MAX_NUM_BATCHED_TOKENS"
        echo "  THOR_LOCAL_VLLM_API_KEY"
        echo "  THOR_VLLM_IMAGE"
        exit 0
    fi
done

load_thor_runtime_config "${1:-}"
prepare_thor_launch_profile "${THOR_MODEL_PROFILE}"

echo ""
echo -e "${BOLD}NemoClaw-Thor Local Model Launcher${NC}"
echo "Repo: ${REPO_ROOT}"
echo ""
print_thor_launch_summary
echo ""

if ! check_thor_launch_prereqs; then
    exit 1
fi

save_thor_runtime_config

if [[ "${THOR_LAUNCH_BACKEND:-vllm}" == "llamacpp" ]]; then
    info "Starting local llama-server..."
    info "OpenClaw sandbox base URL: ${THOR_OPENCLAW_BASE_URL}"
    info "OpenShell provider target: ${THOR_LOCAL_VLLM_BASE_URL}"
    info "Stop with Ctrl-C."
    echo ""
    run_thor_llamacpp_container
else
    info "Starting local vLLM server..."
    info "OpenClaw sandbox base URL: ${THOR_OPENCLAW_BASE_URL}"
    info "OpenShell provider target: ${THOR_LOCAL_VLLM_BASE_URL}"
    info "Stop with Ctrl-C."
    echo ""
    run_thor_vllm_container
fi
