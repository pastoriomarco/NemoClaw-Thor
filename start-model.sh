#!/usr/bin/env bash
# start-model.sh — Start a local Qwen vLLM server for NemoClaw-Thor
#
# Usage:
#   ./start-model.sh [model-profile]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/checks.sh"
source "${SCRIPT_DIR}/lib/config.sh"
source "${SCRIPT_DIR}/lib/launch.sh"

for arg in "$@"; do
    if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
        echo "Usage: ./start-model.sh [model-profile]"
        echo ""
        print_supported_model_profiles
        echo ""
        echo "Environment overrides:"
        echo "  THOR_MAX_MODEL_LEN"
        echo "  THOR_MAX_NUM_SEQS"
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
echo -e "JetsonHacks fork — /home/tndlux/workspaces/thor_llm/src/NemoClaw-Thor"
echo ""
print_thor_launch_summary
echo ""

if ! check_thor_launch_prereqs; then
    exit 1
fi

save_thor_runtime_config

info "Starting local vLLM server..."
info "OpenShell should point to: ${THOR_LOCAL_VLLM_BASE_URL}"
info "Stop with Ctrl-C."
echo ""

run_thor_vllm_container
