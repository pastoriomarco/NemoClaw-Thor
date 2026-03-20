#!/usr/bin/env bash
# configure-local-provider.sh — Configure NemoClaw/OpenShell to use a local vLLM endpoint
#
# Usage:
#   ./configure-local-provider.sh [model-profile]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/checks.sh"
source "${SCRIPT_DIR}/lib/config.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: ./configure-local-provider.sh [model-profile]"
    echo ""
    print_supported_model_profiles
    exit 0
fi

load_thor_runtime_config "${1:-}"

echo ""
echo -e "${BOLD}NemoClaw-Thor Local Provider Configuration${NC}"
echo -e "JetsonHacks fork — /home/tndlux/workspaces/thor_llm/src/NemoClaw-Thor"
echo ""
print_thor_runtime_config
echo ""

if ! command -v openshell &>/dev/null; then
    fail "openshell command not found"
    fix "Install OpenShell and apply the openshell-thor fixes before continuing."
    exit 1
fi

if ! openshell gateway info &>/dev/null; then
    fail "OpenShell gateway is not running"
    fix "Run: openshell gateway start"
    exit 1
fi

if openshell provider get "${THOR_LOCAL_PROVIDER_NAME}" &>/dev/null; then
    pass "Provider ${THOR_LOCAL_PROVIDER_NAME} already exists"
else
    info "Creating provider ${THOR_LOCAL_PROVIDER_NAME}..."
    openshell provider create \
        --name "${THOR_LOCAL_PROVIDER_NAME}" \
        --type openai \
        --credential OPENAI_API_KEY="${THOR_LOCAL_VLLM_API_KEY}" \
        --config OPENAI_BASE_URL="${THOR_LOCAL_VLLM_BASE_URL}"
    pass "Provider ${THOR_LOCAL_PROVIDER_NAME} created"
fi

info "Setting inference route to ${THOR_MODEL_ID}..."
openshell inference set \
    --provider "${THOR_LOCAL_PROVIDER_NAME}" \
    --model "${THOR_MODEL_ID}" \
    --no-verify
pass "Inference route set to ${THOR_LOCAL_PROVIDER_NAME} / ${THOR_MODEL_ID}"

save_thor_runtime_config
pass "Saved runtime config to ${THOR_CONFIG_FILE}"

vllm_response=$(curl -s --max-time 5 \
    -H "Authorization: Bearer ${THOR_LOCAL_VLLM_API_KEY}" \
    "${THOR_HOST_VLLM_MODELS_URL}" 2>/dev/null || echo "")
if [[ -z "${vllm_response}" ]]; then
    warn "Host vLLM endpoint is not reachable yet"
    fix "Start the local model server, then run ./status.sh"
else
    if echo "${vllm_response}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
models = [m['id'] for m in data.get('data', [])]
sys.exit(0 if '${THOR_MODEL_ID}' in models else 1)
" 2>/dev/null; then
        pass "Host vLLM endpoint is serving ${THOR_MODEL_ID}"
    else
        actual=$(echo "${vllm_response}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
models = [m['id'] for m in data.get('data', [])]
print(', '.join(models) if models else 'unknown')
" 2>/dev/null || echo "unknown")
        warn "Host vLLM endpoint is reachable but serving: ${actual}"
        fix "Adjust --served-model-name in the thor_llm launch command, or override THOR_MODEL_ID."
    fi
fi

echo ""
echo -e "${GREEN}${BOLD}  Local provider configuration complete.${NC}"
echo ""
echo "  Next:"
echo "    ./status.sh ${THOR_MODEL_PROFILE}"
echo ""
