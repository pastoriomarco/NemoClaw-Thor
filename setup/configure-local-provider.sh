#!/usr/bin/env bash
# configure-local-provider.sh — Configure NemoClaw/OpenShell to use a local vLLM endpoint
#
# Usage:
#   ./configure-local-provider.sh [model-profile]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/checks.sh"
source "${SCRIPT_DIR}/lib/config.sh"
source "${SCRIPT_DIR}/lib/sandbox-runtime.sh"

THOR_INFERENCE_ROUTE_TIMEOUT_SECONDS="${THOR_INFERENCE_ROUTE_TIMEOUT_SECONDS:-180}"

# Parse flags
_mux_flag_set="false"
while [[ "${1:-}" == --* ]]; do
    case "${1}" in
        --with-manyforge-mux)
            THOR_MANYFORGE_MUX_ENABLED="true"
            _mux_flag_set="true"
            shift
            ;;
        --without-manyforge-mux)
            THOR_MANYFORGE_MUX_ENABLED="false"
            _mux_flag_set="true"
            shift
            ;;
        -h|--help)
            echo "Usage: ./configure-local-provider.sh [OPTIONS] [model-profile]"
            echo ""
            echo "Options:"
            echo "  --with-manyforge-mux      Route inference.local through the ManyForge mux"
            echo "                            proxy (port 8888) for ManyForge MCP tool access"
            echo "  --without-manyforge-mux   Restore direct vLLM routing (port 8000)"
            echo ""
            print_supported_model_profiles
            exit 0
            ;;
        *)
            echo "Unknown flag: ${1}" >&2
            exit 1
            ;;
    esac
done

load_thor_runtime_config "${1:-}"

# When a mux flag was explicitly passed, force the base URL to match.
# This overrides any stale URL from a previous config.env save.
if [[ "${_mux_flag_set}" == "true" ]]; then
    if [[ "${THOR_MANYFORGE_MUX_ENABLED}" == "true" ]]; then
        THOR_LOCAL_VLLM_BASE_URL="http://host.openshell.internal:${THOR_MANYFORGE_MUX_PORT}/v1"
    else
        THOR_LOCAL_VLLM_BASE_URL="http://host.openshell.internal:8000/v1"
    fi
fi

echo ""
echo -e "${BOLD}NemoClaw-Thor Local Provider Configuration${NC}"
echo "Repo: ${SCRIPT_DIR}"
echo ""
print_thor_runtime_config
echo ""

if ! command -v openshell &>/dev/null; then
    fail "openshell command not found"
    fix "Install OpenShell and apply the openshell-thor fixes before continuing."
    exit 1
fi

if ! ensure_openshell_gateway_running; then
    exit 1
fi

sandbox_name=$(resolve_thor_sandbox_name 2>/dev/null || echo "")
if [[ -n "${sandbox_name}" ]]; then
    THOR_MANAGED_SANDBOX_NAME="${sandbox_name}"
fi

if openshell provider get "${THOR_LOCAL_PROVIDER_NAME}" &>/dev/null; then
    info "Updating provider ${THOR_LOCAL_PROVIDER_NAME}..."
    openshell provider update "${THOR_LOCAL_PROVIDER_NAME}" \
        --credential OPENAI_API_KEY="${THOR_LOCAL_VLLM_API_KEY}" \
        --config OPENAI_BASE_URL="${THOR_LOCAL_VLLM_BASE_URL}"
    pass "Provider ${THOR_LOCAL_PROVIDER_NAME} updated"
else
    info "Creating provider ${THOR_LOCAL_PROVIDER_NAME}..."
    openshell provider create \
        --name "${THOR_LOCAL_PROVIDER_NAME}" \
        --type openai \
        --credential OPENAI_API_KEY="${THOR_LOCAL_VLLM_API_KEY}" \
        --config OPENAI_BASE_URL="${THOR_LOCAL_VLLM_BASE_URL}"
    pass "Provider ${THOR_LOCAL_PROVIDER_NAME} created"
fi
THOR_MANAGED_PROVIDER_NAMES=$(csv_append_unique "${THOR_MANAGED_PROVIDER_NAMES:-}" "${THOR_LOCAL_PROVIDER_NAME}")

info "Setting inference route to ${THOR_MODEL_ID}..."
openshell inference set \
    --provider "${THOR_LOCAL_PROVIDER_NAME}" \
    --model "${THOR_MODEL_ID}" \
    --timeout "${THOR_INFERENCE_ROUTE_TIMEOUT_SECONDS}" \
    --no-verify
pass "Inference route set to ${THOR_LOCAL_PROVIDER_NAME} / ${THOR_MODEL_ID} (${THOR_INFERENCE_ROUTE_TIMEOUT_SECONDS}s)"

if [[ -n "${sandbox_name}" ]]; then
    info "Syncing sandbox runtime config for ${sandbox_name}..."
    sync_sandbox_runtime_config "${sandbox_name}"
    pass "Sandbox runtime config synced to ${THOR_MODEL_ID}"
else
    warn "Could not resolve a sandbox name to sync internal NemoClaw config"
    fix "Set THOR_MANAGED_SANDBOX_NAME in ${THOR_CONFIG_FILE}, or rerun install/onboard."
fi

save_thor_runtime_config
pass "Saved runtime config to ${THOR_CONFIG_FILE}"

# Update the host-side NemoClaw registry so `nemoclaw connect` doesn't
# swap the inference route back to the onboard-time cloud provider.
# NemoClaw v0.0.18+ reads provider/model from ~/.nemoclaw/sandboxes.json
# and auto-corrects the route on connect if it doesn't match.
if [[ -n "${sandbox_name}" ]]; then
    local_registry="${HOME}/.nemoclaw/sandboxes.json"
    if [[ -f "${local_registry}" ]]; then
        python3 - "${local_registry}" "${sandbox_name}" "${THOR_LOCAL_PROVIDER_NAME}" "${THOR_MODEL_ID}" <<'PYEOF'
import json, sys
from pathlib import Path

reg_path = Path(sys.argv[1])
sandbox_name = sys.argv[2]
provider_name = sys.argv[3]
model_id = sys.argv[4]

data = json.loads(reg_path.read_text(encoding="utf-8"))
sandboxes = data.get("sandboxes", {})

if sandbox_name in sandboxes:
    entry = sandboxes[sandbox_name]
    if isinstance(entry, dict):
        entry["provider"] = provider_name
        entry["model"] = model_id
    reg_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"  Registry updated: {sandbox_name} → {provider_name}/{model_id}")
else:
    print(f"  Registry: sandbox '{sandbox_name}' not found (skipped)")
PYEOF
    fi
fi

# Apply local-inference policy preset so the sandbox can reach vLLM.
# NemoClaw v0.0.18+ includes the local-inference preset which allows
# egress to host.openshell.internal:8000 (vLLM) and :11434 (Ollama).
if [[ -n "${sandbox_name}" ]]; then
    if nemoclaw "${sandbox_name}" policy-add local-inference 2>/dev/null; then
        pass "Applied local-inference policy preset"
    else
        info "local-inference preset already applied or unavailable"
    fi
fi

info "Ensuring OpenClaw gateway is running inside the sandbox..."
ensure_sandbox_gateway_running "${sandbox_name}" || true
echo ""

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

# Send a warmup request so the first real interaction doesn't stall.
if [[ -n "${vllm_response}" ]]; then
    info "Sending warmup request to ${THOR_MODEL_ID}..."
    warmup_reply=$(curl -s --max-time 120 \
        -H "Authorization: Bearer ${THOR_LOCAL_VLLM_API_KEY}" \
        -H "Content-Type: application/json" \
        "${THOR_LOCAL_VLLM_BASE_URL}/chat/completions" \
        -d "{
            \"model\": \"${THOR_MODEL_ID}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Reply with one word: ready\"}],
            \"max_tokens\": 128
        }" 2>/dev/null || echo "")

    if echo "${warmup_reply}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
msg = data['choices'][0]['message']
content = msg.get('content') or ''
reasoning = msg.get('reasoning') or msg.get('reasoning_content') or ''
sys.exit(0 if content.strip() or reasoning.strip() else 1)
" 2>/dev/null; then
        pass "Model warmup complete"
    else
        warn "Warmup request did not return a reply — first interaction may be slow"
    fi
fi

echo ""
echo -e "${GREEN}${BOLD}  Local provider configuration complete.${NC}"
echo ""
echo "  Next:"
echo "    ./status.sh ${THOR_MODEL_PROFILE}"
echo ""
