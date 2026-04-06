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
source "${SCRIPT_DIR}/lib/egress-firewall.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: ./configure-local-provider.sh [model-profile]"
    echo ""
    print_supported_model_profiles
    exit 0
fi

load_thor_runtime_config "${1:-}"

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
    --no-verify
pass "Inference route set to ${THOR_LOCAL_PROVIDER_NAME} / ${THOR_MODEL_ID}"

if [[ -n "${sandbox_name}" ]]; then
    # --- SSH handshake reconciliation ---
    # What: reads the gateway's SSH handshake secret, compares it to the
    #   sandbox pod's copy, and if they differ, patches the sandbox CRD and
    #   deletes the pod so k8s recreates it with the new secret.
    # Why it existed: on v3 the gateway rotated its handshake secret on every
    #   restart. The sandbox pod kept the old one, so `nemoclaw connect` failed
    #   with a cryptic SSH auth error. This forced a pod restart to resync.
    # Why it may not be needed: OpenShell 0.0.22 sandbox survival means the
    #   pod persists across gateway restarts. If the secret doesn't drift when
    #   the pod survives, this is unnecessary. Needs testing.
    #
    # info "Reconciling sandbox SSH connect state for ${sandbox_name}..."
    # reconcile_sandbox_ssh_handshake_secret "${sandbox_name}"
    # pass "Sandbox SSH connect state is healthy"

    info "Syncing sandbox runtime config for ${sandbox_name}..."
    sync_sandbox_runtime_config "${sandbox_name}"
    pass "Sandbox runtime config synced to ${THOR_MODEL_ID}"

    # --- NemoClaw registry update ---
    # What: writes the current model_id and provider_name into
    #   ~/.nemoclaw/sandboxes.json under the sandbox's entry.
    # Why it existed: the `nemoclaw` CLI reads this file to show which model
    #   a sandbox is using in `nemoclaw list` output. Without it the CLI shows
    #   stale info from the last onboard.
    # Why it may not be needed: purely cosmetic — no runtime effect. The agent
    #   doesn't read this file. Only matters if you care about `nemoclaw list`
    #   showing the correct model name after switching profiles.
    #
    # registry_file="$(thor_nemoclaw_registry_file)"
    # if [[ -f "${registry_file}" ]]; then
    #     python3 - "${registry_file}" "${sandbox_name}" "${THOR_MODEL_ID}" "${THOR_LOCAL_PROVIDER_NAME}" <<'PYEOF'
    # ...
    # PYEOF
    # fi
else
    warn "Could not resolve a sandbox name to sync internal NemoClaw config"
    fix "Set THOR_MANAGED_SANDBOX_NAME in ${THOR_CONFIG_FILE}, or rerun install/onboard."
fi

save_thor_runtime_config
pass "Saved runtime config to ${THOR_CONFIG_FILE}"

# --- OpenClaw gateway lifecycle ---
# What: SSHes into the sandbox, kills any existing openclaw gateway process,
#   pre-creates a device identity + paired.json so the TUI doesn't prompt for
#   manual device approval, then starts `openclaw gateway run --auth none` in
#   the background. Also sets up a host port forward (18789) and probes the
#   gateway to verify it's responding to HTTP.
# Why it existed: on v3 the gateway process died on every pod restart (no
#   sandbox survival). Device identities were lost too (stored in pod tmpfs).
#   Without pre-pairing, the TUI showed "approve this device?" on every
#   reconnect. The --auth none flag bypassed gateway auth entirely.
# Why it may not be needed: with sandbox survival (0.0.22), the gateway
#   process and identity files should persist. Also `nemoclaw connect` may
#   handle gateway startup natively in v0.0.6+. Needs testing — if connect
#   works without this, the entire 200-line function is dead code.
#
# info "Ensuring OpenClaw gateway is running inside the sandbox..."
# ensure_sandbox_gateway_running "${sandbox_name}" || true
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

# --- Egress firewall ---
# What: injects iptables rules into the k3s node's FORWARD chain to DROP all
#   outbound traffic from the sandbox pod except DNS, OpenShell gateway, and
#   host vLLM (172.17.0.1:8000). This is kernel-level enforcement.
# Why it existed: OpenShell policy enforcement is application-level only —
#   it tells the agent "you shouldn't access X" but doesn't block the network.
#   K3s Flannel CNI doesn't enforce NetworkPolicy. Kernel 6.8 doesn't support
#   Landlock network filtering. Without iptables, the sandbox has full internet.
# Why it's commented out: need to first assess what the vanilla onboard
#   sandbox can actually reach. The upstream base policy may have changed in
#   NemoClaw v0.0.6. Also needs testing whether the iptables rules survive
#   pod restarts with sandbox survival. Run ./enforce-egress-firewall.sh
#   separately once we understand the exposure.
#
# cluster_container="$(thor_openshell_cluster_container_name 2>/dev/null || true)"
# if [[ -n "${cluster_container}" && -n "${sandbox_name}" ]]; then
#     info "Enforcing egress firewall on sandbox ${sandbox_name}..."
#     enforce_sandbox_egress_firewall "${cluster_container}" "${sandbox_name}" || \
#         warn "Could not apply egress firewall — sandbox may have unrestricted internet"
# fi

echo ""
echo -e "${GREEN}${BOLD}  Local provider configuration complete.${NC}"
echo ""
echo "  Next:"
echo "    ./status.sh ${THOR_MODEL_PROFILE}"
echo ""
