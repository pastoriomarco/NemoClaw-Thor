#!/usr/bin/env bash
# status.sh — NemoClaw-Thor system health check
#
# Usage:
#   ./status.sh [model-profile]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/checks.sh"
source "${SCRIPT_DIR}/lib/config.sh"
source "${SCRIPT_DIR}/lib/sandbox-runtime.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: ./status.sh [model-profile]"
    echo ""
    print_supported_model_profiles
    exit 0
fi

load_thor_runtime_config "${1:-}"

echo ""
echo -e "${BOLD}NemoClaw-Thor System Status${NC}"
echo "Repo: ${SCRIPT_DIR}"
echo ""
print_thor_runtime_config
echo ""

CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNED=0

record() {
    local ret="$1"
    case "${ret}" in
        0) CHECKS_PASSED=$((CHECKS_PASSED + 1)) ;;
        2) CHECKS_WARNED=$((CHECKS_WARNED + 1)) ;;
        *) CHECKS_FAILED=$((CHECKS_FAILED + 1)) ;;
    esac
}

header "OpenShell Gateway"
echo ""

check_openshell_installed && record 0 || record 1
check_openshell_gateway && record 0 || record 1

header "vLLM Inference Server"
echo ""

vllm_response=$(curl -s --max-time 10 \
    -H "Authorization: Bearer ${THOR_LOCAL_VLLM_API_KEY}" \
    "${THOR_HOST_VLLM_MODELS_URL}" 2>/dev/null || echo "")

if [[ -z "${vllm_response}" ]]; then
    fail "vLLM server is not reachable at ${THOR_HOST_VLLM_MODELS_URL}"
    info "The local model server is not running or is still starting up."
    fix "Start the selected model server: ./start-model.sh ${THOR_MODEL_PROFILE}"
    record 1
else
    if echo "${vllm_response}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
models = [m['id'] for m in data.get('data', [])]
sys.exit(0 if '${THOR_MODEL_ID}' in models else 1)
" 2>/dev/null; then
        pass "vLLM server is running"
        pass "Model serving as: ${THOR_MODEL_ID}"
        record 0
        record 0
    else
        actual=$(echo "${vllm_response}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
models = [m['id'] for m in data.get('data', [])]
print(', '.join(models) if models else 'unknown')
" 2>/dev/null || echo "unknown")
        warn "vLLM server is running but serving unexpected model: ${actual}"
        info "Expected model id: ${THOR_MODEL_ID}"
        fix "Adjust --served-model-name in your thor_llm launch command, or override THOR_MODEL_ID."
        record 2
    fi
fi

header "OpenShell Sandbox"
echo ""

sandbox_name=""
if ! command -v openshell &>/dev/null; then
    fail "openshell not found — cannot check sandbox status"
    record 1
else
    sandbox_list=$(openshell sandbox list 2>/dev/null || echo "")
    sandbox_name=$(resolve_thor_sandbox_name 2>/dev/null || echo "")
    sandbox_phase=""
    if [[ -n "${sandbox_name}" ]]; then
        sandbox_phase=$(echo "${sandbox_list}" \
            | sed 's/\x1b\[[0-9;]*m//g' \
            | awk -v name="${sandbox_name}" 'NR>1 && $1 == name {print $NF; exit}')
    fi

    if [[ -z "${sandbox_name}" ]]; then
        sandbox_count=$(echo "${sandbox_list}" | sed 's/\x1b\[[0-9;]*m//g' | awk 'NR>1 && $1 != "" {count++} END {print count+0}')
        if [[ "${sandbox_count}" -gt 0 ]]; then
            fail "Could not determine which sandbox belongs to this NemoClaw-Thor install"
            fix "Set THOR_MANAGED_SANDBOX_NAME in ${THOR_CONFIG_FILE}, or keep only one sandbox present."
        else
            fail "No sandbox found"
            info "Run ./install.sh ${THOR_MODEL_PROFILE} to create a sandbox."
        fi
        record 1
    elif [[ -z "${sandbox_phase}" ]]; then
        fail "Tracked sandbox '${sandbox_name}' was not found in openshell sandbox list"
        fix "Update THOR_MANAGED_SANDBOX_NAME in ${THOR_CONFIG_FILE}, or re-run install.sh."
        record 1
    elif [[ "${sandbox_phase}" == "Ready" ]]; then
        pass "Sandbox '${sandbox_name}' is Ready"
        record 0
    else
        fail "Sandbox '${sandbox_name}' is in phase: ${sandbox_phase}"
        fix "Check logs: nemoclaw ${sandbox_name} logs --follow"
        fix "Check status: openshell sandbox list"
        record 1
    fi
fi

header "Native Connect Path"
echo ""

if [[ -z "${sandbox_name}" ]]; then
    warn "Skipping native connect check because no sandbox is resolved"
    record 2
else
    gateway_secret=$(gateway_ssh_handshake_secret 2>/dev/null || echo "")
    sandbox_secret=$(sandbox_ssh_handshake_secret "${sandbox_name}" 2>/dev/null || echo "")

    if [[ -z "${gateway_secret}" || -z "${sandbox_secret}" ]]; then
        warn "Could not inspect sandbox SSH handshake state"
        fix "Run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
        record 2
    elif [[ "${gateway_secret}" == "${sandbox_secret}" ]]; then
        pass "Sandbox SSH handshake secret matches the gateway"
        record 0
    else
        fail "Sandbox SSH handshake secret does not match the gateway"
        info "Native 'nemoclaw thor-assistant connect' will fail until the sandbox is resynced."
        fix "Run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
        record 1
    fi
fi

header "Inference Route"
echo ""

if ! command -v openshell &>/dev/null; then
    fail "openshell not found — cannot check inference route"
    record 1
else
    inference=$(openshell inference get 2>/dev/null || echo "")
    provider=$(echo "${inference}" | grep 'Provider:' | awk '{print $2}' || true)
    model=$(echo "${inference}"    | grep 'Model:'    | awk '{print $2}' || true)

    if [[ -z "${provider}" ]]; then
        fail "No inference route configured"
        fix "Run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
        record 1
    elif [[ "${provider}" == "${THOR_LOCAL_PROVIDER_NAME}" ]]; then
        pass "Inference provider: ${provider}"
        if [[ "${model}" == "${THOR_MODEL_ID}" ]]; then
            pass "Inference model:    ${model}"
            record 0
            record 0
        else
            warn "Inference model is '${model}' — expected '${THOR_MODEL_ID}'"
            fix "Run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
            record 0
            record 2
        fi
    else
        warn "Inference provider is '${provider}' — expected '${THOR_LOCAL_PROVIDER_NAME}'"
        info "The stack is not currently pointing at the saved local vLLM provider."
        fix "Run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
        record 2
    fi
fi

header "Sandbox Runtime Config"
echo ""

if [[ -z "${sandbox_name}" ]]; then
    warn "Skipping sandbox runtime config check because no sandbox is resolved"
    record 2
else
    runtime_summary=$(sandbox_runtime_config_summary_json "${sandbox_name}" 2>/dev/null || echo "")
    if [[ -z "${runtime_summary}" ]]; then
        warn "Could not inspect runtime config inside sandbox '${sandbox_name}'"
        fix "Re-run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
        record 2
    else
        onboard_model=$(printf '%s' "${runtime_summary}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
print(data.get("onboard_model") or "")
' 2>/dev/null || echo "")
        primary_model=$(printf '%s' "${runtime_summary}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
print(data.get("openclaw_primary_model") or "")
' 2>/dev/null || echo "")
        openclaw_api=$(printf '%s' "${runtime_summary}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
print(data.get("openclaw_inference_api") or "")
' 2>/dev/null || echo "")
        openclaw_context_window=$(printf '%s' "${runtime_summary}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
value = data.get("openclaw_inference_context_window")
print("" if value is None else value)
' 2>/dev/null || echo "")
        openclaw_parallel_tool_calls=$(printf '%s' "${runtime_summary}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
value = data.get("openclaw_parallel_tool_calls")
print("" if value is None else str(value).lower())
' 2>/dev/null || echo "")
        openclaw_main_max_concurrent=$(printf '%s' "${runtime_summary}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
value = data.get("openclaw_main_max_concurrent")
print("" if value is None else value)
' 2>/dev/null || echo "")
        openclaw_subagents_max_concurrent=$(printf '%s' "${runtime_summary}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
value = data.get("openclaw_subagents_max_concurrent")
print("" if value is None else value)
' 2>/dev/null || echo "")
        openclaw_subagents_max_children=$(printf '%s' "${runtime_summary}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
value = data.get("openclaw_subagents_max_children")
print("" if value is None else value)
' 2>/dev/null || echo "")
        openclaw_subagents_max_spawn_depth=$(printf '%s' "${runtime_summary}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
value = data.get("openclaw_subagents_max_spawn_depth")
print("" if value is None else value)
' 2>/dev/null || echo "")
        openclaw_temperature=$(printf '%s' "${runtime_summary}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
value = data.get("openclaw_temperature")
print("" if value is None else value)
' 2>/dev/null || echo "")

        if [[ "${onboard_model}" == "${THOR_MODEL_ID}" ]]; then
            pass "Sandbox onboard model: ${onboard_model}"
            record 0
        else
            warn "Sandbox onboard model is '${onboard_model:-unknown}' — expected '${THOR_MODEL_ID}'"
            fix "Re-run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
            record 2
        fi

        if [[ "${primary_model}" == "inference/${THOR_MODEL_ID}" ]]; then
            pass "Sandbox primary model: ${primary_model}"
            record 0
        else
            warn "Sandbox primary model is '${primary_model:-unknown}' — expected 'inference/${THOR_MODEL_ID}'"
            fix "Re-run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
            record 2
        fi

        if [[ "${openclaw_api}" == "openai-completions" ]]; then
            pass "Sandbox provider API: ${openclaw_api}"
            record 0
        else
            warn "Sandbox provider API is '${openclaw_api:-unknown}' — expected 'openai-completions'"
            fix "Re-run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
            record 2
        fi

        if [[ "${openclaw_context_window}" == "${THOR_TARGET_MAX_MODEL_LEN}" ]]; then
            pass "Sandbox context window: ${openclaw_context_window}"
            record 0
        else
            warn "Sandbox context window is '${openclaw_context_window:-unknown}' — expected '${THOR_TARGET_MAX_MODEL_LEN}'"
            fix "Re-run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
            record 2
        fi

        if [[ "${openclaw_parallel_tool_calls}" == "false" ]]; then
            pass "Sandbox parallel tool calls disabled"
            record 0
        else
            warn "Sandbox parallel tool calls setting is '${openclaw_parallel_tool_calls:-unknown}' — expected 'false'"
            fix "Re-run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
            record 2
        fi

        if [[ "${openclaw_temperature}" == "0" ]]; then
            pass "Sandbox temperature: ${openclaw_temperature}"
            record 0
        else
            warn "Sandbox temperature is '${openclaw_temperature:-unknown}' — expected '0'"
            fix "Re-run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
            record 2
        fi

        if [[ "${openclaw_main_max_concurrent}" == "${THOR_EFFECTIVE_OPENCLAW_MAIN_MAX_CONCURRENT}" ]]; then
            pass "Sandbox main max concurrent: ${openclaw_main_max_concurrent}"
            record 0
        else
            warn "Sandbox main max concurrent is '${openclaw_main_max_concurrent:-unknown}' — expected '${THOR_EFFECTIVE_OPENCLAW_MAIN_MAX_CONCURRENT}'"
            fix "Re-run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
            record 2
        fi

        if [[ "${openclaw_subagents_max_concurrent}" == "${THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CONCURRENT}" ]]; then
            pass "Sandbox subagent max concurrent: ${openclaw_subagents_max_concurrent}"
            record 0
        else
            warn "Sandbox subagent max concurrent is '${openclaw_subagents_max_concurrent:-unknown}' — expected '${THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CONCURRENT}'"
            fix "Re-run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
            record 2
        fi

        if [[ "${openclaw_subagents_max_children}" == "${THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CHILDREN}" ]]; then
            pass "Sandbox max children per agent: ${openclaw_subagents_max_children}"
            record 0
        else
            warn "Sandbox max children per agent is '${openclaw_subagents_max_children:-unknown}' — expected '${THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CHILDREN}'"
            fix "Re-run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
            record 2
        fi

        if [[ "${openclaw_subagents_max_spawn_depth}" == "${THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_SPAWN_DEPTH}" ]]; then
            pass "Sandbox max spawn depth: ${openclaw_subagents_max_spawn_depth}"
            record 0
        else
            warn "Sandbox max spawn depth is '${openclaw_subagents_max_spawn_depth:-unknown}' — expected '${THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_SPAWN_DEPTH}'"
            fix "Re-run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
            record 2
        fi
    fi
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo ""

total=$((CHECKS_PASSED + CHECKS_WARNED + CHECKS_FAILED))

if [[ "${CHECKS_FAILED}" -eq 0 && "${CHECKS_WARNED}" -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}  All ${total} checks passed — system is healthy.${NC}"
    echo ""
elif [[ "${CHECKS_FAILED}" -eq 0 ]]; then
    echo -e "${YELLOW}${BOLD}  ${CHECKS_PASSED} passed, ${CHECKS_WARNED} warning(s) — system is mostly healthy.${NC}"
    echo ""
    echo "  Review warnings above."
    echo ""
else
    echo -e "${RED}${BOLD}  ${CHECKS_PASSED} passed, ${CHECKS_WARNED} warning(s), ${CHECKS_FAILED} failed.${NC}"
    echo ""
    echo "  Review the failures above and check the fix hints."
    echo ""
    exit 1
fi

if [[ -n "${sandbox_name}" ]]; then
    echo -e "${BOLD}  To verify end-to-end inference:${NC}"
    echo ""
    echo "  Connect to the sandbox:"
    echo "    nemoclaw ${sandbox_name} connect"
    echo ""
    echo "  Then inside the sandbox:"
    echo "    openclaw agent --agent main --local \\"
    echo '      -m "Reply with one word: working" --session-id test'
    echo ""
    echo "  Expected: a short reply from ${THOR_MODEL_ID}."
    echo '  Note: "No reply from agent" on the first attempt can happen'
    echo "  while the model warms up — wait a moment and try again."
    echo ""
    echo "  For temporary outbound access during troubleshooting:"
    echo "    ./apply-policy-additions.sh research-lite"
    echo "  Or use one-off approvals:"
    echo "    openshell term"
    echo ""
fi
