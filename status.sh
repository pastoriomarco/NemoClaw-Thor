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
source "${SCRIPT_DIR}/lib/egress-firewall.sh"

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
cluster_container=""
if ! command -v openshell &>/dev/null; then
    fail "openshell not found — cannot check sandbox status"
    record 1
else
    sandbox_list=$(openshell sandbox list 2>/dev/null || echo "")
    sandbox_name=$(resolve_thor_sandbox_name 2>/dev/null || echo "")
    cluster_container=$(thor_openshell_cluster_container_name 2>/dev/null || echo "")
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

        # parallel_tool_calls is NOT set by configure-local-provider.sh — the
        # OpenAI API default (true) is used. v3 disabled it due to streaming
        # parser bugs; vLLM 0.19 fixes those. Only report current value.
        if [[ "${openclaw_parallel_tool_calls}" == "true" || -z "${openclaw_parallel_tool_calls}" || "${openclaw_parallel_tool_calls}" == "null" ]]; then
            info "Sandbox parallel tool calls: ${openclaw_parallel_tool_calls:-default} (API default)"
            record 0
        else
            info "Sandbox parallel tool calls: ${openclaw_parallel_tool_calls}"
            record 0
        fi

        # Temperature is NOT set by configure-local-provider.sh — vLLM auto-loads
        # model-specific defaults from generation_config.json (e.g. 1.0 for Gemma 4,
        # 0.6 for Qwen 3.5 coding). Any non-null value is fine; only warn if missing.
        if [[ -n "${openclaw_temperature}" && "${openclaw_temperature}" != "null" ]]; then
            info "Sandbox temperature: ${openclaw_temperature} (model default via vLLM)"
            record 0
        else
            info "Sandbox temperature: not set (vLLM will use model's generation_config.json)"
            record 0
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

header "Sandbox OpenClaw Gateway"
echo ""

if [[ -z "${sandbox_name}" ]]; then
    warn "Skipping sandbox gateway check because no sandbox is resolved"
    record 2
else
    forward_line=$(openshell forward list 2>/dev/null \
        | sed 's/\x1b\[[0-9;]*m//g' \
        | awk -v name="${sandbox_name}" '$1 == name && $3 == "18789" && /running/ {found=1} END {print found+0}')

    if [[ "${forward_line}" == "1" ]]; then
        # Send a WebSocket upgrade request to verify the actual gateway process
        # is running, not just the SSH tunnel (which accepts TCP but resets).
        gateway_probe=$(python3 -c "
import socket, sys
try:
    s = socket.create_connection(('127.0.0.1', 18789), timeout=5)
    s.sendall(b'GET / HTTP/1.1\r\nHost: 127.0.0.1:18789\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n')
    s.settimeout(5)
    data = s.recv(1024)
    s.close()
    # Any HTTP response means the gateway is alive (101 Switching Protocols, 400, etc.)
    if data and (b'HTTP/' in data or b'websocket' in data.lower()):
        print('ok')
    elif not data:
        # Empty response — tunnel accepted but nothing behind it
        print('empty')
    else:
        print('no-http')
except ConnectionResetError:
    # SSH tunnel accepted but nothing behind it — gateway not running
    print('reset')
except Exception as e:
    print('error')
" 2>/dev/null || echo "error")

        if [[ "${gateway_probe}" == "ok" ]]; then
            pass "OpenClaw gateway is reachable via host forward on port 18789"
            record 0
        else
            warn "Host forward on port 18789 is active but OpenClaw gateway is not responding"
            info "The gateway process may not be running inside the sandbox."
            fix "Run: ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
            fix "Or inside the sandbox: HOME=/sandbox openclaw gateway run &"
            record 2
        fi
    else
        info "No active host forward on port 18789 — gateway check skipped"
        info "To enable this check: openshell forward start 18789 ${sandbox_name} --background"
        record 0
    fi
fi

# ── Egress firewall ──────────────────────────────────────────
if [[ -n "${cluster_container}" && -n "${sandbox_name}" ]]; then
    # grep -c prints "0" and returns exit code 1 when no matches — use || true
    # to prevent set -e from aborting, then default empty result to 0.
    fw_drop_count=$(docker exec "${cluster_container}" \
        iptables -L "${EGRESS_FW_CHAIN}" -n 2>/dev/null \
        | grep -c "DROP" || true)
    fw_drop_count="${fw_drop_count:-0}"
    if [[ "${fw_drop_count}" -gt 0 ]]; then
        pass "Egress firewall active — sandbox internet access blocked"
        record 0
    else
        fail "Egress firewall NOT active — sandbox has unrestricted internet"
        fix "Run: ./enforce-egress-firewall.sh"
        record 1
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
    echo "  If openclaw tui shows 'gateway disconnected':"
    echo "    ./configure-local-provider.sh ${THOR_MODEL_PROFILE}"
    echo "  Or inside the sandbox:"
    echo "    HOME=/sandbox openclaw gateway run &"
    echo ""
    echo "  For temporary outbound access during troubleshooting:"
    echo "    ./apply-policy-additions.sh research-lite"
    echo "  Or use one-off approvals:"
    echo "    openshell term"
    echo ""
fi
