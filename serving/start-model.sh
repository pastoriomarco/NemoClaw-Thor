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
        echo "  THOR_HF_MODE=auto|offline|latest"
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

# 2026-06-01: per-profile proxy tuning. If THOR_TARGET_PROXY_* vars were
# set by config.sh during load_thor_runtime_config, restart the local
# vllm-proxy with profile-appropriate loop thresholds + thinking force.
# Gated via THOR_RESTART_PROXY=0 to opt out (e.g. when the proxy is
# managed by a separate supervisor or you want to keep an in-flight
# config).
if [[ "${THOR_RESTART_PROXY:-1}" != "0" ]]; then
    REFLECT_AT="${THOR_TARGET_PROXY_LOOP_REFLECT_AT:-4}"
    STOP_AT="${THOR_TARGET_PROXY_LOOP_STOP_AT:-8}"
    FORCE_THINKING="${THOR_TARGET_PROXY_FORCE_ENABLE_THINKING:-}"
    PROXY_SCRIPT="${REPO_ROOT}/manyforge/scripts/proxy/vllm-proxy.py"
    if [[ -f "${PROXY_SCRIPT}" ]]; then
        info "Restarting vllm-proxy with profile tuning: REFLECT_AT=${REFLECT_AT} STOP_AT=${STOP_AT} FORCE_THINKING=${FORCE_THINKING:-(unset)}"
        pkill -9 -f "scripts/proxy/vllm-proxy.py" 2>/dev/null || true
        sleep 1
        # Mirror the production assistant.sh defaults
        export OPENCLAW_PROXY_BIND="${OPENCLAW_PROXY_BIND:-0.0.0.0}"
        export OPENCLAW_PROXY_LISTEN_PORT="${OPENCLAW_PROXY_LISTEN_PORT:-8000}"
        export OPENCLAW_PROXY_UPSTREAM="${OPENCLAW_PROXY_UPSTREAM:-http://127.0.0.1:8050}"
        export OPENCLAW_PROXY_LOG_PATH="${OPENCLAW_PROXY_LOG_PATH:-/tmp/manyforge-assistant-e2e/vllm-proxy.jsonl}"
        export OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS="${OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS:-2048}"
        export OPENCLAW_PROXY_THINKING_TOKEN_BUDGET="${OPENCLAW_PROXY_THINKING_TOKEN_BUDGET:-512}"
        export OPENCLAW_PROXY_LOOP_REFLECT_AT="${REFLECT_AT}"
        export OPENCLAW_PROXY_LOOP_STOP_AT="${STOP_AT}"
        # 2026-06-02: OpenClaw 2026.5.22 rejects responses where
        # `content` is null (treats as `code=incomplete_result`). Any
        # vLLM profile launched with `--reasoning-parser` (cosmos's
        # qwen3 parser; nemotron3-nano's nano_v3 parser) routes all
        # output into the `reasoning` field, leaving `content` null
        # and breaking every chat-completion lane in the new agent.
        # The proxy mirrors `reasoning` → `content` so the contract
        # is satisfied for both old (reasoning-only) and new (content-
        # required) consumers. Default on; set =0 to opt out for
        # bake-off baselines.
        export OPENCLAW_PROXY_PROMOTE_REASONING_TO_CONTENT="${OPENCLAW_PROXY_PROMOTE_REASONING_TO_CONTENT:-1}"
        # 2026-06-02: vLLM's hermes tool-call parser sometimes leaves the
        # `<tool_call>{...}</tool_call>` wrapper in `assistant.tool_calls[*]
        # .arguments` when cosmos/qwen3.6 emit calls in reasoning-on mode.
        # On the next turn, the bridge replays the conversation; vLLM's
        # chat template calls json.loads(arguments) and 400s on a string
        # that starts with `<`. The proxy detects that pattern and unwraps
        # to just the inner arguments JSON. Default on; set =0 to disable
        # per-profile if a model emits legitimate `<tool_call>` content for
        # some other reason.
        export OPENCLAW_PROXY_UNWRAP_TOOL_CALL_ARGS="${OPENCLAW_PROXY_UNWRAP_TOOL_CALL_ARGS:-1}"
        if [[ -n "${FORCE_THINKING}" ]]; then
            export OPENCLAW_PROXY_FORCE_ENABLE_THINKING="${FORCE_THINKING}"
        else
            unset OPENCLAW_PROXY_FORCE_ENABLE_THINKING
        fi
        mkdir -p "$(dirname "${OPENCLAW_PROXY_LOG_PATH}")"
        nohup python3 "${PROXY_SCRIPT}" > /tmp/vllm-proxy.startup.log 2>&1 &
        disown
        sleep 2
        if pgrep -f "scripts/proxy/vllm-proxy.py" >/dev/null; then
            info "vllm-proxy restarted (pid $(pgrep -f scripts/proxy/vllm-proxy.py | head -1))"
        else
            info "WARNING: vllm-proxy did not start; check /tmp/vllm-proxy.startup.log"
        fi
    fi
fi

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
