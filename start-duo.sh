#!/usr/bin/env bash
# start-duo.sh — Co-serve the ManyForge Qwen3.6 profile and Cosmos Reason 2 2B on Thor
#
# Launches two detached vLLM containers:
#   - qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge on port 8000 (reasoning/coding/query_model)
#   - cosmos-reason2-2b                      on port 8001 (physical-AI VLM for BT nodes)
#
# Combined gpu_memory_utilization: 0.32 (Qwen3.6) + 0.12 (Cosmos) = 0.44 of Thor.
# Requires: single-model containers must be stopped first.
#
# Usage:
#   ./start-duo.sh              # starts both
#   ./start-duo.sh stop         # stops both
#   ./start-duo.sh status       # shows health of both endpoints
#   ./start-duo.sh logs qwen    # tails qwen container logs
#   ./start-duo.sh logs cosmos  # tails cosmos container logs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Gated repos (e.g. nvidia/Cosmos-Reason2-2B) require HF_TOKEN at vLLM startup
# to fetch processor_config.json. Mounting ~/.cache/huggingface is not sufficient
# for the Transformers gated-repo check — the env var must be set.
if [[ -z "${HF_TOKEN:-}" && -r "${HOME}/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
fi

QWEN_PROFILE="qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge"
QWEN_PORT="${QWEN_PORT:-8000}"
QWEN_NAME="nemoclaw-qwen36-manyforge"

COSMOS_PROFILE="cosmos-reason2-2b"
COSMOS_PORT="${COSMOS_PORT:-8001}"
COSMOS_NAME="nemoclaw-cosmos-reason2"

wait_for_ready() {
    local port="$1" name="$2" max_s="${3:-600}"
    local started=$(date +%s)
    while true; do
        if curl -s --max-time 2 "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -q '"object":"model"'; then
            echo "  ${name} ready on :${port} (took $(( $(date +%s) - started ))s)"
            return 0
        fi
        if ! docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
            echo "  ${name} container died during startup — check logs:" >&2
            echo "    docker logs ${name}" >&2
            return 1
        fi
        if (( $(date +%s) - started > max_s )); then
            echo "  ${name} did not become ready within ${max_s}s" >&2
            return 1
        fi
        sleep 5
    done
}

start_one() {
    local profile="$1" port="$2" name="$3"
    echo "Starting ${name} (profile: ${profile}, port: ${port})..."
    if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
        echo "  already running — skipping"
        return 0
    fi
    THOR_DETACH=1 \
    THOR_CONTAINER_NAME="${name}" \
    THOR_VLLM_PORT="${port}" \
        "${SCRIPT_DIR}/start-model.sh" "${profile}" >/tmp/start-duo-${name}.log 2>&1
    sleep 2
    if ! docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
        echo "  launch failed — details in /tmp/start-duo-${name}.log" >&2
        tail -20 "/tmp/start-duo-${name}.log" >&2
        return 1
    fi
}

cmd_start() {
    if docker ps --format '{{.Names}}' | grep -v openshell | grep -v "^${QWEN_NAME}$" | grep -v "^${COSMOS_NAME}$" | grep -q .; then
        echo "Other vLLM containers are running — stop them first to avoid GPU contention:" >&2
        docker ps --format '  {{.Names}}  ({{.Image}})' | grep -v openshell >&2
        return 1
    fi

    start_one "${QWEN_PROFILE}" "${QWEN_PORT}" "${QWEN_NAME}"
    start_one "${COSMOS_PROFILE}" "${COSMOS_PORT}" "${COSMOS_NAME}"

    echo ""
    echo "Waiting for endpoints to become ready..."
    wait_for_ready "${QWEN_PORT}" "${QWEN_NAME}" 420 &
    local qwen_pid=$!
    wait_for_ready "${COSMOS_PORT}" "${COSMOS_NAME}" 420 &
    local cosmos_pid=$!
    local rc=0
    wait "${qwen_pid}" || rc=1
    wait "${cosmos_pid}" || rc=1

    echo ""
    if [[ "${rc}" == "0" ]]; then
        echo "Duo serve ready:"
        echo "  Qwen3.6 ManyForge : http://127.0.0.1:${QWEN_PORT}/v1  (${QWEN_NAME})"
        echo "  Cosmos Reason 2   : http://127.0.0.1:${COSMOS_PORT}/v1 (${COSMOS_NAME})"
    else
        echo "One or both endpoints failed to come up." >&2
        return 1
    fi
}

cmd_stop() {
    local stopped=0
    for name in "${QWEN_NAME}" "${COSMOS_NAME}"; do
        if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
            echo "Stopping ${name}..."
            docker stop "${name}" >/dev/null
            stopped=$((stopped + 1))
        fi
    done
    sync
    if command -v sudo &>/dev/null && [[ ${stopped} -gt 0 ]]; then
        sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    fi
    echo "Stopped ${stopped} container(s)."
}

cmd_status() {
    for pair in "${QWEN_NAME}:${QWEN_PORT}" "${COSMOS_NAME}:${COSMOS_PORT}"; do
        local name="${pair%:*}" port="${pair#*:}"
        if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
            if curl -s --max-time 3 "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -q '"object":"model"'; then
                echo "  ${name} (:${port})  READY"
            else
                echo "  ${name} (:${port})  RUNNING but endpoint not ready"
            fi
        else
            echo "  ${name} (:${port})  STOPPED"
        fi
    done
}

cmd_logs() {
    local which="${1:-qwen}"
    local name
    case "${which}" in
        qwen|q) name="${QWEN_NAME}" ;;
        cosmos|c) name="${COSMOS_NAME}" ;;
        *) echo "Usage: $0 logs {qwen|cosmos}" >&2; return 1 ;;
    esac
    docker logs -f --tail 60 "${name}"
}

case "${1:-start}" in
    start) cmd_start ;;
    stop)  cmd_stop ;;
    status) cmd_status ;;
    logs) shift; cmd_logs "$@" ;;
    *) echo "Usage: $0 [start|stop|status|logs {qwen|cosmos}]" >&2; exit 1 ;;
esac
