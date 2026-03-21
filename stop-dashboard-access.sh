#!/usr/bin/env bash
# stop-dashboard-access.sh — Stop the host-local dashboard access helpers

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/config.sh"
source "${SCRIPT_DIR}/lib/sandbox-runtime.sh"

ensure_thor_runtime_path
load_thor_runtime_config ""

STATE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/nemoclaw-thor/dashboard"
STATE_FILE="${STATE_DIR}/access.env"
PID_FILE="${STATE_DIR}/proxy.pid"

if [[ -f "${PID_FILE}" ]]; then
    proxy_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${proxy_pid}" ]] && kill -0 "${proxy_pid}" >/dev/null 2>&1; then
        kill "${proxy_pid}" >/dev/null 2>&1 || true
    fi
    rm -f "${PID_FILE}"
fi

if sudo -n true >/dev/null 2>&1; then
    cluster_container="$(active_openshell_cluster_container_name 2>/dev/null || true)"
    sandbox_name="$(resolve_thor_sandbox_name 2>/dev/null || true)"
    if [[ -n "${cluster_container}" && -n "${sandbox_name}" ]]; then
        sudo docker exec "${cluster_container}" sh -lc \
            "pkill -f '[k]ubectl -n openshell port-forward --address 0.0.0.0 pod/${sandbox_name} ' >/dev/null 2>&1 || true"
    fi
fi

rm -f "${STATE_FILE}"
echo "Dashboard access stopped."
