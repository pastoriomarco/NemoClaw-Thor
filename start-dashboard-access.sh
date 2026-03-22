#!/usr/bin/env bash
# start-dashboard-access.sh — Expose the sandbox OpenClaw dashboard on host localhost
#
# Usage:
#   ./start-dashboard-access.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/config.sh"
source "${SCRIPT_DIR}/lib/sandbox-runtime.sh"

ensure_thor_runtime_path
load_thor_runtime_config ""

LOCAL_PORT="${THOR_DASHBOARD_LOCAL_PORT:-18789}"
CLUSTER_PORT="${THOR_DASHBOARD_CLUSTER_PORT:-18794}"
STATE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/nemoclaw-thor/dashboard"
STATE_FILE="${STATE_DIR}/access.env"
PID_FILE="${STATE_DIR}/proxy.pid"
LOG_FILE="${STATE_DIR}/proxy.log"

mkdir -p "${STATE_DIR}"

sandbox_name="$(resolve_thor_sandbox_name)"
cluster_container="$(thor_openshell_cluster_container_name)"

if ! sudo -n true >/dev/null 2>&1; then
    echo "Passwordless sudo is required for dashboard access helpers." >&2
    exit 1
fi

if ! sudo docker exec "${cluster_container}" kubectl -n openshell exec "${sandbox_name}" -- \
    curl -sfI --max-time 5 http://127.0.0.1:18789 >/dev/null 2>&1; then
    echo "The OpenClaw gateway is not serving inside sandbox ${sandbox_name}." >&2
    echo "Inside the sandbox, run: HOME=/sandbox openclaw gateway run" >&2
    exit 1
fi

cluster_ip="$(sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "${cluster_container}")"
if [[ -z "${cluster_ip}" ]]; then
    echo "Could not determine the OpenShell cluster container IP." >&2
    exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" >/dev/null 2>&1; then
        kill "${old_pid}" >/dev/null 2>&1 || true
        sleep 1
    fi
    rm -f "${PID_FILE}"
fi

if ss -ltn "( sport = :${LOCAL_PORT} )" | grep -q "${LOCAL_PORT}"; then
    echo "Local port ${LOCAL_PORT} is already in use." >&2
    echo "Stop the existing listener or set THOR_DASHBOARD_LOCAL_PORT to another port." >&2
    exit 1
fi

sudo docker exec -d "${cluster_container}" \
    kubectl -n openshell port-forward --address 0.0.0.0 "pod/${sandbox_name}" "${CLUSTER_PORT}:18789" \
    >/dev/null

ready=0
for _ in $(seq 1 20); do
    if curl -sfI --max-time 2 "http://${cluster_ip}:${CLUSTER_PORT}" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done

if [[ "${ready}" != "1" ]]; then
    echo "Cluster-side dashboard forward did not become ready at ${cluster_ip}:${CLUSTER_PORT}." >&2
    exit 1
fi

proxy_pid="$(
python3 - "${SCRIPT_DIR}/dashboard_tcp_proxy.py" "${LOCAL_PORT}" "${cluster_ip}" "${CLUSTER_PORT}" "${LOG_FILE}" <<'PY'
import subprocess
import sys

script, local_port, cluster_ip, cluster_port, log_path = sys.argv[1:6]

with open(log_path, "ab", buffering=0) as log_file:
    proc = subprocess.Popen(
        [
            sys.executable,
            script,
            "--listen-host",
            "127.0.0.1",
            "--listen-port",
            local_port,
            "--target-host",
            cluster_ip,
            "--target-port",
            cluster_port,
        ],
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
        close_fds=True,
    )

print(proc.pid)
PY
)"
echo "${proxy_pid}" > "${PID_FILE}"

sleep 1
if ! kill -0 "${proxy_pid}" >/dev/null 2>&1; then
    echo "Dashboard proxy failed to start. See ${LOG_FILE}." >&2
    exit 1
fi

if ! ss -ltn "( sport = :${LOCAL_PORT} )" | grep -q "${LOCAL_PORT}"; then
    echo "Dashboard proxy did not bind localhost:${LOCAL_PORT}. See ${LOG_FILE}." >&2
    exit 1
fi

if ! curl -sfI --max-time 5 "http://127.0.0.1:${LOCAL_PORT}" >/dev/null 2>&1; then
    echo "Dashboard proxy did not serve http://127.0.0.1:${LOCAL_PORT}/ successfully." >&2
    echo "See ${LOG_FILE}." >&2
    exit 1
fi

token="$(
    sudo docker exec -i "${cluster_container}" kubectl -n openshell exec -i "${sandbox_name}" -- \
        python3 - <<'PY'
import json
from pathlib import Path

cfg = json.loads(Path("/sandbox/.openclaw/openclaw.json").read_text())
print(cfg.get("gateway", {}).get("auth", {}).get("token", ""))
PY
)"

cat > "${STATE_FILE}" <<EOF
LOCAL_PORT=${LOCAL_PORT}
CLUSTER_PORT=${CLUSTER_PORT}
CLUSTER_IP=${cluster_ip}
SANDBOX_NAME=${sandbox_name}
CLUSTER_CONTAINER=${cluster_container}
PROXY_PID=${proxy_pid}
EOF

echo "Dashboard access is ready."
echo "Open:"
echo "  http://127.0.0.1:${LOCAL_PORT}/"
echo "or:"
echo "  http://localhost:${LOCAL_PORT}/"
if [[ -n "${token}" ]]; then
    echo "Token URL:"
    echo "  http://127.0.0.1:${LOCAL_PORT}/#token=${token}"
fi
echo ""
echo "Stop with:"
echo "  ./stop-dashboard-access.sh"
