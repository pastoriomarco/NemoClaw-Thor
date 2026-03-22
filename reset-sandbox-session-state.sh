#!/usr/bin/env bash
# reset-sandbox-session-state.sh — Clear embedded OpenClaw session history inside the tracked sandbox
#
# Usage:
#   ./reset-sandbox-session-state.sh [model-profile]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/checks.sh"
source "${SCRIPT_DIR}/lib/config.sh"
source "${SCRIPT_DIR}/lib/sandbox-runtime.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: ./reset-sandbox-session-state.sh [model-profile]"
    echo ""
    print_supported_model_profiles
    exit 0
fi

load_thor_runtime_config "${1:-}"

sandbox_name="$(resolve_thor_sandbox_name 2>/dev/null || true)"
if [[ -z "${sandbox_name}" ]]; then
    fail "Could not determine the sandbox name for session reset"
    fix "Set THOR_MANAGED_SANDBOX_NAME in ${THOR_CONFIG_FILE}, or keep only one sandbox present."
    exit 1
fi

cluster_container="$(thor_openshell_cluster_container_name 2>/dev/null || true)"
if [[ -z "${cluster_container}" ]]; then
    fail "Could not determine the active OpenShell cluster container"
    fix "Check: openshell gateway info"
    fix "Check: docker ps --format '{{.Names}}'"
    exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

echo ""
echo -e "${BOLD}Reset Sandbox Session State${NC}"
echo "Repo: ${SCRIPT_DIR}"
echo ""
echo "  Sandbox:           ${sandbox_name}"
echo "  Backup suffix:     ${timestamp}"
echo ""

docker exec \
    -i \
    -e THOR_SANDBOX_NAME="${sandbox_name}" \
    -e THOR_RESET_TIMESTAMP="${timestamp}" \
    "${cluster_container}" \
    sh <<'SH'
set -euo pipefail

kubectl -n openshell exec "${THOR_SANDBOX_NAME}" -- env \
    THOR_RESET_TIMESTAMP="${THOR_RESET_TIMESTAMP}" \
    sh -lc '
python3 - <<'"'"'PY'"'"'
import json
import os
from pathlib import Path

base = Path("/sandbox/.openclaw-data/agents/main/sessions")
timestamp = os.environ["THOR_RESET_TIMESTAMP"]
base.mkdir(parents=True, exist_ok=True)

sessions_json = base / "sessions.json"
if sessions_json.exists():
    sessions_json.replace(base / f"sessions.json.bak-{timestamp}")

jsonl_files = sorted(base.glob("*.jsonl"))
for path in jsonl_files:
    path.replace(base / f"{path.name}.bak-{timestamp}")

with (base / "sessions.json").open("w", encoding="utf-8") as f:
    json.dump({}, f, indent=2)
    f.write("\n")
PY
echo "Session store reset under /sandbox/.openclaw-data/agents/main/sessions"
ls -1 /sandbox/.openclaw-data/agents/main/sessions
'
SH

echo ""
echo -e "${GREEN}${BOLD}  Session history reset complete.${NC}"
echo ""
echo "  Use this after a broken tool-call round leaves the embedded agent"
echo "  stuck replaying stale conversation history."
echo ""
