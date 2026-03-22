#!/usr/bin/env bash
# start-gateway.sh — Start or revive the local OpenShell gateway used by NemoClaw-Thor
#
# Usage:
#   ./start-gateway.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/checks.sh"
source "${SCRIPT_DIR}/lib/config.sh"

load_thor_runtime_config

echo ""
echo -e "${BOLD}NemoClaw-Thor Gateway Start${NC}"
echo "Repo: ${SCRIPT_DIR}"
echo ""
echo "  Gateway name: ${THOR_OPENSHELL_GATEWAY_NAME}"
echo ""

ensure_openshell_gateway_running

