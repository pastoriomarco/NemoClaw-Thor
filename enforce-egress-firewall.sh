#!/usr/bin/env bash
# enforce-egress-firewall.sh — Apply iptables egress firewall to sandbox pods
#
# Blocks all outbound internet access from sandbox pods except:
#   - DNS (CoreDNS)
#   - OpenShell gateway
#   - Host vLLM via Docker bridge
#
# Usage:
#   ./enforce-egress-firewall.sh [model-profile]
#   ./enforce-egress-firewall.sh --remove

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/checks.sh"
source "${SCRIPT_DIR}/lib/config.sh"
source "${SCRIPT_DIR}/lib/sandbox-runtime.sh"
source "${SCRIPT_DIR}/lib/egress-firewall.sh"

ACTION="enforce"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: ./enforce-egress-firewall.sh [model-profile]"
    echo "       ./enforce-egress-firewall.sh --remove"
    echo "       ./enforce-egress-firewall.sh --check"
    exit 0
fi

if [[ "${1:-}" == "--remove" ]]; then
    ACTION="remove"
    shift
elif [[ "${1:-}" == "--check" ]]; then
    ACTION="check"
    shift
fi

load_thor_runtime_config "${1:-}"

sandbox_name="$(resolve_thor_sandbox_name 2>/dev/null || true)"
if [[ -z "${sandbox_name}" ]]; then
    fail "Could not determine the sandbox name"
    fix "Set THOR_MANAGED_SANDBOX_NAME in ${THOR_CONFIG_FILE}"
    exit 1
fi

cluster_container="$(thor_openshell_cluster_container_name 2>/dev/null || true)"
if [[ -z "${cluster_container}" ]]; then
    fail "Could not determine the active OpenShell cluster container"
    exit 1
fi

echo ""
echo -e "${BOLD}NemoClaw-Thor Egress Firewall${NC}"
echo ""

case "${ACTION}" in
    enforce)
        enforce_sandbox_egress_firewall "${cluster_container}" "${sandbox_name}"
        echo ""
        echo -e "${GREEN}${BOLD}  Sandbox internet access is now blocked.${NC}"
        echo "  Only local inference (via OpenShell provider route) is permitted."
        ;;
    remove)
        remove_sandbox_egress_firewall "${cluster_container}" "${sandbox_name}"
        echo ""
        echo -e "${YELLOW}${BOLD}  Egress firewall removed. Sandbox has internet access.${NC}"
        ;;
    check)
        check_sandbox_egress_firewall "${cluster_container}"
        ;;
esac

echo ""
