#!/usr/bin/env bash
# apply-host-fixes.sh — Backup the Thor host and apply OpenShell-Thor fixes
#
# Usage:
#   ./apply-host-fixes.sh [--helper-repo-dir <dir>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/checks.sh"
source "${SCRIPT_DIR}/lib/host-state.sh"

HELPER_REPO_DIR="$(thor_default_openshell_thor_dir)"
HELPER_REPO_URL="https://github.com/jetsonhacks/OpenShell-Thor.git"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: ./apply-host-fixes.sh [--helper-repo-dir <dir>]"
            exit 0
            ;;
        --helper-repo-dir)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --helper-repo-dir" >&2
                exit 1
            fi
            HELPER_REPO_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unexpected argument: $1" >&2
            exit 1
            ;;
    esac
done

echo ""
echo -e "${BOLD}Apply Thor Host Fixes${NC}"
echo ""
echo "This wrapper snapshots the current host state, then runs the JetsonHacks"
echo "OpenShell-Thor helper scripts that build iptable_raw and apply the Docker,"
echo "iptables, br_netfilter, and sysctl changes needed for OpenShell on Thor."
echo ""

create_host_backup "pre-openshell"
pass "Host backup saved to ${THOR_HOST_BACKUP_DIR_CREATED}"
echo ""

ensure_openshell_thor_repo "${HELPER_REPO_DIR}" "${HELPER_REPO_URL}"
info "Using helper repo: ${HELPER_REPO_DIR}"
echo ""

if [[ ! -x "${HELPER_REPO_DIR}/build_ip_table_raw.sh" ]]; then
    fail "Expected helper script not found: ${HELPER_REPO_DIR}/build_ip_table_raw.sh"
    exit 1
fi

if [[ ! -x "${HELPER_REPO_DIR}/setup-openshell-network.sh" ]]; then
    fail "Expected helper script not found: ${HELPER_REPO_DIR}/setup-openshell-network.sh"
    exit 1
fi

header "Building iptable_raw"
echo ""
"${HELPER_REPO_DIR}/build_ip_table_raw.sh"
echo ""

header "Applying network and Docker fixes"
echo ""
"${HELPER_REPO_DIR}/setup-openshell-network.sh"
echo ""

header "Clearing stale iptables-nft compatibility rules"
echo ""
if clear_stale_nft_compat_firewall; then
    pass "Cleared stale nft compatibility tables left behind by iptables-nft"
else
    info "No stale iptables-nft compatibility tables detected"
fi
echo ""

pass "Thor host fixes applied"
echo ""
echo "If you need to revert to the pre-change host state:"
echo "  ./restore-host-state.sh"
echo ""
