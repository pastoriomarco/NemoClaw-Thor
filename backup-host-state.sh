#!/usr/bin/env bash
# backup-host-state.sh — Snapshot the Thor host before applying OpenShell fixes
#
# Usage:
#   ./backup-host-state.sh [--label <name>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/checks.sh"
source "${SCRIPT_DIR}/lib/host-state.sh"

LABEL="pre-openshell"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: ./backup-host-state.sh [--label <name>]"
            exit 0
            ;;
        --label)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --label" >&2
                exit 1
            fi
            LABEL="$2"
            shift 2
            ;;
        *)
            echo "Unexpected argument: $1" >&2
            exit 1
            ;;
    esac
done

echo ""
echo -e "${BOLD}Thor Host Backup${NC}"
echo ""
echo "This captures the host state before applying OpenShell-Thor fixes."
echo "It saves Docker config, the current iptables backend selection, current"
echo "iptables rules, and any existing OpenShell-specific persistence files."
echo ""
echo "The firewall snapshot is best restored on a quiet host. If unrelated"
echo "containers or firewall rules change later, exact rule restoration may no"
echo "longer match the current runtime state."
echo ""

create_host_backup "${LABEL}"

pass "Host backup saved to ${THOR_HOST_BACKUP_DIR_CREATED}"
echo ""
echo "Latest backup marker:"
echo "  $(thor_latest_host_backup_file)"
echo ""
