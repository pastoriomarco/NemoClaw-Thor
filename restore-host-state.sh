#!/usr/bin/env bash
# restore-host-state.sh — Restore the Thor host to a saved pre-OpenShell state
#
# Usage:
#   ./restore-host-state.sh [--backup-dir <dir>] [--skip-firewall-restore]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/checks.sh"
source "${SCRIPT_DIR}/lib/host-state.sh"

BACKUP_DIR=""
RESTORE_FIREWALL="1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: ./restore-host-state.sh [--backup-dir <dir>] [--skip-firewall-restore]"
            exit 0
            ;;
        --backup-dir)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --backup-dir" >&2
                exit 1
            fi
            BACKUP_DIR="$2"
            shift 2
            ;;
        --skip-firewall-restore)
            RESTORE_FIREWALL="0"
            shift
            ;;
        *)
            echo "Unexpected argument: $1" >&2
            exit 1
            ;;
    esac
done

BACKUP_DIR="$(resolve_host_backup_dir "${BACKUP_DIR}")"

if [[ -z "${BACKUP_DIR}" || ! -d "${BACKUP_DIR}" ]]; then
    echo "No host backup directory found." >&2
    echo "Run ./backup-host-state.sh or ./apply-host-fixes.sh first." >&2
    exit 1
fi

echo ""
echo -e "${BOLD}Restore Thor Host State${NC}"
echo ""
echo "Backup source:"
echo "  ${BACKUP_DIR}"
echo ""

if [[ "${RESTORE_FIREWALL}" == "1" ]]; then
    echo "This run will also restore the saved iptables snapshots."
    echo "That is most reliable when the host is otherwise quiet and Docker state"
    echo "has not changed substantially since the backup was created."
else
    echo "Firewall snapshot restore is disabled for this run."
fi
echo ""

restore_host_backup_dir "${BACKUP_DIR}" "${RESTORE_FIREWALL}"

pass "Host restore completed from ${BACKUP_DIR}"
echo ""
echo "A reboot is still recommended to ensure the restored module and sysctl"
echo "state matches the expected baseline."
echo ""
