#!/usr/bin/env bash
# apply-policy-profile.sh — Install a static sandbox policy profile into a NemoClaw checkout
#
# Usage:
#   ./apply-policy-profile.sh [policy-profile] [--repo-dir <path>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/checks.sh"
source "${SCRIPT_DIR}/lib/config.sh"
source "${SCRIPT_DIR}/lib/policy.sh"

TARGET_REPO_DIR="${HOME}/NemoClaw"
POLICY_PROFILE_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: ./apply-policy-profile.sh [policy-profile] [--repo-dir <path>]"
            echo ""
            print_supported_policy_profiles
            exit 0
            ;;
        --repo-dir)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --repo-dir" >&2
                exit 1
            fi
            TARGET_REPO_DIR="$2"
            shift 2
            ;;
        *)
            if [[ -n "${POLICY_PROFILE_ARG}" ]]; then
                echo "Unexpected argument: $1" >&2
                exit 1
            fi
            POLICY_PROFILE_ARG="$1"
            shift
            ;;
    esac
done

load_thor_runtime_config
resolve_policy_profile "${POLICY_PROFILE_ARG:-${THOR_POLICY_PROFILE:-}}"

echo ""
echo -e "${BOLD}NemoClaw-Thor Static Policy Installer${NC}"
echo "Repo: ${SCRIPT_DIR}"
echo ""
echo "  Repo dir:           ${TARGET_REPO_DIR}"
echo "  Policy profile:     ${THOR_POLICY_PROFILE}"
echo ""

install_static_policy_profile "${TARGET_REPO_DIR}" "${THOR_POLICY_PROFILE}"
save_thor_runtime_config

echo ""
echo "  Next:"
echo "    Re-run the upstream onboarding flow so the new baseline is applied."
echo ""
