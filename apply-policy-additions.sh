#!/usr/bin/env bash
# apply-policy-additions.sh — Apply session-scoped network policy additions to a running sandbox
#
# Usage:
#   ./apply-policy-additions.sh [additions-profile] [--sandbox-name <name>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/checks.sh"
source "${SCRIPT_DIR}/lib/config.sh"
source "${SCRIPT_DIR}/lib/policy.sh"

POLICY_PROFILE_ARG=""
SANDBOX_NAME_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: ./apply-policy-additions.sh [additions-profile] [--sandbox-name <name>]"
            echo ""
            print_supported_policy_additions_profiles
            exit 0
            ;;
        --sandbox-name)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --sandbox-name" >&2
                exit 1
            fi
            SANDBOX_NAME_ARG="$2"
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
resolve_policy_additions_profile "${POLICY_PROFILE_ARG:-research-lite}"

echo ""
echo -e "${BOLD}NemoClaw-Thor Dynamic Policy Additions${NC}"
echo "Repo: ${SCRIPT_DIR}"
echo ""
echo "  Additions profile:  ${THOR_POLICY_ADDITIONS_PROFILE}"
if [[ -n "${SANDBOX_NAME_ARG}" ]]; then
    echo "  Sandbox:            ${SANDBOX_NAME_ARG}"
fi
echo ""

apply_dynamic_policy_additions "${THOR_POLICY_ADDITIONS_PROFILE}" "${SANDBOX_NAME_ARG}"

echo ""
echo "  Next:"
echo "    Use openshell term to watch and approve any further blocked requests."
echo ""
