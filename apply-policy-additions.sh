#!/usr/bin/env bash
# apply-policy-additions.sh — Apply session-scoped network policy additions to a running sandbox
#
# Usage:
#   ./apply-policy-additions.sh [additions-profile]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/checks.sh"
source "${SCRIPT_DIR}/lib/config.sh"
source "${SCRIPT_DIR}/lib/policy.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: ./apply-policy-additions.sh [additions-profile]"
    echo ""
    print_supported_policy_additions_profiles
    exit 0
fi

load_thor_runtime_config
resolve_policy_additions_profile "${1:-research-lite}"

echo ""
echo -e "${BOLD}NemoClaw-Thor Dynamic Policy Additions${NC}"
echo -e "JetsonHacks fork — /home/tndlux/workspaces/thor_llm/src/NemoClaw-Thor"
echo ""
echo "  Additions profile:  ${THOR_POLICY_ADDITIONS_PROFILE}"
echo ""

apply_dynamic_policy_additions "${THOR_POLICY_ADDITIONS_PROFILE}"

echo ""
echo "  Next:"
echo "    Use openshell term to watch and approve any further blocked requests."
echo ""
