#!/usr/bin/env bash
# lib/policy.sh — Shared sandbox policy helpers for NemoClaw-Thor
#
# Source this file; do not execute it directly.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This file is meant to be sourced, not executed directly." >&2
    exit 1
fi

POLICY_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THOR_POLICY_STATIC_DIR="${POLICY_LIB_DIR%/lib}/policies/static"
THOR_POLICY_DYNAMIC_DIR="${POLICY_LIB_DIR%/lib}/policies/dynamic"

print_supported_policy_profiles() {
    cat <<'EOF'
Supported static policy profiles:
  strict-local
  local-hardened
EOF
}

print_supported_policy_additions_profiles() {
    cat <<'EOF'
Supported dynamic policy additions:
  research-lite
EOF
}

normalize_policy_profile() {
    echo "${1}" | tr '[:upper:]' '[:lower:]'
}

resolve_policy_profile() {
    local requested
    requested="$(normalize_policy_profile "${1:-${THOR_POLICY_PROFILE:-strict-local}}")"
    case "${requested}" in
        strict-local|local-hardened)
            THOR_POLICY_PROFILE="${requested}"
            ;;
        *)
            echo "Unsupported policy profile: ${requested}" >&2
            print_supported_policy_profiles >&2
            return 1
            ;;
    esac
}

resolve_policy_additions_profile() {
    local requested
    requested="$(normalize_policy_profile "${1:-}")"
    case "${requested}" in
        research-lite)
            THOR_POLICY_ADDITIONS_PROFILE="${requested}"
            ;;
        *)
            echo "Unsupported policy additions profile: ${requested}" >&2
            print_supported_policy_additions_profiles >&2
            return 1
            ;;
    esac
}

static_policy_template_path() {
    echo "${THOR_POLICY_STATIC_DIR}/${1}.yaml"
}

dynamic_policy_template_path() {
    echo "${THOR_POLICY_DYNAMIC_DIR}/${1}.yaml"
}

install_static_policy_profile() {
    local target_repo_dir="$1"
    local profile="${2:-${THOR_POLICY_PROFILE:-strict-local}}"
    local src dst backup

    resolve_policy_profile "${profile}" || return 1

    src="$(static_policy_template_path "${THOR_POLICY_PROFILE}")"
    dst="${target_repo_dir}/nemoclaw-blueprint/policies/openclaw-sandbox.yaml"
    backup="${target_repo_dir}/nemoclaw-blueprint/policies/openclaw-sandbox.upstream.yaml"

    if [[ ! -f "${src}" ]]; then
        fail "Static policy template not found: ${src}"
        return 1
    fi

    if [[ ! -f "${dst}" ]]; then
        fail "Target NemoClaw policy file not found: ${dst}"
        return 1
    fi

    if [[ ! -f "${backup}" ]]; then
        cp "${dst}" "${backup}"
    fi

    cp "${src}" "${dst}"
    pass "Installed static policy profile '${THOR_POLICY_PROFILE}' into ${dst}"
    info "Original upstream policy saved as ${backup}"
}

apply_dynamic_policy_additions() {
    local profile="${1:-}"
    local src

    resolve_policy_additions_profile "${profile}" || return 1
    src="$(dynamic_policy_template_path "${THOR_POLICY_ADDITIONS_PROFILE}")"

    if [[ ! -f "${src}" ]]; then
        fail "Dynamic policy template not found: ${src}"
        return 1
    fi

    if ! command -v openshell &>/dev/null; then
        fail "openshell command not found"
        return 1
    fi

    openshell policy set "${src}"
    pass "Applied dynamic policy additions '${THOR_POLICY_ADDITIONS_PROFILE}'"
    info "These additions are session-scoped and reset when the sandbox stops."
}
