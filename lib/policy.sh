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
  strict-local-inference
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
        strict-local|strict-local-inference|local-hardened)
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
    local sandbox_name="${2:-}"
    local src
    local current_policy
    local merged_policy_file

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

    if [[ -z "${sandbox_name}" ]]; then
        sandbox_name=$(resolve_thor_sandbox_name 2>/dev/null || echo "")
    fi

    if [[ -z "${sandbox_name}" ]]; then
        fail "Could not determine a target sandbox for dynamic policy additions"
        fix "Pass a sandbox name explicitly, or set THOR_MANAGED_SANDBOX_NAME in ${THOR_CONFIG_FILE}."
        return 1
    fi

    current_policy=$(openshell policy get --full "${sandbox_name}" 2>/dev/null || echo "")
    if [[ -z "${current_policy}" ]]; then
        fail "Could not read the current policy for sandbox '${sandbox_name}'"
        fix "Check: openshell policy get --full ${sandbox_name}"
        return 1
    fi

    merged_policy_file="$(mktemp)"
    if ! CURRENT_POLICY_RAW="${current_policy}" python3 - "${src}" "${merged_policy_file}" <<'PYEOF'
import os
import re
import sys

src_path = sys.argv[1]
dst_path = sys.argv[2]

raw_current = os.environ.get("CURRENT_POLICY_RAW", "")
_, sep, tail = raw_current.partition("---")
current_policy = tail.lstrip("\n") if sep else raw_current.strip()

with open(src_path, encoding="utf-8") as f:
    additions_policy = f.read()

match = re.search(r"^network_policies:\n([\s\S]*)$", additions_policy, re.MULTILINE)
if not match:
    raise SystemExit("dynamic additions file is missing a network_policies block")

addition_entries = match.group(1).rstrip()

if current_policy and "network_policies:" in current_policy:
    lines = current_policy.splitlines()
    merged_lines = []
    in_network_policies = False
    inserted = False

    for line in lines:
        is_top_level = bool(re.match(r"^\S.*:", line))
        if line.strip() == "network_policies:" or line.strip().startswith("network_policies:"):
            in_network_policies = True
            merged_lines.append(line)
            continue
        if in_network_policies and is_top_level and not inserted:
            merged_lines.append(addition_entries)
            inserted = True
            in_network_policies = False
        merged_lines.append(line)

    if in_network_policies and not inserted:
        merged_lines.append(addition_entries)

    merged_policy = "\n".join(merged_lines).rstrip() + "\n"
elif current_policy:
    if "version:" not in current_policy:
        current_policy = "version: 1\n" + current_policy
    merged_policy = current_policy.rstrip() + "\n\nnetwork_policies:\n" + addition_entries + "\n"
else:
    merged_policy = "version: 1\n\nnetwork_policies:\n" + addition_entries + "\n"

with open(dst_path, "w", encoding="utf-8") as f:
    f.write(merged_policy)
PYEOF
    then
        rm -f "${merged_policy_file}"
        fail "Failed to merge dynamic policy additions for sandbox '${sandbox_name}'"
        return 1
    fi

    openshell policy set "${sandbox_name}" --policy "${merged_policy_file}" --wait
    rm -f "${merged_policy_file}"
    pass "Applied dynamic policy additions '${THOR_POLICY_ADDITIONS_PROFILE}' to sandbox '${sandbox_name}'"
    info "These additions are session-scoped and reset when the sandbox stops."
}
