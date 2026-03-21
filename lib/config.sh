#!/usr/bin/env bash
# lib/config.sh — Shared runtime config for NemoClaw-Thor
#
# Source this file; do not execute it directly.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This file is meant to be sourced, not executed directly." >&2
    exit 1
fi

prepend_path_once() {
    local dir="${1:-}"
    [[ -n "${dir}" && -d "${dir}" ]] || return 0

    case ":${PATH:-}:" in
        *":${dir}:"*) ;;
        *) export PATH="${dir}:${PATH:-}" ;;
    esac
}

ensure_thor_runtime_path() {
    prepend_path_once "${HOME}/.local/bin"
    prepend_path_once "${NVM_BIN:-}"
}

ensure_thor_runtime_path

thor_config_file() {
    local config_home
    config_home="${XDG_CONFIG_HOME:-$HOME/.config}"
    echo "${NEMOCLAW_THOR_CONFIG_FILE:-${config_home}/nemoclaw-thor/config.env}"
}

thor_nemoclaw_registry_file() {
    echo "${HOME}/.nemoclaw/sandboxes.json"
}

csv_append_unique() {
    local csv="${1:-}"
    local value="${2:-}"

    python3 - "${csv}" "${value}" <<'PYEOF'
import sys

csv = sys.argv[1]
value = sys.argv[2].strip()

items = [item for item in csv.split(",") if item]
if value and value not in items:
    items.append(value)

print(",".join(items))
PYEOF
}

csv_lines() {
    local csv="${1:-}"
    if [[ -n "${csv}" ]]; then
        printf '%s\n' "${csv}" | tr ',' '\n' | sed '/^$/d'
    fi
}

recent_nemoclaw_registry_sandbox_name() {
    local registry_file
    registry_file="$(thor_nemoclaw_registry_file)"
    [[ -f "${registry_file}" ]] || return 1

    python3 - "${registry_file}" <<'PYEOF'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)

sandboxes = data.get("sandboxes", {})
if not sandboxes:
    raise SystemExit(1)

def sort_key(item):
    name, meta = item
    created_at = ""
    if isinstance(meta, dict):
        created_at = meta.get("createdAt") or ""
    return (created_at, name)

name, _ = max(sandboxes.items(), key=sort_key)
print(name)
PYEOF
}

single_nemoclaw_registry_sandbox_name() {
    local registry_file
    registry_file="$(thor_nemoclaw_registry_file)"
    [[ -f "${registry_file}" ]] || return 1

    python3 - "${registry_file}" <<'PYEOF'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)

sandboxes = data.get("sandboxes", {})
names = list(sandboxes.keys())
if len(names) != 1:
    raise SystemExit(1)
print(names[0])
PYEOF
}

single_openshell_sandbox_name() {
    command -v openshell &>/dev/null || return 1

    local sandbox_names
    sandbox_names=$(openshell sandbox list 2>/dev/null \
        | sed 's/\x1b\[[0-9;]*m//g' \
        | awk 'NR>1 && $1 != "" {print $1}' || true)

    local count
    count=$(printf '%s\n' "${sandbox_names}" | sed '/^$/d' | wc -l | tr -d ' ')
    [[ "${count}" == "1" ]] || return 1
    printf '%s\n' "${sandbox_names}" | sed '/^$/d'
}

resolve_thor_sandbox_name() {
    if [[ -n "${THOR_MANAGED_SANDBOX_NAME:-}" ]]; then
        printf '%s\n' "${THOR_MANAGED_SANDBOX_NAME}"
        return 0
    fi

    local fallback=""
    fallback=$(single_nemoclaw_registry_sandbox_name 2>/dev/null || true)
    if [[ -n "${fallback}" ]]; then
        printf '%s\n' "${fallback}"
        return 0
    fi

    fallback=$(single_openshell_sandbox_name 2>/dev/null || true)
    if [[ -n "${fallback}" ]]; then
        printf '%s\n' "${fallback}"
        return 0
    fi

    return 1
}

print_supported_model_profiles() {
    cat <<'EOF'
Supported model profiles:
  qwen3.5-122b-a10b-nvfp4-resharded
  qwen3.5-27b-fp8
  qwen3.5-35b-a3b-fp8
  qwen3.5-35b-a3b-nvfp4
EOF
}

normalize_model_profile() {
    echo "${1}" | tr '[:upper:]' '[:lower:]'
}

resolve_model_profile() {
    local requested
    requested=$(normalize_model_profile "${1:-${THOR_MODEL_PROFILE:-qwen3.5-27b-fp8}}")

    case "${requested}" in
        qwen3.5-122b-a10b-nvfp4-resharded)
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="Qwen3.5-122B-A10B-NVFP4-resharded"
            THOR_TARGET_MAX_MODEL_LEN="65536"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="8"
            ;;
        qwen3.5-27b-fp8)
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="Qwen3.5-27B-FP8"
            THOR_TARGET_MAX_MODEL_LEN="65536"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="8"
            ;;
        qwen3.5-35b-a3b-fp8)
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="Qwen3.5-35B-A3B-FP8"
            THOR_TARGET_MAX_MODEL_LEN="65536"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="8"
            ;;
        qwen3.5-35b-a3b-nvfp4)
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="Qwen3.5-35B-A3B-NVFP4"
            THOR_TARGET_MAX_MODEL_LEN="65536"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="8"
            ;;
        *)
            echo "Unsupported model profile: ${requested}" >&2
            print_supported_model_profiles >&2
            return 1
            ;;
    esac

    THOR_MODEL_ID="${THOR_MODEL_ID:-${THOR_MODEL_ID_DEFAULT}}"
    THOR_POLICY_PROFILE="${THOR_POLICY_PROFILE:-strict-local}"
    THOR_LOCAL_PROVIDER_NAME="${THOR_LOCAL_PROVIDER_NAME:-vllm-local}"
    THOR_LOCAL_VLLM_BASE_URL="${THOR_LOCAL_VLLM_BASE_URL:-http://host.openshell.internal:8000/v1}"
    THOR_HOST_VLLM_MODELS_URL="${THOR_HOST_VLLM_MODELS_URL:-http://127.0.0.1:8000/v1/models}"
    THOR_LOCAL_VLLM_API_KEY="${THOR_LOCAL_VLLM_API_KEY:-dummy}"
    THOR_MANAGED_SANDBOX_NAME="${THOR_MANAGED_SANDBOX_NAME:-}"
    THOR_MANAGED_PROVIDER_NAMES="${THOR_MANAGED_PROVIDER_NAMES:-}"
}

load_thor_runtime_config() {
    local selected_profile="${1:-}"
    local env_profile="${THOR_MODEL_PROFILE:-}"
    local env_model_id="${THOR_MODEL_ID:-}"
    local env_provider_name="${THOR_LOCAL_PROVIDER_NAME:-}"
    local env_base_url="${THOR_LOCAL_VLLM_BASE_URL:-}"
    local env_host_models_url="${THOR_HOST_VLLM_MODELS_URL:-}"
    local env_api_key="${THOR_LOCAL_VLLM_API_KEY:-}"
    local env_policy_profile="${THOR_POLICY_PROFILE:-}"
    local env_managed_sandbox_name="${THOR_MANAGED_SANDBOX_NAME:-}"
    local env_managed_provider_names="${THOR_MANAGED_PROVIDER_NAMES:-}"
    local env_target_max_model_len="${THOR_TARGET_MAX_MODEL_LEN:-}"
    local env_target_kv_cache_dtype="${THOR_TARGET_KV_CACHE_DTYPE:-}"
    local env_target_max_num_seqs="${THOR_TARGET_MAX_NUM_SEQS:-}"
    local file_target_max_model_len=""
    local file_target_kv_cache_dtype=""
    local file_target_max_num_seqs=""
    THOR_CONFIG_FILE="$(thor_config_file)"

    if [[ -f "${THOR_CONFIG_FILE}" ]]; then
        # shellcheck source=/dev/null
        source "${THOR_CONFIG_FILE}"
        file_target_max_model_len="${THOR_TARGET_MAX_MODEL_LEN:-}"
        file_target_kv_cache_dtype="${THOR_TARGET_KV_CACHE_DTYPE:-}"
        file_target_max_num_seqs="${THOR_TARGET_MAX_NUM_SEQS:-}"
    fi

    if [[ -n "${env_profile}" ]]; then
        THOR_MODEL_PROFILE="${env_profile}"
    fi

    if [[ -n "${selected_profile}" ]]; then
        THOR_MODEL_PROFILE="${selected_profile}"
    fi

    if [[ -n "${env_model_id}" ]]; then
        THOR_MODEL_ID="${env_model_id}"
    elif [[ -n "${env_profile}" || -n "${selected_profile}" ]]; then
        unset THOR_MODEL_ID
    fi

    [[ -n "${env_policy_profile}" ]] && THOR_POLICY_PROFILE="${env_policy_profile}"
    [[ -n "${env_provider_name}" ]] && THOR_LOCAL_PROVIDER_NAME="${env_provider_name}"
    [[ -n "${env_base_url}" ]] && THOR_LOCAL_VLLM_BASE_URL="${env_base_url}"
    [[ -n "${env_host_models_url}" ]] && THOR_HOST_VLLM_MODELS_URL="${env_host_models_url}"
    [[ -n "${env_api_key}" ]] && THOR_LOCAL_VLLM_API_KEY="${env_api_key}"
    [[ -n "${env_managed_sandbox_name}" ]] && THOR_MANAGED_SANDBOX_NAME="${env_managed_sandbox_name}"
    [[ -n "${env_managed_provider_names}" ]] && THOR_MANAGED_PROVIDER_NAMES="${env_managed_provider_names}"

    resolve_model_profile "${THOR_MODEL_PROFILE:-}"

    [[ -n "${file_target_max_model_len}" ]] && THOR_TARGET_MAX_MODEL_LEN="${file_target_max_model_len}"
    [[ -n "${file_target_kv_cache_dtype}" ]] && THOR_TARGET_KV_CACHE_DTYPE="${file_target_kv_cache_dtype}"
    [[ -n "${file_target_max_num_seqs}" ]] && THOR_TARGET_MAX_NUM_SEQS="${file_target_max_num_seqs}"
    [[ -n "${env_target_max_model_len}" ]] && THOR_TARGET_MAX_MODEL_LEN="${env_target_max_model_len}"
    [[ -n "${env_target_kv_cache_dtype}" ]] && THOR_TARGET_KV_CACHE_DTYPE="${env_target_kv_cache_dtype}"
    [[ -n "${env_target_max_num_seqs}" ]] && THOR_TARGET_MAX_NUM_SEQS="${env_target_max_num_seqs}"

    return 0
}

save_thor_runtime_config() {
    local config_dir
    config_dir="$(dirname "${THOR_CONFIG_FILE}")"
    mkdir -p "${config_dir}"

    {
        printf "THOR_MODEL_PROFILE=%q\n" "${THOR_MODEL_PROFILE}"
        printf "THOR_MODEL_ID=%q\n" "${THOR_MODEL_ID}"
        printf "THOR_POLICY_PROFILE=%q\n" "${THOR_POLICY_PROFILE}"
        printf "THOR_LOCAL_PROVIDER_NAME=%q\n" "${THOR_LOCAL_PROVIDER_NAME}"
        printf "THOR_LOCAL_VLLM_BASE_URL=%q\n" "${THOR_LOCAL_VLLM_BASE_URL}"
        printf "THOR_HOST_VLLM_MODELS_URL=%q\n" "${THOR_HOST_VLLM_MODELS_URL}"
        printf "THOR_LOCAL_VLLM_API_KEY=%q\n" "${THOR_LOCAL_VLLM_API_KEY}"
        printf "THOR_MANAGED_SANDBOX_NAME=%q\n" "${THOR_MANAGED_SANDBOX_NAME:-}"
        printf "THOR_MANAGED_PROVIDER_NAMES=%q\n" "${THOR_MANAGED_PROVIDER_NAMES:-}"
        printf "THOR_TARGET_MAX_MODEL_LEN=%q\n" "${THOR_TARGET_MAX_MODEL_LEN}"
        printf "THOR_TARGET_KV_CACHE_DTYPE=%q\n" "${THOR_TARGET_KV_CACHE_DTYPE}"
        printf "THOR_TARGET_MAX_NUM_SEQS=%q\n" "${THOR_TARGET_MAX_NUM_SEQS}"
    } > "${THOR_CONFIG_FILE}"
}

print_thor_runtime_config() {
    echo "  Model profile:     ${THOR_MODEL_PROFILE}"
    echo "  Served model id:   ${THOR_MODEL_ID}"
    echo "  Policy profile:    ${THOR_POLICY_PROFILE}"
    echo "  Provider name:     ${THOR_LOCAL_PROVIDER_NAME}"
    echo "  OpenShell URL:     ${THOR_LOCAL_VLLM_BASE_URL}"
    echo "  Host models URL:   ${THOR_HOST_VLLM_MODELS_URL}"
    if [[ -n "${THOR_MANAGED_SANDBOX_NAME:-}" ]]; then
        echo "  Tracked sandbox:   ${THOR_MANAGED_SANDBOX_NAME}"
    fi
    if [[ -n "${THOR_MANAGED_PROVIDER_NAMES:-}" ]]; then
        echo "  Tracked providers: ${THOR_MANAGED_PROVIDER_NAMES}"
    fi
    echo "  Planned context:   ${THOR_TARGET_MAX_MODEL_LEN}"
    echo "  Planned KV cache:  ${THOR_TARGET_KV_CACHE_DTYPE}"
    echo "  Planned max seqs:  ${THOR_TARGET_MAX_NUM_SEQS}"
}
