#!/usr/bin/env bash
# lib/config.sh — Shared runtime config for NemoClaw-Thor
#
# Source this file; do not execute it directly.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This file is meant to be sourced, not executed directly." >&2
    exit 1
fi

thor_config_file() {
    local config_home
    config_home="${XDG_CONFIG_HOME:-$HOME/.config}"
    echo "${NEMOCLAW_THOR_CONFIG_FILE:-${config_home}/nemoclaw-thor/config.env}"
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
    requested=$(normalize_model_profile "${1:-${THOR_MODEL_PROFILE:-qwen3.5-35b-a3b-fp8}}")

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
    THOR_CONFIG_FILE="$(thor_config_file)"

    if [[ -f "${THOR_CONFIG_FILE}" ]]; then
        # shellcheck source=/dev/null
        source "${THOR_CONFIG_FILE}"
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

    resolve_model_profile "${THOR_MODEL_PROFILE:-}"
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
    echo "  Planned context:   ${THOR_TARGET_MAX_MODEL_LEN}"
    echo "  Planned KV cache:  ${THOR_TARGET_KV_CACHE_DTYPE}"
    echo "  Planned max seqs:  ${THOR_TARGET_MAX_NUM_SEQS}"
}
