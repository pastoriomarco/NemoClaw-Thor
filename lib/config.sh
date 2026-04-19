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
  qwen3.6-35b-a3b-nvfp4-dflash    (DFlash-15, 45.7 single / 192.5 @8-conc, 256K ctx) ★★ FASTEST
  qwen3.6-35b-a3b-nvfp4-dflash-vl (DFlash + vision enabled, experimental)
  qwen3.6-35b-a3b-fp8-dflash      (DFlash-15, 47.6 tok/s, ~700K KV) ★ MAX THROUGHPUT FP8
  qwen3.6-35b-a3b-nvfp4-tq-mtp   (TQ K8V4 + MTP, 28.6 single / 153.6 @8-conc, 256K ctx) ★ MAX CONTEXT
  qwen3.6-35b-a3b-fp8-mtp-fp8kv   (MTP N=4 + FP8 KV, 25.7 tok/s, 1.44M KV)
  qwen3.6-35b-a3b-fp8-turboquant  (TQ K8V4 + MTP, 26.2 tok/s, 1.89M KV)
  qwen3.5-9b-claude-distilled-nvfp4  (DeltaNet hybrid, 9B Opus-distilled NVFP4)
  gemma4-e4b-it             (vLLM, BF16 MoE, 8B/4B-active, vision+text+tools)
  gemma4-31b-it-nvfp4       (vLLM, NVFP4 quantized, vision+text+tools)
  gemma4-26b-a4b-it         (vLLM, BF16 MoE 128E/8A, vision+text+tools)
EOF
}

normalize_model_profile() {
    echo "${1}" | tr '[:upper:]' '[:lower:]'
}

resolve_model_profile() {
    local requested
    # Default profile: NVFP4 + DFlash-15 (FASTEST — 45.7 tok/s single, 192.5 @ 8-concurrent,
    # 256K context). Users without an HF token for the gated drafter can override via
    # THOR_MODEL_PROFILE or arg, e.g. `./start-model.sh qwen3.6-35b-a3b-fp8-dflash`.
    requested=$(normalize_model_profile "${1:-${THOR_MODEL_PROFILE:-qwen3.6-35b-a3b-nvfp4-dflash}}")

    case "${requested}" in
        qwen3.5-122b-a10b-nvfp4)
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="Sehyo/Qwen3.5-122B-A10B-NVFP4"
            THOR_TARGET_MAX_MODEL_LEN="262144"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="3"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="1"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            ;;
        qwen3.5-9b-claude-distilled-nvfp4)
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="Qwen3.5-9B-Claude-Distilled-NVFP4"
            THOR_TARGET_MAX_MODEL_LEN="131072"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="8"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="2"
            THOR_TARGET_MODEL_REASONING="false"
            THOR_TARGET_MAX_TOKENS="16384"
            ;;
        qwen3.5-27b-claude-distilled-v2-nvfp4)
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="Qwen3.5-27B-Claude-Distilled-v2-NVFP4"
            THOR_TARGET_MAX_MODEL_LEN="262144"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="9"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="3"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            ;;
        qwen3.6-35b-a3b-fp8-dflash)
            # ~700K KV tokens at 0.8 (BF16 KV + DFlash drafter overhead)
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="Qwen3.6-35B-A3B-FP8-DFlash"
            THOR_TARGET_MAX_MODEL_LEN="131072"
            THOR_TARGET_KV_CACHE_DTYPE="bfloat16"
            THOR_TARGET_MAX_NUM_SEQS="4"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="1"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            ;;
        qwen3.6-35b-a3b-fp8-mtp-fp8kv)
            # 1.44M KV tokens at 0.8 (FP8 KV, ~2x compression)
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="Qwen3.6-35B-A3B-FP8-MTP-FP8KV"
            THOR_TARGET_MAX_MODEL_LEN="131072"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="8"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="2"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            ;;
        qwen3.6-35b-a3b-fp8-turboquant)
            # 1.89M KV tokens at 0.8 (TQ K8V4, ~2.6x compression)
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="Qwen3.6-35B-A3B-FP8-TQ"
            THOR_TARGET_MAX_MODEL_LEN="262144"
            THOR_TARGET_KV_CACHE_DTYPE="turboquant_k8v4"
            THOR_TARGET_MAX_NUM_SEQS="6"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="2"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            ;;
        qwen3.6-35b-a3b-nvfp4-dflash)
            # ★★ FASTEST: 45.71 tok/s single, 192.45 aggregate at 8-concurrent.
            # 678K KV tokens at 256K context → ~2 full-context concurrent, more at shorter.
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="Qwen3.6-35B-A3B-NVFP4-DFlash"
            THOR_TARGET_MAX_MODEL_LEN="262144"
            THOR_TARGET_KV_CACHE_DTYPE="bfloat16"
            THOR_TARGET_MAX_NUM_SEQS="5"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="2"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            ;;
        qwen3.6-35b-a3b-nvfp4-dflash-vl)
            # EXPERIMENTAL: DFlash with vision enabled. Validated 2026-04-19:
            # ViT works on SM110 with TORCH_SDPA, DFlash drafter coexists with
            # multimodal prompts, ~2s latency per vision request after warmup.
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="Qwen3.6-35B-A3B-NVFP4-DFlash-VL"
            THOR_TARGET_MAX_MODEL_LEN="262144"
            THOR_TARGET_KV_CACHE_DTYPE="bfloat16"
            THOR_TARGET_MAX_NUM_SEQS="5"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="2"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            ;;
        qwen3.6-35b-a3b-nvfp4-tq-mtp)
            # ★ MAX CONTEXT: 28.0 tok/s, 79% acceptance, 2.22M KV tokens
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="Qwen3.6-35B-A3B-NVFP4-TQ-MTP"
            THOR_TARGET_MAX_MODEL_LEN="262144"
            THOR_TARGET_KV_CACHE_DTYPE="turboquant_k8v4"
            THOR_TARGET_MAX_NUM_SEQS="8"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="2"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            ;;
        # qwen3.6-35b-a3b-nvfp4-mtp-fp8kv removed — crashes under 8-concurrent
        # (CUDA illegal memory in MoE autotuner at M=128 on SM110). Superseded
        # by qwen3.6-35b-a3b-nvfp4-tq-mtp which is strictly better: larger KV
        # budget (2.22M vs 1.68M), higher concurrency (29x vs 10x), works
        # under load (153 tok/s aggregate at 8 concurrent).
        # qwen3.5-35b-a3b-nvfp4 removed — superseded by qwen3.6
        gemma4-e4b-it)
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="gemma-4-E4B-it"
            THOR_TARGET_MAX_MODEL_LEN="131072"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="12"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="3"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            THOR_TARGET_TOOL_CALL_PARSER="gemma4"
            THOR_TARGET_QUANTIZATION=""
            ;;
        gemma4-31b-it-nvfp4)
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="Gemma-4-31B-IT-NVFP4"
            THOR_TARGET_MAX_MODEL_LEN="262144"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="6"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="6"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            THOR_TARGET_TOOL_CALL_PARSER="gemma4"
            THOR_TARGET_QUANTIZATION="modelopt"
            ;;
        gemma4-26b-a4b-it)
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="gemma-4-26B-A4B-it"
            THOR_TARGET_MAX_MODEL_LEN="262144"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="17"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="4"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            THOR_TARGET_TOOL_CALL_PARSER="gemma4"
            THOR_TARGET_QUANTIZATION=""
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
    THOR_MANYFORGE_MUX_ENABLED="${THOR_MANYFORGE_MUX_ENABLED:-false}"
    THOR_MANYFORGE_MUX_PORT="${THOR_MANYFORGE_MUX_PORT:-8888}"
    THOR_DASHBOARD_PORT="${THOR_DASHBOARD_PORT:-${NEMOCLAW_DASHBOARD_PORT:-18789}}"

    # The sandbox/OpenClaw client should always talk to inference.local. The
    # underlying OpenShell provider target is what changes between direct vLLM
    # and ManyForge mux mode.
    THOR_OPENCLAW_BASE_URL="${THOR_OPENCLAW_BASE_URL:-https://inference.local/v1}"

    # When ManyForge mux is enabled, the provider target URL MUST point to the
    # mux proxy — this is not a default, it's a forced override. When disabled,
    # reset to direct vLLM if the URL still points to the mux port (cleanup
    # from a previous mux-enabled run); otherwise keep the saved/user value.
    # THOR_HOST_VLLM_MODELS_URL always points directly to vLLM for health checks.
    if [[ "${THOR_MANYFORGE_MUX_ENABLED}" == "true" ]]; then
        THOR_LOCAL_VLLM_BASE_URL="http://host.openshell.internal:${THOR_MANYFORGE_MUX_PORT}/v1"
    else
        case "${THOR_LOCAL_VLLM_BASE_URL:-}" in
            *":${THOR_MANYFORGE_MUX_PORT}/"*)
                THOR_LOCAL_VLLM_BASE_URL="http://host.openshell.internal:8000/v1"
                ;;
            *)
                THOR_LOCAL_VLLM_BASE_URL="${THOR_LOCAL_VLLM_BASE_URL:-http://host.openshell.internal:8000/v1}"
                ;;
        esac
    fi
    THOR_HOST_VLLM_MODELS_URL="${THOR_HOST_VLLM_MODELS_URL:-http://127.0.0.1:8000/v1/models}"
    THOR_LOCAL_VLLM_API_KEY="${THOR_LOCAL_VLLM_API_KEY:-dummy}"
    THOR_OPENSHELL_GATEWAY_NAME="${THOR_OPENSHELL_GATEWAY_NAME:-nemoclaw}"
    THOR_MANAGED_SANDBOX_NAME="${THOR_MANAGED_SANDBOX_NAME:-}"
    THOR_MANAGED_PROVIDER_NAMES="${THOR_MANAGED_PROVIDER_NAMES:-}"
    THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="${THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT:-1}"
}

resolve_thor_openclaw_concurrency_targets() {
    local resolved=""

    resolved=$(python3 - "${THOR_TARGET_MAX_NUM_SEQS:-}" "${THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT:-1}" <<'PYEOF'
import sys
import math

def parse_positive_int(value, default):
    try:
        parsed = int(str(value).strip() or "")
    except Exception:
        return default
    return parsed if parsed >= 1 else default

max_num_seqs = parse_positive_int(sys.argv[1], 1)
requested_main = parse_positive_int(sys.argv[2], 1)

if max_num_seqs > 1:
    effective_main = min(requested_main, max_num_seqs - 1)
    subagent_slots = max_num_seqs - effective_main
    # Fair share: each main agent can spawn up to ceil(slots / mains) children,
    # capped at 4 to prevent one agent from dominating on high-slot models.
    children_per_agent = min(4, max(1, math.ceil(subagent_slots / effective_main)))
else:
    effective_main = 1
    subagent_slots = 1
    children_per_agent = 1

print("\t".join(
    str(value)
    for value in (
        requested_main,
        effective_main,
        subagent_slots,
        children_per_agent,
        1,
    )
))
PYEOF
)

    IFS=$'\t' read -r \
        THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT \
        THOR_EFFECTIVE_OPENCLAW_MAIN_MAX_CONCURRENT \
        THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CONCURRENT \
        THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CHILDREN \
        THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_SPAWN_DEPTH <<<"${resolved}"
}

load_thor_runtime_config() {
    local selected_profile="${1:-}"
    local env_profile="${THOR_MODEL_PROFILE:-}"
    local env_model_id="${THOR_MODEL_ID:-}"
    local env_provider_name="${THOR_LOCAL_PROVIDER_NAME:-}"
    local env_openclaw_base_url="${THOR_OPENCLAW_BASE_URL:-}"
    local env_base_url="${THOR_LOCAL_VLLM_BASE_URL:-}"
    local env_host_models_url="${THOR_HOST_VLLM_MODELS_URL:-}"
    local env_api_key="${THOR_LOCAL_VLLM_API_KEY:-}"
    local env_gateway_name="${THOR_OPENSHELL_GATEWAY_NAME:-}"
    local env_policy_profile="${THOR_POLICY_PROFILE:-}"
    local env_managed_sandbox_name="${THOR_MANAGED_SANDBOX_NAME:-}"
    local env_managed_provider_names="${THOR_MANAGED_PROVIDER_NAMES:-}"
    local env_manyforge_mux_enabled="${THOR_MANYFORGE_MUX_ENABLED:-}"
    local env_manyforge_mux_port="${THOR_MANYFORGE_MUX_PORT:-}"
    local env_target_model_reasoning="${THOR_TARGET_MODEL_REASONING:-}"
    local env_target_max_model_len="${THOR_TARGET_MAX_MODEL_LEN:-}"
    local env_target_kv_cache_dtype="${THOR_TARGET_KV_CACHE_DTYPE:-}"
    local env_target_max_num_seqs="${THOR_TARGET_MAX_NUM_SEQS:-}"
    local env_openclaw_main_max_concurrent="${THOR_OPENCLAW_MAIN_MAX_CONCURRENT:-}"
    local env_target_openclaw_main_max_concurrent="${THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT:-}"
    local file_target_max_model_len=""
    local file_target_kv_cache_dtype=""
    local file_target_max_num_seqs=""
    local file_target_openclaw_main_max_concurrent=""
    THOR_CONFIG_FILE="$(thor_config_file)"

    if [[ -f "${THOR_CONFIG_FILE}" ]]; then
        # shellcheck source=/dev/null
        source "${THOR_CONFIG_FILE}"
        file_target_max_model_len="${THOR_TARGET_MAX_MODEL_LEN:-}"
        file_target_kv_cache_dtype="${THOR_TARGET_KV_CACHE_DTYPE:-}"
        file_target_max_num_seqs="${THOR_TARGET_MAX_NUM_SEQS:-}"
        file_target_openclaw_main_max_concurrent="${THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT:-}"
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

    [[ -n "${env_provider_name}" ]] && THOR_LOCAL_PROVIDER_NAME="${env_provider_name}"
    [[ -n "${env_openclaw_base_url}" ]] && THOR_OPENCLAW_BASE_URL="${env_openclaw_base_url}"
    [[ -n "${env_base_url}" ]] && THOR_LOCAL_VLLM_BASE_URL="${env_base_url}"
    [[ -n "${env_host_models_url}" ]] && THOR_HOST_VLLM_MODELS_URL="${env_host_models_url}"
    [[ -n "${env_api_key}" ]] && THOR_LOCAL_VLLM_API_KEY="${env_api_key}"
    [[ -n "${env_gateway_name}" ]] && THOR_OPENSHELL_GATEWAY_NAME="${env_gateway_name}"
    [[ -n "${env_policy_profile}" ]] && THOR_POLICY_PROFILE="${env_policy_profile}"
    [[ -n "${env_managed_sandbox_name}" ]] && THOR_MANAGED_SANDBOX_NAME="${env_managed_sandbox_name}"
    [[ -n "${env_managed_provider_names}" ]] && THOR_MANAGED_PROVIDER_NAMES="${env_managed_provider_names}"
    [[ -n "${env_manyforge_mux_enabled}" ]] && THOR_MANYFORGE_MUX_ENABLED="${env_manyforge_mux_enabled}"
    [[ -n "${env_manyforge_mux_port}" ]] && THOR_MANYFORGE_MUX_PORT="${env_manyforge_mux_port}"

    resolve_model_profile "${THOR_MODEL_PROFILE:-}"

    # When a model profile is explicitly selected (via CLI argument or env),
    # its per-model defaults (max_num_seqs, etc.) take precedence over saved
    # config.env values. Saved config only overrides when no profile was
    # explicitly given — this prevents stale values from a previous model
    # from overriding the current profile's tuned parameters.
    local profile_explicitly_selected="false"
    if [[ -n "${selected_profile}" || -n "${env_profile}" ]]; then
        profile_explicitly_selected="true"
    fi

    if [[ "${profile_explicitly_selected}" == "false" ]]; then
        [[ -n "${file_target_max_model_len}" ]] && THOR_TARGET_MAX_MODEL_LEN="${file_target_max_model_len}"
        [[ -n "${file_target_kv_cache_dtype}" ]] && THOR_TARGET_KV_CACHE_DTYPE="${file_target_kv_cache_dtype}"
        [[ -n "${file_target_max_num_seqs}" ]] && THOR_TARGET_MAX_NUM_SEQS="${file_target_max_num_seqs}"
        [[ -n "${file_target_openclaw_main_max_concurrent}" ]] && THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="${file_target_openclaw_main_max_concurrent}"
    fi

    # Explicit env vars always win (user override at invocation time).
    [[ -n "${env_target_model_reasoning}" ]] && THOR_TARGET_MODEL_REASONING="${env_target_model_reasoning}"
    [[ -n "${env_target_max_model_len}" ]] && THOR_TARGET_MAX_MODEL_LEN="${env_target_max_model_len}"
    [[ -n "${env_target_kv_cache_dtype}" ]] && THOR_TARGET_KV_CACHE_DTYPE="${env_target_kv_cache_dtype}"
    [[ -n "${env_target_max_num_seqs}" ]] && THOR_TARGET_MAX_NUM_SEQS="${env_target_max_num_seqs}"
    [[ -n "${env_target_openclaw_main_max_concurrent}" ]] && THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="${env_target_openclaw_main_max_concurrent}"
    [[ -n "${env_openclaw_main_max_concurrent}" ]] && THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="${env_openclaw_main_max_concurrent}"

    resolve_thor_openclaw_concurrency_targets

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
        printf "THOR_OPENCLAW_BASE_URL=%q\n" "${THOR_OPENCLAW_BASE_URL}"
        printf "THOR_LOCAL_VLLM_BASE_URL=%q\n" "${THOR_LOCAL_VLLM_BASE_URL}"
        printf "THOR_HOST_VLLM_MODELS_URL=%q\n" "${THOR_HOST_VLLM_MODELS_URL}"
        printf "THOR_LOCAL_VLLM_API_KEY=%q\n" "${THOR_LOCAL_VLLM_API_KEY}"
        printf "THOR_OPENSHELL_GATEWAY_NAME=%q\n" "${THOR_OPENSHELL_GATEWAY_NAME}"
        printf "THOR_MANAGED_SANDBOX_NAME=%q\n" "${THOR_MANAGED_SANDBOX_NAME:-}"
        printf "THOR_MANAGED_PROVIDER_NAMES=%q\n" "${THOR_MANAGED_PROVIDER_NAMES:-}"
        printf "THOR_TARGET_MAX_MODEL_LEN=%q\n" "${THOR_TARGET_MAX_MODEL_LEN}"
        printf "THOR_TARGET_KV_CACHE_DTYPE=%q\n" "${THOR_TARGET_KV_CACHE_DTYPE}"
        printf "THOR_TARGET_MAX_NUM_SEQS=%q\n" "${THOR_TARGET_MAX_NUM_SEQS}"
        printf "THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT=%q\n" "${THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT}"
        printf "THOR_TARGET_MODEL_REASONING=%q\n" "${THOR_TARGET_MODEL_REASONING}"
        printf "THOR_MANYFORGE_MUX_ENABLED=%q\n" "${THOR_MANYFORGE_MUX_ENABLED:-false}"
        printf "THOR_MANYFORGE_MUX_PORT=%q\n" "${THOR_MANYFORGE_MUX_PORT:-8888}"
    } > "${THOR_CONFIG_FILE}"
}

print_thor_runtime_config() {
    echo "  Model profile:     ${THOR_MODEL_PROFILE}"
    echo "  Served model id:   ${THOR_MODEL_ID}"
    echo "  Policy profile:    ${THOR_POLICY_PROFILE}"
    echo "  Provider name:     ${THOR_LOCAL_PROVIDER_NAME}"
    echo "  Gateway name:      ${THOR_OPENSHELL_GATEWAY_NAME}"
    echo "  Sandbox base URL:  ${THOR_OPENCLAW_BASE_URL}"
    echo "  Provider target:   ${THOR_LOCAL_VLLM_BASE_URL}"
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
    echo "  OpenClaw main:     ${THOR_EFFECTIVE_OPENCLAW_MAIN_MAX_CONCURRENT} (requested ${THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT})"
    echo "  OpenClaw subagent: ${THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CONCURRENT}"
    if [[ "${THOR_MANYFORGE_MUX_ENABLED:-false}" == "true" ]]; then
        echo "  ManyForge mux:     enabled (port ${THOR_MANYFORGE_MUX_PORT:-8888})"
    fi
}
