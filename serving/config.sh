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

  Qwen3.6-35B-A3B (NVFP4 weights, agentic-tuned — recommended for orchestration):
    qwen3.6-35b-a3b-nvfp4-mtp-fp8kv    ★★ DEFAULT — max correctness (TEB 93, 90% IFEval, 19.5 tps)
    qwen3.6-35b-a3b-nvfp4-tq-mtp       ★★ throughput+context (TEB 90, 89% IFEval, 24.8 tps, 2.22M KV)
    qwen3.6-35b-a3b-nvfp4-dflash       heavy coding bursts (DFlash-8, ~v6 87 TEB, peak ~130 tps)
    qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge  ★ ManyForge production: TQ+MTP-2 + VISION, 3×64K, co-serves w/ Cosmos

  Other Qwen3.6:
    qwen3.6-27b-fp8-mtp-kvfp8     dense 27B FP8 + MTP + FP8 KV (TEB 84)

  Distilled / specialized:
    qwen3.5-9b-claude-distilled-nvfp4  DeltaNet hybrid, 9B Opus-distilled, fast control loop (TEB 42)

  Cosmos (NVIDIA physical-AI VLMs — for embodied/spatial reasoning):
    cosmos-reason2-2b         Qwen3-VL-2B base, 32K ctx, 2-conc
    cosmos-reason2-8b         Qwen3-VL-8B base, 64K ctx, 3-conc (TEB 81)

  Nemotron 3 Omni (NVIDIA multimodal reasoning — vision + audio + text):
    nemotron3-nano-omni-30b-a3b-nvfp4  30B-A3B hybrid MoE, 32K ctx, 1-conc (released 2026-04-28)

  Gemma 4 (Google, vision+text+tools):
    gemma4-e4b-it             BF16 MoE, 8B/4B-active
    gemma4-31b-it-nvfp4       NVFP4 quantized
    gemma4-26b-a4b-it         BF16 MoE 128E/8A
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
    requested=$(normalize_model_profile "${1:-${THOR_MODEL_PROFILE:-qwen3.6-35b-a3b-nvfp4-mtp-fp8kv}}")

    case "${requested}" in
        # minimax-m2.7-139b-a10b-nvfp4 profile removed 2026-04-23.
        # Investigation preserved at MINIMAX-M27-INVESTIGATION.md.
        # W4A4 NVFP4 MoE is blocked on SM110 — every fast kernel gated off, MARLIN
        # fallback produced 12 tok/s with degraded output. Not viable for production.
        # qwen3.5-122b-a10b-nvfp4 profile removed 2026-04-24 — superseded by qwen3.6.
        qwen3.5-9b-claude-distilled-nvfp4)
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="qwen3.5-9b-claude-distilled-nvfp4"
            THOR_TARGET_MAX_MODEL_LEN="131072"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="8"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="2"
            THOR_TARGET_MODEL_REASONING="false"
            THOR_TARGET_MAX_TOKENS="16384"
            ;;
        # qwen3.5-27b-claude-distilled-v2-nvfp4 profile removed 2026-04-24 — superseded by qwen3.6.
        qwen3.6-27b-fp8-mtp-kvfp8)
            # EXPERIMENTAL: Qwen/Qwen3.6-27B-FP8 (official FP8 release) +
            # MTP N=1 + FP8 KV cache.
            #
            # Why this over NVFP4 variants: llm-compressor's NVFP4 toolchain
            # silently strips MTP head tensors during quantization
            # (sakamakismile, stepnivlk, prithivMLmods, selimaktas, Abiray all
            # showed 0 MTP tensors). Only the official FP8 release and
            # mmangkad's ModelOpt NVFP4 preserve MTP heads. FP8 is the safer
            # pick: mainstream format, proven Thor kernel path
            # (TritonFp8BlockScaledMMKernel via VLLM_DISABLED_KERNELS), same
            # ~27 GB weight footprint as stepnivlk.
            #
            # Dense hybrid: 64 layers, full_attention_interval=4, head_dim=256
            # on the 16 full-attn layers, 48 DeltaNet linear_attn layers.
            # head_dim=256 forces FlashInfer attention (FA2 crashes SM110).
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="qwen3.6-27b-fp8-mtp-kvfp8"
            THOR_TARGET_MAX_MODEL_LEN="262144"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="9"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="3"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            ;;
        # qwen3.6-27b-fp8-dflash REMOVED 2026-04-28 — see launch.sh
        # qwen3.6-35b-a3b-fp8-dflash REMOVED 2026-04-28 — see launch.sh; heavy
        # coding now via qwen3.6-35b-a3b-nvfp4-dflash (same NVFP4 weights).
        # qwen3.6-35b-a3b-fp8-mtp-fp8kv REMOVED 2026-04-28 — FP8-weights variant
        # of an NVFP4 alternative; use nvfp4-mtp-fp8kv or nvfp4-tq-mtp instead.
        # qwen3.6-35b-a3b-fp8-turboquant REMOVED 2026-04-28 — see launch.sh.
        qwen3.6-35b-a3b-nvfp4-dflash)
            # ★★ FASTEST: 45.71 tok/s single, 192.45 aggregate at 8-concurrent.
            # 678K KV tokens at 256K context → ~2 full-context concurrent, more at shorter.
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="qwen3.6-35b-a3b-nvfp4-dflash"
            THOR_TARGET_MAX_MODEL_LEN="262144"
            THOR_TARGET_KV_CACHE_DTYPE="bfloat16"
            THOR_TARGET_MAX_NUM_SEQS="5"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="2"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            ;;
        # qwen3.6-35b-a3b-nvfp4-dflash-vl REMOVED 2026-04-28 — vision support
        # folded into qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge. See launch.sh.
        # qwen3.6-35b-a3b-prismaquant-dflash REMOVED 2026-04-28 — was the
        # default; default re-pointed to qwen3.6-35b-a3b-nvfp4-mtp-fp8kv (the
        # 93/100 tool-eval-bench winner). PrismaQuant 4.75bpp + DFlash-15 was
        # tuned for throughput on a different methodology and proved agentic-
        # weak on our v7 bench (DFlash N=15 → TEB 40-46 across all DFlash
        # variants).
        qwen3.6-35b-a3b-nvfp4-mtp-fp8kv)
            # EXPERIMENTAL (re-added 2026-04-23 for tool-eval-bench quality testing).
            # NVFP4 weights + MTP N=2 + FP8 KV. MTP N=2 mirrors the 27B-FP8 winning
            # config. Older sibling with MTP N=4 crashed under 8-concurrent; this
            # N=2 variant reduces crash risk but still carries the same dtype mix.
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="qwen3.6-35b-a3b-nvfp4-mtp-fp8kv"
            THOR_TARGET_MAX_MODEL_LEN="262144"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="5"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="2"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            ;;
        # qwen3.6-35b-a3b-nvfp4-mtp-fp8kv-n4 REMOVED 2026-04-28 — variance probe
        # done; TEB 91 confirmed N=2 is the right pick for FP8 KV.
        qwen3.6-35b-a3b-nvfp4-tq-mtp)
            # ★ MAX CONTEXT: 28.0 tok/s, 79% acceptance, 2.22M KV tokens
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="qwen3.6-35b-a3b-nvfp4-tq-mtp"
            THOR_TARGET_MAX_MODEL_LEN="262144"
            THOR_TARGET_KV_CACHE_DTYPE="turboquant_k8v4"
            THOR_TARGET_MAX_NUM_SEQS="8"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="2"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            ;;
        # qwen3.6-35b-a3b-nvfp4-tq-mtp-2 REMOVED 2026-04-28 — TQ + N=2 dominated
        # by TQ + N=4 (TEB 87 vs 90 at same KV).
        # qwen3.6-35b-a3b-nvfp4-mtp-fp8kv removed — crashes under 8-concurrent
        # (CUDA illegal memory in MoE autotuner at M=128 on SM110). Superseded
        # by qwen3.6-35b-a3b-nvfp4-tq-mtp which is strictly better: larger KV
        # budget (2.22M vs 1.68M), higher concurrency (29x vs 10x), works
        # under load (153 tok/s aggregate at 8 concurrent).
        # qwen3.5-35b-a3b-nvfp4 removed — superseded by qwen3.6
        qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge)
            # Production ManyForge profile: 3×64K context, TQ K8V4 + MTP N=2,
            # gpu_mem_util 0.32 so we can co-serve cosmos-reason2-2b on Thor.
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge"
            THOR_TARGET_MAX_MODEL_LEN="65536"
            THOR_TARGET_KV_CACHE_DTYPE="turboquant_k8v4"
            THOR_TARGET_MAX_NUM_SEQS="3"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="2"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="8192"
            THOR_TARGET_QUANTIZATION=""
            ;;
        cosmos-reason2-2b)
            # NVIDIA Cosmos Reason 2 (2B), Qwen3-VL-2B base, VLM physical-AI reasoner.
            # Sized for 2×32K concurrent context with FP8 KV (see launch.sh comment).
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="cosmos-reason2-2b"
            THOR_TARGET_MAX_MODEL_LEN="32768"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="2"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="2"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            THOR_TARGET_QUANTIZATION=""
            ;;
        cosmos-reason2-8b)
            # NVIDIA Cosmos Reason 2 (8B), Qwen3-VL-8B base, VLM physical-AI reasoner.
            # Model supports up to 262144 tokens natively (text_config.max_position_embeddings).
            # Sized for 64K context to accommodate OpenClaw's ~16K system prompt + 16K output
            # (32K ceiling was too tight: prompt+output overflows). FP8 KV keeps footprint low.
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="cosmos-reason2-8b"
            THOR_TARGET_MAX_MODEL_LEN="65536"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="3"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="2"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            THOR_TARGET_QUANTIZATION=""
            ;;
        nemotron3-nano-omni-30b-a3b-nvfp4)
            # NVIDIA Nemotron 3 Nano Omni (released 2026-04-28). 30B-A3B
            # hybrid Mamba-Transformer MoE + multimodal (vision/video/audio
            # encoders). NVFP4 quant 20.9 GB on disk. Replaces the text-only
            # nemotron3-nano-30b-a3b-nvfp4 (TEB 67) with a multimodal variant
            # that may shift the agentic ceiling. See MANYFORGE-ASSISTANT-
            # DEPLOYMENT-PLAN.md § Outcome D for the deployment rationale.
            #
            # ManyForge-pipeline calibration (2026-04-30, see
            # MANYFORGE-PROFILE-CALIBRATION.md for the methodology):
            #   - max_model_len=262144 (256K, model native max).
            #     OpenClaw bootstrap is ~16K + 16K output budget; the
            #     prior 32K target overflowed on turn 1. 256K gives
            #     comfortable multi-turn headroom for hybrid agentic
            #     loops and accumulated tool-result history.
            #   - max_num_seqs=16. Sized to absorb GUI assistant +
            #     parallel BT query_model nodes + future subagents
            #     without queueing. Per spec 480 §2.4 the assistant
            #     surface is single-user but may fan out across
            #     parallel BT branches. 16 covers 4 main agents × 3
            #     subagents with headroom; revise downward if main-
            #     agent budget is capped lower.
            #   - gpu_memory_utilization=0.50 (set in launch.sh, not
            #     here). Empirical KV-pool sizing keeps ~32× supportable
            #     concurrency at 256K (vs the 16 we configure), leaving
            #     ~14 GB freed for Isaac ROS / system / cluster gateway.
            #     See MANYFORGE-PROFILE-CALIBRATION.md for the math.
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="nemotron3-nano-omni-30b-a3b-nvfp4"
            THOR_TARGET_MAX_MODEL_LEN="262144"
            THOR_TARGET_KV_CACHE_DTYPE="fp8"
            THOR_TARGET_MAX_NUM_SEQS="16"
            THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT="1"
            THOR_TARGET_MODEL_REASONING="true"
            THOR_TARGET_MAX_TOKENS="16384"
            THOR_TARGET_TOOL_CALL_PARSER="qwen3_coder"
            THOR_TARGET_QUANTIZATION=""
            ;;
        # cosmos-reason2-8b-reasoning REMOVED 2026-04-28 — broken-tuning
        # experiment, see launch.sh. Use cosmos-reason2-8b instead.
        # nemotron3-nano-30b-a3b-nvfp4 REMOVED 2026-04-28 — TEB 67/100
        # mid-pack on Thor (text-only). Replaced by nemotron3-nano-omni
        # (multimodal Reasoning variant). See PERFORMANCE-V7.md.
        gemma4-e4b-it)
            THOR_MODEL_PROFILE="${requested}"
            THOR_MODEL_ID_DEFAULT="gemma4-e4b-it"
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
            THOR_MODEL_ID_DEFAULT="gemma4-31b-it-nvfp4"
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
            THOR_MODEL_ID_DEFAULT="gemma4-26b-a4b-it"
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
