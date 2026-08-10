#!/usr/bin/env bash
# Launch Entrpi/ds4 with the 0731 base + its matching DSpark drafter.
# Do not add the legacy MTP GGUF here: 0731 has no compatible MTP head.

set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    exec /usr/local/bin/ds4-server --help
fi

if [[ "${1:-}" == "ds4-server" ]]; then
    shift
fi

base="${DS4_MODEL_DIR}/${DS4_BASE_FILE}"
drafter="${DS4_MODEL_DIR}/${DS4_DSPARK_FILE}"

weights_ready() {
    [[ -s "${base}" && -s "${drafter}" ]]
}

if ! weights_ready; then
    if [[ "${DS4_WAIT_FOR_WEIGHTS:-0}" != "1" ]]; then
        echo "DS4 0731 GGUF pair is missing under ${DS4_MODEL_DIR}." >&2
        echo "Run: ./serving/start-ds4.sh download" >&2
        exit 2
    fi
    echo "Waiting for the persistent 0731 base + DSpark drafter download..." >&2
    until weights_ready; do
        sleep 60
    done
fi

mkdir -p "${DS4_KV_DISK_DIR}"

# These are the upstream v0.5.5+ DSpark launch settings.  --no-mtp is
# deliberate: the legacy MTP GGUF must never be loaded with a 0731 base.
export DS4_CONT_MTP_MODE="${DS4_CONT_MTP_MODE:-2}"
export DS4_CONT_DSPARK="${DS4_CONT_DSPARK:-1}"
export DS4_DSPARK_MODEL="${DS4_DSPARK_MODEL:-${drafter}}"

# The current Thor builds use Entrpi v0.5.5+'s repaired streaming selector. The
# historical v0.5.4 Thor profile uses the atomics-free deterministic selector.
# Enable streaming only when either pinned provenance marker proves a safe
# implementation is present. A clean upstream image keeps the safe tree unless
# explicitly enabled, and NO_TOPK_STREAM=1 forces that rollback on any image.
build_profile="$(sed -n 's/^profile=//p' /etc/ds4-build.txt 2>/dev/null | head -n 1)"
topk_stream_default=0
if [[ "${build_profile}" == thor-topkdet256* ||
      "${build_profile}" == thor-upstream055* ||
      "${build_profile}" == thor-upstream0562* ]]; then
    topk_stream_default=1
fi
if [[ "${DS4_CUDA_NO_TOPK_STREAM:-0}" == "1" ]]; then
    export DS4_CUDA_NO_TOPK_STREAM=1
    unset DS4_CUDA_TOPK_STREAM
    selector_mode="safe-tree"
elif [[ "${DS4_CUDA_TOPK_STREAM:-${topk_stream_default}}" == "1" ]]; then
    unset DS4_CUDA_NO_TOPK_STREAM
    export DS4_CUDA_TOPK_STREAM=1
    selector_mode="streaming"
else
    export DS4_CUDA_NO_TOPK_STREAM=1
    unset DS4_CUDA_TOPK_STREAM
    selector_mode="safe-tree"
fi
printf 'DS4 build profile=%s selector=%s\n' \
    "${build_profile:-upstream}" "${selector_mode}" >&2
if [[ "${DS4_CUDA_TOPK_STREAM_VERIFY:-0}" != "1" ]]; then
    unset DS4_CUDA_TOPK_STREAM_VERIFY
fi
if [[ -z "${DS4_CUDA_ATTN_HG_SPLIT_N:-}" ]]; then
    unset DS4_CUDA_ATTN_HG_SPLIT_N
fi

exec /usr/local/bin/ds4-server \
    --cuda \
    -m "${base}" \
    --no-mtp \
    -c "${DS4_CTX}" \
    --mem-floor-gb "${DS4_MEM_FLOOR_GB:-8}" \
    --no-update-check \
    --host "${DS4_HOST}" \
    --port "${DS4_PORT}" \
    --kv-disk-dir "${DS4_KV_DISK_DIR}" \
    --kv-disk-space-mb "${DS4_KV_DISK_SPACE_MB}" \
    "$@"
