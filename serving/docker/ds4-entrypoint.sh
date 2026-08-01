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

# These are the upstream v0.5.1 DSpark launch settings.  --no-mtp is
# deliberate: the legacy MTP GGUF must never be loaded with a 0731 base.
export DS4_CONT_MTP_MODE="${DS4_CONT_MTP_MODE:-2}"
export DS4_CONT_DSPARK="${DS4_CONT_DSPARK:-1}"
export DS4_DSPARK_MODEL="${DS4_DSPARK_MODEL:-${drafter}}"

exec /usr/local/bin/ds4-server \
    --cuda \
    -m "${base}" \
    --no-mtp \
    -c "${DS4_CTX}" \
    --host "${DS4_HOST}" \
    --port "${DS4_PORT}" \
    --kv-disk-dir "${DS4_KV_DISK_DIR}" \
    --kv-disk-space-mb "${DS4_KV_DISK_SPACE_MB}" \
    "$@"
