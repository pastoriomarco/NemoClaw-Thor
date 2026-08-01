#!/usr/bin/env bash
# Download the DS4 0731 model pair into the mounted persistent model volume.
# Files are downloaded to *.part and atomically renamed only after curl exits
# successfully, so an interrupted 5 MB/s transfer is safely resumable.

set -euo pipefail

readonly BASE_REPO="${DS4_BASE_REPO:-antirez/deepseek-v4-gguf}"
readonly DSPARK_REPO="${DS4_DSPARK_REPO:-bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF}"
readonly BASE_PATH="${DS4_MODEL_DIR}/${DS4_BASE_FILE}"
readonly DSPARK_PATH="${DS4_MODEL_DIR}/${DS4_DSPARK_FILE}"
readonly BASE_BYTES=86720111488
readonly DSPARK_BYTES=6971241504
readonly FREE_HEADROOM_BYTES=$((5 * 1024 * 1024 * 1024))

usage() {
    cat <<'EOF'
Usage: ds4-download-weights

Downloads only the matched 0731 base and DSpark drafter. It never downloads
or loads the legacy MTP GGUF, which is incompatible with the 0731 base.
EOF
}

case "${1:-}" in
    "" ) ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

valid_gguf() {
    local file="$1" min_bytes="$2" magic size
    [[ -f "${file}" ]] || return 1
    magic="$(dd if="${file}" bs=4 count=1 status=none 2>/dev/null || true)"
    size="$(stat --format='%s' "${file}" 2>/dev/null || echo 0)"
    [[ "${magic}" == "GGUF" && "${size}" -ge "${min_bytes}" ]]
}

download_one() {
    local repo="$1" filename="$2" target="$3" min_bytes="$4"
    local part="${target}.part"
    local url="https://huggingface.co/${repo}/resolve/main/${filename}"

    if valid_gguf "${target}" "${min_bytes}"; then
        echo "Already present: ${target}"
        return 0
    fi

    echo "Downloading ${repo}/${filename}"
    echo "  destination: ${target}"
    curl --fail --location --continue-at - \
        --retry 8 --retry-all-errors --retry-delay 5 \
        --output "${part}" "${url}"

    valid_gguf "${part}" "${min_bytes}" || {
        echo "Downloaded file is not a complete GGUF: ${part}" >&2
        exit 1
    }
    mv "${part}" "${target}"
}

remaining_bytes() {
    local target="$1" expected_bytes="$2" min_bytes="$3"
    local part="${target}.part" have_bytes=0

    if valid_gguf "${target}" "${min_bytes}"; then
        echo 0
        return
    fi
    if [[ -f "${part}" ]]; then
        have_bytes="$(stat --format='%s' "${part}" 2>/dev/null || echo 0)"
    fi
    # A larger partial is not a valid resume point for this pinned artifact.
    if [[ "${have_bytes}" -gt "${expected_bytes}" ]]; then
        have_bytes=0
    fi
    echo $((expected_bytes - have_bytes))
}

mkdir -p "${DS4_MODEL_DIR}"
base_remaining="$(remaining_bytes "${BASE_PATH}" "${BASE_BYTES}" $((80 * 1024 * 1024 * 1024)))"
dspark_remaining="$(remaining_bytes "${DSPARK_PATH}" "${DSPARK_BYTES}" $((6 * 1024 * 1024 * 1024)))"
remaining_total=$((base_remaining + dspark_remaining))
if [[ "${remaining_total}" -gt 0 ]]; then
    required_bytes=$((remaining_total + FREE_HEADROOM_BYTES))
    available_bytes="$(df --output=avail -B1 "${DS4_MODEL_DIR}" | awk 'NR == 2 { print $1 }')"
    if [[ -z "${available_bytes}" || "${available_bytes}" -lt "${required_bytes}" ]]; then
        echo "Need ${required_bytes} bytes free to finish the model pair with 5 GiB headroom; found ${available_bytes:-0} bytes." >&2
        exit 1
    fi
fi

# The files are 86.72 GB and 6.97 GB respectively (93.69 GB total). The bounds
# catch HTML/error pages without hard-coding a transport-specific byte count.
download_one "${BASE_REPO}" "${DS4_BASE_FILE}" "${BASE_PATH}" $((80 * 1024 * 1024 * 1024))
download_one "${DSPARK_REPO}" "${DS4_DSPARK_FILE}" "${DSPARK_PATH}" $((6 * 1024 * 1024 * 1024))

echo "DS4 0731 model pair is ready in ${DS4_MODEL_DIR}."
