#!/usr/bin/env bash
# Restart DS4 from a clean unified-memory state and run the fixed HTTP matrix.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LABEL="${1:?usage: run-ds4-candidate.sh LABEL}"
MIN_AVAILABLE_GIB="${DS4_BENCH_MIN_AVAILABLE_GIB:-100}"
HEALTH_ATTEMPTS="${DS4_BENCH_HEALTH_ATTEMPTS:-18}"

export DS4_BIND_ADDRESS="${DS4_BIND_ADDRESS:-127.0.0.1}"
export DS4_CTX="${DS4_CTX:-524288}"
export DS4_SERVER_COALESCE_MAX="${DS4_SERVER_COALESCE_MAX:-2}"
export DS4_SERVER_COALESCE_MAX_TOKENS="${DS4_SERVER_COALESCE_MAX_TOKENS:-4096}"
export DS4_CONT_PREFILL_CHUNK="${DS4_CONT_PREFILL_CHUNK:-4096}"

"${SERVING_DIR}/start-ds4.sh" stop
sync
sudo sysctl -w vm.drop_caches=3

available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
required_kib="$((MIN_AVAILABLE_GIB * 1024 * 1024))"
if (( available_kib < required_kib )); then
    printf '[ds4-bench] ERROR: only %.1f GiB available; require at least %s GiB.\n' \
        "$(awk -v kib="${available_kib}" 'BEGIN {print kib / 1024 / 1024}')" \
        "${MIN_AVAILABLE_GIB}" >&2
    exit 1
fi

printf '[ds4-bench] clean start: label=%s ctx=%s banks=%s group_tokens=%s chunk=%s available=%.1f GiB\n' \
    "${LABEL}" "${DS4_CTX}" "${DS4_SERVER_COALESCE_MAX}" \
    "${DS4_SERVER_COALESCE_MAX_TOKENS}" "${DS4_CONT_PREFILL_CHUNK}" \
    "$(awk -v kib="${available_kib}" 'BEGIN {print kib / 1024 / 1024}')"

"${SERVING_DIR}/start-ds4.sh" start
for attempt in $(seq 1 "${HEALTH_ATTEMPTS}"); do
    health="$(docker inspect --format '{{.State.Health.Status}}' nemoclaw-ds4-ds4-1 2>/dev/null || true)"
    if [[ "${health}" == "healthy" ]]; then
        break
    fi
    if (( attempt == HEALTH_ATTEMPTS )); then
        docker logs --tail 100 nemoclaw-ds4-ds4-1 >&2 || true
        printf '[ds4-bench] ERROR: DS4 did not become healthy.\n' >&2
        exit 1
    fi
    sleep 10
done

if docker logs nemoclaw-ds4-ds4-1 2>&1 | grep -q 'cont admit rejected on memory floor'; then
    printf '[ds4-bench] ERROR: continuous admission already hit the memory floor.\n' >&2
    exit 1
fi

args=(
    --label "${LABEL}"
    --base-url "http://127.0.0.1:8050/v1"
    --repeats "${DS4_BENCH_REPEATS:-3}"
    --timeout "${DS4_BENCH_TIMEOUT:-1800}"
)
if [[ -n "${DS4_BENCH_OUTPUT_JSON:-}" ]]; then
    args+=(--output-json "${DS4_BENCH_OUTPUT_JSON}")
fi
if [[ -n "${DS4_BENCH_CASES:-}" ]]; then
    read -r -a cases <<<"${DS4_BENCH_CASES}"
    for bench_case in "${cases[@]}"; do
        args+=(--case "${bench_case}")
    done
fi

"${SCRIPT_DIR}/bench-ds4-http.py" "${args[@]}"

if docker logs nemoclaw-ds4-ds4-1 2>&1 | grep -q 'cont admit rejected on memory floor'; then
    printf '[ds4-bench] ERROR: continuous serving fell back on the memory floor.\n' >&2
    exit 1
fi
if docker logs nemoclaw-ds4-ds4-1 2>&1 | grep -Eq \
    'Xid|illegal memory access|cudaMallocAsync.*failed|CUDA error|graph.*failed'; then
    printf '[ds4-bench] ERROR: CUDA/kernel failure found in server log.\n' >&2
    exit 1
fi
