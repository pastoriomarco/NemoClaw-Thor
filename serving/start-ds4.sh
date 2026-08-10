#!/usr/bin/env bash
# Build, download, and operate Entrpi DS4 as a Dockerized Thor service.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/ds4-compose.yml"
COMPOSE=(docker compose --project-name nemoclaw-ds4 -f "${COMPOSE_FILE}")

export DS4_IMAGE="${DS4_IMAGE:-nemoclaw-thor/ds4:v0.5.6.2-sm110-thor}"
export DS4_BUILD_TARGET="${DS4_BUILD_TARGET:-runtime-thor-v056}"
export DS4_TAG="${DS4_TAG:-v0.5.6.2}"
export DS4_REF="${DS4_REF:-027714a4c290a756ef3e6ca557426528745f2033}"
export DS4_MODEL_DIR="${DS4_MODEL_DIR:-${HOME}/thor-hf-cache/ds4}"
export DS4_HOST_PORT="${DS4_HOST_PORT:-8050}"
export DS4_BIND_ADDRESS="${DS4_BIND_ADDRESS:-127.0.0.1}"
export DS4_UID="${DS4_UID:-$(id -u)}"
export DS4_GID="${DS4_GID:-$(id -g)}"

info() { printf '[ds4] %s\n' "$*"; }
die() { printf '[ds4] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: ./serving/start-ds4.sh [start|build|download|status|logs|smoke|test|stop]

  start     Build the image if needed, resume/download the model pair, then
            start DS4 on 127.0.0.1:8050 (default).
  build     Build the pinned Thor-tuned DS4 v0.5.6.2 image for sm_110 only.
  download  Start or resume the persistent 0731 base + DSpark downloads.
  status    Show Compose service state.
  logs      Follow DS4 server logs.
  smoke     Query /v1/models and make one OpenAI chat request.
  test      Run three output-throughput probes and three deterministic quality checks.
  stop      Stop only the DS4 Compose project; model and KV files remain.

Useful overrides: DS4_MODEL_DIR, DS4_HOST_PORT, DS4_BIND_ADDRESS, DS4_CTX,
DS4_BATCH_VMM_BUDGET_MB, DS4_MEM_FLOOR_GB, DS4_SERVER_COALESCE_MAX,
DS4_SERVER_COALESCE_MAX_TOKENS, DS4_CONT_PREFILL_CHUNK. The upstream rollback
uses DS4_BUILD_TARGET=runtime plus a distinct DS4_IMAGE tag. Diagnostics also
use DS4_CUDA_NO_TOPK_STREAM, DS4_CUDA_TOPK_STREAM, and DS4_CUDA_TOPK_STREAM_VERIFY.
Attention tuning candidates additionally use DS4_CUDA_ATTN_HG_SPLIT_N.
EOF
}

client_base_url() {
    local client_host="${DS4_BIND_ADDRESS}"
    # Wildcard bind addresses are valid listeners but not useful HTTP targets.
    [[ "${client_host}" == "0.0.0.0" || "${client_host}" == "::" ]] && client_host="127.0.0.1"
    printf 'http://%s:%s' "${client_host}" "${DS4_HOST_PORT}"
}

check_host() {
    [[ "$(uname -m)" == "aarch64" ]] || die "DS4 container is scoped to aarch64 Thor."
    command -v docker >/dev/null || die "docker is not installed."
    docker info >/dev/null 2>&1 || die "Docker daemon is unavailable."
    docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is required."
    mkdir -p "${DS4_MODEL_DIR}/kv-cache"
}

check_port() {
    if ss -ltnH "sport = :${DS4_HOST_PORT}" 2>/dev/null | grep -q .; then
        die "${DS4_BIND_ADDRESS}:${DS4_HOST_PORT} is already listening; choose DS4_HOST_PORT=<free port>."
    fi
}

build() {
    info "Building ${DS4_IMAGE} target=${DS4_BUILD_TARGET} (Entrpi/ds4 ${DS4_TAG}@${DS4_REF}, CUDA_ARCH=sm_110)."
    "${COMPOSE[@]}" build ds4
}

download() {
    info "Starting/resuming the 0731 base + matching DSpark drafter download."
    "${COMPOSE[@]}" up -d ds4-download
    info "Follow progress: ${0} logs"
}

start() {
    check_port
    build
    info "Starting downloader and DS4. At 5 MB/s the initial 93.69 GB (87.26 GiB) download takes about 5.21 hours; DS4 waits and starts automatically after it completes."
    "${COMPOSE[@]}" up -d ds4-download ds4
    "${COMPOSE[@]}" ps
    info "Monitor download/server state with: ${0} logs"
}

smoke() {
    local base_url="${DS4_BASE_URL:-$(client_base_url)}"
    curl --fail --silent --show-error "${base_url}/v1/models"
    echo
    curl --fail --silent --show-error \
        -H 'Content-Type: application/json' \
        -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":32,"temperature":0,"reasoning_effort":"off"}' \
        "${base_url}/v1/chat/completions"
    echo
}

command="${1:-start}"
case "${command}" in
    start) check_host; start ;;
    build) check_host; build ;;
    download) check_host; build; download ;;
    status) check_host; "${COMPOSE[@]}" ps ;;
    logs) check_host; "${COMPOSE[@]}" logs --follow --tail=100 ds4-download ds4 ;;
    smoke) smoke ;;
    test) check_host; DS4_BASE_URL="${DS4_BASE_URL:-$(client_base_url)}" "${SCRIPT_DIR}/test-ds4.sh" ;;
    stop) check_host; "${COMPOSE[@]}" stop ds4 ds4-download ;;
    -h|--help|help) usage ;;
    *) die "Unknown command: ${command}. Run ${0} --help." ;;
esac
