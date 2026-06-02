#!/usr/bin/env bash
# run-smoke-all-models.sh
#
# Sequence the full smoke corpus across:
#   - cosmos-reason2-8b
#   - nemotron3-nano-omni-30b-a3b-nvfp4  (omni)
#   - nemotron3-nano-4b-bf16  with thinking ON
#   - nemotron3-nano-4b-bf16  with thinking OFF
#   - qwen3.6-35b-a3b-nvfp4-nvidia  (35B)
#
# Per model:
#   1. Stop vLLM + proxy
#   2. Start vLLM + proxy with the target profile and FORCE_THINKING setting
#   3. Wait for vLLM ready
#   4. Restart bridge (clean session state)
#   5. Run full smoke corpus → /tmp/full-smoke-post-rebuild/<tag>-*.json
#
# Assumes:
#   - sandbox 'my-assistant' is already onboarded (this run reuses it)
#   - composer container manyforge-e2e-composer is up
#   - The MCP 4xx→200 patch is loaded in composer (restart composer first
#     if you just pulled new code)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SMOKE_ROOT="/tmp/full-smoke-post-rebuild"
SMOKE_RUNNER="${REPO_ROOT}/manyforge/scripts/debug/smoke_corpus_runner.py"
BRIDGE="${REPO_ROOT}/manyforge/openclaw_assistant_bridge/.venv/bin/python"

mkdir -p "${SMOKE_ROOT}"

# (profile, vllm_port_override, force_thinking, tag)
RUNS=(
  "cosmos-reason2-8b|8050|on|cosmos"
  "nemotron3-nano-omni-30b-a3b-nvfp4|8050|off|omni"
  "nemotron3-nano-4b-bf16|8050|on|4b-thinking-on"
  "nemotron3-nano-4b-bf16|8050|off|4b-thinking-off"
  "qwen3.6-35b-a3b-nvfp4-nvidia|8050|on|35b"
)

restart_bridge() {
  pkill -f "openclaw_assistant_bridge.service" 2>/dev/null || true
  sleep 2
  cd "${REPO_ROOT}"
  PYTHONPATH="${REPO_ROOT}/manyforge" \
  OPENCLAW_ASSISTANT_AGENT=manyforge-composer \
  nohup "${BRIDGE}" -m openclaw_assistant_bridge.service > /tmp/bridge.log 2>&1 &
  sleep 4
  pgrep -f "openclaw_assistant_bridge.service" >/dev/null || {
    echo "  ! bridge failed to start; see /tmp/bridge.log" >&2
    return 1
  }
}

wait_for_vllm() {
  local timeout=${1:-300}
  local elapsed=0
  while [[ $elapsed -lt $timeout ]]; do
    if curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  return 1
}

for run in "${RUNS[@]}"; do
  IFS='|' read -r PROFILE VLLM_PORT FORCE_THINKING TAG <<< "$run"
  STDOUT="${SMOKE_ROOT}/${TAG}-stdout.txt"
  REPORT="${SMOKE_ROOT}/${TAG}-report.json"

  echo
  echo "═══════════════════════════════════════════════════════════════════════"
  echo " ${TAG}: profile=${PROFILE}  thinking=${FORCE_THINKING}"
  echo "═══════════════════════════════════════════════════════════════════════"

  # 1. Stop current vLLM + proxy
  echo "  → stopping vLLM + proxy"
  pkill -f "scripts/proxy/vllm-proxy.py" 2>/dev/null || true
  for c in $(docker ps --filter "ancestor=nemoclaw-thor/vllm:latest" --format "{{.Names}}"); do
    docker stop "$c" >/dev/null 2>&1 || true
  done
  sleep 2
  # Free unified memory (per memory: vLLM memory leak on Thor)
  sync
  echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

  # 2. Start vLLM + proxy with target profile and thinking setting
  echo "  → starting vLLM + proxy"
  cd "${REPO_ROOT}"
  THOR_VLLM_PORT="${VLLM_PORT}" \
  THOR_TARGET_PROXY_FORCE_ENABLE_THINKING="${FORCE_THINKING}" \
    nohup ./serving/start-model.sh "${PROFILE}" > "/tmp/start-${TAG}.log" 2>&1 &

  # 3. Wait for vLLM ready
  if ! wait_for_vllm 360; then
    echo "  ! vLLM did not come up in 6 minutes; skipping ${TAG}" >&2
    continue
  fi
  echo "  → vLLM up"

  # 4. Restart bridge
  if ! restart_bridge; then
    echo "  ! bridge restart failed; skipping ${TAG}" >&2
    continue
  fi
  echo "  → bridge up"

  # 5. Run the smoke
  echo "  → running smoke corpus (report: ${REPORT})"
  python3 -u "${SMOKE_RUNNER}" --report "${REPORT}" > "${STDOUT}" 2>&1 || true

  # Summary
  echo
  echo "  results for ${TAG}:"
  grep -E "total cases|effective|first-try" "${STDOUT}" | sed 's/^/    /' | tail -5
done

echo
echo "═══════════════════════════════════════════════════════════════════════"
echo "All runs complete. Reports in ${SMOKE_ROOT}/"
echo "═══════════════════════════════════════════════════════════════════════"
ls -la "${SMOKE_ROOT}/"
