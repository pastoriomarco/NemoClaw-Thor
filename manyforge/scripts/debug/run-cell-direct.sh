#!/usr/bin/env bash
# Direct-lane smoke cell for gemma-4-12b-it.
#   composer(:9000 -> nemoclaw :8100) + direct bridge(:8100 -> model :8000)
# Env:
#   FILTER   optional regex passed to smoke_corpus_runner --filter (validation)
#   REPORT   report json path (default /tmp/gemma-smoke/direct-report.json)
set -uo pipefail
MF=/home/tndlux/workspaces/dev_ws/src/manyforge
NT=/home/tndlux/workspaces/dev_ws/src/NemoClaw-Thor
OUT="${SMOKE_CELL_OUT:-/tmp/gemma-smoke}"
RUNNER="$NT/manyforge/scripts/debug/smoke_corpus_runner.py"
REPORT="${REPORT:-$OUT/direct-report.json}"
FILTER="${FILTER:-}"
ENDPOINT="http://127.0.0.1:8100/v1/manyforge/assistant"
mkdir -p "$OUT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }

cleanup_bridge(){ pkill -f "manyforge_assistant_bridge.bridge" 2>/dev/null || true; }

# ---- 1. composer (nemoclaw/direct) -----------------------------------------
log "bringing up composer (direct/nemoclaw -> :8100)"
docker rm -f manyforge-e2e-composer >/dev/null 2>&1 || true
docker run -d --rm --name manyforge-e2e-composer --network host \
  -v "$MF:/workspace" -v manyforge_build-cache:/tmp/manyforge-build -w /workspace \
  -e MANYFORGE_ASSISTANT_PROVIDER=nemoclaw \
  -e MANYFORGE_ASSISTANT_ENDPOINT_URL="$ENDPOINT" \
  -e MANYFORGE_ASSISTANT_TIMEOUT_S=180 \
  manyforge-dev:latest \
  bash -lc "python -m manyforge_composer \
    --catalog-path /workspace/manyforge_behavior/resources/node_catalog.yaml \
    --host 0.0.0.0 --port 9000 --hmi-port 8081 --mcp-http \
    --assistant-provider nemoclaw --assistant-endpoint $ENDPOINT --assistant-timeout-s 180" \
  >/dev/null 2>&1 || { log "composer docker run FAILED"; exit 11; }

log "waiting for composer /api/infra/status"
ok=0
for i in $(seq 1 60); do
  if curl -fsS --max-time 3 http://127.0.0.1:9000/api/infra/status >/dev/null 2>&1; then ok=1; log "composer up after ${i}s"; break; fi
  sleep 2
done
[ "$ok" = 1 ] || { log "composer NEVER came up"; docker logs --tail 40 manyforge-e2e-composer 2>&1 | tail -40; exit 12; }

# ---- 2. direct bridge (child) ----------------------------------------------
cleanup_bridge; sleep 1
log "starting direct bridge -> :8100 (model gemma-4-12b-it @ :8000)"
cd "$MF"
BRIDGE_HOST=127.0.0.1 BRIDGE_PORT=8100 \
BRIDGE_UPSTREAM_BASE_URL=http://127.0.0.1:8000/v1 \
BRIDGE_MANYFORGE_BASE_URL=http://127.0.0.1:9000 \
BRIDGE_MODEL="${SMOKE_MODEL_ID:-gemma-4-12b-it}" \
BRIDGE_REQUEST_TIMEOUT_S=170 BRIDGE_MAX_WALL_TIME_S=170 BRIDGE_MAX_TURNS=16 \
BRIDGE_AUDIT_LOG="$OUT/bridge-direct-audit.jsonl" \
MANYFORGE_STATE_CONTEXT=first_then_every_n MANYFORGE_STATE_CONTEXT_EVERY_N=3 \
"$MF/manyforge_assistant_bridge/.venv/bin/python" -m manyforge_assistant_bridge.bridge \
  >"$OUT/bridge-direct.log" 2>&1 &
BPID=$!
ok=0
for i in $(seq 1 20); do
  if curl -fsS --max-time 3 http://127.0.0.1:8100/healthz >/dev/null 2>&1; then ok=1; log "bridge up after ${i}s"; break; fi
  sleep 1
done
[ "$ok" = 1 ] || { log "bridge NEVER came up"; tail -30 "$OUT/bridge-direct.log"; cleanup_bridge; exit 13; }
kill -0 "$BPID" 2>/dev/null && log "bridge pid=$BPID alive" || { log "bridge died"; exit 14; }

# ---- 3. run corpus ----------------------------------------------------------
ARGS=(--report "$REPORT")
[ -n "$FILTER" ] && ARGS+=(--filter "$FILTER")
log "running smoke corpus: python3 $RUNNER ${ARGS[*]}"
python3 -u "$RUNNER" "${ARGS[@]}" >"$OUT/direct-stdout.txt" 2>&1
RC=$?
log "smoke runner exit=$RC"
echo "----- tail of runner stdout -----"
tail -40 "$OUT/direct-stdout.txt"

# ---- 4. teardown bridge -----------------------------------------------------
cleanup_bridge
log "DONE direct cell (report: $REPORT)"
