#!/usr/bin/env bash
# OpenClaw-lane smoke cell for gemma-4-12b-it.
#   composer(:9000 -> openclaw :8200) + openclaw bridge(:8200 -> sandbox my-assistant -> model :8000)
# Args/Env:
#   MODE     code | tools   (sets sandbox toolSearch.mode + bridge TOOL_SURFACE)
#   FILTER   optional regex for smoke_corpus_runner --filter (validation)
#   REPORT   report json path
set -uo pipefail
MODE="${MODE:-tools}"
MF=/home/tndlux/workspaces/dev_ws/src/manyforge
NT=/home/tndlux/workspaces/dev_ws/src/NemoClaw-Thor
OUT="${SMOKE_CELL_OUT:-/tmp/gemma-smoke}"
SANDBOX="${OPENCLAW_ASSISTANT_SANDBOX:-my-assistant}"
RUNNER="$NT/manyforge/scripts/debug/smoke_corpus_runner.py"
BR="$NT/manyforge/openclaw_assistant_bridge"
REPORT="${REPORT:-$OUT/openclaw-$MODE-report.json}"
FILTER="${FILTER:-}"
ENDPOINT="http://127.0.0.1:8200/v1/manyforge/assistant"
mkdir -p "$OUT"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
cleanup_bridge(){ pkill -f "openclaw_assistant_bridge.service" 2>/dev/null || true; }

# ---- 1. sandbox tool-surface config + gateway restart ----------------------
# OpenClaw 2026.5.22: the offered surface is controlled by tools.codeMode (NOT
# just toolSearch.mode). tools surface => codeMode=false + toolSearch.mode=tools
# (offers tool_search/tool_describe/tool_call). code surface => codeMode=true +
# toolSearch.mode=code (offers tool_search_code). profile=full exposes the full
# surface. The running gateway CACHES config, so it must be killed (it respawns
# and re-reads). Verified 2026-06-05: codeMode=false flips the surface and the
# agent dispatches real composer tools instead of looping on tool_search_code.
if [ "$MODE" = "code" ]; then TS_MODE=code; CODEMODE=True; else TS_MODE=tools; CODEMODE=False; fi
log "setting sandbox surface: toolSearch.mode=$TS_MODE codeMode=$CODEMODE profile=full"
nemoclaw "$SANDBOX" exec --no-tty -- python3 -c "import json,hashlib,pathlib; p=pathlib.Path('/sandbox/.openclaw/openclaw.json'); d=json.loads(p.read_text()); t=d.setdefault('tools',{}); t['toolSearch']={'enabled':True,'mode':'$TS_MODE'}; t['codeMode']=$CODEMODE; t['profile']='full'; p.write_text(json.dumps(d,indent=2)); h=hashlib.sha256(p.read_bytes()).hexdigest(); pathlib.Path('/sandbox/.openclaw/.config-hash').write_text(h+'  openclaw.json\n'); print('surface:',json.dumps(t))" 2>&1 | tail -2
log "restarting in-sandbox openclaw gateway to pick up surface config"
nemoclaw "$SANDBOX" exec --no-tty -- sh -lc 'pkill -9 -x openclaw 2>/dev/null; pkill -f manyforge-mcp-bridge.py 2>/dev/null; sleep 4; true' 2>&1 | tail -1

# ---- 2. composer (openclaw) -------------------------------------------------
log "bringing up composer (openclaw -> :8200)"
docker rm -f manyforge-e2e-composer >/dev/null 2>&1 || true
docker run -d --rm --name manyforge-e2e-composer --network host \
  -v "$MF:/workspace" -v manyforge_build-cache:/tmp/manyforge-build -w /workspace \
  -e MANYFORGE_ASSISTANT_PROVIDER=openclaw \
  -e MANYFORGE_ASSISTANT_ENDPOINT_URL="$ENDPOINT" \
  -e MANYFORGE_ASSISTANT_TIMEOUT_S=180 \
  manyforge-dev:latest \
  bash -lc "python -m manyforge_composer \
    --catalog-path /workspace/manyforge_behavior/resources/node_catalog.yaml \
    --host 0.0.0.0 --port 9000 --hmi-port 8081 --mcp-http \
    --assistant-provider openclaw --assistant-endpoint $ENDPOINT --assistant-timeout-s 180" \
  >/dev/null 2>&1 || { log "composer docker run FAILED"; exit 11; }
ok=0
for i in $(seq 1 60); do
  curl -fsS --max-time 3 http://127.0.0.1:9000/api/infra/status >/dev/null 2>&1 && { ok=1; log "composer up after ${i}s"; break; }
  sleep 2
done
[ "$ok" = 1 ] || { log "composer NEVER came up"; docker logs --tail 40 manyforge-e2e-composer 2>&1 | tail -40; exit 12; }

# ---- 3. openclaw bridge (child) -> :8200 -----------------------------------
cleanup_bridge; sleep 1
log "starting openclaw bridge -> :8200 (sandbox $SANDBOX, TOOL_SURFACE=$MODE)"
cd "$NT/manyforge"
OPENCLAW_ASSISTANT_BRIDGE_HOST=127.0.0.1 OPENCLAW_ASSISTANT_BRIDGE_PORT=8200 \
OPENCLAW_ASSISTANT_COMPOSER_BASE=http://127.0.0.1:9000 \
OPENCLAW_ASSISTANT_SANDBOX="$SANDBOX" \
OPENCLAW_ASSISTANT_AGENT=manyforge-composer \
OPENCLAW_ASSISTANT_LOCAL=false OPENCLAW_ASSISTANT_USE_GATEWAY=false \
OPENCLAW_ASSISTANT_TOOL_SURFACE="$MODE" \
OPENCLAW_ASSISTANT_COMPACT_EVERY_N=2 OPENCLAW_ASSISTANT_COMPACT_TIMEOUT_S=120 \
OPENCLAW_ASSISTANT_TIMEOUT_S=170 \
MANYFORGE_STATE_CONTEXT=first_then_every_n MANYFORGE_STATE_CONTEXT_EVERY_N=3 \
"$BR/.venv/bin/python" -m openclaw_assistant_bridge.service \
  >"$OUT/bridge-openclaw-$MODE.log" 2>&1 &
BPID=$!
ok=0
for i in $(seq 1 30); do
  if curl -fsS --max-time 3 http://127.0.0.1:8200/healthz 2>/dev/null | grep -q '"provider":"openclaw"'; then ok=1; log "openclaw bridge up after ${i}s"; break; fi
  sleep 1
done
[ "$ok" = 1 ] || { log "openclaw bridge NEVER came up"; tail -40 "$OUT/bridge-openclaw-$MODE.log"; cleanup_bridge; exit 13; }
kill -0 "$BPID" 2>/dev/null && log "bridge pid=$BPID alive" || { log "bridge died"; exit 14; }

# ---- 4. run corpus ----------------------------------------------------------
ARGS=(--report "$REPORT")
[ -n "$FILTER" ] && ARGS+=(--filter "$FILTER")
log "running smoke corpus (openclaw/$MODE): ${ARGS[*]}"
python3 -u "$RUNNER" "${ARGS[@]}" >"$OUT/openclaw-$MODE-stdout.txt" 2>&1
log "smoke runner exit=$?"
echo "----- tail of runner stdout -----"; tail -40 "$OUT/openclaw-$MODE-stdout.txt"
cleanup_bridge
log "DONE openclaw-$MODE cell (report: $REPORT)"
