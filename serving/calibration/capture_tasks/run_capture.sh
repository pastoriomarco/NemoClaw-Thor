#!/bin/bash
# Overnight traffic capture for draft-vocabulary and PLE-usage analysis.
# Runs on the laptop. Agent tasks go through the hardened OMP writable sandbox
# (~/omp-readonly-harness, profile qwen3.8_rw: no network except the model relay,
# writes only to its own Docker-volume copy of manyforge/manyforge_specs).
# Reasoning prompts go straight to the capture proxy. The capture itself is the
# proxy's dump directory on Thor (~/fn-capture/dumps, capped at 8 GB).
#
# usage: run_capture.sh [task_timeout_s]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
HARNESS=${HARNESS:-$HOME/omp-readonly-harness}
ENDPOINT=${ENDPOINT:-http://192.168.1.136:8050/v1/chat/completions}
TIMEOUT=${1:-900}
OUT=${OUT:-$HARNESS/exports/capture-$(date +%Y%m%d)}
mkdir -p "$OUT"
LOG=$OUT/run.log
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

reasoning_batch() { # first last (1-based line numbers)
    python3 -I - "$ENDPOINT" "$HERE/reasoning_prompts.txt" "$1" "$2" <<'EOF' 2>&1 | tee -a "$LOG"
import json, sys, time, urllib.request
url, path, first, last = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
prompts = [l.strip() for l in open(path) if l.strip()][first - 1:last]
for i, p in enumerate(prompts, first):
    body = json.dumps({"model": "qwen3.8-flash-next", "stream": True, "max_tokens": 8192,
                       "temperature": 0.6, "top_p": 0.95,
                       "messages": [{"role": "user", "content": "[family:reasoning] " + p}],
                       "chat_template_kwargs": {"enable_thinking": True}}).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            n = sum(1 for _ in r)
        print(f"{time.strftime('%H:%M:%S')} reasoning {i}: {n} lines in {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        print(f"{time.strftime('%H:%M:%S')} reasoning {i}: FAILED {e!r}", flush=True)
EOF
}

omp_task() { # id family prompt
    log "task $1 ($2) start"
    ( cd "$HARNESS" && timeout --kill-after=60 "$TIMEOUT" ./run.sh qwen3.8_rw --print --no-session "[family:$2] $3" </dev/null ) \
        2>&1 | tail -c 200000 > "$OUT/$1.log"
    log "task $1 end rc=${PIPESTATUS[0]} ($(wc -c < "$OUT/$1.log") bytes kept)"
}

log "capture start: timeout ${TIMEOUT}s per task, output $OUT"
# Interleave the four agent families, with a reasoning batch after every 7 tasks.
mapfile -t TASKS < <(python3 -I -c '
import json, sys, itertools
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
fams = {}
for r in rows: fams.setdefault(r["family"], []).append(r)
for group in itertools.zip_longest(*fams.values()):
    for r in group:
        if r: print(json.dumps(r))
' "$HERE/omp_tasks.jsonl")
NR=$(grep -c . "$HERE/reasoning_prompts.txt")
per=$(( (NR + 3) / 4 )); next=1; k=0
for row in "${TASKS[@]}"; do
    id=$(python3 -I -c 'import json,sys;print(json.loads(sys.argv[1])["id"])' "$row")
    fam=$(python3 -I -c 'import json,sys;print(json.loads(sys.argv[1])["family"])' "$row")
    prompt=$(python3 -I -c 'import json,sys;print(json.loads(sys.argv[1])["prompt"])' "$row")
    omp_task "$id" "$fam" "$prompt"
    k=$((k + 1))
    if [ $((k % 7)) -eq 0 ] && [ $next -le "$NR" ]; then
        last=$((next + per - 1)); [ $last -gt "$NR" ] && last=$NR
        log "reasoning batch $next-$last"
        reasoning_batch $next $last; next=$((last + 1))
    fi
done
[ $next -le "$NR" ] && { log "reasoning batch $next-$NR"; reasoning_batch $next "$NR"; }
log "capture done"
