#!/bin/bash
# Acceptance/throughput probe for Laguna-S-2.1-NVFP4 + DFlash on Thor.
# Coding-weighted prompt set; per-request spec-decode metric deltas from
# /metrics (drafts, drafted tokens, accepted tokens -> mean accepted length),
# wall-clock decode tok/s from usage + timing.
# Usage: ./test-laguna-dflash-accept.sh [base_url] [model]
set -u

BASE="${1:-http://localhost:8000}"
MODEL="${2:-poolside/Laguna-S-2.1-NVFP4}"
API="$BASE/v1/chat/completions"
METRICS="$BASE/metrics"

scrape() {
    curl -s "$METRICS" | python3 -c '
import sys
d = {"drafts": 0.0, "drafted": 0.0, "accepted": 0.0}
for line in sys.stdin:
    if line.startswith("#"):
        continue
    if "spec_decode_num_drafts" in line and "per_pos" not in line:
        d["drafts"] += float(line.rsplit(None, 1)[-1])
    elif "spec_decode_num_draft_tokens" in line:
        d["drafted"] += float(line.rsplit(None, 1)[-1])
    elif "spec_decode_num_accepted_tokens" in line and "per_pos" not in line:
        d["accepted"] += float(line.rsplit(None, 1)[-1])
print(d["drafts"], d["drafted"], d["accepted"])'
}

run_case() {
    local name="$1" temp="$2" max_tok="$3" prompt="$4"
    local pre post t0 t1
    pre=$(scrape)
    t0=$(date +%s.%N)
    local resp
    resp=$(curl -s "$API" -H "Content-Type: application/json" -d "$(python3 -c '
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": sys.argv[4]}],
    "max_tokens": int(sys.argv[3]),
    "temperature": float(sys.argv[2]),
    "top_p": 0.95,
}))' "$MODEL" "$temp" "$max_tok" "$prompt")")
    t1=$(date +%s.%N)
    post=$(scrape)
    python3 -c '
import json, sys
name, temp, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
pre = [float(x) for x in sys.argv[5].split()]
post = [float(x) for x in sys.argv[6].split()]
try:
    r = json.loads(sys.argv[7])
    usage = r["usage"]
    content = r["choices"][0]["message"].get("content") or ""
    msg = r["choices"][0]["message"]
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    ctok = usage["completion_tokens"]
except Exception as e:
    print(f"[{name}] REQUEST FAILED: {e}: {sys.argv[7][:300]}")
    sys.exit(0)
wall = t1 - t0
drafts, drafted, accepted = (post[0]-pre[0]), (post[1]-pre[1]), (post[2]-pre[2])
mal = 1 + accepted/drafts if drafts else float("nan")
ar = accepted/drafted if drafted else float("nan")
print(f"[{name}] temp={temp} completion={ctok} tok  wall={wall:.1f}s  ~{ctok/wall:.1f} tok/s")
print(f"    drafts={drafts:.0f} drafted={drafted:.0f} accepted={accepted:.0f}  "
      f"accept_rate={ar:.1%}  mean_accepted_len={mal:.2f} tok/step")
snippet = (reasoning[:120] + " || " if reasoning else "") + content[:200]
print(f"    out: {snippet!r}")
' "$name" "$temp" "$t0" "$t1" "$pre" "$post" "$resp"
    echo
}

echo "=== Laguna-S-2.1 DFlash acceptance probe — $(date -Is) ==="
echo "Waiting for server..."
for i in $(seq 1 300); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && { echo "Server ready."; break; }
    [ "$i" -eq 300 ] && { echo "Timeout waiting for server"; exit 1; }
    sleep 2
done
echo

run_case "warmup" 0.7 64 "Say hello in one short sentence."

# Coding-weighted set
run_case "code-impl-python" 0.7 768 "Write a Python function parse_iso8601_duration(s) that parses an ISO 8601 duration string like 'P3DT4H59M12.5S' into a datetime.timedelta. Handle weeks, fractional seconds, and invalid input (raise ValueError). Include 5 pytest test cases."

run_case "code-impl-cpp" 0.7 768 "Implement a thread-safe bounded MPMC queue in C++20 using std::mutex and std::condition_variable_any with stop_token support for shutdown. Provide push, pop, and a small usage example."

run_case "code-refactor" 0.7 640 "Refactor this Python code to be idiomatic and efficient, keeping behavior identical:

def proc(l):
    r = []
    for i in range(len(l)):
        if l[i] != None:
            if type(l[i]) == str:
                if l[i].strip() != '':
                    r.append(l[i].strip().lower())
    d = {}
    for x in r:
        if x in d:
            d[x] = d[x] + 1
        else:
            d[x] = 1
    return d"

run_case "code-explain" 0.7 512 "Explain what this bash does, line by line, and point out any bugs: for f in \$(ls *.log); do cat \$f | grep ERROR | wc -l > \${f%.log}.count; done"

# Non-coding contrast
run_case "prose" 0.7 512 "Summarize the tradeoffs between speculative decoding with a separate draft model versus draft heads attached to the target model, for a technical blog audience."

# Temperature sensitivity on identical coding prompt (community: temp=0 hurts acceptance)
run_case "code-temp0" 0.0 512 "Write a Python function lru_cache_decorator(maxsize) implementing an LRU cache decorator from scratch using collections.OrderedDict. No functools."
run_case "code-temp07" 0.7 512 "Write a Python function lru_cache_decorator(maxsize) implementing an LRU cache decorator from scratch using collections.OrderedDict. No functools."

echo "=== Server-side SpecDecoding log lines (last 10) ==="
docker logs laguna-s21-dflash-smoke 2>&1 | grep -i "SpecDecoding\|spec.*accept" | tail -10
