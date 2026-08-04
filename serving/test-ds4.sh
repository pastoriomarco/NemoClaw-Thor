#!/usr/bin/env bash
# Small deterministic API/throughput probe for the Dockerized Entrpi DS4 server.

set -euo pipefail

BASE_URL="${DS4_BASE_URL:-http://127.0.0.1:8050}"
MODEL="${DS4_TEST_MODEL:-deepseek-v4-flash}"
# Use no-thinking mode for the repeated output probe. This avoids consuming the
# completion budget on visible reasoning and makes decoder measurements stable.
REASONING_EFFORT="${DS4_TEST_REASONING_EFFORT:-none}"
THROUGHPUT_RUNS="${DS4_TEST_RUNS:-3}"
QUALITY_FAILURES=0
DECODE_RATES=()

die() { printf '[ds4-test] ERROR: %s\n' "$*" >&2; exit 1; }

request() {
    local prompt="$1"
    local max_tokens="$2"
    local reasoning_effort="${3-${REASONING_EFFORT}}"
    local payload

    payload="$(jq -cn \
        --arg model "$MODEL" \
        --arg prompt "$prompt" \
        --arg reasoning_effort "$reasoning_effort" \
        --argjson max_tokens "$max_tokens" \
        '{model: $model, messages: [{role: "user", content: $prompt}],
          temperature: 0, max_tokens: $max_tokens}
         + (if $reasoning_effort == "" then {} else {reasoning_effort: $reasoning_effort} end)')"
    curl --fail --silent --show-error --connect-timeout 5 --max-time 600 \
        -H 'Content-Type: application/json' \
        -d "$payload" \
        "${BASE_URL}/v1/chat/completions"
}

report_timing() {
    local label="$1"
    local response="$2"
    local output_tokens prefill_tokens ttft prefill_rate decode_rate

    output_tokens="$(jq -r '.usage.completion_tokens // 0' <<<"$response")"
    prefill_tokens="$(jq -r '.usage.prompt_tokens // 0' <<<"$response")"
    ttft="$(jq -r '.timings.ttft_ms // 0' <<<"$response")"
    prefill_rate="$(jq -r '.timings.prefill_tok_s // 0' <<<"$response")"
    decode_rate="$(jq -r '.timings.decode_tok_s // 0' <<<"$response")"
    DECODE_RATES+=("$decode_rate")
    printf '[%s] output=%s tokens; TTFT=%sms; prefill=%s tok/s; decode=%s tok/s\n' \
        "$label" "$output_tokens" "$ttft" "$prefill_rate" "$decode_rate"
}

quality_case() {
    local label="$1"
    local prompt="$2"
    local response content passed=0 reasoning_effort=""

    # Default reasoning is deliberately retained for the logic check. The
    # server's no-thinking mode is used for JSON because it follows the
    # structured-output contract without emitting a visible explanation.
    [[ "$label" == "json" ]] && reasoning_effort="none"
    response="$(request "$prompt" 96 "$reasoning_effort")"
    content="$(jq -r '.choices[0].message.content // ""' <<<"$response")"
    case "$label" in
        arithmetic)
            if grep -Eq '(^|[[:space:]])410[[:space:]]*$' <<<"$content"; then passed=1; fi
            ;;
        json)
            if jq -e '(.oldest == "Bo") and (.average_age == 9) and (.tied_youngest == ["Ana", "Cy"])' \
                <<<"$content" >/dev/null; then passed=1; fi
            ;;
        logic)
            # Accept the requested letter with conventional multiple-choice
            # punctuation; both "C" and "C)" express the same selected option.
            if grep -Eq '^C[.)]?$' <<<"$(xargs <<<"$content")"; then passed=1; fi
            ;;
        *)
            die "unknown quality check: ${label}"
            ;;
    esac
    if (( passed )); then
        printf '[quality:%s effort=%s] PASS — %s\n' "$label" "${reasoning_effort:-default}" "$content"
    else
        printf '[quality:%s effort=%s] FAIL — %s\n' "$label" "${reasoning_effort:-default}" "$content" >&2
        QUALITY_FAILURES=$((QUALITY_FAILURES + 1))
    fi
}

curl --fail --silent --show-error --connect-timeout 5 "${BASE_URL}/v1/models" \
    >/dev/null || die "DS4 is not reachable at ${BASE_URL}."
[[ "$THROUGHPUT_RUNS" =~ ^[0-9]+$ ]] || die "DS4_TEST_RUNS must be a non-negative integer."

printf 'DS4 API probe: %s (model=%s, reasoning_effort=%s)\n' \
    "$BASE_URL" "$MODEL" "${REASONING_EFFORT:-default}"
printf '\nThroughput (three deterministic long responses):\n'
throughput_prompt='Return only a JSON array containing every integer from 1 through 100 in ascending order.'
for run in $(seq 1 "$THROUGHPUT_RUNS"); do
    # Tokenization of the compact number list is implementation-dependent;
    # leave room for the closing bracket on every run.
    response="$(request "$throughput_prompt" 512)"
    jq -e '(.choices[0].message.content | fromjson) == [range(1; 101)]' \
        <<<"$response" >/dev/null || die "throughput run ${run} did not return the required JSON sequence."
    report_timing "throughput:${run}" "$response"
done

if (( THROUGHPUT_RUNS > 0 )); then
    average_decode="$(printf '%s\n' "${DECODE_RATES[@]}" | awk '{total += $1} END {printf "%.2f", total / NR}')"
    printf 'Average reported decoder throughput: %s tok/s\n' "$average_decode"
else
    printf 'Throughput probe skipped (DS4_TEST_RUNS=0).\n'
fi

printf '\nDeterministic quality checks:\n'
quality_case \
    arithmetic \
    'Answer only the final integer. A library has 18 shelves with 24 books each. It adds 37 books, then gives away 59. How many books remain?'
quality_case \
    json \
    'Return valid JSON only. Given people Ana=8, Bo=11, Cy=8, return exactly these fields: oldest (name), average_age (number), tied_youngest (names alphabetically).'
quality_case \
    logic \
    'Answer only one letter. Every flerm is a norp. No norp is blue. Miko is a flerm. Which conclusion follows? A) Miko is blue. B) It is unknown whether Miko is blue. C) Miko is not blue. D) Miko is not a flerm.'

if (( QUALITY_FAILURES > 0 )); then
    die "${QUALITY_FAILURES} deterministic quality check(s) failed."
fi

printf '\nAll DS4 API checks passed. This is a small sanity probe, not a benchmark or comprehensive evaluation.\n'
