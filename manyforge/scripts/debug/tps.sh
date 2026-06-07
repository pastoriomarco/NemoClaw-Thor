#!/bin/bash
# tps.sh — extract decode/generation throughput for a smoke run from the model
# container's logs. Handles BOTH serving lanes:
#   - llama.cpp : "eval time ... tokens per second" (decode) + "prompt eval time" (prefill)
#   - vLLM      : "Avg generation throughput: N tokens/s" + "Avg prompt throughput: N"
#
# Usage:
#   tps.sh <container-name> <run-start-file>
#     <container-name>  model container, e.g. manyforge-e2e-vllm
#     <run-start-file>  a file whose contents are the run's start unix timestamp
#                       (the smoke launch writes `date +%s` to one); passed to
#                       `docker logs --since` so only this run's lines are counted.
#
# Prints n / min / median / max tokens-per-second per metric. NOTE: vLLM's
# aggregate "generation throughput" is dragged down by idle windows in
# single-stream smoke runs — treat wall-clock per case as the truer cross-lane
# speed signal. Companion to summarize.py (scoring breakdown from the report JSON).
C="$1"; START=$(cat "$2" 2>/dev/null || echo 0)
echo "container=$C  since(unix)=$START"
LOG=$(docker logs --since "$START" "$C" 2>&1)
# llama.cpp decode (exclude the prompt-eval/prefill lines)
echo "$LOG" | grep -E "eval time" | grep -v "prompt eval" \
  | grep -oE "[0-9.]+ tokens per second" | awk '{print $1}' | sort -n \
  | awk 'NR{a[NR]=$1} END{if(NR)printf "  [llama.cpp] decode t/s : n=%d  min=%.1f  median=%.1f  max=%.1f\n",NR,a[1],a[int((NR+1)/2)],a[NR]}'
echo "$LOG" | grep -E "prompt eval time" \
  | grep -oE "[0-9.]+ tokens per second" | awk '{print $1}' | sort -n \
  | awk 'NR{a[NR]=$1} END{if(NR)printf "  [llama.cpp] prefill t/s: n=%d  min=%.1f  median=%.1f  max=%.1f\n",NR,a[1],a[int((NR+1)/2)],a[NR]}'
# vLLM throughput
echo "$LOG" | grep -oE "generation throughput: [0-9.]+" | awk '{print $3}' | sort -n \
  | awk 'NR{a[NR]=$1} END{if(NR)printf "  [vllm] gen t/s     : n=%d  min=%.1f  median=%.1f  max=%.1f\n",NR,a[1],a[int((NR+1)/2)],a[NR]}'
echo "$LOG" | grep -oE "prompt throughput: [0-9.]+" | awk '{print $3}' | sort -n \
  | awk 'NR{a[NR]=$1} END{if(NR)printf "  [vllm] prompt t/s  : n=%d  min=%.1f  median=%.1f  max=%.1f\n",NR,a[1],a[int((NR+1)/2)],a[NR]}'
