#!/usr/bin/env bash
# Quick learning curve of the static PLE hot set on what is captured so far (minutes, CPU only):
#   ~/fn-capture/scripts/learning_curve.sh [DUMPS_DIR[,DUMPS_DIR...]] [OUT_DIR]
# Defaults: ~/fn-capture/dumps,~/fn-capture/dumps-low -> ~/fn-capture/analysis/learning-curve-<time>/ (learning_curve.md/.json).
# Tokenises the dumps (no counters, no vocabulary), writes a temporary spool, prints the table and verdicts.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
# Dump folders: comma-separated, each its own request stream (default: dumps = effort xhigh, dumps-low = effort low).
DUMP_LIST="${1:-$HOME/fn-capture/dumps,$HOME/fn-capture/dumps-low}"
DUMP_MOUNTS=()
DUMP_ARGS=()
IFS=',' read -ra _dirs <<< "$DUMP_LIST"
for d in "${_dirs[@]}"; do
  [ -d "$d" ] || { echo "skip missing $d" >&2; continue; }
  d="$(realpath "$d")"
  DUMP_MOUNTS+=(-v "$d:/dumps/$(basename "$d"):ro")
  DUMP_ARGS+=("/dumps/$(basename "$d")")
done
OUT="$(realpath -m "${2:-$HOME/fn-capture/analysis/learning-curve-$(date +%Y%m%d-%H%M)}")"
IMAGE="vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e"
MODEL=/hf/linked/fn-fp4-local
mkdir -p "$OUT"
run() {
  docker run --rm --runtime runc -e NVIDIA_VISIBLE_DEVICES=void --network none --user "$(id -u):$(id -g)" -e HOME=/tmp --cpus "${CPUS:-4}" --memory "${MEMORY:-8g}" \
    -v "$S:/scripts:ro" "${DUMP_MOUNTS[@]}" -v "$HOME/thor-hf-cache:/hf:ro" -v "$OUT:/out" \
    --entrypoint nice "$IMAGE" -n 10 python3 -I "$@"
}
run /scripts/analyze.py --dumps "${DUMP_ARGS[@]}" --model "$MODEL" --out /out --draft-vocab /dev/null --draft-builder /dev/null \
  --spool /out/spool --spool-only 2>&1 | grep -v '^W1' | grep -vE '^  [0-9]+/'
run /scripts/learning_curve.py --spool /out/spool --model "$MODEL" --out /out ${LC_ARGS:-} 2>&1 | grep -v '^W1' | grep -v ' k='
rm -rf "$OUT/spool"
echo "== $OUT/learning_curve.md"
