#!/usr/bin/env bash
# Run the whole capture analysis on Thor in the vLLM 0.33 image (CPU only, no network).
#   ~/fn-capture/scripts/run_all.sh [DUMPS_DIR[,DUMPS_DIR...]] [OUT_DIR]
# Defaults: ~/fn-capture/dumps,~/fn-capture/dumps-low -> ~/fn-capture/analysis (REPORT.md, summary.json, CSVs, .npy).
# Steps: PLE id verification, analysis (+ lookup-stream spool), learning curve, Edge-LLM cache replay;
# the last two are appended to REPORT.md. The spool (OUT/work) is deleted unless KEEP_SPOOL=1.
# Env: CPUS (docker --cpus, default 4), MEMORY (docker --memory, default 8g),
#      ANALYZE_ARGS / LC_ARGS / SIM_ARGS (extra flags for analyze.py / learning_curve.py / ple_cache_sim.py).
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
OUT="$(realpath -m "${2:-$HOME/fn-capture/analysis}")"
IMAGE="vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e"
HF="$HOME/thor-hf-cache"
MODEL=/hf/linked/fn-fp4-local
FP8_MODEL=/hf/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47
ASSET="$HOME/fn-assets/draft_vocab_code_47149.npy"
CORPUS_DIR="$HOME/fn-capture/analysis/corpus"   # coding corpus (build_coding_corpus.py output)
mkdir -p "$OUT"

CORPUS_ARGS=()
CORPUS_MOUNT=()
if [ -f "$CORPUS_DIR/coding-source-768.messages.jsonl" ]; then
  CORPUS_MOUNT=(-v "$CORPUS_DIR:/corpus:ro")
  CORPUS_ARGS=(--corpus /corpus/coding-source-768.messages.jsonl)
else
  echo "warning: no coding corpus in $CORPUS_DIR; corpus-based candidates are skipped" >&2
fi
FP8_ARGS=()
[ -f "$HF/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47/model.safetensors.index.json" ] \
  && FP8_ARGS=(--fp8-model "$FP8_MODEL")

run() {
  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
    --cpus "${CPUS:-4}" --memory "${MEMORY:-8g}" \
    -v "$S:/scripts:ro" "${DUMP_MOUNTS[@]}" -v "$HF:/hf:ro" -v "$ASSET:/assets/draft_vocab.npy:ro" \
    "${CORPUS_MOUNT[@]}" -v "$OUT:/out" \
    --entrypoint nice "$IMAGE" -n 10 python3 -I "$@"
}

echo "== PLE id verification"
run /scripts/verify_ple.py "$MODEL" /out/ple_verification.json 2>&1 | grep -v '^W1'
echo "== analysis of ${DUMP_ARGS[*]}"
run /scripts/analyze.py --dumps "${DUMP_ARGS[@]}" --model "$MODEL" --out /out \
  --draft-vocab /assets/draft_vocab.npy --draft-builder /scripts/build_draft_vocab.py \
  "${CORPUS_ARGS[@]}" --spool /out/work/spool ${ANALYZE_ARGS:-} 2>&1 | grep -v '^W1'
echo "== learning curve"
run /scripts/learning_curve.py --spool /out/work/spool --model "$MODEL" --out /out ${LC_ARGS:-} 2>&1 \
  | grep -v '^W1' | grep -E 'VERDICT|done in|k='
echo "== Edge-LLM cached-residency replay"
run /scripts/ple_cache_sim.py --spool /out/work/spool --model "$MODEL" "${FP8_ARGS[@]}" --out /out ${SIM_ARGS:-} 2>&1 \
  | grep -v '^W1' | grep -E '^variant|done in|configs'
cat "$OUT/learning_curve.md" "$OUT/ple_cache_sim.md" >> "$OUT/REPORT.md"
[ "${KEEP_SPOOL:-0}" = 1 ] || rm -rf "$OUT/work"
echo "== report: $OUT/REPORT.md"
du -sh "$OUT"
