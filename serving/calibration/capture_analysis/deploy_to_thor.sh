#!/usr/bin/env bash
# Copy the scripts (and the draft-vocab builder they import) to Thor ~/fn-capture/scripts/.
# Optional: CORPUS=/path/to/coding-source-768.messages.jsonl also copies the coding corpus
# to ~/fn-capture/analysis/corpus/ (build it with ../build_coding_corpus.py, see README).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
THOR="${THOR:-tndlux@192.168.1.136}"
ssh "$THOR" 'mkdir -p ~/fn-capture/scripts ~/fn-capture/analysis/corpus'
scp -q "$HERE"/{ple_hash.py,render.py,analyze.py,verify_ple.py,learning_curve.py,ple_cache_sim.py,run_all.sh,learning_curve.sh,README.md} \
  "$HERE/../../docker/patches/qwen4exp-draft-head/build_draft_vocab.py" "$THOR:fn-capture/scripts/"
if [ -n "${CORPUS:-}" ]; then
  scp -q "$CORPUS" "$THOR:fn-capture/analysis/corpus/"
fi
echo "deployed to $THOR:~/fn-capture/scripts"
