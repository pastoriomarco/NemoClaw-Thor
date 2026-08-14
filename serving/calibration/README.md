# Coding calibration corpus

This directory contains the offline builder and model-specific renderer for the
Qwen coding NVFP4 calibration corpus. Generated data stays outside git by
default, under the existing Thor ModelOpt calibration directory.

The end-to-end release-day procedure, candidate recipes, validation gates, and
cleanup policy are in
[`../docs/QWEN38-27B-NVFP4-QUANTIZATION-PLAN.md`](../docs/QWEN38-27B-NVFP4-QUANTIZATION-PLAN.md).

## Design

- 768 tokenizer-independent source records.
- A deterministic, balanced selection of 512 records for the primary run.
- Exact category mix: implementation 30%, debugging/testing 20%, code review
  15%, architecture/refactoring 15%, shell/tool/JSON 10%, and
  CI/config/docs/repository exploration 10%.
- Natural record lengths, including a deliberate long-context tail. The source
  and default renderer do not truncate records.
- System, user, and assistant messages in every record. Assistant material is
  drawn from committed implementations, changes, tests, investigations, and
  architecture documents rather than generated chain-of-thought.
- Local committed files only. No dataset download or network access.
- Secret-like values are removed, private-key material is rejected, dependency
  vendoring and generated artifacts are excluded, and local home paths are
  normalized.
- Evaluation tasks are not used as a source corpus.
- Historical LLM smoke/evaluation outputs and benchmark corpora are excluded;
  tests and engineering investigations remain eligible.

The fixed seed is an experimental control: every candidate quantization sees
the same records, so corpus sampling variance does not affect the comparison.

## Build now

```bash
export MODELOPT_CALIBRATION_ROOT="${MODELOPT_CALIBRATION_ROOT:-$HOME/thor-hf-cache/modelopt/calibration}"
export CODING_CALIBRATION_DIR="$MODELOPT_CALIBRATION_ROOT/qwen38-coding"

python3 serving/calibration/build_coding_corpus.py \
  --output-dir "$CODING_CALIBRATION_DIR"
```

The build produces:

```text
coding-source-768.messages.jsonl
coding-selected-512.messages.jsonl
manifest.json
```

Re-running against the same three repository revisions produces identical
JSONL hashes.

## Render after the model is published

Use the tokenizer shipped in the local model directory. Network access is off
by default and `--max-tokens 0` preserves every record completely.

```bash
export SOURCE_MODEL_DIR="$HOME/thor-hf-cache/modelopt/models/qwen38-27b-original"

docker run --rm \
  --entrypoint python3 \
  -v "$SOURCE_MODEL_DIR:/models/source:ro" \
  -v "$CODING_CALIBRATION_DIR:/calibration" \
  -v "$PWD/serving/calibration/render_coding_corpus.py:/opt/render_coding_corpus.py:ro" \
  thor-modelopt:0.45.0 \
  /opt/render_coding_corpus.py \
  --source /calibration/coding-selected-512.messages.jsonl \
  --model /models/source \
  --output /calibration/coding-selected-512.rendered.jsonl \
  --max-tokens 0 \
  --trust-remote-code
```

The renderer writes a ModelOpt-compatible JSONL with one `text` field per
record and a sidecar manifest containing the actual token distribution. If the
released architecture cannot calibrate the natural long tail within Thor's
memory, set an explicit cap only after inspecting that manifest.

If `SOURCE_MODEL_DIR` points into Hugging Face's `hub/models--.../snapshots/`
layout, mount the complete `models--...` directory instead of the snapshot
alone. Snapshot entries are relative symlinks into the sibling `blobs/`
directory and cannot resolve when only the snapshot is visible in Docker.
