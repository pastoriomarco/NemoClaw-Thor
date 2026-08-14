# Qwen 27B quality-first NVFP4 quantization on Thor

**Date**: 2026-08-14
**Status**: Draft runbook; countdown model card inspected, checkpoint-dependent gates pending.
**Target**: `Qwen/Qwen3.8-27B`, a dense 27B native vision-language hybrid
Gated DeltaNet/full-attention model. The locked checkpoint files remain
authoritative for exact module and tensor names.

## Objective

Produce a recommended production candidate on Jetson AGX Thor for delegated
coding, software architecture, repository exploration, testing, debugging,
and review under a frontier Codex orchestrator. A second, higher-risk speed
experiment is retained only for an explicitly approved comparison:

1. **Recommended quality/performance candidate** — use
   `nvfp4_qkv_bf16_local_hessian-kv_fp8_cast.yaml`. Apply local-Hessian NVFP4
   W4A4 to safe eligible projections while retaining full-attention Q/K/V
   (including the fused `q_proj` output gate), Gated DeltaNet
   `in_proj_qkv`/`in_proj_z`, MTP, vision, embeddings, norms, routers, and
   `lm_head` in BF16. Serve with an FP8 KV cache. **Quantize and evaluate this
   candidate first; it is the expected production default and the suggested
   best performance/quality ratio.**
2. **Experimental full-NVFP4 speed challenger** — use
   `nvfp4_full_local_hessian-kv_fp8_cast.yaml` only after the recommended
   candidate passes. It quantizes every ModelOpt-eligible linear projection,
   including sensitive gated projections. It is not recommended for the
   initial overnight run or production without a separately approved,
   controlled quality experiment.

> **Recommended starting point:**
> `nvfp4_qkv_bf16_local_hessian-kv_fp8_cast.yaml` is the suggested recipe for
> the best performance/quality ratio. Do not start with the full-NVFP4 recipe.

If the experimental challenger is run, both candidates use the same complete
768-record coding calibration corpus, released tokenizer/chat template,
source-weight revision, and FP8 KV-cache metadata. The challenger cannot be
promoted merely because it is faster.

Matching the existing Qwen3.6-27B result (~17 average decode tok/s, ~38 tok/s
observed peak with MTP) is a useful reference, not a hard requirement. Coding
reliability, non-looping behavior, valid tool calls, and successful tests are
hard gates.

### Why the recommended recipe preserves attention gates

A documented ModelOpt tracker report localized severe Qwen3.5/3.6 NVFP4
degradation to the gate half fused into full-attention `q_proj`. It reported
teacher-forced perplexity of 65.13 for the existing
`Qwen3.6-27B-Text-NVFP4-MTP` checkpoint and reproduced the divergence by
rounding only that gate through NVFP4. The report used a third-party inference
engine and is not yet an end-to-end vLLM repair validation, but the magnitude
is too large to accept for an agent model. Preserving the entire `q_proj` is
estimated to cost only about 552 MiB on the dense 27B model, roughly 4% of the
quantized checkpoint. See
[ModelOpt issue #2091](https://github.com/NVIDIA/Model-Optimizer/issues/2091).

Separately, an open ModelOpt fix documents export failures when packed Gated
DeltaNet `in_proj_qkv` and `in_proj_z` projections are quantized and then split
by the Qwen weight mapper. The recommended recipe therefore keeps both BF16.
See [ModelOpt PR #1936](https://github.com/NVIDIA/Model-Optimizer/pull/1936).

Local-Hessian scale selection improves the quantized matrices that remain, but
it does not justify quantizing a structurally sensitive sigmoid gate. These
BF16 exclusions are mandatory for the first production candidate.

## Decisions already made

- Quantize locally on Thor in Docker; install nothing on the host beyond the
  existing Docker/NVIDIA runtime and ordinary shell tools.
- Use the locally downloaded BF16 checkpoint by path. Never quantize from a
  mutable Hub model name and never allow the quantization job to refresh
  weights from upstream.
- Use ModelOpt local-Hessian FP8-scale sweep for highest practical PTQ quality.
- Do not wait for ModelOpt 0.46. ModelOpt 0.45 works; 0.46 is an optional
  runtime reduction if a stable image is ready before execution.
- Keep MTP and vision in BF16 for the recommended candidate and any later
  challenger.
- Keep full-attention Q/K/V, the fused `q_proj` output gate, and packed Gated
  DeltaNet `in_proj_qkv`/`in_proj_z` in BF16 for the recommended candidate.
- Keep `lm_head`, embeddings, norms, routers, convolution/state modules, and
  ModelOpt's standard unsupported/sensitive modules unquantized.
- Use FP8 KV cache when serving. Do not attempt NVFP4 KV cache in this plan.
- Calibrate with all 768 prepared records at their natural lengths. Do not
  apply a 2K cap. A cap is an OOM fallback, not the default.
- Build and evaluate only the recommended quality/performance candidate first.
  The full-NVFP4 challenger is optional, requires an explicit decision after
  reviewing the recommended candidate, and starts from untouched source
  weights if approved.
- Never overwrite one output with the other and do not delete the BF16 source
  until the final candidate has passed the full validation gate.

## Prepared assets

### Images

| Image | Current state | Role |
|---|---|---|
| `thor-modelopt:0.45.0` | Built and offline smoke-tested | Primary PTQ image |
| `thor-modelopt-src:0.45.0` | Present | Source/debug fallback |
| `vllm/vllm-openai:v0.27.1` | Present | Serving and HF download image |

The 0.45 image has already completed online and offline Qwen3-0.6B NVFP4 smoke
quantization. Its one-time CUDA extension build takes about 142 seconds on
Thor. The known-good `cnn-dailymail-512.jsonl` fixture is retained only for
image regression smoke; it is not used for the final 27B quantization.

### Calibration corpus

Default root:

```text
$HOME/thor-hf-cache/modelopt/calibration/qwen38-coding/
```

| Asset | Purpose | SHA-256 |
|---|---|---|
| `coding-source-768.messages.jsonl` | Complete tokenizer-independent source; final run | `554f2273254ba86eb22de315542d48ffb9ece612caf92c46c00c703712ca86b6` |
| `coding-selected-512.messages.jsonl` | Faster controlled subset | `6b2a002dbe902653ac61e7e0b52af8fc77105f02e17d135a1b861691fddeefac` |
| `manifest.json` | Revisions, composition, hashes, length distribution | generated |

Exact 768-record composition:

| Category | Records |
|---|---:|
| Implementation and multi-file changes | 230 |
| Debugging, failures, logs, and tests | 154 |
| Code review and defect identification | 115 |
| Architecture and refactoring | 115 |
| Shell, tool calls, and structured data | 77 |
| CI, configuration, docs, and repository exploration | 77 |

The corpus is built only from committed material in `NemoClaw-Thor`,
`manyforge`, and `manyforge_specs`. It excludes dependency vendoring,
generated artifacts, historical LLM smoke outputs, benchmark/evaluation
corpora, secret-like values, and private-key material. Assistant-side material
comes from real implementations, patches, tests, investigations, and design
documents rather than generated chain-of-thought.

A preview with the current Qwen3.6 tokenizer proved the 512-record rendering
path without truncation: 2,025,299 tokens total, 3,956 mean, 18,085 maximum.
The actual Qwen release tokenizer is authoritative; render again after the
weights arrive.

Corpus tooling:

- `serving/calibration/build_coding_corpus.py`
- `serving/calibration/render_coding_corpus.py`
- `serving/calibration/README.md`

### Quantization recipes

Two ModelOpt 0.45 recipes are prepared and loader-validated:

- `serving/calibration/recipes/nvfp4_qkv_bf16_local_hessian-kv_fp8_cast.yaml`
- `serving/calibration/recipes/nvfp4_full_local_hessian-kv_fp8_cast.yaml`

**Start with
`serving/calibration/recipes/nvfp4_qkv_bf16_local_hessian-kv_fp8_cast.yaml`.**
This is the recommended quality/performance recipe. It protects conventional
`self_attn.{q,k,v}_proj`, including the sensitive gate fused into `q_proj`, and
hybrid `linear_attn.{in_proj_qkv,in_proj_z}` modules. It also explicitly
preserves MTP and vision. Safe MLP and output projections still use native
NVFP4 W4A4, retaining most of the expected Thor speed and memory benefit.

The full recipe quantizes these sensitive projections when otherwise eligible
and is retained only as an experimental upper-speed bound. Do not select it
for the initial run merely because its filename says `full`.

**Architecture gate:** these patterns are provisional until the released
checkpoint's module names are inspected. If Qwen changes its layer names, edit
and loader-test the recipe before starting PTQ. Do not assume Qwen3.6 names.

## Expected resource envelope

Planning estimates for a dense 27B model:

| Item | Estimate |
|---|---:|
| BF16 source checkpoint | 54–60 GB |
| Recommended quality/performance candidate | approximately 23–30 GB, architecture-dependent |
| Full eligible NVFP4 candidate | approximately 19–22 GB |
| Free disk before starting | at least 120 GB preferred |
| ModelOpt 0.45 quality run, 768 natural-length records | approximately 4–8 hours |
| ModelOpt 0.45 full run | approximately 5–9 hours |

Thor measurement supporting the scale-search estimate:

- ModelOpt 0.45 local-Hessian on a representative 8192×4096 BF16 matrix:
  4.33 seconds after extension compilation.
- Expected 27B scale-search portion: roughly 45–70 minutes.
- The rest is checkpoint I/O and two corpus passes: max calibration followed
  by Hessian accumulation.

ModelOpt 0.46's fused Triton sweep is reported bit-exact and about 34× faster
for one representative large matrix. It should save roughly one hour, not make
the complete job 34× faster. If 0.46 is available, build it as a separate tag
and smoke-test it; never replace the known-good 0.45 image in place.

## Directory contract

Set these once in the terminal or tmux session. The defaults follow the
existing Thor layout but remain fully parameterized.

```bash
export NEMOCLAW_THOR_REPO="${NEMOCLAW_THOR_REPO:-$HOME/workspaces/dev_ws/src/NemoClaw-Thor}"
export THOR_MODELOPT_ROOT="${THOR_MODELOPT_ROOT:-$HOME/thor-hf-cache/modelopt}"
export THOR_HF_ROOT="${THOR_HF_ROOT:-$HOME/thor-hf-cache}"

export SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-$THOR_MODELOPT_ROOT/models/qwen38-27b-original}"
export QUALITY_OUTPUT_DIR="${QUALITY_OUTPUT_DIR:-$THOR_MODELOPT_ROOT/models/qwen38-27b-qkv-bf16-nvfp4-lh}"
export FULL_OUTPUT_DIR="${FULL_OUTPUT_DIR:-$THOR_MODELOPT_ROOT/models/qwen38-27b-full-nvfp4-lh}"
export CALIBRATION_DIR="${CALIBRATION_DIR:-$THOR_MODELOPT_ROOT/calibration/qwen38-coding}"
export QUANT_LOG_DIR="${QUANT_LOG_DIR:-$THOR_MODELOPT_ROOT/logs}"

export TORCH_CACHE_DIR="${TORCH_CACHE_DIR:-$HOME/thor-torch-cache}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$HOME/thor-vllm-cache}"
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-$HOME/thor-flashinfer-cache}"

export MODELOPT_IMAGE="${MODELOPT_IMAGE:-thor-modelopt:0.45.0}"
export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.27.1}"
```

Create the persistent directories:

```bash
mkdir -p \
  "$SOURCE_MODEL_DIR" \
  "$QUALITY_OUTPUT_DIR" \
  "$FULL_OUTPUT_DIR" \
  "$CALIBRATION_DIR" \
  "$QUANT_LOG_DIR" \
  "$TORCH_CACHE_DIR" \
  "$VLLM_CACHE_DIR" \
  "$FLASHINFER_CACHE_DIR"
```

The existing empty `qwen38-27b-nvfp4` placeholder is intentionally not used;
explicit candidate names prevent accidental overwrite or confusion.

## Phase 0 — pre-release preparation

Completed as of 2026-08-14:

- [x] ModelOpt 0.45 image built.
- [x] Online small-model PTQ smoke passed.
- [x] Fully offline small-model PTQ smoke passed.
- [x] Local CNN/DailyMail smoke fixture retained.
- [x] 768-record coding corpus generated and validated.
- [x] Deterministic 512-record subset generated.
- [x] Corpus renderer implemented with network disabled by default.
- [x] Current-Qwen tokenizer preview passed without truncation.
- [x] Quality and speed local-Hessian recipes added and loader-validated on
  ModelOpt 0.45.
- [x] Countdown model card confirms a dense 27B native vision-language model
  with the Qwen3.5/3.6 hybrid Gated DeltaNet/full-attention dimensions.

Pending until release:

- [ ] Exact model repository and immutable revision.
- [ ] Released tokenizer/chat template rendering.
- [ ] Architecture/module-name inspection.
- [ ] Exact vision and MTP tensor prefixes and inference-ready MTP packaging.
- [ ] Multimodal calibration supplement if image/screenshot behavior is a
  required quality gate.

## Phase 1 — acquire and freeze the original checkpoint

### 1. Choose the correct release artifact

Prefer the original BF16 model containing every native feature required for
production:

- coding/reasoning behavior;
- MTP head, if NVIDIA/Qwen releases one;
- vision branch, if vision is required;
- original tokenizer, processor, chat template, and remote-code files.

Do not start from a third-party quant and do not use a text-only derivative if
vision is a production requirement.

Record an immutable Hugging Face commit SHA:

```bash
export MODEL_ID="<publisher>/<exact-27B-model>"
export MODEL_REVISION="<full-Hugging-Face-commit-SHA>"
```

Download from inside the existing vLLM container. This is the only mandatory
large download:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --entrypoint hf \
  -e HF_TOKEN \
  -e HF_HOME=/tmp/huggingface \
  -v "$SOURCE_MODEL_DIR:/models/source" \
  "$VLLM_IMAGE" \
  download "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --local-dir /models/source
```

After download, all later commands use `/models/source`, never `$MODEL_ID`.

### 2. Verify checkpoint completeness and precision

```bash
test -s "$SOURCE_MODEL_DIR/config.json"
test -s "$SOURCE_MODEL_DIR/tokenizer_config.json"
```

```bash
jq '{architectures,model_type,dtype,quantization_config,text_config}' \
  "$SOURCE_MODEL_DIR/config.json"
```

The source must be BF16/FP16 and must not already contain an active weight
`quantization_config`. If it has a safetensors index, verify every referenced
shard exists:

```bash
jq -r '.weight_map | values[]' "$SOURCE_MODEL_DIR/model.safetensors.index.json" \
  | sort -u \
  | while read -r shard; do test -s "$SOURCE_MODEL_DIR/$shard" || exit 1; done
```

Record source provenance locally:

```bash
printf '%s\n' "$MODEL_ID" > "$SOURCE_MODEL_DIR/SOURCE_MODEL_ID"
printf '%s\n' "$MODEL_REVISION" > "$SOURCE_MODEL_DIR/SOURCE_REVISION"
```

Do not modify files under `$SOURCE_MODEL_DIR` after this point.

## Phase 2 — architecture gate

Before quantization, inspect:

- `architectures`, `model_type`, and nested `text_config`;
- dense versus MoE topology;
- standard attention versus hybrid/linear-attention layer types;
- exact Q/K/V and hybrid input-projection names;
- MTP layer count and tensor prefixes;
- vision tower and multimodal-projector prefixes;
- tied versus independent embeddings and `lm_head`;
- `max_position_embeddings` and rope configuration.

List relevant tensor names without loading the 27B weights into RAM:

```bash
python3 - "$SOURCE_MODEL_DIR" <<'PY'
import glob
import json
import os
import struct
import sys

root = sys.argv[1]
needles = ("q_proj", "k_proj", "v_proj", "qkv", "linear_attn", "mtp", "visual", "vision", "lm_head")
for path in sorted(glob.glob(os.path.join(root, "*.safetensors"))):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    for name, metadata in header.items():
        if name != "__metadata__" and any(needle in name for needle in needles):
            print(name, metadata.get("dtype"), metadata.get("shape"))
PY
```

Compare the resulting names to the recommended recipe first. Required
corrections must be made before PTQ and rechecked by loading the YAML through
ModelOpt. Reconcile the experimental recipe only if that later run is
explicitly approved. Key rule:
the recommended candidate must leave every QKV-equivalent projection, fused
attention gate, and packed Gated DeltaNet Z/gating projection BF16. Do not
start PTQ unless the observed names are covered by the quality recipe.

If the model is MoE rather than dense, stop and revise the plan: expert routing,
active-parameter cost, calibration coverage, and serving performance differ
materially from this dense-27B plan.

## Phase 3 — render the full corpus with the released tokenizer

Render all 768 records locally and without truncation:

```bash
docker run --rm \
  --entrypoint python3 \
  --network none \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "$SOURCE_MODEL_DIR:/models/source:ro" \
  -v "$CALIBRATION_DIR:/calibration" \
  -v "$NEMOCLAW_THOR_REPO/serving/calibration/render_coding_corpus.py:/opt/render_coding_corpus.py:ro" \
  "$MODELOPT_IMAGE" \
  /opt/render_coding_corpus.py \
  --source /calibration/coding-source-768.messages.jsonl \
  --model /models/source \
  --output /calibration/coding-source-768.rendered.jsonl \
  --max-tokens 0 \
  --trust-remote-code
```

Inspect the actual distribution:

```bash
jq . "$CALIBRATION_DIR/coding-source-768.rendered.jsonl.manifest.json"
```

Set `CALIB_SEQ` to the next 1,024-token boundary above the reported maximum:

```bash
export CALIB_MAX_TOKENS="$(jq -r '.tokens.maximum' "$CALIBRATION_DIR/coding-source-768.rendered.jsonl.manifest.json")"
export CALIB_SEQ="$(( (CALIB_MAX_TOKENS + 1023) / 1024 * 1024 ))"
printf 'CALIB_MAX_TOKENS=%s CALIB_SEQ=%s\n' "$CALIB_MAX_TOKENS" "$CALIB_SEQ"
```

Do not silently reduce this value. If the longest record causes OOM at batch
one, record the failure and retry with a declared 16K cap; do not reduce the
sample count and length simultaneously.

If the checkpoint includes vision and screenshot/image understanding is a
production requirement, text-only calibration is not a complete validation.
Keep the vision tower BF16 and add a small, separate image-text calibration
supplement that reflects code screenshots, terminal captures, diagrams, and
UI failure states. This asset is not prepared yet because the released
processor and multimodal input contract are unknown.

## Phase 4 — machine preflight

Run the overnight job inside `tmux` so a terminal disconnect does not stop it:

```bash
tmux new -s qwen27b-quant
```

Confirm images, inputs, disk, and output-directory isolation:

```bash
docker image inspect "$MODELOPT_IMAGE" >/dev/null
test -s "$CALIBRATION_DIR/coding-source-768.rendered.jsonl"
df -h "$THOR_MODELOPT_ROOT"
free -h
```

Stop the current vLLM serving container after resolving its exact name:

```bash
docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
docker stop <current-vllm-container-name>
```

Release filesystem page cache immediately before starting PTQ, as required by
the established Thor workflow:

```bash
sudo sysctl -w vm.drop_caches=3
```

Do not run another model server or GPU-heavy workload during calibration.

Before each run, require an empty dedicated output directory. Never delete or
reuse a non-empty directory without inspecting it:

```bash
test -z "$(find "$QUALITY_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)"
```

## Phase 5 — quantize the recommended quality/performance candidate

The container stays in the foreground and all output is duplicated to a
persistent log. No model or dataset download is possible during the run.

```bash
set -o pipefail
```

```bash
docker run --rm \
  --name qwen27b-quant-quality \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  --network none \
  --entrypoint python3 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/data/models/huggingface \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor \
  -v "$SOURCE_MODEL_DIR:/models/source:ro" \
  -v "$QUALITY_OUTPUT_DIR:/models/output" \
  -v "$CALIBRATION_DIR:/calibration:ro" \
  -v "$NEMOCLAW_THOR_REPO/serving/calibration/recipes:/recipes:ro" \
  -v "$THOR_HF_ROOT:/data/models/huggingface" \
  -v "$TORCH_CACHE_DIR:/root/.cache/torch" \
  "$MODELOPT_IMAGE" \
  /opt/modelopt/examples/llm_ptq/hf_ptq.py \
  --pyt_ckpt_path /models/source \
  --recipe /recipes/nvfp4_qkv_bf16_local_hessian-kv_fp8_cast.yaml \
  --dataset /calibration/coding-source-768.rendered.jsonl \
  --calib_size 768 \
  --calib_seq "$CALIB_SEQ" \
  --batch_size 1 \
  --export_path /models/output \
  --trust_remote_code \
  --skip_generate \
  2>&1 | tee "$QUANT_LOG_DIR/qwen27b-quality-local-hessian.log"
```

Check `${PIPESTATUS[0]}` immediately; a successful `tee` must not mask a failed
container:

```bash
test "${PIPESTATUS[0]}" -eq 0
```

ModelOpt 0.45 uses a Python-driven 126-candidate Hessian-weighted sweep for
each eligible weight matrix. Long periods without new terminal lines can be
normal during that phase. Follow from another terminal with:

```bash
tail -F "$QUANT_LOG_DIR/qwen27b-quality-local-hessian.log"
```

## Phase 6 — validate the quality checkpoint before another overnight run

Minimum artifact checks:

```bash
test -s "$QUALITY_OUTPUT_DIR/config.json"
find "$QUALITY_OUTPUT_DIR" -maxdepth 1 -name '*.safetensors' -type f -size +1M | sort
du -sh "$QUALITY_OUTPUT_DIR"
jq '.quantization_config' "$QUALITY_OUTPUT_DIR/config.json"
```

Inspect the exported quantization summary if present:

```bash
find "$QUALITY_OUTPUT_DIR" -maxdepth 1 -name '*quant*summary*' -o -name 'hf_quant_config.json'
```

Tensor-level gate:

- MLP and attention-output weights expected by the recipe are packed NVFP4.
- Conventional Q/K/V, the fused `q_proj` gate, and hybrid
  `in_proj_qkv`/`in_proj_z` weights are BF16.
- MTP weights remain BF16 and are present.
- Vision weights/projector remain BF16 and are present if the source had them.
- `lm_head`, embeddings, norms, routers, conv/state paths remain high precision.
- No shard is missing and the tokenizer/processor/chat-template files survived
  export.

If any expected module is accidentally quantized or absent, reject the
checkpoint and correct the recipe. Do not try to compensate at serving time.

## Optional Phase 7 — full-NVFP4 risk experiment

**Do not run this phase as part of the initial quantization.** Stop after the
recommended candidate has been served and evaluated. Run this experiment only
after explicitly deciding that a possible decode-speed gain justifies testing
the known gated-projection quality risk.

Start from the untouched BF16 directory and use the identical rendered corpus,
order, calibration length, batch size, and ModelOpt version. Confirm the full
output directory is empty first.

```bash
test -z "$(find "$FULL_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)"
```

Repeat the foreground Docker command from Phase 5 with exactly these changes:

```text
--name qwen27b-quant-full
-v "$FULL_OUTPUT_DIR:/models/output"
--recipe /recipes/nvfp4_full_local_hessian-kv_fp8_cast.yaml
tee "$QUANT_LOG_DIR/qwen27b-full-local-hessian.log"
```

Do not reuse the already quantized quality candidate as input. PTQ candidates
must be independently derived from the same frozen BF16 source.

Validate the same preserved native features. For this risk experiment,
QKV-like and gating linear weights will be packed NVFP4 unless a standard
ModelOpt exclusion applies. That behavior is precisely the quality risk under
test. “Full” means **full ModelOpt-eligible**, not embeddings, norms, MTP,
vision, routers, unsupported conv/state modules, or every tensor in the file.

## Optional Phase 8 — ModelOpt 0.46 / AutoQuantize

This phase is not required to obtain either primary checkpoint.

If stable ModelOpt 0.46 is available:

1. Build `thor-modelopt:0.46.0`; keep `thor-modelopt:0.45.0` unchanged.
2. Repeat the small offline CNN/DailyMail smoke.
3. Load-test both custom YAML recipes.
4. If only the faster local-Hessian kernel changed, either use 0.46 for the
   final run or retain 0.45; expected checkpoint quality should be identical.
5. Treat four-over-six and AutoQuantize as new candidates requiring separate
   evaluation, not silent upgrades to the baseline recipe.

The higher-effort third candidate would pin QKV/MTP/vision to BF16 and let
AutoQuantize select NVFP4, FP8, or BF16 for remaining layers at an initial
5.5–6.0 effective-bit budget. NVIDIA's Nemotron evidence favors this type of
mixed precision for near-BF16 median quality, but there is no direct Qwen-27B
coding-agent ablation yet. Do this only if both fixed recipes miss the desired
quality/speed balance.

## Phase 9 — vLLM serving validation

Test each candidate first at batch one and without MTP. This isolates the
target checkpoint from speculative-decoding behavior. Then enable the retained
BF16 MTP head and measure acceptance/speed.

Baseline serving policy:

- local checkpoint path only;
- vLLM 0.27.1 or later version verified on Thor;
- `--quantization modelopt`;
- FlashInfer attention and native NVFP4 linear backend on SM110;
- FP8 KV cache;
- 262,144 maximum model length if supported by the released model;
- `--max-num-seqs 1` for controlled comparison;
- 8,192 batched-token prefill budget;
- prefix caching and chunked prefill;
- 0.80 GPU-memory utilization as the initial value;
- temperature zero for deterministic quality comparison.

Use the current proven flags as a starting point, but gate these on the actual
release:

- confirm the reasoning parser name;
- confirm the tool-call parser name;
- confirm MTP support and `num_speculative_tokens=3` compatibility;
- omit `--language-model-only` when vision must be served;
- include `--language-model-only` only for an intentionally text-only target.

After batch-one comparison, test `--max-num-seqs 4` for production concurrency.
The scheduler can batch concurrent calls, but single-request decode speed and
aggregate throughput must be reported separately.

## Phase 10 — quality and behavior gate

Do not select the winner from MMLU or a short chat response. The production
model is a coding subagent. Use held-out tasks that were excluded from
calibration:

- multi-file implementation with tests;
- diagnosis from compiler errors, stack traces, and failing tests;
- repository exploration followed by a correct localized patch;
- code review with seeded correctness/concurrency/security defects;
- architecture and migration-plan reasoning;
- shell and structured tool calls with strict JSON validation;
- long-context repository/audit tasks;
- repeated-action and no-progress loop detection;
- hallucinated-file and fabricated-test-result detection.

Evaluation controls:

- same prompts and repository revisions;
- same system prompt/tool schema/chat template;
- temperature zero;
- same maximum output budget;
- first compare without MTP, then with the same MTP configuration;
- record completion status, tests, tool validity, loops, latency, prefill, and
  decode statistics.

Hard rejection conditions:

- incoherent or corrupted text;
- new repetitive/looping behavior;
- invalid or unstable tool-call serialization;
- missing vision or MTP functionality promised by the source;
- recurring failure to finish patches or tests;
- significant long-context regression.

Exceptional promotion rule for the full-NVFP4 risk experiment:

- no critical behavioral regression;
- no more than one additional failure in a 50-task fixed coding suite relative
  to the quality candidate;
- tool-call validity and loop rate no worse;
- a meaningful decode-speed gain, preferably at least 10–15%.

The reported gated-attention degradation means this challenger starts with a
presumption of rejection. Even if it passes the short suite, inspect
perplexity, long-context tool use, and loop behavior before considering it. If
it is less than 10% faster, prefer the recommended candidate automatically. If
the recommended candidate is still unreliable relative to BF16, fall back to
a stricter OMLP-only recipe or the optional AutoQuantize path rather than
accepting a fragile agent.

## Performance measurement and planning envelope

Use vLLM metrics/logs without disturbing serving. Report:

- prefill tok/s: average excluding zero samples and maximum;
- decode tok/s: average excluding zero samples and maximum;
- MTP acceptance rate;
- end-to-end latency and task success;
- batch-one versus concurrent aggregate throughput.

Pre-release theoretical ranges, anchored to the existing Qwen3.6 serving:

| Candidate | Approximate average decode | Approximate observed peak |
|---|---:|---:|
| Experimental full ModelOpt-eligible NVFP4 + MTP; not recommended | 16–20 tok/s | 34–42 tok/s |
| **Recommended QKV/gates-BF16 NVFP4 + MTP** | **13–17 tok/s** | **28–37 tok/s** |
| Stricter OMLP-only fallback | 10–15 tok/s | architecture-dependent |

These are planning ranges, not promises. Qwen3.8 topology, hybrid-attention
mix, kernel routing, MTP acceptance, long-context state, and BF16↔NVFP4 kernel
transitions can change the result materially.

## Cleanup policy

Keep until final promotion:

- immutable BF16 source;
- both candidate checkpoints;
- both quantization logs;
- rendered and source calibration corpora;
- corpus and tokenizer-render manifests;
- exact custom recipes;
- serving commands and evaluation results.

After a winner is selected, the losing checkpoint may be deleted explicitly.
Keep the small CNN/DailyMail fixture until the ModelOpt image/version is final;
it costs only about 1.8 MB and is the known-good offline regression input.

Do not upload checkpoints or publish model cards automatically. Any external
publication requires a separate decision after license, provenance, benchmark,
and checksum review.

## Final execution checklist

- [ ] Exact BF16 27B release selected.
- [ ] Immutable Hub revision recorded and downloaded locally.
- [ ] Checkpoint complete; no pre-existing quantization.
- [ ] Architecture, MTP, vision, QKV, hybrid, and `lm_head` names inspected.
- [ ] Recommended recipe patterns reconciled with actual module names,
      including `q_proj` gate, `in_proj_qkv`, and `in_proj_z`.
- [ ] 768 records rendered with actual tokenizer, no truncation.
- [ ] Optional image-text supplement prepared if vision quality is required.
- [ ] At least 120 GB disk free; output directories empty and separate.
- [ ] Existing vLLM stopped; page cache dropped; tmux/logging active.
- [ ] Recommended candidate quantized from BF16 and tensor-level validated.
- [ ] Recommended candidate boots in vLLM without MTP.
- [ ] Recommended candidate boots with retained BF16 MTP, if supported.
- [ ] Optional full-NVFP4 experiment separately approved before it is run.
- [ ] If approved, full candidate independently derived from BF16 and
      tensor-level validated.
- [ ] Vision smoke passes when required.
- [ ] Fixed held-out coding/agent suite completed.
- [ ] Batch-one and concurrency performance captured.
- [ ] Winner selected using the promotion rule.
- [ ] Serving profile added only after promotion.
