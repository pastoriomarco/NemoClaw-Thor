# Cosmos-Reason2 — fine-tune + NVFP4 quantize on Thor

Working plan for improving the historical cosmos-reason2 production default
(`cosmos-reason2-8b` at 9/9 smoke on the ManyForge composer-assistant lane,
flipped 2026-05-07) by either fine-tuning the existing 8B or distilling it
into a tuned 2B, then NVFP4-quantizing the text weights for deployment on
Jetson AGX Thor.

Status as of 2026-06-11: superseded for the clean-start ManyForge assistant
default by `gemma4-12b-it-gguf`. Keep this as a Cosmos-specific research and
reproduction note, not as current default-selection guidance.

This is a **plan + reference**, not a shipped procedure. Steps that have not
been executed are explicitly marked.

---

## Goal

A cosmos-reason2-based profile that:

- maintains or exceeds the current 8B smoke score (9/9 on `smoke_corpus.yaml`),
- is smaller and/or faster than stock 8B on Thor,
- is produced and deployed entirely on Thor (no external GPU dependency),
- carries no third-party model-output licensing risk.

---

## Why fine-tune at all

The historical production cosmos-reason2-8b default reaches 9/9 on the curated smoke
corpus but the corpus is small. Lane-parity work in early May 2026 (see
[`manyforge/docs/LANE-COMPARISON.md`](../../manyforge/docs/LANE-COMPARISON.md))
landed several vendor-side adjustments (sampling, MCP null-arg validation,
removed schema examples) but model-side behavior is still the gating factor
on edge cases. Targeted SFT against the OpenClaw bridge audit log set is
expected to:

- raise first-try success rates on the smoke corpus (currently 49/66 = 74.2%
  at iter 20 on the broader corpus, even with cosmos-reason2-8b),
- tighten tool-calling adherence (schema fidelity, null-arg avoidance),
- preserve the underlying Qwen3-VL physical-AI reasoning prior.

NVFP4 quantization of the **text weights only** (vision tower stays BF16) is
a deployment win independent of fine-tuning: roughly 4× smaller weights, no
KV-cache quantization needed.

---

## Architecture facts (load-bearing)

| Property | cosmos-reason2-8b | cosmos-reason2-2b |
|---|---|---|
| Base model | Qwen3-VL-8B | Qwen3-VL-2B |
| model_type | `qwen3_vl` | `qwen3_vl` |
| HF class | `Qwen3VLForConditionalGeneration` | same |
| head_dim | 128 | 128 |
| KV heads | 8 | (verify at first load) |
| Layers | 36 | (smaller) |
| Native max_position | 262144 | (verify) |
| RoPE | M-RoPE (multimodal) | same |
| License | NVIDIA Open Model License (commercial + derivatives allowed) | same |

Profile registrations already present in this repo:

- [`serving/config.sh:314`](../config.sh#L314) — `cosmos-reason2-2b` (32K ctx, 2-conc, FP8 KV)
- [`serving/config.sh:327`](../config.sh#L327) — `cosmos-reason2-8b` (262K ctx, FP8 KV)
- [`serving/launch.sh:76`](../launch.sh#L76) — `cosmos-reason2-2b` launch case
- `serving/launch.sh` — `cosmos-reason2-8b` launch case at ~line 109

No new profile registration is needed before the stock-2B baseline run.

---

## Official upstream tooling

NVIDIA ships first-class tooling for Cosmos-Reason2 quantization and
fine-tuning at `github.com/nvidia-cosmos/cosmos-reason2`:

| Artifact | Purpose |
|---|---|
| `scripts/quantize.py` | NVFP4 / FP8 / FP8-dynamic quantization wrapper over llm-compressor |
| `examples/cosmos_rl/` | RL fine-tuning recipes (FSDP, AdamW, BF16 params, fp32 master) |
| `configs/cosmos_rl_config.toml` | RL config template |
| `docs/llmcompressor.md` | Quantization option reference |
| `Dockerfile` (with `--extra cu130`) | Documented Jetson AGX + DGX Spark support |

llm-compressor has a direct-match example at
`examples/multimodal_vision/qwen3_vl_example.py` — Cosmos-Reason2 *is* a
Qwen3-VL derivative, so the recipe applies verbatim.

Quantization defaults from `docs/llmcompressor.md`:

```
--precision nvfp4          # smallest/fastest; fp8 / fp8_dynamic also supported
--kv-precision bf16        # fp8 also supported (we run fp8 in production)
--num-samples 512          # calibration samples
--smoothing-strength 0.8   # SmoothQuant strength
```

Required environment on Jetson AGX (per NVIDIA's Dockerfile):

```bash
export TRITON_PTXAS_PATH=$(find / -name ptxas | head -1)
```

---

## Hard constraint: must run on Thor

Thor has 128 GB unified memory (CPU+GPU share the same pool). All training
and quantization peak memory must fit there, with headroom for the kernel,
the build, and any concurrent inference.

### Memory math

| Workload | Peak (GB) | Verdict on 128 GB Thor |
|---|---|---|
| 8B **LoRA** SFT — rank 32, BF16 base, AdamW master fp32, grad-ckpt, bs=1×grad-accum 16 | 28-35 | ✓ comfortable |
| 8B **full** SFT — BF16, AdamW master fp32, grad-ckpt | 120-140 | ✗ exceeds |
| 8B LoRA + KL ref (frozen base loaded twice) | ~50 | ✓ fits |
| 2B **full** SFT — bs=1×grad-accum 16 | 32-40 | ✓ comfortable |
| 2B full SFT + KL ref (frozen 2B base) | ~50 | ✓ fits |
| 2B full SFT + KL ref using 8B teacher | ~60-70 | ✓ fits |
| 8B NVFP4 quantization (llm-compressor, 512 calib, 2K seq) | 30-45 | ✓ fits |
| 2B NVFP4 quantization | 10-15 | ✓ trivial |

**Decision rule**: full SFT is the default at 2B, LoRA is the default at 8B.
Full SFT on 8B is out of scope for Thor — that requires DGX Spark or rented
H100 time, which we are explicitly avoiding.

---

## Path selection: 2B-distilled-from-8B is the recommended first attempt

Three candidate paths and their tradeoffs:

| Path | Quality vs stock-2B-SFT | Effort | Risk |
|---|---|---|---|
| **Distill 8B → 2B** (sequence-level: replay prompts through your own 8B, train 2B on its outputs) | Usually meaningfully better — transfers schema-aligned, smoke-corpus-aligned behavior | Medium | Low — your own 8B has no ToS issues, already matches your prompt distribution |
| Stock 2B + frontier-distilled SFT (Claude/GPT/Gemini outputs as targets) | Strong but brings ToS questions; OpenAI ToS restricts training competing models, Anthropic restricts competing models, Gemini is more permissive | Medium | Medium — license review needed for commercial deployment |
| Stock 2B + bridge-audit-log SFT only | Limited by what real-traffic data you have | Low | Low — but data-bound |
| 8B LoRA + bridge-audit-log SFT | High-quality 8B variant; same deployment cost as today | Medium | Low — fallback if 2B path falls short |
| Pruned 8B (layer or width prune) | Fragile; loses 5-15% on reasoning benches; needs recovery training | High | High — research-grade |

**Recommended ordering**:

1. Establish stock-2B baseline against smoke corpus (no training; ~1 hour).
2. Pilot: distill 8B → 2B with 1K samples; smoke regression. Decide if signal is there.
3. Scale: 8B → 2B distillation at 2-3K curated samples + replay + KL reg; smoke regression.
4. Quantize the resulting 2B to NVFP4; smoke regression.
5. If 2B falls short at any stage, pivot to 8B LoRA using the same dataset.

---

## Distillation strategy: use your own production 8B as teacher

The 8B already in production has three properties no external model has:

- aligned to **your exact tool schemas** (the OpenClaw bridge ones),
- aligned to **your OpenClaw envelope conventions** (max_tokens, sampling, MCP),
- scored 9/9 on the **exact corpus** the student will be judged against.

That alignment transfers to the 2B student in a way no public-data SFT or
frontier-API distillation can match — and there is **no third-party model
output ToS question** because the teacher is your own model.

### Sequence-level distillation (cheap, recommended first)

For each prompt P in your training set:

1. Run P through cosmos-reason2-8b in production-equivalent settings (same
   sampling, same tool schemas, same MCP wrapper).
2. Capture the 8B's full response (text + tool calls + final answer).
3. Filter aggressively (see "Data curation" below).
4. Use the filtered (P, 8B-response) pairs as SFT targets for the 2B student.

This is computationally cheap (one inference pass per training sample) and
captures the teacher's *behavior*. It does not capture probability mass on
non-chosen tokens.

### Logit-level distillation (more expensive, optional)

Capture top-K logits (K=16-64) from the 8B teacher at every token position;
train the 2B student with KL loss against those logits. Bigger lift on hard
cases, ~3-5× more storage per sample (top-K logits per token), ~2× training
memory (teacher resident during training).

Reserve this for a second iteration if sequence-level distillation plateaus.
At Thor scale, sequence-level is the right starting point.

---

## Retaining base capability with fewer samples

Five techniques that compose. Roughly ordered by bang-for-buck:

### a) Replay mixing

Mix general-domain samples into the task-specific data to prevent
catastrophic forgetting on capabilities the smoke corpus doesn't exercise.

Typical ratio: **70-90% task-specific + 10-30% replay**.

Public replay candidates:

| Dataset | Size | Why it fits |
|---|---|---|
| Tulu-3 SFT mix (Allen AI) | ~940K (subsample) | High-quality general instruct; permissive license |
| OpenHermes-2.5 | ~1M | Strong general reasoning |
| UltraChat | ~200K | Multi-turn assistant patterns |
| Cosmos-Reason post-train data | TBD | Best fit if/when NVIDIA releases it |

Practical mix for the first iteration: 2K composer-assistant samples + 300-500
random Tulu-3 samples.

### b) KL regularization to the base model

Add a loss term that penalizes drift from the original cosmos-reason2 logits:

```
loss = α · CE(student, target) + β · KL(student || frozen_base)
```

Starting values: α=0.7, β=0.3. The frozen base acts as a gravity well —
student moves toward the new data but is pulled back toward base capabilities.
Reported sample efficiency gains in published recipes: 2-3× (meaning 1K
samples with KL ≈ 3K samples without).

Cost: 2× training memory (base + student both resident). On a 2B student
that's ~50 GB total — fits Thor. On an 8B-LoRA setup it's ~50 GB — also fits.

### c) LoRA as soft regularization

LoRA constrains the update to a low-rank subspace; the merged result is a
small delta on the base, inherently preserving more of the base distribution.

On 2B, LoRA caps learning capacity — full SFT is preferred by default. If
full SFT is observed to forget too much general capability, rank-128 LoRA
on 2B is a good middle ground.

On 8B (where full SFT is out of reach), LoRA is the only option.

### d) Curriculum, low LR, few epochs

- LR: 5e-5 for full SFT, 2e-4 for LoRA. Lower than typical instruct recipes.
- Epochs: 1-2, not 3-5. Less re-exposure → less overwrite.
- Curriculum: easy → hard. The model anchors on familiar patterns first.

### e) Data quality > data quantity

500 carefully curated samples (validator-filtered, deduped,
schema-version-pinned) routinely beat 5,000 noisy ones. Curation is where
the human work is and where the wins compound.

---

## Data curation (where the actual work is)

### Sources

1. **Bridge audit logs** from `manyforge/openclaw_assistant_bridge/` — real
   composer-assistant traffic. Each entry has (prompt, tool_calls,
   final_response, success/fail). Highest-signal source.
2. **Smoke corpus prompts** (`manyforge/scripts/debug/smoke_corpus.yaml`)
   replayed through the 8B teacher — guaranteed schema fidelity, guaranteed
   evaluator coverage.
3. **Synthetic prompt expansion** — have the 8B teacher generate variations
   (different scenes, different tool sequences, different P1/P2/P3 mode
   boundaries) on each curated seed prompt to broaden coverage cheaply.

### Filters (apply in order)

1. **Schema validation** — every tool call must validate against the bridge's
   wrapper validator. Schema hash must match the deployment schema hash.
2. **Null-arg check** — bridge's null-arg validator (the load-bearing
   safeguard from the schema-examples decision).
3. **Smoke-corpus oracle** — if the prompt is from the smoke corpus, the
   8B response must pass the corpus's expected-outcome check.
4. **Deduplication** — by (prompt-hash, tool-sequence-hash). Keep the highest
   quality variant per duplicate group.
5. **Length filter** — drop traces where the 8B's reasoning chain exceeds
   ~4K tokens. The 2B student can't reproduce that depth; teaching it to
   try produces incoherent truncations.
6. **Failure pairing (optional, for DPO later)** — pair passing traces with
   failed traces on the same prompt for preference learning in a second stage.

### Volume targets

| Iteration | Curated samples | Replay samples | Total | Use case |
|---|---|---|---|---|
| Pilot | 800-1200 | 200 | ~1K-1.4K | Sanity-check the pipeline, decide if 2B can learn |
| Production-quality first run | 2K-3K | 300-500 | ~2.5K-3.5K | Target 9/9 smoke + improved broader corpus |
| Saturation run (if needed) | 5K-8K | 800-1500 | ~6K-10K | If iteration 2 falls short |

The pilot's 1K-1.4K total is achievable in days of curation work; the
production run scales the same pipeline ~3×.

---

## Concrete recipe (recommended first attempt)

### Step 0 — baseline (no code changes)

```bash
./serving/start-model.sh cosmos-reason2-2b
# Run smoke corpus against it; record the baseline score.
# Profile already exists at serving/config.sh:314 and serving/launch.sh:76.
```

If "Context overflow: prompt too large" fires on short prompts, bump the 2B
profile's `THOR_TARGET_MAX_MODEL_LEN` from 32768 → 65536 first — the
OpenClaw gateway's preemptive context-overflow guard is tuned per profile.

### Step 1 — derived fine-tune image (don't pollute v8.1 serving image)

```dockerfile
# serving/docker/Dockerfile.finetune (NEW — not yet created)
FROM nemoclaw-thor/vllm:v8.1
RUN uv pip install --system \
      peft \
      trl \
      llm-compressor \
      accelerate \
      datasets
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

Tag: `nemoclaw-thor/finetune:v8.1-ft`. Build once (~30 min).

Keeping this separate from the serving image means an FT-toolchain bump
(peft/trl/llm-compressor moves fast) cannot destabilize the serving image.

### Step 2 — distillation data generation

```bash
# In the FT image:
python tools/distill_from_8b.py \
  --teacher cosmos-reason2-8b \
  --prompt-set bridge-audit-logs + smoke-corpus + synthetic \
  --output data/distilled-v1.jsonl \
  --num-samples 3000
# Apply filters: schema validation → null-arg → smoke oracle → dedup → length.
# Expect ~60-80% to survive → ~1800-2400 clean samples.
```

(`tools/distill_from_8b.py` is referenced as the future script — not yet
written. It is essentially a batched OpenAI-compatible inference loop
against the production 8B + the filter chain documented above.)

### Step 3 — SFT with KL regularization + replay

```bash
# In the FT image:
python tools/train_sft_kl.py \
  --base nvidia/Cosmos-Reason2-2B \
  --train data/distilled-v1.jsonl \
  --replay data/tulu3-subsample-500.jsonl \
  --kl-beta 0.2 \
  --lr 5e-5 \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 16 \
  --grad-checkpoint \
  --output checkpoints/cosmos-reason2-2b-tuned-v1
# Expected wallclock on Thor: ~3-4 hours for ~2-3K samples.
```

(`tools/train_sft_kl.py` is referenced as the future script — not yet
written. TRL's `SFTTrainer` with a manually added KL term against a frozen
reference model is the standard recipe.)

### Step 4 — NVFP4 quantization

```bash
# In the FT image, using NVIDIA's official script:
python scripts/quantize.py \
  --model checkpoints/cosmos-reason2-2b-tuned-v1 \
  --precision nvfp4 \
  --kv-precision fp8 \
  --num-samples 512 \
  --smoothing-strength 0.8 \
  --output checkpoints/cosmos-reason2-2b-tuned-v1-nvfp4
# Expected wallclock on Thor: ~10-15 minutes for 2B.
```

The vision tower stays BF16 — llm-compressor's qwen3_vl recipe ignores the
visual encoder by default. Preserve that.

### Step 5 — register as a new profile

Add a new case to [`serving/config.sh`](../config.sh) and
[`serving/launch.sh`](../launch.sh), e.g. `cosmos-reason2-2b-tuned-nvfp4`,
pointing at the local checkpoint. Mirror the existing 2B profile's KV
settings (32K ctx, FP8 KV, 2-conc) until measured otherwise.

### Step 6 — smoke regression

```bash
./serving/start-model.sh cosmos-reason2-2b-tuned-nvfp4
# Run the full smoke corpus.
# Required to ship: ≥ 9/9 on the curated set (matches current 8B baseline).
# Nice to have: improvement on the broader 66-case corpus (currently 49/66 at iter 20).
```

If 9/9 holds, flip the profile target. Keep stock cosmos-reason2-8b as
fallback.

---

## Caveats specific to Thor / SM110a

- **No SM110a NVFP4 KV-cache cubins** exist in NVIDIA's artifactory (only
  Sm100a / Sm100f / Sm103a as of v8.1). KV-cache NVFP4 path is blocked
  upstream — this was the failure mode of the v8.2-experimental build.
  **Weight NVFP4 is unaffected** and is what this plan uses.
- **TRITON_PTXAS_PATH** must be set on Jetson AGX or NVIDIA's tooling fails
  silently in unhelpful ways. Bake into the FT image entrypoint.
- **Don't quantize the vision tower** — keep BF16. The Qwen3-VL llm-compressor
  recipe defaults to text-only quantization; preserve that default.
- **Validate against the smoke corpus** (not just generic instruct benches)
  before flipping any production profile. The corpus is the contract.

---

## What this plan deliberately does not include

- **Full SFT on 8B** — exceeds Thor memory. Out of scope.
- **Pruning (layer or width) of 8B** — fragile, research-grade, lower ROI
  than distillation.
- **KV-cache NVFP4** — blocked upstream (see caveats above).
- **DPO / preference learning** — viable second stage after SFT lands.
  Pairing passing vs failing traces on the same prompt is the obvious
  starting point; deferred to a follow-up doc.
- **RL via cosmos_rl** — NVIDIA's RL recipe is available but the SFT path
  must land first to establish a quality floor.
- **Off-Thor training** (DGX Spark, rented H100) — explicit constraint:
  Thor-only.

---

## Status

- [ ] Stock-2B baseline against smoke corpus
- [ ] `Dockerfile.finetune` written and built
- [ ] Distillation script (`tools/distill_from_8b.py`)
- [ ] Pilot data generation (1K samples, filtered)
- [ ] Pilot SFT run on 2B
- [ ] Pilot smoke regression — decision point
- [ ] Production-quality data generation (2-3K samples)
- [ ] Production SFT run + KL reg + replay mix
- [ ] NVFP4 quantization
- [ ] New profile registered in `config.sh` / `launch.sh`
- [ ] Smoke regression on tuned + quantized model
- [ ] Production profile flip (if ≥ 9/9)

---

## References

- NVIDIA Cosmos-Reason2 repo: `github.com/nvidia-cosmos/cosmos-reason2`
- llm-compressor multimodal: `examples/multimodal_vision/qwen3_vl_example.py`
- Cosmos-Reason2 license: NVIDIA Open Model License
- Production default decision: see `VERSIONS.md` §C Phase 2 (2026-05-07 flip)
- Smoke corpus contract: `manyforge/scripts/debug/smoke_corpus.yaml`
- Bridge audit logs: `manyforge/openclaw_assistant_bridge/` mount point
- Lane parity analysis: [`manyforge/docs/LANE-COMPARISON.md`](../../manyforge/docs/LANE-COMPARISON.md)
- 32B quantization (related, larger-target reference): [`COSMOS-REASON2-32B-QUANTIZATION.md`](COSMOS-REASON2-32B-QUANTIZATION.md)
