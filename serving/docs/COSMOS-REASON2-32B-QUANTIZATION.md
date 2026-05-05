# Cosmos-Reason2-32B quantization on Jetson Thor — investigation notes

**Date**: 2026-04-30
**Status**: Pre-quantization research; no run executed yet.
**Goal**: Produce an NVFP4 (or NVFP4A16) quant of `nvidia/Cosmos-Reason2-32B`
that fits Thor's serving budget and preserves Physical AI reasoning quality.
**Why**: The 32B was released 2026-04-29 in BF16 only (~64 GB on disk). At that
size it cannot co-serve with Qwen3.6-MTP on Thor and cannot fit the 30–40 GB
Orin envelope. It also blocks `manyforge_assistant` adoption: the deployment
plan's Outcome A/B/C lineup ([MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md](MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md))
requires quantized perception specialists. NVIDIA published quants for the 2B
and 8B siblings (FP8 on NGC, mixed-precision W4A16 / NVFP4A16 from Embedl, NVFP4
from Firworks for the 8B); for the 32B we'd be the first.

This doc records the recipe options, calibration choices, gotchas, and a
feasibility check for running the quantization pipeline on Thor itself.

---

## Existing quants for the Cosmos-Reason2 family

| Model | Format | Source | Notes |
|---|---|---|---|
| 2-32B (target) | **None — BF16 only** | NVIDIA HF | Released 2026-04-29 |
| 2-8B | FP8 W8A8 + static FP8 KV | `nim/nvidia/cosmos-reason2-8b:1208-fp8-static-kv8` (NGC) | NVIDIA official |
| 2-8B | NVFP4 | `Firworks/Cosmos-Reason2-8B-nvfp4` | llm-compressor; calibrated on `Rombo-Org/Optimized_Reasoning` 256 samples × 4096 tokens |
| 2-2B | W4A16 mixed-precision | `embedl/Cosmos-Reason2-2B-W4A16-Edge2` | **Lossless on Physical AI Bench** (50.58 vs 50.60 BF16) |
| 2-2B | NVFP4A16 (Blackwell) | `embedl/Cosmos-Reason2-2B-NVFP4A16` | Beats W4A16 overall on Physical AI Bench |
| 2-2B | FP8 W8A8 + KV8 | NGC official | — |
| 1-7B | FP8 (script in repo) | `nvidia-cosmos/cosmos-reason1/scripts/quantize_fp8.py` | **NVIDIA's reference recipe** — llm-compressor + SmoothQuant + Flickr30k |

The directly transferable production blueprint is **`RedHatAI/Qwen3-VL-32B-Instruct-NVFP4`**:
Cosmos-Reason2-32B is a post-train of the same backbone (`Qwen3-VL-32B-Instruct`)
and inherits its layer topology, so the RedHat NVFP4 recipe should apply almost
verbatim with only the calibration dataset adjusted for the physical-AI domain.

---

## Recipe options ranked by risk/effort

| Path | Tool | Effort | Risk | Expected outcome |
|---|---|---|---|---|
| **A. Clone RedHat NVFP4 recipe** | `llm-compressor >= 0.9.0` | ~6h pipeline | Low — proven on same architecture | ~16 GB; ~97% MMLU recovery; **possible 2-pt physics drop** (see Embedl 2B precedent) |
| **B. NVFP4A16 mixed-precision** | `llm-compressor` + selective targets | ~10h pipeline | Medium — need to identify sensitive layers | ~17–19 GB; lossless on physics per Embedl |
| **C. PrismaQuant 4.75–5.5 bpp** | `RobTand/PrismQuant` | ~3h on DGX Spark; untested on Thor | Medium — methodology proven on Qwen3.6+ViT; never on Cosmos | ~18–20 GB; optimal Pareto |
| **D. ModelOpt NVFP4** | `nvidia-modelopt v0.35+` | ~6h pipeline | Medium — Qwen3-VL not in ModelOpt support matrix; precedent: `Ex0bit/Qwen3-VLTO-32B-Instruct-NVFP4` succeeded with v0.35.1 | Same outcome as Path A, different toolchain |

**Recommended**: Path A first. Lowest risk, directly transferable evidence,
fastest to validate. If Physical AI Bench shows >2-point drop vs BF16, fall back
to Path B (NVFP4A16) or Path C (PrismaQuant per-Linear allocation).

---

## Recipe template (Path A — RedHat blueprint, Cosmos-tuned)

`llm-compressor >= 0.9.0` is the version floor — earlier versions don't support
Qwen3-VL data-dependent calibration. The package is `py3-none-any.whl` (pure
Python, no aarch64 wheel issue).

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from datasets import load_dataset

model_id = "nvidia/Cosmos-Reason2-32B"

# NVIDIA's own choice for Cosmos-Reason1 FP8 quant; on-domain for VLM.
calib_ds = (
    load_dataset("lmms-lab/flickr30k", split="test")
    .shuffle(seed=42)
    .select(range(512))
)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",          # ViT — engineering reason, see gotcha #1
        "re:model.visual.*",    # ViT under alternate naming
        "re:.*mlp.gate$",       # MoE router gates (defensive; 32B is dense)
    ],
)

oneshot(
    model=model_id,
    dataset=calib_ds,
    recipe=recipe,
    max_seq_length=8192,
    num_calibration_samples=512,
    output_dir="cosmos-reason2-32b-nvfp4",
)
```

Then serve (drop into a new `cosmos-reason2-32b-nvfp4` profile in
[lib/launch.sh](lib/launch.sh)):

```bash
vllm serve cosmos-reason2-32b-nvfp4 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --kernel-config '{"enable_flashinfer_autotune": false}'
```

The `enable_flashinfer_autotune=false` flag is empirically required on the v7
image to dodge the cuDNN apt 9.21.1 / pip 9.20.0 sublib mismatch
([PERFORMANCE-V7.md](PERFORMANCE-V7.md) finding #3). v8 image should drop the
apt cuDNN, at which point this flag becomes optional.

---

## Critical gotchas

### 1. The vision encoder MUST be skipped uniformly

Not for quality reasons — research (MBQ, CVPR 2025) shows the ViT actually
tolerates W4A8 fine, sometimes *improving* benchmark scores. The reason is
**engineering**: ViT hidden dims are not divisible by NVFP4's group size of 16,
which crashes vLLM with:

> Unsupported model when in features size is not multiple of 16

This is documented as [cosmos-reason1 issue #85](https://github.com/nvidia-cosmos/cosmos-reason1/issues/85),
**unresolved upstream**. Every working public recipe (RedHat, Firworks, NVIDIA's
own `quantize_fp8.py`, Embedl) uses the same `re:visual.*` / `re:model.visual.*`
ignore pattern. Don't try to quantize the ViT — it fails the same way every
time.

### 2. Uniform 4-bit costs ~2 points on Physical AI reasoning

The only direct measurement comes from Embedl on the 2B sibling
([embedl.com/knowledge/cosmos-reason-2-without-the-quantization-trade-off](https://www.embedl.com/knowledge/cosmos-reason-2-without-the-quantization-trade-off)):

| Quant | Physical AI Bench Reason Task |
|---|---|
| BF16 baseline | 50.60 |
| Vanilla W4A16 | 48.68 (−1.92, ~3.8% relative drop) |
| **W4A16 mixed-precision (Edge2)** | **50.58 (−0.02, lossless)** |

If the Path A NVFP4 quant of the 32B shows a similar ~2-point drop on the
Reason Task (validation gate below), the fix is mixed-precision: Path B
(NVFP4A16, FP16 activations preserve more dynamic range) or Path C (PrismaQuant
per-Linear). Don't attempt to recover physics quality by tweaking the Path A
calibration set further — the issue is bit-width, not data.

### 3. Calibration data choice is a +5–10% lever

MBQ data: switching from text-only to multimodal calibration improves AWQ W3 by
**+2.1% MMMU and +10.3% SEED**. NVIDIA's own choice for Cosmos-Reason1 was
**Flickr30k**, not the cnn_dailymail default used by ModelOpt for text LLMs.

| Calibration dataset | Provenance | Size |
|---|---|---|
| `lmms-lab/flickr30k` | NVIDIA's own choice in `cosmos-reason1/scripts/quantize_fp8.py` | ~30K image-caption pairs |
| `Rombo-Org/Optimized_Reasoning` | Firworks's choice for Cosmos-Reason2-8B-NVFP4 | Long, reasoning-heavy text |
| `nvidia/Cosmos-Reason1-Benchmark` | NVIDIA's official physical-AI eval set | On-domain but novel as calibration source |
| COCO captions | MBQ paper recommendation | Multimodal baseline |
| `cnn_dailymail` / C4 | Default for text LLMs | **Avoid** — text-only, MBQ shows worst case for VLM |

Default to Flickr30k for the first attempt. If Path A produces a >2-point physics
drop, swap to a mix of Flickr30k + `nvidia/Cosmos-Reason1-Benchmark` samples on
the next pass before falling back to Path B.

### 4. AWQ via llm-compressor is broken on Qwen3-VL

vLLM doesn't recognize the `compressed-tensors` config format produced by
llm-compressor's AWQ path and won't route to the AWQ marlin kernel
([vllm#27991](https://github.com/vllm-project/vllm/issues/27991), closed
not-planned). Use NVFP4 or FP8 instead. Path A targets NVFP4 directly so this
isn't a blocker, but don't try to substitute AWQ "as a smaller alternative".

### 5. v7 image cuDNN packaging issue

Same gotcha that hit `nemotron3-nano-omni-30b-a3b-nvfp4` boot. Required
`--kernel-config '{"enable_flashinfer_autotune": false}'`. The 32B will hit
this too on the v7 image. v8 image (drops apt cuDNN, keeps only pip cuDNN
9.20.0) should resolve it.

---

## Validation gate

Before merging the new profile to `lib/config.sh`:

1. **Boot test** — vLLM serves the quantized model, accepts `/v1/chat/completions`
2. **Physical AI Bench Reason Task** — compare to BF16 baseline (50.60 if 2B
   pattern transfers; the 32B BF16 number from the model card is the actual
   target); fail if drop > 2 points
3. **TEB + IFEval** — same harness as [PERFORMANCE-V7.md](PERFORMANCE-V7.md);
   compare to `cosmos-reason2-8b` (TEB 81 / IFEval 84.9%); the 32B should match
   or beat
4. **Tool-call smoke test** — using `--reasoning-parser qwen3`, verify a small
   tool-call workload doesn't regress vs the 8B
5. **Co-serve test** — boot alongside `qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge`
   under split `gpu_memory_utilization` (Qwen 0.32, Cosmos-32B-NVFP4 0.30);
   verify both endpoints stable for ≥30 min under load

If all five pass, add as `cosmos-reason2-32b-nvfp4` profile in
[lib/config.sh](lib/config.sh) + [lib/launch.sh](lib/launch.sh) and amend the
deployment plan with a Thor-only "Outcome E" entry.

---

## Feasibility on Thor itself

**Verdict: feasible.** All three paths (A, B, C) fit Thor's 128 GB unified-memory
pool. The reason to consider running Paths B and C off-Thor is **software
maturity, not memory capacity** — see "Spark vs Thor" below.

### Memory budget

Confirmed today on this Thor: 122 GB total, 63 GB swap. With vLLM stopped and
the standard `sync && drop_caches` ([AGENTS.md § "End-of-session cleanup"](AGENTS.md#c--end-of-session-cleanup)),
~110 GB is available.

Quantization peak memory for 32B BF16:

| Item | Estimate |
|---|---|
| Model weights (BF16) | ~64 GB |
| Forward-pass activations (seq_len=8192, no grad checkpoint) | ~10–15 GB |
| llm-compressor calibration accumulators | ~5–10 GB |
| Python runtime + buffers | ~3–5 GB |
| **Peak (Path A NVFP4 oneshot)** | **~85–95 GB** |
| Output quantized model written to disk | +16 GB |

Comfortable inside Thor's 110 GB usable. Path B (NVFP4A16 with selective layer
keep-FP16) and Path C (PrismaQuant Fisher probe running multiple format
candidates per Linear) push peak toward 100–110 GB — viable on Thor's RAM alone,
and the 63 GB swap absorbs any transient spike on the host side (model
serialization, dataset materialization, Python interpreter overhead).

### Note on swap

Swap **does** help, but with one important constraint:

- **Host-side allocations** (Python objects, intermediate tensors that live in
  CPU RAM, dataset buffers, safetensors writes) — fully swappable; the 63 GB
  swap is a real safety net for these
- **CUDA-resident tensors** (model weights once `.to('cuda')` happens,
  per-layer activations during forward pass) — **cannot swap**. CUDA pins these
  in physical memory; if the GPU allocator can't satisfy a request, you get a
  CUDA OOM, not a swap-out

In practice for `llmcompressor` oneshot:
- The model is loaded layer-by-layer to GPU during the forward pass; idle
  layers stay in CPU RAM
- Activations and accumulators are GPU-resident during their working window
- Calibration samples are CPU-side until used

So swap helps with the CPU-resident model copy and Python overhead, not with
the calibration-time GPU peak. The 95 GB peak estimate above is mostly CUDA
allocations; treat the 110 GB DRAM usable as the real ceiling and swap as
insurance, not capacity.

### Spark vs Thor — same memory, different ecosystem

DGX Spark (GB10) has the **same 128 GB unified-memory pool** as Thor. It is not
a memory-capacity advantage. The reasons published recipes complete faster on
Spark are software, not hardware:

| Factor | Spark (SM121) | Thor (SM110a) |
|---|---|---|
| Unified memory | 128 GB | 128 GB (122 GB usable) |
| Blackwell-class | Yes (GB10) | Yes (Thor variant) |
| `llmcompressor` validation | Direct, cited in release notes | None — we'd be the first run |
| `PrismaQuant` validation | Repo benchmarks reproduced here | Untested |
| `transformers` Qwen3-VL kernel coverage | Mainstream NVIDIA test target | Edge case (vision encoder paths) |
| Stack maturity for VLM PTQ | RedHat / Firworks recipes published with Spark/H100 numbers | No published Cosmos-2 32B Thor recipe |
| PyTorch aarch64 vs x86_64 | x86_64 mainline | aarch64 with Jetson AI Lab patched wheels |

The Spark advantage is **fewer surprises**, not more memory. If a PrismaQuant
Fisher probe step OOMs or misbehaves on Thor, there's no community-debugged
workaround to fall back on; on Spark there is. For Path A (the simplest
pipeline) this gap is small. For Paths B and C it's the main reason to consider
running off-Thor.

### Software stack

| Component | Status on aarch64 Thor |
|---|---|
| `llmcompressor` 0.9.0+ | Pure-Python `py3-none-any.whl`; pip-installable |
| `compressed-tensors` | Pure Python; pip-installable |
| `transformers >= 4.52` | aarch64 wheel exists |
| `torch` with CUDA 13 | Available via Jetson AI Lab index (`pypi.jetson-ai-lab.io/sbsa/cu130`) |
| `datasets` | Pure Python |
| `lmms-lab/flickr30k` download | ~2 GB; HF token already configured per [AGENTS.md § Workflows A.3](AGENTS.md) |

No fundamental aarch64/cu130 blocker. The ecosystem is one `pip install
llmcompressor` + a Flickr30k download away.

### Time estimate

| Phase | Spark (reference) | Thor (estimate) |
|---|---|---|
| Model download (BF16 ~64 GB) | ~15 min | ~15 min (same gigabit link) |
| Calibration data download | ~2 min | ~2 min |
| Calibration forward passes (512 samples × 8192 tokens) | ~30–60 min | ~45–90 min (similar Blackwell perf) |
| NVFP4 weight quantization | ~10 min | ~15 min |
| Checkpoint export + safetensors write | ~5 min | ~10 min |
| Validation (boot + Physical AI Bench Reason Task) | ~30 min | ~45 min |
| **Total Path A** | **~1.5 h** | **~2.5–3.5 h** |

### Pre-flight checklist

Before kicking off a Thor quantization run:

1. Stop any active vLLM container: `docker stop <name>`
2. `sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`
3. `free -g` — verify ≥110 GB available
4. Confirm HF token at `~/.cache/huggingface/token`
5. Pre-pull the BF16 weights: `hf download nvidia/Cosmos-Reason2-32B`
6. Pre-pull the calibration set: `hf download lmms-lab/flickr30k`
7. Create a Python venv outside the vLLM container; install `llmcompressor>=0.9.0`,
   matching `transformers`, `datasets`, and the Jetson AI Lab `torch+cu130`
8. Disable cron jobs / NemoClaw onboarding wizards that might launch background
   Python processes during the run

### When to do it elsewhere instead

The reason is software-stack maturity, not memory. Consider Spark or H100 cloud
if any of these apply:

- Targeting **Path C (PrismaQuant)** — Fisher probe + multi-format error
  measurement is documented and reproducible on Spark; never validated on Thor.
  A failed run there is debuggable from forum threads; a failed run on Thor is
  on us alone
- Need to **iterate quickly across recipes or calibration sets** — Spark's
  faster turnaround per run (~1.5h vs ~3h) compounds when you're trying 4–5
  variants
- Want the artifact **published to HF for reproducibility** — Spark's
  x86_64 / mainline PyTorch path produces wheels other people can reuse without
  the Jetson AI Lab patches
- Hitting unfamiliar `transformers` / `llmcompressor` errors specific to the
  aarch64 Qwen3-VL vision encoder path

For Path A (single NVFP4 oneshot aimed at our own deployment), Thor is fine —
the recipe is short, well-tested on the same backbone elsewhere, and the
Thor-specific risk is just the cuDNN / FlashInfer flag we already work around
on the serving side.

---

## What this doc does not cover

- **PrismaQuant on Thor specifically** — methodology is documented at
  `github.com/RobTand/prismaquant`; it has never been run on SM110a, and the
  per-Linear Fisher probe is more fragile than oneshot NVFP4. If we go Path C,
  do it on DGX Spark first.
- **TurboQuant K8V4 KV stacking with Cosmos-Reason2-32B-NVFP4** — TQ is
  orthogonal to weight quantization and would multiply concurrent-request
  capacity; out of scope until the Path A artifact exists.
- **MTP for Cosmos-Reason2-32B** — Qwen3-VL backbones don't ship an MTP head;
  no speculative-decoding gain available without training a drafter.
- **Audio support** — Cosmos-Reason2 is vision+text only; for audio the
  candidate is `nemotron3-nano-omni-30b-a3b-nvfp4`
  ([MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md](MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md)
  Outcome D), not Cosmos.

---

## References

- [nvidia/Cosmos-Reason2-32B](https://huggingface.co/nvidia/Cosmos-Reason2-32B) — model card
- [RedHatAI/Qwen3-VL-32B-Instruct-NVFP4](https://huggingface.co/RedHatAI/Qwen3-VL-32B-Instruct-NVFP4) — recipe blueprint
- [Firworks/Cosmos-Reason2-8B-nvfp4](https://huggingface.co/Firworks/Cosmos-Reason2-8B-nvfp4) — Cosmos-family recipe reference
- [embedl/Cosmos-Reason2-2B-W4A16-Edge2](https://huggingface.co/embedl/Cosmos-Reason2-2B-W4A16-Edge2) — physics-quality measurement
- [nvidia-cosmos/cosmos-reason1 quantize_fp8.py](https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/scripts/quantize_fp8.py) — NVIDIA's own VLM quant script
- [llm-compressor 0.9.0 release notes](https://developers.redhat.com/articles/2026/01/16/llm-compressor-090-attention-quantization-mxfp4-support-and-more) — Qwen3-VL data-dependent calibration support
- [MBQ paper (CVPR 2025)](https://arxiv.org/html/2412.19509v1) — modality sensitivity, calibration data
- [cosmos-reason1#85](https://github.com/nvidia-cosmos/cosmos-reason1/issues/85) — ViT NVFP4 group-of-16 incompatibility
- [Embedl — Cosmos Reason 2 Without the Quantization Trade-Off](https://www.embedl.com/knowledge/cosmos-reason-2-without-the-quantization-trade-off) — physics-bench data
