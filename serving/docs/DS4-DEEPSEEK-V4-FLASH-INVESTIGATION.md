# DS4 / DeepSeek-V4-Flash on Jetson Thor — investigation notes

**Date**: 2026-05-12
**Status**: Pre-execution feasibility study; not yet built or benched on this Thor.
**Verdict**: **Buildable in a few hours, mostly download wait.** Single-stream
only — useful as a personal coding-agent or RLM sub-LM backend, **not** a
production serving stack and **not** a vLLM/TRT-Edge-LLM replacement.

## Why we care

DeepSeek-V4-Flash (284B total / 13B active MoE, 1M context) is one of the few
open frontier-class reasoning models that fits a 128 GB unified-memory budget
when paired with the right inference engine and quant. Mitko Vasilev (`mitkox`,
`@iotcoi`) demonstrated it running on a single DGX Spark (GB10, 128 GB unified)
at 1M context for personal coding-agent use, using **antirez/ds4** — a custom C+CUDA
inference engine purpose-built for the DeepSeek V4 architecture, descended from
antirez's earlier ds4-metal Mac engine.

Thor (SM_110a, 128 GB unified, aarch64, CUDA 13) is architecturally close
enough to GB10 (SM_121, 128 GB unified, aarch64, CUDA 13) that the same stack
should work with minimal changes. This document records what's needed to verify
that, and what the realistic positioning of the result would be.

This is a **personal-developer-experience experiment**, not a candidate for
ManyForge serving. ds4 has no continuous batching and no concurrent-request
support — it would not handle the multi-agent dispatch pattern OpenClaw +
ManyForge requires.

---

## The stack

### Engine: `antirez/ds4`

- Repo: <https://github.com/antirez/ds4>
- License: BSD-2-Clause
- Pure C + CUDA + Metal (Mac), no Python in the serving path, no PyTorch
- Single 9981-line `ds4_cuda.cu`, ~140-line Makefile
- Custom KV disk-cache compression (`--kv-disk-dir`), pages cold KV to disk
- Single-stream, greedy-only by design; no continuous batching

**Naming clarification.** The active community uses the path `mitkox/ds4-cuda`
informally because mitkox publicizes it on LinkedIn, but **the actual repo is
`antirez/ds4`** — there is no separate mitkox fork. Mitkox's "merged antirez's
ds4 with my stack" line refers to wiring ds4 into his coding-agent stack, not a
kernel fork.

### Model: DeepSeek-V4-Flash IQ2-imatrix custom quant

- Published checkpoint: <https://huggingface.co/antirez/deepseek-v4-gguf>
- File: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`
  (80.8 GiB)
- Suffix decoded:
  - `IQ2XXS` — imatrix 2-bit (~2.0 bpw) base quant for most weights
  - `w2Q2K` — 2-bit K-quants for some layers
  - `AProjQ8` — attention projections at Q8 (preserved precision)
  - `SExpQ8` — shared experts at Q8
  - `OutQ8` — output layer at Q8
  - `chat-v2` — chat-tuned variant
- Same philosophy as PrismaQuant (aggressive 2-bit base + selective Q8 retention
  for sensitive layers), implemented through GGUF tensor types
- README explicitly states: *"These quants are specific for the DS4 inference
  engine. They may work with other inference engines or not (they should, but
  not the MTP model which requires a specific loader)."* Stay on `ds4-server`
  for now; do not assume llama.cpp swap-in compatibility for the MTP variant

---

## Compatibility analysis

The blockers I expected (custom hand-written kernels with arch-specific
intrinsics, hardcoded SM versions, unpublished model file) are **not actually
present** in this codebase. Concrete evidence:

### CUDA source review

`ds4_cuda.cu` (9981 lines) was inspected for arch gates and Hopper/Blackwell-only
intrinsics:

- **Single `__CUDA_ARCH__` gate** in the entire file, at line 4432, requiring
  `__CUDA_ARCH__ >= 700` (Volta and up) — Thor SM 11.0 satisfies this trivially
- **No Hopper TMA** (no `cp.async.bulk.tensor`)
- **No Blackwell cluster API** (no `wgmma`, no `mma.m64n*` PTX asm)
- **No `__nv_fp8_e4m3` / `__nv_bfloat16` hardware types** —
  `fp8_kv_quantize_kernel` is software-emulated FP8 via a dequant lookup
  function operating on plain `float`
- Only intrinsics used: `__shfl_*_sync`, `__syncwarp`, plain `wmma::mma_sync`,
  one `atomicAdd`, `__dp4a` — all stable since Volta

### Build system

The Makefile's CUDA path (verbatim from <https://github.com/antirez/ds4/blob/main/Makefile>):
```
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= native
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS  ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
```

Notable:
- Default `CUDA_ARCH=native` lets `nvcc` auto-detect. On Thor with CUDA 13 this
  selects `sm_110a` automatically
- The link line **already references `$(CUDA_HOME)/targets/sbsa-linux/lib`** —
  antirez built his Spark binary against the same SBSA aarch64 library tree
  Thor uses. This is not accidental; the Spark is also aarch64
- No `gencode` list, no Hopper-only flags, no Blackwell `sm_120a` requirement

### Thor vs Spark axis comparison

| Axis | DGX Spark GB10 (mitkox) | Jetson AGX Thor (us) | Verdict |
|---|---|---|---|
| Arch family | Blackwell-class | Blackwell-class | Equivalent |
| Compute capability | sm_121 | **sm_110a** (CC 11.0) | Different SM, both supported by `nvcc` 13 |
| Memory | 128 GB unified LPDDR5X | 128 GB unified LPDDR5X | Equivalent |
| CPU ISA | aarch64 (Grace) | aarch64 (Neoverse) | Equivalent — Makefile sbsa-linux path works for both |
| CUDA toolkit | 13.0 (Spark default) | 13.0 (JP7.1 default) | Equivalent |
| `__dp4a` | yes | yes | OK |
| Volta `wmma` (CC ≥ 7.0) | yes | yes | OK |
| Hopper TMA / FP8 hw / Blackwell cluster | not used by ds4 | not used by ds4 | Non-issue |

The earlier Thor NIM/Triton issue (forum thread *"ptxas-blackwell does not
recognize sm_110a"*) affected NVIDIA's pre-built Triton ptxas in their NIM
containers, **not** `nvcc 13` on Thor itself. Building ds4 from source against
the Thor-native CUDA 13 toolkit avoids that class of bug entirely — ds4 is not
using Triton.

---

## Build procedure

```bash
# Prerequisites: JP7.1 Thor with cuda-toolkit-13-0, ~85 GB free disk
git clone https://github.com/antirez/ds4 && cd ds4

# Pull the published GGUF (80.8 GiB — biggest time sink, ~30-90 min on home links)
./download_model.sh q2-imatrix

# Build with explicit Thor arch (avoids native-detection ambiguity)
make CUDA_ARCH=sm_110a

# Serve
./ds4-server -m ds4flash.gguf --host 0.0.0.0 --port 8000 \
             --ctx 200000 \
             --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

### Likely failure modes and one-line fixes

| Symptom | Fix |
|---|---|
| `nvcc -arch=native` picked `sm_110` instead of `sm_110a` | Pass `CUDA_ARCH=sm_110a` explicitly (knob already exists) |
| `__dp4a is undefined` | Same as antirez Issue #75: explicit arch flag forces the right path. `make NVCCFLAGS="-O3 --use_fast_math -Xcompiler -pthread -arch=sm_110a"` |
| OOM at large context (KV growth) | Set `DS4_CUDA_NO_Q8_F16_CACHE=1` per Issue #78. Throughput penalty is small (post-fix on RTX 6000 96 GB: pp 64 t/s, gen 40 t/s, ctx 100k, ~92 GB used) |
| `cuBLAS` link error | Confirm `/usr/local/cuda-13.0/targets/sbsa-linux/lib/libcublas.so` exists. JP7.1 ships it |
| `CUDA host registration skipped: operation not supported` | Benign warning, antirez logs it on Spark too |

### Pre-flight checklist

Before kicking off the build:

1. Stop any active vLLM container so 128 GB memory pool is fully available
2. `sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`
3. `free -g` — verify ≥110 GB available
4. Confirm `/usr/local/cuda-13.0/bin/nvcc --version` returns CUDA 13.0
5. Confirm ~85 GB free disk for the GGUF download + build artifacts

---

## Memory budget

Confirmed today on this Thor: 122 GB total, 63 GB swap.

| Item | Estimate |
|---|---|
| GGUF model file (mmaped) | 80.8 GB |
| KV cache @ 200K context (paged + Q8 f16) | ~10–14 GB |
| KV disk cache spillover | 8 GB on `/tmp/ds4-kv` (configurable) |
| ds4-server runtime + buffers | ~2 GB |
| **Peak GPU-resident at 200K ctx** | **~95 GB** |
| **Peak at 1M ctx (mitkox's config)** | **~105 GB** |

We have 32 GB more headroom than the RTX 6000 Pro 96 GB user from antirez Issue
#78 — the `DS4_CUDA_NO_Q8_F16_CACHE=1` workaround is unlikely to be needed at
200K ctx, and is a fallback for 1M ctx if KV-disk paging doesn't keep up.

---

## Expected throughput

Antirez's reference numbers from <https://github.com/antirez/ds4/issues/34>
on a single DGX Spark (GB10) at q2:

- Prefill: **340 t/s**
- Generation: **14–16 t/s**

Thor's GPU is roughly **half a Spark on raw FP16 TFLOPS**, so plan for:

- Prefill: **~150–200 t/s** (rough estimate)
- Generation: **~7–10 t/s** (rough estimate)

This is **not** datacenter-class throughput. It is ballpark-equivalent to what
mitkox is actually getting on Spark — usable single-stream coding-agent
backend at 1M context on consumer-class Blackwell.

For comparison vs our current production stack:
- vLLM v8.1 with Qwen3.6-35B-A3B-NVFP4-MTP: **19.5 tok/s** (TEB 93 winner)
- vLLM v8.1 with Qwen3.6-35B-A3B-NVFP4-DFlash: **47.6 tok/s**
- TRT-Edge-LLM v0.7 with Nemotron Omni 30B-A3B: **23.22 tok/s** (single-stream)
- ds4 with DeepSeek V4 Flash 284B/13B-active: **~7–10 tok/s** (estimated, single-stream, much larger model)

ds4's value proposition is not throughput — it's **fitting a 284B-class model
with 1M context on a 128 GB box**.

---

## Where this fits in our stack

**Not a vLLM replacement. Not a ManyForge serving backend. Not a TRT-Edge-LLM
alternative.**

### Genuine fit

- **Personal coding-agent backend.** Solo developer running Claude-Code-style
  workflows against a local frontier-class model. Single user, single stream,
  long context — exactly mitkox's pattern
- **RLM sub-LM substrate.** If we adopt rlmgw for long-context audit lanes
  (per [feedback_agent_timeouts.md](../../../../.claude/projects/-home-tndlux-workspaces-nemoclaw/memory/feedback_agent_timeouts.md)),
  DeepSeek V4 Flash on ds4 would be a strong recursive-call backend (the RLM
  paper benchmarks against GPT-5-mini; DeepSeek V4 Flash is in the same league)
- **1M-context experimental lane.** Validation that Thor + custom-CUDA + 80 GB
  GGUF + 1M context all work together — useful prior art for any future
  custom-engine work
- **Existence proof for SM_110a + custom-CUDA + huge MoE.** Baseline for
  measuring whether other custom-engine candidates (Atlas, FlashRT, TokenSpeed)
  can match what a small open-source engine achieves on Thor

### Not a fit

| Workload | Why not |
|---|---|
| ManyForge multi-agent dispatch | ds4 is single-stream, no continuous batching, crashes on concurrent requests like TRT-Edge-LLM v0.7 (worse: it's by design here, not a bug fix away) |
| Smoke corpus / tool-eval-bench | Tool-call reliability is owned by parser correctness; ds4 has no qwen3_xml-class parser surface |
| Composer / OpenClaw production | Multi-agent + tool-call requirements rule it out |
| Vision tasks | DeepSeek V4 Flash is text-only; no Cosmos/Nemotron/Qwen-VL substitution |

---

## Caveats

1. **Single-stream by design.** Same architectural posture as TRT-Edge-LLM v0.7
   pre-acknowledgment (see [TRT-EDGE-LLM-NOTES.md](../docker/TRT-EDGE-LLM-NOTES.md)
   — likely to move to `serving/docs/TRT-EDGE-LLM-INVESTIGATION.md`). The
   difference: TRT-Edge-LLM has NVIDIA roadmap commitment to fix concurrency;
   ds4's single-stream-only nature is intentional and unlikely to change

2. **DeepSeek V4 architecture only.** Custom hand-rolled engine for one model
   family. No Qwen, no Cosmos, no Gemma, no Nemotron. Cannot grow to cover
   our serving lineup

3. **DSML-specific quant format.** Antirez's HF README explicitly warns the
   IQ2-imatrix mixed quant *"may work with other inference engines or not."*
   The MTP variant requires the ds4 loader. Stay on ds4-server; do not attempt
   to swap to llama.cpp without testing

4. **No tool-call parser surface.** ds4-server emits text. Tool-calling
   workloads would need a wrapper similar to the
   `tool_call_proxy.py` we built for TRT-Edge-LLM v0.7

5. **Replaces nothing in production.** Adds a new endpoint serving DeepSeek
   V4 Flash. Doesn't compete with our Qwen3.6 / Cosmos-Reason2 / Nemotron
   lineup. If we run it, it's alongside the v8.1 vLLM container, not instead
   of it. Memory budget needs to account for the swap (vLLM stopped first,
   `drop_caches`, then ds4-server)

6. **Solo project.** antirez maintains it directly; small contributor base
   (a Strix Halo HIP port PR, a Docker envelope PR). Not a community-blessed
   stack like vLLM or even TRT-Edge-LLM. Treat it as personal-tooling-grade

---

## Open questions for execution

If we decide to build this on Thor, things worth verifying experimentally:

1. **Actual generation throughput on Thor SM_110a vs Spark SM_121.** The "half
   the FP16 TFLOPS → half the gen tok/s" estimate is a rough scale-down; reality
   may differ ±30%. Need a measured number before claiming a baseline
2. **KV disk cache behavior on JP7.1.** Spark and Thor have different disk
   subsystems; the `--kv-disk-dir` cold-page eviction loop may stress NVMe
   differently. Watch for I/O contention during long contexts
3. **Whether DS4_CUDA_NO_Q8_F16_CACHE workaround is needed at 1M ctx on Thor.**
   We have 32 GB more headroom than the Issue #78 reporter, so probably not at
   200K ctx; possibly at 1M ctx. Bench at increasing context sizes
4. **Coexistence with vLLM container.** ds4 is not containerized in the upstream;
   we'd need to either accept a host-side install, build our own thin container,
   or wait for PR #86 (docker/compose envelope) to merge

These are not blockers — they're things to record on the first build attempt
so future agents can learn from the result.

---

## References

### Engine and model
- [antirez/ds4](https://github.com/antirez/ds4) — main repo
- [antirez/ds4 Makefile](https://github.com/antirez/ds4/blob/main/Makefile) — CUDA_ARCH knob, sbsa-linux link path
- [antirez/ds4 README](https://github.com/antirez/ds4/blob/main/README.md) — full build + serve docs
- [antirez/deepseek-v4-gguf on HF](https://huggingface.co/antirez/deepseek-v4-gguf) — published IQ2-imatrix and other quants

### Key issues (one-line fixes documented)
- [antirez/ds4#75 `__dp4a` undefined](https://github.com/antirez/ds4/issues/75) — RTX 6000 Pro Blackwell + CUDA 12.8; `CUDA_ARCH=native` Makefile fix; build log shows sbsa-linux already wired
- [antirez/ds4#78 `q8 fp16 cache alloc failed`](https://github.com/antirez/ds4/issues/78) — `DS4_CUDA_NO_Q8_F16_CACHE=1` workaround
- [antirez/ds4#34 CUDA work in progress](https://github.com/antirez/ds4/issues/34) — antirez's GB10 reference numbers (340 t/s pp / 14–16 t/s tg on q2)
- [antirez/ds4 PR #86 docker/compose envelope](https://github.com/antirez/ds4/pull/86) — pending

### Community context
- [Mitkox LinkedIn — DeepSeek V4 1M ctx on GB10](https://www.linkedin.com/posts/mitkox_i-just-removed-my-last-rag-full-rlm-now-activity-7459576886529282048-GgTJ) — original demonstration screenshot
- [Mitkox LinkedIn — "perfect devbox" follow-up](https://www.linkedin.com/posts/mitkox_the-perfect-devbox-doesnt-exi-eyeing-a-ugcPost-7459922701286240257-d1l9) — anchored on single GB10
- [NVIDIA Forum: ds4 on 1× Spark](https://forums.developer.nvidia.com/t/fully-custom-cuda-native-deepseek-4-flash-optimized-for-1x-spark-antirez-ds4/369791) — community thread, zero Thor mentions

### Thor toolchain
- [NVIDIA Jetson AGX Thor CUDA Setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_cuda.html) — CUDA 13 sm_110a confirmed
- [Ridgerun Thor / Blackwell GPU wiki](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_AGX_Thor/Blackwell_GPU)
- [JP7.0 Thor compute_110 forum thread](https://forums.developer.nvidia.com/t/jetpack-7-0-thor-cumotion-curobo-build-fails-nvcc-fatal-unsupported-gpu-architecture-compute-110/348679) — older toolchains can fail; native nvcc 13 on JP7.1 is fine
- [NIM Triton ptxas-blackwell sm_110a issue](https://forums.developer.nvidia.com/t/title-nim-qwen3-5-35b-a3b-1-7-0-variant-fails-on-jetson-agx-thor-triton-ptxas-blackwell-does-not-recognize-sm-110a/364963) — NIM-only, ds4 doesn't use Triton

### Related ManyForge-Thor docs
- [TRT-EDGE-LLM-NOTES.md](../docker/TRT-EDGE-LLM-NOTES.md) — analogous single-stream-on-Thor evaluation; pending move to `serving/docs/TRT-EDGE-LLM-INVESTIGATION.md`
- [MINIMAX-M27-INVESTIGATION.md](MINIMAX-M27-INVESTIGATION.md) — analogous "huge MoE on Thor" feasibility study
- [COSMOS-REASON2-32B-QUANTIZATION.md](COSMOS-REASON2-32B-QUANTIZATION.md) — similar pre-execution feasibility template
