# Thor accelerator map (T5000)

**Status:** Reference (live)
**Written:** 2026-11-07
**Target:** Jetson AGX Thor Developer Kit = **T5000**, `sm_110a`, 128 GB unified
memory
**Purpose:** name every compute engine on the module, state what each is for,
and give the SM/TPC and per-SM arithmetic that decides which of them can
actually move serving throughput. This is a fact sheet plus derivations — the
work list that consumes it is
[`THOR-ACCELERATOR-ROI.md`](THOR-ACCELERATOR-ROI.md).

Sources: NVIDIA's Jetson Thor T5000/T4000 specification table; the NVIDIA PVA
SDK architecture guide; local CUDA device attributes recorded in
[`DS4-THOR-OPTIMIZATION-PLAN.md`](DS4-THOR-OPTIMIZATION-PLAN.md); the
`sm_110a` ISA audit in [`V9.1-TASK4-FP4-UNLOCK.md`](V9.1-TASK4-FP4-UNLOCK.md).
Anything labelled **(derived)** is arithmetic on those numbers, not a
measurement.

---

## 1. Compute engines on the module

| Engine | T5000 configuration | Designed for |
|---|---|---|
| GPU | 2 560 CUDA cores, **20 SMs / 10 TPCs**, 1.57 GHz max, `sm_110a` | Dense linear algebra, attention, everything tensor-shaped |
| 5th-gen Tensor Cores | One `tcgen05` group per SM + tensor memory (TMEM) | FP4 (block-scaled NVFP4/MXFP4), FP8 E4M3/E5M2, INT8, 2:4 structured sparsity |
| CPU | 14× Arm Neoverse-V3AE, 1 MB L2/core, 16 MB shared L3, 2.6 GHz | Control loops, sensor fusion, and the whole CPU side of serving (tokenizer, sampler, scheduler, weight load) |
| PVA v3 | 1 instance (T4000 shares the row) | Programmable VLIW SIMD DSP for classic CV / sensor pre- and post-processing |
| Video encode | 2× NVENCODE | Hardware H.264/HEVC/AV1 encode |
| Video decode | 2× NVDECODE | Hardware decode, many concurrent streams |
| Camera front end | 16-lane MIPI CSI-2 (6 cameras, 32 via virtual channels, C-PHY 2.1 / D-PHY 2.1), plus Holoscan Sensor Bridge over Ethernet | Getting high-rate frames into memory without CPU copies |
| Networking | 4× 25 GbE (devkit exposes 1× QSFP28 + 1× 5 GbE RJ45) | Sensor ingest, multi-node linkage |
| PCIe | Gen5, up to x8 | Module I/O; NVMe |
| MIG | Up to 10 TPCs partitionable | Co-tenancy / QoS isolation |

### Present on AGX Orin and **absent** on Thor

- **DLA.** Orin's table lists 2× NVDLA v2 alongside PVA v2; the Thor table
  lists one PVA v3 and **no DLA**. On Thor the DNN accelerator *is* the GPU
  tensor core path (TensorRT/vLLM/ggml). There is no second DNN engine to
  reclaim, and no `--use-dla`-style offload lane to design around.
- Consequence: any "offload inference to a dedicated accelerator" idea on Thor
  reduces to "use the tensor cores properly".

---

## 2. GPU internal counts

| Attribute | Value | Note |
|---|---:|---|
| Compute capability | 11.0 (`sm_110a`) | Family-specific build required for `tcgen05`/FP4 |
| TPCs | 10 | 2 SMs per TPC |
| SMs | 20 | 2 560 CUDA cores ÷ 128 cores/SM |
| CUDA cores per SM | 128 | 4 partitions × 32 |
| Threads per SM | 1 536 | |
| Registers per SM | 65 536 | |
| Shared memory per SM | 233 472 B | 232 448 B opt-in per block |
| Cooperative launch | supported | |
| Thread-block clusters | supported | Needed for 2-CTA `tcgen05` pairs |

DGX Spark's GB10 reports **48 SMs / 6 144 CUDA cores** (`sm_121`). That is the
single most load-bearing hardware fact when comparing the two boxes: Thor has
~2.4× fewer shader cores.

---

## 3. Tensor core generation

`sm_110a` carries the **full `tcgen05` ISA** — `alloc`/`dealloc`/`ld`/`st`/
`mma`/`mma.ws`/`cp`/`shift`/`commit` — with TMEM, i.e. the same instruction
set as datacenter `sm_100a`. The recorded evidence
([`V9.1-TASK4-FP4-UNLOCK.md`](V9.1-TASK4-FP4-UNLOCK.md)) is:

- PTXAS analysis plus NVIDIA staff confirmation that the ISA is complete,
- a supplementary scheduling table of only 618 bytes versus `sm_100`'s full
  table → "minimal scheduling parameter changes from `sm_100`",
- community-measured **870+ TFLOPS dense FP4** on Thor hardware.

Practical implications:

1. Thor's tensor cores are **not** consumer-style SMs with a bigger pile of
   shader cores. Per-SM FP4 rate is datacenter-class (§4).
2. `mma.ws` (weight-stationary MMA) and 2-CTA pairs are available and, as of
   this writing, unused anywhere in this repo.
3. Almost every FP4 kernel in the stack is gated by a **software allowlist**,
   not by missing silicon — see §8.

### 3.1 What the tensor core will and will not execute

The block-scaled FP4 MMA modes take **FP4 in both operands**. There is no
native "FP4 weights × BF16/FP8 activations" mode, so mixed-precision serving is
necessarily coarse-grained (chosen per matrix) rather than within a single GEMM:

| Weights × activations | Native `tcgen05` mode | What actually runs | Observed on Thor |
|---|---|---|---|
| FP16/BF16 × FP16/BF16 | `f16` | standard MMA | reference path |
| FP8 E4M3 block-scaled × FP8 | `mxf8f6f4` + scales | `CutlassFp8BlockScaledMMKernel` | works (Flash-Next dense side layers) |
| FP4 E2M1 × FP4 E2M1, NVFP4 scales (16-elem block, E4M3 scale) | `mxf4nvf4` | CUTLASS / FlashInfer `scaled_fp4_mm` | **18.9–22 tok/s** |
| FP4 E2M1 × FP4 E2M1, MXFP4 scales (32-elem block, E8M0 scale) | `mxf4` | same rate as NVFP4 | — |
| **FP4 × FP16/BF16 (W4A16)** | **none** | **Marlin**: dequant → FP16 MMA | **8.3 tok/s** |
| FP4 × FP8 (W4A8) | none | dequant/emulation | — |
| INT8 × INT8 | `i8` | unused by these profiles | — |
| 2:4 sparse variants | 2× on any kind | unused | — |

Two consequences:

- **NVFP4 vs MXFP4 is purely an accuracy choice.** Same hardware rate; NVFP4's
  16-element blocks with an E4M3 scale beat MXFP4's 32-element blocks with a
  power-of-two E8M0 scale.
- **A W4A16 checkpoint pays a 2.3× penalty** versus a true W4A4 one on this
  hardware (8.3 vs 18.9–22 tok/s). The repo's decision to force `marlin` for
  NVIDIA W4A16 weights is correct, but such a profile is not exercising the
  FP4 tensor core at all.

### 3.2 The decode path is not only tensor cores

One large component of Flash-Next never reaches an MMA unit: a 47.7 GiB
per-layer embedding table, mapped as raw `uint8` rows from NVMe and gathered per
token. Measured gather cost after `MADV_RANDOM` + parallel reads is 2.06 ms per
64-row gather and 414 ms per 32 768-row gather (see
[`QWEN38-FLASH-NEXT-FAST-THOR.md`](QWEN38-FLASH-NEXT-FAST-THOR.md)). Any
"which engine is underused" analysis that stops at the SMs misses this path —
its cost is page faults and page-cache pressure, not FLOPs, and its width is set
by the *dtype of the table*, not by the kernel.

---

## 4. Per-SM FP4 arithmetic (derived)

T5000 quotes **2 070 TFLOPS FP4 with sparsity** at the 1.57 GHz max clock over
20 SMs.

```
dense FP4          = 2 070 / 2                         = 1 035 TFLOPS
per SM (dense)     = 1 035 / 20                        =   51.75 TFLOPS
per SM per clock   = 51.75e12 / 1.57e9                 =  32 962 FLOP/cycle
MACs per SM/clock  = 32 962 / 2                        ≈  16 384 MAC/cycle (2^14)
```

Cross-check against datacenter Blackwell (B200 as reference: ~9 PFLOPS dense
FP4, ~148 SMs, ~1.97 GHz boost):

```
9e15 / (148 × 1.97e9) = 30 900 FLOP/cycle ≈ 15 450 MAC/cycle/SM ≈ 2^14
```

Same per-SM rate. **Thor's tensor core is an SM100-class unit, 20 of them.**

### Thor vs Spark, per SM (derived)

| Basis for Spark's "~1 PFLOPS FP4" | Spark dense FP4 / SM | Thor dense FP4 / SM | Ratio |
|---|---:|---:|---:|
| quoted with sparsity (÷2) | ≈ 10.4 TFLOPS | 51.75 TFLOPS | **≈ 5.0×** |
| quoted dense | ≈ 20.8 TFLOPS | 51.75 TFLOPS | **≈ 2.5×** |

Either way Thor wins **per SM** by 2.5–5× while losing ~2.4× on SM count. That
is why the module can advertise 2 070 TFLOPS FP4 against Spark's ~1 000 on
one-fifth the die area: it is a different tensor core class, not a bigger one.

### Vector (CUDA core) arithmetic — the other side of the coin

```
Thor  FP32 FMA: 2 560 cores × 2 FLOP × 1.57 GHz ≈  8.0 TFLOPS
Spark FP32 FMA: 6 144 cores × 2 FLOP × ~1.3 GHz ≈ 16.0 TFLOPS
```

**~2× the vector throughput on Spark.** Every kernel that dequantizes into
INT8/FP16 fragments (the DS4 GGUF path, Marlin weight-only FP4) runs on this
side of the ledger — which is exactly the measured gap in
[`DS4-DEEPSEEK-V4-FLASH-INVESTIGATION.md`](DS4-DEEPSEEK-V4-FLASH-INVESTIGATION.md)
(Thor at 31–36 % of Spark prefill, 51–55 % of generation, identical 273 GB/s).

---

## 5. Memory roofline (derived, from measured bandwidth)

Measured on this module (`bench/bw_bench.cu`, `sm_110`): **241–245 GB/s read**,
**227–230 GB/s copy**, against 273 GB/s theoretical. That is 88–90 % of peak —
the DRAM pipe is not where the remaining performance is hiding.

Single-stream decode:

```
tok/s ≈ 245 GB/s ÷ (bytes read per verified pass ÷ accepted tokens per pass)
     ≈ 245 GB/s ÷ (W / k)
```

Using the Flash-Next FP8-hybrid result (34.8 → 46.7 tok/s for 5.42 GiB of BF16
becoming 2.71 GiB of FP8, i.e. −2.71 GB of read per pass) the two data points
solve to:

- **k ≈ 1.5 output tokens per pass**
- **W ≈ 10.6 GB read per pass**

Sanity note: with MTP=3, `k ≈ 1.5` means only ~0.5 drafts accepted per pass
beyond the mandatory bonus token (~17 % per-draft acceptance) — or,
equivalently, that sustained bandwidth sits below the 245 GB/s benchmark. Either
way it flags draft acceptance as a second lever worth instrumenting, alongside
quantization (see `THOR-ACCELERATOR-ROI.md` §7).

Treat those as order-of-magnitude constraints, not measurements — they assume
pure bandwidth-bound decode at sustained 245 GB/s. Their value is directional:

| Target C1 rate | Required bytes per output token | Required W at k≈1.5 |
|---:|---:|---:|
| 46.7 tok/s (today) | 5.25 GB | ≈ 10.6 GB |
| 60 tok/s | 4.08 GB | ≈ 6.1 GB |
| 90 tok/s | 2.72 GB | ≈ 4.1 GB |

Nothing except **fewer bytes per token** or **more accepted tokens per pass**
moves the first column. This single line is the reason the accelerator triage
in §6 comes out the way it does.

---

## 6. Can each engine raise token throughput?

| Engine | Verdict | Reasoning |
|---|---|---|
| 5th-gen Tensor Cores (`tcgen05`) | **Yes — the main lever** | Reduces bytes read per pass when weights move to FP4 and the kernel uses the native path. The repo already proved the mechanism with FP8 (BF16→FP8 = +34 %) |
| FP4 KV cache | **Yes** | More sequences/context per GiB; also lowers KV read traffic. Upstream-blocked: no `sm_110a` cubins (see ROADMAP, "NVFP4 KV cache on SM110a") |
| `mma.ws` / 2-CTA `tcgen05` clusters | Only when **compute**-bound | At batch 1 the limit is DRAM, not MMA issue rate. Real for prefill/TTFT and high batch |
| 2:4 structured sparsity | Only when compute-bound | 2× tensor throughput; dense weights give nothing; needs pruning-aware finetune |
| 14× Neoverse-V3AE | **Indirectly** | Frontend/sampler/tokenizer placement, n-gram drafting. Cannot do transformer math at 273 GB/s shared bandwidth |
| CUDA graphs (not `--enforce-eager`) | **Yes, 10–20 %** | Documented cost in `manyforge/docs/COMPOSER-ASSISTANT-ARCHITECTURE.md`; still on for both `cosmos-reason2-*` profiles |
| MIG (10 TPCs) | **No** | Partitions SMs, **not** LPDDR5X bandwidth — slices contend on the same 273 GB/s. QoS/co-tenancy tool only, and carving 20 SMs hurts |
| PVA v3 | **No** for LLM math | 12-bit fixed point, ~32 px/cycle/VPU. Cannot run matmuls and is far too slow for a ViT patch embed. Usable for image pre/post-processing and radar/lidar, and `cuPVA` submits to **CUDA streams** so it can overlap with decode |
| 2× NVENC / 2× NVDEC | **No** | Video ingest only (VSS). Zero effect on tokens |
| DLA | n/a | Not present on Thor |

---

## 7. What PVA v3 actually is

From the NVIDIA PVA SDK architecture guide:

- A **VLIW SIMD DSP** optimised for *predictable* run-times at low power and
  low latency, aimed at semi-dense/dense regular computation even on small
  data sets.
- Two **Vector Processing Subsystems** (each: 7-way VLIW VPU with 2 scalar ALU,
  2 vector ALU, 3 load/store units; VMEM; a decoupled look-up-table unit
  (DLUT); a dedicated DMA engine) plus a **Cortex-R5 control subsystem** that
  schedules work and is not user-programmable.
- Each VPU SIMD instruction processes **32 pixels at 12 bits per pixel**.
- `cuPVA` operates directly on CUDA device pointers and PVA programs can be
  submitted to CUDA streams, so PVA work can be interleaved with GPU work
  without extra copies.
- Intended domains: image processing, computer vision, radar/lidar
  pre-processing, feature extraction and object localisation ahead of a
  tracker.

What that means on Thor: PVA is a **bandwidth-and-latency offload for the
sensor front end**, not an inference engine. It can strip resize/normalize/
undistort/pyramid/optical-flow work off the SMs. It is worth measuring only if
(a) the served model is a VLM and image preprocessing is a visible share of
TTFT, or (b) the same box must keep SMs free for a real-time perception lane.

Nothing in this repo currently references PVA, NVENC, NVDEC, MIG or CPU
pinning — all four are genuinely unused today.

---

## 8. Software gates that look like hardware limits

Ordered by how often they get mistaken for silicon ceilings:

| Symptom | Real cause | Reference |
|---|---|---|
| `ValueError: ... requires SM100 family (Blackwell: SM100, SM103). Got SM110.` | Kernel-level allowlist in CUTLASS DSL / FlashInfer | [`V9.1-TASK4-FP4-UNLOCK.md`](V9.1-TASK4-FP4-UNLOCK.md) |
| `No supported CUDA architectures found for major versions [10, 12]` | TRTLLM precompiled cubins ship for `sm_100`/`sm_120` only, no `sm_110` | same |
| NVFP4 MoE silently falls back to Marlin (software FP4) on SM110 | Upstream oracle gating; 7 in-place patches unlock `FLASHINFER_CUTLASS` | same, [`PUBLIC-IMAGE-MIGRATION-PLAN.md`](PUBLIC-IMAGE-MIGRATION-PLAN.md) |
| No native FP4 KV cache | `sm_110a` cubins absent in NVIDIA's artifactory for vLLM's FP4-KV path | `ROADMAP.md` |
| FA2/FA4 refuses `head_dim=256` on SM110 | Backend support, not a `tcgen05` tile limit (the earlier tile-shape claim was retracted) | [`DFLASH-INVESTIGATION.md`](DFLASH-INVESTIGATION.md) |

Rule of thumb: on this module, **assume a missing FP4/FP8 path is a
software gate until proven otherwise** — the ISA audit says the instructions
are there.

---

## 9. One-line summary per engine

- **Tensor cores** — the only engine that changes the decode roofline; reach it
  by moving bytes to FP4 and by un-gating kernels.
- **CUDA cores** — where Thor loses to Spark by ~2×; avoid paths that dequantize
  into INT8/FP16 fragments.
- **CPU** — 14 Neoverse-V3AE cores are idle overhead-reducers, not FLOPs.
- **PVA v3** — sensor/CV offload with stream-level overlap; never matmul.
- **NVENC/NVDEC** — video-in only.
- **MIG** — isolation, not throughput.
- **DLA** — absent; do not design for it.
