---
base_model:
- RadixArk/Qwen3.8-Flash-Next-NVFP4
license: other
library_name: vllm
pipeline_tag: image-text-to-text
tags:
- qwen3.8
- jetson-thor
- nvfp4
- fp8
- quantized
- delta-checkpoint
---

# Qwen3.8 Flash-Next NVFP4 + FP8 dense-side-layer delta

This is a **delta, not a standalone model checkpoint**. It contains four
converted safetensors shards and an updated index for the exact base snapshot:

- Model: [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
- Required base revision: `7b719225242aacd3dbd3f9407468c2ee9a9d2594`
- Original model: [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

Unchanged files must be obtained from that exact RadixArk base revision. The
delta cannot be loaded by itself.

## What changed

The conversion rewrites 300 frequently read dense matrices from BF16 to
blockwise FP8 E4M3 with 128x128 blocks and FP32 inverse scales:

- GDN input/output projections;
- QSA query/key/value/output projections;
- shared-expert gate/up/down projections.

Those weights occupy 5.42 GiB in BF16 and 2.71 GiB after conversion.

The following remain unchanged:

- routed experts: ModelOpt NVFP4;
- PLE table: the base checkpoint's FP8 representation, served through mmap;
- vision tower: BF16;
- embedded MTP layer and target vocabulary;
- all other excluded tensors.

KV cache is not stored in model weights. The recommended Thor runtime keeps it
in **BF16** (`--kv-cache-dtype auto`); this delta does not use FP8 KV cache.

## Jetson AGX Thor result

Measured on Jetson AGX Thor (SM110a) with the NemoClaw-Thor `sm110-fast-v3`
image, MTP=3, BF16 KV, 262144 context, three default scheduler slots and 0.92
memory. The recorded C4 benchmark used an explicit four-slot override:

| Measurement | Original side weights | FP8 hybrid |
|---|---:|---:|
| `sgbench` C1 token-weighted output rate | 34.8 tok/s | **46.7 tok/s** |
| Loaded tensor memory | 79.42 GiB | **76.75 GiB** |
| Advertised BF16-KV tokens | 728869 at 0.90 | **842439 at 0.92** |
| Full 262144-token concurrency estimate | 2.78x at 0.90 | **3.21x at 0.92** |

`sgbench` includes TTFT and prefill; it is not an isolated decode-kernel test.

The same complete 88-scenario Tool Eval was run on both variants:

| Tool Eval | Original side weights | FP8 hybrid |
|---|---:|---:|
| Overall | **94/100** | **94/100** |
| Points | 165/176 | **166/176** |
| Hard Mode | 95% | **100%** |
| Passed / partial / failed | 79 / 7 / 2 | **80 / 6 / 2** |
| Median turn | 4.3 s | **4.1 s** |
| Responsiveness | 37/100 | **39/100** |
| Deployability | 77/100 | **78/100** |
| Runtime | 1471.4 s | **1299.6 s** |

The aggregate score therefore showed quality parity in this evaluation, with
better responsiveness. Category results were not identical: the hybrid reached
100% Error Recovery and Hard Mode, while Safety & Boundaries moved from 92% to
85%. This is empirical evidence from one complete run per variant, not a claim
of bitwise equivalence.

## Assemble and serve on Thor

Use the public
[`pastoriomarco/NemoClaw-Thor`](https://github.com/pastoriomarco/NemoClaw-Thor)
repository. The complete pinned image build, delta assembly, validation,
fallback and foreground-serving instructions are in the
[`QWEN38-FLASH-NEXT-FAST-THOR.md`](https://github.com/pastoriomarco/NemoClaw-Thor/blob/main/serving/docs/QWEN38-FLASH-NEXT-FAST-THOR.md)
runbook.

The short path after cloning NemoClaw-Thor is:

```bash
./serving/build-qwen38-flash-next-fast.sh
./serving/download-qwen38-flash-next.sh
./serving/apply-qwen38-flash-next-fp8-delta.sh
./serving/start-qwen38-flash-next-fast.sh
```

The apply helper defaults to this repository at immutable weights revision
`358162ffaaeb4372c79843d9cd1f67d3654593da`. It checks the delta manifest,
SHA256 of every changed file, and exact base revision before creating the
linked local snapshot.

## Conversion and provenance

The Docker-only preparation command and auditable converter are included in
NemoClaw-Thor:

```bash
./serving/prepare-qwen38-flash-next-fp8-hybrid.sh
```

The conversion requires no calibration corpus. It directly calculates a scale
for every 128x128 weight block and retains the base snapshot unchanged. The
converter is derived from
[`Saren-Arterius/qwen3.8-Flash-DGX-AutoRound`](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)
and the fast-path integration builds on
[`blazux/qwen3.8-Flash-DGX`](https://github.com/blazux/qwen3.8-Flash-DGX).

## License

This repository carries only derived weight files and metadata. The source
model and RadixArk checkpoint license and terms continue to apply. Review those
terms before redistribution or commercial use.
