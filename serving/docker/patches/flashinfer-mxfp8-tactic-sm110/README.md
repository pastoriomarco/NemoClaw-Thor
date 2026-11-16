# FlashInfer CUTLASS MXFP8 tactic pin on sm_110 (Jetson AGX Thor)

Read-only one-file overlay for the official `vllm/vllm-openai@sha256:d825baf9…` image (vLLM 0.33.0, FlashInfer
0.7.2.post1). It lets the FlashInfer CUTLASS MXFP8 linear kernel (`FlashInferCutlassMxfp8LinearKernel`, used by R8b
for the 192 MXFP8 side matrices of `Qwen3.8-Flash-Next-FP4-Local`) run a chosen CUTLASS tactic for small M instead of
tactic 0. Plan item C3 in [`../../../docs/FLASH-NEXT-SPEED-PLAN.md`](../../../docs/FLASH-NEXT-SPEED-PLAN.md);
benchmark and raw results in [`../../../benchmarks/mxfp8-tactic-sm110/`](../../../benchmarks/mxfp8-tactic-sm110/).

Why: with `--disable-flashinfer-autotune`, `mm_mxfp8(backend="cutlass")` runs tactic -1 → 0, which is config
index 22 of `CutlassMxfp8GemmRunner::getConfigs()`: tile 128×128×256 with cluster 4×2×1. At decode M (1–16) the 4×2
cluster leaves Thor's DRAM bandwidth unused; tactic 8 (tile 128×64×128, cluster 2×2×1) streams the weights at
~250 GB/s on the large matrices.

## Change

`vllm/model_executor/kernels/linear/mxfp8/flashinfer.py` = the image's file plus two blocks marked `[thor]`:

- import-time parse of `VLLM_MXFP8_TACTIC` and, only when set, a custom op `vllm::thor_mm_mxfp8_tactic` that runs
  FlashInfer's own CUTLASS MXFP8 runner (`get_cutlass_mxfp8_gemm_module(major).cutlass_mxfp8_gemm_runner()`, same
  workspace buffer) with the pinned tactic for M within a bucket, and the stock `flashinfer.mm_mxfp8(...,
  backend="cutlass")` call for larger M;
- `FlashInferCutlassMxfp8LinearKernel.apply_weights` calls that op when the switch is set, the stock op otherwise.

Switch format: `VLLM_MXFP8_TACTIC="<max_m>:<tactic>[,<max_m>:<tactic>...]"`, buckets checked in ascending `max_m`;
M above the last bucket takes the stock path. Unset or empty = image behaviour (op not even registered).
Recommended: `VLLM_MXFP8_TACTIC=256:8`. It applies to every FlashInfer CUTLASS MXFP8 linear in the process; for
FP4-Local those are exactly the 192 side matrices (the MTP layer's MXFP8-eligible parts are BF16 in the checkpoint).

## Use

```
-v $O/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py:ro
-e VLLM_MXFP8_TACTIC=256:8
```

(`$O` = this folder.) Keep `--disable-flashinfer-autotune`. Expected log line (from the process that loads the
model): `[thor] MXFP8 CUTLASS tactic pins (VLLM_MXFP8_TACTIC): M<=256 -> tactic 8; larger M stock`.

## Measured (Thor, GPU-only micro-benchmark, weights streamed from DRAM, CUDA graphs)

Sum over the 192 matrices per forward (µs): default tactic 0 vs tactic 8.

| M | default | tactic 8 | saving | best per shape |
|---:|---:|---:|---:|---:|
| 4 (1 seq, MTP 3 verify) | 14014 | 12209 | −1805 | 12153 |
| 8 (2 seq) | 12704 | 12295 | −409 | 12257 |
| 12 (3 seq) | 13591 | 12193 | −1398 | 12176 |
| 16 (4 seq) | 13931 | 12311 | −1620 | 12294 |
| 256 | 15318 | 14524 | −794 | 14420 |
| 384 | 17355 | not top 5 | — | 15765 |

Tactic 8 is within 0.5 % of the best tactic per shape at every M from 1 to 256, so one bucket is enough; a per-shape
table would gain ≤ 60 µs. Above 256 larger tiles win, so the recommended pin stops at 256 and prefill (M ≥ 512) keeps
the stock call. Every one of the 32 tactics gave bit-identical output to tactic 0 for every shape and M tested; the
overlay was checked bit-identical to the stock call (eager and `torch.compile`, M = 1…8192) by
`benchmarks/mxfp8-tactic-sm110/test_overlay.py`. Not yet measured in a serving run.

## Notes

- Licence: the file is vLLM's (Apache-2.0); only the `[thor]` blocks differ from the image.
- Larger tiles would also speed up prefill (best per shape at M = 8192: −16 ms per forward, −7 %), e.g. adding
  `,2048:18`; not part of the recommendation because chunk sizes between 2048 and 8192 were not swept.
