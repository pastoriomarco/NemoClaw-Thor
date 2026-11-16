# Flash-Next on Thor: next speed steps (N3 and R8b)

Status: **Stages A, C and D done; recipes in the main doc, "Current recipes"; Stage D follow-up next** (2026-11-16). Baseline and run history:
[`QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md`](QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md) (R8b, N1–N3). Analysis: Fable 5.2
(high effort) on the R8b/N3/N1 profiles and the vLLM 0.33 image sources, 2026-11-15; numbers below are from it
unless marked otherwise, labelled measured or estimate.

## Every strategy so far (kept or not)

Speed is the change in decode step time unless noted; "exact" = target output unchanged (the draft may change, which
moves only acceptance). Details in [`QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md`](QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md) (R1–N3)
and in the stages below.

| Strategy | Status | Speed | Output / quality | Where |
|---|---|---|---|---|
| MTP speculative decoding, 3 draft tokens | kept | large (R1 → R3) | exact | R4b, A1 |
| MTP 4 / 5 draft tokens | no | K=4 ties on bench, −8 % on agent replay; K=5 blocked (QSA ring 12 ∤ 848) | — | R4b, A1 |
| PLE table on NVMe + prefetch + vectorized gather | kept | needed to fit; −host time | exact | R5 |
| PLE page-cache prewarm (R7b) | replaced by CLOCK pool | ~0 | exact | R7b |
| CLOCK PLE slot pool 0.5 GiB | kept | −2 ms, +163K KV tokens (frees page-cache reservation) | exact | R8a |
| PLE early fetch (`VLLM_PLE_DISK_EARLY`) | rejected (1 and 4 streams) | +1.6 ms (1 seq); +9.5 ms step, GPU idle 5.5 → 9.1 ms (4 streams, N3 + C1 + C2) | exact | A2, D4d |
| Resident PLE table | no | — | — | does not fit (0.68 GB left) |
| Reduced draft vocabulary 47K (coding) | kept | sampling −ms | draft only | R3 |
| Capture-built draft vocabularies 47K / 65K | no | within noise | draft only | N3 A/B |
| MTP dense layers FP8 (`QWEN4EXP_MTP_DENSE_FP8`, was `…_MXFP8`) | kept | small: ~0.3 ms estimated (0.067 GB less per draft step), not measured alone; A/B queued ("To test later") | draft only | R3 |
| FP8 draft head | kept | ~−1.9 ms | draft only | R5 |
| Draft passes as FULL CUDA graphs | kept | ~0 on step, less host time | exact | R4b, R5 |
| FP8 target head (`QWEN4EXP_LM_HEAD_MXFP8`) | no (superseded) | −2.7 ms sampling | changes logits | R4b |
| Screened target head (FP8 screen + exact BF16 rows) | kept | sampling 5.3 → 2.6 ms | exact except BF16 ties | R6, A3 (99.4 % of calls screened) |
| Routed MoE: Marlin W4A16 (R4b) → FlashInfer CUTLASS (R6) → CuTe DSL fused (R8b) | CuTe DSL kept | R8b −2.7 ms decode, prefill 3.94 → 2.67 s | W4A4 as trained (QAD); BF16-atomic finalize = run-to-run variation | R4b, R6, R8b |
| Side layers on Marlin W8A16 (R4a) | no | −2 ms decode but prefill 5.2 s | — | R4a, R5 |
| Marlin atomic add | off | not measured | non-deterministic | "To test later" |
| NVIDIA NVFP4 checkpoint + online FP8 side layers, QSA `qkv_proj`+indexer BF16 (N3) | kept (N3) | −11 ms vs BF16 side layers | 93 vs N1 91 | N1–N3 |
| All side layers FP8 incl. QSA (N2) | no | −1 ms vs N3 | 86, 5 safety warnings | N2 |
| C1: QSA indexer split, q/k/v/gate FP8 (N3) | candidate | −1.3…−1.5 ms | Tool Eval alone pending (with C2: 94, 94) | Stage C |
| C2: hyperconnections FP8 | candidate on R8b; recheck under concurrency (D4c) | R8b −1.1 ms, N3 0 (GPU −1.3 ms both); prefill +0.13 s; +21–25K KV | 92, 91 (R8b); 94, 94 (N3 + C1) | Stage C |
| C3: MXFP8 GEMM tactic pin (R8b) | kept | −1.8 ms | exact | Stage C |
| Screened-head eligibility widening | no | nothing to win | — | A3 |
| Per-expert FC2 `input_scale` (FP4-Local) | no | — | same error (CPU probe) | analysis |
| MoE tile/PDL tuning, MTP expert kernel tuning, target FULL graphs, `--max-num-batched-tokens 16384`, torch.compile act-quant fusion | no | ≤ 0.5 ms each or blocked | — | analysis, R5 ideas |
| Memory share 0.70 → 0.75 | kept | — | — | R7, R8a |
| Benchmark hygiene: no desktop GPU monitor on Thor | rule | an open `gnome-system-monitor` cost ~3 ms per step | — | Stage C |
| Adaptive draft length, prompt-lookup drafts, NVFP4 draft head, faster prefill kernels | later | see "Later" | | |

## Where the time goes (decode step, MTP 3, short greedy prompt)

| Pool | R8b ms | N3 ms |
|---|---|---|
| Routed MoE FC1 + FC2 (CuTe DSL) + setup | 13.96 | 14.74 |
| Side layers (8-bit; N3 incl. 3.35 ms BF16 QSA `qkv_proj`) | 14.58 | 14.45 |
| Hyperconnections (BF16) | 7.52 | 7.46 |
| Target head (FP8 screen + exact rows) | 2.48 | 2.56 |
| Draft head + MTP dense, 3 passes / MTP experts | 2.36 / 0.60 | 2.36 / 1.05 |
| GDN / QSA / PLE kernels / other | 1.2 / 0.8 / 0.3 / 1.4 | 1.2 / 0.8 / 0.3 / 1.6 |
| GPU busy / idle (PLE gap, cold prompt) | 45.2 / 4.6 (3.5) | 46.5 / 3.7 (2.6) |

About 9 GB of weights are read per step, ~80 % of busy time at 245 GB/s, so both recipes have the same speed
ceiling within ~1 ms. The host runs 13–17 ms ahead of the GPU; the only host gap that reaches the GPU is the PLE
row fetch (0–1 ms at 72–95 % cache hits, 2.5–3.5 ms on new content). Live N3 under Tool Eval: 51.5 ms cycle,
3.05 tokens per step, 59 tok/s, acceptance by draft position 80 / 67 / 58 %.

Not worth doing (analysis): MoE tile/PDL tuning (n128 and n256 tie at 4 tokens; ≤ 0.5 ms left), MTP expert kernel
tuning (Marlin and Triton at their byte floor), FULL graphs for the target (≤ 0.45 ms, blocked by the PLE host
break), `--max-num-batched-tokens 16384` (the 105-token tail comes from the Mamba block-aligned chunk split, not the
budget), per-expert FC2 `input_scale` on FP4-Local (CPU probe on layer 23: same error in range, 0.088 vs 0.088,
more clipping at full range; synthetic inputs).

## Stage A — output-identical (now)

| # | Step | Expected | Output | Gate |
|---|---|---|---|---|
| A1 | MTP 3 / 4 / 5 with CUDA graph capture sizes including 5 and 6 (today `[1, 2, 4, 8]`, so a 5- or 6-token verify runs padded to 8) | +5–10 % decode on agent traffic | identical (greedy verify) | higher K kept at ≥ +4 % tok/s on agent-style replay and thinking-off bench no worse than −2 % |
| A2 | (implemented as a page-cache read-ahead, `VLLM_PLE_DISK_EARLY=1`: earlier tokens' rows are read during the draft passes, the batch's own staging is unchanged; the full per-token staging would add ~0.3–1 ms more and is built only if a gap remains) PLE ids hashed and fetched per token during the draft passes, instead of after the whole verify batch is known (`qwen4exp-ple-disk` + `qwen4exp-spec-step`) | −0.5…−2 ms per step, up to −3.5 ms on new content | identical | greedy outputs identical to baseline; target GPU −1 ms or `prefetch wait − id sync` < 0.3 ms |
| A3 | Count why sampling falls back from the screened head to the full head (logprobs, `min_p`, no `top_k`, logits processors, structured output), then widen eligibility where it stays exact | up to −0.9 ms on Tool-Eval-like traffic | identical | counts first; widening only for exact cases |

**A1 result (2026-11-15, N3, capture sizes 1–8, `~/fn-bench/round13-k{3,4,5}.log`): keep K=3.** Agent replay
(6 captured Cline requests, temperature 0): K=3 60.0 tok/s, acceptance 3.08; K=4 55.1 tok/s, acceptance 3.18. Bench
cycle 48.6 → 53.0 ms: the 4th draft costs +4.4 ms per step (target +2.7 ms GPU from more distinct experts per layer,
one more draft pass) and returns +0.1 tokens on agent traffic (break-even ≈ +0.28). The padding hypothesis is ruled
out (K=4 ran with a 5-token graph). K=5 does not start on stock vLLM 0.33: `QSA ring capacity 12 must divide the
attention block size 848` (`models/qwen4_exp/common/qsa_cache.py:850`). Outputs differ between K=3 and K=4
(different verify batch shapes round differently), so "output-identical" holds for a fixed K only.

**A2 result (2026-11-15, N3 K=3, `~/fn-bench/round14-e0.log` vs `round16-e1.log`): no gain, switch stays off.** With
`VLLM_PLE_DISK_EARLY=1` the host's `id sync` fell from ~14–16 to 1–6 ms per lookup, but the step got slower: bench
cycle 48.9 → 50.5 ms, target GPU 40.9 → 42.2 ms, bench 47.5/71.0/63.8 → 45.9/68.8/61.9 tok/s, warm 7.6K prefill
2.64 → 2.82 s. The host was already ahead of the GPU; the added kernels between draft passes and the early reads
(~1 ms per token, sharing Thor's memory system) cost more than the hidden disk waits. First server start failed
(`Only dense CPU tensors can be pinned`: a pinned buffer built under vLLM's default-device context), fixed with an
explicit `device="cpu"`. The full per-token staging variant is not pursued. Replay text also differs run to run at
temperature 0 with identical settings (CuTe DSL finalize uses BF16 atomics), so exactness checks rely on the unit
tests and acceptance, not on text equality.

**A3 first counts (agent replay at temperature 0):** 1,424 of 1,428 sampling calls screened; fallbacks only
`logit_bias` (2) and `structured_output` (2). **A3 closed, no gain:** with Cline's own settings (no sampling fields,
so the model defaults temperature 1.0 / top_k 20 / top_p 0.95 apply) all 322 calls were screened (`k20`); a brief Tool
Eval start added 18 `structured_output` fallbacks. Since server start 3,303 of 3,323 calls (99.4 %) were screened, so
widening eligibility has nothing to win and the +0.9 ms sample time under Tool Eval does not come from the full head.

Measurement for A1–A3: the bench prompts, step timing, and a replay of captured Cline requests (agent-style
traffic) read from vLLM's counters.

## Stage B — recipe choice (measurement)

B1. Two more Tool Eval runs each on N3 and R8b (with the Stage A settings). A mean gap of ≥ 2 points over three
runs decides; otherwise N3 (precision selectable per matrix, best score so far), with R8b as the leaner fallback
(37 GB smaller, +52K KV tokens).

## Stage C — changes output, Tool Eval required (planned)

| # | Step | Recipe | Expected | Gate |
|---|---|---|---|---|
| C1 | Split the QSA indexer from `qkv_proj` (`fuse_indexer_projection` off; q/k/v/gate FP8, indexer `index_qk_proj` BF16 via `QWEN4EXP_SIDE_FP8_SKIP`) | N3 | −1.5 ms per step | step ≤ 46.5 ms; Tool Eval mean within 1 point of N3 over two runs, no new safety warnings |
| C2 | Hyperconnection weights BF16 → FP8 (not quantized by NVIDIA's QAD nor by the RadixArk hybrid) | both | −2.6…−3.3 ms | ≥ −2.5 ms and Tool Eval within noise over two runs |
| C3 | FlashInfer MXFP8 GEMM tactic for 4-token calls (autotune is off; one fixed 128×128×256 tile today; FP8 block GEMM on the same matrices is 64 vs 70 µs) | R8b | up to −1.2 ms | exact; ≥ −0.5 ms |

C3 is exact (kernel choice only) and listed here only because it matters only if R8b is chosen.

**Stage C results (2026-11-15, `round17.sh`, laptop `stagec.log` / `stagec2.log`).** Bench = 3 greedy prompts
(tok/s e2e), cycle/target in ms, prefill = warm 7.6K prompt, replay = 6 Cline requests.

| Run | Bench tok/s | Cycle | Target GPU | Prefill | Replay tok/s (acc.) | Tool Eval |
|---|---|---|---|---|---|---|
| R8b + C3 (`VLLM_MXFP8_TACTIC=256:8`) | 51.5 / 70.6 / 65.9 | 47.0 (R8b 48.8) | 33.3 | 2.67 s | 61.7 (3.06) | 92 (R8b) |
| R8b + C3 + C2 | 47.1 / 68.8 / 62.9 | 49.4 | 34.9 | 2.98 s | 57.7 (2.98) | 92, 91 |
| N3 (`round13-k3`) | — | 48.6 | 35.5 | 2.69 s | 60.0 (3.08) | 93 |
| N3 + C1 + C2 | 45.2 / 68.2 / 61.9 | 50.8 | 36.5 | 2.97 s | 57.2 (2.99) | 94, 94 |

- **C3: kept for R8b** (−1.8 ms cycle, exact).
**Interleaved re-check (2026-11-15 17:30–18:30, one session, `~/fn-bench/round18-*.log`).** The afternoon runs
above were slowed ~3 ms by an open `gnome-system-monitor` on Thor (plain N3 52.0 ms with it, 48.8 ms after closing
it); N3 first and last bound the drift to 0.5 ms.

| Run | Cycle | GPU step | Prefill | Replay tok/s (acc.) | KV tokens |
|---|---|---|---|---|---|
| N3 (first / last) | 48.8 / 48.3 | 46.8 / 46.3 | 2.69 / 2.64 s | 58.8 (3.03) / 60.5 (3.09) | 422K |
| N3 + C1 | 47.3 | 45.3 | 2.66 s | 56.9 (2.82) | 443K |
| N3 + C1 + C2 | 47.3 | 44.0 | 2.80 s | 60.6 (2.97) | 468K |
| R8b + C3 | 47.3 | 44.3 | 2.62 s | 59.6 (2.99) | 475K |
| R8b + C3 + C2 | **46.2** | **42.9** | 2.75 s | **65.3 (3.17)** | 496K |

- **C1:** −1.3…−1.5 ms per step on N3, as estimated; prefill unchanged. Tool Eval so far only with C2 (94, 94).
- **C2:** GPU −1.3…−1.4 ms per step on both; step −1.1 ms on R8b, unchanged on N3 (the host gap grew from 2.0 to
  3.3 ms there); prefill +0.13 s on both. Earlier shape failures: `fp8_per_block` needs K % 128 (up has K = 320),
  `mxfp8` needs N % 32 (down has N = 336); fixed with per-projection formats (`QWEN4EXP_HC_FP8_FORMAT=
  "<down>[,<up>]"`, default `fp8_per_block,mxfp8`). Re-check under concurrency: D4c.
- **C3:** confirmed (R8b + C3 47.3 ms, morning 47.0).
- Replay tok/s moves with acceptance (2.82–3.17 here, output differs run to run), so cycle and GPU step are the
  speed measures; replay is the agent-traffic check.
- `tests/test_hc_fp8.py` passes with the per-projection formats (CPU container on Thor, 2026-11-15).

Projection for N3 on agent traffic (estimate): 59 tok/s today, 66–70 after Stage A + C1, 70–75 with C2.

## Stage D — up to 4 concurrent sequences (planned, after Stage C's GPU session)

Everything above is measured with `--max-num-seqs 1`. For parallel subagents / users, up to 4 sequences:

| # | Step | Note |
|---|---|---|
| D1 | `--max-num-seqs 4`, CUDA graph capture sizes up to 16, memory share ~0.85 | KV ≈ 29 KB/token BF16 (N3: 11.7 GiB = 427K tokens). 0.85 gives ~800K (N3) / ~850K (R8b). 1M BF16 (~27.4 GiB, share ~0.87–0.88) leaves only 2–3 GiB for the OS, PLE page-cache reads and other containers: too tight. Room exists because the CLOCK pool replaced the ~27 GB PLE page-cache reservation |
| D2 | 1, 2 and 4 parallel agent-replay streams; MTP 3 vs 1 (and 2) | Aggregate and per-stream tok/s from vLLM counters. Spec decoding gains shrink as the batch grows |
| D3 | C3 MXFP8 tactic per batch size (M = 1–16) | From the C3 benchmark (FP4-Local) |
| D4 | Re-check items that were flat for one sequence: MoE tiles n128/n256 (n256 wins at larger batches), tuned Triton config for N3's MTP experts, `--max-num-batched-tokens 16384` (prefill of one stream next to others' decode), CLOCK pool size, screened-head row limit (`MAX_TOKENS=64`, fallback above ~16 users × 4 tokens) | Measure only where D2 shows the pool matters |
| D4b | CLOCK PLE pool 0.5 / 1 / 2 GiB under 4 parallel streams (maintainer's request) | Hit rate, evictions, `waited for the disk` and step time per size; each extra 0.5 GiB costs ~18K KV tokens. Single-sequence replay saturated at 0.5 GiB with 0 evictions |
| D4d | **Done, rejected:** A2 at 4 streams: step 81.8 → 91.3 ms, GPU idle 5.5 → 9.1 ms, target GPU +5.6 ms (`round20-f-n3c1c2-a2.log`). Was: A2 early PLE fetch (`VLLM_PLE_DISK_EARLY=1`) at 4 streams | At 4 streams the GPU idles 4.5–5.5 ms per step (cycle 81.8 vs GPU 76.3 ms, N3 + C1 + C2) against ~2 ms at one; A2 hides exactly that PLE wait. For one sequence it cost +1.6 ms |
| D4c | C2 (hyperconnections FP8) at 1 / 2 / 4 streams, N3 and R8b (maintainer's request) | One sequence: GPU work −1.3…−1.4 ms per step, step time −1.1 ms on R8b and unchanged on N3, prefill +0.13 s (interleaved runs, `round18-*.log`). With more streams the weight reads are shared and the FP8 tensor-core GEMMs may matter more; the per-GEMM activation quantization also grows. Step time per stream count decides |
| D5 | FP8 KV cache (option) | ~850K–950K at 0.75, > 1M at 0.80; changes output, Tool Eval required |

## Stage D results (2026-11-15 evening – 2026-11-16 night)

Servers: `--max-num-seqs 4` (Cline replays, profiles) or 6 (`bench_categories`, llama-benchy, step 4),
`--kv-cache-memory-bytes 29600000000` (1,007,348 tokens on both recipes), memory available 5–13 GiB. Logs on Thor
`~/fn-bench/round19*`–`round28*`, `lb/*.md`, `bcat/results/`, `humaneval/`; tools reviewed before use and run in
CPU-only containers.

**Throughput and quality (single runs unless noted).**

| | N3 + C1 + C2 | R8b + C3 + C2 |
|---|---|---|
| sgbench C1 / C2 / C4, token-weighted tok/s (forum script) | 58.9 / 81.0 / 116.2 | 64.5 / 83.7 / 125.8 |
| `bench_categories` (tonyd2wild 0d507df) c1 / c4 / c6: median per-stream decode; batch total | 66.4 / 39.1 / 30.9; 71.2 / 100.1 at c4 / c6 | 68.0 / 41.7 / 30.1; 72.7 / 87.9 |
| llama-benchy 0.4.0 c1, pp2048 (prefill) at 0 / 32K / 128K history | 1,706 / ~3,000 / 3,050 | 1,130–1,383 / ~3,080 / 3,082 |
| llama-benchy c1, tg128 decode at 0 / 32K / 128K | 43.6 / 47.4 / 51.9 | 51.4 / 53.4 / 48.8 |
| llama-benchy c4 cached history (`--enable-prefix-caching`), tg128 total at 32K / 128K; TTFT | — | 99.8 / 86.4; 3.6 / 4.3 s |
| GPU time per step, 4 streams at full batch (profile) | 100 ms | 95.8 ms |
| HumanEval / HumanEval+ (EvalPlus 0.3.1, greedy, thinking off, 768 tokens) | 154 / 151 | 157 / 155 |
| Tool Eval (6be685f, 2 runs) | 94, 94 | 92, 91 |

At 4 streams the routed MoE takes ~54 ms of the step (each stream reads different experts; ~13 GB per step at
~245 GB/s, the bandwidth floor). A2 (early PLE fetch) was rejected again at 4 streams.

**Concurrent long prompts.** vLLM 0.33 has no decode-first rule; with Mamba "align" (832-token blocks) one stream's
cold prefill takes whole 7,488-token chunks (~2.4 s per step), and other streams get one step per chunk:
- staggered test (`stagger.py`: B's 32K prompt starts 15 s after A's): A decodes at **1.7–2.2 tok/s** during B's
  prefill (~11 s), against 43–60 before;
- llama-benchy cold c4 "t/s (total)" at 32K (15 tok/s) mostly measures the spread of first-token times (the metric
  spans first stream's first token to last stream's last token); the per-request column (13.5 at 32K) is the usable one.

Step 4 (`--long-prefill-token-threshold 832 --long-prefill-token-threshold-adaptive --prefill-schedule-interval 16`,
R8b): staggered A keeps **18–32 tok/s** during B's prefill, B's first token 11 → 19.5 s; cold c4 per-request decode
31 / 28 / 29 tok/s at 0 / 8K / 32K (default 28 / 23 / 14), TTFT ~44 s at 32K for all. With "adaptive" the chunk is
max(832, 8192 / requests) floored to 832 multiples (3,328 at 2 requests, 1,664 at 4, 832 from 5), so prefill slows
with load (estimate ~820 tok/s at 4 requests, ~470 at 5+). Not adopted: the default scheduler stays.

**Corrections from a review of these runs (Fable 5.2):**
- the step-4 Cline replay (106 → 77 tok/s) started with a cold cache (llama-benchy had evicted the histories), the
  default one warm; the comparison says nothing about the settings. The default baseline was a 4-sequence server;
- `vllm:request_prefill_time` runs from admission to first token (includes holds and shared steps), not prefill compute;
- the Cline replay is one conversation (each request extends the previous one), so 4 streams mostly hit the cache.
  Real captured traffic (432 requests, 52 conversations): 3.7K (p50) / 8.5K (p90) new tokens per turn on 49K (p50) to
  155K histories; cold starts 10–17K; 20 of 26 conversation starts ≥ 8K got no cache hit;
- stagger uses thinking off and `ignore_eos` (acceptance 3.8 vs 2.8–3.1 on agent traffic); single runs ±30 %;
- Spark Arena likely measures with cached context, so Thor's comparable rows are the cached ones.

**Forum comparison (best single DGX Spark found; tools and versions differ unless stated).**

| | Thor (best of the two) | Single Spark |
|---|---|---|
| `bench_categories` c1 / c6 (same script) | 68.0 / 100.1 | 67.5–69.8 / 88–92 ("Six weeks…", Oct 22) |
| llama-benchy c1 tg128 | 48–53 | 50–55 (hibrid48, Spark Arena, adaptive draft 3–7) |
| Prefill on a long history | 2,200 (cached, c4 total) – 3,080 (cold) | 532–834 (hibrid48, likely cached) |
| Tool Eval | 94 (6be685f, 88 scenarios) | 93 (v2.8.1, 97 scenarios) |
| HumanEval | 157/164 no-think | 98.17 % thinking xhigh; 158/164 no-think on 2 Sparks |

**Next (Stage D follow-up):** a realistic multi-agent replay (4–6 different captured conversations, turns sent
after each reply plus a tool pause, staggered starts, streaming; per turn TTFT, longest gap, turn time, cached vs
computed tokens; same cache state per config, 3 repetitions), then the scheduler sweep on it (default; threshold 832
non-adaptive with interval 1 / 2 / 4; step 4 for reference), and D2 (draft length 3 / 2 / 1). Why subagent starts
miss the prefix cache. Later, with approval: skipping the per-turn recompute of the previous reply (~28 % of prefill),
adaptive drafting (vLLM PR #60068 port).

## Later (new mechanisms, maintainer approval first)

Adaptive draft length for one sequence — future possibility, not queued (2026-11-15): vLLM fixes K at server start
and 0.33's adaptive verification is DSpark-only and needs FULL graphs, so this is a new mechanism (~1–2 days). It
would pay only on predictable stretches (YAML/code, where per-position acceptance reached 0.94–0.99) while staying at
K≤3 on prose; K=5 would also need the QSA ring/block-size constraint lifted. A cheap first measurement, if revisited:
per-request K=3 vs K=4 decode on the agent replay (`~/fn-bench/replay2.py`). One-chunk prefill
and CuTe DSL GDN prefill, prompt-lookup hybrid drafting, hyperconnection kernel fusion at prefill, NVFP4 draft head
(draft-only, −0.7 ms).
