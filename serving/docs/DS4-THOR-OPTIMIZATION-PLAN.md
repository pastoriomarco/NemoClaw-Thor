# Entrpi DS4 Thor optimization plan

**Started:** 2026-08-04
**Status:** Complete
**Initial target:** Entrpi/ds4 v0.5.4; **current runtime target:** v0.5.6.2
(API/startup smoke accepted; performance baseline remains v0.5.5),
DeepSeek-V4-Flash-0731 plus the matching DSpark drafter, Jetson T5000
(`sm_110`, 128 GB unified memory)

This plan turns the working Dockerized DS4 service into an explicitly
Thor-tuned implementation. It does not change the ManyForge `:8000` serving
lane. Experiments use the existing external weights under
`~/thor-hf-cache/ds4`; source builds and CUDA tools stay inside containers.

## Accepted Thor profile

- Source: Entrpi/ds4 v0.5.6.2 at
  `027714a4c290a756ef3e6ca557426528745f2033`; the fixed performance and
  long-context baseline below was measured on v0.5.5 at
  `2e9799073e08ea8f89eb1e72c47328ee6d90c6e8`.
- Build: CUDA 13, generic `sm_110` SASS. Spark `sm_121`/`sm_121a` is forbidden.
- API: OpenAI-compatible service on port 8050.
- Context allocation: 524,288 tokens.
- Continuous banks: two.
- Prefill chunk: 4,096 tokens.
- Deep selector: Entrpi v0.5.5+'s repaired streaming exact top-512 selector,
  enabled automatically only by the Thor image provenance marker. The v0.5.4
  atomics-free selector and Entrpi's safe tree remain explicit rollbacks.
- Memory protection: deterministic two-bank plan, outstanding-projection
  admission accounting, 12 GiB batch VMM budget, 8 GiB live-memory floor, and
  8 GiB fit headroom.
- Measured fixed HTTP path: 488 tok/s prefill at 2.4K, declining to 465 tok/s
  at 21K; peak easily-speculated DSpark decode is 23-24 tok/s while ordinary
  prose/JSON is about 11-12 tok/s.
- Validated depth: exact needles at 244,518 tokens on v0.5.5 and at 479,817
  tokens on the prior v0.5.4 512K profile.

The DS4 startup log's 43 GB/s estimate is invalid on Thor. Entrpi's own
`bench/bw_bench.cu`, compiled for `sm_110`, measured 241-245 GB/s read and
227-230 GB/s copy. Thor's memory subsystem is therefore not the source of the
Spark gap.

## Hardware facts that drive the work

The local CUDA runtime reports:

| Resource | Thor |
|---|---:|
| Compute capability | 11.0 |
| SM count | 20 |
| Threads per SM | 1,536 |
| Registers per SM | 65,536 |
| Shared memory per SM | 233,472 bytes |
| Opt-in shared memory per block | 232,448 bytes |
| Cooperative launch | supported |

DGX Spark has 48 SMs and 6,144 CUDA cores, compared with Thor's 20 SMs and
2,560 CUDA cores, while both expose 273 GB/s theoretical memory bandwidth.
Thor has datacenter-style TCGen05 Tensor Cores and substantially higher
advertised FP4 throughput, but the current DS4 expert path mostly uses IQ2/Q2
dequantization into legacy INT8 MMA fragments. The optimization objective is
therefore to improve occupancy and make safe use of Thor's native features;
it is not to tune for a nonexistent memory-bandwidth deficit.

## Non-negotiable gates

Every candidate must satisfy all applicable gates before it becomes a default:

1. Build only for Thor: stable `sm_110`, or an explicitly marked experimental
   `sm_110a` feature build. Never emit or run Spark `sm_121` code.
2. Preserve exact/accepted output against the v0.5.4 baseline and pass the
   deterministic API checks in `serving/test-ds4.sh`.
3. Preserve OpenAI non-streaming and streaming responses plus DSpark fallback.
4. Produce no new Xid, illegal access, CUDA graph failure, or server restart.
5. Retain the 256K exact-needle result. A change affecting the selector, VMM,
   or attention path also requires a deep-context replay before adoption.
6. Stay within the 128 GB unified-memory envelope without sustained swap I/O.
7. Report warmup, repetitions, prompt/decode sizes, context allocation,
   prefill chunk, and speculation mode. Do not combine unlike measurements.
8. Adopt a performance change only when the median improvement exceeds normal
   run-to-run noise and no important metric regresses materially.

## Execution phases

### Phase 0 — reproducible measurement

- Add a Docker-only experimental build mode pinned to the same v0.5.4 commit.
- Add a benchmark driver that records image/source identity, runtime settings,
  prompt/decode throughput, process health, memory, and kernel-error evidence.
- Re-measure the unmodified `sm_110` image after warmup.
- Keep the model service stopped between incompatible engine-side tests so two
  80+ GiB mappings are never resident simultaneously.

Exit gate: repeatable baseline with median and range for prefill and decode.

### Phase 1 — safe runtime sweep

Change one value at a time:

| Candidate | Values | Primary question |
|---|---|---|
| Prefill chunk | 2,048 / 4,096 / 8,192 | Which size best fills 20 SMs without memory or latency instability? |
| Context allocation | 262,144 / 524,288 | Does the larger reservation affect short-request performance or only capacity? |
| Continuous banks | 2 | One bank disables Entrpi's continuous/DSpark engine; keep two while tuning it. |
| Memory floor | 8 / 12 GiB | Can a safer floor avoid workspace OOM without rejecting normal solo work? |

The production default must still expose at least 256K. A 512K default remains
preferred if its solo performance is equivalent and its admission behavior is
honest.

**Result (2026-08-04): complete.** At 256K, effective 2K chunks regressed
prefill to 440.1 tok/s at 2.4K and 434.9 at 21K. Effective 8K chunks retained
488.3 tok/s short and improved the 21K leg from 465.2 to 483.0 (+3.8%), but
require both the live chunk and persistent scratch cap to be 8192. At 512K,
that extra scratch left only about 2 GiB available and correctly tripped the
8 GiB memory floor, forcing serial fallback. The 512K production profile
therefore remains 4K/4K; 8K/8K is an optional solo 256K performance profile.

### Phase 2 — Thor device profile and observability

Add a narrowly scoped source patch:

- `DS4_CUDA_THOR` build capability, separate from the generic SM120
  `BLACKWELL_MMA` classification.
- A `cuda-thor` build recipe that keeps the stable lane on `sm_110`.
- Thor-aware startup reporting of architecture, SM count, and chosen kernels.
- Remove or label the invalid calculated LPDDR bandwidth value. Do not use it
  for dispatch.
- Make the safe tree top-512 selector automatic on CC 11.0 until a replacement
  passes the deep-context gate; retain an explicit diagnostic override.

Exit gate: behavior matches the existing environment-based workaround and the
server boots without requiring undocumented Thor flags.

**Result (2026-08-04): complete.** The
`runtime-thor-topk` Docker target now compiles an explicit `DS4_CUDA_THOR`
capability for `sm_110`, records `profile=thor-topkdet256-attnhg` in
`/etc/ds4-build.txt`, and reports architecture, SM count, threads/SM, and
unified-addressing status at startup. The entrypoint reads that marker before
enabling the replacement selector; an unmodified `runtime` image defaults to
the safe tree. The invalid CUDA-attribute-derived 43 GB/s number is suppressed
on the Thor build and is not used for dispatch. The independently measured
bandwidth remains 241-245 GB/s read and 227-230 GB/s copy. The final image
booted with the expected provenance/selector log and passed OpenAI smoke plus
the deterministic arithmetic, JSON, and logic checks.

### Phase 3 — attention split sweep

Entrpi's head-group decode path derives its split count from SM count. With 64
heads it selects about 12 on Spark and 5 on Thor. Add a diagnostic
`DS4_CUDA_ATTN_HG_SPLIT_N` override and sweep 4-12 at short and deep context.
The existing `DS4_CUDA_ATTN_SPLIT_N` controls a different fallback and is not
a substitute.

Exit gate: select the fastest stable split, or retain automatic split=5 when
the sweep shows no repeatable improvement.

**Result (2026-08-04): complete.** Splits 4/5/6/8/10/12 were clean-booted
against identical 2.4K and 21K structured-decode legs. Values 4-6 were flat;
8 regressed slightly; 10 was only 0.1-0.2 tok/s ahead. Three-repeat 21K
medians were 23.6 tok/s for auto=5 (range 23.5-23.7) and 23.8 for split=10
(23.6-23.8). At 63K they were 22.1 and 22.3 respectively with identical
431.2 tok/s prefill. The overlapping ranges and sub-1% deep difference are
below the adoption threshold, so Thor retains the architecture-derived value
5. The override remains available for future diagnostics.

### Phase 4 — Thor D2R expert geometry

Entrpi's dominant IQ2/Q2 D2R kernels were tuned around GB10:

- 256-thread, eight-warp CTAs;
- 64-column routed-expert down tiles;
- 32-column fused gate/up tiles;
- a two-CTA-per-GB10-SM design assumption;
- a fused-tile decision justified for 2,048-token chunks.

Compile isolated variants rather than changing constants blindly:

- fused gate/up TileN 16 and 32 first; TileN 64 only if shared-memory and
  register evidence supports it;
- down TileN 32 and 64;
- launch-bound/occupancy variants appropriate to 20 Thor SMs;
- 2K/4K/8K chunks with `DS4_MMQ_D2R_STATS=1` only on telemetry legs.

Use Entrpi's `cuda/mmq/test` prototypes for kernel-level parity and timing,
then verify winners through the real server. Telemetry synchronizes the CUDA
stream and must never be enabled for published throughput legs.

Exit gate: bit-exact output and a repeatable end-to-end prefill improvement.

**Result (2026-08-04): complete.** The upstream 64-column Q2_K down tile,
32-column fused IQ2 gate/up tile, and two-CTA launch bound remain optimal on
Thor. Fused TileN=16 regressed prefill by 11-12%; down TileN=32 regressed by
1.3-1.4%. A three-CTA launch bound reduced each expert kernel from 128 to 80
registers/thread, but increased stack use from 192 to 472 bytes for fused
gate/up and 48 to 192 bytes for down; resulting spill traffic reduced prefill
by about 5.5-6.2%. All three candidates retained deterministic output hashes
and runtime stability, so the decisions are performance-based.

### Phase 5 — experimental SM110a MXF4 indexer

Entrpi compiles its exact MXF4 score/select implementation only for
`sm_121a`, although CUDA 13 exposes the corresponding TCGen05 feature target
on Thor. Add an experimental `compute_110a,code=sm_110a` build which defines
`DS4_CUDA_HAVE_MXF4` only after ptxas accepts the existing instructions.

- Do not replace the stable `sm_110` image during bring-up.
- Compare selected indices and outputs bit-for-bit with the generic scorer.
- Keep the safe tree top-512 selection during initial tests.
- Measure separately at 2K, 64K, 128K, 256K, and near 512K; the upstream
  five-times claim is for the scorer sub-operation, not total prefill.

Exit gate: compile, exact parity, deep stability, and an end-to-end win. If any
gate fails, document the compiler or semantic boundary and keep generic
`sm_110`.

**Result (2026-08-04): rejected at the compiler gate.** CUDA 13 accepts the
`compute_110a,code=sm_110a` target, but ptxas rejects Entrpi's unchanged
`mma.sync.aligned.kind::mxf4.block_scale` instruction as unsupported on
`.target sm_110a`. The same kernel is an `sm_121a`-only ISA path, not a feature
that can be enabled on Thor by changing the gencode target. No experimental
binary was produced or run; the stable build remains generic `sm_110`.

### Phase 6 — streaming top-512 replacement

Do not make the current 512-thread streaming selector the Thor default. The
first candidate preserved its one-pass atomic algorithm while changing the CTA
geometry from 512 threads x 4 items to 256 threads x 8 items. It reproduced Xid
13 and was rejected. The replacement is an atomics-free 256 x 8 exact merge:
each iteration sorts the previous 512 survivors with 1,536 new candidates in a
fixed 2,048-key block and retains the best 512. Every shared-memory write is
statically bounded. Run upstream's dual selector verification against the safe
tree implementation, then exercise the historical long-context reproducer and
the 126K/247K needle gates.

Exit gate: the previous 84K Xid reproducer and 126K/247K needles pass repeatedly
with no Xid 13.

**Result (2026-08-04): complete.** The atomics-free deterministic selector
matched the safe tree exactly on 1,025 dual-run comparisons through a 244,238
token prompt, including `n_comp=58,368`; the exact needle passed. Independent
84K, 105K, 125K, and 244K stability legs completed without Xid, graph error,
fallback, OOM, restart, or swap growth. At about 105K it improved cold prefill
from 364.1 to 402.6 tok/s (+10.6%) versus the safe tree, with no shallow
regression in the fixed matrix. It is the accepted Thor default; the original
atomic selector remains rejected and the tree remains the rollback.

### Phase 7 — unified-memory admission and scratch reuse

The 512K/two-bank profile has produced a repeated-request
`cudaMallocAsync` workspace OOM even with v0.5.4's live-memory floor.
Investigate and, if needed:

- preallocate/reuse D2R and selector scratch;
- account for worst-case CUDA workspace before admitting another bank;
- evict warm checkpoints before allocation failure;
- expose the reason a request queues or is rejected;
- validate repeated two-request waves, not only one successful pair.

Exit gate: repeated two-bank operation cannot trigger CUDA allocation failure.
Four 512K banks are explicitly out of scope unless memory accounting proves
they fit.

**Result (2026-08-04): complete for the supported two-bank envelope.** The
accepted selector at 512K/4K passed the three-repeat HTTP matrix followed by
five consecutive two-request waves (ten requests) with no allocation failure,
fallback, restart, Xid, or swap growth. About 15 GiB remained available after
the run. Four-bank 512K operation remains explicitly unsupported.

### Phase 8 — adopt Entrpi v0.5.5 and retire the replacement selector

Entrpi v0.5.5 identifies the v0.5.4 Xid 13 root cause: sibling warps could
observe different shared-counter values and disagree about whether to compact.
The upstream fix freezes that verdict under a barrier. Rebuild the pinned tag
for `sm_110`, compare it with the accepted v0.5.4 Thor selector, and promote it
only after real-score parity, deep retrieval, admission, and tool-call gates.

Exit gates:

- upstream stream-vs-tree selections are byte-identical on Thor;
- normal 105K prefill is no slower than the interim Thor selector;
- exact retrieval passes at 244K without Xid, fallback, OOM, or restart;
- two-bank serving survives repeated two- and four-caller waves;
- a tool call cut by `max_tokens` reports `finish_reason=length` in both the
  buffered and SSE OpenAI paths.

**Result (2026-08-05): complete.** Entrpi v0.5.5 at
`2e9799073e08ea8f89eb1e72c47328ee6d90c6e8` passed 513 byte-exact selector
comparisons through 104,676 tokens. Normal 104,738-token prefill reached 411.6
tok/s, 2.2% above the v0.5.4 deterministic selector and about 13% above the
safe tree. The final image retrieved all needles at 46,533 tokens (447.1
tok/s) and 244,518 tokens (348.2 tok/s), with no CUDA or serving failure.
Five two-caller waves and two four-caller waves completed on the two-bank plan.

The release's serial tool-budget fix was not ported to its continuous resolver:
a 26-token cutoff still returned `error`. The image therefore carries a small
server-only patch that preserves the engine's `length` verdict and partial
assistant text. Buffered and SSE gates now pass and never invoke over-budget
recovery. The default is `runtime-thor-v055`; the v0.5.4 atomics-free target
remains available only as rollback evidence.

## Benchmark matrix

The minimum decision matrix is:

| Test | Prompt/depth | Output | Repeats | Measures |
|---|---:|---:|---:|---|
| Short prefill | ~2K | 128 | warmup + 3 | D2R plus normal serving overhead |
| Medium prefill | ~18K | 128 | warmup + 3 | early attention/indexer cost |
| Deep prefill | 64K and 126K | 32-128 | 2 | selector and VMM behavior |
| Short decode | ~2K | 512 | warmup + 3 | sustained DSpark decode |
| Deep decode | 64K+ | 256 | 2 | attention split and KV-read cost |
| Concurrency | ~2K, c=1/2 | 256 | warmup + 3 | aggregate and per-request decode |
| Correctness | deterministic smoke + needles | bounded | every winner | output/API/deep-context parity |

Entrpi's published Spark comparison uses `pp=2048`, `tg=128/512`, depths
0/4096/16384, plus a separate concurrency sweep. Keep that shape for external
comparability. The local corpus must remain fixed because changing text changes
DSpark acceptance and decode speed.

## Result log

Append one row per controlled change. Failed candidates remain recorded so the
same unsafe or unproductive experiment is not repeated.

| Date | Image / change | Runtime delta | Prefill | Decode | Correctness / stability | Decision |
|---|---|---|---:|---:|---|---|
| 2026-08-03 | v0.5.4 baseline | 512K, chunk=4096, banks=2, safe top-K | 489 tok/s @2K | 12.09 tok/s @512 output | HTTP matrix passed; c=4 workspace OOM | Baseline |
| 2026-08-04 | v0.5.4 clean control | 256K, group=4096, chunk=4096, banks=2, floor=8 GiB, safe top-K | 488.5 tok/s @2.4K; 465.3 tok/s @21K | 24.3 tok/s @512 synthetic sequence | Three repeats per leg; continuous path only; no fallback, CUDA error, restart, or swap | Valid control |
| 2026-08-04 | Thor atomic stream 256x8 | 256K, capture=0, verifier=1, ~84K prompt | Failed at 32,768 / 83,983 tokens | N/A | First `n_comp=9216` stream-vs-tree check passed exactly; later Xid 13 out-of-range address poisoned CUDA | Rejected |
| 2026-08-04 | Thor atomic stream 256x8 | Same, verifier=0 | Failed before response | N/A | Reproduced Xid 13 with about 14 GiB available; async error surfaced at following D2R launch | Rejected; verifier and RAM exonerated |
| 2026-08-04 | Thor deterministic stream 256x8 | 256K, capture=0, verifier=1, 104,875-token cold prompt | 347.5 tok/s | 16.8 tok/s on 55-token needle JSON | 513 stream-vs-tree comparisons exact; retrieval passed; no Xid, restart, fallback, or swap | Correctness gate passed; verifier timing is diagnostic only |
| 2026-08-04 | Upstream safe-tree A/B control | 256K, capture=0, 104,591-token cold prompt | 364.1 tok/s | 16.7 tok/s on 59-token needle JSON | Retrieval passed; no Xid, restart, or fallback | Deep control for selector A/B |
| 2026-08-04 | Thor deterministic stream 256x8 | 256K, capture=0, 104,672-token cold prompt | 402.6 tok/s | 16.5 tok/s on 59-token needle JSON | Retrieval passed; no Xid, restart, fallback, or swap | **+10.6% prefill vs safe-tree control; continue** |
| 2026-08-04 | Thor deterministic stream 256x8 | Same with capture=1, 105,051-token cold prompt | 401.9 tok/s | 16.5 tok/s on 56-token needle JSON | Retrieval passed; no Xid, graph error, restart, fallback, or swap | Capture gate passed; no measurable prefill regression |
| 2026-08-04 | Thor deterministic stream 256x8 | 256K, capture=1, 125,082-token cold prompt | 389.0 tok/s | 16.6 tok/s on 54-token needle JSON | Retrieval passed; no Xid, graph error, restart, fallback, or swap | 126K gate passed |
| 2026-08-04 | Thor deterministic stream 256x8 | 256K, capture=1, 244,330-token cold prompt | 327.8 tok/s | 12.9 tok/s on 54-token needle JSON | Runtime stable with no Xid, graph error, restart, fallback, OOM, or swap growth; 2/3 needles exact, first hex value malformed | Stability passed; retrieval unresolved pending deep selector parity |
| 2026-08-04 | Thor deterministic stream 256x8 | 256K, capture=1, verifier=1, 244,238-token cold prompt | 250.9 tok/s diagnostic | 15.1 tok/s on 59-token needle JSON | Retrieval passed; 1,025 stream-vs-tree comparisons exact through `n_comp=58368`; no Xid, graph error, restart, fallback, OOM, or swap growth | Deep correctness gate passed; promote to performance candidate |
| 2026-08-04 | Thor deterministic stream 256x8 | 256K, capture=1, fixed HTTP matrix, three repeats | 488.6 tok/s @2.4K; 465.2 tok/s @21K | 23.4 @128 and 24.4 @512 structured output | Continuous path only; no fallback, CUDA error, restart, or swap; selector does not engage below 8,192 compressed rows | No shallow regression; deep +10.6% remains the adoption evidence |
| 2026-08-04 | Attention HG split sweep | 256K, capture=1, splits 4/5/6/8/10/12; confirmation at 21K and 63K | Unchanged (465.1 @21K; 431.2 @63K) | auto=5: 23.6 @21K, 22.1 @63K; split=10: 23.8, 22.3 | All candidates stable; output hashes identical on accepted structured legs | Retain auto=5; sub-1% difference is within noise |
| 2026-08-04 | D2R fused gate/up TileN=16 | 256K, accepted selector, down TileN=64; three-repeat fixed matrix | 431.6 tok/s @2.4K; 412.2 tok/s @21K | 24.3 @512 structured output | Deterministic output hashes; no fallback, CUDA error, restart, or swap | Rejected: 11-12% prefill regression vs fused TileN=32 control |
| 2026-08-04 | D2R Q2_K down TileN=32 | 256K, accepted selector, fused TileN=32; generalized 32-column Q8 prefetch mapping | 482.1 tok/s @2.4K; 458.7 tok/s @21K | 24.5 @512 structured output | Deterministic output hashes; no fallback, CUDA error, restart, or swap | Rejected: consistent 1.3-1.4% prefill regression vs down TileN=64 control |
| 2026-08-04 | Effective prefill chunk 2K | 256K, coalesce scratch=2048, chunk=2048, accepted selector | 440.1 tok/s @2.4K; 434.9 tok/s @21K | 23.7 @128 structured output | Three repeats; deterministic output hashes; no fallback, CUDA error, restart, or swap | Rejected: 9.9% short and 6.5% medium prefill regression vs 4K |
| 2026-08-04 | Effective prefill chunk 8K | 256K, coalesce scratch=8192, chunk=8192, accepted selector | 488.3 tok/s @2.4K; 483.0 tok/s @21K | 24.3 @512 structured output | Three repeats plus three two-request/21K waves; ~10 GiB available, no fallback, OOM, Xid, restart, or swap growth | Candidate: +3.8% medium prefill vs 4K; requires 512K memory gate |
| 2026-08-04 | D2R three-CTA launch bound | 256K, 4K chunk, accepted 64/32 tile geometry; expert kernels 128 to 80 registers/thread | 461.0 tok/s @2.4K; 436.6 tok/s @21K | 24.3 @512 structured output | Deterministic hashes; no fallback, CUDA error, restart, or swap; substantially higher stack use | Rejected: 5.5-6.2% prefill regression from spill traffic |
| 2026-08-04 | Thor MXF4 feature build | `compute_110a,code=sm_110a`, unchanged Entrpi exact MXF4 scorer | N/A | N/A | ptxas: block-scale MMA instruction unsupported on `.target sm_110a`; no binary run | Rejected at compiler gate; Spark `sm_121a` ISA path cannot be retargeted unchanged |
| 2026-08-04 | Accepted selector at 512K/8K | banks=2, coalesce scratch=8192, chunk=8192, floor=8 GiB | 379.5 tok/s @2.4K serial fallback | 19.9 @128 fallback | Only ~2 GiB available; every continuous admission correctly rejected by memory floor | Rejected: 8K scratch is incompatible with 512K continuous serving on 128 GB |
| 2026-08-04 | Accepted selector at 512K/4K | banks=2, coalesce scratch=4096, chunk=4096, floor=8 GiB | 488.4 tok/s @2.4K; 465.0 tok/s @21K | 24.4 @512 structured output | Three-repeat matrix plus five two-request waves; ~15 GiB available; no fallback, allocation failure, Xid, restart, or swap | Adopt as 512K production profile |
| 2026-08-04 | Accepted selector near 512K | 512K, 4K chunk, 1.32 MiB three-needle archive | 249.9 tok/s @479,817 prompt tokens | 10.5 tok/s on 55-token JSON | Exact retrieval; 1,920.5 s TTFT; continuous path; no fallback, Xid, OOM, restart, or swap growth | Near-limit depth gate passed; document cold-latency cost |
| 2026-08-05 | Entrpi v0.5.5 upstream selector | 512K, 4K chunk, 104,738-token cold prompt | 411.6 tok/s | 18.7 tok/s on 58-token needle JSON | Exact retrieval; no Xid, restart, fallback, or swap | Promote over v0.5.4 selector (+2.2%) |
| 2026-08-05 | Entrpi v0.5.5 selector verifier | 512K, capture=0, 104,676-token cold prompt | 355.2 tok/s diagnostic | 17.0 tok/s | 513 stream-vs-tree comparisons exact; retrieval passed; no bound trip or CUDA failure | Correctness gate passed |
| 2026-08-05 | Entrpi v0.5.5 admission accounting | 512K, two banks, five c=2 waves then two c=4 waves | N/A | c=2 aggregate 12.25-13.05; c=4 aggregate 8.29-8.72 tok/s | All 18 requests completed; no fallback, OOM, Xid, restart, or swap growth | Four callers queue safely over two banks |
| 2026-08-05 | Final v0.5.5 Thor image | 512K, 4K chunk, three-repeat fixed matrix | 488.5 tok/s @2.4K; 465.3 tok/s @21K | 23.4 @128; 24.4 @512 | Deterministic hashes; continuous path; no fallback or CUDA error | No shallow regression |
| 2026-08-05 | Final v0.5.5 Thor image | 512K, 4K chunk, 46,533-token cold prompt | 447.1 tok/s | 19.5 tok/s on 53-token needle JSON | Exact retrieval; default marker engaged repaired selector; no failure | Intermediate depth gate passed |
| 2026-08-05 | Final v0.5.5 Thor image | 512K, 4K chunk, 244,518-token cold prompt | 348.2 tok/s | 16.4 tok/s on 54-token needle JSON | Exact retrieval; 702.4 s TTFT; no fallback, Xid, OOM, restart, or swap growth | Deep stability gate passed; +6.2% vs prior selector |
| 2026-08-05 | v0.5.5 continuous tool-budget patch | OpenAI buffered + SSE, tool emission cut at 26 tokens | N/A | N/A | Both return `finish_reason=length`, partial text, exactly 26 tokens; no recovery continuation | Patch required and accepted |

The 2026-08-04 run followed removal of a 6.7 GiB private-anonymous leak in
GNOME System Monitor. With the leak present, the same class of two-bank 256K
profile could fall below the live-memory floor after lazy CUDA allocation and
silently use the slower serial fallback. After closing the monitor, the loaded
service retained about 13 GiB available and passed every continuous-path leg.
Host memory hygiene is therefore part of a valid comparison, not a model or
kernel optimization.

Decode figures in the two rows use different output corpora. The fixed 2026-08-04
harness requests an easily speculated integer sequence and demonstrates peak
DSpark acceptance; ordinary prose/JSON measured about 11-12 tok/s. Compare
candidate deltas only within the fixed harness and corpus.

## Deliverables

- This live plan and result log.
- Reproducible Docker patch/build inputs; no raw host installation.
- A benchmark command suitable for rerunning after Entrpi updates.
- The best verified Thor configuration in `ds4-compose.yml` and
  `start-ds4.sh`.
- Updated `DS4-ON-THOR.md` describing stable defaults, experimental switches,
  measured consequences, and rollback.
- No commit or push without separate operator authorization.
