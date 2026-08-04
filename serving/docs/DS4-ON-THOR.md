# Serving recipe: DeepSeek-V4-Flash-0731 / Entrpi DS4 on Thor

This is an isolated Dockerized serving option for Jetson Thor. It is not yet a
`serving/config.sh` profile and does not change the existing ManyForge proxy
(`:8000`) or its active backend. The server binds at `127.0.0.1:8050` so it
can be validated independently first.

## Start

```bash
cd ~/workspaces/dev_ws/src/NemoClaw-Thor
./serving/start-ds4.sh
./serving/start-ds4.sh logs
```

The first command builds the pinned Entrpi/ds4 v0.5.4 source inside Docker for
`sm_110`, including the validated Thor deterministic top-512 selector, then
starts a resumable in-container download alongside a DS4 container that waits
quietly for the complete pair. The weights persist outside the container in
`~/thor-hf-cache/ds4/`:

- `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
- `DSpark-drafter-Q2K-Q8-0731.gguf`
- `kv-cache/`

At a 5 MB/s connection, the 93.69 GB (87.26 GiB) model pair takes about 5.21
hours at a sustained line rate. The downloader writes `*.part` files and
resumes them after interruptions; the service starts automatically after both
GGUFs are atomically in place.

The default host bind is loopback only. To expose DS4 on Thor's LAN address for
an external Cline instance, start it with the explicit address:

```bash
export DS4_BIND_ADDRESS=192.168.1.136
./serving/start-ds4.sh start
```

Use `http://192.168.1.136:8050/v1` only after that explicit bind; DS4 has no
API-key enforcement, so do not bind it broadly on an untrusted network. Keep
the export in the shell when using the `smoke` and `test` commands against the
LAN-bound instance (or set `DS4_BASE_URL=http://192.168.1.136:8050`).

Once healthy:

```bash
./serving/start-ds4.sh smoke
./serving/start-ds4.sh test
curl http://127.0.0.1:8050/v1/models
```

`test` runs three validated ~200-token responses to report the server's decoder
tok/s and TTFT, then checks deterministic arithmetic, JSON, and basic logical
inference. It is a quick deployment sanity check, not a model benchmark or a
substitute for the ManyForge tool/concurrency gate below.
The throughput probe and JSON case use `reasoning_effort=none` so JSON follows
the API contract; arithmetic and logic retain the API default. Set
`DS4_TEST_REASONING_EFFORT=` to observe default-mode output throughput instead.

## Invariants

- Source is pinned to Entrpi DS4 `v0.5.4` commit
  `215af2f1245324bcebf9a69a498eff79275aac8e`.
- Build target is exactly `CUDA_ARCH=sm_110`; never use Spark's `sm_121` or
  `sm_121a` target.
- The default image is `nemoclaw-thor/ds4:v0.5.4-sm110-thor`, built from the
  `runtime-thor-topk` target. `/etc/ds4-build.txt` records the source, arch,
  and Thor profile, and the entrypoint enables the replacement selector only
  when this provenance marker is present.
- The 0731 base uses only the matching DSpark drafter. The container passes
  `--no-mtp` so the legacy MTP GGUF cannot be paired with it.
- v0.5.4 trims evicted deep-context VMM pages and adds a live admission floor.
  We retain a 12 GiB (`DS4_BATCH_VMM_BUDGET_MB`) ceiling, an 8 GiB
  (`DS4_MEM_FLOOR_GB`) live-free-memory floor, and default to a 524,288-token
  context. `DS4_SERVER_COALESCE_MAX=2` keeps continuous DSpark serving armed
  with two safe Thor banks. A four-bank, 2K-prompt concurrency probe reached a
  CUDA workspace OOM, so the higher Spark-oriented setting is not safe at this
  512K window. The live memory floor still protects against unsafe additional
  or deep work.
  `DS4_NO_UPDATE_CHECK=1` keeps this LAN service from making the upstream daily
  update request.
- Entrpi's original atomic streaming top-512 indexer reproducibly raised Xid
  13 on `sm_110` once the compressed index crossed 8,192 rows. The Thor image
  replaces it with a bounded 256-thread x 8-item, atomics-free exact merge.
  Upstream's dual-run verifier matched the safe tree on 1,025 selector calls
  through a 244K prompt, the needle passed, and the replacement improved a
  controlled 105K prefill by 10.6% over the tree. Set
  `DS4_CUDA_NO_TOPK_STREAM=1` only to force the slower tree for diagnosis.
- The default continuous-prefill chunk and persistent scratch cap are both
  4,096 tokens. They must move together. At 256K, setting both to 8,192
  improves a controlled 21K prefill by 3.8%; at 512K the larger scratch leaves
  only about 2 GiB available and the memory floor correctly forces slow serial
  fallback. Therefore 512K/4K is the production profile.
- The numeric 43 GB/s value previously printed by DS4 was an invalid estimate
  derived from CUDA attributes on unified-memory Thor, and was never used for
  dispatch. The Thor build suppresses it. Entrpi's bandwidth probe measured
  roughly 241-245 GB/s read and 227-230 GB/s copy on this device.

## HTTP serving benchmark (Thor profile)

On 2026-08-04, the real OpenAI HTTP path was measured three times per leg after
warmup using the accepted deterministic selector, two banks, a 512K allocation,
and effective 4K prefill chunks. A fixed local corpus makes candidate
comparisons reproducible. This does not start engine-side `ds4-bench` alongside
the service, which would map a second copy of the 80+ GiB model.

| Effective prompt | Output limit | Median prefill | Median decode |
|---:|---:|---:|---:|
| 2,414 | 128 | 488.4 tok/s | 23.3 tok/s |
| 2,415 | 512 | 487.7 tok/s | 24.4 tok/s |
| 21,052 | 128 | 465.0 tok/s | 23.1 tok/s |

These decode numbers use an easily speculated integer-sequence output and show
peak DSpark behavior; normal prose/JSON in the earlier comparable HTTP corpus
measured about 11-12 tok/s. Five consecutive two-request waves at 512K/4K all
passed (ten requests) with no allocation failure, fallback, Xid, restart, or
swap growth; about 15 GiB remained available. Four requests previously caused
a `cudaMallocAsync` workspace failure, so the Compose profile deliberately
exposes only two banks. This is two-request admission capacity, not a claim
that two near-512K active sequences can both fit simultaneously.

## Validated near-limit 512K depth

The accepted 512K/4K profile completed a cold exact-retrieval request at
479,817 prompt tokens on 2026-08-04. Three unique values placed at the early,
middle, and late eighths of a 1.32 MiB distractor archive were returned exactly:

| Prompt | Output | TTFT | Prefill | Decode | Wall | Result |
|---:|---:|---:|---:|---:|---:|---|
| 479,817 | 55 | 1,920.5 s | 249.9 tok/s | 10.5 tok/s | 1,930.4 s | PASS |

The request stayed on the continuous path and produced no fallback, Xid, OOM,
restart, or swap growth. It also demonstrates the practical limit: 512K is
real capacity, but a cold ~480K prompt takes about 32 minutes on this Thor.
Normal coding sessions benefit from retained prefixes and the v0.5.4 warm
checkpoint behavior, but clients still need timeouts long enough for the first
deep prefill.

## Optional 256K / 8K performance profile

For a solo coding workload that needs no more than 256K, both effective chunk
knobs may be increased together:

```bash
DS4_CTX=262144 \
DS4_SERVER_COALESCE_MAX_TOKENS=8192 \
DS4_CONT_PREFILL_CHUNK=8192 \
./serving/start-ds4.sh start
```

This measured 483.0 tok/s at a 21K prompt versus 465.2 for 4K chunks (+3.8%)
and passed three two-request waves without fallback or memory errors. Do not
combine 8K chunks with the 512K allocation on this 128 GB Thor.

## Rollback

The unmodified Entrpi binary remains buildable as a distinct image. It keeps
the safe-tree selector by default:

```bash
DS4_BUILD_TARGET=runtime \
DS4_IMAGE=nemoclaw-thor/ds4:v0.5.4-sm110-upstream \
./serving/start-ds4.sh start
```

For a selector-only diagnostic on the normal Thor image, use
`DS4_CUDA_NO_TOPK_STREAM=1`. Never reuse the normal Thor image tag for the
upstream target, because Docker tags—not target names—identify the resulting
runtime image.

## Historical 256K safe-tree depth gate

Before the replacement selector was implemented, the 2026-08-01 depth gate
used the 0731 base and matching DSpark drafter with
`DS4_CTX=262144`, 4,096-token prefill chunks, capture enabled, and only
`DS4_CUDA_NO_TOPK_STREAM=1` changed from the failing baseline:

| Prompt tokens | Result | TTFT | Prefill | Decode | DSpark accept |
|---:|---|---:|---:|---:|---:|
| 84,797 | exact needle | 232.7 s | 364.5 tok/s | 8.3 tok/s | 80.0% |
| 126,215 | exact needle | 382.9 s | 329.7 tok/s | 13.0 tok/s | 100.0% |
| 247,065 | exact needle | 962.1 s | 257.2 tok/s | 9.2 tok/s | 83.3% |

The last row validated practical use of a 256K window while retaining about
15K tokens for output and protocol overhead. The two deeper requests used
retained-prefix cuts of 47,283 and 69,793 tokens even though the API reported
`cached_tokens=0`; their rates are DS4's total-prompt accounting for an
iterative-context workload, not independent cold-prefill benchmarks. The
247K request still took about 16 minutes to return on one Thor. The current
atomics-free selector later matched the tree on 1,025 dual-run calls through a
244K prompt, retrieved the needles, and became the default described above.

For Cline on the laptop, use:

- Base URL: `http://192.168.1.136:8050/v1`
- Model ID: `deepseek-v4-flash` (the ID advertised by `/v1/models`)
- Context Window Size: `524288`
- API key: any non-empty placeholder (DS4 itself does not authenticate)

Keep Cline's maximum output tokens within the remaining context budget; 8,192
or 16,384 is reasonable for normal coding tasks.

## ManyForge integration gate

Do not repoint `manyforge/scripts/proxy/vllm-proxy.py` at this service yet.
First verify `/v1/chat/completions`, streaming, tool-call continuations, and
concurrent requests against DS4. When it passes, the proxy can retain its
public `:8000` contract while using DS4 at `:8050` as its upstream.
