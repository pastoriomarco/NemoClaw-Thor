# DeepSeek-V4-Flash-0731 / Entrpi DS4 on Thor

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

The first command builds `Entrpi/ds4` v0.5.1 inside Docker for `sm_110`, then
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

- Source is pinned to Entrpi DS4 `v0.5.1` commit
  `161b23609ab8a928246d268cec61101007f678b3`.
- Build target is exactly `CUDA_ARCH=sm_110`; never use Spark's `sm_121` or
  `sm_121a` target.
- The 0731 base uses only the matching DSpark drafter. The container passes
  `--no-mtp` so the legacy MTP GGUF cannot be paired with it.
- v0.5.1 trims evicted deep-context VMM pages; we retain a 12 GiB
  (`DS4_BATCH_VMM_BUDGET_MB`) ceiling and default to a 262,144-token context.
  On `sm_110`, v0.5.1's streaming top-512 indexer reproducibly raised Xid 13
  once the compressed index crossed 8,192 rows. The Compose profile sets
  `DS4_CUDA_NO_TOPK_STREAM=1`, Entrpi's built-in forensic escape to its
  output-equivalent legacy chunked-tree indexer. With that single change, the
  same 84,797-token reproducer passed, followed by exact-needle passes at
  126,215 and 247,065 prompt tokens. No new Xid was logged. The 512K allocation
  boots, but 512K with this workaround has not been validated and remains
  experimental.
- The default continuous-prefill chunk is 4,096 tokens. A cold 4,100-token
  Thor request measured 476 tok/s without extra swap use; defer the Spark-like
  8,192-token chunk until the host has substantially more than its current
  ~9 GiB unified-memory headroom.

## Validated 256K profile

The 2026-08-01 depth gate used the 0731 base and matching DSpark drafter with
`DS4_CTX=262144`, 4,096-token prefill chunks, capture enabled, and only
`DS4_CUDA_NO_TOPK_STREAM=1` changed from the failing baseline:

| Prompt tokens | Result | TTFT | Prefill | Decode | DSpark accept |
|---:|---|---:|---:|---:|---:|
| 84,797 | exact needle | 232.7 s | 364.5 tok/s | 8.3 tok/s | 80.0% |
| 126,215 | exact needle | 382.9 s | 329.7 tok/s | 13.0 tok/s | 100.0% |
| 247,065 | exact needle | 962.1 s | 257.2 tok/s | 9.2 tok/s | 83.3% |

The last row validates practical use of a 256K window while retaining about
15K tokens for output and protocol overhead. The two deeper requests used
retained-prefix cuts of 47,283 and 69,793 tokens even though the API reported
`cached_tokens=0`; their rates are DS4's total-prompt accounting for an
iterative-context workload, not independent cold-prefill benchmarks. The
247K request still took about 16 minutes to return on one Thor.

For Cline on the laptop, use:

- Base URL: `http://192.168.1.136:8050/v1`
- Model ID: `deepseek-chat`
- Context Window Size: `262144`
- API key: any non-empty placeholder (DS4 itself does not authenticate)

Keep Cline's maximum output tokens within the remaining context budget; 8,192
or 16,384 is reasonable for normal coding tasks.

## ManyForge integration gate

Do not repoint `manyforge/scripts/proxy/vllm-proxy.py` at this service yet.
First verify `/v1/chat/completions`, streaming, tool-call continuations, and
concurrent requests against DS4. When it passes, the proxy can retain its
public `:8000` contract while using DS4 at `:8050` as its upstream.
