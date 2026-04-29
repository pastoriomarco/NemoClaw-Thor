# agentic-bench — runtime-agnostic bench harness

Standardized benchmarks for comparing LLM serving runtimes on Jetson AGX
Thor (SM110a) for agentic workloads. Hits any OpenAI-compatible
`/v1/chat/completions` endpoint, so it works against vLLM, TRT-Edge-LLM,
or any other server speaking the OAI protocol.

This harness was written during the v0.7.0 TRT-Edge-LLM bake-off in
April 2026. The findings from that session are in
`docker/TRT-EDGE-LLM-NOTES.md`.

## What's here

```
scripts/
├── long_tps_oai.py             # single-stream sustained-tps probe (~3 prompts × 600-800 tokens)
├── run_ifeval_trt.sh           # IFEval (HF Open LLM Leaderboard variant, 541 prompts, ~90 min @ conc=1)
├── run_gsm8k_trt.sh            # GSM8K-CoT zero-shot (250 problems by default, ~30 min @ conc=1)
└── tool_call_proxy.py          # FastAPI middleware that injects structured tool_calls into responses
                                # from runtimes that don't parse them server-side (e.g. TRT-Edge-LLM v0.7.0)
```

The `*_trt.sh` filenames reflect history; they're runtime-agnostic — point
`ENDPOINT=` at any OpenAI-compatible server.

## Setup (first time only)

```bash
cd agentic-bench
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install lm-eval[ifeval] bfcl-eval
.venv/bin/pip install fastapi uvicorn httpx soundfile  # for the proxy + bfcl deps
```

## Running the benches

### Single-stream sustained throughput

```bash
BASE_URL=http://127.0.0.1:8000 \
MODEL=$(curl -s http://127.0.0.1:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])') \
.venv/bin/python scripts/long_tps_oai.py
```

Output is a per-prompt + total tps line, ~3 minutes on a 30B NVFP4 model.

### IFEval (instruction following)

```bash
LIMIT=20 ./scripts/run_ifeval_trt.sh        # quick pipeline test (~3 min)
./scripts/run_ifeval_trt.sh                 # full 541 prompts (~90 min @ conc=1)
```

Default sampling: T=0, max_tokens=1280, num_concurrent=1. Edit the script
to bump concurrency if your runtime supports it (vLLM does, TRT-Edge-LLM
v0.7.0 does NOT — see TRT notes).

Score to publish: `prompt_level_strict_acc` (HF Open LLM Leaderboard
canonical metric).

### GSM8K-CoT zero-shot (math reasoning)

```bash
LIMIT=250 ./scripts/run_gsm8k_trt.sh        # default subset
LIMIT="" ./scripts/run_gsm8k_trt.sh          # full 1319 prompts
```

Score to publish: `gsm8k_cot_zeroshot,exact_match`.

### BFCL (Berkeley Function Calling Leaderboard) — via tool-call proxy

If your runtime returns structured `tool_calls` (vLLM with
`--tool-call-parser`), point BFCL directly at it. If your runtime returns
text-only content (TRT-Edge-LLM v0.7.0), launch the proxy first:

```bash
docker run --rm -d --name tool-call-proxy --network host \
    -v $(pwd):/work \
    --entrypoint bash \
    nemoclaw-thor/trt-edge-llm:latest \
    -c '/opt/trt-venv/bin/pip install fastapi uvicorn httpx >/dev/null 2>&1 && \
        TRT_BASE_URL=http://127.0.0.1:8000 PROXY_PORT=8001 \
        /opt/trt-venv/bin/python /work/scripts/tool_call_proxy.py'
```

Then run BFCL against `:8001` instead of `:8000`. The proxy parses 4
tool-call formats (Nemotron native `<TOOLCALL>[fn(arg=val)]</TOOLCALL>`,
JSON-in-content, code-fenced JSON, XML-tagged JSON) and injects them into
the OpenAI-standard `tool_calls` field.

## Concurrency notes

vLLM continuous-batches concurrent requests automatically.

TRT-Edge-LLM v0.7.0's experimental Python server is NOT thread-safe —
concurrent requests crash the engine context with a Myelin
"already-loaded binary graph" error and require a server restart. Use
`num_concurrent=1` until v0.8 ships continuous batching. See
`docker/TRT-EDGE-LLM-NOTES.md` for full details.

## Cache locations

The lm-eval-harness writes per-task results + per-sample logs to
`results/<task-name>/<model-name>/`. The harness pulls model
configs/tokenizers from `HF_HOME` (defaulted to
`/home/tndlux/thor-hf-cache` in the scripts so it reuses existing weights
rather than re-downloading).

## Pinned bench results from April 2026

For Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 on Jetson AGX Thor:

| Bench | vLLM v0.20.0 | TRT-Edge-LLM v0.7.0 |
|---|---:|---:|
| long_tps_oai (2100 tokens) | 12.71 tps | 23.22 tps |
| IFEval prompt-strict (limit=20) | not run | 85.0% |

Full IFEval/GSM8K/BFCL on both runtimes is deferred until TRT-Edge-LLM
v0.8+ ships proper request batching.
