# TensorRT-Edge-LLM v0.7.0 on Thor — build + bake-off notes

Companion document to `Dockerfile.trt`, `build-trt.sh`, and
`patches/trt_edge_llm_v0.7.0_thor.patch`. Captures what was learned during
the Thor v0.7.0 evaluation in April 2026, why we parked TRT-Edge-LLM as a
deployment runtime, and what's needed to revisit when v0.8+ ships.

---

## TL;DR — bake-off result (Nemotron 3 Nano Omni 30B-A3B NVFP4 on Jetson AGX Thor)

| Metric | vLLM v0.20.0 | TRT-Edge-LLM v0.7.0 | Winner |
|---|---:|---:|---|
| Single-stream sustained tps (2100t / 165s) | **12.71** | **23.22** | TRT 1.83× |
| Concurrent requests (≥ 2) | continuous batching, scales | **CRASHES** (Myelin error) | vLLM (TRT broken) |
| Recovery from request failure | normal | **requires server restart** | vLLM |
| First-token latency, short prompt | ~0.20 s | ~0.18 s | tie |
| Engine load (cached) | ~2 min (HF safetensors) | **5–14 s** (engine on disk) | TRT |
| Engine first-build cost | 0 (no compile step) | ~15 min for 30B NVFP4 + ~10 min ONNX export | vLLM |
| Image disk footprint | 17.4 GB (vLLM stack) | 21.6 GB (standalone, vLLM-independent) | similar |

**Verdict for our manyforge orchestrator use case (2–8 concurrent
agent slots): vLLM stays the production runtime.** TRT-Edge-LLM's
single-stream advantage doesn't apply because we never run a single user.

---

## What we found wrong with TRT-Edge-LLM v0.7.0

### Finding #1 — experimental Python server is not thread-safe

`experimental/server/api_server.py` calls `_runtime.handle_request(request)`
directly from the async FastAPI handler with no request queue or batch
scheduler. Two concurrent HTTP requests collide on the same Myelin
execution context:

```
[ERROR] [TensorRT] IExecutionContext::enqueueV3: Error Code 1: Myelin
([executor.cpp:864: myelinGraphLoad] Called with an already loaded
binary graph. In updateAndLoadGraph at .../graphContext.cpp:118)
```

This fires on the **second** concurrent request, regardless of the
`--max-batch-size` value the engine was built with. We built engines at
batch=1, batch=4, batch=8 — all crash on concurrent requests. The C++
runtime CAN serve batched sequences (engines have multiple worker
streams), but the Python wrapper doesn't route concurrent HTTP requests
to different execution contexts safely.

### Finding #2 — server doesn't recover from request failures

After the first Myelin crash, the engine context stays corrupted. All
subsequent requests (including sequential ones) hang or fail. Server
process must be restarted to recover. There's no internal reset / engine
reload path.

### Finding #3 — no tool-call parsing in the chat-completions endpoint

`api_server.py`'s response always carries the model's raw text in the
`content` field. Standard OpenAI-compatible clients (BFCL, OpenAI SDK
with `tools=`, vLLM benches) expect `tool_calls` as a structured array.
Nemotron Omni emits a Pythonic `<TOOLCALL>[...]</TOOLCALL>` syntax that
the server doesn't extract.

We worked around this with `tool_call_proxy.py` in `~/agentic-bench/scripts/`
(a 200-line FastAPI middleware that handles 4 tool-call formats:
Nemotron native, JSON-in-content, code-fenced JSON, XML-tagged JSON).

### Finding #4 — eight build attempts to discover the right cmake flags

The Dockerfile's header documents the full sequence of cmake flag
discoveries. Highlights:

- `-DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake` + `-DEMBEDDED_TARGET=jetson-thor` to make `find_package(TensorRT)` succeed under Debian apt's multiarch layout
- `-DENABLE_CUTE_DSL=ALL` to link the prebuilt `nvfp4_moe`, `fmha`, `gdn`, `gemm`, `ssd` kernel groups (without this, Mamba-MoE inference crashes at first token with `NvFP4MoEContiguousGemmRunner: decomposed AOT kernels not enabled`)
- `AARCH64_BUILD=1` env var to make `setup_pybind.py` forward the toolchain flags during the pybind build
- `apt install libnvonnxparsers-dev` because `find_package(TensorRT REQUIRED COMPONENTS OnnxParser)` needs ONNX parser headers + lib (NOT pulled by `libnvinfer-dev`)

These are baked into `Dockerfile.trt`. Two `setup_pybind.py` upstream gaps
(env-var forwarding, .so packaging path) are extracted into
`patches/trt_edge_llm_v0.7.0_thor.patch`.

---

## What we kept after the bake-off

```
docker/Dockerfile.trt                          # standalone, vLLM-independent build
docker/build-trt.sh                            # build orchestrator (mirrors build-vllm.sh)
docker/patches/trt_edge_llm_v0.7.0_thor.patch  # 2-hunk fix for setup_pybind.py
~/agentic-bench/                               # bench harness (runtime-agnostic)
    scripts/long_tps_oai.py                    # OAI-API tps probe (works on either runtime)
    scripts/run_ifeval_trt.sh                  # IFEval (lm-eval-harness)
    scripts/run_gsm8k_trt.sh                   # GSM8K-CoT (lm-eval-harness)
    scripts/tool_call_proxy.py                 # parser middleware for Omni tool-calls
    .venv/                                     # bench tools (lm-eval, bfcl-eval, etc.)
```

The Docker image and engine artifacts (~113 GB on disk) were deleted
because they're not reusable across v0.7 → v0.8 (TRT engine files are
TRT-LLM-version-specific) and the build pipeline can regenerate them
from source in ~30 min when needed.

---

## When TRT-Edge-LLM v0.8+ ships

Reproducible retry, in order:

1. Bump the ref:
   ```bash
   cd docker && ./build-trt.sh --trt-ref release/0.8.0
   ```
2. If the patch fails to apply (`git apply --3way` will report .rej
   files), inspect the rejected hunks. The patch's header documents
   *what* each hunk does — usually a 5-minute fix to update line context.
3. If new kernel groups appear in `cmake/CuteDsl.cmake`, `ENABLE_CUTE_DSL=ALL`
   picks them up automatically.
4. Re-test against the architectural blockers above:
   - **Concurrency**: send 2 simultaneous requests; if it doesn't crash,
     v0.8 has fixed Finding #1.
   - **Recovery**: deliberately trigger a request failure; if subsequent
     requests succeed, v0.8 has fixed Finding #2.
   - **Tool-call parsing**: send `tools=[…]` in a chat completion; if
     the response has a structured `tool_calls` field, v0.8 has fixed
     Finding #3 and the `tool_call_proxy.py` middleware can be retired.
5. Run the bench harness against the new server:
   ```bash
   ~/agentic-bench/.venv/bin/lm-eval run --model local-chat-completions \
       --model_args base_url=http://127.0.0.1:8000/v1/chat/completions,... \
       --tasks leaderboard_instruction_following,gsm8k_cot_zeroshot
   ```
6. If v0.8 passes Findings #1–#2, the bake-off question reopens: revisit
   the deployment decision based on multi-stream tps + quality scores.

---

## Pinned numbers from this session (April 2026)

For future regression-detection, the validated numbers on this exact
hardware/model combination:

- Hardware: Jetson AGX Thor, SM110a, 128 GB unified memory, JetPack 7.1, CUDA 13.0
- Model: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`
- vLLM v0.20.0 single-stream (2100t / 165 s):  **12.71 tps**
- TRT-Edge-LLM v0.7.0 single-stream (2100t / 90 s): **23.22 tps**
- vLLM Omni IFEval-20 prompt-strict (limit pipeline test): not run
- TRT-Edge-LLM Omni IFEval-20 prompt-strict (limit pipeline test): **85.0%**
- TRT engine on-disk size (NVFP4, batch=8): ~21 GB
- TRT image size (vLLM-independent): 21.6 GB
