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

---

## v0.8.0 evaluation (2026-06-29, Jetson AGX Thor — JetPack 7.1 / CUDA 13.0 / TRT 10.13 / R38.4)

### Build: use upstream's OWN container build, NOT our v0.7 `Dockerfile.trt`

0.8 ships an official containerized build that supersedes our custom one:
- `experimental/docker/build_container.sh` (base image `nvcr.io/nvidia/pytorch:26.04-py3`, the SBSA/aarch64 NGC PyTorch image — NOT an l4t image; it bundles CUDA+TRT+torch).
- Its defaults already match Thor: `CUDA_CTK_VERSION=13.0`, `CUTE_DSL_ARTIFACT_TAG=sm_110`, `CUTE_DSL_ARCH=aarch64` — run it with no overrides.
- JetPack 7.0/7.1 + CUDA 13.0 is an officially-listed Thor config (no host upgrade from 7.1 needed). The base ships CUDA 13.2 internally → runs under CUDA **Minor-Version-Compatibility** on the 13.0 driver (warns `cuInit()=803`, MVC enabled — **GPU confirmed functional**, builds + serves fine).
- Recipe: `git clone --recursive --branch release/0.8.0 …; experimental/docker/build_container.sh`. CuteDSL sm_110/cuda13 prebuilt ships in-repo (real gzip tarballs, ~1.6 MB — not LFS pointers).
- Our `Dockerfile.trt` / `build-trt.sh` / `trt_edge_llm_v0.7.0_thor.patch` are now **superseded for 0.8** (they were the pre-official-container 0.7 path; the patch's setup_pybind.py target also restructured).

### CRITICAL: trt-edge-llm is ModelOpt-ONLY — it cannot consume `compressed-tensors` NVFP4

The single biggest finding. trt-edge-llm's quant stack is built entirely on NVIDIA **ModelOpt** (`modelopt.torch`, `export_hf_checkpoint`, `hf_quant_config.json`, ModelOpt scale naming `_pre_quant_scale`). It accepts **ModelOpt-format** NVFP4 OR quantizes from **bf16** itself.
- Feeding it `unsloth/Qwen3.6-27B-NVFP4` (`quant_method: compressed-tensors`, `nvfp4-pack-quantized`, **no** `hf_quant_config.json`) **built without error but produced GARBAGE** (cross-script token salad) — the dequant scales were mis-mapped. **The same checkpoint is coherent in vLLM** (vLLM handles compressed-tensors). So: checkpoint fine, *format* unsupported here.
- ✅ Works (**empirically confirmed 2026-06-29**): `nvidia/Qwen3.6-35B-A3B-NVFP4` (`quant_method: modelopt`, has `hf_quant_config.json`) served a coherent answer — same Thor/trt-edge-llm stack that produced garbage on the unsloth 27B, so the difference is purely the checkpoint format. ✅ Works: any bf16 original (trt-edge-llm self-quantizes).
- ❌ Does NOT work: unsloth / RedHat-compressed-tensors / any `compressed-tensors` quant.
- Implication: real portability constraint vs vLLM. The **unsloth 27B is garbage here** (compressed-tensors). A correct ModelOpt 27B **does exist** — `nvidia/Qwen3.6-27B-NVFP4` (dense, has `hf_quant_config.json`, MTP head, ~21 GB) — and serves coherently *vanilla*; but its **MTP export segfaults** (see MTP section), so it is not a usable MTP path either. `nvidia/Qwen3.6-35B-A3B-NVFP4` (MoE, ModelOpt, production anchor) also works vanilla.

### Bug + patch: VL visual export drops `model_config`

For VL models the server's `_export_visual_onnx()` (experimental/server/engine.py) calls `_export_visual()` with 6 args but the signature needs 7 → `TypeError: _export_visual() missing 1 required positional argument: 'model_config'`. (`qwen3_5` IS in `_VISUAL_REGISTRY`, so vision is supported once the arg is supplied.)
- Patch: add `from tensorrt_edgellm.config import ModelConfig` and pass `ModelConfig.from_pretrained(self._model_dir)` as the 7th arg. Apply by **mounting** the fixed file over the image's: `-v <clone>/experimental/server/engine.py:/opt/TensorRT-Edge-LLM/experimental/server/engine.py:ro` (no rebuild).
- Fallback: add `EDGELLM_SKIP_VISUAL=1` (our added env guard on the `_is_vlm` line) to force text-only (vision unneeded for text/tool-call gating).
- Report upstream (clean call-site bug).

### Serving + engine cache

- Pass `--model <LOCAL snapshot dir path>` (NOT the HF id) so `_resolve_model_dir` returns it directly and skips `snapshot_download`; works with `HF_HUB_OFFLINE=1` (no accidental 54 GB pull).
- Engine cache is **dim-keyed**: `cfg_tag = i{max_input_len}_b{batch}_kv{max_kv_cache_capacity}`. Any change → fresh **engine** rebuild (ONNX is reused, dim-agnostic). Artifacts persist per tag under `<model>/.edgellm/engines/<tag>/` (~20 GB each) + the ONNX (~65 GB) — **not auto-pruned, root-owned, inside the HF snapshot**. Prune stale tags after settling; the whole `.edgellm/` can be deleted to reclaim.
- The chat endpoint does NOT validate the request `model` field, so a client can send any name.

### Context sizing (max_input_len vs max_kv_cache_capacity)

- There is **no separate max-output**. Output = **per-request residual** (`capacity − actual_input`). `max_input_len` = the largest single PROMPT accepted; `capacity` = max total (input+gen) and sizes the **pre-allocated KV pool** (no paging, unlike vLLM — ~32 GB runtime at 40 K for the dense 27B; 256 K runtime ≈ ~55–70 GB, fits 128 GB).
- Cline re-sends the FULL accumulated context each turn, so to use N tokens of context, `max_input_len` must be ≈ N.
- Rule: `max_input_len` = largest realistic prompt; reserve `capacity − max_input_len ≥` worst-case single-turn output (think block + answer + tool-call). For Qwen3.6 **thinking**, reserve ~**32 K**.
- Recommended 256 K split: **`--max-input-len 229376 --max-kv-cache-capacity 262144`** (224 K prompt, 32 K output floor). `253952/262144` (8 K floor) is too thin for a thinking model.
- You can also build the engine with larger limits and **cap Cline lower** — Cline self-limits within the engine, so no rebuild is needed to tune the split.
- **Cline config must-dos**: set Context Window = `max_input_len`, Max Output Tokens ≈ reserved residual, and ensure the endpoint returns a correct `usage` object (else Cline under-uses the window or sends `input+max_tokens > capacity` → hard error).

### MTP — DENSE-ONLY; **NOT available for the MoE 35B** (tested 2026-06-29, the blocker)

The headline reason to try trt-edge-llm (~2× MTP) **does not apply to our production model.** Empirically:

**MTP export rejects the MoE.** `tensorrt-edgellm-export <nvidia-35B-A3B> <out> --mtp` fails at base-model load:
```
NotImplementedError: Qwen3.5 dense MTP base is only supported for qwen3_5_text checkpoints.
```
The 35B-A3B is `qwen3_5_moe`. The export header reports `MTP capable: yes` (the checkpoint *has* an `mtp_num_hidden_layers: 1` head), but trt-edge-llm has **not implemented the MoE MTP base** — only dense (`qwen3_5_text`). So **no MTP for the 35B-A3B, or any `qwen3_5_moe`, on trt-edge-llm.** No flag/workaround short of upstream support.

**This flips the vLLM comparison for our model:** vLLM **does** run MTP on this exact MoE (our committed 35B recipe, K=3); trt-edge-llm gives the MoE **vanilla only**. So trt-edge-llm's MTP advantage is dense-only.

**The dense 27B is also blocked — by a *different* cause (tested 2026-06-30).** `nvidia/Qwen3.6-27B-NVFP4` **does** exist (ModelOpt; dense — top-level model_type `qwen3_5` whose promoted `text_config` is `qwen3_5_text`; MTP head present + unquantized per the `ignore: ["mtp*"]` quant list; ~21 GB — this corrects the earlier "no on-disk ModelOpt 27B"). It **clears the MTP gate** (`MTP capable: yes`, `MTP export: yes`). But the MTP-base ONNX export **segfaults `torch.export`** (dynamo, opset 24) while capturing the graph. The **vanilla** export of the same checkpoint succeeds (19 GB ONNX) and 111 GB RAM was free → **not OOM**: a `torch.export` bug on the MTP-base graph (tree-attention + draft head). The dynamo exporter is **hardwired** (`onnx/export.py:656 dynamo=True` + custom translation tables) — no legacy/torchscript fallback to route around it. So **MTP is unavailable for the dense 27B too.**

**Mechanics learned along the way (kept for if a dense model is pursued):**
- `llm_build` (the C++ engine builder MTP/spec needs) is **not in the image** — `experimental/docker/build.sh:214` builds only `--target NvInfer_edgellm_plugin _edgellm_runtime` (plugin + pybind for the server). The source + cmake target **are** present (`examples/llm/llm_build.cpp`, `add_executable(llm_build …)`, root `add_subdirectory(examples)`). Fix: add `llm_build llm_inference` to that `--target` line and rebuild the image, **or** `cmake --build build --target llm_build --parallel $(nproc)` in-container (cmake is pre-configured → compiles in ~minutes; verified it links cleanly on this image).
- The server's engine builder is the **Python `rt.LLMBuilder`** (`LLMBuilderConfig` = `max_input_len/batch/kv` only — **no spec field**) → it builds **vanilla engines only**. `--spec-decode-engine-dir` → `eagle_engine_dir` just **points at a pre-built** spec engine (when set, `_engine_dir = eagle` and `_build_engine()` is skipped) — it does not build one.
- Dense MTP build pipeline would be: `tensorrt-edgellm-export <dense-modelopt> <out> --mtp` → `llm_build --specBase` (base) + `llm_build --specDraft` (draft, both to one `--engineDir`) → serve `--model <m> --spec-decode-engine-dir <eng> --draft-top-k 1 --draft-step 3 --verify-tree-size 4`. The **serve wiring remains UNVERIFIED** (we never reached a working MTP build to test it; docs demonstrate MTP via the C++ `llm_inference`, not the server).

### Status / outcome — **trt-edge-llm 0.8 SHELVED 2026-06-30** (awaiting upstream updates)

- ✅ **Build + run on Thor works**: official `build_container.sh` image builds and runs (CUDA MVC); the `engine.py` visual-export patch + `EDGELLM_SKIP_VISUAL=1` guard work; `llm_build` compiles once added to `build.sh`'s `--target`.
- ✅ **ModelOpt-format NVFP4 is coherent**: vanilla `nvidia/Qwen3.6-35B-A3B-NVFP4` served a clean answer; the dense 27B also exports + would serve vanilla. The unsloth-27B garbage was purely the compressed-tensors↔ModelOpt mismatch, not Thor/trt-edge-llm.
- ❌ **MTP is unreachable for BOTH candidate models — two different causes:**
  - **35B-A3B (MoE):** MTP base not implemented — gate rejects `qwen3_5_moe`.
  - **27B (dense, `nvidia/Qwen3.6-27B-NVFP4`):** gate passes, but the MTP-base ONNX export **segfaults `torch.export`** (not OOM — 111 GB free; vanilla export of the same model succeeds).
- **Verdict:** trt-edge-llm 0.8 cannot deliver its headline MTP speedup for either of our models on this build. The only speed reason to use it was MTP; vanilla dense-27B here is bandwidth-bound (~16–19 tok/s) and loses to vLLM, and **vLLM already runs MTP on the MoE 35B** (committed K=3 recipe). So **vLLM remains the serving + MTP path**; trt-edge-llm is shelved.
- **Cleanup done (2026-06-30):** `tensorrt-edge-llm:experimental` image, scratch `~/trt-edge-mtp` (ONNX/engines), **and the clone `~/workspaces/trt-llm-edge`** all removed. **Kept** the `nvidia/Qwen3.6-27B-NVFP4` checkpoint (correct ModelOpt+dense+MTP-head model, ~21 GB) to avoid the slow re-download when revisiting. To revisit, re-clone the newer release; the `engine.py` patch (visual `model_config` fix + `EDGELLM_SKIP_VISUAL` guard) is documented in "Bug + patch" above for re-application.
- **To revisit on a future release:** re-clone the newer tag, build the image (add `llm_build` to `build.sh:214` if still absent), re-run the MTP export — if `torch.export` no longer segfaults on the MTP-base graph **and/or** MoE-MTP is implemented, resume from the MTP build pipeline documented above. Until one of those lands upstream, there is no MTP path here.

---

## 0.9.1 DFlash/DDTree retest — 2026-07-24 (dense 27B: DDTree blocked by Myelin; vanilla loses to vLLM)

Goal: DFlash+DDTree (new in 0.9.0/0.9.1) on Qwen3.6-27B NVFP4 with the
`z-lab/Qwen3.6-27B-DFlash` drafter (now cached in `~/thor-hf-cache`).

**Image**: `./build-trt.sh --trt-ref release/0.9.1` works after three
`Dockerfile.trt` fixes: (1) upstream deleted `experimental/llm_loader/` —
replaced by a pinned copy of the pyproject `[tools]` extra (nvidia-modelopt,
torchvision, …; re-sync on every bump); (2) torch cu130 force-reinstall bumped
2.10.0→2.12.0 to match upstream pins; (3) `pybind11==2.13.6` added (pyproject
`server` extra, skipped by `--no-deps`). The v0.7.0 Thor patch is **still
required** (upstream `ENABLE_CUTE_DSL` still defaults OFF; per-target
`LIBRARY_OUTPUT_DIRECTORY` still present) and applies cleanly with `--3way`.
Tagged `nemoclaw-thor/trt-edge-llm:0.9.1-thor-sm110`.

**Export pipeline works** (dynamo, opset 24): `--dflash-base`,
`--dflash-tree-base`, `--dflash-draft` all export for the (upstream-unvalidated)
Qwen3.6 pairing; the ModelOpt checkpoint's visual tower also exports.

**Checkpoint findings**:
- unsloth 27B NVFP4: re-confirmed **unusable here** — compressed-tensors layout
  (`weight_packed`/`*_global_scale`) is silently skipped (547/2111 tensors
  loaded) → pipeline runs end-to-end and produces garbage. Loader is
  ModelOpt-only; silent skip should be a hard-fail (upstream report #3).
- nvidia 27B NVFP4: loads fully (1846/2194; 348 skipped = visual scope), but its
  **quantized lm_head crashes the DFlash draft export**
  (`_normalize_dflash_lm_head_tensor_shape`: weight_scale [248320,320] vs
  [248320,5120] — upstream report #2). Workaround that works: modified
  checkpoint copy `/ws/ckpt-nv-fp16head` (built by `/ws/fix_lmhead.py`) grafting
  unsloth's BF16 `lm_head.weight` (the original Qwen head) + `lm_head` added to
  quant ignore.

**The blocker — spec engines do not compile on sm110** (upstream report #1):
both `--dflash-base` and `--dflash-tree-base` graphs from the fixed checkpoint
fail `llm_build` with Myelin `collective_codegen.cpp:1900 CHECK(false)
unsupported shape` → `Could not find any implementation for node
{ForeignNode[n1_385...node__to_copy_466]} [profile 1]` (~9 min into tactic
search). The **plain (non-spec) graph compiles fine** (951 s) ⇒ the gap is
specifically **NVFP4-dense GEMM × spec-verify profile**. DDTree runtime
mechanics are fine — on the (garbage-weight) unsloth engines the branching-tree
decoder initialized, captured CUDA graphs, and generated; only the engine build
blocks the real model.

**llm_inference spec flags** (undocumented constraints hit): DFlash requires
`--specDraftStep 1`; `--specVerifySize` must be ≤ the engine's
`maxVerifyTreeSize` (default 60 fails a 16-tree build); `--specDraftTopK 1` =
linear block, `>1` = DDTree (needs the tree-base engine; a tree-base engine
must not run with topK 1).

**Vanilla measurement** (plain engine, coding prompt, 512 tok, single stream):
coherent code, **11.6 tok/s** vs vLLM MTP-K3 ~16-18 ⇒ vanilla trt-edge-llm
loses; dense NVFP4 GEMMs run on TRT-native kernels (the CuTe DSL groups cover
MoE only). Concurrency (Finding #1) and 0.9.0 server tool-calling were NOT
retested — moot while spec engines don't build.

**Verdict: still shelved for the 27B.** No path beats vLLM until upstream fixes
the spec-profile Myelin failure (or routes dense NVFP4 GEMMs to CuTe DSL).
Build/run logs for the reports are in the session scratchpad
(`build-nvidia-*.log`, `nvidia-{fp16head,linear,plain}.log`).

**Assets kept**: the 0.9.1 image; the drafter in `~/thor-hf-cache`;
`~/tensorrt-edgellm-workspace/` holds `ckpt-nv-fp16head` + `fix_lmhead.py`,
the working nvidia plain engine, the garbage-weight unsloth engines (delete
freely), and regenerable ONNX dirs (large).

**Possible next**: (a) exercise DDTree mechanics on the upstream-validated
Qwen3.5-4B pair (BF16 base → compiles); (b) MTP retest on the now-compiling
dense checkpoint — likely hits the same spec-profile wall.
