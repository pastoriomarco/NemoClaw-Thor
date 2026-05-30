# V9 — Qwen3.6-35B-A3B-NVFP4 (NVIDIA serving recipe)

**Profile name**: `qwen3.6-35b-a3b-nvfp4-nvidia`

**Status (2026-05-30)**: ★★ candidate — matches iter-32 cosmos-reason2-8b on the composer smoke corpus (51/66 = 77.3%) using a 35B-A3B base, ~19 tok/s, 22 GiB footprint.

## TL;DR
- **Weights**: `RedHatAI/Qwen3.6-35B-A3B-NVFP4` (NOT `nvidia/Qwen3.6-35B-A3B-NVFP4` — that variant has an NVFP4-quantized `lm_head` vLLM 0.22 cannot load; see [Why RedHat weights](#why-redhat-weights-not-nvidia))
- **Serving recipe**: NVIDIA's published Spark recipe — MTP K=3 + `moe_backend:triton` + Froggeric chat template + FP8 KV cache
- **Proxy/bridge**: iter-32 production caps (`max_tokens=2048`, `thinking_token_budget=512`, chain-on /compact every 2 prompts)
- **vLLM**: 0.22.1.dev0 from v9 image (FlashInfer 0.6.12, flash-attn-4 b15)

## Profile (launch.sh)
```sh
THOR_LAUNCH_MODEL_SOURCE="RedHatAI/Qwen3.6-35B-A3B-NVFP4"
THOR_LAUNCH_GPU_MEMORY_UTILIZATION="${THOR_GPU_MEMORY_UTILIZATION:-0.85}"
THOR_LAUNCH_CHAT_TEMPLATE_HOST_PATH="${THOR_CHAT_TEMPLATE_HOST_DIR}/qwen-fixed-froggeric.jinja"

THOR_DOCKER_ENV_ARGS+=(
    "-e" "VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass"
    "-e" "VLLM_USE_FLASHINFER_MOE_FP16=0"     # drafter BF16 MoE → Triton (avoid SM100-only tile)
)

THOR_VLLM_ARGS+=(
    "--download-dir" "/data/models/huggingface/hub"
    "--kv-cache-dtype" "fp8"
    "--attention-backend" "flashinfer"
    "--enforce-eager"                          # FP4 GEMM doesn't compose with CUDA graphs
    "--language-model-only"
    "--enable-prefix-caching"
    "--enable-chunked-prefill"
    "--async-scheduling"
    "--max-num-batched-tokens" "8192"
    "--reasoning-parser" "qwen3"
    "--enable-auto-tool-choice"
    "--tool-call-parser" "qwen3_coder"
    "--default-chat-template-kwargs" '{"enable_thinking":true}'
    "--speculative-config" '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
    "--trust-remote-code"
)
```

config.sh entry: `qwen3.6-35b-a3b-nvfp4-nvidia` with `max_model_len=262144`, `max_num_seqs=5`, `kv_cache_dtype=fp8`.

## Stack wiring
```
composer :9000 ──HTTP──▶ assistant bridge :8200 (chain-on, /compact every 2)
                                 │
                                 ▼
                          openclaw-gateway (sandbox SSH :18789)
                                 │
                                 ▼
                          vllm-proxy :8000 (max_tokens=2048, thinking_budget=512)
                                 │
                                 ▼
                          vLLM :8050 (this profile)
```

Required env when starting proxy:
```
OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS=2048
OPENCLAW_PROXY_THINKING_TOKEN_BUDGET=512
```

Required env when starting bridge:
```
OPENCLAW_ASSISTANT_USE_GATEWAY=true
OPENCLAW_ASSISTANT_COMPACT_EVERY_N=2
OPENCLAW_ASSISTANT_COMPACT_TIMEOUT_S=180
```

## Boot characteristics
- Cold boot from clean cache: ~5 min (engine init 199s + chat template + weights 56s)
- Warm boot (safetensors in page cache): ~99s engine init
- Model footprint: **22.61 GiB** (vs ~70 GiB BF16; NVFP4 nearly halves it)
- vLLM oracle picks **VLLM_CUTLASS** NVFP4 MoE backend (out of 7 candidates) — best path for SM110
- NVFP4 linear kernel: **FlashInferCutlassNvFp4LinearKernel**
- Drafter unquantized MoE: **TRITON** (avoids broken SM100-only CUTLASS BF16 tile)
- Attention backend: **FLASHINFER**
- MTP detected, shares target embedding + lm_head with drafter

## Smoke corpus result (2026-05-30)
**51/66 (77.3%)** — matches the iter-32 cosmos-reason2-8b winner exactly.

| Bucket | Pass | Soft | Fail | Total |
|---|---|---|---|---|
| P1-P3 priorities | 3 | 0 | 0 | 3 |
| WRAP/SCENE_add/TREE_insert | 4 | 0 | 1 | 5 |
| INSERT_position | 2 | 0 | 1 | 3 |
| PARALLEL/FALLBACK | 0 | 0 | 5 | 5 |
| UPDATE/REPLACE | 1 | 0 | 4 | 5 |
| DELETE/MOVE/PARAM/BB | 9 | 0 | 0 | 9 |
| SCENE_update/remove | 0 | 0 | 3 | 3 |
| PnP_01-18 (sequence) | 18 | 0 | 0 | 18 |
| PnP_20 (gripper force) | 1 | 0 | 0 | 1 |
| CUR_* (current-frame) | 5 | 1 | 1 | 7 |
| CLARIFY_* | 1 | 3 | 0 | 4 |

### Strengths
- **PnP_01-18: 18/18 clean.** The core agentic pick-and-place loop is rock solid.
- All MOVE / DELETE / BB / PARAM cases clean.
- 4 cases that the harness scored "soft-pass" (tool+state correct, answer text quibble) — meaningful contracts honored.

### Failure clusters
1. **Long-chain compositional, hit token cap** (≈8 cases, 180-275s latency):
   - PARALLEL_concurrent_medium / PARALLEL_generic
   - FALLBACK_retry_specific / FALLBACK_alternate_medium / FALLBACK_generic
   - REPLACE_subtree_specific / UPDATE_params_specific
   - INSERT_position_first_specific
   Long thinking exceeds the 512-token thinking budget (the proxy then truncates). Iter-32 caps were tuned for an 8B model; a 35B with more elaborate chains-of-thought may benefit from `thinking_token_budget=768` or `1024`.
2. **Fast tool-arg-shape mismatches** (≈5 cases, 10-33s):
   - SCENE_remove_specific (10s) / SCENE_update_pose_specific (19s) / SCENE_update_size_medium (57s)
   - UPDATE_params_generic / REPLACE_simple_medium
   Different tool name or arg shape on first try. Worth grep-ing the JSONL bodies (proxy log) for the actual call shape vs schema.
3. **Generic-prompt under-spec** (2 cases): TREE_insert_runtime_generic, PARALLEL_generic — established weakness across all models.

## Why RedHat weights (not NVIDIA)
NVIDIA's `nvidia/Qwen3.6-35B-A3B-NVFP4` quant (produced by ModelOpt v0.44.0) ships `lm_head` as **W4A16_NVFP4** with these tensors:
```
lm_head.weight        U8  [248320, 1024]   ← NVFP4 packed
lm_head.weight_scale  F8  [248320, 128]
lm_head.weight_scale_2 F32 scalar
lm_head.input_scale    F32 scalar
```

vLLM 0.22's `Qwen3_5MoeForCausalLM` only registers `lm_head.weight` on `ParallelLMHead` — no quant method attached. Loader fails at 67% weight load with:
```
ValueError: There is no module or parameter named 'lm_head.input_scale'
in Qwen3_5MoeForCausalLM.
```

RedHat ships the same base model quantized via **llm-compressor** (the upstream tool for vLLM):
```
lm_head.weight        BF16 [248320, 2048]   ← native, vLLM-loadable
```
Format: `compressed-tensors / nvfp4-pack-quantized` (auto-detected from `config.json`'s `quantization_config`; no `--quantization` flag needed).

This is a swap-the-weights, not a recipe change. The NVIDIA Spark serving recipe (MTP K=3, `moe_backend:triton`, FP8 KV) applies cleanly to RedHat's weights — and is what this profile uses.

When vLLM ships a fix for `Qwen3_5MoeForCausalLM`'s `ParallelLMHead` to attach the NVFP4 quant method (it does for other classes), this profile should be re-tested with `nvidia/…` weights to compare quality.

### Iter-4 verification: NVIDIA's published Spark recipe still hits the gap (2026-05-30)
Tried `nvidia/Qwen3.6-35B-A3B-NVFP4` weights with NVIDIA's full DGX Spark recipe applied verbatim from the model card:
- `VLLM_USE_FLASHINFER_MOE_FP4=0` (explicit)
- `VLLM_FP8_MOE_BACKEND=flashinfer_cutlass`
- `FLASHINFER_DISABLE_VERSION_CHECK=1`
- `CUTE_DSL_ARCH=sm_110a` (NVIDIA's value `sm_121a` adjusted for Thor)
- `--quantization modelopt`
- `--moe-backend marlin` (NVIDIA-recommended, explicit)
- `--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'`

vLLM honored `--moe-backend marlin` (log: `Using 'MARLIN' NvFp4 MoE backend`) but the loader still failed at the same point (2/3 shards, 67%):
```
ValueError: There is no module or parameter named 'lm_head.input_scale'
in Qwen3_5MoeForCausalLM. Available: {'lm_head.weight'}
```
Crash path: `qwen3_5.py:525 → utils.py:337`. The MoE backend choice is irrelevant — the failure is in the model class's weight loader registration. **The NVIDIA model card explicitly recommends `vllm/vllm-openai:nightly`**; our v9 image (`vllm 0.22.1.dev0+g0b3ba88f1`) is older and lacks the fix.

**Action**: re-test `nvidia/…` weights when the v10/next vLLM image lands. File the gap upstream if it persists in nightly.

## Known footguns
- **`VLLM_USE_FLASHINFER_MOE_FP4=1` must NOT be set with this recipe.** When set, vLLM's NVFP4 MoE oracle rejects every candidate backend with `NotImplementedError`. Letting the oracle auto-pick lands on `VLLM_CUTLASS` which is faster on SM110.
- **GPU memory cleanup between launches**: after any vLLM crash, run `sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`. The crashed engine pins ~32 GiB of unified memory until released. (Already in `feedback_vllm_memory_leak`.)
- **DNS race on local-inference egress preset**: `host.openshell.internal` doesn't resolve immediately after the preset is applied. If `start_bridge` bails, retry after ~10s. (Noted in ROADMAP.)
- **Froggeric template** is required for 100% prefix-cache hit. The qwen3_coder tool-call parser is paired with it. Do not swap to qwen3_xml (designed for stock template) or hermes (designed for JSON-in-tags) without retesting.

## Fail timing breakdown
| Bucket | n | Total | Mean | Cases |
|---|---|---|---|---|
| **Long (≥150s, bridge transport timeout)** | **6** | **24 min** | 243s | PARALLEL_concurrent_medium (275s, 7 tool calls), FALLBACK_alternate_medium (275s, 2 calls), UPDATE_params_specific (254s, 1 call), REPLACE_subtree_specific (245s, 4 calls), INSERT_position_first_specific (225s, 13 calls), FALLBACK_retry_specific (181s, 4 calls) |
| Medium (60-150s, partial then refused) | 2 | 3.7 min | 111s | CUR_runtime_remove_then_restore_graspable (131s, 6 calls), TREE_insert_runtime_generic (91s, 0 calls) |
| Fast (30-60s, arg-shape mismatch) | 4 | 2.8 min | 41s | SCENE_update_size_medium, PARALLEL_generic, UPDATE_params_generic, FALLBACK_generic |
| Very-fast (<30s, refused expected tool) | 3 | 1 min | 20s | REPLACE_simple_medium (29s), SCENE_update_pose_specific (20s), SCENE_remove_specific (10s) |

4 of the 6 long fails returned `chat HTTP -1` — the bridge transport hit `ASSISTANT_TIMEOUT_S=300`. Two of these emitted **useful tool sequences** (7 and 13 calls) before being killed — the model was making progress, just slowly.

The 3 very-fast fails (<30s) emitted **zero tool calls** — pure prompt/schema mismatch, not capacity. Worth grep-ing `/tmp/manyforge-assistant-e2e/vllm-proxy.jsonl` for the actual emitted text to see what the model said instead.

## Tuning candidates (not yet evaluated)
1. **`ASSISTANT_TIMEOUT_S=600` + `OPENCLAW_PROXY_THINKING_TOKEN_BUDGET=768`** — would let the 6 long-fail cases finish their tool chains (current ASSISTANT_TIMEOUT_S=300 clips them at ~275s). Cost: ~24 min added to slowest fails.
2. **`OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS=4096`** with thinking_budget=1024 — bigger reasoning runway for the medium/long compositional cases.
3. **Drop MTP to K=2** if acceptance dips below ~70% under sustained load (NVIDIA reports K=3 at 85-94% acceptance on dense 27B; we haven't measured on this 35B MoE).
4. **`--reasoning-parser` off + `enable_thinking:false`** for tool-only modes (composer is happiest with tool-first responses; might fix the very-fast schema fails).

## Test plan for next vLLM image (v10+)

The whole point of trying NVIDIA's weights again: τ²-Bench Telecom 94.7 (NVFP4) vs whatever the RedHat llm-compressor variant achieves. If both load, run apples-to-apples comparison.

### Step 1 — verify the load gap is closed
Temporarily switch profile to `THOR_LAUNCH_MODEL_SOURCE="nvidia/Qwen3.6-35B-A3B-NVFP4"` with NVIDIA's full Spark recipe:
```sh
THOR_DOCKER_ENV_ARGS+=(
    "-e" "VLLM_USE_FLASHINFER_MOE_FP4=0"
    "-e" "VLLM_FP8_MOE_BACKEND=flashinfer_cutlass"
    "-e" "FLASHINFER_DISABLE_VERSION_CHECK=1"
    "-e" "CUTE_DSL_ARCH=sm_110a"
)
THOR_VLLM_ARGS+=(
    "--quantization" "modelopt"
    "--moe-backend" "marlin"
    # ... rest of NVIDIA card recipe
)
```
**Pass criterion**: weight load gets past 67% (3/3 shards) without `ValueError: There is no module or parameter named 'lm_head.input_scale'`. The MoE backend line in the log should still read `Using 'MARLIN' NvFp4 MoE backend`.

### Step 2 — verify generation works
Quick chat probe (5 token reply): should return a finite answer, no NaN/garbage. If broken sampling/quantization on lm_head: would see token IDs near edges of vocab, very long incoherent strings, or repeat loops.

### Step 3 — head-to-head vs RedHat on smoke corpus
Both variants share the same base Qwen3.6-35B-A3B, same NVFP4 weight quant family — comparable quality expected. Run the 66-case smoke corpus on both:
- **NVIDIA weights**: capture pass rate, first-try, latencies, any new failure modes
- **RedHat weights** (current 51/66 = 77.3% baseline) — re-run for noise envelope

### Step 4 — head-to-head on agent benchmarks
If quality looks comparable, run `tool-eval-bench` (TEB) and τ²-Bench to compare against NVIDIA's published numbers. The RedHat profile `qwen3.6-35b-a3b-nvfp4-mtp-fp8kv` previously scored TEB 93 — see if NVIDIA's variant matches NVIDIA's claimed τ²-Bench 94.7.

### Step 5 — decision
Keep whichever ships better quality at equal/better throughput. If NVIDIA's wins: rename profile back to `nvidia/...` weights. If RedHat ties or wins: keep current.

### Confirmed upstream fix — vLLM PR #42124 (merged 2026-05-26)
[PR #42124 "Add LM head quantization support for ModelOpt"](https://github.com/vllm-project/vllm/pull/42124), commit `6f5b533241`. Quoting PR body:
> "Add ModelOpt quantized lm_head support for checkpoints with `tie_word_embeddings=false`, where the LM head is exported as a separate quantized `ParallelLMHead`. **Passes `quant_config` into `ParallelLMHead` for Qwen3.5/Qwen3.6 and Nemotron-H causal LM models.**"

Companion still-open PR [#35660](https://github.com/vllm-project/vllm/pull/35660) names our error verbatim: *"weight loading to fail with `ValueError: There is no module or parameter named 'lm_head.input_scale'` when NVFP4 scale tensors were present in the checkpoint."*

**Status**: NOT in v0.22.0 (the release branch was cut just before #42124 landed). NVIDIA's "use `vllm/vllm-openai:nightly`" guidance reduces to "use any commit ≥ `6f5b533241`".

**Two consumption paths for v10**:
1. **Pin to a `main` SHA ≥ 2026-05-27** (clean upstream, picks up other fixes too)
2. **Cherry-pick #42124 onto v0.22.0** (small — `modelopt.py` + Qwen3.5/Nemotron-H model wiring; lower-risk for the Thor SM110a build)

### Other v0.22.0 fixes we already inherit (no action needed)
- PR #26345 — qwen3_xml tool parser bugfix
- PR #40820 — qwen3 reasoning/content streaming routing fix (when enable_thinking=false)
- PR #40700 + #41110 — streaming tool dispatch with `required` and named tool_choice primitives
- PR #43401 — `reasoning_effort` → `enable_thinking` mapping in Completions API
- DeepSeek V4 MTP (#43385), NemotronH non-MTP spec, MRv2 shared MTP weights

### Stack-side streaming gap (unrelated to v10)
v0.21.0+ already ships SSE streaming tool dispatch (PRs #40700, #41110). Our v0.22.0 has it. But our **stack does not consume it**:
- `manyforge/scripts/proxy/vllm-proxy.py:422-429` deliberately buffers SSE responses ("buffering preserves correctness; latency cost acceptable for smoke harness")
- `openclaw_assistant_bridge/adapter.py:754` hard-codes `"stream": False`
- `manyforge_assistant_bridge/bridge.py:1216-1223` parses tool_calls from full buffered response

If we want streaming tool dispatch, it's ~150-300 lines of Python on the bridges (proxy passthrough + bridge SSE event handler), independent of any vLLM bump.

## Files
- Profile: `serving/launch.sh` → case `qwen3.6-35b-a3b-nvfp4-nvidia`
- Defaults: `serving/config.sh` → `resolve_model_profile`
- Chat template: `serving/templates/qwen-fixed-froggeric.jinja`
- Iteration log: `/tmp/35b-iter-log/log.md`
- Smoke corpus report: `/tmp/35b-iter-log/full-smoke-35b.json`
- Smoke corpus log: `/tmp/35b-iter-log/full-smoke-35b.log`

## Iteration history
- **Iter 1** (NVIDIA weights, drop `VLLM_USE_FLASHINFER_MOE_FP4`): vLLM oracle picked MARLIN, loaded 67% then failed on `lm_head.input_scale` — vLLM gap for `ParallelLMHead` NVFP4.
- **Iter 2** (RedHat weights, kept `VLLM_USE_FLASHINFER_MOE_FP4=1`): same NotImplementedError. Env var must come off.
- **Iter 2b**: GPU memory still pinned from iter-1 crash. `sync + drop_caches` recovered.
- **Iter 3** (RedHat weights, env var off): full boot. vLLM oracle picks VLLM_CUTLASS. 22.6 GiB. 18.9 tok/s.
- **Iter 4** (full stack on :8050): bridge supervisor flips waiting→ready, P1 probe passes, full smoke 51/66 = 77.3%.
