# NemoClaw-Thor vLLM Recipe Migration Plan — Custom Image → Public Images

**Status:** Plan (not executed). **Date:** 2026-06-23. **Owner:** Marco Pastorio.
**Fail-closed invariant:** the custom image `nemoclaw-thor/vllm:latest` ("v9.1") is NOT retired until **every in-scope recipe** passes its automated functionality test on the public image it migrates to.

---

## 1. Goal & Scope

### 1.1 Goal
Move every vLLM recipe that currently runs on the custom Thor image `nemoclaw-thor/vllm:latest` onto **public** container images, preserving *every* functionality each recipe exposes (text, vision, audio, video, tools, reasoning, MTP/spec-decode, context length, FP8 KV, NVFP4/FP8 quant). For each migrated recipe, define an **automated test that proves each functionality still works** on the public image before the custom image is allowed to be retired.

### 1.2 In scope (11 vLLM recipes)
All 10 vLLM recipes default to `nemoclaw-thor/vllm:latest` because none of their `launch.sh` case branches override `THOR_VLLM_IMAGE` — they fall through to the global default at `launch.sh:20` (`THOR_VLLM_IMAGE="${THOR_VLLM_IMAGE:-nemoclaw-thor/vllm:latest}"`).

| Recipe | Headline functionalities | Target public image |
|---|---|---|
| `cosmos-reason2-2b` | text, vision (Qwen3-VL ViT), tools (hermes), reasoning (template+proxy) | `vllm/vllm-openai:nightly-aarch64` (pinned) |
| `cosmos-reason2-8b` | text, vision, tools (hermes), reasoning (qwen3), 256K | `vllm/vllm-openai:nightly-aarch64` (pinned) |
| `nemotron3-nano-4b-bf16` | text, tools (qwen3_coder), reasoning (nano_v3 host plugin), Mamba-2 hybrid | `vllm/vllm-openai:nightly-aarch64` (pinned) |
| `nemotron3-nano-30b-a3b-nvfp4` | text, tools (qwen3_coder), NVFP4 MoE, Mamba-2 hybrid | `vllm/vllm-openai:nightly-aarch64` (pinned) |
| `nemotron3-nano-omni-30b-a3b-nvfp4` | text, vision (C-RADIOv4-H), **audio (Parakeet)**, video (EVS), tools, NVFP4 MoE, think-OFF | `vllm/vllm-openai:nightly-aarch64` **+ vllm[audio] overlay** (pinned) |
| `nemotron3-nano-omni-30b-a3b-nvfp4-reasoning` | as above **+ reasoning (nemotron_v3), think-ON** | `vllm/vllm-openai:nightly-aarch64` **+ vllm[audio] overlay** (pinned) |
| `qwen3.6-35b-a3b-nvfp4-nvidia` | text, tools (qwen3_coder), reasoning (qwen3), **MTP K=3**, NVFP4 W4A16 | `vllm/vllm-openai:nightly-aarch64` (pinned) |
| `qwen3.5-9b-claude-distilled-nvfp4` | text, vision (BF16 ViT), tools (qwen3_xml), MTP K=1, NVFP4-MLP, reasoning OFF-by-design | `vllm/vllm-openai:nightly-aarch64` (pinned) |
| `qwen3.6-27b-fp8-mtp-kvfp8` | text, tools (qwen3_coder), reasoning (qwen3), **MTP K=3 (qwen3_next_mtp)**, native FP8 | `vllm/vllm-openai:nightly-aarch64` (pinned) |
| `gemma4-e4b-it` | text, vision (SigLIP2), audio (native), tools (gemma4), reasoning (gemma4) | `ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-thor` (pinned) |
| `gemma4-26b-a4b-it` | text, vision (SigLIP2), tools (gemma4), reasoning (gemma4), BF16 MoE | `ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-thor` (pinned) |

(11 distinct vLLM recipes — the two cosmos, two omni, and two gemma4 variants each count separately.)

### 1.3 Out of scope — GGUF recipes (excluded once, with reason)
The **6 GGUF recipes** (the `*-gguf` family, e.g. `gemma4-12b-it-gguf`, `launch.sh:762+`) are **excluded**: they already run on a **public** image, `ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-thor` (`THOR_LLAMACPP_IMAGE`, `launch.sh:773`), not on the custom vLLM image. They are llama.cpp lanes with no dependency on `nemoclaw-thor/vllm:latest`, so the migration goal does not touch them. They are not mentioned again.

### 1.4 Verified / inferred / untested split (honesty table)
This is the confidence basis for the rollout order in §5. "Verified" = booted on the public image **this session**; "inferred" = not individually booted but same architecture/family as a verified recipe (low risk); "untested" = neither booted nor a verified-family member (higher risk).

| Recipe | Status | Evidence |
|---|---|---|
| `cosmos-reason2-8b` | **VERIFIED** | Vision on nightly: TORCH_SDPA cleared vLLM #38411 (ViT PTX crash on sm110); correct spatial description; ~262 vision tokens through the ViT. |
| `qwen3.6-35b-a3b-nvfp4-nvidia` | **VERIFIED** | MTP on nightly: ~50 tok/s, 55–81% acceptance, tool-calling works. |
| `nemotron3-nano-omni-30b-a3b-nvfp4-reasoning` | **VERIFIED (partial)** | Booted 160s on nightly+vllm[audio] (no `--enforce-eager`); TEXT populated; IMAGE identified; AUDIO path accepted/encoded `input_audio` and the model responded about sound (440Hz tone — **transcription accuracy UNVERIFIED**). Long reasoning can hit `finish_reason=length` leaving `content` empty. |
| `nemotron3-nano-omni-30b-a3b-nvfp4` (base/think-OFF) | **INFERRED** (same checkpoint/image as the reasoning sibling; only the parser/thinking flags differ) | Family of the verified omni. |
| `cosmos-reason2-2b` | **INFERRED** | Same Qwen3-VL family / arch as verified cosmos-8b; #38411 ViT blocker cleared for the whole Qwen3-VL family. |
| `qwen3.5-9b-claude-distilled-nvfp4` | **INFERRED** | DeltaNet-hybrid VLM on the same nightly; ViT + NVFP4-modelopt path; not individually booted. |
| `nemotron3-nano-4b-bf16` | **INFERRED** | Nemotron-3-Nano text model family. |
| `nemotron3-nano-30b-a3b-nvfp4` | **INFERRED** | Nemotron-3-Nano hybrid-MoE text model family. |
| `qwen3.6-27b-fp8-mtp-kvfp8` | **INFERRED** | Native-FP8 Qwen3.6 text model; `qwen3_next_mtp` spec method is the one unverified item. |
| `gemma4-e4b-it` | **UNTESTED (higher risk)** | Rides NVIDIA `ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-thor`; SigLIP2 ViT + gemma4 reasoning/tool parsers + `triton_attn` (head_dim=512) all unverified by us. |
| `gemma4-26b-a4b-it` | **UNTESTED (higher risk)** | Same image; additionally `migration_risk: high` (BF16 MoE 128E/8A ~52GB fit, model-id existence). |

---

## 2. Target Images & Strategy

**Decision (the "v10" strategy): do NOT build vLLM from source.** Digest-pin a thin overlay on the stock public nightly. Two public images cover all in-scope recipes:

### 2.1 Image A — pinned stock nightly (+ optional vllm[audio] overlay)
**`vllm/vllm-openai:nightly-aarch64`** carries vLLM `0.23.1rc1.dev`, which includes `sm_110` in its CUDA-13 targets — so Thor/sm110 is a first-class target in stock.

- **`nightly-aarch64` is a MOVING tag.** It must be **digest-pinned** before adoption or vLLM/FlashInfer/parser versions silently drift between launches. Last observed digest this session:
  `vllm/vllm-openai@sha256:9eb0f4b6be6814dee6742e82f7be872232b943665faadbbcccf1856a5e807b28`.
  This digest is the migration baseline; re-pin (and re-run the full §4 suite) only deliberately.
- **vllm[audio] overlay.** Stock images do **not** ship the audio extra (`librosa`/`soundfile`). The two omni recipes need it for Parakeet. Bake it into the **digest-pinned overlay**, *not* a per-boot `pip install`.

**Vehicle: the existing `serving/docker/Dockerfile.overlay`** (`src/NemoClaw-Thor/serving/docker/Dockerfile.overlay`). It already has the right shape:
```dockerfile
ARG BASE_IMAGE=nemoclaw-thor/vllm:latest
FROM ${BASE_IMAGE}
RUN pip install --no-cache-dir instanttensor
```
Repoint it for v10:
```dockerfile
# v10: thin overlay on the digest-pinned public nightly. NO source build.
ARG BASE_IMAGE=vllm/vllm-openai@sha256:9eb0f4b6be6814dee6742e82f7be872232b943665faadbbcccf1856a5e807b28
FROM ${BASE_IMAGE}
# Audio extra (librosa/soundfile) is absent from stock; bake it once, not per boot.
RUN pip install --no-cache-dir "vllm[audio]" instanttensor
```
Build/tag (the overlay re-tags so `start-model.sh` picks it up automatically):
```
docker build -f docker/Dockerfile.overlay \
  --build-arg BASE_IMAGE=vllm/vllm-openai@sha256:9eb0f4b6...e807b28 \
  -t nemoclaw-thor/vllm:v10 docker/
```
Recipes that need audio (the two omni) point `THOR_VLLM_IMAGE` at the `:v10` overlay tag (or its own digest). Pure-text/vision recipes can point directly at the pinned **base** nightly digest (no overlay needed) — but using the single overlay tag for all nightly recipes is simpler and harmless (the audio extra is inert when unused). **Recommendation:** one overlay tag for all 9 nightly recipes; keep the base-nightly-only option documented for minimalism.

### 2.2 Image B — NVIDIA's purpose-built Gemma4 image
**`ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-thor`** for `gemma4-e4b-it` and `gemma4-26b-a4b-it`. This dedicated image almost certainly exists **because** stock vLLM lacks the gemma4 reasoning/tool parsers and the SigLIP2/head_dim=512 handling. It is **also a moving tag** and must be **digest-pinned** at migration. Note: jetson-ai-lab paired this image with a bg-digitalservices **NVFP4** quant; our recipes use **Google BF16** sources (`google/gemma-4-E4B-it`, `google/gemma-4-26B-A4B-it`) — confirm the parsers and BF16 load on this image (untested by us; see §6).

### 2.3 Why NVFP4 recipes lose nothing on stock
On sm110, native FP4 MoE is gated to the SM100 family upstream, so **NVFP4 MoE falls back to Marlin weight-only on BOTH the custom and the stock image**. Decode is memory-bandwidth-bound and MoE (few active params) decodes fast regardless. Native FP8 (`CutlassFp8BlockScaledMM` path via the disabled-kernels workaround) and BF16 run natively. Net: the NVFP4 recipes (`*-nvfp4*`, `qwen3.6-35b…nvfp4`, `qwen3.5-9b…nvfp4`) are **already Marlin on the custom image** and incur **no additional loss** moving to stock.

---

## 3. Per-Image Migration Sections

For each recipe: current→target image, **exact `launch.sh` change** (what flips, what stays), a functionality-preservation checklist, and sm110 caveats. The single mechanical change shared by all 9 nightly recipes is **setting `THOR_VLLM_IMAGE` to the pinned image** — currently none of them set it. The simplest implementation is to change the **global default** at `launch.sh:20`:

```sh
# launch.sh:20 (was: nemoclaw-thor/vllm:latest)
THOR_VLLM_IMAGE="${THOR_VLLM_IMAGE:-nemoclaw-thor/vllm:v10}"   # v10 = pinned-nightly+audio overlay
```
…and have the two gemma4 branches **explicitly override** `THOR_VLLM_IMAGE` to the pinned gemma4 digest (they currently rely on the default). All `THOR_VLLM_ARGS`, `THOR_DOCKER_ENV_ARGS`, mounts, and `THOR_TARGET_*` settings **stay byte-for-byte the same** unless a caveat below flags a flag-name reconciliation. The launch.sh/config.sh contracts are the source of truth and do not change except the image pointer.

---

### GROUP A — `vllm/vllm-openai:nightly-aarch64` (+ vllm[audio] overlay), digest-pinned

#### A1. `cosmos-reason2-2b` (INFERRED)
- **Image:** `nemoclaw-thor/vllm:latest` → pinned nightly (`:v10`). Branch does not set `THOR_VLLM_IMAGE`; picks up the new default.
- **launch.sh changes:** ONLY the image pointer. **Args that STAY (unchanged):** `--attention-backend flashinfer`, `--enforce-eager`, `--mm-encoder-attn-backend TORCH_SDPA`, `--kv-cache-dtype fp8`, `--max-num-batched-tokens 8192`, `--enable-auto-tool-choice`, `--tool-call-parser hermes`, `--gpu-memory-utilization 0.12`, `--max-model-len 32768`, `--max-num-seqs 2`, `--compilation-config {"custom_ops":["-quant_fp8",...]}`, `--served-model-name cosmos-reason2-2b cosmos-reason2-8b`. **No `--reasoning-parser` is present and none is added** (this branch carries reasoning via the bundled `chat_template.json` + proxy promotion only).
- **Preservation checklist:** text ✓; vision via `--mm-encoder-attn-backend TORCH_SDPA` (Qwen3-VL ViT head_dim=64, #38411) ✓ — **this is the cleared blocker**; tools via hermes parser (Cosmos emits hermes-style `<tool_call>{...}</tool_call>`) ✓; reasoning via auto-loaded bundled `chat_template.json` + `OPENCLAW_PROXY_PROMOTE_REASONING_TO_CONTENT=1` (start-model.sh:90) + `OPENCLAW_PROXY_UNWRAP_TOOL_CALL_ARGS=1` (start-model.sh:100) ✓; context 32768 / FP8 KV ✓; BF16 native (no quant flag) ✓; no spec-decode (do not add) ✓.
- **sm110 caveats:** confirm hermes parser present on the pinned nightly; confirm the bundled `chat_template.json` auto-loads without `--chat-template`; confirm FlashInfer FP8-KV has sm_110a kernels in the image JIT cache; confirm `VLLM_DISABLED_KERNELS` + `ENABLE_TRIATTENTION=0` env switches honored. **Reconcile** the stale `start-model.sh:82-83` comment that calls this a "qwen3 reasoning-parser" profile — the 2B branch sets **no** parser; decide template+proxy-only stays (recommended; matches verified cosmos-8b vision behavior) vs adding an explicit parser.

#### A2. `cosmos-reason2-8b` (**VERIFIED** — vision)
- **Image:** custom → pinned nightly (`:v10`).
- **launch.sh changes:** image pointer only. **Args STAY:** `--attention-backend flashinfer`, `--enforce-eager`, `--mm-encoder-attn-backend TORCH_SDPA`, `--kv-cache-dtype fp8`, `--max-num-batched-tokens 8192`, `--enable-auto-tool-choice`, `--tool-call-parser hermes`, `--override-generation-config {"temperature":0.2,"top_p":0.95}`, `--default-chat-template-kwargs {"enable_thinking":true}`, **`--reasoning-parser qwen3`**, `--gpu-memory-utilization 0.35`, `--max-model-len 262144`, `--max-num-seqs 3`, `--compilation-config …`, `--served-model-name ${THOR_MODEL_ID} cosmos-reason2-8b`. **De-dup opportunity (optional, not required):** `--kv-cache-dtype fp8` is appended twice (branch + common block) and `--max-num-batched-tokens` may double; vLLM is last-wins and values agree, so safe to leave, but a cleanup is welcome.
- **Preservation checklist:** text ✓; vision (TORCH_SDPA, **VERIFIED live on nightly** — correct spatial description, ~262 vision tokens) ✓; tools hermes ✓; reasoning qwen3 — **routes CoT to `message.reasoning_content`, the answer to `content`** (the qwen3 parser was added to stop `<think>` from polluting `content` and breaking hermes tool extraction) ✓; thinking-on ✓; 256K / FP8 KV ✓.
- **sm110 caveats:** gated HF repo — **HF_TOKEN required**; re-verify hermes still matches Cosmos's tool format; **reasoning-channel client issue applies** (see §3 note below) — downstream must read `reasoning_content` for CoT and `content` for the answer; re-validate `--gpu-memory-utilization 0.35` KV-pool fit at 256K on the new image (watch the boot log `GPU KV cache size: NNN tokens`).

#### A3. `nemotron3-nano-4b-bf16` (INFERRED)
- **Image:** custom → pinned nightly (`:v10`).
- **launch.sh changes:** image pointer only. **Args STAY:** `--trust-remote-code`, `--mamba_ssm_cache_dtype float32`, `--enable-auto-tool-choice`, `--tool-call-parser qwen3_coder`, **`--reasoning-parser-plugin /workspace/mods/nano_v3_reasoning_parser.py` + `--reasoning-parser nano_v3`**, `--override-generation-config {"temperature":0.6,"top_p":0.95}`, `--default-chat-template-kwargs {"enable_thinking":false}`, `--max-num-batched-tokens 8192`, plus common `--max-model-len 65536`, `--kv-cache-dtype fp8`, `--max-num-seqs 8`, `--gpu-memory-utilization 0.40`.
- **CRITICAL mount that STAYS:** `-v ${THOR_MODS_HOST_DIR}:/workspace/mods:ro` — the `nano_v3` parser is a **host-mounted plugin**, not baked into any image, so it is image-independent **as long as the public image still supports `--reasoning-parser-plugin` external loading**.
- **Preservation checklist:** text + hybrid Mamba-2 via `--trust-remote-code` ✓; tools qwen3_coder ✓; reasoning nano_v3 host-plugin (thinking OFF by default) ✓; FP8 KV ✓; BF16 native ✓; 65536 ctx (conservative; native is 262144) ✓.
- **sm110 caveats:** confirm the nightly bundles Nemotron-3-Nano hybrid Mamba-2 support and loads the HF custom modeling code under `--trust-remote-code`; **confirm `--mamba_ssm_cache_dtype` (underscore form) is still accepted** (stock may expect `--mamba-ssm-cache-dtype`) — reconcile flag name if boot rejects it; confirm `--reasoning-parser-plugin` external-file loading is supported; re-confirm the 2026-06-01 finding that combining `nano_v3` reasoning + `qwen3_coder` tools does **not** break tool calling (`P1_wrap_root_specific`) via a smoke run.

#### A4. `nemotron3-nano-30b-a3b-nvfp4` (INFERRED)
- **Image:** custom → pinned nightly (`:v10`).
- **launch.sh changes:** image pointer only. **Args STAY:** `--trust-remote-code`, `--enable-auto-tool-choice`, `--tool-call-parser qwen3_coder`, `--override-generation-config {"temperature":0.6,"top_p":0.95}`, `--default-chat-template-kwargs {"enable_thinking":false}`, `--max-num-batched-tokens 8192`, common `--max-model-len 65536`, `--kv-cache-dtype fp8`, `--max-num-seqs 8`, `--gpu-memory-utilization 0.50`. **Env STAYS:** `-e VLLM_USE_FLASHINFER_MOE_FP16=0`.
- **Preservation checklist:** text ✓; tools qwen3_coder ✓; reasoning = **deliberate absence of `--reasoning-parser`** + `enable_thinking:false` → reasoning stays inline in `content` (do NOT add a reasoning parser — HF discussion #3: tool-call + nano_v3 reasoning together breaks tools) ✓; NVFP4 MoE auto-detected → Marlin weight-only on sm110 (same as custom) ✓; FP8 KV ✓.
- **sm110 caveats:** confirm `qwen3_coder` present; confirm `--trust-remote-code` loads the hybrid Mamba-2/attn/MoE arch; confirm `VLLM_USE_FLASHINFER_MOE_FP16=0` still honored and the gated FlashInfer MoE path still exists; **do not introduce a reasoning parser on migration** (`THOR_TARGET_MODEL_REASONING=true` is informational only).

#### A5. `nemotron3-nano-omni-30b-a3b-nvfp4` (base, think-OFF) (INFERRED — family of verified)
- **Image:** custom (v8, no audio extra) → **pinned nightly + vllm[audio] overlay (`:v10`)**.
- **launch.sh changes:** image pointer only. **Args STAY:** `--trust-remote-code`, `--max-num-batched-tokens 8192`, `--enable-auto-tool-choice`, `--tool-call-parser qwen3_coder`, `--override-generation-config '{"temperature":0.6,"top_p":0.95}'`, `--default-chat-template-kwargs '{"enable_thinking":false}'`, `--enable-prefix-caching`, common `--max-model-len 262144`, `--kv-cache-dtype fp8`, `--max-num-seqs 16`, `--gpu-memory-utilization 0.50`, `--compilation-config …`. **MUST preserve the ABSENCE of `--reasoning-parser nemotron_v3`** and keep `enable_thinking:false` — re-enabling either reverts to the -reasoning sibling and empties `message.content`. **Env STAYS:** `-e VLLM_USE_FLASHINFER_MOE_FP16=0`. **Do NOT add `--moe-backend`** (vLLM's NvFP4 oracle auto-picks; triton + flashinfer_cutlass both rejected on sm110). **Do NOT add `--enforce-eager`** (verified not needed on nightly).
- **NEW on migration (an ADD, not a regression):** **audio (Parakeet) lights up** via the vllm[audio] overlay; it was baked-out of the v8 custom image.
- **⚠ REVISION PIN REQUIRED:** the HF `main` ref drifted to a **config-only PARTIAL snapshot (missing `modeling.py`)**, and `HF_HUB_OFFLINE` then crashed `trust_remote_code` on the missing custom-code file. **Fix + standing policy:** pin a COMPLETE `--revision` (e.g. `b5a7a5e3da84cd3db76bc9f4f1e2474fa14a63c3`) for the omni source and do not let offline mode block the custom-code fetch. (Source is currently passed with no `--revision`; revision handling is only a best-effort drift WARNING at `launch.sh:1286-1296`.)
- **Preservation checklist:** text ✓; vision C-RADIOv4-H (verify ViT attention on sm110; may need `--mm-encoder-attn-backend TORCH_SDPA` — currently NOT set) ✓/⚠; **audio Parakeet (ADD)** ✓; video EVS ✓; tools qwen3_coder ✓; reasoning DELIBERATELY DISABLED (keep stripped) ✓; NVFP4→Marlin ✓; 256K / FP8 KV / prefix-caching ✓; sampling T=0.6/top_p=0.95 (load-bearing: prevents the 80+ null-arg tool-call loop) ✓.
- **sm110 caveats:** verify vllm[audio] overlay ABI matches the nightly's vLLM version; re-check whether the v8 cuDNN mismatch returns on the public base (if FlashInfer autotuner fails on boot, re-add `--kernel-config autotune-disable`); confirm C-RADIO ViT works or add TORCH_SDPA.

#### A6. `nemotron3-nano-omni-30b-a3b-nvfp4-reasoning` (**VERIFIED, partial**)
- **Image:** custom (v8.1, audio baked) → **pinned nightly + vllm[audio] overlay (`:v10`)**.
- **launch.sh changes:** image pointer only. **Args STAY:** all of A5's *plus the distinguishing two*: **`--reasoning-parser nemotron_v3`** and **`--default-chat-template-kwargs '{"enable_thinking":true}'`**. Common `--max-model-len 262144`, `--kv-cache-dtype fp8`, `--max-num-seqs 16`, `--gpu-memory-utilization 0.50`. **Env STAYS:** `-e VLLM_USE_FLASHINFER_MOE_FP16=0`.
- **⚠ Same REVISION PIN requirement as A5** — pin a complete `--revision`.
- **Verified live:** booted 160s; TEXT populated; IMAGE identified; AUDIO accepted/encoded a 440Hz tone and the model responded about sound. **Caveats from the live run:** long reasoning can hit `finish_reason=length` leaving `content` empty; **transcription ACCURACY unverified** (tone only).
- **Preservation checklist:** text ✓; vision ✓ (verified identifies image); audio ✓ (path verified, accuracy not); video EVS ✓; tools qwen3_coder ✓; reasoning nemotron_v3 with thinking-ON → `<think>…</think>` split into **`message.reasoning_content`** ✓; NVFP4→Marlin / FP8 KV / 256K / prefix-caching ✓.
- **sm110 + reasoning-channel caveat (applies):** with thinking ON, the answer/CoT route to `reasoning_content`; **downstream bridges must read both `content` and `reasoning_content`** (the 2026-05-06 `openclaw_assistant_bridge` read only `content` and would DROP reasoning). The §4 harness must score this profile on the reasoning channel.

#### A7. `qwen3.6-35b-a3b-nvfp4-nvidia` (**VERIFIED** — MTP)
- **Image:** custom (v9.1, post-PR#42124) → pinned nightly (`:v10`).
- **launch.sh changes:** image pointer only. **Args STAY:** `--quantization modelopt`, `--moe-backend marlin`, `--kv-cache-dtype fp8`, `--attention-backend flashinfer`, `--enforce-eager`, `--language-model-only`, `--enable-prefix-caching`, `--enable-chunked-prefill`, `--async-scheduling`, `--max-num-batched-tokens 8192`, **`--reasoning-parser qwen3`**, `--enable-auto-tool-choice`, `--tool-call-parser qwen3_coder`, `--default-chat-template-kwargs {"enable_thinking":true}`, **`--speculative-config {"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}`**, `--trust-remote-code`, `--gpu-memory-utilization 0.55`, `--max-model-len 262144`, `--max-num-seqs 4`. **Chat-template mount + flag STAY:** `--chat-template /opt/nemoclaw-thor/templates/qwen-fixed-froggeric.jinja` (host-mounted, so image-independent). **Env STAYS (all load-bearing for sm110):** `VLLM_USE_FLASHINFER_MOE_FP4=0`, `VLLM_FP8_MOE_BACKEND=flashinfer_cutlass`, `FLASHINFER_DISABLE_VERSION_CHECK=1`, `CUTE_DSL_ARCH=sm_110a`, `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`, `VLLM_USE_FLASHINFER_MOE_FP16=0`. **Drop/treat-as-inert:** `VLLM_MODS=sm110a-fp4-dsl-unlock` is a **custom-image mod-patch hook** the public nightly will not honor — it is INERT for the W4A16 path, so acceptable, but flag it as a no-op on stock.
- **Verified live on nightly:** ~50 tok/s, 55–81% MTP acceptance, tool-calling works.
- **Preservation checklist:** text (LM-only, ViT dropped) ✓; tools qwen3_coder — **must stay qwen3_coder** (froggeric template emits native XML; do NOT swap to qwen3_xml/hermes) ✓; reasoning qwen3 → `message.reasoning` ✓; MTP K=3 with `moe_backend:triton` (BF16 drafter avoids the SM100-only CUTLASS tile crash) ✓; NVFP4 W4A16 → Marlin (forced) ✓; 256K / FP8 KV ✓.
- **sm110 caveats:** **confirm the pinned digest is post-PR#42124** (LM-head ModelOpt support) — without it the load crashes at `lm_head.input_scale`; confirm `qwen3`/`qwen3_coder` parsers in the registry; confirm the `speculative-config` schema (`method=mtp`, `moe_backend:triton`) is accepted; verify the SM110 env names exist in stock; re-verify the MTP perf number post-swap. This recipe is the manyforge orchestration anchor — re-validate the smoke harness end-to-end before promoting.

#### A8. `qwen3.5-9b-claude-distilled-nvfp4` (INFERRED)
- **Image:** custom → pinned nightly (`:v10`).
- **launch.sh changes:** image pointer only. **Args STAY:** `--attention-backend flashinfer`, `--quantization modelopt`, `--enable-auto-tool-choice`, `--tool-call-parser qwen3_xml`, `--enable-prefix-caching`, `--mm-encoder-attn-backend TORCH_SDPA`, `--max-num-batched-tokens 8192` (note: branch **comment says 4096 but code is 8192** — 8192 is authoritative), `--speculative-config {"method":"mtp","num_speculative_tokens":1}`, `--gpu-memory-utilization 0.4`, `--max-model-len 131072`, `--kv-cache-dtype fp8`, `--max-num-seqs 8`, `--chat-template /opt/nemoclaw-thor/templates/qwen3-tool-call-compat-nothink.jinja` (host-mounted). **Env STAYS:** `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`. **MUST stay ABSENT:** `--language-model-only` (so the BF16 ViT loads) and any `--reasoning-parser` (reasoning DISABLED-BY-DESIGN; the no-think template suppresses `<think>`; adding a parser would split content into `message.reasoning` and starve the embedded agent).
- **Preservation checklist:** text ✓; vision (BF16 ViT, TORCH_SDPA, #38411) ✓; tools qwen3_xml ✓; MTP K=1 ✓; NVFP4-MLP → Marlin ✓; 131072 / FP8 KV / prefix-caching ✓; reasoning OFF-by-design (no parser, no-think template, `THOR_TARGET_PROXY_FORCE_ENABLE_THINKING` intentionally unset) ✓.
- **sm110 caveats:** confirm `qwen3_xml` present; confirm DeltaNet-hybrid (linear+full attention) loads and `--attention-backend flashinfer` covers both layer types on sm110; confirm MTP honored for this hybrid arch; confirm TORCH_SDPA still clears #38411; confirm the no-think template mount path is still mountable.

#### A9. `qwen3.6-27b-fp8-mtp-kvfp8` (INFERRED)
- **Image:** custom → pinned nightly (`:v10`).
- **launch.sh changes:** image pointer only. **Args STAY:** `--attention-backend flashinfer`, `--language-model-only`, **`--reasoning-parser qwen3`**, `--enable-auto-tool-choice`, `--tool-call-parser qwen3_coder`, `--enable-prefix-caching`, `--max-num-batched-tokens 32768`, **`--speculative-config {"method":"qwen3_next_mtp","num_speculative_tokens":3}`**, `--gpu-memory-utilization 0.8`, `--max-model-len 262144`, `--kv-cache-dtype fp8`, `--max-num-seqs 9`, `--compilation-config …`, `--chat-template …/qwen-fixed-froggeric.jinja` (host-mounted). **Env STAYS (load-bearing):** `VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel,CutlassFp8BlockScaledMMKernel` (routes FP8 GEMM off Cutlass to dodge Xid 43 on sm110) + `ENABLE_TRIATTENTION=0`. (Note: `config.sh:322-323` comment says "MTP N=1" but the flag is K=3 — the flag value 3 is authoritative.)
- **Preservation checklist:** text + native FP8 (head_dim=256 forces FlashInfer attention; FA2 crashes sm110) ✓; tools qwen3_coder (coupled with the froggeric template) ✓; reasoning qwen3 → `message.reasoning_content` ✓; **MTP via `qwen3_next_mtp` — HIGHEST-risk item** ✓-pending; 256K / FP8 KV / prefix-caching ✓.
- **sm110 caveats:** **confirm `qwen3_next_mtp` spec method is registered in the pinned nightly** (if absent, the headline MTP feature fails); confirm `VLLM_DISABLED_KERNELS` honored and FlashInfer sm_110a FP8-GEMM JIT kernels exposed (else Xid 43 returns); confirm `ENABLE_TRIATTENTION=0` still recognized; confirm `--language-model-only` and `--attention-backend flashinfer` flag names unchanged.

---

### GROUP B — `ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-thor`, digest-pinned (UNTESTED — highest risk)

#### B1. `gemma4-e4b-it` (UNTESTED; migration_risk medium)
- **Image:** custom → **pinned `ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-thor`**. The branch must now **explicitly set `THOR_VLLM_IMAGE`** to the pinned gemma4 digest (it currently inherits the global default; with the §3 default flipped to nightly, this branch MUST override).
- **launch.sh changes:** **set `THOR_VLLM_IMAGE` (gemma4 digest) in-branch** + everything else STAYS. **Args STAY:** `--attention-backend triton_attn` (head_dim=512 — FlashInfer unsupported; must NOT revert to flashinfer), `--reasoning-parser gemma4`, `--enable-auto-tool-choice`, `--tool-call-parser gemma4`, `--enable-prefix-caching`, `--mm-encoder-attn-backend TORCH_SDPA` (SigLIP2 ViT, #38411), `--max-num-batched-tokens 4096` (≥ MM-item budget floor), `--gpu-memory-utilization 0.4`, `--max-model-len 131072`, `--kv-cache-dtype fp8`, `--max-num-seqs 12`. **MUST stay ABSENT:** `--language-model-only` (so vision + native audio encoders load). **Env STAYS:** `VLLM_DISABLED_KERNELS`, `ENABLE_TRIATTENTION=0`.
- **Preservation checklist:** text ✓; vision SigLIP2 (TORCH_SDPA) ⚠untested; **audio native (full MM load, no flag — untested)** ⚠; tools gemma4 ⚠ (parser must exist in image registry); reasoning gemma4 → `message.reasoning` ⚠; 131072 / FP8 KV / prefix-caching ✓; BF16 native ✓.
- **sm110 + reasoning-channel caveats:** **confirm the image bundles BOTH the gemma4 reasoning parser AND the gemma4 tool-call parser** (likely present since the image is gemma4-purpose-built, but UNVERIFIED — if absent, tools+reasoning silently degrade); confirm `triton_attn` valid for head_dim=512; confirm `--mm-encoder-attn-backend TORCH_SDPA` honored and SigLIP2 ViT does not PTX-crash; confirm audio actually works (declared, never validated by us); reasoning routes to `message.reasoning` so the proxy must read that channel.

#### B2. `gemma4-26b-a4b-it` (UNTESTED; **migration_risk HIGH**)
- **Image:** custom → **pinned `gemma4-jetson-thor`**; **set `THOR_VLLM_IMAGE` in-branch** (same override requirement as B1).
- **launch.sh changes:** set `THOR_VLLM_IMAGE` + everything else STAYS. **Args STAY:** `--attention-backend triton_attn`, `--reasoning-parser gemma4`, `--enable-auto-tool-choice`, `--tool-call-parser gemma4`, `--enable-prefix-caching`, `--mm-encoder-attn-backend TORCH_SDPA`, `--gpu-memory-utilization 0.80`, `--max-model-len 262144`, `--kv-cache-dtype fp8`, `--max-num-seqs 17`. (This branch sets NO `--max-num-batched-tokens` and NO `--chat-template` — unchanged.) **Env STAYS:** `VLLM_DISABLED_KERNELS`, `ENABLE_TRIATTENTION=0`.
- **Preservation checklist:** text ✓; vision SigLIP2 (TORCH_SDPA) ⚠; tools gemma4 ⚠; reasoning gemma4 → `message.reasoning` ⚠; 262144 / FP8 KV / prefix-caching ✓; **BF16 MoE 128E/8A (~52GB) fit at gpu-mem-util 0.80 + 256K + 17 seqs — unverified, primary HIGH-risk concern** ⚠.
- **sm110 caveats:** confirm gemma4 parsers in this specific image; **confirm `google/gemma-4-26B-A4B-it` HF repo exists / matches the image's expected weights** (model-id existence is an open question); validate the BF16 MoE memory fit on the target image's layout; confirm the Thor fused-MoE configs mount (`/opt/nemoclaw-thor/moe-configs`) is honored or unnecessary on this image.

---

### Cross-cutting note — reasoning-channel client issue (applies to A2, A6, A7, A9, B1, B2)
Recipes with `--reasoning-parser` (cosmos→qwen3, omni→nemotron_v3, qwen3.6→qwen3, gemma4→gemma4) route the **answer to `message.reasoning`/`reasoning_content`, leaving `message.content` null/empty for short replies**. OpenAI clients that read only `content` (e.g. Cline) see "empty response." **Mitigation, per client:** (a) read `reasoning`; or (b) drop `--reasoning-parser` (answer lands in `content` with literal `<think>` tags); or (c) thinking-off for content-only clients. **The ManyForge bridge MUST read `reasoning`.** The §4 harness scores these profiles on the reasoning channel (see §4d).

---

## 4. Automated Validation Test (CORE DELIVERABLE)

**Goal:** prove, per recipe, that **each declared functionality works on the public image** — fail-closed gate for retiring the custom image. The harness launches the container on the public image, polls `/v1/models` for readiness, runs the functionality probes matched to that recipe, asserts pass/fail, tears down, and aggregates a machine-readable report.

### 4a. Probe matrix (recipe × functionality)
Legend: ✔ = probe required; — = N/A; ⚑ = reasoning-channel probe (assert answer in `reasoning`/`content` per §4d); ★ = ADD (new on migration).

| Recipe | text | vision | audio | video | tools | reasoning | MTP/spec | ctx-sanity | FP8-KV boot |
|---|---|---|---|---|---|---|---|---|---|
| cosmos-reason2-2b | ✔ | ✔ | — | — | ✔ | ⚑(template) | — | ✔ | ✔ |
| cosmos-reason2-8b | ✔ | ✔ | — | — | ✔ | ⚑ qwen3 | — | ✔(256K) | ✔ |
| nemotron3-nano-4b-bf16 | ✔ | — | — | — | ✔ | ⚑ nano_v3(off-default) | — | ✔ | ✔ |
| nemotron3-nano-30b-a3b-nvfp4 | ✔ | — | — | — | ✔ | (inline, no parser) | — | ✔ | ✔ |
| omni-…-nvfp4 (base) | ✔ | ✔ | ✔★ | ✔ | ✔ | (stripped — assert content non-empty) | — | ✔(256K) | ✔ |
| omni-…-nvfp4-reasoning | ✔ | ✔ | ✔ | ✔ | ✔ | ⚑ nemotron_v3 | — | ✔(256K) | ✔ |
| qwen3.6-35b-a3b-nvfp4-nvidia | ✔ | — | — | — | ✔ | ⚑ qwen3 | ✔(K=3 accept-rate) | ✔(256K) | ✔ |
| qwen3.5-9b-claude-distilled | ✔ | ✔ | — | — | ✔ | (off-by-design — assert content non-empty, no `<think>`) | ✔(K=1) | ✔ | ✔ |
| qwen3.6-27b-fp8-mtp-kvfp8 | ✔ | — | — | — | ✔ | ⚑ qwen3 | ✔(qwen3_next_mtp K=3) | ✔(256K) | ✔ |
| gemma4-e4b-it | ✔ | ✔ | ✔★ | — | ✔ | ⚑ gemma4 | — | ✔ | ✔ |
| gemma4-26b-a4b-it | ✔ | ✔ | — | — | ✔ | ⚑ gemma4 | — | ✔(256K) | ✔ |

**Probe definitions (deterministic, low-temp where the recipe allows):**
- **text:** prompt for a fixed short fact; assert non-empty answer in the correct channel (§4d).
- **vision (describe-a-known-image, assert shapes/positions):** send a bundled fixture image with known content (e.g. a red square top-left, blue circle bottom-right on white); assert the response names both shapes AND their relative positions (string-match on {red/square/left} ∧ {blue/circle/right}); also assert the boot log / usage shows vision tokens (>0) flowed through the ViT.
- **audio (transcribe):** send a bundled WAV via `input_audio`. **Two-tier scoring:** (1) **path probe (gating)** — assert the request is accepted, encoded, and yields a non-empty audio-grounded response (this is what is VERIFIED for omni today); (2) **accuracy probe (non-gating, flagged)** — a short spoken clip with known transcript; fuzzy-match WER < threshold. Accuracy starts **WARN-only** because only the path is proven (440Hz tone); promote to gating once a real spoken-clip baseline is established.
- **video:** send a tiny multi-frame clip (EVS); assert a non-empty grounded response (path probe).
- **tools (tool-call):** define one function (`get_weather(city)`); prompt to call it; assert `choices[0].message.tool_calls[0].function.name == "get_weather"` and arguments parse as JSON with the expected key. (Parser-specific: hermes/qwen3_coder/qwen3_xml/gemma4 — the assertion is parser-agnostic at the `tool_calls` level.)
- **reasoning-channel (⚑):** see §4d.
- **MTP/spec-decode acceptance-rate:** scrape vLLM Prometheus `/metrics` (`vllm:spec_decode_num_accepted_tokens_total` / `…num_draft_tokens_total`) or the boot/serve logs for the acceptance ratio after a fixed generation; assert ratio above a floor (e.g. ≥ 0.5; qwen3.6-35b verified 0.55–0.81). For recipes without MTP, assert the spec-decode metric is **absent/zero** (no accidental introduction).
- **context-length sanity:** request `/v1/models` and assert `max_model_len` equals the recipe's `THOR_TARGET_MAX_MODEL_LEN` (32768 / 65536 / 131072 / 262144); optionally send a prompt sized near the window and assert no overflow error.
- **FP8-KV boot:** assert the boot log contains the KV-cache line and `kv_cache_dtype=fp8` (and capture `GPU KV cache size: NNN tokens` for headroom tracking, §6).

### 4b. Runnable harness skeleton (one representative recipe: `cosmos-reason2-8b`)
Includes the readiness/timeout loop and the reasoning-vs-content extraction. Lives at `serving/test/validate_public_images.py` (see §4c).

```python
#!/usr/bin/env python3
"""validate_public_images.py — fail-closed functionality harness for the public-image migration.
Representative slice for cosmos-reason2-8b (text + vision + tools + reasoning-channel + ctx).
Full version is manifest-driven (see serving/test/manifests/*.yaml)."""
import base64, json, subprocess, sys, time, urllib.request, urllib.error

BASE = "http://127.0.0.1:8000"

def _post(path, payload, api_key=None, timeout=120):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def wait_ready(model_id, timeout_s=900, poll_s=5):
    """Poll /v1/models until the served model appears or timeout. Returns the model record."""
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as r:
                models = json.load(r).get("data", [])
            ids = {m["id"] for m in models}
            if model_id in ids:
                return next(m for m in models if m["id"] == model_id)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_err = e
        time.sleep(poll_s)
    raise TimeoutError(f"{model_id} not ready in {timeout_s}s (last: {last_err})")

def extract_answer(choice):
    """Reasoning-vs-content extraction. Returns (answer_text, channel, content_was_empty)."""
    msg = choice["message"]
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    if content:
        return content, "content", False
    if reasoning:
        # content empty but reasoning present: the model DID answer, just in the reasoning channel;
        # count as a pass but flag it so a content-only client (e.g. Cline) regression stays visible.
        return reasoning, "reasoning", True
    return "", "none", True

def probe_text(model):
    r = _post("/v1/chat/completions", {"model": model, "max_tokens": 64, "temperature": 0,
              "messages": [{"role": "user", "content": "Reply with exactly: NEMOCLAW_OK"}]})
    ans, ch, empty = extract_answer(r["choices"][0])
    ok = "NEMOCLAW_OK" in ans
    return {"probe": "text", "pass": ok, "channel": ch, "content_empty": empty, "answer": ans[:120]}

def probe_vision(model, image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    r = _post("/v1/chat/completions", {"model": model, "max_tokens": 256, "temperature": 0,
              "messages": [{"role": "user", "content": [
                  {"type": "text", "text": "Name the two shapes and where each is."},
                  {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]})
    ans, ch, empty = extract_answer(r["choices"][0])
    low = ans.lower()
    shapes_ok = ("square" in low and "circle" in low)
    pos_ok = (("left" in low or "top" in low) and ("right" in low or "bottom" in low))
    vis_tokens = (r.get("usage", {}).get("prompt_tokens_details", {}) or {}).get("image_tokens")
    return {"probe": "vision", "pass": bool(shapes_ok and pos_ok), "channel": ch,
            "content_empty": empty, "image_tokens": vis_tokens, "answer": ans[:200]}

def probe_tools(model):
    tools = [{"type": "function", "function": {"name": "get_weather",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                             "required": ["city"]}}}]
    r = _post("/v1/chat/completions", {"model": model, "max_tokens": 256, "temperature": 0,
              "tools": tools, "tool_choice": "auto",
              "messages": [{"role": "user", "content": "What's the weather in Turin? Use the tool."}]})
    tcs = r["choices"][0]["message"].get("tool_calls") or []
    ok = False; parsed = None
    if tcs:
        fn = tcs[0]["function"]
        try:
            parsed = json.loads(fn.get("arguments") or "{}")
            ok = (fn.get("name") == "get_weather" and "city" in parsed)
        except json.JSONDecodeError:
            ok = False
    return {"probe": "tools", "pass": ok, "tool_call": (tcs[0] if tcs else None), "args": parsed}

def probe_ctx(model_record, expected_max_len):
    actual = model_record.get("max_model_len")
    return {"probe": "ctx", "pass": actual == expected_max_len,
            "expected": expected_max_len, "actual": actual}

def run(manifest):
    rec = wait_ready(manifest["served_model_name"], timeout_s=manifest.get("ready_timeout_s", 900))
    results = [probe_text(manifest["served_model_name"])]
    if "vision" in manifest["functionalities"]:
        results.append(probe_vision(manifest["served_model_name"], manifest["fixtures"]["image"]))
    if "tools" in manifest["functionalities"]:
        results.append(probe_tools(manifest["served_model_name"]))
    results.append(probe_ctx(rec, manifest["expected_max_model_len"]))
    # empty content + correct reasoning still means the model works; pass it, but record the flag
    report = {"recipe": manifest["recipe_id"], "image": manifest["image"],
              "probes": results,
              "passed": all(p["pass"] for p in results),
              "reasoning_channel_flags": [p["probe"] for p in results if p.get("content_empty")]}
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    with open(sys.argv[1]) as f:  # path to per-recipe manifest (yaml->dict via a loader in full version)
        import yaml; manifest = yaml.safe_load(f)
    sys.exit(run(manifest))
```

The full harness adds: `launch_container(manifest)` (invokes `start-model.sh`/`launch.sh` with `THOR_VLLM_IMAGE` forced to the pinned digest and the recipe's profile), `teardown(container)` (always runs in a `finally`), the audio/video/MTP/FP8-KV probes, and a top-level loop over all manifests writing the aggregate report.

### 4c. Where the harness lives
- `serving/test/validate_public_images.py` — the runner above (path: `src/NemoClaw-Thor/serving/test/validate_public_images.py`; **the `serving/test/` directory does not exist yet and must be created**).
- `serving/test/manifests/<recipe_id>.yaml` — one **per-recipe manifest** declaring: `recipe_id`, `image` (pinned digest), `served_model_name`, `functionalities` (drives which probes run), `expected_max_model_len`, `fixtures` (paths to the known image / audio / video), MTP `accept_floor`, `ready_timeout_s`, and the `THOR_*` profile env to pass to `launch.sh`. The manifest's `functionalities` list is copied directly from each recipe contract's `functionalities` field so the probe matrix (§4a) is generated, not hand-maintained.
- `serving/test/fixtures/` — the known image (red-square/blue-circle), the audio WAV(s) (tone for path-probe + a spoken clip for the accuracy WARN probe), the EVS video clip.
- Aggregate output: `serving/test/reports/<digest>/summary.json` — machine-readable `{recipe: {passed, probes[], reasoning_channel_flags[]}}` plus a roll-up `all_passed` boolean that **is the §5 gate**.

### 4d. Scoring a reasoning model that returns `content=null` but correct `reasoning`
This is the explicit policy encoded in `extract_answer()` above:
- If `message.content` is non-empty and correct → **PASS, channel=content**.
- If `message.content` is empty/null but `message.reasoning_content` (or `reasoning`) is non-empty and contains the correct answer → **PASS on reasoning, channel=reasoning, and set `content_empty=True`** so the recipe report lists the probe under `reasoning_channel_flags`. The recipe still passes the functionality gate (the model **can** answer), but the flag surfaces the §3 client issue: any OpenAI client reading only `content` would see an empty response, so the deployment must read `reasoning` (ManyForge bridge does) or use mitigation (b)/(c).
- If **both** empty → **FAIL** (no answer in any channel). Also FAIL if `finish_reason == "length"` with empty content (the verified omni edge case) — report it distinctly as `truncated_reasoning` so it is not confused with a parser/channel bug.

---

## 5. Rollout Order & Risk Gates

**Each gate = `serving/test/reports/<digest>/summary.json` shows `all_passed: true` for that wave's recipes on the pinned public image.** No wave promotes until its gate is green. **Fail-closed: the custom `nemoclaw-thor/vllm:latest` is NOT retired until ALL in-scope recipes pass.**

1. **Wave 0 — Pin & build.** Digest-pin the nightly (`sha256:9eb0f4b6…e807b28` or a deliberately re-chosen digest), repoint `Dockerfile.overlay` `BASE_IMAGE`, bake `vllm[audio]`, build/tag `nemoclaw-thor/vllm:v10`. Pin the gemma4-jetson-thor digest. Pin the omni `--revision`. Create `serving/test/` + manifests + fixtures. **Gate:** overlay builds; `/v1/models` reachable for a trivial text-only boot.
2. **Wave 1 — VERIFIED first.** `cosmos-reason2-8b` (vision), `qwen3.6-35b-a3b-nvfp4-nvidia` (MTP), `nemotron3-nano-omni-30b-a3b-nvfp4-reasoning` (text+image+audio-path). **Gate:** all three green, including the reasoning-channel flags being *expected* not *failures*, and omni audio **path** probe green (accuracy WARN-only).
3. **Wave 2 — INFERRED (low risk, verified-family).** `cosmos-reason2-2b`, `nemotron3-nano-omni-30b-a3b-nvfp4` (base/think-OFF), `qwen3.5-9b-claude-distilled-nvfp4`, `nemotron3-nano-4b-bf16`, `nemotron3-nano-30b-a3b-nvfp4`, `qwen3.6-27b-fp8-mtp-kvfp8`. **Gate:** all green; pay special attention to `qwen3_next_mtp` availability (27b) and the `--mamba_ssm_cache_dtype` flag name (4b).
4. **Wave 3 — UNTESTED last (gemma4).** `gemma4-e4b-it`, then `gemma4-26b-a4b-it` (HIGH risk). **Gate:** confirm gemma4 reasoning+tool parsers present on the image, SigLIP2 ViT boots with TORCH_SDPA, and the 26B BF16 MoE fits at 0.80/256K/17-seqs.
5. **Wave 4 — Retire.** Only after Waves 1–3 are ALL green: flip the production default away from the custom image and retire `nemoclaw-thor/vllm:latest`. Keep the custom image build artifacts archived (not deleted) for one rollback cycle.

---

## 6. Open Questions / Risks

**A. Gemma4 parser availability (highest single risk, untested).** Does `ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-thor` actually bundle BOTH the `gemma4` reasoning parser AND the `gemma4` tool-call parser? Both flags are mandatory for B1/B2's tools+reasoning contract; if either is missing they fail at startup or silently disable. Likely present (purpose-built image) but UNVERIFIED. Also: does `google/gemma-4-26B-A4B-it` exist and match the image's expected weights, and does the image accept Google BF16 sources (jetson-ai-lab paired it with an NVFP4 quant)?

**B. Audio transcription accuracy (only the path is proven).** Omni audio was verified end-to-end *as a path* (accepted, encoded, model responded about a 440Hz tone). **Transcription accuracy is unverified.** The §4 audio accuracy probe is WARN-only until a real spoken-clip baseline exists; promote to gating afterward. Gemma4 native audio is entirely untested (declared only).

**C. Nightly digest drift.** `vllm/vllm-openai:nightly-aarch64` and `gemma4-jetson-thor` are MOVING tags. The whole contract's reproducibility hinges on the digest pins. Any deliberate re-pin **must re-run the full §4 suite** (the digest is part of the report path: `reports/<digest>/`). Risk: a future digest drops the `qwen3_next_mtp` spec method, the `qwen3`/`hermes`/`qwen3_coder` parsers, PR#42124 (qwen3.6-35b lm_head), or renames `--mamba_ssm_cache_dtype` / `--mm-encoder-attn-backend` / `--attention-backend`.

**D. KV/context headroom when co-serving.** GPU-mem-util values are calibrated against the custom image's KV-pool/slot allocation (cosmos-8b 0.35 @256K; qwen3.6-35b 0.55; gemma4-26b 0.80; omni 0.50 @256K/16-seqs). The public image's KV-pool sizing may differ — capture `GPU KV cache size: NNN tokens` from each boot log (the §4 FP8-KV probe does this) and re-validate `max_num_seqs` headroom, especially for the **duo co-serve** cases (cosmos-8b sized to co-serve with the qwen3.6 manyforge profile).

**E. Reasoning-channel client compatibility (operational, not a boot risk).** A2/A6/A7/A9/B1/B2 route the answer to `message.reasoning`/`reasoning_content`. The ManyForge bridge must read `reasoning`; content-only OpenAI clients (e.g. Cline) need mitigation (b)/(c). The §4d scoring flags this per recipe so it cannot silently regress.

**F. SM110 env-switch validity on stock.** `VLLM_DISABLED_KERNELS`, `ENABLE_TRIATTENTION=0`, `VLLM_USE_FLASHINFER_MOE_FP16=0`, `VLLM_USE_FLASHINFER_MOE_FP4=0`, `CUTE_DSL_ARCH=sm_110a`, `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`, `VLLM_FP8_MOE_BACKEND=flashinfer_cutlass` are partly custom-build-era env names. Confirm each is still recognized by the pinned nightly; `VLLM_MODS=sm110a-fp4-dsl-unlock` is a custom-image-only hook and will be **inert/ignored** on stock (acceptable for the W4A16 path). If FlashInfer FP8-KV lacks sm_110a JIT kernels on the public image, FP8 KV may fail or fall back.

**G. Omni revision/offline interaction.** The complete `--revision` pin (e.g. `b5a7a5e3…`) is mandatory (HF `main` drifted to a config-only partial snapshot missing `modeling.py`); offline mode must not block the custom-code fetch. A pinned, complete revision is the reproducible/desired state.

**H. MTP method registration (qwen3.6-27b).** `qwen3_next_mtp` is the one MTP method not verified live (qwen3.6-35b's `mtp` *is* verified). If absent/renamed on the pinned nightly, the 27b recipe's headline feature fails — Wave 2 gate must check it explicitly.