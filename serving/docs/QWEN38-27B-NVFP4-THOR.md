# Qwen3.8-27B NVFP4 on Jetson AGX Thor

**Status:** verified working on Thor on 2026-08-14

**Profile:** `qwen3.8-27b-nvfp4`

**Served model ID:** `qwen3.8-27b-nvfp4`

This is the reproducible image-build and serving recipe for
`unsloth/Qwen3.8-27B-NVFP4`. It preserves vision and the checkpoint's MTP head,
uses the native NVFP4 path for eligible quantized layers, and exposes the usual
OpenAI-compatible endpoint for ManyForge and NemoClaw.

## One command to serve it

After building the image once, run this from the repository root:

```bash
THOR_VLLM_PORT=8050 ./serving/start-model.sh qwen3.8-27b-nvfp4
```

This is intentionally attached: logs remain in the terminal and `Ctrl-C` stops
the model container. The launcher also starts the ManyForge vLLM proxy on
`:8000`, forwarding to the model on `:8050`.

The profile uses the complete local Hugging Face snapshot without contacting
the Hub when it is present. If the pinned snapshot is absent, it downloads only
that revision. The persistent directories are created automatically.

## Pinned components

| Component | Reproducible pin |
|---|---|
| Target weights | `unsloth/Qwen3.8-27B-NVFP4` |
| Target revision | `9c73e2daee1d0fd494ffbd1d8753f2174a953796` |
| Qwen3.8 vLLM base | `vllm/vllm-openai:qwen38@sha256:4a2f33a884222f7049b983263ad9976f89452bb81affecf5b67d89ad35c1bc31` |
| SM110 extension donor | `vllm/vllm-openai:v0.27.1@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967` |
| FlashInfer cubins | `flashinfer_cubin==0.6.17`, official release wheel |
| Resulting local image | `nemoclaw-thor/vllm-openai:qwen38-thor-sm110` |

The Dockerfile is
[`serving/docker/Dockerfile.qwen38-thor-sm110`](../docker/Dockerfile.qwen38-thor-sm110),
and both upstream images are digest-pinned. Do not replace only one base image:
the copied native extension is validated against this exact pair.

## Why this build is better than either upstream image alone

The image is a small, targeted overlay rather than another multi-hour source
build:

- The `qwen38` preview supplies the new Qwen3.8 architecture, model loader,
  Transformers integration, and its matching CUDA 13 / Torch 2.13 Python
  stack. Stable vLLM 0.27.1 alone does not supply this complete Qwen3.8 path.
- The preview image's vLLM native extension was not built with SM110 code. On
  Thor it reaches a `no kernel image is available` failure. The overlay copies
  the compatible `_C_stable_libtorch` extension from the arm64 v0.27.1 image,
  which contains native SM110 support.
- The preview includes FlashInfer 0.6.17 and its JIT-cache package but omits
  the separately distributed cubin bundle. Installing the official 0.6.17
  cubin wheel makes its precompiled kernel registry available, avoids needless
  compilation for covered shapes, and gives more consistent cold starts.
- It keeps the working Qwen3.8 preview userspace intact. There is no vLLM,
  PyTorch, CUDA, or FlashInfer installation on the Thor host.
- With the model profile, eligible MLP linear layers use hardware-accelerated
  NVFP4 W4A4; protected FP8/BF16 layers retain their checkpoint precision,
  FlashInfer handles attention, FP8 reduces KV memory, and BF16 MTP K=3 raises
  decode throughput without replacing the target model.
- Vision remains enabled: up to four images per prompt are allowed. This is not
  the text-only compromise used by the older Qwen3.6 launch command.

The final overlay is about 21.2 GB locally. Roughly 4.8 GB of its increase over
the preview image is the precompiled FlashInfer cubin package; the copied vLLM
extension is comparatively small. Docker layer sharing avoids duplicating all
base layers when the inputs are already present.

This remains a temporary compatibility image. The binary transplant is not a
general ABI promise: rebuild and smoke-test it if either pinned base changes.
Retire it when an official vLLM arm64 image contains Qwen3.8 support, an SM110
native extension, and the required FlashInfer 0.6.17+ cubins together.

## Build the image

From the repository root:

```bash
./serving/docker/build-qwen38-thor-sm110.sh
```

To use another local image tag:

```bash
QWEN38_THOR_IMAGE=my-registry/vllm:qwen38-thor ./serving/docker/build-qwen38-thor-sm110.sh
```

Use the same tag at launch:

```bash
THOR_QWEN38_VLLM_IMAGE=my-registry/vllm:qwen38-thor THOR_VLLM_PORT=8050 \
  ./serving/start-model.sh qwen3.8-27b-nvfp4
```

On a machine that already has both base images and the FlashInfer download in
Docker's build cache, this overlay build takes minutes. A fresh machine must
first pull the two large arm64 base images and the cubin wheel, so network time
dominates. Model weights are a separate download performed at serving time.

Confirm the result:

```bash
docker image inspect nemoclaw-thor/vllm-openai:qwen38-thor-sm110 \
  --format '{{.Id}} {{.Size}}'
```

## Persistent storage and Hub behavior

Defaults match the existing NemoClaw-Thor recipes:

| Purpose | Host default |
|---|---|
| Hugging Face weights | `$HOME/thor-hf-cache` |
| vLLM cache | `$HOME/thor-vllm-cache` |
| TorchInductor cache | `$HOME/thor-torch-cache` |
| FlashInfer cache | `$HOME/thor-flashinfer-cache` |

They can be relocated per launch without changing the recipe:

```bash
THOR_HF_CACHE_DIR=/fast/models/huggingface \
THOR_VLLM_CACHE_DIR=/fast/cache/vllm \
THOR_TORCH_CACHE_DIR=/fast/cache/torch \
THOR_FLASHINFER_CACHE_DIR=/fast/cache/flashinfer \
THOR_VLLM_PORT=8050 \
  ./serving/start-model.sh qwen3.8-27b-nvfp4
```

Hub policies are:

- `THOR_HF_MODE=auto` (default): use the complete pinned local snapshot;
  download that pinned revision only when missing.
- `THOR_HF_MODE=offline`: fail unless the pinned revision is complete locally.
- `THOR_HF_MODE=latest`: check upstream `main` and fetch changed or missing
  blobs. This deliberately drops the verified model-revision pin.
- `THOR_QWEN38_REVISION=<commit>`: test another explicit revision while
  retaining cache-first behavior.

The verified profile does **not** set `NVIDIA_DISABLE_REQUIRE=true`. JetPack
7.x provides the CUDA 13 runtime expected by the pinned image; keeping the
container runtime's CUDA requirement check enabled catches an incompatible
host instead of hiding it.

## Serving configuration

The matching cases in [`serving/config.sh`](../config.sh) and
[`serving/launch.sh`](../launch.sh) encode:

| Setting | Value |
|---|---|
| Combined context (`prompt + output`) | 262,144 tokens |
| Advertised maximum output | 16,384 tokens |
| Scheduler concurrency | 4 sequences |
| ManyForge/OpenClaw main concurrency | 3, leaving one subagent slot |
| GPU memory utilization | 0.80 |
| KV cache | FP8 (not NVFP4) |
| Batched-token cap | 8,192 |
| Attention | FlashInfer; multimodal encoder uses Torch SDPA |
| Speculative decoding | embedded BF16 MTP, 3 draft tokens |
| Tool/reasoning parsers | `qwen3_coder` / `qwen3` |
| Modalities | text and images; maximum 4 images, video disabled |

NVFP4 describes the eligible model weights and activations. It does not make
the KV cache FP4; the stable and memory-efficient runtime choice here is FP8.
The checkpoint is intentionally mixed precision rather than indiscriminately
quantizing the attention, vision, MTP, and other quality-sensitive paths.

The 16K output value is an allowed per-request ceiling. Every request must
still satisfy `input_tokens + max_tokens <= 262144`. The ManyForge proxy uses
a lower operational default (`max_tokens=2048`, thinking budget 512) for
bounded agent turns; direct clients on `:8050` may request the full 16K.

## ManyForge assistant

The full ManyForge launcher already passes port `8050`, owns the proxy on
`:8000`, drops caches during model swaps, and uses the profile slug as the live
model ID. From the sibling `manyforge` repository:

```bash
MODEL_PROFILE=qwen3.8-27b-nvfp4 ./scripts/demo-assistant-known-good.sh restart
```

For the standalone NemoClaw workflow, start the model with the one-command
recipe above, then configure the provider:

```bash
./setup/configure-local-provider.sh qwen3.8-27b-nvfp4
```

Detached model-only form, when a supervisor will own the process:

```bash
THOR_VLLM_PORT=8050 THOR_DETACH=1 THOR_NO_RM=1 \
THOR_CONTAINER_NAME=manyforge-e2e-vllm \
  ./serving/start-model.sh qwen3.8-27b-nvfp4
```

Follow it with:

```bash
docker logs -f manyforge-e2e-vllm
```

## Verification and observed performance

Proxy-facing readiness:

```bash
curl -s http://127.0.0.1:8000/v1/models | jq -r '.data[].id'
```

Expected output:

```text
qwen3.8-27b-nvfp4
```

The first live text probe on the verified image measured about **22.5 decoded
tokens/s after the first token** with MTP enabled. Treat this as a short-request
sanity number, not a throughput guarantee: prompt length, MTP acceptance,
concurrency, thermals, and agent tool-call cadence all change the observed
rate. Vision, tool parsing, FP8 KV, and MTP were present in the working server.

The initial boot may compile shapes not covered by the cubin/JIT packages.
Keep the four cache mounts persistent; subsequent launches should reuse them.
