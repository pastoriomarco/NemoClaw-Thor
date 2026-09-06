# Qwen3.8-Flash-Next NVFP4 on Jetson AGX Thor

**Status:** historical baseline, superseded by the current best verified recipe.

**Served model ID:** `qwen3.8-flash-next`

**For the recommended setup**, including all image builds and single-command launch, see
[Flash-Next fast Thor recipe](QWEN38-FLASH-NEXT-FAST-THOR.md).
The original Triton recipe below is retained as a fallback and historical baseline.
Its MTP=2/0.85 advice is historical: the recommended fused-CUDA setup uses
MTP=3/0.90 with BF16 KV. The FP8-KV command below is experimental, not the
recommended launch command.

This recipe serves `RadixArk/Qwen3.8-Flash-Next-NVFP4` with its large PLE
n-gram table mapped from NVMe, native NVFP4 routed experts, deterministic QSA
top-k, working prefix caching, and embedded MTP. It is derived from
`blazux/qwen3.8-Flash-DGX`, pinned at commit
`4b723de2e2c465d866738b57ae64bde6e8c07744`, with two Thor-specific changes:

- vLLM's FlashInfer CUTLASS MoE backend gate is extended to SM110. The stock
  automatic backend selected an SM100 vLLM CUTLASS kernel and failed while
  creating TMA descriptors on Thor.
- `VLLM_GDN_DECODE_KERNEL=triton` bypasses the preview image's fused GDN MTP
  CUDA kernel, which has no SM110 code object. MTP remains enabled.

The model snapshot used for the verified run is
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`.

## What was verified

The working `0.85`, BF16-KV, MTP=2 launch reported:

```text
Using FlashInferExperts MoE backend
Available KV cache memory: 12.45 GiB
GPU KV cache size: 440,286 tokens
Maximum concurrency for 262,144 tokens per request: 1.68x
Application startup complete
```

`--max-num-seqs 4` is only a scheduler ceiling. With BF16 KV, four requests
can execute together when their combined cached token count fits in about
440K tokens; it does not provide four independent full 262K contexts.

Across 86 ten-second live metric windows, MTP=2 produced 49.52% weighted draft
acceptance, mean acceptance length 2.03, and average per-position acceptance
of 62.1% / 40.7%. The non-zero aggregate generation windows averaged 17.66
tok/s and peaked at 31.80 tok/s. These numbers include changing context,
prefill interruptions, and occasional concurrency, so they are operational
observations rather than a controlled benchmark.

## Recommended versus capacity-oriented settings

### Quality/speed baseline

- `gpu_memory_utilization=0.85`
- BF16 KV (`--kv-cache-dtype auto`)
- MTP=2
- native 262,144-token maximum
- four scheduler slots, about 440K total KV tokens

This is the verified profile and keeps KV precision at BF16.

### Experimental four-full-context capacity profile

- `gpu_memory_utilization=0.90`
- FP8 E4M3 KV (`--kv-cache-dtype fp8_e4m3`)
- MTP=2
- native 262,144-token maximum
- four scheduler slots
- `--long-prefill-token-threshold 8192`

FP8 KV is required for four simultaneously resident full contexts. Based on
the measured 440K BF16 tokens at `0.85`, `0.90` with BF16 KV would be expected
to hold only about 650K tokens, approximately 2.5 full contexts. The image's
FP8 KV path offers about 1.9x the capacity, which should clear four contexts;
the actual boot log remains authoritative.

This profile is not yet validated on Thor. Qwen3.8-Flash-Next has an upstream
vLLM issue in which mixed-length concurrent prefills allocate a dense
`num_prefills x max_chunk_length` PLE short-conv buffer. Such traffic has OOMed
at `0.85` even with low KV utilization. An 8,192-token long-prefill threshold
does not fix this when the total batched-token budget is already 8,192;
mixed-length padded allocations can still exceed that budget. At `0.90`,
watch host available memory during four mixed-length agent requests; steady
idle memory is not the worst case.

FP8 KV can also have a small quality and speed cost from quantization and
dequantization. There is no FP4 KV-cache implementation in this stack.

### Why MTP remains 2

The observed second-position acceptance of 40.7% makes MTP=3 worth an A/B
test, but not a justified default. A directly comparable DGX Spark sweep for
Qwen3.8-Flash-Next measured MTP=2 at 25.38 tok/s and MTP=3 at 21.01 tok/s.
MTP=3 accepted longer spans but its extra draft work cost more than it saved.
Thor additionally rebuilds GDN attention metadata between draft steps and
uses the Triton GDN fallback, so the likely penalty is at least as large.

To test MTP=3, change only `num_speculative_tokens` from 2 to 3 and compare
the same prompts, context lengths, and concurrency. Speculative acceptance
alone is insufficient; compare delivered generation tok/s.

## Build

Create or update the pinned upstream source checkout:

```bash
git clone https://github.com/blazux/qwen3.8-Flash-DGX.git \
  "$HOME/thor-qwen38-flash-next-vllm/source"
```

```bash
git -C "$HOME/thor-qwen38-flash-next-vllm/source" checkout \
  4b723de2e2c465d866738b57ae64bde6e8c07744
```

Build the upstream layer for SM110:

```bash
docker build \
  --build-arg DET_ARCH=110a \
  -t nemoclaw-thor/qwen38-flash-next-vllm:sm110 \
  "$HOME/thor-qwen38-flash-next-vllm/source"
```

From the NemoClaw-Thor repository root, build the small Thor overlay:

```bash
docker build \
  -f serving/docker/Dockerfile.qwen38-flash-next-sm110 \
  -t nemoclaw-thor/qwen38-flash-next-vllm:sm110-flashinfer-moe \
  .
```

## Persistent paths

```bash
export HF_CACHE="$HOME/thor-hf-cache"
export VLLM_CACHE="$HOME/thor-vllm-cache"
export TORCH_CACHE="$HOME/thor-torch-cache"
export FLASHINFER_CACHE="$HOME/thor-flashinfer-cache"
export FLASHNEXT_MODEL_REV="7b719225242aacd3dbd3f9407468c2ee9a9d2594"
```

```bash
mkdir -p "$HF_CACHE" "$VLLM_CACHE" "$TORCH_CACHE" "$FLASHINFER_CACHE"
```

The checkpoint must exist at:

```text
$HF_CACHE/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594
```

## Foreground launch

The command below is the experimental four-full-context profile. It remains
attached so startup and runtime logs stay visible in the terminal.

```bash
export FLASHNEXT_SPLIT_OPS='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
```

```bash
docker run -it \
  --name qwen38-flash-next-vllm \
  --runtime nvidia \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size 16g \
  -v "$HF_CACHE:/hf" \
  -v "$VLLM_CACHE:/root/.cache/vllm" \
  -v "$TORCH_CACHE:/root/.cache/torch" \
  -v "$FLASHINFER_CACHE:/root/.cache/flashinfer" \
  -e HF_HOME=/hf \
  -e HF_HUB_OFFLINE=1 \
  -e CUTE_DSL_ARCH=sm_110a \
  -e TORCH_CUDA_ARCH_LIST=11.0a \
  -e VLLM_PLE_MMAP=1 \
  -e VLLM_PLE_MMAP_WORKERS=14 \
  -e VLLM_PLE_MMAP_PREWARM=0 \
  -e VLLM_QSA_DET_TOPK=1 \
  -e VLLM_QSA_DET_LIB=/opt/llm/kernel-det/_C_det.so \
  -e VLLM_QSA_EXACT_TOPK=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  --entrypoint vllm \
  nemoclaw-thor/qwen38-flash-next-vllm:sm110-flashinfer-moe \
  serve "/hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/$FLASHNEXT_MODEL_REV" \
  --served-model-name qwen3.8-flash-next \
  --host 0.0.0.0 \
  --port 8050 \
  --load-format safetensors \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --long-prefill-token-threshold 8192 \
  -cc.cudagraph_mode=PIECEWISE \
  -cc.splitting_ops="$FLASHNEXT_SPLIT_OPS" \
  --no-enable-flashinfer-autotune \
  --moe-backend flashinfer_cutlass \
  --kv-cache-dtype fp8_e4m3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

For the verified quality/speed baseline, change only:

```text
--gpu-memory-utilization 0.85
--kv-cache-dtype auto
```

and omit `--long-prefill-token-threshold 8192` if maximum cold-prefill speed
matters more than mixed-concurrency OOM protection.

## Client endpoint

```text
Base URL: http://<thor-ethernet-ip>:8050/v1
Model ID: qwen3.8-flash-next
Context window: 262144
Maximum output: 16384
```
