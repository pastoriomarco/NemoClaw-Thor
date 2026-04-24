# KV Cache Budget — Thor 128 GB Unified Memory

How `max_num_seqs` is calculated for each model profile.
If actual vLLM "Available KV cache memory" differs from estimates below,
divide actual available by the KV/seq column to get the correct max_num_seqs.

## Platform constants

- Total memory: 128 GiB (Jetson AGX Thor unified)
- gpu_memory_utilization: 0.80 (most models), 0.85 (122B), 0.50 (FP8 TQ)
- KV cache dtype: varies by profile (BF16, FP8, TurboQuant K8V4)
- CUDA/framework overhead: ~9 GiB (measured from gemma4-26b vLLM log)
- Target max_model_len: 131072 (DFlash) or 262144 (MTP/TQ/legacy)

## Architecture details

**Qwen 3.6-35B-A3B** is an MoE hybrid with `full_attention_interval=4`.
Only 25% of layers (every 4th, 10 of 40) use traditional KV cache. The other
75% use DeltaNet linear attention with a fixed-size recurrent state (~1 MB/layer/seq).
Key difference from Qwen 3.5: **head_dim=128** (vs 256), enabling native FA2 on SM110.

**Qwen 3.5 models** are DeltaNet hybrids with `full_attention_interval=4`,
same structure as 3.6 but with head_dim=256 (FA2 incompatible on SM110).

**Gemma 4 models** use hybrid SWA + global attention. Pattern: 5 SWA layers
then 1 global, repeating. SWA layers cache only 1024 tokens (sliding window),
global layers cache the full context. Asymmetric head dimensions: global
layers have head_dim=512, SWA layers have head_dim=256.

## Per-model KV cache per sequence

### Qwen 3.6-35B-A3B (production profiles)

Formula: `full_attn_layers x 2(K+V) x kv_heads x head_dim x dtype_bytes x max_model_len`

| KV dtype | Bytes/elem | KV/seq @131K | KV/seq @256K | Notes |
|----------|-----------|-------------|-------------|-------|
| BF16 | 2 | 1.25 GiB | 2.5 GiB | DFlash profiles |
| FP8 | 1 | 0.625 GiB | 1.25 GiB | MTP+FP8KV profiles |
| TQ K8V4 | ~0.75 | ~0.47 GiB | ~0.94 GiB | TurboQuant ~2.6x compression |

Math (BF16 @131K): 10 x 2 x 2 x 128 x 2 x 131072 = 1,342,177,280 bytes = 1.25 GiB
Math (FP8 @131K): 10 x 2 x 2 x 128 x 1 x 131072 = 671,088,640 bytes = 0.625 GiB

Model details: 40 total layers, 10 full attention (every 4th), 2 KV heads, head_dim=128.

### Qwen 3.5 DeltaNet models (legacy)

Formula: `full_attn_layers x 2(K+V) x kv_heads x head_dim x 1(fp8) x 262144`

| Model | Layers (full/total) | KV heads | head_dim | KV/seq (fp8) | DeltaNet state |
|-------|---------------------|----------|----------|--------------|----------------|
| qwen3.5-9b (VLM) | see 9B profile | — | — | — | — |
| qwen3.5-35b-a3b | 10 / 40 | 2 | 256 | 2.5 GiB | 30 MB |

Math for 35B: 10 x 2 x 2 x 256 x 1 x 262144 = 2,684,354,560 bytes = 2.5 GiB

### Gemma 4 hybrid SWA + global models

Formula: global KV + SWA KV (fixed at window=1024)

| Model | Global layers | Global KV heads | Global head_dim | SWA layers | SWA KV heads | SWA head_dim | KV/seq (fp8) |
|-------|--------------|-----------------|-----------------|------------|--------------|--------------|--------------|
| gemma4-31b | 10 / 60 | 4 | 512 | 50 | 16 | 256 | 10.4 GiB |
| gemma4-26b-a4b | 5 / 30 | 2 | 512 | 25 | 8 | 256 | 2.6 GiB |

Math for 31B:
  Global: 10 x 2 x 4 x 512 x 1 x 262144 = 10.0 GiB
  SWA:    50 x 2 x 16 x 256 x 1 x 1024  =  0.4 GiB (fixed, context-independent)
  Total: 10.4 GiB

Math for 26B-A4B:
  Global: 5 x 2 x 2 x 512 x 1 x 262144  =  2.5 GiB
  SWA:   25 x 2 x 8 x 256 x 1 x 1024    =  0.1 GiB (fixed, context-independent)
  Total: 2.6 GiB

## Available KV memory and max_num_seqs

Model weight sizes are estimates (marked ~). Verify by reading the
"Available KV cache memory: XX.XX GiB" line from vLLM startup logs,
then divide by KV/seq above to get exact max_num_seqs.

### Qwen3.6 profiles (v6 container, measured)

| Profile | Quant | KV dtype | gpu_mem | max_model_len | KV tokens | max_num_seqs | Tok/s |
|---------|-------|----------|---------|---------------|-----------|--------------|-------|
| qwen3.6-35b-a3b-nvfp4-dflash | NVFP4 | BF16 | 0.80 | 262144 | 678K | 5 | **45.7 / 192.5 @ 8-conc** |
| qwen3.6-35b-a3b-fp8-dflash | FP8 | BF16 | 0.80 | 131072 | ~700K | 4 | **47.6** |
| qwen3.6-35b-a3b-nvfp4-tq-mtp | NVFP4 | TQ K8V4 | 0.80 | 262144 | 2.22M | 8 | 28.6 (153 @ 8-conc) |
| qwen3.6-35b-a3b-fp8-mtp-fp8kv | FP8 | FP8 | 0.80 | 131072 | 1.44M | 8 | 25.7 |
| qwen3.6-35b-a3b-fp8-turboquant | FP8 | TQ K8V4 | 0.50 | 262144 | 1.89M | 6 | 26.2 |

DFlash profiles include ~0.9 GiB drafter model overhead. KV token counts are
measured from vLLM startup logs, not estimated.

### Legacy profiles

| Profile | Quant | Est weights | gpu_mem | Usable GiB | Est KV avail | KV/seq @256K | max_num_seqs |
|---------|-------|-------------|---------|------------|--------------|--------------|--------------|
| qwen3.5-9b-claude-distilled-nvfp4 | NVFP4 | ~7 GiB | 0.40 | 51.2 | ~35 GiB | — | 8 |
| gemma4-31b-it-nvfp4 | NVFP4 | **31 GiB** | 0.80 | 102.4 | **64.2 GiB** | 10.4 GiB | 6 |
| gemma4-26b-a4b-it | BF16 | ~48.5 GiB | 0.80 | 102.4 | ~45 GiB | 2.6 GiB | 17 |

## Throughput scaling — why concurrency matters

**Thor is bandwidth-bound, not compute-bound.** This has a critical
consequence: adding concurrent sequences does NOT slow down individual
requests. Total throughput scales linearly with the number of concurrent
sequences, up to the KV cache limit.

### Measured scaling (Qwopus 3.5-27B NVFP4, 2026-04-07)

| Concurrent requests | Aggregate tok/s | Per-request tok/s | KV cache used |
|---------------------|-----------------|-------------------|---------------|
| 1 | ~9.3 | ~9.3 | <0.1% |
| 3 | ~37 | ~12 | ~1.6% |
| 6 | **77.0** | ~12.8 | **3.2%** |

Key observations:
- **6 concurrent requests = 8.3x single-request throughput** (near-linear)
- Per-request throughput actually *increases* slightly with batch size
  (better GPU utilization, amortized overhead)
- KV cache usage is minimal for short agentic prompts (3.2% at 6 reqs)
- MTP speculative decode acceptance: 71-93% (avg 82%), +1.7-1.9 tokens/step
- Prefix cache hit rate: 81.9% (good for repeated system prompts)

### Implication for agent design

**Every unused sequence slot is wasted throughput.** An agentic workflow
with 1 main agent on a model that supports 9 slots is using 11% of
available throughput. Spawning subagents for parallel work (file search,
test execution, code review) is essentially free — you get the results
faster without any penalty to the main agent.

Configure OpenClaw to fill slots aggressively:
- `maxConcurrent` for main agents (3 for 9-slot models)
- `maxChildrenPerAgent` for subagents (2 children per main agent)
- `maxSpawnDepth` = 1 (subagents don't spawn sub-subagents)

## Agent concurrency allocation

Slots are split between main agents and subagents. maxChildrenPerAgent
is capped at min(4, ceil(subagent_slots / main)) for fair distribution —
no single agent can hog all subagent slots.

### Qwen3.6 profiles

| Profile | Slots | Main agents | Subagent slots | Children/agent | Depth |
|---------|-------|-------------|----------------|----------------|-------|
| qwen3.6 DFlash profiles | 4 | 1 | 3 | 3 | 1 |
| qwen3.6 MTP/TQ profiles (8 seqs) | 8 | 2 | 6 | 3 | 1 |
| qwen3.6-35b-a3b-fp8-turboquant | 6 | 2 | 4 | 2 | 1 |

### Legacy profiles

| Profile | Slots | Main agents | Subagent slots | Children/agent | Depth |
|---------|-------|-------------|----------------|----------------|-------|
| qwen3.5-9b-claude-distilled-nvfp4 | 8 | 2 | 6 | 3 | 1 |
| gemma4-31b-nvfp4 | 6 | 2 | 4 | 2 | 1 |
| gemma4-26b-a4b | 17 | 4 | 13 | 4 | 1 |

Override main concurrency at launch:
`THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT=N ./configure-local-provider.sh`

## Calibration data (actual vLLM logs)

Use this section to record actual values from vLLM startup logs.
Update max_num_seqs in config.sh if actuals differ significantly.

```
# qwopus3.5-27b-nvfp4 @ gpu_mem_util=0.80 (2026-04-07):
#   Available KV cache memory: 64.13 GiB
#   max_num_seqs = floor(64.13 / 8.0) = 8 (config says 9 — close enough,
#     DeltaNet state overhead accounts for the difference)
#   MTP speculative decode: 1 token (model has mtp_num_hidden_layers=1)
#   MTP acceptance rate: 82% avg (71-93% range)
#   Prefix cache hit rate: 81.9%
#   Throughput scaling:
#     1 req: ~9.3 tok/s
#     3 req: ~37 tok/s (12.3 tok/s per req)
#     6 req:  77 tok/s (12.8 tok/s per req) — near-linear scaling confirmed
#   KV cache usage at 6 concurrent short prompts: 3.2%

# qwen3.5-122b-a10b-nvfp4 (Sehyo) @ gpu_mem_util=0.85 + MTP (2026-04-08):
#   Model loading took 75.17 GiB (includes MTP drafter, shared embed+lm_head)
#   Available KV cache memory: 24.05 GiB
#   GPU KV cache size: 482,080 tokens
#   Maximum concurrency for 262,144 tokens per request: 6.42x
#   max_num_seqs = floor(24.05 / 3.0) = 8 ✓ (config set to 8)
#   MTP speculative decode: 1 token (model has mtp_num_hidden_layers=1)
#   Architecture: Qwen3_5MoeMTP, FlashInfer CUTLASS MoE + FlashInfer attention
#   Drafter loaded in 43s, target in 193s, total warmup ~10 min
#   Note: 8 slots all as main agents (Claude Code orchestration, no subagents)

# qwen3.5-122b-a10b-nvfp4-resharded @ gpu_mem_util=0.85 (2026-04-08):
#   Model loading took 70.47 GiB (no MTP drafter)
#   Available KV cache memory: 21.43 GiB
#   GPU KV cache size: 467,712 tokens
#   Maximum concurrency for 131,072 tokens per request: 11.79x
#   max_num_seqs = floor(21.43 / 3.0) = 7 (config set to 8 — safe, agentic
#     sessions rarely hit full 262K context)

# gemma4-31b-it-nvfp4 @ gpu_mem_util=0.80 (2026-04-07):
#   Model loading took 31.04 GiB (includes ~10 GiB vision encoder)
#   Available KV cache memory: 64.21 GiB
#   max_num_seqs = floor(64.21 / 10.4) = 6 ✓ (matches config)
#   Est weights was ~21 GiB — actual 31 GiB due to SigLIP vision encoder
#   Throughput: 6.7 tok/s single request (triton_attn, no FlashInfer)

# gemma4-26b-a4b-it @ gpu_mem_util=0.85 (old setting, pre-v4):
#   Model loading took 48.5 GiB
#   Available KV cache memory: 51.62 GiB
#   GPU KV cache size: 451,040 tokens
#   Maximum concurrency for 131,072 tokens per request: 33.42x
```

## How to recalculate

If model weights or overhead change (new vLLM version, different quant):

1. Start the model: `./start-model.sh <profile>`
2. Read from vLLM log: `Available KV cache memory: XX.XX GiB`
3. Look up KV/seq from the table above
4. max_num_seqs = floor(available_kv / kv_per_seq)
5. Update THOR_TARGET_MAX_NUM_SEQS in lib/config.sh

## Why DeltaNet/SWA models need far fewer sequence slots than you'd expect

A naive calculation of `max_model_len x max_num_seqs <= total_kv_tokens` assumes
every layer caches the full context. This is wrong for:

- **DeltaNet**: 75% of layers use O(1) recurrent state, not KV cache
- **SWA**: sliding window layers only cache 1024 tokens regardless of context

The correct calculation accounts for the per-architecture KV footprint, which
can be 4-10x smaller than a naive estimate for the same parameter count.
