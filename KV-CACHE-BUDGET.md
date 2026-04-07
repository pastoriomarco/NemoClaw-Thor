# KV Cache Budget — Thor 128 GB Unified Memory

How `max_num_seqs` is calculated for each model profile.
If actual vLLM "Available KV cache memory" differs from estimates below,
divide actual available by the KV/seq column to get the correct max_num_seqs.

## Platform constants

- Total memory: 128 GiB (Jetson AGX Thor unified)
- gpu_memory_utilization: 0.80 (all models except 122B which uses 0.85)
- KV cache dtype: fp8 (1 byte per element)
- CUDA/framework overhead: ~9 GiB (measured from gemma4-26b vLLM log)
- Target max_model_len: 262144 (256K) for all models

## Architecture details

All Qwen 3.5 models are **DeltaNet hybrids** with `full_attention_interval=4`.
Only 25% of layers (every 4th) use traditional KV cache. The other 75% use
DeltaNet linear attention with a fixed-size recurrent state (~1 MB/layer/seq),
which does NOT grow with sequence length.

Gemma 4 models use **hybrid SWA + global attention**. Pattern: 5 SWA layers
then 1 global, repeating. SWA layers cache only 1024 tokens (sliding window),
global layers cache the full context. Gemma 4 also uses asymmetric head
dimensions: global layers have head_dim=512 with fewer KV heads, SWA layers
have head_dim=256 with more KV heads.

## Per-model KV cache per sequence at 256K context (fp8)

### Qwen 3.5 DeltaNet models

Formula: `full_attn_layers x 2(K+V) x kv_heads x head_dim x 1(fp8) x 262144`

| Model | Layers (full/total) | KV heads | head_dim | KV/seq (fp8) | DeltaNet state |
|-------|---------------------|----------|----------|--------------|----------------|
| qwen3.5-27b (all 27B variants) | 16 / 64 | 4 | 256 | 8.0 GiB | 48 MB |
| qwen3.5-35b-a3b | 10 / 40 | 2 | 256 | 2.5 GiB | 30 MB |
| qwen3.5-122b-a10b | 12 / 48 | 2 | 256 | 3.0 GiB | 36 MB |

Math for 27B: 16 x 2 x 4 x 256 x 1 x 262144 = 8,589,934,592 bytes = 8.0 GiB
Math for 35B: 10 x 2 x 2 x 256 x 1 x 262144 = 2,684,354,560 bytes = 2.5 GiB
Math for 122B: 12 x 2 x 2 x 256 x 1 x 262144 = 3,221,225,472 bytes = 3.0 GiB

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

| Profile | Quant | Est weights | gpu_mem_util | Usable GiB | Est KV avail | KV/seq @256K | max_num_seqs |
|---------|-------|-------------|--------------|------------|--------------|--------------|--------------|
| qwen3.5-122b-a10b-nvfp4 | NVFP4 | ~85 GiB | 0.85 | 108.8 | ~15 GiB | 3.0 GiB | 4 |
| qwopus3.5-27b-nvfp4 | NVFP4 | ~20 GiB | 0.80 | 102.4 | ~73 GiB | 8.0 GiB | 9 |
| qwen3.5-27b-claude-nvfp4 | NVFP4 | ~20 GiB | 0.80 | 102.4 | ~73 GiB | 8.0 GiB | 9 |
| qwen3.5-27b-fp8 | FP8 | ~28 GiB | 0.80 | 102.4 | ~65 GiB | 8.0 GiB | 8 |
| qwen3.5-35b-a3b-fp8 | FP8 | ~36 GiB | 0.80 | 102.4 | ~57 GiB | 2.5 GiB | 22 |
| qwen3.5-35b-a3b-nvfp4 | NVFP4 | ~26 GiB | 0.80 | 102.4 | ~67 GiB | 2.5 GiB | 26 |
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

| Profile | Slots | Main agents | Subagent slots | Children/agent | Depth |
|---------|-------|-------------|----------------|----------------|-------|
| qwen3.5-122b-a10b | 4 | 1 | 3 | 3 | 1 |
| qwen3.5-27b-fp8 | 8 | 2 | 6 | 3 | 1 |
| qwen3.5-27b NVFP4 variants | 9 | 3 | 6 | 2 | 1 |
| gemma4-31b-nvfp4 | 6 | 2 | 4 | 2 | 1 |
| gemma4-26b-a4b | 17 | 4 | 13 | 4 | 1 |
| qwen3.5-35b-a3b-fp8 | 22 | 5 | 17 | 4 | 1 |
| qwen3.5-35b-a3b-nvfp4 | 26 | 6 | 20 | 4 | 1 |

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
