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
| gemma4-31b-it-nvfp4 | NVFP4 | ~21 GiB | 0.80 | 102.4 | ~72 GiB | 10.4 GiB | 6 |
| gemma4-26b-a4b-it | BF16 | ~48.5 GiB | 0.80 | 102.4 | ~45 GiB | 2.6 GiB | 17 |

## Agent concurrency allocation

Thor is bandwidth-bound: per-agent throughput stays constant as you add
concurrent sequences. Total throughput scales linearly. So filling
sequence slots is free performance.

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
# gemma4-26b-a4b-it @ gpu_mem_util=0.85 (old setting):
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
