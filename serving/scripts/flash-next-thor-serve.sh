#!/bin/bash
# Qwen3.8-Flash-Next on Jetson AGX Thor: the two current recipes (2026-11-16).
# usage: flash-next-thor-serve.sh n3|r8b single|multi [extra vLLM args...]
#
#   n3     nvidia/Qwen3.8-Flash-Next-NVFP4 + online FP8 side layers (QSA indexer BF16) + C1 + C2
#          (best Tool Eval: 94, 94)
#   r8b    nvidia/Qwen3.8-Flash-Next-FP4-Local + C3 + C2 (fastest: 46.2 ms per step, HumanEval 157/164)
#   single 1 sequence, memory share 0.75 (KV ~468K tokens on n3, ~496K on r8b)
#   multi  up to 6 sequences, KV pool fixed at 29.6 GB (~1.0M tokens on both), memory share 0.90
#
# Details and measured results: serving/docs/QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md, "Current recipes".
# Measurement switches (step timing, PLE and screened-head stats): FN_STATS=1.
#
# Thor overlay folders (read-only mounts) and their sources in this repo:
#   ~/fn-local-patch      vLLM PR #64312 + #64869 (FP4-Local scale decode, WeightsMapper), qsa_indexer.py [thor]
#   ~/fn-local-ple-disk   docker/patches/qwen4exp-ple-disk     (PLE table on NVMe + CLOCK slot pool)
#   ~/fn-draft-head       docker/patches/qwen4exp-draft-head   (draft vocab, FP8 heads, screened head, side FP8, C1, C2)
#   ~/fn-spec-step        docker/patches/qwen4exp-spec-step    (draft FULL graphs, step timing)
#   ~/fn-cutedsl-moe      docker/patches/flashinfer-cutedsl-moe-sm110 (+ prebuilt moe_utils in ~/flashinfer-cache)
#   ~/fn-fi-moe           docker/patches/flashinfer-cutlass-moe-sm110 (mounted in every measured run; not selected)
#   ~/fn-mxfp8-tactic     docker/patches/flashinfer-mxfp8-tactic-sm110 (C3)
#   ~/fn-assets           draft vocabulary (draft_vocab_code_47149.npy)
set -e
RECIPE=$1; MODE=$2; shift 2 || true
case "$RECIPE" in n3|r8b) ;; *) echo "usage: $0 n3|r8b single|multi [extra vLLM args]"; exit 2;; esac
case "$MODE" in single|multi) ;; *) echo "usage: $0 n3|r8b single|multi [extra vLLM args]"; exit 2;; esac

IMAGE=vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e
V=/usr/local/lib/python3.12/dist-packages/vllm
F=/usr/local/lib/python3.12/dist-packages/flashinfer
N=$V/models/qwen4_exp/nvidia
P=$HOME/fn-local-patch/vllm
D=$HOME/fn-local-ple-disk/vllm/models/qwen4_exp
H=$HOME/fn-draft-head/vllm/models/qwen4_exp/nvidia
S=$HOME/fn-spec-step/vllm/v1/worker/gpu
M=$HOME/fn-cutedsl-moe

MNT=(
  -v "$HOME/thor-hf-cache:/root/.cache/huggingface:ro" -v "$HOME/vllm-cache:/root/.cache/vllm"
  -v "$HOME/flashinfer-cache:/root/.cache/flashinfer" -v "$HOME/fn-assets:/opt/qwen4exp:ro"
  -v "$P/model_executor/layers/quantization/modelopt.py:$V/model_executor/layers/quantization/modelopt.py:ro"
  -v "$P/model_executor/layers/quantization/utils/nvfp4_utils.py:$V/model_executor/layers/quantization/utils/nvfp4_utils.py:ro"
  -v "$P/model_executor/models/utils.py:$V/model_executor/models/utils.py:ro"
  -v "$P/model_executor/models/whisper.py:$V/model_executor/models/whisper.py:ro"
  -v "$P/models/qwen4_exp/nvidia/ops/qsa_indexer.py:$N/ops/qsa_indexer.py:ro"
  -v "$D/common/ple_disk.py:$V/models/qwen4_exp/common/ple_disk.py:ro"
  -v "$D/common/ple_disk_table.py:$V/models/qwen4_exp/common/ple_disk_table.py:ro"
  -v "$D/common/ple_disk_cache.py:$V/models/qwen4_exp/common/ple_disk_cache.py:ro"
  -v "$D/nvidia/ngram_embedding.py:$N/ngram_embedding.py:ro"
  -v "$H/draft_head.py:$N/draft_head.py:ro" -v "$H/model.py:$N/model.py:ro"
  -v "$H/screened_head.py:$N/screened_head.py:ro" -v "$H/mtp.py:$N/mtp.py:ro"
  -v "$H/hyperconnection.py:$N/hyperconnection.py:ro" -v "$H/hc_fp8.py:$N/hc_fp8.py:ro"
  -v "$S/spec_step_timing.py:$V/v1/worker/gpu/spec_step_timing.py:ro"
  -v "$S/model_runner.py:$V/v1/worker/gpu/model_runner.py:ro"
  -v "$S/spec_decode/target_dependent_ar/speculator.py:$V/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py:ro"
  -v "$M/flashinfer/fused_moe/cute_dsl/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py:$F/fused_moe/cute_dsl/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py:ro"
  -v "$M/flashinfer/fused_moe/cute_dsl/blockscaled_contiguous_grouped_gemm_finalize_fusion.py:$F/fused_moe/cute_dsl/blockscaled_contiguous_grouped_gemm_finalize_fusion.py:ro"
  -v "$M/flashinfer/jit/moe_utils.py:$F/jit/moe_utils.py:ro"
  -v "$M/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py:$V/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py:ro"
  -v "$HOME/fn-fi-moe/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py:$V/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py:ro"
  -v "$HOME/fn-mxfp8-tactic/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py:$V/model_executor/kernels/linear/mxfp8/flashinfer.py:ro"
)

ENV=(
  -e HF_HUB_OFFLINE=1 -e VLLM_PLE_DISK=1 -e VLLM_PLE_DISK_PREFETCH=1 -e VLLM_PLE_DISK_CACHE_GB=0.5
  -e VLLM_MARLIN_USE_ATOMIC_ADD=0
  -e QWEN4EXP_DRAFT_VOCAB=/opt/qwen4exp/draft_vocab_code_47149.npy -e QWEN4EXP_MTP_DENSE_FP8=1
  -e QWEN4EXP_FP8_FORMAT=fp8_per_block -e QWEN4EXP_DRAFT_HEAD_FP8=1 -e VLLM_SPEC_STEP_DRAFT_FULL_CG=1
  -e QWEN4EXP_LM_HEAD_SCREENED=1
  -e QWEN4EXP_HC_FP8=1                                    # C2 (both recipes)
)
if [ "$RECIPE" = n3 ]; then
  MODEL=/root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47
  ENV+=(-e QWEN4EXP_SIDE_FP8=1 -e QWEN4EXP_QSA_UNFUSE_INDEXER=1                     # side FP8 + C1
        -e QWEN4EXP_SIDE_FP8_SKIP=self_attn.indexer.index_qk_proj)
else
  MODEL=/root/.cache/huggingface/linked/fn-fp4-local
  ENV+=(-e VLLM_MXFP8_TACTIC=256:8)                                                # C3
fi
[ "${FN_STATS:-0}" = 1 ] && ENV+=(-e VLLM_SPEC_STEP_TIMING=200 -e VLLM_PLE_DISK_STATS_SEC=30 -e QWEN4EXP_LM_HEAD_SCREEN_STATS=60)

ARGS=(
  --model "$MODEL" --served-model-name qwen38-flash-next --host 0.0.0.0 --port 8000 --language-model-only
  --max-model-len 262144 --max-num-batched-tokens 8192 --kv-cache-dtype bfloat16
  --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  --performance-mode throughput --optimization-level 3 --async-scheduling --disable-flashinfer-autotune
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml --enable-prompt-tokens-details
)
if [ "$MODE" = single ]; then
  ARGS+=(--max-num-seqs 1 --gpu-memory-utilization 0.75
         --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_num_of_warmups":5}')
else
  ARGS+=(--max-num-seqs 6 --gpu-memory-utilization 0.90 --kv-cache-memory-bytes 29600000000
         --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_num_of_warmups":5,"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,12,16,20,24]}')
fi

docker stop -t 15 fn-local >/dev/null 2>&1 || true
docker rm fn-local >/dev/null 2>&1 || true
sudo -n sync && sudo -n sysctl -w vm.drop_caches=3 >/dev/null   # unified memory: free the page cache before loading
docker run -d --init --name fn-local --runtime nvidia --network host --ipc=host "${ENV[@]}" "${MNT[@]}" "$IMAGE" "${ARGS[@]}" "$@" >/dev/null
echo "fn-local starting: $RECIPE $MODE (load takes 5-8 min; ready when GET :8000/v1/models answers)"
