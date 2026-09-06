#!/usr/bin/env bash
# Thin, offline launcher for the separately built Thor fused-GDN image.
# Does not stop/delete existing containers. See docs/QWEN38-FLASH-NEXT-FAST-THOR.md.
set -euo pipefail
HF_CACHE=${HF_CACHE:-$HOME/thor-hf-cache}
VLLM_CACHE=${VLLM_CACHE:-$HOME/thor-vllm-cache}
TORCH_CACHE=${TORCH_CACHE:-$HOME/thor-torch-cache}
FLASHINFER_CACHE=${FLASHINFER_CACHE:-$HOME/thor-flashinfer-cache}
FLASHNEXT_IMAGE=${FLASHNEXT_IMAGE:-nemoclaw-thor/qwen38-flash-next-vllm:sm110-gdn-cuda}
FLASHNEXT_CONTAINER=${FLASHNEXT_CONTAINER:-qwen38-flash-next-fast}
FLASHNEXT_MODEL_REV=${FLASHNEXT_MODEL_REV:-7b719225242aacd3dbd3f9407468c2ee9a9d2594}
FLASHNEXT_PORT=${FLASHNEXT_PORT:-8050}
FLASHNEXT_MTP=${FLASHNEXT_MTP:-3}
FLASHNEXT_GPU_MEM=${FLASHNEXT_GPU_MEM:-0.90}
FLASHNEXT_KV=${FLASHNEXT_KV:-auto}
FLASHNEXT_GDN=${FLASHNEXT_GDN:-cuda}
FLASHNEXT_MAX_SEQS=${FLASHNEXT_MAX_SEQS:-4}
FLASHNEXT_CONTEXT=${FLASHNEXT_CONTEXT:-262144}
model_rel="hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/$FLASHNEXT_MODEL_REV"
test -s "$HF_CACHE/$model_rel/config.json" || {
    echo "Local model snapshot missing: $HF_CACHE/$model_rel" >&2; exit 1;
}
mkdir -p "$VLLM_CACHE" "$TORCH_CACHE" "$FLASHINFER_CACHE"
split_ops='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
run_mode=(-it)
if [[ ${FLASHNEXT_DETACH:-0} == 1 ]]; then run_mode=(-d); fi
exec docker run "${run_mode[@]}" --pull never \
    --name "$FLASHNEXT_CONTAINER" --runtime nvidia --gpus all \
    --ipc host --network host --shm-size 16g \
    -v "$HF_CACHE:/hf" -v "$VLLM_CACHE:/root/.cache/vllm" \
    -v "$TORCH_CACHE:/root/.cache/torch" -v "$FLASHINFER_CACHE:/root/.cache/flashinfer" \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
    -e VLLM_DISABLE_COMPILE_CACHE=1 -e VLLM_USE_AOT_COMPILE=1 \
    -e CUTE_DSL_ARCH=sm_110a -e TORCH_CUDA_ARCH_LIST=11.0a \
    -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS=14 -e VLLM_PLE_MMAP_PREWARM=0 \
    -e VLLM_PLE_MMAP_MADV_RANDOM="${VLLM_PLE_MMAP_MADV_RANDOM:-1}" \
    -e VLLM_PLE_MMAP_FAST_ROWS="${VLLM_PLE_MMAP_FAST_ROWS:-0}" \
    -e VLLM_QSA_DET_TOPK=1 -e VLLM_QSA_DET_LIB=/opt/llm/kernel-det/_C_det.so \
    -e VLLM_QSA_EXACT_TOPK=0 -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e VLLM_GDN_DECODE_KERNEL="$FLASHNEXT_GDN" \
    --entrypoint vllm "$FLASHNEXT_IMAGE" serve "/hf/$model_rel" \
    --served-model-name qwen3.8-flash-next --host 0.0.0.0 --port "$FLASHNEXT_PORT" \
    --load-format safetensors --max-model-len "$FLASHNEXT_CONTEXT" \
    --max-num-seqs "$FLASHNEXT_MAX_SEQS" --gpu-memory-utilization "$FLASHNEXT_GPU_MEM" \
    --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 \
    -cc.cudagraph_mode=PIECEWISE "-cc.splitting_ops=$split_ops" \
    --no-enable-flashinfer-autotune --moe-backend flashinfer_cutlass \
    --kv-cache-dtype "$FLASHNEXT_KV" --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$FLASHNEXT_MTP}"
