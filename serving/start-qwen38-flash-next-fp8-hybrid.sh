#!/usr/bin/env bash
# Recommended fast FP8-side-layer variant. KV cache deliberately remains BF16.
set -euo pipefail

base_rev=${FLASHNEXT_BASE_REV:-7b719225242aacd3dbd3f9407468c2ee9a9d2594}
export FLASHNEXT_MODEL_REV=${FLASHNEXT_MODEL_REV:-${base_rev}-fp8hybrid}
export FLASHNEXT_CONTAINER=${FLASHNEXT_CONTAINER:-qwen38-flash-next-fast-v3-fp8hybrid}
export FLASHNEXT_GPU_MEM=${FLASHNEXT_GPU_MEM:-0.90}
export FLASHNEXT_KV=auto
export FLASHNEXT_MTP=${FLASHNEXT_MTP:-3}
export FLASHNEXT_MAX_SEQS=${FLASHNEXT_MAX_SEQS:-3}
export FLASHNEXT_CONTEXT=${FLASHNEXT_CONTEXT:-262144}

exec "$(dirname "${BASH_SOURCE[0]}")/start-qwen38-flash-next-fast.sh"
