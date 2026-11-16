#!/bin/bash
# SM110 NVFP4 MoE spike runner (Thor). Usage:
#   ~/fn-moe-spike/code/run.sh gpu  [extra spike.py args]   # needs the GPU free (no vLLM serving)
#   ~/fn-moe-spike/code/run.sh dry  [extra spike.py args]   # CPU only, no GPU, safe while serving
#   ~/fn-moe-spike/code/run.sh prebuild                     # CPU only: build moe_utils for sm_110a
# Everything runs in the stock vLLM 0.33 image; mounts are read-only except the spike's own dirs.
# Docker's default runtime on Thor is nvidia, so dry/prebuild force --runtime runc (no GPU, no driver).
set -euo pipefail
MODE=${1:-dry}; shift || true
IMG=vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e
B=$HOME/fn-moe-spike
V=/usr/local/lib/python3.12/dist-packages/vllm
CSF=$HOME/fn-local-patch/vllm/model_executor/layers/quantization/utils/nvfp4_utils.py   # vLLM PR #64312 CSF decode
mkdir -p "$B/fiws" "$B/runs"
STAMP=$(date +%Y%m%d-%H%M%S)
COMMON=(--rm --network none -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1
  -e FLASHINFER_CUDA_ARCH_LIST=11.0a -e FLASHINFER_WORKSPACE_BASE=/ws -e CUTE_DSL_CACHE_DIR=/ws/cute-dsl-cache
  -v "$HOME/thor-hf-cache:/hf:ro" -v "$CSF:$V/model_executor/layers/quantization/utils/nvfp4_utils.py:ro"
  -v "$B/code:/code:ro" -v "$B/fiws:/ws" -w /code --entrypoint python3)
case "$MODE" in
  gpu)
    if docker ps --format '{{.Names}}' | grep -qx fn-local; then echo "fn-local is running: GPU not free" >&2; exit 1; fi
    OUT=$B/runs/gpu-$STAMP; mkdir -p "$OUT"
    rm -rf "$B/fiws/cute-dsl-cache"   # cold CuTe DSL compile is part of the measurement
    timeout 3600 docker run --name fn-moe-spike --runtime nvidia --ipc=host "${COMMON[@]}" -v "$OUT:/out" \
      $IMG /code/spike.py --out /out "$@" 2>&1 | tee "$OUT/log.txt"
    echo "results: $OUT/results.json  summary: $OUT/summary.md" ;;
  dry)
    OUT=$B/runs/dry-$STAMP; mkdir -p "$OUT"
    timeout 1800 docker run --name fn-moe-spike-dry --runtime runc -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= --cpus 2 --memory 8g "${COMMON[@]}" -v "$OUT:/out" \
      $IMG /code/spike.py --dry-run --out /out "$@" 2>&1 | tee "$OUT/log.txt" ;;
  prebuild)
    timeout 5400 docker run --name fn-moe-prebuild --runtime runc -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= --cpus 3 --memory 8g -e MAX_JOBS=3 "${COMMON[@]}" \
      $IMG /code/prebuild_moe_utils.py 2>&1 | tee "$B/prebuild.log" ;;
  *) echo "mode: gpu | dry | prebuild" >&2; exit 2 ;;
esac
