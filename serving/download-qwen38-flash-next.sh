#!/usr/bin/env bash
# Download the exact verified checkpoint into the persistent Thor HF cache.
set -euo pipefail

hf_cache=${HF_CACHE:-$HOME/thor-hf-cache}
image=${FLASHNEXT_IMAGE:-nemoclaw-thor/qwen38-flash-next-vllm:sm110-fast-v3}
model_rev=${FLASHNEXT_MODEL_REV:-7b719225242aacd3dbd3f9407468c2ee9a9d2594}
model_rel="hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/$model_rev"

if [[ -s "$hf_cache/$model_rel/config.json" ]]; then
    echo "Pinned checkpoint already present: $hf_cache/$model_rel"
    exit 0
fi

mkdir -p "$hf_cache"
docker image inspect "$image" >/dev/null 2>&1 || {
    echo "Missing image $image; run ./serving/build-qwen38-flash-next-fast.sh first." >&2
    exit 1
}

docker run --rm --pull never --user "$(id -u):$(id -g)" \
    -v "$hf_cache:/hf" -e HF_HOME=/hf -e MODEL_REV="$model_rev" \
    --entrypoint python3 "$image" \
    -c 'import os; from huggingface_hub import snapshot_download; snapshot_download("RadixArk/Qwen3.8-Flash-Next-NVFP4", revision=os.environ["MODEL_REV"])'

test -s "$hf_cache/$model_rel/config.json"
echo "Downloaded pinned checkpoint to $hf_cache/$model_rel"
