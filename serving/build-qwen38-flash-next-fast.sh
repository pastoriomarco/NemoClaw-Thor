#!/usr/bin/env bash
# Build the complete pinned Qwen3.8-Flash-Next SM110 serving stack.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${FLASHNEXT_SOURCE:-$HOME/thor-qwen38-flash-next-vllm/source}
source_repo=https://github.com/blazux/qwen3.8-Flash-DGX.git
source_rev=4b723de2e2c465d866738b57ae64bde6e8c07744
final_image=${FLASHNEXT_IMAGE:-nemoclaw-thor/qwen38-flash-next-vllm:sm110-fast-v3}

command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }

if [[ ! -d "$source_dir/.git" ]]; then
    mkdir -p "$(dirname "$source_dir")"
    git clone "$source_repo" "$source_dir"
fi

if ! git -C "$source_dir" diff --quiet || ! git -C "$source_dir" diff --cached --quiet; then
    echo "Tracked changes found in $source_dir; preserve them before building." >&2
    exit 1
fi

if ! git -C "$source_dir" cat-file -e "$source_rev^{commit}" 2>/dev/null; then
    git -C "$source_dir" fetch origin "$source_rev"
fi
git -C "$source_dir" checkout --detach "$source_rev"

docker build --progress=plain --build-arg DET_ARCH=110a \
    -t nemoclaw-thor/qwen38-flash-next-vllm:sm110 "$source_dir"

docker build --progress=plain \
    -f "$repo_root/serving/docker/Dockerfile.qwen38-flash-next-sm110" \
    -t nemoclaw-thor/qwen38-flash-next-vllm:sm110-flashinfer-moe "$repo_root"

docker build --progress=plain \
    -f "$repo_root/serving/docker/Dockerfile.qwen38-flash-next-gdn-sm110" \
    -t nemoclaw-thor/qwen38-flash-next-vllm:sm110-gdn-cuda "$repo_root"

docker build --progress=plain \
    -f "$repo_root/serving/docker/Dockerfile.qwen38-flash-next-fast-v3" \
    -t "$final_image" "$repo_root"

echo "Built $final_image"
echo "Next: ./serving/download-qwen38-flash-next.sh (new machine only)"
echo "Then: ./serving/prepare-qwen38-flash-next-fp8-hybrid.sh"
echo "Then: ./serving/start-qwen38-flash-next-fast.sh"
