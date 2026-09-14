#!/usr/bin/env bash
# Upload only the four converted shards and modified safetensors index to HF Hub.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
hf_cache=${HF_CACHE:-$HOME/thor-hf-cache}
image=${FLASHNEXT_IMAGE:-nemoclaw-thor/qwen38-flash-next-vllm:sm110-fast-v3}
base_rev=${FLASHNEXT_BASE_REV:-7b719225242aacd3dbd3f9407468c2ee9a9d2594}
hybrid_rev=${FLASHNEXT_HYBRID_REV:-${base_rev}-fp8hybrid}
delta_repo=${FLASHNEXT_DELTA_REPO:?Set FLASHNEXT_DELTA_REPO=username/repository}
private=${FLASHNEXT_DELTA_PRIVATE:-1}
snapshot="$hf_cache/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/$hybrid_rev"

test -f "$snapshot/.prepared" || {
    echo "Prepared hybrid snapshot missing: $snapshot" >&2
    exit 1
}

token_env=()
[[ -n ${HF_TOKEN:-} ]] && token_env=(-e HF_TOKEN)
if ! docker run --rm --pull never --network host \
    -v "$hf_cache:/hf-auth:ro" -e HF_HOME=/hf-auth \
    "${token_env[@]}" --entrypoint hf "$image" auth whoami >/dev/null; then
    echo "No valid Hugging Face login in $hf_cache; run the documented 'hf auth login' command." >&2
    exit 1
fi

docker run --rm --pull never --network host \
    -v "$snapshot:/model:ro" -v "$repo_root:/recipe:ro" \
    -v "$hf_cache:/hf-auth" -e HF_HOME=/hf-auth \
    "${token_env[@]}" -e DELTA_REPO="$delta_repo" -e DELTA_PRIVATE="$private" \
    -e BASE_REV="$base_rev" --entrypoint python3 "$image" -c '
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi

files = [
    "model-bf16-00001.safetensors",
    "model-bf16-00010.safetensors",
    "model-bf16-00011.safetensors",
    "model-bf16-00012.safetensors",
    "model.safetensors.index.json",
]
root = Path("/model")
for name in files:
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"expected a real converted file: {path}")

manifest = {
    "format": "nemoclaw-thor-qwen38-flash-next-fp8-delta-v1",
    "base_model": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
    "base_revision": os.environ["BASE_REV"],
    "conversion": "BF16 dense GDN/QSA/shared-expert weights to blockwise FP8 E4M3; 128x128 blocks; FP32 inverse scales",
    "files": {},
}
for name in files:
    digest = hashlib.sha256()
    size = 0
    with open(root / name, "rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    manifest["files"][name] = {"sha256": digest.hexdigest(), "size": size}

repo_id = os.environ["DELTA_REPO"]
api = HfApi(token=os.environ.get("HF_TOKEN") or None)
api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=os.environ["DELTA_PRIVATE"].lower() in ("1", "true", "yes"),
    exist_ok=True,
)
api.upload_file(
    path_or_fileobj="/recipe/serving/tools/qwen38-flash-next/README.fp8-delta.md",
    path_in_repo="README.md", repo_id=repo_id,
    commit_message="Add FP8 hybrid delta model card",
)
api.upload_file(
    path_or_fileobj=(json.dumps(manifest, indent=2) + "\n").encode(),
    path_in_repo="fp8-hybrid-manifest.json", repo_id=repo_id,
    commit_message="Add pinned delta manifest",
)
result = api.upload_folder(
    folder_path=root,
    repo_id=repo_id,
    allow_patterns=files,
    commit_message="Upload FP8 hybrid delta shards",
)
print(f"Uploaded https://huggingface.co/{repo_id}")
print(f"Pin this delta revision in the runbook: {result.oid}")
'
