#!/usr/bin/env bash
# Download a five-file HF delta and assemble the linked local hybrid snapshot.
set -euo pipefail

hf_cache=${HF_CACHE:-$HOME/thor-hf-cache}
image=${FLASHNEXT_IMAGE:-nemoclaw-thor/qwen38-flash-next-vllm:sm110-fast-v3}
base_rev=${FLASHNEXT_BASE_REV:-7b719225242aacd3dbd3f9407468c2ee9a9d2594}
hybrid_rev=${FLASHNEXT_HYBRID_REV:-${base_rev}-fp8hybrid}
delta_repo=${FLASHNEXT_DELTA_REPO:-pastoriomarco/Qwen3.8-Flash-Next-NVFP4-FP8-Hybrid-Delta}
delta_rev=${FLASHNEXT_DELTA_REV:-358162ffaaeb4372c79843d9cd1f67d3654593da}
model_root="hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots"
src_host="$hf_cache/$model_root/$base_rev"
dst_host="$hf_cache/$model_root/$hybrid_rev"
stage_host="${dst_host}.building"
delta_key=${delta_repo//\//--}
delta_dir="$hf_cache/deltas/$delta_key/$delta_rev"

test -s "$src_host/config.json" || {
    echo "Base snapshot missing: $src_host" >&2
    echo "Run ./serving/download-qwen38-flash-next.sh first." >&2
    exit 1
}
if [[ -f "$dst_host/.prepared" ]]; then
    echo "FP8-hybrid snapshot already prepared: $dst_host"
    exit 0
fi
if [[ -e "$dst_host" || -e "$stage_host" ]]; then
    echo "Incomplete destination exists; inspect it before removing it:" >&2
    [[ -e "$dst_host" ]] && echo "  $dst_host" >&2
    [[ -e "$stage_host" ]] && echo "  $stage_host" >&2
    exit 1
fi

token_env=()
[[ -n ${HF_TOKEN:-} ]] && token_env=(-e HF_TOKEN)
mkdir -p "$delta_dir"
docker run --rm --pull never --network host \
    -v "$delta_dir:/delta" "${token_env[@]}" \
    -e DELTA_REPO="$delta_repo" -e DELTA_REV="$delta_rev" \
    --entrypoint python3 "$image" -c '
import os
from huggingface_hub import snapshot_download
files = [
    "model-bf16-00001.safetensors",
    "model-bf16-00010.safetensors",
    "model-bf16-00011.safetensors",
    "model-bf16-00012.safetensors",
    "model.safetensors.index.json",
    "fp8-hybrid-manifest.json",
]
snapshot_download(
    repo_id=os.environ["DELTA_REPO"], revision=os.environ["DELTA_REV"],
    local_dir="/delta", allow_patterns=files,
    token=os.environ.get("HF_TOKEN") or None,
)
'

docker run --rm --pull never \
    -v "$hf_cache:/hf" -v "$delta_dir:/delta:ro" \
    -e BASE_REV="$base_rev" -e STAGE_IN="/hf/$model_root/${hybrid_rev}.building" \
    -e DST_IN="/hf/$model_root/$hybrid_rev" -e SRC_IN="/hf/$model_root/$base_rev" \
    --entrypoint bash "$image" -lc '
set -euo pipefail
files=(
  model-bf16-00001.safetensors model-bf16-00010.safetensors
  model-bf16-00011.safetensors model-bf16-00012.safetensors
  model.safetensors.index.json
)
python3 - <<"PY"
import hashlib, json, os
manifest=json.load(open("/delta/fp8-hybrid-manifest.json"))
if manifest["base_revision"] != os.environ["BASE_REV"]:
    raise SystemExit("delta base revision does not match FLASHNEXT_BASE_REV")
for name, expected in manifest["files"].items():
    digest=hashlib.sha256()
    size=0
    with open("/delta/"+name,"rb") as source:
        while chunk := source.read(8*1024*1024):
            digest.update(chunk); size += len(chunk)
    if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
        raise SystemExit("delta checksum mismatch: "+name)
print("Delta manifest and SHA256 checks passed")
PY
cp -a "$SRC_IN" "$STAGE_IN"
for name in "${files[@]}"; do
    cp --remove-destination "/delta/$name" "$STAGE_IN/$name"
done
touch "$STAGE_IN/.prepared"
mv "$STAGE_IN" "$DST_IN"
'

test -f "$dst_host/.prepared"
echo "Prepared from pinned delta: $dst_host"
