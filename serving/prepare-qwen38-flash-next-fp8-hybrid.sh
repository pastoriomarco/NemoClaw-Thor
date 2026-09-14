#!/usr/bin/env bash
# Create a linked sibling snapshot with only dense side weights rewritten as FP8.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
hf_cache=${HF_CACHE:-$HOME/thor-hf-cache}
image=${FLASHNEXT_IMAGE:-nemoclaw-thor/qwen38-flash-next-vllm:sm110-fast-v3}
base_rev=${FLASHNEXT_BASE_REV:-7b719225242aacd3dbd3f9407468c2ee9a9d2594}
hybrid_rev=${FLASHNEXT_HYBRID_REV:-${base_rev}-fp8hybrid}
model_root="hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots"
src_host="$hf_cache/$model_root/$base_rev"
dst_host="$hf_cache/$model_root/$hybrid_rev"
stage_host="${dst_host}.building"
src_in="/hf/$model_root/$base_rev"
dst_in="/hf/$model_root/$hybrid_rev"
stage_in="${dst_in}.building"

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
docker image inspect "$image" >/dev/null 2>&1 || {
    echo "Missing image $image; run ./serving/build-qwen38-flash-next-fast.sh first." >&2
    exit 1
}

echo "Preparing $hybrid_rev: linked base snapshot plus four converted shards."
docker run --rm --pull never \
    -v "$hf_cache:/hf" -v "$repo_root:/recipe:ro" \
    -e SRC_IN="$src_in" -e DST_IN="$dst_in" -e STAGE_IN="$stage_in" \
    --entrypoint bash "$image" -lc '
set -euo pipefail
cp -a "$SRC_IN" "$STAGE_IN"
cp --remove-destination "$(readlink -f "$STAGE_IN/model.safetensors.index.json")" \
    "$STAGE_IN/model.safetensors.index.json"
python3 /recipe/serving/tools/qwen38-flash-next/fp8_convert.py "$STAGE_IN" \
    | tee "$STAGE_IN/fp8_convert.log"
scale_count=$(python3 -c "import json,os; p=os.environ[\"STAGE_IN\"]+\"/model.safetensors.index.json\"; print(sum(k.endswith(\"weight_scale_inv\") for k in json.load(open(p))[\"weight_map\"]))")
test "$scale_count" -eq 300
chmod 644 "$STAGE_IN"/*.safetensors "$STAGE_IN/model.safetensors.index.json"
touch "$STAGE_IN/.prepared"
mv "$STAGE_IN" "$DST_IN"
'

test -f "$dst_host/.prepared"
echo "Prepared: $dst_host"
echo "Only 4 shards are real files; unchanged checkpoint files link to the base snapshot."
