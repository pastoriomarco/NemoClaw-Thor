#!/usr/bin/env bash
# Patch dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B-NVFP4 (W4A16) config.json
# to add the 62 MoE router gates to the quantization_config.ignore list.
#
# Why: the upstream checkpoint (v1.0 as of 2026-04-19) declares the model as
# fully NVFP4-quantized except for lm_head, but the 62 block_sparse_moe.gate
# layers are actually stored as plain BF16 in the safetensors (never
# quantized). vLLM fails with
#   KeyError: 'layers.0.block_sparse_moe.gate.weight_scale'
# because it looks for NVFP4 scale tensors that don't exist for those layers.
#
# The author applied this exact fix to the sibling W4A4 (GB10) variant in
# v1.1.1 but hasn't propagated it to the W4A16 repo. We fix it locally.
#
# This script is idempotent — re-running it after it's already applied is a
# no-op. If `hf download` re-pulls the blob (unlikely, content-addressed),
# re-run this script.

set -euo pipefail

REPO_DIR="${HOME}/thor-hf-cache/hub/models--dervig--m51Lab-MiniMax-M2.7-REAP-139B-A10B-NVFP4"

if [[ ! -d "${REPO_DIR}" ]]; then
    echo "ERROR: MiniMax W4A16 checkpoint not found at ${REPO_DIR}" >&2
    echo "       Download first: hf download dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B-NVFP4" >&2
    exit 1
fi

# Find the (single) snapshot dir
SNAPSHOT_DIR=$(ls -d "${REPO_DIR}/snapshots/"*/ 2>/dev/null | head -1)
if [[ -z "${SNAPSHOT_DIR}" ]]; then
    echo "ERROR: no snapshots/ dir inside ${REPO_DIR}" >&2
    exit 1
fi

CONFIG_FILE="${SNAPSHOT_DIR}config.json"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: config.json not found at ${CONFIG_FILE}" >&2
    exit 1
fi

echo "Patching ${CONFIG_FILE}"
echo "  (snapshot: $(basename "${SNAPSHOT_DIR%/}"))"

python3 - "${CONFIG_FILE}" <<'PYEOF'
import json
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])

# HF cache stores config.json as a symlink to blobs/<hash>. Editing a symlink
# writes to the blob. That's fine, but if the same blob is referenced by
# another snapshot (rare but possible) we'd mutate that too. Safer: replace
# the symlink with a real file carrying the patched content.
if path.is_symlink():
    real = path.resolve()
    print(f"  (symlink → {real})")

with path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

qc = cfg.get("quantization_config", {})
ignore = qc.get("ignore", [])

num_layers = cfg.get("num_hidden_layers")
if num_layers is None:
    print("ERROR: config.json missing num_hidden_layers", file=sys.stderr)
    sys.exit(1)

gate_entries = [
    f"model.layers.{i}.block_sparse_moe.gate" for i in range(num_layers)
]

existing = set(ignore)
to_add = [g for g in gate_entries if g not in existing]

if not to_add:
    print(f"  ✓ already patched — all {num_layers} gate entries present in ignore list")
    sys.exit(0)

# Preserve order: keep existing entries, append new ones
ignore_patched = list(ignore) + to_add
qc["ignore"] = ignore_patched
cfg["quantization_config"] = qc

# Break the symlink (if any) and write a fresh file to the snapshot path
if path.is_symlink():
    path.unlink()

with path.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")

print(f"  ✓ added {len(to_add)} gate entries to ignore list ({num_layers} total layers)")
print(f"  ignore list now has {len(ignore_patched)} entries")
PYEOF

echo "Done."
