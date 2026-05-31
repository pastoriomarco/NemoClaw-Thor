#!/bin/bash
# Unlock CUTLASS DSL NVFP4 BlockScaledMmaOp + vLLM FlashInfer NVFP4 MoE
# backends on Thor (sm_110a).
#
# Background: CUTLASS DSL 4.5.x hardcodes BlockScaledMmaOp.admissible_archs
# to [sm_100a, sm_103a]. vLLM's NVFP4 MoE backend gates (FlashInfer CuteDSL,
# FlashInfer CUTLASS, FlashInfer CuteDSL Batched, TRT-LLM NVFP4) all check
# `p.is_device_capability_family(100)` exclusively, locking out sm_110a.
#
# Thor (sm_110a) IS Blackwell with native FP4 hardware (verified 875 TF
# dense FP4 by community via CUTLASS C++). Patches:
#   1. CUTLASS DSL: BlockScaledMmaOp admissible_archs += sm_110a
#   2-5. vLLM flashinfer_cutedsl_moe / cutedsl_batched / cutlass / trtllm_nvfp4
#       arch-family gates: add `or is_device_capability_family(110)`
#
# Mirror of cutlass#3096 (DGX Spark sm_121 community patch). Each patch is
# idempotent (marker-based: skips if already applied).
#
# Unlocks native sm_110a FP4 tensor cores → 2x+ throughput vs Marlin
# software-FP4 fallback.
set -euo pipefail

CUTLASS_PKG="/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass"
VLLM_PKG="/usr/local/lib/python3.12/dist-packages/vllm"
MMA="${CUTLASS_PKG}/cute/nvgpu/tcgen05/mma.py"

drop_pyc() {
    local file="$1"
    local pyc_dir
    pyc_dir="$(dirname "${file}")/__pycache__"
    local stem
    stem="$(basename "${file}" .py)"
    if [ -d "${pyc_dir}" ]; then
        rm -f "${pyc_dir}/${stem}.cpython-"*.pyc 2>/dev/null || true
    fi
}

# ---- Patch 1: CUTLASS DSL BlockScaledMmaOp ----
if [ -f "${MMA}" ]; then
    python3 - "${MMA}" <<'PYEOF'
import sys
from pathlib import Path
p = Path(sys.argv[1])
text = p.read_text()
needle = "    admissible_archs = [\n        Arch.sm_100a,\n        Arch.sm_103a,\n    ]"
replacement = "    admissible_archs = [\n        Arch.sm_100a,\n        Arch.sm_103a,\n        Arch.sm_110a,\n    ]"
if "Arch.sm_110a" in text and needle not in text:
    print("sm110a-fp4-dsl-unlock[cutlass-dsl]: already patched; no-op")
    sys.exit(0)
if needle not in text:
    print(f"sm110a-fp4-dsl-unlock[cutlass-dsl]: needle not found in {p}; cutlass-dsl drifted, skipping", file=sys.stderr)
    sys.exit(0)
p.write_text(text.replace(needle, replacement, 1))
print("sm110a-fp4-dsl-unlock[cutlass-dsl]: patched BlockScaledMmaOp.admissible_archs += sm_110a")
PYEOF
    drop_pyc "${MMA}"
else
    echo "sm110a-fp4-dsl-unlock[cutlass-dsl]: ${MMA} not found; skipping" >&2
fi

# ---- Patches 2-5: vLLM FlashInfer NVFP4 MoE backend arch gates ----
patch_vllm_gate() {
    local file="$1"
    local label="$2"
    if [ ! -f "${file}" ]; then
        echo "sm110a-fp4-dsl-unlock[${label}]: file not found; skipping" >&2
        return 0
    fi
    python3 - "${file}" "${label}" <<'PYEOF'
import sys, re
from pathlib import Path
p = Path(sys.argv[1])
label = sys.argv[2]
text = p.read_text()
# Idempotency marker: a (110) check within a few chars of a (100) check
already = re.search(
    r"is_device_capability_family\(100\)[^\n]*\n?[^\n]*is_device_capability_family\(110\)",
    text,
)
if already:
    print(f"sm110a-fp4-dsl-unlock[{label}]: already patched; no-op")
    sys.exit(0)
pat = re.compile(
    r"(?P<indent>[ \t]*)(?P<connector>(?:and|or)\s+)?(?P<expr>[a-zA-Z_][a-zA-Z0-9_.]*\.is_device_capability_family\(100\))",
)
def repl(m):
    indent = m.group("indent")
    connector = m.group("connector") or ""
    expr = m.group("expr")
    obj = expr.split(".is_device_capability_family")[0]
    return f"{indent}{connector}({expr} or {obj}.is_device_capability_family(110))"
new_text, count = pat.subn(repl, text)
if count == 0:
    print(f"sm110a-fp4-dsl-unlock[{label}]: no gate found; skipping", file=sys.stderr)
    sys.exit(0)
p.write_text(new_text)
print(f"sm110a-fp4-dsl-unlock[{label}]: patched {count} arch-family gate(s) to accept sm_110a")
PYEOF
    drop_pyc "${file}"
}

patch_vllm_gate "${VLLM_PKG}/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py" "flashinfer_cutedsl"
patch_vllm_gate "${VLLM_PKG}/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_batched_moe.py" "flashinfer_cutedsl_batched"
patch_vllm_gate "${VLLM_PKG}/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py" "flashinfer_cutlass"
patch_vllm_gate "${VLLM_PKG}/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py" "trtllm_nvfp4"

# ---- Patches 6-7: FlashInfer JIT GEMM + MoE utils arch lists ----
# Several `supported_major_versions=[10]` lines gate sm_100-only JIT kernels
# from compiling for sm_110. Sibling GEMM kernels in the same file already use
# [10, 11, 12], so adding 11 to the [10]-only ones is consistent. Lines:
#   - flashinfer/jit/gemm/core.py:73   (gen_mm_bf16_cublaslt_module)
#   - flashinfer/jit/moe_utils.py:78   (gen_moe_utils_module — CuteDSL NVFP4 path)
patch_flashinfer_arch_list() {
    local file="$1"
    local label="$2"
    if [ ! -f "${file}" ]; then
        echo "sm110a-fp4-dsl-unlock[${label}]: file not found; skipping" >&2
        return 0
    fi
    python3 - "${file}" "${label}" <<'PYEOF'
import sys
from pathlib import Path
p = Path(sys.argv[1])
label = sys.argv[2]
text = p.read_text()
needle = "        supported_major_versions=[10]\n"
replacement = "        supported_major_versions=[10, 11]\n"
if needle not in text:
    if "supported_major_versions=[10, 11]" in text:
        print(f"sm110a-fp4-dsl-unlock[{label}]: already patched; no-op")
    else:
        print(f"sm110a-fp4-dsl-unlock[{label}]: needle not found; skipping", file=sys.stderr)
    sys.exit(0)
p.write_text(text.replace(needle, replacement, 1))
print(f"sm110a-fp4-dsl-unlock[{label}]: patched supported_major_versions to accept major=11")
PYEOF
    drop_pyc "${file}"
}
patch_flashinfer_arch_list "/usr/local/lib/python3.12/dist-packages/flashinfer/jit/gemm/core.py" "flashinfer_gemm_core"
patch_flashinfer_arch_list "/usr/local/lib/python3.12/dist-packages/flashinfer/jit/moe_utils.py" "flashinfer_moe_utils"

# ---- Patch 8: copy Thor MoE configs into vLLM's configs directory ----
# vLLM looks up fused-MoE configs at
# /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/
# Bind-mounting individual files there is awkward, so we copy from a known
# host-mount location (set by launch.sh). The configs dir is writable inside
# the container.
THOR_MOE_CFG_SRC="/opt/nemoclaw-thor/moe-configs"
VLLM_MOE_CFG_DST="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs"
if [ -d "${THOR_MOE_CFG_SRC}" ] && [ -d "${VLLM_MOE_CFG_DST}" ]; then
    count=0
    for cfg in "${THOR_MOE_CFG_SRC}"/*.json; do
        [ -e "${cfg}" ] || continue
        cp -f "${cfg}" "${VLLM_MOE_CFG_DST}/"
        count=$((count + 1))
    done
    if [ ${count} -gt 0 ]; then
        echo "sm110a-fp4-dsl-unlock[moe_configs]: copied ${count} Thor MoE config(s) into vLLM configs dir"
    fi
fi

echo "sm110a-fp4-dsl-unlock: done"
