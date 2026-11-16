"""Build FlashInfer's moe_utils JIT module for sm_110a without a GPU (CPU-only container).

Run with FLASHINFER_CUDA_ARCH_LIST="11.0a" and FLASHINFER_WORKSPACE_BASE=<persistent dir>; the
spike later mounts the same dir, so its GPU window does not pay this build.
Prints the build time; exits non-zero on a build failure (that would be a G0 blocker that a
Python overlay alone cannot lift if the .cu sources reject sm_110).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
assert os.environ.get("FLASHINFER_CUDA_ARCH_LIST") == "11.0a", "set FLASHINFER_CUDA_ARCH_LIST=11.0a"
from sm110_lift import lift_cutedsl_moe_gates
print("lifted:", lift_cutedsl_moe_gates(), flush=True)
import flashinfer.fused_moe.cute_dsl.moe_utils as mu
from flashinfer.jit import env as jit_env
spec = mu.gen_moe_utils_module()
print("jit dir:", jit_env.FLASHINFER_JIT_DIR, "so:", spec.jit_library_path, flush=True)
print("nvcc arch flags:", [f for f in spec.extra_cuda_cflags if "gencode" in f], flush=True)
if spec.jit_library_path.exists():
    print("already built"); sys.exit(0)
t = time.time()
spec.build(verbose=os.environ.get("VERBOSE") == "1")
print(f"BUILT moe_utils for sm_110a in {time.time()-t:.0f} s -> {spec.jit_library_path}", flush=True)
