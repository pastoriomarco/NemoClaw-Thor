"""CPU-only cross-compile probe: FlashInfer 0.7.2.post1 CuTe DSL NVFP4 MoE kernels for sm_110a.

usage: CUTE_DSL_ARCH=sm_110a python3 xcompile_sm110a.py {fc1|fc2} {128|256} [fused|twostage]
Compiles the FC1 (gather + grouped GEMM + SwiGLU + FP4 requant) or FC2 (grouped GEMM + finalize)
kernel class with Flash-Next run-time sizes and fake device addresses (nothing is executed), then
checks the generated PTX target and the block-scaled tcgen05 MMA instruction.
"""
import os, re, sys, tempfile, time

assert os.environ.get("CUTE_DSL_ARCH") == "sm_110a", "export CUTE_DSL_ARCH=sm_110a before starting"
dump = tempfile.mkdtemp(prefix="xc_")
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", dump)
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import make_ptr
from flashinfer.tllm_enums import ActivationType
from flashinfer.fused_moe.cute_dsl.blackwell.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
    BlockScaledContiguousGatherGroupedGemmKernel as Fc1,
)
from flashinfer.fused_moe.cute_dsl.blackwell.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel as Fc2,
)

which, nt = sys.argv[1], int(sys.argv[2])
fin = sys.argv[3] if len(sys.argv) > 3 else "fused"
H, I, E, TOPK, SMS = 2560, 640, 512, 10, 20
gm = lambda dt, al=16: make_ptr(dt, 1 << 20, cute.AddressSpace.gmem, assumed_align=al)  # noqa: E731
F4, F8, F32, I32 = cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, cutlass.Float32, cutlass.Int32
t = time.time()
try:
    if which == "fc1":
        k = Fc1(sf_vec_size=16, mma_tiler_mn=(128, nt), cluster_shape_mn=(1, 1), vectorized_f32=True, topk=TOPK,
                raster_along_m=False, enable_pdl=True, activation_type=ActivationType.Swiglu.value)
        cute.compile(k.wrapper, gm(F4, 32), gm(F4, 32), gm(F8), gm(F8), gm(F4, 32), gm(F8), gm(F32, 4), gm(I32, 4),
                     gm(I32, 4), gm(I32, 4), gm(I32, 4), gm(F32, 4), None, 4, 128 * E, 2 * I, H, E, tile_size=128,
                     scaling_vector_size=16, max_active_clusters=SMS, stream=cuda.CUstream(0))
    else:
        k = Fc2(sf_vec_size=16, mma_tiler_mn=(128, nt), cluster_shape_mn=(1, 1), raster_along_m=False,
                enable_pdl=True, use_fused_finalize=(fin == "fused"))
        cute.compile(k.wrapper, gm(F4, 32), gm(F4, 32), gm(F8), gm(F8), gm(cutlass.BFloat16), gm(F32, 4), gm(I32, 4),
                     gm(I32, 4), gm(I32, 4), gm(I32, 4), gm(F32, 4), None, 128 * E, H, I, E, 4, TOPK, tile_size=128,
                     scaling_vector_size=16, max_active_clusters=SMS, stream=cuda.CUstream(0))
    print(f"COMPILED {which} n{nt} {fin if which == 'fc2' else ''} in {time.time() - t:.1f} s")
except BaseException as e:  # noqa: BLE001
    print(f"FAILED {which} n{nt}: {type(e).__name__}: {str(e)[:800]}")
    sys.exit(1)
for f in os.listdir(dump):
    if f.endswith(".ptx"):
        s = open(os.path.join(dump, f)).read()
        tgt = re.search(r"^\.target (\S+)", s, re.M)
        mma = sorted(set(re.findall(r"tcgen05\.mma[\w:.]*", s)))
        print(f"PTX target={tgt.group(1) if tgt else '?'} mma={mma} griddepcontrol={s.count('griddepcontrol')}")
