"""Check the overlay: logs, bitwise vs stock mm_mxfp8, torch.compile + CUDA graph, timing."""
import os, statistics, torch
os.environ.setdefault("VLLM_MXFP8_TACTIC", "256:8")
import vllm.model_executor.layers.quantization.utils.mxfp8_utils as mu
from vllm.model_executor.kernels.linear.mxfp8 import flashinfer as k
from flashinfer import mm_mxfp8
print("pins", k._THOR_TACTICS)
kern = object.__new__(k.FlashInferCutlassMxfp8LinearKernel)
N, K = 16384, 2560
layer = torch.nn.Module()
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
wq, s2 = mu.mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
layer.weight = torch.nn.Parameter(wq, requires_grad=False)
layer.weight_scale = torch.nn.Parameter(s2, requires_grad=False)
kern.process_weights_after_loading(layer)
f = torch.compile(lambda x: kern.apply_weights(layer, x), fullgraph=True, dynamic=True)
for M in (1, 4, 16, 256, 257, 8192):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    a, asf = mu.mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    stock = mm_mxfp8(a, layer.weight.t(), asf, layer.weight_scale, out_dtype=torch.bfloat16, backend="cutlass")
    eager = kern.apply_weights(layer, x)
    comp = f(x)
    print(f"M={M}: eager==stock {torch.equal(eager, stock)}  compiled==stock {torch.equal(comp, stock)}")
# CUDA graph of the compiled call at M=4, timing vs stock in graph
x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
def tg(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    ts = []
    for _ in range(50):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(ts), out
a, asf = mu.mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
t1, o1 = tg(lambda: f(x))
t0, o0 = tg(lambda: mm_mxfp8(a, layer.weight.t(), asf, layer.weight_scale, out_dtype=torch.bfloat16, backend="cutlass"))
print(f"graph M=4 qkvz (L2-warm, incl. act quant for overlay): overlay {t1:.1f} us, stock gemm {t0:.1f} us, equal {torch.equal(o0, o1)}")
