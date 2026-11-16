"""MXFP8 CUTLASS tactic sweep for the FP4-Local side layers (C3, Thor sm_110).

Calls FlashInfer's CUTLASS MXFP8 runner exactly as vLLM's
FlashInferCutlassMxfp8LinearKernel does (A: MXFP8 [M,K] row-major with
F8_128x4-swizzled 1D scales from vLLM's mxfp8_e4m3_quantize; B: weight [N,K]
contiguous passed as weight.t(); B scales swizzled by vLLM's
swizzle_mxfp8_scale; BF16 out), once per tactic id.

Timing: per (shape, M, tactic) one CUDA graph holding `count` calls, each on a
distinct weight copy (so weights stream from DRAM as in a real step); replay
median of many runs divided by count. Output of every tactic is compared with
tactic -1 (= what vLLM runs with autotune off).

Run inside the vLLM image:  python3 /w/bench.py [--quick] [--ms=1,4,8192] [--tactics=8,9]
"""

import statistics
import sys
import time

import torch

import vllm.model_executor.layers.quantization.utils.mxfp8_utils as mu  # registers ops
from flashinfer.gemm.gemm_base import (
    DEFAULT_WORKSPACE_SIZE,
    _get_cache_buf,
    get_cutlass_mxfp8_gemm_module,
)
from flashinfer import mm_mxfp8

QUICK = "--quick" in sys.argv
dev = torch.device("cuda")
torch.manual_seed(0)

# (name, N, K, matrices per decode step) -- vLLM fused shapes
SHAPES = [
    ("gdn.in_proj_qkvz", 16384, 2560, 36),  # qkv 10240 + z 6144
    ("out_proj+o_proj", 2560, 6144, 48),  # 36 GDN out_proj + 12 QSA o_proj
    ("qsa.qkv_proj", 13952, 2560, 12),  # q+gate 12288 + k 512 + v 512 + indexer 640
    ("shexp.gate_up", 1280, 2560, 48),
    ("shexp.down", 2560, 640, 48),
]
MS = [1, 4, 8192]
for arg in sys.argv[1:]:
    if arg.startswith("--ms="):  # e.g. --ms=8,16,64,256
        MS = [int(v) for v in arg[5:].split(",")]

major, minor = torch.cuda.get_device_capability()
runner = get_cutlass_mxfp8_gemm_module(major).cutlass_mxfp8_gemm_runner()
mod_tactics = runner.get_valid_tactics(None, None)
TACTICS = [-1] + list(mod_tactics)
for arg in sys.argv[1:]:
    if arg.startswith("--tactics="):  # e.g. --tactics=8,9 (default -1 always included)
        TACTICS = [-1] + [int(v) for v in arg[10:].split(",")]
ws = _get_cache_buf("mm_mxfp8_workspace", DEFAULT_WORKSPACE_SIZE, dev)

# Config order in mxfp8_gemm_cutlass_template.h getConfigs(): tile x cluster
# grid reordered so {22,20,29,4,18} come first. Decode for the label only.
TILES = ["128x64x128", "128x256x128", "128x128x256", "128x256x256"]
CLUS = ["1x1", "1x2", "2x1", "2x2", "1x4", "2x4", "4x2", "4x4"]
_order = [22, 20, 29, 4, 18] + [i for i in range(32) if i not in (22, 20, 29, 4, 18)]


def label(t):
    if t < 0:
        return "default(-1)"
    i = _order[t]
    return f"t{t}:{TILES[i // 8]}/c{CLUS[i % 8]}"


def make_weight(N, K):
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    wq, ws2d = mu.mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
    wsw = mu.swizzle_mxfp8_scale(ws2d[:N, : K // 32].contiguous(), M=N, K=K)
    return wq.contiguous(), wsw.contiguous(), ws2d


def call(t, a, asf, w, wsf, out):
    runner.forward([a, w.t(), asf, wsf, torch.bfloat16, out, ws], tactic=t)


def time_graph(fn, reps):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    del g
    return statistics.median(ts)


print(f"device sm_{major}{minor}, SMs {torch.cuda.get_device_properties(0).multi_processor_count}, "
      f"{len(mod_tactics)} CUTLASS tactics", flush=True)
results = {}  # (shape, M) -> {tactic: us_per_call}
exact = {}  # (shape, M) -> {tactic: "bit" | maxdiff}
t_start = time.time()
for name, N, K, cnt in SHAPES:
    copies = 2 if QUICK else cnt
    weights = [make_weight(N, K) for _ in range(copies)]
    for M in MS:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        a, asf = mu.mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        ncalls = min(copies, 4) if M > 512 else copies
        reps = 10 if M > 512 else (30 if QUICK else 100)
        outs = [torch.empty(M, N, device=dev, dtype=torch.bfloat16) for _ in range(ncalls)]
        w0, wsf0, w2d0 = weights[0]
        # reference: vLLM's actual call path (backend cutlass, autotune off)
        ref = mm_mxfp8(a, w0.t(), asf, wsf0, out_dtype=torch.bfloat16, backend="cutlass")
        if M == 4 and not QUICK:  # sanity: default vs dequantized fp32 reference
            a2, asf2 = mu.mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
            dq = (mu.dequant_mxfp8_to_bf16(a2, asf2[:, : K // 32]).float()
                  @ mu.dequant_mxfp8_to_bf16(w0, w2d0[:, : K // 32]).float().t())
            rel = ((ref.float() - dq).norm() / dq.norm()).item()
            print(f"  sanity {name} M=4: rel err vs dequant ref {rel:.2e}", flush=True)
        res, ex = {}, {}
        for t in TACTICS:
            o = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
            try:
                call(t, a, asf, w0, wsf0, o)
                torch.cuda.synchronize()
            except Exception as e:  # tactic not runnable for this shape/arch
                res[t] = None
                ex[t] = type(e).__name__
                torch.cuda.synchronize()
                continue
            if torch.equal(o, ref):
                ex[t] = "bit"
            else:
                ex[t] = f"{(o.float() - ref.float()).abs().max().item():.3g}"

            def fn(t=t):
                for i in range(ncalls):
                    w, wsf, _ = weights[i % copies]
                    call(t, a, asf, w, wsf, outs[i])

            res[t] = time_graph(fn, reps) / ncalls
        results[(name, M)] = res
        exact[(name, M)] = ex
        ok = {t: v for t, v in res.items() if v is not None}
        best = min(ok, key=ok.get)
        print(f"{name:18s} N={N:5d} K={K:4d} M={M:5d}: default {res[-1]:8.1f} us  "
              f"best {label(best)} {ok[best]:8.1f} us  [{ex[best]}]  "
              f"failed={sum(1 for v in res.values() if v is None)}", flush=True)
    del weights
    torch.cuda.empty_cache()

# ---- summary --------------------------------------------------------------
print("\n== per-tactic weighted totals (us per step = sum count*us); decode M=4 and M=1 ==")
for M in MS:
    tot = {}
    for t in TACTICS:
        s = 0.0
        for name, N, K, cnt in SHAPES:
            v = results[(name, M)].get(t)
            if v is None:
                s = None
                break
            s += cnt * v
        tot[t] = s
    ok = {t: v for t, v in tot.items() if v is not None}
    order = sorted(ok, key=ok.get)
    print(f"M={M}: default {tot[-1]:.0f} us; top5: " +
          ", ".join(f"{label(t)}={ok[t]:.0f}" for t in order[:5]))
    # best per-shape table
    bs = 0.0
    for name, N, K, cnt in SHAPES:
        r = {t: v for t, v in results[(name, M)].items() if v is not None}
        b = min(r, key=r.get)
        bs += cnt * r[b]
    print(f"M={M}: best-per-shape total {bs:.0f} us (saving {tot[-1] - bs:.0f} us)")

print("\n== per-shape top 4 at each M (us/call, exactness vs default) ==")
for name, N, K, cnt in SHAPES:
    for M in MS:
        r = {t: v for t, v in results[(name, M)].items() if v is not None}
        order = sorted(r, key=r.get)[:4]
        print(f"{name:18s} M={M:5d} def={r[-1]:8.1f} | " +
              " | ".join(f"{label(t)} {r[t]:.1f} [{exact[(name, M)][t]}]" for t in order))

nonbit = sorted({(label(t), v) for k, d in exact.items() for t, v in d.items() if v != "bit"})
print("\nnon-bitwise or failed tactics (label, maxdiff/exception):", nonbit[:40])
print(f"elapsed {time.time() - t_start:.0f} s")
