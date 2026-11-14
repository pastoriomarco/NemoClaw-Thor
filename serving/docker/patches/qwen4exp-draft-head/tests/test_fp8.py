"""CPU round-trip error of the lm_head / MTP-dense FP8 choices on random data.

MXFP8 mirrors vLLM 0.33 _mxfp8_e4m3_quantize_torch (e8m0 = ceil(log2(amax/448))
per 32 inputs). PTPC = per-row weight + per-token activation e4m3 (stock
Fp8PtpcOnlineLinearMethod). Per-tensor = one weight scale + per-tensor dynamic act.
"""
import torch
torch.manual_seed(0)
F8 = torch.float8_e4m3fn; FMAX = 448.0

def mx(x):  # quantize-dequantize along the last dim in blocks of 32
    xb = x.float().reshape(*x.shape[:-1], -1, 32)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=torch.finfo(torch.float32).tiny)
    s = torch.exp2(torch.ceil(torch.log2(amax / FMAX)))
    return ((xb / s).to(F8).float() * s).reshape(x.shape)

def rowwise(x):
    s = x.float().abs().amax(-1, keepdim=True).clamp(min=1e-12) / FMAX
    return (x.float() / s).to(F8).float() * s

def tensorwise(x):
    s = x.float().abs().amax().clamp(min=1e-12) / FMAX
    return (x.float() / s).to(F8).float() * s

def rel(a, b): return ((a - b).norm() / b.norm()).item()

# lm_head-like weight: 32768 rows (subset of 248320) x 2560, row norms vary 4x,
# 0.1% outlier inputs channels x8, a few huge rows (like rare/special tokens).
N, K, M = 32768, 2560, 8
W = torch.randn(N, K) * 0.02 * torch.exp(torch.randn(N, 1) * 0.35)
W[:, torch.randperm(K)[:3]] *= 8
W[torch.randperm(N)[:16]] *= 10
W = W.to(torch.bfloat16).float()
h = torch.randn(M, K); h[:, torch.randperm(K)[:8]] *= 20   # activation outlier channels
h = h.to(torch.bfloat16).float()
ref = h @ W.t()
print(f"{'scheme':38s} {'w rel err':>10s} {'logit rel':>10s} {'top1 agree':>10s} {'max|dlogit|':>11s}")
for name, wq, aq in [
    ("MXFP8 W8A8 (FlashInferCutlassMxfp8)", mx, mx),
    ("MXFP8 weight-only (Marlin W8A16)", mx, lambda x: x),
    ("FP8 PTPC per-row W / per-token A", rowwise, rowwise),
    ("FP8 per-tensor W / per-tensor A", tensorwise, tensorwise),
]:
    Wq = wq(W); out = aq(h) @ Wq.t()
    print(f"{name:38s} {rel(Wq, W):10.4f} {rel(out, ref):10.4f} "
          f"{(out.argmax(-1) == ref.argmax(-1)).float().mean().item():10.3f} {(out-ref).abs().max().item():11.4f}")

# MTP dense shapes (fc_hidden 2560x2560 3-D input, shared-expert down 2560x640)
for (n, k) in [(2560, 2560), (13952, 2560), (2560, 6144), (2560, 640)]:
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16).float()
    x = torch.randn(4, 4, k).to(torch.bfloat16).float()   # [T, hc, K] like fc_hidden
    y = x @ w.t(); yq = mx(x.reshape(-1, k)).reshape(x.shape) @ mx(w).t()
    print(f"MTP dense MXFP8 W8A8 {n}x{k}: weight rel err {rel(mx(w), w):.4f}, output rel err {rel(yq, y):.4f}")

# sanity: e8m0 scale exactness / zero block / dequant identity on representable values
z = torch.zeros(2, 64); assert torch.equal(mx(z), z)
v = torch.tensor([[1.0, -2.0, 0.5, 448.0] * 8]); assert torch.equal(mx(v), v)
print("MXFP8 reference sanity OK")
