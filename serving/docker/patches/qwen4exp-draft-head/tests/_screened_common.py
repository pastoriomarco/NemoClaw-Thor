"""Shared stubs and the torch reference for the [screened-head] CPU tests."""
import os, sys, types
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, "..", "vllm", "models", "qwen4_exp", "nvidia")


def stub(name, **attrs):
    m = sys.modules.get(name) or types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, *a):
        self.lines.append(a[0] % a[1:])
        print("LOG", self.lines[-1])

    warning = info


LOG = _Log()
import triton
import triton.language as tl

stub("vllm"); stub("vllm.logger", init_logger=lambda n: LOG)
stub("vllm.triton_utils", tl=tl, triton=triton)
stub("vllm.envs", VLLM_BATCH_INVARIANT=False)
sys.path.insert(0, PATCH)
import screened_head as sh  # noqa: E402


def make_head_weight(V, H, seed=0, dtype=torch.bfloat16):
    """lm_head-like rows: N(0, 0.02) with log-normal row norms, a few outlier
    hidden channels (x8) and 0.1% outlier rows (x4)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(V, H, generator=g) * 0.02
    w *= torch.exp(0.3 * torch.randn(V, 1, generator=g))
    ch = torch.randperm(H, generator=g)[:8]
    w[:, ch] *= 8.0
    rows = torch.randperm(V, generator=g)[: max(V // 1000, 1)]
    w[rows] *= 4.0
    return w.to(dtype)


def screen_ref(head, x, top_k):
    """Torch emulation of the three kernels: same math, other summation order.
    Returns (logits [T, V] bf16 with -inf outside, keep mask)."""
    T, V = x.shape[0], head.vocab_size
    xf = x.float()
    s = (xf @ head.w_fp8.float().T) * head.scale
    bound = xf.norm(dim=1, keepdim=True) * head.err
    lb, ub = s - bound, s + bound
    nb = head.block_max.shape[1]
    pad = torch.full((T, nb * head.BLOCK_V - V), float("-inf"))
    block_max = torch.cat([lb, pad], 1).view(T, nb, head.BLOCK_V).amax(-1)
    t = block_max.topk(top_k, dim=-1).values[:, -1:]
    keep = ub >= t - head.tie * t.abs()
    exact = (xf @ head.weight.float().T).to(x.dtype)
    return torch.where(keep, exact, torch.full_like(exact, float("-inf"))), keep
