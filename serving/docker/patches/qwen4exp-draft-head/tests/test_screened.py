"""CPU tests for the [screened-head] lm_head (stubs stand in for vLLM).

1. Certification on a head shaped like the real one (248320 x 2560 BF16 with
   outlier rows/channels): the real ScreenedLMHead.refresh() builds the FP8 copy
   and bounds; the three kernels are emulated in torch (tests/_screened_common.py,
   same math). Hidden states: flat, peaked, crowded top-20 and near-tie rows.
   "flat" and "near_tie" rows are adversarial (no peak; the bound then keeps
   thousands of rows). Checked per row and k in (1, 20, 32): the full head's top k (BF16 logits and
   the fp64 pre-rounding logits) is never screened out, greedy argmax matches,
   and kept logits are within one BF16 ulp of the fp64 truth.
2. Sampling / rejection semantics: vLLM's top-k/top-p (apply_top_k_top_p_pytorch
   logic) gives bit-identical target probabilities from the screened and the
   full logits, for every draft token, including drafts outside the top k.
3. required_top_k(), the Sampler.add_request / GPUModelRunner.sample hooks and
   the env switches.

Run: python -B tests/test_screened.py [num_rows]  (default 4096 rows, ~1-3 min)
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _screened_common import LOG, make_head_weight, screen_ref, sh, stub  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(max(os.cpu_count() or 1, 1))
V, H = 248320, 2560
NUM_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
BATCH = 64

# ---------------------------------------------------------------- 1. certification
t0 = time.time()
W = make_head_weight(V, H)
head = sh.ScreenedLMHead(W, V, BATCH, 32)
print(f"refresh (FP8 copy + fp64 bounds) on CPU: {time.time() - t0:.1f} s")
assert head.w_fp8.dtype == torch.float8_e4m3fn and head.w_fp8.data_ptr() != W.data_ptr()
assert head.weight.data_ptr() == W.data_ptr(), "the BF16 weight must not be copied"
Wf = W.float()
deq = head.w_fp8.float() * head.scale[:, None]
assert torch.all(head.err >= (W.double() - deq.double()).norm(dim=1).float())


def hidden_states(kind, n, g):
    """Final-norm-like activations: N(0,1) with 16 outlier channels (x5), scaled
    so that logits have std ~2.5, then shaped per kind."""
    x = torch.randn(n, H, generator=g)
    x[:, :16] *= 5.0
    x *= 2.5 / (x @ Wf[:4096].T).std()
    if kind == "flat":
        return x
    if kind == "peaked":  # one row pushed to +10..25 (LLM-like top-1: 4-10 sigma)
        j = torch.randint(0, V, (n,), generator=g)
        d = 10 + 15 * torch.rand(n, 1, generator=g)
        wj = Wf[j]
        return x + d * wj / wj.square().sum(1, keepdim=True)
    if kind == "crowded":  # 30 rows pushed to +12..16 (a crowded top 20)
        for i in range(n):
            j = torch.randint(0, V, (30,), generator=g)
            d = 12 + 4 * torch.rand(30, 1, generator=g)
            x[i] += (d * Wf[j] / Wf[j].square().sum(1, keepdim=True)).sum(0)
        return x
    if kind == "near_tie":  # best two rows pulled to a near tie (PR #60566 test)
        logits = x @ Wf.T
        top2 = logits.topk(2, dim=-1)
        diff = Wf[top2.indices[:, 0]] - Wf[top2.indices[:, 1]]
        gap = top2.values[:, 0] - top2.values[:, 1]
        shrink = 1 - torch.rand(n, 1, generator=g) ** 4  # mostly very close
        return x - shrink * (gap / diff.square().sum(1))[:, None] * diff
    raise ValueError(kind)


g = torch.Generator().manual_seed(1)
kinds = ["flat", "peaked", "crowded", "near_tie"]
stats = {}
ulp_rows = fp8_flips = 0
t0 = time.time()
for b in range(NUM_ROWS // BATCH):
    kind = kinds[b % len(kinds)]
    x = hidden_states(kind, BATCH, g).bfloat16()
    pre = x.float() @ Wf.T  # pre-rounding logits (fp32 accumulation)
    full = pre.bfloat16()  # BF16 head output
    fp8_flips += int((((x.float() @ deq.T)).argmax(1) != full.argmax(1)).sum())
    for k in (1, 20, 32):
        s, keep = screen_ref(head, x, k)
        st = stats.setdefault((kind, k), dict(rows=0, cand=[], miss=0, miss64=0, arg=0, worse=0, ulp=0))
        st["rows"] += BATCH
        st["cand"] += keep.sum(1).tolist()
        st["miss"] += int((~keep.gather(1, full.float().topk(k, 1).indices)).sum())
        st["miss64"] += int((~keep.gather(1, pre.topk(k, 1).indices)).sum())
        # Top-k with ties, as vLLM's sampler applies it: every row >= k-th value.
        kth = full.float().topk(k, 1).values[:, -1:]
        st["miss"] += int((~keep & (full.float() >= kth)).sum())
        if k == 1:
            sa = s.float().argmax(1, keepdim=True)
            st["arg"] += int((sa[:, 0] != full.float().argmax(1)).sum())
            st["worse"] += int((full.float().gather(1, sa) < full.float().max(1, keepdim=True).values).sum())
        # Kept logits vs the fp64 dot product: within one BF16 ulp plus the
        # fp32 accumulation bound gamma * sum|x_i w_i| that any fp32-accumulating
        # GEMM (cuBLAS included) has; it matters only for cancelling, tiny logits.
        if k == 32:
            gamma = H * 2.0**-24 / (1 - H * 2.0**-24)
            for i in range(BATCH):
                idx = keep[i].nonzero()[:, 0]
                t64 = W[idx].double() @ x[i].double()
                _, e = torch.frexp(t64)
                tol = torch.ldexp(torch.ones_like(t64), e - 8)
                tol += gamma * (W[idx].double().abs() @ x[i].double().abs())
                st["ulp"] += int(((s[i, idx].double() - t64).abs() > tol).sum())
print(f"{NUM_ROWS} rows x k in (1,20,32) in {time.time() - t0:.0f} s")
print(f"plain per-row FP8 head (W8A16, what the screen scores with): argmax flips {fp8_flips}/{NUM_ROWS}")
print(f"{'kind':9} {'k':>3} {'rows':>5} {'exact rows/token avg':>21} {'max':>6} {'top-k screened out':>19} {'argmax diff':>12} {'>1ulp+acc':>10}")
for (kind, k), st in sorted(stats.items(), key=lambda kv: (kv[0][1], kinds.index(kv[0][0]))):
    c = np.array(st["cand"])
    print(f"{kind:9} {k:>3} {st['rows']:>5} {c.mean():>21.1f} {c.max():>6} {st['miss'] + st['miss64']:>19} "
          f"{(str(st['arg']) + ' (' + str(st['worse']) + ' lower)') if k == 1 else '-':>12} {st['ulp']:>10}")
    assert st["miss"] == 0 and st["miss64"] == 0, (kind, k, st)
    assert st["worse"] == 0 and st["ulp"] == 0, (kind, k, st)
    assert c.min() >= k

# ---------------------------------------------------- 2. sampling / rejection semantics
def top_k_top_p(logits, k, p):
    """vLLM 0.33 apply_top_k_top_p_pytorch (topk_topp_sampler.py:524-565)."""
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    m = logits_sort.gather(1, (logits_sort.size(1) - k.long()).unsqueeze(1))
    logits_sort.masked_fill_(logits_sort < m, -float("inf"))
    probs_sum = torch.cumsum(logits_sort.softmax(dim=-1), dim=-1)
    mask = probs_sum <= 1 - p.unsqueeze(1)
    mask[:, -1] = False
    logits_sort.masked_fill_(mask, -float("inf"))
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


x = torch.cat([hidden_states(kd, 16, g) for kd in kinds]).bfloat16()
full = (x.float() @ Wf.T).bfloat16()
temp, top_k, top_p = 1.0, 20, 0.95
s, keep = screen_ref(head, x, top_k)
# Isolate the screening: kept rows carry the full head's values.
s_iso = torch.where(keep, full, torch.full_like(full, float("-inf")))
p_full = top_k_top_p(full.float() / temp, torch.full((x.shape[0],), top_k), torch.full((x.shape[0],), top_p)).softmax(-1)
p_scr = top_k_top_p(s_iso.float() / temp, torch.full((x.shape[0],), top_k), torch.full((x.shape[0],), top_p)).softmax(-1)
assert torch.equal(p_full, p_scr), "target probabilities differ"
# Draft tokens: the full head's top 5, ranks 15..25 (around top_k), and random ids.
drafts = torch.cat([full.float().topk(25, 1).indices, torch.randint(0, V, (x.shape[0], 25), generator=g)], 1)
outside_kept = int((~keep.gather(1, drafts)).sum())
outside_topk = int((p_full.gather(1, drafts) == 0).sum())
assert torch.equal(p_full.gather(1, drafts), p_scr.gather(1, drafts))
# Greedy verification compares drafts with the target argmax.
assert torch.equal(s_iso.float().argmax(1), full.float().argmax(1))
print(f"rejection semantics: {x.shape[0]} rows, {drafts.numel()} draft ids, {outside_topk} with target p=0 "
      f"(outside top-k/top-p), {outside_kept} outside the exact set: target p identical for all")

# --------------------------------------------- 3. required_top_k, hooks, env switches
class UVA:
    def __init__(self, a): self.np = np.array(a)

class FakeStates:
    def __init__(self, temp, top_k, min_p):
        self.temperature, self.top_k, self.min_p = UVA(temp), UVA(top_k), UVA(min_p)
        self.vocab_size = V
    def add_request(self, i, sp):
        self.temperature.np[i], self.top_k.np[i], self.min_p.np[i] = sp["t"], sp["k"], sp.get("min_p", 0.0)
        return sp["t"] != 0.0 and (sp["t"] != 1.0 or sp["k"] != V)

class FakeProc:
    def __init__(self, key): self.key = key
    def add_request(self, i, sp): return bool(sp.get(self.key))

class FakeLogprobState:
    def add_request(self, i, sp): pass

class Sampler:  # attributes of vLLM 0.33 Sampler that the hooks read
    def __init__(self):
        self.sampling_states = FakeStates([0.0] * 4, [V] * 4, [0.0] * 4)
        self.thinking_budget_state = FakeProc("thinking")
        self.logits_processors = [FakeProc("bias"), FakeProc("penalty")]
        self.logprob_token_ids_state = FakeLogprobState()
        self.trace_replay_state = None
        self.needs_logits_processing = np.zeros(4, dtype=bool)
        self.compute_nans = False
        self.return_sampling_mask = False
        self.logprobs = np.zeros(4, dtype=bool)
    def add_request(self, i, sp): raise AssertionError("must be replaced by the hook")
    def get_logprobs_dims(self, idx): return (1, 0) if self.logprobs[idx].any() else None

calls = []
class GPUModelRunner:
    def sample(self, hidden_states, input_batch, grammar_output):
        calls.append(find_head().top_k)
        return "out"

stub("vllm.v1"); stub("vllm.v1.worker"); stub("vllm.v1.worker.gpu")
stub("vllm.v1.worker.gpu.model_runner", GPUModelRunner=GPUModelRunner)
stub("vllm.v1.worker.gpu.sample"); stub("vllm.v1.worker.gpu.sample.sampler", Sampler=Sampler)
sh._install_hooks()
sh._install_hooks()  # idempotent
assert GPUModelRunner.sample.__wrapped__ is not None and Sampler.add_request.__name__ == "add_request"

small = sh.ScreenedLMHead(make_head_weight(4096, 256), 4096, 10, 32)
class LM: screened_head = small
class Model:
    def get_language_model(self): return LM()
find_head = lambda: sh.find_screened_head(Model())
assert find_head() is small

sampler = Sampler()
for i, sp in enumerate([dict(t=0.0, k=V), dict(t=1.0, k=20), dict(t=1.0, k=20, penalty=1), dict(t=1.0, k=40)]):
    sampler.add_request(i, sp)
assert sampler.uses_logits_processors.tolist() == [False, False, True, False]
assert sampler.needs_logits_processing.tolist() == [False, True, True, True]
rk = lambda idx, n=5: small.required_top_k(sampler, np.array(idx), n)
assert rk([0]) == 1                       # greedy
assert rk([0, 1]) == 20                   # greedy + Qwen sampling: certify top 20
assert rk([1], 10) == 20 and rk([1], 11) is None  # rows beyond the buffers
assert rk([2]) is None                    # penalties (a logits processor)
assert rk([3]) is None                    # top_k 40 > SCREEN_K 32
sampler.add_request(3, dict(t=0.7, k=V))  # no top_k: top_p/temperature read everything
assert rk([3]) is None
sampler.add_request(3, dict(t=1.0, k=20, min_p=0.1)); assert rk([3]) is None
sampler.add_request(3, dict(t=1.0, k=20)); sampler.logprobs[3] = True; assert rk([3]) is None
sampler.logprobs[3] = False
small.min_top_k = 3; assert rk([0]) == 3 and rk([1]) == 20; small.min_top_k = 1  # fly verifier

class Batch:
    def __init__(self, idx, n):
        self.idx_mapping_np = np.array(idx)
        self.logits_indices = torch.zeros(n, dtype=torch.long)
runner = GPUModelRunner(); runner.model = Model(); runner.sampler = sampler; runner.batch_sharder = None
assert runner.sample(None, Batch([0, 1], 5), None) == "out" and calls[-1] == 20 and small.top_k is None
runner.sample(None, Batch([0, 1], 5), object()); assert calls[-1] is None   # structured output
runner.sample(None, Batch([2], 5), None); assert calls[-1] is None
sampler.compute_nans = True; runner.sample(None, Batch([0], 5), None); assert calls[-1] is None
sampler.compute_nans = False
runner.sample(None, Batch([0], 5), None); assert calls[-1] == 1

class Cfg:  # the vllm_config fields screened_head_settings reads
    def __init__(self, seqs=2, spec=4, lora=None, tp=1):
        self.lora_config = lora
        self.parallel_config = type("P", (), {"tensor_parallel_size": tp})()
        self.scheduler_config = type("S", (), {"max_num_seqs": seqs})()
        self.speculative_config = type("Sp", (), {"num_speculative_tokens": spec})() if spec else None
sh.torch.cuda.is_available = lambda: True
for k in ("QWEN4EXP_LM_HEAD_SCREENED", "QWEN4EXP_LM_HEAD_MXFP8", "QWEN4EXP_LM_HEAD_SCREEN_K"):
    os.environ.pop(k, None)
assert sh.screened_head_settings(Cfg()) is None
os.environ["QWEN4EXP_LM_HEAD_SCREENED"] = "1"
assert sh.screened_head_settings(Cfg()) == (10, 32)
assert sh.screened_head_settings(Cfg(seqs=64, spec=3)) == (64, 32)
assert sh.screened_head_settings(Cfg(spec=0)) == (2, 32)
os.environ["QWEN4EXP_LM_HEAD_SCREEN_K"] = "20"; assert sh.screened_head_settings(Cfg())[1] == 20
os.environ["QWEN4EXP_LM_HEAD_SCREEN_K"] = "0"
try: sh.screened_head_settings(Cfg()); raise AssertionError
except ValueError as e: print("OK raises:", e)
del os.environ["QWEN4EXP_LM_HEAD_SCREEN_K"]
for bad in (dict(lora=object()), dict(tp=2)):
    try: sh.screened_head_settings(Cfg(**bad)); raise AssertionError
    except ValueError as e: print("OK raises:", e)
os.environ["QWEN4EXP_LM_HEAD_MXFP8"] = "1"
try: sh.screened_head_settings(Cfg()); raise AssertionError
except ValueError as e: print("OK raises:", e)
os.environ["QWEN4EXP_LM_HEAD_SCREENED"] = "0"; assert sh.screened_head_settings(Cfg()) is None
# Check mode: counts against the full head on the same hidden states.
chk = sh.ScreenCheck(2)
xs = torch.randn(5, 256).bfloat16()
fs = (xs.float() @ small.weight.float().T).bfloat16()
ss, kk = screen_ref(small, xs, 20)
for _ in range(2):
    chk.update(ss, fs, 20, kk.sum(1))
rows, arg, worse, missing, diff, cand = chk.counts.tolist()
assert (rows, worse, missing, diff) == (10, 0, 0, 0) and cand == 2 * int(kk.sum())
assert LOG.lines[-1].startswith("[screened-head] check after 2 calls: rows=10")
print("ALL OK")
