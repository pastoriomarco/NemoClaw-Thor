"""CPU tests for the [draft-head] vocabulary logic (stubs stand in for vLLM)."""
import ast, hashlib, os, sys, types
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, "..", "vllm", "models", "qwen4_exp", "nvidia")
ASSET = os.path.join(HERE, "..", "..", "..", "assets", "qwen38-flash-next", "draft_vocab_code_47149.npy")
V = 248320

# --- stub the vLLM imports of draft_head.py
def stub(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m
class _Log:
    def info(self, *a): print("LOG", a[0] % a[1:])
stub("vllm"); stub("vllm.config"); stub("vllm.config.quantization", QuantizationConfigArgs=dict)
stub("vllm.logger", init_logger=lambda n: _Log())
stub("vllm.model_executor"); stub("vllm.model_executor.layers")
stub("vllm.model_executor.layers.quantization", QuantizationConfig=object)
stub("vllm.model_executor.layers.quantization.online")
stub("vllm.model_executor.layers.quantization.online.base", OnlineQuantizationConfig=lambda a: a)
stub("vllm.model_executor.layers.quantization.online.mxfp8", Mxfp8OnlineLinearMethod=object)
stub("vllm.model_executor.layers.quantization.online.fp8", Fp8PerBlockOnlineLinearMethod=object)
stub("vllm.model_executor.layers.vocab_parallel_embedding", ParallelLMHead=object)
sys.path.insert(0, PATCH)
import draft_head as dh

assert hashlib.sha256(open(ASSET, "rb").read()).hexdigest() == "36cd2455dadd4f8362c1137a6e4b54305adb5fd640fc7653cfb3815584177a37"
raw = np.load(ASSET)
os.environ["QWEN4EXP_DRAFT_VOCAB"] = ASSET
ids = dh.load_draft_vocab(V)
assert ids.dtype == torch.int64 and ids.numel() == 47232 and ids.numel() % 128 == 0
assert torch.all(ids[1:] > ids[:-1]) and int(ids.max()) < V
assert set(raw.tolist()) <= set(ids.tolist())
extra = sorted(set(ids.tolist()) - set(raw.tolist()))
print("padded 47149 -> %d, added ids %s..%s" % (ids.numel(), extra[:3], extra[-3:]))
assert dh.pad_draft_vocab(np.arange(256), V).size == 256  # already aligned: unchanged
del os.environ["QWEN4EXP_DRAFT_VOCAB"]; assert dh.load_draft_vocab(V) is None

# MTP dense regex (stock online matcher uses re.match on the layer prefix)
import re
pat = dh.MTP_DENSE_TARGET[3:]
hit = ["mtp.fc_embedding", "mtp.fc_hidden", "mtp.layers.48.self_attn.qkv_proj",
       "mtp.layers.48.self_attn.o_proj", "mtp.layers.48.mlp.shared_expert.gate_up_proj",
       "mtp.layers.48.mlp.shared_expert.down_proj"]
miss = ["mtp.layers.48.self_attn.indexer.index_qk_proj", "mtp.layers.48.mlp.gate",
        "mtp.layers.48.mlp.shared_expert_gate", "mtp.hyper_connection_mixer.input_mix_weight_up",
        "mtp.layers.48.attn_hyper_connection.input_mix_weight_down_block_inject",
        "model.layers.3.self_attn.qkv_proj", "language_model.model.layers.3.mlp.shared_expert.down_proj",
        "mtp.layers.48.mlp.experts", "lm_head", "mtp.draft_head"]
assert all(re.match(pat, p) for p in hit) and not any(re.match(pat, p) for p in miss)

# --- exercise the real compute_logits / load-weights hunks from the patched mtp.py
src = open(os.path.join(PATCH, "mtp.py")).read()
cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpMTP")
fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_logits")
ns = {"torch": torch}; exec(compile(ast.Module([fn], []), "mtp.py", "exec"), ns)
compute_logits = ns["compute_logits"]

class LP:  # LogitsProcessor(vocab) with an unquantized head: hidden @ W^T
    def __init__(self, n): self.org_vocab_size = n
    def __call__(self, head, h): return (h @ head.t())[..., : self.org_vocab_size]

torch.manual_seed(0)
Vs, H, M = 4096, 64, 5   # small vocab for speed; ids = a random subset
W = torch.randn(Vs, H).to(torch.bfloat16)
sub = torch.as_tensor(dh.pad_draft_vocab(np.random.default_rng(0).choice(Vs, 1000, replace=False), Vs))
h = torch.randn(M, H).to(torch.bfloat16)

# load path: draft_head.weight = lm_head.weight.index_select(0, ids)
draft_W = W.index_select(0, sub)
self_full = types.SimpleNamespace(draft_vocab_ids=None, logits_processor=LP(Vs), lm_head=W)
self_red = types.SimpleNamespace(draft_vocab_ids=sub, logits_processor=LP(Vs), lm_head=W,
                                 draft_logits_processor=LP(sub.numel()), draft_head=draft_W)
full = compute_logits(self_full, h)
red = compute_logits(self_red, h)
assert red.shape == full.shape == (M, Vs) and red.dtype == full.dtype
assert torch.equal(red[:, sub], full[:, sub])           # kept columns bit-identical
mask = torch.ones(Vs, dtype=torch.bool); mask[sub] = False
assert torch.isneginf(red[:, mask]).all()               # others -inf
assert torch.equal(red.argmax(-1), full.masked_fill(mask, float("-inf")).argmax(-1))
assert set(red.argmax(-1).tolist()) <= set(sub.tolist())   # drafts are target ids
p = torch.softmax(red.float(), -1); assert torch.allclose(p.sum(-1), torch.ones(M)) and (p[:, mask] == 0).all()
print("compute_logits scatter: OK (kept logits identical, rest -inf, greedy/probabilistic ids in set)")
print("ALL VOCAB TESTS PASSED")
