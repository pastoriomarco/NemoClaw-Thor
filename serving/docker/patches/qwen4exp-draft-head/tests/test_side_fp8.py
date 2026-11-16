"""CPU tests for [side-fp8] (QWEN4EXP_SIDE_FP8), against real vLLM 0.33 code.

Runs inside the vLLM 0.33 image with the overlay files mounted over the
package (see README "Tests"), so it uses the real ModelOpt MIXED_PRECISION
config, the real online-target matcher (resolve_quant_method) and the real
model class mappings. Checkpoint metadata only: config.json, the index and
the safetensors headers (no tensor data is read).

usage: python3 -B test_side_fp8.py NVIDIA_SNAPSHOT [FP4_LOCAL_SNAPSHOT]
"""

import ast
import json
import math
import os
import re
import struct
import sys
import types
from collections import Counter, defaultdict

import torch

import vllm.model_executor.layers.quantization.online.fp8 as online_fp8
import vllm.model_executor.layers.quantization.online.mxfp8 as online_mxfp8
from vllm.model_executor.layers.linear import (
    LinearBase,
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.quantization.base_config import resolve_quant_method
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.models.qwen4_exp.nvidia import draft_head as dh
from vllm.models.qwen4_exp.nvidia import model as qmodel

NV = sys.argv[1]
FP4 = sys.argv[2] if len(sys.argv) > 2 else None
CLS = qmodel.Qwen4ExpForConditionalGeneration

# The online methods read the model dtype from the current vLLM config at
# construction; outside an engine there is none.
_cfg = types.SimpleNamespace(model_config=types.SimpleNamespace(dtype=torch.bfloat16))
online_fp8.get_current_vllm_config = lambda: _cfg
online_mxfp8.get_current_vllm_config = lambda: _cfg

LOG: list[str] = []


class _Log:
    def info(self, msg, *args):
        LOG.append(msg % args)
        print("LOG", msg % args)

    warning = info


dh.logger = _Log()


def set_env(**kv):
    for k in ("QWEN4EXP_SIDE_FP8", "QWEN4EXP_FP8_FORMAT", "QWEN4EXP_SIDE_FP8_SKIP",
              "QWEN4EXP_QSA_UNFUSE_INDEXER"):
        os.environ.pop(k, None)
    for k, v in kv.items():
        os.environ[k] = v


def quant_config(snapshot):
    """The target's checkpoint quant config, as the loader builds it."""
    cfg = json.load(open(os.path.join(snapshot, "config.json")))
    qc = ModelOptMixedPrecisionConfig.from_config(cfg["quantization_config"])
    configure_quant_config(qc, CLS)  # hf_to_vllm names + packed mapping
    return qc


def headers(snapshot):
    """{tensor name: (dtype, shape)} from the safetensors headers."""
    index = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))
    out = {}
    for f in sorted(set(index["weight_map"].values())):
        with open(os.path.join(snapshot, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        out.update({k: (v["dtype"], v["shape"]) for k, v in h.items()})
    return out


# Checkpoint shard name -> runtime fused module (packed_modules_mapping inverse;
# the QSA indexer projection fuses into self_attn.qkv_proj without LoRA).
SHARD_TO_FUSED = {
    "q_proj": "qkv_proj", "k_proj": "qkv_proj", "v_proj": "qkv_proj",
    "indexer.index_qk_proj": "qkv_proj",
    "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj",
    "in_proj_qkv": "in_proj_qkvz", "in_proj_z": "in_proj_qkvz",
    "in_proj_b": "in_proj_ba", "in_proj_a": "in_proj_ba",
    "key_proj": "kv_proj", "value_proj": "kv_proj",
    "input_mix_weight_down": "input_mix_weight_down_block_inject",
    "block_inject_weight": "input_mix_weight_down_block_inject",
}


def runtime_linears(hdr, unfused_indexer=False):
    """Runtime module prefix -> [(checkpoint name, dtype, shape)] for every 2-D
    weight of the language model and the MTP block (visual is not loaded).
    unfused_indexer: QWEN4EXP_QSA_UNFUSE_INDEXER on the target (not the MTP)."""
    mods = defaultdict(list)
    for name, (dtype, shape) in hdr.items():
        if not name.endswith(".weight") or len(shape) != 2 or ".visual." in name:
            continue
        if ".experts." in name or ".ngram_embedding." in name:
            continue  # routed experts: checked separately; PLE row table
        stem = name[: -len(".weight")]
        mapped = CLS.hf_to_vllm_mapper._map_name(stem) if not stem.startswith("mtp.") else stem
        if mapped is None:
            continue
        for shard, fused in SHARD_TO_FUSED.items():
            if unfused_indexer and shard == "indexer.index_qk_proj" and not mapped.startswith("mtp."):
                continue
            if mapped.endswith("." + shard):
                mapped = mapped[: -len(shard)] + fused
                break
        mods[mapped].append((name, dtype, shape))
    return mods


def resolve(qc, prefix):
    cls = ReplicatedLinear if prefix.endswith(".indexer.index_qk_proj") else MergedColumnParallelLinear
    layer = object.__new__(cls)  # isinstance(LinearBase)
    return type(resolve_quant_method(qc, layer, prefix)).__name__


# ---------------------------------------------------------------- NVIDIA
hdr = headers(NV)
mods = runtime_linears(hdr)
target = {k: v for k, v in mods.items() if not k.startswith("mtp.")}
print(f"NVIDIA: {len(target)} runtime 2-D linear/embedding modules outside MTP")

# switch off: stock behaviour, nothing attached
set_env()
qc = quant_config(NV)
assert dh.maybe_quantize_side_layers(qc) is False and qc.online_quantization_config is None
for v in ("0", ""):
    set_env(QWEN4EXP_SIDE_FP8=v)
    assert dh.maybe_quantize_side_layers(qc) is False and qc.online_quantization_config is None
stock = {p: resolve(qc, p) for p in target}

# switch on (default fp8_per_block)
set_env(QWEN4EXP_SIDE_FP8="1")
qc = quant_config(NV)
assert dh.maybe_quantize_side_layers(qc) is True
assert qc.online_quantization_config.args.targets == {dh.side_fp8_target(): "fp8_per_block"}
assert "indexer" not in dh.side_fp8_target()  # fused: the indexer is no kind of its own
side = {p: resolve(qc, p) for p in target}
conv = sorted(p for p, m in side.items() if m == "Fp8PerBlockOnlineLinearMethod")
kinds = Counter(dh._layer_kind(p) for p in conv)
print("converted:", dict(sorted(kinds.items())))
assert kinds == {
    "linear_attn.in_proj_qkvz": 36, "linear_attn.out_proj": 36,
    "self_attn.qkv_proj": 12, "self_attn.o_proj": 12,
    "mlp.shared_expert.gate_up_proj": 48, "mlp.shared_expert.down_proj": 48,
}, kinds
# every converted module is BF16 in the checkpoint and unquantized by ModelOpt
for p in conv:
    assert stock[p] == "UnquantizedLinearMethod", (p, stock[p])
    assert all(dt == "BF16" for _, dt, _ in target[p]), target[p]
    for _, _, (n, k) in target[p]:
        assert n % 128 == 0 and k % 128 == 0, (p, n, k)  # 128x128 blocks
# everything else keeps the stock method
for p, m in side.items():
    if p not in conv:
        assert m == stock[p], (p, m, stock[p])
left = Counter(dh._layer_kind(p) for p in target if p not in conv)
print("left as before:", dict(sorted(left.items())))
for kind in ("linear_attn.in_proj_ba", "mlp.gate", "mlp.shared_expert_gate",
             "attn_hyper_connection.input_mix_weight_down_block_inject",
             "mlp_hyper_connection.input_mix_weight_up", "ple.kv_proj",
             "language_model.model.embed_tokens", "lm_head"):
    assert any(k == kind or k.endswith(kind) for k in left), kind
# routed experts (already NVFP4) never match; the drafter never matches
online = qc.online_quantization_config
from vllm.model_executor.layers.fused_moe import RoutedExperts  # noqa: E402

for i in range(48):
    assert online.resolve_quant_method_cls(
        RoutedExperts, f"language_model.model.layers.{i}.mlp.experts") is None
for p in [k for k in mods if k.startswith("mtp.")] + [
    "mtp.layers.48.self_attn.qkv_proj", "mtp.layers.48.mlp.shared_expert.down_proj",
    "mtp.fc_embedding", "mtp.fc_hidden",
]:
    for n in (p, p.replace("mtp.layers.0.", "mtp.layers.48.")):
        assert online.resolve_quant_method_cls(LinearBase, n) is None, n

# bytes per step (weights read once per target forward)
ckpt = [t for p in conv for t in target[p]]
bf16 = sum(2 * n * k for _, _, (n, k) in ckpt)
fused = [(sum(s[0] for _, _, s in target[p]), target[p][0][2][1]) for p in conv]
fp8 = sum(dh._fp8_bytes("fp8_per_block", n, k) for n, k in fused)
mx = sum(dh._fp8_bytes("mxfp8", n, k) for n, k in fused)
radix = [t for t in ckpt if "index_qk_proj" not in t[0]]
print(f"checkpoint matrices: {len(ckpt)} ({len(radix)} of the RadixArk class + "
      f"{len(ckpt) - len(radix)} indexer index_qk_proj fused into qkv_proj)")
print(f"bytes/step: BF16 {bf16/1e9:.3f} GB -> fp8_per_block {fp8/1e9:.3f} GB "
      f"(saves {(bf16-fp8)/1e9:.3f} GB), mxfp8 {mx/1e9:.3f} GB")
assert len(ckpt) == 312 and len(radix) == 300
assert abs(sum(2 * n * k for _, _, (n, k) in radix) / 2**30 - 5.42) < 0.01  # RadixArk 5.42 GiB

# summary log line on the real module names
LOG.clear()
kernel = type("CutlassFp8BlockScaledMMKernel", (), {})()
root = torch.nn.Module()
for p in conv:
    m = torch.nn.Module()
    m.quant_method = object.__new__(online_fp8.Fp8PerBlockOnlineLinearMethod)
    m.quant_method.fp8_linear = kernel
    m.output_size_per_partition = sum(s[0] for _, _, s in target[p])
    m.input_size_per_partition = target[p][0][2][1]
    parent = root
    for part in p.split(".")[:-1]:
        if not hasattr(parent, part):
            parent.add_module(part, torch.nn.Module())
        parent = getattr(parent, part)
    parent.add_module(p.split(".")[-1], m)
other = torch.nn.Module()
other.quant_method = object()  # e.g. the BF16 router: not counted
root.add_module("router", other)
dh.log_side_layers(root)
line = LOG[-1]
assert line.startswith("[side-fp8] Target side layers: 192 matrices to online fp8_per_block")
assert "linear_attn.in_proj_qkvz 36" in line and "mlp.shared_expert.down_proj 48" in line
assert f"{bf16/1e9:.3f} GB BF16 -> {fp8/1e9:.3f} GB" in line
assert line.endswith("via CutlassFp8BlockScaledMMKernel"), line

# mxfp8 format and plumbing errors
set_env(QWEN4EXP_SIDE_FP8="1", QWEN4EXP_FP8_FORMAT="mxfp8")
qc = quant_config(NV)
assert dh.maybe_quantize_side_layers(qc) is True
assert qc.online_quantization_config.args.targets == {dh.side_fp8_target(): "mxfp8"}
assert resolve(qc, conv[0]) == "Mxfp8OnlineLinearMethod"
try:
    dh.maybe_quantize_side_layers(qc)  # an online config is already set
    raise AssertionError("expected ValueError")
except ValueError as e:
    assert "--quantization-config" in str(e)
set_env(QWEN4EXP_SIDE_FP8="1", QWEN4EXP_FP8_FORMAT="fp4")
try:
    dh.maybe_quantize_side_layers(quant_config(NV))
    raise AssertionError("expected ValueError")
except ValueError as e:
    assert "QWEN4EXP_FP8_FORMAT" in str(e)
set_env(QWEN4EXP_SIDE_FP8="1")
LOG.clear()
assert dh.maybe_quantize_side_layers(None) is False and "no-op" in LOG[-1]

# ---------------------------------------------------------------- skip kinds
def side_run(snapshot, unfuse, skip=None, mods_=None):
    """{prefix: method} for the target with QWEN4EXP_SIDE_FP8=1, as the model
    init applies it (maybe_unfuse_qsa_indexer, then maybe_quantize_side_layers)."""
    env = {"QWEN4EXP_SIDE_FP8": "1"}
    if unfuse:
        env["QWEN4EXP_QSA_UNFUSE_INDEXER"] = "1"
    if skip is not None:
        env["QWEN4EXP_SIDE_FP8_SKIP"] = skip
    set_env(**env)
    qc = quant_config(snapshot)
    assert dh.maybe_unfuse_qsa_indexer(qc) is unfuse
    on = dh.maybe_quantize_side_layers(qc)
    return qc, on, {p: resolve(qc, p) for p in mods_}


def fp8_kinds(methods):
    return Counter(dh._layer_kind(p) for p, m in methods.items()
                   if m == "Fp8PerBlockOnlineLinearMethod")


ALL6 = dict(kinds)
N3 = {k: v for k, v in ALL6.items() if k != "self_attn.qkv_proj"}
IDX = dh.QSA_INDEXER_KIND

# today's N3 (indexer fused): 180 matrices, qkv_proj + indexer BF16
_, on, n3 = side_run(NV, False, "self_attn.qkv_proj", target)
assert on and fp8_kinds(n3) == N3, fp8_kinds(n3)
print("N3 (fused, skip qkv_proj):", sum(N3.values()), "matrices")

# unfused: the indexer is its own runtime module
umods = runtime_linears(hdr, unfused_indexer=True)
utarget = {k: v for k, v in umods.items() if not k.startswith("mtp.")}
idx = sorted(p for p in utarget if p.endswith(".self_attn.indexer.index_qk_proj"))
assert len(idx) == 12, idx
for p in idx:
    (name, dt, shape), = utarget[p]
    assert name.endswith(".self_attn.indexer.index_qk_proj.weight") and dt == "BF16" and shape == [640, 2560]
    q = p.replace(".indexer.index_qk_proj", ".qkv_proj")
    assert sorted(n.rsplit(".", 2)[-2] for n, _, _ in utarget[q]) == ["k_proj", "q_proj", "v_proj"]
    assert sum(sh[0] for _, _, sh in utarget[q]) == 13312  # 13,952 - 640 indexer rows
# the MTP layer keeps the fused layout
assert all(".indexer." not in p for p in umods if p.startswith("mtp."))
assert {p for p in utarget} == {p for p in target} | set(idx)

# unfused, mapping applied, switch on, nothing skipped: 192 + 12 indexer
qc, on, u_all = side_run(NV, True, None, utarget)
assert qc.packed_modules_mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
assert on and fp8_kinds(u_all) == {**ALL6, IDX: 12}, fp8_kinds(u_all)
assert all(u_all[p] == "Fp8PerBlockOnlineLinearMethod" for p in idx)

# C1: unfused, skip only the indexer -> qkv_proj (q/gate,k,v) FP8, indexer BF16
qc, on, c1 = side_run(NV, True, IDX, utarget)
assert on and fp8_kinds(c1) == ALL6, fp8_kinds(c1)
for p in idx:
    assert c1[p] == "UnquantizedLinearMethod", (p, c1[p])
target_re = qc.online_quantization_config.args.targets
print("C1 target:", next(iter(target_re)))
assert IDX not in next(iter(target_re))
# stock method for everything not converted (same as the fused stock run)
for p, m in c1.items():
    if m != "Fp8PerBlockOnlineLinearMethod":
        assert m == stock.get(p, "UnquantizedLinearMethod"), (p, m)

# unfused, skip qkv_proj only: the indexer alone FP8 (works thanks to the
# mapping; with the stock mapping the online matcher sees an unmatched shard)
_, on, u_q = side_run(NV, True, "self_attn.qkv_proj", utarget)
assert on and fp8_kinds(u_q) == {**N3, IDX: 12}, fp8_kinds(u_q)
set_env(QWEN4EXP_SIDE_FP8="1", QWEN4EXP_QSA_UNFUSE_INDEXER="1", QWEN4EXP_SIDE_FP8_SKIP="self_attn.qkv_proj")
qc = quant_config(NV)  # no maybe_unfuse_qsa_indexer: stock qkv_proj shards
assert dh.maybe_quantize_side_layers(qc)
try:
    resolve(qc, idx[0].replace(".indexer.index_qk_proj", ".qkv_proj"))
    raise AssertionError("expected ValueError")
except ValueError as e:
    assert "unmatched shards" in str(e), e

# config errors: indexer skip without unfusing, unknown kinds
for env, msg in (
    (dict(QWEN4EXP_SIDE_FP8_SKIP=IDX), "QWEN4EXP_QSA_UNFUSE_INDEXER=1"),
    (dict(QWEN4EXP_SIDE_FP8_SKIP="self_attn.bogus"), "unknown kinds"),
    (dict(QWEN4EXP_SIDE_FP8_SKIP="self_attn.bogus", QWEN4EXP_QSA_UNFUSE_INDEXER="1"), "unknown kinds"),
):
    set_env(QWEN4EXP_SIDE_FP8="1", **env)
    try:
        dh.maybe_quantize_side_layers(quant_config(NV))
        raise AssertionError(f"expected ValueError for {env}")
    except ValueError as e:
        assert msg in str(e), (env, e)
# switch off: the unfuse hook does nothing
for v in (None, "0", ""):
    set_env(**({} if v is None else {"QWEN4EXP_QSA_UNFUSE_INDEXER": v}))
    qc = quant_config(NV)
    before = qc.packed_modules_mapping
    assert dh.maybe_unfuse_qsa_indexer(qc) is False and qc.packed_modules_mapping is before
    assert "indexer.index_qk_proj" in qc.packed_modules_mapping["qkv_proj"]

# bytes per step: C1 vs N3 (weights read once per target forward)
def step_bytes(methods, mods_):
    total = 0
    for p, m in methods.items():
        n = sum(s[0] for _, _, s in mods_[p])
        k = mods_[p][0][2][1]
        if m == "Fp8PerBlockOnlineLinearMethod":
            total += dh._fp8_bytes("fp8_per_block", n, k)
        elif dh._layer_kind(p) in dh.SIDE_FP8_KINDS + (IDX,):
            total += 2 * n * k
    return total


b_n3, b_c1, b_all = step_bytes(n3, target), step_bytes(c1, utarget), step_bytes(u_all, utarget)
print(f"side-layer bytes/step: N3 {b_n3/1e9:.3f} GB, C1 {b_c1/1e9:.3f} GB "
      f"(-{(b_n3-b_c1)/1e9:.3f} GB = -{(b_n3-b_c1)/245e9*1e3:.2f} ms at 245 GB/s), "
      f"all FP8 unfused {b_all/1e9:.3f} GB")
assert b_c1 < b_n3

# the summary line on the unfused C1 names
LOG.clear()
root = torch.nn.Module()
for p in sorted(p for p, m in c1.items() if m == "Fp8PerBlockOnlineLinearMethod"):
    m = torch.nn.Module()
    m.quant_method = object.__new__(online_fp8.Fp8PerBlockOnlineLinearMethod)
    m.quant_method.fp8_linear = kernel
    m.output_size_per_partition = sum(s[0] for _, _, s in utarget[p])
    m.input_size_per_partition = utarget[p][0][2][1]
    parent = root
    for part in p.split(".")[:-1]:
        if not hasattr(parent, part):
            parent.add_module(part, torch.nn.Module())
        parent = getattr(parent, part)
    parent.add_module(p.split(".")[-1], m)
dh.log_side_layers(root)
print("C1 log:", LOG[-1])
assert LOG[-1].startswith("[side-fp8] Target side layers: 192 matrices to online fp8_per_block")
assert "self_attn.qkv_proj 12" in LOG[-1] and "index_qk_proj" not in LOG[-1]

# ---------------------------------------------------------------- FP4-Local
if FP4:
    set_env(QWEN4EXP_SIDE_FP8="1")
    qc = quant_config(FP4)
    LOG.clear()
    assert dh.maybe_quantize_side_layers(qc) is False
    assert qc.online_quantization_config is None
    assert "no-op" in LOG[-1] and "MXFP8" in LOG[-1], LOG
    # Why the guard: the same targets on this checkpoint would requantize its
    # MXFP8 side layers (dequantize + FP8), not skip them.
    from vllm.config.quantization import QuantizationConfigArgs  # noqa: E402
    from vllm.model_executor.layers.quantization.online.base import (  # noqa: E402
        OnlineQuantizationConfig,
    )

    p = "language_model.model.layers.0.linear_attn.in_proj_qkvz"
    assert resolve(qc, p) != "UnquantizedLinearMethod"
    qc.online_quantization_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={dh.side_fp8_target(): "fp8_per_block"}))
    layer = object.__new__(MergedColumnParallelLinear)
    m = resolve_quant_method(qc, layer, p)
    assert m.requantization_source is not None
    print("FP4-Local: no-op (side layers already MXFP8); forcing the targets would requantize")
    # with the unfuse switch too: side FP8 stays a no-op, the unfused indexer and
    # qkv_proj keep the checkpoint's MXFP8 method
    hdr4 = headers(FP4)
    t4 = {k: v for k, v in runtime_linears(hdr4, unfused_indexer=True).items()
          if not k.startswith("mtp.")}
    qc4, on, m4 = side_run(FP4, True, IDX, t4)
    assert on is False and qc4.online_quantization_config is None
    idx4 = sorted(p for p in t4 if p.endswith(".indexer.index_qk_proj"))
    assert len(idx4) == 12
    for p in idx4:
        q = p.replace(".indexer.index_qk_proj", ".qkv_proj")
        assert m4[p] != "UnquantizedLinearMethod" and m4[p] == m4[q], (p, m4[p], m4[q])
    print("FP4-Local + unfuse: no-op;", m4[idx4[0]], "for indexer and qkv_proj")

# ---------------------------------------------------------------- model.py wiring
src = open(qmodel.__file__).read()
cls = next(n for n in ast.parse(src).body
           if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpForCausalLM")
init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
calls = [c.func.id for c in sorted(
    (c for c in ast.walk(init) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)),
    key=lambda c: (c.lineno, c.col_offset))]
order = [c for c in calls if c in ("maybe_unfuse_qsa_indexer", "maybe_quantize_side_layers",
                                   "Qwen4ExpModel", "log_side_layers")]
assert order == ["maybe_unfuse_qsa_indexer", "maybe_quantize_side_layers", "Qwen4ExpModel",
                 "log_side_layers"], order
# the decoder layer picks the unfused builder for target layers only
assert 'qsa_unfuse_indexer() and "mtp" not in prefix.split(".")' in src

# [qsa-unfuse] builder: the stock __init__'s fusion flag is forced off and the
# module ends up as the stock class. The image's qsa.py must still decide the
# fusion with that one assignment and have the unfused paths.
import inspect  # noqa: E402

from vllm.models.qwen4_exp.nvidia import qsa as qqsa  # noqa: E402

qsrc = inspect.getsource(qqsa.Qwen4ExpQSAAttention)
assert qsrc.count("self.fuse_indexer_projection = vllm_config.lora_config is None") == 1
assert qsrc.count("self.fuse_indexer_projection") == 4, qsrc.count("self.fuse_indexer_projection")
assert "projected_qk, _ = self.indexer.index_qk_proj(hidden_states)" in qsrc
assert "del self.indexer.index_qk_proj" in qsrc
stock_init = qqsa.Qwen4ExpQSAAttention.__init__


def fake_init(self, **kw):
    torch.nn.Module.__init__(self)
    self.fuse_indexer_projection = kw["vllm_config"].lora_config is None
    self.other = True


qqsa.Qwen4ExpQSAAttention.__init__ = fake_init
try:
    a = qmodel._Qwen4ExpQSAUnfusedIndexerInit.build(
        vllm_config=types.SimpleNamespace(lora_config=None))
    assert type(a) is qqsa.Qwen4ExpQSAAttention and a.fuse_indexer_projection is False and a.other
    b = qqsa.Qwen4ExpQSAAttention(vllm_config=types.SimpleNamespace(lora_config=None))
    assert b.fuse_indexer_projection is True  # the stock class is untouched
finally:
    qqsa.Qwen4ExpQSAAttention.__init__ = stock_init
assert qmodel.Qwen4ExpQSAAttention is qqsa.Qwen4ExpQSAAttention
print("ALL SIDE-FP8 TESTS PASSED")
