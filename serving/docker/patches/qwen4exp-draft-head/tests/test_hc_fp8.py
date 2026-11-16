"""CPU tests for [hc-fp8] (QWEN4EXP_HC_FP8), against real vLLM 0.33 code.

Runs inside the vLLM 0.33 image with the overlay files mounted over the
package (see README "Tests"): the real overlay GatedResidual, the real vLLM
Linear classes, OnlineQuantizationConfig, online FP8 methods and the online
layer-wise loader. Only the GEMM kernel object is a stub (a torch emulation of
the W8A8 block-scaled GEMM), since there is no GPU.

Reads the checkpoints' config.json, index and safetensors headers, plus the
real HC tensors of a few layers (CPU, ~27 MB per layer).

usage: python3 -B test_hc_fp8.py NVIDIA_SNAPSHOT FP4_LOCAL_SNAPSHOT
"""

import ast
import json
import os
import re
import struct
import sys
import types
from collections import Counter

import torch
import torch.nn.functional as F
from safetensors import safe_open

import vllm.model_executor.layers.quantization.online.fp8 as online_fp8
import vllm.model_executor.layers.quantization.online.mxfp8 as online_mxfp8
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.qwen4_exp.nvidia import draft_head as dh
from vllm.models.qwen4_exp.nvidia import hc_fp8
from vllm.models.qwen4_exp.nvidia import hyperconnection as hcmod
from vllm.models.qwen4_exp.nvidia import model as qmodel

SNAPS = {"NVIDIA": sys.argv[1], "FP4-Local": sys.argv[2]}
F8 = torch.float8_e4m3fn
FMAX = 448.0
torch.manual_seed(0)

# The online methods read the model dtype from the current vLLM config at
# construction; outside an engine there is none.
_cfg = types.SimpleNamespace(model_config=types.SimpleNamespace(dtype=torch.bfloat16))
online_fp8.get_current_vllm_config = lambda: _cfg
online_mxfp8.get_current_vllm_config = lambda: _cfg
# Single process, no distributed init: TP rank 0 of 1.
import vllm.model_executor.layers.linear as _linear  # noqa: E402
import vllm.model_executor.parameter as _parameter  # noqa: E402

for _mod in (_linear, _parameter):
    for _fn, _val in (("get_tensor_model_parallel_rank", 0),
                      ("get_tensor_model_parallel_world_size", 1)):
        if hasattr(_mod, _fn):
            setattr(_mod, _fn, lambda _v=_val: _v)
# The fused BF16 down+SiLU kernel is chosen on sm>=90; report a GPU-like platform
# so the test sees the decision the server makes.
hcmod.current_platform = types.SimpleNamespace(has_device_capability=lambda c: True)


def act_q_block(x):
    """Per-token 1x128 dynamic e4m3 (QuantFP8 group (1,128)), dequantized."""
    m, k = x.shape
    kp = -(-k // 128) * 128
    xp = torch.zeros(m, kp)
    xp[:, :k] = x.float()
    xb = xp.view(m, -1, 128)
    s = xb.abs().amax(-1, keepdim=True).clamp(min=1e-10) / FMAX
    return ((xb / s).to(F8).float() * s).view(m, kp)[:, :k]


def w_deq_block(w, s):
    """Dequantize a 128x128 block-scaled e4m3 weight [N, K] with scales [N/128, K/128]."""
    n, k = w.shape
    full = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
    return w.float() * full


def mx(x):
    """MXFP8 quantize-dequantize along the last dim (0.33 torch rule)."""
    xb = x.float().reshape(*x.shape[:-1], -1, 32)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=torch.finfo(torch.float32).tiny)
    s = torch.exp2(torch.ceil(torch.log2(amax / FMAX)))
    return ((xb / s).to(F8).float() * s).reshape(x.shape)


class CutlassFp8BlockScaledMMKernel:  # stub named like the server's kernel
    def process_weights_after_loading(self, layer):
        pass

    def apply_weights(self, layer, x, bias=None):
        w = w_deq_block(layer.weight, layer.weight_scale_inv)
        return (act_q_block(x) @ w.t()).to(torch.bfloat16)


class FlashInferCutlassMxfp8LinearKernel:  # stub named like the server's kernel
    def process_weights_after_loading(self, layer):
        pass


online_fp8.init_fp8_linear_kernel = lambda **kw: CutlassFp8BlockScaledMMKernel()
online_mxfp8.init_mxfp8_linear_kernel = lambda **kw: FlashInferCutlassMxfp8LinearKernel()

LOG: list[str] = []


class _Log:
    def info(self, msg, *args):
        LOG.append(msg % args)
        print("LOG", msg % args)

    warning = info


hc_fp8.logger = _Log()
dh.logger = _Log()


def set_env(**kv):
    for k in ("QWEN4EXP_HC_FP8", "QWEN4EXP_HC_FP8_FORMAT", "QWEN4EXP_FP8_FORMAT", "QWEN4EXP_SIDE_FP8"):
        os.environ.pop(k, None)
    os.environ.update(kv)


def headers(snapshot):
    index = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))
    out = {}
    for f in sorted(set(index["weight_map"].values())):
        with open(os.path.join(snapshot, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        out.update({k: (v["dtype"], tuple(v["shape"])) for k, v in h.items()})
    return out, index["weight_map"]


def text_config(snapshot):
    cfg = json.load(open(os.path.join(snapshot, "config.json")))
    return cfg.get("text_config", cfg)


tc = text_config(SNAPS["NVIDIA"])
HS, HC, RANK, L = tc["hidden_size"], tc["hc_count"], tc["hc_lowrank"], tc["num_hidden_layers"]
assert (HS, HC, RANK, L) == (2560, 4, 320, 48), (HS, HC, RANK, L)
HCFG = hcmod.HyperConnectionConfig(
    hc_count=HC, hidden_size=HS, params_dtype=torch.bfloat16, hc_lowrank=RANK,
    rms_norm_eps=tc["rms_norm_eps"], hc_per_branch_norm=True,
)
D = HC * HS  # 10240
PAD = (-(RANK + HC)) % 16  # 12

# ------------------------------------------------------------ 1. inventory
print("== 1. HC matrices in the checkpoints")
for label, snap in SNAPS.items():
    assert text_config(snap)["hc_lowrank"] == RANK
    hdr, _ = headers(snap)
    kinds = Counter()
    for name, (dtype, shape) in hdr.items():
        if "hyper_connection" not in name or len(shape) != 2:
            continue
        kind = re.sub(r"\.layers\.\d+\.", ".layers.N.", name)
        kinds[(kind, dtype, shape)] += 1
    for (kind, dtype, shape), c in sorted(kinds.items()):
        print(f"  {label:9s} {c:3d} x {kind} {dtype} {list(shape)}")
    t = "model.language_model."
    exp = {}
    for b in ("attn", "mlp"):
        exp[(f"{t}layers.N.{b}_hyper_connection.input_mix_weight_down.weight", "BF16", (RANK, D))] = L
        exp[(f"{t}layers.N.{b}_hyper_connection.block_inject_weight.weight", "BF16", (HC, D))] = L
        exp[(f"{t}layers.N.{b}_hyper_connection.input_mix_weight_up.weight", "BF16", (D, RANK))] = L
    exp[(f"{t}hyper_connection_mixer.input_mix_weight_down.weight", "BF16", (RANK, D))] = 1
    exp[(f"{t}hyper_connection_mixer.input_mix_weight_up.weight", "BF16", (D, RANK))] = 1
    target = {k: v for k, v in kinds.items() if k[0].startswith(t)}
    assert target == exp, (label, target)
    n_target = sum(target.values())
    bf16 = sum(2 * s[0] * s[1] * c for (_, _, s), c in target.items())
    print(f"  {label}: {n_target} target HC matrices (all BF16), {bf16 / 1e9:.3f} GB read per step")
    assert n_target == 6 * L + 2

# ------------------------------------------------------------ helpers
def build(prefix, use_combine=True):
    return hcmod.GatedResidual(HCFG, use_combine=use_combine, prefix=prefix)


def methods(gr):
    names = ["input_mix_weight_up"] + (
        ["input_mix_weight_down_block_inject"] if gr.use_combine else ["input_mix_weight_down"]
    )
    return {n: type(getattr(gr, n).quant_method).__name__ for n in names}


T = "language_model.model."  # Qwen4ExpForConditionalGeneration runtime prefix
TARGET_PREFIXES = (
    [f"{T}layers.{i}.{b}_hyper_connection" for i in range(L) for b in ("attn", "mlp")]
    + [f"{T}hyper_connection_mixer"]
)
MTP_PREFIXES = ["mtp.layers.48.attn_hyper_connection", "mtp.layers.48.mlp_hyper_connection",
                "mtp.hyper_connection_mixer", "model.mtp.layers.48.attn_hyper_connection",
                "model.mtp.hyper_connection_mixer"]

# ------------------------------------------------------------ 2. switch off
print("== 2. switch off: stock modules")
for v in (None, "0", ""):
    set_env(**({} if v is None else {"QWEN4EXP_HC_FP8": v}))
    assert not hc_fp8.hc_fp8_enabled()
    for p in TARGET_PREFIXES + MTP_PREFIXES:
        for leaf in ("input_mix_weight_down_block_inject", "input_mix_weight_down",
                     "input_mix_weight_up"):
            assert hc_fp8.hc_fp8_quant_config(f"{p}.{leaf}") is None
set_env()
for p, comb in [(TARGET_PREFIXES[0], True), (TARGET_PREFIXES[-1], False)]:
    gr = build(p, comb)
    assert set(methods(gr).values()) == {"UnquantizedLinearMethod"}, methods(gr)
    if comb:
        w = gr.input_mix_weight_down_block_inject.weight
        assert w.shape == (RANK + HC + PAD, D) and w.dtype == torch.bfloat16 and not w.is_meta
        assert gr._use_hc_down_silu is True  # stock fused BF16 down+SiLU
        assert gr.input_mix_weight_down_block_inject.quant_config is None
        assert not hasattr(gr.input_mix_weight_down_block_inject, "hc_fp8_pad_rows")
    assert gr.input_mix_weight_up.quant_config is None
LOG.clear()
hc_fp8.log_hc_fp8(torch.nn.Module())
assert LOG == []  # no log line when off
print("  off (unset, 0, empty): every HC Linear stock (quant_config None, fused down+SiLU)")


# ------------------------------------------------------------ 3. switch on: selection, counts, log
def tree(fmt):
    set_env(QWEN4EXP_HC_FP8="1", **({"QWEN4EXP_HC_FP8_FORMAT": fmt} if fmt else {}))
    hc_fp8._hc_quant_config.cache_clear()
    root = torch.nn.Module()
    for p in TARGET_PREFIXES + MTP_PREFIXES[:3]:
        parent = root
        rel = p[len(T):] if p.startswith(T) else p
        parts = rel.split(".")
        for part in parts[:-1]:
            if not hasattr(parent, part):
                parent.add_module(part, torch.nn.Module())
            parent = getattr(parent, part)
        parent.add_module(parts[-1], build(p, use_combine=not p.endswith(f"{T}hyper_connection_mixer")))
    return root


METHOD = {"fp8_per_block": "Fp8PerBlockOnlineLinearMethod", "mxfp8": "Mxfp8OnlineLinearMethod"}
KERNEL = {"fp8_per_block": "CutlassFp8BlockScaledMMKernel", "mxfp8": "FlashInferCutlassMxfp8LinearKernel"}
# QWEN4EXP_HC_FP8_FORMAT="<down>[,<up>]"; default fp8_per_block (down, K % 128) +
# mxfp8 (up, N % 32): the only pair whose GPU kernels accept both HC shapes.
for fmt, f_down, f_up in [(None, "fp8_per_block", "mxfp8"),
                          ("fp8_per_block", "fp8_per_block", "fp8_per_block"),
                          ("mxfp8,fp8_per_block", "mxfp8", "fp8_per_block")]:
    print(f"== 3. switch on, format {fmt or 'fp8_per_block,mxfp8 (default)'}")
    root = tree(fmt)
    conv = Counter()
    for name, m in root.named_modules():
        qm = getattr(m, "quant_method", None)
        if qm is None:
            continue
        if name.startswith("mtp."):
            assert type(qm) is UnquantizedLinearMethod, name
            assert m.quant_config is None
            continue
        method = METHOD[f_down if "input_mix_weight_down" in name else f_up]
        assert type(qm).__name__ == method, (name, type(qm).__name__)
        assert m.weight.is_meta and m.weight.dtype == torch.bfloat16
        conv[hc_fp8._hc_kind(name)] += 1
    print("  converted:", dict(sorted(conv.items())))
    assert conv == {
        "attn_hyper_connection.input_mix_weight_down_block_inject": L,
        "attn_hyper_connection.input_mix_weight_up": L,
        "mlp_hyper_connection.input_mix_weight_down_block_inject": L,
        "mlp_hyper_connection.input_mix_weight_up": L,
        "hyper_connection_mixer.input_mix_weight_down": 1,
        "hyper_connection_mixer.input_mix_weight_up": 1,
    }, conv
    for gr in root.modules():
        if isinstance(gr, hcmod.GatedResidual) and gr.use_combine:
            target = "mtp" not in gr.input_mix_weight_down_block_inject.prefix
            assert gr._use_hc_down_silu is (not target)  # FP8 layers: Linear + hc_silu
            assert getattr(gr.input_mix_weight_down_block_inject, "hc_fp8_pad_rows", 0) == (
                PAD if target else 0)
    # the stock config records the matched prefixes; the drafter's are absent
    qc = hc_fp8._hc_quant_config(f_down, f_up)
    assert len(qc.quantized_layers) == 4 * L + 2
    assert not any("mtp" in k for k in qc.quantized_layers)
    # the plain ForCausalLM naming (model.layers.N) matches too
    assert hc_fp8.hc_fp8_quant_config("model.layers.3.mlp_hyper_connection.input_mix_weight_up") is qc
    # the summary line
    LOG.clear()
    hc_fp8.log_hc_fp8(root)
    line = LOG[-1]
    bf16 = 2 * L * (2 * (RANK + HC) * D + 2 * D * RANK) + 2 * (2 * RANK * D)
    n_down, n_up = RANK + HC + PAD, D
    fp8 = 2 * L * (dh._fp8_bytes(f_down, n_down, D) + dh._fp8_bytes(f_up, n_up, RANK)) + (
        dh._fp8_bytes(f_down, RANK, D) + dh._fp8_bytes(f_up, D, RANK))
    kernel = "/".join(sorted({KERNEL[f_down], KERNEL[f_up]}))
    assert line.startswith(f"[hc-fp8] Hyperconnection GEMMs: {4 * L + 2} matrices to online "
                           f"{f_down} (down) / {f_up} (up) ("), line
    assert f"{bf16 / 1e9:.3f} GB BF16 -> {fp8 / 1e9:.3f} GB per step via {kernel}" in line, line
    print(f"  bytes/step: BF16 {bf16 / 1e9:.3f} GB -> {f_down}/{f_up} {fp8 / 1e9:.3f} GB "
          f"(saves {(bf16 - fp8) / 1e9:.3f} GB = {(bf16 - fp8) / 245e9 * 1e3:.2f} ms at 245 GB/s)")

for bad in ("fp8", "fp8_per_block,mxfp8,mxfp8", "fp8_per_block,"):
    set_env(QWEN4EXP_HC_FP8="1", QWEN4EXP_HC_FP8_FORMAT=bad)
    try:
        hc_fp8.hc_fp8_quant_config("model.layers.3.mlp_hyper_connection.input_mix_weight_up")
    except ValueError:
        continue
    raise AssertionError(f"format {bad!r} accepted")
print("  bad QWEN4EXP_HC_FP8_FORMAT values refused")

# the side-layer summary does not count HC layers (both switches on)
root = tree(None)
side = online_fp8.Fp8PerBlockOnlineLinearMethod.__new__(online_fp8.Fp8PerBlockOnlineLinearMethod)
side.fp8_linear = CutlassFp8BlockScaledMMKernel()
op = torch.nn.Module()
op.quant_method, op.output_size_per_partition, op.input_size_per_partition = side, 2560, 6144
root.layers.get_submodule("0").add_module("linear_attn", torch.nn.Module())
root.layers.get_submodule("0").linear_attn.add_module("out_proj", op)
LOG.clear()
dh.log_side_layers(root)
assert "Target side layers: 1 matrices" in LOG[-1] and "linear_attn.out_proj 1" in LOG[-1], LOG
print("  side-fp8 summary ignores the HC layers")

# ------------------------------------------------------------ 4. real weights through the online loader
print("== 4. real HC weights through the online loader (fp8_per_block) and W8A8 error")


def load_real(snap, wmap, prefix_ckpt, names):
    out = {}
    for n in names:
        key = f"{prefix_ckpt}.{n}"
        with safe_open(os.path.join(snap, wmap[key]), "pt") as fh:
            out[n] = fh.get_tensor(key)
    return out


def rmsnorm_streams(x, w, eps):
    xs = x.float().view(-1, HC, HS)
    xs = xs * torch.rsqrt(xs.square().mean(-1, keepdim=True) + eps)
    return (xs.view(-1, D) * (1.0 + w.float())).to(torch.bfloat16)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


set_env(QWEN4EXP_HC_FP8="1", QWEN4EXP_HC_FP8_FORMAT="fp8_per_block")  # MXFP8 emulated below
hc_fp8._hc_quant_config.cache_clear()
# hc_silu is a Triton kernel; the torch reference of the same math
hcmod.hc_silu = lambda x, hc: (F.silu(x.float() / hc)).to(x.dtype)
M = 8  # decode rows (1 + K drafts, padded)
RESULTS = []
print(f"  {'checkpoint':9s} {'layer':28s} {'W down':>7s} {'W inj':>7s} {'W up':>7s} "
      f"{'silu(low)':>9s} {'inj logit':>9s} {'gate':>7s} {'block_in':>8s} {'mx block_in':>11s}")
for label, snap in SNAPS.items():
    _, wmap = headers(snap)
    for li in (0, L // 2, L - 1, None):
        for b in (("attn",) if li is None else ("attn", "mlp")):
            if li is None:
                ck, rt, comb = "model.language_model.hyper_connection_mixer", f"{T}hyper_connection_mixer", False
            else:
                ck = f"model.language_model.layers.{li}.{b}_hyper_connection"
                rt, comb = f"{T}layers.{li}.{b}_hyper_connection", True
            names = ["hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight"]
            names += ["block_inject_weight.weight"] if comb else []
            t = load_real(snap, wmap, ck, names)
            gr = build(rt, comb)
            down = gr.input_mix_weight_down_block_inject if comb else gr.input_mix_weight_down
            # model.load_weights: stacked shards 0 (down) and 1 (inject); up plain
            if comb:
                down.weight.weight_loader(down.weight, t["input_mix_weight_down.weight"], 0)
                assert not getattr(down, "_already_called_process_weights_after_loading", False)
                down.weight.weight_loader(down.weight, t["block_inject_weight.weight"], 1)
            else:
                down.weight.weight_loader(down.weight, t["input_mix_weight_down.weight"])
            up = gr.input_mix_weight_up
            up.weight.weight_loader(up.weight, t["input_mix_weight_up.weight"])
            # quantized at load (the pad rows count as loaded), not deferred to finalize
            for lin in (down, up):
                assert lin._already_called_process_weights_after_loading
                assert lin.weight.dtype == F8 and not lin.weight.is_meta
            n_down = RANK + (HC + PAD if comb else 0)
            assert down.weight.shape == (n_down, D) and down.weight_scale_inv.shape == (-(-n_down // 128), D // 128)
            assert up.weight.shape == (D, RANK) and up.weight_scale_inv.shape == (D // 128, -(-RANK // 128))
            wd = w_deq_block(down.weight, down.weight_scale_inv)
            wu = w_deq_block(up.weight, up.weight_scale_inv)
            if comb:
                assert torch.equal(wd[RANK + HC:], torch.zeros(PAD, D))  # pad rows exactly 0
            ref_d = t["input_mix_weight_down.weight"].float()
            ref_u = t["input_mix_weight_up.weight"].float()
            e_d, e_u = rel(wd[:RANK], ref_d), rel(wu, ref_u)
            e_i = rel(wd[RANK:RANK + HC], t["block_inject_weight.weight"]) if comb else float("nan")
            # one mix on synthetic multi-stream states (unit RMS per stream, a few
            # outlier channels), with the real hc_norm weight
            x = torch.randn(M, D)
            x[:, torch.randperm(D)[:16]] *= 8
            xn = rmsnorm_streams(x, t["hc_norm.weight"], HCFG.rms_norm_eps)
            # reference: BF16 weights, fp32 accumulation, BF16 outputs (stock math)
            ref_lr = F.linear(xn.float(), ref_d).to(torch.bfloat16)
            ref_low = F.silu(ref_lr.float() / HC).to(torch.bfloat16)
            ref_gate = F.linear(ref_low.float(), ref_u).to(torch.bfloat16)
            ref_inj = (F.linear(xn.float(), t["block_inject_weight.weight"].float()).to(torch.bfloat16)
                       if comb else None)

            def mix(gate):
                g = torch.sigmoid(gate.float()).view(-1, HC, HS)
                return (g * xn.float().view(-1, HC, HS)).mean(1)

            # FP8 path: the converted layers through the real GatedResidual code
            if comb:
                low, inj = gr._down_and_inject(xn)
            else:
                low, inj = hcmod.hc_silu(down(xn), HC), None
            gate = up(low)
            e_low, e_gate = rel(low, ref_low), rel(torch.sigmoid(gate.float()), torch.sigmoid(ref_gate.float()))
            e_inj = rel(inj, ref_inj) if comb else float("nan")
            e_bi = rel(mix(gate), mix(ref_gate))
            # MXFP8 emulation of the same mix (weights per 32 along K, activations per 32)
            mlow = F.silu((mx(xn) @ mx(ref_d).t()).to(torch.bfloat16).float() / HC).to(torch.bfloat16)
            mgate = (mx(mlow) @ mx(ref_u).t()).to(torch.bfloat16)
            e_mx = rel(mix(mgate), mix(ref_gate))
            name = "final mixer" if li is None else f"layer {li} {b}"
            print(f"  {label:9s} {name:28s} {e_d:7.4f} {e_i:7.4f} {e_u:7.4f} {e_low:9.4f} "
                  f"{e_inj:9.4f} {e_gate:7.4f} {e_bi:8.5f} {e_mx:11.5f}")
            assert e_d < 0.05 and e_u < 0.05 and (not comb or e_i < 0.1), (e_d, e_i, e_u)
            RESULTS.append((label, name, e_bi, e_mx))
# W8A8 at 2.6% weight error gives a few % on the block input on these synthetic
# states; a broken scale (e.g. a pad block from uninitialized memory) or a wrong
# shard order would be far above this.
worst = max(RESULTS, key=lambda r: r[2])
print(f"  worst block-input rel err: fp8_per_block {worst[2]:.4f} ({worst[0]} {worst[1]}), "
      f"mxfp8 max {max(r[3] for r in RESULTS):.4f}")
assert worst[2] < 0.06, worst

# ------------------------------------------------------------ 5. wiring
print("== 5. wiring")
src = open(qmodel.__file__).read()
cls = next(n for n in ast.parse(src).body
           if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpForCausalLM")
init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
calls = [c.func.id for c in sorted(
    (c for c in ast.walk(init) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)),
    key=lambda c: (c.lineno, c.col_offset))]
order = [c for c in calls if c in ("Qwen4ExpModel", "log_hc_fp8")]
assert order == ["Qwen4ExpModel", "log_hc_fp8"], order
# switch off needs no new mount: model.py imports hc_fp8 only inside the switch
top = [n for n in ast.parse(src).body if isinstance(n, ast.ImportFrom)]
assert not any(n.module == "hc_fp8" for n in top)
assert 'if os.environ.get("QWEN4EXP_HC_FP8", "0") not in ("", "0"):\n            from .hc_fp8 import log_hc_fp8' in src
hsrc = open(hcmod.__file__).read()
assert "quant_config=None" not in hsrc and hsrc.count("hc_fp8_quant_config(") == 3
print("  model.py logs after Qwen4ExpModel; hyperconnection.py: all 3 HC Linears take hc_fp8_quant_config")
print("ALL HC-FP8 TESTS PASSED")
