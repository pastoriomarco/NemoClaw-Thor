"""SM110 NVFP4 MoE spike (Stage 0b of serving/docs/SM110-NVFP4-MOE-PLAN.md).

One process, no server. Compares, on identical Qwen3.8-Flash-Next-shaped tensors (H 2560, I 640,
512 experts, top-10), the routed-expert MoE region of
  (i)  FlashInfer CUTLASS MoE, called exactly as vLLM 0.33 FlashInferExperts calls it in R7b, and
  (ii) FlashInfer 0.7.2.post1 CuTe DSL NVFP4 MoE (vLLM backend FLASHINFER_CUTEDSL), called exactly as
       vLLM 0.33 FlashInferCuteDSLExperts calls it, with the sm_110 gates lifted in this process
       (sm110_lift.py), tactics n128 / n256, fused (atomic) and deterministic (two-stage) finalize.
The measured region of both is "input NVFP4 quant + experts", i.e. what changes between the two
backends; router/top-k is identical and excluded.

Weights: one real layer (FP4-Local after the same CSF decode vLLM applies, or the NVIDIA NVFP4
checkpoint), prepared by vLLM's own prep functions for each backend; or random NVFP4 (--weights random).

Modes:
  default       GPU run: phases env, weights, jit, numerics, timing, determinism, profile, jitwarm
  --dry-run     CPU only: imports, lifts, weight loading (a few experts), vLLM weight prep for both
                layouts, torch reference, kernel-call argument binding, and a cross-compile of the
                CuTe DSL FC1/FC2 kernels for sm_110a (CUTE_DSL_ARCH=sm_110a). No kernel launch.
Outputs (in --out): results.json, summary.md, profile_*.json (kernel totals only), log.txt.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import inspect
import json
import math
import os
import platform
import re
import statistics as st
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

H, I, E, TOPK = 2560, 640, 512, 10
DECODE_T = [1, 4, 5, 8, 16, 64, 512]  # graph replay; 4 = one MTP-3 verify step at batch 1
PREFILL_T = [2048, 7488, 8192]  # eager (7488 = R7b chunk, 8192 = max batched tokens)
NUMERICS_T = [1, 4, 64, 7488]
EDGE_CASES = [(127, "same10"), (128, "same10"), (129, "same10"), (4, "zero_weight_row")]
SKEW_ZIPF_S = 1.1


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------------------------


def load_layer_weights(ckpt: str, layer: int, n_experts: int):
    """Return checkpoint-format tensors (CPU) for one layer's routed experts, as vLLM's modelopt
    NVFP4 MoE method holds them before process_weights_after_loading:
      w13 [E, 2I, H/2] u8 ([w1=gate; w3=up]), w13_scale [E, 2I, H/16] e4m3, w13_scale_2 [E, 2],
      a13 [E, 2], w2 [E, H, I/2] u8, w2_scale [E, H, I/16] e4m3, w2_scale_2 [E], a2 [E].
    FP4-Local CSF scale streams are decoded with vLLM PR #64312's decode_csf_scale (overlay)."""
    import torch
    from safetensors import safe_open

    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    pre = f"model.language_model.layers.{layer}.mlp.experts."
    csf = None
    want = []
    for e in range(n_experts):
        for p in ("gate_proj", "up_proj", "down_proj"):
            for k in ("weight", "weight_scale_2", "input_scale"):
                want.append(f"{pre}{e}.{p}.{k}")
            n = f"{pre}{e}.{p}.weight_scale"
            if n in idx:
                want.append(n)
            else:
                streams = [s for s in idx if s.startswith(n + ".")]
                if not streams:
                    raise KeyError(n)
                want += streams
                csf = True
    by_shard = {}
    for n in want:
        by_shard.setdefault(idx[n], []).append(n)
    t = {}
    t0 = time.time()
    for sh, names in by_shard.items():
        with safe_open(os.path.join(ckpt, sh), "pt") as f:
            for n in names:
                t[n] = f.get_tensor(n)
    log(f"  read {len(t)} tensors from {len(by_shard)} shards in {time.time()-t0:.1f}s")
    ncsf = 0
    if csf:
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (  # PR #64312 overlay
            CSF_SCALE_ENCODINGS,
            decode_csf_scale,
        )

        for e in range(n_experts):
            for p in ("gate_proj", "up_proj", "down_proj"):
                base = f"{pre}{e}.{p}.weight_scale"
                if base in t:
                    continue
                for enc, (names, _) in CSF_SCALE_ENCODINGS.items():
                    if all(f"{base}.{s}" in t for s in names):
                        t[base] = decode_csf_scale(enc, {s: t.pop(f"{base}.{s}") for s in names})
                        ncsf += 1
                        break
                else:
                    raise KeyError(f"no CSF stream set for {base}")
        log(f"  decoded {ncsf} CSF scale tensors")

    def g(e, p, k):
        return t[f"{pre}{e}.{p}.{k}"]

    w13 = torch.stack([torch.cat([g(e, "gate_proj", "weight"), g(e, "up_proj", "weight")]) for e in range(n_experts)])
    w13s = torch.stack(
        [torch.cat([g(e, "gate_proj", "weight_scale"), g(e, "up_proj", "weight_scale")]) for e in range(n_experts)]
    ).view(torch.float8_e4m3fn)
    w13s2 = torch.stack(
        [torch.stack([g(e, "gate_proj", "weight_scale_2"), g(e, "up_proj", "weight_scale_2")]) for e in range(n_experts)]
    ).float()
    a13 = torch.stack(
        [torch.stack([g(e, "gate_proj", "input_scale"), g(e, "up_proj", "input_scale")]) for e in range(n_experts)]
    ).float()
    w2 = torch.stack([g(e, "down_proj", "weight") for e in range(n_experts)])
    w2s = torch.stack([g(e, "down_proj", "weight_scale") for e in range(n_experts)]).view(torch.float8_e4m3fn)
    w2s2 = torch.stack([g(e, "down_proj", "weight_scale_2") for e in range(n_experts)]).float().reshape(-1)
    a2 = torch.stack([g(e, "down_proj", "input_scale") for e in range(n_experts)]).float().reshape(-1)
    assert w13.shape == (n_experts, 2 * I, H // 2) and w13.dtype == torch.uint8, w13.shape
    assert w13s.shape == (n_experts, 2 * I, H // 16), w13s.shape
    assert w2.shape == (n_experts, H, I // 2) and w2s.shape == (n_experts, H, I // 16), (w2.shape, w2s.shape)
    meta = dict(
        source=ckpt,
        layer=layer,
        experts=n_experts,
        csf_decoded=ncsf,
        w13_scale_2_gate_eq_up=bool(torch.equal(w13s2[:, 0], w13s2[:, 1])),
        a13_distinct=int(a13.unique().numel()),
        a2_distinct=int(a2.unique().numel()),
        a2_min_max=[float(a2.min()), float(a2.max())],
    )
    return dict(w13=w13, w13_scale=w13s, w13_scale_2=w13s2, a13=a13, w2=w2, w2_scale=w2s, w2_scale_2=w2s2, a2=a2), meta


def random_layer_weights(n_experts: int, seed: int = 0):
    """Random NVFP4 codes with realistic scales (magnitudes taken from FP4-Local layer 23)."""
    import torch

    g = torch.Generator().manual_seed(seed)
    w13 = torch.randint(0, 256, (n_experts, 2 * I, H // 2), dtype=torch.uint8, generator=g)
    w2 = torch.randint(0, 256, (n_experts, H, I // 2), dtype=torch.uint8, generator=g)
    # E4M3 block scales around 100..448 (typical for amax-calibrated blocks)
    w13s = (torch.rand(n_experts, 2 * I, H // 16, generator=g) * 348 + 100).to(torch.float8_e4m3fn)
    w2s = (torch.rand(n_experts, H, I // 16, generator=g) * 348 + 100).to(torch.float8_e4m3fn)
    s2 = torch.rand(n_experts, generator=g) * 2.2e-4 + 2.5e-5
    w13s2 = torch.stack([s2, s2], 1)
    a13 = torch.full((n_experts, 2), 0.006045)
    a2 = torch.rand(n_experts, generator=g) * 0.0157 + 0.002
    w2s2 = torch.rand(n_experts, generator=g) * 1.1e-4 + 2.3e-5
    meta = dict(source="random", experts=n_experts, seed=seed)
    return dict(w13=w13, w13_scale=w13s, w13_scale_2=w13s2, a13=a13, w2=w2, w2_scale=w2s, w2_scale_2=w2s2, a2=a2), meta


def _layer_ns(n_experts):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    return SimpleNamespace(
        activation=MoEActivation.SILU,
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(enable_eplb=False),
            intermediate_size_per_partition=I,
            hidden_dim=H,
            hidden_dim_unpadded=None,
            num_experts=n_experts,
        ),
    )


def prep_cutlass(ck, dev):
    """vLLM 0.33 FLASHINFER_CUTLASS: modelopt.process_weights_after_loading ->
    convert_to_nvfp4_moe_kernel_format -> prepare_nvfp4_moe_layer_for_fi_or_cutlass ->
    make_nvfp4_moe_quant_config -> FlashInferExperts.process_weights_after_loading (alpha folding)."""
    import torch
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        prepare_nvfp4_moe_layer_for_fi_or_cutlass,
    )

    n = ck["w13"].shape[0]
    w13, w13s, w13s2, a13, w2, w2s, w2s2, a2 = prepare_nvfp4_moe_layer_for_fi_or_cutlass(
        backend=NvFp4MoeBackend.FLASHINFER_CUTLASS,
        layer=_layer_ns(n),
        w13=ck["w13"].to(dev).clone(),
        w13_scale=ck["w13_scale"].to(dev).clone(),
        w13_scale_2=ck["w13_scale_2"][:, 0].contiguous().to(dev),
        a13_scale=ck["a13"].to(dev),
        w2=ck["w2"].to(dev).clone(),
        w2_scale=ck["w2_scale"].to(dev).clone(),
        w2_scale_2=ck["w2_scale_2"].to(dev).clone(),
        a2_scale=ck["a2"].to(dev),
        is_act_and_mul=True,
    )
    g1 = (w13s2 * a13).float()  # FlashInferExperts.process_weights_after_loading
    g2 = (w2s2 * a2).float()
    return SimpleNamespace(
        w1=w13, w2=w2, w1_scale=w13s, w2_scale=w2s, g1_alphas=g1, g2_alphas=g2,
        a1_gscale=(1.0 / a13).float(), a2_gscale=(1.0 / a2).float(),
    )


def prep_cutedsl(ck, dev):
    """vLLM 0.33 FLASHINFER_CUTEDSL: prepare_nvfp4_moe_layer_for_flashinfer_cutedsl ->
    nvfp4_swizzled_scale_to_cutedsl_mma_view -> FlashInferCuteDSLExperts.process_weights_after_loading."""
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        nvfp4_swizzled_scale_to_cutedsl_mma_view,
        prepare_nvfp4_moe_layer_for_flashinfer_cutedsl,
    )

    n = ck["w13"].shape[0]
    layer = _layer_ns(n)
    w13, w13s, w13s2, a13, w2, w2s, w2s2, a2 = prepare_nvfp4_moe_layer_for_flashinfer_cutedsl(
        layer=layer,
        w13=ck["w13"].to(dev).clone(),
        w13_scale=ck["w13_scale"].to(dev).clone(),
        w13_scale_2=ck["w13_scale_2"][:, 0].contiguous().to(dev),
        a13_scale=ck["a13"].to(dev),
        w2=ck["w2"].to(dev).clone(),
        w2_scale=ck["w2_scale"].to(dev).clone(),
        w2_scale_2=ck["w2_scale_2"].to(dev).clone(),
        a2_scale=ck["a2"].to(dev),
    )
    assert layer.moe_config.hidden_dim == H and layer.moe_config.intermediate_size_per_partition == I, (
        "unexpected padding", layer.moe_config)
    w13s = nvfp4_swizzled_scale_to_cutedsl_mma_view(w13s)
    w2s = nvfp4_swizzled_scale_to_cutedsl_mma_view(w2s)
    g1 = (w13s2 * a13).float()
    g2 = (w2s2 * a2).float()
    return SimpleNamespace(
        w1=w13, w2=w2, w1_scale=w13s, w2_scale=w2s, g1_alphas=g1, g2_alphas=g2,
        a1_gscale=(1.0 / a13).float(), a2_gscale=(1.0 / a2).float(),
    )


# ----------------------------------------------------------------------------------------------
# Torch reference ("what the checkpoint means", activation-scale variant A as in R7b)
# ----------------------------------------------------------------------------------------------


def _dequant_rows(codes_u8, sf_e4m3, scale_2):
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import break_fp4_bytes

    import torch

    n, kh = codes_u8.shape
    v = break_fp4_bytes(codes_u8, torch.float32)
    v = v.reshape(n, kh * 2 // 16, 16) * sf_e4m3.float().unsqueeze(-1)
    return v.reshape(n, kh * 2) * float(scale_2)


def reference_moe(ck, x, ids, wts, quant_act=True):
    """fp32 reference: x -> NVFP4 q/dq with the shared FC1 global scale (max input_scale over
    experts) -> per expert dequantized W13 ([gate; up]) -> silu(gate)*up -> NVFP4 q/dq with the
    shared FC2 global scale -> W2 -> * router weight -> sum over top-k. quant_act=False gives the
    unquantized-activation (W4A16) reference."""
    import torch
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import ref_nvfp4_quant

    dev = x.device
    a13 = ck["a13"].max().float().to(dev)
    a2 = ck["a2"].max().float().to(dev)

    def qdq(t, a):
        if not quant_act:
            return t.float()
        gs = (1.0 / a).reshape(1).float()
        q, s = ref_nvfp4_quant(t.float(), gs, 16)
        return (q.reshape(t.shape[0], -1, 16) * (s.unsqueeze(-1) / gs)).reshape(t.shape)

    xq = qdq(x, a13)
    out = torch.zeros(x.shape[0], H, dtype=torch.float32, device=dev)
    for e in ids.unique().tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        w13 = _dequant_rows(ck["w13"][e].to(dev), ck["w13_scale"][e].to(dev), ck["w13_scale_2"][e, 0])
        h = xq[tok] @ w13.t()
        gate, up = h[:, :I], h[:, I:]
        act = torch.nn.functional.silu(gate) * up
        w2 = _dequant_rows(ck["w2"][e].to(dev), ck["w2_scale"][e].to(dev), ck["w2_scale_2"][e])
        y = qdq(act, a2) @ w2.t()
        out.index_add_(0, tok, y * wts[tok, slot].float().unsqueeze(-1))
    return out


def compare(y, ref):
    import torch

    y = y.float()
    ref = ref.float()
    rel = (torch.linalg.vector_norm(y - ref) / torch.linalg.vector_norm(ref).clamp_min(1e-30)).item()
    cos = torch.nn.functional.cosine_similarity(y, ref, dim=-1)
    nz = ref.abs().sum(-1) > 0
    return dict(
        rel_l2=rel,
        cos_min=float(cos[nz].min()) if nz.any() else None,
        cos_median=float(cos[nz].median()) if nz.any() else None,
        max_abs=float((y - ref).abs().max()),
        max_abs_out=float(y.abs().max()),
        nonfinite=int((~torch.isfinite(y)).sum()),
    )


# ----------------------------------------------------------------------------------------------
# Routing and inputs
# ----------------------------------------------------------------------------------------------


def make_routing(T, mode, n_experts, gen, dev):
    import torch

    if mode in ("uniform", "zero_weight_row"):
        logits = torch.rand(T, n_experts, generator=gen)
    elif mode == "skewed":  # Zipf popularity over a random expert permutation
        perm = torch.randperm(n_experts, generator=gen)
        pop = 1.0 / torch.arange(1, n_experts + 1).float() ** SKEW_ZIPF_S
        p = torch.empty(n_experts)
        p[perm] = pop / pop.sum()
        ids = torch.stack([torch.multinomial(p, TOPK, replacement=False, generator=gen) for _ in range(T)])
        logits = None
    elif mode == "same10":  # all tokens to the same 10 experts: group size T per expert
        ids = torch.arange(TOPK).repeat(T, 1)
        logits = None
    else:
        raise ValueError(mode)
    if logits is not None:
        ids = logits.topk(TOPK, dim=-1).indices
    w = torch.softmax(torch.randn(T, TOPK, generator=gen), -1)  # softmax top-k, renormalised
    if mode == "zero_weight_row":
        w[0] = 0
    return ids.to(torch.int32).to(dev), w.float().to(dev)


def make_x(T, ck, gen, dev):
    import torch

    amax = float(ck["a13"].max()) * 448.0 * 6.0  # calibrated FC1 input amax
    x = torch.randn(T, H, generator=gen) * (amax / 6.0)
    return x.to(torch.bfloat16).to(dev)


# ----------------------------------------------------------------------------------------------
# The two candidates (call exactly as the vLLM 0.33 experts classes do)
# ----------------------------------------------------------------------------------------------


class Cutlass:
    name = "fi_cutlass"

    def __init__(self, p):
        self.p = p
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import activation_to_flashinfer_type

        self.act = activation_to_flashinfer_type(MoEActivation.SILU)

    def kwargs(self, xq, xsf, ids, wts, out):
        p = self.p
        import torch

        return dict(
            input=xq, token_selected_experts=ids.to(torch.int), token_final_scales=wts,
            fc1_expert_weights=p.w1.view(torch.long), fc2_expert_weights=p.w2.view(torch.long),
            fc1_expert_biases=None, fc2_expert_biases=None, swiglu_alpha=None, swiglu_beta=None,
            swiglu_limit=None, output=out, output_dtype=torch.bfloat16,
            quant_scales=[p.a1_gscale, p.w1_scale.view(torch.int32), p.g1_alphas, p.a2_gscale,
                          p.w2_scale.view(torch.int32), p.g2_alphas],
            input_sf=xsf, tp_size=1, tp_rank=0, ep_size=1, ep_rank=0, activation_type=self.act,
            use_deepseek_fp8_block_scale=False, use_mxfp8_act_scaling=False, use_w4_group_scaling=False,
        )

    def __call__(self, x, ids, wts, out):
        from vllm import _custom_ops as ops
        from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe

        xq, xsf = ops.scaled_fp4_quant(x, self.p.a1_gscale, is_sf_swizzled_layout=True)
        flashinfer_cutlass_fused_moe(**self.kwargs(xq, xsf, ids, wts, out))
        return out


class CuteDsl:
    def __init__(self, p, n_tile, fused, tactic_explicit=True):
        self.p, self.n_tile, self.fused, self.tactic_explicit = p, n_tile, fused, tactic_explicit
        self.name = f"fi_cutedsl_n{n_tile}_{'fused' if fused else 'det'}" + ("" if tactic_explicit else "_vllmdefault")
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import activation_to_flashinfer_int

        self.act = activation_to_flashinfer_int(MoEActivation.SILU)
        self.tactic = (128, ((128, n_tile), (1, 1), False), ((128, n_tile), (1, 1), False))

    def kwargs(self, xq, xsf, ids, wts, out):
        import torch

        p = self.p
        kw = dict(
            x=xq, x_sf=xsf.unsqueeze(-1), token_selected_experts=ids.to(torch.int32),
            token_final_scales=wts.float(), w1_weight=p.w1, w1_weight_sf=p.w1_scale, w1_alpha=p.g1_alphas,
            fc2_input_scale=p.a2_gscale, w2_weight=p.w2, w2_weight_sf=p.w2_scale, w2_alpha=p.g2_alphas,
            num_experts=p.w1.shape[0], top_k=TOPK, num_local_experts=p.w1.shape[0], local_expert_offset=0,
            moe_output=out, activation_type=self.act, use_fused_finalize=self.fused, quant_mode="w4a4",
        )
        if self.tactic_explicit:
            kw["tactic"] = self.tactic
        return kw

    def __call__(self, x, ids, wts, out):
        from vllm import _custom_ops as ops
        from flashinfer.fused_moe import cute_dsl_fused_moe

        xq, xsf = ops.scaled_fp4_quant(x, self.p.a1_gscale, is_sf_swizzled_layout=False)
        cute_dsl_fused_moe(**self.kwargs(xq, xsf, ids, wts, out))
        return out


def edge_llm_fallback(*_a, **_k):
    """Candidate (iii) of the plan: Edge-LLM 0.12 K1/K2 JIT-compiled through kernel.wrapper.
    Documented stub only: run only if every (ii) variant fails G0 for FlashInfer-specific reasons.
    What it would take (plan 6.2): vendor E/kernelSrcs/nvfp4_moe_cutedsl/{K1,K2,custom_pipeline,
    cute_utils,moe_compat}.py; compile with cute.compile(kernel.wrapper, ...) as in
    export_fc1_kernel.py:84-164 minus export_to_c; FC1 A-row map = permuted_to_expanded // top_k;
    reuse FlashInfer moe_sort for the layout; alphas = folded g1/g2, input_global_scale = ones,
    down_input_scale = shared w2 input scale."""
    raise NotImplementedError("Edge-LLM fallback (iii) is a documented stub; see plan section 6.2")


# ----------------------------------------------------------------------------------------------
# Timing helpers
# ----------------------------------------------------------------------------------------------


def time_graph(fn_calls, replays, warm=10):
    """Capture all calls in one CUDA graph; return per-replay ms list."""
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fn_calls:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fn_calls:
            f()
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(replays)]
    for a, b in ev:
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    return g, [a.elapsed_time(b) for a, b in ev]


def time_eager(f, iters, warm=3):
    import torch

    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for a, b in ev:
        a.record()
        f()
        b.record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a, b in ev]


def pstats(v, div=1):
    v = sorted(x / div for x in v)
    return dict(median=st.median(v), p10=v[len(v) // 10], p90=v[(9 * len(v)) // 10], n=len(v))


KCAT = [
    ("quant_in", r"cvt_fp16_to_fp4|fp4_quant"),
    ("cutlass_prefix_sums", r"ExpertPrefixSum"),
    ("cutlass_expand_rows", r"expandInputRows"),
    ("cutlass_strides", r"computeStridesTma"),
    ("cutlass_activation", r"doActivationKernel"),
    ("cutlass_finalize", r"finalizeMoeRouting"),
    ("cutlass_grouped_gemm", r"GroupProblemShape"),
    ("dsl_fc1_gather_swiglu", r"GatherGroupedGemm|gather_grouped_gemm"),
    ("dsl_fc2_finalize", r"FinalizeFusion|finalize_fusion"),
    ("dsl_unpermute", r"[Uu]npermute"),
    ("dsl_memset", r"[Mm]emset"),
    ("dsl_sort_routing", r"routing|Routing|moeSort|moe_sort|sort"),
]


def kcat(name):
    for c, rx in KCAT:
        if re.search(rx, name):
            return c
    return "other"


def profile_calls(run_replays, label, outdir, n_calls):
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        run_replays()
        torch.cuda.synchronize()
    tot, names = {}, {}
    for e in prof.events():
        if e.device_type is not None and str(e.device_type).endswith("CUDA") and getattr(e, "device_time", 0):
            c = kcat(e.name)
            tot[c] = tot.get(c, 0.0) + e.device_time / 1e3
            names[e.name[:160]] = names.get(e.name[:160], 0.0) + e.device_time / 1e3
    per = {k: v / n_calls for k, v in sorted(tot.items(), key=lambda x: -x[1])}
    top = dict(sorted(((k, v / n_calls) for k, v in names.items()), key=lambda x: -x[1])[:25])
    json.dump(dict(label=label, ms_per_call_by_category=per, top_kernels_ms_per_call=top),
              open(os.path.join(outdir, f"profile_{label}.json"), "w"), indent=1)
    return per


# ----------------------------------------------------------------------------------------------
# Phases
# ----------------------------------------------------------------------------------------------


def phase_env(R):
    import torch

    env = dict(python=platform.python_version(), torch=torch.__version__, cuda_runtime=torch.version.cuda)
    for mod in ("vllm", "flashinfer", "cutlass"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            env[mod] = f"import failed: {e}"
    try:
        from importlib.metadata import version

        env["nvidia-cutlass-dsl"] = version("nvidia-cutlass-dsl")
        env["flashinfer-python"] = version("flashinfer-python")
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(device=p.name, cc=f"{p.major}.{p.minor}", sms=p.multi_processor_count,
                   mem_gb=round(p.total_memory / 2**30, 1),
                   smem_per_block_optin=getattr(p, "shared_memory_per_block_optin", None),
                   smem_per_sm=getattr(p, "shared_memory_per_multiprocessor", None),
                   l2_bytes=getattr(p, "L2_cache_size", None))
        try:  # the CuTe DSL MoE kernels size their pipelines from the sm_100 table (232448 B)
            from cutlass.memory import get_smem_capacity_in_bytes

            env["dsl_smem_table_sm_100"] = get_smem_capacity_in_bytes("sm_100")
        except Exception as e:  # noqa: BLE001
            env["dsl_smem_table_sm_100"] = f"?: {e}"
    env["CUTE_DSL_ARCH"] = os.environ.get("CUTE_DSL_ARCH")
    env["CUTE_DSL_CACHE_DIR"] = os.environ.get("CUTE_DSL_CACHE_DIR")
    env["FLASHINFER_WORKSPACE_BASE"] = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    from sm110_lift import lift_cutedsl_moe_gates

    env["lifted"] = lift_cutedsl_moe_gates()
    from cutlass.base_dsl.enums import Arch

    env["dsl_has_sm_110a"] = hasattr(Arch, "sm_110a")
    R["env"] = env
    log("env:", json.dumps(env))


def phase_jit(R, cands):
    """Cold JIT: moe_utils load/build, then the first call of every CuTe DSL variant (compile)."""
    import torch

    j = {}
    t = time.time()
    import flashinfer.fused_moe.cute_dsl.moe_utils as mu

    spec = mu.gen_moe_utils_module()
    j["moe_utils_prebuilt"] = spec.jit_library_path.exists()
    mu._get_moe_utils_module()
    j["moe_utils_build_or_load_s"] = round(time.time() - t, 1)
    dev = "cuda"
    gen = torch.Generator().manual_seed(1)
    for c in cands:
        ck = R["_ck"]
        x = make_x(4, ck, gen, dev)
        ids, w = make_routing(4, "uniform", R["_E"], gen, dev)
        out = torch.empty(4, H, dtype=torch.bfloat16, device=dev)
        torch.cuda.synchronize()
        t = time.time()
        try:
            c(x, ids, w, out)
            torch.cuda.synchronize()
            j[c.name] = dict(first_call_s=round(time.time() - t, 2), ok=True)
        except Exception as e:  # noqa: BLE001
            j[c.name] = dict(first_call_s=round(time.time() - t, 2), ok=False, error=f"{type(e).__name__}: {e}"[:2000],
                             tb=traceback.format_exc()[-3000:])
            log(f"  FIRST CALL FAILED {c.name}: {e}")
        log(f"  jit {c.name}: {j[c.name]['first_call_s']} s ok={j[c.name]['ok']}")
    R["jit"] = j


def phase_numerics(R, cands, args):
    import torch

    dev = "cuda"
    ck = R["_ck"]
    num = {}
    cases = [(T, "uniform") for T in NUMERICS_T] + list(EDGE_CASES)
    for T, mode in cases:
        gen = torch.Generator().manual_seed(100 + T)
        x = make_x(T, ck, gen, dev)
        ids, w = make_routing(T, mode, R["_E"], gen, dev)
        ref = reference_moe(ck, x, ids, w)
        ref16 = reference_moe(ck, x, ids, w, quant_act=False)
        key = f"T{T}_{mode}"
        num[key] = {"w4a4_ref_vs_w4a16_ref": compare(ref, ref16)}
        for c in cands:
            if not R["jit"].get(c.name, {}).get("ok", True):
                continue
            out = torch.full((T, H), float("nan"), dtype=torch.bfloat16, device=dev)
            try:
                c(x, ids, w, out)
                torch.cuda.synchronize()
                m = compare(out, ref)
                if mode == "zero_weight_row":
                    m["zero_row_max_abs"] = float(out[0].float().abs().max())
                num[key][c.name] = m
            except Exception as e:  # noqa: BLE001
                num[key][c.name] = dict(error=f"{type(e).__name__}: {e}"[:1500])
        log(f"  numerics {key}: " + ", ".join(
            f"{k}={v['rel_l2']:.4g}" if "rel_l2" in v else f"{k}=ERR {v.get('error', '?')[:200]}"
            for k, v in num[key].items()))
        del x, ids, w, ref, ref16
    R["numerics"] = num


def phase_timing(R, cands, args):
    import torch

    dev = "cuda"
    ck = R["_ck"]
    tim = {}
    for mode in args.routings:
        for T in args.decode_t + args.prefill_t:
            graph = T in args.decode_t
            ncalls = args.calls_per_graph if graph else 1
            gen = torch.Generator().manual_seed(1000 + T)
            inputs = []
            distinct = []
            for _ in range(ncalls):
                x = make_x(T, ck, gen, dev)
                ids, w = make_routing(T, mode, R["_E"], gen, dev)
                distinct.append(int(ids.unique().numel()))
                inputs.append((x, ids, w, torch.empty(T, H, dtype=torch.bfloat16, device=dev)))
            key = f"{mode}_T{T}"
            tim[key] = dict(mode=mode, T=T, graph=graph, calls_per_graph=ncalls,
                            distinct_experts_mean=st.mean(distinct))
            for c in cands:
                if not R["jit"].get(c.name, {}).get("ok", True):
                    continue
                try:
                    calls = [(lambda c=c, a=a: c(*a)) for a in inputs]
                    if graph:
                        g, ms = time_graph(calls, args.replays)
                        tim[key][c.name] = pstats(ms, ncalls)
                        del g
                    else:
                        ms = time_eager(calls[0], args.eager_iters)
                        tim[key][c.name] = pstats(ms)
                except Exception as e:  # noqa: BLE001
                    tim[key][c.name] = dict(error=f"{type(e).__name__}: {e}"[:1500], tb=traceback.format_exc()[-2000:])
                    log(f"  TIMING FAILED {key} {c.name}: {e}")
                torch.cuda.synchronize()
            log(f"  timing {key} (distinct experts {tim[key]['distinct_experts_mean']:.1f}): " + ", ".join(
                f"{c.name}={tim[key][c.name]['median']:.3f}ms" for c in cands if "median" in tim[key].get(c.name, {})))
            del inputs
            gc.collect()
            torch.cuda.empty_cache()
    R["timing"] = tim


def phase_determinism(R, cands, args):
    import torch

    dev = "cuda"
    ck = R["_ck"]
    det = {}
    for T in (4, 7488):
        gen = torch.Generator().manual_seed(7 + T)
        x = make_x(T, ck, gen, dev)
        ids, w = make_routing(T, "uniform", R["_E"], gen, dev)
        for c in cands:
            if not R["jit"].get(c.name, {}).get("ok", True):
                continue
            out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
            try:
                if T <= 512:
                    g, _ = time_graph([lambda: c(x, ids, w, out)], 1, warm=0)
                    run = g.replay
                else:
                    run = lambda: c(x, ids, w, out)  # noqa: E731
                run()
                torch.cuda.synchronize()
                first = out.clone()
                neq, maxd = 0, 0.0
                for _ in range(args.det_runs - 1):
                    run()
                    torch.cuda.synchronize()
                    if not torch.equal(out, first):
                        neq += 1
                        maxd = max(maxd, float((out.float() - first.float()).abs().max()))
                det[f"T{T}_{c.name}"] = dict(runs=args.det_runs, runs_differing=neq, max_abs_diff=maxd,
                                             mode="graph replay" if T <= 512 else "eager")
            except Exception as e:  # noqa: BLE001
                det[f"T{T}_{c.name}"] = dict(error=f"{type(e).__name__}: {e}"[:1000])
        del x, ids, w
    R["determinism"] = det
    log("  determinism:", json.dumps(det))


def phase_profile(R, cands, args):
    import torch

    dev = "cuda"
    ck = R["_ck"]
    prof = {}
    for T in (4, 7488):
        gen = torch.Generator().manual_seed(55 + T)
        n = args.calls_per_graph if T <= 512 else 1
        inputs = []
        for _ in range(n):
            x = make_x(T, ck, gen, dev)
            ids, w = make_routing(T, "uniform", R["_E"], gen, dev)
            inputs.append((x, ids, w, torch.empty(T, H, dtype=torch.bfloat16, device=dev)))
        for c in cands:
            if not R["jit"].get(c.name, {}).get("ok", True):
                continue
            try:
                calls = [(lambda c=c, a=a: c(*a)) for a in inputs]
                reps = 5
                if T <= 512:
                    g, _ = time_graph(calls, 1)
                    per = profile_calls(lambda: [g.replay() for _ in range(reps)], f"T{T}_{c.name}", args.out, n * reps)
                    if not per:  # CUPTI did not report graph kernels: profile eager calls instead
                        per = profile_calls(lambda: [f() for _ in range(reps) for f in calls], f"T{T}_{c.name}_eager",
                                            args.out, n * reps)
                else:
                    calls[0]()
                    per = profile_calls(lambda: [calls[0]() for _ in range(reps)], f"T{T}_{c.name}", args.out, reps)
                prof[f"T{T}_{c.name}"] = per
            except Exception as e:  # noqa: BLE001
                prof[f"T{T}_{c.name}"] = dict(error=f"{type(e).__name__}: {e}"[:1000])
        del inputs
    R["profile"] = prof
    log("  profile (ms per call by category):", json.dumps(prof, indent=1))


def phase_jitwarm(R, args):
    """Second process with the same CuTe DSL file cache: what a server restart would pay."""
    cmd = [sys.executable, os.path.abspath(__file__), "--phase-jitwarm-child", "--out", args.out,
           "--weights", "random", "--experts", "64"]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    R["jit_warm"] = dict(returncode=r.returncode, wall_s=round(time.time() - t, 1),
                         child=r.stdout.strip().splitlines()[-1:] if r.stdout else [], stderr_tail=r.stderr[-1500:])
    log("  jit warm:", R["jit_warm"])


# ----------------------------------------------------------------------------------------------
# G0 verdict
# ----------------------------------------------------------------------------------------------


def verdict(R):
    v = {}
    tim, num, jit = R.get("timing", {}), R.get("numerics", {}), R.get("jit", {})
    base = Cutlass.name
    names = [k for k in jit if k.startswith("fi_cutedsl")]
    cold_total = jit.get("moe_utils_build_or_load_s", 0) + sum(
        jit[n]["first_call_s"] for n in names if isinstance(jit.get(n), dict))
    for n in names:
        ok_run = jit[n].get("ok") and all(
            n in t and "median" in t[n] for t in tim.values()) and not any(
            "error" in d.get(n, {}) for d in num.values())
        tol = []
        for key, d in num.items():
            if base in d and n in d and "rel_l2" in d[n] and "rel_l2" in d[base]:
                tol.append(d[n]["rel_l2"] <= 1.5 * d[base]["rel_l2"] + 1e-6 and d[n]["nonfinite"] == 0)
        num_ok = bool(tol) and all(tol)
        dec = tim.get("uniform_T4", {})
        pre = tim.get("uniform_T7488", {})

        def ratio(d):
            try:
                return d[n]["median"] / d[base]["median"]
            except (KeyError, ZeroDivisionError, TypeError):
                return None

        rd, rp = ratio(dec), ratio(pre)
        perf_ok = rd is not None and rp is not None and (
            (rd <= 0.90 and rp <= 1.0) or (rp <= 0.80 and rd <= 1.0))
        v[n] = dict(
            runs_all_token_counts=bool(ok_run), numerics_within_1p5x_cutlass=num_ok,
            decode_T4_ratio_vs_cutlass=rd, prefill_T7488_ratio_vs_cutlass=rp,
            decode_saving_ms_per_step_48_layers=(None if rd is None else
                                                 round((dec[base]["median"] - dec[n]["median"]) * 48, 2)),
            prefill_saving_ms_per_chunk_48_layers=(None if rp is None else
                                                   round((pre[base]["median"] - pre[n]["median"]) * 48, 1)),
            perf_gate=perf_ok, cold_jit_total_s=round(cold_total, 1), cold_jit_ok=cold_total <= 300,
            G0_pass=bool(ok_run and num_ok and perf_ok and cold_total <= 300),
        )
    R["G0"] = v
    return v


def write_summary(R, out):
    L = ["# SM110 MoE spike summary", "", f"weights: {json.dumps(R.get('weights_meta', {}))}", ""]
    L.append("## env")
    L.append("```\n" + json.dumps(R.get("env", {}), indent=1) + "\n```")
    L.append("## cold JIT (s)\n```\n" + json.dumps(R.get("jit", {}), indent=1)[:4000] + "\n```")
    if "timing" in R:
        L.append("## timing, ms per MoE call (one layer; x48 for a step/chunk)")
        cols = sorted({k for d in R["timing"].values() for k in d if k.startswith("fi_")})
        L.append("| case | distinct experts | " + " | ".join(cols) + " |")
        L.append("|---|---|" + "---|" * len(cols))
        for key, d in R["timing"].items():
            row = [f"{d[c]['median']:.3f}" if "median" in d.get(c, {}) else ("ERR" if c in d else "-") for c in cols]
            L.append(f"| {key} | {d['distinct_experts_mean']:.1f} | " + " | ".join(row) + " |")
    if "numerics" in R:
        L.append("\n## numerics vs torch reference (rel L2 / min cosine)")
        for key, d in R["numerics"].items():
            L.append(f"- {key}: " + "; ".join(
                f"{k} {v['rel_l2']:.4f}/{v['cos_min'] if v['cos_min'] is None else round(v['cos_min'],4)}"
                if "rel_l2" in v else f"{k} {v.get('error','?')[:120]}" for k, v in d.items()))
    for sec in ("determinism", "profile", "jit_warm", "G0"):
        if sec in R:
            L.append(f"\n## {sec}\n```\n" + json.dumps(R[sec], indent=1)[:6000] + "\n```")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------------------------
# Dry run (CPU only)
# ----------------------------------------------------------------------------------------------


def dry_run(args, R):
    import torch

    log("DRY RUN (CPU): no kernel launches")
    torch.Tensor.cuda = lambda self, *a, **k: self  # vLLM's swizzle_blockscale calls .cuda()
    from sm110_lift import lift_cutedsl_moe_gates

    R["lifted"] = lift_cutedsl_moe_gates()
    n = args.experts
    if args.weights == "random":
        ck, meta = random_layer_weights(n)
    else:
        ck, meta = load_layer_weights(args.ckpt, args.layer, n)
    R["weights_meta"] = meta
    log("weights:", meta)
    pc = prep_cutlass(ck, "cpu")
    pd = prep_cutedsl(ck, "cpu")
    R["prep_cutlass"] = {k: [list(v.shape), str(v.dtype)] for k, v in vars(pc).items()}
    R["prep_cutedsl"] = {k: [list(v.shape), str(v.dtype)] for k, v in vars(pd).items()}
    log("prep cutlass:", R["prep_cutlass"])
    log("prep cutedsl:", R["prep_cutedsl"])
    gen = torch.Generator().manual_seed(0)
    T = 4
    x = make_x(T, ck, gen, "cpu")
    ids, w = make_routing(T, "uniform", n, gen, "cpu")
    t = time.time()
    ref = reference_moe(ck, x, ids, w)
    ref16 = reference_moe(ck, x, ids, w, quant_act=False)
    R["reference_T4"] = dict(seconds=round(time.time() - t, 2), w4a4_vs_w4a16=compare(ref, ref16),
                             out_abs_max=float(ref.abs().max()))
    log("reference:", R["reference_T4"])
    # argument plumbing: bind the exact kwargs against the installed signatures
    from flashinfer.fused_moe import cute_dsl_fused_moe, cutlass_fused_moe

    xq = torch.zeros(T, H // 2, dtype=torch.uint8)
    xsf = torch.zeros(T, H // 16, dtype=torch.float8_e4m3fn)
    out = torch.empty(T, H, dtype=torch.bfloat16)
    binds = {}
    for c, fn in ((Cutlass(pc), cutlass_fused_moe), (CuteDsl(pd, 128, True), cute_dsl_fused_moe),
                  (CuteDsl(pd, 256, False), cute_dsl_fused_moe)):
        try:
            inspect.signature(getattr(fn, "__wrapped__", fn)).bind(**c.kwargs(xq, xsf, ids, w, out))
            binds[c.name] = "ok"
        except TypeError as e:
            binds[c.name] = f"BIND FAILED: {e}"
    R["arg_binding"] = binds
    log("arg binding:", binds)
    # cross-compile the CuTe DSL kernels for sm_110a in a child (CUTE_DSL_ARCH must be set before import)
    xc = {}
    for which, nt, fin in (("fc1", 128, ""), ("fc1", 256, ""), ("fc2", 128, "fused"), ("fc2", 256, "fused"),
                           ("fc2", 128, "twostage")):
        envv = dict(os.environ, CUTE_DSL_ARCH="sm_110a")
        r = subprocess.run([sys.executable, os.path.join(HERE, "xcompile_sm110a.py"), which, str(nt), fin or "fused"],
                           capture_output=True, text=True, env=envv, timeout=900)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith(("COMPILED", "FAILED", "PTX"))]
        xc[f"{which}_n{nt}_{fin}"] = lines or [r.stderr[-800:]]
    R["xcompile_sm110a"] = xc
    log("xcompile:", json.dumps(xc, indent=1))


# ----------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/out")
    ap.add_argument("--weights", choices=["fp4local", "nvfp4", "random"], default="fp4local")
    ap.add_argument("--ckpt", default=None, help="checkpoint dir (default by --weights)")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--experts", type=int, default=None, help=f"default {E} (GPU) / 16 (dry run)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--routings", nargs="+", default=["uniform", "skewed"])
    ap.add_argument("--decode-t", type=int, nargs="+", default=DECODE_T)
    ap.add_argument("--prefill-t", type=int, nargs="+", default=PREFILL_T)
    ap.add_argument("--replays", type=int, default=200)
    ap.add_argument("--eager-iters", type=int, default=20)
    ap.add_argument("--calls-per-graph", type=int, default=8,
                    help="distinct MoE calls (inputs/routings) per captured graph, like successive layers")
    ap.add_argument("--det-runs", type=int, default=20)
    ap.add_argument("--variants", nargs="+", default=["n128_fused", "n256_fused", "n128_det", "n256_det", "vllmdefault"])
    ap.add_argument("--phases", nargs="+",
                    default=["jit", "numerics", "timing", "determinism", "profile", "jitwarm"])
    ap.add_argument("--phase-jitwarm-child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.ckpt is None:
        args.ckpt = {"fp4local": "/hf/linked/fn-fp4-local",
                     "nvfp4": "/hf/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/"
                              "fc694b54fb0174e0913e6adf86691ef85a4ead47"}.get(args.weights)
    os.makedirs(args.out, exist_ok=True)
    if args.experts is None and not args.dry_run:
        args.experts = E
    R = dict(args=vars(args), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    t_all = time.time()
    res_path = os.path.join(args.out, "dryrun.json" if args.dry_run else
                            "jitwarm_child.json" if args.phase_jitwarm_child else "results.json")

    def dump():
        R["wall_s"] = round(time.time() - t_all, 1)
        json.dump({k: v for k, v in R.items() if not k.startswith("_")}, open(res_path, "w"), indent=1, default=str)

    try:
        if args.dry_run:
            args.experts = args.experts or 16  # >= top-k
            phase_env(R)
            dry_run(args, R)
            return
        import torch

        assert torch.cuda.is_available(), "GPU run needs the GPU (use --dry-run otherwise)"
        if args.phase_jitwarm_child:
            from sm110_lift import lift_cutedsl_moe_gates

            lift_cutedsl_moe_gates()
            ck, _ = random_layer_weights(args.experts)
            R["_ck"], R["_E"] = ck, args.experts
            pd = prep_cutedsl(ck, "cuda")
            c = CuteDsl(pd, 128, True)
            t = time.time()
            import flashinfer.fused_moe.cute_dsl.moe_utils as mu

            mu._get_moe_utils_module()
            gen = torch.Generator().manual_seed(1)
            x = make_x(4, ck, gen, "cuda")
            ids, w = make_routing(4, "uniform", args.experts, gen, "cuda")
            c(x, ids, w, torch.empty(4, H, dtype=torch.bfloat16, device="cuda"))
            torch.cuda.synchronize()
            print(f"WARM first call n128_fused incl. moe_utils load: {time.time()-t:.2f} s")
            return
        phase_env(R)
        t = time.time()
        if args.weights == "random":
            ck, meta = random_layer_weights(args.experts)
        else:
            ck, meta = load_layer_weights(args.ckpt, args.layer, args.experts)
        meta["load_s"] = round(time.time() - t, 1)
        R["weights_meta"] = meta
        R["_ck"], R["_E"] = ck, args.experts
        log("weights:", meta)
        cands = [Cutlass(prep_cutlass(ck, "cuda"))]
        pd = prep_cutedsl(ck, "cuda")
        for v in args.variants:
            if v == "vllmdefault":
                cands.append(CuteDsl(pd, 128, True, tactic_explicit=False))
            else:
                nt, fin = v.split("_")
                cands.append(CuteDsl(pd, int(nt[1:]), fin == "fused"))
        R["candidates"] = [c.name for c in cands]
        R["edge_llm_fallback"] = "not run (documented stub; only if every CuTe DSL variant fails G0)"
        if "jit" in args.phases:
            phase_jit(R, cands)
        else:
            R["jit"] = {}
        dump()
        for ph, fn in (("numerics", phase_numerics), ("timing", phase_timing),
                       ("determinism", phase_determinism), ("profile", phase_profile)):
            if ph in args.phases:
                log(f"== {ph}")
                try:
                    fn(R, cands, args)
                except Exception as e:  # noqa: BLE001
                    R[f"{ph}_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}"
                    log(f"PHASE {ph} FAILED: {e}")
                dump()
        if "jitwarm" in args.phases:
            phase_jitwarm(R, args)
        verdict(R)
        log("G0:", json.dumps(R["G0"], indent=1))
    finally:
        dump()
        if not args.dry_run and not args.phase_jitwarm_child:
            write_summary(R, args.out)
        log(f"done in {time.time()-t_all:.0f} s -> {res_path}")


if __name__ == "__main__":
    main()
