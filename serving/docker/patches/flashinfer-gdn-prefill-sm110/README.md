# FlashInfer GDN prefill on sm_110 (Jetson AGX Thor)

Read-only two-file overlay for the official `vllm/vllm-openai@sha256:d825baf9…` image (vLLM 0.33.0, FlashInfer
0.7.2.post1). It lets the Gated DeltaNet (GDN) prefill of Qwen3.5+/Flash-Next run FlashInfer's SM100 CuTe DSL
kernel on sm_110 instead of Triton/FLA. CuTe DSL compiles that kernel natively for `sm_110a`; only arch checks
blocked it. A minimal backport of the non-CP part of the open upstream PR flashinfer-ai/flashinfer#7827, plus the
vLLM selector change its author plans as a follow-up (vLLM issue #65054).

Status (2026-11-16): prepared, not run yet. Opt-in only: without the flag below nothing changes.

## Why

Flash-Next has 36 GDN layers with 16 key / 48 value heads, K = V = 128, the shapes benchmarked on Thor in vLLM
issue #65054. Kernel time per layer there, 8,192 tokens: FLA 11.4 ms, FlashInfer 1.7 ms. For 36 layers about
0.41 s vs 0.06 s per 8K tokens of prefill (our profile measured 430 ms for GDN prefill). Expected: about −14 %
prefill at 8K and −16 % at 32K. Decode is not affected.

## Change

Each file is the image's own file with the hunks marked `[thor]`:

| File | Change |
|---|---|
| `flashinfer/gdn_prefill.py` (sha256 of the base `9a3e122b…ae17d`, tag v0.7.2.post1) | `chunk_gated_delta_rule`: compute capability major 11 takes the SM100 non-CP CuTe DSL path (`_arch_major in (10, 11)`), including `state_indices`, the in-kernel q/k L2 norm and the align-mode state checkpoints that path already supports. Context-parallel (CP) stays off on sm_110: the `use_cp="auto"` heuristic does not match major 11, and #7827's own Thor rule never picks CP for 48 value heads. |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (base sha256 `c6c037ad…17f9`, tag v0.33.0) | `_resolve_gdn_prefill_backend`: on family 110 with `head_k_dim == 128` and CUDA 13, FlashInfer is allowed when explicitly requested. `auto` keeps Triton/FLA. |

## Use

Mount over the image's files, read-only, and request the backend through vLLM 0.33's additional config:

```
-v $O/flashinfer/gdn_prefill.py:/usr/local/lib/python3.12/dist-packages/flashinfer/gdn_prefill.py:ro
-v $O/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro
... --additional-config '{"gdn_prefill_backend":"flashinfer"}'
```

Expected log line: `Using FlashInfer GDN prefill kernel (requested=flashinfer, head_k_dim=128).` The first
prefill JIT-compiles the CuTe DSL kernel.

## Verify before use (planned)

1. Image files equal the base checksums above (otherwise rebase the overlay).
2. FlashInfer's `tests/gdn/test_prefill_delta_rule.py` and `test_prefill_state_indices.py` on Thor in a container,
   with the overlay mounted.
3. R8b and N3 with and without the flag, alternated: cold prefill at 8K and 32K, decode step time (unchanged),
   prefix-cache hits with align mode.
4. Tool Eval on both recipes (different kernel, rounding-level differences).

Drop this overlay once a FlashInfer release has #7827 and vLLM's pin and selector include it.
