"""In-process lifts of the sm_110 gates that block FlashInfer 0.7.2.post1's CuTe DSL NVFP4 MoE.

Nothing on disk is changed: each lift rebinds one name inside an already imported module, in this
process only. Every lift corresponds to one line that a Python-file overlay would change; the
file:line references are for the files inside vllm/vllm-openai@sha256:d825baf9... (FlashInfer 0.7.2.post1).

  L1 flashinfer/fused_moe/cute_dsl/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py:686-692
     `if major != 10: raise ValueError(... requires SM10x family ...)`  (FC1 wrapper)
  L2 flashinfer/fused_moe/cute_dsl/blockscaled_contiguous_grouped_gemm_finalize_fusion.py:511-517
     same check in the FC2 wrapper
  L3 flashinfer/jit/moe_utils.py:86-88
     `get_nvcc_flags_list(supported_major_versions=[9, 10], ...)` -> RuntimeError
     "No supported CUDA architectures found" on sm_110 (moe_sort/memset/unpermute JIT module;
     no prebuilt sm110a moe_utils.so exists in flashinfer_jit_cache, so it is built from source)

Not lifted because they do not block a direct call (see README):
  - @supported_compute_capability([100, 103, 107]) on cute_dsl_fused_moe(_nvfp4) and the wrapper
    class (fused_moe.py:942,1502,1863) only attach metadata (flashinfer/utils.py:1209-1216);
  - require_cute_dsl_arch(native_only=True) (fused_moe.py:1692) passes: DSL 4.8.0 has Arch.sm_110a;
  - vLLM FlashInferCuteDSLExperts._supports_current_device (family 100) is not on this path.
"""

from __future__ import annotations

import functools

LIFTED: list[str] = []


def _cc_as_sm100(real):
    @functools.wraps(real)
    def wrapped(device=None):
        major, minor = real(device)
        if major == 11:
            return 10, 0  # only consulted for `major != 10` and the SM107 (minor == 7) checks
        return major, minor

    return wrapped


def lift_cutedsl_moe_gates() -> list[str]:
    """Apply L1-L3. Idempotent. Returns the list of lifted gates."""
    if LIFTED:
        return LIFTED
    import flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion as fc1w
    import flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion as fc2w

    fc1w.get_compute_capability = _cc_as_sm100(fc1w.get_compute_capability)
    LIFTED.append("L1 FC1 wrapper major!=10")
    fc2w.get_compute_capability = _cc_as_sm100(fc2w.get_compute_capability)
    LIFTED.append("L2 FC2 wrapper major!=10")

    import flashinfer.jit.moe_utils as jmu

    real_gen = jmu.gen_moe_utils_module
    if not getattr(real_gen, "_sm110_lift", False):
        ctx = jmu.current_compilation_context

        @functools.wraps(real_gen)
        def gen_with_11():
            orig = ctx.get_nvcc_flags_list

            def flags(supported_major_versions=None, map_sm107_to_100f=False):
                if supported_major_versions and 10 in supported_major_versions:
                    supported_major_versions = sorted(set(supported_major_versions) | {11})
                return orig(supported_major_versions=supported_major_versions,
                            map_sm107_to_100f=map_sm107_to_100f)

            ctx.get_nvcc_flags_list = flags
            try:
                return real_gen()
            finally:
                del ctx.get_nvcc_flags_list  # back to the class method

        gen_with_11._sm110_lift = True
        jmu.gen_moe_utils_module = gen_with_11
        import flashinfer.fused_moe.cute_dsl.moe_utils as mu  # imported the name at module load

        mu.gen_moe_utils_module = gen_with_11
        mu._get_moe_utils_module.cache_clear() if hasattr(mu._get_moe_utils_module, "cache_clear") else None
    LIFTED.append("L3 moe_utils JIT arch list [9,10] -> [9,10,11]")
    return LIFTED
