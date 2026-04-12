# SM110 (Thor) capability family patch for NVFP4 / FlashInfer MoE kernels.
#
# SM110 (Jetson AGX Thor) is part of the Blackwell enterprise family alongside
# SM100 (B200/B300) and SM103 (GB300). They share tcgen05 tensor operations
# and the same CUTLASS/FlashInfer kernel implementations. However, vLLM's
# is_device_capability_family(100) returns False for SM110 because 110//10=11
# != 100//10=10. This causes NVFP4 MoE to fall back to slower Marlin kernels.
#
# This patch makes is_device_capability_family(100) return True on SM110,
# enabling native FlashInfer NVFP4 kernels (which have SM110a cubins in the
# JIT cache).
#
# References:
#   - vllm-project/vllm#33333, #33416 (SM120 variant of the same bug)
#   - patrickbdevaney/qwen-3.5-35b-a3b-vllm-resources (SM110 patch)
#   - NVIDIA/cutlass#2802 (SM110a in tcgen05 allowed list)

import logging

logger = logging.getLogger(__name__)

# SM110 belongs to the SM100 enterprise Blackwell family.
_SM110_FAMILY_ALIASES = {100}


def patch():
    from vllm.platforms import current_platform

    platform_cls = type(current_platform)
    original = platform_cls.is_device_capability_family.__func__

    @classmethod  # type: ignore[misc]
    def _patched(cls, capability, device_id=0):
        if original(cls, capability, device_id):
            return True
        # On SM110, also match requests for SM100 family.
        if capability in _SM110_FAMILY_ALIASES:
            return original(cls, 110, device_id)
        return False

    platform_cls.is_device_capability_family = _patched
    logger.info("SM110 capability family patch applied (SM110 -> SM100 family)")
