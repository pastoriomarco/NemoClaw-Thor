# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.flashinfer import has_flashinfer, has_flashinfer_cutedsl

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

# [thor] begin: optional CUTLASS MXFP8 tactic pin by M bucket (default off).
# VLLM_MXFP8_TACTIC="<max_m>:<tactic>[,<max_m>:<tactic>...]", e.g. "256:8": calls with
# M <= 256 run FlashInfer CUTLASS tactic 8; larger M keep the stock mm_mxfp8 call.
# Kernel choice only: every tactic computes bit-identical output (see README).
import functools
import os

from vllm.logger import init_logger

_thor_logger = init_logger(__name__)
_THOR_TACTICS = sorted(
    (int(m), int(t))
    for m, t in (
        b.split(":") for b in os.environ.get("VLLM_MXFP8_TACTIC", "").split(",") if b
    )
)
if _THOR_TACTICS:
    _thor_logger.info(
        "[thor] MXFP8 CUTLASS tactic pins (VLLM_MXFP8_TACTIC): %s; larger M stock",
        ", ".join(f"M<={m} -> tactic {t}" for m, t in _THOR_TACTICS),
    )

    @functools.cache
    def _thor_cutlass_runner(major: int):
        from flashinfer.gemm.gemm_base import get_cutlass_mxfp8_gemm_module

        return get_cutlass_mxfp8_gemm_module(major).cutlass_mxfp8_gemm_runner()

    @torch.library.custom_op(
        "vllm::thor_mm_mxfp8_tactic", mutates_args=[], device_types="cuda"
    )
    def _thor_mm_mxfp8_tactic(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        m = A.shape[0]
        tactic = next((t for max_m, t in _THOR_TACTICS if m <= max_m), None)
        if tactic is None:
            from flashinfer import mm_mxfp8

            return mm_mxfp8(
                A, B, A_scale, B_scale, out_dtype=out_dtype, backend="cutlass"
            )
        from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE, _get_cache_buf

        out = torch.empty((m, B.shape[1]), device=A.device, dtype=out_dtype)
        workspace = _get_cache_buf("mm_mxfp8_workspace", DEFAULT_WORKSPACE_SIZE, A.device)
        runner = _thor_cutlass_runner(torch.cuda.get_device_capability(A.device)[0])
        runner.forward(
            [A, B, A_scale, B_scale, out_dtype, out, workspace], tactic=tactic
        )
        return out

    @torch.library.register_fake("vllm::thor_mm_mxfp8_tactic")
    def _thor_mm_mxfp8_tactic_fake(A, B, A_scale, B_scale, out_dtype):
        return torch.empty(A.shape[0], B.shape[1], dtype=out_dtype, device=A.device)
# [thor] end


def _check_mm_mxfp8_shape(weight_shape: tuple[int, int]) -> tuple[bool, str | None]:
    N, K = weight_shape
    if K < 128:
        return False, f"mm_mxfp8 requires K >= 128, got K={K}."
    if K % MXFP8_BLOCK_SIZE != 0:
        return (
            False,
            f"mm_mxfp8 requires K to be divisible by {MXFP8_BLOCK_SIZE}, got K={K}.",
        )
    if N < 128:
        return False, f"mm_mxfp8 requires N >= 128, got N={N}."
    return True, None


class FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer CUTLASS (SM100+)."""

    supports_pre_processed_weights = True

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda() and current_platform.has_device_capability(100)
        ):
            return False, "requires >=sm_100 (Blackwell)"
        if not has_flashinfer():
            return False, "requires FlashInfer"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        can_implement, reason = _check_mm_mxfp8_shape(c.weight_shape)
        if not can_implement:
            return False, reason
        N, _ = c.weight_shape
        if current_platform.is_device_capability_family(120) and N % 32 != 0:
            return (
                False,
                f"mm_mxfp8 on SM12x requires N to be divisible by 32, got N={N}.",
            )
        return True, None

    def input_quant_key(self) -> QuantKey:
        # Activations use FlashInfer's F8_128x4 swizzled scale layout.
        return kMxfp8Dynamic

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)

        replace_parameter(layer, "weight", weight.contiguous())
        replace_parameter(layer, "weight_scale", weight_scale_swizzled.contiguous())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        N, K = weight.shape

        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            input_mxfp8, input_scale = qa.data, qa.scale
            out_dtype, input_shape = qa.orig_dtype, qa.orig_shape
        else:
            assert isinstance(x, torch.Tensor)
            out_dtype, input_shape = x.dtype, x.shape
            input_mxfp8, input_scale = mxfp8_e4m3_quantize(
                x.view(-1, K), is_sf_swizzled_layout=True
            )

        if not weight.is_contiguous():
            weight = weight.contiguous()

        if _THOR_TACTICS:  # [thor] tactic pin by M bucket
            output = torch.ops.vllm.thor_mm_mxfp8_tactic(
                input_mxfp8, weight.t(), input_scale, weight_scale, out_dtype
            )
        else:
            output = vllm_flashinfer.mm_mxfp8(
                input_mxfp8,
                weight.t(),
                input_scale,
                weight_scale,
                out_dtype=out_dtype,
                backend="cutlass",
            )

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)


class FlashInferCutedslMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer CuTe-DSL (SM100/SM103)."""

    supports_pre_processed_weights = True

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            return False, "requires sm_100/sm_103 (Blackwell)"
        if not has_flashinfer_cutedsl():
            return False, "requires FlashInfer CuTe-DSL module"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return _check_mm_mxfp8_shape(c.weight_shape)

    def input_quant_key(self) -> QuantKey:
        # Activations use FlashInfer's F8_128x4 swizzled scale layout.
        return kMxfp8Dynamic

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)

        # Store weight column-major [K, N] as mm_mxfp8 expects for operand B.
        replace_parameter(layer, "weight", weight.contiguous().t())
        replace_parameter(layer, "weight_scale", weight_scale_swizzled.contiguous())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight  # [K, N], column-major
        weight_scale = layer.weight_scale
        K, N = weight.shape

        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            input_mxfp8, input_scale = qa.data, qa.scale
            out_dtype, input_shape = qa.orig_dtype, qa.orig_shape
        else:
            assert isinstance(x, torch.Tensor)
            out_dtype, input_shape = x.dtype, x.shape
            input_mxfp8, input_scale = mxfp8_e4m3_quantize(
                x.view(-1, K), is_sf_swizzled_layout=True
            )

        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight,
            input_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend="cute-dsl",
        )

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)


class FlashInferTrtllmMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer's TensorRT-LLM wrapper."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            return False, "requires SM100-family GPU"
        if not has_flashinfer():
            return False, "requires FlashInfer"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        if hasattr(layer, "_mxfp8_trtllm_output_size") and layer.weight_scale.ndim == 1:
            return

        weight = layer.weight.data  # [N, K]
        N, K = weight.shape
        if K % 256 != 0:
            raise ValueError(
                f"FlashInfer TRTLLM MXFP8 requires K to be divisible by 256, got K={K}."
            )

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()
        padded_n = ((N + 127) // 128) * 128
        if padded_n != N:
            padded_weight = weight.new_zeros((padded_n, K))
            padded_weight[:N] = weight
            weight = padded_weight

            padded_scale = weight_scale.new_zeros((padded_n, scale_k))
            padded_scale[:N] = weight_scale
            weight_scale = padded_scale
        else:
            weight = weight.contiguous()

        replace_parameter(
            layer,
            "weight",
            shuffle_matrix_a(weight, 128).reshape(padded_n, K),
        )
        replace_parameter(
            layer,
            "weight_scale",
            shuffle_matrix_sf_a(
                weight_scale,
                128,
                num_elts_per_sf=MXFP8_BLOCK_SIZE,
            ).reshape(-1),
        )
        layer._mxfp8_trtllm_output_size = N

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.dtype == torch.bfloat16, (
            f"FlashInfer TRTLLM MXFP8 requires bfloat16 activations, got {x.dtype}."
        )

        weight = layer.weight  # shuffled [padded N, K]
        weight_scale = layer.weight_scale
        _, K = weight.shape
        output_size = layer._mxfp8_trtllm_output_size
        input_shape = x.shape
        input_2d = x.view(-1, K)

        input_mxfp8, input_scale = vllm_flashinfer.flashinfer_mxfp8_quantize_8x4(
            input_2d
        )
        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=x.dtype,
            backend="trtllm",
            use_8x4_sf_layout=True,
        )
        if output.shape[-1] != output_size:
            output = output[:, :output_size].contiguous()

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], output_size)
        return output.view(output_shape)
