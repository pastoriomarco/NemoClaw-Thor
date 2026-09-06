#!/usr/bin/env python3
"""Allow vLLM's FlashInfer CUTLASS fused-MoE backend on Jetson Thor."""

from pathlib import Path


path = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "fused_moe/experts/flashinfer_cutlass_moe.py"
)
text = path.read_text()

old = """                or p.is_device_capability_family(100)
                # SM110 excluded: flashinfer-ai/flashinfer#3134
                or p.is_device_capability_family(120)
"""
new = """                or p.is_device_capability_family(100)
                # Thor: FlashInfer 0.6.17 maps SM110 to its SM100 CUTLASS
                # implementation and enables CUDA major version 11 for it.
                or p.is_device_capability_family(110)
                or p.is_device_capability_family(120)
"""

if new in text:
    print("SM110 FlashInfer CUTLASS MoE gate already enabled")
elif text.count(old) == 1:
    path.write_text(text.replace(old, new, 1))
    print("Enabled FlashInfer CUTLASS MoE backend on SM110")
else:
    raise SystemExit(
        f"Refusing to patch {path}: expected vLLM gate was not found exactly once"
    )
