#!/usr/bin/env python3
"""Add DSpark K=7 verification widths to vLLM's FlashInfer warmup pass."""

from pathlib import Path


path = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/warmup/kernel_warmup.py"
)

before = '''_FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS = 32


def _flashinfer_autotune_token_counts(runner: "GPUModelRunner") -> tuple[int, ...]:
    max_tokens = runner.scheduler_config.max_num_batched_tokens
    linear_backend = runner.vllm_config.kernel_config.linear_backend
    if (
        linear_backend == "flashinfer_cutedsl"
        and max_tokens > _FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS
    ):
        return max_tokens, _FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS
    return (max_tokens,)
'''

after = '''_FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS = 32
# Qwen3.8 DSpark K=7 verifies nine target tokens per active sequence. Tune
# every flattened width reachable by the profile's four scheduler slots.
_FLASHINFER_DSPARK_VERIFY_TOKEN_COUNTS = (9, 18, 27, 36)


def _flashinfer_autotune_token_counts(runner: "GPUModelRunner") -> tuple[int, ...]:
    max_tokens = runner.scheduler_config.max_num_batched_tokens
    linear_backend = runner.vllm_config.kernel_config.linear_backend
    token_counts = [max_tokens]
    if (
        linear_backend == "flashinfer_cutedsl"
        and max_tokens > _FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS
    ):
        token_counts.append(_FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS)
    token_counts.extend(_FLASHINFER_DSPARK_VERIFY_TOKEN_COUNTS)
    # Preserve order and avoid profiling a shape twice.
    return tuple(dict.fromkeys(token_counts))
'''

source = path.read_text()
matches = source.count(before)
if matches != 1:
    raise SystemExit(
        f"expected exactly one FlashInfer autotune block in {path}, found {matches}"
    )
path.write_text(source.replace(before, after, 1))
