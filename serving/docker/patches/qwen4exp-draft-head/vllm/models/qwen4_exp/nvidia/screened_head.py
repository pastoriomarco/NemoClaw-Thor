# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[screened-head] Exact top-k target logits from an FP8 screen of the lm_head.

Backport of vLLM PR #60566 ("[MRV2][Perf] Screened lm_head") to the 0.33 Qwen4Exp
model files, switched by ``QWEN4EXP_LM_HEAD_SCREENED=1`` instead of
``--screened-lm-head``.

An FP8 copy of the lm_head (one scale per row) scores every row j as s_j. The
logit the BF16 head computes before rounding, c_j, satisfies
|c_j - s_j| <= ||x|| * err_j (Cauchy-Schwarz on W_j - W8_j plus the fp32
accumulation error of both GEMMs; err_j is computed in fp64 at load). Let t be
the k-th largest per-block maximum of s_j - ||x|| err_j: at least k rows have
c_j >= t. A row whose upper bound is below t - 2 eps |t| sits more than one
output ulp below each of them, so it can neither enter nor tie the full head's
top k. The remaining rows get exact BF16-weight logits (fp32 accumulation) and
every other row -inf.

The target sampler only reads the top k logits when every request in the batch
is greedy (k = 1) or samples with top_k <= QWEN4EXP_LM_HEAD_SCREEN_K (top_p and
temperature act inside the top k), and nothing else touches the logits. Those
batches use the screen; all others (logprobs, penalties, logit bias, bad words,
allowed ids, thinking budget, custom logits processors, min_p, structured
outputs, no top_k, more rows than the buffers) use the full BF16 head. The
rejection sampler applies the same top-k/top-p to the target logits before it
reads target probabilities (spec_decode/rejection_sampler.py:264), so a draft
token outside the top k has target probability 0 with either head.
"""

import functools
import os
import time

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

SCREENED_ENV = "QWEN4EXP_LM_HEAD_SCREENED"
SCREEN_K_ENV = "QWEN4EXP_LM_HEAD_SCREEN_K"
SCREEN_CHECK_ENV = "QWEN4EXP_LM_HEAD_SCREEN_CHECK"

# Largest sampling top_k served by the screen. Qwen's recommended sampling uses
# top_k=20; PR #60566 measured top-32 still 23% faster than the BF16 head at 64
# rows on GB10. A larger k lowers the threshold and keeps more exact rows.
DEFAULT_SCREEN_K = 32
# Upstream cap on rows per screened call: beyond ~96 rows the BF16 GEMM is no
# longer bandwidth bound and the screen loses (GB10, same DRAM class as Thor).
MAX_TOKENS = 64

_QUANT_CHUNK = 4096
_FP8_MAX = 448.0  # float8_e4m3fn


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "")
    return int(value) if value else default


def screened_head_settings(vllm_config) -> tuple[int, int] | None:
    """(max rows per call, max top_k) when QWEN4EXP_LM_HEAD_SCREENED is set,
    else None. Raises on combinations the screen cannot serve."""
    if os.environ.get(SCREENED_ENV, "0") in ("", "0"):
        return None
    if os.environ.get("QWEN4EXP_LM_HEAD_MXFP8", "0") not in ("", "0"):
        raise ValueError(
            f"{SCREENED_ENV}=1 and QWEN4EXP_LM_HEAD_MXFP8=1 are mutually "
            "exclusive: the screen needs the BF16 lm_head for its exact pass. "
            "Unset one of them (QWEN4EXP_DRAFT_HEAD_FP8 may stay set)."
        )
    max_top_k = _env_int(SCREEN_K_ENV, DEFAULT_SCREEN_K)
    if not 1 <= max_top_k <= 1024:
        raise ValueError(f"{SCREEN_K_ENV} must be in 1..1024, got {max_top_k}")
    import vllm.envs as envs

    reason = None
    if not torch.cuda.is_available():
        reason = "CUDA"
    elif envs.VLLM_BATCH_INVARIANT:
        # Whether a request is screened depends on the rest of its batch.
        reason = "VLLM_BATCH_INVARIANT=0"
    elif vllm_config.lora_config is not None:
        reason = "no LoRA"
    elif vllm_config.parallel_config.tensor_parallel_size > 1:
        reason = "an lm_head without TP sharding"
    if reason is not None:
        raise ValueError(f"{SCREENED_ENV}=1 requires {reason}.")
    spec = vllm_config.speculative_config
    num_spec = spec.num_speculative_tokens if spec is not None else 0
    max_rows = vllm_config.scheduler_config.max_num_seqs * (1 + (num_spec or 0))
    return min(max_rows, MAX_TOKENS), max_top_k


@triton.jit
def _screen_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    err_ptr,
    ub_ptr,
    block_max_ptr,
    num_tokens,
    vocab_size,
    hidden_size,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    block = tl.program_id(0)
    rows = block * BLOCK_V + tl.arange(0, BLOCK_V)
    toks = tl.arange(0, BLOCK_N)
    row_mask = rows < vocab_size
    tok_mask = toks < num_tokens
    acc = tl.zeros([BLOCK_N, BLOCK_V], dtype=tl.float32)
    sq = tl.zeros([BLOCK_N], dtype=tl.float32)
    for h0 in range(0, hidden_size, BLOCK_H):
        cols = h0 + tl.arange(0, BLOCK_H)
        col_mask = cols < hidden_size
        x = tl.load(
            x_ptr + toks[:, None] * hidden_size + cols[None, :],
            mask=tok_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + rows[None, :].to(tl.int64) * hidden_size + cols[:, None],
            mask=row_mask[None, :] & col_mask[:, None],
            other=0.0,
        )
        acc = tl.dot(x, w.to(x.dtype), acc)
        xf = x.to(tl.float32)
        sq += tl.sum(xf * xf, axis=1)
    s = acc * tl.load(scale_ptr + rows, mask=row_mask, other=0.0)[None, :]
    bound = tl.sqrt(sq)[:, None] * tl.load(err_ptr + rows, mask=row_mask, other=0.0)
    mask = tok_mask[:, None] & row_mask[None, :]
    lb = tl.where(mask, s - bound, float("-inf"))
    tl.store(
        block_max_ptr + toks * tl.num_programs(0) + block,
        tl.max(lb, axis=1),
        mask=tok_mask,
    )
    tl.store(
        ub_ptr + toks[:, None].to(tl.int64) * vocab_size + rows[None, :],
        s + bound,
        mask=mask,
    )


@triton.jit
def _select_kernel(
    ub_ptr,
    t_ptr,
    logits_ptr,
    cand_ptr,
    num_cand_ptr,
    vocab_size,
    TIE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < vocab_size
    t = tl.load(t_ptr + tok)
    ub = tl.load(ub_ptr + tok * vocab_size + cols, mask=mask, other=float("-inf"))
    keep = mask & (ub >= t - TIE * tl.abs(t))
    tl.store(logits_ptr + tok * vocab_size + cols, float("-inf"), mask=mask)
    num_keep = tl.sum(keep.to(tl.int32), axis=0)
    if num_keep > 0:
        base = tl.atomic_add(num_cand_ptr + tok, num_keep)
        offs = base + tl.cumsum(keep.to(tl.int32), axis=0) - 1
        tl.store(cand_ptr + tok * vocab_size + offs, cols, mask=keep)


@triton.jit
def _exact_kernel(
    x_ptr,
    w_ptr,
    cand_ptr,
    num_cand_ptr,
    logits_ptr,
    vocab_size,
    hidden_size,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    num_cand = tl.load(num_cand_ptr + tok)
    step = tl.num_programs(1) * BLOCK_R
    for start in range(tl.program_id(1) * BLOCK_R, num_cand, step):
        r = start + tl.arange(0, BLOCK_R)
        r_mask = r < num_cand
        rows = tl.load(cand_ptr + tok * vocab_size + r, mask=r_mask, other=0)
        rows = rows.to(tl.int64)
        acc = tl.zeros([BLOCK_R], dtype=tl.float32)
        for h0 in range(0, hidden_size, BLOCK_H):
            cols = h0 + tl.arange(0, BLOCK_H)
            col_mask = cols < hidden_size
            x = tl.load(x_ptr + tok * hidden_size + cols, mask=col_mask, other=0.0)
            w = tl.load(
                w_ptr + rows[:, None] * hidden_size + cols[None, :],
                mask=r_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
        tl.store(
            logits_ptr + tok * vocab_size + rows,
            acc.to(logits_ptr.dtype.element_ty),
            mask=r_mask,
        )


class ScreenedLMHead:
    """lm_head logits for batches that only read the top k: exact on every row
    that can be among them, -inf elsewhere. Not an nn.Module: the BF16 weight
    is the lm_head's own tensor (no copy); only the FP8 copy is new."""

    BLOCK_V = 64
    BLOCK_H = 256
    SELECT_BLOCK = 4096
    EXACT_PROGRAMS = 64

    def __init__(
        self,
        weight: torch.Tensor,
        vocab_size: int,
        max_num_tokens: int,
        max_top_k: int,
        min_top_k: int = 1,
    ):
        self.weight = weight[:vocab_size]
        self.vocab_size, self.hidden_size = self.weight.shape
        self.max_num_tokens = max_num_tokens
        num_blocks = triton.cdiv(self.vocab_size, self.BLOCK_V)
        self.max_top_k = min(max_top_k, num_blocks)
        self.min_top_k = min_top_k
        # One output ulp is at most eps * |logit|.
        self.tie = 2 * torch.finfo(weight.dtype).eps
        device = weight.device
        self.w_fp8 = torch.empty_like(self.weight, dtype=torch.float8_e4m3fn)
        self.scale = torch.empty(self.vocab_size, device=device)
        self.err = torch.empty(self.vocab_size, device=device)
        self.ub = torch.empty(max_num_tokens, self.vocab_size, device=device)
        self.cand = torch.empty_like(self.ub, dtype=torch.int32)
        self.block_max = torch.empty(max_num_tokens, num_blocks, device=device)
        # Set by the sample() hook for the duration of one target sample call.
        self.top_k: int | None = None
        self.num_cand: torch.Tensor | None = None
        self.check: ScreenCheck | None = None
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the FP8 copy (one scale per row) and its error bounds."""
        u = 2.0**-24
        # fp32 accumulation error over hidden_size terms, relative to
        # sum_i |x_i w_i| <= ||x|| ||w||.
        gamma = self.hidden_size * u / (1 - self.hidden_size * u)
        for start in range(0, self.vocab_size, _QUANT_CHUNK):
            rows = slice(start, start + _QUANT_CHUNK)
            w32 = self.weight[rows].float()
            amax = w32.abs().amax(dim=1)
            scale = torch.where(amax > 0, amax / _FP8_MAX, torch.ones_like(amax))
            self.w_fp8[rows] = (w32 / scale[:, None]).clamp_(-_FP8_MAX, _FP8_MAX).to(
                torch.float8_e4m3fn
            )
            self.scale[rows] = scale
            w = self.weight[rows].double()
            deq = self.w_fp8[rows].double() * scale.double()[:, None]
            err = (
                (w - deq).norm(dim=1)
                + gamma * w.norm(dim=1)
                + (gamma + 2 * u) * deq.norm(dim=1)
            )
            # Slack for the fp32 norm of x and the fp32 bound arithmetic.
            self.err[rows] = err * (1 + 4 * gamma)

    def required_top_k(self, sampler, idx_mapping_np: np.ndarray, num_tokens: int):
        """How many top logits sampling this batch reads (1 when all greedy),
        or None when it needs more than the screen certifies."""
        states = sampler.sampling_states
        if (
            not 0 < num_tokens <= self.max_num_tokens
            or np.any(sampler.uses_logits_processors[idx_mapping_np])
            or sampler.get_logprobs_dims(idx_mapping_np) is not None
        ):
            return None
        sampled = idx_mapping_np[states.temperature.np[idx_mapping_np] != 0.0]
        if sampled.size == 0:
            return self.min_top_k
        # Without top-k, top-p reads the whole distribution; min-p is applied
        # before top-k.
        top_k = int(states.top_k.np[sampled].max())
        if top_k > self.max_top_k or np.any(states.min_p.np[sampled] != 0.0):
            return None
        return max(top_k, self.min_top_k)

    def __call__(self, x: torch.Tensor, top_k: int = 1) -> torch.Tensor:
        num_tokens = x.shape[0]
        vocab_size = self.vocab_size
        assert num_tokens <= self.max_num_tokens
        x = x.contiguous()
        ub = self.ub[:num_tokens]
        cand = self.cand[:num_tokens]
        block_max = self.block_max[:num_tokens]
        # A single token block, so the FP8 weights are read once.
        block_n = max(16, triton.next_power_of_2(num_tokens))
        _screen_kernel[(triton.cdiv(vocab_size, self.BLOCK_V),)](
            x,
            self.w_fp8,
            self.scale,
            self.err,
            ub,
            block_max,
            num_tokens,
            vocab_size,
            self.hidden_size,
            BLOCK_N=block_n,
            BLOCK_V=self.BLOCK_V,
            BLOCK_H=self.BLOCK_H,
        )
        t = block_max.topk(top_k, dim=-1).values[:, -1].contiguous()
        logits = torch.empty(num_tokens, vocab_size, device=x.device, dtype=x.dtype)
        num_cand = torch.zeros(num_tokens, device=x.device, dtype=torch.int32)
        _select_kernel[(num_tokens, triton.cdiv(vocab_size, self.SELECT_BLOCK))](
            ub,
            t,
            logits,
            cand,
            num_cand,
            vocab_size,
            TIE=self.tie,
            BLOCK=self.SELECT_BLOCK,
        )
        _exact_kernel[(num_tokens, self.EXACT_PROGRAMS)](
            x,
            self.weight,
            cand,
            num_cand,
            logits,
            vocab_size,
            self.hidden_size,
            BLOCK_R=4,
            BLOCK_H=self.BLOCK_H,
        )
        self.num_cand = num_cand
        return logits

    @classmethod
    def build(cls, lm_head, logits_processor, vocab_size: int, settings, vllm_config):
        """Check the head and build the screen right after the weights loaded,
        so the FP8 copy is part of the model memory that vLLM profiles."""
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
            VocabParallelEmbedding,
        )

        reason = None
        if not isinstance(lm_head, VocabParallelEmbedding) or not isinstance(
            lm_head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        ):
            reason = "an unquantized lm_head"
        elif lm_head.tp_size > 1 or getattr(lm_head, "bias", None) is not None:
            reason = "an lm_head without TP sharding or bias"
        elif lm_head.weight.dtype not in (torch.bfloat16, torch.float16):
            reason = "a BF16 or FP16 lm_head"
        elif (
            logits_processor.logits_as_input
            or logits_processor.soft_cap is not None
            or logits_processor.scale != 1.0
            or logits_processor.head_dtype not in (None, lm_head.weight.dtype)
        ):
            reason = "logits that are the plain lm_head projection"
        if reason is not None:
            raise ValueError(f"{SCREENED_ENV}=1 requires {reason}.")
        max_num_tokens, max_top_k = settings
        spec = vllm_config.speculative_config
        # The "fly" verifier reads the entropy of the target's top
        # fly_entropy_top_k logits, also for greedy requests.
        min_top_k = (
            spec.fly_entropy_top_k
            if spec is not None and spec.rejection_sample_method == "fly"
            else 1
        )
        start = time.perf_counter()
        head = cls(lm_head.weight.data, vocab_size, max_num_tokens, max_top_k, min_top_k)
        # Compile every kernel specialization now, not on the first request.
        weight = head.weight
        for num_tokens in range(1, head.max_num_tokens + 1):
            head(torch.randn_like(weight[:num_tokens]), head.min_top_k)
        torch.cuda.synchronize()
        if os.environ.get(SCREEN_CHECK_ENV, "0") not in ("", "0"):
            head.check = ScreenCheck(int(os.environ[SCREEN_CHECK_ENV]))
        _install_hooks()
        logger.info(
            "[screened-head] lm_head %d x %d: FP8 screen ready in %.1f s "
            "(FP8 copy %.3f GB, buffers %.1f MB, BF16 head kept for the exact "
            "rows); screens greedy and top_k <= %d batches of <= %d rows%s.",
            head.vocab_size,
            head.hidden_size,
            time.perf_counter() - start,
            head.w_fp8.numel() / 1e9,
            (head.ub.numel() * 8 + head.block_max.numel() * 4) / 1e6,
            head.max_top_k,
            head.max_num_tokens,
            f", min top_k {min_top_k} (fly verifier)" if min_top_k > 1 else "",
        )
        return head


class ScreenCheck:
    """QWEN4EXP_LM_HEAD_SCREEN_CHECK=N: also compute the full BF16 logits for
    every screened batch, count disagreements on device, log every N calls.
    The served logits stay the screened ones. Debug only (reads the head twice,
    one host sync per log line)."""

    def __init__(self, every: int):
        self.every = max(every, 1)
        self.calls = 0
        # rows, argmax differs, chosen row strictly below the full max,
        # true top-k rows screened out, kept logits that differ, kept rows
        self.counts = None

    def update(self, screened, full, top_k: int, num_cand) -> None:
        full = full[:, : screened.shape[1]]
        s_arg = screened.argmax(dim=-1, keepdim=True)
        full_max = full.max(dim=-1, keepdim=True).values
        top_idx = full.topk(top_k, dim=-1).indices
        kept = screened.isfinite()
        stats = torch.stack(
            [
                torch.full((), screened.shape[0], device=screened.device),
                (s_arg != full.argmax(dim=-1, keepdim=True)).sum(),
                (full.gather(-1, s_arg) < full_max).sum(),
                screened.gather(-1, top_idx).isneginf().sum(),
                ((screened != full) & kept).sum(),
                num_cand.sum(),
            ]
        ).to(torch.int64)
        self.counts = stats if self.counts is None else self.counts + stats
        self.calls += 1
        if self.calls % self.every == 0:
            rows, arg, worse, missing, diff, cand = self.counts.tolist()
            logger.info(
                "[screened-head] check after %d calls: rows=%d argmax_diff=%d "
                "(strictly_lower=%d) topk_screened_out=%d kept_logits_differing="
                "%d of %d (avg %.1f exact rows per token)",
                self.calls,
                rows,
                arg,
                worse,
                missing,
                diff,
                cand,
                cand / max(rows, 1),
            )


def find_screened_head(model) -> ScreenedLMHead | None:
    get_language_model = getattr(model, "get_language_model", None)
    language_model = get_language_model() if get_language_model else model
    return getattr(language_model, "screened_head", None)


@functools.cache
def _install_hooks() -> None:
    """The two runner hunks of PR #60566, applied at load as wrappers so that
    this overlay does not ship model_runner.py / sampler.py copies (the
    qwen4exp-spec-step overlay already mounts its own model_runner.py)."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    def add_request(self, req_idx, sampling_params) -> None:
        # Sampler.add_request (sampler.py:96-108) with PR #60566's split: like
        # needs_logits_processing, minus temperature, top-k, top-p, min-p.
        if not hasattr(self, "uses_logits_processors"):
            self.uses_logits_processors = np.zeros_like(self.needs_logits_processing)
        needs_processing = self.sampling_states.add_request(req_idx, sampling_params)
        uses_processors = self.thinking_budget_state.add_request(
            req_idx, sampling_params
        )
        for processor in self.logits_processors:
            uses_processors |= processor.add_request(req_idx, sampling_params)
        self.uses_logits_processors[req_idx] = uses_processors
        self.needs_logits_processing[req_idx] = needs_processing or uses_processors
        self.logprob_token_ids_state.add_request(req_idx, sampling_params)
        if self.trace_replay_state is not None:
            self.trace_replay_state.add_request(req_idx, sampling_params)

    Sampler.add_request = add_request

    orig_sample = GPUModelRunner.sample

    @functools.wraps(orig_sample)
    def sample(self, hidden_states, input_batch, grammar_output):
        # GPUModelRunner.sample (model_runner.py:1633-1656) computes the target
        # logits with self.model.compute_logits; the model's compute_logits uses
        # the screen while head.top_k is set.
        head = find_screened_head(self.model)
        sampler = self.sampler
        if (
            head is None
            or grammar_output is not None
            or self.batch_sharder is not None
            or type(sampler) is not Sampler
            or sampler.compute_nans
            or sampler.trace_replay_state is not None
            or sampler.return_sampling_mask
            or not hasattr(sampler, "uses_logits_processors")
        ):
            return orig_sample(self, hidden_states, input_batch, grammar_output)
        head.top_k = head.required_top_k(
            sampler, input_batch.idx_mapping_np, input_batch.logits_indices.shape[0]
        )
        try:
            return orig_sample(self, hidden_states, input_batch, grammar_output)
        finally:
            head.top_k = None

    GPUModelRunner.sample = sample
