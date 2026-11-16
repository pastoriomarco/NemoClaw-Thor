"""[screened-head] QWEN4EXP_LM_HEAD_SCREEN_STATS: per-reason counts of screened
and full-head target sample() calls, through the real hooks (stubs stand in for
the vLLM 0.33 Sampler / GPUModelRunner attributes they read).

Checks: every fallback reason is reported as the first disqualifying one, the
logits-processor reason names the specific processor (pipeline order wins),
screened calls are counted by top_k, rows come from logits_indices, the log
line appears at most once per interval, the switch validates its value, and
with the switch off nothing is recorded.

Run: python -B tests/test_screen_stats.py   (seconds; needs torch, numpy, triton)
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _screened_common import LOG, make_head_weight, sh, stub  # noqa: E402

V = 4096  # vocab of the small head and of the sampler states


class UVA:
    def __init__(self, a): self.np = np.array(a)


class SP:
    """SamplingParams fields the hooks and the fake states read."""
    def __init__(self, t=1.0, k=20, min_p=0.0, logprobs=False, thinking=None,
                 allowed_token_ids=None, logit_bias=None, min_tokens=0,
                 repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0,
                 bad_words=None, custom=False):
        self.__dict__.update(locals()); del self.__dict__["self"]


class States:
    def __init__(self, n):
        self.temperature, self.top_k, self.min_p = UVA([0.0] * n), UVA([V] * n), UVA([0.0] * n)
        self.vocab_size = V
    def add_request(self, i, sp):
        self.temperature.np[i], self.top_k.np[i], self.min_p.np[i] = sp.t, sp.k, sp.min_p
        return sp.t != 0.0


# Named like the vLLM classes: the reason is taken from the class name.
class LogitBiasState:
    def add_request(self, i, sp):
        return bool(sp.allowed_token_ids or sp.logit_bias or sp.min_tokens)

class PenaltiesState:
    def add_request(self, i, sp):
        return (sp.repetition_penalty != 1.0 or sp.frequency_penalty != 0.0
                or sp.presence_penalty != 0.0)

class BadWordsState:
    def add_request(self, i, sp): return bool(sp.bad_words)

class MyProcessor:
    def add_request(self, i, sp): return sp.custom

class ThinkingBudgetState:
    def add_request(self, i, sp): return sp.thinking is not None

class LogprobState:
    def add_request(self, i, sp): pass


class Sampler:
    def __init__(self, n=8):
        self.sampling_states = States(n)
        self.thinking_budget_state = ThinkingBudgetState()
        self.logits_processors = [LogitBiasState(), PenaltiesState(), BadWordsState(), MyProcessor()]
        self.logprob_token_ids_state = LogprobState()
        self.trace_replay_state = None
        self.needs_logits_processing = np.zeros(n, dtype=bool)
        self.compute_nans = False
        self.return_sampling_mask = False
        self.logprobs = np.zeros(n, dtype=bool)
    def get_logprobs_dims(self, idx): return (1, 0) if self.logprobs[idx].any() else None


calls = []
class GPUModelRunner:
    def sample(self, hidden_states, input_batch, grammar_output):
        calls.append(head.top_k)
        return "out"

stub("vllm.v1"); stub("vllm.v1.worker"); stub("vllm.v1.worker.gpu")
stub("vllm.v1.worker.gpu.model_runner", GPUModelRunner=GPUModelRunner)
stub("vllm.v1.worker.gpu.sample"); stub("vllm.v1.worker.gpu.sample.sampler", Sampler=Sampler)
sh._install_hooks()

head = sh.ScreenedLMHead(make_head_weight(V, 256), V, 10, 32)
class LM: screened_head = head
class Model:
    def get_language_model(self): return LM()

class Batch:
    def __init__(self, idx, n):
        self.idx_mapping_np = np.array(idx)
        self.logits_indices = torch.zeros(n, dtype=torch.long)

def runner(sampler):
    r = GPUModelRunner(); r.model = Model(); r.sampler = sampler; r.batch_sharder = None
    return r

# ------------------------------------------------ switch off: nothing recorded
assert sh._STATS is None
s0 = Sampler()
s0.add_request(0, SP(presence_penalty=1.5))
assert not hasattr(s0, "screen_processor_reason")
assert s0.uses_logits_processors[0]
runner(s0).sample(None, Batch([0], 5), None)
assert calls[-1] is None
# Without the per-request codes the reason stays generic.
assert head.screen_decision(s0, np.array([0]), 5, detail=True) == (None, "logits_processor")

# ------------------------------------------------ switch on
clock = [1000.0]
sh.ScreenStats.clock = staticmethod(lambda: clock[0])
stats = sh._STATS = sh.ScreenStats(60)
s = Sampler()
reqs = [
    SP(t=0.0, k=V),                         # 0 greedy
    SP(t=1.0, k=20),                        # 1 Qwen sampling
    SP(presence_penalty=1.5),               # 2
    SP(repetition_penalty=1.1, logit_bias={5: -1.0}),   # 3 logit_bias first (pipeline order)
    SP(thinking=100, min_tokens=4),         # 4 min_tokens before thinking_budget
    SP(t=0.7, k=V),                         # 5 no top_k
    SP(t=1.0, k=20, min_p=0.05),            # 6
    SP(t=1.0, k=40),                        # 7 top_k 40 > SCREEN_K 32
]
for i, sp in enumerate(reqs):
    s.add_request(i, sp)
codes = [sh.PROCESSOR_REASONS[c] for c in s.screen_processor_reason]
assert codes == [None, None, "presence_penalty", "logit_bias", "min_tokens", None, None, None], codes
s.add_request(5, SP(t=0.7, k=V, thinking=10)); assert sh.PROCESSOR_REASONS[s.screen_processor_reason[5]] == "thinking_budget"
s.add_request(5, SP(t=1.0, k=20, bad_words=[[1]])); assert sh.PROCESSOR_REASONS[s.screen_processor_reason[5]] == "bad_words"
s.add_request(5, SP(t=1.0, k=20, custom=True)); assert sh.PROCESSOR_REASONS[s.screen_processor_reason[5]] == "custom_processor"
s.add_request(5, SP(t=1.0, k=20, allowed_token_ids=[1, 2])); assert sh.PROCESSOR_REASONS[s.screen_processor_reason[5]] == "allowed_token_ids"
s.add_request(5, SP(t=1.0, k=20, frequency_penalty=0.5)); assert sh.PROCESSOR_REASONS[s.screen_processor_reason[5]] == "frequency_penalty"
s.add_request(5, SP(t=0.7, k=V))
assert s.screen_processor_reason[5] == 0

dec = lambda idx, n=5: head.screen_decision(s, np.array(idx), n, detail=True)
assert dec([0]) == (1, None) and dec([0, 1]) == (20, None)
assert dec([1], 0) == (None, "no_rows") and dec([1], 11) == (None, "rows_over_max")
assert dec([1, 2]) == (None, "presence_penalty")
assert dec([2, 3]) == (None, "logit_bias")           # first in pipeline order over the batch
assert dec([4, 2]) == (None, "min_tokens")
assert dec([5]) == (None, "no_top_k") and dec([7]) == (None, "top_k_over_max")
assert dec([6]) == (None, "min_p")
assert dec([2, 5], 11) == (None, "rows_over_max")     # rows are checked first
s.logprobs[1] = True
assert dec([1, 5]) == (None, "logprobs") and dec([1, 2]) == (None, "presence_penalty")
s.logprobs[1] = False
# required_top_k is unchanged: same top_k, None on any reason.
assert head.required_top_k(s, np.array([0, 1]), 5) == 20 and head.required_top_k(s, np.array([6]), 5) is None

r = runner(s)
lines0 = len(LOG.lines)
for idx, n in [([0], 5), ([0], 5), ([0, 1], 10), ([2], 5), ([5], 1), ([6], 5), ([7], 5), ([3, 4], 2)]:
    assert r.sample(None, Batch(idx, n), None) == "out"
    assert head.top_k is None
r.sample(None, Batch([0], 5), object())           # structured output
s.compute_nans = True; r.sample(None, Batch([0], 5), None); s.compute_nans = False
r.batch_sharder = object(); r.sample(None, Batch([0], 5), None); r.batch_sharder = None
assert calls[-11:] == [1, 1, 20, None, None, None, None, None, None, None, None], calls[-11:]
assert len(LOG.lines) == lines0, "no line before the interval"
assert stats.calls == 11 and stats.rows == 5 + 5 + 10 + 5 + 1 + 5 + 5 + 2 + 15
assert stats.screened == {1: [2, 10], 20: [1, 10]}
assert stats.full == {
    "presence_penalty": [1, 5], "no_top_k": [1, 1], "min_p": [1, 5], "top_k_over_max": [1, 5],
    "logit_bias": [1, 2], "structured_output": [1, 5], "compute_nans": [1, 5], "batch_sharder": [1, 5],
}, stats.full
clock[0] += 60.0
r.sample(None, Batch([0], 5), None)
assert len(LOG.lines) == lines0 + 1
line = LOG.lines[-1]
assert line == (
    "[screened-head] stats 60s calls/rows: all=12/58 screened=4/25 full=8/33 "
    "| screened by top_k: k1=3/15 k20=1/10 "
    "| full by reason: batch_sharder=1/5 compute_nans=1/5 min_p=1/5 presence_penalty=1/5 "
    "structured_output=1/5 top_k_over_max=1/5 logit_bias=1/2 no_top_k=1/1"
), line
clock[0] += 59.0
r.sample(None, Batch([0], 5), None)
assert len(LOG.lines) == lines0 + 1, "at most one line per interval"
clock[0] += 1.0
r.sample(None, Batch([0], 5), None)
assert len(LOG.lines) == lines0 + 2 and "all=14/68" in LOG.lines[-1]

# Empty tables print "-".
assert sh.ScreenStats(5).line(clock[0]).endswith("| screened by top_k: - | full by reason: -")
for bad in (0.0, -1.0):
    try: sh.ScreenStats(bad); raise AssertionError
    except ValueError as e: print("OK raises:", e)
sh._STATS = None
print("ALL OK")
