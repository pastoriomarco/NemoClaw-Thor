#!/bin/bash
# BranchSpec: Multi-candidate speculative decoding for DFlash on Thor.
#
# Standard spec decode accepts draft tokens that exactly match the target's top-1.
# BranchSpec stores the drafter's top-K candidates at each position and accepts
# if the target's choice is anywhere in the top-K set.
#
# Patches 4 files: metadata.py, eagle.py, gpu_model_runner.py, rejection_sampler.py

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
EAGLE_PY="$SITE_PACKAGES/vllm/v1/spec_decode/eagle.py"
REJECTION_PY="$SITE_PACKAGES/vllm/v1/sample/rejection_sampler.py"
METADATA_PY="$SITE_PACKAGES/vllm/v1/spec_decode/metadata.py"
RUNNER_PY="$SITE_PACKAGES/vllm/v1/worker/gpu_model_runner.py"

BRANCH_K="${BRANCH_K:-3}"
echo "Applying BranchSpec multi-candidate patch (K=$BRANCH_K)..."

# Patch 1: SpecDecodeMetadata — add draft_topk_ids field (after logits_indices)
python3 - "$METADATA_PY" <<'PYEOF1'
import sys
path = sys.argv[1]
with open(path, "r") as f: c = f.read()
old = "    logits_indices: torch.Tensor"
new = "    logits_indices: torch.Tensor\n    draft_topk_ids: torch.Tensor | None = None"
if old in c and "draft_topk_ids" not in c:
    c = c.replace(old, new, 1)
    with open(path, "w") as f: f.write(c)
    print("  [1/4] Added draft_topk_ids to SpecDecodeMetadata")
else:
    print("  [1/4] SKIP")
PYEOF1

# Patch 2: _greedy_sample — compute and store top-K alongside greedy
python3 - "$EAGLE_PY" "$BRANCH_K" <<'PYEOF2'
import sys
path, K = sys.argv[1], int(sys.argv[2])
with open(path, "r") as f: c = f.read()
old = '''    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy-sample draft tokens from hidden states."""
        if self.use_local_argmax_reduction:
            return self.model.get_top_tokens(hidden_states)'''
new = f'''    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy-sample draft tokens from hidden states."""
        # BranchSpec: accumulate top-K across sequential MTP calls
        logits = self.model.compute_logits(hidden_states)
        _topk = torch.topk(logits, k=min({K}, logits.shape[-1]), dim=-1).indices
        if not hasattr(self, '_branchspec_topk_parts'):
            self._branchspec_topk_parts = []
        self._branchspec_topk_parts.append(_topk)
        self._branchspec_topk = torch.cat(self._branchspec_topk_parts, dim=0)
        if self.use_local_argmax_reduction:
            return self.model.get_top_tokens(hidden_states)'''
if old in c:
    c = c.replace(old, new, 1)
    print(f"  [2a] Patched _greedy_sample to accumulate top-{K}")
else:
    print("  [2a] SKIP")

# Also add reset at start of propose()
old_propose = '''    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,'''
new_propose = '''    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,'''
# We patch by adding a reset line right after the propose body starts
old_batch = '        batch_size = common_attn_metadata.batch_size()'
new_batch = '        self._branchspec_topk_parts = []  # BranchSpec: reset accumulator\n        batch_size = common_attn_metadata.batch_size()'
if old_batch in c and '_branchspec_topk_parts = []' not in c:
    c = c.replace(old_batch, new_batch, 1)
    print("  [2b] Added topk accumulator reset in propose()")

with open(path, "w") as f: f.write(c)
PYEOF2

# Patch 3: gpu_model_runner — thread top-K from proposer to metadata
python3 - "$RUNNER_PY" <<'PYEOF3'
import sys
path = sys.argv[1]
with open(path, "r") as f: c = f.read()
old = """        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )"""
new = """        _topk = None
        if hasattr(self, 'proposer') and hasattr(self.proposer, '_branchspec_topk'):
            _raw = self.proposer._branchspec_topk
            if _raw is not None and _raw.shape[0] >= draft_token_ids.shape[0]:
                _topk = _raw[:draft_token_ids.shape[0]].to(dtype=torch.int32)
        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
            draft_topk_ids=_topk,
        )"""
if old in c:
    c = c.replace(old, new, 1)
    with open(path, "w") as f: f.write(c)
    print("  [3/4] Threaded top-K to SpecDecodeMetadata")
else:
    print("  [3/4] SKIP")
PYEOF3

# Patch 4: rejection_sampler — full call chain modification
python3 - "$REJECTION_PY" "$BRANCH_K" <<'PYEOF4'
import sys
path, K = sys.argv[1], int(sys.argv[2])
with open(path, "r") as f: c = f.read()
ok = True

# 4a: Add helper function
func = '''
def _branchspec_greedy_topk(output_token_ids, cu_num_draft_tokens, draft_topk_ids,
                            target_argmax, bonus_token_ids, is_greedy, max_spec_len, K):
    """BranchSpec: accept if target token is in drafter top-K."""
    batch_size = output_token_ids.shape[0]
    for req_idx in range(batch_size):
        if is_greedy is not None and not is_greedy[req_idx]:
            continue
        start = 0 if req_idx == 0 else cu_num_draft_tokens[req_idx - 1].item()
        end = cu_num_draft_tokens[req_idx].item()
        rejected = False
        for pos in range(end - start):
            target_id = target_argmax[start + pos].item()
            output_token_ids[req_idx, pos] = target_id
            if not rejected:
                if target_id not in draft_topk_ids[start + pos].tolist():
                    rejected = True
        if not rejected:
            output_token_ids[req_idx, end - start] = bonus_token_ids[req_idx].item()

'''
marker = "class RejectionSampler(nn.Module):"
if marker in c and "def _branchspec_greedy_topk" not in c:
    c = c.replace(marker, func + marker, 1)
    print("  [4a] Added _branchspec_greedy_topk function")
else:
    print("  [4a] SKIP"); ok = False

# 4b: Add draft_topk_ids param to rejection_sample
old_sig = "    sampling_metadata: SamplingMetadata,\n) -> torch.Tensor:"
new_sig = "    sampling_metadata: SamplingMetadata,\n    draft_topk_ids: torch.Tensor | None = None,\n) -> torch.Tensor:"
if old_sig in c and "draft_topk_ids: torch.Tensor" not in c:
    c = c.replace(old_sig, new_sig, 1)
    print("  [4b] Added draft_topk_ids param to rejection_sample")
else:
    print("  [4b] SKIP")

# 4c: Replace greedy rejection with top-K check
old_check = """    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids"""

new_check = f"""    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_logits.argmax(dim=-1)
        if draft_topk_ids is not None:
            _branchspec_greedy_topk(
                output_token_ids, cu_num_draft_tokens, draft_topk_ids,
                target_argmax, bonus_token_ids, is_greedy, max_spec_len, {K},
            )
        else:
            rejection_greedy_sample_kernel[(batch_size,)](
                output_token_ids, cu_num_draft_tokens, draft_token_ids,
                target_argmax, bonus_token_ids, is_greedy, max_spec_len,
            )
        if sampling_metadata.all_greedy:
            return output_token_ids"""

if old_check in c:
    c = c.replace(old_check, new_check, 1)
    print(f"  [4c] Patched greedy check for top-{K}")
else:
    print("  [4c] SKIP")

# 4d: Thread topk in RejectionSampler.forward call to rejection_sample
old_fwd = """        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
        )"""
new_fwd = """        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            draft_topk_ids=getattr(metadata, 'draft_topk_ids', None),
        )"""
if old_fwd in c:
    c = c.replace(old_fwd, new_fwd, 1)
    print("  [4d] Threaded topk through forward→rejection_sample")
else:
    print("  [4d] SKIP")

with open(path, "w") as f: f.write(c)
PYEOF4

echo "  BranchSpec patch complete."
