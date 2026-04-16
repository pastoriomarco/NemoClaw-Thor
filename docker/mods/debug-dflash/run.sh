#!/bin/bash

# Debug DFlash: comprehensive runtime instrumentation for diagnosing 0% acceptance.
# Hooks into DFlashProposer.propose() and DFlashQwen3Model.precompute_and_store_context_kv()
# to log tensor shapes, values, and the actual draft vs target comparison.

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
DFLASH_PY="$SITE_PACKAGES/vllm/v1/spec_decode/dflash.py"
EAGLE_PY="$SITE_PACKAGES/vllm/v1/spec_decode/eagle.py"
QWEN3_DFLASH_PY="$SITE_PACKAGES/vllm/model_executor/models/qwen3_dflash.py"

echo "Applying DFlash debug instrumentation..."

# --- Instrument set_inputs_first_pass in dflash.py for block_size + slots ---
python3 - "$DFLASH_PY" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

# Add logging before the return statement (after new_cad is created)
old = "        return num_query_total, token_indices_to_sample, new_cad"
new = """        # DEBUG: log block_size, slot mapping, and seq_lens
        if not hasattr(self, '_bs_log_count'):
            self._bs_log_count = 0
        self._bs_log_count += 1
        if self._bs_log_count <= 3:
            import logging
            _bslog = logging.getLogger("dflash.debug")
            _bslog.setLevel(logging.INFO)
            if not _bslog.handlers:
                _bslog.addHandler(logging.StreamHandler())
            _ctx_n = min(8, num_context)
            _q_n = min(9, num_query_total)
            _bslog.info(
                f"SLOTS #{self._bs_log_count}: block_size={self.block_size}, "
                f"num_context={num_context}, num_query={num_query_total}, "
                f"cad.seq_lens={cad.seq_lens.tolist()[:2]}, "
                f"new_seq_lens={new_cad.seq_lens.tolist()[:2]}, "
                f"ctx_slots[:8]={self._context_slot_mapping_buffer[:_ctx_n].tolist()}, "
                f"query_slots[:9]={query_slot_mapping[:_q_n].tolist()}, "
                f"ctx_pos[:8]={self._context_positions_buffer[:_ctx_n].tolist()}, "
                f"query_pos[:9]={self.positions[:_q_n].tolist()}"
            )
        return num_query_total, token_indices_to_sample, new_cad"""
if old in content:
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print("  [0/4] Instrumented set_inputs_first_pass for block_size/slots")
else:
    print("  [0/4] WARNING: set_inputs_first_pass slot pattern not found")

PYEOF

# --- Instrument propose() in eagle.py ---
# Hook into the propose() method where combine_hidden_states is called.
# This runs BEFORE the drafter forward pass.
python3 - "$EAGLE_PY" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

# Instrument after combine_hidden_states and before set_inputs_first_pass
old = """            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size"""

new = """            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size
            # DEBUG: Log combined hidden states
            if not hasattr(self, '_dbg_propose_count'):
                self._dbg_propose_count = 0
            self._dbg_propose_count += 1
            if self._dbg_propose_count <= 3:
                import logging
                _dbg = logging.getLogger("dflash.debug")
                _dbg.setLevel(logging.INFO)
                if not _dbg.handlers:
                    _dbg.addHandler(logging.StreamHandler())
                _hs = target_hidden_states.float()
                _dbg.info(
                    f"PROPOSE #{self._dbg_propose_count}: "
                    f"combined_hidden shape={target_hidden_states.shape}, "
                    f"dtype={target_hidden_states.dtype}, "
                    f"mean={_hs.mean().item():.6f}, "
                    f"std={_hs.std().item():.6f}, "
                    f"min={_hs.min().item():.6f}, "
                    f"max={_hs.max().item():.6f}, "
                    f"has_nan={_hs.isnan().any().item()}, "
                    f"has_inf={_hs.isinf().any().item()}"
                )"""

if old in content:
    content = content.replace(old, new, 1)
    print("  [1/4] Instrumented propose() after combine_hidden_states")
else:
    print("  [1/4] WARNING: propose() pattern not found")

# Instrument the parallel_drafting early exit to log draft tokens
old2 = """        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids = self._greedy_sample(sample_hidden_states)
            return draft_token_ids.view(-1, self.num_speculative_tokens)"""

new2 = """        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids = self._greedy_sample(sample_hidden_states)
            # DEBUG: Log draft output
            if hasattr(self, '_dbg_propose_count') and self._dbg_propose_count <= 3:
                import logging
                _dbg = logging.getLogger("dflash.debug")
                _shs = sample_hidden_states.float()
                _dbg.info(
                    f"DRAFT #{self._dbg_propose_count}: "
                    f"sample_hidden shape={sample_hidden_states.shape}, "
                    f"mean={_shs.mean().item():.6f}, "
                    f"std={_shs.std().item():.6f}, "
                    f"draft_tokens={draft_token_ids.view(-1).tolist()[:24]}, "
                    f"parallel_drafting={self.parallel_drafting}"
                )
            return draft_token_ids.view(-1, self.num_speculative_tokens)"""

if old2 in content:
    content = content.replace(old2, new2, 1)
    print("  [2/4] Instrumented parallel_drafting early exit")
else:
    print("  [2/4] WARNING: parallel_drafting early exit pattern not found")

with open(path, "w") as f:
    f.write(content)

PYEOF

# --- Instrument precompute_and_store_context_kv ---
python3 - "$QWEN3_DFLASH_PY" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

# Add debug logging to precompute_and_store_context_kv
old = """        num_ctx = context_states.shape[0]
        L = self._num_attn_layers
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads"""

new = """        num_ctx = context_states.shape[0]
        L = self._num_attn_layers
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads
        # DEBUG: Log context states entering precompute (skip dummy runs)
        if context_slot_mapping is not None:
            if not hasattr(self, '_dbg_precompute_count'):
                self._dbg_precompute_count = 0
            self._dbg_precompute_count += 1
        if context_slot_mapping is not None and hasattr(self, '_dbg_precompute_count') and self._dbg_precompute_count <= 3:
            import logging
            _dbg = logging.getLogger("dflash.debug")
            _dbg.setLevel(logging.INFO)
            if not _dbg.handlers:
                _dbg.addHandler(logging.StreamHandler())
            _cs = context_states.float()
            _dbg.info(
                f"PRECOMPUTE #{self._dbg_precompute_count}: "
                f"context shape={context_states.shape}, dtype={context_states.dtype}, "
                f"positions shape={context_positions.shape}, "
                f"slot_mapping={'None' if context_slot_mapping is None else context_slot_mapping.shape}, "
                f"num_layers={L}, kv_size={kv}, head_dim={hd}, num_kv_heads={nkv}, "
                f"context mean={_cs.mean().item():.6f}, std={_cs.std().item():.6f}, "
                f"positions[:8]={context_positions[:8].tolist()}"
            )"""

if old in content:
    content = content.replace(old, new, 1)
    print("  [3/4] Instrumented precompute_and_store_context_kv")
else:
    print("  [3/4] WARNING: precompute pattern not found")

with open(path, "w") as f:
    f.write(content)

PYEOF

# --- Instrument combine_hidden_states to log pre-fc values ---
python3 - "$QWEN3_DFLASH_PY" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

old = """    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        result = self.model.fc(hidden_states)
        if needs_squeeze:
            result = result.squeeze(0)
        return result"""

new = """    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        # DEBUG: Log pre-fc and post-fc values
        if not hasattr(self, '_dbg_combine_count'):
            self._dbg_combine_count = 0
        self._dbg_combine_count += 1
        if self._dbg_combine_count <= 3:
            import logging
            _dbg = logging.getLogger("dflash.debug")
            _dbg.setLevel(logging.INFO)
            if not _dbg.handlers:
                _dbg.addHandler(logging.StreamHandler())
            _hs = hidden_states.float()
            _dbg.info(
                f"COMBINE #{self._dbg_combine_count}: "
                f"pre-fc shape={hidden_states.shape}, dtype={hidden_states.dtype}, "
                f"mean={_hs.mean().item():.6f}, std={_hs.std().item():.6f}, "
                f"min={_hs.min().item():.6f}, max={_hs.max().item():.6f}"
            )
        result = self.model.fc(hidden_states)
        if self._dbg_combine_count <= 3:
            _r = result.float()
            _dbg.info(
                f"COMBINE #{self._dbg_combine_count}: "
                f"post-fc shape={result.shape}, "
                f"mean={_r.mean().item():.6f}, std={_r.std().item():.6f}, "
                f"min={_r.min().item():.6f}, max={_r.max().item():.6f}"
            )
        if needs_squeeze:
            result = result.squeeze(0)
        return result"""

if old in content:
    content = content.replace(old, new, 1)
    print("  [4/4] Instrumented combine_hidden_states")
else:
    print("  [4/4] WARNING: combine_hidden_states pattern not found")

with open(path, "w") as f:
    f.write(content)

PYEOF

# --- Instrument forward() to log per-layer outputs and input_ids ---
python3 - "$QWEN3_DFLASH_PY" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

old = """    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states"""

new = """    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds

        # DEBUG: Log drafter forward inputs and per-layer outputs
        # Skip dummy runs (all-zero input_ids = warmup/profiling)
        _is_real = input_ids.any().item()
        if not hasattr(self, '_dbg_fwd_count'):
            self._dbg_fwd_count = 0
        if _is_real:
            self._dbg_fwd_count += 1
        _do_log = _is_real and self._dbg_fwd_count <= 3
        if _do_log:
            import logging
            _dbg = logging.getLogger("dflash.debug")
            _dbg.setLevel(logging.INFO)
            if not _dbg.handlers:
                _dbg.addHandler(logging.StreamHandler())
            _dbg.info(
                f"FWD #{self._dbg_fwd_count}: input_ids={input_ids.tolist()[:12]}, "
                f"positions={positions.tolist()[:12]}, "
                f"embed mean={hidden_states.float().mean().item():.6f}, "
                f"embed std={hidden_states.float().std().item():.6f}, "
                f"has_nan={hidden_states.isnan().any().item()}"
            )

        residual = None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if _do_log:
                _h = hidden_states.float()
                _dbg.info(
                    f"FWD #{self._dbg_fwd_count} L{layer_idx}: "
                    f"mean={_h.mean().item():.6f}, std={_h.std().item():.6f}, "
                    f"has_nan={_h.isnan().any().item()}"
                )
        hidden_states, _ = self.norm(hidden_states, residual)
        if _do_log:
            _h = hidden_states.float()
            _dbg.info(
                f"FWD #{self._dbg_fwd_count} FINAL: "
                f"mean={_h.mean().item():.6f}, std={_h.std().item():.6f}, "
                f"has_nan={_h.isnan().any().item()}"
            )
        return hidden_states"""

if old in content:
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print("  [5/5] Instrumented drafter forward() with per-layer logging")
else:
    print("  [5/5] WARNING: drafter forward pattern not found")

PYEOF

echo "  DFlash debug mod complete."
