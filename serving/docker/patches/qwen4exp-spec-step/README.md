# qwen4exp-spec-step: per-step speculative-decoding overhead (vLLM 0.33.0, V2 runner)

Qwen3.8-Flash-Next ("Qwen4Exp") with the built-in MTP drafter on Jetson AGX Thor
(sm_110), served by the official vLLM 0.33.0 image in `cudagraph_mode=PIECEWISE`
with breakable CUDA graphs. 0.33 turns breakable graphs on by itself for
`Qwen4Exp*`: `config/vllm.py:76-112`, `:842-869`.

This overlay adds three things. All are off unless an env variable is set, and with
none of the variables set the behaviour is stock:

1. **`VLLM_SPEC_STEP_DRAFT_FULL_CG=1`**: capture the MTP draft *decode* steps
   (draft positions 1..K-1) as FULL CUDA graphs even though the target stays
   PIECEWISE. In stock 0.33 they run eagerly in that case.
2. **`VLLM_SPEC_STEP_TIMING=<N>`**: a per-step phase timing log that averages over N
   decode steps. It records CUDA events and host timestamps and never synchronizes.
3. **`VLLM_PLE_DISK_EARLY=1`**: the hooks of the `qwen4exp-ple-disk` early row fetch
   (plan step A2). After sampling and after each draft pass but the last, the
   runner and speculator hand the new token of the next verify batch to
   `ple_early_begin` / `ple_early_token`, which hash it and read its PLE rows
   during drafting. What, why and how to verify: that overlay's README,
   "Early row fetch".

Line numbers below refer to the pristine 0.33.0 files.

## What one speculative step does in stock 0.33 (MTP K=3, 1 sequence, PIECEWISE + breakable)

**Target verify forward** (`v1/worker/gpu/model_runner.py`)
1. `execute_model` (`:1764`): request-state updates, then `gather_batch_req_state` and
   `dispatch_cg_and_sync_dp` (`:1832`), which gives a PIECEWISE descriptor for 4 tokens.
   `prepare_inputs` (`:1862`), `prepare_attn` (block tables, slot mappings),
   `model_state.preprocess_state` (Mamba align pre-copy, `:1870`), then
   `model_state.prepare_attn` (`:1967`, `model_states/mamba_hybrid.py:230`). The last
   one builds metadata for every group: GDN (spec-decode split, CPU-side `.item()` on
   CPU tensors in `v1/attention/backends/gdn_attn.py:344-461`), QSA main (FA2
   builder), and the two QSA side caches (`models/qwen4_exp/common/qsa_cache.py:651`,
   one Triton kernel). No GPU→host sync happens here.
2. `cudagraph_manager.run_pw_graph` (`:2092`) → `BreakableCUDAGraphWrapper._replay`
   (`compilation/breakable_cudagraph.py:415`) replays a list of segments
   (`:213-215`). For qwen4_exp the eager breaks are:
   - 36 × `qwen_gdn_attention_core[_fused_norm_packed]`
     (`model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:2243`, `:2291`): conv1d
     update, CUDA spec-decode recurrence and gated norm, launched from Python on
     every replay.
   - 12 × `Qwen4ExpQSAAttention._run_qsa` (`models/qwen4_exp/nvidia/qsa.py:562`):
     indexer (side-cache writes, compressed-key paged top-k over the whole page
     table), K/V write and the QSA Triton kernel. All of it is eager.
   - PLE: `start_prefetch` (`models/qwen4_exp/common/ngram_embedding.py:738`),
     `_finalize_prefetch` (`:783`) and `_short_conv` (`models/qwen4_exp/nvidia/ple_layer.py:378`).
     With the `qwen4exp-ple-disk` overlay, the row gather runs on the host after a
     D2H copy of this step's ids. Those ids depend on the previous step's sampled and
     draft tokens, so **every step blocks the host on the GPU at the PLE layer**.
   - That is about 51 eager breaks and about 52 graph replays per target forward.
     Everything between two breaks (projections, MoE, HC, norms) is graphed.
3. `sample_tokens` → `sample` (`:2200`): lm_head (248,320 × 2,560) on 4 rows, then the
   rejection sampler (no host sync). `AsyncOutput` (`:2247`) starts the D2H copy on
   the copy stream. Then `postprocess_sampled` (`:2286`).

**Drafting** (`v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py`, `propose` `:164`)
4. `prepare_prefill_inputs` (one Triton kernel, `:226`), then `dispatch_cg_and_sync_dp`
   for the draft prefill (`:258`, PIECEWISE).
5. **Draft pass 0** (draft prefill, 4 tokens) `_prefill` (`:285`). It reuses the
   target's attention metadata. The MTP model runs under its own breakable wrapper
   (`run_pw_graph`, `:425`): 1 QSA eager break and 2 graph segments. Then
   `compute_logits` (full 248K-vocab lm_head) and argmax or Gumbel run **eagerly**
   (`:476`).
6. `prepare_decode_inputs` (one kernel, `:312`), then the decode dispatch (`:334`).
   Because the target mode is PIECEWISE, `init_cudagraph_manager` (`:98-102`) gave the
   draft decode manager `CUDAGraphMode.NONE`, so the descriptor is NONE.
7. `supports_decode_metadata_update` is False. The QSA side builder
   (`QSAMetadataBuilder`, `qsa_cache.py:599`) has no `update_draft_decode_metadata`,
   so `speculator.py:312-327` logs "cannot update draft decode metadata in place".
   That selects `_multi_step_decode` (`:492`). **For each of draft passes 1..K-1**:
   `compute_slot_mappings` + `build_slot_mappings_by_layer` +
   `_build_uniform_attn_metadata` (`spec_decode/speculator.py:572` → `:329`). This is
   a full host-side rebuild: CPU tensors, the FA builder `build`, and the QSA side
   `build` with its Triton kernel, twice (raw and compressed caches). After that comes
   `_generate_draft` (`:634`): **a fully eager MTP forward** (embed, 2 norms, 2 MXFP8
   linears, HC, one QSA decoder layer with indexer top-k, NVFP4 MoE with 512 experts
   and top-10, shared expert, HC mixer), an eager lm_head over 248K vocab plus
   argmax, and `update_draft_inputs` (one kernel).
8. Back in `sample_tokens`: `draft_tokens` go to `req_states`, `set_draft_tokens`, return.

Host↔GPU syncs per step: no `.item()` on GPU tensors in the runner, metadata
builders, rejection sampler or speculator. All of those operate on CPU mirrors. The
**only** per-step blocking sync is the PLE host lookup. `AsyncOutput.get_output`
waits on its copy event later, outside the step.

### Avoidable overhead per step, ranked (estimates, not measured; measure with `VLLM_SPEC_STEP_TIMING`)

Context: without MTP the step was 45 ms, 18 ms of it PLE host time
(`docs/QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md`). With MTP K=3 at 60–80K context it is about
170 ms.

| # | Overhead | Est. cost / step | Fix |
|---|---|---|---|
| 1 | PLE host lookup on the critical path. It syncs the host to the GPU every step, so the CPU no longer runs ahead and every host launch cost after layer 1 becomes visible | 10–20 ms direct, and it multiplies items 2 and 3 | `qwen4exp-ple-disk` overlay, not this one. Ideas: overlap the lookup with the previous step's drafting, or prefetch from draft tokens |
| 2 | K-1 **eager** MTP draft forwards, op by op from Python, about 100+ launches each including the 248K lm_head | about 4–10 ms host per pass, i.e. 8–20 ms at K=3 and 16–40 ms at K=5 | **this overlay: `VLLM_SPEC_STEP_DRAFT_FULL_CG=1`** |
| 3 | About 51 eager breaks plus 52 segment replays in the target forward (GDN spec-decode cores, QSA indexer+attention) | about 8–15 ms host | Needs a target graph that captures GDN/QSA for uniform decode while keeping only PLE as a break. That is a new mechanism (persistent-buffer metadata in a breakable mode). Not done here |
| 4 | Byte cost hidden in "sampling": lm_head is 1.27 GB in BF16 and is read 1 + K times per step | about 6 ms per read on Thor, so about 25 ms at K=3 | Reduced draft vocabulary (`flash-next-fast-v2/patch_mtp_draft_vocab.py`). Bytes, not overhead |
| 5 | Per draft pass, a host rebuild of attention metadata (`_build_uniform_attn_metadata` + slot mappings) | about 0.3–1 ms × (K-1) | Fused draft decode. Blocked by `QSAMetadataBuilder` (see "Not implemented") |
| 6 | Draft pass 0 eager tail (lm_head + argmax) and 2 segment replays | about 0.5–1.5 ms | Not worth it alone |
| 7 | Target metadata build (GDN/QSA/FA builders, CPU only) | about 1–2 ms host | Needed every step |

Items 2, 3, 5 and 6 cost host time. They add to the step time only while the CPU is
not ahead of the GPU, and item 1 guarantees that it is not. The timing log shows
this directly: when a phase's host time is about equal to its GPU time, the phase
is launch-bound.

## Mjolnir patches (GoBrrr-ai/mjolnir, vLLM 0.33 base): verdicts

| Patch | Verdict | Why |
|---|---|---|
| `50885-flashinfer-full-cudagraph.patch` | **Not applicable** | FlashInfer-only policy that grants UNIFORM_BATCH FULL graphs to the target verify batch. This model's attention is QSA Triton, GDN and FA2 metadata, not FlashInfer. The target cannot take a FULL graph here anyway: the PLE disk lookup is a host-side eager break, and inside a FULL capture `eager_break_during_capture` runs inline (`breakable_cudagraph.py:106-109`), so the host gather would be baked into the graph. |
| `49652-draft-decode-capture.patch` | **Already in 0.33 (equivalent)** and irrelevant here | The Dynamic-SD expansion in `CudaGraphManager._init_candidates` (`v1/worker/gpu/cudagraph_utils.py:276-281`) already requires `decode_query_len > num_speculative_tokens`, which is false for the 1-token draft decode manager. Dynamic SD is not used. |
| `thor-fused-draft-decode-fi-native-update.patch` | **Applicable only with adaptation, in a file this overlay may not touch** | The idea (GPU-side `update_draft_decode_metadata` so `_fused_multi_step_decode` runs the K-1 loop as one graph) maps to qwen4_exp. The FA2 main builder already supports it (its update is a no-op without an AOT scheduler, `flash_attn.py:741`, `:1035`). The blocker is `QSAMetadataBuilder` in `models/qwen4_exp/common/qsa_cache.py`. The hunk is described below. Gain over item 2's fix: (K-2) metadata rebuilds and (K-2) replays, about 0.5–2 ms at K=3–5. |
| `thor-fused-draft-t5-plan-refresh.patch` | **Not applicable** | Fixes Mjolnir's host-computed T5 split-KV work plan baked into the fused FULL graph. The stock QSA/indexer kernels here take only persistent GPU buffers and constant host ints for a 1-token decode (grids come from the static page-table width, `ops/qsa_indexer.py:541-559`), so there is no per-cycle host plan to refresh. The same hazard class was checked for item 2's FULL graph; see Risks. |
| `thor-draft-cg-gate-sm110.patch` | **Not applicable as code** (DSpark/dflash speculator) | Its finding still matters: a graphed tiny draft measured 7–16% slower than eager at N=1 on sm_110. Here the draft is a full MoE + QSA layer plus a 248K lm_head, much heavier than their 5-layer draft, so FULL should win. That is why the switch defaults off and the timing log exists to settle it. |

## Files

Paths mirror the package, relative to `vllm/`.

| File | Kind | Hunks (all marked `[spec-step]`) |
|---|---|---|
| `vllm/v1/worker/gpu/spec_step_timing.py` | new | `SpecStepTimer`: `begin_step` / `set_label` / `mark(name)` / `end_step`. A phase is the interval since the previous mark, with host time (`perf_counter`) and GPU time (CUDA events on the current stream). Completed steps are resolved with `Event.query()` and never synchronize. Every N `decode` steps it logs averages. Module singleton `SPEC_STEP_TIMER` is `None` unless `VLLM_SPEC_STEP_TIMING>0`. |
| `vllm/v1/worker/gpu/model_runner.py` | full copy of 0.33 (sha256 `9145e260…0121`) | Import of `SPEC_STEP_TIMER`. `execute_model`: `begin_step` (non-dummy), `set_label("decode")` when the batch is a uniform spec-decode batch (`uniform_tok_count == decode_query_len`), marks `prep` (before the forward) and `target` (after it). `sample_tokens`: marks `gap` (entry), `sample` (after `self.sample`), `post` (before `propose`), `tail`, then `end_step`. Each call site is one `is not None` check. Early PLE fetch: `import os`, the `_PLE_EARLY` switch, and in `sample_tokens` after `postprocess_sampled` (before the `post` mark), with one request and drafts to propose, `ple_early_begin(all_token_ids, num_computed_tokens, last_sampled_tokens, idx_mapping)`. |
| `vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py` | full copy of 0.33 (sha256 `511b337a…e318`) | (a) `init_cudagraph_manager`: with `VLLM_SPEC_STEP_DRAFT_FULL_CG=1`, a non-NONE target mode, and draft attention support ≥ `UNIFORM_SINGLE_TOKEN_DECODE`, the draft decode manager gets `FULL_DECODE_ONLY` instead of `NONE`, with a one-line log. Everything downstream is stock: capture (`capture()` → `decode_cudagraph_manager.capture(_generate_draft)`), dispatch and `_multi_step_decode`'s `run_fullgraph` branch are the path 0.33 already takes whenever the target is FULL_AND_PIECEWISE. (b) Timing marks: `draft_prep`, `draft0_<mode>`, `draft_dec_prep`, `draft<s>_meta`, `draft<s>_<mode>` per decode pass, and `draft_fused_*` on the fused path. (c) Early PLE fetch: the `_PLE_EARLY` switch; `ple_early_token(draft_tokens[:1, 0])` after draft pass 0 when K ≥ 2, and `ple_early_token(draft_tokens[:1, step])` after decode pass `step` for `step < K - 1` (outside the graphs, between replays). |

The draft prefill (pass 0) is unchanged: it stays PIECEWISE and reuses the target's
metadata.

## Docker mounts

Mount all three together: both modified files import the new module. With
`VLLM_PLE_DISK_EARLY=1` the `qwen4exp-ple-disk` mounts are needed too (the hooks
import `vllm.models.qwen4_exp.common.ple_disk` at the first step). The container
package root is `/usr/local/lib/python3.12/dist-packages/vllm`.

```bash
P=${HOME}/workspaces/dev_ws/src/NemoClaw-Thor/serving/docker/patches/qwen4exp-spec-step
V=/usr/local/lib/python3.12/dist-packages/vllm
  -v $P/vllm/v1/worker/gpu/spec_step_timing.py:$V/v1/worker/gpu/spec_step_timing.py:ro \
  -v $P/vllm/v1/worker/gpu/model_runner.py:$V/v1/worker/gpu/model_runner.py:ro \
  -v $P/vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py:$V/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py:ro \
```

Before mounting, check that the image's originals match the 0.33.0 copies these were
made from (sha256 above, e.g. `docker run --rm --entrypoint sha256sum <image> $V/v1/worker/gpu/model_runner.py`).
No other overlay touches these files.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_SPEC_STEP_DRAFT_FULL_CG` | `0` | `1` captures and replays FULL graphs for MTP draft decode passes 1..K-1 while the target stays PIECEWISE. No effect with `cudagraph_mode=NONE` / `--enforce-eager`, or when the target is already FULL for decode. |
| `VLLM_SPEC_STEP_TIMING` | `0` | `N>0` logs one line per N uniform spec-decode steps. Other steps (prefill chunks, mixed) are counted but not averaged. |
| `VLLM_PLE_DISK_EARLY` | `0` | `1` calls the early PLE row fetch hooks (`qwen4exp-ple-disk`). Each call site is one module-level bool check when off. |

vLLM will warn "Unknown vLLM environment variable detected" for these. That is harmless.

## Verifying from the log

- Draft FULL graphs:
  `[spec-step] draft decode: FULL CUDA graphs (target mode PIECEWISE, draft attention UNIFORM_BATCH)`.
  At startup there is also a second progress bar
  `Capturing decode CUDA graphs (FULL)` (one graph, size 1, at max_num_seqs=1) after
  `Capturing prefill CUDA graphs (PIECEWISE)`. The stock line "Attention backend(s)
  QWEN4_EXP_EXP_QSA_STATE cannot update draft decode metadata in place …" still
  appears, which is expected: it selects the per-pass replay path.
- Timing (example shape):
  `[spec-step] 200 decode steps (+3 other): cycle 171.2 ms (5.84 steps/s), step 168.9/169.5 ms host/gpu | prep a/b | target a/b | gap … | sample … | post … | draft_prep … | draft0_PIECEWISE … | draft_dec_prep … | draft1_meta … | draft1_NONE … | draft2_meta … | draft2_NONE … | tail …`
  Each phase is `host_ms/gpu_ms`. GPU time is the event span on the stream, idle gaps
  included. If host time is about equal to GPU time, the phase is host-bound. If
  host time is much smaller than GPU time, the GPU is the bottleneck, or the phase
  waited (for example `target` contains the PLE stall). With the switch on, the
  `draft<s>_NONE` entries become `draft<s>_FULL`. Compare them directly; that is the
  sm_110 question from the Mjolnir gate patch. `cycle` is the host time between step
  entries and includes engine-core work between steps.
- Correctness A/B: same greedy prompt (temperature 0) with the switch off and on.
  The output text must be identical, and the spec-decode acceptance metrics in the
  engine stats should match.

## Risks

- **Silent baked host state in the draft FULL graph.** This was checked in 0.33. Every
  pass is a 1-token, 1-request decode (constant `num_actual_tokens`, `num_decodes`,
  `decode_query_len`, `max_query_len` → `use_prefill_config=False`). The rebuild
  before each replay writes the same persistent buffers the graph reads: input-buffer
  `query_start_loc` / `seq_lens` / `positions`, block-table and slot-mapping buffers,
  and the QSA builder's `token_to_req` / `logical_positions` / `visible_blocks` /
  `slot_mapping` / `k_work_metadata`. The indexer's decode top-k grid uses the static
  page-table width. Nothing in the draft path reads `max_seq_len` for decode. This is
  the same path stock 0.33 takes for qwen4_exp whenever the target mode is
  FULL_AND_PIECEWISE. It is only new on Thor and only in combination with a
  PIECEWISE target.
- **Capture failure** shows up at startup, not silently. If an op in the MTP layer
  syncs (e.g. a Thor-only kernel path), capture raises and the engine does not come
  up. Then unset the variable.
- **Token identity.** The same kernels at the same M=1 shape replay the same work, so
  the drafts and therefore the outputs should be bit-identical. The A/B above is the
  check. If outputs drift, drafts differ (e.g. a library picking a different tactic
  under capture). Output under greedy should still equal the target's argmax, but the
  rule here is identical tokens, so treat any diff as a failure.
- **Perf can be negative** at N=1 on sm_110 (Mjolnir's finding). Measure before
  making it the default.
- **Memory:** one extra FULL graph per captured size ≤ max_num_seqs, which is small.
- The timer creates one CUDA event per mark (about 15 per step) only when enabled.
  Steps whose events have not completed are resolved on a later step.

## Not implemented (owned files): fused draft decode for qwen4_exp

`models/qwen4_exp/common/qsa_cache.py` belongs to another agent and must not be edited
here. With item 2 enabled, this hunk would let `_fused_multi_step_decode` run all K-1
passes as one FULL replay, with one metadata build per step instead of K-1:

- `QSAMetadataBuilder`: set `supports_draft_decode_metadata_update = True` (class
  attribute, next to `_cudagraph_support`, `:602`).
- In `build()` (`:651`), keep `common_attn_metadata` and the `build_qsa_metadata`
  keyword arguments, e.g. `self._draft_args = (common_attn_metadata, k_work_metadata if build_k_work else None, request_capacity)`.
- Add `update_draft_decode_metadata(self, metadata)` that calls `build_qsa_metadata`
  again with the stored common metadata and the builder's own persistent buffers.
  This is capture-safe for a 1-token decode: all inputs (`query_start_loc`,
  `seq_lens`, `slot_mapping`, `block_table_tensor`) are the draft's persistent GPU
  buffers, which `update_draft_inputs` / `compute_slot_mappings` advance between
  passes. The host ints (`num_reqs`, `num_mapped_tokens = query_start_loc_cpu[-1]`)
  are constant across passes. It writes into `metadata.token_to_req` /
  `logical_positions` / `visible_blocks` / `slot_mapping` / `k_work_metadata`, the
  same views.

The FA2 main builder already reports support (`flash_attn.py:741`). Nothing else
is needed in this overlay: `_configure_decode_metadata_update` will then pick
`_fused_multi_step_decode`, and the timing log will show `draft_fused_FULL`.

## Checks run

- `python3 -m py_compile` on all three files: OK.
- `spec_step_timing.py` driven on CPU with a stub `torch.cuda.Event` (7 synthetic
  steps, window 3, one non-decode step): averages and labels as expected, marks
  outside a step ignored.
- Early PLE fetch hunks (2026-11-15): the patched runner and speculator import in
  the 0.33 image (CPU container) with `VLLM_PLE_DISK_EARLY=1`. The hook logic is
  tested in `qwen4exp-ple-disk/tests/test_ple_disk_early.py`.
- **Not run (original hunks, at writing):** no import of the patched runner or
  speculator inside the 0.33 image, no server start, no capture, no device timing,
  no A/B. **Not run (early hunks):** no server start, no GPU run, no A/B.
