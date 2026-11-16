# qwen4exp-draft-head: cheaper logits and drafter for MTP speculative decoding (vLLM 0.33.0)

Qwen3.8-Flash-Next ("Qwen4Exp") served from `nvidia/Qwen3.8-Flash-Next-FP4-Local`
with the built-in MTP drafter on Jetson AGX Thor (sm_110), official vLLM 0.33.0 image,
V2 model runner, `cudagraph_mode=PIECEWISE` with breakable CUDA graphs.

Decode on this board is limited by the bytes read per step. The checkpoint keeps
`lm_head` (248,320 x 2,560, 1.27 GB) and the dense MTP parts in BF16, and the lm_head
is read on every target step and on every one of the K draft steps.

This overlay adds six features. Each has its own env variable, all are off by
default, and with none set the behaviour is stock 0.33.0. Item 5 is for the
`nvidia/Qwen3.8-Flash-Next-NVFP4` checkpoint, whose dense side layers are BF16;
on FP4-Local it logs a no-op. Item 6 works the same on both checkpoints.

| # | Variable | What it does | Target output |
|---|---|---|---|
| 1 | `QWEN4EXP_DRAFT_VOCAB=/path/ids.npy` | The MTP draft step scores only these token ids, using its own reduced copy of the lm_head rows. | Unchanged. Target verification keeps the full vocabulary, so only the acceptance length can drop. |
| 2 | `QWEN4EXP_LM_HEAD_MXFP8=1` | Online MXFP8 for the target lm_head. The drafter shares that head. With item 1 set, the reduced draft head is MXFP8 too. | Changes slightly (see "Precision of item 2"). |
| 3 | `QWEN4EXP_MTP_DENSE_FP8=1` (named `QWEN4EXP_MTP_DENSE_MXFP8` before 2026-11-16) | Online 8-bit (`QWEN4EXP_FP8_FORMAT`, default blockwise FP8) for the BF16 dense MTP projections: `fc_embedding`, `fc_hidden`, the QSA `qkv_proj` (which carries the indexer's `index_qk_proj`), `o_proj`, and `shared_expert.gate_up_proj` / `down_proj`. | Unchanged, because only the drafter changes. |
| 5 | `QWEN4EXP_SIDE_FP8=1` | Online 8-bit (`QWEN4EXP_FP8_FORMAT`, default blockwise FP8) for the target's dense side layers that a ModelOpt MIXED_PRECISION checkpoint leaves in BF16: GDN `in_proj_qkvz` / `out_proj`, QSA `qkv_proj` (with the fused indexer `index_qk_proj`) / `o_proj`, shared-expert `gate_up_proj` / `down_proj`. The online form of the earlier RadixArk FP8 hybrid. No-op when the checkpoint already quantizes them (FP4-Local: MXFP8). `QWEN4EXP_SIDE_FP8_SKIP` (comma-separated kinds, e.g. `self_attn.qkv_proj`) keeps those kinds in BF16; unknown kinds refuse to start. On NVIDIA NVFP4, all kinds cost Tool Eval 5 points (86/100); skipping `self_attn.qkv_proj` (QSA projections + indexer) gave 93/100. With `QWEN4EXP_QSA_UNFUSE_INDEXER=1` the indexer is its own kind, `self_attn.indexer.index_qk_proj`, so the skip can keep only it in BF16 (see "Item 5: QSA indexer unfused"). | Changes slightly (FP8 W8A8 GEMMs in every layer). |
| 6 | `QWEN4EXP_HC_FP8=1` | Online 8-bit (`QWEN4EXP_HC_FP8_FORMAT="<down>[,<up>]"`, default `fp8_per_block,mxfp8`) for the target's hyperconnection GEMMs: per layer the `attn_`/`mlp_hyper_connection` merged down+inject and up projections, plus the final mixer's down and up (194 matrices, 1.279 GB BF16 per step). Both checkpoints keep them in BF16. Measured on Thor: GPU −1.3…−1.4 ms per step, step −1.1 ms on R8b, unchanged on N3, prefill +0.13 s. See "Item 6". | Changes slightly (FP8 W8A8 GEMMs in every layer). |
| 4 | `QWEN4EXP_LM_HEAD_SCREENED=1` | Screened target lm_head (backport of vLLM PR #60566): an FP8 copy scores all 248,320 rows with a rigorous error bound, and only the rows that can be in the top k get exact logits from the resident BF16 head. Used for greedy and top_k <= `QWEN4EXP_LM_HEAD_SCREEN_K` (default 32) batches; all other batches use the full BF16 head. Mutually exclusive with item 2. | Unchanged: the sampled token and every target probability that the sampler or rejection sampler reads come from the full head's top k, up to one-ulp summation-order differences (see "Item 4"). |

None of this needs a new checkpoint snapshot. Item 2 uses vLLM's stock
online MXFP8 linear method (`Mxfp8OnlineLinearMethod`); item 3 uses the stock online method of
`QWEN4EXP_FP8_FORMAT` (`Fp8PerBlockOnlineLinearMethod` by default, or `Mxfp8OnlineLinearMethod`). It loads the BF16 rows onto
a meta-device weight and quantizes each layer as soon as its rows have arrived, so
the BF16 copy never stays resident. That suits unified memory. There is no offline
conversion script, and no disk or conversion time is needed: the quantization runs
at load and takes well under a second on the GPU.

## Files

All paths are package-relative. Line numbers refer to the pristine 0.33.0 files.
Every hunk is marked `[draft-head]`.

| File | Base | Change |
|---|---|---|
| `vllm/models/qwen4_exp/nvidia/draft_head.py` | new | `make_lm_head()` builds a `ParallelLMHead`. With item 2 it passes `quant_method=Mxfp8OnlineLinearMethod()` and logs the kernel. `pad_draft_vocab()` / `load_draft_vocab()`: sorted unique ids from the `.npy`, topped up with the lowest unused ids to a multiple of 128, so there is no vocab padding and GEMM rows stay aligned; the result is a device tensor. `maybe_quantize_mtp_dense()` puts a stock `OnlineQuantizationConfig(targets={MTP_DENSE_TARGET: <QWEN4EXP_FP8_FORMAT>})` on the drafter's quant config. `log_mtp_dense()` logs the matched layers. |
| `vllm/models/qwen4_exp/nvidia/model.py` | full copy of 0.33 (sha256 `e9d9c2a6…a1c0`) | `Qwen4ExpForCausalLM.__init__` (`:730-734`): `ParallelLMHead(...)` becomes `make_lm_head(...)` with the same arguments. The unused `ParallelLMHead` import is dropped (`:33`), and `make_lm_head` is imported. |
| `vllm/models/qwen4_exp/nvidia/mtp.py` | full copy of 0.33 (sha256 `86126a83…a9e`) | (a) `_make_draft_vllm_config` (after `:154-160`): `maybe_quantize_mtp_dense(draft_quant_config)`. (b) `Qwen4ExpMultiTokenPredictor.__init__` (before `:231`): `log_mtp_dense(...)`. (c) `Qwen4ExpMTP.__init__` (after `:437`): registers the non-persistent buffer `draft_vocab_ids` (`None` when off). When it is set, the init also creates `draft_head = make_lm_head(len(ids), hidden)` and `draft_logits_processor = LogitsProcessor(len(ids))`. Only on the last PP rank, and only for untied embeddings. (d) `compute_logits` (`:465`): when ids are set, it computes logits with the reduced head, writes them into a `[T, 248320]` tensor filled with `-inf` (`index_copy_`), and returns that. (e) `load_weights` (`:471`): for checkpoint `lm_head.weight`, it also yields `draft_head.weight = lm_head.weight.index_select(0, ids)` and logs it. |
| `build_draft_vocab.py` | port of `flash-next-fast-v2/build_draft_vocab.py` | Builds a vocabulary from a corpus (frequency ranking after byte, special and added tokens, then id-order fill). `--verify` checks an asset against a tokenizer. |
| `vllm/models/qwen4_exp/nvidia/screened_head.py` | new `[screened-head]` | Item 4. `screened_head_settings()` (env, mutual exclusion, rows per call = `min(max_num_seqs * (1 + num_speculative_tokens), 64)`), the three Triton kernels and the bound of PR #60566 unchanged, `ScreenedLMHead` (FP8 copy, `required_top_k()` / `screen_decision()` with the fallback reason, `build()` with checks, kernel warm-up and log line), `ScreenCheck` (item 4 check mode), `ScreenStats` (item 4 stats), `_install_hooks()` (the PR's two runner hunks as wrappers, see "Item 4"). |
| `vllm/models/qwen4_exp/nvidia/model.py` (item 4) | same copy | `[screened-head]` hunks: import (`:85`); `Qwen4ExpForCausalLM.__init__` (after `:736`): `screened_head = None`, `_screened_settings = screened_head_settings(vllm_config)` (raises on bad combinations at startup); `compute_logits` (`:903`): uses the screen while the sample hook has set `screened_head.top_k`, else stock; `load_weights` (`:917-925`): after the stock loader, `ScreenedLMHead.build(...)` from the loaded BF16 `lm_head`. `Qwen4ExpForConditionalGeneration` reaches both through `language_model`. |
| `vllm/models/qwen4_exp/nvidia/draft_head.py` (item 5) | same new file | `[side-fp8]`: `side_fp8_target()` (the runtime-name regex minus `QWEN4EXP_SIDE_FP8_SKIP`), `maybe_quantize_side_layers()` (env, no-op checks, puts a stock `OnlineQuantizationConfig(targets={side_fp8_target(): format})` on the target's checkpoint quant config), `log_side_layers()` (the one-line summary). `[qsa-unfuse]`: `QSA_INDEXER_KIND`, `qsa_unfuse_indexer()`, `maybe_unfuse_qsa_indexer()`. |
| `vllm/models/qwen4_exp/nvidia/model.py` (item 5) | same copy | `[side-fp8]` hunks: import; `Qwen4ExpForCausalLM.__init__`: `maybe_quantize_side_layers(self.quant_config)` before `Qwen4ExpModel(...)` is built, `log_side_layers(self.model)` after. `[qsa-unfuse]` hunks: `_Qwen4ExpQSAUnfusedIndexerInit` (after the imports); `Qwen4ExpDecoderLayer.__init__` builds target QSA layers with it when the switch is on; `maybe_unfuse_qsa_indexer(self.quant_config)` before `maybe_quantize_side_layers`. |
| `tests/test_side_fp8.py` | new | Item 5 tests against the real vLLM 0.33 code (see "Tests"). |
| `vllm/models/qwen4_exp/nvidia/hyperconnection.py` (item 6) | full copy of 0.33 (sha256 `9561e232…6f4a`) | `[hc-fp8]` hunks in `GatedResidual.__init__`: the three HC Linears take `quant_config=hc_fp8_quant_config(prefix)` instead of `None`; a converted merged down+inject layer gets its pad rows loaded as zeros (`load_hc_pad`) and `_use_hc_down_silu=False`. |
| `vllm/models/qwen4_exp/nvidia/hc_fp8.py` (item 6) | new `[hc-fp8]` | `HC_FP8_TARGET` (runtime-name regex), `hc_fp8_quant_config()`, `load_hc_pad()`, `log_hc_fp8()` (the one-line summary). |
| `vllm/models/qwen4_exp/nvidia/model.py` (item 6) | same copy | `[hc-fp8]`: `import os`; `Qwen4ExpForCausalLM.__init__`, after `Qwen4ExpModel(...)`: `log_hc_fp8(self.model)` when the switch is set (imported there, so the switch-off overlay needs no new mount). |
| `vllm/models/qwen4_exp/nvidia/draft_head.py` (item 6) | same new file | `[hc-fp8]`: `log_side_layers` skips HC modules (they use the same method class). |
| `tests/test_hc_fp8.py` | new | Item 6 tests against the real vLLM 0.33 code (see "Tests"). |
| `tests/test_vocab.py`, `tests/test_fp8.py`, `tests/test_screened.py`, `tests/test_screened_kernels.py`, `tests/test_screen_stats.py`, `tests/_screened_common.py` | new | CPU tests (see "Tests"). |

The `qwen4exp-spec-step` and `qwen4exp-ple-disk` overlays touch none of these files,
so they combine. Neither do the overlays of `modelopt.py`, `nvfp4_utils.py`,
`models/utils.py`, `whisper.py`, `ops/qsa_indexer.py`, `ngram_embedding.py` or
`ple_disk*.py`.

## How each item hooks into 0.33

**Item 1, reduced draft vocabulary.** In the V2 runner, each draft step calls
`Speculator.compute_draft_logits` → `self.model.compute_logits(h)`
(`v1/worker/gpu/spec_decode/speculator.py:434-437`). `sample_draft` (`:444-478`) then
takes the argmax (the default, `draft_sample_method="greedy"`), or a Gumbel sample
that is cached for probabilistic rejection.

Logits stay full width with `-inf` outside the set. As a result:
- every proposed id is already a target id, so no id mapping is needed anywhere;
- the probabilistic path sees p=0 outside the set, so rejection sampling stays exact.

The drafter's own `lm_head` is still replaced by the target's head
(`v1/worker/gpu/spec_decode/eagle/utils.py:106-128`; stock, unchanged), but with item
1 set it is no longer used for draft logits. The reduced head is a separate module,
`draft_head`, which `load_eagle_model` does not touch. It is loaded during the
drafter's normal weight load, so it is ready before CUDA graph capture. That avoids
the old fork's lazy first-call build.

In PIECEWISE mode, `compute_logits` runs eagerly: the draft prefill graphs only the
model forward, and the draft decodes run with `NONE`
(`target_dependent_ar/speculator.py:98-120`). The ops added here have static shapes
and no host syncs (`new_full`, `index_copy_` with a device index buffer), so they are
also safe inside a FULL graph, e.g. with `VLLM_SPEC_STEP_DRAFT_FULL_CG=1` from
`qwen4exp-spec-step`. They do not depend on K.

The old fork's asset is valid for this tokenizer. `draft_vocab_code_47149.npy`
(sha256 `36cd2455…7a37`, in `serving/docker/assets/qwen38-flash-next/`) holds 47,149
sorted unique int32 ids, range 0..248076. All ids exist in the FP4-Local
`tokenizer.json`, which has a 248,044-entry BPE vocab plus 33 added tokens up to id
248076. The asset contains:
- all 256 byte ids;
- all 33 added and special tokens (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`,
  `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, …);
- typical code tokens (`def`, `Ġdef`, `Ġreturn`, `import`, `Ġself`, `ĠNone`,
  `Ġconst`, `(){`, `Ċ`, `ĠĠĠ`).

At runtime it is topped up to 47,232 rows with the 83 lowest unused ids, 36454..36564.
The asset is coding-oriented, so for other workloads build a new one (see "Draft
vocabulary").

**Item 2, MXFP8 lm_head.** `ParallelLMHead` takes a preselected `quant_method`
(`model_executor/layers/vocab_parallel_embedding.py:598-622`), and the online linear
methods work on it: the stock `online/lm_head.py` builds its copies the same way. The
online layer-wise loader (`model_loader/reload/layerwise.py:120-224`, finalized in
`model_loader/base_loader.py:106`) quantizes the head the moment its 635 M elements
have been loaded.

`LogitsProcessor._apply_head` calls `lm_head.quant_method.apply`
(`model_executor/layers/logits_processor.py:142-146`). The drafter shares this module
(`eagle/utils.py:106-121`), so without item 1 the draft logits use it too.

The stock option `draft_lm_head_quantization` (`config/speculative.py:484`, applied
in `eagle/utils.py:111-116` via `online/lm_head.quantized_lm_head_copy`) does work
for qwen4_exp MTP on the V2 runner, but it does not give item 1 or item 2:
- It quantizes a drafter-only copy of the full 248K head. The target keeps its BF16
  head, so the target step gains nothing.
- The BF16 head stays resident next to the extra copy (+0.64 GB for fp8, about
  +0.36 GB for nvfp4).
- Its `fp8` is PTPC W8A8, which rejects Marlin. No FP8 scaled-mm kernel has been seen
  running on sm_110 in this server.

Leave it unset. With item 2 it raises, because it needs an unquantized head. With
item 1 its copy would never be used.

**Item 3, 8-bit MTP dense.** vLLM's stock way to quantize unquantized layers of a
pre-quantized checkpoint online is `--quantization-config '{"targets": …}'`
(composition in `model_executor/layers/quantization/base_config.py:295-352`). That
flag does not reach the MTP drafter: `SpeculativeConfig` builds the draft
`ModelConfig` without `quantization_config` (`config/speculative.py:1406-1430`). The
hook therefore attaches the same stock `OnlineQuantizationConfig` to the drafter's
own `ModelOptMixedPrecisionConfig`.

The checkpoint's `ignore` patterns keep those layers BF16 (`UnquantizedLinearMethod`),
and `resolve_quant_method` then swaps in the online method of `QWEN4EXP_FP8_FORMAT`
(`Fp8PerBlockOnlineLinearMethod` by default) for prefixes that match `MTP_DENSE_TARGET`. Target layers never match (they are `model.layers.N…` and
use the checkpoint's own MXFP8 anyway).

Not converted:
- the HC mixers, which are built with `quant_config=None` and partly run through
  custom CuTeDSL kernels (`HcDownSiluMma`);
- the router `mlp.gate` and `shared_expert_gate` (N=512 and N=1);
- the norms.

**Item 5, side-layer FP8 (NVIDIA NVFP4 checkpoint).** The checkpoint's
`quantization_config` is ModelOpt `MIXED_PRECISION` (`modelopt_mixed`): only the
routed experts (NVFP4), the PLE table (FP8) and the MTP experts (FP8 blocks) are in
`quantized_layers`; `ignore` lists, per layer, `linear_attn*`, `self_attn*`,
`mlp.shared_expert*`, `mlp.gate`, `mlp.shared_expert_gate` and the HC mixers, plus
`embed_tokens` and `lm_head`. Every other dense linear is unquantized as well
(`ModelOptMixedPrecisionConfig.get_quant_method` falls back to
`UnquantizedLinearMethod`). So under 0.33 these target matrices are BF16:

| Runtime module (per layer) | Layers | Shape (N x K) | BF16 | Item 5 |
|---|---|---|---|---|
| `linear_attn.in_proj_qkvz` (ckpt `in_proj_qkv` + `in_proj_z`) | 36 | 16384 x 2560 | 3.020 GB | FP8 |
| `linear_attn.out_proj` | 36 | 2560 x 6144 | 1.132 GB | FP8 |
| `self_attn.qkv_proj` (ckpt `q/k/v_proj` + `indexer.index_qk_proj`) | 12 | 13952 x 2560 | 0.857 GB | FP8 |
| `self_attn.o_proj` | 12 | 2560 x 6144 | 0.377 GB | FP8 |
| `mlp.shared_expert.gate_up_proj` | 48 | 1280 x 2560 | 0.315 GB | FP8 |
| `mlp.shared_expert.down_proj` | 48 | 2560 x 640 | 0.157 GB | FP8 |
| `linear_attn.in_proj_ba` (ckpt `in_proj_b` + `in_proj_a`) | 36 | 96 x 2560 | 0.018 GB | BF16 (N=96: no 128-blocks; small) |
| `mlp.gate` (router) | 48 | 512 x 2560 | 0.126 GB | BF16 (routing precision) |
| `mlp.shared_expert_gate` | 48 | 1 x 2560 | 0.0005 GB | BF16 (built with `quant_config=None`) |
| HC mixers (`*_hyper_connection.input_mix_weight_down_block_inject` / `_up`, final mixer) | 97 x 2 | 324(+pad) x 10240, 10240 x 320 | 1.28 GB | BF16 (custom CuTeDSL kernels, precision); item 6 converts them |
| `ple.kv_proj` (layer 1) | 1 | 12800 x 2560 | 0.066 GB | BF16 (not in the RadixArk class) |
| `embed_tokens`, `lm_head` | 1 + 1 | 248320 x 2560 | 1.27 GB each | BF16 (the embedding is a gather; the head has items 2 and 4) |

The 192 converted modules hold 312 checkpoint matrices: the 300 of the RadixArk FP8
hybrid (`linear_attn.in_proj_qkv/in_proj_z/out_proj`, `self_attn.q/k/v/o_proj`,
`mlp.shared_expert.gate/up/down_proj`; 5.42 GiB) plus the 12 indexer `index_qk_proj`
(640 x 2560), which vLLM fuses into the QSA `qkv_proj` GEMM; a fused layer takes one
method, so the indexer projection goes along. FP4-Local quantizes it too (MXFP8).

The hook is the stock composition of item 3, on the target's quant config:
`maybe_quantize_side_layers` sets `online_quantization_config` with
`targets={side_fp8_target(): format}` at the start of `Qwen4ExpForCausalLM.__init__`.
Each `LinearBase` then calls `resolve_quant_method`
(`model_executor/layers/quantization/base_config.py:295-352`): the checkpoint method
is `UnquantizedLinearMethod`, the prefix matches, so it gets
`Fp8PerBlockOnlineLinearMethod` (or `Mxfp8OnlineLinearMethod`). The regex is anchored
on `model.layers.N.`, so the drafter (`mtp.layers.N`, own quant config, item 3) and
the experts never match. The online loader quantizes each layer once its shards
have arrived; no BF16 copy stays resident.

The stock `--quantization-config '{"targets": …}'` would reach the target too, but
it does not check that the matched layers are unquantized: on FP4-Local it would
requantize the MXFP8 side layers. The env switch makes that check. No-op cases (one
log line, nothing changed):
- the checkpoint is not ModelOpt `MIXED_PRECISION`;
- the checkpoint's `quantized_layers` already lists any side matrix (FP4-Local lists
  all 312 as MXFP8). Without this check the targets would *requantize* those layers
  (dequantize MXFP8, then FP8), because `resolve_quant_method` applies an online
  target to quantized layers too.
With `--quantization-config` also set, startup raises.

**Item 5: QSA indexer unfused (`QWEN4EXP_QSA_UNFUSE_INDEXER=1`, `[qsa-unfuse]`).**
Without LoRA, 0.33 fuses the indexer projection into the QSA `qkv_proj` GEMM
(`qsa.py:508-516`, `self.fuse_indexer_projection = vllm_config.lora_config is None`;
`Qwen4ExpQSAQKVIndexerLinear`, 13,952 x 2,560 of which 640 indexer rows), so one
method covers both. N3 keeps the whole GEMM in BF16 (3.35 ms per step) to keep the
indexer exact. With the switch, the target's QSA layers take qsa.py's own unfused
(LoRA) layout: `qkv_proj` is the stock `QKVParallelLinear` (q/gate, k, v: 13,312 x
2,560) and `indexer.index_qk_proj` stays its own `ReplicatedLinear` (640 x 2,560),
run separately in `forward` (`qsa.py:634-640`); `load_weights` then uses no indexer
mapper, so the checkpoint's `self_attn.indexer.index_qk_proj.weight` loads into it
by name and q/k/v stack into `qkv_proj` as usual. No `qsa.py` overlay:
`Qwen4ExpDecoderLayer` builds those layers through `_Qwen4ExpQSAUnfusedIndexerInit`,
a subclass whose `__setattr__` turns that one assignment into `False`, and resets
the module's class to the stock `Qwen4ExpQSAAttention` right after `__init__`; at
run time the module is the stock class on its unfused path. The drafter
(`mtp.layers.N`) keeps the fused GEMM. `maybe_unfuse_qsa_indexer` also gives the
target's quant config a `qkv_proj -> [q_proj, k_proj, v_proj]` packed mapping
(it was `[…, indexer.index_qk_proj]`): the quant configs match a fused module by its
shards, and with the stock mapping a target on the indexer without one on
`qkv_proj` makes the online matcher refuse ("unmatched shards").

The side-FP8 kind `self_attn.indexer.index_qk_proj` exists only with the switch on,
so `QWEN4EXP_SIDE_FP8_SKIP=self_attn.indexer.index_qk_proj` keeps exactly the 12
indexer projections in BF16 and `self_attn.qkv_proj` (now q/gate, k, v) goes to
online FP8 with the other side layers. Naming that kind without the switch refuses
to start (the indexer would otherwise silently go FP8 inside `qkv_proj`). The
switch alone (side FP8 off) changes no values, only splits one GEMM in two. On
FP4-Local, side FP8 stays a no-op and the unfused indexer and `qkv_proj` keep the
checkpoint's MXFP8 method, so there the switch only costs the extra GEMM: leave it
off.

Bytes per target step for the side layers (weights only, `fp8_per_block`): N3 3.359
GB, C1 (unfused, indexer skipped) 2.950 GB, i.e. -0.409 GB = -1.67 ms at 245 GB/s,
minus 12 extra small GEMM launches; all kinds FP8 unfused 2.930 GB (= N2).

**Item 4, screened lm_head.** A backport of vLLM PR #60566 ("[MRV2][Perf] Screened
lm_head", open, not in 0.33). The bound, the three Triton kernels (`_screen_kernel`,
`_select_kernel`, `_exact_kernel`) and the eligibility rule are the PR's, unchanged.
The changes are in how it is wired in:

| PR #60566 | This overlay |
|---|---|
| `--screened-lm-head` (`ModelConfig.screened_lm_head`) | `QWEN4EXP_LM_HEAD_SCREENED=1`; no config or CLI file is patched. |
| `ScreenedLMHead.from_model()` in `GPUModelRunner.load_model` | `ScreenedLMHead.build()` at the end of `Qwen4ExpForCausalLM.load_weights`, so the FP8 copy is in the model memory that vLLM profiles before sizing the KV cache. |
| hunk in `GPUModelRunner.sample` (`model_runner.py:1655-1656`): screen instead of `compute_logits` | the same decision in a wrapper around `GPUModelRunner.sample`. It sets `screened_head.top_k` for the call, and `Qwen4ExpForCausalLM.compute_logits` then uses the screen. The wrapper is installed at load by `_install_hooks()`, because the `qwen4exp-spec-step` overlay already mounts its own full `model_runner.py`; with a wrapper both overlays combine. |
| hunk in `Sampler.add_request` (`sampler.py:96-108`): new `uses_logits_processors` | the same body, installed the same way (it calls the same states in the same order and only adds that array; with `QWEN4EXP_LM_HEAD_SCREEN_STATS` also a per-request reason code). |
| FP8 copy via `ops.scaled_fp8_quant` (per row) | the same per-row e4m3 scale in plain torch (`amax / 448`); the bound is computed from the stored values either way. |
| `MAX_TOP_K = 32` | `QWEN4EXP_LM_HEAD_SCREEN_K` (default 32). |
| kernel warm-up for 1, 17, 33 rows | every row count 1..max, so no Triton JIT happens on a live request. |
| weight-transfer / `reload_weights` refresh | not wired (not used here); a reload through `load_weights` rebuilds it. |
| — | `QWEN4EXP_LM_HEAD_SCREEN_CHECK=N`: in-server comparison on the same hidden states (below). |
| — | `QWEN4EXP_LM_HEAD_SCREEN_STATS=<sec>`: per-reason counts of screened and full-head calls (below). |

*How it works.* The FP8 copy (e4m3, one fp32 scale per row) scores every row as s_j.
For each row an fp64 bound err_j is computed at load, so that
`|c_j - s_j| <= ||x|| * err_j`, where c_j is the logit the BF16 head computes before
rounding. err_j covers the FP8 rounding (Cauchy-Schwarz) and the fp32 accumulation
error of both GEMMs. t is the k-th largest of the per-64-row-block maxima of
`s_j - ||x|| err_j`, so at least k rows have `c_j >= t`. A row whose upper bound is
below `t - 2 eps |t|` lies more than one BF16 ulp below all of them, so it can neither
enter nor tie the top k. All other rows get exact logits: the BF16 weight row against
x with fp32 accumulation, rounded to BF16. Every other logit is `-inf`.

There is no fixed candidate count. The number of exact rows depends on the data, and
the set always contains the true top k with ties. PR #60566 measured, on real
Flash-Next rows, about 15 exact rows per token for greedy and about 1,200 for top-20.
A fixed K (for example the 64 or 128 best FP8 scores) cannot guarantee that. It also
does not bound the cost: on flat rows the bound keeps thousands of rows. Plain per-row
FP8 flips the greedy argmax on 413 of the 4,096 synthetic test rows (10%), and on 0
with the screen.
`QWEN4EXP_LM_HEAD_SCREEN_K` is therefore the largest *sampling* top_k that the screen
serves. The default is 32: it covers Qwen's top_k=20, and the PR measured top-32 still
23% faster than the BF16 head at 64 rows. A larger value keeps more exact rows, since
the threshold t drops.

*When the screen is used* (`ScreenedLMHead.required_top_k`, per target `sample()`
call, from host-side state only):
- every request in the batch is greedy, or has temperature > 0 with
  `top_k <= SCREEN_K`, and `min_p == 0`;
- no request asks for logprobs;
- no request uses a logits processor: logit bias, allowed token ids, penalties, bad
  words, thinking budget or a custom processor;
- there is no structured-output bitmask;
- the batch has at most `min(max_num_seqs * (1 + num_speculative_tokens), 64)` logit
  rows;
- the sampler is the plain `Sampler` (no NaN counting, trace replay or sampling mask).

Otherwise that call uses the full BF16 head (`QWEN4EXP_LM_HEAD_SCREEN_STATS` counts
how often, and why). With the `fly` rejection method, greedy
batches certify `fly_entropy_top_k` rows instead of 1, because that verifier reads the
entropy of the target's top rows.

*Sampling semantics.* The sampler applies temperature, then top-k (it keeps every row
>= the k-th value), then top-p inside the top k (`sampler.py:226-262`). In the stock
path the logits outside the top k are therefore already `-inf` before softmax or
Gumbel. The screen keeps a superset of the top k with exact values, so:
- the processed logits are the stock ones;
- greedy picks the same argmax;
- seeded Gumbel sampling picks the same token for the same seed;
- the FlashInfer top-k/top-p sampler sees the same top-k distribution (it is only
  statistically equivalent to stock in either case);
- temperature and top_p act only inside the top k, so they are unaffected.

What is lost is only what the sampler never reads with these settings: logits outside
the top k. That is why logprobs, min_p, no-top_k sampling and logits processors fall
back to the full head.

*Rejection sampler (MTP).* `RejectionSampler._verify` calls
`sampler.apply_sampling_params` without `skip_top_k_top_p`
(`spec_decode/rejection_sampler.py:264`). The target probabilities are therefore
computed after top-k/top-p, and with the full head a draft token outside the top k
already has target probability exactly 0. With the screen it has the same 0. Greedy
verification compares the draft with the target argmax, which is in the exact set.

Draft tokens therefore do not need to be added to the exact set. Their target
probability is either the exact value (inside the top k) or exactly the stock 0.
`tests/test_screened.py` checks this with 3,200 draft ids, including ids outside the
top k and outside the exact set. The draft probabilities `q` come from the drafter's
own logits and are not touched.

*Exactness of kept logits.* The exact rows are fp32 dot products rounded to BF16. They
equal the cuBLAS BF16 GEMM up to summation order. PR #60566 measured 0.047% of kept
entries differing, 99% of those by one ulp, and 12 greedy flips in 1.36M live rows.
All 12 were exact BF16 rounding ties, where the screened value was the correctly
rounded one. Greedy text A/B runs therefore match except at such ties. Use
`QWEN4EXP_LM_HEAD_SCREEN_CHECK` to tell a tie from a real error.

*CUDA graphs, MTP and prefill.* `GPUModelRunner.sample` and `compute_logits` run
eagerly after the PIECEWISE graphs. The screen:
- needs no host sync: the decision uses host-side numpy state, and the candidate
  count stays on the device (the exact kernel loops to it);
- uses persistent buffers sized at load (`ub` fp32 and `cand` int32,
  `[rows, 248320]`, plus `block_max`);
- has static grids, so it would also be capturable.

MTP K=3..5 gives 4..6 logit rows per request. A prefill or mixed step has one logit
row per request (the last token), so it takes the same path. Draft steps are not
affected: they use `Qwen4ExpMTP.compute_logits` (the shared BF16 head, or item 1's
reduced head).

*Memory (unified pool).*
- The BF16 lm_head stays resident: the exact pass reads its rows, and
  `ScreenedLMHead.weight` is a view of it, not a copy.
- New: the FP8 copy, 0.636 GB.
- New: scales and bounds, 2 MB.
- New: the buffers, `8 B x 248,320 x rows`. Rows is
  `min(max_num_seqs * (1 + K), 64)`, for example 19.9 MB for 2 sequences at K=4.
- Net: about +0.66 GB against stock.
- At load, the fp64 bound computation adds a transient 4,096-row chunk of about
  0.3 GB.

## Item 6: hyperconnection FP8 (`QWEN4EXP_HC_FP8`)

*What exists.* Every decoder layer has two `GatedResidual` modules
(`attn_hyper_connection`, `mlp_hyper_connection`), and the model ends with one
`hyper_connection_mixer` (`use_combine=False`). Shapes for this model (hidden 2,560,
`hc_count` 4, `hc_lowrank` 320, so the multi-stream width is 10,240):

| Runtime module | Count | Checkpoint matrices | Runtime shape (N x K) | Kind | BF16 per step |
|---|---|---|---|---|---|
| `layers.N.{attn,mlp}_hyper_connection.input_mix_weight_down_block_inject` (`MergedColumnParallelLinear`, `disable_tp`) | 96 | `input_mix_weight_down` 320 x 10240 + `block_inject_weight` 4 x 10240 (+ 12 pad rows, not in the checkpoint) | 336 x 10240 | GEMM; at M <= 48 the fused CuTeDSL down+SiLU kernel (`hc_down_silu`, BF16 weight), else the Linear + `hc_silu` | 0.637 GB (324 rows; the fused kernel skips the pad) |
| `layers.N.{attn,mlp}_hyper_connection.input_mix_weight_up` (`ReplicatedLinear`) | 96 | `input_mix_weight_up` 10240 x 320 | 10240 x 320 | GEMM (Linear dispatch) | 0.629 GB |
| `hyper_connection_mixer.input_mix_weight_down` / `_up` (`ReplicatedLinear`) | 1 + 1 | 320 x 10240, 10240 x 320 | same | GEMM | 0.013 GB |
| `*.hc_norm.weight` | 97 | 10240 (BF16; FP32 in FP4-Local) | - | elementwise (`grouped_gemma_rmsnorm`, `hc_combine_norm`) | 2 MB |
| combine, gate mix (`hc_combine`, `hc_gate_mix`, `hc_silu`) | - | - | - | elementwise on activations | no weights |

All 290 target checkpoint matrices are BF16 in both NVIDIA NVFP4 and FP4-Local
(`tests/test_hc_fp8.py` reads the headers), 1.279 GB read per target step. The stock
model builds the three Linears with `quant_config=None`, so the checkpoint's quant
config never reaches them. The drafter has the same modules (`mtp.layers.48.*`,
`mtp.hyper_connection_mixer`, 0.040 GB per draft pass); item 6 leaves them BF16.
Only the GEMM weights are worth converting; the norms are 2 MB and the elementwise
kernels (most of the 490 ms at a 7.5K prefill chunk) read activations, not weights.

*How it hooks in.* `hyperconnection.py` (full copy) passes
`hc_fp8_quant_config(prefix)` as `quant_config`. With the switch off, or for a
prefix that is not a target HC projection (the drafter), that is `None`: the stock
code path. With it on, it is one stock `OnlineQuantizationConfig(targets=
{HC_DOWN_TARGET: <down>, HC_UP_TARGET: <up>})` from `QWEN4EXP_HC_FP8_FORMAT`, and
`LinearBase` resolves it to the stock `Fp8PerBlockOnlineLinearMethod` or
`Mxfp8OnlineLinearMethod`: meta weight, online
quantization once the layer's rows have arrived, no resident BF16 copy. Two hunks
are needed beyond that:
- The merged down+inject layer has 12 pad rows that the checkpoint never fills.
  `load_hc_pad` feeds them zeros through the layer's own (online-wrapped) loader at
  construction. Without it the layer would quantize only at the final
  "delayed processing", from `torch.empty` rows that share a 128-row scale block
  with real rows (rows 256..335 = 64 low-rank rows, the 4 injection rows and the pad).
- `_use_hc_down_silu` is False for a converted layer: the fused CuTeDSL kernel takes
  a BF16 weight. The layer runs as Linear (FP8 GEMM) + the stock `hc_silu` kernel,
  the path stock already uses above M = 48.
Nothing else changes: the same split, gate mix and combine kernels, no host syncs,
and the FP8 linears are the ones the side layers already run inside PIECEWISE
graphs. The online methods are independent of the checkpoint's quant config, so
item 6 composes with item 5, `QWEN4EXP_QSA_UNFUSE_INDEXER` and `--quantization-config`.

*Bytes and expected saving (target step).*

| | Bytes per step | Saving at 245 GB/s |
|---|---|---|
| stock BF16 | 1.279 GB | - |
| item 6, `fp8_per_block` (e4m3 + fp32 per 128x128; the pad rows are read) | 0.652 GB | -0.628 GB, -2.56 ms |
| item 6, `mxfp8` (e4m3 + e8m0 per 32) | 0.672 GB | -0.607 GB, -2.48 ms |
| item 6, default `fp8_per_block,mxfp8` (the only pair the GPU kernels accept) | 0.661 GB | -0.618 GB, -2.52 ms |

Resident memory drops by the same ~0.63 GB. Costs that offset part of it at decode:
one dynamic activation-quant kernel per converted GEMM (194 per step), the separate
`hc_silu` (96 per step) and losing the fused split-K down+SiLU kernel. The stock
up GEMM runs at ~170 GB/s (3.7 ms for 0.63 GB), so a better FP8 GEMM efficiency
there would add to the saving. Expected net: about -1.5 to -2.8 ms per target step,
so the C2 gate (>= -2.5 ms) is not certain. At prefill the down GEMM gains one extra
pass over its activation for the quantization (~150 MB per call at 7,488 tokens), so
a 7.5K chunk may get ~40-60 ms slower (estimate); measure it.

*Precision (CPU, real weights).* Through the real online loader, every converted
matrix has a weight relative error of 0.023-0.027 (e4m3, about half an ulp), the
injection rows inside the shared 128-row block included. One mix on synthetic
multi-stream states (unit RMS per stream, outlier channels, the real `hc_norm`
weights) gives 1.7-4.7% relative error on the block input (`fp8_per_block`; `mxfp8`
emulated: 1.9-4.2%), layers 0, 24, 47 and the final mixer of both checkpoints. That
is the usual W8A8 level, but it applies to every layer's residual mixing: check Tool
Eval before keeping it.

*Server run (either checkpoint).*
```bash
  # the four mounts of "Docker mounts", plus
  -v $P/vllm/models/qwen4_exp/nvidia/hyperconnection.py:$V/models/qwen4_exp/nvidia/hyperconnection.py:ro \
  -v $P/vllm/models/qwen4_exp/nvidia/hc_fp8.py:$V/models/qwen4_exp/nvidia/hc_fp8.py:ro \
  -e QWEN4EXP_HC_FP8=1 \
```
`QWEN4EXP_HC_FP8_FORMAT="<down>[,<up>]"` (one value = both; default
`fp8_per_block,mxfp8`) is separate from `QWEN4EXP_FP8_FORMAT`. Only the default pair
starts on sm_110: `fp8_per_block` needs K % 128 (the up layers have K = 320,
`the last dimension of x 320 must be divisible by group_size 128`), `mxfp8` needs
N % 32 (the down layers have N = 336, `Expected n to be divisible by 32`). Log line
(one, at model construction):
`[hc-fp8] Hyperconnection GEMMs: 194 matrices to online fp8_per_block (down) / mxfp8 (up) (attn_hyper_connection.input_mix_weight_down_block_inject 48, attn_hyper_connection.input_mix_weight_up 48, hyper_connection_mixer.input_mix_weight_down 1, hyper_connection_mixer.input_mix_weight_up 1, mlp_hyper_connection.input_mix_weight_down_block_inject 48, mlp_hyper_connection.input_mix_weight_up 48): 1.279 GB BF16 -> 0.661 GB per step via Cutlass`. `0 matrices`
means `hyperconnection.py` is not mounted. `Model loading took … GiB` is about
0.6 GiB lower.

*Measured on Thor (2026-11-15, default formats, interleaved runs in one session).*
GPU work per step −1.3 ms (N3 + C1) and −1.4 ms (R8b + C3); step time 47.3 -> 46.2 ms on
R8b + C3, unchanged at 47.3 ms on N3 + C1 (the host gap grew); warm 7.6K prefill +0.13 s
on both. Tool Eval in noise (R8b 92, 91; N3 + C1 94, 94). Details:
`serving/docs/FLASH-NEXT-SPEED-PLAN.md` (Stage C).

*Risks.*
- **Kernel shapes.** The HC GEMMs are not 128-aligned: down N = 336, up K = 320.
  Each GPU kernel rejects one of them (above), hence the per-projection default.
- **Small-M speed.** Two quant kernels + an unfused SiLU per layer pair may eat a
  third of the byte saving (above). Measure decode tok/s and the target GPU time.
- **Quality.** New quantization (neither NVIDIA's QAD nor the RadixArk hybrid
  touched these matrices); the error compounds through the residual streams of all
  48 layers. Tool Eval decides.
- **Drafter unchanged.** The MTP mixers stay BF16 with the fused kernel (its warm-up
  still runs); converting them would only change the draft.

## Precision of item 2 (explicit)

Item 2 uses MXFP8 W8A8:
- the weight is stored as e4m3 with one e8m0 (power-of-two) scale per 32 inputs;
- activations are quantized the same way, dynamically, per token row and 32-block;
- the output is BF16.

The kernel is `FlashInferCutlassMxfp8LinearKernel`, which is first in the CUDA MXFP8
priority list (`model_executor/kernels/linear/__init__.py:568-578`). It needs sm>=100
and N>=128 (`kernels/linear/mxfp8/flashinfer.py:41-69`), and this server already
selects it for its MXFP8 side layers (startup log: `Using FlashInferCutlassMxfp8LinearKernel
for MXFP8 GEMM`). Marlin MXFP8 is only picked for N<128 layers.

Other options and why they were not used:

| Option | Why not |
|---|---|
| per-tensor FP8 | One scale over 635 M values. It is also not more accurate than per-row in the test below, and this server runs no FP8 scaled-mm kernel on sm_110, so it would be an untested kernel. |
| per-row/per-token FP8 (PTPC, the stock `draft_lm_head_quantization` scheme) | Similar error, but the same unproven-kernel issue. |

On random lm_head-like data (CPU reference), MXFP8 W8A8 gives a weight relative
error of 0.027 and a logit relative error of 0.037. That is about 1/3 of a e4m3 ulp,
the same order as PTPC (0.035) and per-tensor (0.037), and top-1 agreed on every
sampled row.

The target logits change slightly, so greedy outputs can flip on near-ties. That is
why item 2 has its own switch.

## Expected bytes read per step (weights only)

Assumptions: M = 1 sequence. Each verify cycle does 1 target logits step and K draft
steps, all with the same head.

| Configuration | Target step head | Draft step head | Draft step MTP dense |
|---|---|---|---|
| stock | 1.271 GB | 1.271 GB | 0.139 GB |
| item 1 | 1.271 GB | 0.242 GB (47,232 x 2,560 BF16) | 0.139 GB |
| item 2 | 0.656 GB (0.636 values + 0.020 scales) | 0.656 GB | 0.139 GB |
| items 1+2 | 0.656 GB | 0.125 GB | 0.139 GB |
| items 1+2+3 | 0.656 GB | 0.125 GB | 0.072 GB |
| item 4 | 0.65–0.68 GB (see below) | 1.271 GB (unchanged) | 0.139 GB |
| items 1+4 (+3) | 0.65–0.68 GB | 0.242 GB, or 0.125 GB with `QWEN4EXP_DRAFT_HEAD_FP8=1` | 0.139 (0.072) GB |

The MTP dense layers come to 69.4 M parameters in BF16: fc_embedding 6.6 M, fc_hidden
6.6 M, qkv+indexer 13,952 x 2,560 = 35.7 M, o_proj 15.7 M, shared expert 4.9 M.

Per verify cycle at K=4, the head plus MTP dense weights drop from 6.92 GB (stock) to
1.44 GB (all three). The savings break down as:
- item 1: 1.03 GB per draft step;
- item 2: 0.62 GB on the target step;
- item 3: 0.067 GB per draft step.

Item 4, target step at T logit rows (T = 1 + K = 5):

| Part | Bytes |
|---|---|
| FP8 screen | 0.636 GB values + 2 MB scales and bounds |
| `ub` write + read | 2 x 4 B x 248,320 x T = 9.9 MB |
| exact rows, greedy | ~15 rows x 5 KB x T ≈ 0.4 MB (PR's real-data count) |
| exact rows, top-20 | ~1,200 rows x 5 KB x T ≈ 31 MB (PR's real-data count) |
| logits write | 2.5 MB (same as stock) |
| total | 0.65 GB (greedy) to 0.68 GB (top-20), against 1.271 GB |

At about 245 GB/s, that is 2.6–2.8 ms instead of about 5.2 ms. The expected saving is
2.3–2.6 ms per target step, about 4–5% of a 55 ms step, if the Triton screen reaches
DRAM bandwidth on sm_110. PR #60566 measured -62% at 1 row and about -45% at 16–28
rows on GB10.

Item 2 saves about the same on the target step, but changes the logits. Item 4 does
not help the draft steps; combine it with item 1 (and `QWEN4EXP_DRAFT_HEAD_FP8`) for
those.

Resident memory:

| Item | Effect |
|---|---|
| item 4 | +0.66 GB (FP8 copy + buffers; the BF16 head stays) |
| item 2 | -0.62 GB |
| item 1 | +0.24 GB (BF16) or +0.125 GB (MXFP8) for `draft_head` |
| item 3 | -0.067 GB |

Item 5, target step (all six kinds, weights only; M = 1 + K tokens share one read):

| | Bytes per target step |
|---|---|
| NVIDIA NVFP4, stock (BF16 side layers) | 5.859 GB |
| NVIDIA NVFP4 + item 5, `fp8_per_block` (e4m3 + fp32 per 128x128) | 2.930 GB (-2.929 GB) |
| NVIDIA NVFP4 + item 5, `mxfp8` (e4m3 + e8m0 per 32) | 3.021 GB |
| FP4-Local (MXFP8 as shipped) | 3.021 GB, plus `in_proj_ba` 0.009 GB MXFP8 |

With item 5 the NVIDIA checkpoint reads about what FP4-Local reads for these
layers (the remaining `in_proj_ba` difference is 9 MB). At about 245 GB/s, -2.93 GB
is about -12 ms per target step. Resident memory drops by the same 2.93 GB. The
drafter steps do not change (item 3 covers the MTP dense layers).

## Docker mounts

Mount the four model files together: `model.py` and `mtp.py` both import the new
`draft_head.py`, and `model.py` imports the new `screened_head.py` (also with item 4
off). The asset is only needed for item 1. Before mounting, check that the
image's `model.py` and `mtp.py` match the sha256 values above.

```bash
P=${HOME}/workspaces/dev_ws/src/NemoClaw-Thor/serving/docker/patches/qwen4exp-draft-head
A=${HOME}/workspaces/dev_ws/src/NemoClaw-Thor/serving/docker/assets/qwen38-flash-next
V=/usr/local/lib/python3.12/dist-packages/vllm
  -v $P/vllm/models/qwen4_exp/nvidia/draft_head.py:$V/models/qwen4_exp/nvidia/draft_head.py:ro \
  -v $P/vllm/models/qwen4_exp/nvidia/model.py:$V/models/qwen4_exp/nvidia/model.py:ro \
  -v $P/vllm/models/qwen4_exp/nvidia/mtp.py:$V/models/qwen4_exp/nvidia/mtp.py:ro \
  -v $P/vllm/models/qwen4_exp/nvidia/screened_head.py:$V/models/qwen4_exp/nvidia/screened_head.py:ro \
  -v $A/draft_vocab_code_47149.npy:/opt/qwen4exp/draft_vocab.npy:ro \
```

## Environment and flags

```bash
  -e QWEN4EXP_DRAFT_VOCAB=/opt/qwen4exp/draft_vocab.npy \
  -e QWEN4EXP_LM_HEAD_MXFP8=1 \
  -e QWEN4EXP_MTP_DENSE_FP8=1 \
  ... vllm serve /root/.cache/huggingface/linked/fn-fp4-local \
      --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
      <the usual flags>
```

- Unset, empty or `0` means off.
- The variables are deliberately not `VLLM_*`, so vLLM's unknown-env warning stays
  quiet. They are read in the engine-core process, which inherits the container
  environment.
- K=3..5 all work; nothing depends on K.
- Keep `cudagraph_mode` PIECEWISE (breakable graphs are auto-enabled for Qwen4Exp).
- Do not set `draft_lm_head_quantization`.
- Item 4 instead of item 2 (the two together raise at startup):
  ```bash
    -e QWEN4EXP_DRAFT_VOCAB=/opt/qwen4exp/draft_vocab.npy \
    -e QWEN4EXP_DRAFT_HEAD_FP8=1 \
    -e QWEN4EXP_LM_HEAD_SCREENED=1 \
    -e QWEN4EXP_MTP_DENSE_FP8=1 \
  ```
  `QWEN4EXP_LM_HEAD_SCREEN_K` (default 32) is the largest sampling top_k that the
  screen serves; requests with a larger or no top_k use the full head.
  `QWEN4EXP_LM_HEAD_SCREEN_CHECK=N` (debug) also computes the full BF16 logits for
  every screened batch and logs a comparison every N screened calls. The served
  logits stay the screened ones, the head is read twice, and there is one host sync
  per log line.
  `QWEN4EXP_LM_HEAD_SCREEN_STATS=<sec>` (for example `60`) counts every target
  `sample()` call and its logit rows as screened (by top_k) or full head (by the
  first disqualifying reason) and logs the totals since load at most every `<sec>`
  seconds, on the first call after the interval (no line while idle). Host-side
  only: no GPU sync, no device read; unset or `0` records nothing. It needs
  `QWEN4EXP_LM_HEAD_SCREENED=1` and raises at startup on a negative value.
  Item 4 also raises at startup with LoRA, TP > 1 or
  `VLLM_BATCH_INVARIANT=1`. No vLLM flag changes; keep the sampling defaults
  (`temperature 1.0, top_p 0.95, top_k 20` from the generation config) or greedy,
  without logprobs, to get the screen.
- `draft_sample_method` may be `greedy` (the default) or `probabilistic`.
- Item 5 with the NVIDIA checkpoint (no new mount; it is in `draft_head.py` and
  `model.py`):
  ```bash
    -e QWEN4EXP_SIDE_FP8=1 \
    -e QWEN4EXP_FP8_FORMAT=fp8_per_block \
    -e QWEN4EXP_SIDE_FP8_SKIP=self_attn.qkv_proj \
  ```
  C1 (NVIDIA only): `qkv_proj` FP8, only the QSA indexer projection BF16 (no new
  mount; `draft_head.py` and `model.py`):
  ```bash
    -e QWEN4EXP_SIDE_FP8=1 \
    -e QWEN4EXP_FP8_FORMAT=fp8_per_block \
    -e QWEN4EXP_QSA_UNFUSE_INDEXER=1 \
    -e QWEN4EXP_SIDE_FP8_SKIP=self_attn.indexer.index_qk_proj \
  ```
  It composes with items 1-4 and `QWEN4EXP_DRAFT_HEAD_FP8`; `QWEN4EXP_FP8_FORMAT`
  applies to all of them. Do not pass `--quantization-config` with it.

## Verify from the server log

| Item | Expected log lines |
|---|---|
| 1 | `[draft-head] MTP draft vocabulary: 47232 of 248320 ids from /opt/qwen4exp/draft_vocab.npy`, then while loading the drafter `[draft-head] Loading 47232 draft_head rows from lm_head.` The spec-decode metrics should show a mean acceptance length close to the stock run; compare with the same prompts. |
| 2 | `[draft-head] language_model.lm_head: online MXFP8 (248320 x 2560) via FlashInferCutlassMxfp8LinearKernel`. The prefix may read `lm_head`, depending on the wrapper class. With item 1 there is also `[draft-head] draft_head: online MXFP8 (47232 x 2560) via …`. `Model loading took … GiB` should be about 0.6 GiB lower than stock. |
| 3 | `[draft-head] MTP dense projections: online fp8_per_block (re:…)` and `[draft-head] MTP online fp8_per_block layers: mlp.shared_expert.down_proj: 1 (from targets: …, fp8_per_block); mlp.shared_expert.gate_up_proj: 1 …; mtp.fc_embedding: 1 …; mtp.fc_hidden: 1 …; self_attn.o_proj: 1 …; self_attn.qkv_proj: 1 …` (6 layers). |

| 5 | `[side-fp8] Target side layers: online fp8_per_block (re:…)`, then `[side-fp8] Target side layers: 192 matrices to online fp8_per_block (linear_attn.in_proj_qkvz 36, linear_attn.out_proj 36, mlp.shared_expert.down_proj 48, mlp.shared_expert.gate_up_proj 48, self_attn.o_proj 12, self_attn.qkv_proj 12): 5.859 GB BF16 -> 2.930 GB per step via CutlassFp8BlockScaledMMKernel`, and the stock `Online quantization as part of model loading took … seconds`. `Model loading took … GiB` is about 2.7 GiB lower. With `QWEN4EXP_SIDE_FP8_SKIP=self_attn.qkv_proj`: 180 matrices, without the `self_attn.qkv_proj 12` entry. With `QWEN4EXP_QSA_UNFUSE_INDEXER=1`, first `[side-fp8] QSA indexer unfused: target self_attn.qkv_proj holds q/gate, k, v; self_attn.indexer.index_qk_proj is its own linear.`; with `QWEN4EXP_SIDE_FP8_SKIP=self_attn.indexer.index_qk_proj` (C1) the regex has no `indexer` and the summary is `192 matrices to online fp8_per_block (linear_attn.in_proj_qkvz 36, linear_attn.out_proj 36, mlp.shared_expert.down_proj 48, mlp.shared_expert.gate_up_proj 48, self_attn.o_proj 12, self_attn.qkv_proj 12): 5.820 GB BF16 -> 2.911 GB per step via CutlassFp8BlockScaledMMKernel` (the 0.039 GB difference is the BF16 indexer); without a skip, 204 matrices with `self_attn.indexer.index_qk_proj 12`. On FP4-Local: `[side-fp8] QWEN4EXP_SIDE_FP8 is a no-op: the checkpoint already quantizes 312 side-layer matrices (MXFP8, e.g. language_model.model.layers.0.linear_attn.in_proj_qkv).` |
| 6 | `[hc-fp8] Hyperconnection GEMMs: 194 matrices to online fp8_per_block (…): 1.279 GB BF16 -> 0.652 GB per step via CutlassFp8BlockScaledMMKernel` (see "Item 6"). |
| 4 | Once at load: `[screened-head] lm_head 248320 x 2560: FP8 screen ready in … s (FP8 copy 0.636 GB, buffers … MB, BF16 head kept for the exact rows); screens greedy and top_k <= 32 batches of <= N rows.` `Model loading took … GiB` is about 0.66 GiB higher than stock. With `QWEN4EXP_LM_HEAD_SCREEN_CHECK=N`: `[screened-head] check after … calls: rows=… argmax_diff=a (strictly_lower=b) topk_screened_out=c kept_logits_differing=d of e (avg … exact rows per token)`. b and c must be 0. a counts only BF16 ties (both rows have the same full logit). d should be a small fraction (one-ulp summation-order differences). The average should be in the tens for greedy and in the low thousands for top-20; much larger values cost speed, not correctness. With `QWEN4EXP_LM_HEAD_SCREEN_STATS=<sec>`, at most every `<sec>` s: `[screened-head] stats <s>s calls/rows: all=C/R screened=C/R full=C/R \| screened by top_k: k1=C/R k20=C/R \| full by reason: presence_penalty=C/R logprobs=C/R …` (totals since load; C = `sample()` calls, R = their logit rows; entries sorted by rows; `-` when empty). Reasons, checked in this order, the first one counts: `structured_output`, `batch_sharder`, `sampler_subclass`, `compute_nans`, `trace_replay`, `sampling_mask`, `hook_missing`, `no_rows`, `rows_over_max`, then the first logits processor of any request in the batch in pipeline order (`allowed_token_ids`, `logit_bias`, `min_tokens` (stop ids masked while below `min_tokens`), `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `bad_words`, `custom_processor`, `thinking_budget`), then `logprobs` (also per-request logprob token ids), and for the sampled requests `no_top_k`, `top_k_over_max` (top_k > `SCREEN_K`), `min_p`. A request keeps its processor reason for its whole life (for example `min_tokens` also after the minimum is reached), as vLLM keeps the flag. |

If an item is off, its line is absent.

How to A/B item 4 against the BF16 head:
1. Run fixed prompts greedy (`temperature 0`), with and without
   `QWEN4EXP_LM_HEAD_SCREENED=1`, everything else identical, with the Marlin
   atomic-add off as in the other determinism runs. Expect identical text. A
   divergence can only start at an exact BF16 tie between the top two logits;
   confirm it with one run with `QWEN4EXP_LM_HEAD_SCREEN_CHECK=1` on the same prompt
   (`strictly_lower=0`, `topk_screened_out=0`).
2. For sampling, use a fixed `seed` with `temperature 1.0, top_p 0.95, top_k 20`. The
   seeded Gumbel path then gives identical tokens unless a one-ulp difference moves
   a sample.
3. Speed: compare decode tok/s, or the `sample` phase of `VLLM_SPEC_STEP_TIMING`
   from `qwen4exp-spec-step`.

How to confirm the target output is unchanged under items 1 and 3: run the same
greedy prompts with spec decode on and with it off. The outputs must be identical up
to the batch-invariance noise of the stock server. A difference appears only with
item 2.

## Draft vocabulary (ManyForge or other domains)

Run this in the official vLLM container on Thor. The image already has transformers.
The tokenizer comes from the served snapshot, so no GPU is needed.

```bash
docker run --rm --pull never --user "$(id -u):$(id -g)" --entrypoint python3 \
  -v $P:/p:ro -v $A:/out -v ~/thor-hf-cache:/root/.cache/huggingface:ro -v /path/to/corpus:/corpus:ro \
  <vllm-0.33.0-image> /p/build_draft_vocab.py \
  /root/.cache/huggingface/linked/fn-fp4-local /out/draft_vocab_manyforge.npy \
  --size 47232 --corpus /corpus
# verify any asset (optionally with --corpus for coverage)
... /p/build_draft_vocab.py /root/.cache/huggingface/linked/fn-fp4-local /out/draft_vocab_code_47149.npy --verify
```

Using a multiple of 128 for the size avoids the runtime top-up. Each 10K ids cost
about 51 MB in BF16 or 26 MB in MXFP8 per draft step.

## Tests (CPU, `tests/`)

Run with any Python that has torch and numpy, e.g.
`python -B tests/test_vocab.py && python -B tests/test_fp8.py`. They use stubs for
the vLLM imports. Item 4 tests also need `triton` (`pip install triton`): run
`python -B tests/test_screened.py [rows]` (default 4,096 rows, about 13 min on 32 CPU
cores), `python -B tests/test_screened_kernels.py` (Triton CPU interpreter) and
`python -B tests/test_screen_stats.py` (seconds).

- `test_vocab.py`:
  - checks the asset sha256;
  - loads it with the real `draft_head.load_draft_vocab` and checks the 47,232 rows
    (sorted, superset, in range), the aligned no-op case and the env-off case;
  - checks that the `MTP_DENSE_TARGET` regex hits the 6 intended drafter prefixes and
    misses the indexer pre-fusion layer, router, gates, HC mixers, experts, target
    layers, `lm_head` and `draft_head`;
  - executes the real patched `Qwen4ExpMTP.compute_logits` (extracted from `mtp.py`)
    against a full head. Kept columns are bit-identical, the others are `-inf`, the
    greedy argmax equals the masked full argmax, the ids stay in the set, and the
    softmax puts 0 outside the set.
- `test_screened.py` (item 4):
  - builds the real `ScreenedLMHead` (FP8 copy and fp64 bounds) on a random
    248,320 x 2,560 BF16 head with log-normal row norms, 8 outlier channels (x8) and
    0.1% outlier rows (x4); the three kernels are emulated in torch with the same
    math;
  - 4,096 hidden states in four kinds: flat, peaked (top-1 at +10..25), crowded (30
    rows at +12..16) and near-tie (the PR's construction);
  - for k = 1, 20 and 32 it checks, per row:
    - no row of the full head's top k (with ties, as vLLM's top-k keeps them) is
      screened out, against both the BF16 and the unrounded logits;
    - the greedy argmax matches;
    - the kept logits are within one BF16 ulp of the fp64 dot product, plus the
      fp32-accumulation bound that every fp32 GEMM has;
  - it also reports the exact rows per token;
  - for the rejection sampler, vLLM's top-k/top-p (`apply_top_k_top_p_pytorch`)
    gives bit-identical target probabilities from the screened and the full logits
    for 3,200 draft ids (the top 25 plus random ids, many outside the top k and
    outside the exact set), and the greedy argmax matches;
  - `required_top_k` cases: greedy, Qwen sampling, rows over the buffer, penalties,
    top_k 40 > 32, no top_k, min_p, logprobs, fly minimum;
  - the `Sampler.add_request` and `GPUModelRunner.sample` wrappers on stub
    classes: `uses_logits_processors`, `top_k` set during the call and cleared
    after, structured output, NaN counting, idempotent install;
  - the env switches: off, rows formula, `SCREEN_K` parsing and range, LoRA, TP,
    and the MXFP8 mutual exclusion;
  - check-mode counters and the log line.
- `test_screen_stats.py` (item 4, seconds): the stats counters through the real
  hooks on stub classes named like vLLM's: switch off records nothing (and the
  reason stays the generic `logits_processor`); per-request processor reasons for
  each processor (pipeline order wins within a request and across the batch);
  every runner and batch reason in check order; screened calls by top_k; rows
  from `logits_indices`; `required_top_k` unchanged; the exact log line, at most
  one per interval (fake clock); empty tables; the value check.
- `test_screened_kernels.py` (item 4): runs the real Triton kernels under the
  Triton CPU interpreter (`TRITON_INTERPRET=1`) against the torch emulation, for
  rows 1..20 and k 1..32 on a 4,133 x 640 head with padded rows. The exact set and
  the candidate counts must be identical, the kept values within one BF16 ulp, and
  the greedy argmax equal. The interpreter misreads bf16 and fp8 loads, so the test
  passes fp32 tensors that hold the exact bf16/fp8 values. The on-GPU dtype
  conversions are therefore not covered.
- `test_side_fp8.py` (item 5): runs in the vLLM 0.33 image with the overlay
  mounted, against the real `ModelOptMixedPrecisionConfig`, `resolve_quant_method`,
  online-target matcher and model-class mappings, on the checkpoints' `config.json`,
  index and safetensors headers (no tensor data):
  - switch off (unset, `0`, empty): no online config;
  - on, NVIDIA: every runtime linear prefix derived from the index resolves as
    expected: exactly the 192 modules of the six kinds become
    `Fp8PerBlockOnlineLinearMethod`, each was `UnquantizedLinearMethod`, all BF16
    in the checkpoint and 128-divisible; all others keep their stock method
    (in_proj_ba, router, shared_expert_gate, HC mixers, PLE kv_proj, embed_tokens,
    lm_head); experts and every `mtp.*` prefix never match;
  - 312 checkpoint matrices = 300 of the RadixArk class (5.42 GiB, as recorded)
    + 12 indexer; bytes before/after; the summary line on the real names;
  - `mxfp8` selects `Mxfp8OnlineLinearMethod`; a bad format and a second online
    config raise; `quant_config=None` is a no-op;
  - FP4-Local: no-op with the MXFP8 log line, and forcing the targets there would
    requantize (so the check is needed);
  - `model.py`: the hooks run before `Qwen4ExpModel` (`maybe_unfuse_qsa_indexer`,
    then `maybe_quantize_side_layers`) and the log after it;
  - `[qsa-unfuse]`: today's N3 (fused, skip `self_attn.qkv_proj`) is 180 matrices
    and its regex has no indexer; unfused, the 12 `self_attn.indexer.index_qk_proj`
    prefixes map to the checkpoint's BF16 640 x 2,560 tensors and each `qkv_proj`
    holds exactly q/k/v (13,312 rows), the MTP keeps the fused layout; unfused + skip
    indexer: the 192 modules of the six kinds go FP8 and the indexer resolves
    `UnquantizedLinearMethod` (`ReplicatedLinear`); unfused + no skip: 204; unfused +
    skip `self_attn.qkv_proj`: indexer alone FP8 (and, without the packed-mapping
    change, vLLM's "unmatched shards" error, which is why it is there); the indexer
    skip without the switch and unknown kinds refuse; switch off (unset, `0`, empty)
    leaves the mapping untouched; bytes N3 vs C1 and the C1 summary line; FP4-Local
    with the switch: still a no-op, indexer and `qkv_proj` keep the checkpoint's
    `ModelOptLinearMethod`; the builder forces the flag off and returns the stock
    class (stock `__init__` stubbed), and the image's `qsa.py` still has the one
    fusion assignment, the `del`, and the unfused forward call.
  ```bash
  H=<overlay dir>; C=$HOME/fn-local-patch/vllm; V=/usr/local/lib/python3.12/dist-packages/vllm; N=$V/models/qwen4_exp/nvidia
  docker run --rm --runtime runc -e NVIDIA_VISIBLE_DEVICES=void --network none --entrypoint python3 \
   -v $C/model_executor/layers/quantization/modelopt.py:$V/model_executor/layers/quantization/modelopt.py:ro \
   -v $C/model_executor/layers/quantization/utils/nvfp4_utils.py:$V/model_executor/layers/quantization/utils/nvfp4_utils.py:ro \
   -v $C/model_executor/models/utils.py:$V/model_executor/models/utils.py:ro \
   -v $H/vllm/models/qwen4_exp/nvidia/draft_head.py:$N/draft_head.py:ro -v $H/vllm/models/qwen4_exp/nvidia/model.py:$N/model.py:ro \
   -v $H/vllm/models/qwen4_exp/nvidia/screened_head.py:$N/screened_head.py:ro -v $H/vllm/models/qwen4_exp/nvidia/mtp.py:$N/mtp.py:ro \
   -v $H/tests:/t:ro -v $HOME/thor-hf-cache:/root/.cache/huggingface:ro <vllm-0.33.0-image> \
   -B /t/test_side_fp8.py /root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47 \
   /root/.cache/huggingface/linked/fn-fp4-local
  ```
- `test_hc_fp8.py` (item 6): same image and mounts as `test_side_fp8.py` plus
  `hyperconnection.py` and `hc_fp8.py`; the real overlay `GatedResidual`, vLLM Linear
  classes, online FP8 methods and online layer-wise loader; only the GEMM kernel
  object is a torch emulation. Checks:
  - headers of both checkpoints: the 290 target HC matrices (48 x 2 x down /
    inject / up + final mixer down / up), all BF16, 1.279 GB;
  - switch off (unset, `0`, empty): `quant_config=None`, `UnquantizedLinearMethod`,
    fused down+SiLU kept, no pad load, no log line;
  - on (default `fp8_per_block,mxfp8`, `fp8_per_block` for both, and
    `mxfp8,fp8_per_block`) on the full set of real runtime prefixes: the 194 target
    Linears get the down / up method, the drafter's (`mtp.*`) keep BF16 and
    the fused kernel; converted down layers have the zero pad and the unfused path;
    the summary line with the exact bytes; the side-layer summary ignores HC layers; bad
    `QWEN4EXP_HC_FP8_FORMAT` values are refused;
  - real tensors of layers 0, 24, 47 and the final mixer of both checkpoints through
    the online loader with the model's shard ids: quantized at load (not deferred),
    pad rows exactly 0, per-matrix error, and one mix through the real
    `_down_and_inject` / up Linear against the BF16 reference (table above);
  - `model.py` logs after `Qwen4ExpModel` and imports `hc_fp8` only under the switch;
    `hyperconnection.py` has no `quant_config=None` left.
  ```bash
  # as for test_side_fp8.py, plus
   -v $H/vllm/models/qwen4_exp/nvidia/hyperconnection.py:$N/hyperconnection.py:ro    -v $H/vllm/models/qwen4_exp/nvidia/hc_fp8.py:$N/hc_fp8.py:ro    ... -B /t/test_hc_fp8.py <NVIDIA snapshot> /root/.cache/huggingface/linked/fn-fp4-local
  ```
- `test_fp8.py`: round-trip error of MXFP8 (mirrors 0.33's
  `_mxfp8_e4m3_quantize_torch` scale rule), PTPC and per-tensor on lm_head-like and
  MTP-dense shapes, with outlier rows and channels.

Not tested here (no GPU, no weights):
- the online loader path on the real checkpoint;
- the FlashInfer CUTLASS MXFP8 GEMM on sm_110 at M ≤ 6 with N = 248,320 and 47,232;
- the online fused QSA qkv+indexer linear (the target uses serialized MXFP8 for it);
- `[qsa-unfuse]` on a real model: building `Qwen4ExpQSAAttention` (needs the
  engine), the weight load into the unfused modules, the forward, online FP8 on the
  QSA `qkv_proj` without the indexer, and the step time;
- acceptance rates;
- end-to-end speed;
- `build_draft_vocab.py` (it needs transformers; it parses);
- item 4 on sm_110: kernel speed, Triton FP8/BF16 conversions and `tl.dot`, and the
  exact-row count on real hidden states (use `QWEN4EXP_LM_HEAD_SCREEN_CHECK`).

## Risks

- **Online quant on real loader paths.** Online MXFP8 on a `ParallelLMHead` and on
  the custom `Qwen4ExpQSAQKVIndexerLinear` loader is new for these modules. The log
  lines above show whether it ran. If the head is never loaded, the online finalize
  would quantize uninitialized rows, which the `Loading … draft_head rows` line rules
  out for item 1.
- **Small-M GEMM speed.** The CUTLASS MXFP8 GEMM pads M to a tile. At M ≤ 6 it should
  stay bandwidth-bound, but measure the decode tok/s. There is no lm_head-only switch
  to Marlin W8A16: `VLLM_DISABLED_KERNELS` would affect every MXFP8 layer.
- **Transient memory at load.** The drafter's own BF16 `lm_head` is loaded and then
  dropped when it shares the target head; this is stock behaviour. The online
  quantization briefly needs the BF16 head as well. These are load-time transients
  only, and the caching allocator keeps those blocks for reuse.
- **Domain fit.** The asset is coding-oriented, so acceptance can drop on other
  domains. Use the builder.
- **Item 4: exact-row count on real data.** The bound is Cauchy-Schwarz, so flat
  distributions keep thousands of rows. The synthetic test kept about 900 rows per
  token for greedy and about 14,000 for top-20 on flat rows. The cost then grows
  (5 KB per row per token) but the result stays exact. PR #60566's real-data
  numbers (about 15 and about 1,200) are what the speed estimate assumes. Check the
  live average with the check mode.
- **Item 4: Triton on sm_110.** The kernels were tuned on GB10/GB200 (64-row
  blocks, `tl.dot` with 16 token rows). The bound assumes fp32 accumulation in the
  BF16 MMA, which holds on Blackwell. The first load compiles every row-count
  specialization (included in the "ready in … s" time).
- **Item 4: runtime wrappers.** `GPUModelRunner.sample` and `Sampler.add_request`
  are replaced in the worker process at load. Both are pinned to the 0.33 bodies
  (`model_runner.py` sha256 `9145e260…`, `sampler.py` sha256 `7f9941f9…`). If
  either file changes, re-check the wrappers. The replaced `add_request` keeps the
  stock call order and only adds `uses_logits_processors`.
- **Item 4: ties.** Greedy text can diverge from a BF16-head run at an exact BF16
  tie of the top two logits (12 in 1.36M rows upstream). This is summation order,
  not the screen; any GEMM change does the same.
- **Item 5: new online paths.** Online FP8 on the GDN `in_proj_qkvz` (four
  logical shards loaded from two checkpoint tensors, one with a tuple shard id) and
  `out_proj` is new; the QSA fused `qkv_proj`, `o_proj` and shared expert already
  run this way in the drafter (item 3). The summary line and the stock
  "Online quantization … took" line show that it ran.
- **Item 5: quality.** Every target layer changes (W8A8, per-token-group 128
  activations). The RadixArk hybrid held 94/100 Tool Eval with the same 300
  matrices; the 12 indexer projections are extra (they feed the QSA top-k
  selection; NVIDIA's FP4-Local ships them as MXFP8). Re-run Tool Eval.
- **Item 5: small-M kernel.** `CutlassFp8BlockScaledMMKernel` at M = 1 + K was
  measured only through the RadixArk hybrid (whole recipe +34%); measure decode tok/s.
- **Untested setups.** TP>1, PP>1 and tied embeddings are untested. Item 1 is skipped
  for tied embeddings.
