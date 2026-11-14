# ManyForge Assistant: Fine-Tuned Small Model Plan

Status: **Draft — TO REVIEW** (design only; nothing downloaded, generated or trained).
- Rev. 1, 2026-11-10.
- Companion to [`ASSISTANT-SANDBOX-REFACTOR-PLAN.md`](ASSISTANT-SANDBOX-REFACTOR-PLAN.md): that plan owns where the model server runs and how it is contained; this plan owns which model it serves and how that model is made.

## 1. Goal

A small model, fine-tuned for the ManyForge assistant's narrow scope, that is:

1. **Accurate in scope.** At minimum it builds every ManyForge example scenario (§4). More scenario families can be added later.
2. **Fast on Thor and Orin.** On Thor it runs on the tcgen05 fast path (FP8 or NVFP4 GEMMs, speculative decoding). On AGX Orin (sm_87, no FP8/FP4) it runs a 4-bit weight build. AMD64 with an NVIDIA dGPU is covered by whichever of the two fits its GPU.
3. **Kept in step with ManyForge.** The assistant tool contract is **not frozen**. The training data lives outside the ManyForge repo, and an on-demand checker reports which traces no longer match the current contract (§6).

## 2. Base model

**Nemotron 3.5 Bolt E4B** (NVIDIA post-training of Google Gemma 4 E4B; 8B total, about 4B effective; released 2026-10-23).

| Artifact | Use |
|---|---|
| `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16` (14.9 GB, also on NGC) | Fine-tuning base |
| `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16-DFlash-NVFP4` (580 MB; default revision W4A16, `w4a4` revision W4A4) | Stock drafter for the baseline; retrained after fine-tuning (§7.4) |
| `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB` (Q4_K_M GGUF, 6.2 GB) | Orin baseline |

Why Bolt:
- **Dense transformer.** All weights go through standard GEMMs, which is what tcgen05 accelerates. The hybrid alternatives (Qwen 3.5 small with Gated DeltaNet, LFM3 with convolution layers, Nemotron-H with Mamba) spend part of each step in non-tensor-core kernels and are less mature on Orin.
- **Best fine-tuning support.** NVIDIA published the full post-training recipe (NeMo-RL, NeMo-Gym; documented in NeMo AutoModel) and the datasets, including tool-calling and instruction-following sets. We train with the same recipe and mix their data with ours.
- **Both machines covered.** ModelOpt FP8/NVFP4 and vLLM on Thor; official GGUF and TensorRT Edge-LLM (a community Gemma 4 E4B build exists) on Orin.

No FP8 or NVFP4 build of the Bolt model itself is published; we quantize our fine-tuned build ourselves (§7.5).

Fallbacks, only if Bolt fails the baseline (§7.1): Qwen 3.5 4B (stronger on Tau2-Bench per NVIDIA's own table, but hybrid), LFM3-1.7B (fastest, but Liquid says it is weak at multi-step reasoning, and the LFM Open License needs a legal check).

Licenses: Bolt is OpenMDW-1.1 over Gemma 4 (Apache 2.0). Both allow commercial use; confirm before publication.

## 3. The contract the model is trained against

The source of truth stays in ManyForge; this plan copies nothing from it.

- **Tool schemas.** Headless serves the active mode projection at `GET /api/assistant/modes/composer-assistant` (and `/context`). It is generated from the node catalog by `manyforge_headless_api/assistant_instruments.py`, and carries per-tool schema hashes and a source content digest.
- **What the model actually sees.** The bridge (`manyforge_assistant_bridge`) builds the system prompt (`prompt.py`), narrows the catalog (`catalog.py`), applies constrained decoding and regeneration (`constrained.py`, `node_catalog.py`), and keeps history (`history.py`).

So the training traces are captured **between the bridge and the model**, with the existing evidence proxy (`manyforge/scripts/debug/model_capture_proxy.py`). Each trace is then exactly a request the deployed model will receive: system prompt, tool list, messages, tool results, and the teacher's answer.

**Contract fingerprint** of a trace = the projection's source content digest + the per-tool schema hashes used + a digest of the bridge system prompt.

## 4. Scenario coverage

**Baseline (requirement A):** every example in `manyforge/examples/`. Today that is 26 `.scenario` bundles and 13 `.program.yaml` files.

For each example scenario:
- the **task** is to build its Program, starting from the scenario with an empty or scaffold Program;
- the **check** is structural equivalence with the example's Program tree (node kinds, order, parameters within tolerance, scene resources), run through headless on the Mock lane.

Each example becomes a **scenario family**:
- **Training side:** many variants (rephrased requests, different detail levels, different scene parameters and starting states, partial builds, "continue from here", error recovery, correct refusals).
- **Test side:** the canonical request plus held-out rephrasings and starting states. Test variants never appear in training.

Real-lane and Sim-lane examples are built on Mock (same Program, Mock robot), so no live hardware is needed.

**Edit skills:** the 71 cases of `manyforge/scripts/debug/smoke_corpus_v021.yaml` (wrap, insert, move, update, scene edits, queries) become further families, with the existing outcome scorer (`outcome_scorer.py`) as their checker.

## 5. Where things live

| What | Where |
|---|---|
| Plan and decisions | This file (NemoClaw-Thor); a short pointer in `manyforge_specs` only if a ManyForge change is ever needed |
| Generator, checker, scenario families, test set, training/quantization scripts, dataset manifests | `NemoClaw-Thor/finetune/` (new) |
| Generated bulk traces and checkpoints | Outside git, on Thor under `~/manyforge-finetune/` (datasets, runs, checkpoints), each with a disk cap; optionally mirrored to a private HF dataset |
| Contract (tool schemas, prompt) | ManyForge, unchanged; read at check time |

Trace storage format is model-neutral JSON: messages, tool calls (name + arguments), tool results, plus the fingerprint, the family, the teacher, and the check result. Rendering to Bolt's chat template happens only at training time, so changing the base model does not invalidate the data.

## 6. Contract checker (requirement B)

`finetune/check_corpus.py`, run on demand. It reads the contract from a ManyForge source tree (offline, by importing the projection generator) or from a running Mock headless.

**Static level (fast, offline):**
- every tool called still exists;
- arguments validate against the current JSON schema (required fields, enums, renamed or removed parameters);
- recorded tool results still match the result schema;
- the system prompt digest is unchanged (a changed prompt marks traces as "prompt drift", not invalid).

The report groups failures by cause ("tool X renamed", "new required parameter Y", "prompt changed") and by family, and states the fingerprint it checked against.

**Replay level (on demand, needs Mock):** re-runs the tool calls of each trace against Mock ManyForge and re-applies the family checker. This catches behaviour changes the schemas do not show.

**Repair:**
- simple renames through a small mapping file (`finetune/contract_migrations.yaml`);
- everything else by regenerating the affected families through the teacher pipeline (§7.2).

Every dataset build records the fingerprint it was checked against. A model build is only accepted against the contract it was trained on.

## 7. Pipeline

### 7.1 Baseline (needs Thor and Orin time; ask first)
- Stock Bolt BF16 + NVFP4 DFlash drafter on Thor, vLLM v0.33 (`sha256:d825baf9…`), BF16 KV, scored on the test set against Qwen3.6-35B-A3B-NVFP4.
- `nsys` profile on Thor: confirm the GEMM and attention kernels are sm_100-family/sm_110 tcgen05 kernels, not fallbacks. Known risk: some FlashInfer TRT-LLM-gen kernels match only SM 100/103/107 (seen with Flint); Gemma 4 attention (large head dimension on global layers, sliding window mixed with global) may hit that.
- Stock GGUF on Orin with llama.cpp; TensorRT Edge-LLM build if llama.cpp is too slow.

### 7.2 Corpus generation
- **Teacher:** Qwen3.6-35B-A3B-NVFP4 served locally on Thor (Apache 2.0, our tool-call format, free and unlimited), through the real bridge and Mock ManyForge, with the capture proxy recording traces (§3).
- **Generator:** samples a family, a variant (request wording, detail level, starting state) and runs the agent loop.
- **Rejection sampling:** keep only traces whose final state passes the family checker. Keep error-recovery traces and correct refusals.
- **Mix:** about 20–30% NVIDIA post-training data (tool calling, instruction following) to prevent forgetting.
- **Size:** 5–20k traces to start; split by family so no test variant leaks.

### 7.3 Training (Thor, 122 GiB unified memory)
- NVIDIA's Bolt recipe (NeMo AutoModel), LoRA first. Full SFT of the effective 4B with the per-layer embedding tables frozen if LoRA plateaus.
- Then a short **GRPO** stage (NeMo-RL) with a NeMo-Gym environment wrapping Mock ManyForge; reward = family checker pass.

### 7.4 Drafter
- Regenerate drafter training data from the fine-tuned model; retrain DFlash with ModelOpt (NVIDIA's recipe; five Qwen3-style layers sharing the target's embeddings).
- Quantize the drafter to NVFP4 (W4A16, plus a W4A4 candidate).

### 7.5 Quantization

| Target | Build |
|---|---|
| Thor | ModelOpt FP8 (no calibration); NVFP4 W4A4 candidate calibrated on our traces, if vLLM v0.33 has an sm_110 NVFP4 W4A4 linear kernel |
| Orin | Q4_K_M GGUF or W4A16 (Marlin) |
| AMD64 | FP8 on Ada/Hopper/Blackwell, GGUF otherwise |

### 7.6 Acceptance gate
A build is accepted when, on the test set at its contract fingerprint:
- it builds **every** example scenario (requirement A);
- overall pass rate is at least Qwen3.6-35B-A3B's;
- latency targets are met on Thor and Orin (targets set from the baseline);
- the Thor kernel profile shows no fallback kernels on the hot path.

### 7.7 Deployment
The model server of the sandbox plan: hardened no-egress container, image and weights pinned by digest, the bridge's constrained decoding against the tool schemas. The model build records the contract fingerprint it was trained against.

## 8. Work split

Up to five Haiku 5.5 agents, in parallel on separate files, once this plan is approved:
1. scenario-family definitions and the example-scenario checker (§4);
2. the corpus generator around the bridge, Mock and the capture proxy (§7.2);
3. `check_corpus.py` and the migrations file (§6);
4. the NeMo-Gym environment for GRPO (§7.3);
5. quantization and export scripts (§7.5).

**Claude is not used to write training data** (Anthropic's terms restrict using outputs to train AI models). Agents write tooling and the held-out test variants only; training traces come from the open-weight teacher.

Every download, generation run, training run and live run on Thor or Orin is asked for first.

## 9. Open questions for review

1. Teacher choice: Qwen3.6-35B-A3B-NVFP4 only, or also a larger open-weight teacher for the hardest families?
2. Example equivalence tolerance: what counts as "built the example" (exact node ids? parameter tolerances?).
3. Whether `NemoClaw-Thor/finetune/` is the right home, or the future `manyforge-assistant` component repo from the sandbox plan.
4. Latency targets per machine (set after the baseline).
5. License confirmation for OpenMDW-1.1 before publication.
