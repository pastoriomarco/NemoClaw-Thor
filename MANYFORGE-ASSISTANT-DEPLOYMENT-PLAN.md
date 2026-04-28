# ManyForge Assistant — LLM Stack Deployment Plan for Thor and Jetson Orin AGX

**Status:** PENDING — tests deferred until the manyforge-assistant agent is
operative on Thor. This document captures the future evaluation plan.

**Target hardware:**

- Thor: development/evaluation target with Blackwell-class GPU behavior and
  substantially more memory headroom.
- Jetson Orin AGX Dev Kit: constrained deployment target with 64 GB unified
  memory and Ampere SM86 behavior.

**Memory budget:** on Orin, target **30-40 GB total for the LLM stack** so
ManyForge, NemoClaw/OpenClaw services, Isaac ROS, controller/runtime services,
and perception models such as FoundationPose, RT-DETR, and nvblox have usable
headroom on the same device.

**ManyForge contract references:**

- `/home/tndlux/workspaces/dev_ws/src/manyforge/docs/plans/AI_ASSISTANT_INTEGRATION_PLAN.md`
- `/home/tndlux/workspaces/dev_ws/src/manyforge/docs/reference/ASSISTANT_BACKEND_CONTRACT.md`

ManyForge owns the assistant provider contract, tool contracts, proposal
review/apply boundary, and workflow tests. This document owns model-serving
profiles, memory budgets, and platform-specific deployment tradeoffs.

---

## Goal

Decide which Thor-validated profile, or combination of profiles, is the right
fit for the manyforge-assistant agent across Thor and Orin AGX.

Thor remains the primary development and evaluation platform. Orin AGX is the
strict deployment budget case: any profile intended for field deployment must
fit the 30-40 GB LLM envelope while leaving enough memory for the robotics and
perception stack.

The Thor v7 bench already established the agentic-quality ceiling for each
candidate (see `PERFORMANCE-V7.md`). What's outstanding is
**workload-specific validation against the manyforge-assistant agent's actual
tool calls and pipeline workflows** and **whether each profile fits the Orin
AGX shared-memory budget with the rest of the stack present**.

---

## Candidate outcomes

Three plausible deployments, ranked by deployment simplicity:

### Outcome A: Cosmos-Reason2-8B alone

| Aspect | Value |
|---|---|
| Profile | `cosmos-reason2-8b` |
| Weights | BF16 (~17 GB) |
| KV at 64K ctx, FP8 KV | ~7 GB |
| ViT activations | ~1.5 GB peak |
| **Estimated steady-state on Orin** | **~26 GB** |
| Headroom inside 40 GB | ~14 GB (35%) |
| Thor TEB score | 81 ★★★★ Good |
| Thor IFEval | 84.9% |
| Strengths | Smallest footprint; physical-AI VLM specialty; surprisingly competent at general agentic; single-model simplicity |
| Risks | 81 TEB is below the 90+ tier — may struggle on the most complex agentic chains (TC-52..69 in tool-eval-bench: research workflows, schema+tool combos) |

**Pick if:** the manyforge-assistant tool-call set is dominated by
spatial/embodied queries with light tool calling. Cosmos's TEB 81 is
"adequate" and its 8B size leaves the most room for Isaac ROS.

### Outcome B: ManyForge profile alone (with vision)

| Aspect | Value |
|---|---|
| Profile | `qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge` |
| Weights | NVFP4 (~17 GB) + ViT BF16 (~830 MB) |
| KV at 3×64K ctx, TQ K8V4 | ~10 GB |
| MTP drafter | ~0.9 GB |
| Activation buffer | ~3-5 GB |
| **Estimated steady-state on Orin** | **~32-34 GB** |
| Headroom inside 40 GB | ~6-8 GB (15-20%) |
| Thor TEB score | (not benched — same family as 90/93 winners) |
| Strengths | Strongest agentic correctness; vision in the same model; one endpoint; matches the v7 winner family |
| Risks | Tightest budget — Orin's smaller pool + non-SM110 kernel paths may push us over; need to verify NVFP4 + TQ K8V4 + MTP all work on SM86; ViT activation spikes under multi-image load could OOM |

**Pick if:** the manyforge-assistant tool-call set demands the strongest
agentic correctness and vision is required. Single-model architecture
keeps orchestration simple.

### Outcome C: ManyForge (text only) + Cosmos-Reason2-2B (split, slimmed)

| Aspect | Value |
|---|---|
| Profile A | `qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge` (vision stripped — re-add `--language-model-only`, smaller KV budget) |
| Profile B | `cosmos-reason2-2b` (small Cosmos for spatial subagent) |
| Weights | NVFP4 35B (~17 GB) + Cosmos-2B BF16 (~5 GB) |
| Combined KV | trimmed — manyforge to ~2-3 GB at lower max_num_seqs/ctx, Cosmos to ~1 GB |
| Activations | ~3-4 GB combined |
| **Estimated steady-state on Orin** | **~28-32 GB** |
| Headroom inside 40 GB | ~8-12 GB (20-30%) |
| Thor TEB scores | Qwen 90 + Cosmos 81 (split) |
| Strengths | Best of both: agentic LLM specialist + physical-AI VLM specialist. Each profile slimmed to its specific role. Roles are clean (no overlap). |
| Risks | Two endpoints to orchestrate. Slimming may sacrifice context/concurrency. Co-serve interaction on Orin not yet validated (Thor co-serve verified). |

**Pick if:** workload demands BOTH strong agentic AND strong physical-AI
reasoning, AND the slimming to fit 40 GB is achievable without breaking
either role's quality threshold.

### Outcome D: Nemotron 3 Nano Omni (added 2026-04-28 — released same day)

| Aspect | Value |
|---|---|
| Profile | new — `nemotron3-nano-omni-30b-a3b-nvfp4` (TBD) |
| HF repo | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` (open license, no gated access) |
| Weights | NVFP4 (~20.9 GB on disk, 4.98 effective bpw — −0.38 pts vs BF16 across 9 multimodal benchmarks) |
| Architecture | 30B-A3B hybrid Mamba-Transformer MoE — same family as the (text-only) Nemotron 3 Nano we tested earlier |
| Native modalities | **vision (C-RADIOv4-H), video (EVS frame compression), audio (NVIDIA Parakeet, ≥8 kHz, up to 1 h), text** |
| Context | 16K → 49K → **262K** native (32K Thor recipe, 131K Spark recipe — pick by deployment) |
| KV at 32K, FP8 KV | ~10 GB |
| Activations | ~3-5 GB (vision + audio buffers) |
| **Estimated steady-state on Orin (40 GB target)** | weights 21 + KV 10 + activations 4 ≈ **~35 GB** at `--max-model-len 32768`, `gpu_memory_utilization 0.6` (vs Thor's recommended 0.65 on its 122 GB pool) |
| Headroom inside 40 GB | ~5 GB (12%) — tight but feasible |
| Thor TEB / IFEval | not yet benched; the text-only Nemotron 3 Nano scored TEB 67 on our Thor v7 bench — the Omni variant may behave differently due to multimodal pretraining |
| Strengths | Single endpoint for vision + audio + text; built-in reasoning mode with explicit `thinking_token_budget`; Jetson AI Lab has an official Thor recipe; NVFP4 quant officially supported on Blackwell (SM110); commercial-friendly license |
| Risks | Same architecture class as the Nano text model that scored 67/100 on agentic — quality may not match Qwen3.6-MTP. C-RADIOv4-H vision encoder is new (different from SigLIP2 in Cosmos / Qwen3-VL); SM110 kernel paths unverified. Audio support requires `pip install vllm[audio]` extra in the container (our v7 image doesn't ship it). Tighter Orin budget (~5 GB headroom) than Outcomes A/B/C. |

**Pick if:** the manyforge-assistant needs **vision AND audio** (e.g.,
voice control of robotics, multi-modal scene understanding) AND tool
calling AND reasoning, all in one endpoint. Folds Outcomes A, B, and C
into a single model — but only worth it if its agentic quality on the
manyforge-assistant workflow tests beats what the existing Qwen3.6-MTP
+ Cosmos split achieves.

**Thor reference recipe (from Jetson AI Lab):**
```
sudo docker run -it --rm --pull always --runtime=nvidia --network host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.20.0-ubuntu2404 \
  bash -c "pip install -q 'vllm[audio]' && \
    vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
      --trust-remote-code \
      --gpu-memory-utilization 0.65 \
      --max-model-len 32768 \
      --reasoning-parser nemotron_v3 \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder"
```

**Recommended sampling (per NVIDIA model card):**
- Reasoning mode: T=0.6, top_p=0.95, `thinking_token_budget=16384+1024`
- Instruct mode (no thinking): **T=0.2, top_k=1** (notably *not* the
  Qwen3.6 generic recommendation of 0.7/0.8 — Nemotron Omni is calibrated
  much tighter for non-thinking output)
- ASR (audio in): T=0.2, top_k=1

Note: our T=0 deterministic baseline finding from
`PERFORMANCE-V7.md` § "Recommended-sampling probe" likely still applies
— vendor-recommended sampling consistently cost 2–4 TEB points on the
Qwen3.6-MTP tests. The Omni model's `T=0.2 top_k=1` "Instruct" mode is
much closer to deterministic and may not show the same gap.

---

## Tests to run (once manyforge-assistant is operative)

### Per-profile boot test on Orin AGX

For each candidate, verify on Orin AGX (NOT Thor):

1. Profile boots inside 40 GB allocation
2. KV cache lands at usable size (≥1× context-budget × concurrency)
3. NVFP4 GEMM works on SM86 (Ampere) — may need `VLLM_NVFP4_GEMM_BACKEND` re-evaluation
4. TurboQuant K8V4 KV works on SM86 — `fix-pr39931-turboquant` mod is target-arch-agnostic but unverified on Ampere
5. flash_attn / flashinfer paths on SM86 (different from SM110)
6. Vision encoder (SigLIP2) works on SM86 — already validated per Thor 9B-VLM profile

### Per-profile manyforge-assistant workflow tests

For each candidate that boots, run the agent's actual production
workflows:

- **BT (Behavior Tree) node selection / dispatch** — does the LLM
  choose the right BT node from the available set? Multi-step BT chains?
- **Visual scene reasoning** — object detection + spatial relationships
  ("which object is closest to the gripper"); only relevant for
  outcomes A, B, and D (vision present)
- **Audio reasoning** (Outcome D only) — voice-command parsing,
  spoken-instruction → BT-node dispatch
- **Tool calling for robot APIs** — gripper open/close, pose commands,
  trajectory queries — tool-name correctness, argument schema
  compliance
- **Code generation for new BT nodes** — synthesize Python/C++ for a
  new node given a spec; quality vs latency
- **Multi-step planning / task decomposition** — turn a high-level
  goal into a sequence of BT-executable steps
- **Error recovery / retry logic** — handle empty tool results,
  malformed responses, tool-not-found, conflicting state
- **Latency-critical control loop** — measure TTFT and steady-state
  decode for short tool-call responses (target: <1 s TTFT, ≥15 tps
  decode for the closed-loop budget)

### Decision criteria

A candidate is **acceptable** if it meets all of:

1. Boots inside 40 GB on Orin AGX
2. Passes ≥ 90% of the manyforge-assistant workflow tests
3. Latency-critical loop meets the closed-loop budget
4. Vision works correctly when needed (A, B, or D; via Cosmos-2B in C)
5. (Outcome D only) Audio works for voice-control workflows if those
   are part of the agent's tool-call set

A candidate is the **winner** if:

- It's acceptable AND has the most headroom for Isaac ROS, OR
- All three are acceptable AND it's the simplest deployment

---

## Implementation notes per outcome

### If Outcome A wins (Cosmos-8B alone)

- Move `cosmos-reason2-8b` profile from `lib/launch.sh` to a new
  `orin/lib/launch-orin.sh` (or similar) with Orin-tuned defaults
- Drop manyforge profile from the Orin deployment entirely
- ManyForge orchestrator routes ALL queries to Cosmos-8B endpoint

### If Outcome B wins (manyforge alone with vision)

- Replicate `qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge` to an Orin variant
  (likely with different `gpu_mem_util`, possibly smaller `max_model_len`
  to fit the 40 GB budget)
- Verify TurboQuant K8V4 mod works on Ampere — if not, fall back to
  FP8 KV and lose ~40% of the KV budget
- Test ViT memory under concurrent vision requests (worst case)

### If Outcome C wins (split + slimmed)

- Adapt `start-duo.sh` for Orin (different ports OK, but tuned
  `gpu_mem_util` per device)
- Strip vision from manyforge-Orin variant: re-add `--language-model-only`
- Tune Cosmos-2B `gpu_mem_util` down (from Thor's 0.12 of 122 GB ≈ 14 GB
  to Orin's ~10 GB equivalent — lower max_num_seqs and max_model_len)
- Validate that the orchestrator's routing logic correctly distinguishes
  agentic queries (→ port 8000) from spatial-physics queries (→ port 8001)

### If Outcome D wins (Nemotron 3 Nano Omni)

- Add new profile `nemotron3-nano-omni-30b-a3b-nvfp4` to `lib/launch.sh`
  + `lib/config.sh` (single source of truth for both Thor + Orin)
- Vision encoder is C-RADIOv4-H (not SigLIP2) — boot may need a
  different `--mm-encoder-attn-backend` than the SDPA workaround used
  for Cosmos / Qwen3-VL. Probe at first boot.
- Audio support requires `pip install vllm[audio]` extra; either bake
  into a v8 image rebuild OR install at container boot (slower)
- Use Jetson AI Lab Thor recipe as starting point: `gpu_memory_utilization=0.65`,
  `--max-model-len 32768`, parsers `nemotron_v3` + `qwen3_coder`
- Drop the manyforge profile + Cosmos pair from Orin deployment;
  single endpoint serves all agentic + perception + voice
- Sampling default for the orchestrator: T=0.2, top_k=1 (NVIDIA's
  "Instruct" recommendation — much closer to deterministic than
  the Qwen3.6 generic T=0.7)

---

## Open questions

1. **NVFP4 on Ampere SM86**: vLLM's NVFP4 GEMM backend depends on
   FlashInfer-CUTLASS kernels. Need to confirm SM86 instantiations exist
   for the relevant tile shapes. If not, NVFP4 falls back to a slower
   path or fails outright. (Affects A's headroom and especially B/C/D
   that depend on NVFP4 35B / 30B-A3B weights.)
2. **TurboQuant K8V4 on Ampere**: the `fix-pr39931-turboquant` mod
   removes a SM110-related guard, but TQ K8V4 itself may have arch
   assumptions. Worth a boot probe before committing to Outcome B or C.
3. **Cosmos-Reason2-8B vision on Ampere**: Thor needs `TORCH_SDPA`
   workaround for SM110 ViT FA2 PTX crash; SM86 may not need this
   workaround (could try FA2 for better ViT throughput).
4. **C-RADIOv4-H vision encoder on Thor SM110 / Orin SM87**: this is
   NVIDIA's own encoder, different from the SigLIP2/Qwen-VL ViTs we've
   validated. Untested on either platform. If it crashes the same way
   Cosmos's ViT did on SM110 (vllm #38411), we'd need a similar SDPA
   fallback; Jetson AI Lab Thor recipe doesn't mention one, suggesting
   it may "just work".
5. **Audio extras in the container**: `pip install vllm[audio]` is not
   in our v7 image. Decide whether to bake into v8 (preferred for cold
   starts) or install at container boot (slower but doesn't require a
   rebuild).
6. **Nemotron 3 family agentic ceiling**: the text-only Nemotron 3 Nano
   landed mid-pack on TEB (67/100). The Omni variant uses the same
   architecture class but with multimodal pretraining — does that
   shift agentic tool-calling quality up, down, or sideways? Only the
   manyforge-assistant workflow tests will tell.
7. **ManyForge-assistant agent test harness**: the workflow tests above
   need a runnable harness — defer until the agent itself is at least
   prototyped.

---

## Reference: Thor (v7) numbers

For context, the per-candidate Thor v7 results that underpin this plan:

| Profile | TEB | IFEval | Median tps | Notes |
|---|---:|---:|---:|---|
| `cosmos-reason2-8b` | 81 ★★★★ | 84.9% | 14.5 | Surprise general-agentic competence |
| `qwen3.6-35b-a3b-nvfp4-mtp-fp8kv` | 93 ★★★★★ | 90.4% | 19.5 | Tool-eval-bench winner |
| `qwen3.6-35b-a3b-nvfp4-tq-mtp` | 90 ★★★★★ | 89.0% | 24.8 | manyforge family; +27% tps, +1.4× ctx |

Full bench results in `PERFORMANCE-V7.md`.
