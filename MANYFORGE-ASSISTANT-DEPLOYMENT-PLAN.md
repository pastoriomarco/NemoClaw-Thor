# ManyForge Assistant — LLM stack deployment plan for Jetson Orin AGX

**Status:** PENDING — tests deferred until the manyforge-assistant agent is
operative on Thor. This document captures the future evaluation plan.

**Target hardware:** Jetson Orin AGX Dev Kit (64 GB unified memory, SM86
Ampere — *different from Thor SM110a*).

**Memory budget:** **≤ 40 GB total for the LLM stack** so ≥ 24 GB stays
free for the Isaac ROS perception/control pipeline running on the same
device.

---

## Goal

Decide which Thor-validated profile (or combination) is the right fit
for the manyforge-assistant agent on Orin AGX, given the 40 GB constraint.

The Thor v7 bench already established the agentic-quality ceiling for
each candidate (see `PERFORMANCE-V7.md`). What's outstanding is
**workload-specific validation against the manyforge-assistant agent's
actual tool calls and pipeline workflows** — and **whether each profile
fits the Orin AGX 40 GB budget at all**.

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
  outcomes A and B (vision present)
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
4. Vision works correctly when needed (A or B; or via Cosmos-2B in C)

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

---

## Open questions

1. **NVFP4 on Ampere SM86**: vLLM's NVFP4 GEMM backend depends on
   FlashInfer-CUTLASS kernels. Need to confirm SM86 instantiations exist
   for the relevant tile shapes. If not, NVFP4 falls back to a slower
   path or fails outright.
2. **TurboQuant K8V4 on Ampere**: the `fix-pr39931-turboquant` mod
   removes a SM110-related guard, but TQ K8V4 itself may have arch
   assumptions. Worth a boot probe before committing to Outcome B or C.
3. **Cosmos-Reason2-8B vision on Ampere**: Thor needs `TORCH_SDPA`
   workaround for SM110 ViT FA2 PTX crash; SM86 may not need this
   workaround (could try FA2 for better ViT throughput).
4. **ManyForge-assistant agent test harness**: the workflow tests above
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
