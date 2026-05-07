# NemoClaw-Thor — Current Versions

Single source of truth across the three scopes this repository owns. When a
version changes, update **only this file**; `README.md`, `AGENTS.md`, and
`USER_QUICKSTART_MANUAL.md` link here rather than duplicating tables.

| Scope | Where it lives | What it pins |
|---|---|---|
| A | `setup/` | NemoClaw / OpenShell / OpenClaw control-plane tools |
| B | `serving/` | The vLLM container image and its dependency stack |
| C | `manyforge/` | The ManyForge ↔ NemoClaw integration phase |

---

## A. Setup / control plane

External CLIs and sandbox runtime that NemoClaw-Thor's `setup/` scripts target.
Verified by booting a clean profile end-to-end with these versions.

| Component | Verified version | Audit date |
|---|---|---|
| NemoClaw CLI (host) | `v0.0.31` | 2026-04-30 |
| OpenShell CLI | `0.0.36` | 2026-04-30 |
| OpenShell cluster image | `0.0.36` | 2026-04-30 |
| OpenClaw (in-sandbox) | `v2026.4.24` | 2026-04-30 |

Detailed onboarding workflow: [`setup/NEMOCLAW-OPENCLAW-WORKFLOW.md`](setup/NEMOCLAW-OPENCLAW-WORKFLOW.md).

---

## B. Model-serving container

| Image generation | Status | Notes |
|---|---|---|
| **`v8.1`** | staged in source, not yet built | vLLM/FlashInfer/flash-attn-4/cuDNN/tvm-ffi bumps (see table below) |
| `v8` | last shipped (2026-04-29) | hygiene release on top of v7 (apt cuDNN drop, audio deps, transformers 5.7.0) |
| `v7` | superseded | full-rebuild generation; introduced TurboQuant + DFlash on SM110 |

Build invocation for the canonical v8.1 image:

```bash
./serving/docker/build-vllm.sh --vllm-ref v0.20.1 --flashinfer-ref v0.6.10
```

Per-pin status:

| Pin | v8 (shipped) | **v8.1 (staged)** | Notes |
|---|---|---|---|
| vLLM | `v0.20.0` | **`v0.20.1`** | bump — PTX FP32→FP4 codegen, CUDA-graph batched-token capture fix, KV-block override fix |
| FlashInfer | `v0.6.9` | **`v0.6.10`** | bump — NVFP4 KV cache (SM80+), autotuner correctness + bucketing perf, vLLM OOB fix |
| flash-attn-4 | `4.0.0b10` | **`4.0.0b12`** | bump — b11 added first-class hd256 in CUTE DSL, 3–9% hd256 perf, SM100 MLA stream + empty-tile fixes; b12 is the rolling post-b11 fix tail |
| nvidia-cudnn-cu13 | `9.20.0.48` | **`9.21.1.3`** | bump (both stages) — safe now that v8 fixed the apt-vs-pip mismatch |
| apache-tvm-ffi | `0.1.10` | **`0.1.11`** | single patch release |
| transformers | `5.7.0` | **`5.8.0`** | bump — rolling minor; smoke-test 5.7.0-known-good areas (Qwen3.5 GDN, Gemma4 rotary, KV-dedup ≥16K, NVFP4+torchao) after build |
| nvidia-nvshmem-cu13 | `3.6.5` | `3.6.5` | already-latest |
| nvidia-cutlass-dsl | `4.4.2` | **`4.5.0`** | bump to stable — 4.5.0 release-line headline ("optimal codegen for CUDA 13.2") does NOT apply at CUDA 13.0.3, but accumulated 4.4.2 → 4.5.0 bug fixes do; revisit codegen win after JetPack 7.2 |
| fastsafetensors | `0.3` | **`0.3.1`** | single patch release |
| instanttensor | `0.1.8` | `0.1.8` | already-latest |
| triattention | `@325297218a` | `@325297218a` | held — HEAD is one README-only commit ahead |
| torch | `2.13.0.dev20260426+cu130` | `2.13.0.dev20260426+cu130` | held — CUDA-coupled |
| torchvision | `0.27.0.dev20260426+cu130` | (held) | tied to torch pin |
| torchaudio | `2.11.0.dev20260426+cu130` | (held) | tied to torch pin |
| CUDA base image | `nvidia/cuda:13.0.3-devel-ubuntu24.04` | (held) | bump with JetPack 7.2 |

Authoritative source for the build pipeline:
[`serving/docker/Dockerfile.vllm`](serving/docker/Dockerfile.vllm) header.
Patches/audit history: [`serving/docker/PATCHES-AUDIT.md`](serving/docker/PATCHES-AUDIT.md),
[`serving/docker/NOTES.md`](serving/docker/NOTES.md).

---

## C. ManyForge ↔ NemoClaw integration

No semver yet — phase-tagged as work lands.

| Phase | Status | Landed | Notes |
|---|---|---|---|
| **Phase 1 — MCP integration** | shipped | 2026-05-04 (`1136a16`) | Custom egress preset for `host.openshell.internal:9000`; idempotent provisioner stages the `manyforge-composer` skill and registers it as an MCP server in the `my-assistant` sandbox |
| Phase 2 — OpenClaw assistant-provider adapter | experimental live route validated | 2026-05-05 | `manyforge/openclaw_assistant_bridge/` speaks the Composer assistant-provider HTTP contract and invokes `openclaw agent` in `my-assistant`; Composer chat → OpenClaw → mode-scoped MCP → `/api/assistant/bridge/tools/{toolId}` was live-smoked with `catalog.read` and `tree.draft.wrap_node`. Direct vLLM remains the known-good default until the A/B harness qualifies reliability and latency. |

Active artifacts:

- Provisioner: [`manyforge/setup-manyforge-assistant.sh`](manyforge/setup-manyforge-assistant.sh)
- Egress preset: [`manyforge/policies/manyforge-composer.preset.yaml`](manyforge/policies/manyforge-composer.preset.yaml)
- Experimental OpenClaw adapter: [`manyforge/openclaw_assistant_bridge/`](manyforge/openclaw_assistant_bridge/)
- Bridge audit log mount point: [`manyforge/bridge/`](manyforge/bridge/) (the bridge service code itself lives in the sibling `manyforge` repo at `manyforge_assistant_bridge/`)

Wire contract / spec authority: in the sibling `manyforge_specs` repo, not here. See:

- `manyforge_specs/docs/spec/480-assistant-modes-and-bounded-autonomy.md`
- `manyforge_specs/docs/spec/485-assistant-bridge-architecture.md`
- `manyforge_specs/docs/reference/ASSISTANT_BACKEND_CONTRACT.md`

Companion docs in this repo: [`manyforge/docs/MANYFORGE-MCP-INTEGRATION.md`](manyforge/docs/MANYFORGE-MCP-INTEGRATION.md),
[`manyforge/docs/MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md`](manyforge/docs/MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md),
[`manyforge/docs/MANYFORGE-PROFILE-CALIBRATION.md`](manyforge/docs/MANYFORGE-PROFILE-CALIBRATION.md).

---

## Maintenance

- Update on each tested upgrade. The audit dates carry the load-bearing
  meaning: "this combination was verified end-to-end on this date."
- "Held" pins document a deliberate decision, not neglect — keep the rationale
  inline so a future reader doesn't have to re-derive it.
- When a scope's authoritative document moves or is replaced, update the
  links here in the same commit as the move.
