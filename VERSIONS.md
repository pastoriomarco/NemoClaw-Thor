# NemoClaw-Thor — Current Versions

## Repository version

| Field | Value |
|---|---|
| Repository SemVer | `0.1.0` (see [`VERSION`](VERSION)) |
| Compatible manyforge | `0.1.x` |
| License | MIT — see [`LICENSE`](LICENSE) |

The `VERSION` file is the source of truth for "what NemoClaw-Thor release
this checkout represents." It moves on the SemVer schedule documented in
[`CHANGELOG.md`](CHANGELOG.md). The sections below pin external
dependencies independently — those pins move when external upstreams
release, not on NemoClaw-Thor's own release cadence.

---

Single source of truth across the three external-dependency scopes this
repository owns. When a pin changes, update **only this file**;
`README.md`, `AGENTS.md`, and `USER_QUICKSTART_MANUAL.md` link here
rather than duplicating tables.

| Scope | Where it lives | What it pins |
|---|---|---|
| A | `setup/` | NemoClaw / OpenShell / OpenClaw control-plane tools |
| B | `serving/` | The vLLM container image and its dependency stack |
| C | `manyforge/` | The ManyForge ↔ NemoClaw integration phase |

---

## A. Setup / control plane

External CLIs and sandbox runtime that NemoClaw-Thor's `setup/` scripts target.
Verified by booting a clean profile end-to-end with these versions.

| Component | Verified version | Audit date | Notes |
|---|---|---|---|
| NemoClaw CLI (host) | `lkg` (= `v0.0.73`) | 2026-07-08 | NVIDIA's last-known-good alias moved `v0.0.55` → `v0.0.73` (upstream latest was `v0.0.76`; we track `lkg`). Bumped in-place on the `~/NemoClaw` checkout (`git checkout v0.0.73 && npm install`). Full delta + procedure: [Jul-08 control-plane upgrade](manyforge/docs/CONTROL-PLANE-UPGRADE-2026-07-08.md). |
| OpenShell CLI / gateway / sandbox | `0.0.71` | 2026-07-08 | Blueprint pins `min == max == 0.0.71`; NemoClaw `v0.0.73` **refuses** 0.0.44 at sandbox-create time. Installed by `~/NemoClaw/scripts/install-openshell.sh` (fetches all three aarch64 binaries into `~/.local/bin`, no sudo). **After the binary swap you MUST restart the host gateway** (`nemoclaw <sandbox> recover`) — the in-place install leaves the old 0.0.44 daemon running in memory, and sandboxes created under it crash-loop on `no sandbox token source available`. See upgrade record. |
| OpenShell driver | `docker` (no k3s) | 2026-07-08 | 0.0.71 adds a TLS-secured supervisor relay + per-sandbox auth token; a sandbox created under a stale/old gateway will not provision. Gateway endpoint is still plaintext HTTP on `127.0.0.1:8080`. |
| OpenClaw (in-sandbox) | `v2026.5.22` | 2026-07-08 | **Unchanged** across `v0.0.55` → `v0.0.73`: sandbox image digest (`sha256:b3d832b596…`) and `min_openclaw_version` (`2026.3.11`) are byte-identical in both blueprints — this upgrade forces **no** OpenClaw sandbox rebuild. |
| Hermes agent (Hermes lane) | `v0.17.0` (calver `v2026.6.19`) | 2026-07-08 | Derived from the NemoClaw pin (`agents/hermes/Dockerfile.base` `HERMES_SEMVER`); moved `v0.14.0` → `v0.17.0` with the CLI bump. Required three `setup-hermes.sh` re-baseline fixes (Dockerfile COPY-source staging, dashboard port, secret-boundary env) — see upgrade record. |

Prior audit (kept for diff context):

| Component | Verified version | Audit date |
|---|---|---|
| NemoClaw CLI (host) | `lkg` (= `v0.0.55`) | 2026-06-02 |
| OpenShell CLI | `0.0.44` | 2026-06-02 |
| OpenClaw (in-sandbox) | `v2026.5.22` | 2026-06-02 |
| NemoClaw CLI (host) | `v0.0.31` | 2026-04-30 |
| OpenShell CLI | `0.0.36` | 2026-04-30 |
| OpenShell cluster image | `0.0.36` | 2026-04-30 |
| OpenClaw (in-sandbox) | `v2026.4.24` | 2026-04-30 |

Detailed onboarding workflow: [`setup/NEMOCLAW-OPENCLAW-WORKFLOW.md`](setup/NEMOCLAW-OPENCLAW-WORKFLOW.md).

---

## B. Model-serving container

| Image generation | Status | Notes |
|---|---|---|
| **Entrpi DS4 v0.5.6.2 + Thor profile** | **healthy; Chat and Responses smoke passed (2026-08-10), with v0.5.5 deep/concurrency baseline retained** | Isolated opt-in Docker service on `:8050`, not wired into ManyForge `:8000`. Source pin `027714a4c290a756ef3e6ca557426528745f2033`; CUDA 13.0.1 bases are digest-pinned and compilation is strictly `sm_110`. Uses the 0731 base + matching DSpark drafter only (`--no-mtp`). The default `runtime-thor-v056` image retains Entrpi's repaired streaming top-512 selector and only the Thor profile/attention diagnostic patches; the prior local tool-budget patch is upstream. The profile remains 512K context, two banks, 4K effective chunks, 12 GiB VMM budget, and an 8 GiB live-memory floor. v0.5.6.2 adds continuous OpenAI Responses/Anthropic serving and agent reconnect, transcript, and stream-idle fixes. The v0.5.5 baseline remains the deep evidence: exact 244,518-token retrieval, five two-caller and two four-caller waves without fallback/OOM/Xid/restart, plus the prior 479,817-token capacity proof. Runbook: [`serving/docs/DS4-ON-THOR.md`](serving/docs/DS4-ON-THOR.md). |
| **`v9.1`** | **last shipped (2026-05-30)** | Same dependency stack as v9, but vLLM is pinned to `main @ 3fd9d2d35` (0.22.1rc1.dev22, 180 commits past the v0.22.0 tag) to pick up [PR #42124](https://github.com/vllm-project/vllm/pull/42124) "Add LM head quantization support for ModelOpt" (merged 2026-05-26). v9 (vLLM 0.22.0) crashes at 67% loading `nvidia/Qwen3.6-35B-A3B-NVFP4`; #42124 adds `quant_config` to `ParallelLMHead` for Qwen3.5/Qwen3.6 and Nemotron-H. Tagged `nemoclaw-thor/vllm:latest` (the `start-model.sh` default). See [`serving/docs/V9.1-IMAGE-NOTES.md`](serving/docs/V9.1-IMAGE-NOTES.md). |
| `v9` | built, superseded by v9.1 | Major bump (vLLM 0.20.1 → 0.22.0 skipping v0.21 as waypoint) + FlashInfer / flash-attn-4 / transformers / cuDNN minor bumps. See per-pin table below. Superseded because vLLM 0.22.0 cannot load Qwen3.6-35B-A3B-NVFP4 (pre-#42124). |
| `v8.1` | superseded (2026-05-06) | vLLM 0.20.1 + FlashInfer 0.6.10 + flash-attn-4 b12 + transformers 5.8.0 + cutlass-dsl 4.5.0 + cuDNN 9.21.1.3. Carries the v8 baseline (sm_110 build target, SM100+ spec-decode fix, TQ+FA prefill, MRv2 acceptance) plus PTX FP32→FP4 codegen and NVFP4 KV path. |
| `v8` | superseded (2026-04-29) | hygiene release on top of v7 (apt cuDNN drop, audio deps, transformers 5.7.0) |
| `v7` | superseded | full-rebuild generation; introduced TurboQuant + DFlash on SM110 |

Build invocation for the shipped v9.1 image (vLLM `main` snapshot for #42124,
reusing the cached FlashInfer 0.6.12 wheel):

```bash
IMAGE_GEN=v9.1 ./serving/docker/build-vllm.sh \
  --vllm-ref 3fd9d2d35714e80b4cb3fcd3c408a0398fa2525f --skip-flashinfer
```

The canonical v9 (tagged-release) invocation it derives from:

```bash
./serving/docker/build-vllm.sh --vllm-ref v0.22.0 --flashinfer-ref v0.6.12
```

> **Pin-it-back-to-a-tag opportunity:** vLLM `v0.23.0` (stable, 2026-06-15)
> includes #42124, so a future v10 could drop the `main`-commit pin and return
> to a tagged release while keeping the Qwen3.6-35B-A3B-NVFP4 fix. Gate on the
> smoke corpus before promotion (0.23.0 flips MRv2-default for Llama/Mistral
> dense and carries DeepSeek-V4 churn).

Per-pin status (v8.1 → v9 transitions; v9.1 changes only the vLLM ref, all
other pins identical to v9):

| Pin | v8.1 (shipped) | **v9 (staged)** | Notes |
|---|---|---|---|
| vLLM | `v0.20.1` | **`v0.22.0`** | major bump (skipping v0.21 as a waypoint) — batch-invariant Cutlass FP8 path (+28.9% E2E on FP8-KV decode, directly applies to cosmos-reason2-8b production), CutlassFP8 padding pre-processing (+13.5% TTFT), Qwen3.5 ViT full CUDA graph, stream-aware allocator (long-running gateway stability), streaming tool dispatch primitives (#40700, #41110 — basis for Bridge Tier 1.1), XGrammar 0.2.0 structural tags (#40894), FP8-on-Thor formalization (#39712 — removes launch.sh SM-guard workarounds), Qwen3-VL deepstack heavy-load fix. Breaking changes (benign): C++20 compiler (GCC 13 supports), transformers v4 deprecation. |
| FlashInfer | `v0.6.10` | **`v0.6.12`** | bump — XQA kernel fixes for multi-iteration scenarios (connects to FlashRT-on-Thor XQA FP8-KV transferable opt from audit); per-token NVFP4 quantization kernel perf; "limit sm110 builds to aarch64" CI hardening. NOTE: vLLM 0.22 bundles 0.6.11.post2; we override to 0.6.12 via wheel-cache override (see PATCHES-AUDIT.md). |
| flash-attn-4 | `4.0.0b12` | **`4.0.0b15`** | bump — b15 (2026-05-27) added "Include sm_110 in Blackwell-family arch gating" (first commit-level signal of first-class Thor support); b13/b14 carry varlen blocksparsity and varlen-paged-KV split-config fixes. |
| transformers | `5.8.0` | **`5.9.0`** | bump — "support for custom field prefilling (reasoning_content, thinking, etc.) in chat template handling" — template-side primitive for Bridge Tier 2 reasoning-block preservation pattern. Bridge can opt into reasoning_content prefill via chat_template_kwargs (subject to per-model template support; verify Qwen3-VL template support on cosmos-reason2-8b after build). |
| nvidia-cudnn-cu13 | `9.21.1.3` | **`9.23.0.39`** | minor bump (both stages). Pin subject to vLLM's transitive cuDNN constraint at install time; documents requested version as current latest. |
| nvidia-cutlass-dsl | `4.5.0` | **`4.5.2`** | bump — two patch releases on the 4.5.0 stable line; accumulated bug fixes. CUDA-13.2 codegen win still gated on JetPack 7.2. |
| nvidia-nvshmem-cu13 | `3.6.5` | `3.6.5` | already-latest (held) |
| apache-tvm-ffi | `0.1.11` | `0.1.11` | already-latest (held) |
| fastsafetensors | `0.3.1` | **`0.3.2`** | bump — single patch release; paired with instanttensor 0.1.9 for the fast-load codepath. |
| instanttensor | `0.1.8` | **`0.1.9`** | bump — single patch release; paired with fastsafetensors 0.3.2 for the --load-format instanttensor boot path. |
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
| Phase 2 — OpenClaw assistant-provider adapter | **production default lane** | first live: 2026-05-05; lane default flip: 2026-05-07; model default aligned 2026-06-11 | `manyforge/openclaw_assistant_bridge/` speaks the Composer assistant-provider HTTP contract and dispatches into the OpenClaw runtime in `my-assistant`. Composer chat → OpenClaw gateway → mode-scoped MCP → `/api/assistant/bridge/tools/{toolId}` is the production assistant request path on this stack. Current clean-start default is `demo-assistant-known-good.sh ASSISTANT_PROVIDER=openclaw MODEL_PROFILE=gemma4-12b-it-gguf`. The historical Cosmos anchor remains available for reruns; current default evidence is the 2026-06-07/06-09 Gemma QAT smoke corpus (75 total / 66 scored) in `manyforge/docs/PHASE-5-PRODUCTION-DECISION.md`. The direct model lane (`direct`, with `nemoclaw` accepted as a legacy alias) remains supported as a faster fallback path but is no longer the default. |

Active artifacts:

- Provisioner: [`manyforge/setup-manyforge-assistant.sh`](manyforge/setup-manyforge-assistant.sh)
- Egress preset: [`manyforge/policies/manyforge-composer.preset.yaml`](manyforge/policies/manyforge-composer.preset.yaml)
- **OpenClaw-lane assistant-provider adapter (production default)**: [`manyforge/openclaw_assistant_bridge/`](manyforge/openclaw_assistant_bridge/) on `:8200`
- Bridge audit log mount point: [`manyforge/bridge/`](manyforge/bridge/)
- **Direct-lane bridge** (`manyforge_assistant_bridge`, `:8100`) lives in the sibling `manyforge` repo and is NOT in this repo. It is the supported fallback transport, not the default.

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
