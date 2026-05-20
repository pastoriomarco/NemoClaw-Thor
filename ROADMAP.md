# NemoClaw-Thor roadmap

This is the direction document for **NemoClaw-Thor's actual scope**: the
Jetson AGX Thor deployment helper that ships the vLLM container, the
model profiles and launch scripts, the OpenShell sandbox onboarding
workflow, and the OpenClaw-lane assistant bridge implementation.

For per-component pinned versions, see [`VERSIONS.md`](VERSIONS.md). For
what's actually shipped, see [`CHANGELOG.md`](CHANGELOG.md). For the
authoritative spec and contract documents that this repo conforms to,
see [`manyforge_specs/docs/`](../../../dev_ws/src/manyforge_specs/docs/)
in the sibling workspace — particularly
[`open-points.md`](../../../dev_ws/src/manyforge_specs/docs/open-points.md)
for upstream pending decisions, and
[`cross-workspace-conventions.md`](../../../dev_ws/src/manyforge_specs/docs/cross-workspace-conventions.md)
for the authority/ownership map across the three workspaces.

Items are organized by **concern lane**, not by release version.
NemoClaw-Thor's SemVer is decoupled from image generation (the
container pins move on their own cadence in
[`VERSIONS.md`](VERSIONS.md)), so version-tagged sections would create
false precision. Each item below describes the outcome and what "done"
looks like, with no timeline commitment.

Current state context:
[`cosmos-reason2-8b`](https://huggingface.co/nvidia/Cosmos-Reason2-8B)
is the production assistant model on the OpenClaw lane (default since
2026-05-07; see
[`manyforge/docs/LANE-COMPARISON-direct-vs-openclaw.md`](manyforge/docs/LANE-COMPARISON-direct-vs-openclaw.md)).
Smoke corpus iter 32 production recipe scores 51/66 = 77.3% with the
bridge-fired `/compact` every 2 prompts.

---

## Serving lane

Container builds, model profiles, launch scripts, profile calibration.

### v8.2 vLLM container

**Goal.** Move from `v8.1` (vLLM 0.20.1) to `v8.2` (vLLM 0.21.0) plus
the small dependent pin bumps, on the same Dockerfile pipeline.

**Why.** vLLM 0.21.0 brings several upstream features directly relevant
to this stack:

- Qwen3-VL deepstack heavy-load fix (matches cosmos-reason2-8b in production)
- Streaming tool-call dispatch with `required` and named tool choice
  (#40700, #41110) — the upstream primitive the deferred Bridge work
  builds on
- XGrammar 0.2.0 structural tags for strict tool calling (#40894)
- FP8-on-Thor formalization (#39712), removing the runtime SM-guard
  workarounds we currently rely on
- Accumulated bug fixes from 0.20.1 → 0.20.2 → 0.21.0

NVFP4 KV cache (#40177) is explicitly **not** in scope for Thor in this
generation — the kernels target Blackwell SM100 only and no SM110a
cubins exist upstream. Filing an RFE upstream is a Notes-section item
(see below), not a milestone.

**Acceptance criteria.**

- `nemoclaw-thor/vllm:v0.21.0-...-v8.2` builds clean from
  [`serving/docker/Dockerfile.vllm`](serving/docker/Dockerfile.vllm) on
  the existing pipeline (no Thor-specific patches beyond what v8.1
  already carries).
- The two breaking changes in 0.21.0 pass cleanly: C++20 compiler
  requirement (GCC 13 in the base image), transformers v4 deprecation
  (we're already on v5.x).
- Smoke corpus regression at the iter-32 production recipe: ≥ 51/66 on
  the existing cosmos-reason2-8b production profile.
- Image generation table in [`VERSIONS.md`](VERSIONS.md) updated to
  reflect v8.2 as shipped.

**Companion pin bumps** (riding along on the rebuild): FlashInfer
`0.6.11.post2`, transformers `5.8.1`, flash-attn-4 `4.0.0.beta13`.
See [`VERSIONS.md`](VERSIONS.md) for the authoritative pin set.

---

## Bridge lane

OpenClaw-lane assistant-provider adapter at
[`manyforge/openclaw_assistant_bridge/`](manyforge/openclaw_assistant_bridge/),
the implementation owned by this repo per
[`cross-workspace-conventions.md`](../../../dev_ws/src/manyforge_specs/docs/cross-workspace-conventions.md).

Recently-shipped bridge work (per-request thinking controls,
single-reasoning-parser audit, Prometheus metrics, circuit breaker) is
tracked in [`CHANGELOG.md`](CHANGELOG.md), not here. This lane lists
only deferred items.

### Streaming tool dispatch at the bridge layer

**Goal.** Surface tool calls to the composer as soon as the model
finishes parsing them, rather than at end-of-turn. Maps directly to the
streaming primitive vLLM 0.21.0 adds upstream (#40700).

**Depends on.** v8.2 container shipped (the upstream feature is what
the bridge will consume).

**Acceptance criteria.**

- Bridge flips `"stream": False` to `"stream": True` in the
  gateway-path chat-completions request and parses the resulting SSE
  stream (currently
  [`adapter.py:754`](manyforge/openclaw_assistant_bridge/adapter.py#L754)).
- Bridge surfaces `tool_call_dispatch` SSE events to composer in real
  time without waiting for `finish_reason: "tool_calls"`.
- Smoke corpus regression at iter-32 recipe: no quality loss on
  cosmos-reason2-8b production profile.
- Tool-call surfacing latency measurable via the Prometheus metrics
  shipped in 2026-05-11 (per-stage histograms).

### Prefix-stable session management (replacing blanket `/compact`)

**Goal.** Replace the current iter-32 "fire `/compact` every 2 prompts"
recipe with targeted, prefix-preserving history management. Recover the
TTFT win that `/compact` destroys without losing the quality `/compact`
maintains.

**Status.** Deliberate **A/B candidate** — not a commitment to ship. The
current `/compact` recipe is load-bearing for the 51/66 score; any
replacement must hold or improve that number without sacrificing the
prefix-cache hit rate.

**Acceptance criteria for a successful A/B.**

- Static prefix audit: system prompt + tool schemas verified
  byte-identical across turns (no timestamps, no per-turn IDs).
- Alternative pruning strategy implemented (tool-output body pruning,
  batch turn drop at stable boundaries) gated by env var so the
  `/compact` recipe remains the default.
- Side-by-side smoke run: tuned strategy ≥ 51/66 on the iter-32 corpus.
- Measured TTFT win: `vllm:prefix_cache_hit_rate` materially higher
  than the `/compact` baseline (target: > 50% mid-session).

If A/B fails the smoke threshold, the `/compact` recipe stays — the
direction was wrong, not the work.

---

## Fine-tuning lane (investigation)

Exploratory. Not a committed roadmap track. Reference plan:
[`serving/docs/COSMOS-REASON2-FINETUNE-PLAN.md`](serving/docs/COSMOS-REASON2-FINETUNE-PLAN.md).

The plan documents a Thor-on-device path for tuning Cosmos-Reason2-2B
or -8B via distillation from the 8B production model, followed by
NVFP4 weight quantization. It exists because the OpenClaw-lane
composer-assistant workload would benefit from a tool-call-tuned
variant, and because NVIDIA's official tooling
([`github.com/nvidia-cosmos/cosmos-reason2`](https://github.com/nvidia-cosmos/cosmos-reason2))
supports Jetson AGX directly.

**Why exploratory, not committed.**

- The data pipeline (bridge audit-log curation + 8B-teacher
  distillation) is the gating cost, not the training itself.
- Whether the 2B base has the capacity to reach the 8B's smoke score
  is an open question — the plan's Step 0 (stock-2B baseline) is
  cheap; the larger investment is gated on that signal.
- The deferred Bridge work (streaming dispatch + prefix-stable
  sessions) likely produces meaningful TTFT wins on the **same model**;
  worth measuring those before investing in model-side improvements.

If the investigation produces a tuned-and-quantized 2B variant that
matches or exceeds 9/9 smoke on the curated set, it graduates to a
profile addition in the Serving lane. Until then, it stays a plan.

---

## Stack lane

External version tracking. Items in this lane are dependencies on
upstream releases; the work is validation + integration, not
implementation.

### OpenShell `0.0.36 → 0.0.41` bump

**Goal.** Update the OpenShell CLI and cluster image past the breaking
v0.0.37 release (pluggable compute drivers + Helm chart) to one of the
v0.0.38–v0.0.41 stable points.

**Why this is non-trivial.** v0.0.37 is a major refactor of the
compute-driver abstraction. The sandbox-onboarding workflow in
[`setup/NEMOCLAW-OPENCLAW-WORKFLOW.md`](setup/NEMOCLAW-OPENCLAW-WORKFLOW.md)
and the network-policy preset in
[`manyforge/policies/manyforge-composer.preset.yaml`](manyforge/policies/manyforge-composer.preset.yaml)
need to be re-validated against the new driver interface.

**Acceptance criteria.**

- Sandbox onboarding succeeds end-to-end on the new OpenShell version.
- Manyforge MCP wrapper registration with the gateway behaves
  identically.
- Smoke corpus regression at iter-32 holds.
- [`VERSIONS.md`](VERSIONS.md) §A updated to the new pinned versions.

### OpenClaw `2026.4.24 → 2026.5.12` bump

**Goal.** Update the in-sandbox OpenClaw agent past the multiple
release line since the v8.1 verification date.

**Why.** Relevant upstream changes include `response_format` forwarding
through agent stream params (could affect bridge tool-calling
correctness) and binding hardening for principal/conversation
correlation.

**Acceptance criteria.**

- Bridge → gateway tool-call flow validated against the new OpenClaw
  release; no regression in the live `bind_principal` callback path.
- Smoke corpus regression at iter-32 holds.
- [`VERSIONS.md`](VERSIONS.md) §A updated.

### JetPack 7.2 SBSA transition

**Goal.** Once JetPack 7.2 ships SBSA-mode support for Thor (target
Q2 2026), restructure the container build so we can drop the
SM110-specific pin set in favor of NVIDIA's official aarch64 packages.

**Acceptance criteria.**

- vLLM container builds on the SBSA path without the SM110-specific
  Dockerfile patches v8.x has accumulated.
- Held pins documented in [`VERSIONS.md`](VERSIONS.md) (currently held
  for "JP7.2 transition" — CUDA base image, possibly the torch
  nightly) move to NVIDIA-official refs.
- Fine-tuning lane gains real footing on Thor (SDFT/QAT via NVIDIA's
  official toolchain — see the FT investigation note above).

The transition is the **single largest open dependency** on this
roadmap. Until it ships, several adjacent items stay parked.

### TRT-Edge-LLM v0.8 (watchlist)

**Goal.** Not committed work. Re-evaluate TRT-Edge-LLM as a possible
inference path when the NVIDIA-acknowledged v0.8 blockers
(concurrency, tool-call) ship.

v0.7.1 (released 2026-05-20) did **not** address either blocker. The
retest pipeline is preserved for a half-day rebuild whenever v0.8
arrives. No expected work in this lane before then.

---

## Notes (informational, not milestones)

Items the team is aware of but not committing to as roadmap deliverables.

- **NVFP4 27B VLM candidate** —
  [`natfii/Qwen3.6-27B-VLM-NVFP4-MTP`](https://huggingface.co/natfii/Qwen3.6-27B-VLM-NVFP4-MTP)
  is a community-built NVFP4 quantization of Qwen3.6-27B with bundled
  MTP weights. Authored against the SM120 target, not validated on
  SM110a. A boot experiment on Thor is a candidate for one-day
  investigation; not roadmap-tracked.
- **Cosmos-Reason2 successors** as NVIDIA ships them (32B variant
  exists; tracking documented in
  [`serving/docs/COSMOS-REASON2-32B-QUANTIZATION.md`](serving/docs/COSMOS-REASON2-32B-QUANTIZATION.md)).
  Profile additions follow `MANYFORGE-PROFILE-CALIBRATION.md`
  methodology when they happen; not a roadmap commitment until a
  successor proves operationally relevant.
- **NVFP4 KV cache on SM110a** — upstream blocked. PR #40177 landed in
  vLLM 0.21 for Blackwell SM100 only; SM110a cubins don't exist in
  NVIDIA's artifactory as of the v8.1 / v8.2 audits. Filing an RFE
  upstream is the next step here, not building around it.
- **Dynamo / TokenSpeed / SMG** — agentic-harness frameworks discussed
  as architectural references for the Bridge lane work. Not
  Thor-compatible today; revisit after JP7.2 SBSA.

---

## Out of scope / explicitly deferred

- **ManyForge kernel, ROS, planning, behavior runtime, composer UI.**
  These belong in
  [`manyforge_specs/`](../../../dev_ws/src/manyforge_specs/) and
  [`manyforge/`](../../../dev_ws/src/manyforge/). NemoClaw-Thor is the
  deployment helper, not the platform.
- **Wire-contract authority.** The assistant-provider HTTP contract
  envelope is owned by
  [`manyforge_specs/docs/reference/ASSISTANT_BACKEND_CONTRACT.md`](../../../dev_ws/src/manyforge_specs/docs/reference/ASSISTANT_BACKEND_CONTRACT.md)
  and
  [`manyforge/manyforge_composer/backend/assistant_provider.py`](../../../dev_ws/src/manyforge/manyforge_composer/backend/assistant_provider.py).
  Changes to the contract are coordinated through `manyforge_specs`
  and `manyforge`, not here.
- **Non-Thor hardware support.** The repo name says "Thor"; non-Thor
  ports (Orin, x86 + discrete GPU) belong in a sibling repo. Patches
  that keep the build tolerant of non-SM110 targets are welcome,
  roadmap items they are not.
- **Cloud-only deployments.** NemoClaw-Thor is a self-hosted inference
  stack; cloud-API drop-ins (OpenAI, Anthropic, Gemini) are out of
  scope for this repo.
- **Independent product SemVer.** Per [`VERSIONS.md`](VERSIONS.md), the
  three scopes this repo owns (setup / serving / manyforge integration)
  are pinned independently. NemoClaw-Thor's own `VERSION` moves when
  the repo's public surface changes, not on every vLLM bump.

---

## Authority pointers (so this roadmap stays grounded)

| Concern | Authoritative source |
|---|---|
| Pin set, image generations | [`VERSIONS.md`](VERSIONS.md) |
| What's shipped | [`CHANGELOG.md`](CHANGELOG.md) |
| Cross-workspace ownership map | [`cross-workspace-conventions.md`](../../../dev_ws/src/manyforge_specs/docs/cross-workspace-conventions.md) |
| Upstream pending decisions | [`open-points.md`](../../../dev_ws/src/manyforge_specs/docs/open-points.md) |
| Assistant-provider HTTP contract | [`manyforge_specs/docs/reference/ASSISTANT_BACKEND_CONTRACT.md`](../../../dev_ws/src/manyforge_specs/docs/reference/ASSISTANT_BACKEND_CONTRACT.md) |
| Mode taxonomy + bridge architecture | [`manyforge_specs/docs/spec/480-...md`](../../../dev_ws/src/manyforge_specs/docs/spec/) and `485-...md` |
| OpenClaw-lane bridge implementation (this repo) | [`manyforge/openclaw_assistant_bridge/`](manyforge/openclaw_assistant_bridge/) |
| Setup workflow | [`setup/NEMOCLAW-OPENCLAW-WORKFLOW.md`](setup/NEMOCLAW-OPENCLAW-WORKFLOW.md) |

If a concern is not in this table, the relevant workspace's
`AGENTS.md` is the entry point.
