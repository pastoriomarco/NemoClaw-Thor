# NemoClaw-Thor roadmap

This is the public direction document for NemoClaw-Thor — the
Thor-hardware-specific layer that hosts vLLM serving, the OpenClaw
sandbox bridge, and the ManyForge integration glue. For per-component
pinned versions, see [`VERSIONS.md`](VERSIONS.md). For what's actually
shipped, see [`CHANGELOG.md`](CHANGELOG.md).

The roadmap is structured by **what** before **when**: items below
describe outcomes Thor is working toward, in rough priority order. Each
milestone names what it depends on and what "done" looks like.

## Companion roadmap

ManyForge (the orchestration layer Thor serves) publishes its own
roadmap focused on the kernel, scene authoring, and assistant modes.
See
[manyforge ROADMAP](https://github.com/pastoriomarco/manyforge/blob/main/ROADMAP.md).
The two roadmaps coordinate version pins via `VERSIONS.md` §C; check
both before planning a release that crosses the boundary.

## 0.2.x — v8.1 vLLM container shipped

**Goal.** Move the staged `v8.1` container from in-progress to
shipped, with `gpt-5.5` reachable from the OpenClaw bridge.

**Acceptance criteria.**

- vLLM 0.20.1 + FlashInfer 0.6.10 + flash-attn-4 4.0.0b12 produce
  a clean cold-boot on Thor SM110a (target: ~3 min after the
  one-time ~69-min FlashInfer JIT compile completes).
- gpt-5.5 model profile validated end-to-end: ChatGPT-auth Codex
  CLI 0.130.0+ users can hit `model = "gpt-5.5"` without "requires
  newer version" errors.
- Smoke corpus retest at iter-32 production recipe maintains or
  improves the 51/66 = 77.3% baseline.

## 0.2.x — Cosmos-Reason2 successor + multimodal lane

**Goal.** Track Cosmos-Reason2 successors as they ship. Add a
qualified multimodal lane (vision input) usable from the Composer
assistant for scene reasoning over Isaac sim renders.

**Acceptance criteria.**

- New profile in `serving/config.sh` for the multimodal model.
- vLLM container handles image input transports
  (`inline_data_url`, `http_url`) without bridge regression.
- Lane-comparison probe shows the multimodal lane within 2× of the
  text-only lane on assistant tasks that don't need vision.
- ManyForge `inspect_isaac_scene` tool wired to an actual screenshot
  capture (currently text-only — the multimodal lane unlocks the
  visual branch).

## 0.3.x — Thor SBSA transition (JP7.2, Q2 2026)

**Goal.** Once JetPack 7.2 ships SBSA-mode support for Thor,
restructure the container build so NemoClaw-Thor uses official
NVIDIA aarch64 packages instead of the custom SM110 pin set.

**Acceptance criteria.**

- vLLM container builds on the SBSA path without custom CMake
  flags for SM110.
- TurboQuant SM110 port is unnecessary (or transitions to a
  vLLM-upstream PR).
- Fine-tuning lane: SDFT/QAT becomes available on Thor hardware
  using NVIDIA Dynamo + the SBSA toolchain. Tracking link:
  manyforge ROADMAP §"0.4.x — fine-tuning + agentic loop".
- Broader NVIDIA agentic-orchestration tooling (Dynamo, TRT-Edge-LLM
  v0.8+) can be evaluated as a vLLM replacement once
  [`project_trt_edge_llm_roadmap`](https://github.com/pastoriomarco/manyforge_specs)
  blockers are resolved upstream.

## 0.3.x — bridge observability + audit

**Goal.** Promote the OpenClaw assistant bridge from "validated"
to "production-supervisable."

**Acceptance criteria.**

- Bridge `/healthz` reports structured component readiness instead
  of a binary OK/fail.
- `audit.jsonl` log format documented in `manyforge/openclaw_assistant_bridge/README.md`
  with a stable schema versioned per spec 485.
- Bridge-fired `/compact` cadence configurable per deployment
  (currently pinned at N=2 for OpenClaw).
- Smoke harness output emits per-task JSON for downstream tooling.

## 0.4.x — fine-tuning lane

**Goal.** Use the JP7.2 SBSA transition to host SDFT/QAT fine-tuning
of the assistant model on operator-edited program corpora.

**Acceptance criteria depend on Thor SBSA + manyforge's
operator-corpora collection lane (manyforge ROADMAP §"0.4.x").**

## 1.0.0 — production readiness

The bar for 1.0:

1. v8.1 container stable on Thor SM110a with `gpt-5.5` and the
   multimodal lane both qualified.
2. The OpenClaw production lane has been operating against real
   robot deployments (not just smoke corpora) for at least one
   minor release cycle.
3. `VERSIONS.md` pins are NVIDIA-stable refs (release tags), not
   commit hashes or staging branches.
4. SECURITY.md threat model has been reviewed for any deployment
   exposing the bridge beyond an isolated subnet.
5. Bridge wire contract (`docs/reference/ASSISTANT_BACKEND_CONTRACT.md`
   in the manyforge_specs repo) is version-tagged and stable enough
   that breaking changes are 1.x → 2.x transitions.

## Out of scope / explicitly deferred

- **Independent serving versioning.** `VERSIONS.md` already pins
  serving, setup, and manyforge integrations independently. The
  repo SemVer (`VERSION`) moves when the public surface of
  NemoClaw-Thor itself changes — not every time vLLM bumps.
- **Non-Thor hardware support.** The repo name says "Thor"; non-Thor
  ports (Orin, x86 + discrete GPU) would belong in a sibling repo,
  not here. Patches to make the build tolerant of non-SM110 targets
  are welcome but not roadmap items.
- **Cloud-only deployments.** NemoClaw-Thor is a self-hosted
  inference stack; cloud-API drop-in replacements (OpenAI, Anthropic,
  Gemini) are out of scope.

## Contributing to the roadmap

Roadmap items are not "claimed" — multiple contributors can work in
parallel on the same milestone. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the branch + PR workflow. New milestones go through an issue with
the `roadmap` label first.
