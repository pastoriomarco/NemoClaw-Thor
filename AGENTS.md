# AGENTS.md — NemoClaw-Thor scope and intended workflows

> **For LLMs (Claude / ChatGPT) helping the user with this repo:** read this
> file first. It defines what this repo does and does not own, points you
> to the operational docs, and lists the boundary rules that protect the
> user from repeating known mistakes. Read referenced docs *before*
> editing code or running scripts.

---

## Purpose

This repository provides a tested, reproducible setup for running
**NemoClaw** on **Jetson AGX Thor (SM110)** with local LLM inference, in
service of the **ManyForge assistant pipeline**.

It scopes to two concerns:

1. **NemoClaw onboarding on Thor** — a documented (ideally LLM-assisted)
   path from a clean Thor host to a working sandbox running OpenClaw or
   Hermes against a local vLLM endpoint.
2. **LLM serving for NemoClaw** — Thor-tuned vLLM and TRT-Edge-LLM
   container builds, model profiles, and launch scripts that expose an
   OpenAI-compatible `/v1` endpoint consumable by NemoClaw and (downstream)
   ManyForge.

The downstream consumer is the ManyForge composer running an assistant
agent backed by these models. This repo also hosts the **assistant-provider
bridge** that translates ManyForge's contract into model dispatches; see
"ManyForge integration" below.

---

## What this repo owns vs consumes

**Owns** (edit freely, version in this repo):

- `docker/` — Thor-specific vLLM and TRT-Edge-LLM container builds and
  Dockerfile-side runtime patches under `docker/mods/`.
- `lib/` — model profiles, vLLM launch arg matrix, sandbox prereqs.
- `start-model.sh`, `start-duo.sh`, `configure-local-provider.sh` — host
  entry points wrapping the above.
- `agentic-bench/` — runtime-agnostic benchmark harness.
- Operational docs in this repo (workflow, deployment plan, performance
  reports, investigations).

This repo does **not** own the ManyForge assistant-provider bridge
service. The bridge implements ManyForge's HTTP contract and lives in
the ManyForge repo (`manyforge/manyforge_assistant_bridge/`); it
consumes the vLLM endpoint this repo's launch scripts expose. The
bridge's architectural design lives alongside the contract in
`manyforge_specs/docs/spec/485-assistant-bridge-architecture.md`.

**Consumes** (don't reimplement; don't fork; configure and wrap):

- **NemoClaw** (`NVIDIA/NemoClaw`) — host CLI + sandbox onboarding.
- **OpenShell** (`NVIDIA/OpenShell`) — host CLI + cluster gateway
  container providing the k3s + sandbox runtime.
- **OpenClaw** (`openclaw/openclaw`) — in-sandbox coding agent.
- **Hermes Agent** (`NousResearch/hermes-agent`) — alternative
  in-sandbox agent (NemoClaw added support pre-v0.0.18; status: viable
  but bumpy — prefer OpenClaw for production today).
- **vLLM**, **TRT-Edge-LLM**, **HuggingFace models** — upstream binaries.

**Sibling workspaces** (read-but-don't-edit boundaries; cross-cutting
work happens via the coordination protocol in
`manyforge_specs/docs/cross-workspace-conventions.md`):

- **`manyforge`** at `dev_ws/src/manyforge/` — ManyForge composer +
  behavior runtime + assistant-provider HTTP contract code +
  `manyforge_assistant_bridge/` runtime service. The wire shape and
  runtime services for the assistant pipeline live there, not here.
- **`manyforge_specs`** at `dev_ws/src/manyforge_specs/` —
  architectural specs + ADRs + contract specifications. Normative for
  both this repo and `manyforge`. When a behaviour question crosses
  workspace boundaries, the spec layer is the tie-breaker.

This repo's role in the trio is the **deployment helper** layer:
hardware-specific build/launch/onboard scripts and Thor-tuned model
profiles. It does not own the ManyForge contract, the bridge
implementation, or any runtime service that fulfils the
assistant-provider contract.

---

## Scope shift — 2026-04 reboot, do not regress

In v1–v3 this repo overrode large parts of NemoClaw config to force
agentic workflow interaction. **v4 (2026-04-06)** rebooted to the
current scope. The override approach is gone and stays gone.

Concretely:

- **Don't** re-add a restream proxy. NemoClaw + OpenShell L7 proxy
  handle routing now.
- **Don't** patch NemoClaw's npm sources. If a feature is missing,
  file an upstream issue; if a bug bites us, document the *workaround*
  (not a fix) and check upstream for a release.
- **Don't** write scripts to bypass `nemoclaw onboard`. Onboarding can
  require manual intervention (license acceptance, HF token, device
  attestation) — that's acceptable.
- **Do** keep our wrappers thin and our docs explicit. The ratio of
  `# explanation` to `command` in our shell scripts is intentional.

---

## Stack chain

```
┌─────────────────────────────────────────────────────────────────┐
│ Host (Jetson AGX Thor, SM110, JetPack 7.x, Docker + sudo)        │
│                                                                  │
│  ┌─ nemoclaw CLI (npm, host)                                     │
│  │     ↓ onboard / dispatch / policy / channels                  │
│  ├─ openshell CLI (host)  ────►  openshell-cluster-nemoclaw      │
│  │                                (Docker, embeds k3s + gateway) │
│  │                                       │                       │
│  │                                       ▼                       │
│  │                                   sandbox pods                 │
│  │                                   ┌─ <sandbox-name> ───────┐  │
│  │                                   │  Landlock + seccomp    │  │
│  │                                   │  in-sandbox agent:     │  │
│  │                                   │   OpenClaw (default)   │  │
│  │                                   │   Hermes (optional)    │  │
│  │                                   │  in-sandbox gateway    │  │
│  │                                   │   :18789 (OpenClaw)    │  │
│  │                                   │   :8642  (Hermes)      │  │
│  │                                   └────────┬───────────────┘  │
│  │                                            │ HTTP via         │
│  │                                            │ host.openshell.  │
│  │                                            │ internal:8000    │
│  │                                            ▼                  │
│  └─ vLLM container (this repo)  ────►  /v1/chat/completions      │
│        nemoclaw-thor/vllm:latest        served at 127.0.0.1:8000 │
│        (this repo: docker/, lib/, start-model.sh)                │
└─────────────────────────────────────────────────────────────────┘

                                       ▲
                                       │ ManyForge composer (downstream)
                                       │ POSTs assistant-provider contract
                                       │ to the bridge service (lives in
                                       │ manyforge/manyforge_assistant_bridge/,
                                       │ NOT this repo); the bridge
                                       │ enforces the active mode's catalog
                                       │ and dispatches to the local vLLM
                                       │ endpoint above. The sandbox is for
                                       │ interactive OpenClaw use, not the
                                       │ assistant request path. See
                                       │ manyforge_specs spec 485 for the
                                       │ bridge architecture.
```

**Verified working version pins.** Update on each tested upgrade. The
`NEMOCLAW-OPENCLAW-WORKFLOW.md` version table is the more detailed
record; this is the entry-point summary.

| Component | Verified version | Audit date |
|---|---|---|
| NemoClaw CLI (host) | `v0.0.31` | 2026-04-30 |
| OpenShell CLI | `0.0.36` | 2026-04-30 |
| OpenShell cluster image | `0.0.36` | 2026-04-30 |
| OpenClaw (in-sandbox) | `v2026.4.24` | 2026-04-30 |

Latest-at-audit is the same as verified for these components on this
date. Drift will reappear; expect to do a small upgrade pass periodically.

---

## Workflows by intent

### A — First-time NemoClaw onboarding on a clean Thor

Manual steps (these stay manual; do not script around them):

1. Install Docker, accept NVIDIA runtime prompts.
2. Install Node.js + the `nemoclaw` and `openshell` CLIs. Pin to the
   verified versions above unless explicitly upgrading. The
   NemoClaw repo ships `scripts/install-openshell.sh` which handles the
   OpenShell CLI bump within the supported version range.
3. Save HF token to `~/.cache/huggingface/token` for gated repos.
4. Accept gated-model licenses on huggingface.co for the profiles you
   intend to use.
5. Pre-pull model weights with `hf download <repo>` — vLLM cold starts
   faster from a populated cache.
6. Start the model first (`./start-model.sh <profile>`), then run
   `nemoclaw onboard` — the onboarding wizard probes the inference
   endpoint as part of step 3/8.

When NemoClaw's onboarding wizard hangs or fails on a specific step,
the answer is to upgrade NemoClaw upstream rather than work around it.
Most onboarding regressions are fixed within 1–2 monthly NemoClaw
releases.

The detailed wizard walkthrough — exact prompt answers that produce a
working `my-assistant` sandbox on Thor (inference option, base URL,
served-model name, sandbox name, web-search/messaging skip, policy tier
choice, minimum-egress preset selection, dashboard token handling) —
lives in [NEMOCLAW-OPENCLAW-WORKFLOW.md](NEMOCLAW-OPENCLAW-WORKFLOW.md)
under "First-time NemoClaw onboard recipe". Read it before running
`nemoclaw onboard` on a clean Thor.

### B — Per-session: serve a model and wire NemoClaw to it

```bash
cd /home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor

# 1. Start the model (one of the profiles in lib/config.sh)
./start-model.sh <profile-slug>

# 2. Wire OpenShell to route the sandbox at it
./configure-local-provider.sh <profile-slug>

# 3. Use it (interactive)
nemoclaw <sandbox-name> connect

# Or scripted — see NEMOCLAW-OPENCLAW-WORKFLOW.md "Scripted / non-interactive"
```

**Naming invariant.** The served-model-name advertised by vLLM equals
the profile slug. A consumer (ManyForge, OpenShell route, BFCL, etc.)
only needs to know the slug. The mechanism lives in `lib/config.sh`'s
case-statement branches and `lib/launch.sh`'s `--served-model-name`
flag construction.

### C — End-of-session cleanup

```bash
docker stop <vllm-container-name>
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
free -g     # should report ~110+ GB available
```

The `drop_caches` step is part of the Thor memory protocol — vLLM can
leave unified-memory pages pinned after a stop or crash.

### D — Bench candidates

See `agentic-bench/README.md` for the harness, current candidate
shortlist, and bench-menu rationale.

---

## ManyForge assistant pipeline integration

ManyForge is the downstream consumer of this repo's serving stack.
The integration has three pieces, all of which are user-driven — not
automated by this repo's scripts:

1. **Model serving** — owned by this repo. Status: working.
2. **Sandbox + agent runtime** — onboard and configure via the
   workflows above. Status: working with the verified version pins.
3. **Bridge service** that translates ManyForge's
   `manyforge.assistant.provider_request.v0` envelope into a model
   dispatch (and back) — owned by this repo when built. Status:
   not yet implemented.

### Where to read ManyForge's expectations

ManyForge's repos live under `/home/tndlux/workspaces/dev_ws/src/`.
For assistant-integration questions, look at three places, in this
order:

- **`manyforge/manyforge_composer/backend/`** — the
  `NemoClawAssistantProvider` Python class is the executable spec
  for the contract: what ManyForge sends, what it expects back, what
  it rejects. When the doc disagrees with the code, **the code wins**.
- **`manyforge/docs/reference/`** — human-readable description of the
  contract and the surrounding architecture.
- **`manyforge_specs/`** (sibling repo) — the architectural
  trajectory: skill catalog, intervention sessions, deployment-artifact
  schema, design ADRs. This is where Phase 3+ features (multi-step
  recovery, GUI tree builder) get specced *before* code lands.

These doc paths shift release-to-release; treat the directories as
durable but not the file names.

### Architecture decisions (locked)

- **Pattern B** — bridge service (in the ManyForge repo, not here) translates ManyForge's
  contract into model dispatches with a deployment-scoped tool catalog.
  The bridge is the enforcement point for the bounded-autonomy
  invariant. Pattern C (MCP) is a future evolution gated on upstream
  MCP support in NemoClaw / OpenClaw; not in scope today.
- **Hybrid autonomy** — the LLM operates with multi-step latitude but
  *only over a finite set of pre-approved skills, primitive nodes, and tools*.
  The bridge guarantees the model never sees instruments outside the
  active mode catalog, so any action it proposes is a certifiable
  outcome by construction.
- **Mode-driven instrument catalogs** — the user activates a named
  mode in ManyForge's GUI (e.g. composer-assistant, error-recovery,
  query) which selects which instruments are exposed for that request.
  Modes have per-mode review policies.
- **Skill-declared tools** — each ManyForge skill ships with its own
  `requires_tools` declaration. ManyForge composes the catalog at
  request time from the active skill set + active mode.
- **No tight-latency LLM path** — assistants are for offline
  program-building and recovery decision-making. Real-time control
  (and any future VLA/policy work) is backend territory and out of
  scope for this assistant pipeline.

Detail and rationale live in `manyforge_specs/docs/spec/485-assistant-bridge-architecture.md`.
Mode taxonomy and bounded-autonomy spec live in `manyforge_specs/docs/spec/480-...md`.

---

## Boundary rules for LLM agents working in this repo

1. **Read before edit.** Before modifying any file under `lib/`,
   `docker/`, or the top-level `start-*.sh` scripts, read the workflow
   doc and any open `*-INVESTIGATION.md` notes that touch the same
   area. The repo's value is mostly in the comments — they record
   traps that took hours to find.

   **Read-before-claim rule for cross-workspace work.** Before
   implementing or claiming scope for anything that crosses a
   workspace boundary (assistant-provider envelope, bridge behaviour,
   deployment-artifact field, skill manifest field, model-serving
   profile referenced from manyforge_specs), read the relevant source
   code in the OTHER workspace, not just its docs. Wire shapes,
   validation rules, and runtime semantics are in code; documentation
   summarises them but can lag. See
   `manyforge_specs/docs/cross-workspace-conventions.md` for the full
   protocol and the authority-by-concern table.

2. **Don't re-implement upstream.** If something feels missing in
   NemoClaw / OpenShell / OpenClaw, the answer is *"check upstream
   first"*. File the issue, work around with a documented manual
   step, wait for the next monthly release. Don't add a script that
   forks or patches their behaviour.

3. **Onboarding can be manual.** The user explicitly accepts manual
   intervention during `nemoclaw onboard`. Do not write scripts that
   try to bypass the wizard, simulate prompts, or pre-populate
   credentials in non-standard locations.

4. **Don't bake secrets, models, or HF tokens into images or scripts.**
   Tokens live in `~/.cache/huggingface/token` and are exported by
   `start-model.sh` / `start-duo.sh` from there. Models live in HF
   cache. The build pipeline pulls neither.

5. **Profile changes go in pairs.** A new profile is two edits:
   `lib/config.sh` (the runtime config) and `lib/launch.sh` (the vLLM
   args). Both must use the same case-statement label. The served name
   is the label.

6. **Be honest about what a benchmark measures.** This repo has been
   burned multiple times by metrics that don't capture the workload
   in question. When proposing a benchmark, name what it measures *and*
   what it doesn't.

7. **Confirm destructive actions.** `nemoclaw onboard` rebuilds the
   sandbox pod and loses workspace state. Container `docker stop`
   before `drop_caches` is fine; `docker system prune` is a separate
   conversation. Stops, restarts, and config wipes against a live
   sandbox are user-confirmed-only.

8. **Match the response style the user prefers.** Terse, no trailing
   summaries, no emoji unless asked. When in doubt, match the diff
   style of nearby commits.

---

## Pointers to operational docs

Keep these high-level — file names shift, scope endures.

- **`NEMOCLAW-OPENCLAW-WORKFLOW.md`** — canonical end-to-end recipe
  (start model, wire OpenShell, dispatch agent). Read first when
  answering anything operational.
- **`MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md`** — profile-selection
  plan for the ManyForge assistant on Thor and Orin (which model fits
  which deployment budget).
- **`MANYFORGE-PROFILE-CALIBRATION.md`** — sizing methodology for
  `max_model_len` / `max_num_seqs` / `gpu_memory_utilization` on
  ManyForge-pipeline-targeted profiles. Read before adding a new
  profile or changing those knobs on an existing one.
- **`agentic-bench/README.md`** — bench harness and candidate plan.
- **`docker/`** — Thor-specific build notes and patch rationale,
  including any active TRT-Edge-LLM evaluation.
- **Performance reports** — `PERFORMANCE-V*.md`, `TOOL-EVAL-BENCH-THOR.md`,
  and other dated bench docs capture point-in-time numbers; treat them
  as historical evidence, not live state.
- **Architectural investigations** — `*-INVESTIGATION.md` files
  document durable lessons from substantial deep-dives (e.g.
  speculative-decoding behaviour on SM110). These stay even after
  the immediate work is done.

**Transient incident docs** (file names matching `*-HANG-*.md` or
`*-INVESTIGATION-*.md` tied to a single specific incident) are problem
reports, not durable knowledge. They are deleted in the same commit
that closes the issue — the commit message + workflow doc updates carry
the resolution. Don't archive them as "Resolved" entries; don't list
them in permanent-pointer tables.

---

## Maintaining this file

When the verified-version pins shift, the scope shifts, or a new
operational doc is added, update this file. It is the entry point for
new contributors (human or LLM); stale information here costs more
than stale information elsewhere.

This file should contain only **durable** operational guidance.
"Known issues today" tables, dated experiment results, and references
to transient incident docs do **not** belong here — they age fast and
turn the entry point into a graveyard. Use commit messages, dated
bench docs, or upstream issue trackers for that.
