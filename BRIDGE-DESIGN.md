# BRIDGE-DESIGN.md — ManyForge assistant-provider bridge

> Architectural design doc for the bridge service that connects
> ManyForge's assistant-provider contract to a local LLM dispatch
> running on this Thor host. Concept-focused and durable; specific
> implementation choices (file paths, version pins, exact request
> field names) belong elsewhere and are deliberately not pinned here.

---

## Purpose

The bridge service is the local enforcement point that mediates between
ManyForge's assistant-provider contract and the model + agent runtime
on Thor. It exists for one structural reason: to **guarantee that the
LLM only ever acts within a certifiable surface of pre-approved
instruments**, regardless of what tools/skills exist in the broader
deployment.

The bridge is owned by this repo (`NemoClaw-Thor/`). It runs on the
Thor host, not inside the OpenShell sandbox.

---

## Position in the stack

```
ManyForge composer (downstream)
    │
    │ POST <bridge-url>
    │ body: assistant-provider request envelope
    │       (version, schemaVersion, message, mode, assistantMode,
    │        requestedTools, context, runtime, tools[], constraints)
    │       — version       = manyforge.assistant.provider_request.v0
    │                         (stable wire family, per 480 §2.1)
    │       — schemaVersion = "0.1.0" or higher; assistantMode mandatory
    │                         for >= 0.1.0 (per 480 §2.1)
    │       — mode          = mock | provider (existing dispatch field)
    │       — assistantMode = active assistant-mode id (per 480 §2.1)
    ▼
┌──────────────────────────────────────────┐
│ Bridge service (this repo)               │
│  • catalog verification + enforcement    │
│  • mode → instrument enforcement         │
│  • multi-step agent loop (hybrid)        │
│  • response normalization + audit        │
└──────────────────────────────────────────┘
    │
    │ HTTP POST /v1/chat/completions
    │ body: messages, tools (filtered), model (slug)
    ▼
vLLM (this repo)  ─►  served-model-name = profile slug
```

The bridge sits between ManyForge and vLLM. OpenClaw and the
sandbox are *not* on the assistant request path. The sandbox provides
an interactive coding-agent path for direct user use
(`nemoclaw <sandbox> connect`); the bridge does not depend on it.

---

## Locked design decisions

The decisions below are settled and the bridge implementation must
respect them. Where the bridge has latitude, it's called out
explicitly under "Open implementation questions" further down.

### 1. Bridge owns the agent loop

The bridge implements the multi-step agent loop directly. It calls vLLM
with `tools=[...]`, parses tool calls, **invokes only `effect: read_only`
tools** during the loop (e.g. scene/program inspection), feeds results
back to the model, iterates until the LLM declines further tool calls
or a turn/wall-time limit is reached. The bridge is responsible for
kill switches: max-iterations, max-wall-time, max-tokens-emitted.

`effect: proposal` and `effect: runtime_mutating` tools are **action
descriptors** the LLM can reference but which the bridge does not
invoke. The bridge records them as part of the response's
`proposals[]` array and returns them to ManyForge. ManyForge applies
proposals on user approval, and executes runtime_mutating descriptors
under its own authority (typically `delegated_recovery` per
manyforge_specs 460).

This is not a sandbox-isolation decision. ManyForge's tools are
proposal-shaped (the LLM proposes; the user reviews and applies); they
do not execute arbitrary code in the sandbox. The sandbox boundary
exists for a different threat model (interactive coding agent) and is
not load-bearing here.

### 2. Bounded autonomy via mode-scoped instrument catalogs

Every request has an active **mode** identifier (the `assistantMode`
field in the request envelope; see manyforge_specs spec 480). ManyForge
composer **composes the resolved catalog before sending the request**:
it has direct access to the deployment artifact, the loaded skill
catalog, and the active program in-process. The composed catalog ships
inside the request envelope, so the bridge sees exactly the instrument
set the mode authorizes.

The bridge does **not** read the deployment artifact independently. It
relies on the embedded catalog and treats it as the authoritative
allowlist for that single request.

If the model proposes a tool/skill/node not present in the embedded
catalog, the bridge **rejects** the call (out-of-mode rejection). The
LLM never sees out-of-catalog instruments because ManyForge filtered
them before the request was sent; rejection covers the
defense-in-depth case where a model hallucinates a reference.

The catalog composition rules — including the load-time
mode-skill compatibility check that prevents an enabled mode from
referencing skills with unsatisfied `requires_tools` — live in
manyforge_specs and ManyForge composer. The bridge enforces, it does
not compose.

### 3. Skill-declared tools

The bridge does not own the skill→tools mapping; ManyForge does.
The bridge consumes whatever the deployment artifact + skill manifests
declare. The format is specified in `manyforge_specs/`.

### 4. Review gating is enforced

`mutated: true` in any response is rejected — same as the upstream
contract. All `proposals` are emitted with `status: draft` and
`requiresReview: true`. The bridge does not make exceptions even for
modes that allow auto-apply within recovery profiles; auto-apply is a
*downstream* (ManyForge composer) concern, applied to draft proposals
by the same review machinery. The bridge always returns drafts.

### 5. Audit trail

Every request emits an audit record containing, at minimum: timestamp,
active mode, catalog hash (so the exact instrument surface can be
reconstructed later), model identity, request-id, response shape
summary (tool-call names, proposal types — not user content),
duration, exit reason. The audit log is the certifiable evidence that
the LLM operated within the declared surface for that request.

Where the audit log is written is an implementation detail; it must
not be the only place — also expose recent records via a small
diagnostic endpoint so an operator can inspect without grepping logs.

---

## What the bridge does *not* do

- It does not implement the ManyForge contract semantics beyond what
  the schema requires. Proposal validation, scene safety checks, and
  apply/save logic remain in ManyForge.
- It does not modify ManyForge state. It returns proposals; ManyForge
  decides what to do with them.
- **It does not invoke `runtime_mutating` tools.** Those are action
  descriptors that ManyForge executes; the bridge only records the
  LLM's references to them as proposals.
- It does not host a tool catalog or skill manifests of its own. It
  consumes the catalog embedded by ManyForge in the request envelope.
- It does not read the deployment artifact independently — that
  duplicates the source of truth. ManyForge composer is the trusted
  catalog authority on the local Thor host.
- It does not orchestrate across deployments. One bridge instance
  serves one deployment.
- It does not depend on OpenClaw for the assistant request path.
  OpenClaw is for the interactive sandbox-agent path.

### Trust model (v0.1)

ManyForge composer and the bridge run as peer processes on the same
operator-controlled host. The bridge trusts the catalog ManyForge
sends; there is no inter-process auth or catalog signing in the
initial v0.1 assistant deployment.
Future hardening for multi-host or untrusted-peer deployments —
signed catalogs, mutual TLS between composer and bridge, or
relocating composition into the bridge with a declared
deployment-artifact source — is parked under "Open implementation
questions" below.

---

## Migration path: B → C if MCP support lands

Pattern C (Model Context Protocol) is the future evolution if
NemoClaw / OpenClaw ship MCP support upstream. The migration is mostly
additive:

- The bridge gains an MCP client capability.
- ManyForge runs an MCP server exposing tools dynamically.
- The instrument catalog is sourced from the MCP server at request
  time instead of the deployment artifact.
- The bridge's enforcement logic (mode → catalog filtering) does not
  change — only the source of the catalog moves.

Mode taxonomy, bounded-autonomy invariant, review gating, and audit
trail are all preserved. The bridge does not need a v2 contract; it
gains an alternative catalog source.

Track upstream (NemoClaw / OpenClaw) for MCP client support in monthly
releases. Until that ships, Pattern C is not actionable from this repo
without forking, which violates a boundary rule.

---

## Open implementation questions

These are not blocking the bridge skeleton, but should be resolved
before the bridge is considered production-ready. Captured here so
they don't get lost.

### Q — Read-only callback surface

The tool-effect split is fixed by the ManyForge specs: the bridge may
execute only `effect: read_only` tools during the loop, while
`effect: proposal` and `effect: runtime_mutating` entries are recorded
as proposals/descriptors and returned to ManyForge.

The unresolved implementation question is how the bridge obtains data
for read-only tools:

- which ManyForge HTTP endpoints or local APIs provide scene/program/
  runtime data,
- how bridge-to-ManyForge calls are authenticated under the v0.1
  same-host trust model,
- timeout/retry limits for in-loop read-only calls,
- how partial failures are represented to the model, user, and audit
  trail.

### Q — Multi-step loop termination conditions

Hybrid autonomy means the LLM can take multiple turns. Termination
conditions to define:

- Max turns per request (suggest: small single-digit default, mode-overridable).
- Max wall time per request (suggest: 30–60s default).
- Max tokens emitted per request.
- LLM declines further tool calls (returns plain message).
- LLM emits an unrecognized tool — bridge rejects, ends loop.

Each termination should be reflected in the audit `exit reason` field.

### Q — Cancellation contract

Separate from natural termination: how does ManyForge cancel a
mid-flight bridge agent loop? Common triggers: user dismisses the
assistant pane, mode switches, deployment shuts down, supervisor
revokes the session. The current ManyForge contract does not specify
a cancellation endpoint or signal; the bridge needs:

- a way to receive a cancellation signal from ManyForge mid-request,
- guarantees about partial-state cleanup (the in-flight model call to
  vLLM, any open in-loop tool invocations),
- a defined response shape for cancelled requests (probably an envelope
  with `exit_reason: cancelled` and no proposals).

This is bridge-skeleton scope; resolution probably also lands a small
extension to the assistant-provider HTTP contract.

### Q — Catalog hash semantics

The audit record includes a "catalog hash" so the active surface is
reconstructible. The hash should be deterministic over: the mode name,
the active skills, the resolved tool set, the review policy in effect.
Decide hash algorithm and field ordering when implementing — and
document it so consumers (ManyForge audit viewer, certification
auditors) can recompute it.

### Q — Stateless vs. stateful bridge

Per request the bridge could be fully stateless (every request
re-resolves catalog from manyforge config) or cache the resolved
catalogs by `(deployment_version, mode)`. Stateless is simpler to
reason about; cached is faster.

Recommendation: start stateless, add caching when measurements show
catalog resolution is a measurable fraction of latency.

### Q — Bridge↔ManyForge communication

The bridge probably needs to make a few calls back to ManyForge during
a loop turn (read scene state, validate a proposal, query skill
manifests). What's the auth model, the URL discovery, the failure
handling for these inner calls? Not yet specified; coordinate with
ManyForge's API surface.

### Q — Error and partial-failure shapes

If a tool call inside the loop fails (e.g. bridge can't reach
ManyForge for an inspection tool), what does the response look like?
Options: bubble up as a `warning` and let the model continue;
terminate the loop with a partial response; treat as fatal. Likely
mode-dependent.

---

## Cross-references

- Mode taxonomy, bounded-autonomy principle, skill self-declaration:
  `manyforge_specs/`.
- Assistant-provider contract (request/response envelope, validation
  rules): `manyforge/docs/reference/` and the `NemoClawAssistantProvider`
  source under `manyforge/manyforge_composer/backend/`. When the docs
  and the code disagree, the code wins.
- Stack chain and version pins for the Thor host: [AGENTS.md](AGENTS.md).
- Operational workflows (start model, configure provider, dispatch):
  [NEMOCLAW-OPENCLAW-WORKFLOW.md](NEMOCLAW-OPENCLAW-WORKFLOW.md).
- Profile selection (which model fits which deployment budget):
  [MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md](MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md).

---

## Maintaining this file

Concept-level updates only. When the bridge gains a new mechanism
(e.g. MCP catalog source), update the relevant section here. Don't
add operational instructions, version pins, or commit references —
those rot fast and belong in the workflow doc, AGENTS.md, or commit
messages respectively.

If a section's content becomes "what we did last release" rather than
"what the bridge is", move that content out of this file.
