# ManyForge integration — agent landing

This subtree under `NemoClaw-Thor/manyforge/` holds the **integration-side**
artifacts for running ManyForge's Composer-assistant pipeline on Thor:
egress preset, sandbox provisioner, OpenClaw bridge service, agent
workspace files, lane-parity debug tooling, and operational docs.

## Authority chain (read in this order before changing anything)

The three-repo split is intentional. Each repo owns a different
question; agents should land in the right one for the change being
made:

| Question | Authoritative repo | AGENTS.md |
|---|---|---|
| **What is the contract / spec / ADR?** ("what should this do?") | `dev_ws/src/manyforge_specs/` | `manyforge_specs/AGENTS.md` |
| **What's in the implementation code / tests?** ("how is it written today?") | `dev_ws/src/manyforge/` | `manyforge/AGENTS.md` (a one-page redirect to `manyforge_specs`) |
| **What does NemoClaw-Thor own for serving + sandbox + integration?** ("how does it run on Thor?") | `nemoclaw/src/NemoClaw-Thor/` | [`NemoClaw-Thor/AGENTS.md`](../AGENTS.md) — the parent of this file |
| **This subtree (integration-only artifacts)** | here | this file |

If your change spans repos (most do): start at
`manyforge_specs/AGENTS.md`, walk down to the implementation, then
back to here for the runtime artifacts. Don't write spec-level
content in this directory; it belongs in `manyforge_specs`.

## What this subtree owns

- `setup-manyforge-assistant.sh` — sandbox provisioner (idempotent).
- `start-openclaw-assistant-bridge.sh` — runs the OpenClaw assistant
  bridge on `:8200` (the production lane).
- `policies/manyforge-composer.preset.yaml` — OpenShell egress / SSRF
  policy.
- `agent-workspace/AGENTS.md` — workspace file injected into every
  OpenClaw agent run (NOT a development AGENTS.md — runtime artifact).
- `openclaw_assistant_bridge/` — bridge service source (Python).
- `docs/` — operational docs (runbook, lane comparison, MCP integration).
- `scripts/debug/` — proxy + harness for lane parity debugging.

## Production default

`openclaw` lane + `cosmos-reason2-8b` served model. Bring-up commands
and runbook are in [`docs/COMPOSER-ASSISTANT-RUNBOOK.md`](docs/COMPOSER-ASSISTANT-RUNBOOK.md).
The benchmark behind this default is in
[`docs/LANE-COMPARISON-direct-vs-openclaw.md` §8](docs/LANE-COMPARISON-direct-vs-openclaw.md).

## Files that must stay in sync

When the served model or the deployment catalog changes:

- `policies/manyforge-composer.preset.yaml` (this repo)
- `setup-manyforge-assistant.sh` (this repo)
- `agent-workspace/AGENTS.md` (this repo)
- `openclaw_assistant_bridge/adapter.py` (this repo)
- `dev_ws/src/manyforge/examples/*.deployment.yaml` (sibling repo —
  the source of truth for what the agent actually sees)

To detect drift, run `scripts/debug/lane-parity-diff.py` on a
known-good prompt. Any divergence between the two lanes shows up as
a field-by-field diff.
