# ManyForge ↔ NemoClaw integration

This directory holds the integration-side artifacts for the ManyForge
Composer-assistant pipeline:

- **`setup-manyforge-assistant.sh`** — idempotent provisioner. Installs
  the egress preset, the `manyforge-composer` skill, the `manyforge`
  MCP server, and the agent profile inside a NemoClaw sandbox. Re-run
  any time you rebuild the sandbox or change the served model.
- **`start-openclaw-assistant-bridge.sh`** — launches the
  `openclaw_assistant_bridge` HTTP service on `:8200`. This is the
  Composer-facing endpoint when the production lane is active.
- **`policies/manyforge-composer.preset.yaml`** — the OpenShell SSRF /
  L7 policy that allows the in-sandbox MCP bridge to reach Composer
  (`:9000`) and vLLM (`:8000`) via `host.openshell.internal`.
- **`agent-workspace/`** — workspace fragment that the provisioner
  composes into the in-sandbox workspace AGENTS.md. Canonical content
  is in the sibling `manyforge` repo at
  `agent-skills/manyforge-composer/workspace-AGENTS.md`; this repo
  contributes only the optional `openclaw-overlay.md`. The system
  prompt the OpenClaw runner sees on every turn is the canonical
  workspace AGENTS.md plus (if present) the overlay.
- **`openclaw_assistant_bridge/`** — the bridge service source (Python
  FastAPI). The provisioner and the launcher both reference this path.
- **`docs/`** — operational docs (see *Where to read* below).
- **`scripts/proxy/`** — `vllm-proxy.py`, the HTTP reverse proxy that
  logs every request/response between the chat-completions client (the
  bridge or OpenClaw gateway) and vLLM, and optionally mutates outbound
  request bodies (max_tokens injection, thinking budget, tool_choice
  overrides, …). Part of the iter-20 production recipe. See
  [`docs/COMPOSER-ASSISTANT-ARCHITECTURE.md`](docs/COMPOSER-ASSISTANT-ARCHITECTURE.md)
  for the proxy's role in the full request flow.
- **`scripts/debug/`** — lane-parity debugging harnesses:
  `lane-parity-diff.py` (single-prompt structural diff between the two
  lanes), `lane-3x3-smoke.py` (3-round × 3-prompt × 2-lane reliability
  smoke), and the smoke corpus runner (`smoke_corpus_runner.py`). README
  under `scripts/debug/`.

## Bringing up the production default

Production default = OpenClaw lane + `gemma4-12b-it-gguf`.
After the local model server is running (`./serving/start-model.sh` from the repo root),
the four commands that bring up the Composer-assistant pipeline:

```bash
# 1. Provision the sandbox (skill + policy + MCP register + agent profile)
./manyforge/setup-manyforge-assistant.sh my-assistant

# 2. Start the OpenClaw bridge on :8200
./manyforge/start-openclaw-assistant-bridge.sh &

# 3. Start Composer in OpenClaw mode (defaults to ASSISTANT_PROVIDER=openclaw)
cd ~/workspaces/dev_ws/src/manyforge
./scripts/demo-assistant-known-good.sh start
```

To swap to the direct lane (fast-path for simple prompts only — sandbox
bypass; the bridge runs its own loop with a `tool_choice` pin):

```bash
ASSISTANT_PROVIDER=direct ./scripts/demo-assistant-known-good.sh restart
```

## Where to read

| Topic | Doc |
|---|---|
| One-screen production-default summary + bring-up | [docs/COMPOSER-ASSISTANT-RUNBOOK.md](docs/COMPOSER-ASSISTANT-RUNBOOK.md) (top block) |
| Per-gate debugging (10 gates of the request chain) | [docs/COMPOSER-ASSISTANT-RUNBOOK.md](docs/COMPOSER-ASSISTANT-RUNBOOK.md) §1–§4 |
| Why OpenClaw + Cosmos-8B is the default + benchmark + end-to-end reproduction recipe | [docs/LANE-COMPARISON.md §8](docs/LANE-COMPARISON.md) |
| Lane-parity debug tooling (proxies + harness + JSONL format) | [docs/LANE-COMPARISON.md §7](docs/LANE-COMPARISON.md), [scripts/debug/README.md](scripts/debug/README.md) |
| MCP integration deep-dive (the `manyforge` server, tool-name mangling, principal binding) | [docs/MANYFORGE-MCP-INTEGRATION.md](docs/MANYFORGE-MCP-INTEGRATION.md) |
| Per-model sampling / profile calibration history | [docs/MANYFORGE-PROFILE-CALIBRATION.md](docs/MANYFORGE-PROFILE-CALIBRATION.md), [docs/archive/WORKSPACE-PROMPT-OPTIMIZATION.md](docs/archive/WORKSPACE-PROMPT-OPTIMIZATION.md) |
| Local-model ManyForge smoke evaluation: Cosmos3 Nano, DS4 and Cosmos3 Edge (2026-08-06) | [docs/smoke-evidence/2026-08-06-manyforge-local-model-smoke/REPORT.md](docs/smoke-evidence/2026-08-06-manyforge-local-model-smoke/REPORT.md) |
| Deployment plan (delivered phases + open follow-ups) | [docs/MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md](docs/MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md) |

## Files that must stay in sync

When the served model or the deployment catalog changes, these need
to update together:

- `policies/manyforge-composer.preset.yaml` — SSRF L7 policy.
- `setup-manyforge-assistant.sh` — MCP server config + agent profile.
- Canonical workspace AGENTS.md (role + vocabulary + tool routing) —
  in the sibling `manyforge` repo at
  `agent-skills/manyforge-composer/workspace-AGENTS.md`.
- `agent-workspace/openclaw-overlay.md` (this repo, optional) —
  OpenClaw-specific addendum appended after the canonical content.
- `openclaw_assistant_bridge/adapter.py` — session-key derivation +
  tool alias filter.
- `dev_ws/src/manyforge/examples/*.deployment.yaml` (different repo) —
  `assistant_modes.<mode>.catalog.tools` is the only source of truth
  for what the agent actually sees.

If any of these drift, run `./manyforge/scripts/debug/lane-parity-diff.py`
on a known-good prompt and the divergence will surface in the diff.
