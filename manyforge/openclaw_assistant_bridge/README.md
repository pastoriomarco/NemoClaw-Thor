# openclaw_assistant_bridge

Experimental Phase 2 adapter for routing the ManyForge Composer assistant
provider through the OpenClaw agent running in the NemoClaw `my-assistant`
sandbox.

The adapter speaks the same HTTP provider contract Composer already uses:

- `POST /v1/manyforge/assistant`
- `POST /v1/manyforge/assistant/{requestId}/cancel`
- `GET /healthz`

It does not execute ManyForge tools itself. It invokes `openclaw agent` in the
sandbox; OpenClaw reaches ManyForge through the mode-scoped `manyforge` MCP
server provisioned by `manyforge/setup-manyforge-assistant.sh`. ManyForge
remains the authority for mode allowlists, catalog hashes, draft mutation, and
audit.

## Run

Preferred launcher:

```bash
cd /home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor
./manyforge/start-openclaw-assistant-bridge.sh
```

The launcher defaults to the sandbox-local `manyforge-composer` agent profile
installed by `manyforge/setup-manyforge-assistant.sh`.

Manual service run:

```bash
cd /home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor/manyforge/openclaw_assistant_bridge
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

PYTHONPATH="$PWD/.." \
OPENCLAW_ASSISTANT_SANDBOX=my-assistant \
OPENCLAW_ASSISTANT_AGENT=manyforge-composer \
OPENCLAW_ASSISTANT_BRIDGE_PORT=8200 \
.venv/bin/python -m openclaw_assistant_bridge.service
```

Then point Composer at:

```bash
MANYFORGE_ASSISTANT_PROVIDER=openclaw
MANYFORGE_ASSISTANT_ENDPOINT_URL=http://127.0.0.1:8200/v1/manyforge/assistant
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENCLAW_ASSISTANT_SANDBOX` | `my-assistant` | OpenShell sandbox/pod name |
| `OPENCLAW_ASSISTANT_NAMESPACE` | `openshell` | Kubernetes namespace inside the OpenShell cluster |
| `OPENCLAW_ASSISTANT_CONTAINER` | `agent` | Sandbox container name |
| `OPENCLAW_ASSISTANT_CLUSTER_CONTAINER` | `openshell-cluster-nemoclaw` | Docker container that hosts k3s/kubectl |
| `OPENCLAW_ASSISTANT_AGENT` | launcher default: `manyforge-composer`; service default: `main` | OpenClaw agent id |
| `OPENCLAW_ASSISTANT_TIMEOUT_S` | `180` | Default per-request agent timeout |
| `OPENCLAW_ASSISTANT_LOCAL` | launcher default: `true`; service default: `false` | Add `--local` to `openclaw agent` when set. The launcher defaults this on because the validated Thor path uses the sandbox-local OpenClaw runner. |
| `OPENCLAW_ASSISTANT_THINKING` | `off` | Passed as `openclaw agent --thinking ...`. The Phase 2 route defaults to `off` because local Omni otherwise spends the Composer provider timeout even on trivial prompts. |
| `OPENCLAW_ASSISTANT_AUTO_TOOL_WINDOW` | `true` | When enabled, the adapter writes a short-lived sandbox tool-window file so the ManyForge MCP wrapper exposes a request-sized tool window for obvious tree/scene edits. Broad or ambiguous prompts fail open to the full mode surface. |
| `OPENCLAW_ASSISTANT_ALLOWED_TOOLS_FILE` | `/tmp/manyforge-openclaw-allowed-tools.txt` | Sandbox file used to pass the request-scoped tool window to the mode-scoped MCP wrapper. A file is used because OpenClaw's configured MCP server env is static. |
| `OPENCLAW_ASSISTANT_BRIDGE_HOST` | `127.0.0.1` | HTTP bind host |
| `OPENCLAW_ASSISTANT_BRIDGE_PORT` | `8200` | HTTP bind port |

## Current status

This is an experimental Phase 2 endpoint. On 2026-05-05, Composer chat routing
through this adapter was live-smoked against the sandbox OpenClaw agent:

- `catalog.read` completed through mode-scoped MCP and
  `/api/assistant/bridge/tools/catalog.read`.
- `tree.draft.wrap_node` completed through the same bounded path and mutated
  the Composer draft root to `repeat -> pick_and_place`.

The direct vLLM bridge remains the known-good demo default until the A/B
harness qualifies reliability and latency for the OpenClaw route.

Follow-up measurements on 2026-05-05 showed that shelling into the sandbox is
sub-second, but the OpenClaw agent turn remains minute-scale even with thinking
off and a narrowed ManyForge MCP tool window. The next optimization surface is
a persistent OpenClaw runner or provider-side speed work, not more ManyForge
callback tuning. Use `manyforge/smoke-openclaw-assistant-reliability.py` to
collect comparable timing records while iterating.
