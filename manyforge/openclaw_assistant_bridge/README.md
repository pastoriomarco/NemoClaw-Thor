# openclaw_assistant_bridge

> **Lane status (2026-05-06):** this is the **default assistant-provider
> lane** in the ManyForge demo launcher
> (`scripts/demo-assistant-known-good.sh`, `ASSISTANT_PROVIDER=openclaw`).
> Lane parity with the direct-vLLM lane was verified on this date — all
> 5 probe tasks pass on both lanes within 1.3× of each other, OpenClaw
> faster on `scene_inspect` and `scene_add`. The in-tree
> `manyforge_assistant_bridge` (`nemoclaw` provider id, port :8100) is
> retained as a backup for fast local iteration when the sandbox layer
> is not needed.
>
> The trio of fixes that closed the lane gap:
>
> 1. vLLM-side vendor sampling (`--override-generation-config
>    '{"temperature":0.6,"top_p":0.95}'` +
>    `--default-chat-template-kwargs '{"enable_thinking":false}'`) —
>    baked into the matching profile in `serving/launch.sh`.
> 2. The custom MCP wrapper (`scripts/manyforge-mcp-bridge.py` in the
>    manyforge repo) now rejects null/missing-required-arg tool calls
>    with a structured error.
> 3. Tool input schemas
>    (`manyforge_composer/backend/assistant_tool_schemas.py`) ship
>    JSON-Schema-standard `examples` arrays on
>    `scene.draft.add_object` and `tree.draft.wrap_node`.
>
> Full diagnosis + reproducer:
> [`manyforge/docs/LANE-COMPARISON-direct-vs-openclaw.md`](../docs/LANE-COMPARISON-direct-vs-openclaw.md)
> §9.

Phase 2 adapter for routing the ManyForge Composer assistant provider
through the OpenClaw agent running in the NemoClaw `my-assistant` sandbox.

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
cd ${HOME}/workspaces/nemoclaw/src/NemoClaw-Thor
./manyforge/start-openclaw-assistant-bridge.sh
```

The launcher defaults to the sandbox-local `manyforge-composer` agent profile
installed by `manyforge/setup-manyforge-assistant.sh`.

Manual service run:

```bash
cd ${HOME}/workspaces/nemoclaw/src/NemoClaw-Thor/manyforge/openclaw_assistant_bridge
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

**Production lane (default, 2026-05-06).** Lane parity with direct vLLM is
verified — see `manyforge/docs/LANE-COMPARISON-direct-vs-openclaw.md` §9
for the per-task numbers and the trio of fixes that closed the gap.

Routing is end-to-end live across all 5 probe tasks (scene_inspect,
program_read, scene_add, tree_wrap, root_query) — both lanes pass with
OpenClaw faster on two of them. The `manyforge-composer` agent profile,
the bounded-autonomy MCP wrapper at
`/api/assistant/bridge/tools/{toolId}`, and the gateway-HTTP transport
are all on the validated path.

### Sampling defaults — owned by vLLM, not the bridge

As of 2026-05-06 this bridge **no longer reads or injects** sampling
fields. vLLM owns them server-side via `--override-generation-config`
and `--default-chat-template-kwargs` (see the matching profile in
`nemoclaw-thor/serving/launch.sh`). `AdapterConfig`'s
`gateway_temperature` / `gateway_top_k` / `gateway_top_p` /
`gateway_enable_thinking` fields default to `None` and are kept for
backward compatibility — when `None`, no body field is added and
vLLM's defaults apply.

The previous YAML-driven path
(`manyforge/agent-sampling-defaults.yaml` → `service.py::_load_sampling_defaults_for_model`)
was retired. The YAML file is kept as reference documentation for
per-model empirical sampling notes.

Use `manyforge/smoke-openclaw-assistant-reliability.py` for ongoing
reliability snapshots and the live profiler in
`manyforge/docs/WORKSPACE-PROMPT-OPTIMIZATION.md §6` for per-task
latency / token / cache-hit measurements.
