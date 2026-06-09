# Composer-Assistant Runbook

A single-place reference for the OpenClaw-backed ManyForge composer-
assistant lane: how it's wired, the gates a request passes through,
and how to debug each stage.

Scope: the in-sandbox **OpenClaw gateway** path with the **manyforge
MCP bridge**. The fallback "direct vLLM" lane is documented in
[MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md](./MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md).

## Production default (2026-05-07)

| | |
|---|---|
| **Lane** | `openclaw` (`scripts/demo-assistant-known-good.sh:48`) |
| **Model** | `cosmos-reason2-8b` — `nvidia/Cosmos-Reason2-8B`, FP8 KV, hermes parser, 64K ctx (`serving/config.sh:189` default fallback) |
| **vLLM sampling** | `temperature=0.2, top_p=0.95` server-side, `enable_thinking=false` from the model's chat template |
| **Why this combo** | OpenClaw lane achieves 9/9 on the 3-prompt × 3-round parity smoke (Qwen3.6 OpenClaw was 1/9; Nemotron 0/9). See [LANE-COMPARISON-direct-vs-openclaw.md §8](./LANE-COMPARISON-direct-vs-openclaw.md) for the full benchmark and reproduction recipe. |

To bring the default stack up after a reboot:

```bash
cd $HOME/workspaces/dev_ws/src/NemoClaw-Thor && ./serving/start-model.sh
$HOME/workspaces/dev_ws/src/NemoClaw-Thor/manyforge/setup-manyforge-assistant.sh my-assistant
$HOME/workspaces/dev_ws/src/NemoClaw-Thor/manyforge/start-openclaw-assistant-bridge.sh &
cd $HOME/workspaces/dev_ws/src/manyforge && ./scripts/demo-assistant-known-good.sh start
```

To switch back to the direct lane (fast-path for simple prompts only —
P3-style compound prompts race the 60s budget on this lane):

```bash
ASSISTANT_PROVIDER=nemoclaw ./scripts/demo-assistant-known-good.sh restart
```

---

## Three-lane bring-up (THREE-LANE-MIGRATION-PLAN.md Phase 2/3/4)

### Direct lane (`ASSISTANT_PROVIDER=nemoclaw`)

Canonical bring-up via the launcher:

```bash
cd $HOME/workspaces/dev_ws/src/manyforge
ASSISTANT_PROVIDER=nemoclaw \
  THOR_VLLM_PORT=8050 THOR_RESTART_PROXY=0 MANYFORGE_NON_INTERACTIVE=1 \
  ./scripts/launch.sh start --lane manyforge-only --assistant on \
  --scenario ur10e-scene-authoring --non-interactive --yes
```

Brings up composer (:9000), vLLM (:8050), vllm-proxy (:8000), direct
bridge (:8100). The shared core (`manyforge.common.projection`) is
auto-imported by the bridge when `MANYFORGE_THOR_ROOT` points at the
NemoClaw-Thor checkout (defaults to
`/home/tndlux/workspaces/dev_ws/src/NemoClaw-Thor/manyforge`).

Diagnostic helper: `./scripts/setup-direct.sh` walks the six gates
(vLLM → proxy → composer → bridge venv → shared core → bridge healthz)
and reports the first that fails.

Baseline on cosmos-reason2-8b: **28/66 effective** (Phase 0 D-1; see
[PHASE-0-LANE-BASELINE.md](./archive/PHASE-0-LANE-BASELINE.md)). Failure
pattern: `args_contain[...] got '<MISSING>'` — cosmos calls the right
tool but doesn't fill arguments.

### OpenClaw lane (`ASSISTANT_PROVIDER=openclaw`)

Canonical bring-up (after a clean sandbox onboard):

```bash
# 1. Onboard a fresh sandbox with restricted policy + local-inference preset.
NEMOCLAW_PROVIDER=custom \
  NEMOCLAW_ENDPOINT_URL=http://127.0.0.1:8000/v1 \
  NEMOCLAW_MODEL=cosmos-reason2-8b \
  NEMOCLAW_PROVIDER_KEY=dummy \
  NEMOCLAW_POLICY_TIER=restricted \
  NEMOCLAW_POLICY_PRESETS=local-inference \
  nemoclaw onboard --non-interactive --yes --fresh --name my-assistant \
                   --recreate-sandbox --yes-i-accept-third-party-software

# 2. Setup the manyforge layer (skill, MCP server, agent profile).
$HOME/workspaces/dev_ws/src/NemoClaw-Thor/manyforge/setup-manyforge-assistant.sh my-assistant

# 3. Bring up via launcher.
cd $HOME/workspaces/dev_ws/src/manyforge
ASSISTANT_PROVIDER=openclaw \
  THOR_VLLM_PORT=8050 THOR_RESTART_PROXY=0 MANYFORGE_NON_INTERACTIVE=1 \
  ./scripts/launch.sh start --lane manyforge-only --assistant on \
  --scenario ur10e-scene-authoring --non-interactive --yes
```

iter-32 production baseline: **51/66 effective** on cosmos-reason2-8b
with the chain-on recipe (bridge fires `/compact` every 2 prompts).

### Hermes lane (`ASSISTANT_PROVIDER=hermes`) — Phase 4

Inert until Phase 4 lands. The launcher rejects `ASSISTANT_PROVIDER=hermes`
unless `HERMES_LANE_PHASE4_ENABLED=true` is also set, surfacing the
gap clearly instead of falling back to Direct (the pre-Phase-1
foot-gun). Composer-side `LANE_REGISTRY` carries the entry with
`inert=True` for forward visibility.

---

## 1. The full chain

A user message in the Composer UI travels through ten distinct gates.
Any one of them being misconfigured produces a "timeout" or
"hallucinated answer" symptom in the UI:

```
1.  UI (AssistantOverlay)
        ↓ POST /api/assistant/chat
2.  Composer backend (routes_assistant.chat)
        ↓ HTTP to assistant_endpoint
3.  Bridge service (openclaw_assistant_bridge.service:8200)
        ↓ build_gateway_chat_completions_command + curl
4.  OpenClaw gateway (in-sandbox, port 18789 → forwarded to host)
        ↓ /v1/chat/completions with x-openclaw-session-key
5.  vLLM (host.openshell.internal:8000)
        ↓ (model emits tool_calls)
6.  OpenClaw runner (bundle-mcp materialize → callTool)
        ↓ stdio JSON-RPC tools/call
7.  manyforge-mcp-bridge.py subprocess
        ↓ HTTP through OpenShell egress proxy (10.200.0.1:3128)
8.  Composer /api/assistant/bridge/tools/<toolId>
        ↓ runs the actual ManyForge tool
9.  Result back through 7 → 6 → next vLLM turn
10. Final assistant message → bridge service → Composer → UI
```

---

## 2. The ten gates and how to verify each

### Gate 1 — UI sends the right shape

The UI POST body is:
```json
{
  "requestId": "...", "conversationId": "...", "message": "...",
  "mode": "provider", "providerId": "openclaw",
  "assistantMode": "composer-assistant"  // or "query"
}
```

Verify: `docker logs --since 5m manyforge-e2e-composer | grep "POST /api/assistant/chat"`.

### Gate 2 — Composer dispatches to the bridge service

Composer is started with `--assistant-provider <id> --assistant-endpoint
<bridge-url> --assistant-timeout-s 300`. The 300 s is the hard wall: if
the agent loop runs longer, the UI sees "NemoClaw assistant timed out
after 300.000s". (Older docs reference 180s; raised to 300s on
2026-05-06 because legitimate OpenClaw runs occasionally take 100-200s.)

Default provider is `openclaw` (sandboxed gateway lane on `:8200`,
2026-05-06 — lane parity verified). Set `ASSISTANT_PROVIDER=nemoclaw`
on the launcher for the direct-vLLM backup on `:8100`. See
[`LANE-COMPARISON-direct-vs-openclaw.md`](./LANE-COMPARISON-direct-vs-openclaw.md)
§9 for the data behind the default switch and the trio of fixes
(vendor sampling at vLLM, MCP wrapper null-arg validation, schema
worked examples).

### Gate 3 — Bridge service is up and reachable

```bash
curl -sS http://127.0.0.1:8200/healthz
```
Expected: `{"status":"ok","provider":"openclaw","sandbox":"my-assistant",...}`.

**OPENCLAW_ASSISTANT_USE_GATEWAY is now `false` by default** as of the
2026-06-03 route fix (see [PHASE-0-LANE-BASELINE.md](./archive/PHASE-0-LANE-BASELINE.md)).
OpenClaw 2026.5.22's HTTP server does not expose `/v1/chat/completions`,
so the legacy `gateway_http` transport returns 404 in 50ms. The bridge
now uses `cli_shell_out` (invokes `openclaw agent` via `nemoclaw exec`)
which works against 2026.5.22. Set
`OPENCLAW_ASSISTANT_USE_GATEWAY=true` only to force the legacy path
(useful if upstream OpenClaw exposes `/v1/chat/completions` in a future
release; currently broken).

If timeouts hit ~120s, the service was started with the default
`OPENCLAW_ASSISTANT_TIMEOUT_S=120` — restart with a higher value. The
iter-32 production recipe sets `OPENCLAW_ASSISTANT_TIMEOUT_S=300`.

### Gate 4 — Gateway is running in the sandbox SSH netns

```bash
openshell sandbox exec -n my-assistant --no-tty -- bash -c \
  'ps -ef | grep openclaw-gateway | grep -v grep'
```

Restart with `setup/sandbox-runtime.sh::ensure_sandbox_gateway_running my-assistant`.
The gateway MUST run in the SSH-session network namespace (not the
pod root) for the host port-forward to reach it.

### Gate 5 — Gateway can reach vLLM

The OpenShell SSRF policy at `manyforge-composer.preset.yaml` allows
`host.openshell.internal:8000` with `allowed_ips: ["172.17.0.0/16"]`.
The "allowed_ips" field is mandatory because OpenShell's SSRF guard
otherwise rejects the docker-bridge IP 172.17.0.1.

Smoke: from inside the sandbox, with proxy envs set,
```python
import urllib.request
urllib.request.urlopen(
  "http://host.openshell.internal:8000/v1/models", timeout=5
).status   # → 200
```

### Gate 6 — vLLM tool-call parser matches the model

For `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`:
```
--tool-call-parser qwen3_coder --reasoning-parser nemotron_v3
```

Quick wire test (bypasses OpenClaw entirely): see
`/tmp/test-vllm-direct-tool.py` — a healthy parser returns
`{"tool_calls":[{"function":{"name":"manyforge__scene-inspect", "arguments":"{}"}}]}`.

### Gate 7 — OpenClaw bundle-mcp policy allows tool execution

The agent profile `manyforge-composer` MUST set:
```python
"tools": {"profile": "minimal", "alsoAllow": ["bundle-mcp"]}
```

The "minimal" core profile alone does NOT include `bundle-mcp`
(verified in `tool-policy-DArLXMH2.js`). Without `alsoAllow`, all
manyforge MCP tool calls return "Tool failed" and the agent loops.

### Gate 8 — MCP server config carries proxy envs

OpenClaw spawns MCP server processes with a SCRUBBED environment.
The bridge subprocess therefore needs HTTP_PROXY / HTTPS_PROXY /
NO_PROXY explicitly forwarded in the MCP server config:
```json
{"command":"python3","args":[...],"env":{
  "HTTP_PROXY":"http://10.200.0.1:3128",
  "HTTPS_PROXY":"http://10.200.0.1:3128",
  "NO_PROXY":"127.0.0.1,localhost,::1",
  ...
}}
```

The provisioner does this. To verify after a manual `openclaw mcp
set`, read `/proc/<bridge_pid>/environ`.

### Gate 9 — SSRF policy path patterns match

The L7 enforcer does NOT honor `{name}`-style placeholders. Use
`/**`:
```yaml
- allow: { method: GET,  path: /api/assistant/modes/** }
- allow: { method: POST, path: /api/assistant/bridge/tools/** }
```
Verified 2026-05-05 — `{mode}` failed with "denied by L7 policy".

### Gate 10 — Deployment manifest has the tools the agent needs

Read-write modes must be a strict superset of read-only modes'
read tools, otherwise the model loops trying to call obvious-
named tools that aren't actually exposed (e.g. `program.read`).

Verify:
```bash
diff <(curl -sS http://127.0.0.1:9000/api/assistant/modes/query \
        | jq -r '.tools[].id' | sort) \
     <(curl -sS http://127.0.0.1:9000/api/assistant/modes/composer-assistant \
        | jq -r '.tools[].id' | sort)
```

If `composer-assistant` is missing any read-only tool that `query`
has, fix the deployment YAML's
`assistant_modes.composer-assistant.catalog.tools` list.

---

## 3. Symptom → most-likely gate

| Symptom in UI | Most-likely gate |
|---|---|
| "Could you provide the session key?" | Workspace files (AGENTS.md/TOOLS.md) missing or stale; `tools.profile=minimal` without `alsoAllow: ["bundle-mcp"]` |
| Hallucinated answer (no tools called) | Same as above |
| "NemoClaw assistant timed out after 180.000s" | Gate 7 (policy blocks tools), Gate 8 (proxy missing → bridge can't reach Composer), or Gate 10 (model loops on a tool that doesn't exist in this mode) |
| `Connection refused` in gateway log | Gate 8 (proxy envs missing on MCP server) |
| `denied by L7 policy` in 403 body | Gate 9 (`{name}` placeholder syntax instead of `/**`) |
| `policy_denied` in 403 body, curl from non-allowed binary | Binary not in `manyforge_composer.binaries` allow-list |
| `Manyforge X-Y failed` in gateway, no POST in Composer audit | Gate 7 (allow list blocks bundle-mcp) or Gate 10 (tool not in mode catalog) |
| Tool name registered as `manyforge__scene-inspect` but model calls `manyforge__scene_inspect` | Workspace TOOLS.md has wrong mangling — dots become **dashes**, underscores stay |
| Bridge returns "tool not exposed by assistant mode" | Tool id is not in the active assistant mode catalog, or the lane is pointed at the wrong `MANYFORGE_ASSISTANT_MODE` / Composer instance |

---

## 4. Quick smoke chain (every gate in 30 seconds)

```bash
# 3 — bridge service
curl -sS http://127.0.0.1:8200/healthz | jq .status

# 5 — vLLM reachable (must run inside sandbox)
openshell sandbox exec -n my-assistant --no-tty -- python3 -c \
  'import urllib.request as u; print(u.urlopen("http://host.openshell.internal:8000/v1/models",timeout=3).status)'

# 6 — vLLM tool parser correct
python3 ${HOME}/workspaces/dev_ws/src/NemoClaw-Thor/manyforge/openclaw_assistant_bridge/tests/test_adapter.py  # run unit tests

# 9 — SSRF policy syntax
openshell policy get my-assistant --full | grep -A2 'method: GET' | grep '/api/assistant/modes'
# expect: path: /api/assistant/modes/**

# 10 — deployment manifests
curl -sS http://127.0.0.1:9000/api/assistant/modes/composer-assistant | jq -r '.tools[].id' | sort | grep program.read
# expect: program.read

# end-to-end (full chain)
curl -sS -X POST http://127.0.0.1:9000/api/assistant/chat \
  -H "content-type: application/json" --max-time 200 \
  -d '{"requestId":"smoke","conversationId":"smoke","message":"What is in the scene?","mode":"provider","providerId":"openclaw","assistantMode":"composer-assistant"}'
# expect: a real scene description in <130s
```

---

## 5. Files that must stay in sync

| File | Owns |
|---|---|
| `manyforge/policies/manyforge-composer.preset.yaml` | SSRF L7 policy + binary allow-list |
| `manyforge/setup-manyforge-assistant.sh` | MCP server config (proxy envs) + agent profile (`alsoAllow: bundle-mcp`) + workspace file install |
| `<manyforge-repo>/agent-skills/manyforge-composer/workspace-AGENTS.md` (in the sibling `manyforge` repo) | Canonical workspace AGENTS.md — role, vocabulary, tool surface, long-form guardrails. The provisioner composes the in-sandbox `workspace/AGENTS.md` from this canonical file plus the optional overlay below. |
| `manyforge/agent-workspace/openclaw-overlay.md` (this repo, optional) | OpenClaw-specific overlay appended to the canonical workspace AGENTS.md by `setup-manyforge-assistant.sh`. Empty/absent is fine. |
| `manyforge/openclaw_assistant_bridge/adapter.py` | Session-key derivation + leakage filter |
| `dev_ws/src/manyforge/examples/*.deployment.yaml` (different repo) | `assistant_modes.<mode>.catalog.tools` — the only source of truth for what the agent actually sees |

If you change tool-name mangling or allowed tools in any of these,
the others may need to follow. The smoke chain above catches the
common drifts.
