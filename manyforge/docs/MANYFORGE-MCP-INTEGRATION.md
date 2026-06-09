# ManyForge ↔ OpenClaw MCP Integration on Thor

End-to-end runbook for routing the ManyForge composer-assistant through the
OpenClaw agent runtime in the NemoClaw `my-assistant` sandbox, using
official NemoClaw / OpenClaw / OpenShell extension points only. The goal is
to exploit the agent's skill mechanism and the Model Context Protocol (MCP)
tool surface so the in-sandbox agent — not direct vLLM chat-completions —
plans the model's work, while ManyForge remains the enforcement boundary
for tools, modes, and draft mutations.

Status:
- **Phase 1 — sandbox can call ManyForge tools via MCP, mode-scoped:**
  capability proof against the broad `/api/mcp` surface validated
  end-to-end on 2026-05-04. On 2026-05-05 the path was narrowed in
  code to the mode-scoped bridge endpoint: the in-sandbox MCP wrapper
  translates every tool call into a `/api/assistant/bridge/tools/{toolId}`
  call with the bounded-autonomy envelope (`assistantMode`,
  `catalogHash`, `requestId`, `conversationId`, `principal`).
  Server-side enforcement — the same gates the in-Composer assistant
  uses — is the source of truth. The mode-scoped path was
  **wrapper-layer-validated end-to-end on 2026-05-05** (host probe,
  Composer-reload, provisioner against `my-assistant`, sandbox-side
  `tools/list` returning 23 mode-permitted tools with zero broad-MCP
  leakage); see "Validation log" at the bottom for evidence. The
  full agent → `tools/call` round-trip was subsequently validated
  through the Phase 2 adapter.
  Provisioning artifacts are in this repo; setup is reproducible via
  `manyforge/setup-manyforge-assistant.sh`.
- **Phase 2 — composer's chat endpoint routes through OpenClaw:**
  adapter landed and was live-smoked on 2026-05-05 under
  `manyforge/openclaw_assistant_bridge/`. It speaks the existing
  assistant-provider HTTP contract, invokes `openclaw agent` inside
  `my-assistant`, and translates OpenClaw JSON output back into the
  Composer envelope. Live Composer chat validation passed for
  `catalog.read` and `tree.draft.wrap_node`; the latter mutated the
  Composer draft through `/api/assistant/bridge/tools/tree.draft.wrap_node`.
  The route is still experimental because latency is high and the A/B
  reliability harness is not implemented yet.
- **Phase 3 — A/B harness comparing direct-vLLM vs OpenClaw-skill paths:**
  designed, not yet implemented.

This is the deployment-side companion to the runtime tree-mutation
hardening that lives in the manyforge repo's bridge service
(`manyforge_assistant_bridge/`) and Composer backend
(`manyforge_composer/backend/routes_assistant.py`).

---

## Why this exists

The ManyForge composer's assistant flow currently runs:

```
Composer UI
  → composer.assistant_provider (HTTP, contract: provider_request.v0)
  → manyforge_assistant_bridge (HTTP, on host :8100)
  → vLLM /v1/chat/completions (host :8000) — direct prompt + tool list
```

The 30B local model emits invalid tool calls more often than is acceptable
for a video demo (catalog-id invention, wrong-tool selection across
overloaded vocabulary, fabricated state). Improving the prompt has hit
diminishing returns on the JSON-Schema-description path. The remaining
leverage is putting the procedural context — vocabulary lock, tool routing
rules, anti-patterns, recovery protocols — in the agent's *skill* layer,
which OpenClaw already supports natively, rather than fragmented across
every tool description.

The NemoClaw stack already provisions OpenClaw inside `my-assistant`
sandbox; the agent has its own subagent infrastructure, sandboxed
filesystem, and 256k-token context. None of that is currently used by the
manyforge-assistant flow. This integration wires it in.

The architectural rule we keep: **OpenClaw / Hermes plan and reason;
ManyForge stays the authority** for modes, allowlists, draft mutations,
and tool effects. The agent calls ManyForge tools via MCP; ManyForge
enforces every call exactly as it does today.

---

## Phase 1 — what's deployed

```
Composer (host :9000) — bound to 0.0.0.0
  ▲                                             │
  │  manyforge.assistant.provider_request.v0   │  /api/assistant/modes/{mode}
  │  (unchanged today; will route through       │     (HTTP GET, manifest)
  │   Phase 2 adapter once it lands)            │  /api/assistant/bridge/tools/{toolId}
  │                                             │     (HTTP POST, bounded-autonomy
  │                                             │      envelope: mode + catalogHash
  │                                             │      + requestId + conversationId)
  │                                             ▼
manyforge_assistant_bridge       my-assistant sandbox (OpenClaw)
  → vLLM (today's path)            ├─ skill: manyforge-composer
                                   │    (procedural knowledge + bridge)
                                   ├─ MCP: manyforge (stdio)
                                   │    └─ python3 manyforge-mcp-bridge.py
                                   │       (mode-scoped wrapper: fetches the
                                   │        manifest, exposes only mode-allowed
                                   │        tools, stamps each call with the
                                   │        bridge envelope)
                                   │
                                   └─ vLLM (host :8000) for inference
```

Phase 1 only sets up the right side. Phase 2 redirects the left side
through it.

### Files (all versioned in source repos)

| Path | Role |
|---|---|
| `manyforge/agent-skills/manyforge-composer/SKILL.md` | OpenClaw skill — vocabulary, tool routing, canonical ids, anti-patterns, recovery protocol, worked examples. Frontmatter `metadata.contract` declares the assistant mode the skill is rev'd against; the provisioner refuses to install if the running Composer doesn't expose that mode. |
| `manyforge/agent-skills/manyforge-composer/manyforge-mcp-bridge.py` | Symlink → `manyforge/scripts/manyforge-mcp-bridge.py` (single-source). Bundled into the skill at install time. The script is a *mode-scoped MCP wrapper*: it fetches the manifest from `/api/assistant/modes/{mode}`, exposes only the tools that mode permits, and forwards each `tools/call` to `/api/assistant/bridge/tools/{toolId}` with a full bounded-autonomy envelope. |
| `dev_ws/src/NemoClaw-Thor/manyforge/policies/manyforge-composer.preset.yaml` | NemoClaw custom egress preset opening `host.openshell.internal:9000` to the agent's permitted binaries. |
| `dev_ws/src/NemoClaw-Thor/manyforge/setup-manyforge-assistant.sh` | Idempotent provisioner. Verifies that Composer exposes the configured assistant mode (refuses to install otherwise), applies the preset, stages the skill, installs it, registers the MCP server with the mode + principal env. |

### What the provisioner does — the four official routes

Each step uses an officially supported NemoClaw / OpenClaw command. No
filesystem patches, no kubectl `cp`, no /tmp persistence.

1. **Egress preset** — `nemoclaw <sandbox> policy-add --from-file
   policies/manyforge-composer.preset.yaml`. The preset declares a
   `network_policies.manyforge_composer` block scoped to
   `host.openshell.internal:9000` for the same trusted binaries the
   built-in `local-inference` preset already permits (`/usr/local/bin/openclaw`,
   `/usr/local/bin/node`, `/usr/bin/python3`). Idempotent: re-run skips when
   already applied (status visible in `policy-list`).

2. **Skill install** — `nemoclaw <sandbox> skill install <staging>`. The
   provisioner creates a `mktemp -d` staging directory, dereferences the
   skill's symlinked bridge with `cp -L`, runs the install, and removes
   the staging dir on `trap EXIT`. Files land in
   `/sandbox/.openclaw/skills/manyforge-composer/` inside the sandbox PVC.

3. **MCP server registration** — `openclaw mcp set manyforge '{...}'`
   inside the sandbox (via `kubectl exec ... su sandbox -c`). Config
   shape per the OpenClaw docs:

   ```json
   {
     "command": "python3",
     "args": ["/sandbox/.openclaw/skills/manyforge-composer/manyforge-mcp-bridge.py"],
     "env": {
       "MANYFORGE_COMPOSER_BASE": "http://host.openshell.internal:9000",
       "MANYFORGE_ASSISTANT_MODE": "composer-assistant",
       "MANYFORGE_PRINCIPAL": "openclaw-my-assistant"
     }
   }
   ```

   This persists into `/sandbox/.openclaw/openclaw.json`. The bridge is
   spawned as a stdio child by OpenClaw on agent runs. On each
   `tools/list` it fetches the mode manifest from
   `/api/assistant/modes/${MANYFORGE_ASSISTANT_MODE}`; on each
   `tools/call` it POSTs to
   `/api/assistant/bridge/tools/${tool_id}` with the bounded-autonomy
   envelope. A 409 catalog-hash mismatch (deployment hot-reloaded)
   triggers exactly one manifest refresh + retry.

4. **Composer launch flags** — the demo launch script passes
   `--host ${COMPOSER_BIND_HOST:-0.0.0.0}` (sandbox-reachable bind via
   `host.openshell.internal:9000`). The mode-scoped MCP wrapper reaches
   `/api/assistant/modes/{mode}` and `/api/assistant/bridge/tools/{toolId}`,
   which are part of Composer's standard assistant routes (always mounted).
   The `--mcp-http` flag is no longer required for the bounded-autonomy
   path — keep it on if you want the broad `/api/mcp` surface available
   for operator tooling, but the agent does not use it.

### Verification commands and expected outputs

Run these in order after `setup-manyforge-assistant.sh` to confirm
end-to-end reachability.

**a. Composer is bound on the sandbox-reachable interface:**

```bash
ss -tlnp | grep ":9000"
# expect: 0.0.0.0:9000  (not 127.0.0.1:9000)
```

**b. Mode manifest is reachable from the host:**

```bash
curl -fsS http://localhost:9000/api/assistant/modes/composer-assistant | \
  python3 -m json.tool | head -20
# expect: a JSON object with `mode`, `catalogHash`, `tools[]`, `nodes[]`.
# `tools[]` should include tree.draft.wrap_node, scene.draft.add_object,
# program.read, etc., each with `description`, `effect`, `inputSchema`.
```

**c. Sandbox can reach the mode manifest (the wrapper's discovery call):**

```bash
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell my-assistant -c agent \
  -- su sandbox -c \
  "curl -fsS --max-time 5 \
    http://host.openshell.internal:9000/api/assistant/modes/composer-assistant | head -c 400"
# expect: same JSON body, truncated.
```

**d. Stdio wrapper exposes the mode-scoped tool list:**

```bash
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell my-assistant -c agent \
  -- su sandbox -c \
  "echo '{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"initialize\",\
         \"params\":{\"protocolVersion\":\"2024-11-05\",\
         \"clientInfo\":{\"name\":\"x\",\"version\":\"1\"},\
         \"capabilities\":{}}}\n\
{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n\
{\"jsonrpc\":\"2.0\",\"id\":\"2\",\"method\":\"tools/list\"}' | \
   MANYFORGE_COMPOSER_BASE=http://host.openshell.internal:9000 \
   MANYFORGE_ASSISTANT_MODE=composer-assistant \
   python3 /sandbox/.openclaw/skills/manyforge-composer/manyforge-mcp-bridge.py"
# expect: only the tools the mode permits (tree.draft.*, scene.draft.*,
# program.read, catalog.read, skills.read). NOT the broad operator tools
# (manyforge_runtime_override, manyforge_intervention, manyforge_program_save).
```

**e. The OpenClaw agent sees the manyforge MCP server and can call it:**

```bash
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell my-assistant -c agent \
  -- su sandbox -c \
  "openclaw agent --agent main \
    --message 'Use the manyforge MCP server. Call program.read and report the tree root node name.' \
    --json --timeout 120"
# expect: toolSummary.tools includes 'manyforge__program.read' or similar;
# finalAssistantVisibleText reports the root node name from the loaded program.
# The audit log on Composer's side carries the assistant mode + catalog hash
# + the openclaw-* requestId for that call (visible in bridge audit records).
```

The skill is loaded on demand, not eagerly: only the skill metadata
(name + description) is in the agent's prompt context until a manyforge
request prompts the agent's `read` tool to fetch the SKILL.md body. We
verified this by asking *"List your installed skill names"* (no `read`
calls) and *"What does the manyforge-composer skill say about Repeat?"*
(one `read` call, correct extracted answer).

### Industrial properties

- **Bounded autonomy is enforced server-side.** Every tool call from
  OpenClaw transits `/api/assistant/bridge/tools/{toolId}`, which checks
  `assistantMode`, validates `catalogHash`, enforces the mode tool
  allowlist, gates effect-vs-mode (`composer_draft_mutating` only in
  composer-assistant mode), enforces the node-kind allowlist for
  tree-edit tools, and applies the same recovery-hint payloads the
  in-Composer assistant sees. The MCP wrapper narrows the *visible*
  surface client-side; the bridge endpoint is the source of truth.
- **Request identity is preserved.** The wrapper assigns a fresh
  `requestId` per tool call and a stable `conversationId` per stdio
  session, both stamped onto the bridge envelope. Composer's bridge
  audit records carry these alongside `assistantMode`, `catalogHash`,
  `principal`, and `stuckTool` / `stuckRepeatCount` when applicable —
  so OpenClaw-driven calls are attributable in the same audit trail
  as direct-vLLM calls.
- **Skill-vs-runtime compatibility check at install time.** The skill's
  frontmatter declares the assistant mode it expects; the provisioner
  GETs `/api/assistant/modes/{mode}` before installing the skill and
  refuses if the mode is not loaded or the endpoint is unreachable. This
  catches the "stale skill, hot-reloaded deployment" failure mode at
  install rather than at first MCP call.
- **No /tmp persistence.** Source files live in versioned repos.
  `mktemp -d` is used only at install time inside
  `setup-manyforge-assistant.sh`, removed via `trap EXIT`. The skill's
  persistent home in the sandbox is `/sandbox/.openclaw/skills/<name>/`,
  which is OpenClaw's documented user-skill path and persists in the
  sandbox PVC across restarts.
- **Single-source repo files.** `manyforge-mcp-bridge.py` lives at one
  canonical path (`manyforge/scripts/manyforge-mcp-bridge.py`); the skill
  bundles it via symlink. The provisioner dereferences the symlink at
  install time so the upload is a flat directory, but no copy lands in
  any persistent repo path.
- **Idempotent provisioning.** `setup-manyforge-assistant.sh` checks
  `policy-list` before applying the preset and re-runs `skill install`
  (which OpenClaw treats as an update). MCP registration overwrites the
  same key in `openclaw.json`. Re-running the script is safe.
- **No model-side filesystem access.** OpenClaw's `read` tool runs
  inside the sandbox (Node.js); the model only consumes prompt text. The
  skill's content is delivered via tool-result messages to the model,
  never via direct disk access from the model.
- **Officially supported routes only.** `policy-add --from-file`,
  `skill install`, `openclaw mcp set` are all documented CLI commands
  with stable surfaces. No NemoClaw upstream patches.

### Known gaps (deployment-side)

- The `/api/mcp` and `/api/assistant/mcp/{mode}` endpoints are
  unauthenticated. Acceptable for single-developer experimentation against
  `host.openshell.internal`. Required before multi-tenant or non-loopback
  deployment: shared-secret token bound to a principal name, or a
  stronger network boundary (Unix socket / loopback bind + sandbox-side
  proxying). Tracked in open-points.

### Reproduce from a clean lane

```bash
cd ${HOME}/workspaces/dev_ws/src/NemoClaw-Thor
./manyforge/setup-manyforge-assistant.sh         # default sandbox: my-assistant

# Plus, in the manyforge demo lane:
cd ${HOME}/workspaces/dev_ws/src/manyforge
REBUILD_FRONTEND_ON_COMPOSER_RELOAD=false \
  ./scripts/demo-assistant-known-good.sh restart
```

Then run the verification commands above.

---

## Phase 2 — composer's chat endpoint routes through OpenClaw

**Status:** experimental live route validated on 2026-05-05 for
`composer-assistant`.

The composer's existing `openclaw` provider at
`manyforge/manyforge_composer/backend/assistant_provider.py:584` accepts
any HTTP endpoint speaking the `manyforge.assistant.provider_request.v0`
contract. Today it points at `manyforge_assistant_bridge` (which runs the
agent loop directly against vLLM). Phase 2 introduces a new endpoint that
runs the agent loop through OpenClaw inside the sandbox instead.

### Scope

A small adapter service — **`openclaw_assistant_bridge`** — that:

- Listens on HTTP for `manyforge.assistant.provider_request.v0` requests,
  same shape composer already speaks.
- For each request, invokes the OpenClaw agent inside the sandbox via
  `kubectl exec ... openclaw agent --local --thinking off --agent main
  --session-id <conversation-or-request-id> --message <user-msg>
  --json --timeout <t>` (or the equivalent gateway WebSocket call).
- Parses the JSON output from stdout or stderr: extracts
  `payloads[].text`, `finalAssistantVisibleText`, the tool summary, and
  any tool-call envelopes.
- Translates these into the composer's expected envelope: proposals,
  `toolCalls[]`, `draftMutated` flag, warnings, `requestId`, etc.
- Returns the standard 200 envelope (or `_envelope_error` shape) to the
  composer.

### Where it lives

Under the post-refactor layout, sibling to the existing
`manyforge/bridge/` audit-log mount in this repo:

```
dev_ws/src/NemoClaw-Thor/
└── manyforge/                                   # ManyForge integration scope
    ├── bridge/                                  # audit-log mount for the
    │                                            # in-ManyForge bridge service
    │                                            # (vLLM path runs there)
    ├── openclaw_assistant_bridge/               # new — Phase 2
    │   ├── README.md
    │   ├── adapter.py                           # prompt/output translation
    │   ├── service.py                           # FastAPI HTTP layer
    │   ├── requirements.txt
    │   ├── README.md
    │   └── tests/
    ├── policies/manyforge-composer.preset.yaml
    ├── setup-manyforge-assistant.sh
    └── docs/
        └── MANYFORGE-MCP-INTEGRATION.md         # this doc
```

Versioned with the deployment recipe. The only manyforge-side runtime
selection is pointing `MANYFORGE_ASSISTANT_ENDPOINT_URL` at the new
service and setting `MANYFORGE_ASSISTANT_PROVIDER=openclaw`.

### Composer-side switch (no code change)

```bash
cd ${HOME}/workspaces/dev_ws/src/NemoClaw-Thor
./manyforge/start-openclaw-assistant-bridge.sh

# in the Composer launch environment:
MANYFORGE_ASSISTANT_PROVIDER=openclaw
MANYFORGE_ASSISTANT_ENDPOINT_URL=http://127.0.0.1:8200/v1/manyforge/assistant
```

The composer's `openclaw` provider id (already in
`assistant_provider.py:648-657`) is a synonym of `nemoclaw` with the same
contract — the only change is the endpoint URL.

### Adapter contract sketch

Request (existing, unchanged):

```json
{
  "version": "manyforge.assistant.provider_request.v0",
  "requestId": "...",
  "providerId": "openclaw",
  "conversationId": "...",
  "message": "<user prompt>",
  "tools": [...],            // composer-resolved tool catalog (visible to model)
  "context": {...},
  "runtime": {...},
  "constraints": {...}
}
```

Response (existing, unchanged):

```json
{
  "version": "manyforge.assistant.provider_request.v0",
  "schemaVersion": "0.1.0",
  "requestId": "...",
  "message": "<final assistant text>",
  "toolCalls": [...],
  "proposals": [],
  "warnings": [],
  "mutated": false,
  "draftMutated": <bool>,
  "requiresReview": true
}
```

Adapter internals:

```python
async def assistant(req: Request):
    payload = await req.json()
    # 1. Construct the agent prompt: include the composer-visible tool list
    #    in a structured preamble (the agent will route through MCP, so we
    #    don't pass the tools as OpenAI tools — they reach the agent via
    #    the manyforge MCP server already).
    agent_prompt = build_prompt(payload)
    # 2. Invoke the in-sandbox agent through docker/kubectl exec.
    result = await invoke_openclaw_agent(
        sandbox="my-assistant",
        agent="main",
        message=agent_prompt,
        timeout_s=payload.get("timeoutSeconds", 180),
    )
    # 3. Translate the agent's tool-call summary into the composer envelope.
    return translate_to_envelope(result, request_id=payload["requestId"])
```

### What stays unchanged (important)

- ManyForge's mode-scoped assistant callback surface is the canonical
  authority. Every ManyForge tool the agent calls goes through the
  provisioned `manyforge` MCP wrapper, which forwards to
  `host.openshell.internal:9000/api/assistant/bridge/tools/{toolId}`
  with `assistantMode`, `catalogHash`, `requestId`, `conversationId`,
  and `principal`. No tool effect can reach state without ManyForge's
  mode allowlists, catalog validation, scope rules, and draft-boundary
  enforcement.
- The composer's `assistant_provider.py` is unchanged. The provider is
  just being pointed at a different endpoint.
- The `manyforge_assistant_bridge` (vLLM path) stays in the repo as the
  baseline for A/B comparison. We do not delete it during Phase 2.

### Estimated work

Initial skeleton:

- FastAPI service translating the contract.
- Dependency-light adapter module with unit tests for prompt building,
  command construction, JSON extraction, tool-name canonicalization, and
  envelope normalization.
- `manyforge/start-openclaw-assistant-bridge.sh` launcher.

Remaining validation:

- Live smoke proving OpenClaw emits an MCP `tools/call`, the MCP wrapper
  forwards it to `/api/assistant/bridge/tools/{toolId}`, and the adapter
  returns the expected Composer envelope.

### Spec note

The normative bridge spec currently locks the direct-vLLM bridge as the
v0.1 assistant request path and explicitly says that bridge does not
delegate to the sandbox runtime. This Phase 2 adapter is therefore an
experimental deployment-side alternate provider until the specs are
updated or the experiment is retired.

---

## Phase 3 — A/B harness

**Status:** designed, not implemented.

After Phase 2 lands, run the same prompt suite through both endpoints and
compare. Metrics from the existing audit log (already populated for the
vLLM path; populated by the new adapter for the OpenClaw path):

| Metric | Source |
|---|---|
| Success rate (final draft matches expected mutation) | composer's `/api/program/tree` after each request |
| Tool call count per request | bridge audit `toolCallCount` |
| Wall time | bridge audit `durationMs` |
| Invalid tool calls (4xx) | bridge audit `failures` |
| Stuck-loop exits | bridge audit `exit_reason: stuck_loop` |
| Unwanted draft mutations (collateral edits) | composer scene/program diff |
| Final assistant message coherence | manual or LLM-judge review |

Suite: the three demo prompts plus the synthetic regression set used in
`test_assistant_routes.py` rephrased as natural-language requests. Run
each prompt N=10 times against each endpoint to capture variance.

The harness is a simple Python script that invokes
`POST /api/assistant/chat` with the same `message` payload, varying
`MANYFORGE_ASSISTANT_PROVIDER` between runs (or running side-by-side
composer instances on different ports).

---

## Validation log

### 2026-05-05 — Phase 1 mode-scoped path, wrapper-layer end-to-end

Stack at validation time: vLLM `nemotron3-nano-omni-30b-a3b-nvfp4` on
`:8000`; Composer reloaded against the new code with
`REBUILD_FRONTEND_ON_COMPOSER_RELOAD=false` (2 s startup); assistant
bridge on `:8100`; sandbox `my-assistant` running OpenClaw against the
local vLLM endpoint; deployment YAML
`examples/assistant_modes_scene_authoring.deployment.yaml` loaded into
Composer.

Passes:

- **1a — host probe of `GET /api/assistant/modes/composer-assistant`**:
  HTTP 200; 23 mode-permitted tools (including `tree.draft.wrap_node`),
  12 allowed node kinds, catalogHash `710496b1e801997d…`.
- **1b — `manyforge/setup-manyforge-assistant.sh my-assistant`**:
  runtime-compat precheck found the same catalogHash; egress preset
  applied; skill staged + installed; MCP server registered with
  `{MANYFORGE_COMPOSER_BASE, MANYFORGE_ASSISTANT_MODE,
  MANYFORGE_PRINCIPAL}` env shape; sandbox-side fetch returned the
  same catalogHash. (One small staging-loop hardening landed during
  this run: skip `__pycache__/*.pyc/node_modules` and bare directories.)
- **1c — stdio wrapper inside the sandbox** (`initialize` +
  `tools/list`): wrapper reports `manyforge-composer 0.2.0
  mode=composer-assistant`; returns the same 23 tools the manifest
  exposed; **zero broad-MCP leakage** (no `manyforge_*` operator
  tools).

Composer's HTTP audit log during the Phase 1 wrapper run showed only
mode-scoped manifest fetches from `172.18.0.2` (sandbox); no `/api/mcp`
traffic and no broad-tool exposure.

### Phase 2 live validation — 2026-05-05

After the adapter fixes for `--thinking off`, stderr JSON parsing, and
per-request/per-conversation OpenClaw session ids:

- **2a — Composer configured with provider `openclaw`:** `GET
  /api/assistant/providers` returned `configuredProviderId: openclaw`;
  adapter `/healthz` returned `status: ok` on `:8200`.
- **2b — read-only tool round-trip through Composer chat:** prompt forced
  `manyforge__catalog-read` for `remove_collision_object`. Composer logs
  showed `POST /api/assistant/bridge/tools/catalog.read 200`; the chat
  response returned HTTP 200 with message "The catalog entry id for
  remove_collision_object is remove_collision_object."
- **2c — draft-mutating tool round-trip through Composer chat:** prompt
  forced `manyforge__tree-draft-wrap_node` with `targetName: @root` and
  wrapper `{id: repeat, name: repeat_node}`. Composer logs showed `POST
  /api/assistant/bridge/tools/tree.draft.wrap_node 200`; the provider
  response returned HTTP 200 with `draftMutated: true` and a completed
  `tree.draft.wrap_node` tool call; `GET /api/program/tree` then showed
  `repeat/repeat_node` as root with `sequence/pick_and_place` as its
  child.

Known Phase 2 limitations:

- The sandbox MCP server is currently registered for `composer-assistant`;
  a query-mode smoke asking for `program.read` timed out because that was
  not the mode-scoped tool surface under test.
- Latency is high on the local Omni route. Follow-up measurements after
  request-scoped MCP tool-window support showed:
  - host-to-sandbox shell launch (`docker exec` -> `kubectl exec` -> `su`) is
    about 220-260 ms and is not the bottleneck;
  - a trivial direct vLLM call with a capped response can complete in under one
    second;
  - a trivial OpenClaw agent turn with `--local --thinking off` and a narrowed
    ManyForge MCP tool window still takes about 58-61 s;
  - an unconstrained Composer root-wrap prompt through OpenClaw still timed out
    at the 180 s Composer provider budget before issuing a ManyForge callback.
  Direct vLLM remains the known-good demo default until the Phase 3 A/B harness
  quantifies reliability and latency.
- The OpenClaw adapter now passes request-scoped tool windows via a short-lived
  sandbox file (`/tmp/manyforge-openclaw-allowed-tools.txt`) because OpenClaw's
  configured MCP server environment is static and does not inherit
  per-invocation environment variables.
- The adapter also filters prompt-visible tool descriptions to the same
  request-scoped window. A one-tool `catalog.read` probe dropped the adapter
  prompt from ~7.6k chars to ~1.3k chars, but still timed out at 120 s; prompt
  size reduction alone does not solve local OpenClaw turn latency.
- `manyforge/setup-manyforge-assistant.sh` now installs a dedicated
  `manyforge-composer` OpenClaw agent profile (`skills: [manyforge-composer]`,
  `tools.profile: minimal`). The Phase 2 bridge launcher defaults to this
  agent. A direct "Reply exactly OK" probe through that profile still took
  ~50 s with a ~9.6k-char OpenClaw system prompt, so remaining work is in
  OpenClaw runner/provider behavior rather than ManyForge callback handling.
- Timeout/cancel cleanup must kill the sandbox-side `openclaw agent` process by
  session id; killing only the host `docker exec` process can leave a model
  request running after Composer has timed out.
- The active vLLM container reported 0% prefix-cache hit rate during these
  probes. The Nemotron-3 Omni launch profile now adds
  `--enable-prefix-caching` on the next model restart unless disabled with
  `THOR_ENABLE_PREFIX_CACHING=0`. A 2026-05-05 post-restart check confirmed
  the running `manyforge-e2e-vllm` container includes the flag, but a direct
  `manyforge-composer` OpenClaw "OK" turn still took ~81 s wall time and vLLM
  still stayed slow. Direct vLLM controls prove prefix caching works on this
  server: an identical ~10k-token prompt took 3.6 s on the first call, then
  ~0.58 s on repeated calls with ~8.5k cached tokens. Repeated OpenClaw
  same-session calls also produced partial cache hits (~2.1k cached tokens per
  turn), but still took ~58 s and ~123 s. Before changing `max_num_seqs` or
  `max_num_batched_tokens`, inspect OpenClaw output-token budgets, hidden
  reasoning/request shaping, dynamic prompt sections, and whether OpenClaw is
  issuing multiple model turns for simple requests.
- OpenClaw's JSON output may omit detailed tool arguments/results even
  when the bounded ManyForge callback succeeded; Composer audit logs remain
  the source of truth for exact callback payloads in this phase.

### Phase 2 latency leap — 2026-05-05 evening (canonical fix)

After the post-restart prefix-caching probe pinpointed the cost as
OpenClaw bootstrap (not vLLM, not the prompt), we switched the adapter
from per-request `openclaw agent` shell-out to the persistent gateway's
`/v1/chat/completions` endpoint.

#### What we shipped (all via official routes — no workarounds)

- **OpenClaw config**: enabled `gateway.http.endpoints.chatCompletions.enabled
  = true`; pruned 4 unused plugins (`browser`, `device-pair`,
  `phone-control`, `talk-voice`). Gateway boots with `1 plugin: acpx;
  2.6s` instead of `5 plugins; 3.2s`.
- **OpenShell policy**: shipped `manyforge-composer.preset.yaml` with the
  canonical `allowed_ips` field on both endpoints
  (`host.openshell.internal:8000` for vLLM and `:9000` for Composer),
  per [OpenShell policy schema](https://docs.nvidia.com/openshell/latest/reference/policy-schema.html)
  and `OpenShell/examples/private-ip-routing`.
  **REVISED 2026-06-03**: the provisioner now applies BOTH
  `local-inference` AND `manyforge-composer` (it used to remove
  `local-inference` — see "Local-inference removal was wrong" below
  for the empirical evidence that prompted the change).
  Per [THREE-LANE-MIGRATION-PLAN.md §4.6](./THREE-LANE-MIGRATION-PLAN.md),
  the canonical post-Phase-1 layout splits this into three files:
  `manyforge-egress-shared.yaml` (egress rules only) + a per-lane
  binary-whitelist overlay (`manyforge-openclaw.overlay.yaml` or
  `manyforge-hermes.overlay.yaml`). The fused
  `manyforge-composer.preset.yaml` is retained for backward
  compatibility until Phase 5 retires it.
- **Adapter**: [openclaw_assistant_bridge/adapter.py](../openclaw_assistant_bridge/adapter.py)
  has a gateway-HTTP path gated by `OPENCLAW_ASSISTANT_USE_GATEWAY=true`,
  using a host-side curl through the openshell port-forward tunnel
  (`127.0.0.1:18789`) — the path the canonical
  `configure-local-provider.sh` already maintains.

#### Measured results

Same prompt ("Reply with just OK"), through the full canonical stack
(host → SSH tunnel → SSH-session gateway → vLLM):

- Cold call: ~47-57 s.
- Warm calls: 5-12 s (best 5.5 s observed; matches the speed the user
  sees in the TUI). **10-20× speedup over the original 65-130 s CLI
  shell-out baseline.**

Composer audit log + OpenShell policy log show only allowed traffic;
no SSRF DENY events fire on the canonical lane after the policy update.

#### Local-inference removal was wrong (REVISED 2026-06-03)

The earlier theory — that `manyforge-composer` was a strict superset
of `local-inference` and therefore the latter could be removed — turned
out to be empirically wrong for the chat-completion POST path. With
`local-inference` removed and only `manyforge-composer` active, the
OpenShell network proxy (10.200.0.1:3128) denies sandbox→host:8000
POSTs with `policy_denied`. Verified during the Phase 0 O-1 baseline
investigation: see [PHASE-0-LANE-BASELINE.md](./PHASE-0-LANE-BASELINE.md)
and [PIPELINE-TRACE-2026-06-03.md](./PIPELINE-TRACE-2026-06-03.md).

The correct configuration is to apply BOTH presets:

- `local-inference` (built-in) — covers the proxy's L7 allowlist for
  the inference endpoint (this is the field whose absence was causing
  the denials).
- `manyforge-composer` (custom) — adds the Composer :9000 endpoint
  with mode-scoped path rules and the `allowed_ips` for the SSRF guard.

`setup-manyforge-assistant.sh` was updated to apply both — see the
commit history for the `Step 1/5` change.

The earlier SSRF reasoning still applies at the SSRF-guard layer, but
the OpenShell proxy's L7 policy rejection (the new finding) is a
different enforcer that requires the built-in preset's contribution
to function. Both layers must be satisfied for the route to work.

This is configure-only. No NemoClaw / OpenShell / OpenClaw upstream
patches required. A future upstream improvement worth proposing would
be adding `allowed_ips` to the built-in `local-inference` preset by
default, or making the SSRF engine union allowlists across matching
presets.

#### Side findings (not blockers, recorded for completeness)

- `openclaw gateway restart` does not auto-respawn the gateway in
  this sandbox — there is no systemd/launchd inside it, and the only
  CMD is `sleep infinity`. The canonical
  `ensure_sandbox_gateway_running` function in
  `setup/sandbox-runtime.sh` handles spawn via `nohup openclaw
  gateway run`; if the gateway crashes, re-running
  `configure-local-provider.sh` brings it back. No daemon supervisor
  exists in the design.
- The OpenClaw gateway listens in the network namespace it was spawned
  in. The canonical (SSH-session) namespace is reachable from the host
  via the openshell port-forward tunnel; fresh `kubectl exec` shells
  enter a different namespace and cannot see it. This matters when
  testing — always probe via host curl, not `kubectl exec curl`.

### Phase 3 A/B harness — first run, 2026-05-05 evening

After the canonical-fix wins landed, ran [`manyforge/ab-direct-vs-openclaw.py`](../ab-direct-vs-openclaw.py)
with N=3 over 3 prompts (trivial "OK", short factual, decorator
description) against both inference paths. The full results live in
`/tmp/ab-results.json`; aggregate:

| path | runs | success | P50 | P95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| direct_vllm | 9 | 100% | **2.14 s** | 2.78 s | 0.35 s | 2.76 s |
| openclaw_gw | 9 | 100% | **15.04 s** | 67.30 s | 4.92 s | 61.19 s |

Reliability: 18/18 calls returned non-empty content on both paths.
**The Nemotron #71847 null-content fingerprint is gone** after we
applied `chat_template_kwargs: {enable_thinking: false,
force_nonempty_content: true}` via
`agents.defaults.models.<id>.params.chat_template_kwargs` — that is
the canonical OpenClaw config knob per
[docs.openclaw.ai/providers/vllm](https://docs.openclaw.ai/providers/vllm).

Latency: direct vLLM is **~7× faster at P50, ~24× faster at P95**.
OpenClaw's agent loop adds a roughly constant 5-7 s on simple prompts
(visible as the trivial "OK" minimum) plus high variance (5-67 s)
that scales with how much internal reasoning the model decides to do
even with `enable_thinking: false`. The variance is the next thing to
investigate before declaring the OpenClaw lane production-default;
direct vLLM remains the known-good demo path.

Side observation: model accuracy is similar across paths (both
hallucinate "MCP" answers) but OpenClaw's responses tend to be shorter
and more conservative ("the context does not specify"), consistent
with the slim `manyforge-composer` skill's prose biasing the answer
shape. Worth quantifying with a real prompt suite (Phase 3 expansion).

#### How to reproduce

```bash
cd dev_ws/src/NemoClaw-Thor
./manyforge/ab-direct-vs-openclaw.py --runs 5 --json /tmp/ab.json
```

Default uses 3 prompts × N runs × 2 paths. `--paths direct_vllm` or
`--paths openclaw_gw` runs only one side. Results: per-call wall +
content + success table, plus aggregates.

---

## Phase 4 — Hermes lane mcp_servers (landed)

The Hermes lane reaches the same composer `/api/assistant/bridge/tools/{toolId}`
mutation path as OpenClaw, but via Hermes' **native** `mcp_servers` config rather
than an OpenClaw `mcp set`. No wrapper code — the integration is ~30 lines of
YAML ([`lanes/hermes/mcp_servers_config.yaml`](../lanes/hermes/mcp_servers_config.yaml))
pointing Hermes at the lane-neutral `manyforge-mcp-bridge.py` (which runs with
`MANYFORGE_LANE=hermes`, `MANYFORGE_PRINCIPAL=hermes-<sandbox>`). Hermes registers
the tools with an `mcp_manyforge_` prefix; the bridge's progress observer strips
it for cross-lane audit parity (`common.tool_calls.strip_mcp_prefix`).

**Emission strategy (Q7):** the direct-config-write path — `setup-hermes.sh`
renders the YAML (env-substituted) into `/sandbox/.hermes/config.yaml` **before**
the gateway starts. This is required because (spike probe 3 online finding) the
`mcp_servers` auto-reload watcher only runs under Hermes' interactive CLI, **not**
the gateway. So: inject before start; any later change needs a gateway `recover`
(NOT `rebuild`, which wipes the config). The `hermes-config.ts` overlay remains a
valid alternative for upstream-friendliness but is not required for the lane to
work. `NO_PROXY` must NOT include `host.openshell.internal` — the bridge reaches
Composer only through the OpenShell proxy at `10.200.0.1:3128`. Full bring-up:
[PHASE-4-HERMES-LONGITUDINAL.md](./PHASE-4-HERMES-LONGITUDINAL.md).

---

## Cross-references

- Runtime tree-mutation hardening (manyforge side):
  `dev_ws/src/manyforge/manyforge_assistant_bridge/` and
  `dev_ws/src/manyforge/manyforge_composer/backend/routes_assistant.py`.
- Wire contract that both paths preserve:
  [`ASSISTANT_PROVIDER_CONTRACT.md`](https://github.com/pastoriomarco/manyforge/blob/main/docs/reference/ASSISTANT_PROVIDER_CONTRACT.md).
- NemoClaw onboarding workflow this integration plugs into:
  `../../setup/NEMOCLAW-OPENCLAW-WORKFLOW.md`.
- Profile selection for the assistant model:
  `MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md`.
- OpenClaw skill format and `mcp set` semantics: `https://docs.openclaw.ai/`.

---

## Removal criteria

This integration is intended to be permanent. The Phase-1 wiring becomes
the production deployment shape once Phase 2 lands; the direct-vLLM path
in `manyforge_assistant_bridge/` may eventually retire after Phase 3 shows
the OpenClaw-routed path matches or exceeds the baseline on every metric.

The custom egress preset (`../policies/manyforge-composer.preset.yaml`) is
the only piece that may need updating if the composer's MCP endpoint moves
or if NemoClaw's preset format changes upstream. That's a small,
localized change.
