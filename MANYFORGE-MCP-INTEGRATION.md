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
  validated end-to-end on 2026-05-04 against the broad `/api/mcp`
  surface (capability proof). On 2026-05-05 the path was narrowed to the
  mode-scoped bridge endpoint: the in-sandbox MCP wrapper now translates
  every tool call into a `/api/assistant/bridge/tools/{toolId}` call with
  the bounded-autonomy envelope (`assistantMode`, `catalogHash`,
  `requestId`, `conversationId`, `principal`). Server-side enforcement —
  the same gates we use for the in-Composer assistant — is the source of
  truth. Provisioning artifacts are in this repo; setup is reproducible
  via `scripts/setup-manyforge-assistant.sh`.
- **Phase 2 — composer's chat endpoint routes through OpenClaw:** designed,
  not yet implemented. Sketch and contract are in this doc.
- **Phase 3 — A/B harness comparing direct-vLLM vs OpenClaw-skill paths:**
  designed, not yet implemented.

This is the deployment-side companion to the runtime hardening documented
in `manyforge_specs/docs/implementation/composer-assistant-tree-mutation-hardening.md`
(sibling repo to `manyforge`).

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
| `nemoclaw/src/NemoClaw-Thor/policies/manyforge-composer.preset.yaml` | NemoClaw custom egress preset opening `host.openshell.internal:9000` to the agent's permitted binaries. |
| `nemoclaw/src/NemoClaw-Thor/scripts/setup-manyforge-assistant.sh` | Idempotent provisioner. Verifies that Composer exposes the configured assistant mode (refuses to install otherwise), applies the preset, stages the skill, installs it, registers the MCP server with the mode + principal env. |

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

### Known gaps tracked in `manyforge_specs/docs/open-points.md`

- The `/api/mcp` and `/api/assistant/mcp/{mode}` endpoints are
  unauthenticated. Acceptable for single-developer experimentation against
  `host.openshell.internal`. Required before multi-tenant or non-loopback
  deployment: shared-secret token bound to a principal name, or a
  stronger network boundary (Unix socket / loopback bind + sandbox-side
  proxying). Tracked in open-points.

### Reproduce from a clean lane

```bash
cd /home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor
./scripts/setup-manyforge-assistant.sh         # default sandbox: my-assistant

# Plus, in the manyforge demo lane:
cd ../../../dev_ws/src/manyforge
REBUILD_FRONTEND_ON_COMPOSER_RELOAD=false \
  ./scripts/demo-assistant-known-good.sh reload-composer
```

Then run the verification commands above.

---

## Phase 2 — composer's chat endpoint routes through OpenClaw

**Status:** designed, not implemented.

The composer's existing `openclaw` provider at
`manyforge/manyforge_composer/backend/assistant_provider.py:584` accepts
any HTTP endpoint speaking the `manyforge.assistant.provider_request.v0`
contract. Today it points at `manyforge_assistant_bridge` (which runs the
agent loop directly against vLLM). Phase 2 introduces a new endpoint that
runs the agent loop through OpenClaw inside the sandbox instead.

### Scope

A small adapter service — provisional name **`openclaw_assistant_bridge`** — that:

- Listens on HTTP for `manyforge.assistant.provider_request.v0` requests,
  same shape composer already speaks.
- For each request, invokes the OpenClaw agent inside the sandbox via
  `kubectl exec ... openclaw agent --agent main --message <user-msg>
  --json --timeout <t>` (or the equivalent gateway WebSocket call).
- Parses the JSON output: extracts `finalAssistantVisibleText`, the tool
  summary, and any tool-call envelopes.
- Translates these into the composer's expected envelope: proposals,
  `toolCalls[]`, `draftMutated` flag, warnings, `requestId`, etc.
- Returns the standard 200 envelope (or `_envelope_error` shape) to the
  composer.

### Where it lives

Sibling to the existing `bridge/` directory in this repo:

```
nemoclaw/src/NemoClaw-Thor/
├── bridge/                          # the existing assistant_bridge (vLLM path)
└── openclaw_assistant_bridge/       # new — Phase 2
    ├── README.md
    ├── service.py                   # the adapter (FastAPI, ~200 LoC)
    ├── pyproject.toml
    └── tests/
```

Versioned with the deployment recipe; no manyforge-side changes required
beyond pointing `MANYFORGE_ASSISTANT_ENDPOINT_URL` at the new service.

### Composer-side switch (no code change)

```bash
# in scripts/demo-assistant-known-good.sh, the BRIDGE_URL becomes:
BRIDGE_URL="${BRIDGE_URL:-http://127.0.0.1:8200}"   # openclaw_assistant_bridge
# and the env passed to the composer container:
-e MANYFORGE_ASSISTANT_PROVIDER=openclaw \
-e MANYFORGE_ASSISTANT_ENDPOINT_URL="${BRIDGE_URL}/v1/manyforge/assistant" \
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

Adapter internals (sketch):

```python
async def assistant(req: Request):
    payload = await req.json()
    # 1. Construct the agent prompt: include the composer-visible tool list
    #    in a structured preamble (the agent will route through MCP, so we
    #    don't pass the tools as OpenAI tools — they reach the agent via
    #    the manyforge MCP server already).
    agent_prompt = build_prompt(payload)
    # 2. Invoke the in-sandbox agent.
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

- ManyForge's MCP tool surface is the canonical authority. Every tool the
  agent calls goes back through `host.openshell.internal:9000/api/mcp` →
  composer's enforcement (mode allowlists, catalog validation, scope
  rules, draft boundaries). No tool effect can reach state without
  ManyForge's say-so.
- The composer's `assistant_provider.py` is unchanged. The provider is
  just being pointed at a different endpoint.
- The `manyforge_assistant_bridge` (vLLM path) stays in the repo as the
  baseline for A/B comparison. We do not delete it during Phase 2.

### Estimated work

~1 person-day:

- ~150 LoC FastAPI service translating the contract.
- Smoke test exercising one end-to-end agent call.
- Provisioning hook (systemd unit or `start-` script) sibling to the
  existing assistant bridge launcher.

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

## Cross-references

- Runtime hardening (manyforge side):
  `manyforge_specs/docs/implementation/composer-assistant-tree-mutation-hardening.md`
- Spec contract that both paths preserve:
  `manyforge_specs/docs/spec/430-deployment-artifact-schema.md` §3.5F,
  `480-assistant-modes-and-bounded-autonomy.md` §5.1,
  `485-assistant-bridge-architecture.md` §3.
- NemoClaw onboarding workflow this integration plugs into:
  `NEMOCLAW-OPENCLAW-WORKFLOW.md`.
- Profile selection for the assistant model:
  `MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md`.
- OpenClaw skill format and `mcp set` semantics: `https://docs.openclaw.ai/`.

---

## Removal criteria

This integration is intended to be permanent. The Phase-1 wiring becomes
the production deployment shape once Phase 2 lands; the direct-vLLM path
in `manyforge_assistant_bridge/` may eventually retire after Phase 3 shows
the OpenClaw-routed path matches or exceeds the baseline on every metric.

The custom egress preset (`policies/manyforge-composer.preset.yaml`) is
the only piece that may need updating if the composer's MCP endpoint moves
or if NemoClaw's preset format changes upstream. That's a small,
localized change.
