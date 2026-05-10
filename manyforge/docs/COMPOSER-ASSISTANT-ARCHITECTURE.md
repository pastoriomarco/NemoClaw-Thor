# Composer Assistant — Architecture Map for Refinement Work

Companion document to [`SMOKE-CORPUS.md`](./SMOKE-CORPUS.md). Captures the
end-to-end runtime that the smoke corpus exercises, where each piece of the
tool surface lives, and what changes when. Written 2026-05-10 to support the
post-iter-20 refinement plan (Pattern A/B/C/E from the failure analysis).

---

## TL;DR

- **Composer** (the user-facing FastAPI + React UI) sends a `manyforge.assistant.provider_request.v0`
  envelope over HTTP to a *bridge*. The bridge runs the multi-turn LLM loop
  and dispatches tool calls back into Composer's state.
- **Two bridges** can implement the bridge endpoint. They listen on different
  ports and use the same wire protocol; the deployment YAML's
  `model_endpoints.assistant_model.base_url` selects which one Composer hits:
  - `manyforge_assistant_bridge` (port 8100) — direct lane.
  - `openclaw_assistant_bridge` (port 8200) — OpenClaw lane (production default).
- **`vllm-proxy`** is independent of either bridge. It sits between
  whatever client makes chat-completion calls and vLLM, and is the only
  component that sees per-turn agent-loop traffic when the OpenClaw lane is
  active (because OpenClaw runs the agent loop server-side, not in the bridge).
- **Tool schemas** (the JSON-schemas for `tree_draft_insert_node`, etc.)
  live in `manyforge_composer/backend/assistant_tool_schemas.py`. They are
  served by Composer's `/api/assistant/bridge/tools` endpoint and reach the
  model through the bridge → vLLM `tools[]` array.
- **Node catalog** (the per-node descriptions that appear in the prompt's
  `nodeCatalog`, e.g. `upsert_collision_object`'s description) lives in
  `manyforge_behavior/resources/node_catalog.yaml`.
- **Smoke corpus** drives Composer's `/api/assistant/chat` endpoint with 74
  cases and grades each as pass / soft-pass / fail. **Best config (iter 20)**:
  Cosmos-Reason2-8B, proxy with `max_tokens=2048` injected, `enable_thinking:true`
  default, no `tool_choice` mutation, smoke runner with `--no-chain-session`.
  **49/66 effective (74.2 %), 45/66 first-try (68.2 %).**

---

## Repository layout

Two source trees are live in this work:

| Path | Role |
|------|------|
| `/home/tndlux/workspaces/dev_ws/src/manyforge` | **Manyforge core dev workspace** — Composer backend, bridges, behavior catalog, MCP wrapper. Mounted into the `manyforge-e2e-composer` container as `/workspace`. **Most refinement edits land here.** |
| `/home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor` | **NemoClaw-Thor session repo** — model serving (vLLM launchers), the OpenClaw bridge variant, smoke runner, this doc. |

The two are *separate repos* checked out side-by-side. Composer reads from
`dev_ws/src/manyforge`; the smoke runner and `vllm-proxy` live in
`NemoClaw-Thor/manyforge/scripts/debug/`.

---

## Component map (iter-20 production setup)

```
                             host machine (Thor)
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  ┌───────────────────────┐                                                 │
│  │ smoke_corpus_runner   │  (HTTP)                                         │
│  │ NemoClaw-Thor/.../    │ ───────────► http://127.0.0.1:9000              │
│  │ scripts/debug/        │              /api/assistant/chat                │
│  └───────────────────────┘                      │                          │
│                                                  ▼                         │
│   ┌──────────────────────────── manyforge-e2e-composer container ──────┐  │
│   │ Composer (React + FastAPI)                                          │  │
│   │ /workspace = /home/tndlux/workspaces/dev_ws/src/manyforge           │  │
│   │ Backend: manyforge_composer/backend/{routes_assistant.py,           │  │
│   │   assistant_provider.py, assistant_tool_schemas.py}                 │  │
│   │ Catalog: manyforge_behavior/resources/node_catalog.yaml             │  │
│   └─────────────┬───────────────────────────────────────────────────────┘  │
│                 │  manyforge.assistant.provider_request.v0 (HTTP POST)     │
│                 │  base_url from deployment YAML                           │
│                 ▼                                                          │
│   ┌──────── openclaw_assistant_bridge (host:8200) ─────────┐               │
│   │ NemoClaw-Thor/manyforge/openclaw_assistant_bridge/      │               │
│   │ adapter.py + service.py                                 │               │
│   │ Translates envelope → OpenClaw agent invocation         │               │
│   └─────────────┬───────────────────────────────────────────┘               │
│                 │  HTTP (gateway mode) over SSH tunnel host:18789           │
│                 ▼                                                          │
│   ┌──────── OpenClaw gateway (sandbox / cluster pod) ──────┐               │
│   │ Runs the agent loop with up to 3 concurrent subagents.  │               │
│   │ Holds the manyforge MCP wrapper (tools/list).           │               │
│   └─────────────┬───────────────────────────────────────────┘               │
│                 │  OpenAI-compatible chat-completions with tools[]          │
│                 ▼                                                          │
│   ┌──────── vllm-proxy (host:8000) ────────────────┐               │
│   │ NemoClaw-Thor/manyforge/scripts/debug/                  │               │
│   │ vllm-proxy.py                                   │               │
│   │ Logs every request/response as JSONL.                   │               │
│   │ Mutates outbound requests (max_tokens, enable_thinking, │               │
│   │ tool_choice, user-message suffix, …) when env vars set. │               │
│   └─────────────┬───────────────────────────────────────────┘               │
│                 │  forwards to upstream                                      │
│                 ▼                                                          │
│   ┌──────── vLLM (manyforge-e2e-vllm container, host:8050) ──┐             │
│   │ Cosmos-Reason2-8B (or other model) served as             │             │
│   │ `cosmos-reason2-8b`.                                      │             │
│   │ NemoClaw-Thor/serving/{launch.sh,start-model.sh}          │             │
│   └───────────────────────────────────────────────────────────┘             │
└────────────────────────────────────────────────────────────────────────────┘
```

Key port assignments in iter-20 production setup:

| Port | Process |
|------|---------|
| 9000 | Composer (frontend + backend, in `manyforge-e2e-composer` container) |
| 8200 | `openclaw_assistant_bridge` |
| 18789 | SSH tunnel to OpenClaw gateway in cluster sandbox |
| 8000 | `vllm-proxy` (the mutator/logger) |
| 8050 | vLLM serving the model |

---

## The two assistant lanes

Composer's deployment YAML
([`assistant_modes_scene_authoring.deployment.yaml`](/home/tndlux/workspaces/dev_ws/src/manyforge/examples/assistant_modes_scene_authoring.deployment.yaml))
points at one bridge:

```yaml
model_endpoints:
  assistant_model:
    base_url: http://127.0.0.1:8100/v1/manyforge/assistant   # direct lane
    # or
    base_url: http://127.0.0.1:8200/v1/manyforge/assistant   # OpenClaw lane
```

Both bridges expose the same routes:

- `GET  /healthz`
- `POST /v1/manyforge/assistant`              (the provider request)
- `POST /v1/manyforge/assistant/{rid}/cancel` (mid-flight cancel)

But what they do internally is very different.

### Direct lane — `manyforge_assistant_bridge`

Source: `/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_assistant_bridge/bridge.py`.

- The bridge runs the **multi-turn LLM loop locally**: it sends chat-completion
  requests to the upstream model, reads the response, dispatches tool calls
  by calling back into Composer (`POST /api/assistant/bridge/tools/...`),
  and feeds the tool result back to the model.
- The bridge sees every turn. The smoke runner's per-prompt request maps to
  N (≈ 1–6) bridge → upstream chat-completion calls.
- Upstream is configured via `BRIDGE_UPSTREAM_BASE_URL` env (default
  `http://127.0.0.1:8000/v1`).
- Tool-call format conversion (manyforge schema → OpenAI tools[] → back) is
  done here.

### OpenClaw lane — `openclaw_assistant_bridge`

Source: `/home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor/manyforge/openclaw_assistant_bridge/{adapter.py,service.py}`.

- The bridge **does NOT run the LLM loop**. It translates the
  `manyforge.assistant.provider_request.v0` envelope into a single OpenClaw
  agent invocation.
- OpenClaw runs its own agent loop server-side (with up to 3 concurrent
  subagents) and emits its own chat-completion calls to vLLM.
- The bridge sees ONE call per smoke-runner prompt. The per-turn
  agentic traffic is invisible to the bridge — it's between OpenClaw and vLLM.
- Two execution modes:
  - **Gateway mode** (`OPENCLAW_USE_GATEWAY=true`): persistent OpenClaw
    process holds the MCP wrapper; bridge POSTs to the gateway URL over the
    SSH tunnel. **This is the production default.**
  - **CLI mode**: bridge launches a fresh OpenClaw subprocess per request.
    Used for ad-hoc debugging.

#### Bridge-side periodic `/compact` (iter 32)

The bridge tracks a per-process counter keyed by gateway session-key
(`derive_gateway_session_key(payload)` = `conversationId + catalogHash + programRevision`).
When `OPENCLAW_ASSISTANT_COMPACT_EVERY_N=N` is set in the environment,
the bridge fires a `/compact` slash command to the gateway BEFORE
forwarding every Nth user prompt on that session-key (skipping #1).
Sequential — bridge waits for the compaction call to return before
forwarding the actual user message.

Rationale: OpenClaw's built-in auto-compaction (triggered by
`agents.defaults.contextTokens` overflow precheck) hits an
`already_compacted_recently` cooldown after the first overflow and
stops working. Spacing compactions at known boundaries from the bridge
sidesteps the cooldown entirely. Verified iter 32: 9 successive
compactions, all succeeded, no cooldown blocks. The compaction model
is configured via `agents.defaults.compaction.model` in
`/sandbox/.openclaw/openclaw.json` — set to `inference/cosmos-reason2-8b`
to route compaction through the local Cosmos model rather than the
unreachable default `gpt-5.5`.

Tunables:

- `OPENCLAW_ASSISTANT_COMPACT_EVERY_N` (int, default 0 = off): bridge
  fires `/compact` on session-counts `N+1, 2N+1, 3N+1, …`. The first
  request on a session-key never compacts (nothing to compact yet).
- `OPENCLAW_ASSISTANT_COMPACT_TIMEOUT_S` (float, default 120): timeout
  for the compact call itself. If exceeded the failure is logged and
  the user request still goes through.

Telemetry (`_log_event` JSONL on bridge stdout):

- `openclaw_compact_fire_started` — when the compact call is dispatched
- `openclaw_compact_fire_succeeded` — when the compact call returns
- `openclaw_compact_fire_failed` — when the compact call raises (with
  the exception class + message)

### Production default = OpenClaw lane

Per memory `project_lane_parity_cosmos8b.md` (2026-05-07): OpenClaw lane
beats the direct lane 9/9 vs 1/9 on the lane-parity probe with Cosmos-8B.
The smoke corpus best result (iter 32, 51/66 = 77.3 % under chain-session-on
with bridge-fired periodic `/compact`) was measured on the OpenClaw lane.

---

## The vllm-proxy (logger + mutator)

Source: `/home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor/manyforge/scripts/proxy/vllm-proxy.py`.
(Was `scripts/proxy/vllm-proxy.py` before iter 21 — renamed +
relocated when its role grew from pure logging into the load-bearing
mutator that ships max_tokens injection in the production recipe.)

A single-process Python HTTP reverse proxy that sits between *any*
chat-completions caller and vLLM. It's not a bridge — it doesn't understand
the manyforge envelope or run an agent loop. It just forwards HTTP requests
to vLLM, with two responsibilities:

1. **Logging (always on).** Every chat-completion request and response is
   appended to a JSONL file with the full bodies (parsed when JSON, raw
   excerpt otherwise), mutation diffs, response status and headers, and
   per-call wall-clock latency. The smoke runner and ad-hoc debugging
   workflows both read this file by byte-offset diff per request to scope
   the per-call view. **Logging is the proxy's original purpose and is
   never disabled.**
2. **Mutation (opt-in via env vars).** When configured, the proxy rewrites
   the outbound request body before forwarding to vLLM. The single
   load-bearing mutation in the production recipe is `max_tokens=2048`
   injection — without it, OpenClaw → vLLM traffic carries no bound and
   thinking-on generations run for many minutes per turn. Other knobs
   (thinking_token_budget, tool_choice, user-message suffix, …) are used
   for ad-hoc experiments without rebuilding any stack component.

### What it observes

| Lane | Proxy sees per smoke-prompt |
|------|------------------------------|
| Direct (manyforge_assistant_bridge) | every per-turn chat-completion call (the bridge runs the loop and proxies through this proxy if `BRIDGE_UPSTREAM_BASE_URL=http://127.0.0.1:8000/v1`) |
| OpenClaw | every per-turn chat-completion call from the OpenClaw agent loop. The bridge isn't involved per-turn. |

In iter 20's setup, the proxy logs **183–286 chat-completions per 66-case
run** (avg 3-4 turns per case, depending on agent-loop depth).

### Mutation knobs (env vars)

Active per request when set; pure pass-through when unset.

| Env var | Effect |
|---------|--------|
| `OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS=N` | Inject or rewrite `max_tokens` (or `max_completion_tokens`) to N. **Inject is load-bearing**: OpenClaw → vLLM omits the field, so without injection vLLM defaults to model max context (32K) and runs unbounded. |
| `OPENCLAW_PROXY_THINKING_TOKEN_BUDGET=N` | Inject `chat_template_kwargs.thinking_token_budget=N`. Soft cap on internal CoT length when the chat template honors it. |
| `OPENCLAW_PROXY_FORCE_ENABLE_THINKING=on\|off\|alternating-…` | Inject `chat_template_kwargs.enable_thinking`. Alternating modes use turn parity (counted from `messages[]` assistant-role count). |
| `OPENCLAW_PROXY_FORCE_TOOL_CHOICE=required\|auto\|required-first\|alternating[-on-even]` | Override `tool_choice` per call. Turn-aware modes count assistant messages. |
| `OPENCLAW_PROXY_USER_MESSAGE_SUFFIX="…"` | Append text to the last user message. Iter-16's "read-first" hint was injected this way. |
| `OPENCLAW_PROXY_OVERRIDE_TEMPERATURE`, `_TOP_P`, `_USER_SUFFIX_FIRST_TURN_ONLY` | Misc overrides. |

### Infrastructure fixes (shipped iter 18b)

- `_OVERRIDE_MAX_TOKENS` was rewrite-only before iter 18b — silently no-op
  against OpenClaw's missing-field requests. Now injects with `"injected": true`
  in the audit record.
- Per-request socket timeout is 200 s (was 600 s). Smoke runner's case timeout
  is 244 s; the proxy's 200 s ensures it fails first and releases the upstream
  KV slot before the runner gives up. Prevents zombie-thread accumulation.

### Log file

JSONL at `/tmp/<custom>_proxy.jsonl` (set with `OPENCLAW_PROXY_LOG_PATH`).
Each entry has `request.{method,path,headers,body,mutation}` and
`response.{status,headers,body,duration_ms}`. The mutation block lists what
the proxy changed and the before/after values, so each run is auditable.

---

## Tool surface — where each piece lives

### 1. Tool schema (what the model sees as a JSON Schema)

`/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_composer/backend/assistant_tool_schemas.py`

Defines `_TREE_DRAFT_INSERT_NODE_SCHEMA`,
`_TREE_DRAFT_CHANGE_NODE_KIND_SCHEMA`, `_SCENE_DRAFT_*_SCHEMA`, etc.
Each is a Python dict with a `description` field that the model reads
as the OpenAI tool description.

The **registry** (line ~1148) maps tool ids to schemas:

```python
TOOL_SCHEMAS = {
    "tree_draft_insert_node":      _TREE_DRAFT_INSERT_NODE_SCHEMA,
    "tree_draft_change_node_kind": _TREE_DRAFT_CHANGE_NODE_KIND_SCHEMA,
    ...
}
```

`enrich_assistant_tool_descriptor()` (line ~1172) merges these schemas onto
the per-mode tool list before serving them to the bridge.

### 2. Tool handlers (what runs when the model calls a tool)

`/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_composer/backend/routes_assistant.py`

Each tool has an `_apply_<tool_id>` function, e.g. `_apply_tree_draft_insert_node`
(line 2333). The dispatcher at line ~1299 routes tool ids to handlers.

Helpers:

- `_find_tree_node_ref(tree, name)` — returns `(node, parent, sibling_index)`
  for any name in the tree. Will be reused for `afterName`/`beforeName`.
- `_resolve_node_name_alias("@root", tree)` — root-name shortcut.
- `_collect_tree_node_names(tree)` — for 4xx response hints.
- `_insert_index(arguments, child_count)` — bounds-checks a passed `index`.

### 3. Tool recovery (4xx response shaping)

`/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_composer/backend/assistant_recovery.py`

When a tool call fails with a structural error, recovery hints get attached
to the error response so the model can self-correct on the next turn.
Existing recovery classes: `arity_insufficient`, `unknown_parent_name`,
`unknown_node_kind`, …. **Adding the "X-is-not-a-parent → use afterName" hint
plugs in here.**

### 4. Node catalog (the runtime BT primitives)

`/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_behavior/resources/node_catalog.yaml`

YAML list of behavior-tree node kinds. Each entry has `id`, `name`, `category`,
`description`, `parameters`, etc. The 6 runtime collision-object kinds we
need to retag:

| Line | id |
|------|-----|
| 447 | `add_collision_object` |
| 517 | `upsert_collision_object` |
| 589 | `remove_collision_object` |
| 624 | `update_collision_object_pose` |
| 669 | `attach_object_to_link` |
| 723 | `detach_object_from_link` |

The `description` field of each entry is what surfaces in the prompt's
`nodeCatalog` array (id + kind + description only — `category`,
`tags`, `parameters` etc. are stripped before reaching the model).

### 5. Mode allowlist (which tools/node-kinds are exposed per mode)

`/home/tndlux/workspaces/dev_ws/src/manyforge/examples/assistant_modes_scene_authoring.deployment.yaml`

Each `assistant_modes.<mode>.catalog.{tools,nodes}` is the soft allowlist
the bridge sends to the model. **Tool renames here flow through to all
downstream lists.**

---

## Smoke corpus mechanics

Source:
`/home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor/manyforge/scripts/debug/smoke_corpus_runner.py`
+ `smoke_corpus.yaml`.

### Per-case loop

For each case in `smoke_corpus.yaml`:

1. Optionally reset Composer state (`/api/program/load forceDiscardOverrides:true`).
2. POST `{message: <user_prompt>}` to `<composer>/api/assistant/chat` with
   a per-case session key derived from
   `conversationId + catalogHash + programRevision`.
3. Parse the response: extract `toolCalls[].tool` and any answer text.
4. Run the case's `expected.tools_called` / `expected.forbidden_tools`
   / `expected.answer_must_contain` / `expected.state_after` rubrics.
5. Record one of:
   - `pass` — all expected tools fired with right args; no forbidden tools;
     state_after matches; required answer-text substrings present.
   - `recovered-pass` (`🛟`) — passed only after a recovery turn was injected.
   - `soft-pass` (`🟡`) — answer-text rubric matched but tool-call rubric
     didn't (used for clarification cases that emitted prose without firing
     tools).
   - `fail` (`❌`) — anything else.
6. Emit one log line: `<emoji> <case_id>  <wall_s>s  status=<…>  fail: [...]`.

### `--no-chain-session` flag

Default behavior is **chain-session ON**: PnP_01 → PnP_02 → … all share the
same `conversationId`, so OpenClaw's session memory persists across chain
steps. A failure in PnP_06 leaves broken state in the session and the
remaining PnP_07–PnP_20 cascade-fail.

`--no-chain-session` overrides this: every chain step gets a fresh
conversationId. Failures stay independent. This is the iter-16 fix that
unlocked iter 20's win.

### Effective rate vs first-try rate

```
first-try = pass / total
effective = (pass + recovered-pass + soft-pass) / total
```

Iter 20: 45/66 first-try (68.2 %), 49/66 effective (74.2 %).

---

## Iter-20 production recipe (best config, 49/66 = 74.2 %)

End-to-end commands to reproduce. Stop any conflicting containers/processes
first (`docker rm -f manyforge-e2e-vllm`, `pkill -f vllm-proxy`).

```bash
# 1. vLLM with thinking-on default
cd /home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor
THOR_DETACH=1 \
THOR_CONTAINER_NAME=manyforge-e2e-vllm \
THOR_VLLM_PORT=8050 \
  ./serving/start-model.sh cosmos-reason2-8b
# (waits ~2-3 min for first-time model load; subsequent restarts ~30 s)

# 2. Mutator proxy with cap and thinking budget injected
cd /home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor/manyforge/scripts/debug
OPENCLAW_PROXY_LISTEN_PORT=8000 \
OPENCLAW_PROXY_BIND=0.0.0.0 \
OPENCLAW_PROXY_UPSTREAM=http://127.0.0.1:8050 \
OPENCLAW_PROXY_LOG_PATH=/tmp/iter20_proxy.jsonl \
OPENCLAW_PROXY_THINKING_TOKEN_BUDGET=512 \
OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS=2048 \
  nohup python3 vllm-proxy.py >> /tmp/iter20_proxy_stdout.log 2>&1 &

# 3. Smoke runner (no chain-session)
cd /home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor/manyforge/scripts/debug
nohup python3 -u smoke_corpus_runner.py --no-chain-session \
  > /tmp/instrumented_run/iter20.log 2>&1 &
```

Composer (`manyforge-e2e-composer` container, host:9000) and
`openclaw_assistant_bridge` (host:8200) are assumed to be already running
from earlier session bring-up. The runner's `<composer>/api/assistant/chat`
calls into Composer; Composer dispatches to whichever bridge is configured
in the deployment YAML.

Wall-clock: ~41 minutes for the full 66-case corpus.

---

## Refinement plan — what landed (2026-05-10)

The post-iter-20 plan, ranked by leverage and shipped together as the
iter-27 candidate:

### Action A — runtime collision-object catalog descriptions (Pattern C, ~3 cases)

[`manyforge_behavior/resources/node_catalog.yaml`](/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_behavior/resources/node_catalog.yaml).
6 entries rewritten (`add_collision_object`, `upsert_collision_object`,
`remove_collision_object`, `update_collision_object_pose`,
`attach_object_to_link`, `detach_object_from_link`). Each description
now leads with a "Behavior-tree leaf — runtime collision-object
operation" framing and explicitly contrasts against the corresponding
`scene_draft_*` tool, so the model's lexical pattern match against the
word "scene" no longer pulls it toward the static-scene tool family.
YAML-only change; Composer rebuilds the catalog on next request.

### Action B — `afterName` / `beforeName` / `position` on `tree_draft_insert_node` (Pattern B, ~5 cases)

Two files, ~110 LoC of new handler logic:

1. [`assistant_tool_schemas.py`](/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_composer/backend/assistant_tool_schemas.py):
   schema now declares `afterName`, `beforeName`, and `position: "first" |
   "last"` alongside the existing `parentName` + `index`. `required` is
   now just `["node"]`; the description spells out four mutually
   exclusive positional forms and which natural-language patterns map
   to each. `parentName`'s and `index`'s descriptions explicitly mark
   their mutual exclusivity with the new forms.
2. [`routes_assistant.py`](/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_composer/backend/routes_assistant.py):
   new `_resolve_insert_position` helper translates the chosen form
   into a (parent_name, optional_index) pair before the existing
   parent-resolution path runs. `afterName`/`beforeName` use
   `_find_tree_node_ref` to read the sibling's parent + index out of
   the live tree; `position: "first"` resolves to index 0 and
   `"last"` to the parent's child count. The 404 response now carries
   an explicit `hint` when the model passed a known *leaf* node as
   `parentName` (suggesting `afterName`/`beforeName` instead).
   `_apply_tree_draft_insert_node` was lightly refactored around
   the new helper.

### Action C — rename `tree_draft_swap_node` → `tree_draft_change_node_kind` + `move_node` cross-ref (Pattern E, ~1 case)

Clean rename, no alias. The new name ends the lexical trap where
"swap the order" routed to the wrong tool because both share the
word "swap". Files updated:

1. [`assistant_tool_schemas.py`](/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_composer/backend/assistant_tool_schemas.py):
   `_TREE_DRAFT_SWAP_NODE_SCHEMA` → `_TREE_DRAFT_CHANGE_NODE_KIND_SCHEMA`,
   registry key updated. Description gains an explicit anti-example
   sentence that names `tree_draft_move_node` as the right tool for
   sibling reordering.
2. `_TREE_DRAFT_MOVE_NODE_SCHEMA` description gains the matching
   forward reference: this is the right tool when the user says
   "swap the order", "swap A with B", or "reorder".
3. [`routes_assistant.py`](/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_composer/backend/routes_assistant.py):
   `_apply_tree_draft_swap_node` → `_apply_tree_draft_change_node_kind`,
   dispatcher updated, internal error messages reflect the new name.
4. [`assistant_recovery.py`](/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_composer/backend/assistant_recovery.py):
   recovery hint text updated.
5. [`assistant_modes_scene_authoring.deployment.yaml`](/home/tndlux/workspaces/dev_ws/src/manyforge/examples/assistant_modes_scene_authoring.deployment.yaml):
   mode-tools allowlist + the tool registry block, both renamed.
6. [`manyforge_assistant_bridge/bridge.py`](/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_assistant_bridge/bridge.py):
   intent-inference heuristic updated. "swap the order" / "swap order"
   / "reorder" now route to `tree_draft_move_node`; bare "swap"
   continues to route to `tree_draft_change_node_kind`.
7. [`manyforge_behavior/manyforge_behavior/catalog.py`](/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_behavior/manyforge_behavior/catalog.py):
   `swap_class` field comment updated.
8. [`test_deployment.py`](/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_behavior/tests/test_deployment.py):
   tool-list assertion updated; comment carries the historical pointer.
9. [`smoke_corpus.yaml`](../scripts/debug/smoke_corpus.yaml): three
   forbidden-tools list entries renamed.

The OpenClaw bridge ([`adapter.py`](../openclaw_assistant_bridge/adapter.py))
doesn't name-match on tool ids, so no edit was needed there.

---

## After applying the plan

1. Re-run iter 20's recipe verbatim. Same model, same proxy config, same
   `--no-chain-session`. The only differences are the source-code edits.
2. Compare against iter 20's failure set (15 consistent fails listed in
   SMOKE-CORPUS.md). Predicted lift: 9 cases, putting the corpus at ~58/66
   (~88 %).
3. Document the result as iter 27 in SMOKE-CORPUS.md.
4. If the lift lands as predicted, the remaining fails are split between
   genuinely-hard multi-arg cases (Pattern D, 2) and corpus-rubric
   choices (Pattern A, 2) — neither addressable from infra.

---

## Glossary

- **Bridge**: the HTTP service that consumes Composer's
  `manyforge.assistant.provider_request.v0` envelope and runs (or
  delegates) the LLM agent loop.
- **Lane**: one of the two bridge implementations.
- **Mutator**: the `vllm-proxy` running with mutation env vars set.
- **Direct lane**: bridge runs the agent loop locally.
- **OpenClaw lane**: bridge delegates to OpenClaw which runs the agent loop.
- **Chain-session**: Composer's per-conversation session key; when the smoke
  runner reuses it across chain steps, OpenClaw retains the prior turn
  history.
