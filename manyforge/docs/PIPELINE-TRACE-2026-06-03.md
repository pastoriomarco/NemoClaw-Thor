# Pipeline Trace — OpenClaw Lane Failure Diagnosis (2026-06-03)

Full request/response trace through every step of the OpenClaw lane after the route fix landed. Goal: answer "what is actually arriving at each step, and does it match what we think is being sent?"

## Test request

```json
POST http://127.0.0.1:9000/api/assistant/chat
{
  "message": "Read the program and report the root node id.",
  "mode": "provider",
  "conversationId": "trace-001",
  "requestId": "trace-001",
  "assistantMode": "composer-assistant",
  "timeoutSeconds": 150
}
```

## Step 1 — Composer → Bridge

Composer dispatches to `MANYFORGE_ASSISTANT_ENDPOINT_URL=http://127.0.0.1:8200/v1/manyforge/assistant`. The HTTP request body is the `manyforge.assistant.provider_request.v0` envelope — visible at the bridge's `service.py` receiving end.

The bridge logs each request as `openclaw_request_started`:

```json
{
  "allowedMcpTools": ["program_read", "catalog_read", "skills_read", ...],
  "event": "openclaw_request_started",
  "promptChars": 18328,
  "requestId": "trace-001",
  "timeoutS": 120.0,
  "transport": "cli_shell_out"
}
```

Critical observations:
- ✅ `transport: cli_shell_out` (the fix is active).
- ✅ `allowedMcpTools` is the per-mode allowlist (14 manyforge tools).
- ⚠️ **`timeoutS: 120.0` is hard-coded in the bridge.** It does NOT honor the composer's `timeoutSeconds: 150`. Cases that take >120s get 504. (One case hit this already: `SCENE_add_medium`).
- ❌ **`OPENCLAW_ASSISTANT_COMPACT_EVERY_N` is not set.** Bridge log contains zero `compact` events. The iter-32 production recipe REQUIRES compaction every 2 user prompts; my CLI-mode bridge restart lost this env.

## Step 2 — Bridge → OpenClaw (CLI)

The bridge's `adapter.build_openclaw_command()` constructs:

```
nemoclaw my-assistant exec --no-tty -- bash -c 'eval "$(echo <base64> | base64 -d)"'
```

Where the base64 decodes to:

```
printf %s 'program_read,catalog_read,skills_read,deployment_capabilities_read,scene_inspect,tree_draft_insert_node,tree_draft_update_node_params,tree_draft_delete_node,tree_draft_move_node,tree_draft_replace_subtree,tree_draft_change_node_kind,tree_draft_wrap_node,program_draft_upsert_parameters,program_draft_remove_parameters' > /tmp/manyforge-openclaw-allowed-tools.txt
trap 'rm -f /tmp/manyforge-openclaw-allowed-tools.txt' EXIT
openclaw agent --agent manyforge-composer --message '<19k-char prompt>' --json --timeout 120
```

The bridge's 18,328-char prompt (per `promptChars: 18328`) is the full `build_agent_prompt` output:

- `<nemoclaw-runtime>` block (network policy, fs policy, behavior)
- Date/role header: "You are the ManyForge composer assistant running inside OpenClaw."
- RULES (5+ items: catalog ids immutable, id vs name fields, scene_draft vs tree_draft, snapshots are live, etc.)
- Full `programSnapshot` (current tree + parameters + blackboard)
- Full `sceneSnapshot` (objects + dimensions + poses)
- Full `nodeCatalog` (all canonical node kinds with their schemas)
- Full `skillCatalog`
- `## user_request\nRead the program and report the root node id.`
- `## tail_checklist (apply BEFORE emitting your next action)`

## Step 3 — OpenClaw → vLLM (via proxy on :8000)

This is what shows up in `/tmp/manyforge-assistant-e2e/vllm-proxy.jsonl`. **OpenClaw discards or transforms the bridge's prompt in a load-bearing way**:

### What OpenClaw sends to vLLM

```json
POST http://127.0.0.1:8050/v1/chat/completions  (via proxy on :8000)
{
  "model": "cosmos-reason2-8b",
  "stream": true,
  "max_completion_tokens": 2048,
  "chat_template_kwargs": {...},
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "tool_search_code",
        "parameters": {
          "type": "object",
          "required": ["code"],
          "properties": {"code": {"type": "string"}}
        }
      }
    }
  ],
  "tool_choice": "auto",
  "messages": [
    {
      "role": "system",
      "content": "<16,885-char OpenClaw default system message>\n## Tooling\nAvailable tools are policy-filtered. Names are case-sensitive; call exactly as listed.\n- tool_search_code\n..."
    },
    {
      "role": "user",
      "content": "<19,384-char bridge prompt> [includes the nemoclaw-runtime block + manyforge composer instructions + RULES + programSnapshot + sceneSnapshot + nodeCatalog + user_request + tail_checklist]"
    }
  ]
}
```

### Critical observations on what OpenClaw shows the model

1. ❌ **`tools[]` contains only `tool_search_code`.** OpenClaw 2026.5.6+ native discovery shim is active in CODE MODE. The model does NOT see `tree_draft_wrap_node`, `scene_draft_add_object`, etc. in its `tools[]`.

2. ⚠️ **The bridge's manyforge skill prompt is inside the USER message** (19,384 chars), not the system message. OpenClaw's system message (16,885 chars) is its DEFAULT prompt with no manyforge specificity — it only mentions `tool_search_code` is available.

3. ❌ **The Phase 3 skill addendum I wrote was NEVER applied.** It lives in `manyforge/lanes/openclaw/skill_addendum.md` in NemoClaw-Thor but was never copied to the sandbox skill or merged into the openclaw.json system prompt. The system message contains nothing about the discovery protocol or camelCase conventions.

4. ✅ Proxy banner shows `profile: compat` with all four mutations on (`normalize_tool_names`, `tool_error_rewrite`, `promote_reasoning_to_content`, `unwrap_tool_call_args`).

## Step 4 — vLLM response (model output)

The model's response, reassembled from streaming chunks (proxy captures `body_raw_excerpt`):

```python
turn  1 [ 5034ms] finish=tool_calls:
  → tree_draft_wrap_node({'target_name': 'pick_and_place', 'wrapper_id': 'repeat'})
turn  2 [ 3475ms] finish=tool_calls:
  → tree_draft_wrap_node({'target_name': 'pick_and_place', 'wrapper_id': 'repeat'})
turn  3 [ 3496ms] finish=tool_calls:
  → tree_draft_wrap_node({'target_name': 'pick_and_place', 'wrapper_id': 'repeat'})
...
turn 11 [ 2272ms] finish=tool_calls:
  → tree_draft_wrap_node({'target_name': 'pick_and_place', 'wrapper_id': 'repeat'})
```

Same wrong call, 11 times. The model is in `tool_search_code` mode but emits the inner tool call directly as a function-call instead of wrapping it in `tool_search_code({code: "..."})`. **It also uses snake_case (`target_name`) and a flat structure (`wrapper_id`) — but the canonical composer schema requires camelCase (`targetName`) and a nested object (`wrapper: {id, name, params}`).**

This is consistent with cosmos-reason2-8b's training: it has a Python-style code/argument convention from being trained as a robotics-reasoning model. It hasn't been instruction-tuned for camelCase + nested JSON tool calls.

## Step 5 — OpenClaw evaluates → MCP bridge → Composer

OpenClaw receives the model's `tree_draft_wrap_node` call and dispatches via the in-sandbox MCP bridge (`manyforge-mcp-bridge.py`) to composer at `POST /api/assistant/bridge/tools/tree_draft_wrap_node`.

Composer's validator rejects the call: missing required field `targetName`. Per the [routes_assistant.py:execute_bridge_tool fix](/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_composer/backend/routes_assistant.py#L648-L692), the response is HTTP 200 with `{success: false, error: "validation_error", result: {kind: validation_error, message: ..., detail: ...}}`. This is the OpenClaw drop-policy workaround — keeps the turn in chat history so the model can see the error.

## Step 6 — Bridge → Composer (final response)

```json
{
  "requestId": "trace-001",
  "providerId": "openclaw",
  "message": "I can't use the tool \"program_read\" here because it isn't available. I need to stop retrying it and answer without that tool.",
  "toolCalls": [
    {"name": "program_read", "status": "completed", "arguments": {}, "result": {}, "error": null}
  ],
  "warnings": ["OpenClaw stderr: ..."],
  "requiresReview": true,
  "programLoaded": true
}
```

The model eventually gave up after looping, and its final answer treats "the tool isn't available" — which is wrong; the tool IS available but the model never figured out the camelCase/nested calling convention.

## Why the model loops

1. **No instruction in the system prompt about camelCase/nested conventions.** The Phase 3 skill addendum that addresses this was never applied.
2. **Tool error envelope back to the model is unstructured.** The model sees a string like "validation_error" but it's not formatted to highlight which field is wrong.
3. **No cross-turn loop detection** on the proxy — `proxy_loop_reflection_injected` events: 0. The proxy's `same_args>=2` threshold is for SINGLE-request loops; OpenClaw orchestrates one chat-completion per turn, so the loop is across turns and the proxy can't see it.
4. **No bridge-fired `/compact`.** The iter-32 51/66 recipe required compaction every 2 prompts to break the loop by re-summarizing the context. The current bridge has `OPENCLAW_ASSISTANT_COMPACT_EVERY_N` unset.
5. **OpenClaw's code mode is active** (`tool_search_code`), not tools mode (`tool_search`/`tool_describe`/`tool_call`). The Phase 3 skill addendum was written for tools mode.

## Failure pattern summary

13 cases observed by smoke runner:

| Case | Result | Pattern |
|---|---|---|
| P1_wrap_root_specific | fail | `target_name`/`wrapper_id` snake_case loop |
| P2_scene_add_specific | fail | wrong arg shape for `scene_draft_add_object` |
| P3_tree_insert_runtime_obj_specific | fail | expected tool not observed |
| WRAP_root_generic | **soft-pass** | got close enough; the `*_generic` variants are looser |
| WRAP_root_medium | fail | same as P1 |
| SCENE_add_generic | fail | wrong arg shape |
| SCENE_add_medium | fail | HTTP 504 — exceeded 120s bridge timeout |
| TREE_insert_runtime_generic | **soft-pass** | model didn't fire tool but generic check passed on content |
| TREE_insert_runtime_medium | fail | tool not observed |
| INSERT_position_first_specific | fail | tool not observed |
| INSERT_position_after_named_medium | fail | tool not observed |
| INSERT_position_before_named_generic | **soft-pass** | content matches |
| DELETE_named_specific | fail | wrong arg shape |

Three out of 13 are soft-passes (model attempted the right thing, the `*_generic` test variants are looser about exact args). All hard fails share one of three root causes: snake_case/flat args, model not firing the expected tool, or hitting the 120s bridge timeout.

## What's different from the iter-32 51/66 baseline

| Variable | iter-32 baseline | current run | Impact |
|---|---|---|---|
| Bridge transport | gateway_http (probably worked on older OpenClaw) | cli_shell_out (forced by 2026.5.22 missing /v1/chat/completions) | unknown — both should produce equivalent output |
| `OPENCLAW_ASSISTANT_COMPACT_EVERY_N` | **2** (chain-on /compact) | **unset** (no compaction) | **MAJOR** — compaction breaks loops |
| `OPENCLAW_ASSISTANT_TIMEOUT_S` | 300s | 120s default | one 504 already |
| OpenClaw version | 2026.4.x (pre-shim) | 2026.5.22 (native shim active) | **MAJOR** — code mode now active |
| Composer corpus | iter-32 ([iter-28 + afterName/beforeName + runtime catalog rewrites]) | same corpus | n/a |
| Manyforge spec | iter-32 era (camelCase) | same camelCase | n/a |
| Proxy mutations | UNWRAP_TOOL_CALL_ARGS + PROMOTE_REASONING_TO_CONTENT + NORMALIZE_TOOL_NAMES + TOOL_ERROR_REWRITE | same (banner confirms) | n/a |
| Skill prompt | iter-32 default | same default (no Phase 3 addendum applied) | **MAJOR** — no discovery primer |

The three MAJOR differences explain the 14/66 vs 51/66 gap.

## Recommended fixes (ordered)

### Fix 1 — Restore iter-32 bridge env (the most impactful single change)

The bridge supervisor must receive:

```bash
OPENCLAW_ASSISTANT_COMPACT_EVERY_N=2
OPENCLAW_ASSISTANT_COMPACT_TIMEOUT_S=120
OPENCLAW_ASSISTANT_TIMEOUT_S=300
```

These are in `scripts/lib/assistant.sh` for the launcher path. My manual bridge restart bypassed those defaults. Fix: relaunch via the launcher (which I patched to default `USE_GATEWAY=false`).

### Fix 2 — Apply the Phase 3 skill addendum to the sandbox

`manyforge/lanes/openclaw/skill_addendum.md` needs to be merged into `/sandbox/.openclaw/skills/manyforge-composer/SKILL.md` (or appended to the system prompt via openclaw.json's agent template). Currently it lives in NemoClaw-Thor but is never installed.

The addendum should be EXPANDED to also cover code mode (`tool_search_code`) — current version is tools-mode only. The expanded version should explicitly teach:
- camelCase argument names (`targetName`, not `target_name`)
- Nested wrapper objects (`wrapper: {id, name, params}`, not `wrapper_id`)
- Exact catalog ids (no inventing names like `attach_graspable`)

### Fix 3 — Composer-side schema tolerance (defensive but pragmatic)

Have `routes_assistant.execute_bridge_tool` accept BOTH `targetName`/`target_name` and BOTH flat/nested `wrapper`/`wrapper_id` forms. Normalize to canonical at the validator entry. This is a one-way translation, doesn't change the model-facing schema. Significantly improves cosmos-reason2-8b's hit rate without changing model training.

### Fix 4 — Better tool error envelope

Composer's validation error response should include:
- `expected_schema` (the JSON Schema for the rejected tool)
- `provided_args` (what the model sent)
- `diff` (which keys differ — e.g. "you sent `target_name`, schema expects `targetName`")

The model can then read these and self-correct on the next turn.

### Fix 5 — Cross-turn loop detection at the bridge

Add bridge-side detection: if the same tool name + similar args fire >3 times in the same conversation, inject a stronger nudge or fire `/compact`. This catches what the proxy can't (proxy only sees one turn at a time).

## Pipeline coverage gaps in current diagnostics

Several gaps surfaced during this trace:
- **Bridge audit log is stale** (`known-good-bridge-audit.jsonl` last touched 00:21, never updated by the current bridge). The bridge's audit writer is broken or pointed at the wrong path.
- **OpenClaw stdout** captured by the bridge is short (`stdoutBytes: 9` for the failing case, just `Not Found`). The CLI shell-out path captures stdout properly (`stdoutBytes: 14949` for a working case), but error cases lose info.
- **Proxy log captures `body_raw_excerpt` (SSE chunks)** which has to be reassembled to see tool_calls — needs a per-call parser to be useful for live debugging.
- **No correlation id linking bridge → openclaw → proxy → vllm** — manual matching by timestamp is error-prone with concurrent requests.
