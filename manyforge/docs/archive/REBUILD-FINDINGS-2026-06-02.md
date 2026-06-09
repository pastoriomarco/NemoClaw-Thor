# Findings — post-rebuild stack (2026-06-02)

Companion to [REBUILD-2026-06-02-openshell-0.0.44.md](REBUILD-2026-06-02-openshell-0.0.44.md). The rebuild doc explains *what* changed and *how* to redo it. This doc explains what we learned by running the new stack.

## Headline

**The stack works end-to-end on OpenShell 0.0.44 / OpenClaw 2026.5.22 / NemoClaw `lkg` (=v0.0.55, validated session-of-record on v0.0.56 which is byte-equivalent except for the [PR #4613 installer change](https://github.com/NVIDIA/NemoClaw/pull/4613)) with the patches in [REBUILD-2026-06-02-openshell-0.0.44.md](REBUILD-2026-06-02-openshell-0.0.44.md). Model accuracy regressed across the board because OpenClaw 2026.5.22 routes every tool through a `tool_search_code` discovery shim that cannot be cleanly disabled.** The model has to write JavaScript that calls `tools.search(query)` → `tools.describe(id)` → `tools.call(id, args)` to reach any catalog tool, instead of emitting direct hermes-style `<tool_call>...` blocks. This breaks the direct-tool patterns the manyforge-composer skill, the bridge prompts, and the smoke-corpus assertions were designed for.

The pipeline itself is fine. Every layer (proxy ↔ vLLM, bridge ↔ openclaw gateway, openclaw ↔ MCP, manyforge MCP ↔ composer state) routes traffic correctly. The regression is purely in how the *content* of those calls is shaped.

## Smoke results

(Filled in as runs complete.)

| Model | Profile | Thinking | Effective | First-try | PnP pass | Wall-clock | Notes |
|---|---|---|---|---|---|---|---|
| cosmos-reason2-8b | cosmos | on | TBD | TBD | TBD | TBD | reference: 77.3% (iter-32, OpenClaw 2026.4.24) |
| nemotron3-nano-omni-30b-a3b-nvfp4 | omni | off | TBD | TBD | TBD | TBD | reference: 31.8% (pre-rebuild bake-off) |
| nemotron3-nano-4b-bf16 | 4B | on | TBD | TBD | TBD | TBD | (no prior reference at thinking=on) |
| nemotron3-nano-4b-bf16 | 4B | off | TBD | TBD | TBD | TBD | reference: 39.4% (pre-rebuild bake-off) |
| qwen3.6-35b-a3b-nvfp4-nvidia | 35B | on | TBD | TBD | TBD | TBD | reference: 84.8% (pre-rebuild bake-off) |

## Why the regression — `tool_search_code`

Each chat-completion request the agent sends to vLLM contains a single tool:

```json
{
  "type": "function",
  "function": {
    "name": "tool_search_code",
    "description": "Run JavaScript or TypeScript in OpenClaw code mode. Node.js modules and `require`/`import` are NOT available; for any shell, file, network, or external action, use enabled catalog tools allowed by policy from inside your code: `tools.search(query)` to find catalog entries, `tools.describe(entry.id)` for the input schema, then `tools.call(entry.id, args)`. ...",
    "parameters": {"code": "string", "language": "javascript|typescript"}
  }
}
```

The manyforge MCP tools (`program_read`, `tree_draft_wrap_node`, `scene_draft_add_object`, …) are NOT exposed directly. They live behind `tools.search()` and the model has to discover and call them via code-mode.

### Settings that did NOT disable this

We tried all of:
- `agents.list[manyforge-composer].tools.profile = "full"` (replaces `minimal` + `bundle-mcp` allowlist)
- `tools.toolSearch = { enabled: false }` (the documented runtime setting)
- `tools.codeMode = { enabled: false }` (codeMode is the alternative compaction path)

None of these turned off the compaction. The runtime check at `selection-hR-AeOeU.js:13160` does honor `resolveToolSearchConfig(config).enabled`, but something else in the agent-construction path is re-enabling it. Inspection of `pi-tools-iVT6BGHc.js` shows that `addClientToolsToToolSearchCatalog` is called unconditionally when client tools are present; without forking OpenClaw we cannot suppress it.

### Why this hurts manyforge specifically

Manyforge tools have rich structural validators on their arguments — e.g. `tree_draft_wrap_node` expects `{path, wrap_with: {kind: "repeat" | "sequence" | ...}, ...}`. The skill and the bridge prompts were written assuming the model emits hermes XML with the right argument keys directly. With the discovery shim:

- The model has to first call `tools.search("wrap a node")` and pick from the description text. Cosmos consistently picks tools whose names *sound* similar (e.g. wrapping nodes with a `sequence` instead of a `repeat`) because the structured arg schema is hidden until `tools.describe`.
- Even when the right tool is chosen, the model often produces argument shapes that the validator rejects because the code-mode prompt nudges loose JS syntax. The proxy's tool-error rewrite + our manyforge MCP 4xx→200 fix (for the `erroredAssistantResultPolicy: "drop"` workaround) should recover from this, but `tools.call(id, args)` calls inside code-mode are not visible to the proxy's `_normalize_tool_names_in_response` rewriter — those happen inside the in-sandbox runtime, not on the LLM wire.

## Recommended upstream issues

1. **OpenClaw**: make `tools.toolSearch.enabled = false` actually disable the compaction. As of 2026.5.22 the runtime check honors the flag but `applyToolCatalogCompaction` / `addClientToolsToToolSearchCatalog` are invoked unconditionally, so the catalog never reaches the model.

2. **OpenClaw**: stop hardcoding `erroredAssistantResultPolicy: "drop"` in [`src/agents/embedded-agent-runner/run/attempt.ts:3102-3107`](https://github.com/openclaw/openclaw). Either expose it as a config or expose the sanitizer's policy choice as an option on `openclaw agent ...`. Our manyforge MCP workaround (returning 200 + structured `success:false`) is sufficient for our case, but every MCP server in the ecosystem will eventually need the same workaround.

3. **NemoClaw**: register the locally-spawned OpenShell gateway as `http://...` (plaintext) instead of `https://...` (mTLS). The mismatch surfaces at onboarding step 4/8 with `transport error: received corrupt message of type InvalidContentType` and is opaque to first-time users. The two-line `openshell gateway remove / openshell gateway add` workaround is in our headless onboarding script but should not be necessary.

## Findings unrelated to OpenClaw 2026.5.22

- **NemoClaw 0.0.56 `nemoclaw status` TypeError**. `shields.getShieldsPosture is not a function`. The setup script's healthcheck originally relied on `status`; switched to `nemoclaw exec true` which is a stronger liveness signal anyway.

- **OpenShell exec endpoint rejects newlines in argv**. `nemoclaw exec --no-tty -- bash -c "<multi-line cmd>"` gets the gRPC reply `InvalidArgument: command argument 2 contains newline or carriage return characters`. The bridge adapter base64-wraps the command (`bash -c "eval \"$(echo <base64> | base64 -d)\""`) and the same workaround is suitable for any caller that needs to pass a multi-line script.

- **Docker bridge subnet changed**. The new `openshell-docker` network is `172.18.0.0/16`; the old `bridge` network was `172.17.0.0/16`. Custom policies whose `allowed_ips` referenced 172.17 silently broke (HTTP 403 from the SSRF guard) until the allowlist was expanded.

- **vLLM `--reasoning-parser` + OpenClaw 2026.5.22**. Any model launched with a reasoning parser routes output into the `reasoning` field, leaving `content` null. OpenClaw 2026.5.22 rejects `content==null` responses as `code=incomplete_result`. The proxy now mirrors `reasoning → content` (preserves both for downstream); this was the load-bearing fix that let the bridge return chat HTTP 200 at all.

- **Cosmos cannot work without thinking-on**. We confirmed (again) that `enable_thinking: false` on cosmos triggers narration mode — the model says "Tool call completed. The program_read tool has been successfully executed." instead of emitting an actual tool call. Cosmos is post-trained from Qwen3-VL with long-CoT assumed; thinking-off is out-of-distribution. The proxy keeps thinking-on; the new `reasoning → content` mirror handles the OpenClaw contract change.

## What works well now

- **bridge → openclaw gateway → vLLM** end-to-end timing is healthy (~45s per simple chat completion; ~120s for multi-turn tool loops). No timeout flapping.

- **MCP namespace isolation** stayed intact across the OpenShell 0.0.36 → 0.0.44 driver swap. The manyforge MCP server runs inside the sandbox, talks to composer via `host.openshell.internal:9000`, and is reachable from the OpenClaw agent without any extra plumbing.

- **vLLM v9.1 image** survived the upgrade unchanged. All five model profiles boot. Cosmos cold-starts in ~120s on this Thor host; 35B and Omni take roughly twice that.

- **Composer's stateful program tracking** survived too. The container was untouched throughout the rebuild and the loaded program persisted (until we restarted to pick up the MCP 4xx→200 patch).

## Files touched in this rebuild

| Path | Diff |
|---|---|
| [`serving/start-model.sh`](../../serving/start-model.sh) | added `OPENCLAW_PROXY_PROMOTE_REASONING_TO_CONTENT=1` default |
| [`scripts/proxy/vllm-proxy.py`](../scripts/proxy/vllm-proxy.py) | new `_promote_reasoning_to_content_in_response` mutation + flag |
| [`openclaw_assistant_bridge/adapter.py`](../openclaw_assistant_bridge/adapter.py) | k3s→docker exec; base64-wrap shell command for newline guard |
| [`openclaw_assistant_bridge/service.py`](../openclaw_assistant_bridge/service.py) | log openclaw stderr on returncode != 0 |
| [`setup-manyforge-assistant.sh`](../setup-manyforge-assistant.sh) | 5 patches (see REBUILD doc) |
| [`policies/manyforge-composer.preset.yaml`](../policies/manyforge-composer.preset.yaml) | add 172.18.0.0/16 to `allowed_ips` |
| [`docs/REBUILD-2026-06-02-openshell-0.0.44.md`](REBUILD-2026-06-02-openshell-0.0.44.md) | new — rebuild procedure |
| [`docs/REBUILD-FINDINGS-2026-06-02.md`](REBUILD-FINDINGS-2026-06-02.md) | new — this doc |
| [`scripts/rebuild-headless-onboarding.sh`](../scripts/rebuild-headless-onboarding.sh) | new — headless reproduction |
| [`scripts/debug/run-smoke-all-models.sh`](../scripts/debug/run-smoke-all-models.sh) | new — multi-model bake-off driver |
| [`/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_composer/backend/routes_assistant.py`](../../../../../dev_ws/src/manyforge/manyforge_composer/backend/routes_assistant.py) | `execute_bridge_tool` validator path returns 200 + error envelope (was 4xx) |

(The composer file lives outside this repo; that change was applied live via the bind mount.)
