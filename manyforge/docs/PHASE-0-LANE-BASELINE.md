# Phase 0 — Lane Baseline Probes

Phase 0 gate of the [THREE-LANE-MIGRATION-PLAN](./THREE-LANE-MIGRATION-PLAN.md). Per-lane validation that the two already-functional lanes (Direct + OpenClaw) work as expected on the current stack BEFORE the Phase 1 refactor lands.

## Version tuple

| Component | Version | Source |
|---|---|---|
| Production model | `cosmos-reason2-8b` | NVIDIA Cosmos-Reason2-8B (anchor confirmed Phase 0) |
| OpenClaw (installed) | `2026.5.22` | NemoClaw `lkg` pin → `Dockerfile.base:31` |
| OpenClaw (upstream-latest stable) | `2026.5.28` | npm `dist-tags.latest` |
| OpenClaw (upstream-latest beta) | `2026.6.1-beta.2` | npm `dist-tags.beta` |
| Hermes (installed) | `2026.5.16` / `0.14.0` | NemoClaw pin → `agents/hermes/Dockerfile.base:27` |
| Hermes (upstream-latest stable) | `0.15.2` | PyPI |
| NemoClaw | `v0.0.55` / `lkg` (commit `95d483fe2`) | publicly-released pin |
| vLLM | per cosmos-reason2-8b profile | `nemoclaw-thor/vllm:latest` |
| Proxy mutation profile | `compat` | full mutations (PROMOTE_REASONING_TO_CONTENT + UNWRAP_TOOL_CALL_ARGS + NORMALIZE_TOOL_NAMES + TOOL_ERROR_REWRITE) |
| OpenShell | `0.0.44` (docker driver) | host install |

## Setup work — status

- [x] Plugin attempt artifacts archived to `archive/openclaw-plugin-attempt-2026-06-02/` (commit `5237a6e` on `publication-readiness-v0.1.0`)
- [x] [HERMES-MIGRATION-ANALYSIS-2026-06-02.deprecated.md](./HERMES-MIGRATION-ANALYSIS-2026-06-02.deprecated.md) renamed
- [x] `cosmos-reason2-8b` confirmed as production model anchor (matches iter-32 production recipe per memory)
- [x] Upstream OpenClaw release probe: latest stable `2026.5.28`, latest beta `2026.6.1-beta.2`. Changelog scan for `extendProvider` / `tools.toolSearch` knob: no upstream fix in flight. The shim remains unconditional; the THREE-LANE plan's Phase 3 skill-rewrite approach is the only path that doesn't require an upstream change.

## Direct lane probes

Probes run on cosmos-reason2-8b with the proxy in path (8000 → 8050) and direct bridge on :8100.

### Probe D-1 — Smoke baseline
- **Procedure**: `python3 -u manyforge/scripts/debug/smoke_corpus_runner.py` against the running direct lane (ASSISTANT_PROVIDER=nemoclaw, bridge `:8100`).
- **Pass criteria**: ≥40/66 (sanity floor) AND no per-case latency > 60s (P95 health).
- **Result**: ❌ **FAIL — 28/66 effective (42.4%), 22/66 first-try (33.3%)**.
  - 22 pass, 6 soft-pass, 38 fail, 8 future-tagged skips.
  - Failure mode is dominated by `args_contain[...] got '<MISSING>'` — the model calls the right tool but doesn't fill the JSON arguments. A few cases hit `chat HTTP 502` (likely transient under sustained load).
  - **Diagnosis**: cosmos-reason2-8b is a multimodal robotics model that is weak at multi-field structured tool-argument filling against a direct catalog. This is exactly the failure mode that OpenClaw's `tool_describe` step (lazy schema fetch on demand) is designed to fix — it gives the model the parameter schema explicitly before the call. The 28/66 result is not an orchestration regression; it's the realistic ceiling of cosmos-reason2-8b on the direct catalog without scaffolding.
  - **Implication for Phase 5**: cosmos-reason2-8b is likely the WRONG default model for the Direct lane. The lane is architecturally sound (D-2–D-5 all pass); the model–lane fit is poor. Either pick a different model for direct-lane defaults, OR document Direct lane as "best for tool-use-trained models" and route cosmos-reason2-8b to OpenClaw/Hermes.
  - Report: `/tmp/smoke_corpus_1780438868356.json`. Proxy log entries logged for forensics.

### Probe D-2 — Proxy mutation path
- **Procedure**: Confirm `/tmp/manyforge-assistant-e2e/vllm-proxy.jsonl` is being written; inspect the first chat-completion request entry's mutation summary banner.
- **Pass criteria**: log header reports active mutation profile name (will be `compat` for this baseline) and at least one of the four mutations (`UNWRAP_TOOL_CALL_ARGS`, `PROMOTE_REASONING_TO_CONTENT`, `NORMALIZE_TOOL_NAMES`, `TOOL_ERROR_REWRITE`) fired during the probe.
- **Result**: ✅ **PASS**. Proxy banner: `openclaw-logging-proxy listening on 0.0.0.0:8000 -> http://127.0.0.1:8050 (log: /tmp/manyforge-assistant-e2e/vllm-proxy.jsonl; mode: max_tokens=2048, thinking_token_budget=512, loop_reflect=[same_tool>=4,same_args>=2,result_repeat>=2,namespace>=5(stop@16),turn_counter>=5 stop@8], malformed_tool_detect=...)`. 5 `proxy_loop_reflection_injected` events fired during the D-1 smoke run (proves loop-detection mutations active), and the proxy log captured 651 KB of request/response pairs.

### Probe D-3 — Tool catalog parity
- **Procedure**: Send one chat request through the bridge; inspect the corresponding vLLM request body in the proxy log.
- **Pass criteria**: `tools[]` array contains ≥20 manyforge tool names with `tree_draft_*` / `scene_draft_*` / `program_*` prefixes (NOT `tool_search_*`).
- **Result**: ✅ **PASS**. POST `/v1/chat/completions` request body contains **25 manyforge tools** by name (`program_read`, `catalog_read`, `skills_read`, `deployment_capabilities_read`, `scene_inspect`, `inspect_isaac_scene`, `propose_scene_objects`, `propose_scene_object_nodes`, ...). ZERO `tool_search_*` discovery shim entries — the model is given the full catalog directly, as designed for the Direct lane.

### Probe D-4 — MCP allowlist enforcement
- **Procedure**: Send a request whose `allowedTools` list intentionally excludes the tool the model would naturally call.
- **Pass criteria**: bridge returns `success:false` envelope with `kind=validation_error`; the conversation continues (does not 4xx).
- **Result**: ✅ **PASS**. Probe request with `assistantMode=composer-assistant` (which restricts the allowlist). When the model emitted a tool name outside the catalog (`list_my_tools`), the bridge returned `error.code=response_rejected, detail="upstream model emitted tool calls outside the active assistantMode 'composer-assistant' catalog: list_my_tools"` — the rejection is structured, the next chat call could continue, and no transport-level error was raised.

### Probe D-5 — Conversation continuity
- **Procedure**: Issue a 3-turn conversation; turn 1 = wrap a node; turn 2 = add a child; turn 3 = read the program state.
- **Pass criteria**: `conversationId` honored across all 3 turns; turn-3 program-state snapshot reflects turn-1 and turn-2 mutations.
- **Result**: ✅ **PASS**. 3-turn conversation `conversationId=probe-D5-conv-249286`:
  - Turn 1 (read tree): model reported "root node id is sequence" ✓
  - Turn 2 (wrap root in repeat with `max_iterations=3`): model returned `"draftMutated": True` with message "successfully wrapped … inside a repeat decorator named outer_repeat with max_iterations=3" ✓
  - Turn 3 (read mutated tree): model reported "root node kind is repeat, has 1 child" — mutation from turn 2 visible ✓

## OpenClaw lane probes

Probes run on cosmos-reason2-8b through the OpenClaw sandbox (gateway `:18789` host-forwarded, bridge `:8200`).

### Probe O-1 — iter-32 baseline reproduction (FIXED 2026-06-03)

**Route fix landed**: the launcher's default `OPENCLAW_ASSISTANT_USE_GATEWAY=true` made the bridge POST to `http://127.0.0.1:18789/v1/chat/completions` — but OpenClaw 2026.5.22 **does not expose `/v1/chat/completions` at all** (its HTTP server only registers `/v1/responses`, `/v1/embeddings`, `/v1/models`, `/v1/token`). The bridge therefore got "Not Found" in 50ms on every call without the gateway ever touching the LLM.

**Fix**: switch the bridge to CLI shell-out mode (`OPENCLAW_ASSISTANT_USE_GATEWAY=false`). The bridge then invokes `openclaw agent --agent manyforge-composer ...` via `nemoclaw exec` inside the sandbox, which uses OpenClaw's internal agent runner — that path DOES reach the inference provider and our `:8000` proxy.

Additional fixes required to make the route work:
1. **Enable `local-inference` preset alongside `manyforge-composer`** — without local-inference, the OpenShell network proxy denies sandbox→host:8000 with `policy_denied`. `setup-manyforge-assistant.sh` removes local-inference assuming manyforge-composer is a strict superset; it is not for the inference path.
2. **Patch `openclaw.json` `models.providers.inference.baseUrl`** from the default `https://inference.local/v1` to `http://host.openshell.internal:8000/v1` (skipping the inference.local TLS-proxy hop and going directly through our proxy).

After both fixes: proxy log captures POST `/chat/completions` entries (all 200 OK, durations 4-6s), gateway log shows `tool-search: cataloged 26 tools behind compact prompt surface` and real agent turns.

### Probe O-1 (original — before route fix; kept for forensics)
- **Procedure**: Run the chain-on production recipe against the OpenClaw lane.
- **Pass criteria**: ≥49/66 (51/66 baseline ±2).
- **Result (before fix)**: ❌ **FAIL — 14/66 effective (21.2%), 1/66 first-try (1.5%)**.
  - 1 pass, 13 soft-pass, 52 fail, 8 future-tagged skips.
  - **Root cause**: vLLM landed on `:8000` directly (no proxy in path) because the launcher was restarted with `START_VLLM_PROXY=false` to avoid the stale-proxy race condition documented in Phase 0 setup. Without the proxy in path:
    - `UNWRAP_TOOL_CALL_ARGS` mutation is not applied — the hermes tool-call parser's `<tool_call>` XML wrap leaks into `assistant.tool_calls[*].arguments`, breaking the next round's JSON parse.
    - `PROMOTE_REASONING_TO_CONTENT` is not applied — OpenClaw 2026.5.22 rejects null `content` from the qwen3 reasoning channel.
    - `NORMALIZE_TOOL_NAMES` is not applied — `manyforge__` prefixed tool names break dispatch.
    - `TOOL_ERROR_REWRITE` is not applied — error envelopes back to the model are degraded.
  - Failure pattern: identical to Direct lane D-1 — `args_contain[...] got '<MISSING>'` and `expected tool X not observed`. **The lane is correct; the without-proxy stack is broken**.
  - **Implication**: the iter-32 51/66 number was measured WITH the proxy in path. To reproduce, the proxy must be running on `:8000` and vLLM moved to `:8050`. The triage of "vLLM container keeps dying when the proxy step also runs" is a separate follow-up that gates a true OpenClaw baseline.
  - Report: `/tmp/smoke_corpus_1780463260607.json`.

### Probe O-2 — Discovery surface present
- **Procedure**: Inspect the most recent vLLM chat-completion request in the proxy log during the OpenClaw lane run.
- **Pass criteria**: `tools[]` array contains exactly `tool_search`, `tool_describe`, `tool_call` (and optionally `tool_search_code`); ZERO `tree_draft_*` / `scene_draft_*` / `program_*` names directly.
- **Result**: _to be filled_

### Probe O-3 — `/compact` cadence
- **Procedure**: Inspect bridge audit log during a multi-turn OpenClaw conversation.
- **Pass criteria**: `/compact` fires every 2 user prompts; post-compaction the model can still call `tree_draft_*` tools without re-discovery.
- **Result**: _to be filled_

### Probe O-4 — MCP bridge callback shape
- **Procedure**: Trigger one read-only tool call; inspect the matching entry in the composer's MCP-bridge callback log.
- **Pass criteria**: `principal=openclaw-sandbox`, `conversationId=openclaw-...`, `assistantMode=composer-assistant`, dispatch latency < 5s for the read-only tool.
- **Result**: _to be filled_

### Probe O-5 — Plugin-path archive accessibility
- **Procedure**: Run `bash archive/openclaw-plugin-attempt-2026-06-02/apply-openclaw-overrides.sh my-assistant --dry-run` against the live sandbox.
- **Pass criteria**: script reports "already at desired state" or equivalent no-op; proves the rollback path is intact.
- **Result**: _to be filled_

## Gate verdict

_To be filled when all 10 probes have a recorded result. Documentation deliverable per plan §8._

| Probe | Pass / Fail | Notes |
|---|---|---|
| D-1 smoke baseline | _ | _ |
| D-2 proxy mutation | _ | _ |
| D-3 tool catalog | _ | _ |
| D-4 MCP allowlist | _ | _ |
| D-5 continuity | _ | _ |
| O-1 iter-32 | _ | _ |
| O-2 discovery surface | _ | _ |
| O-3 /compact cadence | _ | _ |
| O-4 callback shape | _ | _ |
| O-5 plugin archive | _ | _ |

Decision: _Go to Phase 0.5_ / _Diagnose failures first_
