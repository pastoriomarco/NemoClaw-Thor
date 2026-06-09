# Three-Lane Architecture & Migration Plan

> **Replaces** [HERMES-MIGRATION-ANALYSIS-2026-06-02.deprecated.md](./HERMES-MIGRATION-ANALYSIS-2026-06-02.deprecated.md).
> Authored 2026-06-02 after the OpenClaw 2026.5.6+ tool-search-shim rebuild and the four-agent inventory pass that grounds the recommendations below.
> **Revised 2026-06-02 (rev. 2)** after first external review: Hermes natively supports `mcp_servers` (verified against Hermes 0.14.0 wheel — `cli.py:2691, 9314+`, `tools/mcp_tool.py`), so the custom MCP-to-Hermes-tool wrapper is dropped in favor of Hermes' native MCP path; the Hermes lane uses Hermes' native session/runs APIs rather than `/v1/chat/completions`; policy preset is split into shared egress + per-lane binary overlay; OpenClaw plugin/build artifacts are archived (not deleted) until after Phase 3 gate; per-lane opt-in for synthetic short-circuits; explicit composer provider registry task in Phase 1; proxy env-var rename + per-lane mutation profiles. Per-phase gates corrected for the 66-case corpus.
>
> **Revised 2026-06-02 (rev. 3)** after second external review: stale contradictions cleaned up (principle #2 wording, §3 findings, §7 target layout for `policies/` and the MCP-integration doc cross-reference, §13 timeline/question counts); new **Phase 0.5 Hermes contract spike** added to de-risk the Hermes lane before the big refactor by probing `mcp_servers` injection, both session APIs, and the vLLM tool-call parser end-to-end; the `mcp_servers.manyforge` config snippet in §5.3 expanded to include `MANYFORGE_LANE`/`MANYFORGE_PRINCIPAL`/`MANYFORGE_ASSISTANT_MODE` + outbound proxy envs; an explicit **lane-neutralization task for `manyforge-mcp-bridge.py`** in Phase 1 (it has OpenClaw defaults baked in today); Phase 1 marked as a **behavior-preserving refactor** with no concurrent OpenClaw native/plugin behavior changes; Hermes progress observer documented as **best-effort augmentation**, with composer bridge-tools callbacks as the hard correctness source.
>
> **Revised 2026-06-02 (rev. 4)** after verifying gaps in test/verification coverage and documentation discipline: **Phase 0 expanded** with 5 Direct-lane probes + 5 OpenClaw-lane probes mirroring Phase 0.5's shape, so all three lanes get explicit pre-refactor validation; **every phase now lists explicit documentation deliverables** as part of its gate (per-lane bring-up runbook sections, package READMEs, lane-comparison numbers rows, phase-specific result docs) — documentation is no longer an afterthought; new **load-bearing principle #4** added requiring `manyforge_specs` adherence before any modification to manyforge code, with per-phase specs-check sub-sections in Phase 1 (envelope/MCP wrapper/catalog hash), Phase 2 (cross-workspace bridge location), and Phase 5 (composer lane router contract). Total time bumps to 17-23 working days.
>
> **Revised 2026-06-03 (rev. 5)** after debugging the OpenClaw lane to first green smoke: §4.6 **split-policy guidance retracted** — `nemoclaw policy-add` REPLACES same-keyed `network_policies.<name>` blocks rather than additively merging, so applying `manyforge-egress-shared.yaml` then `manyforge-openclaw.overlay.yaml` silently dropped the port-9000 endpoints. The corrected shape is one merged file per lane (`policies/manyforge-composer.merged.yaml` for OpenClaw, `policies/manyforge-composer-hermes.merged.yaml` for Hermes), each carrying both endpoints AND binary subjects. Two additional findings folded into §4.6: (a) the MCP bridge runs in an isolated network namespace — direct TCP to `host.openshell.internal` times out, so `NO_PROXY` must list loopback only and the bridge must route through the OpenShell proxy at `10.200.0.1:3128`; (b) OpenShell's policy enforcer matches binary subjects by *resolved* exe path, so the whitelist must include `/usr/bin/python3.13` (or whichever versioned interpreter `readlink /proc/<pid>/exe` produces on the active base image), not only the `/usr/bin/python3` symlink. `setup-manyforge-assistant.sh` now applies the merged file with hard-fail post-apply verification on both invariants. End-to-end validated: 3/3 P-series smoke cases (P1 228s / P2 69s / P3 148s) on qwen3.6-35b-a3b-nvfp4 via OpenClaw tools-mode.
>
> **Revised 2026-06-08 (rev. 6)** after Hermes native-MCP pipeline validation: the Hermes lane keeps a **lean prompt catalog** and fetches parameterized node schemas through `catalog_read`; full inline node params are not the baseline because they match quality on the insert probe while adding prompt cost. The Hermes dispatcher now terminates only on `run.*` lifecycle events, and a repeated `catalog_read` loop-breaker prevents successful read-only discovery loops. Next publishable evidence is a clean 75-case Hermes run with self-heal on, followed by same-day restored OpenClaw and Direct baselines.

## TL;DR

Stop treating any single lane as the production target. Build three first-class lanes — **Direct vLLM**, **OpenClaw**, **Hermes Agents** — each running in the configuration its upstream intends, behind one shared core of tooling. Then route per request shape and benchmark per lane on the metric that matches the lane's nature.

## 1. Three load-bearing principles

These are not negotiable. Every architectural choice below derives from them.

1. **All three lanes must work correctly, each as the upstream intends.** No fighting the OpenClaw tool-search shim, no shoehorning Hermes into a stateless cycle, no privileging the direct lane just because it was easiest to wire. If OpenClaw demands a three-trip `tool_search → tool_describe → tool_call` discovery for every tool, we pay it. If Hermes wants memory and skills active, we leave them active. The cost shows up in benchmarks; the lane-routing layer decides where to send each request.
2. **Tooling behind the lanes must be the same or as close as possible.** One vLLM proxy, one smoke corpus, one tool catalog source of truth, one projection library, one MCP bridge, one set of shared egress rules (with thin per-lane binary-whitelist overlays — see §4.6), one observability format. Lane-specific code lives at the thin adapter at the top of the stack; everything below is shared.
3. **Each lane is benchmarked on the metric that matches its nature.** Direct and OpenClaw are deterministic per-turn — judged on per-case pass rate against the smoke corpus. Hermes is a learning agent — judged on session-level outcomes over an N-interaction longitudinal run where memory and skills can compound. Comparing them on a single number is a category error.
4. **`manyforge_specs` is authoritative; respect it before changing anything in the manyforge tree.** The composer, the MCP bridge, the bridge envelope (`manyforge.assistant.provider_request.v0`), the catalog hash discipline, the policy contracts — these are specs-driven, not free-form. Before modifying any file under [`dev_ws/src/manyforge/`](/home/tndlux/workspaces/dev_ws/src/manyforge/), check the relevant section of [`dev_ws/src/manyforge_specs/`](/home/tndlux/workspaces/dev_ws/src/manyforge_specs/) — particularly [docs/cross-workspace-conventions.md](/home/tndlux/workspaces/dev_ws/src/manyforge_specs/docs/cross-workspace-conventions.md), [docs/agent-playbook.md](/home/tndlux/workspaces/dev_ws/src/manyforge_specs/docs/agent-playbook.md), and any ADR / kernel / port spec touching the file in question. If the refactor REQUIRES a spec change (e.g. moving the bridge code introduces a new contract), the spec change must land first, with explicit justification, and reviewed before any implementation work touches the dependent code. Silent drift from specs is the failure mode this principle blocks.

## 2. The three lanes — intent and role

| Lane | Wire path | Intent | When it should win |
|---|---|---|---|
| **Direct** | Composer → `manyforge_assistant_bridge:8100` → `vllm-proxy:8000` → `vllm:8050` | Lowest-latency stateless turn. Full manyforge catalog exposed directly to the model. Deterministic, single round-trip per tool call. | Known-workflow operator runs; CI smoke parity; the lane that proves whether a regression is model-side vs orchestration-side. |
| **OpenClaw** | Composer → `openclaw_assistant_bridge:8200` → OpenClaw gateway `:18789` → `vllm-proxy:8000` → `vllm:8050` | Structured agent harness, MCP runtime, native discovery surface (`tool_search`/`tool_describe`/`tool_call`), policy presets, session continuity, `/compact`-driven turn management. Each tool costs ~3 LLM round-trips at first invocation. | Multi-step plans where the discovery surface keeps the model's working surface small; CLI-style sessions; the lane that exercises OpenClaw's session/skills/audit features. |
| **Hermes** | Composer → `hermes_assistant_bridge:8300` → Hermes `:8642` → `vllm-proxy:8000` → `vllm:8050` | Self-improving agent with persistent memory, hot-reloadable skills, cron, todo, delegation. Lossy determinism by design — session state compounds across turns. | Repeat users with preference patterns ("always wrap in retry-3"); long-running interactions; emergent skill capture; the lane that bets non-determinism is worth it for compounding capability. |

The composer-side routing layer (Section 9) picks a lane per request. The user can pin a lane manually. The default rotates based on benchmarks, not architectural preference.

## 3. Findings that drive the design

Carrying forward from the prior hermes-migration analysis (preserved per the inventory), the rebuild work, and the four inventory agents:

- **OpenClaw 2026.5.6 introduced the native tool-search shim** as a baked-in feature ([dist/pi-tools-iVT6BGHc.js](file:///usr/local/lib/node_modules/openclaw/dist/pi-tools-iVT6BGHc.js):1014–1063 in 2026.5.22). It is **not toggleable** — `tools.toolSearch=false` in `openclaw.json` is silently ignored. NemoClaw's `patch-openclaw-tool-catalog.js` is a no-op on 2026.5.6+ ([scripts/patch-openclaw-tool-catalog.js:244](file:///home/tndlux/NemoClaw/scripts/patch-openclaw-tool-catalog.js#L244)).
- **`tool_search` is a substring tokenizer, not semantic search.** Score weights: exact match ×20, name substring ×8, id ×6, label ×4, description ×2 (`scoreEntry` at `pi-tools:633-714`). Zero-hit when query terms don't substring-match any of those fields. **Skill prompts must list exact tool names**, not paraphrase what the tool does.
- **Default mode is `"code"` but auto-falls-back to `"tools"`** when the Node runtime lacks `--permission` support — which NemoClaw sandboxes lack (`pi-tools:329-331`). The structured `tool_search`/`tool_describe`/`tool_call` triplet is what we actually get.
- **`tool_call` wraps the real tool result**: `{ tool: <compact entry>, result: <real result> }` (`pi-tools:728-743`). The model must be taught to read `.result.content[].text`, not the wrapper.
- **`tool_describe` accepts a bare tool name as `id`** (`pi-tools:findEntry`). So 3-trips collapse to 2 whenever the skill prompt has pre-named the tools.
- **System-prompt teaching from OpenClaw is essentially nil** — two short strings in `thread-lifecycle:1511,1576`. The discovery protocol must be taught by the manyforge skill prompt itself.
- **The bundled NemoClaw `nemoclaw` plugin owns `api.registerProvider({id:"inference"})` unconditionally** ([nemoclaw plugin index.js:178](file:///sandbox/.openclaw/extensions/nemoclaw/dist/index.js#L178)). Plugin-side `normalizeToolSchemas` overrides collide with no upstream `extendProvider` API. The right answer is not a plugin; it is the skill rewrite.
- **Hermes consumes OpenAI-style structured `tool_calls`**, not XML. vLLM's `--tool-call-parser hermes` translates Nous Hermes XML emitted by trained models into OpenAI JSON before Hermes Agent ever sees it. The wire shape into Hermes Agent is identical to the wire shape into the direct lane.
- **Hermes natively supports MCP servers via top-level `mcp_servers` config** ([upstream docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp); verified against the 0.14.0 wheel — `hermes_agent/cli.py:2691, 9314+` auto-reloads on `mcp_servers` mtime changes, `tools/mcp_tool.py` does the registration, `tools/mcp_oauth_manager.py` reads `mcp_servers.<name>.oauth`). NemoClaw's current `hermes-config.ts` simply doesn't emit `mcp_servers` — it only sets `plugins.enabled = ["nemoclaw"]`. **We add the `mcp_servers.manyforge` entry ourselves**, pointing at the lane-neutral `manyforge-mcp-bridge.py`; no custom Hermes plugin wrapper is needed.
- **Hermes also exposes native session/runs APIs** (`/v1/runs`, `/api/sessions/{id}/chat`, `/v1/responses` with `X-Hermes-Session-Id` / `X-Hermes-Session-Key` headers — [API server docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server/)). The Hermes lane uses those, not `/v1/chat/completions`, so Hermes owns its agent loop, memory writes, skill emergence, and progress events. The bridge submits and observes — it does NOT shuttle per-turn tool calls.
- **Hermes ships memory + skills + cron + todo + delegation as first-class capabilities** (`agents/hermes/config/hermes-config.ts`). Per the explicit ask, those stay enabled in the Hermes lane — bake-off must account for this.
- **`adapter.py` has a natural split at ~line 680.** Top half: lane-agnostic envelope translation (projection, prompt assembly, MCP allowlist, session keys). Bottom half: OpenClaw-specific command-builders and response-parsers. This is the extraction seam.
- **The projection mirror is real and documented.** `_build_program_summary` etc. exist twice — in NemoClaw-Thor's `adapter.py` and dev_ws's `manyforge_assistant_bridge/bridge.py`. Without a shared package, Hermes would be the third copy.
- **`vllm-proxy.py`, `smoke_corpus.yaml`, and `smoke_corpus_runner.py` are already lane-agnostic.** No work needed (except the proxy env-var rename in §4.4); they become the universal verifier surface.
- **`policies/manyforge-composer.preset.yaml` is NOT lane-agnostic** — it whitelists `/usr/local/bin/openclaw`, Node, and Python as allowed subjects. Egress rules are lane-agnostic but subject whitelists are lane-specific. Split required (§4.6).
- **`scripts/manyforge-mcp-bridge.py` had OpenClaw defaults baked in** — principal `openclaw-sandbox` and conversation/request prefixes `openclaw-...`. Phase 1 parameterizes the bridge with `MANYFORGE_LANE` and `MANYFORGE_PRINCIPAL` so the same script serves Hermes (`MANYFORGE_LANE=hermes`, `MANYFORGE_PRINCIPAL=hermes-<sandbox>`) and any future lane. Request-scoped MCP tool-window files were removed 2026-06-04; the wrapper exposes the assistant-mode catalog and lets Composer/MCP policy enforce authorization.
- **Cosmos-specific synthetic bypasses were retired on 2026-06-04.** `/compact` orchestration (~80 lines) remains trapped inside the OpenClaw `service.py`; lift that to a shared `assistant_session` layer if Direct or Hermes need it.
- **Baseline yardstick to beat is empirical, not aspirational.** Cosmos-Reason2-8B iter-32 chain-on production recipe: 51/66 (77.3%), per memory. The Qwen3.6-35B 56/66 (84.8%) from the pre-rebuild bake-off is a peer reference but not the production model — keep both visible, anchor decisions on the cosmos number.
- **Hermes memory is an evaluation hazard.** `/sandbox/.hermes/{memories,sessions,runtime/state.db}` must be reset between bake-off runs for determinism comparisons. For the Hermes-native longitudinal harness, it stays.
- **Hermes' prompt catalog is deliberately lean.** `nodeCatalog` is a
  node-kind chooser, not a full parameter schema dump. Parameterized node
  schemas come from `catalog_read`; repeated identical catalog reads are capped
  so the model cannot spend a whole run on successful read-only discovery.

## 4. The universal core — what is shared

This is principle #2 made concrete. Everything in this list has **one implementation** consumed by all three lanes.

### 4.1 Tool catalog source of truth

The composer's `GET /api/assistant/modes/composer-assistant` is the single source of truth. The four artifacts that consume it today (direct bridge, OpenClaw plugin Dockerfile build, OpenClaw bridge prompt assembly, smoke runner verification) become one consumer with one cached representation. The OpenClaw lane no longer bakes a copy into a Dockerfile; the Hermes lane fetches at startup.

### 4.2 The `manyforge_assistant_common` library (new)

Extracted from the top half of `openclaw_assistant_bridge/adapter.py`. Pure-function modules, zero transport dependencies, used by all three lane adapters and any future lane:

- `projection.py` — `build_program_summary`, `build_scene_summary`, `project_tree_node`, `collect_ancestor_paths`, `project_node_catalog`, `project_skill_catalog`. Single canonical implementation of the program/scene/catalog projection that today is mirror-duplicated in two repos.
- `prompt.py` — `build_agent_prompt(envelope, *, discovery_mode: Lane)`. The preamble + rules + tail-checklist block. The `discovery_mode` parameter swaps in the OpenClaw discovery primer (Section 6) or the direct-catalog header for the other two lanes. Same checklist, same RULES block.
- `envelope.py` — `error_envelope`, `request_id_from_payload`, `is_action_shaped_prompt`, `derive_session_key`. The DTOs and helpers around the Composer `manyforge.assistant.provider_request.v0` envelope. Lane-agnostic.
- `tool_calls.py` — `extract_tool_calls`, `canonical_tool_name`, `dedupe_known`, `unwrap_openclaw_envelope` (handles the `{tool, result}` wrap from OpenClaw's `tool_call`). The OpenClaw envelope unwrap is one helper here; lanes that don't need it just don't call it.
- `mcp_catalog.py` — mode-catalog inspection helpers only. Prompt-keyword tool-window inference is intentionally removed; the mode-scoped MCP wrapper remains the mutation authorization point.

This library lives in NemoClaw-Thor at `manyforge/common/` and is `pip install -e`'d by every bridge venv. The dev_ws sibling repo's `manyforge_assistant_bridge` imports it instead of keeping its mirror. The projection mirror dies.

### 4.3 `assistant_session` layer (new)

Extracted from service.py. Owns the orchestration logic that should never have been transport-coupled. Each policy below is **per-lane opt-in via a `SessionPolicy` config** — defaults preserve the iter-32 OpenClaw behavior, but Hermes and Direct lanes can disable individual policies if their nature already handles the concern.

- **Compaction policy.** Today's iter-32 recipe is "bridge-fires `/compact` every 2 prompts." Default on for OpenClaw. For Hermes, **OFF by default** — Hermes owns its own session/memory lifecycle via the runs API and we should let it (per principle #1: don't fight upstream's intended behavior). For Direct, the bridge does its own truncation strategy; default on with a different threshold.
- **Synthetic clarification + retry-loop detector.** Cosmos-specific patches that today are in `service.py:355-508` — they apply where the *model* is what's broken, not the transport. Default on for OpenClaw (proven there). For Hermes and Direct, **opt-in** — flag them per-config and benchmark with/without before defaulting on.
- **Principal binding + session key derivation.** Lane-agnostic, always on. Just a pure-function move.
- **Circuit breaker + cancellation.** Lane-agnostic, always on.

The `SessionPolicy` config is per-lane and lives in `manyforge/lanes/<lane>/policy.yaml`. The session orchestrator reads it at startup.

### 4.4 The vLLM proxy

`scripts/proxy/vllm-proxy.py` stays where it is and remains lane-agnostic in spirit — its mutations (`UNWRAP_TOOL_CALL_ARGS`, `NORMALIZE_TOOL_NAMES`, `PROMOTE_REASONING_TO_CONTENT`, `TOOL_ERROR_REWRITE`) target wire shape coming back from vLLM, which is identical regardless of caller. But two cleanups are required:

- **Rename `OPENCLAW_PROXY_*` env vars to `MANYFORGE_PROXY_*`** (with `OPENCLAW_PROXY_*` kept as deprecated aliases for one release cycle). Today's names imply OpenClaw-only applicability and would confuse Hermes/Direct operators.
- **Define explicit per-lane mutation profiles**: `native` (only mutations needed by model wire shape, no agent-level rewrites), `compat` (current full set, preserves OpenClaw's tool-error rewriting), `prod` (whatever wins in the bake-off). Record the active profile name in every audit entry so smoke-corpus reports can correlate behavior to profile. The profile is selected per-bridge at startup via env var, defaulting to `compat` for OpenClaw, `native` for Direct/Hermes.

### 4.5 Smoke corpus + runner

`scripts/debug/smoke_corpus.yaml` and `smoke_corpus_runner.py` stay where they are. The runner hits the composer at `:9000` and is lane-agnostic by construction — whichever provider Composer is pointed at is what gets exercised. The runner becomes the universal verifier surface across all three lanes for per-turn comparison.

The three two-lane A/B harnesses (`ab-direct-vs-openclaw.py`, `lane-3x3-smoke.py`, `lane-parity-diff.py`) collapse into one parametric `scripts/debug/compare_lanes.py --lanes direct,openclaw,hermes`. Same corpus, three lane runs, side-by-side report.

### 4.6 Policy preset — one merged file per lane (revised 2026-06-03)

A single preset is unsafe. The current `policies/manyforge-composer.preset.yaml` whitelists `/usr/local/bin/openclaw`, Node, and Python — OpenClaw-shaped subjects. Hermes' `policy-additions.yaml` whitelists Hermes/Python and currently lacks Composer port 9000 egress. Mixing them would either grant OpenClaw to a Hermes sandbox or strand Hermes without composer access.

Rev. 2 of this plan proposed splitting into a shared `manyforge-egress-shared.yaml` plus a per-lane `*.overlay.yaml`, with sandbox setup scripts applying both at onboarding. **That shape does NOT work with current `nemoclaw policy-add` semantics — `policy-add` REPLACES same-keyed `network_policies.<name>` blocks rather than merging additively.** Applying `manyforge-egress-shared.yaml` then `manyforge-openclaw.overlay.yaml` produced a final policy with only the overlay's `binaries:` block — the shared file's port-9000 endpoints were silently dropped. The MCP bridge's proxy request to `host.openshell.internal:9000/api/assistant/modes/composer-assistant` returned HTTP 403 ("not permitted by policy"); OpenClaw's `bundle-mcp` logged "Request timed out" because the bridge's `tools/list` handler never received the mode manifest; the model saw "cataloged 1 tools" (OpenClaw core only) and every `tool_call(...)` failed with "Unknown tool id". Confirmed end-to-end 2026-06-03 by reading `openshell policy get my-assistant --full` after each apply step (versions 18 / 19 / 20).

The correct shape (revised):

- **`policies/manyforge-composer.merged.yaml`** — the OpenClaw lane's single source of truth. Contains BOTH the egress endpoints (host.openshell.internal:9000 for `/api/mcp`, `/api/mcp/sse`, `/api/assistant/modes/**`, `/api/assistant/bridge/tools/**`; host.openshell.internal:8000 for vLLM `/v1/*`) AND the binary subject whitelist (`/usr/local/bin/openclaw`, `/usr/local/bin/node`, `/usr/bin/python3`, plus every Python interpreter the bridge has actually been observed to resolve to — currently `/usr/bin/python3.11/12/13`). Applied by `setup-manyforge-assistant.sh` with hard-fail post-apply verification that the active policy contains both `port: 9000` and `/usr/bin/python3.13`.
- **`policies/manyforge-composer-hermes.merged.yaml`** (Phase 4, same shape) — Hermes lane equivalent. Endpoints identical to the OpenClaw file; binaries swap OpenClaw/Node for `/usr/local/bin/hermes` (and any Hermes-specific egress, e.g. Nous Portal broker, if Phase 4 enables managed-tool gateways).
- **`policies/manyforge-egress-shared.yaml` + `policies/manyforge-{openclaw,hermes}.overlay.yaml`** — kept in `policies/` as documentation of the conceptual decomposition (endpoints vs. binary subjects) and as source material for the merged files. They are NOT applied at onboarding. A future `nemoclaw policy-add` that grows additive-merge semantics could re-enable the split-apply path; until then the merged file is authoritative.
- **Direct lane** is host-side, no sandbox, no preset needed.

**Why `/usr/bin/python3.13` specifically.** OpenShell's policy enforcer matches binary subjects by *resolved* exe path (`readlink /proc/<pid>/exe`), not the symlink the spawn command used. The MCP bridge launches as `python3 bridge.py`, but the kernel records the underlying interpreter — currently `/usr/bin/python3.13` on the OpenShell `my-assistant` base image. Without the versioned entry, the proxy denies the bridge's egress and the MCP catalog never registers. The merged file lists both the symlink (forwards-compatibility for any spawn path that records it) and every interpreter we have actually observed in production.

**Why the bridge must route through the proxy at all.** OpenClaw spawns the MCP bridge in an isolated network namespace. Direct TCP from inside that netns to `host.openshell.internal` (which resolves to `172.18.0.1` — the Docker bridge gateway) times out — that IP is not reachable from the netns. The only path back to Composer is via OpenShell's egress proxy at `10.200.0.1:3128`, which is reachable. Hence `NO_PROXY` in the bridge env must cover loopback only — adding `host.openshell.internal` to it sends urllib down the direct-connect path and the bridge hangs on `tools/list` for the full `MANYFORGE_REQUEST_TIMEOUT` window.

### 4.7 Observability

One JSONL audit format, written by all three bridges in identical shape:

```
{ ts, lane, requestId, conversationId, principal, model, transport,
  toolsObserved[], toolsExpected[], compactionFires, latencyMs,
  exitReason, errorChain }
```

`/tmp/manyforge-assistant-e2e/{direct,openclaw,hermes}-bridge-audit.jsonl` parallel files. The vLLM proxy log stays where it is — it's a single shared resource because there's a single proxy. Smoke runner ingests all three audit streams for compare reports.

## 5. Lane adapters — what is deliberately different

Each adapter is small (target: <500 lines) and does exactly the lane-specific work and nothing else.

### 5.1 Direct lane adapter

Already lives in `dev_ws/src/manyforge/manyforge_assistant_bridge/`. After the refactor it imports from `manyforge_assistant_common` and `assistant_session`. The adapter's only job is to format the OpenAI `/v1/chat/completions` request with `tools=[...real manyforge catalog...]`, dispatch, parse `choices[0].message.tool_calls`, run them through the MCP allowlist, post tool results back as `role: tool` messages, loop until done. No discovery, no wrapping, no session memory.

### 5.2 OpenClaw lane adapter

Lives at `manyforge/lanes/openclaw/`. After the extraction, the OpenClaw bridge is one transport strategy (`build_command`, `dispatch`, `parse_response`, `normalize_response`) implementing the shared `AssistantTransport` interface. The orchestration logic comes from `assistant_session`. The lane-specific work is exactly:

1. Build the discovery-primer prompt (Section 6).
2. Call the OpenClaw gateway at `:18789/v1/chat/completions` with model name prefixed `openclaw/manyforge-composer`.
3. Parse the response — three control tools (`tool_search`, `tool_describe`, `tool_call`) may have fired any number of times; the universal `tool_calls.unwrap_openclaw_envelope` extracts the real tool result from `{tool, result}` wrappers.
4. Audit the discovery turns vs the real-tool turns separately so the bake-off can show the round-trip overhead honestly.

The current `manyforge-direct-tools` OpenClaw plugin and the `openclaw-overrides/*.json` manifests are **retired**. They were workarounds for the shim; we now embrace the shim. The `apply-openclaw-overrides.sh` script is retired. The `Dockerfile.manyforge-sandbox{,-prebuilt}` and `build-manyforge-sandbox-image.sh` are retired. The plugin/catalog bake-in goes away. What survives: the egress preset, the skill+MCP registration, the agent profile.

### 5.3 Hermes lane adapter

Lives at `manyforge/lanes/hermes/`. Mostly new code. Per the explicit ask, Hermes runs with memory + skills + cron + todo enabled — and per principle #1, we let Hermes own its agent loop rather than shuttling per-turn over `/v1/chat/completions`. The adapter is deliberately thin:

1. **Provider registration.** Composer's `build_assistant_provider()` only knows `nemoclaw` and `openclaw` ([assistant_provider.py:614](/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_composer/backend/assistant_provider.py#L614)); the launcher's routing in [assistant.sh:74](/home/tndlux/workspaces/dev_ws/src/manyforge/scripts/lib/assistant.sh#L74) falls back to Direct for unknown providers. Phase 1 adds an explicit lane registry — a `LANE_REGISTRY` dict in both files keyed by provider id with `(bridge_endpoint, default_port, transport_class)` per entry. The `hermes` provider id lands here.
2. **NemoClaw config emission.** `hermes-config.ts` currently emits no `mcp_servers` block. We add a fork or NemoClaw-Thor overlay that emits:
   ```yaml
   mcp_servers:
     manyforge:
       command: python3
       args: ["/sandbox/manyforge/scripts/manyforge-mcp-bridge.py"]
       env:
         MANYFORGE_LANE: "hermes"
         MANYFORGE_PRINCIPAL: "hermes-${SANDBOX_NAME}"
         MANYFORGE_ASSISTANT_MODE: "composer-assistant"
         MANYFORGE_COMPOSER_BASE: "http://host.openshell.internal:9000"
         HTTP_PROXY: "${HTTP_PROXY}"
         HTTPS_PROXY: "${HTTPS_PROXY}"
         NO_PROXY: "${NO_PROXY}"   # loopback only; do NOT add host.openshell.internal — see below
   ```
   The bridge script is the lane-neutral MCP server (after the Phase 1 parameterization). Hermes discovers and registers its tools at startup via its native MCP path. Auto-reload is built in (`cli.py:9314+` watches `mcp_servers` mtime). The proxy envs match the OpenClaw provisioner's allowlist so the bridge can reach `host.openshell.internal:9000` through OpenShell's outbound proxy — without these, the bridge calls fail closed and Hermes surfaces no MCP tools at all.

   **Do not add `host.openshell.internal` to `NO_PROXY`.** The bridge documents this at [`manyforge-mcp-bridge.py:75-81`](file:///home/tndlux/workspaces/dev_ws/src/manyforge/scripts/manyforge-mcp-bridge.py#L75-L81): "the only network path to Composer from inside the sandbox runs through the proxy at 10.200.0.1:3128 — bypassing it yields 'Connection refused'." Match the OpenClaw provisioner pattern — proxy stays enabled for `host.openshell.internal`; `NO_PROXY` covers loopback hosts only.
3. **Hermes bridge service** (`:8300`). Submits work to Hermes via its **native session APIs** — `POST /api/sessions/{session_id}/chat` or `POST /v1/runs` (whichever the bake-off probe in Phase 4 confirms is the right one for our streaming + cancellation needs). Sets `X-Hermes-Session-Id` from the composer's `conversationId` and `X-Hermes-Session-Key` for session continuity. **Bridge does NOT loop over tool calls** — Hermes runs its own agent loop, fires its own MCP tools (which through our `mcp_servers.manyforge` config dispatch to the lane-neutral bridge script, which in turn POSTs to composer `/api/assistant/bridge/tools/{toolId}` as the mode-scoped mutation path).
4. **Observation.** The bridge consumes Hermes' progress event stream (SSE on the runs API) and emits the universal audit shape for every observable event: tool calls (visible as Hermes' `mcp_manyforge_<tool>` prefixed names — strip the prefix for parity with the other lanes), memory writes, skill creations, cron fires, delegation. **Progress observation is best-effort augmentation, not correctness-bearing.** The composer's `/api/assistant/bridge/tools/{toolId}` callback log is the hard source of truth for what tools ran and with what effect — it's the path the smoke runner already verifies against. Pinned Hermes may not expose every internal event in the shape the plan assumes; if a progress event is missing or malformed, drop it from the audit and continue. Final response goes back through the composer envelope when Hermes signals run completion.
5. **Memory and skills lifecycle.** The composer's `conversationId` is the Hermes session key. Memory accumulates across turns of the same conversation. `/sandbox/.hermes/memories` is cleared between bake-off runs only when `--reset-hermes-state` is passed.
6. **Asymmetric metric collection.** Per-event audit goes in the universal shape (one entry per Hermes-emitted event). In addition, the lane emits `hermes-session-events.jsonl` capturing skill creations, memory writes, cron-firing events, and delegation calls. The longitudinal harness (Section 9) reads this.

The custom MCP-to-Hermes-tool wrapper from rev 1 of this plan (~400 lines) is dropped. Hermes' native MCP support replaces it. The remaining new code in the lane is mostly the session-dispatch + progress-stream observer (~150 lines projected).

### 5.4 The shared `AssistantTransport` interface

```python
class AssistantTransport(Protocol):
    lane: Literal["direct", "openclaw", "hermes"]
    def build_request(self, prompt: AgentPrompt, ctx: SessionCtx) -> WireRequest: ...
    async def dispatch(self, req: WireRequest, *, timeout_s: float) -> WireResponse: ...
    def parse_response(self, resp: WireResponse) -> ParsedResponse: ...
    def normalize_tool_calls(self, parsed: ParsedResponse) -> list[ToolCall]: ...
```

One FastAPI service factory wraps any `AssistantTransport`. Three implementations: `DirectTransport`, `OpenClawTransport`, `HermesTransport`. The bridge supervisor (`scripts/launch.sh`) instantiates the right one based on `ASSISTANT_PROVIDER`.

## 6. Wire format translation

The wire shapes the three lanes consume:

| Concern | Direct | OpenClaw | Hermes |
|---|---|---|---|
| Bridge → agent endpoint | OpenAI `/v1/chat/completions` on vLLM proxy `:8000` | `POST /v1/chat/completions` on OpenClaw gateway `:18789` with model id `openclaw/<agent>` | `POST /api/sessions/{conversationId}/chat` (or `/v1/runs` — TBD Phase 4 probe) on Hermes `:8642` with `Authorization: Bearer $API_SERVER_KEY` and `X-Hermes-Session-Id`/`X-Hermes-Session-Key` headers |
| Who owns the agent loop | The bridge (per-turn) | OpenClaw (per-turn, but with discovery overhead) | **Hermes** (multi-turn, with memory + skills + cron compounding) |
| `tools[]` shape (to model) | Full manyforge catalog (~25 tools) | Three discovery tools (`tool_search`, `tool_describe`, `tool_call`) | MCP-registered manyforge tools from `mcp_servers.manyforge`; prompt carries a lean `nodeCatalog`, while full parameterized node schemas are fetched with `catalog_read` |
| Reasoning channel | Qwen3 `<think>` blocks parsed by vLLM `--reasoning-parser qwen3` into `reasoning_content` | Same (the proxy promotes reasoning→content on streamed responses) | Same |
| Tool-call shape | OpenAI structured `tool_calls[]` | OpenAI structured `tool_calls[]`, but real-tool dispatch is wrapped in OpenClaw's `tool_call` envelope | OpenAI structured `tool_calls[]` (vLLM's `--tool-call-parser hermes` translates Nous Hermes XML model output to JSON before Hermes sees it) |
| Tool result envelope | OpenAI `role: tool, tool_call_id, content` | Same back from `tool_call`, but content is `{tool, result}` — universal `unwrap_openclaw_envelope` strips the wrapper | OpenAI `role: tool` (Hermes handles internally; bridge never sees individual tool results, only the run's final response + progress events) |
| Streaming | OpenAI SSE deltas | Same | Hermes runs API progress-event stream (SSE); event types include tool invocations, memory writes, skill emergences, partial responses |
| Compaction trigger | Bridge-fired truncation (per session policy) | Bridge-fired `/compact` (the iter-32 recipe) — `SessionPolicy.compaction=on` | **Hermes-managed** (`SessionPolicy.compaction=off` — Hermes owns its session lifecycle; we don't override) |
| MCP transport | Bridge calls composer `/api/assistant/bridge/tools/{toolId}` directly | OpenClaw's MCP runtime calls the composer endpoint via the mode-scoped wrapper | Hermes' native MCP runtime calls the lane-neutral `manyforge-mcp-bridge.py` configured under `mcp_servers.manyforge`, which calls composer `/api/assistant/bridge/tools/{toolId}` |
| Memory | None | OpenClaw session continuity, opt-in `/compact` | Memory writes persisted to `/sandbox/.hermes/memories`, accumulating across the conversation |
| Composer effect tracking | Bridge audit (per-turn) | Bridge audit (per-turn) | **Track via Composer's `/api/assistant/bridge/tools/{toolId}` callbacks**, not Hermes-visible tool names (Hermes prefixes MCP tools as `mcp_manyforge_<tool>`; strip prefix for parity in audit/comparison reports) |

**Proxy mutations work the same on all three lanes** — they target the wire shape coming back from vLLM, which is identical regardless of who called it. Per §4.4, each lane selects a mutation profile (`native`/`compat`/`prod`) at startup; profile name is recorded in every audit entry.

### The OpenClaw skill rewrite (the only "creative" piece of lane-specific prompting)

The OpenClaw lane needs the manyforge skill prompt rewritten to teach the discovery protocol. Concrete additions:

- **Discovery primer paragraph** explaining `tool_search` → `tool_describe` → `tool_call` and the `.result` unwrap.
- **Pre-named tool list** of the full manyforge catalog. The model is told it can skip `tool_search` and go straight to `tool_describe({id:"tree_draft_wrap_node"})` whenever it already knows the name — collapses 3 trips to 2.
- **Describe-once-per-session policy** so the same tool's schema isn't re-fetched.
- **Envelope-read instruction** so the model reads `.result.content[].text` and ignores the `tool` wrapper.
- **Multi-term search hints** for the cases where the model genuinely doesn't know a tool's name.

This rewrite is the only place where "embracing the shim" shows up as user-visible code. It lives at `manyforge/lanes/openclaw/skill_addendum.md`, appended to the shared `manyforge_assistant_common/skill_base.md`.

## 7. Repository refactor — target layout

```
NemoClaw-Thor/manyforge/
├── common/                          # SHARED — Python package, pip install -e
│   ├── projection.py                # was adapter.py L1-180
│   ├── prompt.py                    # was adapter.py L180-500 (parameterized by lane)
│   ├── envelope.py                  # was adapter.py L500-680
│   ├── tool_calls.py                # was adapter.py L1571-1700 + new unwrap helper
│   ├── mcp_catalog.py               # mode-catalog helpers, no prompt inference
│   ├── skill_base.md                # the lane-agnostic skill body
│   └── tests/
│
├── assistant_session/               # SHARED — orchestration layer
│   ├── compaction.py                # was service.py L42-78, L608-662
│   ├── circuit_breaker.py           # moved from openclaw_assistant_bridge/
│   ├── session_key.py               # was adapter.py session-key derivation
│   └── tests/
│
├── lanes/
│   ├── direct/                      # imports common+session; transport adapter
│   │   ├── transport.py
│   │   ├── service.py               # FastAPI on :8100
│   │   ├── skill_addendum.md        # tiny — direct-catalog header
│   │   └── tests/
│   ├── openclaw/                    # imports common+session
│   │   ├── transport.py
│   │   ├── service.py               # FastAPI on :8200
│   │   ├── skill_addendum.md        # the discovery primer
│   │   └── tests/
│   └── hermes/                      # imports common+session
│       ├── transport.py
│       ├── service.py               # FastAPI on :8300
│       ├── session_dispatcher.py    # Hermes runs/sessions API client (~150 lines)
│       ├── progress_observer.py     # SSE progress-event consumer → universal audit
│       ├── mcp_servers_config.yaml  # the mcp_servers.manyforge block emitted into Hermes config
│       ├── policy.yaml              # SessionPolicy: compaction=off
│       ├── skill_addendum.md        # tiny — direct-catalog header, memory note
│       └── tests/
│
├── scripts/
│   ├── proxy/vllm-proxy.py          # UNCHANGED — already lane-agnostic
│   ├── debug/
│   │   ├── smoke_corpus.yaml        # UNCHANGED
│   │   ├── smoke_corpus_runner.py   # UNCHANGED
│   │   ├── compare_lanes.py         # NEW — parametric replacement for the 3 A/B harnesses
│   │   └── longitudinal_hermes.py   # NEW — multi-session Hermes harness (Section 9)
│   ├── setup-direct.sh              # NEW — bring up direct lane (was undocumented)
│   ├── setup-openclaw.sh            # was setup-manyforge-assistant.sh — pruned to non-shim concerns
│   ├── setup-hermes.sh              # NEW — bring up Hermes lane
│   └── launch.sh                    # UPDATED — handle ASSISTANT_PROVIDER ∈ {direct, openclaw, hermes}
│
├── policies/
│   ├── manyforge-composer.merged.yaml         # OpenClaw lane — APPLIED at onboarding (endpoints + binaries incl. python3.13)
│   ├── manyforge-composer-hermes.merged.yaml  # Hermes lane — APPLIED at onboarding (Phase 4; same shape, hermes binary)
│   ├── manyforge-egress-shared.yaml           # documentation / source material — NOT applied (see §4.6 rev. 3)
│   ├── manyforge-openclaw.overlay.yaml        # documentation / source material — NOT applied
│   └── manyforge-hermes.overlay.yaml          # documentation / source material — NOT applied
│
└── docs/
    ├── THREE-LANE-MIGRATION-PLAN.md  # this file
    ├── HERMES-MIGRATION-ANALYSIS-2026-06-02.deprecated.md  # rename after Phase 0
    ├── COMPOSER-ASSISTANT-ARCHITECTURE.md  # durable, updated for three-lane
    ├── COMPOSER-ASSISTANT-RUNBOOK.md       # durable, per-lane runbook sections
    ├── LANE-COMPARISON-direct-vs-openclaw.md  # extended to three lanes
    ├── MANYFORGE-MCP-INTEGRATION.md     # durable; add Hermes mcp_servers section + lane-neutral bridge envs
    ├── SMOKE-CORPUS.md                  # durable
    ├── SMOKE-ITER-RUNBOOK.md            # durable
    # (BLOCKER-openclaw-plugin-2026-06-02.md deleted 2026-06-08 — plugin path
    #  superseded by the validated native-discovery lane; rationale retained
    #  in §3/§10 of this doc, artifacts in archive/openclaw-plugin-attempt-2026-06-02/)
```

**Files archived (NOT yet deleted from the live tree — see Phase 0 / Phase 3 gates):**

These move to `archive/openclaw-plugin-attempt-2026-06-02/` AND remain symlinked into the live tree for rollback purposes. Final deletion only happens after Phase 3's OpenClaw-native discovery-surface result is measured and accepted (or after a documented decision to keep both paths permanently).

- `openclaw-plugins/` entire directory
- `openclaw-overrides/*.json`
- `scripts/apply-openclaw-overrides.sh`
- `scripts/build-manyforge-sandbox-image.sh`
- `Dockerfile.manyforge-sandbox{,-prebuilt}`
- `REBUILD-*.md` (already memory-flagged as transient; kept in `docs/` for forensics)

## 8. Migration phases with go/no-go gates

Each phase ends with an explicit pass/fail gate. No phase chains into the next without confirmation.

### Phase 0 — Spike & baseline (1-2 days)

**Why this phase exists.** Establish that the two already-functional lanes (Direct + OpenClaw) still work as expected on the current stack before we refactor anything. Mirrors the shape of Phase 0.5 for Hermes — each lane gets a small dedicated probe set so we know what we're starting from.

**Setup work.** Archive the abandoned plugin attempt to `archive/openclaw-plugin-attempt-2026-06-02/` — files remain symlinked into the live tree pending Phase 3 result. Rename HERMES-MIGRATION-ANALYSIS to `.deprecated.md` (already done). Authorial decision: confirm cosmos-reason2-8b is the production model anchor. Also probe upstream OpenClaw releases — `2026.5.28` is current latest stable; NemoClaw pins `2026.5.22`; note any relevant changelogs that bear on the shim or `extendProvider` API.

**Direct lane probes.** Five probes on cosmos-reason2-8b (~30 min once vLLM is warm):
1. **Smoke baseline.** Run the full 66-case smoke corpus through the direct lane. Record per-case pass/fail and total tokens. PASSES if ≥40/66 (sanity floor) AND no per-case latency > 60s (P95 health).
2. **Proxy mutation path.** Verify the vLLM proxy is in the request path by checking `/tmp/manyforge-assistant-e2e/vllm-proxy.jsonl` is being written. PASSES if log shows the active mutation profile name (the new `MANYFORGE_PROXY_PROFILE`) and at least one of the four mutations fired.
3. **Tool catalog parity.** Confirm the model sees the real manyforge catalog. PASSES if a recorded vLLM request's `tools[]` array contains ≥20 manyforge tool names with `tree_draft_*` / `scene_draft_*` / `program_*` prefixes (not `tool_search_*`).
4. **MCP allowlist enforcement.** Send one request that would invoke a tool NOT in the conversation's `allowedTools` list. PASSES if the bridge returns a `success:false` envelope with `kind=validation_error`, model can read it, conversation continues.
5. **Conversation continuity.** Send a 3-turn conversation. PASSES if `conversationId` is honored across all 3 turns and the program-state snapshot in turn 3 reflects mutations from turn 2.

**OpenClaw lane probes.** Five probes on cosmos-reason2-8b (~45 min, includes vLLM warm + OpenClaw gateway warm):
1. **iter-32 baseline reproduction.** Run the iter-32 chain-on recipe (51/66 production baseline). PASSES if reproduces within ±2 cases.
2. **Discovery surface present.** Confirm tool_search shim is on. PASSES if a recorded vLLM request's `tools[]` array contains exactly `tool_search`, `tool_describe`, `tool_call` (and optionally `tool_search_code` in code mode) and zero `tree_draft_*` / `scene_draft_*` / `program_*` names directly.
3. **`/compact` cadence.** PASSES if the bridge log shows `/compact` firing every 2 user prompts (the iter-32 recipe) AND post-compaction the model can still call `tree_draft_*` tools without re-discovery.
4. **MCP bridge callback shape.** PASSES if a recorded composer callback log entry shows `principal=openclaw-sandbox`, `conversationId=openclaw-...`, `assistantMode=composer-assistant`, and the tool dispatch returned within 5s for read-only tools.
5. **Plugin-path archive accessibility.** PASSES if the archived `apply-openclaw-overrides.sh` still runs to a no-op (already-applied state) from the archive directory — proves the rollback path is intact.

**Gate.** All 10 lane probes PASS, OR specific failures documented with diagnosis (e.g. "probe 3 failed because NemoClaw bumped lkg and dropped /compact"). THREE-LANE plan landed and HERMES doc deprecated; plugin artifacts archived (not deleted). **Documentation deliverable**: `docs/archive/PHASE-0-LANE-BASELINE.md` recording probe results, exact version tuple `(OpenClaw, Hermes, NemoClaw, vLLM, model, proxy-profile)`, and any flagged regressions. **Go = proceed to Phase 0.5. No-go = re-investigate the failed probes before any refactor.**

### Phase 0.5 — Hermes contract spike (1-2 days)

**Why this phase exists.** The reviewer's strongest implementation-time concern: don't wait until Phase 4 to discover that pinned Hermes doesn't behave as the plan assumes. We de-risk by running a tiny end-to-end Hermes probe before the universal-core refactor, so that any surprises (missing config field, different session-API shape, MCP discovery quirk) surface while the codebase is still in its pre-refactor state and easy to back out from.

**Work.**
- Provision a one-off Hermes sandbox (fresh, no production data). Apply `manyforge-egress-shared.yaml` + `manyforge-hermes.overlay.yaml` policy together.
- Inject `mcp_servers.manyforge` config (the snippet from §5.3, hand-written into `/sandbox/.hermes/config.yaml` for this probe — not yet through `hermes-config.ts`).
- Restart Hermes; verify it lists `mcp_manyforge_*` tools via its tool-list API.
- Call one **read-only** manyforge tool (e.g. `program_read`) through the composer's `/api/assistant/bridge/tools/{toolId}` callback path. Confirm the callback fires with the expected envelope and `MANYFORGE_PRINCIPAL=hermes-<sandbox>` is recorded.
- Probe **both** session APIs head-to-head — `POST /api/sessions/{id}/chat` and `POST /v1/runs` — with the same 3-message conversation. Compare: streaming shape, cancellation behavior, session-key headers respected, progress events received. Pick the one for Phase 4 implementation.
- Probe `--tool-call-parser hermes` on vLLM for cosmos-reason2-8b — 5 simple cases — to confirm the wire shape Hermes Agent receives.

**Gate.** All four probes return a recordable answer (not necessarily a happy one — "Hermes doesn't support X yet" is a valid finding that surfaces here, not in Phase 4). **Documentation deliverable**: `docs/archive/PHASE-0.5-HERMES-SPIKE.md` recording per-probe findings, the chosen session API (`/api/sessions/{id}/chat` vs `/v1/runs`), the verified `mcp_manyforge_*` tool count, and any flagged gaps. **Go = proceed to Phase 1, OR raise specific gaps for resolution before Phase 1, OR re-scope Hermes lane based on what we learned (e.g. defer Hermes lane indefinitely while still shipping the universal core + Direct + OpenClaw lanes).**

### Phase 1 — Extract universal core + lane registry (3-4 days)

**Behavior-preserving constraint.** Phase 1 is a refactor and a registry add only. It does NOT change OpenClaw native vs plugin behavior, does NOT change the active skill prompt, does NOT alter compaction policy defaults. The goal is: identical runtime behavior, identical smoke numbers, with the codebase reshaped to enable Phase 3 and Phase 4 work. Rolling Phase 1 back must be a clean `git revert` with no semantic side-effects.

**Specs check (per principle #4).** Before touching `dev_ws/manyforge_assistant_bridge/bridge.py`, `manyforge_composer/backend/assistant_provider.py`, or `manyforge_composer/backend/routes_assistant.py`, read the corresponding section of [`manyforge_specs/`](/home/tndlux/workspaces/dev_ws/src/manyforge_specs/). The `manyforge.assistant.provider_request.v0` envelope, the mode-scoped MCP wrapper contract, and the catalog-hash discipline are specs-driven; the refactor must preserve their semantics. If the refactor needs a spec change (e.g. `LANE_REGISTRY` introducing a new provider-id field), the spec change lands first with explicit justification, then the implementation. **Documentation deliverable** for the specs-check: a `docs/archive/PHASE-1-SPECS-AUDIT.md` listing every spec section consulted and whether it was respected, amended, or extended.

**Work.**
- Create `manyforge/common/` and `manyforge/assistant_session/` packages with the extracted code. Make them pip-installable. Refactor the existing OpenClaw bridge to import from them — zero behavior change.
- Delete the projection mirror in `dev_ws/manyforge_assistant_bridge/bridge.py` and have it import from the new package.
- **Add explicit lane registry to Composer + launcher.** Today Composer's `build_assistant_provider()` ([assistant_provider.py:614](/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_composer/backend/assistant_provider.py#L614)) only knows `nemoclaw`/`openclaw`; the launcher's [assistant.sh:74](/home/tndlux/workspaces/dev_ws/src/manyforge/scripts/lib/assistant.sh#L74) falls back to Direct for unknown providers. Introduce `LANE_REGISTRY` keyed by provider id with `(bridge_endpoint, default_port, transport_class)` entries for `direct`, `openclaw`, and `hermes` (Hermes entry inert until Phase 4 wires it). Eliminates the "unknown provider falls back silently" foot-gun before bridge code moves.
- Rename `OPENCLAW_PROXY_*` env vars to `MANYFORGE_PROXY_*` with deprecated aliases.
- Merge `policies/manyforge-composer.preset.yaml` into `policies/manyforge-composer.merged.yaml` (single applied file containing both endpoints and the binary subject whitelist incl. `/usr/bin/python3.13`). Retain the `egress-shared` + `<lane>.overlay` files as documentation of the conceptual split — per §4.6 rev. 3, `policy-add` does not additively merge same-keyed blocks, so applying them separately drops the endpoints.
  - **Lane-neutralize `manyforge-mcp-bridge.py`.** Five concrete changes (verified against the current script). Reference implementation for the top-level constants:

  ```python
  LANE = (os.environ.get("MANYFORGE_LANE", "openclaw").strip() or "openclaw")
  PRINCIPAL = os.environ.get("MANYFORGE_PRINCIPAL", f"{LANE}-sandbox")
  ASSISTANT_MODE = os.environ.get("MANYFORGE_ASSISTANT_MODE", "composer-assistant")
  CONVERSATION_ID = f"{LANE}-{uuid.uuid4().hex[:12]}"
  # request_id inside the request loop:
  request_id = f"{LANE}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
  ```
  `LANE` is sanitized via `.strip() or "openclaw"` so an empty env doesn't corrupt IDs/log columns. Existing OpenClaw deployments are preserved: with the lane/principal envs unset, `LANE="openclaw"`, `PRINCIPAL="openclaw-sandbox"`, and the two ID prefixes resolve to `"openclaw-..."`.

  Maps to:
  - **Env-readable defaults**: `PRINCIPAL` reads `MANYFORGE_PRINCIPAL` and defaults to `f"{LANE}-sandbox"` so any lane that sets only `MANYFORGE_LANE` gets correct identity/audit attribution for free.
  - **Introduce `MANYFORGE_LANE` env read** at module top (new env, defaults to `"openclaw"`).
  - **Replace hardcoded `CONVERSATION_ID` constant** ([line 115](/home/tndlux/workspaces/dev_ws/src/manyforge/scripts/manyforge-mcp-bridge.py#L115)): currently `f"openclaw-{uuid...}"`; lane-prefix instead.
  - **Replace hardcoded `request_id` prefix** ([line 421](/home/tndlux/workspaces/dev_ws/src/manyforge/scripts/manyforge-mcp-bridge.py#L421)): currently `f"openclaw-{int(time...)}-{uuid...}"`; lane-prefix instead.
  - **Update docstrings** (lines 32, 44, 53): change `default: openclaw-sandbox` → `default: <lane>-sandbox` (or remove the hardcoded value), change `/tmp/manyforge-openclaw-allowed-tools.txt` → `/tmp/manyforge-<lane>-allowed-tools.txt`, and reword `Usage in openclaw.json mcpServers:` to a lane-agnostic phrasing covering both `openclaw.json` and Hermes' `config.yaml mcp_servers:`.
  - **Extend the audit-log row** beyond just `lane`. Today the per-tool-call audit row records timing + tool name + status. Add `lane`, `principal`, `conversationId`, `assistantMode` so the bridge's local log is self-contained for cross-lane debugging — without these, you have to cross-reference Composer's bridge-tools log to attribute a row to a session. All four values are already module-level constants in the bridge; emitting them costs nothing.

  **Compatibility note for the Phase 1 commit message.** While the change is behavior-preserving for tool-call semantics, the audit-log row gains 4 columns and the `CONVERSATION_ID`/`request_id` formats change from the hardcoded `openclaw-...` to lane-derived (still `"openclaw-..."` when `LANE="openclaw"`, so identical for OpenClaw deployments by content, but the format is now a variable). Log-parsing tools that grep for the literal string `"openclaw-"` in the bridge log are unaffected; tools that match on a *position* (e.g. column count) need to know about the four new columns.

**Gate.** OpenClaw lane smoke matches iter-32 baseline within ±2 cases after refactor; direct lane unchanged; zero duplicated projection logic across the two repos (grep proof); `LANE_REGISTRY` accepts `direct`/`openclaw`/`hermes` and rejects unknown ids with a clear error; MCP bridge accepts the new envs without breaking the existing OpenClaw deployment. **Documentation deliverables**: (a) `manyforge/common/README.md` and `manyforge/assistant_session/README.md` describing each new package's contract and intended consumers; (b) [COMPOSER-ASSISTANT-ARCHITECTURE.md](./COMPOSER-ASSISTANT-ARCHITECTURE.md) updated to reflect the three-lane structure + `AssistantTransport` interface; (c) [MANYFORGE-MCP-INTEGRATION.md](./MANYFORGE-MCP-INTEGRATION.md) updated with the lane-neutral bridge envs (`MANYFORGE_LANE` etc.) and the audit-row schema; (d) `setup-openclaw.sh` renamed from `setup-manyforge-assistant.sh` with header pointing at the new docs. **Go = proceed to Phase 2.**

### Phase 2 — Direct lane on the shared core (1 day)

**Specs check (per principle #4).** Moving `manyforge_assistant_bridge/` out of `dev_ws/manyforge/` is potentially a cross-workspace contract change. Before the move, read [`manyforge_specs/docs/cross-workspace-conventions.md`](/home/tndlux/workspaces/dev_ws/src/manyforge_specs/docs/cross-workspace-conventions.md) and any `manyforge_specs` section that references the direct bridge's location, port, or supervisor. If the move requires a spec amendment (likely yes — the bridge's home directory is a cross-workspace concern), the amendment must land first.

**Work.** Move `manyforge_assistant_bridge/` from dev_ws/manyforge to `manyforge/lanes/direct/` (or formalize the cross-repo import — TBD per [Open question Q1](#12-open-questions)). Implement the `AssistantTransport` interface for the direct lane. Update launcher.

**Gate.** Direct lane smoke ≥ pre-refactor direct-lane number on cosmos-reason2-8b. **Documentation deliverables**: (a) `setup-direct.sh` script that takes a clean host to a working direct lane (no manual steps); (b) new "Direct lane bring-up" section in [COMPOSER-ASSISTANT-RUNBOOK.md](./COMPOSER-ASSISTANT-RUNBOOK.md) covering start/stop/smoke; (c) [LANE-COMPARISON-direct-vs-openclaw.md](./LANE-COMPARISON-direct-vs-openclaw.md) renamed to `LANE-COMPARISON-three-lane.md` with a Direct-lane numbers row added (and Hermes row reserved as `TBD Phase 4`). **Go = proceed to Phase 3.**

### Phase 3 — OpenClaw skill rewrite for discovery surface (3-4 days)

**Work.** Rewrite the manyforge skill addendum for the discovery protocol. Run smoke corpus through OpenClaw lane on the native discovery path (no plugin). Tune skill until pass rate is close to iter-32 baseline — this is the "embrace the shim, accept the overhead, see how close we get" test. Measure: per-turn pass rate, latency, average round-trips per real tool call, total tokens consumed.

**Gate.** OpenClaw native lane achieves **≥46/66 (≈70%)** on a clean discovery-surface run — within ~5 cases of the iter-32 51/66 baseline. If yes, the archived plugin artifacts can be deleted in Phase 5. If no, **document the gap, keep the archived artifacts available as a production rollback path**, and proceed to Phase 4 either way — the architectural shape doesn't change, only the production routing default does. **Documentation deliverables**: (a) `manyforge/lanes/openclaw/skill_addendum.md` (the final tuned discovery-protocol skill prompt that achieved the result); (b) new or updated "OpenClaw lane bring-up" section in [COMPOSER-ASSISTANT-RUNBOOK.md](./COMPOSER-ASSISTANT-RUNBOOK.md); (c) `LANE-COMPARISON-three-lane.md` OpenClaw row populated with `(pass-rate, P50/P95 latency, avg round-trips per real tool call, tokens consumed)`; (d) updated `docs/PHASE-3-OPENCLAW-NATIVE-RESULT.md` recording the iteration history of skill-prompt tunings and final numbers.

### Phase 4 — Hermes lane bring-up (5-7 days)

**Work.**
- Spin up Hermes in a fresh sandbox. Apply `policies/manyforge-composer-hermes.merged.yaml` (single merged file — see §4.6 rev. 3). Provision `API_SERVER_KEY` via NemoClaw secret-store path (TBD Q6).
- Emit `mcp_servers.manyforge` into Hermes config (fork or NemoClaw-Thor overlay on `hermes-config.ts`). Confirm Hermes discovers manyforge MCP tools at startup (Hermes will prefix them as `mcp_manyforge_<tool>`).
- **Probe**: confirm `--tool-call-parser hermes` on vLLM 0.x produces OpenAI structured `tool_calls[]` for cosmos-reason2-8b — 5-case probe (open question Q3). If not, fall back to default parser + verify Hermes still consumes the catalog correctly.
- **Probe**: decide between `/api/sessions/{id}/chat` and `/v1/runs` based on streaming + cancellation + session-key behavior — 3-case probe.
- Wire the bridge on `:8300`. Implement `HermesTransport` against the chosen session API. Implement the SSE progress-event observer that emits universal audit entries.
- Run per-turn smoke corpus through Hermes lane with memory disabled for parity comparison against the other two lanes.
- Run the **longitudinal harness** (Section 9) with memory + skills + cron + todo + delegation enabled — the Hermes-native bake-off.

**Gate.** Per-turn smoke ≥40/66 with memory disabled (Hermes is not optimized for stateless turns; sanity floor only). Longitudinal harness shows measurable session-over-session improvement OR explicit "no improvement" finding documented with diagnosis. **Documentation deliverables**: (a) `setup-hermes.sh` script that takes a clean host to a working Hermes lane (provisions sandbox, applies `manyforge-composer-hermes.merged.yaml` with post-apply verification, emits `mcp_servers.manyforge`, sets `API_SERVER_KEY`); (b) new "Hermes lane bring-up" section in [COMPOSER-ASSISTANT-RUNBOOK.md](./COMPOSER-ASSISTANT-RUNBOOK.md); (c) new "Hermes mcp_servers" section in [MANYFORGE-MCP-INTEGRATION.md](./MANYFORGE-MCP-INTEGRATION.md) covering the config emission strategy chosen for Q7 (fork / overlay / direct config write); (d) `LANE-COMPARISON-three-lane.md` Hermes row populated with both per-turn (memory off) and longitudinal (memory on) numbers; (e) `docs/PHASE-4-HERMES-LONGITUDINAL.md` documenting the longitudinal corpus design + measured skill emergences, memory hit-rates, and turns-to-completion across the session sequence. **Go = proceed to Phase 5.**

### Phase 5 — Lane routing + production decision (2-3 days)

**Specs check (per principle #4).** The composer-side lane router (Section 9) introduces a new `lane_routing.yaml` config and a new dispatch path through `routes_assistant.py`. Before implementing, read `manyforge_specs/` sections on Composer's assistant-mode contract, the provider-id discipline, and any spec describing how Composer chooses between providers today. If the router design conflicts with a spec, escalate before coding.

**Work.** Implement the composer-side lane router (Section 9). Pick a production default based on numbers from Phase 3 + Phase 4. If Phase 3 didn't pass and the archived plugin path is acceptable as rollback, keep it as the OpenClaw lane's plugin-mode (feature-flagged via `OPENCLAW_LANE_MODE=plugin|native`). Ship.

**Gate.** Lane router routes correctly on a 20-case manual probe; user-override works; rollback flag flips production back to a known-good lane in <60 seconds. **Documentation deliverables**: (a) `lane_routing.yaml` shipped with explanatory header + production default recorded; (b) "Lane routing" section in [COMPOSER-ASSISTANT-RUNBOOK.md](./COMPOSER-ASSISTANT-RUNBOOK.md) covering rollback playbook + `OPENCLAW_LANE_MODE=plugin|native` feature flag semantics if Phase 3 didn't fully pass; (c) `docs/PHASE-5-PRODUCTION-DECISION.md` recording the chosen default, the empirical basis (Phase 3 + Phase 4 numbers), the version tuple `(OpenClaw, Hermes, NemoClaw, vLLM, model, proxy-profile)`, and the next re-baseline trigger. **Go = production rollout.**

**Total time: 17-23 working days.** Phases are sequential; no parallelism assumed. (Phase 0 expanded to 1-2 days for per-lane probes; Phase 0.5 added 1-2 days for the Hermes spike. Every phase has documentation as an explicit deliverable, not an afterthought.)

## 9. Observability, benchmarking, and lane routing

### 9.1 Universal audit format

Already described in 4.7. The bake-off scripts ingest these three JSONL streams + the shared vLLM proxy log. The proxy log is the single source of truth for what reached the model on each lane — same proxy, same shape, lane is inferable from `requestId` namespace.

### 9.2 Per-turn bake-off

`scripts/debug/compare_lanes.py --lanes direct,openclaw,hermes --corpus smoke_corpus.yaml --model cosmos-reason2-8b`. Runs the 66-case corpus through each lane with `--reset-hermes-state` for parity. Reports: per-case pass/fail per lane, latency P50/P95, average round-trips per real tool call, total tokens consumed. This is the deterministic-comparison surface.

### 9.3 Longitudinal Hermes harness

`scripts/debug/longitudinal_hermes.py --sessions 10 --turns-per-session 8 --corpus longitudinal_corpus.yaml`. A new corpus designed for session compounding — repeated patterns ("wrap pick_and_place in retry-3 every time"), preference accumulation, scheduled-by-cron checks. Reports: skill emergences per session, memory hit-rate, average turns-to-task-completion across the session sequence. The metric Hermes is allowed to win on.

### 9.4 Composer-side lane router

The composer keeps a `lane_routing.yaml`:

```yaml
default_lane: openclaw    # set by bake-off result
overrides:
  - match: { route: "/chat", user_pinned: "direct" }
    lane: direct
  - match: { request_shape: "long_running" }
    lane: hermes
  - match: { time_budget_ms: { lt: 5000 } }
    lane: direct
```

Composer's assistant route consumes this and dispatches accordingly. User can pin a lane per-conversation. The routing layer is the place where the three lanes' differing natures show up as real product behavior.

## 10. Version pin policy

The OpenClaw 2026.4.24→2026.5.22 surprise that started this whole rebuild is the worked example for why we need a pin policy.

- **OpenClaw**: track NemoClaw's `lkg` pin (currently `2026.5.22`). Upstream-latest stable at time of writing is `2026.5.28`; latest beta is `2026.6.1-beta.2` (npm `dist-tags.beta`). The full in-flight series spans `2026.5.28-beta.{1,3,4}` → `2026.5.30-beta.1` → `2026.5.31-beta.{1..4}` → `2026.6.1-beta.{1,2}`. Probe upstream changelogs on each NemoClaw bump for relevant shim or `extendProvider` changes — given the velocity, NemoClaw's `lkg` will likely bump within the Phase 3 window. Re-baseline the smoke corpus when NemoClaw bumps the pin. Document the version in every smoke report.
- **Hermes**: track NemoClaw's `lkg` pin (currently `2026.5.16` / upstream `v0.14.0`). Upstream-latest is `0.15.2`. The `mcp_servers` schema has been stable across 0.14.x → 0.15.x per upstream docs, so a bump is likely low-risk for our integration — but re-baseline anyway.
- **NemoClaw itself**: follow `lkg` for the public installer path. If `lkg` and `v0.0.<latest>` diverge, prefer `lkg` for production sandboxes until both are baselined together.
- **vLLM**: stays on whatever the manyforge profile pins (currently per-model). Independent of agent-lane choice.
- **Bake-off baselines must record the (OpenClaw, Hermes, NemoClaw, vLLM, model, proxy-mutation-profile) tuple.** Reports without that tuple are not comparable across time.

## 11. Risk and rollback

### Per-phase rollback

| Phase | What can break | Rollback |
|---|---|---|
| 0 | None — read-only spike | n/a |
| 1 | Refactor introduces regression in OpenClaw lane | `git revert` the refactor commits; both bridges fall back to the pre-refactor adapter.py |
| 2 | Direct lane breaks | Revert to dev_ws/manyforge_assistant_bridge code path; lane router pins to OpenClaw |
| 3 | Skill rewrite degrades OpenClaw lane | Keep skill addendum behind a feature flag; revert flag |
| 4 | Hermes lane fails to bring up OR breaks sandbox | Lane router excludes Hermes; bridge supervisor doesn't start it |
| 5 | Lane router misroutes | `default_lane: openclaw` env-var override; user pin always wins |

### Cross-cutting risks

- **Hermes MCP discovery drift.** If a manyforge MCP tool's schema changes, Hermes re-registers it on the next config-watcher cycle (`cli.py:9314+` watches `mcp_servers` section mtime). Mitigation: trigger a config touch on every manyforge-mcp-bridge restart; smoke parity test catches schema drift; the `mcp_manyforge_<tool>` prefix in audit logs makes drift visible per-tool.
- **Hermes memory poisoning.** A run that fails messily can leave Hermes memory in a bad state. Mitigation: `--reset-hermes-state` flag in the runner; weekly cron cleanup of `/sandbox/.hermes/memories` if the longitudinal harness shows degradation.
- **Composer provider-id explosion.** Adding `hermes` as a third provider id means the composer's `build_assistant_provider()` and `routes_assistant.py` mode handling get a third branch. Risk of branch drift. Mitigation: the three branches share the same `AssistantTransport` consumer-side interface; composer doesn't care which lane returned the envelope.

## 12. Open questions

These are real decision gates, not vague worries. Each blocks a specific phase.

1. **(Phase 2)** Do we move `manyforge_assistant_bridge` from `dev_ws/src/manyforge/` to `NemoClaw-Thor/manyforge/lanes/direct/`, or do both repos depend on the new `manyforge_assistant_common` package via PyPI / file path? Affects deploy story.
2. **(Phase 3)** Does the OpenClaw discovery surface achieve ≥46/66 cases (≥70%) with a well-tuned skill prompt, or does the 3-trip overhead fundamentally cap accuracy lower? Empirical question; Phase 3 is the answer.
3. **(Phase 4)** Does `--tool-call-parser hermes` on vLLM 0.x produce the OpenAI structured `tool_calls[]` that Hermes Agent expects, when the model is cosmos-reason2-8b (not a Hermes-trained model)? Needs a 5-case probe before MCP-config emission.
4. **(Phase 4)** Which Hermes session API is the right one for our needs — `/api/sessions/{id}/chat` or `/v1/runs`? Differs on streaming semantics, cancellation, and how progress events are emitted. Needs a 3-case probe.
5. **(Phase 4)** What's the right Hermes session lifecycle for a composer conversation? Per-conversation session, per-turn session, or one long session per user? Affects how memory accumulates.
6. **(Phase 4)** How is `API_SERVER_KEY` issued and rotated? Manual env var, NemoClaw secret store, or generated per sandbox?
7. **(Phase 4)** Where do we patch `hermes-config.ts` to emit `mcp_servers.manyforge`? Options: (a) fork in NemoClaw-Thor and ship as an overlay, (b) upstream PR to NemoClaw to add `extraMcpServers` config field, (c) write the `config.yaml` directly post-onboard and rely on Hermes' auto-reload. Affects maintenance burden vs upstream-friendliness.
8. **(Phase 5)** Default lane decision — based on per-turn comparison alone, longitudinal Hermes alone, or a weighted combination? Affects the production target metric.
9. **(Phase 5)** Composer UI for user-pin: a dropdown, a chat command, both? Affects product surface.

## 13. What replaces the prior hermes-migration analysis

The prior doc's binary plugin-vs-Hermes framing was the right framing for the rebuild week but is the wrong framing for a durable architecture. Specifically:

- The "asymmetric: plugin now, Hermes later" recommendation is **dropped**. Path A (the OpenClaw plugin) is dropped entirely — we abandoned it after confirming the bundled `nemoclaw` plugin owns the `inference` provider unconditionally and OpenClaw exposes no `extendProvider` API. Path C (rewrite the skill for the discovery API) was dismissed there as institutionalized degradation; here it is the deliberate design for the OpenClaw lane.
- The "1-4 weeks plugin / 3-6 months Hermes" timeline is **dropped**. Replaced by the 17-23 working day phased plan above (Phase 0 has 10 per-lane probes for Direct + OpenClaw; Phase 0.5 has 5 Hermes contract probes; every phase has documentation as an explicit deliverable).
- The yardstick is anchored on **cosmos-reason2-8b iter-32 (51/66, 77.3%)**, not the Qwen3.6-35B 84.8% pre-rebuild number. The 35B number stays as a peer reference.
- The bridge port `:8300` for Hermes is **kept** (the choice was unobjectionable; sticking with it avoids re-renaming).
- The eight open questions about Hermes are **preserved and expanded to nine** (one new question on where to patch `hermes-config.ts` to emit `mcp_servers`) as Phase 4 gates above.

Everything else is superseded.

---

**Authoring note.** This plan was drafted after a four-agent parallel inventory pass (HERMES-doc context, NemoClaw-Thor structure, Hermes agent, OpenClaw shim mechanism). All claims about file paths, line numbers, OpenClaw internals, and Hermes wire format are verifiable from the inventory transcripts; please file an issue if any of them have rotted.
