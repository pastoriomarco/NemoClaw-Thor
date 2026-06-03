# ManyForge AI Assistant — Architecture Reference (Thor)

Canonical reference for the **ManyForge AI assistant pipeline running on Jetson
AGX Thor**. Target audience: a human operator or LLM agent that has never seen
this stack but needs to operate, debug, and tune it. This is the entry point
for architectural questions; runbooks, lane comparisons, and MCP integration
notes are referenced from here.

Cross-references:
- Operational gates + per-symptom debugging: [COMPOSER-ASSISTANT-RUNBOOK.md](./COMPOSER-ASSISTANT-RUNBOOK.md)
- Smoke corpus mechanics + per-iter history: [SMOKE-CORPUS.md](./SMOKE-CORPUS.md)
- Cold-start order of operations: [SMOKE-ITER-RUNBOOK.md](./SMOKE-ITER-RUNBOOK.md)
- Lane-parity benchmark (why OpenClaw + Cosmos-8B is the default): [LANE-COMPARISON-direct-vs-openclaw.md](./LANE-COMPARISON-direct-vs-openclaw.md)
- MCP integration deep-dive: [MANYFORGE-MCP-INTEGRATION.md](./MANYFORGE-MCP-INTEGRATION.md)
- Profile calibration methodology: [MANYFORGE-PROFILE-CALIBRATION.md](./MANYFORGE-PROFILE-CALIBRATION.md)
- Deployment plan: [MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md](./MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md)

---

## Recent changes (2026-05-31 / 2026-06-01)

The pipeline accumulated several behavior-changing patches in late May / early
June 2026. The list below is the rapid-reference; each item is fully described
later in the doc.

- **Synthetic clarification inverted to default-OFF (2026-06-01).** Was
  hard-coded ON before. Opt-in via `OPENCLAW_ASSISTANT_ENABLE_SYNTHETIC=1`;
  legacy `OPENCLAW_ASSISTANT_DISABLE_SYNTHETIC=1` preserved as the back-compat
  off-switch. The bypass was hiding model-specific ability to ask clarification
  on `add a <kind>` patterns; default-off makes the smoke corpus a fair
  comparator across models. Implementation: `service.py:397-407`. See §B.3 +
  §D.4.
- **CONSECUTIVE (not LIFETIME) tail-run loop counting (2026-06-01).** The
  proxy's `vllm-proxy.py:403-442` now counts run-length at the tail of the
  assistant-turn sequence — any different tool resets the run to 1, a no-tool
  assistant turn resets it to 0. Lifetime counting was the major bug that
  hard-stopped every chained smoke case (PnP_01..PnP_20) after the 3rd or 4th
  retry because the shared `conversationId` history accumulated faster than
  the case-local loop length. Documented in §D.1.
- **Cascading loop defense (round 10, 2026-06-01).** Reflection-injection
  at `_REFLECT_AT=4` + hard stop at `_STOP_AT=8`. Two-stage: the model gets
  exactly one shot to act on the reflection prompt before being cut off. Pairs
  with the consecutive counting fix above. Documented in §D.1.
- **Rule 11 NO_REPLY guard + Rule 11a Missing-WHERE clause (2026-06-01).**
  `adapter.py:513-536`. Rule 11 explicitly forbids the model from replying the
  literal `NO_REPLY` string on action-shaped prompts; 11a names
  `PARALLEL_generic` / `FALLBACK_generic` / `UPDATE_params_generic` as the
  patterns that MUST elicit a clarification question instead of a tool call.
  This is the prompt-side replacement for the now-default-off synthetic
  clarification bypass. Documented in §B.2-prompt and §D.4.
- **Per-profile proxy tuning (2026-06-01).** `serving/config.sh` carries
  `THOR_TARGET_PROXY_LOOP_REFLECT_AT` / `_STOP_AT` / `_FORCE_ENABLE_THINKING`
  per profile case branch (5 active branches);
  `serving/start-model.sh:56-95` auto-restarts the local vllm-proxy with
  those values whenever a new profile boots. Opt out with
  `THOR_RESTART_PROXY=0` (e.g. when a separate supervisor owns the proxy).
  Documented in §B.1.
- **Proxy top-level enable_thinking mirror (2026-05-31).**
  `vllm-proxy.py:285-292`. When `OPENCLAW_PROXY_FORCE_ENABLE_THINKING` is
  set, the proxy writes the target value to BOTH
  `chat_template_kwargs.enable_thinking` AND the top-level `enable_thinking`
  field. vLLM treats the top-level as source of truth and ignores
  `chat_template_kwargs` when both are present; without the mirror, Composer's
  top-level `enable_thinking:false` silently cancels every profile's
  thinking-on default. Now fed automatically from the per-profile
  `THOR_TARGET_PROXY_FORCE_ENABLE_THINKING` via `start-model.sh`. See §C.
- **Schema fix: `node.name` accepted in tree_draft_insert_node (2026-06-01).**
  `dev_ws/src/manyforge/manyforge_composer/backend/assistant_tool_schemas.py:488-528`.
  The `_TREE_NODE_PAYLOAD_FOR_INSERT_SCHEMA` now accepts an optional
  `node.name` as a duplicate of the top-level `nodeName` argument
  (mirroring the older `_TREE_NODE_SCHEMA` shape used by `tree_draft_wrap_node`).
  Handler still reads from the top-level `nodeName`; `node.name` is
  accepted-and-ignored. Stops validator death spirals when models
  emit the legacy duplicated shape. See §B-Composer.
- **Smoke runner `alt_names` support (2026-06-01).**
  `smoke_corpus_runner.py:391-424`. Corpus cases may declare an `alt_names`
  list for tools with equivalent effect (e.g. `scene_draft_upsert_objects`
  as an alt for `scene_draft_add_object`). When the primary `name` is
  missing but an alt fired with 2xx, the case is recorded as `soft-pass`
  instead of failing. Documented in §B.5 and §F.7.
- **Smoke corpus rebalance (2026-06-01).** 4 cases now expect ASK
  (clarification, no tool emission) rather than a tool call:
  `PARALLEL_generic`, `FALLBACK_generic`, plus 2 cases labeled
  `category: clarification` (smoke_corpus.yaml lines 1170, 1196, 1214).
  These rebalances are tied to the default-synthetic-off measurement design —
  they probe whether the model itself asks (via Rule 11a) without the
  bridge bypass.

---

## TL;DR

- A user message in **Composer** (FastAPI + React UI on `:9000`) traverses
  five hops on the way to vLLM and back: **Composer → bridge → OpenClaw
  gateway → vllm-proxy → vLLM**. Each hop is a separate process; each can
  inject, strip, or mutate fields independently.
- **Two assistant lanes** exist; selection is one env var
  (`ASSISTANT_PROVIDER=openclaw|nemoclaw`) and the deployment YAML's
  `base_url`. **OpenClaw is the production default since 2026-05-07**
  ([`LANE-COMPARISON-direct-vs-openclaw.md` §8](./LANE-COMPARISON-direct-vs-openclaw.md)).
- The **vllm-proxy** sits in front of vLLM as a logger/mutator. Even with
  zero mutation env vars set, it logs every request/response to JSONL; with
  env vars set, it injects caps, budgets, tool_choice, user-suffixes, and a
  cascading **same-tool-loop defense** (reflection at 4 calls, hard stop
  at 8).
- The **production knobs that matter most**, ranked:
  1. `--default-chat-template-kwargs '{"enable_thinking":…}'` per profile in
     `serving/launch.sh` (and the proxy's top-level mirror — see §C).
  2. `OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS=2048` (load-bearing — without it,
     thinking-on generations are unbounded).
  3. `OPENCLAW_PROXY_THINKING_TOKEN_BUDGET=512` (caps the `<think>`
     envelope; Qwen3-VL sweet-spot).
  4. `OPENCLAW_ASSISTANT_COMPACT_EVERY_N=2` (bridge-fired `/compact` keeps
     chain-on sessions from overflowing).
  5. `OPENCLAW_PROXY_LOOP_REFLECT_AT=4` / `OPENCLAW_PROXY_LOOP_STOP_AT=8`
     (round-10 cascading loop defense, **CONSECUTIVE** counting since
     2026-06-01).
- **Synthetic clarification is OFF by default since 2026-06-01.** The
  bypass that used to short-circuit `add a <kind>` prompts is now an
  opt-in (`OPENCLAW_ASSISTANT_ENABLE_SYNTHETIC=1`). The prompt-side
  replacement is Rule 11a (Missing-WHERE) in `adapter.py:519-536`, which
  instructs the model to ask "which parent and where in its children?"
  for those patterns without bypass.
- **Production default**: `cosmos-reason2-8b` (8B Qwen3-VL VLM, FP8 KV,
  hermes tool parser, qwen3 reasoning parser, thinking-on, 256K ctx).

---

## A. End-to-end request flow

### A.1 The hops, named

```
                                  ┌─────────────────────────────┐
                                  │ User in Composer UI         │
                                  │ (React, browser)            │
                                  └────────┬────────────────────┘
                                           │ HTTP POST /api/assistant/chat
                                           ▼
┌─────────────────────────────── Thor host ─────────────────────────────────┐
│                                                                            │
│  ┌────────────────────────────── Hop 1 ──────────────────────────────┐    │
│  │  manyforge-e2e-composer (Docker container, :9000)                  │    │
│  │  routes_assistant.chat → builds manyforge.assistant.provider_      │    │
│  │  request.v0 envelope (catalog snapshot, scene snapshot, tools,     │    │
│  │  mode allowlist, session key, top-level enable_thinking=false)     │    │
│  └──────────────────┬─────────────────────────────────────────────────┘    │
│                     │ HTTP POST  base_url from deployment YAML              │
│                     │ openclaw lane → http://127.0.0.1:8200/v1/manyforge/…  │
│                     │ direct  lane → http://127.0.0.1:8100/v1/manyforge/…  │
│                     ▼                                                       │
│  ┌────────────────────────── Hop 2 (lane-specific) ──────────────────┐    │
│  │  openclaw_assistant_bridge :8200  (FastAPI, this repo)             │    │
│  │  ─ synthetic clarif for "add a <kind>" — DEFAULT OFF since         │    │
│  │      2026-06-01 (opt-in: OPENCLAW_ASSISTANT_ENABLE_SYNTHETIC=1)    │    │
│  │  ─ Rule 11 NO_REPLY guard + 11a Missing-WHERE injected into RULES  │    │
│  │      block (adapter.py:513-536) — prompt-side replacement          │    │
│  │  ─ detects 5+ same-tool calls across turns → synthetic stop (r 8)  │    │
│  │  ─ fires /compact every Nth user prompt on this session key        │    │
│  │  ─ builds OpenClaw agent invocation (gateway mode = HTTP)          │    │
│  │  ─ derives session key = conversationId + catalogHash + progRev    │    │
│  │  ─                                                                 │    │
│  │  manyforge_assistant_bridge :8100  (FastAPI, sibling manyforge)    │    │
│  │  ─ runs its OWN agent loop in-process (no OpenClaw)                │    │
│  │  ─ tool_choice pin, inline-snapshot context for compound prompts   │    │
│  └──────────────────┬─────────────────────────────────────────────────┘    │
│                     │ HTTP POST to in-sandbox OpenClaw gateway              │
│                     │ host:18789 (port-forward → SSH netns inside pod)     │
│                     ▼                                                       │
│  ┌────────────────────────────── Hop 3 ──────────────────────────────┐    │
│  │  OpenClaw gateway (in `my-assistant` sandbox, SSH netns, :18789)   │    │
│  │  ─ runs the multi-turn LLM agent loop server-side                  │    │
│  │  ─ up to 3 concurrent subagents (per `agents.defaults.maxConcurr…`)│    │
│  │  ─ MCP wrapper subprocess registered at gateway boot, holds        │    │
│  │    the manyforge tool catalog (tools/list)                         │    │
│  │  ─ each turn emits an OpenAI-compatible /v1/chat/completions       │    │
│  │  ─ forwards `x-openclaw-session-key` so server-side cache hits     │    │
│  └──────────────────┬─────────────────────────────────────────────────┘    │
│                     │ HTTP POST /v1/chat/completions                        │
│                     │ → host.openshell.internal:8000                        │
│                     ▼                                                       │
│  ┌────────────────────────────── Hop 4 ──────────────────────────────┐    │
│  │  vllm-proxy :8000  (single-process Python HTTP reverse proxy)      │    │
│  │  ─ LOGS every req/resp to JSONL (always-on, never disabled)        │    │
│  │  ─ injects max_tokens, thinking_token_budget, enable_thinking…     │    │
│  │  ─ mirrors top-level enable_thinking to chat_template_kwargs (r10) │    │
│  │  ─ cascading loop defense (round 10, 2026-06-01):                  │    │
│  │      4 CONSECUTIVE same-tool calls → injects reflection prompt     │    │
│  │      8 CONSECUTIVE same-tool calls → synthesizes SSE assistant stop│    │
│  │      (consecutive ≠ lifetime — different tool or no-tool resets)   │    │
│  │  ─ 200s per-request socket timeout (fails before smoke's 244s)     │    │
│  └──────────────────┬─────────────────────────────────────────────────┘    │
│                     │ forward to upstream                                   │
│                     ▼                                                       │
│  ┌────────────────────────────── Hop 5 ──────────────────────────────┐    │
│  │  vLLM container :8050  (`manyforge-e2e-vllm`, served via Docker)   │    │
│  │  ─ runs the model (cosmos-reason2-8b by default)                   │    │
│  │  ─ flags from `serving/launch.sh` per-profile case branch          │    │
│  │  ─ per-profile: tool-call parser, reasoning parser, kv dtype,      │    │
│  │    moe backend, default chat_template_kwargs, quantization, …     │    │
│  │  ─ exposes /v1/chat/completions, /v1/models, /metrics              │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────────┘

                       ⇡ Tool-call return path (Hops 6-10) ⇡

      vLLM → gateway: response with `tool_calls[]`
      gateway → MCP wrapper subprocess: stdio JSON-RPC tools/call
      MCP wrapper → HTTP through OpenShell egress proxy (10.200.0.1:3128)
      egress proxy → Composer /api/assistant/bridge/tools/<toolId>
      Composer runs the actual ManyForge tool, returns the result
      result back through reverse path, next vLLM turn …
      final assistant message → bridge → Composer → UI
```

### A.2 What each hop modifies vs strips vs preserves

| Hop | Owns | Always modifies | Strips | Never touches |
|-----|------|-----------------|--------|----------------|
| **1. Composer backend** | request shape | adds `nodeCatalog`, `programSnapshot`, `sceneSnapshot`, mode allowlist, principal binding | request body parts not in envelope | per-request sampling fields (always omitted; vLLM owns them) |
| **2. Bridge (openclaw)** | session key + compaction + loop short-circuits | adds `prompt` preamble that mirrors gateway-internal MCP catalog | per-request `tool_choice`, `temperature`, `top_k`, `top_p` (None → omit) | top-level `enable_thinking` (passed through if Composer sets it) |
| **2. Bridge (direct)** | agent loop in-process | adds `tools[]`, `tool_choice`, inline scene snapshot | nothing of note | nothing |
| **3. OpenClaw gateway** | per-turn LLM loop, MCP dispatch, subagent fan-out | adds `x-openclaw-session-key`, `tools[]`, transformed history | empty CoT envelopes (depending on parser) | streaming SSE shape |
| **4. vllm-proxy** | logging + opt-in mutation | injects `max_tokens`, `thinking_token_budget`, etc. when env set; reflection user message at 4 same-tool calls | nothing (proxy preserves all headers) | `messages[]` ordering, system prompt |
| **5. vLLM** | inference | applies chat template + parsers; emits `reasoning` / `content` / `tool_calls` | nothing (it's the producer) | n/a |

**Key gotchas** to keep in mind when reading the code:

- The **direct lane** runs the agent loop in the bridge (`manyforge_assistant_bridge/bridge.py`) and emits N chat-completions per Composer prompt — every per-turn call passes through hop 4 (vllm-proxy).
- The **OpenClaw lane** runs the agent loop **server-side in OpenClaw**. The bridge sees one call per Composer prompt; the per-turn traffic between gateway and vLLM is invisible to the bridge but still passes through the vllm-proxy.
- **Composer always sends top-level `enable_thinking: False`** in its provider envelope (see `manyforge_composer/backend/assistant_provider.py`), regardless of the model's chat template. vLLM treats the top-level field as the source of truth and ignores `chat_template_kwargs.enable_thinking` when both are present. The proxy mirrors the chat_template_kwargs decision back to the top level when `OPENCLAW_PROXY_FORCE_ENABLE_THINKING` is set — without that mirror, thinking-on profiles silently revert to thinking-off (observed empirically on cosmos-reason2-8b 2026-05-31).
- The bridge's **session key** is `derive_gateway_session_key(payload)` = `conversationId + catalogHash + programRevision`. All bridge-side counters (compact counter, loop detector) are keyed by this. Catalog rotation = new session = counters reset.

---

## B. Configurable knobs, by impact

Knobs are grouped by **owning component**. Within each group they are listed in **rough order of operational impact** for the current production stack.

### B.1 vLLM launch flags (per-profile in `serving/launch.sh`)

Each profile is a case branch in `serving/launch.sh`. Adding a profile requires
matching edits to `serving/config.sh` (sizing) AND `serving/launch.sh` (vLLM
args). The slug must be identical between the two files; it is also the
`served-model-name` advertised by vLLM.

| Flag | Default for cosmos-reason2-8b | What it controls | When to change |
|------|-------------------------------|------------------|---------------|
| `--tool-call-parser` | `hermes` | Parser that extracts `tool_calls[]` from raw decode output. Must match the chat-template's tool emission format. | Switch to `qwen3_coder` for Qwen3.6-family or Nemotron-Omni (XML tool calls); `qwen3_xml` for stock Qwen3.6 template; `gemma4` for Gemma-4. Wrong choice → tool_calls always empty → infinite agent loop. |
| `--reasoning-parser` | `qwen3` | Parser that extracts the `<think>…</think>` envelope into the `reasoning` field. | Match to the model family. Empty (no flag) lets thinking bleed into `content`. **The bridges only consume `choices[0].message.content`** — if you use a reasoning parser, thinking is hidden from the bridge. Often that is what you want (clean tool-call extraction); for the omni instruct profile we explicitly drop the parser so content carries everything. |
| `--default-chat-template-kwargs` | `{"enable_thinking":true}` | Default value of `chat_template_kwargs.enable_thinking` when the caller omits it. | This is the **server-side default** — clients override per-request. See §C for the full thinking subsystem. |
| `--override-generation-config` | `{"temperature":0.2,"top_p":0.95}` | Server-wide sampling defaults. Caller-supplied values still win. | Match to the model card's vendor recipe; the historical regression cases for omni and 35B were caused by greedy (`top_k=1` / `temperature=0`) sampling on thinking-off, which produces null-arg tool-call loops. |
| `--moe-backend` | _(unset, default oracle)_ | MoE kernel selection. On SM110 NVFP4 the oracle picks `FLASHINFER_TRTLLM` or `FLASHINFER_CUTEDSL`; `triton` and `flashinfer_cutlass` are rejected (verified 2026-06-01). For NVIDIA W4A16 quant we force `marlin` because CUTEDSL requires full-NVFP4 activations. | Only override when a profile comment explicitly says so. **SM110 has limited support** — verify with vLLM oracle logs before relying on a hand-picked backend. |
| `--kv-cache-dtype` | `fp8` | KV-cache precision. `fp8` halves footprint vs `auto` (bf16) at a small quality cost; `auto` for highest fidelity. | Stay on `fp8` for all production profiles on Thor (KV pool is the binding constraint). Bump to `auto` only on a sizing experiment. |
| `--quantization` | _(unset, weights are pre-quantized)_ | Forces a specific quant scheme for runtime. `modelopt` for NVIDIA ModelOpt-quantized weights; `nvfp4` is implied by file format on most quants. | Set when the model card requires it (e.g. NVIDIA Qwen3.6-35B-A3B-NVFP4 needs `modelopt`). |
| `--enable-prefix-caching` | _(on by default for 35B / Gemma / Nemotron, off for Cosmos 2B/8B)_ | Server-side prefix-cache for shared turn prefixes. | Leave alone unless you measure a regression — turning it on for chat workloads is nearly free. |
| `--enable-chunked-prefill` | _(varies)_ | Splits prefill across decode steps. | Required for some MoE profiles (35B) to avoid scheduler stalls. Profile comments call this out. |
| `--enforce-eager` | _(on for cosmos profiles + 35B)_ | Disables CUDA graphs. Costs 10-20% throughput in steady state but bounds first-token latency. | Default on for VLM profiles where the ViT path doesn't graph cleanly. Turn off only after measuring a clean CUDA-graphs run. |
| `--mamba_ssm_cache_dtype` | _(required `float32` for Nemotron-H hybrids)_ | SSM cache precision for Mamba layers. | **Mandatory** for `nemotron3-nano-*-bf16` profiles — wrong dtype crashes the SSM kernel at boot. |
| `--attention-backend` | `flashinfer` | Attention kernel selection. SM110 requires FlashInfer for FP8 KV regardless of head_dim. | Stay on `flashinfer`. `flash_attn` works only at head_dim=128 in BF16 KV mode; crashes on FP8. |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}` for 35B | Speculative decoding params. MTP heads are baked into the weights for Qwen3.6 35B and Qwen3.6 27B-FP8; absent for Nemotron Nano family (see memory `project_nemotron3_mtp_availability.md`). | Match to model card. K=3 is NVIDIA's Spark recommendation; K=2 has historically been TEB-cleaner on FP8 KV. |
| Env: `VLLM_USE_FLASHINFER_MOE_FP4` | `0` for 35B-NVIDIA | Toggles FlashInfer NVFP4 MoE backends. NVIDIA W4A16 needs `=0` so the oracle picks Marlin. | Profile comments are the ground truth — don't flip without reading the rationale. |
| Env: `VLLM_USE_FLASHINFER_MOE_FP16` | `0` for omni + 35B-NVIDIA | Routes unquantized BF16 MoE through Triton (dodges SM100-only CUTLASS tile crash on SM110). | Always 0 on Thor for these profiles. |
| Env: `VLLM_FP8_MOE_BACKEND` | `flashinfer_cutlass` for 35B-NVIDIA | FP8 MoE kernel selection. | NVIDIA Spark recipe. Leave alone unless explicitly probing alternatives. |
| Env: `VLLM_NVFP4_GEMM_BACKEND` | `flashinfer-cutlass` for 35B-NVIDIA | NVFP4 GEMM kernel. SM110-compatible path. | Stay on `flashinfer-cutlass` for SM110 NVFP4. |
| Env: `CUTE_DSL_ARCH` | `sm_110a` for 35B-NVIDIA | Override CUTE-DSL arch detection. JetPack reports the GPU arch inconsistently; this pins it. | Set to `sm_110a` for any FlashInfer CUTEDSL path on Thor. |
| Env: `FLASHINFER_DISABLE_VERSION_CHECK` | `1` for 35B-NVIDIA | Bypasses FlashInfer's version sanity check (which trips on Thor's bundled libs). | Leave on for SM110 profiles using CUTEDSL. |

### B.2 Proxy mutations (`manyforge/scripts/proxy/vllm-proxy.py`)

All proxy mutations are **opt-in**. With every env var unset, the proxy is a
pure logger. Most production stacks set 2-3 of the knobs below; the iter-32
production recipe sets `OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS=2048` and
`OPENCLAW_PROXY_THINKING_TOKEN_BUDGET=512` only.

| Env var | Default | What it controls | When to use it |
|---------|---------|------------------|----------------|
| `OPENCLAW_PROXY_LISTEN_PORT` | `18790` (prod sets `8000`) | Listen port. | Override to bind on the same port your gateway/bridge dial. |
| `OPENCLAW_PROXY_BIND` | `127.0.0.1` (prod sets `0.0.0.0`) | Listen address. **Must be `0.0.0.0` if anything in a Docker container (e.g. Composer) needs to reach it via `host.openshell.internal`**. | Always `0.0.0.0` in production. |
| `OPENCLAW_PROXY_UPSTREAM` | `http://127.0.0.1:18789` (prod sets `http://127.0.0.1:8050`) | URL where vLLM (or another proxy) is listening. | Point at the vLLM container's port. |
| `OPENCLAW_PROXY_LOG_PATH` | `/tmp/openclaw_proxy.jsonl` | JSONL audit log path. Truncated on proxy start. | Set a per-iter path so you can compare runs side-by-side. |
| `OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS=N` | unset | Rewrite **or inject** `max_tokens` / `max_completion_tokens` to N on every chat-completions request. The **injection** path is load-bearing: OpenClaw → vLLM omits the field, so without injection vLLM defaults to the model's full context window and runs unbounded under thinking-on (one turn ≈ tens of minutes). | **Always set in production**, value 2048. Bump to 4096 only if measuring a model that truncates answers; reducing below 1024 cuts off legitimate tool-call payloads. |
| `OPENCLAW_PROXY_THINKING_TOKEN_BUDGET=N` | unset | Inject `chat_template_kwargs.thinking_token_budget=N` on every call. Soft cap on the `<think>…</think>` envelope when the chat template honors it (Qwen3-VL, Cosmos, Gemma-4 honor it; Nemotron does not). | **Set to 512 in production** (Qwen3-VL technical report sweet spot for 8B-class robotics tool calls — ~95% accuracy, ~half the latency of unbounded; 256 drops ~6 pts accuracy; 1024 is overkill). |
| `OPENCLAW_PROXY_FORCE_ENABLE_THINKING` | unset | Inject `chat_template_kwargs.enable_thinking` AND top-level `enable_thinking` (mirror added 2026-05-31). Modes: `on`, `off`, `alternating-off-on-even` (don't mutate odd turns / leave vLLM default; force false on even turns). | **Off by default.** Use `on` to force thinking everywhere (override Composer's top-level `enable_thinking:False`); use `off` to silence a thinking-on profile during a tool-call sweep; use the alternating mode only for `thinking-on→tool-emit` patterns. |
| `OPENCLAW_PROXY_FORCE_TOOL_CHOICE` | unset | Override `tool_choice` per call. Modes: `required` (always inject `required`), `auto` (always inject `auto`), `required-first` (only on turn 1 of a conversation, then pass through — lets the loop exit), `alternating` (odd turns), `alternating-on-even` (even turns). | Use `required` only during iter-experiments — the iter-34 negative result showed `required` every turn prevents the agent from ever exiting. `required-first` is the safer A/B. |
| `OPENCLAW_PROXY_OVERRIDE_TEMPERATURE` | unset | Overwrite `temperature` on every call. | Set to 0 to force determinism while debugging a flaky case; leave unset in production (vLLM owns the default). |
| `OPENCLAW_PROXY_OVERRIDE_TOP_P` | unset | Overwrite `top_p` on every call. | Same as above — debug only. |
| `OPENCLAW_PROXY_USER_MESSAGE_SUFFIX="…"` | unset | Append a fixed string to the LAST user message on every call. Idempotent (skips appending if the suffix is already there). | Used in iter 16 ("read first" hint) to inject a cross-cutting plan-then-execute nudge. Today it is deprecated in favour of in-prompt rule blocks; keep it available for one-shot A/B probes. |
| `OPENCLAW_PROXY_USER_SUFFIX_FIRST_TURN_ONLY` | unset | When set to `1` / `true`, only inject the user-suffix on the first turn of a conversation (no prior assistant messages). | Pair with `_USER_MESSAGE_SUFFIX` when chains are long enough that a per-turn nudge is overkill. |
| `OPENCLAW_PROXY_LOOP_REFLECT_AT` | `4` | When the same tool name has been called this many times across the conversation, **inject** a reflection user message after the last tool result: "STOP. You called X N times… choose (a) different tool, (b) change the failing arg, or (c) clarifying question." Marked with `[loop-reflection]` so it never injects twice. **Round 10, 2026-06-01.** | Lower (e.g. 3) for faster intervention; raise to disable (`0`). Threshold counts ALL turns in `messages[]`, so chain-on sessions accumulate quickly. |
| `OPENCLAW_PROXY_LOOP_STOP_AT` | `8` | When the same tool name has been called this many times across the conversation, **hard-stop** the agent loop: synthesize an SSE assistant response with content "I have called X N times… stopping to avoid runaway." OpenClaw treats this as a normal text completion and exits the loop. | Lower to fail-faster (smoke triages); raise (`0` to disable). Round 10's two-stage design: reflection at 4 gives the model one chance after fresh advice; hard stop at 8 caps total GPU spend. |
| `OPENCLAW_PROXY_LOOP_TOOL_THRESHOLD` | `0` (legacy) | **Legacy single-threshold env**, still honored. If set to N, maps to `_STOP_AT=N` and disables reflection (preserves pre-round-10 behavior). | Only set in a back-compat test harness. New deployments use the two-knob form. |

**Per-request socket timeout** is hard-coded to 200 s in
[`vllm-proxy.py:619`](../scripts/proxy/vllm-proxy.py). The smoke runner's
case timeout is 244 s; the proxy's 200 s ensures it fails first and releases
the upstream KV slot. Don't raise either without raising both.

#### Per-profile proxy tuning (2026-06-01)

Every model profile carries its own preferred proxy knobs in
`serving/config.sh`. The launcher reads these during
`load_thor_runtime_config` and `serving/start-model.sh:56-95` then
**auto-restarts** the local vllm-proxy with those values whenever a new
profile boots. Opt out (e.g. when the proxy is managed by a separate
supervisor) with `THOR_RESTART_PROXY=0`.

| Profile | `THOR_TARGET_PROXY_LOOP_REFLECT_AT` | `THOR_TARGET_PROXY_LOOP_STOP_AT` | `THOR_TARGET_PROXY_FORCE_ENABLE_THINKING` | Rationale |
|---------|--------------------------------------|----------------------------------|--------------------------------------------|-----------|
| `cosmos-reason2-8b` | `4` | `8` | _(unset — server default thinking-on dominates)_ | Production default; the proxy mirror picks up server-side thinking-on so no force needed. |
| `cosmos-reason2-2b` | `4` | `8` | _(unset)_ | Same as 8B with smaller footprint. |
| `nemotron3-nano-omni-30b-a3b-nvfp4` | `3` | `6` | `on` | Tighter loop because the model loops faster; force thinking-on because Composer's top-level `enable_thinking:false` would otherwise cancel the thinking budget. |
| `qwen3.6-35b-a3b-nvfp4-nvidia` | `4` | `8` | _(unset)_ | Heavier model; default thresholds suffice. |
| _other profiles_ | `4` | `8` | _(unset)_ | Defaults until a profile-specific calibration is added. |

The `start-model.sh` auto-restart preserves any `OPENCLAW_PROXY_*` env vars
the caller set explicitly (it sources `:-…` defaults so caller values win).
Combined with `assistant.sh`'s baseline (`OVERRIDE_MAX_TOKENS=2048`,
`THINKING_TOKEN_BUDGET=512`), the production proxy boot is:

```bash
OPENCLAW_PROXY_BIND=0.0.0.0
OPENCLAW_PROXY_LISTEN_PORT=8000
OPENCLAW_PROXY_UPSTREAM=http://127.0.0.1:8050
OPENCLAW_PROXY_LOG_PATH=/tmp/manyforge-assistant-e2e/vllm-proxy.jsonl
OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS=2048
OPENCLAW_PROXY_THINKING_TOKEN_BUDGET=512
OPENCLAW_PROXY_LOOP_REFLECT_AT=<profile value>
OPENCLAW_PROXY_LOOP_STOP_AT=<profile value>
[OPENCLAW_PROXY_FORCE_ENABLE_THINKING=<profile value if set>]
```

When adding a new profile (recipe §F.5), the per-profile proxy vars are
now part of the profile contract — set them in `config.sh` even if they
match the default-4/8 values, so future re-tuning has a clear baseline.

### B.3 Bridge config (`manyforge/openclaw_assistant_bridge/`)

Loaded in `service.py::_config_from_env()` and (for cluster identifiers) at
module import. Bridge restart required after env changes; Composer + gateway
unchanged.

| Env var | Default | What it controls | When to use it |
|---------|---------|------------------|----------------|
| `OPENCLAW_ASSISTANT_BRIDGE_HOST` | `127.0.0.1` | Bind address. | `0.0.0.0` if Composer is in a remote container and dialing across the bridge. |
| `OPENCLAW_ASSISTANT_BRIDGE_PORT` | `8200` | Listen port for the provider HTTP contract. | Always `8200` in production. |
| `OPENCLAW_ASSISTANT_USE_GATEWAY` | `false` | When `true`, dispatch via the in-sandbox **persistent gateway** (HTTP at `:18789`). When `false`, shell-out to `openclaw` CLI per request. | **Always `true` in production.** CLI mode is a debugging fallback. |
| `OPENCLAW_ASSISTANT_AGENT` | `main` | OpenClaw agent profile name (the agent ID in `openclaw.json`). | Always `manyforge-composer` in production. |
| `OPENCLAW_ASSISTANT_TIMEOUT_S` | `120` | End-to-end timeout for the OpenClaw agent invocation. | `300` in production — legitimate runs can hit 100-200 s under thinking-on. |
| `OPENCLAW_ASSISTANT_GATEWAY_PORT` | `18789` | Port inside the sandbox where the gateway listens (forwarded to host by SSH netns). | Leave alone. |
| `OPENCLAW_ASSISTANT_GATEWAY_MAX_TOKENS` | `4096` | `max_tokens` value forwarded to the gateway request envelope. Note: this is **a different field** than the proxy's `_OVERRIDE_MAX_TOKENS`; the proxy still rewrites/injects on the chat-completions hop. | Leave alone — `4096` here is the bridge → gateway hint; the proxy's `2048` is what reaches vLLM. |
| `OPENCLAW_ASSISTANT_GATEWAY_TEMPERATURE` | unset | Optional per-request `temperature` for the gateway envelope. Default None → not added. | Probe-only. The bridge intentionally omits sampling fields so vLLM's `--override-generation-config` owns them. |
| `OPENCLAW_ASSISTANT_GATEWAY_TOP_K`, `_TOP_P` | unset | Same as above. | Probe-only. |
| `OPENCLAW_ASSISTANT_GATEWAY_ENABLE_THINKING` | unset | Optional boolean. If set, added to `chat_template_kwargs` in the gateway envelope. Note: **the proxy's mirror is the proven path** — this flag exists as an alternative but is less tested. | Use the **proxy's** `OPENCLAW_PROXY_FORCE_ENABLE_THINKING` for production toggles. This bridge-side var is here for symmetry. |
| `OPENCLAW_ASSISTANT_COMPACT_EVERY_N` | `0` (disabled) | When set to N>0, the bridge POSTs `/compact` to the gateway **before** every Nth user prompt on this session key (skipping #1). Counter resets on bridge restart. **Iter-32 production setting.** | `2` in production — keeps chain-on sessions from overflowing the 256K context. Disable only if measuring an isolated baseline. |
| `OPENCLAW_ASSISTANT_COMPACT_TIMEOUT_S` | `120` | Timeout for the `/compact` call itself. If exceeded the failure is logged and the user request still goes through. | Leave alone — compaction normally completes in 10-30 s. |
| `OPENCLAW_ASSISTANT_LOOP_TOOL_THRESHOLD` | `5` | Bridge-side **cross-turn loop detector** (round 8, 2026-05-31). When the request's `messages[]` history has 5+ assistant turns calling the same tool, the bridge short-circuits with a synthetic stop response (status 200, message "I have called X N times… stopping to prevent a runaway loop.", `warnings: ["loop_detected_stopped: …"]`). NOTE: Composer→bridge sends ONE user message per turn — `messages[]` is rarely populated as a history — so the load-bearing duplicate-tool detector is the **proxy's**. Leave at 5 as a belt-and-braces backstop. | This catches loops the proxy's per-conversation threshold would miss when Composer restarts conversations indefinitely. Leave at 5; raise to `0` to disable. |
| `OPENCLAW_ASSISTANT_ENABLE_SYNTHETIC` | _(unset = off)_ | **NEW 2026-06-01.** Opt-in for the bridge-side synthetic clarification short-circuit (see §D.4). When set (1/true/yes/on), prompts matching the narrow gate `add a <kind>` / `insert a <kind>` / `wrap with <kind>` (≤4 words, kind ∈ {parallel, fallback, sequence, repeat, retry, inverter}) are answered with a canned "Which parent? Which position?" without invoking OpenClaw. Default OFF means the model itself must ask (Rule 11a in `adapter.py`). | Production: leave OFF. Enable only when running a smoke against a model that demonstrably cannot pass Rule 11a and you need the bypass to ship. |
| `OPENCLAW_ASSISTANT_DISABLE_SYNTHETIC` | _(unset = off)_ | **Legacy back-compat opt-out** (was the only knob before 2026-06-01, when synthetic was hard-coded ON). When set (1/true/yes/on), forces the bypass off even if `_ENABLE_SYNTHETIC` was also set. Honored for backwards compatibility with launcher scripts that referenced this var. | Use only if you have an old script that sets ENABLE but you want to verify the bypass is off. New deployments should leave it unset and rely on the new opt-in. |
| `OPENCLAW_ASSISTANT_SANDBOX` | `my-assistant` | NemoClaw sandbox name. | Match the actual sandbox; default is the one the provisioner installs. |
| `OPENCLAW_ASSISTANT_NAMESPACE` | `openshell` | K8s namespace inside the cluster gateway container. | Leave alone. |
| `OPENCLAW_ASSISTANT_CONTAINER` | `agent` | K8s container name. | Leave alone. |
| `OPENCLAW_ASSISTANT_CLUSTER_CONTAINER` | `openshell-cluster-nemoclaw` | Docker container hosting the k3s cluster + gateway runtime. | Leave alone unless you renamed it. |
| `OPENCLAW_ASSISTANT_SANDBOX_USER` | `sandbox` | Linux user for `kubectl exec` shell. | Leave alone. |
| `OPENCLAW_ASSISTANT_BIN` | `openclaw` | Path to the `openclaw` binary on the host (CLI fallback). | Override when testing a non-default install. |
| `OPENCLAW_ASSISTANT_LOCAL` | `false` | When `true`, run the agent locally instead of via cluster `kubectl exec`. | Debug-only — bypasses the sandbox's Landlock + seccomp + egress policy. |
| `OPENCLAW_ASSISTANT_THINKING` | `off` | Adapter-side hint for OpenClaw's `--thinking` CLI flag. **Only used in CLI mode.** | Gateway mode ignores this. |
| `OPENCLAW_ASSISTANT_AUTO_TOOL_WINDOW` | `true` | When `true` in CLI mode, the bridge narrows the per-request allowlist by inferring tools the prompt needs. | Leave on for CLI mode. Gateway mode does not narrow (incompatible with the persistent MCP wrapper). |
| `OPENCLAW_ASSISTANT_ALLOWED_TOOLS_FILE` | `/tmp/manyforge-openclaw-allowed-tools.txt` | Where the CLI-mode allowlist is written for the MCP wrapper to read. | Leave alone. Stale entries cause the symptom in the runbook §3 ("not exposed by this request's tool window"). |
| `OPENCLAW_ASSISTANT_COMPOSER_BASE` | `http://127.0.0.1:9000` | Composer URL for principal-binding registration (live tool-call streaming). | Override when Composer runs on a non-default port. |
| `OPENCLAW_ASSISTANT_CIRCUIT_BREAKER_ENABLED` | `false` | Opt-in: after N consecutive failures, fail fast with 503 instead of dispatching to a sick gateway. | Enable in production reliability runs; leave off for development. |
| `OPENCLAW_ASSISTANT_CIRCUIT_BREAKER_THRESHOLD` | `5` | Consecutive failures before opening. | Tune by environment. |
| `OPENCLAW_ASSISTANT_CIRCUIT_BREAKER_COOLDOWN_S` | `30` | Half-open probe interval. | Tune by environment. |
| `OPENCLAW_ASSISTANT_METRICS_ENABLED` | `false` | When `true`, mount `/metrics` Prometheus endpoint on the bridge port. | Enable for production observability. |
| `OPENCLAW_ASSISTANT_LOG_LEVEL` | `info` | Uvicorn log level. | `debug` for new-incident triage. |

**Bridge-side synthetic-clarification short-circuit** (`service.py:370-446`,
round 7 of 2026-05-31; **default-OFF since 2026-06-01**): pattern-matches
`add a <kind>` / `insert a <kind>` / `wrap with <kind>` (kind ∈ parallel,
fallback, sequence, repeat, retry, inverter; word count ≤ 4); returns a
canned "Which parent? Which position?" clarification without ever invoking
OpenClaw.

The 2026-06-01 inversion: **opt-in via `OPENCLAW_ASSISTANT_ENABLE_SYNTHETIC=1`**;
default off so smoke benchmarks measure the model's actual ability to ask
clarification on those patterns (the bypass was hiding model-specific
behavior). Legacy `OPENCLAW_ASSISTANT_DISABLE_SYNTHETIC=1` opt-out preserved
for backwards compatibility — if set, forces off even when ENABLE is set.

The prompt-side replacement is **Rule 11a Missing-WHERE** (adapter.py:519-536)
which instructs the model to ask "which parent and where in its children?"
for `PARALLEL_generic` / `FALLBACK_generic` / `UPDATE_params_generic`
patterns. See §D.4 for full details on both mechanisms.

### B.4 Composer config (sibling `dev_ws/src/manyforge/`)

Composer is owned by the manyforge repo; only the parts that interact with
this pipeline are documented here. Production runs use the
`scripts/lib/assistant.sh` launcher.

| Env var / setting | Default | What it controls |
|-------------------|---------|------------------|
| `ASSISTANT_PROVIDER` | `openclaw` | Selects which bridge Composer hits. Values: `openclaw` (production, `:8200`) or `nemoclaw` (direct lane, `:8100`). |
| `MODEL_PROFILE` | `cosmos-reason2-8b` | Profile slug for `serving/start-model.sh` and `configure-local-provider.sh`. The `served-model-name` advertised by vLLM equals this slug. |
| `START_VLLM_PROXY` | `true` | Whether the launcher starts the vllm-proxy mutator. When `false`, the proxy is assumed externally managed. |
| `OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS` | `2048` | Forwarded to the proxy. **Load-bearing.** See §B.2. |
| `OPENCLAW_PROXY_THINKING_TOKEN_BUDGET` | `512` | Forwarded to the proxy. See §B.2. |
| `OPENCLAW_ASSISTANT_COMPACT_EVERY_N` | `2` | Forwarded to the bridge. See §B.3. |
| `OPENCLAW_ASSISTANT_COMPACT_TIMEOUT_S` | `120` | Forwarded to the bridge. |
| `OPENCLAW_ASSISTANT_METRICS_ENABLED` | `false` | Forwarded to the bridge. |
| `ASSISTANT_TIMEOUT_S` | `300` | End-to-end UI timeout (Composer-side). After this Composer surfaces "NemoClaw assistant timed out after 300.000s". |
| `ASSISTANT_MAX_TURNS` | `16` | Direct-lane only: maximum agent loop turns. |
| `DROP_CACHES` | `true` | Whether `drop_caches` runs after stop (Thor unified-memory hygiene). |
| `PROVISION_OPENCLAW_SANDBOX` | `true` | Whether the launcher re-runs the provisioner before bringing up the bridge. |
| `VLLM_CONTAINER` | `manyforge-e2e-vllm` | Docker container name for vLLM. |
| `VLLM_MODEL_READY_TIMEOUT_S` | `900` | First-launch grace (15 min); first-time NVFP4 JIT compile can take 60+ min on a fresh image (see memory `project_v81_first_launch_timing.md`). |

**Composer's `enable_thinking` invariant** (load-bearing gotcha): Composer's
`NemoClawAssistantProvider` always sends `enable_thinking: false` at the
**top level** of the provider envelope, regardless of the model's chat
template. This is a deliberate design decision (Composer treats thinking
as opt-in). vLLM treats the top-level field as the source of truth and
**ignores** `chat_template_kwargs.enable_thinking` when both are present.
The proxy's `_FORCE_ENABLE_THINKING` mode mirrors its chat_template_kwargs
decision to the top level (`vllm-proxy.py:285-292`); without that mirror,
thinking-on profiles silently revert to thinking-off when Composer is the
client. The per-profile `THOR_TARGET_PROXY_FORCE_ENABLE_THINKING` in
`serving/config.sh` feeds this mirror through `start-model.sh`'s
auto-restart path so each profile gets its preferred thinking posture
without manual env setup.

**Composer tool schema reuse: `node.name` accepted in `tree_draft_insert_node`**
(`manyforge_composer/backend/assistant_tool_schemas.py:488-528`, 2026-06-01).
The `_TREE_NODE_PAYLOAD_FOR_INSERT_SCHEMA` schema now accepts an optional
`node.name` field as a duplicate of the top-level `nodeName` argument
(mirroring the older `_TREE_NODE_SCHEMA` shape used by
`tree_draft_wrap_node` / `tree_draft_replace_subtree`). Background: small
models routinely emit both shapes — top-level `nodeName: foo` AND
`node.name: foo` inside the payload — because they've seen `_TREE_NODE_SCHEMA`
in the wrap/replace tools. Before the fix, `additionalProperties: false`
on the insert payload rejected `node.name` and the model entered a
validator death-spiral (verified during Cosmos-Reason2-8B smoke runs). The
handler still reads `nodeName` from the top level; `node.name` is
accepted-and-ignored. If a model passes mismatched values, top-level
`nodeName` wins. The fix is comment-documented in the schema source at
lines 510-528.

### B.5 Smoke corpus knobs (`manyforge/scripts/debug/smoke_corpus_runner.py`)

| Flag | Default | What it controls |
|------|---------|------------------|
| `--corpus` | `scripts/debug/smoke_corpus.yaml` | Path to the corpus YAML. |
| `--composer` | `http://127.0.0.1:9000` | Composer base URL. |
| `--filter <regex>` | unset | Only run cases whose `id` matches the regex. Used to scope to a single failure pattern. |
| `--include-future` | off | Include cases marked `future:` true in the corpus (P3+, future-feature cases). |
| `--runtime-flags <csv>` | empty | Enable specific runtime tier names for per-case gating. |
| `--skip-fixture-cases` | off | Skip cases marked as fixture probes. |
| `--report <path>` | unset | Write JSON report to this path (pass/fail per case + aggregates). |
| `--verbose` | off | Print per-case detail to stdout. |
| `--enable-recovery-turn` | off (default-on in production runbook) | When a case fails AND chat returned 200, send one generic follow-up. Cases that pass on the recovery turn are scored `recovered-pass` ([SMOKE-ITER-RUNBOOK.md `--enable-recovery-turn` section](./SMOKE-ITER-RUNBOOK.md)). Iter 33 measured +10 cases salvaged this way. |
| `--no-chain-session` | off | Give each chain step its own `conversationId`. Without this, PnP_01..PnP_20 share a session and one early failure can cascade. **Iter 32 + bridge compaction made chain-on viable**; only use `--no-chain-session` for chain-off baseline comparison. |

**Per-case `alt_names`** (2026-06-01, `smoke_corpus_runner.py:391-424`):
each `expected_tool_calls[]` entry can carry an `alt_names: [...]` list of
tool names that produce an equivalent effect (e.g.
`scene_draft_upsert_objects` as an alt for `scene_draft_add_object` — both
land the same resource via different verbs). The matcher tries the primary
`name` first; on miss, walks `alt_names` looking for any with a 2xx tool
result. A successful alt match records the entry as a **soft-pass**
(`alt-tool '<x>' used instead of '<y>' (equivalent effect)`) rather than a
hard fail. The outer runner's report rolls those into `soft-pass` status
which counts toward the "effective" tally but not "first-try". Use this to
accept legitimate variant behavior without rewriting corpus expectations.

**Smoke corpus rebalance (2026-06-01)**: 4 cases now expect ASK
(clarification, no tool emission) rather than a tool call:
`PARALLEL_generic`, `FALLBACK_generic`, plus 2 cases labeled
`category: clarification` (smoke_corpus.yaml lines 1170, 1196, 1214). The
rebalance is tied to the default-synthetic-OFF design — these cases probe
whether the model itself (via Rule 11a) asks for parent/position without
the bridge bypass.

The runner reconfigures stdout to line-buffered at import (`smoke_corpus_runner.py:48`) so `tail -f /tmp/iterN_runner.log` streams verdicts realtime — see [SMOKE-ITER-RUNBOOK.md §4](./SMOKE-ITER-RUNBOOK.md) for the rationale.

---

## C. The "thinking" subsystem

The single most important behavior the operator can break by accident.
Behavior depends on the interaction of **four** layers:

1. **Model chat template** — owns the `<think>…</think>` envelope shape.
2. **vLLM server-side default** — `--default-chat-template-kwargs '{"enable_thinking":…}'`.
3. **Per-request mutation in the proxy** — `OPENCLAW_PROXY_FORCE_ENABLE_THINKING`.
4. **Caller-supplied top-level `enable_thinking`** in the request body (Composer always sets `false`).

### C.1 vLLM precedence rules (verified 2026-05-31 on cosmos-reason2-8b)

When the request body contains BOTH `chat_template_kwargs.enable_thinking` AND
a top-level `enable_thinking`, **vLLM uses the top-level value**. The
chat_template_kwargs path is honored only when top-level is absent. This is
why Composer's `enable_thinking:false` at the top level cancels a profile's
thinking-on default unless the proxy mirrors a `true` to the top level too.

The 2026-05-31 patch (`vllm-proxy.py:285-292`) handles this by writing both
fields together when `_FORCE_ENABLE_THINKING` is set:

```python
ctk["enable_thinking"] = target_value          # chat_template_kwargs (legacy path)
parsed["enable_thinking"] = target_value       # top-level (vLLM honors this)
```

If you only mutate one, you get silent thinking-off on a thinking-on profile.

### C.2 `reasoning_parser` vs `reasoning_effort`

- `--reasoning-parser` is a **vLLM server flag**. It tells vLLM how to split
  the model's output into `reasoning` and `content`. Set per profile in
  `serving/launch.sh`. Available parsers: `qwen3`, `nano_v3`, `nemotron_v3`,
  `deepseek_r1`, `gemma4`.
- `reasoning_effort` is a **per-request field** (OpenAI-style). **Not plumbed
  through this stack** today — Composer and the bridges never set it. If you
  need per-request reasoning control, use `chat_template_kwargs.enable_thinking`
  / `thinking_token_budget` instead.

### C.3 `thinking_token_budget`

Soft cap on the `<think>…</think>` envelope. Honored by:

- Qwen3-VL family chat template (Cosmos 2B/8B, Qwen3.5/3.6, …)
- Gemma-4 chat template
- Some Nemotron templates (verify per profile)

**Not** honored by:

- Nemotron-3-Nano-Omni (the template doesn't gate on this kwarg)
- Stock OpenAI / Llama templates

Production value: **512** for 8B-class robotics tool calls (Qwen3-VL tech
report sweet spot — ~95% accuracy, ~half the latency of unbounded).

### C.4 Per-profile thinking defaults (current)

| Profile | `--default-chat-template-kwargs` | Reasoning parser | Notes |
|---------|--------------------------------|------------------|-------|
| `cosmos-reason2-2b` | _(unset)_ | _(none)_ | Template defaults apply; thinking opt-in. |
| `cosmos-reason2-8b` | `{"enable_thinking":true}` | `qwen3` | **Thinking-on** since iter 17 (2026-05-09). Matches post-training distribution. Production default. |
| `nemotron3-nano-4b-bf16` | `{"enable_thinking":false}` | _(none)_ | Tool-call regime per HF card. |
| `nemotron3-nano-omni-30b-a3b-nvfp4` | `{"enable_thinking":false}` | _(none, dropped 2026-05-06)_ | Bridges consume `content`; reasoning parser would route output into `reasoning` and the lane would return empty. |
| `nemotron3-nano-omni-30b-a3b-nvfp4-reasoning` | `{"enable_thinking":true}` | `nemotron_v3` | Reasoning variant; consumers must read `reasoning` not `content`. Today the bridges do not. |
| `qwen3.6-35b-a3b-nvfp4-nvidia` | `{"enable_thinking":true}` | `qwen3` | Iter-32 production sampling recipe + MTP K=3. |
| `gemma4-31b-it-nvfp4` / `gemma4-26b-a4b-it` / `gemma4-e4b-it` | _(template defaults)_ | `gemma4` | Tool-call parser `gemma4`. |
| `qwen3.5-9b-claude-distilled-nvfp4` | _(unset)_ | _(none)_ | No-think variant of the template. |
| `qwen3.6-27b-fp8-mtp-kvfp8` | _(template defaults)_ | _(none)_ | EXPERIMENTAL; MTP K=1. |

### C.5 Session learnings on thinking (2026-05-31)

- **Cosmos-Reason2-8B** needs thinking-on per iter 17. Running thinking-off is
  OOD for the model and produces narration-mode collapse on action prompts.
  The qwen3 reasoning parser was added 2026-05-31 to extract `<think>` blocks
  before hermes parses content — without it, thinking bled into `content` and
  produced ~30% narration / ~47% nodeName-dropped failures on v9.
- **Nemotron-3-Nano-Omni instruct** (non-reasoning) profile uses thinking-OFF
  deliberately: the bridges consume `content`, and the `nemotron_v3` reasoning
  parser would otherwise route output into `reasoning`, leaving messages
  empty.
- **35B-NVIDIA** benefits from thinking-on + MTP K=3 + Marlin MoE. The
  Spark recipe is the source of truth.
- **Gemma-4 family** uses `gemma4` parsers for both tool calls and reasoning;
  template-driven thinking is the default.

---

## D. Loop detection / reflection injection (round 10, 2026-06-01)

The **cascading defense** sits at two layers because the symptom appears at
two layers:

1. **Within an OpenClaw agent loop** (one Composer prompt → many vLLM turns):
   OpenClaw enforces a `per-turn 15-cap` budget, but the cap is per Composer
   prompt. The model can still spend 15 turns re-trying the same tool with
   the same args inside one prompt.
2. **Across Composer prompts** (chain-on smoke / real-user multi-turn UI):
   each Composer prompt is a fresh OpenClaw budget. The smoke runner / a
   patient user can extend a same-tool-same-error loop indefinitely.

The proxy and the bridge each defend one layer.

### D.1 Proxy defense (within one chat-completions call)

`vllm-proxy.py:362-510`. **CONSECUTIVE tail-run counting since 2026-06-01**.
The proxy walks the assistant turns in `messages[]` in order, tracking the
**run length** of consecutive same-tool calls at the **tail** of the
sequence:

- A turn that calls a tool with the **same** name as the running `top_name`
  increments `consecutive_count` by 1.
- A turn that calls a **different** tool resets the run to 1 with the new
  name as `top_name`.
- A turn with **no tool call** resets `consecutive_count = 0` and
  `top_name = None`.

This replaced the prior lifetime-counter implementation (using
`Counter.most_common`). Lifetime counting was the major bug fixed 2026-06-01:
when Composer shared a `conversationId` across chained smoke cases (e.g.
`PnP_01`..`PnP_20`), the same tool's lifetime call count crossed `_STOP_AT`
by the 3rd or 4th case and hard-stopped every subsequent case on turn 1 —
even though no case was actually in a same-tool loop. The new semantics
only fire when the model is **actually** stuck in a same-tool retry pattern
within the current case's turn stream.

| Trigger | Action |
|---------|--------|
| `consecutive_count >= OPENCLAW_PROXY_LOOP_REFLECT_AT` (default 4) | **Inject** a `[loop-reflection]`-marked user message right after the last tool result, urging the model to (a) call a different tool, (b) change the failing argument, or (c) ask the user. Forward the mutated body. Marker prevents double-injection. |
| `consecutive_count >= OPENCLAW_PROXY_LOOP_STOP_AT` (default 8) | **Hard-stop**: synthesize an SSE `chat.completion.chunk` with finish_reason=stop and content "I have called X N times… stopping to avoid runaway. Please refine the request…". OpenClaw treats this as a normal text completion and exits the loop. NEVER forwarded upstream — no GPU spend. |

Telemetry events appended to the proxy JSONL:

- `proxy_loop_reflection_injected` — count crossed `_REFLECT_AT`, mutated
  body forwarded
- `proxy_loop_hard_stop` — count crossed `_STOP_AT`, synthetic SSE returned,
  no GPU spend

Why **both** triggers exist instead of one: a single hard stop at low N
denies the model a chance to recover after seeing fresh advice. The two-stage
design (reflect → wait one turn → stop) means the model gets exactly one shot
at the reflection prompt before being cut off, capping total GPU spend
predictably at `STOP_AT` turns per case.

**Legacy back-compat**: the older single-knob `OPENCLAW_PROXY_LOOP_TOOL_THRESHOLD`
env var is still honored — when set, it maps to `_STOP_AT` and DISABLES
reflection (sets `_REFLECT_AT=0`) unless `OPENCLAW_PROXY_LOOP_REFLECT_AT` is
also explicitly set. New deployments should use the two-knob form.

### D.2 Bridge defense (across multiple Composer prompts)

`service.py:412-473`. Same counter logic, but in the bridge — fires
**before** OpenClaw is invoked at all. Returns status 200 with a
`warnings: ["loop_detected_stopped: …"]` payload that Composer renders as a
normal model message and the smoke runner scores as a fail.

| Trigger | Action |
|---------|--------|
| `top_count >= OPENCLAW_ASSISTANT_LOOP_TOOL_THRESHOLD` (default 5) | Synthesize a 200 response with a canned "I have called X N times… stopping to prevent a runaway loop. Please refine the request…" message. No OpenClaw invocation. `bridge_synthetic_loop_break` telemetry event. |

### D.3 Why this matters

- OpenClaw's built-in per-turn 15-cap was insufficient because real
  conversations span many turns.
- Without these defenses a single bad smoke case could spend 25-28 turns
  in OpenClaw at 6-10 s each (observed on cosmos-8b 2026-05-31), exhausting
  the case's 275 s budget AND blocking the next case via stale KV.
- The two-stage proxy design + bridge cross-turn floor reliably bound
  total time-per-case at ~60-80 s even in the worst loop.

### D.4 Bridge synthetic-clarification + the Rule 11a prompt-side equivalent

There are **two** mechanisms in the pipeline that attempt to make the model
ask the right "which parent / where in its children?" clarification on
narrow `add a <kind>` patterns. They overlap in purpose but operate at
different layers:

| Mechanism | Where | Cost | Gating | Today |
|-----------|-------|------|--------|-------|
| **Bridge synthetic clarification** | `service.py:370-446` | 0 GPU — bypass before OpenClaw is invoked | Opt-in env `OPENCLAW_ASSISTANT_ENABLE_SYNTHETIC=1` | **DEFAULT OFF since 2026-06-01** |
| **Prompt-side Rule 11 NO_REPLY + Rule 11a Missing-WHERE** | `adapter.py:513-536` | Normal GPU spend (model invocation) | Always-on (part of every RULES block) | Production replacement for the bypass |

**Bridge synthetic clarification (round 7, 2026-05-31)** — bypass OpenClaw
entirely on prompts matching a narrow gate, return a canned answer. Narrow
gate (`service.py:407-415`):

- `add a <kind>` / `insert a <kind>` / `wrap with <kind>`
- `<kind>` ∈ {parallel, fallback, sequence, repeat, retry, inverter}
- Total word count ≤ 4 (compound forms like `add a parallel that ...`
  refused)

Returns a canned "Which parent node? Which position? For example: 'as the
first child of pick_and_place', 'after gripper_close', or 'as a new root
wrapping the existing tree'." answer. Smoke's `answer_must_contain` rubric
checks for "which" and "where" tokens; the synthetic answer contains both.

Default inverted to **OFF** on 2026-06-01 because the bypass gave every
candidate model the same free pass on `PARALLEL_generic` /
`FALLBACK_generic`, masking model-specific ability (or lack thereof) to ask
clarification. Enable with `OPENCLAW_ASSISTANT_ENABLE_SYNTHETIC=1` for a
production lane where the model demonstrably cannot pass Rule 11a (e.g.
Cosmos-Reason2-8B on chain-off first-turn action prompts). Legacy
`OPENCLAW_ASSISTANT_DISABLE_SYNTHETIC=1` is preserved as a back-compat
opt-out — if set, forces off even when ENABLE is set.

**Rule 11 NO_REPLY guard + Rule 11a Missing-WHERE
(2026-06-01, `adapter.py:513-536`)** — the prompt-side replacement. Both
rules sit in the per-turn RULES block injected into every gateway-mode
agent prompt:

- **Rule 11**: "NEVER reply with the literal string `NO_REPLY` when the
  user's prompt contains an action verb — `NO_REPLY` is reserved for
  genuinely empty contexts (silence, continuation prompts), NOT for action
  requests. If the request is action-shaped, either emit a tool call or
  ask a clarifying question with real text content." Some Qwen-family
  models default to `NO_REPLY` on unfamiliar action prompts; this guard
  forbids that escape hatch.
- **Rule 11a (Missing-WHERE)**: "A prompt is ambiguous if it names the
  operation and node kind but NOT where to place the result (parent /
  position / target sibling). For these, output ONLY a clarification
  question — do NOT call any tool." The rule names the patterns
  explicitly: `PARALLEL_generic`, `FALLBACK_generic`,
  `UPDATE_params_generic`. The smoke corpus mirrors these names in case
  IDs so the rule maps directly to test outcomes.

The two rules together are designed to make the model do what the bypass
used to do, **without** requiring the bypass — letting the smoke corpus
fairly measure cross-model clarification quality. When designing a new
profile, the question to ask is: "does this model satisfy Rule 11a on the
4 ASK cases without the bypass?" If yes, leave synthetic OFF. If no, the
bypass exists as a shipping crutch.

There is also a separate **self_check appendix** at `adapter.py:575-610`
that appends a `\n\n## self_check (apply BEFORE responding)` block to the
user_request when the prompt is ≤5 words, starts with `add a `/`insert a `
/`wrap with `/`wrap the root with `, contains a control-flow kind, and has
**no** locator keyword (none of: `after `, `before `, `child of`,
`under `, `inside `, `position `, `first child`, `last child`, `at index`,
`somewhere`, …). The self_check still routes through the model (unlike
the synthetic bypass which short-circuits), but biases it toward asking
rather than guessing.

---

## E. Profile catalog (`serving/config.sh` + `serving/launch.sh`)

Adding or modifying a profile requires matching edits to **both** files using
the same case-statement label.

| Slug | Model source | Quant | Footprint | Ctx | Tool parser | Reasoning parser | Default thinking | BFCL / smoke | Recommended use |
|------|-------------|-------|-----------|-----|-------------|-----------------|------------------|---------------|-----------------|
| **`cosmos-reason2-8b`** | `nvidia/Cosmos-Reason2-8B` (Qwen3-VL-8B base) | _(none, BF16)_ | ~16 GB weights + FP8 KV pool | 262 144 | `hermes` | `qwen3` | on | smoke 9/9 OpenClaw lane | **Production default (2026-05-07).** VLM, physical-AI reasoner, agentic. |
| `cosmos-reason2-2b` | `nvidia/Cosmos-Reason2-2B` (Qwen3-VL-2B base) | _(none, BF16)_ | ~4.3 GB weights | 32 768 | `hermes` | _(none)_ | template default | _(not benched)_ | VLM small-footprint, low concurrency. Co-serve candidate with 8B for fan-out. |
| `nemotron3-nano-4b-bf16` | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | _(none, BF16)_ | ~8 GB weights | 65 536 (conservative; native 262K) | `qwen3_coder` | _(none, intentional)_ | off | BFCL v3 = 61.1 | NVIDIA's explicit Jetson Thor agentic default. Hybrid Mamba+attn. Tool-call-trained. Edge-class. |
| `nemotron3-nano-omni-30b-a3b-nvfp4` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | NVFP4 | 20.9 GB on disk | 262 144 | `qwen3_coder` | _(none, dropped 2026-05-06)_ | off | TEB 80/100 ★★★★ Good, IFEval 87.7% (vendor regime) | Multimodal (vision+audio+video), MoE, instruct mode. Lane parity hit 0/9 vs cosmos 9/9 — use cosmos for production. |
| `nemotron3-nano-omni-30b-a3b-nvfp4-reasoning` | same weights | NVFP4 | 20.9 GB | 262 144 | `qwen3_coder` | `nemotron_v3` | on | _(not benched on assistant pipeline)_ | Reasoning variant; **bridges would need to read `reasoning` field — they don't today**. Don't route Composer through this without a bridge change. |
| `qwen3.6-35b-a3b-nvfp4-nvidia` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | NVFP4 (W4A16, ModelOpt) + Marlin MoE | ~18 GB weights | 262 144 | `qwen3_coder` | `qwen3` | on | NVIDIA card: τ²-Bench Telecom 94.7 NVFP4 vs 95.5 BF16. Smoke: 56/66 (2026-05-31, head-to-head Task 4) | EXPERIMENTAL. Heavier model with MoE 3B active. MTP K=3 + Marlin. Use when 8B quality isn't enough and 65 min wall-clock per smoke is acceptable. |
| `qwen3.6-27b-fp8-mtp-kvfp8` | `Qwen/Qwen3.6-27B-FP8` | FP8 | ~27 GB weights | 262 144 | _(profile default)_ | _(none)_ | template default | _(not benched on assistant pipeline)_ | EXPERIMENTAL dense hybrid; preserves MTP heads (NVFP4 toolchains strip them on this model). Slower than 8B. |
| `qwen3.5-9b-claude-distilled-nvfp4` | _(Claude 4.6-distilled VLM)_ | NVFP4 + FP8 KV | ~9 GB | 131 072 | _(profile default)_ | _(none)_ | off (no-think template) | _(internal eval)_ | Fast-control / no-think VLM; vision + text + tools. |
| `gemma4-e4b-it` | _(Gemma-4 E4B IT)_ | _(none)_ | ~4 GB | 131 072 | `gemma4` | `gemma4` | template default | _(not benched on assistant pipeline)_ | Small Gemma-4 edge profile. |
| `gemma4-31b-it-nvfp4` | _(Gemma-4 31B IT NVFP4)_ | NVFP4 (modelopt) | ~16 GB | 262 144 | `gemma4` | `gemma4` | template default | _(not benched)_ | Medium Gemma-4 quantized. |
| `gemma4-26b-a4b-it` | _(Gemma-4 26B A4B IT)_ | _(none)_ | ~26 GB | 262 144 | `gemma4` | `gemma4` | template default | _(not benched)_ | Gemma-4 MoE variant. |

**Removed profiles** (kept as historical comments in `config.sh` / `launch.sh`
with rationale): `minimax-m2.7-139b-a10b-nvfp4`, `qwen3.5-122b-a10b-nvfp4`,
several Qwen3.6 35B sub-variants (mtp-fp8kv, n4, tq-mtp, tq-mtp-2,
dflash{,-vl}, fp8-mtp-fp8kv, fp8-turboquant, prismaquant-dflash),
`qwen3.5-35b-a3b-nvfp4`, `nemotron3-nano-30b-a3b-nvfp4` (text-only superseded
by omni), `cosmos-reason2-8b-reasoning` (broken tuning). Look in
[`serving/docs/`](../../serving/docs/) for the dated investigations behind
the removals.

---

## F. Common workflows (cookbook)

### F.1 Boot a new model and run targeted-9 smoke

```bash
cd ~/workspaces/nemoclaw/src/NemoClaw-Thor

# 1. Stop any existing vLLM container (assumes detached run)
docker rm -f manyforge-e2e-vllm 2>/dev/null

# 2. Start the new profile
THOR_DETACH=1 THOR_CONTAINER_NAME=manyforge-e2e-vllm THOR_VLLM_PORT=8050 \
  ./serving/start-model.sh <profile-slug>

# 3. Wait for first-launch (NVFP4 JIT can take 60+ min on a fresh image — see
#    project_v81_first_launch_timing.md). Watch the log:
docker logs -f manyforge-e2e-vllm | head -200

# 4. Start the proxy with production caps:
OPENCLAW_PROXY_BIND=0.0.0.0 \
OPENCLAW_PROXY_LISTEN_PORT=8000 \
OPENCLAW_PROXY_UPSTREAM=http://127.0.0.1:8050 \
OPENCLAW_PROXY_LOG_PATH=/tmp/probe_proxy.jsonl \
OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS=2048 \
OPENCLAW_PROXY_THINKING_TOKEN_BUDGET=512 \
  nohup python3 manyforge/scripts/proxy/vllm-proxy.py \
    >> /tmp/probe_proxy_stdout.log 2>&1 &

# 5. Run targeted-9 (subset filter on the corpus):
cd manyforge
nohup python3 scripts/debug/smoke_corpus_runner.py \
  --corpus scripts/debug/smoke_corpus.yaml \
  --filter '^(P[123]_|TREE_|WRAP_|PARALLEL_|FALLBACK_|REPEAT_)' \
  --enable-recovery-turn \
  --report /tmp/probe_targeted9.json \
  > /tmp/probe_targeted9.log 2>&1 &

# 6. Watch verdicts stream live (runner is line-buffered):
tail -f /tmp/probe_targeted9.log
```

### F.2 Swap models without losing in-flight runs

The cleanest path is to stop the smoke, restart vLLM with the new profile,
restart bridge/gateway, then resume:

```bash
# Halt anything still running
pkill -f smoke_corpus_runner.py
pkill -f vllm-proxy
pkill -f openclaw_assistant_bridge.service

# Stop vLLM container, drop caches (Thor unified memory hygiene)
docker rm -f manyforge-e2e-vllm
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Restart with the new profile via the production launcher
cd ~/workspaces/dev_ws/src/manyforge
MODEL_PROFILE=<new-slug> ./scripts/demo-assistant-known-good.sh restart
```

In-flight Composer UI sessions will see "assistant timed out" on their last
request; new prompts work as soon as the bridge `/healthz` returns ok.

### F.3 A/B test a proxy knob (e.g., reflection threshold)

```bash
# Baseline run — keep the production defaults:
OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS=2048 \
OPENCLAW_PROXY_THINKING_TOKEN_BUDGET=512 \
OPENCLAW_PROXY_LOG_PATH=/tmp/iterA_proxy.jsonl \
  python3 manyforge/scripts/proxy/vllm-proxy.py … &
# (smoke run, save report)

# Treatment run — lower reflect_at to 3, keep stop_at at 8:
pkill -f vllm-proxy
OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS=2048 \
OPENCLAW_PROXY_THINKING_TOKEN_BUDGET=512 \
OPENCLAW_PROXY_LOOP_REFLECT_AT=3 \
OPENCLAW_PROXY_LOOP_STOP_AT=8 \
OPENCLAW_PROXY_LOG_PATH=/tmp/iterB_proxy.jsonl \
  python3 manyforge/scripts/proxy/vllm-proxy.py … &
# (smoke run, save report)

# Compare reports
python3 -c "
import json
a = json.load(open('/tmp/iterA_report.json'))
b = json.load(open('/tmp/iterB_report.json'))
print('A pass:', sum(c['status']=='pass' for c in a['cases']))
print('B pass:', sum(c['status']=='pass' for c in b['cases']))
"

# And cross-check the per-case loop-break events:
grep -E '(proxy_loop_reflection_injected|proxy_loop_hard_stop)' \
  /tmp/iterB_proxy.jsonl | jq -c '.event, .path' | sort | uniq -c
```

Change ONE knob at a time. The corpus is sensitive enough that two
simultaneous changes interact unpredictably (see `SMOKE-CORPUS.md` iter
33 negative result).

### F.4 Diagnose a runaway retry loop from proxy log

```bash
# Symptom: case timed out at 275 s with "chat HTTP -1" or "Tool failed" loop
LOG=/tmp/iterN_proxy.jsonl

# 1. Did the proxy short-circuit anything?
grep -E '(proxy_loop_reflection_injected|proxy_loop_hard_stop|proxy_upstream_error)' "$LOG" | jq .

# 2. What tools were being called repeatedly?
jq -c 'select(.request.path == "/v1/chat/completions") |
       [.request.body.messages[] | select(.role == "assistant") | .tool_calls // [] | .[].function.name]' "$LOG" \
  | sort | uniq -c | sort -rn | head

# 3. What was the last tool error before the loop?
jq -c 'select(.request.path == "/v1/chat/completions") |
       .request.body.messages | reverse | map(select(.role == "tool")) | .[0].content // empty' "$LOG" \
  | head -1

# 4. Did max_tokens get injected?
jq -c 'select(.request.mutation) | .request.mutation' "$LOG" | head
```

If the loop happens **before** the reflection-injection fires (count < 4),
either the threshold is too high for this corpus or the model is varying
tool names enough that the counter never accumulates. Lower
`OPENCLAW_PROXY_LOOP_REFLECT_AT` to 3 and re-run.

If the loop happens **across** Composer prompts (each prompt's history
shows the same tool but the per-conversation counter resets at proxy
level), the bridge's `OPENCLAW_ASSISTANT_LOOP_TOOL_THRESHOLD` is the
right defense — confirm it's `5` (default) and that
`bridge_synthetic_loop_break` events fire in the bridge log.

### F.5 Add a new model profile

Two files, same case label:

1. `serving/config.sh` — add a new `case` branch with `THOR_MODEL_PROFILE`,
   `THOR_MODEL_ID_DEFAULT`, `THOR_TARGET_MAX_MODEL_LEN`,
   `THOR_TARGET_KV_CACHE_DTYPE`, `THOR_TARGET_MAX_NUM_SEQS`,
   `THOR_TARGET_OPENCLAW_MAIN_MAX_CONCURRENT`,
   `THOR_TARGET_MODEL_REASONING`, `THOR_TARGET_MAX_TOKENS`,
   `THOR_TARGET_TOOL_CALL_PARSER` (if non-default),
   `THOR_TARGET_QUANTIZATION`. **Also set per-profile proxy tuning**
   (new 2026-06-01): `THOR_TARGET_PROXY_LOOP_REFLECT_AT` (default `"4"`),
   `THOR_TARGET_PROXY_LOOP_STOP_AT` (default `"8"`),
   `THOR_TARGET_PROXY_FORCE_ENABLE_THINKING` (`"on"` / `"off"` /
   `""` to leave the proxy untouched). Read
   [MANYFORGE-PROFILE-CALIBRATION.md](./MANYFORGE-PROFILE-CALIBRATION.md)
   for the sizing methodology.

2. `serving/launch.sh` — add the matching `case` branch with the actual
   vLLM args: `THOR_LAUNCH_MODEL_SOURCE`,
   `THOR_LAUNCH_GPU_MEMORY_UTILIZATION`, then a `THOR_VLLM_ARGS+=(…)`
   block with `--tool-call-parser`, `--reasoning-parser` (or omit),
   `--default-chat-template-kwargs`, `--override-generation-config`,
   `--enable-auto-tool-choice`, plus any model-specific flags.
   Don't forget `THOR_DOCKER_ENV_ARGS+=(…)` for env vars
   (`VLLM_USE_FLASHINFER_MOE_FP4`, etc.) if your weights need them.

3. **`serving/start-model.sh` proxy auto-restart** picks up the per-profile
   proxy tuning from step 1 automatically (lines 56-95). No edits needed
   unless you want to opt out (`THOR_RESTART_PROXY=0` — useful when a
   separate supervisor owns the proxy lifecycle).

4. Validate:
   ```bash
   ./serving/start-model.sh <new-slug>           # boots vLLM
   curl -s http://127.0.0.1:8000/v1/models | jq -r '.data[].id'   # → new-slug
   curl -s http://127.0.0.1:8000/v1/chat/completions \
        -H 'content-type: application/json' \
        -d '{"model":"<new-slug>","messages":[{"role":"user","content":"hi"}],"max_tokens":16}' \
     | jq -r '.choices[0].message.content'
   ```

5. Then drive a targeted-9 smoke (recipe F.1) to verify before adding to
   any benchmark sweep.

The slug must be identical in both files; `serving/start-model.sh`
errors out with `Unsupported model profile` if `config.sh` doesn't
recognize it. The `served-model-name` advertised by vLLM equals the
slug.

### F.6 Bake fixes into a new vLLM image

`serving/docker/` holds the Thor-specific build context. Active patches are
in `Dockerfile.thor*` and the bundled `mods/` overlay. Workflow:

1. Identify the upstream PR / commit that fixes your blocker (e.g.
   `sm110a-fp4-dsl-unlock` for the SM110 NVFP4 oracle gate).
2. Update `Dockerfile.thor*` to pull the fixed vLLM commit (or apply a
   patch overlay under `mods/`).
3. Build: `docker build -f serving/docker/Dockerfile.thor* serving/docker/`
4. Tag and update `VERSIONS.md` (single source of truth).
5. Smoke-run the production recipe (recipe F.7) before declaring done.
6. Update `serving/docs/PERFORMANCE-V*.md` with the dated outcome.

Half-day rebuilds are normal when a vLLM minor version ships
(see memory `project_trt_edge_llm_roadmap.md`).

### F.7 Run the full smoke corpus + interpret the report

```bash
# Stand up the production-default stack (Cosmos-8B + OpenClaw lane)
cd ~/workspaces/dev_ws/src/manyforge
./scripts/demo-assistant-known-good.sh start
./scripts/demo-assistant-known-good.sh smoke   # blocks on /healthz

# Run the corpus
cd ~/workspaces/nemoclaw/src/NemoClaw-Thor/manyforge
nohup python3 scripts/debug/smoke_corpus_runner.py \
  --corpus scripts/debug/smoke_corpus.yaml \
  --enable-recovery-turn \
  --report /tmp/iter_full.json \
  > /tmp/iter_full.log 2>&1 &

# Wall-clock: ~41 min chain-off, ~75 min chain-on with COMPACT_EVERY_N=2.
# Stream verdicts:
tail -f /tmp/iter_full.log
```

Interpreting the report (`/tmp/iter_full.json`):

```python
import json
r = json.load(open('/tmp/iter_full.json'))
status = [c['status'] for c in r['cases']]
print('total:', len(status))
print('first-try pass:', status.count('pass'))
print('effective (pass + recovered + clarified + soft):',
      sum(s in {'pass','recovered-pass','clarified-pass','soft-pass'} for s in status))
print('fails:', [c['id'] for c in r['cases'] if c['status']=='fail'])
```

Reference baselines (from [SMOKE-CORPUS.md](./SMOKE-CORPUS.md)):

| Iter | Recipe summary | First-try | Effective |
|------|---------------|-----------|-----------|
| 20 | chain-off, no recovery turn | 45/66 (68.2%) | 49/66 (74.2%) |
| 28 | chain-off, recovery turn, schema refactor | 49/66 (74.2%) | 51/66 (77.3%) |
| **32** | **chain-on + COMPACT_EVERY_N=2 + recovery turn (prior prod)** | **47/66 (71.2%)** | **51/66 (77.3%)** |
| 33+ (2026-06-01) | iter-32 recipe **+ synthetic OFF + Rule 11a + alt_names + consecutive-counting + 4 cases rebalanced to ASK** | (re-baselining in progress) | (re-baselining in progress) |

The 2026-06-01 corpus rebalance moves 4 cases (`PARALLEL_generic`,
`FALLBACK_generic`, 2 `category: clarification` cases at smoke_corpus.yaml
lines 1170/1196/1214) from "expect a tool call" to "expect a clarification
question". This couples the corpus to the default-synthetic-OFF design:
under the bypass, those 4 cases were guaranteed to pass via the canned
answer; under Rule 11a, they probe whether the model itself asks. Older
iter-N numbers are NOT directly comparable to runs after the rebalance —
re-baseline on each model when comparing recipes.

---

## G. Known-good production config (snapshot)

The single source of truth is
[`dev_ws/src/manyforge/scripts/lib/assistant.sh`](/home/tndlux/workspaces/dev_ws/src/manyforge/scripts/lib/assistant.sh).
The block below mirrors it for documentation purposes; if the launcher
changes, **the launcher wins**.

### G.1 Stack diagram

```
Composer (container) :9000          ─┐
openclaw_assistant_bridge :8200      │   ┌─ proxy log: /tmp/manyforge-assistant-e2e/vllm-proxy.jsonl
                                      ├─►│  bridge log: /tmp/manyforge-assistant-e2e/known-good-bridge.log
vllm-proxy :8000 (mutator + logger)  │   │  bridge audit: /tmp/manyforge-assistant-e2e/known-good-bridge-audit.jsonl
vLLM container :8050  cosmos-8b      ─┘
```

### G.2 Env vars (every load-bearing one)

```bash
# Composer
ASSISTANT_PROVIDER=openclaw
MODEL_PROFILE=cosmos-reason2-8b
START_VLLM_PROXY=true
ASSISTANT_TIMEOUT_S=300
DROP_CACHES=true
PROVISION_OPENCLAW_SANDBOX=true

# Proxy
OPENCLAW_PROXY_BIND=0.0.0.0
OPENCLAW_PROXY_LISTEN_PORT=8000
OPENCLAW_PROXY_UPSTREAM=http://127.0.0.1:8050
OPENCLAW_PROXY_LOG_PATH=/tmp/manyforge-assistant-e2e/vllm-proxy.jsonl
OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS=2048        # LOAD-BEARING
OPENCLAW_PROXY_THINKING_TOKEN_BUDGET=512       # production sweet-spot
# (loop defense uses defaults: REFLECT_AT=4, STOP_AT=8)

# Bridge
# REVISED 2026-06-03: USE_GATEWAY defaults to false because OpenClaw
# 2026.5.22 does not expose /v1/chat/completions on the gateway HTTP
# server. The bridge uses CLI shell-out (openclaw agent via nemoclaw
# exec) which works against 2026.5.22.
OPENCLAW_ASSISTANT_USE_GATEWAY=false           # cli_shell_out (route fix)
OPENCLAW_ASSISTANT_AGENT=manyforge-composer
OPENCLAW_ASSISTANT_LOCAL=false
OPENCLAW_ASSISTANT_TIMEOUT_S=300
OPENCLAW_ASSISTANT_BRIDGE_HOST=127.0.0.1
OPENCLAW_ASSISTANT_BRIDGE_PORT=8200
OPENCLAW_ASSISTANT_COMPACT_EVERY_N=2           # iter-32 chain-on enabler
OPENCLAW_ASSISTANT_COMPACT_TIMEOUT_S=120
OPENCLAW_ASSISTANT_METRICS_ENABLED=false       # turn on for /metrics
# (loop detector uses default LOOP_TOOL_THRESHOLD=5)

# vLLM (set via serving/launch.sh per-profile case branch)
--attention-backend flashinfer
--enforce-eager
--mm-encoder-attn-backend TORCH_SDPA
--kv-cache-dtype fp8
--max-num-batched-tokens 8192
--enable-auto-tool-choice
--tool-call-parser hermes
--override-generation-config '{"temperature":0.2,"top_p":0.95}'
--default-chat-template-kwargs '{"enable_thinking":true}'
--reasoning-parser qwen3
```

### G.3 Bring-up (single command per role)

```bash
cd ~/workspaces/dev_ws/src/manyforge
./scripts/demo-assistant-known-good.sh start         # boots vLLM + proxy + provisioner + bridge + supervisor
./scripts/demo-assistant-known-good.sh smoke         # blocks on /healthz, full readiness probe
./scripts/demo-assistant-known-good.sh stop          # tears everything down + drop_caches
```

To swap to the **direct lane** (fast-path for simple prompts only):

```bash
ASSISTANT_PROVIDER=nemoclaw ./scripts/demo-assistant-known-good.sh restart
```

---

## H. Session learnings (2026-05-31 / 2026-06-01)

Recorded under `/tmp/35b-iter-log/rounds/`. Round structure:

| Round | Hypothesis | Result |
|-------|-----------|--------|
| **0** | Cosmos-8B parser-only fix (qwen3 reasoning parser; without it, thinking bled into content) | Necessary precondition; alone insufficient to unblock the 9 hard cases. |
| **1-2** | Per-profile sampling overrides (temperature, top_p tweaks) | Neutral on the hard set; baseline TEB unchanged. |
| **3.1** | "Thinking fully on" (top-level + chat_template_kwargs mirror) for cosmos-8b | Discovered the top-level-precedence bug; the mirror landed in `vllm-proxy.py:285-292`. Cleared the silent thinking-off regression. **Wired through per-profile `THOR_TARGET_PROXY_FORCE_ENABLE_THINKING` on 2026-06-01 so each profile boots with its preferred posture.** |
| **3.2** | Repro that the proxy mirror actually fires for cosmos under Composer's `enable_thinking:false` payload | Confirmed via `proxy.jsonl` mutations field. |
| **4** | Bigger thinking budget (1024 → 2048) | Net-negative on cosmos-8b; 512 stays the sweet-spot. |
| **5** | `OPENCLAW_PROXY_FORCE_TOOL_CHOICE=required-first` | Aborted — interacted poorly with chain-on session memory; reverted to no force. |
| **6** | `OPENCLAW_PROXY_USER_MESSAGE_SUFFIX` plan-then-execute nudge | Ignored by the model on first-turn action prompts. Suffix deprecated for this stack. |
| **7** | Bridge synthetic clarification short-circuit for `add a <kind>` | Lands the answer-text rubric for the 4 affected smoke cases; very narrow gate intentionally. **Default-INVERTED to OFF 2026-06-01** so smoke fairly measures cross-model clarification quality. Opt-in via `OPENCLAW_ASSISTANT_ENABLE_SYNTHETIC=1`. |
| **8** | Bridge cross-turn `OPENCLAW_ASSISTANT_LOOP_TOOL_THRESHOLD` fail-fast | Stops the 25-28-turn same-tool runaway pattern. Tight defaults (5). |
| **10** | Proxy cascading loop defense (reflect_at=4 + stop_at=8) | Bounds total time-per-case at ~60-80 s in the worst loop; lets the model recover once after seeing the reflection prompt. **2026-06-01 follow-up: CONSECUTIVE tail-run counting replaces lifetime counting** — fixes false-positive hard-stops on chain-on smoke (every PnP_NN case after the 3rd was being shut down on turn 1). |
| **11** (2026-06-01) | Rule 11 NO_REPLY guard + Rule 11a Missing-WHERE in `adapter.py:513-536` | Prompt-side replacement for the synthetic bypass — instructs the model directly to ask "which parent / where in its children?" on `PARALLEL_generic` / `FALLBACK_generic` / `UPDATE_params_generic` patterns. Lets the smoke corpus fairly measure cross-model clarification quality. |
| **12** (2026-06-01) | Corpus rebalance: 4 cases switched from "expect tool call" → "expect ASK" | Couples the corpus to the default-synthetic-OFF design. Cases that used to pass via the canned bypass answer now probe whether the model itself asks via Rule 11a. |
| **13** (2026-06-01) | Smoke runner `alt_names` support (`smoke_corpus_runner.py:391-424`) | Cases declare equivalent-effect tool aliases; soft-pass when an alt fires with 2xx. Stops false-fails on legitimate variant tool choices (e.g. `scene_draft_upsert_objects` for `scene_draft_add_object`). |
| **14** (2026-06-01) | Schema fix: `node.name` accepted in `tree_draft_insert_node` payload | Stops validator death-spirals when models duplicate `nodeName` into `node.name` (legacy shape from `_TREE_NODE_SCHEMA`). Handler still reads top-level `nodeName`; `node.name` is accepted-and-ignored. |

**Per-model takeaways**:

- **Cosmos-Reason2-8B** is the production winner. Thinking-on + qwen3 reasoning
  parser + proxy mirror + max_tokens=2048 + thinking_budget=512 + compact-every-2.
  9/9 on the lane-parity probe; 51/66 (77.3%) on the iter-32 smoke corpus.
- **Nemotron-3-Nano-Omni-30B-A3B** (instruct, non-reasoning) was the 2026-04-30 candidate
  but lost the lane-parity head-to-head 0/9 vs cosmos-8b 9/9. The reasoning variant has not been
  retried since the bridge would need to read `reasoning` (not `content`).
- **Qwen3.6-35B-A3B-NVIDIA** + MTP K=3 + Marlin MoE wins quality (56/66) vs RedHat-quant
  (53/66) at 65 min vs 82 min, but at 8B-class-quality-only level after the round-3.1 cosmos fix.
- **Nemotron-3-Nano-4B-BF16** is the NVIDIA Jetson Thor explicit default per the HF card
  but ranks BFCL v3 = 61.1 — adequate for tool-call but not for ManyForge's composer-assistant
  pattern complexity.

---

## Glossary

- **Bridge** — the HTTP service that consumes Composer's
  `manyforge.assistant.provider_request.v0` envelope and runs (or delegates) the
  LLM agent loop. Two implementations: `openclaw_assistant_bridge` (gateway-delegating, production) and `manyforge_assistant_bridge` (in-process loop, fallback).
- **Lane** — one of the two bridge implementations.
- **Mutator** — the `vllm-proxy` running with mutation env vars set.
- **Direct lane** — `manyforge_assistant_bridge` on `:8100`; runs the agent loop in-process.
- **OpenClaw lane** — `openclaw_assistant_bridge` on `:8200`; delegates to the OpenClaw gateway in the sandbox.
- **Chain-session** — Composer's per-conversation session key. When the smoke runner reuses it across chain steps (default), OpenClaw retains prior turn history.
- **Session key** — `derive_gateway_session_key(payload)` = `conversationId + catalogHash + programRevision`. Bridge counters (compact, loop) are keyed by this.
- **Compact** — OpenClaw's `/compact` slash-command. Rolls the agent's accumulated conversation up into a summary; preserves chain memory while preventing context overflow.
- **Reflection injection** — proxy mutation that adds a `[loop-reflection]`-marked user message after N same-tool calls, urging the model to change tactics. Forwarded to the model.
- **Hard stop** — proxy mutation that synthesizes an SSE assistant response after M same-tool calls, no GPU spend. OpenClaw treats it as a normal text completion.
- **Synthetic clarification** — bridge-side short-circuit for very narrow `add a <kind>` prompts; returns a canned "which parent? which position?" without invoking OpenClaw.

---

## Maintaining this doc

This file is the **architectural entry point** for the assistant pipeline. Keep it durable:

- When a new round / iter introduces a knob, document it in §B (with default + when-to-use) and in §C, §D, or §E as appropriate.
- When a profile lands or retires, update §E and `serving/config.sh` / `serving/launch.sh` comments in lockstep.
- Date-specific results live in [SMOKE-CORPUS.md](./SMOKE-CORPUS.md) and dated `serving/docs/PERFORMANCE-V*.md` files — not here.
- Operational symptoms + their gates live in [COMPOSER-ASSISTANT-RUNBOOK.md](./COMPOSER-ASSISTANT-RUNBOOK.md). When that runbook adds a new gate, link it here only if it's about an architectural component (not a one-off incident).
- Per memory `feedback_agents_md_durability.md`, transient incident docs do **not** belong in this file's reference table. Add findings here only when they have outlived the incident.
