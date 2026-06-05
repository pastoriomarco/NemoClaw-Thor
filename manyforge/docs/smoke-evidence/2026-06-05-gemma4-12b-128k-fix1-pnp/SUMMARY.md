# PnP chain re-run — gemma-4-12b-it (GGUF) — validates bridge fix #1, exposes root-cause #2

**Date:** 2026-06-05 (afternoon; follows the morning 3-lane matrix in
`../2026-06-05-gemma4-12b-it-gguf/`).
**Scope:** `openclaw-tools` lane, **PnP subset only** (`--filter ^PnP_`, chained
session). This run targets the pick-and-place chain that produced the
"16-case cascade" in the morning matrix (finding 3 there).

**Config delta vs the morning matrix:**

| | morning matrix | this run |
|---|---|---|
| context | 64K | **128K** (`THOR_LLAMACPP_CTX`) |
| KV cache | default | **q8_0 K+V + flash-attn on** |
| mutator proxy | off (raw llama.cpp) | **on** (vLLM-style mutator `:8000`) |
| composer / bridge timeout | 180s / 170s | **320s / 300s** (matches direct lane) |
| bridge loop-history | persists across turns | **cleared on successful (200) turn — fix #1** |

Fix #1 = `openclaw_assistant_bridge/service.py` `_loop_history_clear()` on the
200 success path (committed `7a14274`).

## Result

`openclaw-tools`, PnP chain, 300s budget, fix #1:

**9 pass / 10 fail / 1 future-skip → effective 9/19 (47.4%).**

| case | result | dur | note |
|---|---|---|---|
| PnP_01..05 (scene + tree_root) | ✅ ×5 | 26–49s | |
| PnP_06_approach | ✅ | 77.4s | **recovered** (morning: instant-502 cascade) |
| PnP_07_descend | ✅ | 263.4s | recovered; needed >180s — *would have failed the old cap*. Final answer was itself a **gateway 409 loop**, but the required `insert_node` landed 2xx earlier so the assertion passed |
| PnP_08_close_gripper … PnP_17_home | ❌ ×10 | 35–121s | all: `expected tool 'tree_draft_insert_node' not observed (or never reached 2xx)` |
| PnP_18_repeat_root | ✅ | 61.8s | root-level op (`tree_draft_wrap_node`) |
| PnP_19_safety_parallel | ⏭ future | | |
| PnP_20_grip_force | ✅ | 76.8s | param op (`tree_draft_update_node_params`) |

## Finding 1 — bridge loop-history cascade is ELIMINATED (fix #1 validated)

Across the entire chain: **`bridge_loop_detected_stop` = 0**;
`bridge_loop_history_cleared_on_success` = 10 (exactly one per passing case;
failing cases never reach the 200 path, so genuine repeatedly-failing-loop
detection is preserved).

This **corrects finding 3 of the morning matrix.** That "16-case
gateway-session cascade" of *instant* (0.0–0.1s) HTTP 502s was **not** a gateway
session crash — it was the bridge's own per-conversation loop detector: keyed
only on `(conversationId, assistantMode)` and persisting across turns, it
accumulated fingerprints from the legitimately-repeated PnP tool family and
tripped `bridge_loop_detected_stop` mid-chain, after which every later chained
turn short-circuited (409 → composer 502) **without invoking the model**. With
fix #1 the cases now actually run (35–263s of real work), not 0.1s.

## Finding 2 — root-cause #2: gemma emits a MALFORMED tool id (dash-mangled / unprefixed), NOT a missing id

With the cascade gone, the PnP_08–17 block fails for a **genuine model reason**.
The model's *self-explanation* (in 8 of 10 final answers) is:

> "the `tool_call` was being sent without the required `id` field … the OpenClaw
> tool dispatcher requires the tool name to be passed as the `id`"

**This self-diagnosis is wrong — a hallucination.** The raw gateway log
(`raw-rejected-tool-ids.log`) shows the `id` is **present in every rejected
envelope**; it is just **malformed**:

| what gemma emitted | canonical (accepted) form | defect |
|---|---|---|
| `manyforge__scene-draft-add-object` | `manyforge__scene_draft_add_object` | segments kebab-cased |
| `manyforge__tree-draft-insert-node` | `manyforge__tree_draft_insert_node` | segments kebab-cased |
| `manyforge__tree-draft-wrap_node` | `manyforge__tree_draft_wrap_node` | mixed dash + underscore |
| `tree_draft_insert_node` (code surface) | `manyforge__tree_draft_insert_node` | `manyforge__` prefix dropped |

e.g. `tool_call failed: Unknown tool id: manyforge__scene-draft-add-object
raw_params={"id":"manyforge__scene-draft-add-object","args":{…}}` — id present,
args well-formed, only the **separator/prefix** is wrong. The dispatcher does an
exact-match lookup on the canonical double-underscore id → `Unknown tool id` →
never 2xx → the model retries the same mangled id → the **in-sandbox OpenClaw
gateway** (a *different* loop detector from finding 1's bridge one) returns
`409 loop detected: tool 'tool_call' repeated 8 times`.

This is **not** timeout (35–121s ≪ 300s), **not** the bridge cascade (0
occurrences), and **not** tree-targeting / chain-state degradation — root-level
ops (PnP_18) and param ops (PnP_20) pass *after* the failing block, ruling out
context poisoning. The failures track the **tool-id string the model emits**,
not chain position. The 6/6 early passes are cases where gemma happened to emit
the canonical id (and/or landed the required call before drifting to a mangled
one — PnP_07 passed despite ending in a gateway 409).

### Worse sub-variant: hallucinated completion (PnP_14, PnP_17)

In 2 of the 10, the model gives up on tool calls and emits a confident prose
"the program is complete / fully built" final answer **without ever landing the
inserts**. Silent fabricated success — more dangerous than an honest error.

## Fix #2 — REFRAMED: the normalizer already exists but never fired

The "narrow id normalizer at the dispatch boundary" already exists in the
proxy: `scripts/proxy/vllm-proxy.py` `_NORMALIZE_NESTED_MCP_IDS`
(`OPENCLAW_PROXY_NORMALIZE_NESTED_MCP_IDS`, default-on, added 2026-06-04). It
rewrites nested ManyForge ids (bare / dashed / MCP-locator) to canonical
`manyforge__<underscored>` via text-level regex over the response body before
OpenClaw's dispatcher parses the tool call. The bridge also carries a primer
rule (`adapter.py` Rule 0b) telling the model to copy the `name` field verbatim.

**Yet on this gemma run it never fired.** Proxy mutation log
(`/tmp/manyforge-assistant-e2e/vllm-proxy.jsonl`, 865 mutation records):
**zero** nested-id rewrite-rule hits (`strip_mcp_locator` / prepend / dash→
underscore), while the mangled forms reached the wire anyway —
`scene-draft-add-object` ×139, `program-read` ×288 (canonical forms also present
×870/880; gemma emits both). The proxy IS in the path: the in-sandbox gateway's
`baseUrl` is `http://host.openshell.internal:8000/v1` → proxy `:8000` → model
`:8050`.

So fix #2 is **debug-and-extend, not greenfield.** Three concrete defects:

1. **Non-firing on the response path.** `manyforge__scene-draft-add-object`
   *should* match `_NESTED_DASHED_ID_PATTERN` (`scene-draft[-_]…`) yet produced
   0 rewrites. Either the response-side rewrite is not applied under the
   `compat` proxy profile / streaming shape, or gemma emits the tool call in a
   shape the regex misses (e.g. id inside chat *content*, not the escaped
   `\"id\":\"…\"` tool-call arguments string). **Needs the actual response body
   inspected to see the exact wire shape of the id.**
2. **Pattern gap — flat/non-draft dashed tools.** The dashed pattern only
   handles `<surface>-draft-…`; flat tools (`program_read`, `scene_inspect`,
   `catalog_read`, …) are covered only in underscore form, so
   `manyforge__program-read` (×288) slips through entirely.
3. **Code mode entirely unprotected.** In `tool_search_code` the id is a JS
   string literal (`openclaw.tools.call('tree_draft_insert_node', …)`), not a
   `"id":"…"` JSON field, so no pattern fires. This is why the **code lane
   scored 1/19 (5.3%)** — only PnP_01 passed; PnP_02–20 fast-failed in 6–12s.

Recommended sequence: (a) capture one raw *response* body containing a mangled
id to fix defect 1 at the source; (b) extend the dashed/bare patterns to flat
tools (defect 2); (c) add a code-surface JS-string id rewrite (defect 3) or
accept code mode as unviable for gemma. Each rewrite already logs an
original→rewritten pair for audit.

Alternatives: **primer hardening** (Rule 0b already exists; gemma ignores it) is
proven weak. **Accept as a gemma limitation** and prefer the **direct lane**
(no gateway, no nested-id ABI, PnP-immune) remains a legitimate fallback if the
proxy rewrite can't be made to fire reliably.

## Reproduce

`run-cell-openclaw.sh` (archived here) with `MODE=tools FILTER='^PnP_'`. Model
serving on `:8000` (128K, q8_0 KV, FA), mutator proxy on, composer + bridge at
320s/300s. Artifacts: `pnp-300-tools-report.json`, `pnp-300-tools-stdout.txt`,
`bridge-openclaw-tools.log`.
