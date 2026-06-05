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

## Proposed fix #2 (for discussion — NOT yet implemented)

The earlier "infer `id` = tool name" idea is **rejected**: the `id` is present,
and the top-level OpenAI function name is just `tool_call` (the ManyForge tool
id lives in the envelope's `id`/`args`), so there is nothing to infer — the
present id is merely mis-separated. The evidence supports a **narrow, tested id
normalizer at the dispatch boundary** (protocol repair, not intent shaping):

- Scope: OpenClaw discovery `tool_call` / `tool_describe` / `tool_search_code`
  id resolution only.
- Canonicalize the supplied id: replace `-` with `_` in the tool segment
  (preserve the `manyforge__` double-underscore prefix); if a bare known tool
  name is given, prepend `manyforge__`.
- Resolve against the known tool registry; **repair only if it maps to exactly
  one known tool.** Reject on unknown / ambiguous / conflicting shapes
  (no silent guessing).
- Log every repair as `tool_call_id_repaired` (from→to) for auditability.

**Before writing it:** confirmed the failing shape from one raw envelope
(`raw-rejected-tool-ids.log`); the next step is to locate the exact dispatch
reject point (the `Unknown tool id` thrower) and add the normalizer there with
a unit test over the captured id variants.

Alternatives considered: **primer hardening** (inject the canonical-id rule) is
weak/model-specific; **accepting as a gemma limitation** (prefer the direct
lane, which has no gateway and is PnP-immune) is premature given a clean
protocol-repair path.

## Reproduce

`run-cell-openclaw.sh` (archived here) with `MODE=tools FILTER='^PnP_'`. Model
serving on `:8000` (128K, q8_0 KV, FA), mutator proxy on, composer + bridge at
320s/300s. Artifacts: `pnp-300-tools-report.json`, `pnp-300-tools-stdout.txt`,
`bridge-openclaw-tools.log`.
