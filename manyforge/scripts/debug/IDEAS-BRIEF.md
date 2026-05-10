# Brief for an Outside LLM — "Help Me Find New Cross-Cutting Fixes for the ManyForge Smoke Corpus"

> Paste this entire document into a fresh chat with a capable model
> (Claude Opus, GPT-4-class, etc.) and ask the question at the bottom.
> The brief is self-contained — the receiving model has no access to
> the codebase. Treat its suggestions as candidates to evaluate against
> the constraints, not directives to apply.

---

## What is being tested

ManyForge Composer is a robot-program authoring UI with an embedded
"Composer-assistant" that takes natural-language requests ("add a
move-to-pose that approaches above the graspable") and turns them into
mutations to a behavior tree, scene, parameters, or blackboard. The
assistant runs through an OpenClaw gateway → bridge → vLLM stack.
Active model: `cosmos-reason2-8b` at temperature 0.2.

The smoke corpus is a 74-case YAML suite (66 active + 8 future) that
dispatches each case as a `/api/assistant/chat` request and asserts on:

- `tools_called` — multiset of tools the assistant called, each with optional `args_contain` and `allow_retries`
- `state_after` — dotted-path asserts against `/api/program` + `/api/program/tree` + `/api/scene/state`
- `forbidden_tools`, `answer_must_contain`, `answer_must_not_contain`

A case is **pass** if all asserts hold, **soft-pass** if only the
answer-text assert held, **fail** otherwise.

Cases are tagged `detail_level: specific | medium | generic`. Specific
cases give the model concrete coordinates / names; generic cases are
intentionally vague to test bounded autonomy (does the model ask, or
does it fire mutations on a pronoun?).

A subset of cases form same-conversation **chains** (`chain_id` +
`chain_step`). The flagship chain is `pnp_build` (PnP_01 → PnP_20),
which builds a pick-and-place program from an empty fixture.

## Architecture cheat-sheet

```
user prompt → Composer UI → /api/assistant/chat → OpenClaw bridge →
gateway (in sandbox) → vLLM (cosmos-reason2-8b) → tool calls →
MCP wrapper validator (in bridge) → /api/assistant/bridge/tools/<name>
on Composer → mutation applied → state snapshot returned → next turn.
```

Two key behaviors of the bridge:
- **MCP wrapper validator** rejects malformed tool calls with HTTP 400 + a structured `detail` payload (e.g. `validParentNames: [...]`, `allowedNodeKinds: [...]`).
- **STUCK_LOOP_HINT**: after **3 identical-arg failures** of the same tool, the bridge injects a synthetic message inlining the structured recovery fields in plain prose. Below 3 retries, the model only sees the JSON envelope.

## Hard constraints — non-negotiable

1. **No commits during the iteration cycle.** Stage and re-evaluate each round.
2. **No per-prompt hints to the model.** Prompts must be user-style natural language. No "use parentName=pick_and_place" baked into the prompt.
3. **No tightening of the model's RULES system message** to fix smoke-corpus failures. RULES is shared production surface.
4. **Fixes must be cross-cutting.** Corpus structure / harness logic / deployment YAML / timeouts are fair game. Per-case workarounds are not.
5. **Specific > generic.** Specific cases are the priority. Generic cases that genuinely warrant clarification can stay yellow.
6. **Corpus must remain extensible.** No one-off helpers tied to a single case.
7. **Schema examples in tool descriptions are excluded.** Past attempt: model copy-pasted example `objectId` verbatim → 116-call upsert loop.
8. **Bridge / Composer code changes are out of corpus scope** unless explicitly authorized.
9. **Timeouts can be tuned**, but an upstream 502 fires around 5 min of model loop — past that the request is killed regardless.
10. **Always restart the demo between iterations** to clear zombie sessions.

## What was tried — five-iteration scoreboard

Each iteration applied one or more cross-cutting changes and re-ran the
full 66-case active set. Demo restart between iterations.

| Iter | Pass (effective) | Rate | Cross-cutting change |
|---|---|---|---|
| 1 | 34/66 | 51.5 % | harness gate fix (expanded_node_allowlist false-skip) + tools_called semantic split (missing field vs `[]`) + default timeout 140 → 240 s |
| 2 | 31/66 | 47.0 % | (regression) PnP_05 reframed as "rename root, expecting `tree_draft_update_node_params`" — model couldn't reliably emit the rename; broke the chain anchor |
| 3 | 36/66 | 54.5 % | iter-2 rolled back; pre-named the empty-fixture root `pick_and_place` so PnP_06+ has a real anchor on turn 1; state_after pose path `pose.position_m` → `pose_in_universe.position_m` |
| 4 | 39/66 | 59.1 % | harness multiset tool match (was order-sensitive — was false-failing when model fired the right tools in a slightly different order) |
| 5 | 51/66 | **77.3 %** | repurposed PnP_05 from no-op rename to "verify program state, expecting `program_read`"; default timeout 240 → 360 s |

The PnP_05 fix in iter 5 unblocked all 12 of PnP_06–PnP_17 simultaneously.

### What worked
- **Tool-use mode persistence**: making PnP_05 a real tool call (any
  read-only tool) kept the model in act-mode through the chain.
  Without it, the no-op text turn put the model into narration mode
  for the next 12 turns.
- **Multiset matching**: removed false fails from order-sensitive
  matching when model fired the right tools in a different order.
- **Fixture pre-rename**: pre-naming the root `pick_and_place` removed
  a flaky rename step the model couldn't reliably emit.
- **State-key drift discovery**: `pose_in_universe.position_m` is the
  live response shape; the corpus had been asserting on the wrong key.

### What regressed
- Iter 2 made PnP_05 require a rename call → cosmos-8b couldn't fire
  the right tool reliably → cascade-broke the chain. Lesson: don't
  ask the model to do something it can't already do.
- Iter 5 timeout 240 → 360 s caused 3 cases to hit a 5-minute
  upstream-502 circuit-breaker (PnP_18, FALLBACK_alternate_medium,
  REPLACE_subtree_specific). Lesson: stay below 270 s default; use
  per-case overrides only.

## Remaining 15 failures (iter 5 final)

Grouped by root cause. Numbers are the count we want to reduce.

### Group A — model fires right tool but malformed args (5 cases)
`INSERT_position_first_specific`, `TREE_insert_runtime_medium`, `MOVE_reorder_medium`, `FALLBACK_retry_specific`, `WRAP_root_medium`

**Symptom**: model emits `tree_draft_insert_node` with missing
`parentName`, or with the JSON wrapped in a `_raw` envelope (parser
failure on the model output), or with the wrong `targetName`. Bridge
returns HTTP 400 with `validParentNames` / `validNodeNames` in the
detail. Model paraphrases the error in text instead of retrying. After
3 identical retries the bridge would inject `STUCK_LOOP_HINT`, but the
model usually doesn't repeat the exact same failure call 3 times — it
mutates slightly and so the dedupe doesn't fire.

### Group B — bounded-autonomy gap on generic prompts (6 cases)
`PARALLEL_concurrent_medium`, `PARALLEL_generic`, `FALLBACK_generic`, `MOVE_generic`, `CLARIFY_scene_remove_pronoun`, `CLARIFY_tree_wrap_pronoun`

**Symptom**: prompt is intentionally vague ("rerun anything that fails up to 3 times" or "add a parallel"). Model fires mutations 3-15× when the expected behavior is to ask "which node?" / "which tools?". This is genuine model behavior on cosmos-8b. Deprioritized — these stay yellow.

### Group C — upstream 502 timeout casualties (3 cases)
`PnP_18_repeat_root`, `FALLBACK_alternate_medium`, `REPLACE_subtree_specific`

**Symptom**: case ran 299 s and hit `chat HTTP 502` from upstream. Caused by iter 5's 240 → 360 s default bump. Easy fix: revert default to ~270 s, add `chain_timeout_s: 360` per-case only where genuinely needed. Expected recovery: all 3.

### Group D — single-case bounded-autonomy on specific prompt (1 case)
`CUR_runtime_remove_then_restore_graspable`

**Symptom**: case asks the model to remove a graspable then restore it. Expected `scene_inspect` first, then mutations. Model skips inspection. Could try the PnP_05-style trick (precede with a chain step that requires inspection), but the case isn't currently chained.

## Existing inventory of fixes you should NOT re-suggest (already considered)

- Tighten RULES system message — forbidden (constraint 3).
- Per-prompt hints — forbidden (constraint 2).
- Schema examples in tool descriptions — actively harmful (constraint 7).
- Bigger model — out of scope; corpus measures cosmos-8b on purpose.
- Lower the bridge's `STUCK_LOOP_HINT` trigger from 3 → 1 — interesting but requires bridge code change (constraint 8). Worth flagging but not the primary ask.
- Force `tool_choice: "required"` always — already enabled by the bridge for tool-targeted prompts.
- Add `tree_inspect`-style turn before every INSERT case — would help Group A but feels like per-case workarounds dressed up; might violate constraint 6 (extensibility) if applied widely.

## What we want from you

We're at 51/66 (77.3 %) under the constraints above. We've identified
~3-4 cheap recoveries via the timeout revert (Group C). Beyond that,
we're hitting a wall.

**Question**: given the constraints, the iteration history, and the
remaining failure groups — what *cross-cutting* ideas haven't we
considered that could lift Group A (5 cases, malformed args) and/or
Group D (1 case, bounded-autonomy on specific) without crossing the
constraint lines? Generic ideas about Group B are also welcome but
lower priority.

Be concrete: name the change (corpus structure / harness logic /
deployment YAML / fixture / timeout policy), name the cases it would
target, predict the expected pass-rate delta, and call out the risk of
regression on the other 51 currently-passing cases.

Brainstorm wide first, then prioritize. We will evaluate each
suggestion against the constraints; ideas that violate a constraint
are still useful to hear *if* the violation is small and worth
discussing — just call it out explicitly.
