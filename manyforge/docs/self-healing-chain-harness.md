# Self-Healing Chain Harness — design, plan & documentation

> Status: **implemented for PnP (2026-06-07); end-to-end model-driven validation pending.**
> The generator, `self_heal.py`, the `--self-heal` runner hook, and the PnP golden
> trajectory are built and unit-validated against the live stack. This file is both the
> design record and the §9 mechanism documentation. See §11 for the as-built details.

Scope: the **smoke test harness only** (`manyforge/scripts/debug/`). No live
component (bridge / composer / openclaw / proxy) is modified. The
"fabricate-success" capability described here exists **only in test code** and is
unreachable in production. This is a hard invariant (see §10).

---

## 1. Why this exists (motivation)

Chained corpus cases — primarily the `PnP_*` pick-and-place build (and any case with
a `precondition.chain_id`) — are a **sequence that shares one program + one
conversation**. Each step builds on the previous step's *actual* output:
`precondition: { chain_id: pnp_build, chain_step: N }`, and the corpus requires the
steps to "execute in order on the SAME program" (the runner reuses the first step's
`conversationId` for all steps).

Consequence: **if an early step fails, every downstream step inherits a
broken/incomplete program** and is set up to fail through no fresh fault of its own.
The per-case effective rate then conflates two very different things:

- genuine per-step capability ("can the model do step N from a correct state?"), and
- inherited-state cascade ("step N had no chance because step N-1 left garbage").

Observed directly in the 2026-06-06 `nemotron3-nano-4b-gguf` run: PnP front-half
(01–10) passed 9/10, back-half (11–18) ~1/7, and the deep analysis showed several of
those back-half misses were *not* clean capability failures. The existing
`MANYFORGE_SMOKE_DETECT_CHAIN_CONTAMINATION` detector only catches **one narrow
type** of contamination (the bridge's loop-break sentinel poisoning the shared
conversation — failures at ~0.1s with the model never invoked), not the
program-state cascade, and it is **off by default**.

**Goal:** measure *true per-step capability*. Each chained step should be evaluated
from the **canonical (correct) cumulative state**, and the model should **not even
notice that a prior step failed** — so its behaviour on step N+1 is not perturbed by
seeing its own failure or a visible "correction". We want the model to experience an
unbroken, apparently-successful build.

---

## 2. What it does (behaviour)

After each chained step the runner already runs the case assertion. We add:

- **Pass** → continue normally (real state, real session). Genuine multi-turn flow is
  preserved while the model is succeeding.
- **Fail** → *self-heal* before the next step:
  1. **State**: reset the live program+scene to the **canonical post-N** state.
  2. **History**: **splice a golden turn** into the openclaw session transcript so it
     reads as if step N succeeded.
  3. The model continues the **same conversation**, never seeing the failure.

This isolates each step's capability while keeping the run a coherent multi-turn
chain.

---

## 3. Key findings (validated during design)

1. **State reaches the model via per-turn snapshot re-injection.** The bridge
   (`openclaw_assistant_bridge/adapter.py`, `build_agent_prompt`) injects the current
   `programSnapshot` + `sceneSnapshot` into the **system prompt on every turn**,
   sourced from the composer's *live* state. → Resetting the live state is sufficient
   for the model's view of state; history is secondary for state purposes.
2. **Conversation history lives in openclaw's on-disk session store**, not the
   bridge. The bridge runs `openclaw agent --session-id <id>` (via docker/kubectl
   exec into the sandbox); the transcript persists at
   `/sandbox/.openclaw/agents/<agent>/sessions/<conversationId>.jsonl`
   (+ `<…>.trajectory.jsonl` trace, + `<…>.jsonl.lock`, + `<…>.checkpoint.*.jsonl`
   when compacted). A chained PnP run is **one growing `.jsonl`** (all steps reuse the
   first step's id). The runner already knows the `conversationId`.
3. **🟢 Feasibility spike PASSED (make-or-break).** A turn was sent, a *fabricated*
   exchange was injected directly into the `.jsonl`, and a follow-up turn was sent on
   the same `conversationId`. The model recalled the injected-only fact (`ZEBRA-9`),
   proving **openclaw resumes from a runner-edited `.jsonl` and the model treats
   injected history as genuine.** In-place edit (same id) sufficed.
4. **Transcript schema captured** (see §7).

---

## 4. Design decisions

- **Replay-from-base on failure** (not stored snapshots). Single source of truth:
  the **base PnP program** + an **ordered list of golden changes** (complete
  tool-calls, one per step). To reach canonical post-N, apply golden changes 1..N to
  the base via the **real composer tools**. Coherent by construction
  (`state[N] = state[N-1] + step N's canonical mutation`), structurally identical to
  real tool output (so later prompts that reference "the approach node" find exactly
  what they expect), and matched to the prompts (the golden actions *are* the
  mutations the prompts describe). Replay cost is negligible (~2 ms/tool POST).
- **Golden changes drive BOTH** state replay AND the JSONL splice → state and history
  are **mutually coherent** (the transcript says "I did X" and the program is
  base+changes[1..N]). One source means they cannot drift.
- **Coherence gate**: while generating the trajectory, each golden change applied to
  the prior golden state must satisfy that step's own `state_after` assertion. If it
  doesn't, the corpus is internally inconsistent (a defect to fix, e.g. the P2 /
  INSERT_position_first_specific class). Generation doubles as corpus validation.
- **Test-only boundary**: all logic in the smoke harness; zero live-component edits.

---

## 5. Architecture / data flow

```
smoke_corpus_runner.py  (TEST-ONLY)
  │  send_chat → POST {composer}/api/assistant/chat  {conversationId, message, …}
  ▼
composer (:9000) ──► bridge (/v1/manyforge/assistant) ──► openclaw agent --session-id
  ▲                         │  per-turn programSnapshot+sceneSnapshot → system prompt
  │                         ▼
  │                   openclaw session .jsonl  (in sandbox)  ◄── runner splices here on fail
  │
  └ state mutations via /api/assistant/bridge/tools/<tool>  (reset_program + apply_fixtures)
```

On failure the runner (a) drives the canonical state through the existing tool
endpoints (replay-from-base), and (b) edits the openclaw `.jsonl` in the sandbox via
`exec`, under the `.lock`, between steps (no openclaw process running for that
session at that moment).

---

## 6. Implementation plan (build order)

- **A. Golden-trajectory generator** (`tools/golden_trajectory.py`, test-only):
  given the base + per-step golden changes, replay through the real tools,
  **assertion-gate each step**, and emit validated golden-change specs (+ optionally
  materialized snapshots for debugging). Getting it running also resolves §item-set
  below (base + scene-clear).
- **B. Splice-turn renderer**: golden change → the three parent-linked `.jsonl`
  entries (`assistant: thinking?+toolCall` → `toolResult` → `assistant: text`) per
  the §7 schema. Handles the compacted-session edge.
- **C. Runner self-heal** in `run_case` / the chain loop: track per-chain prior-step
  result; on fail → replay-from-base to canonical state + splice golden turn into the
  session `.jsonl`; **delete the loop-poison detector** (subsumed).
- **D. Golden changes (content)**: author complete golden tool-calls per chained
  step. **Start with `PnP_*` (chain_id `pnp_build`)** as the proving ground; then
  enumerate and cover any other chains. (Authored by us — the corpus author — since
  the corpus only carries partial `args_contain` matchers today.)
- **E. Validate end-to-end on PnP**: force a mid-chain failure and confirm downstream
  steps are evaluated from canonical state, the model doesn't notice, and the
  effective rate reflects per-step capability.

### Open items to resolve while building
- **Base + scene-clear**: base = PnP_01 `fresh_program` (empty program) +
  deployment. Confirm `empty_program_path`/`deployment_path` and **how the scene
  clears to empty** before replay (one-shot scene load vs clear+reseed via
  `scene_draft_*`).
- **Sandbox write path/perms** for rewriting the `.jsonl` under its `.lock`.
- **Compaction**: long chains emit a `compaction` entry + `.checkpoint.*.jsonl`; the
  renderer/splice must target the live `.jsonl` correctly in that state.
- **Bridge in-memory state** (loop-history / poison keyed by conversationId): in-place
  edit worked in the spike; for the loop-break sub-case, decide between leaving it
  (mostly harmless — next step is a different call) or using a fresh id. No live change
  either way.

---

## 7. Schemas (captured)

**Session `.jsonl`** = parent-linked event log. Entry types seen: `session`,
`model_change`, `thinking_level_change`, `custom` (e.g. `model-snapshot`), `message`,
`compaction`. Every entry has `id` (8-hex), `parentId`, `timestamp`, `type`.

**Message entry**: `{type:"message", id, parentId, timestamp, message:{role, content:[parts]}}`.
- roles: `user`, `assistant`, `toolResult`.
- content part types: `text`, `thinking`, `toolCall`.

**Assistant tool-call** content part:
```json
{"type":"toolCall","id":"<~32-char call id>","name":"tool_call",
 "arguments":{"id":"manyforge__<tool>","args":{…REAL ARGS…}},
 "partialArgs":"<JSON.stringify(arguments)>"}
```
**Tool-result entry** (separate message, `parentId` = the toolCall message's id):
```json
{"type":"message","id":"<8hex>","parentId":"<toolCall msg id>","timestamp":"…",
 "message":{"role":"toolResult","content":[{"type":"text","text":"<JSON result string>"}]}}
```

**Golden turn** to splice = `[assistant: (thinking?)+toolCall(manyforge__<tool>, args)]`
→ `[toolResult: success JSON]` → `[assistant: brief "done" text]`, parent-chained,
appended at the live tail (the failed turn's entries removed first).

**Golden change schema (corpus, per chained step):**
```yaml
golden:
  tool_calls: [{ name: tree_draft_insert_node, args: {…COMPLETE args…} }]
  assistant_text: "Added the approach move."
```

---

## 8. Risks / invariants checklist
- [ ] Generator assertion-gate green for every PnP step (corpus is internally coherent).
- [ ] Scene clears to base deterministically.
- [ ] Splice respects `.lock`; only edits between steps.
- [ ] Renderer handles compacted sessions.
- [ ] Sandbox write perms sorted.
- [ ] Golden trajectory regenerated whenever the corpus changes (derived artifact).

---

## 9. Documentation deliverable (REQUIRED)

Thoroughly document this mechanism **for PnP cases**:
- **Why** it was created — the chained-cascade confound + the evidence (the nemotron
  PnP back-half analysis), and why the existing loop-poison detector is insufficient.
- **How it works** — self-heal on failure: canonical state replay-from-base +
  golden-turn transcript splice; "model doesn't notice"; per-turn snapshot re-injection.
- **How it's implemented** — the generator, the renderer, the runner self-heal hook,
  the schemas, and the test-only boundary; how golden changes are authored and kept in
  sync; how to regenerate the trajectory; how to run a PnP self-heal validation.

This file (§1–§8) is the seed of that documentation — expand each section into
thorough prose as the implementation lands, and keep it accurate to the code.

---

## 10. Test-only / safety invariants
- **No live-component code changes.** Bridge / composer / openclaw / proxy untouched.
- The fabricate-success path exists **only** in `smoke_corpus_runner.py` + the
  generator, both run only during smoke tests. Production cannot fake history because
  the code to do so is not in any live component.
- Editing openclaw session files happens **only** against sandbox sessions during a
  test run, between steps, under the session lock.

---

## 11. As-built (2026-06-07)

### Files
- **`scripts/debug/golden_trajectory.py`** — generator. Resets to base, applies each
  golden change via the real bridge tools, and gates each step: (a) tool-level
  `success` (the bridge wraps failures as `success:false` inside HTTP 200), (b)
  *node-landed* (tree node count must rise on inserts — a 200 can still add nothing),
  (c) `expected.state_after` where the corpus declares one. Run:
  `python3 golden_trajectory.py --chain pnp_build`.
- **`scripts/debug/golden_trajectories.yaml`** — the authored golden changes per
  chained step (currently `pnp_build`, 20 steps), in RAW tool-arg form.
- **`scripts/debug/self_heal.py`** — `replay_to_canonical()` (reset-to-base + apply
  golden 1..N), `splice_golden_turn()` (rewrite the failed step's turn in the openclaw
  session `.jsonl` inside the sandbox via `docker exec`), and `self_heal()` (both).
- **`scripts/debug/smoke_corpus_runner.py`** — `--self-heal` hook in the main loop:
  on a chained-step **hard fail** (and not `--no-chain-session`), restore canonical
  state + splice the golden turn; the failed step keeps its own `fail` score (we only
  stop the cascade to step N+1). Flags: `--self-heal-golden`, `--self-heal-container`
  (default `openshell-my-assistant`), `--self-heal-agent` (default `manyforge-composer`).

### Validated against the live stack
- Spike: openclaw replays a runner-edited `.jsonl`; the model treats injected history
  as real (recalled an injected-only codename).
- Generator: full `pnp_build` chain `COHERENT` — 20 steps, all gates green (12
  manipulation nodes under a `repeat` root, 4 scene objects).
- `self_heal`: `replay_to_canonical(...,7)` rebuilt the canonical post-step-7 tree;
  `splice_golden_turn(...)` rewrote the transcript to a golden `toolCall→toolResult→
  text` turn; the model then **recalled the spliced action as its own**
  ("I used `manyforge__tree_draft_insert_node` to add the `approach_above` node").

### Key schemas learned (authoring reference)
- move (`move_manipulator_action`) pose-goal: `params:{move_id, motion_type:pose_goal,
  pose_goal:{position_m:[x,y,z], orientation_rpy_deg:[180,0,-180]}, velocity_scale}`.
- gripper (`command_gripper`): `{gripper_id, position (0.0 open / 0.66 closed),
  max_effort, duration_s, timeout_s}`.
- `upsert_collision_object`: FLAT `{object_id, shape_type:box, box_dimensions_m, pose}`
  (a nested `shape:{}` is rejected — silently, 200 with `success:false`).
- wrap (`tree_draft_wrap_node`): `{targetName, wrapper:{id:repeat, params:{num_cycles:-1}}}`.
- session `.jsonl` golden turn: assistant `{content:[{type:toolCall, name:"tool_call",
  arguments:{id:"manyforge__<tool>", args}, partialArgs}]}` → `{role:toolResult,
  content:[{type:text, text:<result json>}]}` → assistant `{content:[{type:text,text}]}`,
  each `parentId`-chained off the failed step's last user message.

### How to use
- Validate / regenerate the golden trajectory (whenever the corpus changes):
  `python3 golden_trajectory.py --chain pnp_build` → must print `COHERENT ✅`.
- Run a self-healing PnP smoke: `python3 smoke_corpus_runner.py --self-heal --filter '^PnP_'`.

### Changed/removed
- The opt-in chain-contamination detector (`MANYFORGE_SMOKE_DETECT_CHAIN_CONTAMINATION`)
  is removed — superseded by self-heal. The `"contaminated"` status remains as an inert
  vestige in the marker/summary tables (nothing emits it now).

### Pending
- **End-to-end model-driven validation**: run the PnP chain through a model with
  `--self-heal`, force/observe a mid-chain failure, and confirm the next step is
  evaluated from canonical state with the model unaware (to be done after commit, so
  the mechanism is rebuildable).
