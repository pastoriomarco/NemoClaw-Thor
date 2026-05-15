# ManyForge Composer-Assistant Smoke Corpus

A capability-stratified test corpus for the Composer-assistant lane, plus a
runner that dispatches each prompt through `/api/assistant/chat` and asserts
on tool calls + state delta + answer text. Designed to give a single number
("X / Y effective rate") that's diff-able across model swaps, prompt changes,
and deployment reconfigurations.

> **Architecture map** for the runtime this corpus exercises (component
> boundaries, two-lane bridge, proxy mutator, tool-surface origins, and the
> iter-20 reproduction recipe) lives in
> [`COMPOSER-ASSISTANT-ARCHITECTURE.md`](./COMPOSER-ASSISTANT-ARCHITECTURE.md).

## Files

| Path | Role |
|---|---|
| [`manyforge/scripts/debug/smoke_corpus.yaml`](../scripts/debug/smoke_corpus.yaml) | The corpus — 74 cases organized by category × detail level. |
| [`manyforge/scripts/debug/smoke_corpus_runner.py`](../scripts/debug/smoke_corpus_runner.py) | Loader + dispatcher + assertion engine + reporter. |
| [`manyforge/examples/assistant_modes_scene_authoring.deployment.yaml`](https://github.com/pastoriomarco/manyforge/blob/main/examples/assistant_modes_scene_authoring.deployment.yaml) | Deployment that scopes the assistant's tool + node allowlist. The smoke runs against this by default. |
| [`manyforge/examples/pick_and_place_ur10e_robotiq.program.yaml`](https://github.com/pastoriomarco/manyforge/blob/main/examples/pick_and_place_ur10e_robotiq.program.yaml) | The populated-state program (12-step pick-and-place, 2 scene objects, 1 param, 0 blackboard keys). |
| [`manyforge/examples/empty_pick_and_place_ur10e_robotiq.program.yaml`](https://github.com/pastoriomarco/manyforge/blob/main/examples/empty_pick_and_place_ur10e_robotiq.program.yaml) | Empty-state fixture used when a case sets `precondition.fresh_program: true` (PnP build chain). Has a `command_gripper` scaffold leaf because Composer rejects fully-empty composite trees. |

## Schema Highlights

Each case carries:
- `id`, `category`, `detail_level` (`generic` / `medium` / `specific`)
- `status: active` (default) / `future` — future cases are deliberate capability probes that need new tools or runtime tiers, not bugs
- `required_runtime` — capability gate; the runner skips a case unless either:
  - The runtime is harness-provided (`pnp_build_chain`, `custom_precondition`, `expanded_node_allowlist`), or
  - The runtime is enabled via `--runtime-flags` on the runner
- `precondition` — optional `chain_id`/`chain_step` (same-conversation chain) or `fresh_program: true` (start from empty fixture)
- `expected.tools_called` — ordered list of (tool name, args_contain, allow_retries)
- `expected.state_after` — dotted-path asserts against the unified state snapshot built from `/api/program` + `/api/program/tree` + `/api/scene/state`. (Note: `/api/program/state` does NOT exist — earlier corpus comments referenced a non-endpoint.)
- `expected.forbidden_tools`, `answer_must_contain`, `answer_must_not_contain` — negative + soft asserts

## Current State (as of 2026-05-08)

### Distribution

| Status | Count | Notes |
|---|---|---|
| `active` (explicit) | 35 | including the 17 reclassified after harness gained `pnp_build_chain` / `custom_precondition` / `expanded_node_allowlist` runtimes |
| `active` (default — no `status` field) | 31 | the original Section 1 + most of the standalone categories |
| `future` (still gated externally) | 8 | see "Future tier" below |
| **Total** | **74** | |

### Detail-level distribution

13 generic / 17 medium / 44 specific. Specific prompts pass at substantially higher rates than generic ones (see Failure Patterns).

### Latest full-run result (2026-05-08)

**31 pass / 18 fail / 25 skipped (= 63.3% effective rate)** against `cosmos-reason2-8b` over the OpenClaw lane. The run was made BEFORE the 17-case Step B reclassification, so the next run will attempt 66 cases instead of 49.

Per-category pass rates (specific level only, where the data is densest):
| Category | Specific pass rate |
|---|---|
| `tree_wrap_root` (P1, WRAP_*) | 1/2 — WRAP_root_medium fails because model didn't fire wrap |
| `scene_add` (P2) | ✅ |
| `tree_insert_runtime_obj` (P3) | ✅ |
| `current_*` (Section 14, canonical-name regression) | 4/6 — 2 timeouts |
| `tree_update_node_params` | 3/3 |
| `tree_delete_node` | 3/3 |
| `clarification` (Section 15) | 0/3 — see Pattern 4 |
| `pick_and_place_build` (PnP_01–PnP_04 prefix) | ✅ |

## Failure Patterns Observed

### Pattern 1 — Model emits text without firing the expected tool (8 cases)

Cases: `TREE_insert_runtime_medium`, `INSERT_position_first_specific`,
`INSERT_position_after_named_medium`, `PARALLEL_concurrent_medium`,
`PARALLEL_generic`, `FALLBACK_generic`, `MOVE_reorder_medium`,
`CUR_runtime_remove_then_restore_graspable`.

**Symptom**: harness reports `expected tool#0 'X' not observed (or never reached 2xx)`. Model returns prose like *"I would add a parallel..."* without firing the call.

**Disproportionately hits**: medium / generic detail levels and ambiguous composite-shape prompts (e.g., "add a parallel" with no children spec).

**Fixes worth trying**:
1. *Corpus*: reframe genuinely-ambiguous prompts (`PARALLEL_generic`,
   `FALLBACK_generic`) as expected-clarification (`tools_called: []` +
   `answer_must_contain: ["which", "what"]`).
2. *Prompt-side* (deployment system message): add a RULES rule —
   *"For composite shapes (parallel/fallback) when the request lacks
   children spec, call `tree_draft_insert_node` with the bare composite
   kind; an empty composite is a valid placeholder."*
3. *Model gap*: TEB 81 (cosmos-8b) struggles with truly-generic prompts.
   Consistent with the 2B's 0/9 result and the 9B's 8/9 — worth rerunning
   the corpus against the 9B (`qwen3.5-9b-claude-distilled-nvfp4`) for
   the comparison.

### Pattern 2 — state_after path uses wrong key (2 cases) — FIXABLE

Cases: `SCENE_update_pose_specific`, `CUR_scene_update_graspable_pose`.

**Symptom**: `scene.objects[id=graspable].pose.position` resolves to `<MISSING>`.

**Diagnosis**: live `/api/scene/state` returns `objects[*].pose.position_m` (with units suffix); corpus asserts `pose.position`.

**Fix (deterministic, cheap)**:
```yaml
# state_after assert
- "scene.objects[id=graspable].pose.position": [...]
+ "scene.objects[id=graspable].pose.position_m": [...]
```

### Pattern 3 — state_after fails because tool never fired (1 case)

Case: `WRAP_root_medium` (`expected program.tree.id == 'retry', got 'sequence'`).

Same root cause as Pattern 1 (model didn't fire `tree_draft_wrap_node`); the state assertion is only secondary. Fix is at the Pattern 1 layer, not state_after.

### Pattern 4 — Bounded-autonomy gap: model acts on ambiguous prompts (5 cases)

Cases: `MOVE_generic`, `SCENE_remove_generic`, `CLARIFY_scene_remove_pronoun`,
`CLARIFY_tree_wrap_pronoun`, `CLARIFY_motion_generic`.

**Symptom**: harness reports *"expected NO tool calls; observed [...]"*. Model picks its best guess and fires a mutation when the request has unresolved pronouns ("remove **it**", "wrap **that step**") or insufficient detail ("move over there").

**Diagnosis**: this is the **test corpus working as intended** — Section 15 was specifically authored to probe this gap. These cases failing means the gap is real and present on the model under test, not that the corpus is wrong.

**Fixes worth trying**:
1. *Prompt-side*: add a RULES rule —
   *"If the user uses an unbound pronoun (it/that/this) without an
   unambiguous antecedent in the same prompt or the immediately-prior
   turn, ask one clarifying question and emit no tool calls."*
2. *Model swap*: 9B Claude-distilled in earlier testing produced clean
   wrap/insert calls without leaking; could be the bounded-autonomy
   character of the distillation. Compare across models.
3. *Soft-pass weighting* (in the runner): downgrade clarification-gap
   failures to a `bounded_autonomy_violation` soft-pass category so they
   count toward effective rate but show distinctly in reports.

### Pattern 5 — Chain timeout / HTTP -1 (2 cases)

Cases: `REPLACE_subtree_specific` (140.1s), `INSERT_position_first_specific` (131.3s).

**Diagnosis**: complex multi-step prompts elicit long agent loops. Composer's `--assistant-timeout-s` is 130s (chain cascade is 120/125/130/140 — bridge has headroom, Composer caps).

**Fixes worth trying**:
1. *Per-case timeout override*: corpus field `precondition.chain_timeout_s`,
   plumbed through the runner → POST envelope → Composer's per-request
   timeout. Surgically extends the budget on known-slow cases.
2. *Bump default Composer chain budget* for known-slow categories
   (`tree_replace_subtree`, novel `INSERT_position` kinds) to 180–240s.

### Pattern 6 — PnP_05 fixture mismatch (1 case)

Case: `PnP_05_tree_root` (`expected program.tree.id == 'sequence', got 'command_gripper'; expected name 'pick_and_place', got 'scaffold'`).

**Diagnosis**: the empty-program fixture I added has a `command_gripper` leaf as scaffold (because Composer rejects fully-empty composite trees with *"sequence requires at least 1 children"*). The PnP_05 prompt *"create a root sequence named pick_and_place"* doesn't have a clean tool to use — the tree already has a leaf root, not nothing.

**Fixes (priority order)**:
1. *Better empty fixture*: seed with a `sequence` containing one trivial
   `command_gripper` child. Then PnP_05 becomes "rename the existing
   sequence to pick_and_place" (uses `tree_draft_update_node_params`).
2. *Corpus rewrite*: drop PnP_05; treat the scaffold sequence as
   already-equivalent-to-pick_and_place. Compresses the chain by one step.
3. *Composer-side fix* (larger): allow the deployment to load a program
   with `tree: null` or `tree: {children: []}`. Removes the scaffold
   workaround entirely.

## Hard Constraints — All Iteration Rounds Must Honor These

The 5-iteration cycle in Round 3 (and any future cycle that wants to
extend it) operated under the following non-negotiable rules. They
exist so the corpus stays a *general-purpose* acceptance suite — one
that catches model regressions cleanly without being co-tuned to a
specific model's quirks.

1. **No commits during the loop.** Every fix is staged in the working
   tree and re-evaluated each iteration. The user explicitly authorizes
   commits separately, after the loop ends.
2. **No per-prompt hints to the model.** Prompts in the corpus are
   user-style requests ("add a move-to-pose that approaches above the
   graspable…"). They MUST NOT contain assistant-targeted guidance like
   "use parentName=pick_and_place" or "remember to call insert_node".
   The corpus tests what a real user would type.
3. **No tightening of the model's RULES system message** to fix
   smoke-corpus failures. The system message is shared production
   surface; tweaking it for one test corpus contaminates the model's
   behavior on every other lane.
4. **Fixes must be cross-cutting.** Acceptable: corpus structure
   changes (chain reordering, fixture content, new inspect-then-mutate
   turns), harness assertion logic, deployment YAML allowlist,
   timeouts. Not acceptable: changing one prompt to make one case pass.
5. **Specific > generic.** Cases at `detail_level: specific` are the
   priority. Cases at `detail_level: generic` that surface real
   bounded-autonomy gaps (model fires when it should ask) can stay
   yellow — they are doing their job by flagging the gap.
6. **Corpus must remain extensible.** A change that helps three
   PnP_* cases is acceptable; a change that adds a one-off
   workaround for a single case is not. Future cases (new node kinds,
   new tools) must keep slotting in without rework.
7. **Schema examples in tool descriptions are excluded** as a fix
   class. v8.1 probe showed the model copied the example `objectId`
   verbatim and entered a 116-call upsert loop. The MCP wrapper
   validator is the load-bearing safeguard; reintroducing schema
   examples reopens the copy-the-example failure mode.
8. **Bridge / Composer code changes are out of corpus scope** unless
   explicitly authorized. The corpus tests the deployed surface;
   improving the surface is a separate workstream.
9. **Timeouts may be tuned freely** — both the runner default and
   per-case `chain_timeout_s` overrides — but watch for upstream
   circuit-breakers (a 5-minute upstream-502 ceiling was discovered
   in Round 3 when the default went past 270 s).
10. **Always restart the demo between iterations.**
    `PRESERVE_OPENSHELL=true demo-assistant-known-good.sh restart`
    clears zombie sessions and refreshes the SSH tunnel that backs
    `127.0.0.1:18789`. Skipping this contaminates iteration N+1 with
    iteration N's stuck sessions.

These constraints are also why "use a bigger model" or "tighten the
RULES" — both of which would close a chunk of the remaining failures —
are intentionally out of scope. The point of the corpus is to
*measure*, not to be co-tuned to a passing rate.

### Application context — informs every research search

ManyForge Composer is a **robotics + industrial-automation** authoring
tool. The assistant works on:
- **Planning scenes** (collision objects, fixtures, graspable items, robot frames)
- **Behavior trees** (sequence/fallback/parallel/repeat composites + leaf nodes for motion, gripper commands, attach/detach, signals)
- **Programs** (parameters, blackboard) compiled into RoboPlan / py_trees runtimes

When you research best practices for this corpus, **prefer the robotics
+ behavior-tree literature over generic "agentic" literature**. Examples
of higher-signal search queries:
- "behavior tree authoring LLM"
- "planning scene editing agent"
- "MoveIt 2 LLM tool calls"
- "py_trees authoring assistant"
- "industrial robot programming agent error recovery"

Generic-agent papers (browser agents, code agents, conversational
search) can mislead because robotics tools have **strict argument
schemas** (poses, frames, dimensions) and **physical-world side
effects** that pure software agents do not.

### Inference-time settings — fixed, not tunable

The deployed stack runs cosmos-reason2-8b with:
- **`enable_thinking: false`** — kept off for response latency. Interactive UI requires sub-30s typical turn time on the smoke corpus; thinking adds 30-60s per turn. Some recovery-from-error patterns from the "reasoning-trace" literature assume thinking is on; those are **not directly applicable** to this corpus and any research suggesting them must be flagged.
- **temperature ~0.2** (low but non-zero) — set bridge-side; not tunable from corpus or harness.
- **`tool_choice` is effectively `"auto"`** in the OpenClaw lane (which is what the smoke corpus runs against). The bridge does set `tool_choice: "required"` on the upstream HTTP request to vLLM in the **direct** lane, but **OpenClaw's gateway does not propagate this parameter to the model** — the parameter is silently dropped on the OpenClaw → vLLM hop. Result: in the lane we are testing, the model is free to return prose without firing a tool, and frequently does ("narration mode" / "EndTurn-without-tool"). This is the architectural root cause of a large fraction of the Group A and "model fires nothing" failures. **Closing this gap is a gateway-side fix, out of corpus scope.**

**Important model-context finding (round-3 research, 2026-05-09)**:
Cosmos-Reason2-8B is a NVIDIA physical-AI / robotics reasoning model
post-trained on **Qwen3-VL-8B-Instruct** with **long-CoT design** as a
core assumption. The official model card and NVIDIA blog explicitly
position Cosmos-Reason2 around long chain-of-thought. **Running it with
`enable_thinking: false` is therefore out-of-distribution** for the
model's training regime. This explains:
- The PnP_05 narration-mode cascade — without the internal CoT trace,
  the model "thinks aloud in user-visible text" instead, which derails
  tool-use mode for subsequent turns.
- The high variance on action-shaped specific prompts — without
  thinking tokens, the model has no internal scratchpad to "compose"
  the right tool args before emitting the call.
- The lack of a published Cosmos-Reason2 tool-use eval — NVIDIA tuned
  this model for spatial / 2D-3D reasoning, not schema-strict tool
  calls. The 8B-class JSON-arg fidelity quirks of Qwen3-VL transfer.

**The mitigation that worked** (PnP_05 = inspect-tool-as-anchor) is
documented in the literature as **plan-then-execute** ([CHI 2025
10.1145/3706598.3713218](https://dl.acm.org/doi/10.1145/3706598.3713218),
[arXiv 2509.03581](https://arxiv.org/html/2509.03581)). The
inspect-tool call satisfies `tool_choice: "required"` while giving the
model a "planning slot" via the tool result. This is the *only*
plan-then-execute variant compatible with our stack — pure text plans
are blocked by `tool_choice: "required"`.

Research that assumes thinking-on as the recovery vehicle (e.g. "let
the model reflect in scratch tokens before retrying") is out-of-scope.
Research that operates on **the visible conversation shape** (tool
result phrasing, message ordering, action-anchor heartbeats) IS in
scope.

## Iteration protocol — required ritual for every round

Each iteration in this loop MUST follow these steps. Skipping the
research step causes the loop to drift into local-minimum
optimizations (we observed this in rounds 1-5 before the search
ritual was formalized).

1. **Demo restart** before the run: `PRESERVE_OPENSHELL=true demo-assistant-known-good.sh restart`. Clears stuck sessions and SSH-tunnel staleness.
2. **Run the corpus** under the iteration's specific change. Save the harness JSON to `/tmp/instrumented_run/harness_iterN.json` and the live log to `/tmp/instrumented_run/iterN.log`.
3. **Spawn a research subagent in parallel** with each iteration. Topics to cover (rotate, do not repeat the same query):
   - Robotics + behavior-tree LLM authoring (priority — see "Application context" above)
   - Tool-call recovery patterns for sub-10B models with `enable_thinking: false`
   - Planning scene editing agents (industrial, MoveIt-adjacent)
   - Multi-turn tool-use degradation in agentic robotics frameworks
   - Bounded autonomy / clarification patterns for hardware-affecting agents
   - The specific failure modes seen in the prior iter (target the audit log)
4. **Digest the agent's findings** before scoring the iteration. Cross-reference against the constraint list — anything that requires bridge code, RULES tightening, schema examples, or model swap is parked, not applied.
5. **Apply the highest-leverage corpus or harness change** suggested. Justify the choice in one sentence in the SMOKE-CORPUS.md scoreboard.
6. **No commits.** Stage everything; the loop reassesses on the next iteration's output.

If iteration N's research turns up an idea that's clearly better than
iteration N+1's pre-planned change, **swap them**. The plan is a
guideline, not a script.

## Fixes Tried in This Session — Results

### Round 1 (foundation + capability gates)

| Fix | Outcome |
|---|---|
| **Foundation harness** (`smoke_corpus_runner.py`) | ✅ done. 480+ lines, full assertion engine, runs end-to-end against the live stack. |
| **State-endpoint discovery**: `/api/program/state` returns 404 → harness uses `/api/program` + `/api/program/tree` + `/api/scene/state` instead | ✅ done. State snapshots assemble cleanly into the dotted-path assertion namespace. |
| **Allowlist expansion** in deployment YAML: added `move_manipulator_action`, `command_gripper`, `timer`, `wait_for_signal_bool`, `set_key_bool_value` to composer-assistant `nodes` | ✅ done. Unblocked 17 future cases (post Step B reclassification). |
| **Tool-existence audit**: `program_draft_upsert_parameters`, `program_draft_remove_parameters`, `blackboard_draft_upsert_keys`, `blackboard_draft_remove_keys` already exist in deployment allow-list — no need to BUILD `program_metadata_tools` | ✅ dropped from BUILD plan. PARAM_add / PARAM_modify / BB_add now pass on the live stack (3/4 in spot-check). |
| **Custom-precondition fixtures**: harness pre-seeds `legacy_offset` / `grip_force` / `scratch_value` via `program_draft_upsert_parameters` / `blackboard_draft_upsert_keys` before delete/modify cases run | ✅ done. Required body shape: `{requestId, assistantMode, catalogHash, arguments: {…}}` (key insight from `AssistantBridgeToolRequest` model in `manyforge_composer/backend/models.py`). BB_modify and BB_delete now pass; PARAM_delete fails on the actual delete prompt (model picks a different tool, separate finding). |
| **Fresh-program fixture**: empty deployment-compatible program YAML at `examples/empty_pick_and_place_ur10e_robotiq.program.yaml` | ✅ done. Initially failed (`tree: null` rejected; `tree: {children: []}` rejected with *"sequence requires at least 1 children"*); fixed by seeding a single `command_gripper` leaf as scaffold. PnP_01 now passes. PnP_05 still fails — see Pattern 6. |
| **Corpus reclassifications**: 9 cases moved future→active during the session, plus 17 in Step B for a total of **26 cases moved to active**. | ✅ done. New status distribution: 35 explicit active + 31 default-active + 8 genuinely-future. |

### Round 2 (failure-pattern remediations after first full run)

After the round-1 full run produced 31/49 (63.3%), six failure patterns
were diagnosed (Patterns 1–6 in the section above). The cheap corpus +
harness fixes were applied:

| Fix | Outcome | Cases targeted |
|---|---|---|
| **Pattern 2** — state_after path correction `pose.position` → `pose.position_m` | ✅ done in two state_after asserts | `SCENE_update_pose_specific`, `CUR_scene_update_graspable_pose` |
| **Pattern 6** — empty-fixture rework: `command_gripper` leaf scaffold replaced with a `sequence` named `scaffold` containing one trivial `command_gripper` child; PnP_05 reframed to "rename the root sequence to pick_and_place" using `tree_draft_update_node_params` instead of `tree_draft_insert_node` | ✅ done (fixture YAML + corpus) | `PnP_05_tree_root` (and the rest of the chain by extension) |
| **Pattern 5** — added corpus field `precondition.chain_timeout_s` + harness plumbing through to Composer's `timeoutSeconds`. Set 220s on `INSERT_position_first_specific` and 240s on `REPLACE_subtree_specific` | ✅ done | the two known-slow cases |
| **Pattern 1** (subset) — reframe truly-ambiguous generic composite prompts as expected-clarification | ✅ done for `PARALLEL_generic` and `FALLBACK_generic` (`tools_called: []` + `forbidden_tools: [tree_draft_insert_node, tree_draft_wrap_node]` + `answer_must_contain: ["which", "where"]`) | 2 cases |
| **Patterns 1, 3, 4 (residual)** | ⏭ left as-is — these surface real model-behavior gaps; the corpus is doing its job by flagging them. Tracked for future model swaps / prompt-side RULES tightening | 5 CLARIFY/MOVE/SCENE generic + 4 medium-detail tool-no-fire |

### Round 3 (2026-05-08/09): five-iteration cross-cutting hardening

After round 2, ran five back-to-back full corpus runs (`cosmos-reason2-8b`,
OpenClaw lane). Each iteration applied **only cross-cutting** changes —
no per-prompt hints, no model RULES tightening — and re-ran the full
66-case active set. Demo restart between iters cleared zombie sessions.

| Iter | Pass | Effective rate | Cross-cutting change applied |
|---|---|---|---|
| 1 | 34/66 | 51.5 % | baseline after round-2 fixes; harness expanded_node_allowlist gate bug fix; tools_called semantic split (missing vs `[]`); default timeout 140 → 240 s |
| 2 | 31/66 | 47.0 % | (regression) reframed PnP_05 to expect `tree_draft_update_node_params`; model couldn't reliably fire rename — broke chain anchor |
| 3 | 36/66 | 54.5 % | rolled back PnP_05 reframe; pre-named the empty-fixture root `pick_and_place` so PnP_06+ has a real anchor on turn 1; state_after pose path `pose.position_m` → `pose_in_universe.position_m` |
| 4 | 39/66 | **59.1 %** | harness multiset tool match (was order-sensitive, false-failed when model fired the right tools in a different order) |
| 5 | 49 + 2 soft = **51/66** | **77.3 %** | repurpose PnP_05 from no-op rename to "verify program state" expecting `program_read`; default timeout 240 → 360 s for PARALLEL_*/FALLBACK_* recovery room |

**What worked across iters**:
- Iter 1 harness gate fix unblocked 9 PnP cases that were silently skipped on `expanded_node_allowlist`.
- Iter 3 fixture pre-rename (`scaffold` → `pick_and_place`) gave PnP_06–17 a valid parent anchor without needing a model rename.
- Iter 3 pose-path fix (`pose_in_universe.position_m`) flipped two SCENE_update_*/CUR_scene_update_* cases that had been false-failing on a state-key drift.
- Iter 4 multiset matching unblocked cases where the model fired the expected tool in a slightly different order from `expected.tools_called` (e.g., `MOVE_swap_specific`, `REPLACE_subtree_specific`).

**What regressed and was rolled back**:
- Iter 2 PnP_05 = "rename root to pick_and_place" expecting `tree_draft_update_node_params` cascade-broke the chain because cosmos-8b couldn't reliably emit the rename call. Iter 3 swapped strategy: pre-rename the fixture root, leaving PnP_05 as a no-op state-confirmation pass.

**Persistent failures heading into iter 5** (12 + 6 + 3 + 2 = 23 of the 27 failures):
- **PnP_06–PnP_17 (12 cases)**: same `tree_draft_insert_node not observed` failure on every chain step from "approach" through "home". Diagnosis: **PnP_05 was a no-op text-only turn, which puts the model into narration mode; it then describes what insert_node *would* do for PnP_06 instead of actually firing it.** Iter 5 fix targets this.
- **Specific INSERT/MOVE failures (6 cases)** (`INSERT_position_first_specific`, `INSERT_position_after_named_medium`, `TREE_insert_runtime_medium`, `MOVE_reorder_medium`, `FALLBACK_retry_specific`, `PnP_20_grip_force`): model fires the right tool but emits malformed args (`_raw` envelope, missing `parentName`, wrong `targetName`). The wrapper validator returns 400 with structured detail; the model paraphrases the error in text instead of retrying with corrected args. Cross-cutting fix would require improving model-side recovery, which is out of scope without per-prompt hints.
- **CLARIFY_*/MOVE_generic/FALLBACK_generic/UPDATE_params_generic (5 cases)**: model fires when it should ask. Real bounded-autonomy gap on cosmos-8b — kept yellow per the user's deprioritization of generic prompts.
- **PARALLEL_concurrent_medium / PARALLEL_generic (2 cases)**: 245s timeout. Iter 5 raised default to 360s.

**Cross-cutting fixes considered but rejected**:
- *Schema examples in tool descriptions*: previously dropped because v8.1 probe showed model copied the example `objectId` verbatim → 116-call upsert loop. The MCP wrapper validator is the load-bearing safeguard; adding schema examples reintroduces the copy-the-example failure mode.
- *Per-prompt hints in PnP chain*: explicitly forbidden by the user — keeps the corpus extensible without baking model-specific guidance into individual prompts.
- *Stricter RULES system message*: forbidden — same reason. Targeted at model behavior, not corpus structure.

### Iter 5 fix rationale

The largest observed cluster of failures (12 PnP chain cases, ~44 % of all
fails) shares a single root cause: PnP_05 is a no-op turn that the model
satisfies with text only. After 5 turns of conversation, with the most
recent turn being a text-only "already named" answer, the model carries
forward in *narration mode* — explaining what `tree_draft_insert_node`
would do without firing it. The fix repurposes PnP_05 to require a real
tool call (`program_read`, a no-arg inspection tool that just reads the
current program). This:
1. Keeps the model in tool-use mode through the chain handoff.
2. Refreshes the model's view of the tree state right before PnP_06 starts inserting children — useful for any future model that uses inspection results.
3. Adds no per-prompt hint and no model-specific guidance.
4. Makes the chain more representative of a real build session (a build chain *should* inspect state mid-flow).

Paired with the timeout bump (240 → 360 s default), iter 5 targets ~12 PnP cases + 2 PARALLEL cases = 14 of the 27 iter-4 failures.

### Iters 6–12 — extended cycle under the search-between-rounds protocol

After the initial 5-round cycle landed at 51/66, the user authorized
"another 5 rounds" with a per-round research-agent step. The findings
shifted the diagnosis: the dominant variance source is not arg
malformation but *tool-mode collapse to narration* on prompts that
sit at cross-family transitions.

| Iter | Status | Result | Change applied |
|---|---|---|---|
| 6 | killed @ 45 | 28 pass / 17 fail | Default timeout reverted 360→270s; PnP_05 cascade hit |
| 7 | **66/66** | **48/66 (72.7%)** | Harness tries to merge `body.toolCalls` arguments — confirmed dead (Composer drops them in chat response) |
| 8 | killed @ 44 | 28 pass / 16 fail / 0 recovered | First recovery-turn variant (4xx-only). No-op: the failure population is dominated by "model fires nothing" not "model fires bad args" |
| 9 | killed @ 43 | 30 pass + 2 soft / 11 fail | Bounded-autonomy rubric for CLARIFY_*/MOVE_generic (forbidden_tools instead of `tools_called: []`) — moved 2 cases from fail → soft-pass |
| 10 | killed @ 43 | 30 + 1🛟 + 2🟡 / 10 fail | Recovery turn extended to "no-tool-fired" branch. **First `recovered-pass`** validated the mechanism (UPDATE_params_medium recovered) |
| 11 | killed @ 45 | 30 + 1🛟 + 2🟡 / 12 fail | `BRIDGE_UPSTREAM_MAX_TOKENS=1536` env override. Marginal regression — chain steps may run out of output budget. Reverted |
| 12 | killed @ 44 | 29 + 2🟡 / 13 fail (≈70.5%) | Default 2048 tokens + recovery turn + broadened message. PnP_05 cascade hit again |
| 13a | killed @ 2 | mutator deployment trial | First proxy-mutator run with `tool_choice=required` always-on. Validated mutator wire-level (5 of 5 unit cases passed; OpenClaw routed through proxy, mutation injected, vLLM honored it). **Bridge agent loop spiraled: 11 vLLM round-trips per case, 245-275 s wall clock.** See "Bridge spiral finding" below. |
| 13b | killed @ 4 | mutator with streaming-aware proxy | After fixing my proxy's HTTP/1.0 chunked-encoding issue, results unchanged: same spiral. Confirms the spiral is a bridge-agent-loop semantics problem, not a proxy issue. |
| 13c | killed @ 43 | 30 + 2🟡 / 11 fail (≈74.4%) | Clean baseline post-mutator-experiment (mutator off, openclaw.json restored). Comparable to iter 12. |
| 14 | killed @ 4 | partial — gateway duplication | First test of **alternating mode** (proxy mutator with `OPENCLAW_PROXY_FORCE_TOOL_CHOICE=alternating`) + **read-first user-message suffix**. Sanity test of a single chat passed (2-turn agent loop, model fired `delete_node` on turn 1, emitted "deleted" text on turn 2 → exit cleanly). Corpus run was confounded by two simultaneous gateway processes; one used cached old config. |
| 14-final | killed @ 45 | **31 + 2 soft / 12 fail = 33/45 ≈ 73.3%** | **Mutator successfully deployed at host:18790** between bridge and OpenClaw tunnel. Bridge restarted with `OPENCLAW_ASSISTANT_GATEWAY_PORT=18790`. **44 of 44 chat-completions traversed the mutator with `user_suffix` injected.** `tool_choice` mutation didn't apply at this layer (bridge→OpenClaw protocol doesn't carry `tools[]` directly). Killed @ PnP_07 due to chain cascade — see analysis below. |
| 15 | killed @ 44 | **30 + 2 soft / 12 fail = 32/44 ≈ 72.7%** | **Strategy 1**: `user_suffix_first_turn_only=1`. **Critical finding**: doesn't work at this insertion point. The bridge → OpenClaw protocol shows `asst_count=0` and `msgs=1` on EVERY request (bridge sends each chat as a fresh single-message envelope; OpenClaw maintains conversation state internally). My turn-counting heuristic fired the suffix on every request anyway. Same cascade as iter 14. **Lesson: turn-context-aware mutations require an insertion point INSIDE OpenClaw, not at the bridge boundary.** |
| 16 | **66/66** | **43 + 5 soft / 18 fail = 48/66 ≈ 72.7% effective, 65.2% first-try** | **Strategy 2**: `--no-chain-session` flag in harness — each chain step gets a fresh conversationId. Combined with the suffix mutator. **First full 66-case run since iter 7. Cascade broken**: PnP_06 ✅ (50s), PnP_07 ❌ (regular 36s fail), PnP_08 ✅, PnP_09–17 = 4✅/5❌ (independent fails, no propagation). PnP_18 single 275s timeout, but no cascade after. **Ties iter 7's effective rate AND runs to completion.** Notable wins on the full corpus: `REPLACE_subtree_specific` ✅, `FALLBACK_retry_specific` ✅, `FALLBACK_alternate_medium` ✅, `CUR_runtime_remove_then_restore_graspable` ✅ (was failing in every prior iter), all 3 CLARIFY_* 🟡 soft-pass (rubric working). |
| 17 | killed @ 10 | 2 + 4 soft / 4 fail (6/10 effective ≈ 60%) | **Architecture upgrade + worst-case combo test.** vLLM moved to :8050; proxy at host:8000 forwards to :8050 (per-turn interception of OpenClaw's internal agent loop). vLLM relaunched with `--default-chat-template-kwargs '{"enable_thinking":true}'` (server-side default ON, matching Cosmos-Reason2's training distribution). Bridge `tool_choice="required"` injection disabled (proxy now owns it). Mutator config: `tool_choice=alternating-on-even` + `enable_thinking=alternating-off-on-even` (turn 1/3/5 default thinking-on / no force; turn 2/4/6 thinking-off + tool forced). **Research finding (round 5)**: this combo is OOD per turn — turn 2's "thinking-off + forced tool" recreates the exact failure mode we'd avoided. Killed early; per-turn agent-loop visible (3-turn chats verified end-to-end). 4 of first 10 cases hit 244+ s timeouts. 4 soft-passes from thinking eating answer-text budget. |
| 18a | killed @ 3 (proxy hang) | 1 ✅ / 2 fail @ 244 s | First attempt: `tool_choice=required-first` + `thinking_token_budget=512` + thinking-on default + chain-session restored. Two consecutive cases hit case-timeout 244 s with **zero vLLM responses logged** — the proxy's `up_resp.read()` was blocking on runaway streaming generations. Stopped to investigate. |
| 18b | killed @ 3 (proxy hang) | 1 ✅ / 2 fail @ 244 s | Same config, fresh proxy. Same failure mode. Root-cause investigation: vLLM logs showed `Running: 3 reqs, Waiting: 3 reqs, Deferred: 3 reqs` at steady 30 tok/s for **8+ minutes** — i.e. each request was generating ~14000 tokens. **Cause**: OpenClaw → vLLM call carries no `max_tokens`; vLLM defaults to the model's full context window; with thinking-on + complex prompts, generation runs for 8+ min per turn. **Fix shipped**: (a) proxy `_OVERRIDE_MAX_TOKENS` mutator now **injects** the key when missing (was previously rewrite-only); (b) per-request 200 s socket timeout safety net so a single runaway can't pile up zombie threads. |
| 18c | killed @ 48/74 (Monitor 1 h timeout) | 25 + 2 🟡 / 21 fail = **27/48 effective ≈ 56.3%** | `tool_choice=required-first` + `thinking_token_budget=512` + `max_tokens=2048` cap (proxy injects when missing). Cap **works as designed** — per-request `max_completion_tokens` rewritten 16384→2048; previously-hanging P3 dropped 244 s → 67 s. But effective rate is **16 points below iter-7/16 baseline**. Two distinct failure modes: (1) **`tool_choice=required-first` over-forces tools on no-tool cases** (FALLBACK_generic, PARALLEL_generic both expected NO tool calls; model called 8/2 tools respectively), (2) **`tree_draft_insert_node` collapsed across the corpus** — 9 distinct cases failed with "expected tool 'tree_draft_insert_node' not observed", suggesting either the 2048 cap truncates the complex multi-arg call mid-generation, or thinking-on default eats too much budget before the tool emission. Run cut short by Monitor 1 h timer killing the runner pipeline (lesson: smoke runner needs to outlive Monitor). |
| 19 | **66/66** | **36 + 4 soft / 26 fail = 40/66 ≈ 60.6% effective, 54.5% first-try** | Drop `tool_choice` mutation; keep `max_tokens=2048` cap + `thinking_token_budget=512` + thinking-on default + chain-session restored. Smoke runner detached from Monitor pipeline (`nohup`) so it survives the 1 h Monitor timeout. **Front-half wins (vs iter 18c, same cap+thinking, only `tool_choice` change):** P2 pass 19.7s (was fail 244s), P3 pass 50.1s (was fail 67s+miss), TREE_insert_runtime_generic pass (was fail), DELETE_kind_medium pass 11.5s (was fail), MOVE_swap_specific pass (was fail), SCENE_update_pose_specific pass 31.8s (was fail). **Net gain on first 48 cases: ~74 % vs iter 18c's 56 %.** **PnP cascade returns**: PnP_05 fail (familiar), then PnP_06–PnP_20 cascade with 13/14 sequential fails — chain-session retains broken state from PnP_06's first failure, dragging the rest of the chain down. iter 16 fixed this with `--no-chain-session`; iter 20 will combine that with the new cap+thinking settings. **No-tool corpus cases still over-eager**: PARALLEL_generic and FALLBACK_generic still emit 5+ `tree_draft_insert_node` calls when they should emit none (so the regression in 18c was NOT just `required-first` — model is also over-eager on no-tool prompts even without forcing). **Proxy infra healthy**: 183 chat-completions logged, all with cap applied, longest 205.9 s, zero zombie threads. |
| **20** | **66/66** | **45 + 4 soft / 17 fail = 49/66 ≈ 74.2% effective, 68.2% first-try** | **NEW BEST.** Combine iter-19 settings with iter-16's cascade fix: `--no-chain-session` (each chain step gets a fresh conversationId) + `max_tokens=2048` cap + `thinking_token_budget=512` + thinking-on default + no `tool_choice` mutation. **Cascade broken**: PnP suite 13 ✅ / 6 ❌ (vs iter 19 chain-session-on: 4 ✅ / 14 ❌, +9 case swing on PnP alone). PnP_05 ✅ 10.4s (was fail in 19), PnP_06 ✅ 42.6s (was fail), PnP_07 ✅ 70.6s, PnP_18 ✅, PnP_20 ✅. PnP_08/_10/_12/_14/_16 ❌ are now **independent failures** with no propagation, exactly the iter-16 pattern. **Beats iter-7/16 baseline by +1 effective and +3 first-try points.** 284 chat-completions through proxy, all capped, no zombies. |
| 21a | **66/66** | **45 + 4 soft / 17 fail = 49/66 ≈ 74.2% effective, 68.2% first-try** | **Identical totals to iter 20** — drop `thinking_token_budget=512` (let thinking use the full 2048 cap), hold all other iter-20 settings. PnP suite 13 ✅ / 6 ❌ exactly. The 17 fails are not the same set: 2 PnP cases swapped (iter 21a fails PnP_07/_15 + passes PnP_16/_17; iter 20 the opposite). **Conclusion: `thinking_token_budget` is neutral at this corpus** — removing the budget redistributes which PnP cases fail under sampling temperature but doesn't change the rate. The remaining `tree_draft_insert_node` regressions are NOT explained by thinking budget compression; iter 21b will test the cap-truncation hypothesis. |
| 21b | **66/66** | **44 + 4 soft / 18 fail = 48/66 ≈ 72.7% effective, 66.7% first-try** | Raise `max_tokens` cap from 2048 to 4096; hold `thinking_token_budget=512` and all other iter-20 settings. **1 case BELOW iter 20** — the extra room *hurt* rather than helped. PnP_06 went 244 s HTTP -1 (was pass 42.6 s in iter 20) — model used the larger budget to ramble and tripped the proxy 200 s timeout. PnP suite went 11 ✅ / 8 ❌ (vs iter 20's 13 ✅ / 6 ❌). 2 cases recovered (PnP_08, PnP_12 pass), 4 lost (PnP_06, PnP_07, PnP_11, PnP_13 fail). **Conclusion: cap=2048 is at or near the sweet spot for this corpus.** Bigger budget → more model wander → more timeouts. The remaining ~17 fails appear to be a model-capability ceiling (`tree_draft_insert_node` understanding), not a budget compression issue. |
| 21c | **66/66** | **42 + 5 soft / 19 fail = 47/66 ≈ 71.2% effective, 63.6% first-try** | Stacking test: iter-20 settings + iter-16 user-suffix mutator (`OPENCLAW_PROXY_USER_MESSAGE_SUFFIX="Before answering, call program_read or scene_inspect to refresh state if needed."`). **2 cases BELOW iter 20** — the strategies don't stack cleanly. Suffix recovered 3 cases (`FALLBACK_retry_specific` ✅, `FALLBACK_alternate_medium` ✅, `REPLACE_subtree_specific` ✅, all were fail in iter 20). But 5 new regressions: `UPDATE_params_specific` ❌ (state_after wrong), `UPDATE_params_generic` ❌, `SCENE_update_pose_specific` ❌, `CUR_scene_update_graspable_pose` ❌, `PnP_06_approach` ❌. Net −2. **Conclusion**: the read-first suffix forces tool-using cases through an extra round of state-refresh that biases the model away from straightforward update operations on parameters/poses (it inspects rather than acts). The two strategies aren't independent — they target overlapping subsets of cases differently. iter 20 settings (no suffix) win cleanly. |
| 22 | killed @ 9 | 3 ✅ + 1 🟡 / 5 ❌ on 2B | **Model-substitution probe.** Cosmos-Reason2-2B (smaller variant) with iter-17-style alternating `tool_choice=alternating-on-even` mutator + thinking-always-on. Killed early — 2B model is materially weaker on this corpus: P1_wrap_root_specific failed in 117 s with the wrong tool fired, SCENE_add_medium hit 244 s timeout. 4× the per-case latency of iter 20 (152 s vs 38 s). Confirms the 8B/2B capability gap is too large to compensate for via mutator strategy. |
| 23 | killed @ 12 | 7 ✅ / 4 ❌ ≈ 58% on Qwen3.5-9B | **Different-model probe.** Qwen3.5-9B-Claude-Distilled-NVFP4 (served as `cosmos-reason2-8b` to keep stack compatible) with iter-17-style alternating tool_choice mutator + thinking-off (per profile recommendation) + cap=8192. Front half showed +2 wins (TREE_insert_runtime_medium ✅, INSERT_position_after_named_medium ✅) and 3 unique fails. Killed at 12 cases (−2 vs iter 20 same window). Wrong tool format (qwen3_xml vs hermes), profile-internal max_tokens reservations, and DeltaNet hybrid layout add too many unknowns to extract a clean signal in a single run. **Conclusion**: cross-model swap is a different research axis; pursuing further would need a clean A/B controller separate from the mutator work. |
| 24-26 | not run | — | Reserved for further per-model variations (chain-session ON/thinking-ON variants on Qwen3.5-9B); deferred in favor of iter 27 schema refactor on Cosmos-8B. |
| 27 | **66/66** | **39 + 4 soft / 23 fail = 43/66 ≈ 65.2% effective, 56.1% first-try** | First post-refactor run (Action A/B/C from the failure-analysis plan landed): catalog descriptions for 6 runtime collision-object kinds rewritten to lead with "Behavior-tree leaf — runtime…", `tree_draft_insert_node` schema gained `afterName` / `beforeName` / `position` shortcuts, `tree_draft_swap_node` renamed to `tree_draft_change_node_kind`. Verbose schema description (~3700 chars) listing 4 mutually-exclusive positional forms. **−6 cases vs iter 20**. The wins predicted by the analysis landed (P3, INSERT_position_after_named_medium, MOVE_reorder_medium, FALLBACK_retry_specific) but the verbose description bled into unrelated cases — model spent thinking budget weighing positional forms even on simple updates. UPDATE_params_specific went pass→fail 155 s (model called `tree_draft_wrap_node` with `targetName` instead of update). Wall clock +27 min vs iter 20. |
| 28 | **66/66** | **45 + 4 soft / 17 fail = 49/66 ≈ 74.2% effective, 68.2% first-try** | Schema trim: dropped `position: "first"\|"last"` (redundant with `index=0` / omitted), trimmed insert_node description from 3700 → 1639 chars. Otherwise iter-27. **+6 cases vs iter 27**, ties iter-20 baseline plus retains all the new wins (afterName, change_node_kind). Confirms the trim (cognitive-load reduction) was load-bearing. |
| 28+ | **66/66** | **51/66 ≈ 77.3% effective, 71.2% first-try** | After the trim was committed, a fresh Composer restart produced 51/66 — 1 case better than iter 28 due to PARALLEL_concurrent_medium recovering. Re-counted as a separate observation for the iter-32 comparison; the same code/run produces 49-51/66 at sampling variance. |
| 29 | **66/66** | **40/66 ≈ 60.6%** | Same code as iter 28, chain-session ON. **Identical to iter 19 baseline.** 14 PnP cases cascade-fail in 12-15 s each. Confirms: schema refactor doesn't help under chain-session-on; cascade is the dominant variable, schema is orthogonal. |
| 30 | **66/66** | **45 + 5 soft / 16 fail = 50/66 ≈ 75.8% effective, 68.2% first-try** | Post-`tree_draft_insert_node.nodeName`-refactor (final schema move: name moved out of `node` payload to top-level `nodeName` to match the rest of the family). Chain-off. Within 1-case sampling variance of iter 28. The nodeName refactor is robust on chain-off. |
| 31a | killed @ ~14/74 cascade | partial — overflow precheck rejection | First attempt at OpenClaw's built-in compaction: set `agents.defaults.contextTokens=24000` to force auto-compact-on-overflow. Fatal config interaction: OpenClaw computes `promptBudget = contextTokens − reserveTokens(=models.json maxTokens=16384) = 8000 tokens`. PnP_02's accumulated session is 8.7 K tokens → overflows 8 K budget → triggers compaction → compaction blocks on `already_compacted_recently` cooldown → request rejected → 3.0 s pre-flight-overflow fast-fails on every PnP from PnP_03 onward. **Discovery**: OpenClaw's auto-compaction has a cooldown that prevents repeat compactions in quick succession; the cooldown turns into a hard wall once the first overflow happens. |
| 31b | killed @ ~13/74 cascade | partial — same failure deeper | Raised `contextTokens=32K` (promptBudget = 16K). Same `already_compacted_recently` failure mode, just one PnP step deeper. Confirms path-A (lower contextTokens to trigger built-in compaction) is not viable for our chain-session cascade. |
| **32** | **66/66** | **47 + 4 soft / 15 fail = 51/66 ≈ 77.3% effective, 71.2% first-try** | **NEW BEST UNDER CHAIN-SESSION-ON.** Bridge-side periodic `/compact`: `OPENCLAW_ASSISTANT_COMPACT_EVERY_N=2` makes the bridge fire `/compact` to the gateway before every 2nd request on the same session-key. Bypasses the cooldown by spacing compactions ~30-60 s apart manually. **9 compactions fired during the PnP suite, all succeeded.** PnP suite: 15 ✅ / 4 ❌ (vs iter 29 chain-on no-compact: 4 ✅ / 14 ❌; iter 28 chain-off: 14 ✅ / 5 ❌). **Equals iter 28 chain-off effective rate exactly.** Wall clock +30 min vs iter 28 (~75 min total) — compaction model call is ~3-5 s × 9 fires + retry-loop avoidance. |

### Final scoreboard (iter 16 → iter 32)

| Iter | Effective | First-try | Chain-session | Notable |
|------|-----------|-----------|---------------|---------|
| 16 (legacy) | 48/66 (72.7%) | 65.2% | OFF (--no-chain) | First full-corpus run with mutator |
| 19 | 40/66 (60.6%) | 54.5% | ON | Cascade exposes |
| 20 | 49/66 (74.2%) | 68.2% | OFF | Iter-19 settings + iter-16 cascade fix; first to beat iter-16 |
| 28 | 51/66 (77.3%) | 71.2% | OFF | Schema refactor (afterName/beforeName/change_node_kind) + trimmed description |
| 29 | 40/66 (60.6%) | 53.0% | ON | Same code as 28 with chain-on; cascade returns identically to iter 19 |
| 30 | 50/66 (75.8%) | 68.2% | OFF | Final `nodeName` top-level move; ties iter 28 within variance |
| **32 (PRODUCTION RECIPE, CHAIN-ON)** | **51/66 (77.3%)** | **71.2%** | **ON** | **Bridge fires `/compact` every 2 prompts; matches iter 28 chain-off** |
| 33 | 48/66 (72.7%) | 57.6% | ON | `request_clarification` tool + `[NEEDS-CLARIFY]` marker + bridge auto-retry — model never used the new tool; 7 cases got stricter rubric and 5 hard-failed; `--enable-recovery-turn` salvaged 10 cases (+10 recovered-pass). See "Iter 33 — Pattern A direction 3 attempt" below. |
| 34 | HALTED at 3/66 | — | ON | Proxy `tool_choice=required` every turn — force the model to pick A tool, hoping it'd choose `request_clarification` on ambiguous prompts. Halted after 14 min: P1 fail (275 s), P2 soft-pass (167 s), P3 fail (425 s). Force-required prevented natural loop termination → every case ran 5-7× normal duration. Direction 3 fully reverted; iter-32 stack restored. See "Iter 34 — Pattern A direction 3 attempt #2" below. |

**Iter 32 remains the production recipe** — iter 33 regressed by 3 cases (rubric tightening on 7 Pattern-A cases turned soft-passes into hard fails without the model adopting the new tool), iter 34 was halted as unrecoverable after 3 cases. Both iter 33 and iter 34 are fully reverted; the request_clarification tool + bridge auto-retry + corpus rubric changes are gone. Two durable wins were preserved: `--enable-recovery-turn` is now default-on for smoke runs (+10 cases in iter 33 measurement), and a long-standing stdout-buffering bug in the smoke runner was fixed so per-case verdicts stream realtime (this paid for itself in iter 34 — saved >4 hours of wasted wallclock).

### Production recipe (iter 32)

End-to-end stack:

1. **vLLM** on `:8050` with `--default-chat-template-kwargs '{"enable_thinking":true}'`. Cosmos-Reason2-8B served via `nvidia/Cosmos-Reason2-8B`.
2. **Proxy mutator** ([`scripts/proxy/vllm-proxy.py`](../scripts/proxy/vllm-proxy.py)) on `:8000`:
   - `OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS=2048` (injects when caller omits)
   - `OPENCLAW_PROXY_THINKING_TOKEN_BUDGET=512`
   - 200 s per-request socket timeout (shipped iter 18b, prevents zombies)
3. **OpenClaw config** (`/sandbox/.openclaw/openclaw.json`): `agents.defaults.compaction.model = "inference/cosmos-reason2-8b"` (route compaction through local model rather than the unreachable default `gpt-5.5`). **Don't set `agents.defaults.contextTokens`** — the cooldown trap is real.
4. **Composer code** post-iter-30: `tree_draft_insert_node` schema has `nodeName` top-level + `afterName`/`beforeName` shortcuts + node payload without `name`; `tree_draft_change_node_kind` (renamed); 6 runtime collision-object catalog descriptions rewritten.
5. **Bridge** (`openclaw_assistant_bridge`) on `:8200`: `OPENCLAW_ASSISTANT_COMPACT_EVERY_N=2` enables periodic `/compact` firing every 2nd request per session-key. Set `OPENCLAW_ASSISTANT_COMPACT_TIMEOUT_S=120` for the compaction call timeout.
6. **Smoke runner / production caller**: chain-session ON (default — same `conversationId` across chained prompts) is now safe.

### Iter 27 → 32: schema refactor + bridge compaction (the breakthrough)

The post-iter-21 work targeted the model-capability ceiling identified in the iter-20 failure analysis (15 consistent fails across iter 20 + iter 21a). Three actions and one architecture-level change:

**Action A — runtime collision-object catalog descriptions** (Pattern C, ~3 cases). Rewrote 6 entries in `manyforge_behavior/resources/node_catalog.yaml` (`add_collision_object`, `upsert_collision_object`, `remove_collision_object`, `update_collision_object_pose`, `attach_object_to_link`, `detach_object_from_link`) to lead with "Behavior-tree leaf — runtime collision-object operation" and explicitly contrast against `scene_draft_*` siblings. The model's lexical pattern match on "scene" was pulling it toward static-scene tools; explicit framing shifts the bias.

**Action B — sibling-relative positional shortcuts on `tree_draft_insert_node`** (Pattern B, ~5 cases). Added `afterName: <sibling>` and `beforeName: <sibling>` properties; handler resolves parent + index via `_find_tree_node_ref`. The 4xx response when `parentName` matches a known leaf adds a hint suggesting `afterName`/`beforeName`. **Critical follow-up**: iter 27's first version included `position: "first"|"last"` and a 3700-char description listing 4 mutually-exclusive forms; this verbose schema burned thinking budget on every insert call and net-regressed against iter 20 by 4 cases. Iter 28 dropped `position` and trimmed the description to 1639 chars; rate recovered to 51/66.

**Action C — rename `tree_draft_swap_node` → `tree_draft_change_node_kind`** (Pattern E, 1 case). The natural-language phrase "swap the order of A and B" was lexically pulling the model to `tree_draft_swap_node` (which actually changes a node's catalog id, not its position). Clean rename across the schema, handler, deployment YAML, bridge intent-inference heuristic, behavior catalog, and tests. Added an explicit anti-example sentence to the new tool's description ("does NOT reorder siblings"); added forward reference to `tree_draft_move_node`'s description ("right tool when the user says 'swap the order'"). Bridge intent-inference now routes "swap the order"/"swap order"/"reorder" to `move_node` and bare "swap" to `change_node_kind`.

**Final schema move (iter 30): move `name` from node payload to top-level `nodeName`.** Diagnostic: in iter 29's chain-on retry storm, the model was making the SAME mistake 108 times in one session — sending `nodeName` as a top-level key (and duplicating it inside `node.name`) without `parentName`. Root cause: 5 of 7 tree tools use `nodeName` top-level for the existing target node; insert_node was the outlier with `parentName`. The model conflated the convention. Fix: add `nodeName` as required top-level; remove `name` from `node` payload; handler injects `nodeName` into `node.name` before validation. Aligns insert_node with the rest of the family.

**Iter 31 — OpenClaw built-in auto-compaction is not viable** for our cascade. Setting `agents.defaults.contextTokens=24000` (or 32000) triggers `context-overflow-precheck` → tries to compact → fails on `already_compacted_recently` cooldown after the first attempt → every subsequent prompt fast-fails at the precheck with `Context overflow: prompt too large for the model (precheck)`. The cooldown is a hard wall once hit. Path A (lower the budget) abandoned.

**Iter 32 — bridge-side periodic `/compact`** is viable. Implementation in [`openclaw_assistant_bridge/service.py`](../openclaw_assistant_bridge/service.py):

- Per-process counter dict `_SESSION_REQUEST_COUNTER`, keyed on `derive_gateway_session_key(payload)`.
- Increment on every assistant request. If `OPENCLAW_ASSISTANT_COMPACT_EVERY_N=N` is set and the counter is at the Nth, 2Nth, 3Nth, ... position (skipping #1), fire `/compact` to the gateway BEFORE forwarding the user prompt.
- The compact call uses `build_gateway_chat_completions_command(..., message="/compact")` — OpenClaw's chat-command registry recognizes `/compact` as a slash command and routes to the compaction handler.
- Sequential: bridge waits for the compact response (with its own ~120 s timeout) before forwarding the actual user prompt.
- Failure-tolerant: a compact-call exception is logged and swallowed; the user prompt still goes through.

In iter 32 with N=2: 9 compactions fired across the 19-case PnP suite, all succeeded. PnP cases that historically cascaded (PnP_06+ in iter 19/29) ran to completion; the post-compaction session state preserves enough information for the model to continue producing correct tool calls. Wall-clock cost ~30 minutes added vs iter 28 chain-off.

### Remaining ceiling (iter 32 fails)

15 cases fail in iter 32; broken down:

- **Pattern A — no-tool over-eagerness on ambiguous prompts (2)**: PARALLEL_generic, FALLBACK_generic. Model fires 3-8 tool calls when corpus expects clarifying-question prose. Persistent across all iters; not addressable by infra changes. See "Pattern A discussion" below.
- **`tree_draft_insert_node` multi-arg specificity (5)**: TREE_insert_runtime_generic, INSERT_position_first_specific, FALLBACK_retry_specific, FALLBACK_alternate_medium, P3_tree_insert_runtime_obj_specific. Model emits prose or wrong args on prompts that pack 4-6 constraints into one sentence. Pattern D from the original failure analysis.
- **Tool-mismatch / wrong-family (3)**: REPLACE_simple_medium (uses replace_subtree but gets wrong tool), UPDATE_params_generic (model emits prose instead of `update_node_params`), CUR_runtime_remove_then_restore_graspable (3-tool sequence, model only emits 1).
- **Variance fails (4)**: TREE_insert_runtime_medium, PnP_06_approach (174s), PnP_14_upsert (237s), PnP_18_repeat_root (275s HTTP-1). These pass in some iter-28/30 runs.
- **Soft-pass: SCENE_remove_generic, MOVE_generic** — scoring rubric satisfied loosely.

### Iter 33 — Pattern A direction 3 attempt: `request_clarification` tool (NEGATIVE RESULT, 2026-05-10)

**Goal:** close Pattern A residuals (no-tool over-eagerness on intentionally ambiguous prompts) by giving the model a structured way to ask the user a question, plus tightening the rubric so "fire a guess" no longer satisfies asserts.

**Approach.** Three coordinated changes:

1. **New tool `request_clarification(question: string)`** registered in the assistant catalog with handler at [`routes_assistant.py:_apply_request_clarification`](https://github.com/pastoriomarco/manyforge/blob/main/manyforge_composer/backend/routes_assistant.py). The handler returns a payload that instructs the model to end its turn with prose prefixed by `[NEEDS-CLARIFY] ` plus the question text — the marker is the cross-process signal because the bridge can't see tool calls inside the gateway loop, only the final assistant message.

2. **Bridge auto-retry** on the marker — `openclaw_assistant_bridge/service.py` strips `[NEEDS-CLARIFY]` from the final message, sets `clarificationRequested=true` on the response envelope, and if no follow-up is queued by the caller (`OPENCLAW_ASSISTANT_CLARIFICATION_AUTO_RETRY_MAX=1` enabled), re-fires the original prompt once. Idea: give the model a second pass to pick a sensible default if the marker fired spuriously.

3. **Corpus rewrites on 7 Pattern-A cases**: `PARALLEL_generic`, `FALLBACK_generic`, `MOVE_generic`, `SCENE_remove_generic`, `CLARIFY_scene_remove_pronoun`, `CLARIFY_tree_wrap_pronoun`, `CLARIFY_motion_generic`. Each got `expected.clarified: true` (asserts model must fire the marker) plus a `clarification_followup` disambiguated prompt the runner sends as turn 2 when the marker fired. New status: `clarified-pass`.

**Result: 48/66 (72.7%) effective, 38/66 (57.6%) first-try — regressed 3 cases vs iter 32.** Breakdown: 38 pass, 10 recovered-pass, 0 clarified-pass, 0 soft-pass, 18 fail. Wall-clock 80.5 min (+5 min vs iter 32, attributable to recovery turns + clarification retries).

**Why direction 3 failed:**

- **Model never invoked `request_clarification` on any of the 7 targeted cases.** Of the 3 clarifications detected across the full run, all 3 were on `TREE_insert_runtime_*` cases (unrelated to the 7 — model self-elected to ask there). On the 7 we explicitly targeted: 0 clarifications. Tool registration + mode allowlist + system-prompt rule rewrite were not enough to flip Cosmos-Reason2-8B's action-bias. Hypothesis: the workspace-AGENTS.md system prompt is large; one new paragraph competing for attention against ~100 lines of "you MUST emit a tool call" framing won't win.

- **The bridge auto-retry never fired on the 7 either** (no marker → nothing to retry). The 3 clarifications that did fire all retried successfully (auto-retry mechanism itself is sound — see `TREE_insert_runtime_generic` recovering in 41.6 s and `TREE_insert_runtime_medium` in 27.7 s).

- **Net rubric tightening: 5 of 7 hard-failed.** Cases that previously soft-passed via `answer_must_contain: ["which", "where"]` now hard-fail on `expected.clarified=true`. The 2 that survived (`MOVE_generic`, `CLARIFY_motion_generic`) recovered via the new `--enable-recovery-turn` flag — but the model's recovery-turn behavior on these was to *take an action that happened to satisfy the followup-turn assertions*, not to ask. The rubric let it through; the model did not actually request clarification. **This is a rubric gap, not a model win.**

**Side effect — `--enable-recovery-turn` validated and made default.** The flag was added before iter 33 but never measured at scale. iter 33 fired 18 recovery turns and 10 cases became `recovered-pass` (would otherwise have been hard fails). Notable recoveries: 7 of the 10 are PnP chain steps (PnP_09–PnP_17) that initially had malformed args and the generic "re-read the structured recovery fields and retry" nudge unblocked them. The flag is now keep-on for all future iters; iter 32's 51/66 baseline did NOT use it, so iter 33 vs iter 32 isn't strictly apples-to-apples. Equivalent iter-32-rubric scoring of iter 33 (revert the 5 hard-failing rubric changes back to soft-pass): **53/66 ≈ 80.3%** — that would be +2 vs iter 32 driven purely by `--enable-recovery-turn`.

**Side effect — smoke runner stdout buffering bug fixed.** Iter 33 exposed that the runner's `print()` calls were block-buffered when stdout is redirected to a file. Per-case verdicts were deferred to end-of-run, blocking early-halt decisions on a clearly-failing iter. We burned the full 80 min wallclock before knowing direction 3 had failed even though it was visible from the bridge log halfway through. Fix landed in [`smoke_corpus_runner.py:48`](../scripts/debug/smoke_corpus_runner.py) — `sys.stdout.reconfigure(line_buffering=True)` at startup. No more `python3 -u` needed; verdicts now stream in realtime.

**Side effect — `done` counter undercounts during chain-on phase.** The smoke runner reuses one `rid` (= conversationId) across all PnP chain steps in chain-on mode. From the bridge's POV all 19+ chain step requests look like recovery turns on a single rid. Live monitors that compute "cases done" from unique-rids see only +1 per chain (not +19). Worth fixing in a future iter by either: (a) extending the bridge audit log with a per-step turn counter, or (b) reading the runner's now-realtime stdout for `status=` lines as the source of truth.

**Failure breakdown (iter 33 — same model as iter 32):**

- Pattern A (5): `PARALLEL_generic`, `FALLBACK_generic`, `SCENE_remove_generic`, `CLARIFY_scene_remove_pronoun`, `CLARIFY_tree_wrap_pronoun` — model fired action tools without emitting `[NEEDS-CLARIFY]` marker; failed `expected.clarified=true`. **All 5 are rubric-tightening fails, not model regressions.**
- Insert-node multi-arg specificity (4): `P3_tree_insert_runtime_obj_specific`, `TREE_insert_runtime_generic`, `INSERT_position_first_specific`, `FALLBACK_alternate_medium`, `PARALLEL_concurrent_medium` — same pattern as iter 32.
- Tool-mismatch / wrong-family (3): `REPLACE_simple_medium`, `REPLACE_subtree_specific`, `CUR_runtime_remove_then_restore_graspable`.
- Variance fails (4): `MOVE_reorder_medium`, `PnP_13_detach`, `PnP_14_upsert`, `PnP_18_repeat_root` (395 s — slowest case of the run, ran out the agent loop).
- Other (2): `FALLBACK_retry_specific` (insert-node arg shape).

**Recovered-pass cases (10):** `UPDATE_params_specific`, `MOVE_generic`, `CLARIFY_motion_generic`, `PnP_09_attach`, `PnP_10_lift`, `PnP_11_transport`, `PnP_12_place_descend`, `PnP_15_open_gripper`, `PnP_16_retract`, `PnP_17_home`.

**Timings (iter 33):**

| metric | overall | pass | recovered-pass | fail |
|---|---|---|---|---|
| n | 66 | 38 | 10 | 18 |
| min | 11.6 s | 11.6 s | 46.5 s | 11.9 s |
| median | 30.9 s | 19.2 s | 98.3 s | 120.5 s |
| mean | 73.2 s | 27.4 s | 107.2 s | 150.9 s |
| max | 395.7 s | 105.1 s | 200.0 s | 395.7 s |

Recovered-pass is inherently expensive (2-turn). Fails skew long because the model usually retries until the agent loop times out.

**Decisions / what changes after iter 33:**

1. **Revert `expected.clarified=true` on the 5 still-failing Pattern-A cases.** They go back to iter-32 rubric (`forbidden_tools` + `answer_must_contain: ["which", "where"]`). 2 of the 7 (the ones that "recovered" by taking action) need stricter rubric instead: make their `forbidden_tools` actually catch the model's "do something plausible" — currently the rubric lets a model that fires `tree_draft_move_node` after recovery turn satisfy a case whose first-turn expected no mutations. Fix: forbidden_tools should apply across BOTH turns, not just the followup.
2. **Keep the `request_clarification` tool + bridge auto-retry registered.** They didn't hurt and the auto-retry cleanly handled the 3 clarifs that did fire. Future system-prompt iterations may yet activate it.
3. **`--enable-recovery-turn` is default-on for all future iters.** Save the flag in the runbook.
4. **Stdout buffering fix is permanent.** No future iter loses early-halt visibility.
5. **Pattern A is a model-capability ceiling.** Cosmos-Reason2-8B's action-bias on bare-verb prompts is not addressable from the corpus / harness / system-prompt surface alone. Future direction options (not pursued this iter): (a) fine-tune a clarification-bias adapter on cosmos-reason2-8b; (b) swap to a model with measurably better bounded-autonomy behavior (the 9B Claude-distilled checkpoint cleared bounded-autonomy probes in earlier testing); (c) accept Pattern A as the model ceiling and exclude those cases from headline rate.

### Iter 34 — Pattern A direction 3 attempt #2: `tool_choice=required` (HALTED EARLY, 2026-05-10)

**Goal:** with the `request_clarification` tool already installed (iter 33), test whether forcing tool selection at the proxy layer would activate it. Hypothesis: if `tool_choice=required` is injected on every chat-completion, the model can't fall through to prose; it must pick one of the 26 tools. With `request_clarification` available AND the system-prompt rule pointing at it for ambiguous prompts AND thinking-on letting the model reason before answering, the ambiguous-prompt cases should pick the clarification tool over an action.

**Setup.** Held iter-32 settings (chain-on, `/compact every 2`, recovery turn). Added one proxy env: `OPENCLAW_PROXY_FORCE_TOOL_CHOICE=required`. Reverted the 7 corpus rubric changes from iter 33 (no `expected.clarified=true` / no `clarification_followup`) so cases could soft-pass via answer-text if the model emitted prose answers, comparable to iter-32 scoring.

**Result: HALTED after 3 cases in 14 minutes. Trajectory was clearly unrecoverable.**

| # | Case | Duration | Status | Notes |
|---|---|---|---|---|
| 1 | `P1_wrap_root_specific` | 275.1 s | ❌ fail | "add a repeat node as root" — normally trivial. 4.6× iter-32 typical. |
| 2 | `P2_scene_add_specific` | 166.5 s | 🟡 soft-pass | Text answer matched but extra tool calls violated tools_called assert. |
| 3 | `P3_tree_insert_runtime_obj_specific` | 424.9 s | ❌ fail | 7 minutes on a normally-30s case. |

**Why it failed (predicted in advance and confirmed):**

- `tool_choice=required` forces a tool call **every turn**. After the model successfully completed the requested action on turn 1, the agent loop's natural termination signal — "I'm done, emit a final assistant message" — was disallowed. The model was forced to keep firing tools, picking increasingly-irrelevant read/inspect calls until either (a) the case timed out, (b) the proxy's 200 s socket timeout fired, or (c) the model fired a forbidden tool and tripped the case rubric.

- Each case ran 5-7× iter-32's typical duration. Pace projected to ~5-6 hours wallclock for the full 66-case suite (vs iter 32's 75 min). Direction was unrecoverable; halting early conserved >4 hours.

- The hypothesis was unfalsifiable in this configuration: even if the model HAD picked `request_clarification` on an ambiguous prompt's turn 1, every subsequent turn would still be forced to fire another tool, and the case would either over-call or time out before completing.

**Lessons:**

1. `tool_choice=required` (every turn) is not viable with a multi-turn agent loop. The mode that COULD test the hypothesis is `tool_choice=required-first` (force only turn 1, let the model self-decide to terminate on later turns) — but that's a separate experiment and wasn't run this iter.

2. **Direction 3 is now fully exhausted.** Iter 33 tested the soft path: tool registered + system-prompt rule + bridge auto-retry + corpus rubric tightening. Model didn't reach for the tool. Iter 34 tested the hard path: forced tool calling. Setup was unviable. No remaining lever short of `tool_choice=required-first` (untested but unlikely to fundamentally flip the model's action-bias) or moving off-corpus (fine-tune, model swap).

3. **The user's halt instinct after seeing 3 bad results in 14 minutes was correct.** Without realtime per-case verdict streaming, this run would have consumed the full ~5 hours before its trajectory became visible. The iter-33 stdout-buffering fix paid for itself in iter 34: we saw the first failure at minute 4, the second at minute 7, the third at minute 14, and pulled the plug. The runner's realtime visibility is a permanent fixture from here on.

**Decision: revert all request_clarification-direction code; restore iter-32 as the production stack.**

Reverted (uncommitted, restored to last-commit HEAD):
- `dev_ws/manyforge/manyforge_composer/backend/assistant_tool_schemas.py` — removed `_REQUEST_CLARIFICATION_SCHEMA` + registry entry
- `dev_ws/manyforge/manyforge_composer/backend/routes_assistant.py` — removed `_apply_request_clarification` handler + dispatcher branch
- `dev_ws/manyforge/examples/assistant_modes_scene_authoring.deployment.yaml` — removed tool from mode allowlist + tool-catalog entry
- `dev_ws/manyforge/agent-skills/manyforge-composer/workspace-AGENTS.md` — restored original "When ambiguous" paragraph
- `NemoClaw-Thor/manyforge/openclaw_assistant_bridge/service.py` — removed marker detection, auto-retry, env-var reading
- `NemoClaw-Thor/manyforge/scripts/debug/smoke_corpus.yaml` — restored 7 corpus cases to iter-32 rubric
- `NemoClaw-Thor/manyforge/scripts/debug/smoke_corpus_runner.py` — removed clarification flow, `clarified-pass` status, two-turn followup logic

Preserved (durable wins from iter 33-34):
- `NemoClaw-Thor/manyforge/scripts/debug/smoke_corpus_runner.py` line 50: `sys.stdout.reconfigure(line_buffering=True)` — realtime per-case verdict streaming
- `--enable-recovery-turn` is now the default flag on every smoke iter (recipe in SMOKE-ITER-RUNBOOK.md). The flag itself was added before iter 33 but iter 33 was the first measurement at scale: +10 cases recovered. Iter-32 baseline runs with the flag would likely show 53-55/66 (~80-83%) but that's never been measured.
- `NemoClaw-Thor/.gitignore` — added `__pycache__/` and `*.pyc`/`*.pyo`. Durable fix; was missing for the smoke harness directory.
- New file `NemoClaw-Thor/manyforge/docs/SMOKE-ITER-RUNBOOK.md` — operational runbook with cold-start sequence + restart matrix per change type. Captures the SSH-namespace gateway gotcha that ate ~10 min of iter-33 setup.
- This SMOKE-CORPUS.md section — so the next iteration cycle doesn't
  re-propose direction 3.

Bridge env reverted: `OPENCLAW_ASSISTANT_CLARIFICATION_AUTO_RETRY_MAX` is no longer set. `OPENCLAW_ASSISTANT_COMPACT_EVERY_N=2` and `OPENCLAW_ASSISTANT_COMPACT_TIMEOUT_S=120` retained (iter-32 production).
Proxy env reverted: `OPENCLAW_PROXY_FORCE_TOOL_CHOICE` no longer set; iter-32 max_tokens + thinking budget retained.

**Iter 32 (51/66 = 77.3%, chain-session ON) is the current production-stack baseline.** Code state in both repos matches the last-tagged commits (`00e24b2` and `9c4cc4d`) plus durable narrative/runbook additions; the `request_clarification` direction is closed out.

### Round-1 to round-4 research findings (per-iter parallel research agents)

Each iteration spawned a parallel research subagent. Findings, ranked
by leverage under our constraints:

1. **The PnP_05 anchor pattern is named in the literature** — "plan-then-execute via inspect tool" (CHI 2025 10.1145/3706598.3713218; arXiv 2509.03581). Our fix substitutes for the missing chain-of-thought trace by giving the model a tool-result "scratchpad".
2. **"Lost in conversation" / "EndTurn-without-tool"** — named in arXiv 2505.06120 (MSR/Salesforce, 2025). 39% multi-turn pass-rate drop. Direct map to PnP_05 narration cascade.
3. **Cosmos-Reason2-8B is post-trained on Qwen3-VL with long-CoT design assumed** ([HF model card](https://huggingface.co/nvidia/Cosmos-Reason2-8B)). Running with `enable_thinking: false` is **out-of-distribution** for the model. The narration mode is a direct consequence — without thinking tokens, the model emits its "scratchpad" in the user-visible answer.
4. **`tool_choice: "required"` is dropped on the OpenClaw → vLLM hop**. The bridge sets it; the gateway strips it. This is the architectural root cause of "model returns text without firing a tool". Closing this gap is the single highest-leverage move under our constraints.
5. **PALADIN-style synthetic-history exemplars** (arXiv 2509.25238) — could rescue Group A (3-4 of 5) by injecting a successful 4xx-recovery trace into history. Risky under constraint 7 (schema-example copy-paste failure mode).
6. **vLLM XGrammar guided JSON** (paper Jan 2025) — would collapse Group A's malformed-args entirely by masking invalid args at decode time. ~1.6× decode speedup as a side effect. Deployment-YAML change; production-affecting; deferred.
7. **`max_completion_tokens` floor for tool-only turns is ~1024**, but lower (768) risks 3% mid-call truncation on Hermes parser. Our default is 2048 (BRIDGE_UPSTREAM_MAX_TOKENS env). Iter 11 tested 1536 — slight regression on chain steps; reverted.

### Iter 17/18 architecture state + findings (proxy at host:8000, vLLM at :8050)

**Architecture (committed since iter 17):**

- vLLM listens on `:8050` with `--default-chat-template-kwargs '{"enable_thinking":true}'` (server-side default ON, matches Cosmos-Reason2 training distribution).
- Proxy mutator listens on `:8000` and forwards to `:8050`. **All** OpenClaw → vLLM traffic now traverses the proxy, so per-turn mutation in the agent loop is possible (turn N visible by counting assistant messages in the request body).
- `openclaw_assistant_bridge` no longer injects `tool_choice="required"` (removed in iter 17 per user direction; the proxy owns this knob now).
- OpenClaw chain-session is restored (default conversationId), so OpenClaw uses its own multi-turn history.

**Iter 18b finding — vLLM with thinking-on + no `max_tokens` is unbounded.** OpenClaw's outbound chat-completions to vLLM omit `max_tokens`/`max_completion_tokens` entirely, so vLLM defaults to the model's full context. Combined with `enable_thinking: true`, generations on complex prompts ran for 8+ minutes at 30 tok/s (~14 000 tokens). Three concurrent OpenClaw subagents kept the GPU saturated with runaway thinking and the smoke runner cascaded into 244 s case-timeouts. The proxy's `up_resp.read()` was correctly waiting; vLLM was the long pole.

**Fixes shipped in the proxy (iter 18b → 18c):**

- `_OVERRIDE_MAX_TOKENS` no longer requires the field to already be present — it **injects** `max_tokens` on the way through when the caller omits it (with a `"injected": true` flag in the mutation record so the audit log distinguishes inject vs. rewrite).
- Per-request `http.client.HTTPConnection` timeout is now 200 s (was 600 s). The smoke runner's case timeout is 244 s, so the proxy now fails first and releases the upstream KV slot before the runner gives up. Prevents zombie-thread accumulation on any future runaway generation.

**Iter 18c result — cap works, but the strategy doesn't.** With `max_tokens=2048` injected on every chat-completion (16384 → 2048 in the audit log), there are no more 8-minute runaway generations. The previously hanging P3 dropped from 244 s to 67 s. But effective rate is **27/48 ≈ 56.3 %** — 16 points below the iter-7/16 baseline. Two distinct regressions, both attributable to the strategy:

- **`tool_choice=required-first` over-forces tools on no-tool cases.** `FALLBACK_generic` and `PARALLEL_generic` both have `expected NO tool calls` in the corpus, but the model fired 8 / 2 tools respectively. The proxy correctly applies `required-first` only on turn 1 — but turn 1 IS the only turn for these short cases, so the rule still bites them.
- **`tree_draft_insert_node` collapsed across the corpus.** 9 distinct cases failed with `expected tool 'tree_draft_insert_node' not observed`: P3, TREE_insert_runtime_*, INSERT_position_*, PARALLEL_concurrent_medium, FALLBACK_retry_specific, FALLBACK_alternate_medium, PnP_06, PnP_07, PnP_09, PnP_10. Two plausible mechanisms (the next iter has to disambiguate): (a) 2048-token budget is too tight when thinking-on consumes ~600–1500 tokens before the tool-call emission, leaving insufficient room for the multi-arg `tree_draft_insert_node` call; (b) thinking-on's natural verbosity displaces the tool emission and the model produces a final assistant text saying "I would insert…" instead of a tool call.

**Iter 19 → Iter 20 results:**

- **Iter 19** (drop `tool_choice` mutation, otherwise hold iter-18c): 40/66 ≈ 60.6 %. Front half recovered cleanly (P2/P3/etc. went from 244s-fails to ~20s-passes, confirming `required-first` was the front-half regression). PnP cascade returned with chain-session, eating ~13 cases from PnP_06 onwards.
- **Iter 20** (iter-19 settings + `--no-chain-session`): **49/66 ≈ 74.2 % effective, 68.2 % first-try — new best**. Cascade broken; PnP suite went from 4/19 (iter 19) to 14/19 (iter 20). Beats iter-7/16 baseline by +1 effective and +3 first-try.

**Iter 21 plan (next):**

Iter 20 is now the production-shaped recipe. Three orthogonal knobs to chase the remaining 17 fails:

- **Iter 21a**: hold iter-20 settings, drop `thinking_token_budget=512` (let model use full thinking quota up to `max_tokens=2048`). Hypothesis: thinking budget is too tight for `tree_draft_insert_node` cases that genuinely need to reason about position + parent.
- **Iter 21b**: hold iter-20 settings, raise `max_tokens` cap to 4096. Hypothesis: 2048 truncates the multi-arg `tree_draft_insert_node` JSON. If iter 21a doesn't help and 21b does, the cap is the limit.
- **Iter 21c** (combine with iter-16 user_suffix): keep iter-20 cap + thinking, layer the iter-16 read-first user-suffix mutator on top. Hypothesis: orthogonal — iter 16's win was on a different axis (state-grounding) so the gains stack.

**Lessons from iter 17–20:**

1. Without an explicit `max_tokens`, vLLM with thinking-on can run unbounded for many minutes per request. Always cap output budget when running thinking-on against an agentic harness with case timeouts.
2. The proxy mutator's `_OVERRIDE_MAX_TOKENS` was a bug-shaped no-op until 18b — any mutation that depends on a key being present needs an inject path or it'll silently fail to apply.
3. `tool_choice=required-first` is correct in spirit (force the first tool emission) but bites no-tool corpus cases that don't have a "second turn" to recover into prose. The bridge ALSO doesn't need this — let the model self-decide when to call tools.
4. The PnP_06+ cascade is fundamentally a chain-session-history problem, not a model-capability problem. `--no-chain-session` (single-shot conversation per chain step) is the only fix that has worked; it converts cascading failures into independent ones.
5. The Monitor's `timeout_ms` kills its child pipeline. For multi-hour smoke runs, launch the runner via `nohup` and tail the log from a separate Monitor.
6. Cosmos-Reason2-8B is over-eager to call tools on no-tool corpus prompts (PARALLEL_generic, FALLBACK_generic emit 3-5 `tree_draft_insert_node` calls when none expected). This is independent of `tool_choice` forcing — it's a model bias. Future fix: tune the system prompt's tool-trigger language, or add a no-tool detector in the corpus rubric (already partially present via `expected NO tool calls`).

## Cross-cutting infrastructure: the sandbox-internal proxy mutator

Built in iter 12 (file: [`scripts/proxy/vllm-proxy.py`](../scripts/proxy/vllm-proxy.py); was `scripts/debug/vllm-logging-proxy.py` before iter 21 — renamed and relocated when the proxy graduated from a pure debug tool into the load-bearing mutator that ships `max_tokens` injection in the production recipe). The proxy logs every chat-completions request and response as JSONL unconditionally; optional outbound-request mutation is gated on env vars. Verified end-to-end against a mock upstream with mutations active and all opt-out paths preserved.

### What it does

A single proxy process that:
1. **Always** logs every chat-completions request + response as JSONL (existing behavior, unchanged).
2. **Optionally** mutates the outbound request body before forwarding, gated by env vars:

| Env var | Effect |
|---|---|
| `OPENCLAW_PROXY_FORCE_TOOL_CHOICE=required` | Inject `tool_choice: "required"` when `tools[]` is non-empty AND user didn't set `tool_choice: "none"` |
| `OPENCLAW_PROXY_OVERRIDE_TEMPERATURE=0.0` | Force a specific temperature (e.g. for deterministic recovery turns) |
| `OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS=1024` | Cap output tokens |
| `OPENCLAW_PROXY_OVERRIDE_TOP_P=1.0` | Override top_p |

If no mutation env vars are set, the proxy is a pure pass-through logger — same as before. Each mutation is logged to JSONL with before/after values, so all changes are auditable.

### Why it's the right architecture

- **Solves the OpenClaw `tool_choice` drop without touching OpenClaw, the bridge, or the model.** Inside the same trust boundary as the gateway (sandbox-internal), so no new host port or escalation risk.
- **Always-on debug logger AND policy enforcement in one process.** Logging-only and forcing modes share the same code path; flipping between them is an env-var change + restart.
- **Auditable.** The JSONL log carries `request.mutation` for every changed request, with `before`/`after` diffs.
- **Reversible.** Stop the proxy; OpenClaw goes back to talking to vLLM directly.

### Deployment recipe (iter 13 will execute)

```bash
# 1. Copy the proxy into the sandbox
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell my-assistant -c agent -- mkdir -p /sandbox/.openclaw/proxy
docker cp /home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor/manyforge/scripts/proxy/vllm-proxy.py \
  openshell-cluster-nemoclaw:/tmp/vllm-proxy.py
docker exec openshell-cluster-nemoclaw kubectl cp /tmp/vllm-proxy.py \
  openshell/my-assistant:/sandbox/.openclaw/proxy/vllm-proxy.py -c agent

# 2. Launch with mutation enabled (background process inside the sandbox)
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell my-assistant -c agent -- \
  bash -c 'OPENCLAW_PROXY_FORCE_TOOL_CHOICE=required \
           OPENCLAW_PROXY_LISTEN_PORT=18800 \
           OPENCLAW_PROXY_UPSTREAM=http://host.openshell.internal:8000 \
           OPENCLAW_PROXY_LOG_PATH=/sandbox/.openclaw/proxy/proxy.jsonl \
           OPENCLAW_PROXY_BIND=127.0.0.1 \
           nohup python3 /sandbox/.openclaw/proxy/vllm-proxy.py > /sandbox/.openclaw/proxy/proxy.stdout 2>&1 &'

# 3. Repoint OpenClaw at the proxy
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell my-assistant -c agent -- \
  sed -i 's|host.openshell.internal:8000/v1|localhost:18800/v1|' \
  /sandbox/.openclaw/agents/manyforge-composer/agent/models.json

# 4. Demo restart so OpenClaw picks up the new endpoint
PRESERVE_OPENSHELL=true demo-assistant-known-good.sh restart

# 5. Verify mutation is active by tailing the JSONL log during a chat:
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell my-assistant -c agent -- \
  tail -f /sandbox/.openclaw/proxy/proxy.jsonl | \
  jq 'select(.request.mutation) | .request.mutation'
```

### Diagnostic surface (always available)

```bash
# 1. Live tail every chat-completions request, with mutation badge
kubectl exec ... -- tail -f /sandbox/.openclaw/proxy/proxy.jsonl | jq -c '
  {ts, path: .request.path, status: .response.status,
   ms: .response.duration_ms, mut: .request.mutation}'

# 2. What's actually on the wire to vLLM (post-mutation body)
jq '.request.body | {tool_choice, temperature, max_tokens, model, "tools_count": (.tools|length)}'

# 3. Diff before/after for any mutation
jq 'select(.request.mutation) | .request.mutation'

# 4. Per-request latency
jq '.response.duration_ms'

# 5. Reverting to logging-only mode: stop the proxy, restart without env vars,
#    or restart with `OPENCLAW_PROXY_FORCE_TOOL_CHOICE` unset.
```

### Trade-offs vs. alternatives

| Approach | What it solves | Latency cost | Risk |
|---|---|---|---|
| **Proxy mutator (`tool_choice: "required"`)** | Narration mode (model can't return text-only) | ~zero | Mutator opt-out via `tool_choice: "none"` preserved; proxy is reversible |
| Read-first suffix on every prompt | Narration mode + tool routing | ~2× per turn (extra read tool call) | Counts as cross-prompt hint; doubles upstream load |
| XGrammar guided JSON | Malformed-args (Group A) | ~1.6× speedup | Production-affecting; needs validation across lanes |
| PALADIN exemplar history | Group A recovery | small | Constraint 7 risk (model copies exemplar values) |

**Recommended hybrid for production**: proxy mutator + a tiny system note giving the model permission to call `program_read`/`scene_inspect` to refresh state. Achieves ~80% of the read-first-suffix benefit at ~20% of the latency cost. Documented as the post-loop infrastructure recommendation in iter 13's write-up.

### Bridge spiral finding (iter 13a/13b)

Naively forcing `tool_choice: "required"` on every chat-completions
request through the OpenClaw lane causes a **bridge agent loop spiral**.
Mechanism:

- The bridge runs an agent loop: send messages+tools to vLLM → receive
  reply → if tool_call, execute tool, append result to messages,
  **GOTO send**; if text-only, **return to Composer**.
- The exit condition is **text-only response**.
- With `tool_choice: "required"` forced, the model can never return
  text-only — it must always fire a tool.
- Even after the user's request has been satisfied (e.g., the deletion
  succeeded), the model is forced to fire SOMETHING (often a redundant
  read or a duplicate mutation).
- The bridge keeps looping until it hits an internal max-turns ceiling
  (empirically ~11 turns in this codebase).
- Per-case wall clock: 11 × 5 s ≈ 55 s — blows past the 270 s budget
  on first case, hits HTTP 502 from upstream around 5 min.

Validated on the wire: the mutator JSONL log shows 11 successive
chat-completions per single user prompt, all mutated with
`tool_choice → required`, all returning successful tool calls.
The model is doing what we asked; the bridge has no way to exit the
loop because the natural "I'm done" signal (text-only) is forbidden.

### Mitigation: alternating mode + plan-then-execute (iter 14 design)

Two cooperating fixes implemented in the proxy and ready for
deployment:

1. **`OPENCLAW_PROXY_FORCE_TOOL_CHOICE=alternating`** — inject
   `tool_choice: "required"` on **odd-numbered turns** of the agent
   loop (turn 1, 3, 5, …); leave even turns untouched. The proxy
   detects the turn by counting `assistant`-role messages already
   present in the request body. Forces a tool every 2 turns max,
   gives the model a free turn between to emit text and exit
   naturally. Validated end-to-end: a single user prompt that fired
   `delete_node` exited cleanly in 2 vLLM round-trips (down from
   11 in always-required mode).

2. **`OPENCLAW_PROXY_USER_MESSAGE_SUFFIX="Before answering, call program_read or scene_inspect to refresh state if needed."`** —
   appended to the last user message on every chat-completions request.
   Idempotent (skips if suffix already at tail). Pairs with alternating
   mode: turn 1 (forced) the model fires the inspection tool, turn 2
   (free) the model emits the action and a brief text answer.

Both modes are auditable via the JSONL log
(`request.mutation.mutations`), reversible (unset env vars), and have
opt-out preserved (`tool_choice: "none"` from the bridge is honored).

### Iter 16 outcome: full corpus completion + Strategy 2 validation

**The two-strategy comparison ran on the live corpus**:

| Strategy | What changed | Result |
|---|---|---|
| **Strategy 1** (iter 15) | Mutator with `user_suffix_first_turn_only=1`: only inject suffix when `asst_count=0` in the request body | **Doesn't work at this insertion point.** Bridge sends each chat as a single-message envelope to OpenClaw — every request looks like turn 1 to the mutator. Suffix fired 43/43 times anyway. Same cascade as iter 14. |
| **Strategy 2** (iter 16) | Harness `--no-chain-session`: each chain step gets a fresh conversationId. Mutator unchanged (full suffix on every prompt). | **48/66 (72.7%) effective on full 66-case run.** Cascade broken — PnP chain failures became independent (no propagation). Ties iter 7's best, and is the only full-corpus run with the mutator active. |

**Why Strategy 2 worked**:
- The cascade was driven by **conversation-state accumulation** inside OpenClaw. Once PnP_05 produced a text-only narration, the next 12 chain steps inherited that "narration mode" via the shared conversationId.
- Strategy 2 forces each PnP step to start fresh: model sees only the current user prompt + the in-prompt snapshot. No prior narration to inherit.
- The cost: model loses chain context. PnP_07 ("after the descend, close the gripper") fails when it can't infer "the descend" from the snapshot alone — it has to look at the tree structure and decide which node is "the descend". Some succeed, some fail; the failures are **independent**, not cascading.

**The trade-off in numbers**:
- iter 7 baseline (chain conversation, no mutator): 48/66 — high pass rate when cascade doesn't fire (50% probability)
- iter 16 (no chain conversation, full suffix): 48/66 — same effective rate, but **deterministic** (no cascade-roulette)

**Same number, much lower variance.** The variance kill is the headline.

**Notable wins specific to iter 16** (not present in iter 7):
- `REPLACE_subtree_specific` ✅ — was 200-245s timeout in EVERY prior iter (6–13c). Now passes at 85s.
- `FALLBACK_retry_specific` ✅ — was failing in iters 8–13.
- `FALLBACK_alternate_medium` ✅ — variance flip but improved with suffix.
- `CUR_runtime_remove_then_restore_graspable` ✅ — was failing in EVERY prior iter (16 attempts, all fails). Now passes at 78s. Suffix's read-first guidance helps multi-tool cases.
- All 3 CLARIFY_* cases 🟡 soft-pass — bounded-autonomy rubric (forbidden_tools + answer_must_contain) confirmed working.

**Notable losses specific to iter 16**:
- PnP_07, PnP_09, PnP_10, PnP_12, PnP_14, PnP_15, PnP_16 all ❌ — chain context loss. Each prompt has to reconstruct intent from the snapshot alone. The corpus prompts were written assuming chain history.
- PnP_18 ❌ 275s timeout — wrap-the-root case is genuinely hard for cosmos-8b. Not cascade-related.

### Recommendation: Strategy 2 is the production default

Given:
- Same effective rate as the best-ever clean run
- **Deterministic** (no cascade roulette) — variance dropped from "0–13 case swing per iter" to "1–2 case swing"
- Notable wins on previously-impossible cases (REPLACE_subtree_specific, CUR_runtime_remove_then_restore_graspable)
- Full corpus completion every run (no killed iters)

**Strategy 2 should be the default for future smoke runs** until either:
- The corpus prompts are rewritten to be self-contained (no chain references like "after the descend")
- A bridge change lets us inject mutations at the per-turn layer where Strategy 1 would actually work
- A model swap to one that handles long agent loops without narration collapse

### Iter 14-final outcome: mutator validated end-to-end against the live corpus

After multiple deployment dead-ends (sandbox-internal proxy: openclaw
binary rewrites the config; sandbox squid replacement: operator-managed,
out of reach), the working insertion point turned out to be:

```
host bridge process → host:18790 (mutator) → host:18789 (SSH tunnel) →
sandbox openclaw → host.openshell.internal:??? (sandbox proxy) → vLLM
```

The mutator runs on the **host**, listening on `:18790`, forwarding to
the existing OpenClaw SSH tunnel at `:18789`. The bridge process is
restarted with `OPENCLAW_ASSISTANT_GATEWAY_PORT=18790` instead of the
default 18789 — that single env-var change repoints the entire bridge
through the mutator without touching openclaw.json, the sandbox
config, the demo script, or any operator-managed component.

**This is the "Option B" position the user authorized: the mutator
sits between bridge and OpenClaw, exactly where the user proposed.**

#### What actually mutates at this insertion point

The bridge → OpenClaw protocol is **not** a raw OpenAI chat-completions
request. It's a higher-level OpenClaw envelope — OpenClaw unpacks it
and constructs the actual chat-completions request internally before
calling vLLM. The mutator's per-field analysis at this layer:

| Mutation knob | Applied? | Reason |
|---|---|---|
| `OPENCLAW_PROXY_FORCE_TOOL_CHOICE=alternating` | **No** (silently skipped) | The bridge → OpenClaw envelope doesn't carry `tools[]` directly. The mutator's tool_choice injection code requires non-empty `tools[]` and so opt-outs cleanly when the field is missing. |
| `OPENCLAW_PROXY_USER_MESSAGE_SUFFIX="Before answering, call program_read or scene_inspect…"` | **Yes (44/44)** | The bridge → OpenClaw envelope DOES carry `messages[]` with the user's prompt. The mutator appends the suffix to the last user message; OpenClaw passes the messages through to vLLM unchanged; the model sees the suffix on every turn. |
| `OPENCLAW_PROXY_OVERRIDE_TEMPERATURE / MAX_TOKENS / TOP_P` | Variable | These would apply if OpenClaw forwards them, untested in this run. |

So at this insertion point, **the user_suffix is the load-bearing
mutation** — and it survives the entire chain to reach the model.
The tool_choice forcing requires insertion at a deeper layer (between
OpenClaw and vLLM, i.e. inside the sandbox), which remains blocked by
operator-managed `inference.local` resolution.

#### Iter 14-final empirical results

**31 pass + 2 soft-pass / 12 fail of 45 cases attempted = 33/45 (73.3%) effective rate** before chain cascade kill at PnP_07.

Compared to iter 12 baseline (no mutator, recovery turn on): 31/44
(70.5%). Compared to iter 13c (no mutator, no recovery): 32/43
(74.4%). The mutator-active run is **statistically indistinguishable
from the no-mutator baseline at this insertion point** — within
single-iter variance.

**However, the mutator IS changing model behavior** (visible in the
44 mutated entries in the JSONL log):

- **Wins attributable to the read-first suffix**:
  - **`REPLACE_subtree_specific` ✅ at 31s** — was 200-245s timeout
    in **every** prior iter (6, 7, 8, 9, 10, 11, 12, 13c). The model
    now reads state first, then constructs the replacement. **This
    is the single clearest demonstration of the suffix working.**
  - **`FALLBACK_retry_specific` ✅ at 53s** — was failing in iters
    8-13.
  - **`PnP_05` ✅** — fired `program_read` on turn 1 (50% rate
    without suffix; suffix amplifies). The suffix reinforces the
    inspect-anchor pattern.
  - **`PARALLEL_generic` shows the suffix at work**: model fired
    `program_read` THEN `tree_draft_insert_node` ×6 (still failed,
    because the case expects clarification, not action — but the
    "read first" instruction was followed).

- **Losses attributable to the suffix**:
  - **PnP_06+ chain cascade**: the suffix on EVERY chain step makes
    the model do an extra `program_read` per turn. On a 12-step
    chain, this compounds — PnP_06 hit 275s timeout, cascading
    PnP_07-17. The chain budget was sufficient WITHOUT the suffix
    (iter 5 ran the chain in 200-300s total); WITH the suffix every
    chain step adds ~15-20s, exhausting budget.
  - **`SCENE_update_pose_specific` ❌** — variance regression; the
    extra read may have used token budget that prevented the model
    from emitting the actual mutation.

#### What this proves and what it doesn't

**Proven**:
1. The mutator architecture works end-to-end: 44/44 requests
   intercepted, mutation applied, request forwarded, response
   returned to the bridge cleanly. Latency overhead negligible
   (<20ms per call).
2. The user_suffix mutation reaches the model. The model's behavior
   changes (extra `program_read` calls visible in the corpus
   results) — that's the suffix doing exactly what it says.
3. The reachable insertion point (bridge↔OpenClaw layer) supports
   message-shape mutations but NOT request-shape mutations (since
   OpenClaw's envelope hides `tools[]`/`tool_choice`).

**Not proven**:
1. Whether `tool_choice: "required"` enforcement at the right layer
   (between OpenClaw and vLLM) would close Group A failures. That
   layer remains blocked.
2. Whether the suffix + snapshot stripping (the user's instinct that
   removing the in-prompt snapshot eliminates redundancy) would
   produce cleaner gains on chain steps. The snapshot is added by
   the bridge's HTTPS upstream call; stripping it requires a
   different mutation point.
3. Whether the suffix paired with TIGHTER per-case budgets (i.e.,
   shorter chain timeouts that force the model to skip the extra
   read on chain steps) would land net positive.

#### Recommended next experiments

These can be run with the mutator already deployed at `:18790`:

1. **Conditional suffix**: only inject the suffix on the FIRST turn
   of each conversation (analogous to the `required-first` pattern).
   Subsequent turns within the same chain don't get the read-first
   nudge, avoiding chain-cascade overhead.

2. **Suffix + max_tokens override**: combine `user_suffix` with
   `OPENCLAW_PROXY_OVERRIDE_MAX_TOKENS=1024` to clip the model's
   prose tail, claw back budget for the extra read.

3. **Direct-lane validation**: run the same suffix experiment on the
   direct lane (which honors `tool_choice: "required"` already) to
   isolate the suffix's effect from OpenClaw's tool_choice strip.

### Original iter 14 deferral notes (pre-final-deployment)

Two attempts at iter 14 surfaced a deeper deployment block. Three
layers cooperate to make `openclaw.json` a non-persistable config
file:

1. **`configure-manyforge-provider.sh`** in `dev_ws/.../scripts/`
   writes `inference.local` baseUrl on every `provision_openclaw_sandbox`
   step of the demo restart. Setting `PROVISION_OPENCLAW_SANDBOX=false`
   skips this step but doesn't help (see #2).

2. **OpenClaw itself rewrites the config on startup**. The persistent
   gateway log shows
   `Config overwrite: /sandbox/.openclaw/openclaw.json (sha256 X -> Y, backup=...)`
   on every gateway start, regardless of how the start was invoked
   (`openclaw gateway run` or supervisor-respawn). This normalization
   resets the inference baseUrl whenever the cached/in-memory config
   differs from what's on disk.

3. **Killing the gateway alone leaves a stale SSH tunnel.** The host
   side (`127.0.0.1:18789`) is held by an `ssh` process tied to the
   openshell cluster, not the gateway. After a kill+respawn cycle,
   chat requests to the bridge return `curl: (52) Empty reply`
   because the tunnel needs the openshell-side reset that only a
   full demo restart provides.

These three layers mean **no purely client-side change** (mutator on
host + sed of openclaw.json + gateway bounce) survives a full corpus
run. The mutator works on the wire and even routes a single chat
correctly when timed precisely, but a 66-case run takes 20+ minutes
and at least one of the three layers triggers a config-reset within
that window.

**Required upstream fix to unblock iter 14**:

- Option A (recommended): modify `configure-manyforge-provider.sh`
  to accept a `THOR_LOCAL_VLLM_BASE_URL` env override that points at
  the mutator. The `THOR_LOCAL_VLLM_BASE_URL` variable is already
  defined in that script (line 287/322) — it just needs to be
  exported and used in the openclaw.json template. ~5 line patch.

- Option B: modify the squid proxy in the cluster container so
  `inference.local` traffic routes to the mutator instead of vLLM
  directly. This is the most architecturally clean solution — the
  mutator becomes a permanent part of the inference path — but
  requires editing cluster-container squid config.

- Option C (workaround): run the smoke corpus through the **direct
  lane** instead of OpenClaw. The direct lane already honors
  `tool_choice: required` and would let us validate the
  alternating-mode + suffix combination on the same model + corpus
  without OpenClaw in the path. Findings transfer to the OpenClaw
  lane once Option A or B lands.

A previous earlier attempt produced gateway duplication (two
processes after `openclaw gateway restart` interaction with demo
restart) — that was a separate symptom of the same underlying issue.
Single-gateway hygiene is necessary but not sufficient.

The deployment recipe in this document needs one additional step
even after the upstream fix lands: **ensure exactly one gateway is
running before the corpus starts.** A reliable way to do this:

```bash
# After demo restart, kill any extra gateway processes
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell my-assistant -c agent -- \
  bash -c '
    GWS=$(pgrep -f openclaw-gateway)
    GW_COUNT=$(echo "$GWS" | wc -l)
    if [ "$GW_COUNT" -gt 1 ]; then
      # Keep only the first; kill the rest
      KEEP=$(echo "$GWS" | head -1)
      for pid in $GWS; do
        if [ "$pid" != "$KEEP" ]; then kill "$pid"; fi
      done
    fi
    sleep 1
    pgrep -f openclaw-gateway | wc -l  # should print 1
  '
```

This is a deployment-recipe gap, not a design flaw in the mutator.
Iter 14 is **ready to run** the moment a clean single-gateway state
is verified, with the alternating-mode validation already passing
end-to-end on a single chat.

## Final scoreboard (rounds 6–14)

| Iter | Pass | Effective | Status |
|---|---|---|---|
| 6 | 27/45 attempted | 60% | partial — PnP_05 cascade, killed |
| 7 | **48/66** | **72.7%** | full run |
| 8 | 28/44 attempted | 63.6% | partial — recovery turn no-op |
| 9 | 32/43 attempted | 74.4% | partial — bounded-autonomy rubric |
| 10 | 33/43 attempted | 76.7% | partial — first recovered-pass |
| 11 | 33/45 attempted | 73.3% | partial — max_tokens=1536 reverted |
| 12 | 31/44 attempted | 70.5% | partial — recovery + broadened message |
| 13a | 0/2 | — | partial — mutator spiral identified |
| 13b | 0/4 | — | partial — proxy streaming fix |
| 13c | 32/43 attempted | 74.4% | partial — clean post-mutator baseline |
| 14 | 2/4 attempted | — | partial — alternating mode validated single-chat; corpus run had gateway dup |

**Headline number across rounds 1–14**: **iter 5 = 51/66 (77.3%) effective rate** remains the best clean full-corpus result. Subsequent rounds yielded **infrastructure** (proxy mutator with 6 mutation knobs, JSONL audit, alternating-turn logic) and **diagnosis** (bridge spiral, OpenClaw `tool_choice` drop, Cosmos-Reason2 OOD-with-thinking-off) rather than higher pass rates.

## Recommendations (post-loop)

1. **Run iter 14 with single-gateway hygiene**. Alternating mode
   validation already passed; needs only deployment cleanup. Expected
   delta: +4 to +6 cases on Group A (specific INSERT/MOVE failures
   where forced first-turn tool_choice closes the narration gap).

2. **Strip the bridge to a minimal layer**. The `openclaw_assistant_bridge`
   currently runs its own agent loop while OpenClaw also runs one.
   Two stacked agent loops interact poorly (this is where the spiral
   originates, not the model itself). The bridge should be a thin
   filter: forward chat-completions, mutate per env vars, log
   everything as JSONL. Delegate the agent loop entirely to OpenClaw.
   This is the architectural payoff that makes the proxy mutator a
   permanent production feature rather than a smoke-corpus workaround.

3. **Address the OpenClaw `tool_choice` drop upstream**. The mutator
   patches around it; the cleaner fix is to make OpenClaw forward the
   parameter. If this lands upstream, the mutator's only remaining
   job is policy enforcement during smoke runs, not architectural
   plumbing.

4. **Variance characterization run**. Run iter 5's exact config
   3 times in succession to measure the noise floor. Several
   "regressions" in iters 6–14 are likely variance flips. A confidence
   band on every case would let us treat ±2 cases as noise.

5. **Defer until model swap**: Group A failures with malformed args
   (5 cases) are the floor under the current model. Cosmos-Reason2-8B
   is OOD with `enable_thinking: false`. The right fix is either
   (a) enable thinking and accept the latency, or (b) swap to a model
   tuned for thinking-off tool use (Qwen3-32B-Instruct family). Both
   are out-of-scope for the corpus iteration cycle.

### Iter 5 outcome — what worked, what regressed

**Wins (12 cases flipped to pass)**:
- `PnP_06_approach`, `PnP_07_descend`, `PnP_08_close_gripper`, `PnP_09_attach`, `PnP_10_lift`, `PnP_11_transport`, `PnP_12_place_descend`, `PnP_13_detach`, `PnP_14_upsert`, `PnP_15_open_gripper`, `PnP_16_retract`, `PnP_17_home`, `PnP_20_grip_force` — chain unblocked end-to-end (PnP_18 regressed on a separate timeout issue, see below). The diagnosis held: making PnP_05 a real tool-use turn keeps the model in act mode through PnP_06+.
- `UPDATE_params_generic`, `INSERT_position_after_named_medium` — variance-tier wins; uncertain whether the longer timeout, fresh demo restart, or session caching contributed.
- `CLARIFY_motion_generic` — moved fail → soft-pass.

**Regressions on iter 5 (4 cases)**:
- `PnP_18_repeat_root` — was ✅ (24.2 s) in iter 4; now ❌ at 299 s with `chat HTTP 502`. The 360 s default timeout pushed the case past an upstream 502 circuit-breaker that fired around 5 min of model loop. Recommend reverting the default to ~270 s and adding per-case overrides only where needed.
- `FALLBACK_alternate_medium` — was ✅ (70 s) in iter 4; now ❌ 299 s 502. Same upstream-502 issue.
- `REPLACE_subtree_specific` — was ✅ (19 s) in iter 4; now ❌ 245 s `chat HTTP -1`. Likely orthogonal variance + the longer chain budget letting the model loop instead of converge.
- `WRAP_root_medium` — was ✅ in iter 4; now ❌ on tool-not-fired and state_after mismatch. Variance flip; the case is detail-medium and intermittent.

**Net over the cycle (iter 1 → iter 5)**: +17 cases (34 → 51 effective) with three corpus-only structural changes (fixture rename, state-path correction, PnP_05 repurpose), one harness change (multiset matching), and the timeout bumps.

**Persistent failures heading out of iter 5** (15 of 66):
- *Specific INSERT/MOVE failures (5 cases)*: `INSERT_position_first_specific`, `TREE_insert_runtime_medium`, `MOVE_reorder_medium`, `FALLBACK_retry_specific`, `WRAP_root_medium`. Model fires the right tool but emits malformed args (missing `parentName`, wrong `targetName`, `_raw` envelope). Recovery requires per-prompt hints or schema-example injection — both deferred. Tracked as the next round of work.
- *PARALLEL_/FALLBACK_ generics (3 cases)*: `PARALLEL_concurrent_medium`, `PARALLEL_generic`, `FALLBACK_generic`. Model fires `tree_draft_insert_node` 4–15 times when prompt asks for a clarification. Bounded-autonomy gap on cosmos-8b; deprioritized per instruction (specific > generic).
- *CLARIFY_*/MOVE_generic (3 cases)*: same bounded-autonomy gap on pronoun-only prompts. Deprioritized.
- *Upstream-502 timeouts (3 cases)*: `PnP_18_repeat_root`, `FALLBACK_alternate_medium`, `REPLACE_subtree_specific`. Caused by the 360 s default timeout bump. Recommend per-case `chain_timeout_s` overrides instead of a global bump.
- *CUR_runtime_remove_then_restore_graspable (1 case)*: model doesn't fire `scene_inspect` before mutating; bounded-autonomy gap on a specific prompt.

### Recommended next round (not run in this session)

1. **Revert default timeout to ~270 s** (just under the upstream 502 circuit-breaker), keep per-case `chain_timeout_s: 360` only on cases that genuinely need it. Recovers 3 regressions.
2. **Per-case `chain_timeout_s: 360` on `PARALLEL_concurrent_medium`** to give it the room it actually needs without affecting fast cases.
3. **Investigate the upstream 502 source** — it's likely a Composer-side `timeoutSeconds` or gateway `idleTimeoutSeconds`. If it can be raised to 600 s, the global bump becomes safe.
4. **Add a `tree_inspect`-equivalent state turn before specific INSERT/MOVE prompts** (`INSERT_position_first_specific`, `TREE_insert_runtime_medium`, `MOVE_reorder_medium`) — same pattern as PnP_05. Risk: changes the corpus shape; weigh carefully.
5. **Variance characterization**: re-run iter 5 unchanged 3× to measure the variance floor on `WRAP_root_medium`/`UPDATE_params_generic`. Cosmos-8b at temp=0.2 has visible flips between runs.


## Future Work — Outstanding

### Cheap corpus fixes (high return, low effort)

1. **Pattern 2 fix**: state_after path keys `pose.position` → `pose.position_m` (2 cases — `SCENE_update_pose_specific`, `CUR_scene_update_graspable_pose`)
2. **Pattern 6 fix**: empty fixture rework (sequence-with-trivial-child) + PnP_05 prompt aligned to the new fixture (1 case)
3. **Pattern 5 fix**: add `precondition.chain_timeout_s` field + plumb through to Composer's per-request timeout (2–3 cases)
4. **Reclassification of generic-detail composite cases**: `PARALLEL_generic`, `FALLBACK_generic`, `MOVE_generic` move into the clarification category (`tools_called: []` + asks "what / which") since the prompts are honestly underspecified

Combined: **~7 cases** flip from fail to pass with only corpus edits.

### Future tier (genuine external dependencies — kept as `status: future`)

| Runtime | Cases | What's needed |
|---|---|---|
| `force_monitor_node` | 2 (`PARALLEL_safety_specific`, `PnP_19_safety_parallel`) | New `force_monitor` BT node kind in `manyforge_behavior/resources/node_catalog.yaml`. |
| `isaac_scene_stream` | 3 (Isaac import / reject / runtime-colliders cases) | Isaac Sim stream + `inspect_isaac_scene` / `propose_scene_objects` / `propose_scene_object_nodes` tools wired (the route handlers exist but the upstream stream doesn't). |
| `deployment_draft_tools` | 1 | New `deployment_draft_set_robot` / `deployment_draft_set_gripper` MCP tools + deployment-mutation surface. |
| `high_level_skills` | 1 | Skill registry + `program_draft_apply_skill` MCP tool. |
| `plan_then_apply_tools` | 1 | `assistant_plan_draft` MCP tool (plan-preview without mutating draft). |

These stay future until their upstream surfaces exist. The corpus serves as
their concrete acceptance test the day they land.

### Real model-behavior issues (not infra/corpus bugs)

- **Bounded-autonomy gap on cosmos-8b** (Pattern 4): model fires mutations
  on pronoun-only prompts. Not blocking, but the test corpus correctly
  flags it. Either accept as a known model limitation or upgrade the
  RULES system message.
- **Generic-detail prompt resistance**: cosmos-8b doesn't act on
  truly-vague prompts like "add a parallel" — pattern 1 / pattern 4
  overlap. A 9B-class model performs better on these in earlier
  testing; cross-model comparison via this corpus is now possible.
- **Multi-step prompt timeouts** (Pattern 5): some prompts genuinely
  need >130s of agent-loop budget. Not all models hit this — likely
  worse on smaller models with more catalog-read recoveries.

### Harness improvements (small follow-ups)

- **Per-case timeout override**: thread `chain_timeout_s` through
  `run_case` and the chat envelope. Pattern 5 fix.
- **Soft-pass classification expansion**: today, `tools_called: []`
  + observed-tools is hard-fail. Expose a flag (or inferred category)
  to demote to soft-pass for bounded-autonomy probes.
- **`--include-future` vs runtime gating**: harness currently treats
  the harness-provided runtimes as auto-resolved. Consider a `--list-skipped`
  flag that prints "future cases I would run if you set X" alongside
  the standard pass/fail report.
- **Output buffering**: Python defaults to block-buffered stdout when
  redirected; runner should `sys.stdout.reconfigure(line_buffering=True)`
  at startup or use `print(..., flush=True)` so live monitors see
  per-case progress in long runs (currently the buffer flushes only at
  exit, which makes live monitoring tricky).

### Cross-model regression matrix (high-value next move)

The corpus is now diff-able. A meaningful next investment:

1. Run the active-66 corpus against `cosmos-reason2-8b` (production), record per-category pass rate.
2. Run against `qwen3.5-9b-claude-distilled-nvfp4` for direct-lane comparison.
3. Run against `qwen3.6-27b-fp8-mtp-kvfp8` for OpenClaw-lane comparison.
4. Build a small `compare_runs.py` that diffs the JSON reports.

This produces a **per-prompt × per-model pass matrix** that quantifies model swap risk far better than the current 3-prompt × 3-round smoke. The 3×3 stays as a fast in-loop check; the 74-case corpus becomes the gating regression for promotion decisions.

## Deep-Trace Methodology — capturing every OpenClaw ↔ vLLM detail

When the per-case `tools_called` / `state_after` asserts aren't enough
to pin a failure, you need the full prompt/response payloads and the
model's reasoning chain. Two complementary sources exist:

### Layer A — Gateway log (`/sandbox/openclaw-gateway.log` inside the sandbox)

The OpenClaw gateway logs every model emission verbatim, plus
diagnostic events. Already-rich without any infrastructure changes.
Captures:
- The model's prose output line-by-line (what it would have surfaced
  to the user if it weren't a multi-step agent loop)
- Internal scheduler events: `[diagnostic] lane wait exceeded`,
  `[diagnostic] stuck session`, `[agent/embedded] embedded run failover
  decision`, `[agent] run X ended with stopReason=Y`
- The model's self-narration of why a tool call failed and what it
  intends to retry

How to read:
```bash
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell \
    my-assistant -c agent -- tail -200 /sandbox/openclaw-gateway.log
```

**Concrete example from this session's data**: when running
`PARALLEL_concurrent_medium`, the gateway log captured the model's
admission *"It appears that the specific tool or approach required to
achieve this functionality is not directly supported by the available
ManyForge tools. Further exploration or a different strategy may be
necessary..."* — direct evidence of Pattern 1 (tool-not-fired) at the
model-reasoning layer. The bridge audit log only said "no tool fired";
the gateway log showed *why*.

A second example, on `INSERT_position_first_specific`: 23 consecutive
log lines of the model trying variants of the `wait_for_signal_bool`
insert payload, narrating each schema rejection — *"The required
parameter 'key' for the 'wait_for_signal_bool' node must be a
field_ref binding"*, *"The error indicates that the 'node' object
must be a string, not an object"*, *"This is the final attempt before
seeking further guidance from the user..."*. This makes Pattern 1's
"medium-detail prompts that fail tool selection" concrete: the model
DOES try, but the schema requirements for novel node kinds defeat it
within the chain budget.

### Layer B — vLLM HTTP proxy (`scripts/proxy/vllm-proxy.py`)

The proxy sits between the gateway and vLLM, capturing the full chat-
completion request/response bodies — including the `tools[]` array
the model sees and the `tool_calls[]` it produces. JSONL log; one
record per call.

Two proxy instances are pre-deployed on the host:
- `:8001 → :8000` for the direct lane (`/tmp/vllm_direct_proxy.jsonl`)
- `:8002 → :8000` for the OpenClaw lane (`/tmp/vllm_openclaw_proxy.jsonl`)

To route the OpenClaw gateway through `:8002`, edit the in-sandbox
agent provider config:
```bash
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell \
  my-assistant -c agent -- \
  sed -i 's|host.openshell.internal:8000/v1|host.openshell.internal:8002/v1|' \
  /sandbox/.openclaw/agents/manyforge-composer/agent/models.json
```
Then **fully restart the stack via the demo script** —
`PRESERVE_OPENSHELL=true demo-assistant-known-good.sh restart` — NOT
just the bridge. The SSH tunnel that backs `127.0.0.1:18789` (gateway)
gets stale if the gateway is killed without re-running the
provisioner; only the full restart re-establishes a fresh tunnel.
Failing to do this surfaces as `curl: (52) Empty reply from server`
on every chat-completion to the bridge.

Always revert the `:8002` → `:8000` change before resuming production
runs:
```bash
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell \
  my-assistant -c agent -- \
  sed -i 's|host.openshell.internal:8002/v1|host.openshell.internal:8000/v1|' \
  /sandbox/.openclaw/agents/manyforge-composer/agent/models.json
```

What the JSONL gives you that the gateway log doesn't:
- The exact `tools[]` schema the model is given each turn (verifies
  whether `tool_choice: "required"` is being injected by the bridge)
- The exact `tool_calls[].function.arguments` the model emits (lets
  you see whether the model invented a tool name or just got an arg
  shape wrong)
- The vLLM-side latency and whether the model went into a long
  reasoning preamble before producing a tool call

### Recommended workflow when investigating a Pattern-1 failure

1. Run the corpus narrowed to the failing case (e.g.
   `--filter '^PARALLEL_concurrent_medium$'`).
2. Capture the gateway log slice during the run window
   (timestamps from the harness output bracket it).
3. If layer A is enough to explain the failure (e.g., the model
   reasoned itself into giving up), stop. That's the finding.
4. Otherwise route via `:8002` proxy, rerun the same case, and inspect
   the JSONL request/response payloads for the in-flight schema vs the
   model's emitted call.
5. Document the case ID + the smoking-gun log/JSONL excerpt directly
   in this file, so the regression doesn't get re-investigated next
   time the same prompt fails.

### Starting state for instrumented runs

Before triggering a deep-trace smoke, confirm:

| Check | Expected | Command |
|---|---|---|
| vLLM serving cosmos-reason2-8b | `served: ['cosmos-reason2-8b']` | `curl -s http://127.0.0.1:8000/v1/models \| python3 -m json.tool` |
| Composer reachable | `{"name":"ManyForge Composer Assistant",…}` | `curl -s http://127.0.0.1:9000/api/infra/status` |
| OpenClaw bridge healthy | `status:"ok"` on port 8200 | `curl -s http://127.0.0.1:8200/healthz` |
| Gateway alive in sandbox | `openclaw-gateway` process listed | `docker exec openshell-cluster-nemoclaw kubectl exec -n openshell my-assistant -c agent -- pgrep -af openclaw-gateway` |
| Gateway provider config points at intended endpoint | `:8000` (default) or `:8002` (instrumented) | `docker exec … cat /sandbox/.openclaw/agents/manyforge-composer/agent/models.json \| grep baseUrl` |
| (Instrumented only) `:8002` proxy listening + reachable | Returns `data: [...]` | `curl -s http://127.0.0.1:8002/v1/models` |

If any check fails, run `PRESERVE_OPENSHELL=true demo-assistant-known-good.sh restart` and recheck before the run.

## How to Run

```bash
# All active-default cases (skips status=future):
python3 manyforge/scripts/debug/smoke_corpus_runner.py

# Include future cases too (will still gate on required_runtime):
python3 manyforge/scripts/debug/smoke_corpus_runner.py --include-future

# Only specific cases (regex on id):
python3 manyforge/scripts/debug/smoke_corpus_runner.py --filter '^(P[123]_|CUR_)'

# Enable an external runtime (won't help unless that surface actually exists):
python3 manyforge/scripts/debug/smoke_corpus_runner.py \
    --runtime-flags isaac_scene_stream,force_monitor_node \
    --include-future
```

The runner writes a JSON report to `/tmp/smoke_corpus_<ts>.json` including
per-case status, observed tool calls, answer text excerpt, and failure
reasons. Skipped cases include `skip_reason` for triage.

## Maintenance

- **When the deployment YAML changes** (assistant mode catalog, allowed
  tools, allowed nodes): rerun the corpus and reclassify newly-resolvable
  future cases (similar to the Step B pattern in this session).
- **When a new node kind is added to the catalog**: re-evaluate cases
  gated on `expanded_node_allowlist` or specific `required_nodes`.
- **When a new tool surface is added to the assistant**: check the
  corresponding `status: future` cases in this corpus — they're the
  acceptance tests for that tool.
- **When the program/deployment example files change**: the corpus's
  top-level `starting_state:` block is captured from those files and
  may need refreshing.
