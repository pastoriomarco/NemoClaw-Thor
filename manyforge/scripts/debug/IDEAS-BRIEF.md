# Brief for an Outside LLM — "Help Me Find New Cross-Cutting Fixes for the ManyForge Smoke Corpus"

> Paste this entire document into a fresh chat with a capable model
> (Claude Opus, GPT-4-class, etc.) and ask the question at the bottom.
> The brief is self-contained — the receiving model has no access to
> the codebase. Treat its suggestions as candidates to evaluate against
> the constraints, not directives to apply.
>
> **Refresh date: 2026-05-10, after iter 33.** Replaces an earlier
> version that scoped the question at iter 5 / 51-66.

---

## What is being tested

ManyForge Composer is a robot-program authoring UI with an embedded
"Composer-assistant" that takes natural-language requests ("add a
move-to-pose that approaches above the graspable") and turns them into
mutations to a behavior tree, scene, parameters, or blackboard. The
assistant runs through an OpenClaw gateway → bridge → vLLM stack.
**Active model: `cosmos-reason2-8b` at temperature 0.2, thinking-on
(Cosmos is post-trained on Qwen3-VL and assumes long-CoT).**

The smoke corpus is a 74-case YAML suite (66 active + 8 future) that
dispatches each case as a `/api/assistant/chat` request and asserts on:

- `tools_called` — multiset of tools the assistant called, each with optional `args_contain` and `allow_retries`
- `state_after` — dotted-path asserts against `/api/program` + `/api/program/tree` + `/api/scene/state`
- `forbidden_tools`, `answer_must_contain`, `answer_must_not_contain`

A case is **pass** if all asserts hold on the first turn, **recovered-pass** if the runner's generic `--enable-recovery-turn` nudge let it through on turn 2, **soft-pass** if only the answer-text assert held, **fail** otherwise.

Cases are tagged `detail_level: specific | medium | generic`. Specific cases give the model concrete coordinates / names; generic cases are intentionally vague to test bounded autonomy.

A subset of cases form same-conversation **chains** (`chain_id` + `chain_step`). The flagship chain is `pnp_build` (PnP_01 → PnP_20), building a pick-and-place program from an empty fixture. Chain-session ON is the production default.

## Architecture cheat-sheet

```
user prompt → Composer UI → /api/assistant/chat → openclaw_assistant_bridge →
gateway (in sandbox, SSH-namespace) → vLLM proxy (host:8000, mutator) →
vLLM (cosmos-reason2-8b, host:8050) → tool calls → MCP wrapper validator
(in bridge) → /api/assistant/bridge/tools/<name> on Composer → mutation
applied → state snapshot returned → next turn.
```

Key bridge / runtime behaviors:

- **MCP wrapper validator** rejects malformed tool calls with HTTP 400 + a structured `detail` payload (e.g. `validParentNames: [...]`, `allowedNodeKinds: [...]`).
- **STUCK_LOOP_HINT**: after **3 identical-arg failures** of the same tool, the bridge injects a synthetic message inlining the structured recovery fields in plain prose. Below 3 retries, the model only sees the JSON envelope.
- **Bridge-fired `/compact`** (iter 32 production recipe): every N=2 requests on the same session-key, the bridge slips a `/compact` slash-command into the gateway before forwarding the user prompt. Spaces compactions deliberately so OpenClaw's `already_compacted_recently` cooldown never trips. Without this the chain-session-ON PnP chain cascades from PnP_06+.
- **`request_clarification` tool + `[NEEDS-CLARIFY]` marker + bridge auto-retry** — **TRIED IN ITER 33-34, REVERTED.** Was an attempt to give the model a structured way to ask the operator a question on ambiguous prompts. Iter 33 (soft path: tool + system-prompt rule + corpus rubric) showed the model never invoked the tool on the 7 Pattern-A cases that needed it. Iter 34 (hard path: `tool_choice=required` proxy-injection to force tool selection) was halted after 3 cases / 14 min because force-tool every turn disallowed natural loop termination — each case ran 5-7× normal duration. All plumbing fully reverted; direction 3 is closed out. See "Iter 33 + 34 negative result" in SMOKE-CORPUS.md for full analysis.
- **`--enable-recovery-turn`** (default since iter 33): when initial asserts fail and the chat returned 200, the runner sends ONE generic "previous turn failed — re-read structured recovery fields and retry" message in the same conversation, then re-asserts on the combined log. Salvaged 10 cases in iter 33 (mostly PnP chain steps with first-turn malformed args).

## Files at a glance

Two source trees are live, checked out side-by-side. Composer reads from `dev_ws`; the smoke runner and OpenClaw bridge live in `NemoClaw-Thor`.

| File | Role |
|---|---|
| `NemoClaw-Thor/manyforge/scripts/debug/smoke_corpus.yaml` | The 74-case test corpus. Each case has `id`, `prompt`, `expected.{tools_called,state_after,forbidden_tools,answer_must_contain}`. |
| `NemoClaw-Thor/manyforge/scripts/debug/smoke_corpus_runner.py` | The runner — loads the corpus, POSTs each case to `/api/assistant/chat`, parses tool-call audit log, runs asserts, writes JSON report. Owns `--enable-recovery-turn` logic. Stdout is line-buffered at startup so per-case verdicts stream realtime. |
| `NemoClaw-Thor/manyforge/openclaw_assistant_bridge/service.py` | Bridge HTTP service — handles `request_started`/`request_complete`, fires `/compact` every N requests. |
| `NemoClaw-Thor/manyforge/openclaw_assistant_bridge/adapter.py` | Bridge ↔ gateway shell-out / curl builder. Owns the `tools[]` array sent to the gateway and the chat-completions response parser. |
| `NemoClaw-Thor/manyforge/scripts/proxy/vllm-proxy.py` | vLLM mutator proxy — injects `max_tokens`, applies `thinking_token_budget`, logs every chat-completion. Also supports per-turn `tool_choice` mutation via `OPENCLAW_PROXY_FORCE_TOOL_CHOICE` (tried in iter 34, found unviable in `required`-every-turn mode). The single source of truth for per-turn vLLM-side tunables. |
| `NemoClaw-Thor/manyforge/setup-manyforge-assistant.sh` | Sandbox provisioner — installs the MCP wrapper inside the OpenClaw sandbox, configures the gateway. Re-run when Composer rotates the tool catalog. |
| `NemoClaw-Thor/serving/launch.sh` | vLLM model launcher — Cosmos-Reason2-8B with `enable_thinking=true` default. |
| `dev_ws/src/manyforge/manyforge_composer/backend/assistant_tool_schemas.py` | JSON-schemas for every tool the model can call (`tree_draft_insert_node`, `tree_draft_wrap_node`, etc.). Edit here to add a tool or change its arg shape. |
| `dev_ws/src/manyforge/manyforge_composer/backend/routes_assistant.py` | Tool dispatcher — per-tool `_apply_*` handlers. Edit here to change tool behavior or add a handler. |
| `dev_ws/src/manyforge/manyforge_behavior/resources/node_catalog.yaml` | Per-node-kind descriptions surfaced to the model in the prompt's `nodeCatalog`. Edit here to reframe a kind (Pattern C "runtime collision-object" rewrites). |
| `dev_ws/src/manyforge/examples/assistant_modes_scene_authoring.deployment.yaml` | Deployment YAML — scopes the assistant's tool allowlist + node allowlist for the `composer-assistant` mode. New tools must be added here. |
| `dev_ws/src/manyforge/examples/pick_and_place_ur10e_robotiq.program.yaml` | The populated demo program (12-step pick-and-place, 2 scene objects, 1 param, 1 blackboard key). Used by most non-PnP smoke cases. |
| `dev_ws/src/manyforge/examples/empty_pick_and_place_ur10e_robotiq.program.yaml` | Empty-fixture program used when a case sets `precondition.fresh_program: true` — the PnP_01 → PnP_20 build chain starts here. |
| `dev_ws/src/manyforge/agent-skills/manyforge-composer/workspace-AGENTS.md` | The shared workspace system prompt (Role, Vocabulary, Output protocol, Tool surface, Guardrails). Production surface — see Constraint 3 before proposing edits. |
| `NemoClaw-Thor/manyforge/docs/SMOKE-CORPUS.md` | Per-iter history (1 → 33) with failure-pattern analysis. |
| `NemoClaw-Thor/manyforge/docs/SMOKE-ITER-RUNBOOK.md` | Operational cold-start sequence + per-change restart matrix. |
| `NemoClaw-Thor/manyforge/docs/COMPOSER-ASSISTANT-ARCHITECTURE.md` | Runtime topology + repo layout + iter-32 production-recipe map. |

The smoke runner is invoked with:

```bash
cd NemoClaw-Thor/manyforge
nohup python3 scripts/debug/smoke_corpus_runner.py \
  --corpus scripts/debug/smoke_corpus.yaml \
  --enable-recovery-turn \
  --report /tmp/smoke_corpus_iterN.json \
  > /tmp/iterN_runner.log 2>&1 &
```

A full 66-case run takes ~75-80 min wallclock under the iter-32 production recipe.

## Hard constraints — non-negotiable

1. **No commits during the iteration cycle.** Stage and re-evaluate each round.
2. **No per-prompt hints to the model.** Prompts must be user-style natural language. No "use parentName=pick_and_place" baked into the prompt.
3. **No tightening of the model's RULES system message** to fix smoke-corpus failures. RULES is shared production surface.
4. **Fixes must be cross-cutting.** Corpus structure / harness logic / deployment YAML / timeouts / new tools / new bridge mechanisms are fair game. Per-case workarounds are not.
5. **Specific > generic.** Specific cases are the priority. Generic cases that genuinely warrant clarification can stay yellow.
6. **Corpus must remain extensible.** No one-off helpers tied to a single case.
7. **Schema examples in tool descriptions are excluded.** Past attempt: model copy-pasted example `objectId` verbatim → 116-call upsert loop.
8. **Bridge / Composer code changes are in scope** (was previously out of scope). Iter 27-32 landed schema refactors and bridge-side `/compact`. The iter-33 `request_clarification` tool and iter-34 `tool_choice=required` proxy injection were also code changes but they're since reverted — the principle remains that code change is on the table when the corpus surface is exhausted, but the bar is "predicted lift large enough to justify the post-revert hygiene if it fails."
9. **Timeouts can be tuned**, but an upstream 502 fires around 5 min of model loop — past that the request is killed regardless.
10. **Always restart the demo between iterations** to clear zombie sessions. See `SMOKE-ITER-RUNBOOK.md` for the exact restart sequence.

## Where we are now — iter-33 snapshot

**Production recipe: iter 32 = 51/66 (77.3%) effective, 71.2% first-try, chain-session ON.** First chain-on setup that matches the chain-off rate; runs in ~75 min wallclock. See [`SMOKE-CORPUS.md`](../../docs/SMOKE-CORPUS.md) for the full iter-1 → iter-33 history and per-iter changelog.

**Most recent runs: iter 33 = 48/66 (72.7%, regressed -3); iter 34 halted at 3/66.**

*Iter 33* tested the `request_clarification` tool direction (see Architecture cheat-sheet) — registered a new tool, instructed the model via system-prompt rule to emit `[NEEDS-CLARIFY]` marker on ambiguous prompts, added bridge auto-retry on the marker, tightened the corpus rubric on 7 Pattern-A cases with `expected.clarified=true`. **Model never invoked the tool** on any of the 7 targeted cases; 5 hard-failed because the rubric tightened without the model adopting the new behavior. Reverting just the rubric tightening would put iter 33 at ~53/66 (80.3%) — the +2 over iter 32 attributable to `--enable-recovery-turn` salvaging 10 cases.

*Iter 34* tested the harder variant: with the tool installed, proxy-inject `tool_choice=required` on every chat-completion, forcing the model to pick one of the 26 tools. Hypothesis: with thinking-on (model can reason about ambiguity) and `request_clarification` available, forcing tool selection would push it toward the clarification tool. **Halted after 3 cases / 14 min: P1 fail (275 s, 4.6× normal), P2 soft-pass (167 s), P3 fail (425 s).** Root cause: `tool_choice=required` every turn disallows the model's natural "I'm done, emit a final message" signal. After the model successfully completed the requested action on turn 1, it was forced to fire more tools, picking increasingly irrelevant calls until the case timed out. The mode that *might* test the hypothesis cleanly is `tool_choice=required-first` (force turn 1 only) — that's untested.

**Both iter 33 and iter 34 are fully reverted.** All request_clarification plumbing — tool schema, handler, deployment-YAML allowlist entry, system-prompt rule, bridge marker detection + auto-retry, corpus rubric changes — is removed from the production stack. Direction 3 is closed out for the cosmos-reason2-8b + corpus/harness surface alone.

**Key durable wins from iter 27-33** (all retained):

- `tree_draft_insert_node` schema: top-level `nodeName`, `afterName`/`beforeName` sibling shortcuts, trimmed description (1639 chars). Closed 6 cases.
- `tree_draft_swap_node` → `tree_draft_change_node_kind` rename + bridge intent-inference heuristic for "swap the order" → `move_node`. Closed 1 case.
- 6 runtime collision-object catalog descriptions rewritten ("Behavior-tree leaf — runtime collision-object operation"). Closed ~3 cases.
- Bridge-fired `/compact` every 2 prompts. Enables chain-session-ON without cascade. (Architecture-level win, not measurable per-case directly.)
- `--enable-recovery-turn`. +10 cases salvaged in iter 33.
- Runner stdout line-buffering at startup. Operational: per-case verdicts stream realtime so a bad iter can be halted early.

## Remaining 18 failures (iter 33) — by root cause

### Group A — Pattern A residuals: bounded-autonomy gap on intentionally-ambiguous prompts (5 cases)

`PARALLEL_generic`, `FALLBACK_generic`, `SCENE_remove_generic`, `CLARIFY_scene_remove_pronoun`, `CLARIFY_tree_wrap_pronoun`.

**Symptom**: prompt is intentionally vague ("add a parallel", "remove it from the scene", "wrap that step in a retry"). Model fires 1-8 mutations when the expected behavior is to ask "which node?" / "which object?". Persistent across all iters since iter 5; **iter 33 confirmed that giving the model a structured clarification tool + system-prompt rule doesn't flip the behavior** — the model simply won't invoke the tool.

**What was tried**:
- iter 33 direction 3: `request_clarification` tool + `[NEEDS-CLARIFY]` marker + bridge auto-retry. **Failed: model never invoked the tool on any of the 7 targeted cases.** Side effect: 5 of the 7 became hard fails because the rubric was tightened on `expected.clarified=true`.
- Pattern-4 fixes from earlier iters (system-prompt rule against pronouns without antecedents) — partially mitigated MOVE_generic and CLARIFY_motion_generic but the same pattern persists on the remaining 5.

**Open question**: is this a model-capability ceiling (cosmos-reason2-8b is over-eager by training) or is there a corpus-rubric / harness mechanism that can convert "model fires action when it should ask" into a measurable signal that's both fair to the model and useful in production?

### Group B — `tree_draft_insert_node` multi-arg specificity (5 cases)

`P3_tree_insert_runtime_obj_specific`, `TREE_insert_runtime_generic`, `INSERT_position_first_specific`, `FALLBACK_retry_specific`, `FALLBACK_alternate_medium`, `PARALLEL_concurrent_medium`.

**Symptom**: model emits `tree_draft_insert_node` with missing `parentName`, wrong `targetName`, or wraps the JSON in a `_raw` envelope (parser failure on output). Bridge returns 400 with structured hints; model paraphrases the error in text instead of retrying. The model usually mutates args slightly between retries so the bridge's 3-identical-failure dedupe doesn't fire.

**What was tried**:
- iter 27-28: trimmed insert_node description from 3700 → 1639 chars + added afterName/beforeName shortcuts. Closed ~6 cases but these 5 persisted.
- iter 32 production recipe ships with the trimmed schema.

### Group C — Tool-mismatch / wrong-family (3 cases)

`REPLACE_simple_medium`, `REPLACE_subtree_specific`, `CUR_runtime_remove_then_restore_graspable`.

**Symptom**: model picks the wrong tool family (e.g., `tree_draft_insert_node` for a replace request, or only emits 1 of 3 required tools in a multi-step ask).

### Group D — Variance fails (4 cases)

`MOVE_reorder_medium`, `PnP_13_detach`, `PnP_14_upsert`, `PnP_18_repeat_root` (395 s — slowest case, ran out the 5-minute upstream-502 budget).

**Symptom**: pass intermittently on iter-28/30 runs, fail on iter-33. Borderline behavior; sometimes the model gets it right, sometimes the tool args are subtly wrong or the chain-step compaction loses the relevant context.

### Recovered-pass (10 cases, +10 vs iter 32 baseline)

`UPDATE_params_specific`, `MOVE_generic`, `CLARIFY_motion_generic`, `PnP_09_attach`, `PnP_10_lift`, `PnP_11_transport`, `PnP_12_place_descend`, `PnP_15_open_gripper`, `PnP_16_retract`, `PnP_17_home`. All recovered via the generic `--enable-recovery-turn` nudge.

## Existing inventory of fixes you should NOT re-suggest (already considered)

- **Tighten RULES system message** — forbidden (constraint 3).
- **Per-prompt hints** — forbidden (constraint 2).
- **Schema examples in tool descriptions** — actively harmful (constraint 7).
- **Bigger model** — out of scope; corpus measures cosmos-reason2-8b on purpose. (Future direction: 9B Claude-distilled is queued for a comparison run; suggestions about it are welcome but call it out as model-swap.)
- **Lower STUCK_LOOP_HINT trigger from 3 → 1** — interesting but the model mutates args slightly between retries so the dedupe doesn't fire on Group B cases anyway. May help Group A if combined with a more aggressive nudge.
- **Force `tool_choice: "required"` always** — tried in iter 17-18; over-forces tools on no-tool cases, regressed PARALLEL_generic and FALLBACK_generic from soft-pass to hard-fail. Already disabled.
- **`request_clarification` tool variants** — tested in iter 33 (tool + system prompt + bridge auto-retry + corpus rubric tightening). Model never invoked it. Re-engineering the same idea with different prompt wording is unlikely to flip the behavior without deeper changes (fine-tune, model swap, or harness-level intervention).
- **Add a `tree_inspect`-style turn before every INSERT case** — would help Group B but feels like per-case workarounds dressed up; may violate constraint 6 (extensibility) if applied widely.
- **OpenClaw built-in auto-compaction** (`agents.defaults.contextTokens`) — tested in iter 31, hits an `already_compacted_recently` cooldown that turns into a hard wall. Bridge-fired `/compact` is the working replacement.
- **Drop `max_tokens` injection** — tried in early iter-17 variants; vLLM with thinking-on goes unbounded for many minutes. Cap must stay (2048 in production).

## Unexplored or partially-explored directions worth brainstorming

- **vLLM XGrammar guided JSON** (paper Jan 2025) — would collapse Group B's malformed-args by masking invalid args at decode time. Side-effect ~1.6× decode speedup. Deployment-YAML change; production-affecting. Not tested yet.
- **PALADIN-style synthetic-history exemplars** (arXiv 2509.25238) — could rescue Group B (3-4 of 5) by injecting a successful 4xx-recovery trace into history. Risky under constraint 7 (schema-example copy-paste mode failure observed in v8.1 probe).
- **Bridge-side proactive marker injection on detected ambiguity** — could the bridge pre-flight short bare-verb prompts ("add a parallel", "wrap that step") and inject a synthetic "ask me which" pre-prompt before forwarding to the model? Would bypass the model's tool-bias without changing the model. Constraint 4-compatible (cross-cutting), but raises a UX-fairness question for production users who do mean "add a parallel right here as a sibling of the current cursor".
- **Rubric refinement that punishes "did something plausible" satisfying asserts when the case wanted clarification.** Currently `forbidden_tools` applies first-turn only; followup_expected asserts are independent. For Pattern A cases that have `clarification_followup`, an action on turn 1 that happens to match followup_expected on turn 2 currently counts as recovered-pass. Should that be a hard fail?
- **Two-stage prompt with explicit "is this ambiguous?" precheck.** Pre-flight pass: model sees the prompt alone (no tools, just decide "ambiguous yes/no"). Main pass: if ambiguous, ask; otherwise act. Adds ~1 turn per case but might unlock Pattern A cleanly without retraining.
- **Fine-tune a small clarification-bias adapter on cosmos-reason2-8b.** Tens-of-examples LoRA on "ambiguous prompt → ask, not act". Production-affecting; expensive to validate but the only direction that directly addresses the model's training bias.

## What we want from you

We're at 51/66 (77.3%) under the iter-32 production recipe. The model's action-bias on intentionally-ambiguous prompts (Group A) appears to be a training-distribution ceiling we can't shift from the corpus / harness / system-prompt surface alone. Group B (insert_node multi-arg) is the cleanest remaining infra target.

**Question**: given the constraints, the iteration history (especially iter 33's negative result on the `request_clarification` direction), and the remaining failure groups — what *cross-cutting* ideas haven't we considered, or are worth re-exploring with a fresh framing, that could lift:

1. **Group B** (5 cases, malformed insert_node args) — most promising target; harness/deployment fix space
2. **Group A** (5 cases, model won't ask) — hardest; requires either model swap, fine-tune, harness-level "decide if ambiguous before acting" precheck, or rubric reframe
3. **Group D** (4 cases, variance) — quick wins possible via per-case timeout / chain-step tuning

Be concrete: name the change (corpus structure / harness logic / deployment YAML / fixture / timeout policy / bridge or composer code change), name the cases it would target, predict the expected pass-rate delta, and call out the risk of regression on the other ~48 currently-passing cases.

Brainstorm wide first, then prioritize. We will evaluate each suggestion against the constraints; ideas that violate a constraint are still useful to hear *if* the violation is small and worth discussing — just call it out explicitly. **Especially valuable**: ideas that take a different angle on Group A than direction 3 (the tool-based clarification approach) took. We've established that giving the model a clarification tool doesn't get it to ask; what would?
