# Three-lane smoke: scorer strictness vs functional correctness

**Date:** 2026-06-09 · **Model:** gemma4-12b-it QAT (gemma4-12b-it-gguf), temp=1.0,
single llama.cpp slot · **Corpus:** smoke_corpus.yaml (75 cases, 9 future-skipped
→ 66 scored) · **Heal:** on.

## Headline results

| Lane | First-try | Effective | Fails | Per-case median | Full 66-case run |
|---|---|---|---|---|---|
| Hermes | 78.8% (52/66) | 81.8% (54/66) | 12 | 67.5s | ~80 min |
| OpenClaw | 66.7% (44/66) | 77.3% (51/66) | 15 | 36.8s | ~53 min |
| Direct — original scorer | 50.0% (33/66) | 57.6% (38/66) | 28 | **11.4s** | **~30 min** |
| **Direct — corrected scorer** | **59.1% (39/66)** | **71.2% (47/66)** | 19 | **11.4s** | **~30 min** |

**Speed:** Direct's *typical* case is **~6× faster than Hermes** and **~3× faster
than OpenClaw** (median 11.4s vs 67.5s vs 36.8s). Direct runs the tool loop
**in-process** (bridge → vLLM, no hops); the gateway lanes route every turn
through a sandbox gateway agent (extra network hops + an agent loop + MCP
round-trips). That same in-process rapid-fire is also why Direct alone hits the
proxy loop-stop / heavy-generation upstream errors — it hammers the single vLLM
slot back-to-back while the gateways pace themselves.

**With the corrected scorer the three lanes are functionally comparable**
(Direct 71.2% ≈ OpenClaw 77.3%, approaching Hermes 81.8%), and Direct is by far
the fastest. The corrected-scorer Direct run's 19 residual fails are **all
genuine**: 7 clarification over-acts (which fail on *all three* lanes), 5 wrong
arg value/id, 4 model-loop / heavy-generation upstream errors (Direct-specific),
3 wrong/missing tool.

Reports: `/tmp/{direct,hermes-full-smoke,openclaw}-baseline.json`. Comparison:
`/tmp/three-lane-comparison.txt`. Re-scorer: `/tmp/rescore.py`.

## The Direct number is depressed by scorer strictness, not model capability

Direct exposes the **acceptance surface too literally**. The scene/tree tool
schemas advertise flat aliases (`shape_type`, `dimensions`, `box_dimensions_m`,
`radius`, …) and require only `objectId`. Composer **accepts and normalises**
those flat forms. In the Direct lane the tool schema sits directly in the
OpenAI `tools[]`, so the model rationally emits the **shorter flat form**. The
gateway lanes (OpenClaw/Hermes), driving the tool through an MCP agent, happen
to emit the **nested** `shape.type` / `shape.box_dimensions_m` form.

**The test scorer is stricter than Composer.** Verified from
`/tmp/direct-fixcheck.json`: `scene_draft_add_object` returns **200/completed**
with the correct dimensions and `state_after` **passes** (the object exists with
`shape.box_dims = [1.0,0.02,0.25]`; the model's own answer says "Box (1.0m ×
0.02m × 0.25m)"). The case still FAILS — only because the corpus
`args_contain` golden credits `shape.type` / `shape.box_dims` but **not** the
Composer-accepted top-level `box_dimensions_m`, top-level `shape_type`, or
nested `shape.shape_type`. The same class affects `afterName`: Composer treats
`afterName` and `parentName` as mutually exclusive, but the corpus checks
`parentName: pick_and_place` for an "after X" insert.

## Corrected re-score (semantic-effect-first) — existing logs, no new run

Re-credit a failed case as PASS **only** when its sole hard failures are
`args_contain[...]` **and** the case has a `state_after` expectation that
passed (Composer completed the op and the resulting state is correct).
Cases with `args_contain`-only failures but **no** `state_after` check are
**not** auto-credited — they are flagged NEEDS-ARGS-EVIDENCE (raw model args
were not retained in the report). Applied identically to all three lanes:

| Lane | Original eff. | Corrected (state-proven) | Needs-evidence | Genuine fails |
|---|---|---|---|---|
| Hermes | 81.8% | 81.8% (54/66) | 0 | 12 |
| OpenClaw | 77.3% | 77.3% (51/66) | 0 | 15 |
| **Direct** | 57.6% | **62.1% (41/66)** | **12** | **13** |

**The correction is entirely Direct-lane** (3 cases re-credited, 12 flagged; the
gateway lanes have *zero* args-structure false-negatives because their model
output matches the golden's nested form). If the 12 NEEDS-EVIDENCE cases are
also accepted-alias false-negatives — which captured args strongly suggest for
the scene/insert subset (top-level `box_dimensions_m`, `shape.shape_type`,
`afterName`) — Direct's true effective rate is ~**80%**, on par with the
gateway lanes. Confirming the remaining 12 requires either a targeted re-run or
adding `state_after` checks to those cases.

### Direct case-by-case audit (28 non-pass)

- **State-proven PASS (scorer false-negative): 3** — P2_scene_add_specific,
  BB_add_specific, CUR_wrap_existing_motion_with_retry.
- **Needs-args-evidence (args-only fail, no state_after to confirm): 12** —
  SCENE_add_generic, SCENE_add_medium, TREE_insert_runtime_medium,
  INSERT_position_after_named_medium (the `afterName` case), BB_modify_medium,
  PnP_01/02/08/14/20, CUR_scene_add_static_fixture, CUR_runtime_update_pose.
  Captured args show these complete (200) with Composer-accepted aliases →
  *likely* false-negatives, but not state-proven here.
- **Genuine fails: 13**
  - clarification over-act (model mutates when it should ask) — **6**:
    WRAP_root_generic, TREE_insert_runtime_generic,
    INSERT_position_before_named_generic, PARALLEL_generic, FALLBACK_generic,
    UPDATE_params_generic. **These also fail on Hermes/OpenClaw** — genuine
    model behaviour, not lane-specific.
  - pipeline (proxy loop-stop 409→bridge 502, or upstream timeout) — **4**:
    INSERT_position_first_specific, PARALLEL_concurrent_medium,
    FALLBACK_alternate_medium, REPLACE_subtree_specific. Direct-lane bridge
    rough edge (in-process tool loop hammers the single slot; the bridge maps
    the proxy's loop-stop 409 to a bare 502).
  - wrong/missing tool — **3**: REPLACE_simple_medium, SCENE_update_pose_specific,
    CUR_runtime_remove_then_restore_graspable.

## Recommended scorer-contract changes (the real lever)

1. **Semantic-effect-first.** If the tool completed and `state_after` proves the
   object/node exists with the right normalised fields, do not fail solely
   because the args used a Composer-accepted alias. (Implies: add `state_after`
   coverage to the scene/tree cases that currently lack it, so the effect is
   checkable.)
2. **Accepted-form aliases in `args_contain`.** Credit the Composer-accepted
   scene forms: top-level `shape_type`, `box_dimensions_m`, `dimensions`,
   `radius`, `sphere_radius_m`, `cylinder_*`, and nested `shape.shape_type`.
3. **`afterName`/`beforeName`.** Stop expecting `parentName` for "after/before X"
   inserts (mutually exclusive in Composer); verify final tree placement instead.
4. **Longer term — split accepted vs preferred schema.** The runtime can keep
   accepting aliases, but the *model-facing* schema should be canonical and less
   ambiguous (one way to say "box dims") so outputs are consistent across lanes.

## The `enrich_assistant_tool_descriptor` change (parity cleanup, NOT a score fix)

`manyforge_composer/backend/assistant_tool_schemas.py` was changed so the
provider-request envelope (direct lane) carries the **same curated descriptor
source** as the mode manifest (gateway lanes) — overriding the deployment's
legacy reduced `input_schema` with the canonical `ASSISTANT_TOOL_INPUT_SCHEMAS`
entry and dropping the legacy keys. This is a **single-source-of-truth parity
cleanup**. It is explicitly **NOT** proven to change smoke scores: the affected
cases already complete (200) and pass `state_after` with the reduced schema;
they fail the *scorer*, which this change does not touch. Full-corpus
regression validation of this change is pending (validated on 7 cases only).
The catalog hash is keyed on tool ids, so the change does not perturb it.

## Implementation status (2026-06-09)

The first two contract changes are **implemented in
`scripts/debug/smoke_corpus_runner.py` and validated** (applied to every lane):

- **Accepted-form aliases** — `_ARG_PATH_ALIASES` now credits top-level
  `box_dimensions_m` / `shape_type` / `sphere_radius_m` and nested
  `shape.shape_type`.
- **Semantic-effect-first** — `demote_args_when_state_proven()` demotes residual
  `args_contain` mismatches to *soft* when the case declares a `state_after`
  block that PASSED and no other hard failure exists. Over-acts / wrong-tool /
  failed-state stay hard.

Plus **type-inferred-from-dims** (a box specified by `box_dimensions_m` IS a box
even if `shape.type` is omitted — Composer derives it), **list-of-dict subset
matching** (blackboard `keys` may carry extra accepted fields), the corpus
`afterName` fix for the INSERT case (verify placement, not `parentName`), and a
distinguishing `state_after` on `CUR_scene_add_static_fixture`.

**Full-corpus validation (Direct lane, all fixes, 0 regressions):** 50.0/57.6 →
**59.1/71.2** (effective +13.6 pts). The 19 residual fails are all genuine
(7 over-act / 5 wrong value-id / 4 loop-or-heavy-gen / 3 wrong-tool). Controls
held throughout: over-acts (`WRAP_root_generic`, `UPDATE_params_generic`, …) and
wrong-value cases (`PnP_08` gripper 1.0≠0.66, `BB_modify` id `grip_force_key`)
**stay fail** — the demotion is correctly scoped to accepted-alias-with-proof.

**Failure-attribution fix (reporting only).** The bridge surfaces model-behaviour
exits as structured codes (`upstream_loop_stop` [proxy 409 loop hard-stop],
`stuck_loop`, `max_turns_exceeded`), but the Composer's
`_provider_error_status_code` collapsed every non-cancel/non-timeout provider
error to **502** ("bad gateway"), so genuine model looping read as a transport
fault in the harness. Fixed: those three codes now map to **409**, and the smoke
runner surfaces the structured reason in its failure label instead of a bare
`chat HTTP 502`. Does not change pass/fail (these cases still fail); it stops
mis-attributing model loops as infra faults. (`bridge.py` B1 already maps the
proxy's upstream 409 to a `upstream_loop_stop` envelope rather than 502.)

**Remaining (next, reviewed increment) — state_after coverage is case-by-case:**
- `CUR_scene_add_static_fixture` got a distinguishing `state_after` (specific
  dims). The same can be added to other needs-evidence cases **only where a
  distinguishing feature exists** (e.g. PnP_01/02 specific objects).
- `SCENE_add_generic` ("add a small box") **cannot** be verified this way — the
  initial scene already contains a box (`graspable`), so a `type==box` check
  passes vacuously. Needs an object-count-delta primitive in the scorer.
- `INSERT_position_after_named_medium` — implement `afterName`/`beforeName` →
  **verify final tree placement** (state_after on the tree) instead of
  `parentName`.
- `BB_modify_medium` — the model's key is a *superset* (`id`/`type` correct +
  extra `description`/`key`); needs subset-matching for `args_contain` on
  list-of-dict, or a blackboard `state_after`.
- `TREE_insert_runtime_medium` — **genuine** (model omits `node.id` /
  `node.params.object_id`); stays fail correctly.

## Genuine, lane-attributable findings (independent of scorer strictness)

- **Clarification over-act is a real model weakness on all three lanes** at
  temp=1.0 — the dominant *genuine* failure class.
- **Direct's pipeline rough edges** (409→502 mapping; long in-process loops
  timing out) are worth hardening regardless of scoring.
- **Long-chain context bloat:** the 20-step PnP chain grows to ~118k tokens
  (near the 131k ctx), causing tail timeouts (most visible on OpenClaw today).
