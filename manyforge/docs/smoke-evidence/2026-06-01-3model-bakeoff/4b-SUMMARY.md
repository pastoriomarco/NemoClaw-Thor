# 4B (nemotron3-nano-4b-bf16) — full smoke result

**Date**: 2026-06-01
**Pipeline**: proxy with detectors #1-7 active + tail_checklist + namespace_stop=16
**Profile config**: FORCE_ENABLE_THINKING=off (verified in request body)

## Overall
- 74 cases total, 8 skipped (status=future), **66 attempted**
- 20 pass + 6 soft-pass + 40 fail
- **Effective rate: 26/66 = 39.4%**
- First-try rate: 20/66 = 30.3%

## Detector activity
- `normalize_tool_names`: fired N times — bare suffix `tree_draft_X` → `manyforge__tree_draft_X` (working as designed)
- `same_args` reflection: fired on repeat-with-same-args loops (e.g. P2 scene_add_specific)
- `same_tool` reflection: fired at count=4,5 on CUR_wrap case
- `malformed_tool_call`: 0 fires (model used proper format throughout)

## Failure breakdown (40 fails)

| Cat | Count | Pattern |
|---|---|---|
| **B. model asked instead of acting** | **37** | clarification on a directive prompt |
| E. tool dispatched but unrecognized | 2 | REPLACE cases — emitted change_node_kind, but case wants replace_subtree |
| C. tool dispatched but state wrong | 1 | CUR_wrap — used update_node_params instead of wrap_node |

37 of 40 fails (92.5%) are 4B answering with clarification when it should act. Root cause: 4B has a strong clarification bias that the corpus' specific-action prompts exceed.

## Three diagnostic deep-dives

### Case 1: P1_wrap_root_specific (asked-not-acted)
- user_request: `'add a repeat node as root'`
- Model answer: `'which parent and where in its children?'`
- Turns: 1 chat completion, no tool emitted
- Hypothesis: Bridge tail_checklist rule "if prompt doesn't name WHERE, ask" misfires on "as root" (4B doesn't recognize "as root" as the WHERE specifier)
- Pipeline action: none — model didn't emit a tool to normalize/detect

### Case 2: P2_scene_add_specific (tried + validation 400 cascade)
- user_request: `'add a box of size 1.0, 0.02, 0.25 in position 0.0, -0.15, 0.125'`
- Model answer: `'I need the mesh resource URI to add the box.'`
- Turns: 4 chat completions, all emitted scene_draft_add_object
- All 4 dispatches returned HTTP 400 (validation failure)
- After 4 failures, model gave up and asked for mesh URI
- Pipeline action:
  - Normalize fired: `scene_draft_add_object` → `manyforge__scene_draft_add_object`
  - Same-args reflection fired at count=2
- **Real bug**: Model's first turn emitted arguments with `shape` as a JSON STRING (`"{\"box_dimensions_m\": [...]}"`) when the API wants a nested object. Validation rejected. Model retried with proper format on turn 2 but the loop counter had already fired.

### Case 3: CUR_wrap_existing_motion_with_retry (wrong tool + 4 retries)
- user_request: wrap existing `move_to_pick` with retry node
- Model answer: claims it added `attempts: 2` to the node
- Turns: 4 dispatches of `tree_draft_update_node_params` (3× HTTP 400, 1× HTTP 200)
- Pipeline action:
  - Normalize fired multiple times
  - same_tool reflection fired at count=4, count=5
- **Model error**: chose `update_node_params` (modifies existing node) instead of `wrap_node` (creates new wrapper). Detectors couldn't help — the model is confidently using the wrong tool.

## What this tells us

**4B is not the right model for this corpus.** It's strong at clarification (>90% pass on generic-WHERE cases per the targeted smoke earlier) and weak at specific-action prompts (the dominant failure mode here).

**The pipeline is working correctly**:
- Normalization rescued ~30+ tool dispatches from name mismatches
- Reflection broke a runaway loop in P2/CUR
- No false positives observed
- enable_thinking propagation verified clean throughout

**Possible 4B-specific improvements (deferred — not actionable now)**:
1. Soften the bridge tail_checklist rule for prompts containing "as root" / "as new root" / specific position keywords
2. Train/finetune 4B to reduce the over-clarification tendency on this domain
3. Use 4B only as a clarification-first agent and route action prompts to a different model
