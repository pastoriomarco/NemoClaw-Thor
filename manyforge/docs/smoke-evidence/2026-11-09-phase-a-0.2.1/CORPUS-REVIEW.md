# Smoke corpus review for ManyForge 0.2.1 (Phase A)

Date: 2026-11-09.
- Scope: the assistant smoke corpus (`manyforge/scripts/debug/smoke_corpus.yaml`, 66 active cases, baseline iter-32: 51/66 = 77.3% on 2026-06-09) against ManyForge 0.2.1 (`src/manyforge` dev, checkpoint 478f74ee, plus the fixes below).
- Result: all 66 cases kept, ported to `smoke_corpus_v021.yaml`; none proposed for dropping.

## 1. What changed since the baseline

**Transport.**
- `/api/assistant/chat` is gone (0.2.0). The AI tab path is `POST /api/assistant/requests` with status polling, then the bridge (:8100), then the model named by the Deployment's `ai_model` component.
- The completed response carries `toolCalls[] {toolId, status, effect, arguments, draftMutated}`. The old runner's Composer docker-log scraping is no longer needed.
- Reset is `POST /api/program/discard` (Draft back to Applied). Fixtures use `POST /api/program/draft/mutations`, the same Draft engine the assistant's tools use. `/api/program/load` with deployment paths no longer exists.

**Program format.**
- `manyforge.program.v3` and node catalog v3 3.0.0. Tree nodes are `{node_id, instance_id, parameters, children}` in a `forest`.
- Scene resources are `{scene_resource_id, shape{type, box_dimensions_m|sphere_radius_m|…}, pose{frame_ref, position_m, orientation{xyzw|rpy_deg}}, resource_roles}`.
- Parameters are `{parameter_id, type: float64|…, default}`. Blackboard is `blackboard.keys[{key, type, access, persistence}]`.

**Tool arguments.**
- The tool set is unchanged: the same 24 tools.
- Renamed arguments:
  - `parentName` → `parent_instance_id`; `afterName` → `after_instance_id`; `position` → `index` (omitted means append); `nodeName` → `instance_id`; `newParentName` → `new_parent_instance_id`; `targetName` → `target_instance_id`.
  - `wrapper.id/params` → `wrapper.node_id/parameters`; `subtree_root.kind` → `subtree.node_id`.
  - `objectId` → `scene_resource.scene_resource_id`; `objectIds` → `scene_resource_ids`; `names` → `parameter_ids`; blackboard `ids` → `keys`.
- `@root` is still accepted for target and parent ids.

**Node ids.**
- `move_manipulator_action` → `motion.move_to_pose` / `motion.move_to_joint_target` / `motion.follow_cartesian_path`; `command_gripper` → `gripper.command_position`.
- `upsert_collision_object` / `remove_collision_object` / `update_collision_object_pose` → `scene.upsert_object` / `scene.remove_object` / `scene.update_object_pose`.
- `attach_object_to_link` / `detach_object_from_link` → `scene.attach_carried_object` / `scene.detach_carried_object`; `wait_for_signal_bool` → `logic.wait_for_value`.
- The same 52 catalog entries; the assistant may author 23 of them (`maintained_assistant_profiles.v1.json`).

**References (OD-28).**
- Roles are gone. Targets are `{component_id, command_target_id}`; frames are `{component_id, frame_id}`.
- Gripper values are typed (`position {kind, value_rad|value_m|name}`, `max_effort {kind, value_n|value_nm}`).
- There is no `universe` frame; scene poses use `frame_ref.kind: cell_frame` or `component_frame`.

**Bridge limits.** 16 model turns, 180 s wall time, `max_output_tokens` 8192 from the profile.

**Existing tests.**
- All bridge and Composer assistant tests are mocked. 41 bridge tests pass on Thor, including the new one.
- Before tonight no test exercised a real model. The only live checks are the two `scripts/smoke-assistant-*.py` smokes.

## 2. Assistant-path defects found and fixed

1. **Every request failed with `assistant_projection_stale`.**
   - Cause: the Headless assistant projection wraps the Manager section's live observation fields (`receiver_age_ns`, generations, the Manager section digest) in `source`, and those change on every Manager tick (`manyforge_headless/src/headless_api_core.cpp:3851-3884`). The bridge compared the whole forwarded projection, `source` included, with its own later read.
   - Fix: the bridge compares the projection content without `source` (`manyforge_assistant_bridge/bridge.py`, `_projection_content`). That content is the part the semantic digest covers, as the existing stale-projection test already defines it.
   - Regression test: `test_bridge_accepts_a_projection_whose_source_observation_moved_on`.
2. **The scene tools' `frame_ref` still described the role shape** (`component_frame {component_role, model_role, frame_role}`).
   - Program v3 accepts only `component_frame {frame: {component_id, frame_id}}` (`manyforge_behavior/program.py:2518-2520`). A model following the schema could author only `cell_frame` poses.
   - Fix: the generator takes the catalog's `frame_reference` type (`manyforge_deployment/tools/generate_assistant_instrument_registry.py`, `_initial_scene_pose_schema`), and `assistant_instrument_registry_source.v1.json` was regenerated. Only the three scene tools changed.

## 3. Node definitions the assistant sees

The check was run live against the 0.2.1 workbench on Thor (`node_check.py`, `node_check2.py`).
- For each of the 23 allowed nodes, a complete instance was built (taken from the Program where one exists, otherwise from the schema), validated against the assistant's `tree_draft_insert_node` schema, and inserted through Composer's Draft mutation path, then discarded.
- **23/23 are schema-valid and accepted by Composer.**
- The catalog entries for these nodes contain no stale role, old node-id or `universe` wording.
- The assistant schema enforces complete nodes: a required parameter must come from `parameters` or `value_sources`. Composer's Draft accepts incomplete nodes. This is by design, not a defect.

## 4. Case-by-case review

Every active case still fits the 0.2.1 architecture once its expectations are ported. Port rules:
- Assertions are kept at the same strength as before. A case that asserted only the tool name still asserts only the tool name.
- Prompts are kept verbatim, except five that name concepts that no longer exist (listed below).
- New value forms: `one_of a|b` where v3 offers two equivalent nodes, and index `end`, which means omitted index, index equal to the child count, or after the last child.

| Group | Cases | Port |
|---|---|---|
| Clarification (expect no tool) | WRAP_root_generic, TREE_insert_runtime_generic, INSERT_position_before_named_generic, PARALLEL_concurrent_medium, PARALLEL_generic, FALLBACK_alternate_medium, FALLBACK_generic, UPDATE_params_generic, MOVE_generic, SCENE_remove_generic, CLARIFY_* (3) | Unchanged |
| Wrap | P1, WRAP_root_medium, CUR_wrap_existing_motion_with_retry, PnP_18 | `target_instance_id`, `wrapper.node_id/parameters`; state `program.tree.node_id` |
| Scene add / update / remove | P2, SCENE_add_*, SCENE_update_*, SCENE_remove_specific, CUR_scene_*, PnP_01..04 | `scene_resource.*`, `box_dimensions_m`, `position_m`; state `scene.objects[scene_resource_id=…]` |
| Tree insert | P3, TREE_insert_runtime_medium, INSERT_*, FALLBACK_retry_specific, CUR_runtime_*, PnP_06..17 | v3 node ids; `node.parameters.*`; `parent_instance_id` / `after_instance_id` / `index` |
| Update / delete / move / replace | UPDATE_params_*, DELETE_*, MOVE_*, REPLACE_*, CUR_update_existing_motion_speed, PnP_20 | `instance_id`, `parameters.*`, `new_parent_instance_id`, `subtree` |
| Parameters / blackboard | PARAM_*, BB_* | `parameter_id` / `float64`; blackboard `key` / `float64` / `stored_pose`; fixtures through Draft mutations |

Prompts ported, with the original kept as `original_prompt`:
- **INSERT_position_first_specific** and **FALLBACK_retry_specific:** they named removed node ids (`wait_for_signal_bool`, `move_manipulator_action`). Now `wait_for_value` and `move_to_pose`.
- **PnP_09_attach** and **PnP_13_detach:** they named link `grasp_frame`. The UR10e/2F-140 model has no such frame, and the v3 attach uses the active TCP or a `{component_id, frame_id}` carrier. Now "attach graspable to the gripper" and "detach graspable".
- **CUR_scene_update_graspable_pose:** the `universe` frame is now `cell_frame`.

Expectations relaxed to match v3 semantics:
- **TREE_insert_runtime_medium** and **CUR_runtime_update_graspable_pose_each_cycle:** accept `scene.upsert_object` or `scene.update_object_pose`. v3 has a dedicated pose-update node that fits "re-place".
- **PnP_17:** accepts `move_to_pose` or `move_to_joint_target` for "return home".
- **PnP_08 / PnP_15:** the old numeric gripper asserts (0.66, 40, 1.0, 0.0) came from the old untyped parameters and are not in the prompt. They are dropped; the node id is still asserted.
- **PnP_09:** the old `link_frame` assert is dropped.
- **CUR_runtime_remove_then_restore:** `scene_inspect` also accepts `program_read`. Both are read tools that expose the scene in v3.

Behaviour that differs now and may move scores:
- **SCENE_remove_specific:** Composer accepts removing `graspable` even though `attach_graspable` and `detach_graspable` reference it. The Draft stays incomplete until the user fixes it. The expected outcome is unchanged.
- **PnP_20 ("40 N"):** the 2F-140 effort is angular (N·m). A model that sets `value_n` gets a typed Composer refusal; only the tool name is asserted.
- **Typed gripper and frame values and v3 node schemas are larger than the old ones.** `tree_draft_insert_node` alone is about 62 KB of schema, and the 24 tools are about 280 KB. This weighs on context and on small models.

Proposed drops: none.

## 5. What remains for Phase 3

- Move the runner and corpus to `manyforge-assistant`. Retire the legacy runner's OpenClaw-specific code (log scraping, self-heal splice).
- Re-baseline Gemma 4 12B QAT for comparison with iter-32.
- Add cases for v3-only features: `component_frame` scene poses, `value_sources`, `follow_cartesian_path`, `change_node_kind`, `rename_node`.
