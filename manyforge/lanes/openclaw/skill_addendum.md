# Discovery-protocol primer for the OpenClaw lane

This addendum is appended to the lane-agnostic skill body (planned for `manyforge/common/skill_base.md`). It teaches the model how to use OpenClaw 2026.5.6+'s native tool-search compaction surface — the discovery protocol that replaces the direct tool catalog with three control tools (`tool_search`, `tool_describe`, `tool_call`).

The protocol exists because OpenClaw's gateway, by design, only exposes three tool-shaped "control verbs" in the model's `tools[]` array. The real ManyForge catalog (program_read, tree_draft_wrap_node, scene_draft_add_object, …) is *discoverable* via these verbs but never visible to the model directly. Without this primer, the model wastes turns guessing tool names that don't appear in its visible surface.

## The protocol

You see three control tools in your `tools[]`:

1. **`tool_search({query: <keywords>})`** — returns a list of *compact* tool descriptors (`{id, name, label, description}` — NO parameter schema yet). The query is **substring-tokenized over name/id/label/description**, not semantic. Search scoring weights: exact match ×20, name substring ×8, id ×6, label ×4, description ×2. Default limit 8, max 50. If your query terms don't substring-match any field, the search returns ZERO hits even when the right tool is present.

2. **`tool_describe({id: <tool name or id>})`** — given a tool name or id from a search result, returns the FULL parameter schema (the JSON Schema for the tool's arguments). You need this before you can call the tool.

3. **`tool_call({id: <tool name or id>, args: {…}})`** — finally invokes the discovered tool. Returns `{tool: <compact entry>, result: <real result>}`. **The real tool output is in `.result`** — `.tool` is just a wrapper acknowledging which tool ran.

## How to be efficient

The naive flow is 3 round-trips per real tool invocation: search → describe → call. That's expensive (and slow). Use these rules to collapse it:

### Rule 1 — Pre-named tools skip `tool_search`

`tool_describe` accepts a bare tool name as `id`. If you already know the tool name (from this primer or from earlier in the conversation), go straight to `tool_describe({id: "tree_draft_wrap_node"})` — no `tool_search` needed.

### Rule 2 — Describe once per conversation

Once you have a tool's schema in this conversation's working memory, **do not call `tool_describe` for it again**. Just `tool_call` directly. The full catalog is registered server-side at session start and all real tools remain callable for the entire session — even after compactions.

### Rule 3 — Read the envelope correctly

`tool_call` returns `{tool, result}`. Treat `result.content[].text` (or `result.details`) as the actual tool output. Ignore the `tool` wrapper.

### Rule 4 — Use multi-term queries when name is unknown

When you genuinely don't know a tool's name, search with TWO terms — both name fragments and descriptions. The substring tokenizer matches all terms, so `tool_search({query: "tree wrap"})` will hit `tree_draft_wrap_node` (name match) AND any tool with "wrap" in its description.

### Rule 5 — On error, read the envelope

If `tool_call` fails (bad args, missing required field), the error flows back through `result.content` with `isError: true` and a human-readable message. **Read the error and adjust your args** — do NOT retry with the same args. The composer-side validator returns HTTP 200 with the structured error envelope (per the OpenClaw drop-policy workaround in `routes_assistant.py:execute_bridge_tool`), so the conversation continues even on validation failures.

### Rule 6 — Tool arguments are camelCase + nested (NOT snake_case / flat)

ManyForge tool schemas use **camelCase** for top-level keys and **nested objects** for compound payloads. The model's training corpus often produces snake_case Python-style argument names — those are rejected by the validator. Use the canonical names below; the validator's error envelope also includes a `diff.hint` line that names the exact rename when this rule is violated.

| Tool | Wrong (snake_case / flat) | Correct (camelCase / nested) |
|---|---|---|
| `tree_draft_wrap_node` | `target_name`, `wrapper_id`, `wrapper_name`, `wrapper_params` | `targetName`, `wrapper: { id, name, params }` |
| `tree_draft_insert_node` | `parent_name`, `node_id`, `node_name`, `node_params`, `after_name`, `before_name` | `parentName`, `node: { id, name, params }`, `position: { afterName, beforeName, index }` |
| `tree_draft_update_node_params` | `node_name`, `merge:True` | `nodeName`, `merge:true`, `params` |
| `tree_draft_delete_node` | `node_name` | `nodeName` |
| `tree_draft_move_node` | `node_name`, `new_parent`, `new_position` | `nodeName`, `newParentName`, `position` |
| `tree_draft_replace_subtree` | `target_name`, `replacement_id` | `targetName`, `replacement: { id, name, params, children }` |
| `tree_draft_change_node_kind` | `node_name`, `new_kind` | `nodeName`, `newKind`, `params` |
| `scene_draft_add_object` | `box_dims`, `position`, `orientation` | `shape: { type, box_dims }`, `pose: { position, orientation_quat }` |
| `scene_draft_update_object` | `object_id`, `pose_position` | `objectId`, `pose: { position, orientation_quat }` |
| `scene_draft_remove_object` | `object_id` | `objectId` |

**General rule**: camelCase for top-level keys; nested objects for compound payloads (`wrapper`, `node`, `shape`, `pose`, `replacement`, `position`). When in doubt, call `tool_describe` first and use the EXACT keys from its returned schema — do not transliterate Python-style names. The validator returns a structured diff naming the rename; read it and reapply.

## Pre-named tool list (this skill's vocabulary)

For ManyForge robot programming tasks, the most common tools by name. Most of these you can skip directly to `tool_describe`:

### Program/scene read (always available; no preconditions)

- `program_read` — read the loaded behavior-tree program's state.
- `program_validate` — validate the current draft.
- `scene_inspect` — inspect the loaded scene.
- `catalog_read` — read the node + skill catalog (which tools/nodes are available).
- `skills_read` — read available robot skills.
- `deployment_capabilities_read` — what the runtime can do.
- `status_read` — runtime status.

### Tree mutations (draft layer)

- `tree_draft_insert_node` — insert a node under a parent at a position.
- `tree_draft_wrap_node` — wrap an existing node with a decorator.
- `tree_draft_remove_node` — remove a node by name.
- `tree_draft_update_node_params` — patch a node's params.
- `tree_draft_replace_node` — replace a node by name.
- `tree_draft_move_node` — move a node to a new parent/position.
- `tree_draft_change_node_kind` — change a node's `id` (kind) and re-shape its params.
- `tree_draft_replace_subtree` — replace a node AND its entire subtree.

### Scene mutations (draft layer)

- `scene_draft_add_object` — add a collision object to the scene draft.
- `scene_draft_update_object` — update an existing object's pose/shape.
- `scene_draft_remove_object` — remove an object by id.

### Blackboard/runtime config

- `blackboard_draft_upsert_namespaces` — upsert key namespaces.
- `blackboard_draft_set` — set a blackboard key value.
- `program_draft_upsert_parameters` — upsert program parameters.
- `program_draft_remove_parameters` — remove program parameters.
- `program_draft_replace_blackboard` — replace the entire blackboard config.

When in doubt about an exact name, `tool_search` is always safe — but prefer the direct-name path above whenever possible.

## What success looks like

A correct sequence for "wrap the pick_and_place sequence in a retry-3 decorator":

```
turn 1: tool_describe({id: "tree_draft_wrap_node"})
        → returns parameter schema
turn 2: tool_call({id: "tree_draft_wrap_node",
                   args: {targetName: "pick_and_place",
                          wrapper: {id: "retry",
                                    name: "pick_and_place_retry",
                                    params: {num_attempts: 3}}}})
        → returns {tool, result} — the program tree is now wrapped.
```

Two turns, one real mutation. That's the efficient path; the model that calls `tool_search` first ("tree wrap") adds a turn for nothing if it already knows `tree_draft_wrap_node`.
