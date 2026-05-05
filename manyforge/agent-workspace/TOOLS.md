# Tool routing for the ManyForge Composer Assistant

Your primary tool surface is the **manyforge** MCP server. OpenClaw
mangles tool ids (dots → **dashes**, `manyforge__` prefix); the
canonical ManyForge id is in parentheses.

> Tool-name mangling rule (verified 2026-05-05): OpenClaw's bundle-mcp
> registers `scene.inspect` as `manyforge__scene-inspect`. The dot
> becomes a **dash**, not an underscore. Use the dash form below
> verbatim — calling `manyforge__scene_inspect` will fail with
> "tool not available".

## State-reading tools (call FIRST for any state question)

- `manyforge__scene-inspect`        — read current scene resources (`scene.inspect`)
- `manyforge__program-read`         — read full program: tree + scene + parameters (`program.read`)
- `manyforge__catalog-read`         — read node catalog entries (`catalog.read`)
- `manyforge__skills-read`          — read declared skills (`skills.read`)
- `manyforge__deployment-capabilities-read` — read deployment capabilities (`deployment.capabilities.read`)
- `manyforge__status-read`          — read runtime status (`status.read`)

## Scene-resource edits (compile-time; affect program scene)

- `manyforge__scene-draft-add-object`     (`scene.draft.add_object`)
- `manyforge__scene-draft-update-object`  (`scene.draft.update_object`)
- `manyforge__scene-draft-upsert-objects` (`scene.draft.upsert_objects`)
- `manyforge__scene-draft-remove-objects` (`scene.draft.remove_objects`)

## Behavior-tree edits (runtime; affect program tree)

- `manyforge__tree-draft-insert-node`        (`tree.draft.insert_node`)
- `manyforge__tree-draft-update-node-params` (`tree.draft.update_node_params`)
- `manyforge__tree-draft-delete-node`        (`tree.draft.delete_node`)
- `manyforge__tree-draft-move-node`          (`tree.draft.move_node`)
- `manyforge__tree-draft-replace-subtree`    (`tree.draft.replace_subtree`)
- `manyforge__tree-draft-wrap-node`          (`tree.draft.wrap_node`)

## Program parameters / blackboard

- `manyforge__program-draft-upsert-parameters`     (`program.draft.upsert_parameters`)
- `manyforge__program-draft-remove-parameters`     (`program.draft.remove_parameters`)
- `manyforge__blackboard-draft-upsert-namespaces`  (`blackboard.draft.upsert_namespaces`)
- `manyforge__blackboard-draft-remove-namespaces`  (`blackboard.draft.remove_namespaces`)
- `manyforge__blackboard-draft-upsert-keys`        (`blackboard.draft.upsert_keys`)
- `manyforge__blackboard-draft-remove-keys`        (`blackboard.draft.remove_keys`)

## What to ignore

- **`session_status`**: this is OpenClaw's internal session monitor.
  It is NEVER the right tool for a ManyForge question. The user does
  not have a "session key" to give you. Use `manyforge__scene-inspect`
  or `manyforge__program-read` for any state question.

## Routing rules

| User wants… | Call this | NOT this |
|---|---|---|
| "what's in the scene" / "describe the scene" / "show me the scene" | `manyforge__scene-inspect` | `session_status`, no clarification question |
| "show the program" / "what does the tree look like" / "what's the root" | `manyforge__program-read` | `session_status` |
| "add a box" / "add a static object" | `manyforge__scene-draft-add-object` | tree-edit tools |
| "wrap the program with a repeat" / "make repeat the new root" | `manyforge__tree-draft-wrap-node` with `targetName: "@root"` | `tree_draft_replace_subtree` |
| "what catalog ids exist" / "what node kinds are valid" | `manyforge__catalog-read` | session_status |

## Anti-patterns (never do these)

1. Asking the user "which session?" or "session key?" in response to a
   ManyForge question. ManyForge has no sessions. Call the relevant
   read tool instead.
2. Inventing catalog ids with descriptive suffixes (`repeat_root`,
   `repeat_node`). Use the literal id from the catalog.
3. Setting `wrapper.children` on `tree-draft-wrap-node`. Omit the field;
   the handler attaches the existing target subtree automatically.
4. Fabricating object poses. Call `scene-inspect` and copy from the
   result.
5. Identical-call retries. If a call fails, read the structured error
   fields in the response — do not re-send the same arguments.
6. Calling tools with underscore form (`manyforge__scene_inspect`).
   The mangled MCP names use **dashes**. Use the names exactly as
   listed above.
