"""Program / scene / tree / catalog projection helpers.

These functions translate the raw Composer state snapshots
(``/api/program``, ``/api/program/tree``, ``/api/scene/state``,
``/api/assistant/modes/<mode>``) into the compact summaries that the
agent prompt embeds.

Phase 1 status: this module currently re-exports from
``openclaw_assistant_bridge.adapter`` to preserve behavior bit-by-bit
while the package structure lands. Phase 2 will move the implementations
here so the Direct lane can import the same functions without the
historical mirror in ``dev_ws/manyforge_assistant_bridge/bridge.py``.

The mirror is documented at adapter.py:183 ("Mirror of
manyforge_assistant_bridge.bridge._build_program_summary") and dies in
Phase 1's grep-proof gate.
"""
from __future__ import annotations

from openclaw_assistant_bridge.adapter import (  # noqa: F401
    _build_program_summary as build_program_summary,
    _build_scene_summary as build_scene_summary,
    _collect_ancestor_paths as collect_ancestor_paths,
    _collect_tree_node_names as collect_tree_node_names,
    _project_node_catalog as project_node_catalog,
    _project_scene_object as project_scene_object,
    _project_skill_catalog as project_skill_catalog,
    _project_tree_node as project_tree_node,
)

__all__ = [
    "build_program_summary",
    "build_scene_summary",
    "collect_ancestor_paths",
    "collect_tree_node_names",
    "project_node_catalog",
    "project_scene_object",
    "project_skill_catalog",
    "project_tree_node",
]
