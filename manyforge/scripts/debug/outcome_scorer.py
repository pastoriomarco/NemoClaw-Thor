"""Outcome scorer for the assistant smoke corpus v2 (manyforge.assistant_smoke_corpus.v2).

A case passes when the operator's request is fulfilled in the Program Draft:
declarative checks on the Draft after the conversation, no collateral edits,
no new error diagnostics and an answer consistent with the actual change.
Tool choice, argument shapes, call counts and time are not scored.

Pure functions over Draft snapshots ``{"authoring": ..., "diagnostics": [...]}``
as returned by Composer ``GET /api/program/draft``; no I/O here.

Entity identity used by the ``unchanged`` diff:
  node            instance_id; content = the node without ``children``;
                  location = parent instance_id (None at top level) and the
                  order among siblings that are in both Drafts under the same parent
  scene resource  scene_resource_id
  parameter       parameter_id
  blackboard key  key;  blackboard namespace: namespace_id
  contact policy  policy_id
  metadata        every other authoring field except ``positions`` (GUI layout only)
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable

TOL = 1e-3
COMPARATORS = ("max_exclusive", "min_exclusive", "max_inclusive", "min_inclusive")
INCOMPLETE_CODES = ("authoring_node_children_incomplete", "authoring_parameter_required")
CHECK_KINDS = ("node", "node_absent", "scene_resource", "parameter", "blackboard_key",
               "contact_policy", "any_of", "differ", "subtree_order")
NODE_FIELDS = ("instance_id", "node_id", "parent", "parent_node_id", "ancestor", "ancestor_node_id",
               "after", "before", "value_sources",
               "last_child", "index", "parameters", "effective_parameters", "children_node_ids",
               "children_order", "may_remove")
META_SKIP = ("forest", "scene", "parameters", "blackboard", "contact_policies", "positions")


# --- values ------------------------------------------------------------------
def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _quat(o: Any) -> list[float] | None:
    """A unit quaternion [x, y, z, w] from ``{xyzw}`` or ``{rpy_deg}`` (ROS fixed-axis
    roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll))."""
    if not isinstance(o, dict):
        return None
    if isinstance(o.get("xyzw"), list) and len(o["xyzw"]) == 4 and all(map(_num, o["xyzw"])):
        q = [float(v) for v in o["xyzw"]]
    elif isinstance(o.get("rpy_deg"), list) and len(o["rpy_deg"]) == 3 and all(map(_num, o["rpy_deg"])):
        r, p, y = (math.radians(v) / 2 for v in o["rpy_deg"])
        cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
        q = [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
             cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy]
    else:
        return None
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q] if n > 0 else None


def same_rotation(a: Any, b: Any) -> bool:
    qa, qb = _quat(a), _quat(b)
    if qa is None or qb is None:
        return False
    sign = 1.0 if sum(x * y for x, y in zip(qa, qb)) >= 0 else -1.0
    return all(abs(x - sign * y) <= TOL for x, y in zip(qa, qb))


def _is_orientation(v: Any) -> bool:
    return isinstance(v, dict) and set(v) in ({"xyzw"}, {"rpy_deg"})


def values_equal(a: Any, b: Any, key: str | None = None) -> bool:
    """Symmetric equality: numbers within TOL, orientations as rotations."""
    if key == "orientation" and _is_orientation(a) and _is_orientation(b):
        return same_rotation(a, b)
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if _num(a) and _num(b):
        return abs(a - b) <= TOL
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(values_equal(a[k], b[k], k) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b))
    return a == b


def diff_paths(a: Any, b: Any, prefix: str = "", key: str | None = None) -> list[str]:
    """Dotted paths where two values differ (leaf-most dict level)."""
    if isinstance(a, dict) and isinstance(b, dict) and not (
            key == "orientation" and _is_orientation(a) and _is_orientation(b)):
        out: list[str] = []
        for k in sorted(set(a) | set(b)):
            p = f"{prefix}.{k}" if prefix else k
            if k not in a or k not in b:
                out.append(p)
            else:
                out.extend(diff_paths(a[k], b[k], p, k))
        return out
    return [] if values_equal(a, b, key) else [prefix or "<value>"]


ABSENT = {"absent": True}


def _cond_eval(c: Any, env: dict[str, bool]) -> bool | None:
    """Truth value of an interrupt condition ``{all:[C]}|{any:[C]}|{not:C}|{compare:{operand, op, value}}``
    under ``env`` (atom -> bool); None when the condition is malformed."""
    if not isinstance(c, dict) or len(c) != 1:
        return None
    kind, item = next(iter(c.items()))
    if kind in ("all", "any"):
        if not isinstance(item, list) or not item:
            return None
        vals = [_cond_eval(x, env) for x in item]
        return None if None in vals else (all(vals) if kind == "all" else any(vals))
    if kind == "not":
        v = _cond_eval(item, env)
        return None if v is None else not v
    if kind == "compare" and isinstance(item, dict):
        atom, flip = _cond_atom(item)
        return None if atom is None else env[atom] != flip
    return None


def _cond_atom(row: dict) -> tuple[str | None, bool]:
    """A compare's boolean atom and whether the compare is its negation. A bool literal compared
    with == / != is the operand itself (``x == false`` is ``not x``); any other compare is opaque."""
    import json
    if not {"operand", "op", "value"} <= set(row):
        return None, False
    if isinstance(row["value"], bool) and row["op"] in ("==", "!="):
        return json.dumps(row["operand"], sort_keys=True), (row["op"] == "!=") != (row["value"] is False)
    return json.dumps(row, sort_keys=True), False


def _cond_atoms(c: Any, out: set[str]) -> None:
    if isinstance(c, dict) and len(c) == 1:
        kind, item = next(iter(c.items()))
        if kind in ("all", "any") and isinstance(item, list):
            for x in item:
                _cond_atoms(x, out)
        elif kind == "not":
            _cond_atoms(item, out)
        elif kind == "compare" and isinstance(item, dict):
            atom, _ = _cond_atom(item)
            if atom is not None:
                out.add(atom)


def conditions_equivalent(a: Any, b: Any) -> bool:
    """Two interrupt conditions are the same boolean function of their atoms (all/any/not/==/!=
    rewrites, single-item all/any)."""
    import itertools
    atoms: set[str] = set()
    _cond_atoms(a, atoms)
    _cond_atoms(b, atoms)
    names = sorted(atoms)
    if not names or len(names) > 12:
        return False
    for bits in itertools.product((False, True), repeat=len(names)):
        env = dict(zip(names, bits))
        va, vb = _cond_eval(a, env), _cond_eval(b, env)
        if va is None or vb is None or va != vb:
            return False
    return True


def matches(expected: Any, actual: Any, key: str | None = None) -> bool:
    """Expected-value match: dicts as subsets, numbers within TOL, orientations as
    rotations, ``{max_exclusive|min_exclusive|max_inclusive|min_inclusive: x}`` bounds,
    ``{condition_equivalent: C}`` (an interrupt condition with the same truth table as C).
    ``{absent: true}`` as a dict member: that key is not set."""
    if isinstance(expected, dict) and set(expected) == {"condition_equivalent"}:
        return conditions_equivalent(expected["condition_equivalent"], actual)
    if isinstance(expected, dict) and expected and set(expected) <= set(COMPARATORS):
        if not _num(actual):
            return False
        lim = expected
        return (("max_exclusive" not in lim or actual < lim["max_exclusive"] - TOL / 10)
                and ("min_exclusive" not in lim or actual > lim["min_exclusive"] + TOL / 10)
                and ("max_inclusive" not in lim or actual <= lim["max_inclusive"] + TOL)
                and ("min_inclusive" not in lim or actual >= lim["min_inclusive"] - TOL))
    if _is_orientation(expected) and (key == "orientation" or _is_orientation(actual)):
        return same_rotation(expected, actual)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k not in actual if v == ABSENT else (k in actual and matches(v, actual[k], k))
            for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            matches(e, a) for e, a in zip(expected, actual))
    return values_equal(expected, actual)


_SEG = re.compile(r"[^.\[\]]+|\[-?\d+\]")


def _split(path: str) -> list[Any]:
    return [int(s[1:-1]) if s.startswith("[") else (int(s) if s.isdigit() else s)
            for s in _SEG.findall(path)]


def _resolve(obj: Any, segs: list[Any]) -> tuple[bool, Any]:
    for s in segs:
        if isinstance(s, int) and isinstance(obj, list) and -len(obj) <= s < len(obj):
            obj = obj[s]
        elif isinstance(obj, dict) and s in obj:
            obj = obj[s]
        else:
            return False, None
    return True, obj


def path_matches(container: Any, path: str, expected: Any) -> bool:
    """``a.b.c: v`` against a nested dict. ``...orientation.xyzw`` / ``.rpy_deg`` and the
    v1 leaves ``orientation_xyzw`` / ``orientation_rpy_deg`` compare as rotations."""
    segs = _split(path)
    key = segs[-1] if segs else None
    if len(segs) >= 2 and segs[-1] in ("xyzw", "rpy_deg") and segs[-2] == "orientation":
        segs, expected, key = segs[:-1], {segs[-1]: expected}, "orientation"
    elif segs and segs[-1] in ("orientation_xyzw", "orientation_rpy_deg"):
        expected = {segs[-1][len("orientation_"):]: expected}
        segs, key = segs[:-1] + ["orientation"], "orientation"
    found, actual = _resolve(container, segs)
    if expected == ABSENT:
        return not found
    return found and matches(expected, actual, key if isinstance(key, str) else None)


# --- Draft index ----------------------------------------------------------------
class Draft:
    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.raw = snapshot
        a = snapshot.get("authoring") or {}
        self.authoring = a
        self.forest: list[dict] = a.get("forest") or []
        self.diagnostics: list[dict] = snapshot.get("diagnostics") or []
        self.nodes: dict[str, dict] = {}
        self.parent: dict[str, str | None] = {}
        self.children: dict[str | None, list[str]] = {None: []}
        stack = [(n, None) for n in reversed(self.forest)]
        while stack:
            node, pid = stack.pop()
            iid = node.get("instance_id")
            self.nodes[iid] = node
            self.parent[iid] = pid
            self.children.setdefault(pid, []).append(iid)
            self.children.setdefault(iid, [])
            stack.extend((c, iid) for c in reversed(node.get("children") or []))
        self.scene = {o.get("scene_resource_id"): o for o in a.get("scene") or []}
        self.params = {p.get("parameter_id"): p for p in a.get("parameters") or []}
        bb = a.get("blackboard") or {}
        self.keys = {k.get("key"): k for k in bb.get("keys") or []}
        self.namespaces = {n.get("namespace_id"): n for n in bb.get("namespaces") or []}
        self.policies = {p.get("policy_id"): p for p in a.get("contact_policies") or []}
        self.meta = {k: v for k, v in a.items() if k not in META_SKIP}

    def content(self, iid: str) -> dict:
        return {k: v for k, v in self.nodes[iid].items() if k != "children"}

    def ancestors(self, iid: str) -> list[str]:
        out, p = [], self.parent.get(iid)
        while p is not None:
            out.append(p)
            p = self.parent.get(p)
        return out

    def preorder(self, root: str) -> list[str]:
        out, stack = [], [root]
        while stack:
            iid = stack.pop()
            out.append(iid)
            stack.extend(reversed(self.children.get(iid, [])))
        return out

    def effective(self, iid: str) -> dict:
        """Node parameters with ``program_parameter`` value sources resolved to the
        Program parameter default; other sources stay as ``{"value_source": ...}``."""
        import copy
        node = self.nodes[iid]
        params = copy.deepcopy(node.get("parameters") or {})
        for path, source in (node.get("value_sources") or {}).items():
            if isinstance(source, dict) and source.get("kind") == "program_parameter" \
                    and source.get("reference") in self.params:
                value = copy.deepcopy(self.params[source["reference"]].get("default"))
            else:
                value = {"value_source": source}
            segs = _split(path)
            target = params
            for s in segs[:-1]:
                target = target.setdefault(s, {}) if isinstance(target, dict) else {}
            if isinstance(target, dict):
                target[segs[-1]] = value
        return params


# --- matching context -------------------------------------------------------------
class Match:
    """Entities the checks matched; the ``unchanged`` diff ignores them."""

    def __init__(self) -> None:
        self.nodes: dict[str, set[str] | None] = {}   # iid -> covered content prefixes; None = all
        self.removable: dict[str, set[str]] = {}       # iid -> parameter paths the case lets the edit remove
        self.scene: set[str] = set()
        self.params: set[str] = set()
        self.keys: set[str] = set()
        self.policies: set[str] = set()

    def add_node(self, iid: str, prefixes: set[str] | None) -> None:
        if iid in self.nodes and self.nodes[iid] is None:
            return
        if prefixes is None:
            self.nodes[iid] = None
        else:
            self.nodes[iid] = (self.nodes.get(iid) or set()) | prefixes

    def merge(self, other: "Match") -> None:
        for iid, p in other.nodes.items():
            self.add_node(iid, p)
        for iid, r in other.removable.items():
            self.removable[iid] = self.removable.get(iid, set()) | r
        self.scene |= other.scene
        self.params |= other.params
        self.keys |= other.keys
        self.policies |= other.policies


def _node_changed(before: Draft, after: Draft, iid: str) -> bool:
    return (iid not in before.nodes
            or before.parent.get(iid) != after.parent.get(iid)
            or not values_equal(before.content(iid), after.content(iid)))


def _sibling_ref(draft: Draft, iid: str | None, ref: str) -> bool:
    return iid is not None and (iid == ref or draft.nodes[iid].get("node_id") == ref)


def node_mismatch(spec: dict, draft: Draft, iid: str) -> str | None:
    """The first criterion of a node check this node fails, or None."""
    node = draft.nodes[iid]
    pid = draft.parent.get(iid)
    siblings = draft.children.get(pid, [])
    pos = siblings.index(iid)
    if "instance_id" in spec and iid != spec["instance_id"]:
        return "instance_id"
    if "node_id" in spec:
        want = spec["node_id"]
        if node.get("node_id") not in (want if isinstance(want, list) else [want]):
            return "node_id"
    if "parent" in spec and pid != spec["parent"]:
        return f"parent (is {pid})"
    if "parent_node_id" in spec and (pid is None or draft.nodes[pid].get("node_id") != spec["parent_node_id"]):
        return "parent_node_id"
    if "ancestor" in spec and spec["ancestor"] not in draft.ancestors(iid):
        return "ancestor"
    if "ancestor_node_id" in spec:
        want = spec["ancestor_node_id"]
        want = want if isinstance(want, list) else [want]
        if not any(draft.nodes[a].get("node_id") in want for a in draft.ancestors(iid)):
            return "ancestor_node_id"
    if "after" in spec and not (pos > 0 and _sibling_ref(draft, siblings[pos - 1], spec["after"])):
        return f"after (previous sibling {siblings[pos - 1] if pos > 0 else None})"
    if "before" in spec and not (pos + 1 < len(siblings) and _sibling_ref(draft, siblings[pos + 1], spec["before"])):
        return f"before (next sibling {siblings[pos + 1] if pos + 1 < len(siblings) else None})"
    if spec.get("last_child") and pos != len(siblings) - 1:
        return f"last_child (index {pos} of {len(siblings)})"
    if "index" in spec:
        want = spec["index"]
        if (want == "end" and pos != len(siblings) - 1) or (isinstance(want, int) and pos != (
                want if want >= 0 else len(siblings) + want)):
            return f"index (is {pos})"
    if "children_node_ids" in spec:
        have = [draft.nodes[c].get("node_id") for c in draft.children.get(iid, [])]
        if not _children_match(spec["children_node_ids"], have, spec.get("children_order") == "any"):
            return f"children_node_ids (has {have})"
    for path, value in (spec.get("parameters") or {}).items():
        if not path_matches(node.get("parameters") or {}, path, value):
            return f"parameters.{path}"
    for path, value in (spec.get("value_sources") or {}).items():
        if not matches(value, (node.get("value_sources") or {}).get(path)):
            return f"value_sources.{path}"
    if spec.get("effective_parameters"):
        eff = draft.effective(iid)
        for path, value in spec["effective_parameters"].items():
            if not path_matches(eff, path, value):
                return f"effective_parameters.{path}"
    return None


def _children_match(want: list, have: list, any_order: bool) -> bool:
    """Each wanted child is a node_id or a list of acceptable node_ids."""
    if len(want) != len(have):
        return False
    ok = lambda w, h: h in (w if isinstance(w, list) else [w])
    if not any_order:
        return all(ok(w, h) for w, h in zip(want, have))
    import itertools
    return any(all(ok(w, h) for w, h in zip(want, perm)) for perm in itertools.permutations(have))


def _covered_prefixes(spec: dict) -> set[str]:
    out = {f"parameters.{p}" for p in spec.get("parameters") or {}}
    for p in spec.get("value_sources") or {}:
        out |= {f"value_sources.{p}", f"parameters.{p}"}
    for p in spec.get("effective_parameters") or {}:
        out |= {f"parameters.{p}", f"value_sources.{p}", f"value_sources.{_split(p)[0]}",
                f"parameters.{_split(p)[0]}"}
    return out


def _describe(check: dict) -> str:
    kind = next(iter(check))
    body = check[kind]
    if isinstance(body, dict):
        short = ", ".join(f"{k}={v}" for k, v in body.items() if k not in ("parameters", "effective_parameters"))
        return f"{kind}{{{short}}}"
    return kind


def evaluate(check: dict, before: Draft, after: Draft) -> tuple[str | None, Match]:
    """One check on the after-Draft: (failure text or None, matched entities)."""
    kind = next(iter(check))
    spec = check[kind]
    m = Match()
    if kind == "any_of":
        reasons = []
        for alt in spec:
            fail, mm = evaluate_all(alt, before, after)
            if fail is None:
                return None, mm
            reasons.append(fail)
        return f"any_of: no alternative holds ({'; '.join(reasons)})", m
    if kind == "node":
        if spec.get("instance_id") in before.nodes and spec["instance_id"] not in after.nodes:
            # a named node that was removed matches its equal-content copy (renamed_equivalent)
            copies = aliases(before, after, spec["instance_id"])
            hits = [i for i in copies if node_mismatch({**spec, "instance_id": i}, after, i) is None]
        else:
            hits = [i for i in after.nodes if node_mismatch(spec, after, i) is None]
        if not hits:
            near = [(i, node_mismatch(spec, after, i)) for i in after.nodes
                    if "node_id" not in spec or node_mismatch({"node_id": spec["node_id"]}, after, i) is None]
            hint = "; ".join(f"{i}: {r}" for i, r in near[:3])
            return f"{_describe(check)}: no match" + (f" ({hint})" if hint else ""), m
        changed = [i for i in hits if _node_changed(before, after, i)]
        prefixes = _covered_prefixes(spec)
        for i in changed or hits[:1]:
            # a check naming the node's new kind covers the kind change and its parameters
            kind_changed = (i in before.nodes and "node_id" in spec
                            and before.nodes[i].get("node_id") != after.nodes[i].get("node_id"))
            m.add_node(i, None if i not in before.nodes or kind_changed else prefixes)
            if spec.get("may_remove"):
                m.removable[i] = m.removable.get(i, set()) | {f"parameters.{p}" for p in spec["may_remove"]}
        return None, m
    if kind == "node_absent":
        if "instance_id" in spec:
            if spec["instance_id"] in after.nodes:
                return f"{_describe(check)}: still present", m
            m.add_node(spec["instance_id"], None)
            return None, m
        pid = spec.get("parent")
        present = [i for i in after.children.get(pid, []) if after.nodes[i].get("node_id") == spec.get("node_id")]
        if present:
            return f"{_describe(check)}: still present ({present})", m
        for i in before.children.get(pid, []):
            if before.nodes[i].get("node_id") == spec.get("node_id"):
                m.add_node(i, None)
        return None, m
    if kind == "scene_resource":
        def ok(o: dict) -> bool:
            return all(k not in spec or matches(spec[k], o.get(k), k) for k in ("shape", "pose"))
        if spec.get("id") is not None:
            o = after.scene.get(spec["id"])
            m.scene.add(spec["id"])
            if not spec.get("present", True):
                return (None if o is None else f"{_describe(check)}: still present"), m
            if o is None:
                return f"{_describe(check)}: absent", m
            if spec.get("new") and spec["id"] in before.scene:
                return f"{_describe(check)}: not new", m
            return (None if ok(o) else f"{_describe(check)}: shape/pose differ ({ {k: o.get(k) for k in ('shape', 'pose')} })"), m
        cands = [i for i, o in after.scene.items()
                 if ok(o) and not (spec.get("new") and i in before.scene)]
        if not cands:
            return f"{_describe(check)}: no matching resource", m
        changed = [i for i in cands if i not in before.scene or not values_equal(before.scene[i], after.scene[i])]
        m.scene.update(changed or cands[:1])
        return None, m
    if kind in ("parameter", "blackboard_key"):
        ident = spec.get("id") if kind == "parameter" else spec.get("key")
        table = after.params if kind == "parameter" else after.keys
        old = before.params if kind == "parameter" else before.keys
        if ident is None:  # any new row with these fields (the model chooses the name)
            fields = [f for f in ("type", "default", "initial") if f in spec]
            hits = [i for i, row in table.items() if i not in old
                    and all(matches(spec[f], row.get(f), f) for f in fields)]
            if not hits:
                return f"{_describe(check)}: no new matching {kind}", m
            (m.params if kind == "parameter" else m.keys).update(hits[:1])
            return None, m
        (m.params if kind == "parameter" else m.keys).add(ident)
        row = table.get(ident)
        if not spec.get("present", True):
            return (None if row is None else f"{_describe(check)}: still present"), m
        if row is None:
            return f"{_describe(check)}: absent", m
        for f in ("type", "default", "initial"):
            if f in spec and not matches(spec[f], row.get(f), f):
                return f"{_describe(check)}: {f} is {row.get(f)!r}", m
        return None, m
    if kind == "contact_policy":  # by id, or any new policy with this mode (the model chooses the id)
        ident = spec.get("id")
        if ident is None:
            hits = [i for i, row in after.policies.items() if i not in before.policies
                    and ("mode" not in spec or row.get("mode") == spec["mode"])]
            if not hits:
                return f"{_describe(check)}: no new matching contact_policy", m
            m.policies.update(hits[:1])
            return None, m
        m.policies.add(ident)
        row = after.policies.get(ident)
        if not spec.get("present", True):
            return (None if row is None else f"{_describe(check)}: still present"), m
        if row is None:
            return f"{_describe(check)}: absent", m
        if "mode" in spec and row.get("mode") != spec["mode"]:
            return f"{_describe(check)}: mode is {row.get('mode')!r}", m
        return None, m
    if kind == "differ":
        ids = spec["instances"]
        missing = [i for i in ids if i not in after.nodes]
        if missing:
            return f"differ: nodes missing {missing}", m
        vals = [_resolve(after.effective(i), _split(spec["parameter"])) for i in ids]
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if vals[a][0] and vals[b][0] and values_equal(vals[a][1], vals[b][1]):
                    return f"differ: {ids[a]} and {ids[b]} have the same {spec['parameter']}", m
                if not (vals[a][0] and vals[b][0]):
                    return f"differ: {spec['parameter']} unset on {ids[a] if not vals[a][0] else ids[b]}", m
        return None, m
    if kind == "subtree_order":
        root = spec.get("root")
        if root is None:  # root_node_id: the node with that node_id, preferring a new one
            want = spec["root_node_id"]
            want = want if isinstance(want, list) else [want]
            roots = sorted((i for i, n in after.nodes.items() if n.get("node_id") in want),
                           key=lambda i: i in before.nodes)
            root = roots[0] if roots else None
        if root not in after.nodes:
            return f"subtree_order: {root or spec.get('root_node_id')} absent", m
        order = after.preorder(root)

        def at(entry: Any) -> int:
            if isinstance(entry, str):
                return next((order.index(i) for i in aliases(before, after, entry) if i in order), -1)
            return next((k for k, i in enumerate(order) if node_mismatch(entry, after, i) is None), -1)
        pos = [at(e) for e in spec["order"]]
        if -1 in pos or pos != sorted(pos):
            return f"subtree_order: {spec['order']} not in this order under {root}", m
        return None, m
    raise ValueError(f"unknown check kind {kind}")


def evaluate_all(checks: list[dict], before: Draft, after: Draft) -> tuple[str | None, Match]:
    total = Match()
    for check in checks or []:
        fail, m = evaluate(check, before, after)
        if fail is not None:
            return fail, total
        total.merge(m)
    return None, total


# --- unchanged ------------------------------------------------------------------
def aliases(before: Draft, after: Draft, iid: str) -> list[str]:
    """``iid`` itself, or, when it was removed, the new nodes with identical content
    (node_id, parameters, value_sources, children recursively; only instance ids differ)."""
    if iid in after.nodes or iid not in before.nodes:
        return [iid]
    shape = _shape(before.nodes[iid])
    return [i for i in after.nodes if i not in before.nodes and values_equal(_shape(after.nodes[i]), shape)]


def _shape(node: dict) -> dict:
    """A subtree without instance ids: what the node does, not what it is called."""
    return {**{k: v for k, v in node.items() if k not in ("instance_id", "children")},
            "children": [_shape(c) for c in node.get("children") or []]}


def renamed_equivalents(before: Draft, after: Draft, m: Match) -> dict[str, str]:
    """Removed nodes whose identical subtree (all content, children recursively; only
    instance ids differ) now sits at a location a check matched: inside or as a new
    matched node. The operator sees the same Program; the rename is recorded."""
    added_m = {i for i in m.nodes if i not in before.nodes}
    copies = [i for i in after.nodes if i not in before.nodes
              and (i in m.nodes or set(after.ancestors(i)) & added_m)]
    out: dict[str, str] = {}
    for iid in before.nodes:
        if iid in after.nodes or iid in m.nodes:
            continue
        shape = _shape(before.nodes[iid])
        hit = next((c for c in copies if c not in out.values() and values_equal(_shape(after.nodes[c]), shape)), None)
        if hit:
            out[iid] = hit
    return out


def collateral(before: Draft, after: Draft, m: Match, renamed: dict[str, str] | None = None) -> list[str]:
    """Draft differences outside the matched entities. ``renamed`` (filled when given)
    receives removed nodes accepted as renamed equivalents."""
    out: list[str] = []
    added_m = {i for i in m.nodes if i not in before.nodes}
    removed_m = {i for i in m.nodes if i not in after.nodes}
    equiv = renamed_equivalents(before, after, m)
    if renamed is not None:
        renamed.update(equiv)

    for iid in list(before.nodes) + [i for i in after.nodes if i not in before.nodes]:
        b, a = iid in before.nodes, iid in after.nodes
        if iid in m.nodes:
            covered = m.nodes[iid]
            if covered is None or not (a and b):
                continue
            # a declared removable parameter may be gone (removed, not changed)
            gone = {r for r in m.removable.get(iid, ()) if not _resolve(after.content(iid), _split(r))[0]}
            stray = [p for p in diff_paths(before.content(iid), after.content(iid))
                     if not any(p == c or p.startswith(c + ".") or c.startswith(p + ".") for c in covered | gone)]
            if stray:
                out.append(f"node {iid} changed outside the check: {stray[0]}")
            continue
        if b and not a:
            if iid in equiv or set(before.ancestors(iid)) & set(equiv):
                continue
            if not set(before.ancestors(iid)) & removed_m:
                out.append(f"node {iid} ({before.nodes[iid].get('node_id')}) removed")
            continue
        if a and not b:
            if not set(after.ancestors(iid)) & added_m:
                out.append(f"node {iid} ({after.nodes[iid].get('node_id')}) added under {after.parent[iid]}")
            continue
        paths = diff_paths(before.content(iid), after.content(iid))
        if paths:
            out.append(f"node {iid} changed: {paths[0]}")
        bp, ap = before.parent[iid], after.parent[iid]
        if bp != ap and ap not in added_m and bp not in removed_m:
            out.append(f"node {iid} moved from {bp} to {ap}")
    for pid in set(before.children) | set(after.children):
        if (pid in m.nodes and m.nodes[pid] is None) or (pid is not None and (pid not in before.nodes or pid not in after.nodes)):
            continue
        keep = lambda d: [i for i in d.children.get(pid, []) if i not in m.nodes
                          and before.parent.get(i, "<none>") == after.parent.get(i, "<none>") == pid]
        if keep(before) != keep(after):
            out.append(f"children of {pid or 'the top level'} reordered: {keep(before)} -> {keep(after)}")

    # a new namespace that holds a matched new key ("ns.key") belongs to that key
    new_keys = [k for k in m.keys if k not in before.keys and isinstance(k, str)]
    ns_matched = {n for n in after.namespaces if n not in before.namespaces and isinstance(n, str)
                  and any(k.startswith(n + ".") for k in new_keys)}
    for label, bt, at, matched in (("scene resource", before.scene, after.scene, m.scene),
                                   ("parameter", before.params, after.params, m.params),
                                   ("blackboard key", before.keys, after.keys, m.keys),
                                   ("blackboard namespace", before.namespaces, after.namespaces, ns_matched),
                                   ("contact policy", before.policies, after.policies, m.policies),
                                   ("program field", before.meta, after.meta, set())):
        for ident in sorted(set(bt) | set(at), key=str):
            if ident in matched:
                continue
            if ident not in at:
                out.append(f"{label} {ident} removed")
            elif ident not in bt:
                out.append(f"{label} {ident} added")
            elif not values_equal(bt[ident], at[ident]):
                out.append(f"{label} {ident} changed: {diff_paths(bt[ident], at[ident])[:1]}")
    return out


def _lcs(a: list[str], b: list[str]) -> set[str]:
    """Members of a longest common subsequence of two id lists."""
    t = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            t[i][j] = t[i + 1][j + 1] + 1 if a[i] == b[j] else max(t[i + 1][j], t[i][j + 1])
    out, i, j = set(), 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.add(a[i])
            i, j = i + 1, j + 1
        elif t[i + 1][j] >= t[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def reordered(before: Draft, after: Draft) -> dict[str, list[str]]:
    """Nodes that changed position among siblings that stayed under the same parent: the
    fewest nodes whose moves explain the new order, each with every sibling whose index
    changed (a swap has no single "moved" node; naming any of them describes it)."""
    out: dict[str, list[str]] = {}
    for pid, kids in after.children.items():
        if pid is not None and pid not in before.nodes:
            continue
        same = lambda d: [i for i in d.children.get(pid, [])
                          if before.parent.get(i, "<none>") == after.parent.get(i, "<none>") == pid]
        b, a = same(before), same(after)
        if b != a:
            shifted = [i for k, i in enumerate(a) if b[k] != i]
            for i in set(a) - _lcs(b, a):
                out[i] = shifted
    return out


def changes(before: Draft, after: Draft) -> list[dict[str, Any]]:
    """Primary changes (what the answer must name): top of added/removed subtrees,
    nodes edited or moved, and scene resources, parameters and keys."""
    out = []
    moved = reordered(before, after)
    for iid in after.nodes:
        if iid not in before.nodes:
            if after.parent[iid] in before.nodes or after.parent[iid] is None:
                out.append({"kind": "node", "change": "added", "id": iid, "node_id": after.nodes[iid].get("node_id")})
        elif before.parent[iid] != after.parent[iid] or not values_equal(before.content(iid), after.content(iid)):
            out.append({"kind": "node", "change": "updated", "id": iid, "node_id": after.nodes[iid].get("node_id")})
        elif iid in moved:
            out.append({"kind": "node", "change": "moved", "id": iid, "node_id": after.nodes[iid].get("node_id"),
                        "shifted": moved[iid]})
    for iid in before.nodes:
        if iid not in after.nodes and (before.parent[iid] is None or before.parent[iid] in after.nodes):
            out.append({"kind": "node", "change": "removed", "id": iid, "node_id": before.nodes[iid].get("node_id")})
    for kind, bt, at in (("scene_resource", before.scene, after.scene), ("parameter", before.params, after.params),
                         ("blackboard_key", before.keys, after.keys),
                         ("contact_policy", before.policies, after.policies)):
        for ident in sorted(set(bt) | set(at), key=str):
            if ident not in at:
                out.append({"kind": kind, "change": "removed", "id": ident, "row": bt[ident]})
            elif ident not in bt or not values_equal(bt[ident], at[ident]):
                out.append({"kind": kind, "change": "added" if ident not in bt else "updated", "id": ident,
                            "row": at[ident]})
    return out


def draft_changed(before: Draft, after: Draft) -> bool:
    return bool(changes(before, after)) or bool(collateral(before, after, Match()))


# --- diagnostics -----------------------------------------------------------------
_DIAG_PATH = re.compile(r"^forest\[(\d+)\]((?:\.children\[\d+\])*)(.*)$")


def _diag_key(d: dict, draft: Draft) -> tuple:
    path = d.get("path") or ""
    m = _DIAG_PATH.match(path)
    anchor, rest = None, path
    if m:
        nodes = draft.forest
        idx = [int(m.group(1))] + [int(x) for x in re.findall(r"\d+", m.group(2))]
        node = None
        for i in idx:
            if not isinstance(nodes, list) or i >= len(nodes):
                node = None
                break
            node = nodes[i]
            nodes = node.get("children") or []
        if node is not None:
            anchor, rest = node.get("instance_id"), m.group(3)
    return d.get("code"), anchor, rest


def new_error_diagnostics(before: Draft, after: Draft) -> list[dict]:
    errs = lambda d: [x for x in d.diagnostics if x.get("severity", "error") == "error"]
    seen: dict[tuple, int] = {}
    for x in errs(before):
        k = _diag_key(x, before)
        seen[k] = seen.get(k, 0) + 1
    out = []
    for x in errs(after):
        k = _diag_key(x, after)
        if seen.get(k):
            seen[k] -= 1
        else:
            out.append(x)
    return out


# --- answer ----------------------------------------------------------------------
_VERBS = (r"(?:added|inserted|created|removed|deleted|updated|changed|moved|wrapped|replaced|renamed|modified|"
          r"set|placed|reordered|swapped|increased|decreased|reduced|lowered|raised|doubled|halved)")
# A claim of the assistant's own change: first person ("I added"), passive ("was added",
# "has been set") or a sentence/bullet starting with the verb ("Added a timer").
# Not "the resources you added", which reports the operator's earlier edits.
_CHANGE = re.compile(r"\b(?:i|i've|we|we've)\b[^.]*?\b" + _VERBS + r"\b"
                     r"|\b(?:has|have|was|were|been|is|are)\s+(?:now\s+|also\s+)?" + _VERBS + r"\b"
                     r"|^[\s*•\-✅]*" + _VERBS + r"\b", re.I)
_NEG = re.compile(r"(\b(not|no|never|unable|cannot|can't|couldn't|without|would|could|should|will|shall|if|"
                  r"nothing|none|want)\b|n't\b)", re.I)
_EDIT = r"(?:add|insert|create|remove|delete|update|change|move|wrap|replace|set|apply|modify|place|make|edit)"
# Only the assistant's own request-level failure counts, never node or Program behaviour
# ("the node fails", "if the first attempt fails, the retry runs it again").
_FAIL = re.compile(
    r"\b(?:i|we)\b[^.]*?\b(?:couldn't|could not|cannot|can't|was unable|were unable|wasn't able|was not able|"
    r"failed|did not manage|didn't manage)\b"
    r"|\b(?:the|my|this|that|your)\s+(?:edit|edits|change|changes|request|update|insert|insertion|mutation|"
    r"operation|tool call|call)\s+(?:has\s+|have\s+)?(?:failed|did not succeed|was rejected|was refused)\b"
    r"|\b(?:failed|unable|not able) to " + _EDIT + r"\b"
    r"|\bno changes? (?:were|was|have been|has been) (?:made|applied)\b", re.I)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def claims_change(answer: str) -> bool:
    return any(_CHANGE.search(s) and not _NEG.search(s) and not s.endswith("?") for s in _sentences(answer))


def claims_failure(answer: str) -> bool:
    return not claims_change(answer) and any(_FAIL.search(s) and not s.endswith("?") for s in _sentences(answer))


def is_question(answer: str) -> bool:
    return "?" in (answer or "")


_SHAPE_WORDS = {"box": ["box", "cube", "cuboid", "block"], "sphere": ["sphere", "ball"],
                "cylinder": ["cylinder"], "mesh": ["mesh"]}


def _norm(text: str) -> str:
    """Lower case words: underscores, dots, hyphens (any Unicode dash), markdown and punctuation read as spaces."""
    return " " + " ".join(re.findall(r"[^\W_]+", (text or "").lower())) + " "


def _names(change: dict) -> list[str]:
    """What may name a changed entity: its id, its node_id or a part of it (e.g. "move_to_pose",
    "timer"), or the scene resource's shape type (with plain-word synonyms such as "cube")."""
    words = [change["id"], *change.get("shifted", [])]
    if change.get("node_id"):
        nid = change["node_id"]
        words += [nid, *nid.split(".")]
    row = change.get("row") or {}
    shape = row.get("shape") if isinstance(row, dict) else None
    if isinstance(shape, dict) and shape.get("type"):
        words += _SHAPE_WORDS.get(shape["type"], [shape["type"]])
    return [_norm(w).strip() for w in words if isinstance(w, str) and len(w) >= 3]


def answer_failures(answer: str, before: Draft, after: Draft, spec: dict, checks_hold: bool) -> list[str]:
    spec = spec or {}
    out = []
    text = (answer or "").strip()
    if not text:
        return ["answer: empty"]
    low = text.lower()
    for w in spec.get("must_contain") or []:
        if str(w).lower() not in low:
            out.append(f"answer: must contain {w!r}")
    if not spec.get("consistent", True):
        return out
    ch = changes(before, after)
    if not ch and claims_change(text):
        out.append("answer: claims a change but the Draft is unchanged")
    if ch and checks_hold and claims_failure(text):
        out.append("answer: claims failure although the request was fulfilled")
    norm = _norm(text)
    for c in ch:
        if not any(f" {n} " in norm for n in _names(c)):
            out.append(f"answer: does not name the {c['change']} {c['kind']} {c['id']}")
    out += location_claim_failures(text, after, [c for c in ch if c["kind"] == "node" and c["change"] == "added"])
    return out


_PLACE = re.compile(r"\b(after|before)\s+(?:the\s+)?[`'\"]?([A-Za-z_][\w.]*)")


def location_claim_failures(text: str, after: Draft, added: list[dict]) -> list[str]:
    """With exactly one added node: an answer saying it went "after X" / "before X" for a
    sibling X under the same parent must match the Draft. Other wording is not judged."""
    if len(added) != 1:
        return []
    iid = added[0]["id"]
    pid = after.parent.get(iid)
    sibs = after.children.get(pid, [])
    pos = sibs.index(iid)
    for word, ref in _PLACE.findall(text):
        ref = ref.rstrip(".")
        if ref == iid or ref not in after.nodes or after.parent.get(ref) != pid:
            continue
        actual = (sibs[pos - 1] if pos > 0 else None) if word.lower() == "after" else (
            sibs[pos + 1] if pos + 1 < len(sibs) else None)
        if actual != ref:
            return [f"answer: says {iid} is {word.lower()} {ref}, but the Draft has it {word.lower()} {actual}"]
    return []


# --- one judgement -----------------------------------------------------------------
def judge(outcome: dict, before_snap: dict, after_snap: dict, answer: str,
          notes: dict | None = None) -> list[str]:
    """Failures, in check order: draft checks, unchanged, diagnostics, answer.
    ``notes`` (when given) receives ``renamed_equivalent``: {removed id: identical copy id}."""
    before, after = Draft(before_snap), Draft(after_snap)
    fails: list[str] = []
    fail, matched = evaluate_all(outcome.get("draft") or [], before, after)
    if fail:
        fails.append("draft: " + fail)
    elif outcome.get("unchanged", True):
        glue = set(outcome.get("allow_new_node_ids") or [])
        for iid, n in after.nodes.items():
            if iid not in before.nodes and n.get("node_id") in glue:
                matched.add_node(iid, None)
        renamed: dict[str, str] = {}
        fails += ["unchanged: " + c for c in collateral(before, after, matched, renamed)]
        if renamed and notes is not None:
            notes["renamed_equivalent"] = renamed
    if outcome.get("diagnostics", "no_new_errors") == "no_new_errors":
        fails += [f"diagnostics: new {d.get('code')} at {d.get('path')}" for d in new_error_diagnostics(before, after)]
    fails += answer_failures(answer, before, after, outcome.get("answer") or {}, fail is None)
    return fails


# --- conversation flow (clarify modes) -------------------------------------------------
def run_conversation(case: dict, before_snap: dict,
                     submit: Callable[[str, int], dict], read: Callable[[], dict]) -> dict[str, Any]:
    """Drive one case's turns and judge it.

    ``submit(message, turn)`` returns ``{"state", "answer", "needs_clarification"}``;
    ``read()`` returns the Draft snapshot. Status: PASS | PASS_CLARIFIED | FAIL.
    """
    outcome = case.get("outcome") or {}
    mode = outcome.get("clarify", "none")
    reply = outcome.get("clarification_answer")
    turn = submit(case["prompt"], 1)
    after = read()
    res: dict[str, Any] = {"clarify": mode, "asked": False, "clarified": False, "after": after}
    if turn.get("state") != "completed":
        return {**res, "status": "FAIL", "failures": [f"request {turn.get('state')}"], "answer": turn.get("answer", "")}
    changed = draft_changed(Draft(before_snap), Draft(after))
    asked = bool(turn.get("needs_clarification")) or (not changed and is_question(turn.get("answer", "")))
    res["asked"] = asked
    answer = turn.get("answer", "")
    if mode == "required":
        if not asked:
            return {**res, "status": "FAIL", "answer": answer,
                    "failures": ["clarify: required, but the model did not ask"]}
        if changed:
            return {**res, "status": "FAIL", "answer": answer,
                    "failures": ["clarify: required, but the Draft changed before the answer"]}
        if reply is None:
            fails = [] if answer.strip() else ["answer: empty"]
            return {**res, "status": "FAIL" if fails else "PASS", "failures": fails, "answer": answer}
    elif asked and not judge(outcome, before_snap, after, answer, notes := {}):
        # The request is already fulfilled: a trailing question or offer is normal.
        return {**res, **notes, "status": "PASS", "failures": [], "answer": answer}
    elif asked and reply is None:
        if mode == "allowed" and not changed:
            return {**res, "status": "PASS", "failures": [], "answer": answer}
        # the request is not fulfilled (checked above): report why first, then that it asked
        return {**res, "status": "FAIL", "answer": answer,
                "failures": judge(outcome, before_snap, after, answer)
                + ["clarify: the model asked instead of acting (no clarification_answer)"]}
    if asked:
        turn = submit(reply, 2)
        after = read()
        answer = turn.get("answer", "")
        res.update(clarified=True, after=after)
        if turn.get("state") != "completed":
            return {**res, "status": "FAIL", "failures": [f"clarification turn {turn.get('state')}"], "answer": answer}
    notes: dict = {}
    fails = judge(outcome, before_snap, after, answer, notes)
    res.update(notes)
    # PASS_CLARIFIED only for clarify: none (passed only after asking); for required and
    # allowed, asking is an expected path and the result is a plain PASS (``clarified`` records it).
    status = "FAIL" if fails else ("PASS_CLARIFIED" if res["clarified"] and mode == "none" else "PASS")
    return {**res, "status": status, "failures": fails, "answer": answer}


# --- chain repair --------------------------------------------------------------------
class RepairUnsupported(Exception):
    pass


def _nest(dotted: dict) -> dict:
    out: dict = {}
    for path, value in dotted.items():
        if isinstance(value, dict) and value and set(value) <= set(COMPARATORS):
            raise RepairUnsupported(f"bound {value} has no single value")
        segs = [str(s) for s in _split(path)]
        if segs[-1] in ("orientation_xyzw", "orientation_rpy_deg"):
            segs = segs[:-1] + ["orientation", segs[-1][len("orientation_"):]]
        t = out
        for s in segs[:-1]:
            t = t.setdefault(s, {})
        t[segs[-1]] = value
    return out


def _deep_merge(base: dict, patch: dict) -> dict:
    import copy
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not _is_orientation(v):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def restore_mutations(target_snap: dict, current_snap: dict) -> list[dict]:
    """Mutations that bring the current Draft's tree, scene, parameters, keys and contact policies back to
    target (one pass; extra top-level roots are handled by the caller re-reading)."""
    tgt, cur = Draft(target_snap), Draft(current_snap)
    muts: list[dict] = []
    if not values_equal(tgt.forest, cur.forest):
        if len(cur.forest) > 1:
            return [{"tool": "tree_draft_delete_node", "arguments": {"instance_id": cur.forest[-1]["instance_id"]}}]
        if tgt.forest and cur.forest:
            muts.append({"tool": "tree_draft_replace_subtree",
                         "arguments": {"instance_id": cur.forest[0]["instance_id"], "subtree": tgt.forest[0]}})
    for tool_rm, arg_rm, tool_up, arg_up, bt, ct in (
            ("scene_draft_remove_objects", "scene_resource_ids", "scene_draft_upsert_objects", "scene_resources",
             tgt.scene, cur.scene),
            ("program_draft_remove_parameters", "parameter_ids", "program_draft_upsert_parameters", "parameters",
             tgt.params, cur.params),
            ("blackboard_draft_remove_keys", "keys", "blackboard_draft_upsert_keys", "keys", tgt.keys, cur.keys),
            ("program_draft_remove_contact_policies", "contact_policy_ids", "program_draft_upsert_contact_policies",
             "contact_policies", tgt.policies, cur.policies)):
        extra = sorted(i for i in ct if i not in bt)
        if extra:
            muts.append({"tool": tool_rm, "arguments": {arg_rm: extra}})
        rows = [bt[i] for i in sorted(bt) if i not in ct or not values_equal(bt[i], ct[i])]
        if rows:
            muts.append({"tool": tool_up, "arguments": {arg_up: rows}})
    return muts


_CHILD_REF = re.compile(r"^@child:([^:]+):(-?\d+)$")


def resolve_refs(value: Any, draft: Draft) -> Any:
    """``"@child:<parent>:<index>"`` in an explicit repair names the instance at that
    position (the model chooses its own instance ids in earlier chain steps)."""
    if isinstance(value, dict):
        return {k: resolve_refs(v, draft) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_refs(v, draft) for v in value]
    m = _CHILD_REF.match(value) if isinstance(value, str) else None
    if m:
        kids = draft.children.get(m.group(1), [])
        i = int(m.group(2))
        if not -len(kids) <= i < len(kids):
            raise RepairUnsupported(f"{value}: {m.group(1)} has {len(kids)} children")
        return kids[i]
    return value


def repair_mutations(case: dict, snap: dict) -> list[dict]:
    """Mutations that apply a chain step's expected result to the Draft it started from.
    ``outcome.repair`` (fixture form) wins; otherwise they are derived from the checks
    (first ``any_of`` alternative). Raises RepairUnsupported for forms with no single result."""
    outcome = case.get("outcome") or {}
    if "repair" in outcome:
        return [{**m, "arguments": resolve_refs(m.get("arguments") or {}, Draft(snap))} for m in outcome["repair"]]
    draft = Draft(snap)
    slug = re.sub(r"[^a-z0-9]+", "_", case["id"].lower()).strip("_")
    muts: list[dict] = []
    seq = [0]

    def new_id(prefix: str) -> str:
        seq[0] += 1
        return f"repair_{slug}_{seq[0]}"

    def flat(checks: list[dict]) -> list[dict]:
        out = []
        for c in checks:
            if "any_of" in c:
                out += flat(c["any_of"][0])
            else:
                out.append(c)
        return out

    for check in flat(outcome.get("draft") or []):
        kind, spec = next(iter(check.items()))
        fail, _ = evaluate(check, draft, draft)
        if fail is None:
            continue
        if kind == "node":
            params = _nest({**(spec.get("parameters") or {}), **(spec.get("effective_parameters") or {})})
            nid = spec.get("node_id")
            nid = nid[0] if isinstance(nid, list) else nid
            iid = spec.get("instance_id")
            if iid in draft.nodes:
                if params:
                    muts.append({"tool": "tree_draft_update_node_params",
                                 "arguments": {"instance_id": iid, "expected_node_id": draft.nodes[iid]["node_id"],
                                               "parameters": params, "merge": True}})
                # A location check on an existing node is usually met by an earlier repair
                # mutation (wrap, insert); the verification after the repair reports it if not.
                continue
            if nid is None:
                raise RepairUnsupported(f"node check without node_id: {spec}")
            node = {"node_id": nid, "instance_id": iid or new_id(nid), "parameters": params, "children": []}
            if "parent" in spec and spec["parent"] is None:
                kids = spec.get("children_node_ids") or []
                if len(draft.forest) == 1 and kids == [draft.forest[0].get("node_id")]:
                    wrapper = {k: v for k, v in node.items() if k != "children"}
                    muts.append({"tool": "tree_draft_wrap_node",
                                 "arguments": {"target_instance_id": draft.forest[0]["instance_id"], "wrapper": wrapper}})
                    continue
                raise RepairUnsupported("top-level node other than a root wrap")
            pid = spec.get("parent")
            if pid is None and "parent_node_id" in spec:
                pid = next((i for i, n in draft.nodes.items() if n.get("node_id") == spec["parent_node_id"]), None)
            if pid is None or pid not in draft.nodes:
                raise RepairUnsupported(f"node check without an existing parent: {spec}")
            kids = draft.children.get(pid, [])

            def find(ref: str) -> int:
                for k, c in enumerate(kids):
                    if _sibling_ref(draft, c, ref):
                        return k
                raise RepairUnsupported(f"sibling {ref} not under {pid}")
            if isinstance(spec.get("index"), int):
                index = spec["index"] if spec["index"] >= 0 else len(kids) + spec["index"] + 1
            elif "after" in spec:
                index = find(spec["after"]) + 1
            elif "before" in spec:
                index = find(spec["before"])
            else:
                index = len(kids)
            muts.append({"tool": "tree_draft_insert_node",
                         "arguments": {"parent_instance_id": pid, "index": index, "node": node}})
            for k, child in enumerate(spec.get("children_node_ids") or []):
                child = child[0] if isinstance(child, list) else child
                muts.append({"tool": "tree_draft_insert_node",
                             "arguments": {"parent_instance_id": node["instance_id"], "index": k,
                                           "node": {"node_id": child, "instance_id": new_id(child), "parameters": {},
                                                    "children": []}}})
        elif kind == "node_absent":
            if "instance_id" not in spec:
                raise RepairUnsupported("node_absent by node_id")
            muts.append({"tool": "tree_draft_delete_node", "arguments": {"instance_id": spec["instance_id"]}})
        elif kind == "scene_resource":
            ident = spec.get("id")
            if not spec.get("present", True):
                muts.append({"tool": "scene_draft_remove_objects", "arguments": {"scene_resource_ids": [ident]}})
                continue
            base = draft.scene.get(ident) or {
                "pose": {"frame_ref": {"kind": "cell_frame"}, "position_m": [0.0, 0.0, 0.0],
                         "orientation": {"rpy_deg": [0.0, 0.0, 0.0]}}}
            if "shape" not in spec and "shape" not in base:
                raise RepairUnsupported(f"new scene resource without shape: {spec}")
            row = _deep_merge(base, {k: spec[k] for k in ("shape", "pose") if k in spec})
            row["scene_resource_id"] = ident or new_id("scene")
            muts.append({"tool": "scene_draft_upsert_objects", "arguments": {"scene_resources": [row]}})
        elif kind in ("parameter", "blackboard_key") and "id" not in spec and "key" not in spec:
            raise RepairUnsupported(f"{kind} without a name")
        elif kind == "parameter":
            if not spec.get("present", True):
                muts.append({"tool": "program_draft_remove_parameters", "arguments": {"parameter_ids": [spec["id"]]}})
                continue
            row = {**(draft.params.get(spec["id"]) or {}), "parameter_id": spec["id"],
                   **{k: spec[k] for k in ("type", "default") if k in spec}}
            if isinstance(row.get("default"), dict) and set(row["default"]) <= set(COMPARATORS):
                raise RepairUnsupported("parameter default given as a bound")
            muts.append({"tool": "program_draft_upsert_parameters", "arguments": {"parameters": [row]}})
        elif kind == "contact_policy":
            if spec.get("present", True) or "id" not in spec:
                raise RepairUnsupported("a new contact policy needs its full shape: give outcome.repair")
            muts.append({"tool": "program_draft_remove_contact_policies",
                         "arguments": {"contact_policy_ids": [spec["id"]]}})
        elif kind == "blackboard_key":
            if not spec.get("present", True):
                muts.append({"tool": "blackboard_draft_remove_keys", "arguments": {"keys": [spec["key"]]}})
                continue
            row = {"access": "read_write", "persistence": "run", **(draft.keys.get(spec["key"]) or {}),
                   "key": spec["key"], **{k: spec[k] for k in ("type", "initial") if k in spec}}
            muts.append({"tool": "blackboard_draft_upsert_keys", "arguments": {"keys": [row]}})
        else:
            raise RepairUnsupported(f"{kind} has no single result")
    return muts


# --- corpus schema validator -------------------------------------------------------------
CASE_KEYS = {"id", "category", "detail_level", "intent", "prompt", "preconditions", "outcome",
             "original_prompt", "port_notes", "scenario", "status", "exclusion_reason"}
PRE_KEYS = {"chain_id", "chain_step", "fresh_program", "fixtures", "chain_timeout_s"}
OUTCOME_KEYS = {"clarify", "clarification_answer", "draft", "unchanged", "diagnostics", "answer", "repair",
                "allow_new_node_ids"}


def _validate_check(c: Any, where: str, out: list[str]) -> None:
    if not isinstance(c, dict) or len(c) != 1 or next(iter(c)) not in CHECK_KINDS:
        out.append(f"{where}: a check is one of {CHECK_KINDS}, got {str(c)[:120]}")
        return
    kind, spec = next(iter(c.items()))
    if kind == "any_of":
        if not isinstance(spec, list) or len(spec) < 2 or not all(isinstance(a, list) for a in spec):
            out.append(f"{where}: any_of needs >= 2 check lists (an empty list: nothing further required)")
            return
        for i, alt in enumerate(spec):
            for j, cc in enumerate(alt):
                _validate_check(cc, f"{where}.any_of[{i}][{j}]", out)
        return
    if not isinstance(spec, dict):
        out.append(f"{where}: {kind} needs a mapping")
        return
    allowed = {"node": set(NODE_FIELDS), "node_absent": {"instance_id", "node_id", "parent"},
               "scene_resource": {"id", "new", "present", "shape", "pose"},
               "parameter": {"id", "present", "type", "default"},
               "blackboard_key": {"key", "present", "type", "initial"},
               "contact_policy": {"id", "present", "mode"},
               "differ": {"instances", "parameter"}, "subtree_order": {"root", "root_node_id", "order"}}[kind]
    unknown = set(spec) - allowed
    if unknown:
        out.append(f"{where}: {kind} has unknown fields {sorted(unknown)}")
    if kind == "node":
        if not ({"instance_id", "node_id"} & set(spec)):
            out.append(f"{where}: node needs instance_id or node_id")
        if "parent" in spec and "parent_node_id" in spec:
            out.append(f"{where}: node has both parent and parent_node_id")
        for f in ("parameters", "effective_parameters"):
            if f in spec and not isinstance(spec[f], dict):
                out.append(f"{where}: node.{f} must be a mapping of dotted paths")
        if "children_node_ids" in spec and not (isinstance(spec["children_node_ids"], list) and all(
                isinstance(c, str) or (isinstance(c, list) and c and all(isinstance(x, str) for x in c))
                for c in spec["children_node_ids"])):
            out.append(f"{where}: children_node_ids items are node_ids or lists of node_ids")
        if "children_order" in spec and (spec["children_order"] != "any" or "children_node_ids" not in spec):
            out.append(f"{where}: children_order: any needs children_node_ids")
        if "may_remove" in spec and not (isinstance(spec["may_remove"], list) and spec["may_remove"]
                                          and all(isinstance(p, str) and p for p in spec["may_remove"])):
            out.append(f"{where}: may_remove is a non-empty list of parameter paths")
        if "index" in spec and not (isinstance(spec["index"], int) or spec["index"] == "end"):
            out.append(f"{where}: index must be an integer or 'end'")
    elif kind == "node_absent":
        if "instance_id" not in spec and not {"node_id", "parent"} <= set(spec):
            out.append(f"{where}: node_absent needs instance_id or node_id + parent")
    elif kind == "scene_resource":
        if spec.get("present", True) is False and not spec.get("id"):
            out.append(f"{where}: scene_resource present: false needs id")
    elif kind in ("parameter", "blackboard_key"):
        ident = "id" if kind == "parameter" else "key"
        if "present" not in spec:
            out.append(f"{where}: {kind} needs present")
        if ident not in spec and spec.get("present") is False:
            out.append(f"{where}: {kind} present: false needs {ident}")
    elif kind == "differ":
        if not isinstance(spec.get("instances"), list) or len(spec["instances"]) < 2 or "parameter" not in spec:
            out.append(f"{where}: differ needs instances (>= 2) and parameter")
    elif kind == "subtree_order":
        if ("root" in spec) == ("root_node_id" in spec) or not isinstance(spec.get("order"), list):
            out.append(f"{where}: subtree_order needs one of root/root_node_id, and order")


def validate_corpus(corpus: dict) -> list[str]:
    out: list[str] = []
    if corpus.get("schema_version") != "manyforge.assistant_smoke_corpus.v2":
        out.append(f"schema_version is {corpus.get('schema_version')!r}")
    seen: set[str] = set()
    chains: dict[str, list[int]] = {}
    for n, case in enumerate(corpus.get("cases") or []):
        cid = case.get("id", f"#{n}")
        w = f"{cid}"
        if cid in seen:
            out.append(f"{w}: duplicate id")
        seen.add(cid)
        unknown = set(case) - CASE_KEYS
        if unknown:
            out.append(f"{w}: unknown keys {sorted(unknown)}" + (" (v2 uses preconditions)" if "precondition" in unknown else ""))
        for k in ("id", "category", "intent", "prompt", "outcome"):
            if not case.get(k):
                out.append(f"{w}: missing {k}")
        pre = case.get("preconditions") or {}
        if set(pre) - PRE_KEYS:
            out.append(f"{w}: unknown preconditions {sorted(set(pre) - PRE_KEYS)}")
        for i, fx in enumerate(pre.get("fixtures") or []):
            if not isinstance(fx, dict) or set(fx) != {"tool", "arguments"}:
                out.append(f"{w}: fixtures[{i}] needs exactly tool and arguments")
        if pre.get("chain_id"):
            chains.setdefault(pre["chain_id"], []).append(pre.get("chain_step"))
            if pre.get("chain_step") == 1 and not pre.get("fresh_program"):
                out.append(f"{w}: chain step 1 without fresh_program")
        o = case.get("outcome") or {}
        if set(o) - OUTCOME_KEYS:
            out.append(f"{w}: unknown outcome keys {sorted(set(o) - OUTCOME_KEYS)}")
        if o.get("clarify", "none") not in ("none", "required", "allowed"):
            out.append(f"{w}: clarify must be none|required|allowed")
        if "clarification_answer" in o and not (isinstance(o["clarification_answer"], str) and o["clarification_answer"].strip()):
            out.append(f"{w}: clarification_answer must be non-empty text")
        if o.get("diagnostics", "no_new_errors") not in ("no_new_errors", "allow_incomplete"):
            out.append(f"{w}: diagnostics must be no_new_errors|allow_incomplete")
        if not isinstance(o.get("unchanged", True), bool):
            out.append(f"{w}: unchanged must be a boolean")
        a = o.get("answer", {})
        if not isinstance(a, dict) or set(a) - {"consistent", "must_contain"}:
            out.append(f"{w}: answer must be {{consistent, must_contain}}")
        draft = o.get("draft")
        if draft is not None and not isinstance(draft, list):
            out.append(f"{w}: outcome.draft must be a list")
        for j, c in enumerate(draft or []):
            _validate_check(c, f"{w}.draft[{j}]", out)
        if o.get("clarify") != "required" and not draft:
            out.append(f"{w}: no draft checks (only clarify: required may omit them)")
        for i, fx in enumerate(o.get("repair") or []):
            if not isinstance(fx, dict) or set(fx) != {"tool", "arguments"}:
                out.append(f"{w}: repair[{i}] needs exactly tool and arguments")
    for chain, steps in chains.items():
        if sorted(steps) != list(range(1, len(steps) + 1)) and sorted(steps)[0] != 1:
            out.append(f"chain {chain}: steps {sorted(steps)} do not start at 1")
        if len(set(steps)) != len(steps):
            out.append(f"chain {chain}: duplicate steps")
    return out
