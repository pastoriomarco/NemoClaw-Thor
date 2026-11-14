#!/usr/bin/env python3
"""Unit tests for outcome_scorer.py and the v2 runner's flow (plain python3, no ROS).

    python3 -m unittest test_outcome_scorer -v
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import outcome_scorer as s  # noqa: E402


def node(iid, nid, children=(), **params):
    return {"node_id": nid, "instance_id": iid, "parameters": params, "children": list(children)}


def program():
    motion = lambda iid, z: {**node(iid, "motion.move_to_pose", goal={"position_m": [0, 0, z],
                                                                      "orientation": {"rpy_deg": [180, 0, -180]}}),
                             "value_sources": {"velocity_scale": {"kind": "program_parameter",
                                                                  "reference": "velocity_scale"}}}
    root = node("pick_and_place", "sequence", [
        node("open_gripper", "gripper.command_position", position={"kind": "angular_rad", "value_rad": 0.0}),
        motion("approach", 0.1),
        motion("pick", 0.025),
        node("close_gripper", "gripper.command_position", position={"kind": "angular_rad", "value_rad": 0.66}),
    ])
    return {"authoring": {
        "name": "demo", "forest": [root], "positions": {},
        "parameters": [{"parameter_id": "velocity_scale", "type": "float64", "default": 0.5}],
        "scene": [{"scene_resource_id": "graspable", "shape": {"type": "box", "box_dimensions_m": [0.1, 0.01, 0.01]},
                   "pose": {"frame_ref": {"kind": "cell_frame"}, "position_m": [0.3, -0.3, 0.15],
                            "orientation": {"xyzw": [0, 0, -0.3826834323650898, 0.9238795325112867]}}}],
        "blackboard": {"keys": [], "namespaces": []}}, "diagnostics": []}


def kids(snap, iid="pick_and_place"):
    return s.Draft(snap).nodes[iid]["children"]


class FakeComposer:
    """Applies the Draft mutations the harness uses (setup, restore, repair), simply."""

    def __init__(self, snap):
        self.snap = copy.deepcopy(snap)
        self.log = []

    def draft(self):
        return copy.deepcopy(self.snap)

    def _find(self, iid):
        a = self.snap["authoring"]
        if iid == "@root":
            iid = a["forest"][0]["instance_id"]
        stack = [(a["forest"], None)]
        while stack:
            lst, parent = stack.pop()
            for i, n in enumerate(lst):
                if n["instance_id"] == iid:
                    return n, lst, i
                stack.append((n["children"], n))
        raise RuntimeError(f"no node {iid}")

    def mutate(self, tool, args):
        self.log.append((tool, copy.deepcopy(args)))
        a = self.snap["authoring"]
        if tool == "tree_draft_insert_node":
            n = {"children": [], "parameters": {}, **copy.deepcopy(args["node"])}
            self._find(args["parent_instance_id"])[0]["children"].insert(args["index"], n)
        elif tool == "tree_draft_wrap_node":
            target, lst, i = self._find(args["target_instance_id"])
            lst[i] = {**copy.deepcopy(args["wrapper"]), "children": [target]}
        elif tool == "tree_draft_replace_subtree":
            _, lst, i = self._find(args["instance_id"])
            lst[i] = copy.deepcopy(args["subtree"])
        elif tool == "tree_draft_delete_node":
            n, lst, i = self._find(args["instance_id"])
            lst.pop(i)
            if lst is a["forest"]:
                lst[i:i] = n["children"]
            else:
                a["forest"][len(a["forest"]):] = n["children"]
        elif tool == "tree_draft_update_node_params":
            n = self._find(args["instance_id"])[0]
            n["parameters"] = s._deep_merge(n["parameters"], args["parameters"])
        elif tool == "scene_draft_upsert_objects":
            by = {o["scene_resource_id"]: o for o in a["scene"]}
            for o in args["scene_resources"]:
                by[o["scene_resource_id"]] = copy.deepcopy(o)
            a["scene"] = [by[k] for k in sorted(by)]
        elif tool == "scene_draft_remove_objects":
            a["scene"] = [o for o in a["scene"] if o["scene_resource_id"] not in args["scene_resource_ids"]]
        elif tool == "program_draft_upsert_parameters":
            by = {p["parameter_id"]: p for p in a["parameters"]}
            by.update({p["parameter_id"]: copy.deepcopy(p) for p in args["parameters"]})
            a["parameters"] = list(by.values())
        elif tool == "program_draft_remove_parameters":
            a["parameters"] = [p for p in a["parameters"] if p["parameter_id"] not in args["parameter_ids"]]
        elif tool == "blackboard_draft_upsert_keys":
            by = {k["key"]: k for k in a["blackboard"]["keys"]}
            by.update({k["key"]: copy.deepcopy(k) for k in args["keys"]})
            a["blackboard"]["keys"] = list(by.values())
        elif tool == "blackboard_draft_remove_keys":
            a["blackboard"]["keys"] = [k for k in a["blackboard"]["keys"] if k["key"] not in args["keys"]]
        elif tool == "program_draft_upsert_contact_policies":
            by = {p["policy_id"]: p for p in a.get("contact_policies") or []}
            by.update({p["policy_id"]: copy.deepcopy(p) for p in args["contact_policies"]})
            a["contact_policies"] = list(by.values())
        elif tool == "program_draft_remove_contact_policies":
            a["contact_policies"] = [p for p in a.get("contact_policies") or []
                                     if p["policy_id"] not in args["contact_policy_ids"]]
        else:
            raise RuntimeError(f"unsupported {tool}")
        return {}


def insert(snap, parent, index, n):
    f = FakeComposer(snap)
    f.mutate("tree_draft_insert_node", {"parent_instance_id": parent, "index": index, "node": n})
    return f.draft()


class Values(unittest.TestCase):
    def test_tolerance_and_bool(self):
        self.assertTrue(s.matches(0.3, 0.3004))
        self.assertFalse(s.matches(0.3, 0.302))
        self.assertFalse(s.matches(1, True))
        self.assertTrue(s.matches([1.0, 2.0], [1.0005, 2]))

    def test_quaternion_sign_and_rpy(self):
        q = {"xyzw": [0, 0, -0.3826834, 0.9238795]}
        self.assertTrue(s.same_rotation(q, {"xyzw": [0, 0, 0.3826834, -0.9238795]}))
        self.assertTrue(s.same_rotation(q, {"rpy_deg": [0, 0, -45]}))
        self.assertFalse(s.same_rotation(q, {"rpy_deg": [0, 0, 45]}))
        self.assertTrue(s.same_rotation({"rpy_deg": [180, 0, -180]}, {"xyzw": [0, 1, 0, 0]}))
        pose = {"orientation": {"rpy_deg": [0, 0, -45]}}
        self.assertTrue(s.path_matches(pose, "orientation.xyzw", [0, 0, 0.3826834, -0.9238795]))
        self.assertTrue(s.path_matches(pose, "orientation_xyzw", [0, 0, -0.3826834, 0.9238795]))
        self.assertTrue(s.values_equal({"orientation": {"xyzw": [0, 0, 0, 1]}}, {"orientation": {"rpy_deg": [0, 0, 0]}}))

    def test_bounds(self):
        self.assertTrue(s.matches({"max_exclusive": 0.5}, 0.25))
        self.assertFalse(s.matches({"max_exclusive": 0.5}, 0.5))
        self.assertTrue(s.matches({"min_inclusive": 0.5, "max_inclusive": 1}, 0.5))
        self.assertFalse(s.matches({"max_exclusive": 0.5}, "x"))


class Checks(unittest.TestCase):
    def setUp(self):
        self.before = program()
        self.after = insert(self.before, "pick_and_place", 2,
                            node("wait1", "timer", duration_s=2.0))

    def run_checks(self, checks, after=None, **outcome):
        return s.judge({"draft": checks, **outcome}, self.before, after or self.after, "Added timer wait1.")

    def test_node_location_forms(self):
        for spec in ({"parent": "pick_and_place", "index": 2}, {"after": "approach"}, {"before": "pick"},
                     {"parent_node_id": "sequence"}, {"ancestor": "pick_and_place"},
                     {"node_id": ["timer", "fallback"]}, {"after": "motion.move_to_pose"}):
            self.assertEqual(self.run_checks([{"node": {"node_id": "timer", **spec, "parameters": {"duration_s": 2}}}]),
                             [], spec)

    def test_node_location_failures(self):
        f = self.run_checks([{"node": {"node_id": "timer", "last_child": True}}])
        self.assertTrue(f[0].startswith("draft: node{node_id=timer, last_child=True}: no match"), f)
        self.assertIn("last_child", f[0])
        self.assertTrue(self.run_checks([{"node": {"node_id": "timer", "parameters": {"duration_s": 3}}}]))
        self.assertTrue(self.run_checks([{"node": {"node_id": "timer", "parent": None}}]))

    def test_children_node_ids_and_order(self):
        par = node("par", "parallel", [node("g", "gripper.check_status"), node("o", "gripper.command_position")])
        after = insert(self.before, "pick_and_place", 4, par)
        ordered = {"node": {"node_id": "parallel", "children_node_ids": ["gripper.command_position", "gripper.check_status"]}}
        self.assertTrue(self.run_checks([ordered], after))
        alts = {"node": {"node_id": "parallel", "children_node_ids": [
            ["gripper.check_status", "gripper.wait_for_status"], "gripper.command_position"]}}
        self.assertFalse([f for f in self.run_checks([alts], after) if f.startswith(("draft", "unchanged"))])
        alts["node"]["children_node_ids"].reverse()
        self.assertTrue(self.run_checks([alts], after))
        alts["node"]["children_order"] = "any"
        self.assertFalse([f for f in self.run_checks([alts], after) if f.startswith(("draft", "unchanged"))])
        ordered["node"]["children_order"] = "any"
        self.assertFalse([f for f in self.run_checks([ordered], after) if f.startswith(("draft", "unchanged"))])

    def test_effective_parameters_and_differ(self):
        f = FakeComposer(self.before)
        f.mutate("tree_draft_update_node_params", {"instance_id": "approach", "parameters": {"velocity_scale": 0.3}})
        after = f.draft()
        s.Draft(after).nodes["approach"].pop("value_sources")
        eff = {"node": {"instance_id": "approach", "effective_parameters": {"velocity_scale": 0.3}}}
        self.assertEqual([x for x in self.run_checks([eff], after) if not x.startswith("answer")], [])
        self.assertEqual(s.Draft(self.before).effective("pick")["velocity_scale"], 0.5)
        self.assertEqual(s.evaluate({"differ": {"instances": ["approach", "pick"], "parameter": "goal"}},
                                    s.Draft(self.before), s.Draft(self.before))[0], None)
        self.assertIsNotNone(s.evaluate({"differ": {"instances": ["open_gripper", "close_gripper"], "parameter": "goal"}},
                                        s.Draft(self.before), s.Draft(self.before))[0])

    def test_node_absent_and_subtree_order(self):
        f = FakeComposer(self.before)
        f.mutate("tree_draft_delete_node", {"instance_id": "open_gripper"})
        after = f.draft()
        self.assertEqual(self.run_checks([{"node_absent": {"instance_id": "open_gripper"}}], after,
                                         answer={"consistent": False}), [])
        self.assertTrue(self.run_checks([{"node_absent": {"instance_id": "pick"}}], after))
        self.assertEqual(self.run_checks([{"node_absent": {"node_id": "gripper.command_position", "parent": "pick_and_place"}}],
                                         after, answer={"consistent": False})[0][:6], "draft:")  # close_gripper still there
        order = {"subtree_order": {"root": "pick_and_place", "order": ["approach", "pick", "close_gripper"]}}
        self.assertIsNone(s.evaluate(order, s.Draft(after), s.Draft(after))[0])
        order["subtree_order"]["order"].reverse()
        self.assertIsNotNone(s.evaluate(order, s.Draft(after), s.Draft(after))[0])

    def test_scene_parameter_key(self):
        f = FakeComposer(self.before)
        f.mutate("scene_draft_upsert_objects", {"scene_resources": [
            {"scene_resource_id": "box1", "shape": {"type": "box", "box_dimensions_m": [1, 0.02, 0.25]},
             "pose": {"frame_ref": {"kind": "cell_frame"}, "position_m": [0, -0.15, 0.125],
                      "orientation": {"xyzw": [0, 0, 0, -1]}}}]})
        f.mutate("program_draft_upsert_parameters", {"parameters": [{"parameter_id": "speed", "type": "float64", "default": 0.2}]})
        f.mutate("blackboard_draft_upsert_keys", {"keys": [{"key": "target_pose", "type": "stored_pose"}]})
        after = f.draft()
        checks = [{"scene_resource": {"new": True, "shape": {"type": "box", "box_dimensions_m": [1.0, 0.02, 0.25]},
                                      "pose": {"position_m": [0, -0.15, 0.125], "orientation": {"rpy_deg": [0, 0, 0]}}}},
                  {"scene_resource": {"id": "graspable", "present": True}},
                  {"parameter": {"id": "speed", "present": True, "type": "float64", "default": 0.2}},
                  {"blackboard_key": {"key": "target_pose", "present": True, "type": "stored_pose"}}]
        self.assertEqual(s.judge({"draft": checks}, self.before, after, "Added box1, the speed parameter and target_pose."), [])
        self.assertTrue(s.judge({"draft": [{"scene_resource": {"id": "graspable", "new": True}}]}, self.before, after, "x"))
        self.assertTrue(s.judge({"draft": [{"scene_resource": {"id": "box1", "present": False}}]}, self.before, after, "x"))
        self.assertTrue(s.judge({"draft": [{"blackboard_key": {"key": "target_pose", "present": True, "type": "int64"}}]},
                                self.before, after, "x"))

    def test_contact_policy(self):
        f = FakeComposer(self.before)
        f.mutate("program_draft_upsert_contact_policies", {"contact_policies": [
            {"policy_id": "grip_part", "mode": "contact_grasp", "zones": []}]})
        after = f.draft()
        self.assertEqual(s.judge({"draft": [{"contact_policy": {"present": True, "mode": "contact_grasp"}}]},
                                 self.before, after, "Declared the grip_part contact policy."), [])
        self.assertEqual(s.judge({"draft": [{"contact_policy": {"id": "grip_part", "mode": "contact_grasp"}}]},
                                 self.before, after, "Declared grip_part."), [])
        # a new policy is a change: unchecked it is collateral, and the answer must name it
        self.assertIn("unchanged: contact policy grip_part added",
                      s.judge({"draft": []}, self.before, after, "Declared grip_part."))
        self.assertIn("answer: does not name the added contact_policy grip_part",
                      s.judge({"draft": [{"contact_policy": {"present": True}}]}, self.before, after, "Done, added it."))
        self.assertTrue(s.judge({"draft": [{"contact_policy": {"present": True, "mode": "contact_release"}}]},
                                self.before, after, "x"))
        self.assertTrue(s.judge({"draft": [{"contact_policy": {"id": "grip_part", "present": False}}]},
                                self.before, after, "x"))
        self.assertEqual(s.restore_mutations(self.before, after), [
            {"tool": "program_draft_remove_contact_policies", "arguments": {"contact_policy_ids": ["grip_part"]}}])

    def test_any_of(self):
        alt = {"any_of": [[{"node": {"node_id": "fallback"}}], [{"node": {"node_id": "timer", "index": 2}}]]}
        self.assertEqual(self.run_checks([alt]), [])
        bad = {"any_of": [[{"node": {"node_id": "fallback"}}], [{"node": {"node_id": "timer", "index": 0}}]]}
        f = self.run_checks([bad])
        self.assertTrue(f[0].startswith("draft: any_of: no alternative holds"), f)


class G4Forms(unittest.TestCase):
    def setUp(self):
        self.before = program()
        glue = node("seq", "sequence", [{**node("chk", "io.check_input", expected_value=True),
                                         "value_sources": {"condition": {"kind": "blackboard_key", "reference": "k"}}},
                                        node("alarm", "io.set_output", value=True)])
        self.after = insert(self.before, "pick_and_place", 4, node("fb", "fallback", [glue]))
        f = FakeComposer(self.after)
        f.mutate("blackboard_draft_upsert_keys", {"keys": [{"key": "chosen", "type": "stored_pose"}]})
        f.mutate("program_draft_upsert_parameters", {"parameters": [{"parameter_id": "p2", "type": "bool", "default": True}]})
        self.after = f.draft()

    def test_forms(self):
        checks = [{"node": {"node_id": "io.check_input", "ancestor_node_id": ["fallback", "retry"],
                            "value_sources": {"condition": {"kind": "blackboard_key"}}}},
                  {"node": {"node_id": "fallback", "parent": "pick_and_place"}},
                  {"subtree_order": {"root_node_id": "fallback", "order": [{"node_id": "io.check_input"}, "alarm"]}},
                  {"blackboard_key": {"present": True, "type": "stored_pose"}},
                  {"parameter": {"present": True, "type": "bool"}}]
        self.assertEqual(s.judge({"draft": checks, "answer": {"consistent": False}}, self.before, self.after, "ok"), [])
        bad = [{"subtree_order": {"root_node_id": "fallback", "order": ["alarm", {"node_id": "io.check_input"}]}}]
        self.assertTrue(s.judge({"draft": bad}, self.before, self.after, "ok"))
        self.assertTrue(s.judge({"draft": [{"blackboard_key": {"present": True, "type": "int64"}}]}, self.before, self.after, "x"))

    def test_allow_new_node_ids(self):
        checks = [{"node": {"node_id": "io.check_input", "ancestor_node_id": "fallback"}},
                  {"node": {"node_id": "io.set_output", "ancestor_node_id": "fallback"}},
                  {"blackboard_key": {"key": "chosen", "present": True}}, {"parameter": {"id": "p2", "present": True}}]
        out = {"draft": checks, "answer": {"consistent": False}}
        self.assertTrue(any("fallback" in f for f in s.judge(out, self.before, self.after, "ok")))
        self.assertEqual(s.judge({**out, "allow_new_node_ids": ["fallback", "sequence"]}, self.before, self.after, "ok"), [])

    def test_negative_index_path(self):
        self.assertTrue(s.path_matches({"path": [{"p": [1, 2]}, {"p": [3, 4]}]}, "path[-1].p", [3, 4]))
        self.assertTrue(s.path_matches({"goal": {"position_m": [0, 0, 0.2]}}, "goal.position_m[2]", 0.2))


class Unchanged(unittest.TestCase):
    def setUp(self):
        self.before = program()
        self.timer = [{"node": {"node_id": "timer", "parent": "pick_and_place"}}]

    def collateral(self, after, checks):
        fail, m = s.evaluate_all(checks, s.Draft(self.before), s.Draft(after))
        self.assertIsNone(fail)
        return s.collateral(s.Draft(self.before), s.Draft(after), m)

    def test_clean_insert(self):
        after = insert(self.before, "pick_and_place", 1, node("t", "timer", duration_s=1))
        self.assertEqual(self.collateral(after, self.timer), [])

    def test_collateral_param_edit_delete_reorder_scene(self):
        after = insert(self.before, "pick_and_place", 1, node("t", "timer", duration_s=1))
        f = FakeComposer(after)
        f.mutate("tree_draft_update_node_params", {"instance_id": "pick", "parameters": {"goal": {"position_m": [0, 0, 0.5]}}})
        self.assertIn("node pick changed: parameters.goal.position_m", self.collateral(f.draft(), self.timer))
        f = FakeComposer(after)
        f.mutate("tree_draft_delete_node", {"instance_id": "close_gripper"})
        self.assertIn("node close_gripper (gripper.command_position) removed", self.collateral(f.draft(), self.timer))
        swapped = copy.deepcopy(after)
        ch = swapped["authoring"]["forest"][0]["children"]
        ch[2], ch[3] = ch[3], ch[2]
        self.assertTrue(any("reordered" in c for c in self.collateral(swapped, self.timer)))
        f = FakeComposer(after)
        f.mutate("scene_draft_remove_objects", {"scene_resource_ids": ["graspable"]})
        self.assertIn("scene resource graspable removed", self.collateral(f.draft(), self.timer))
        f = FakeComposer(after)
        f.mutate("program_draft_upsert_parameters", {"parameters": [{"parameter_id": "velocity_scale", "type": "float64", "default": 0.2}]})
        self.assertTrue(any(c.startswith("parameter velocity_scale changed") for c in self.collateral(f.draft(), self.timer)))
        extra = insert(after, "pick_and_place", 0, node("t2", "timer"))
        self.assertTrue(any("added" in c for c in self.collateral(extra, [{"node": {"node_id": "timer", "instance_id": "t"}}])))

    def test_wrap_and_move_into_new_composite(self):
        f = FakeComposer(self.before)
        f.mutate("tree_draft_wrap_node", {"target_instance_id": "pick_and_place",
                                          "wrapper": {"node_id": "repeat", "instance_id": "rep", "parameters": {}}})
        checks = [{"node": {"node_id": "repeat", "parent": None, "children_node_ids": ["sequence"]}},
                  {"node": {"instance_id": "pick_and_place", "parent_node_id": "repeat"}}]
        self.assertEqual(self.collateral(f.draft(), checks), [])
        f = FakeComposer(self.before)
        f.mutate("tree_draft_wrap_node", {"target_instance_id": "pick",
                                          "wrapper": {"node_id": "retry", "instance_id": "r", "parameters": {"num_attempts": 3}}})
        self.assertEqual(self.collateral(f.draft(), [{"node": {"node_id": "retry", "after": "approach",
                                                               "children_node_ids": ["motion.move_to_pose"]}}]), [])
        # the moved node keeps its content unless a check covers it
        moved = f.draft()
        s.Draft(moved).nodes["pick"]["parameters"]["goal"]["position_m"] = [9, 9, 9]
        self.assertTrue(self.collateral(moved, [{"node": {"node_id": "retry"}},
                                                {"node": {"instance_id": "pick", "parent_node_id": "retry"}}]))

    def test_renamed_equivalent_copy_inside_new_composite(self):
        # FALLBACK_alternate_medium: fallback before close_gripper with a COPY of pick
        # (new id) plus an alternative, original deleted: equivalent to a wrap.
        pick = copy.deepcopy(s.Draft(self.before).nodes["pick"])
        alt = copy.deepcopy(pick)
        alt["instance_id"] = "alt_pick"
        alt["parameters"]["goal"]["position_m"] = [0.02, 0, 0.025]
        fb = node("fb", "fallback", [{**copy.deepcopy(pick), "instance_id": "pick_child"}, alt])
        f = FakeComposer(insert(self.before, "pick_and_place", 3, fb))
        f.mutate("tree_draft_delete_node", {"instance_id": "pick"})
        after = f.draft()
        out = {"draft": [{"node": {"node_id": "fallback", "before": "close_gripper",
                                   "children_node_ids": ["motion.move_to_pose", "motion.move_to_pose"]}}],
               "answer": {"consistent": False}}
        notes = {}
        self.assertEqual(s.judge(out, self.before, after, "ok", notes), [])
        self.assertEqual(notes, {"renamed_equivalent": {"pick": "pick_child"}})
        # any parameter difference keeps the failure
        bad = copy.deepcopy(after)
        s.Draft(bad).nodes["pick_child"]["parameters"]["execution_timeout_s"] = 99
        self.assertIn("unchanged: node pick (motion.move_to_pose) removed", s.judge(out, self.before, bad, "ok"))
        # a copy outside any matched location is not accepted
        moved = FakeComposer(insert(self.before, "pick_and_place", 0, {**copy.deepcopy(pick), "instance_id": "p2"}))
        moved.mutate("tree_draft_delete_node", {"instance_id": "pick"})
        self.assertTrue(any("removed" in x for x in s.judge({"draft": [{"node": {"node_id": "sequence"}}],
                                                             "answer": {"consistent": False}},
                                                            self.before, moved.draft(), "ok")))
        # the conversation result carries the evidence
        res = s.run_conversation({"prompt": "p", "outcome": out}, self.before,
                                 lambda m, t: {"state": "completed", "answer": "ok"}, lambda: after)
        self.assertEqual((res["status"], res["renamed_equivalent"]), ("PASS", {"pick": "pick_child"}))

    def test_named_node_matches_its_equal_copy(self):
        # REPLACE_subtree_specific: the three named nodes re-created as copies inside pick_routine
        d = s.Draft(self.before)
        kids = [{**copy.deepcopy(d.nodes[i]), "instance_id": i + "_c"} for i in ("approach", "pick")]
        snap = copy.deepcopy(self.before)
        ch = snap["authoring"]["forest"][0]["children"]
        ch[1:3] = [node("pick_routine", "fallback", kids)]
        out = {"draft": [{"node": {"instance_id": "pick_routine", "after": "open_gripper", "before": "close_gripper"}},
                         {"node": {"instance_id": "approach", "ancestor": "pick_routine"}},
                         {"node": {"instance_id": "pick", "ancestor": "pick_routine"}},
                         {"subtree_order": {"root": "pick_routine", "order": ["approach", "pick"]}}],
               "answer": {"consistent": False}}
        notes = {}
        self.assertEqual(s.judge(out, self.before, snap, "ok", notes), [])
        self.assertEqual(notes["renamed_equivalent"], {"approach": "approach_c", "pick": "pick_c"})
        out["draft"][3]["subtree_order"]["order"].reverse()
        self.assertTrue(s.judge(out, self.before, snap, "ok"))
        s.Draft(snap).nodes["pick_c"]["parameters"]["goal"]["position_m"] = [1, 1, 1]
        out["draft"][3]["subtree_order"]["order"].reverse()
        self.assertTrue(s.judge(out, self.before, snap, "ok")[0].startswith("draft: node{instance_id=pick"))

    def test_covered_parameter_only(self):
        f = FakeComposer(self.before)
        f.mutate("tree_draft_update_node_params", {"instance_id": "close_gripper",
                                                   "parameters": {"position": {"value_rad": 0.5}, "speed": 1}})
        checks = [{"node": {"instance_id": "close_gripper", "parameters": {"position.value_rad": 0.5}}}]
        self.assertEqual(self.collateral(f.draft(), checks), ["node close_gripper changed outside the check: parameters.speed"])


class ScorerExtensions2026_11_10(unittest.TestCase):
    """New namespace of a matched key, declared removable parameters, {absent: true},
    {condition_equivalent: C}."""

    def setUp(self):
        self.before = program()

    def test_namespace_of_matched_new_key_is_matched(self):
        after = copy.deepcopy(self.before)
        bb = after["authoring"]["blackboard"]
        bb["namespaces"].append({"namespace_id": "planning", "owner": "program"})
        bb["keys"].append({"key": "planning.pick_plan", "type": "pick_planning_result"})
        check = [{"blackboard_key": {"present": True, "type": "pick_planning_result"}}]
        self.assertEqual(s.judge({"draft": check, "answer": {"consistent": False}}, self.before, after, "ok"), [])
        # a namespace no matched key lives in is still an unrequested edit
        bb["namespaces"].append({"namespace_id": "other", "owner": "program"})
        self.assertEqual(s.judge({"draft": check, "answer": {"consistent": False}}, self.before, after, "ok"),
                         ["unchanged: blackboard namespace other added"])
        # and so is the namespace when its key is not matched by a check
        lone = copy.deepcopy(self.before)
        lone["authoring"]["blackboard"]["namespaces"].append({"namespace_id": "planning", "owner": "program"})
        self.assertTrue(s.judge({"draft": [{"node": {"instance_id": "pick"}}], "answer": {"consistent": False}},
                                self.before, lone, "ok"))

    def _rebound(self, drop=("target_mode",), mode=None):
        after = copy.deepcopy(self.before)
        pick = s.Draft(after).nodes["pick"]
        pick["parameters"].pop("goal")
        pick["parameters"]["target_mode"] = "relative"     # set up: before has it, after drops it
        for k in drop:
            pick["parameters"].pop(k, None)
        if mode:
            pick["parameters"]["target_mode"] = mode
        pick["value_sources"]["goal"] = {"kind": "blackboard_field", "reference": "plan", "field": "pick_pose"}
        before = copy.deepcopy(self.before)
        s.Draft(before).nodes["pick"]["parameters"]["target_mode"] = "relative"
        return before, after

    def test_may_remove(self):
        spec = {"instance_id": "pick", "value_sources": {"goal": {"kind": "blackboard_field", "field": "pick_pose"}}}
        before, after = self._rebound()
        out = lambda sp: s.judge({"draft": [{"node": sp}], "answer": {"consistent": False}}, before, after, "ok")
        self.assertIn("unchanged: node pick changed outside the check: parameters.target_mode", out(spec))
        self.assertEqual(out({**spec, "may_remove": ["target_mode"]}), [])
        # declared removable does not allow a changed value
        before, after = self._rebound(mode="absolute")
        self.assertTrue(out({**spec, "may_remove": ["target_mode"]}))
        # {absent: true} asserts the removal
        before, after = self._rebound()
        self.assertEqual(out({**spec, "may_remove": ["target_mode"], "parameters": {"target_mode": s.ABSENT}}), [])
        before, after = self._rebound(drop=())
        self.assertTrue(out({**spec, "parameters": {"target_mode": {"absent": True}}}))
        problems = s.validate_corpus({"schema_version": "manyforge.assistant_smoke_corpus.v2", "cases": [
            {"id": "X", "category": "c", "intent": "i", "prompt": "p",
             "outcome": {"draft": [{"node": {"node_id": "timer", "may_remove": "goal"}}]}}]})
        self.assertIn("may_remove", json.dumps(problems))

    def test_condition_equivalent(self):
        sig = {"signal": {"component_id": "cell_io", "signal_name": "touch_probe"}}
        cmp = lambda op, v, o=sig: {"compare": {"operand": o, "op": op, "value": v}}
        want = {"condition_equivalent": cmp("==", True)}
        for same in (cmp("==", True), cmp("!=", False), {"not": cmp("==", False)}, {"all": [cmp("==", True)]},
                     {"any": [cmp("==", True), cmp("==", True)]}, {"not": {"not": cmp("==", True)}}):
            self.assertTrue(s.matches(want, same), same)
        other = {"signal": {"component_id": "cell_io", "signal_name": "start_button"}}
        for diff in (cmp("==", False), cmp("!=", True), {"all": [cmp("==", True), cmp("==", True, other)]},
                     {"any": [cmp("==", True), cmp("==", True, other)]}, cmp("==", True, other), None, {"all": []}):
            self.assertFalse(s.matches(want, diff), diff)
        two = {"condition_equivalent": {"all": [cmp("==", True), cmp("==", False, other)]}}
        self.assertTrue(s.matches(two, {"not": {"any": [cmp("==", False), cmp("==", True, other)]}}))  # De Morgan
        self.assertTrue(s.path_matches({"interrupt_condition": {"all": [cmp("==", True)]}}, "interrupt_condition", want))


class Diagnostics(unittest.TestCase):
    def test_new_errors_and_shifted_paths(self):
        before = program()
        before["diagnostics"] = [{"code": "authoring_parameter_required", "path": "forest[0].children[3].parameters.target",
                                  "severity": "error", "message": "m"}]
        after = insert(before, "pick_and_place", 0, node("p", "parallel"))
        after["diagnostics"] = [
            {"code": "authoring_parameter_required", "path": "forest[0].children[4].parameters.target", "severity": "error", "message": "m"},
            {"code": "authoring_node_children_incomplete", "path": "forest[0].children[0].children", "severity": "error", "message": "x"}]
        new = s.new_error_diagnostics(s.Draft(before), s.Draft(after))
        self.assertEqual([d["code"] for d in new], ["authoring_node_children_incomplete"])
        out = {"draft": [{"node": {"node_id": "parallel"}}]}
        self.assertTrue(any(f.startswith("diagnostics") for f in s.judge(out, before, after, "Added parallel p.")))
        self.assertEqual(s.judge({**out, "diagnostics": "allow_incomplete"}, before, after, "Added parallel p."), [])


class Answer(unittest.TestCase):
    def setUp(self):
        self.before = program()
        self.after = insert(self.before, "pick_and_place", 4, node("wait_2s", "timer", duration_s=2))

    def test_names_by_id_node_id_or_shape(self):
        # PnP_06 (cap A/B): the node_id in its own words, with Unicode hyphens and markdown
        after = insert(self.before, "pick_and_place", 4, node("move_to_above_graspable", "motion.move_to_pose"))
        out = {"draft": [{"node": {"node_id": "motion.move_to_pose", "index": 4}}]}
        ans = "Added a **move\u2011to\u2011pose** node as the first child of `pick_and_place`."
        self.assertEqual(s.judge(out, self.before, after, ans), [])
        self.assertEqual(s.judge(out, self.before, after, "Added a motion step at the end."), [])
        self.assertTrue(any("does not name" in f for f in s.judge(out, self.before, after, "Done, it is there now.")))
        # SCENE_add_generic: "cube" names a box
        f = FakeComposer(self.before)
        f.mutate("scene_draft_upsert_objects", {"scene_resources": [{"scene_resource_id": "small_box",
            "shape": {"type": "box", "box_dimensions_m": [0.05, 0.05, 0.05]},
            "pose": {"frame_ref": {"kind": "cell_frame"}, "position_m": [0.4, 0, 0.025], "orientation": {"rpy_deg": [0, 0, 0]}}}]})
        out = {"draft": [{"scene_resource": {"new": True}}]}
        cube = "A 5 cm cube has been added to the scene at position (0.4 m, 0.0 m, 0.025 m) in the cell frame."
        self.assertEqual(s.judge(out, self.before, f.draft(), cube), [])
        self.assertTrue(any("does not name" in x for x in s.judge(out, self.before, f.draft(), "An object has been added.")))
        # a word inside another word is not a name ("timer" vs "timers" is, "set" in "settings" is not)
        self.assertFalse(any(f" {n} " in s._norm("new settings") for n in s._names({"id": "set", "node_id": "io.set"})))

    def test_location_claim(self):
        out = {"draft": [{"node": {"node_id": "timer"}}]}
        self.assertEqual(s.judge(out, self.before, self.after, "Added timer wait_2s after close_gripper."), [])
        self.assertIn("answer: says wait_2s is after open_gripper, but the Draft has it after close_gripper",
                      s.judge(out, self.before, self.after, "Added timer wait_2s after `open_gripper`."))

    def test_same_parent_reorder_is_a_change(self):
        # MOVE_swap_specific: pick moved before approach under the same parent.
        after = copy.deepcopy(self.before)
        ch = after["authoring"]["forest"][0]["children"]
        ch.insert(1, ch.pop(2))
        self.assertTrue(s.draft_changed(s.Draft(self.before), s.Draft(after)))
        ch = s.changes(s.Draft(self.before), s.Draft(after))
        self.assertEqual([(c["change"], sorted(c["shifted"])) for c in ch], [("moved", ["approach", "pick"])])
        out = {"draft": [{"node": {"instance_id": "pick", "parent": "pick_and_place", "index": 1}}]}
        self.assertEqual(s.judge(out, self.before, after, "I moved pick before approach."), [])
        self.assertTrue(any(f.startswith("answer: does not name the moved node") for f in s.judge(out, self.before, after, "Done.")))
        # an unmatched reorder is still collateral
        self.assertTrue(any("reordered" in f for f in s.judge({"draft": [{"node": {"node_id": "sequence"}}]},
                                                             self.before, after, "I moved pick.")))

    def test_claims(self):
        self.assertTrue(s.claims_change("I added the timer."))
        self.assertFalse(s.claims_change("I did not add anything."))
        self.assertFalse(s.claims_change("Should I add a timer?"))
        self.assertFalse(s.claims_change("It already contains the scene resources you added (ground, pads)."))
        self.assertTrue(s.claims_change("The timer was added after pick."))
        self.assertTrue(s.claims_change("- Added `wait_2s` at the end"))
        self.assertTrue(s.claims_change("I've wrapped the root in a repeat."))
        self.assertTrue(s.claims_failure("The insert failed."))
        self.assertFalse(s.claims_failure("The first insert failed, so I added it after pick."))
        # CLARIFY_tree_wrap_pronoun: node behaviour is not a failed request
        retry = ("The close_gripper step is now wrapped in a retry decorator. If the first attempt fails, the retry "
                 "node runs the command again. If both attempts fail, the node fails.")
        self.assertFalse(s.claims_failure(retry))
        self.assertFalse(s.claims_failure("If both attempts fail, the node fails."))
        for text in ("I couldn't find that node.", "I was unable to insert the timer.", "The edit failed.",
                     "Failed to add the box.", "No changes were made.", "The request was rejected by the Draft."):
            self.assertTrue(s.claims_failure(text), text)

    def test_answer_rules(self):
        out = {"draft": [{"node": {"node_id": "timer"}}]}
        self.assertEqual(s.judge(out, self.before, self.after, "I added a 2 s timer at the end."), [])
        self.assertEqual(s.judge(out, self.before, self.after, "Inserted wait_2s."), [])
        self.assertIn("answer: does not name the added node wait_2s",
                      s.judge(out, self.before, self.after, "Done."))
        self.assertIn("answer: claims failure although the request was fulfilled",
                      s.judge(out, self.before, self.after, "Unable to insert the timer node."))
        self.assertEqual(s.judge(out, self.before, self.after,
                                 "Inserted wait_2s. If both attempts fail, the node fails."), [])
        self.assertIn("answer: claims a change but the Draft is unchanged",
                      s.judge({"draft": [{"node": {"node_id": "sequence"}}]}, self.before, self.before, "I added the timer."))
        self.assertEqual(s.judge(out, self.before, self.after, ""), ["answer: empty"])
        self.assertIn("answer: must contain 'end'", s.judge({**out, "answer": {"must_contain": ["end"]}},
                                                             self.before, self.after, "Added timer."))


class Conversation(unittest.TestCase):
    def setUp(self):
        self.before = program()

    def drive(self, outcome, turns):
        """turns: list of (answer, needs_clarification, draft-after-turn)."""
        sent, draft = [], [self.before]
        script = list(turns)

        def submit(message, turn):
            sent.append((turn, message))
            answer, nc, after = script.pop(0)
            draft[0] = after
            return {"state": "completed", "answer": answer, "needs_clarification": nc}
        res = s.run_conversation({"prompt": "p", "outcome": outcome}, self.before, submit, lambda: draft[0])
        return res, sent

    def timer_after(self):
        return insert(self.before, "pick_and_place", 4, node("t", "timer", duration_s=1))

    def test_none_direct_pass_and_clarified(self):
        out = {"draft": [{"node": {"node_id": "timer", "last_child": True}}], "clarification_answer": "at the end"}
        res, sent = self.drive(out, [("Added timer t.", False, self.timer_after())])
        self.assertEqual((res["status"], len(sent)), ("PASS", 1))
        res, sent = self.drive(out, [("Where should it go?", False, self.before), ("Added timer t.", False, self.timer_after())])
        self.assertEqual((res["status"], sent[1]), ("PASS_CLARIFIED", (2, "at the end")))
        res, _ = self.drive({"draft": out["draft"]}, [("Where?", True, self.before)])
        self.assertEqual(res["status"], "FAIL")

    def test_trailing_question_after_fulfilled_request(self):
        # PnP_05: a read-only request, Draft unchanged, answer ends with an offer.
        out = {"draft": [{"node": {"instance_id": "pick_and_place", "parent": None}}],
               "answer": {"must_contain": ["pick_and_place"]}}
        ans = "The root is the pick_and_place sequence. What children would you like to insert?"
        res, sent = self.drive(out, [(ans, False, self.before)])
        self.assertEqual((res["status"], len(sent)), ("PASS", 1))
        res, sent = self.drive(out, [(ans, True, self.before)])
        self.assertEqual((res["status"], len(sent)), ("PASS", 1))
        # fulfilled action plus a follow-up question: no clarification turn
        act = {"draft": [{"node": {"node_id": "timer"}}], "clarification_answer": "x"}
        res, sent = self.drive(act, [("Added timer t. Anything else?", True, self.timer_after())])
        self.assertEqual((res["status"], len(sent)), ("PASS", 1))
        # a question while the checks do not hold is still "asked"
        res, _ = self.drive({"draft": [{"node": {"node_id": "timer"}}]}, [("Where should it go?", False, self.before)])
        self.assertTrue(res["failures"][0].startswith("draft: node{node_id=timer}: no match"), res["failures"])
        self.assertEqual(res["failures"][-1], "clarify: the model asked instead of acting (no clarification_answer)")

    def test_required(self):
        out = {"clarify": "required"}
        self.assertEqual(self.drive(out, [("Which node?", True, self.before)])[0]["status"], "PASS")
        self.assertEqual(self.drive(out, [("Added timer t.", False, self.timer_after())])[0]["status"], "FAIL")
        self.assertEqual(self.drive(out, [("Added it. Which one next?", True, self.timer_after())])[0]["status"], "FAIL")
        out = {"clarify": "required", "clarification_answer": "a timer",
               "draft": [{"node": {"node_id": "timer", "last_child": True}}]}
        res, sent = self.drive(out, [("Which?", True, self.before), ("Added timer t.", False, self.timer_after())])
        self.assertEqual((res["status"], res["clarified"], len(sent)), ("PASS", True, 2))

    def test_allowed(self):
        out = {"clarify": "allowed", "draft": [{"node": {"node_id": "timer"}}]}
        self.assertEqual(self.drive(out, [("Where should it go?", False, self.before)])[0]["status"], "PASS")
        self.assertEqual(self.drive(out, [("I read it as the end: added timer t.", False, self.timer_after())])[0]["status"], "PASS")
        out["clarification_answer"] = "at the end"
        res, _ = self.drive(out, [("Where?", False, self.before), ("I could not do it.", False, self.before)])
        self.assertEqual(res["status"], "FAIL")


class ChainRepair(unittest.TestCase):
    def fresh(self):
        snap = program()
        snap["authoring"]["forest"] = [node("pick_and_place", "sequence", [node("scaffold", "timer", duration_s=0.1)])]
        snap["authoring"]["scene"] = []
        return snap

    def test_repair_keeps_scaffold_last_and_restores(self):
        fake = FakeComposer(self.fresh())
        steps = [
            {"id": "S1", "outcome": {"draft": [{"scene_resource": {"present": True, "shape": {"type": "box", "box_dimensions_m": [2, 2, 0.01]},
                                                                   "pose": {"position_m": [0, 0, 0]}}}]}},
            {"id": "S2", "outcome": {"draft": [{"node": {"node_id": "motion.move_to_pose", "parent": "pick_and_place", "index": 0,
                                                         "parameters": {"goal.position_m": [0.4, 0, 0.2]}}}]}},
            {"id": "S3", "outcome": {"draft": [{"any_of": [
                [{"node": {"node_id": "gripper.command_position", "parent": "pick_and_place", "index": 1,
                           "parameters": {"position.kind": "named_preset", "position.name": "closed"}}}],
                [{"node": {"node_id": "gripper.command_position", "parent": "pick_and_place", "index": 1,
                           "parameters": {"position.value_rad": 0.66}}}]]}]}},
            {"id": "S4", "outcome": {"draft": [{"parameter": {"id": "speed", "present": True, "type": "float64", "default": 0.1}},
                                               {"blackboard_key": {"key": "k", "present": True, "type": "int64"}}]}},
            {"id": "S5", "outcome": {"draft": [{"node": {"node_id": "repeat", "parent": None, "parameters": {"num_cycles": -1},
                                                         "children_node_ids": ["sequence"]}},
                                               {"node": {"instance_id": "pick_and_place", "parent_node_id": "repeat"}}]}},
        ]
        for step in steps:
            before = fake.draft()
            # the model's partial wrong edit, which the repair must undo
            fake.mutate("tree_draft_insert_node", {"parent_instance_id": "pick_and_place", "index": 0,
                                                   "node": {"node_id": "timer", "instance_id": "junk_" + step["id"]}})
            for _ in range(10):
                muts = s.restore_mutations(before, fake.draft())
                if not muts:
                    break
                for m in muts:
                    fake.mutate(m["tool"], m["arguments"])
            self.assertTrue(s.values_equal(fake.draft()["authoring"]["forest"], before["authoring"]["forest"]))
            start = fake.draft()
            for m in s.repair_mutations(step, start):
                fake.mutate(m["tool"], m["arguments"])
            fail, _ = s.evaluate_all(step["outcome"]["draft"], s.Draft(start), s.Draft(fake.draft()))
            self.assertIsNone(fail, step["id"])
        d = s.Draft(fake.draft())
        self.assertEqual([d.nodes[i]["node_id"] for i in d.children["pick_and_place"]],
                         ["motion.move_to_pose", "gripper.command_position", "timer"])
        self.assertEqual(d.children["pick_and_place"][-1], "scaffold")
        self.assertEqual(d.forest[0]["node_id"], "repeat")

    def test_explicit_repair_and_unsupported(self):
        case = {"id": "X", "outcome": {"repair": [{"tool": "t", "arguments": {}}], "draft": []}}
        self.assertEqual(s.repair_mutations(case, self.fresh()), [{"tool": "t", "arguments": {}}])
        self.assertEqual(s.repair_mutations({"id": "E", "outcome": {"repair": [], "draft": [
            {"node": {"node_id": "fallback", "parent": "pick_and_place"}}]}}, self.fresh()), [])
        ref = {"id": "R", "outcome": {"repair": [{"tool": "u", "arguments": {"instance_id": "@child:pick_and_place:-1",
                                                                          "x": ["@child:pick_and_place:0"]}}]}}
        self.assertEqual(s.repair_mutations(ref, self.fresh())[0]["arguments"],
                         {"instance_id": "scaffold", "x": ["scaffold"]})
        with self.assertRaises(s.RepairUnsupported):
            s.repair_mutations({"id": "R2", "outcome": {"repair": [{"tool": "u", "arguments": {
                "instance_id": "@child:pick_and_place:5"}}]}}, self.fresh())
        bad = {"id": "Y", "outcome": {"draft": [{"parameter": {"id": "v", "present": True, "default": {"max_exclusive": 1}}}]}}
        with self.assertRaises(s.RepairUnsupported):
            s.repair_mutations(bad, self.fresh())


class Validator(unittest.TestCase):
    def test_validator(self):
        good = {"id": "A", "category": "c", "intent": "i", "prompt": "p",
                "outcome": {"clarify": "allowed", "clarification_answer": "x",
                            "draft": [{"any_of": [[{"node": {"node_id": "timer"}}], [{"scene_resource": {"new": True}}]]}]}}
        corpus = {"schema_version": "manyforge.assistant_smoke_corpus.v2", "cases": [good]}
        self.assertEqual(s.validate_corpus(corpus), [])
        bad = copy.deepcopy(good)
        bad.update(id="B", precondition={})
        bad["outcome"]["draft"] = [{"node": {"node_id": "t", "where": 1}}, {"all_nodes": {}},
                                   {"scene_resource": {"present": False}}]
        bad["outcome"]["clarify"] = "maybe"
        problems = s.validate_corpus({**corpus, "cases": [good, bad, good]})
        text = json.dumps(problems)
        for needle in ("precondition", "unknown fields ['where']", "all_nodes", "present: false needs id",
                       "clarify must be", "duplicate id"):
            self.assertIn(needle, text)


class Runner(unittest.TestCase):
    def test_log_line_and_summary(self):
        import smoke_corpus_v2 as runner
        r = {"case_id": "C", "status": "FAIL", "failures": ["draft: x", "answer: y"],
             "metrics": {"wall_s": 12.34, "calls": 3, "failed_calls": 1}, "category": "k", "clarify": "none"}
        self.assertEqual(runner.log_line(r), "FAIL C 12.3 3/1 draft: x")
        res = [r, {**r, "case_id": "D", "status": "PASS_CLARIFIED", "failures": []},
               {**r, "case_id": "E", "status": "PASS_CLARIFIED", "clarify": "allowed", "failures": []},
               {"case_id": "F", "status": "NOT_RUN"}]
        sm = runner.summarize(res, {})
        self.assertEqual((sm["pass"], sm["pass_after_clarification"], sm["fail"], sm["not_run"]), (1, 1, 1, 1))
        self.assertEqual(sm["efficiency"]["calls"], {"n": 3, "p50": 3, "max": 3})
        rows = [{"response": {"usage": {"total_tokens": 10, "completion_tokens_details": {"reasoning_tokens": 7}},
                              "choices": [{"message": {"reasoning_content": "abc"}, "finish_reason": "tool_calls"}]}},
                {"response": {"usage": {"total_tokens": 5, "completion_tokens_details": {"reasoning_tokens": 4}},
                              "choices": [{"message": {}, "finish_reason": "length"}]}},
                {"response": {"usage": {"total_tokens": 1}, "choices": [{"finish_reason": "stop"}]}}]
        self.assertEqual(runner.model_metrics(rows), {"model_calls": 3, "total_tokens": 16, "reasoning_tokens": 11,
                                                      "length_truncated_calls": 1,
                                                      "first_call_reasoning_tokens": 7, "first_call_reasoning_chars": 3})
        self.assertIsNone(runner.model_metrics([{"response": {"usage": {}}}])["reasoning_tokens"])
        sm2 = runner.summarize([{**r, "status": "PASS", "metrics": {"length_truncated_calls": 2}}], {})
        self.assertEqual(sm2["length_truncated_calls_total"], 2)

    def test_endpoint_refusal(self):
        import smoke_corpus_v2 as runner
        ep = {"base_url": "http://127.0.0.1:8010/v1", "model_id": "m",
              "request_contract": {"reasoning": "on", "max_output_tokens": 2048}}
        self.assertIsNone(runner.endpoint_refusal(ep, 8010, "on", 2048))
        self.assertIsNone(runner.endpoint_refusal(ep, 8010, None, None))
        self.assertIn("max_output_tokens is 2048", runner.endpoint_refusal(ep, 8010, "on", 8192))
        self.assertIn("reasoning is 'on'", runner.endpoint_refusal(ep, 8010, "off", None))
        self.assertIn("capture proxy", runner.endpoint_refusal(ep, 8011, None, None))


class Scenarios(unittest.TestCase):
    def test_v3_one_corpus_two_scenarios(self):
        import yaml
        import smoke_corpus_v2 as runner
        corpus = yaml.safe_load(open(Path(__file__).resolve().parent / "smoke_corpus_v3.yaml"))
        self.assertEqual(s.validate_corpus(corpus), [])
        by = {}
        for c in corpus["cases"]:
            by.setdefault(runner.scenario_of(c, corpus), []).append(c["id"])
        self.assertEqual(set(by), {"mock_ur10e_assistant_scene_authoring", "sim_ur10e_assistant_scene_authoring"})
        self.assertIn("CAT_io_unavailable_generic", by["sim_ur10e_assistant_scene_authoring"])
        self.assertEqual(len(corpus["cases"]), 131)


class OfflineRescore(unittest.TestCase):
    def test_rejudge_from_report(self):
        import smoke_corpus_v2 as runner
        before = program()
        ans = ("The root is the pick_and_place sequence. It already contains the scene resources you added. "
               "What children would you like to insert?")
        turn = {"turn": 1, "state": "completed", "needs_clarification": False,
                "status": {"response": {"message": ans}}}
        case = {"id": "PnP_05", "prompt": "p", "outcome": {"draft": [{"node": {"instance_id": "pick_and_place"}}],
                                                         "answer": {"must_contain": ["pick_and_place"]}}}
        old = {"case_id": "PnP_05", "status": "FAIL", "failures": ["clarify: the model asked instead of acting"],
               "asked": True, "turns": [turn], "draft_before": before, "draft_after": before}
        (r,) = runner.rescore_report({"cases": [case]}, {"results": [old]})
        self.assertEqual((r["old_status"], r["status"]), ("FAIL", "PASS"))
        # two turns, the first unchanged-Draft question: its Draft is the before-Draft
        after = insert(before, "pick_and_place", 4, node("t", "timer"))
        t2 = {"turn": 2, "state": "completed", "status": {"response": {"message": "Added timer t."}}}
        case2 = {"id": "C", "prompt": "p", "outcome": {"draft": [{"node": {"node_id": "timer"}}],
                                                     "clarification_answer": "end"}}
        old2 = {"case_id": "C", "status": "PASS_CLARIFIED", "asked": True,
                "turns": [{**turn, "status": {"response": {"message": "Where?"}}}, t2],
                "draft_before": before, "draft_after": after}
        (r2,) = runner.rescore_report({"cases": [case2]}, {"results": [old2]})
        self.assertEqual(r2["status"], "PASS_CLARIFIED")
        (r3,) = runner.rescore_report({"cases": [{**case2, "outcome": {**case2["outcome"], "clarify": "required"}}]},
                                      {"results": [old2]})
        self.assertEqual((r3["old_status"], r3["status"]), ("PASS_CLARIFIED", "PASS"))
        old2["turns"][0]["needs_clarification"] = True
        self.assertEqual(runner.rejudge(case2, old2)["status"], "NOT_RECONSTRUCTABLE")
        self.assertEqual(runner.rejudge(case2, {"turns": []})["status"], "NOT_RECONSTRUCTABLE")


class RunnerCase(unittest.TestCase):
    """run_case wiring with a fake Cell: setup, turns, chain repair record."""

    class Cell(FakeComposer):
        def __init__(self, snap, script):
            super().__init__(snap)
            self.applied = copy.deepcopy(snap)
            self.script = script

        def mark(self, *a):
            pass

        def discard(self):
            self.snap = copy.deepcopy(self.applied)

        def submit(self, message, conversation, budget):
            answer, edits = self.script.pop(0)
            for tool, args in edits:
                self.mutate(tool, args)
            calls = [{"toolId": t, "status": "completed", "arguments": a} for t, a in edits]
            return "completed", {"response": {"message": answer, "toolCalls": calls}}, 1.5

    def test_chain_fail_then_repair_then_pass(self):
        import smoke_corpus_v2 as runner
        header = {"fresh_program_fixture": {"tree": node("pick_and_place", "sequence", [node("scaffold", "timer")])}}
        steps = [{"id": f"S{i}", "category": "c", "prompt": "p",
                  "preconditions": {"chain_id": "c", "chain_step": i, **({"fresh_program": True} if i == 1 else {})},
                  "outcome": {"draft": [{"node": {"node_id": "timer", "parent": "pick_and_place", "index": i - 1,
                                                  "parameters": {"duration_s": float(i)}}}]}} for i in (1, 2)]
        cell = self.Cell(program(), [
            ("I added a timer.", [("tree_draft_insert_node", {"parent_instance_id": "pick_and_place", "index": 1,
                                                              "node": {"node_id": "timer", "instance_id": "bad",
                                                                       "parameters": {"duration_s": 1.0}}})]),
            ("Added timer t2.", [("tree_draft_insert_node", {"parent_instance_id": "pick_and_place", "index": 1,
                                                             "node": {"node_id": "timer", "instance_id": "t2",
                                                                      "parameters": {"duration_s": 2.0}}})])])
        chains = {}
        r1 = runner.run_case(cell, steps[0], header, chains, 10)
        self.assertEqual(r1["status"], "FAIL")
        self.assertTrue(r1["repair"]["ok"], r1["repair"])
        r2 = runner.run_case(cell, steps[1], header, chains, 10)
        self.assertEqual(r2["status"], "PASS", r2["failures"])
        self.assertEqual(s.Draft(cell.draft()).children["pick_and_place"][-1], "scaffold")
        self.assertEqual(runner.log_line(r2), "PASS S2 1.5 1/0  [fresh conversation after repair]")
        self.assertIn("draft_after", r2["turns"][0])
        self.assertEqual(runner.rejudge(steps[1], r2)["status"], "PASS")

    def test_fresh_conversation_after_repair(self):
        import smoke_corpus_v2 as runner
        header = {"fresh_program_fixture": {"tree": node("pick_and_place", "sequence", [node("scaffold", "timer")])}}
        steps = [{"id": f"S{i}", "category": "c", "prompt": "p",
                  "preconditions": {"chain_id": "c", "chain_step": i, **({"fresh_program": True} if i == 1 else {})},
                  "outcome": {"draft": [{"node": {"node_id": "timer", "parent": "pick_and_place", "index": i - 1,
                                                  "parameters": {"duration_s": float(i)}}}]}} for i in (1, 2, 3)]
        header["cases"] = steps
        ins = lambda i: ("Added timer t%d." % i, [("tree_draft_insert_node", {
            "parent_instance_id": "pick_and_place", "index": i - 1,
            "node": {"node_id": "timer", "instance_id": "t%d" % i, "parameters": {"duration_s": float(i)}}})])
        cell = self.Cell(program(), [ins(1), ("Nothing.", []), ins(3)])
        convs = []
        orig = cell.submit
        cell.submit = lambda m, conv, b: (convs.append(conv), orig(m, conv, b))[1]
        chains = {}
        r = [runner.run_case(cell, c, header, chains, 10) for c in steps]
        self.assertEqual([x["status"] for x in r], ["PASS", "FAIL", "PASS"])
        self.assertTrue(r[1]["repair"]["ok"])
        self.assertEqual(convs[0], convs[1])          # shared until the repair
        self.assertNotEqual(convs[1], convs[2])       # new conversation after it
        self.assertTrue(r[2]["fresh_after_repair"])
        self.assertNotIn("fresh_after_repair", r[1])
        self.assertIn("[fresh conversation after repair]", runner.log_line(r[2]))

    def test_failed_repair_breaks_the_rest_of_the_chain(self):
        import smoke_corpus_v2 as runner
        header = {"fresh_program_fixture": {"tree": node("pick_and_place", "sequence", [node("scaffold", "timer")])}}
        bad_repair = [{"tool": "no_such_mutation", "arguments": {}}]
        steps = [{"id": f"S{i}", "category": "c", "prompt": "p",
                  "preconditions": {"chain_id": "c", "chain_step": i, **({"fresh_program": True} if i == 1 else {})},
                  "outcome": {"draft": [{"node": {"node_id": "fallback", "parent": "pick_and_place", "index": i - 1}}],
                              "repair": bad_repair}} for i in (1, 2)]
        header["cases"] = steps
        cell = self.Cell(program(), [("Nothing to do.", [])])
        chains = {}
        r1 = runner.run_case(cell, steps[0], header, chains, 10)
        self.assertEqual(r1["status"], "FAIL", r1)
        self.assertFalse(r1["repair"]["ok"])
        r2 = runner.run_case(cell, steps[1], header, chains, 10)
        self.assertEqual(r2["status"], "HARNESS_ERROR")
        self.assertIn("repair of S1 failed", r2["failures"][0])

    def test_filtered_chain_is_seeded_not_run_on_applied(self):
        import smoke_corpus_v2 as runner
        steps = [{"id": f"S{i}", "category": "c", "prompt": "p",
                  "preconditions": {"chain_id": "c", "chain_step": i, **({"fresh_program": True} if i == 1 else {})},
                  "outcome": {"draft": [{"node": {"node_id": "timer", "parent": "pick_and_place", "index": i - 1,
                                                  "parameters": {"duration_s": float(i)}}}]}} for i in (1, 2, 3)]
        header = {"fresh_program_fixture": {"tree": node("pick_and_place", "sequence", [node("scaffold", "timer")])},
                  "cases": steps}
        cell = self.Cell(program(), [
            ("Added timer t3.", [("tree_draft_insert_node", {"parent_instance_id": "pick_and_place", "index": 2,
                                                             "node": {"node_id": "timer", "instance_id": "t3",
                                                                      "parameters": {"duration_s": 3.0}}})])])
        r = runner.run_case(cell, steps[2], header, {}, 10)
        self.assertEqual(r["status"], "PASS", r.get("failures"))
        self.assertEqual(r["chain_seed"]["steps"], ["S1", "S2"])
        d = s.Draft(cell.draft())
        self.assertNotIn("open_gripper", d.nodes)  # not the Applied Program
        self.assertEqual([d.nodes[i]["parameters"].get("duration_s") for i in d.children["pick_and_place"]],
                         [1.0, 2.0, 3.0, None])
        self.assertIn("[seeded from S1..S2]", runner.log_line(r))



class KindChangeCoverage(unittest.TestCase):
    def test_check_naming_the_new_kind_covers_the_rewrite(self):
        before = s.Draft(_kind_program("motion.move_to_pose", {"goal": {"position_m": [0, 0, 0]}}))
        after = s.Draft(_kind_program("motion.follow_cartesian_path", {"path": [{"position_m": [0, 0, 0]}]}))
        check = {"node": {"instance_id": "move_to_drop", "node_id": "motion.follow_cartesian_path"}}
        fail, m = s.evaluate(check, before, after)
        self.assertIsNone(fail)
        self.assertEqual(s.collateral(before, after, m), [])

    def test_check_without_node_id_still_flags_a_kind_change(self):
        before = s.Draft(_kind_program("motion.move_to_pose", {"goal": {"position_m": [0, 0, 0]}}))
        after = s.Draft(_kind_program("motion.follow_cartesian_path", {"goal": {"position_m": [0, 0, 0]}}))
        fail, m = s.evaluate({"node": {"instance_id": "move_to_drop"}}, before, after)
        self.assertIsNone(fail)
        self.assertTrue(any("node_id" in c for c in s.collateral(before, after, m)))


def _kind_program(node_id, parameters):
    return {"authoring": {"forest": [{"node_id": "sequence", "instance_id": "root", "parameters": {}, "children": [
        {"node_id": node_id, "instance_id": "move_to_drop", "parameters": parameters, "children": []}]}]}}

if __name__ == "__main__":
    unittest.main()
