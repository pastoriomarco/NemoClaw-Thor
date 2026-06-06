#!/usr/bin/env python3
"""Tests for the fail-open history-budget trim ladder in vllm-proxy.py.

Run: python3 test_history_trim.py   (exits non-zero on failure)

Rung 1 (keep-latest read elision) always runs; rungs 2-3 escalate only while
over budget. Only RE-FETCHABLE reads are elided; mutation/state results and
message envelopes (role + tool_call_id) are preserved.
"""
import importlib.util
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("vllm_proxy", _HERE / "vllm-proxy.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_fail = []
def check(name, cond, detail=""):
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + ("" if cond else f" — {detail}"))
    if not cond:
        _fail.append(name)

STUB = _mod._READ_ELIDED_STUB
RSTUB = _mod._REASONING_ELIDED_STUB

def tool_read(name, size, tcid):
    """OpenClaw-style tool result envelope for read tool `name`."""
    content = json.dumps({
        "tool": {"id": f"mcp:bundle-mcp:manyforge__{name}", "name": f"manyforge__{name}"},
        "result": {"blob": "X" * size},
    })
    return {"role": "tool", "tool_call_id": tcid, "content": content}

def asst(reasoning, tcid):
    return {"role": "assistant", "content": reasoning,
            "tool_calls": [{"id": tcid, "type": "function",
                            "function": {"name": "tool_call", "arguments": "{}"}}]}

def body(msgs):
    return {"model": "m", "messages": msgs}

def size(b):
    return len(json.dumps(b, separators=(",", ":")))

# --- 1. Rung 1: two catalog reads, generous budget -> old stubbed, latest kept
b = body([{"role": "system", "content": "s"},
          tool_read("catalog_read", 20000, "c1"),
          asst("ok", "a1"),
          tool_read("catalog_read", 20000, "c2")])
rungs, shed = _mod._trim_history_to_budget(b, 10_000_000)  # never over budget
check("rung1: only stub_old_reads", rungs == ["stub_old_reads=1"], f"rungs={rungs}")
check("rung1: first catalog stubbed", b["messages"][1]["content"] == STUB)
check("rung1: latest catalog kept", b["messages"][3]["content"] != STUB)
check("rung1: tool_call_id preserved", b["messages"][1]["tool_call_id"] == "c1")
check("rung1: role preserved", b["messages"][1]["role"] == "tool")
check("rung1: shed > 0", shed > 15000, f"shed={shed}")

# --- 2. Rung 2: single big catalog, budget below it -> latest also stubbed
b = body([{"role": "system", "content": "s"}, tool_read("catalog_read", 50000, "c1")])
rungs, _ = _mod._trim_history_to_budget(b, 1000)
check("rung2: stub_all_reads fired", any(r.startswith("stub_all_reads") for r in rungs), f"rungs={rungs}")
check("rung2: the only catalog stubbed", b["messages"][1]["content"] == STUB)

# --- 3. Rung 3: reads stubbed not enough -> old reasoning stubbed, latest kept
b = body([{"role": "system", "content": "s"},
          asst("X" * 40000, "a1"),                 # old reasoning (huge)
          tool_read("program_read", 5000, "p1"),
          asst("Y" * 40000, "a2")])                # latest reasoning (huge)
rungs, _ = _mod._trim_history_to_budget(b, 50000)
check("rung3: stub_old_reasoning fired", any(r.startswith("stub_old_reasoning") for r in rungs), f"rungs={rungs}")
check("rung3: old reasoning stubbed", b["messages"][1]["content"] == RSTUB)
check("rung3: latest reasoning kept", b["messages"][3]["content"] != RSTUB)
check("rung3: tool_calls preserved on stubbed asst", "tool_calls" in b["messages"][1] and b["messages"][1]["tool_calls"][0]["id"] == "a1")

# --- 4. Mutation/state results are NEVER elided, even under tight budget
b = body([{"role": "system", "content": "s"},
          tool_read("tree_draft_insert_node", 50000, "m1")])  # NOT a read tool
mut_before = b["messages"][1]["content"]
rungs, _ = _mod._trim_history_to_budget(b, 1000)
check("safety: mutation result untouched", b["messages"][1]["content"] == mut_before, f"rungs={rungs}")

# --- 5. Small reads (< MIN_ELIDABLE) are not touched
small = tool_read("catalog_read", 100, "s1")
b = body([{"role": "system", "content": "s"}, small, tool_read("catalog_read", 100, "s2")])
rungs, _ = _mod._trim_history_to_budget(b, 10_000_000)
check("small reads ignored (below MIN_ELIDABLE)", rungs == [], f"rungs={rungs}")

# --- 6. Per-kind keep-latest: two kinds each keep their own latest
b = body([tool_read("catalog_read", 20000, "c1"),
          tool_read("program_read", 20000, "p1"),
          tool_read("catalog_read", 20000, "c2"),
          tool_read("program_read", 20000, "p2")])
_mod._trim_history_to_budget(b, 10_000_000)
got = [m["content"] == STUB for m in b["messages"]]
check("per-kind keep-latest", got == [True, True, False, False], f"got={got}")

# --- 7. classifier: dashed/prefixed names resolve; non-reads return None
check("classify: mcp-locator catalog_read",
      _mod._refetchable_read_kind(tool_read("catalog_read", 5000, "x")) == "catalog_read")
check("classify: mutation -> None",
      _mod._refetchable_read_kind(tool_read("tree_draft_insert_node", 5000, "x")) is None)
check("classify: non-tool message -> None",
      _mod._refetchable_read_kind({"role": "user", "content": "hi"}) is None)

# --- 8. Rung 4: state-heavy (mutation results, not elidable) -> drop oldest,
#        keep system + recent, repair orphan tool results, stay under budget
msgs = [{"role": "system", "content": "sys"}]
for i in range(6):
    msgs.append(asst(f"step {i}", f"a{i}"))
    msgs.append(tool_read("tree_draft_insert_node", 20000, f"a{i}"))  # mutation result (big, not re-fetchable)
b = body(msgs)
full = size(b)
rungs, _ = _mod._trim_history_to_budget(b, full // 2)  # force past rungs 1-3
check("rung4: truncate_oldest fired", any(r.startswith("truncate_oldest") for r in rungs), f"rungs={rungs}")
check("rung4: under budget after truncation", size(b) <= full // 2, f"size={size(b)} budget={full//2}")
check("rung4: system prompt preserved", b["messages"][0].get("role") == "system")
check("rung4: kept >= 2 tail messages", len(b["messages"]) >= 3)
_live = {tc["id"] for m in b["messages"] if m.get("role") == "assistant" for tc in m.get("tool_calls", [])}
_orphans = [m for m in b["messages"] if m.get("role") == "tool" and m.get("tool_call_id") not in _live]
check("rung4: no orphan tool results", _orphans == [], f"orphans={len(_orphans)}")

# --- 9. Reads-first: rung 4 should NOT fire if eliding reads already fits
msgs = [{"role": "system", "content": "sys"},
        tool_read("catalog_read", 80000, "c1"),
        asst("ok", "a1"),
        tool_read("catalog_read", 80000, "c2")]
b = body(msgs)
rungs, _ = _mod._trim_history_to_budget(b, 20000)
check("reads-first: no truncate_oldest when reads suffice",
      not any(r.startswith("truncate_oldest") for r in rungs), f"rungs={rungs}")

print()
if _fail:
    print(f"FAILED ({len(_fail)}): " + ", ".join(_fail)); sys.exit(1)
print("ALL TESTS PASSED")
