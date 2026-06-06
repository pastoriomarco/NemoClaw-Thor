#!/usr/bin/env python3
"""Tests for the streaming-aware nested-MCP-id normalizer in vllm-proxy.py.

Run: python3 test_nested_id_streaming.py   (exits non-zero on failure)

Covers the 2026-06-05 fix: OpenClaw streams `tool_calls[*].function.arguments`
one token per SSE `data:` event, so the contiguous `"id":"…"` the old
text-regex needs never appears in the body. The streaming variant reassembles
the fragments, canonicalizes the nested id, and rewrites the chunks.
"""
import importlib.util
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("vllm_proxy", _HERE / "vllm-proxy.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_failures = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    if not cond:
        _failures.append(f"{name}: {detail}")
    print(f"[{status}] {name}" + (f" — {detail}" if (not cond and detail) else ""))


def _is_json(s):
    try:
        json.loads(s); return True
    except Exception:
        return False


def _sse_fragmented(args_str, *, frag=3, name="tool_call"):
    """Build an SSE body that streams `args_str` as `function.arguments`
    split into `frag`-char fragments, one per `data:` chunk — exactly the
    shape OpenClaw emits."""
    events = []
    # first chunk also carries id/type/name
    first = {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "call_x", "type": "function",
         "function": {"name": name, "arguments": args_str[:frag]}}]}}]}
    events.append("data: " + json.dumps(first, separators=(",", ":")))
    pos = frag
    while pos < len(args_str):
        ch = {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": args_str[pos:pos + frag]}}]}}]}
        events.append("data: " + json.dumps(ch, separators=(",", ":")))
        pos += frag
    events.append("data: [DONE]")
    return "\n\n".join(events) + "\n\n"


def _reassemble_args(body):
    """Reassemble the resulting `function.arguments` from an SSE body (what
    OpenClaw's gateway does downstream)."""
    out = ""
    for line in body.split("\n"):
        s = line.strip()
        if not s.startswith("data:"):
            continue
        p = s[len("data:"):].strip()
        if not p or p == "[DONE]":
            continue
        try:
            obj = json.loads(p)
        except Exception:
            continue
        for c in obj.get("choices", []):
            for tc in (c.get("delta") or {}).get("tool_calls", []) or []:
                out += (tc.get("function") or {}).get("arguments") or ""
    return out


# --- 1. The headline case: dashed id fragmented across SSE chunks ----------
target = '{"id":"manyforge__scene-draft-add-object","args":{"objectId":"ground_plane"}}'
body = _sse_fragmented(target, frag=3)
# precondition: the contiguous mangled id is NOT in the raw body (the bug)
check("precondition: mangled id not contiguous in SSE body",
      "manyforge__scene-draft-add-object" not in body,
      "test fixture failed to fragment the id")
new_body, rewrites = _mod._normalize_nested_mcp_ids_streaming(body)
check("dashed id: rewrite fired", len(rewrites) == 1, f"rewrites={rewrites}")
reasm = _reassemble_args(new_body)
check("dashed id: reassembled arguments canonical",
      '"id":"manyforge__scene_draft_add_object"' in reasm, f"reasm={reasm!r}")
check("dashed id: args payload preserved",
      '"objectId":"ground_plane"' in reasm, f"reasm={reasm!r}")
check("dashed id: result is valid JSON", _is_json(reasm), f"reasm={reasm!r}")


# --- 2. Flat dashed tool: program-read (was an uncovered pattern gap) -------
for mangled, canon in [
    ('manyforge__program-read', 'manyforge__program_read'),
    ('manyforge__scene-inspect', 'manyforge__scene_inspect'),
    ('manyforge__inspect-isaac-scene', 'manyforge__inspect_isaac_scene'),
    ('program-read', 'manyforge__program_read'),            # bare + dashed
    ('tree_draft_insert_node', 'manyforge__tree_draft_insert_node'),  # bare
]:
    t = '{"id":"%s","args":{}}' % mangled
    b = _sse_fragmented(t, frag=4)
    nb, rw = _mod._normalize_nested_mcp_ids_streaming(b)
    r = _reassemble_args(nb)
    check(f"flat/bare normalize: {mangled} -> {canon}",
          ('"id":"%s"' % canon) in r, f"got {r!r}")


# --- 3. Idempotency: canonical id untouched, no rewrites -------------------
canon_t = '{"id":"manyforge__tree_draft_insert_node","args":{"x":1}}'
cb = _sse_fragmented(canon_t, frag=5)
ncb, crw = _mod._normalize_nested_mcp_ids_streaming(cb)
check("idempotent: canonical id yields no rewrites", crw == [], f"rw={crw}")
check("idempotent: reassembled unchanged",
      _reassemble_args(ncb) == canon_t, f"got {_reassemble_args(ncb)!r}")


# --- 4. Non-SSE body: streaming variant is a no-op (text-regex handles it) --
plain = '{"choices":[{"message":{"tool_calls":[{"function":{"arguments":"{\\"id\\":\\"manyforge__program-read\\"}"}}]}}]}'
nb2, rw2 = _mod._normalize_nested_mcp_ids_streaming(plain)
check("non-SSE: streaming variant no-ops", nb2 == plain and rw2 == [], f"rw={rw2}")
# and the text-regex variant DOES handle the escaped plain-JSON form
_, rw3 = _mod._normalize_nested_mcp_ids_in_text(plain)
check("non-SSE: text-regex handles escaped flat dashed", len(rw3) == 1, f"rw={rw3}")


# --- 5. History budget guard: explicit failure-shaped envelope -------------
budget_body = {
    "messages": [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "x" * 50},
        {"role": "assistant", "content": "ok"},
    ]
}
stats = _mod._message_stats(budget_body)
check("budget stats: message count", stats["message_count"] == 3, f"stats={stats}")
check("budget stats: role counts", stats["role_counts"]["user"] == 1, f"stats={stats}")
check("budget stats: largest message", stats["largest_message_chars"] == 50, f"stats={stats}")
err = json.loads(
    _mod._history_budget_error_body(
        body_chars=1234,
        max_chars=1000,
        body_json=budget_body,
    )
)
check("budget error: code", err["error"]["code"] == "history_budget_exceeded", f"err={err}")
check("budget error: type",
      err["error"]["type"] == "manyforge_proxy_history_budget_exceeded", f"err={err}")
check("budget error: includes stats",
      err["error"]["historyBudget"]["largest_message_chars"] == 50, f"err={err}")


print()
if _failures:
    print(f"FAILED ({len(_failures)}):")
    for f in _failures:
        print("  -", f)
    sys.exit(1)
print("ALL TESTS PASSED")
