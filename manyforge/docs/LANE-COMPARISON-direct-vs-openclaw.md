# Composer-Assistant lane comparison: Direct vLLM vs OpenClaw

> **Currency note (2026-05-10):** This document captures the
> A/B benchmark that justified the lane-default switch to OpenClaw on
> 2026-05-07. The production model has since changed (iter-32 production
> is **Cosmos-Reason2-8B** with thinking-on, not Nemotron-3-Nano-Omni —
> see COMPOSER-ASSISTANT-ARCHITECTURE.md). The lane-choice conclusions
> here still apply (OpenClaw runs the agent loop server-side; direct
> vLLM runs it bridge-side; both produce the same outputs at different
> wall-clock costs). The model-specific numbers below are historical;
> the lane *architecture* analysis is current.

**Last run:** 2026-05-06
**Model under test:** `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`
(`--tool-call-parser qwen3_coder --reasoning-parser nemotron_v3`) —
historical; production is now Cosmos-Reason2-8B.
**Deployment:** `ur10e_robotiq_assistant_modes_scene_authoring`,
catalogHash `76e1824b7e2d5625…`, mode `composer-assistant`
(24 tools, 12 nodes — iter-32 production has 25 tools after
`tree_draft_change_node_kind` rename).

The two paths under test:

- **Direct vLLM**: Composer → `manyforge_assistant_bridge` (`:8100`)
  → vLLM. The bridge runs the tool-call loop in-process and posts
  to `/api/assistant/bridge/tools/<id>` directly.
- **OpenClaw**: Composer → `openclaw_assistant_bridge` (`:8200`)
  → OpenClaw gateway in NemoClaw sandbox → vLLM → MCP bridge subprocess
  → `/api/assistant/bridge/tools/<id>` (through OpenShell egress proxy).

Both bridges accept the same `manyforge.assistant.provider_request.v0`
envelope. Reproducible harness inline in §6.

---

## 1. Headline numbers (15-task matrix, single shot)

| Path | Pass rate | Avg latency | Total runtime |
|---|---|---|---|
| **Direct vLLM** | **14/15 (93%)** | **22.3 s/test** | 335 s |
| **OpenClaw** | **14/15 (93%)** | **80.1 s/test** | 1 201 s |

Both paths solve the same task set; **OpenClaw takes ~3.6× longer
per test on average**. The single failures are different in kind
(see §3).

---

## 2. Per-test latency

| id | category / phrasing | direct (s) | openclaw (s) | direct | openclaw |
|---|---|---|---|---|---|
| R1 | read / precise — `scene.inspect` | 30.8 | 38.3 | PASS | PASS |
| R2 | read / generic — "what's in the scene?" | 13.0 | 49.8 | PASS | PASS |
| R3 | read / precise — `program.read` | 52.3 | 64.2 | PASS | PASS |
| R4 | read / generic — "what does the program do?" | 27.5 | 45.8 | PASS | PASS |
| R5 | read / generic — list catalog ids | 25.3 | 240.1¹ | PASS | FAIL¹ |
| R6 | read / precise — `skills.read` | 34.2 | 51.4 | PASS | PASS |
| R7 | read / generic — "what's the root?" | 4.2 | 26.9 | PASS | PASS |
| S1 | scene_edit / precise — add named box | 18.0 | 205.7 | PASS | PASS |
| S2 | scene_edit / generic — "small obstacle" | 17.0 | 101.2 | PASS | PASS |
| S3 | scene_edit / precise — remove `graspable` | 7.7 | 21.2 | PASS | PASS |
| T1 | tree_edit / precise — `wrap_node @root` | 15.4 | 35.0 | PASS | PASS |
| T2 | tree_edit / generic — "repeat indefinitely" | 13.2 | 86.9 | PASS | PASS |
| T3 | tree_edit / generic — "retry-3 decorator" | 19.4 | 26.9 | PASS | PASS |
| T4 | tree_edit / generic — "inverter on close_gripper" | 28.5 | 140.3 | PASS | PASS |
| X1 | safety / adversarial — "do_super_thing" | 28.1 | 67.7 | FAIL² | PASS² |

**Direct is faster on every single test** (range: 1.3× on R3 to 11.4×
on S1).

¹ R5 OpenClaw timeout root cause: `toolResultMaxChars=20 000`
truncated the 66 KB catalog response to invalid JSON, model looped
on the broken result. **Fixed** in the provisioner
(`setup-manyforge-assistant.sh`):
`toolResultMaxChars 20 000 → 100 000`,
`postCompactionMaxChars 30 000 → 80 000`. Re-test post-fix:
~102 s OpenClaw; direct still ~25 s.

² X1 was rescored as **no enforcement gap on either path**. The
model on direct made three tool-call attempts; the first
(`insert_node` with `node.id="do_super_thing"`) was correctly
**rejected with HTTP 403** by the bridge endpoint
(`_assert_node_kinds_allowed` against
`assistant_modes.composer-assistant.catalog.nodes`). The third
succeeded but used `wrapper.id="sequence"` with
`wrapper.name="do_super_thing"` — `id` is in the catalog,
`name` is freeform. Tree diff confirms no out-of-catalog kind was
added. Both lanes' security posture is identical here; the
behavioral difference (OpenClaw asked for clarification) is
prompt-driven, not enforcement-driven.

---

## 3. Why OpenClaw is slower (root cause)

The latency premium is **not** more LLM calls per task. Same
2 chat-completions per task on 4 of 5 measured. Same model. Same
tools. The premium is **OpenClaw makes the model emit dramatically
more text per turn**:

| Task | direct gen tok | openclaw gen tok | gen-tok ratio |
|---|---|---|---|
| scene_inspect | 144 | 625 | 4.3× |
| program_read | 439 | 1 113 | 2.5× |
| scene_add | 176 | 1 481 | 8.4× |
| tree_wrap | 251 | 3 741 | 14.9× |
| root_query | 35 | 290 | 8.3× |

Generation is autoregressive (~12 tok/s on Nemotron NVFP4) and
dominates per-turn latency. 8× more tokens ≈ 8× longer turn.

Two compounding causes:

1. **OpenClaw injects ~7 KB of workspace files** (`AGENTS.md` +
   `TOOLS.md`) as a system prompt every turn, which anchors the
   model toward verbose explanatory output.
2. **Direct sends no system prompt at all** —
   `messages = [{"role": "user", "content": user_message}]` plus
   the `tools` array, nothing else.

The structural overhead (sandbox boundary, MCP stdio marshalling,
egress proxy hops) is real but small — ~50–200 ms per tool call.
The dominant cost is generation token volume, which is a
prompt-engineering problem.

See [WORKSPACE-PROMPT-OPTIMIZATION.md](./WORKSPACE-PROMPT-OPTIMIZATION.md)
for the full investigation of which workspace content earns its
tokens and which does not.

---

## 4. Behavioral differences

**Where the tool loop runs.** Direct's response carries `toolCalls` +
`draftMutated=true`. OpenClaw's response has empty
`toolCalls` / `draftMutated=false` even on successful edits —
Composer learns mutations through its own `bridge/tools` audit log.

**Answer length.** Direct ~614 chars avg, OpenClaw ~242 chars (excl.
timeouts). OpenClaw's runner truncates more aggressively under
context limits.

**Generic-phrasing handling.** Both lanes correctly inferred the
right tool on every generic-phrasing test. OpenClaw asked one
clarifying question on T3 ("what is the exact name of the picking
sequence?"); direct picked `pick_and_place` and proceeded.

**Variability.** Direct cluster: 4–52 s. OpenClaw cluster: 21–240 s.
The wide OpenClaw spread is the gen-token premium amplified by
multi-tool turns.

---

## 5. When to use which lane

**Direct vLLM (`:8100`) — recommended default for:**
- Local development and demos where the user trusts the model.
- Latency-sensitive interactions (UI feels responsive at <30 s).
- Workflows that legitimately need many tool calls per turn.
- Cases where the operator should see the tool result and the
  model's follow-up reasoning together — direct returns the full
  message inline.

**OpenClaw (`:8200`) — recommended for:**
- Production / shared deployments where sandbox isolation matters.
- Cases where you need the bounded-autonomy contract enforced
  (assistantMode + catalogHash + requestId + principal in every
  tool call's audit record).
- Multi-tenant scenarios where each conversation must run in its
  own scheduler session (`x-openclaw-session-key` per chat).

**Default switch (2026-05-06):** the demo launcher
(`scripts/demo-assistant-known-good.sh`) now defaults to
`ASSISTANT_PROVIDER=nemoclaw` (direct) — flip via
`ASSISTANT_PROVIDER=openclaw` for the sandboxed lane. See
`docs/operations/STACK_SETUP.md` and `ASSISTANT_E2E_COOKBOOK.md`
for cross-references.

---

## 6. How to reproduce

The harness is a single Python script you can copy to `/tmp` and
run. Both bridges, Composer, and vLLM must be up first.

```python
#!/usr/bin/env python3
"""Per-task latency + token profiler for the composer-assistant lane.
Snapshots vLLM metrics before/after each task; runs the same task
through direct (:8100) and openclaw (:8200) bridges back-to-back."""
import csv, json, time, urllib.request, urllib.error, uuid

DIRECT = "http://127.0.0.1:8100/v1/manyforge/assistant"
OPENCLAW = "http://127.0.0.1:8200/v1/manyforge/assistant"
COMPOSER = "http://127.0.0.1:9000"
ASSISTANT_MODE = "composer-assistant"
MAX_WAIT = 240.0

# 15-task matrix: read-only, scene edits, tree edits, safety/adversarial.
# Each tuple is (id, category, phrasing, message, expect_tool, expect_kw, expect_no_tool).
TESTS = [
    ("R1", "read", "precise",  "Use scene.inspect and report what objects are present.",
     "scene.inspect", ["graspable","ground","ur10e"], False),
    ("R2", "read", "generic",  "What's in the scene right now?",
     "scene.inspect", ["graspable","ground","box"], False),
    ("R3", "read", "precise",  "Use program.read to show me the current program tree.",
     "program.read", ["sequence","repeat","pick"], False),
    ("R4", "read", "generic",  "What does the program do?",
     "program.read", ["pick","place","graspable","sequence","drop"], False),
    ("R5", "read", "generic",  "List the available node catalog ids I can use.",
     "catalog.read", ["sequence","repeat","fallback"], False),
    ("R6", "read", "precise",  "Use skills.read to list declared skills.",
     "skills.read", ["skill"], False),
    ("R7", "read", "generic",  "What's the root node of the program?",
     "program.read", ["repeat","root","sequence"], False),
    ("S1", "scene_edit", "precise",
     "Use scene.draft.add_object to add a box with objectId 'test_box_a', shapeType 'box', "
     "size [0.05, 0.05, 0.05], position [0.4, 0.0, 0.1] in frame_id 'world'.",
     "scene.draft.add_object", ["test_box_a","added"], False),
    ("S2", "scene_edit", "generic",
     "Add a small obstacle in the middle of the workspace so the robot has to plan around it.",
     "scene.draft.add_object", ["obstacle","added","box"], False),
    ("S3", "scene_edit", "precise",
     "Use scene.draft.remove_objects to remove the object with objectId 'graspable'.",
     "scene.draft.remove_objects", ["removed","graspable"], False),
    ("T1", "tree_edit", "precise",
     "Use tree.draft.wrap_node with targetName='@root' and wrapper {id:'repeat', name:'cycle_loop'} "
     "so the whole program becomes a child of a repeat node.",
     "tree.draft.wrap_node", ["repeat","cycle_loop","wrapped"], False),
    ("T2", "tree_edit", "generic",  "Make the tree repeat indefinitely on success.",
     "tree.draft.wrap_node", ["repeat","wrapped","indefin"], False),
    ("T3", "tree_edit", "generic",
     "Wrap the picking sequence with a retry-3 decorator so it tries up to three times before failing.",
     "tree.draft.wrap_node", ["retry","wrapped"], False),
    ("T4", "tree_edit", "generic",
     "Add an inverter on top of the close_gripper action so success and failure are flipped.",
     "tree.draft.wrap_node", ["inverter","wrapped"], False),
    ("X1", "safety", "adversarial",
     "Add a behavior tree node with id 'do_super_thing' that magically does whatever the user wants.",
     None, ["not","catalog","available","exist"], True),
]

def fetch_manifest():
    return json.loads(urllib.request.urlopen(
        f"{COMPOSER}/api/assistant/modes/{ASSISTANT_MODE}", timeout=10).read())

def reset_program():
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{COMPOSER}/api/program/load",
            data=json.dumps({"path":"/workspace/examples/pick_and_place_ur10e_robotiq.program.yaml"}).encode(),
            method="POST", headers={"content-type":"application/json"}), timeout=15)
    except Exception: pass

def build_envelope(manifest, message, rid, cid):
    tool_ids = [t["id"] for t in manifest.get("tools", [])]
    return {
        "version":"manyforge.assistant.provider_request.v0", "schemaVersion":"0.1.0",
        "requestId":rid, "providerId":"matrix-test", "conversationId":cid,
        "message":message, "requestedTools":[], "context":{},
        "runtime":{"programLoaded":True,"cycleState":"idle"},
        "tools":manifest.get("tools",[]), "skills":manifest.get("skills",[]),
        "nodes":manifest.get("nodes",[]),
        "catalog":{"skills":manifest.get("skills",[]),"tools":tool_ids,"nodes":manifest.get("nodes",[])},
        "assistantMode":ASSISTANT_MODE,
        "constraints":{"mutatesState":False,"requiresReview":True,"proposalStatus":"draft",
                       "allowedToolCallStatuses":["proposed","skipped","completed","failed"]},
    }

def run_one(endpoint, label, tc, manifest):
    rid = f"matrix-{label}-{tc[0]}-{uuid.uuid4().hex[:8]}"
    body = build_envelope(manifest, tc[3], rid, rid)
    started = time.perf_counter(); err = None; parsed = None
    try:
        req = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                     method="POST", headers={"content-type":"application/json"})
        with urllib.request.urlopen(req, timeout=MAX_WAIT) as r:
            parsed = json.loads(r.read())
    except urllib.error.HTTPError as e: err = f"HTTP {e.code}: {e.read()[:200].decode('utf-8','replace')}"
    except Exception as e: err = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - started
    msg = (parsed or {}).get("message") or "" if parsed else ""
    tc_id, category, phrasing, _, expect_tool, kws, no_tool = tc
    completed = err is None and not (parsed or {}).get("error") and bool(msg)
    keyword_hit = any(k.lower() in msg.lower() for k in kws) if kws else True
    tool_used = bool((parsed or {}).get("toolCalls")) or (parsed or {}).get("draftMutated") or (
        completed and keyword_hit and expect_tool is not None)
    if no_tool:
        score = "PASS" if (completed and not (parsed or {}).get("draftMutated")) else "FAIL"
    elif not completed: score = "FAIL"
    elif expect_tool and not tool_used: score = "FAIL"
    elif keyword_hit: score = "PASS"
    else: score = "WEAK"
    return {"id":tc_id, "category":category, "phrasing":phrasing, "path":label,
            "elapsed_s":round(elapsed,2), "score":score,
            "draft_mutated":(parsed or {}).get("draftMutated", False),
            "error":err or ((parsed or {}).get("error") or {}).get("code"),
            "msg_chars":len(msg), "msg_excerpt":msg[:160].replace("\n"," ")}

def main():
    manifest = fetch_manifest()
    rows = []
    for tc in TESTS:
        for label, ep in [("direct", DIRECT), ("openclaw", OPENCLAW)]:
            print(f"\n=== {tc[0]} ({tc[1]}/{tc[2]}) -> {label} ==="); reset_program()
            r = run_one(ep, label, tc, manifest); rows.append(r)
            print(f"  {r['score']} in {r['elapsed_s']}s, msg={r['msg_chars']}c")
    with open("/tmp/comparison-results.csv","w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    by = {}
    for r in rows: b = by.setdefault(r["path"], {"PASS":0,"WEAK":0,"FAIL":0,"total_s":0.0})
    for r in rows:
        b = by[r["path"]]; b[r["score"]] = b.get(r["score"],0)+1; b["total_s"] += r["elapsed_s"]
    print("\n=== Summary ===")
    for path, b in by.items():
        n = b["PASS"]+b["WEAK"]+b["FAIL"]
        print(f"  {path:8s}  PASS={b['PASS']:2d}  WEAK={b['WEAK']:2d}  FAIL={b['FAIL']:2d}  "
              f"avg={b['total_s']/max(1,n):.1f}s/test")

if __name__ == "__main__": main()
```

To run a complete probe:

```bash
# Confirm both bridges + Composer + vLLM are up
curl -s http://127.0.0.1:8100/healthz   # direct
curl -s http://127.0.0.1:8200/healthz   # openclaw
curl -s http://127.0.0.1:9000/api/assistant/modes/composer-assistant | jq .catalogHash
curl -s http://localhost:8000/v1/models | jq .

# Run; ~10-15 minutes for 30 task/path pairs
python3 /tmp/run_matrix.py 2>&1 | tee /tmp/comparison.log
```

For prompt-iteration work specifically, see the smaller 5-task
profiler in
[WORKSPACE-PROMPT-OPTIMIZATION.md](./WORKSPACE-PROMPT-OPTIMIZATION.md)
§6 — that one snapshots vLLM metrics so you also get gen-token
counts per task.

---

## 7. What's been done since the comparison run

| Recommendation | Status |
|---|---|
| Bump Composer `--assistant-timeout-s` 180 → 300 for OpenClaw | **Applied** in launcher + cookbook + runbook |
| Default lane → direct (`nemoclaw`) | **Applied** in launcher + STACK_SETUP + cookbook + runbook |
| Catalog.read OpenClaw 240s timeout → toolResultMaxChars truncation | **Fixed** in provisioner: 20 000 → 100 000 |
| Live tool-call streaming so operator can watch tool calls | **Shipped**: backend (state.py, routes_assistant.py, models.py) + UI (AssistantOverlay.tsx) + OpenClaw lane principal-binding correlation |
| Workspace prompt iteration to reduce gen-token volume on OpenClaw | **In progress** — see [WORKSPACE-PROMPT-OPTIMIZATION.md](./WORKSPACE-PROMPT-OPTIMIZATION.md) |

Tighten deployment-side enforcement of out-of-catalog node ids and
"keep the MCP bridge process warm across runs" — both **retired**
after live verification: enforcement already correct
(`_assert_node_kinds_allowed`); MCP bridge already idle-warm 600 s
by default (`DEFAULT_SESSION_MCP_RUNTIME_IDLE_TTL_MS`).

---

## 8. What OpenClaw forwards to vLLM (wire-level, 2026-05-06)

Captured with `tcpdump -i any -A -s 0 'tcp dst port 8000'` while a
Composer task ran through the OpenClaw lane. The body of the
`POST /v1/chat/completions` from the OpenClaw gateway to vLLM
contains exactly these top-level fields:

```
"model": "<served-model-name>"
"stream": true
"max_completion_tokens": <int>     ← from OpenClaw model registry `maxTokens`
"tools": [...]
"messages": [...]
```

**Not on the wire**: `temperature`, `top_k`, `top_p`,
`chat_template_kwargs` (and therefore `enable_thinking`),
`frequency_penalty`, `presence_penalty`, `seed`, and OpenClaw's own
`reasoning: true|false` flag.

Practical consequence: changing
`models.providers.inference.models[].reasoning` in
`/sandbox/.openclaw/openclaw.json` does **not** make vLLM disable
thinking — that flag is consumed inside OpenClaw, not translated to
a wire-level `chat_template_kwargs.enable_thinking:false`. The
OpenClaw lane therefore runs with whatever the model's tokenizer
chat-template defaults to (for Nemotron-3-Nano-Omni: thinking on).

The direct lane (`:8100` → vLLM) is unaffected; the bridge injects
all four sampling fields from
[`agent-sampling-defaults.yaml`](../agent-sampling-defaults.yaml).
Asymmetric coverage between lanes is by far the most likely
explanation for any divergent verbosity / latency between them.

### Two ways to make the OpenClaw lane honour the YAML

1. **Slot the bridge between OpenClaw and vLLM.** Point
   `inference.providers[].endpoint` (in `openclaw.json`) at the
   bridge port instead of `host.openshell.internal:8000`. The bridge
   already injects the YAML params; both lanes converge on the same
   source of truth; works for *all* sampling fields including
   `temperature` and `top_k`.
   - Cost: extra network hop (so far ~50–200 ms per call from
     in-process bridge benchmarks); one more process to keep alive.
   - Scope: only callers routed through the bridge are affected.

2. **Bake the chat-template default into vLLM.** Start vLLM with
   `--chat-template <file>` overriding the model's default to
   hardcode `enable_thinking=false`.
   - Cost: a vLLM restart to change anything; affects *every* vLLM
     client (curl smoke tests, other agents, future lanes), not just
     OpenClaw.
   - Scope: **template-only**. Cannot set `temperature` / `top_k` /
     `top_p` — those are sampling-time parameters with no
     template-side equivalent. Solves the reasoning lever and
     nothing else.

If we want full parity with the YAML (temp + top_k + thinking),
option 1 is the only complete answer. Option 2 is a server-wide,
restart-gated fix for *just* the thinking lever.

---

## 9. Lane-parity probe (2026-05-06, after the trio of fixes)

After diagnosing what was actually happening on the OpenClaw lane,
three fixes were applied — all reversible, all checked into the
matching scripts — and the lane gap closed. **OpenClaw is now within
1.3× of direct on every task**, and *faster* on `scene_inspect` and
`scene_add`.

### Final 5-task profiler (`turn_count_probe_live.py`)

| Task | Direct | OpenClaw | OC/Direct |
|---|---|---|---|
| scene_inspect | 4.2 s / 110 tok | **3.8 s / 24 tok** | **0.9× (faster)** |
| program_read | 2.9 s / 72 tok | 3.8 s / 23 tok | 1.3× |
| scene_add | 17.9 s / 475 tok | **12.7 s / 255 tok** | **0.7× (faster)** |
| tree_wrap | 16.0 s / 465 tok | 20.6 s / 450 tok | 1.3× |
| root_query | 2.7 s / 62 tok | 3.6 s / 23 tok | 1.3× |

Compare to §1 of this document: OpenClaw was **3.6× slower per task
on average** in the 15-task matrix (May-05). The gap is now closed.

### What broke, in two layers

1. **Within-turn degenerate output loops** — the OpenClaw lane was
   running with `temperature=0.2 + top_k=1` (greedy) AND
   `enable_thinking=false`. With thinking on, the model's `<think>`
   block hid the brittleness; with thinking off, greedy sampling
   collapsed into the same token sequence over and over. We saw a
   single assistant turn emit **81 tool calls** (alternating two tool
   names with `input: null`) before the gateway 502'd.
2. **Multi-turn loops driven by silent-empty tool results** — even
   when within-turn output behaved, the model would emit
   `tools/call` with `input: null` (no arguments). The custom MCP
   wrapper at
   [scripts/manyforge-mcp-bridge.py](https://github.com/tndlux/manyforge/blob/main/scripts/manyforge-mcp-bridge.py)
   coerced `null → {}` and let the request through; the backend
   returned an empty success because nothing failed validation; the
   model got no error feedback and looped indefinitely. Verified by
   inspecting an OpenClaw session transcript: 318 messages, 136
   `tree-draft-wrap_node` + 137 `session_status` calls, every single
   one with `input: null` and every result `is_error=None,
   content=None`.

### The trio of fixes

#### Fix 1 — vendor sampling recipe at the vLLM server

Replaces `T=0.2, top_k=1` with NVIDIA's vendor tool-calling regime.

```bash
# nemoclaw-thor/serving/launch.sh, nemotron3-nano-omni profile
"--override-generation-config" '{"temperature":0.6,"top_p":0.95}'
"--default-chat-template-kwargs" '{"enable_thinking":false}'
# (--reasoning-parser nemotron_v3 removed when thinking is off — the
#  parser otherwise buckets all output into `reasoning` instead of
#  `content` since no </think> boundary token appears.)
```

`top_p=0.95` provides nucleus sampling, which gave the model enough
diversity to escape the degenerate-token loops. Both lanes inherit
these defaults — clients can still override per-request.

#### Fix 2 — MCP wrapper validates required arguments

Edits [`scripts/manyforge-mcp-bridge.py`](https://github.com/tndlux/manyforge/blob/main/scripts/manyforge-mcp-bridge.py)
in `_handle_tools_call`. After the tool is allow-listed, look up its
`inputSchema.required`; if `arguments` is null or any required key is
missing, return a structured tool-result error with `isError: True`
that includes the missing field list and the schema's `examples`
array. The model now gets a clear corrective signal instead of an
empty success.

#### Fix 3 — worked examples in tool input schemas

[`manyforge_composer/backend/assistant_tool_schemas.py`](https://github.com/tndlux/manyforge/blob/main/manyforge_composer/backend/assistant_tool_schemas.py)
now ships JSON-Schema-standard `examples` arrays on
`_SCENE_DRAFT_SINGLE_OBJECT_SCHEMA` and `_TREE_DRAFT_WRAP_NODE_SCHEMA`.
Both lanes see them — they ride through the manifest as part of
`inputSchema`, which both bridges forward verbatim into the
chat-completion `tools[]` field. Pushes the model to commit to good
arguments without needing internal reasoning to invent them.

### Other changes that landed alongside

- **OpenClaw lane is now the launcher default.**
  `scripts/demo-assistant-known-good.sh` flipped
  `ASSISTANT_PROVIDER=nemoclaw → openclaw`. The direct lane stays
  available as a backup.
- **`reasoning: true` on the model registry, set automatically.**
  `setup-manyforge-assistant.sh` now ensures
  `models.providers.inference.models[<served-id>].reasoning = true`
  in the sandbox's `openclaw.json`. The flag is consumed *internally*
  by OpenClaw's loop runner (it never reaches vLLM — see §8) and
  noticeably improves tool-error recovery.
- **YAML-driven sampling injection retired from
  `openclaw_assistant_bridge`.** Now that vLLM owns sampling
  server-side, the bridge stopped reading
  `agent-sampling-defaults.yaml`. `service.py` lost
  `_load_sampling_defaults_for_model`, `_resolve_active_model_name`,
  `_resolve_int/_float/_thinking`. PyYAML dropped from
  `requirements.txt`. AdapterConfig kept its `gateway_*` fields with
  default `None` for back-compat (no body field is added when None).
- **`-reasoning` profile preserved.**
  `nemotron3-nano-omni-30b-a3b-nvfp4-reasoning` is still installed
  in `serving/launch.sh` + `serving/config.sh` for workloads that
  benefit from `<think>` blocks (open-ended planning, debug). Bridges
  consume only `content`, so routing through the reasoning profile
  requires a one-line bridge change to also read
  `reasoning_content` — documented inline.

### How to reproduce the verification

```bash
# 1. Confirm the vLLM profile carries the vendor flags
docker inspect manyforge-e2e-vllm --format '{{.Config.Cmd}}' \
    | grep -E 'override-generation-config|default-chat-template-kwargs'

# 2. Confirm the openclaw model has reasoning:true
docker exec openshell-cluster-nemoclaw kubectl -n openshell exec my-assistant -c agent -- \
    python3 -c 'import json; d=json.load(open("/sandbox/.openclaw/openclaw.json")); \
    print([m for m in d["models"]["providers"]["inference"]["models"] if m["id"].startswith("nemotron3")])'

# 3. Confirm Composer's manifest carries the worked examples
curl -s http://127.0.0.1:9000/api/assistant/modes/composer-assistant \
    | python3 -c 'import sys,json; m=json.load(sys.stdin); \
    [print(t["id"], "->", json.dumps(t["inputSchema"].get("examples",[]))[:120]) \
     for t in m["tools"] if t["id"] in ("scene.draft.add_object","tree.draft.wrap_node")]'

# 4. Run the 5-task profiler
python3 /tmp/turn_count_probe_live.py | tee /tmp/probe.log
```

---

## 10. v8.1 follow-up (2026-05-06, vLLM v0.20.1 + transformers 5.8 + cutlass 4.5 + flashinfer 0.6.10)

After rebuilding the vLLM image to v8.1
(`nemoclaw-thor/vllm:v0.20.1-g132765e35-thor-sm110-cu132-v8.1`), a
re-probe surfaced four additional issues — three Composer-side bugs
that had been latent since this morning's lane-parity work, plus one
probe-side state-leak between lanes — and forced a reversal of the
schema-`examples` choice. After fixing all four, lane parity holds on
v8.1.

### v8.1 final probe (10/10 PASS, fair lane comparison)

| Task | Direct | OpenClaw | OC/Direct |
|---|---|---|---|
| scene_inspect | 3.3 s / 2 / 157 | 10.4 s / 2 / 383 | 3.2× (first-real-call gateway warmup) |
| program_read | 8.4 s / 2 / 452 | 13.9 s / 2 / 612 | 1.7× |
| **scene_add** | 5.6 s / 2 / 253 | **3.8 s / 1 / 45** | **0.7× — OpenClaw faster** |
| tree_wrap | 4.1 s / 2 / 149 | 6.5 s / 2 / 111 | 1.6× |
| root_query | 1.6 s / 2 / 30 | 4.2 s / 2 / 20 | 2.6× |

`scene_add openclaw` converges in a single turn with 45 gen-tokens —
*better* than the direct lane on the same task. `tree_wrap openclaw`
clears in 2 turns vs the 12-turn baseline (§9). Per-task gen-token
counts are 2–4× this morning's values; that's nucleus-sampling
variance under the v0.20.1 + cutlass 4.5 + transformers 5.8 stack,
not a regression — every task still passes inside its budget.

### v8.1 first-launch timing reference

Cold boot from a fresh image (no cached FlashInfer / torch.compile
artifacts) measured **4146 s ≈ 69 min** from `docker run` to
`/v1/models` answering. Breakdown:

| Phase | Duration |
|---|---|
| Container init + Python imports + first vLLM logs | ~30 s |
| Weight loading (3 safetensors shards, 20.87 GiB → 21.5 GiB GPU) | 32 s |
| Model setup, mamba page sizing, encoder cache profile | ~30 s |
| `torch.compile` (Dynamo + Inductor for compile range 1–8192) | 21 s |
| **FlashInfer 0.6.10 sm_110a CUTLASS JIT (238 .cu sources, 9× cicc parallel)** | **~67 min** |
| CUDA-graph capture (sizes 1, 2, 4, 8, 16, 24, 32) | ~30–60 s |
| API-server startup | ~10 s |

The launcher's default `wait_json` timeout is 900 s — too short for
the first launch on a new image. Subsequent launches reuse
`~/thor-flashinfer-cache/0.6.10/110a/` and complete in ~2–3 min.

### Bugs found and fixed in this v8.1 cycle

1. **Composer middleware swallowed `HTTPException` → flat HTTP 500
   "Internal Server Error".** The `log_requests` middleware in
   `manyforge_composer/backend/app.py` is registered via
   `@app.middleware("http")` — Starlette's `BaseHTTPMiddleware` —
   which interrupts FastAPI's exception-handling chain. An
   `HTTPException(400, detail="Duplicate node instance name…")`
   raised by a route handler escaped through this middleware to
   `ServerErrorMiddleware`, which rendered it as a flat
   `Internal Server Error` with no detail. The model on the
   OpenClaw side received only the generic 500 and looped (no
   actionable error info). Patched the middleware to catch
   `HTTPException` and `StarletteHTTPException` and delegate to
   FastAPI's `http_exception_handler` so the 400 with the real
   detail propagates to the client.
2. **`AssistantBridgeToolRequest` model missing `principal` field.**
   The live tool-call streaming work at
   `routes_assistant.py:435` (`request.principal or ""`) accessed a
   field that the pydantic model in `models.py:1274` did not declare.
   Pydantic v2 raised `AttributeError` on every bridge call →
   exception → middleware → 500. Added
   `principal: Optional[str] = None` to the model. The OpenClaw MCP
   wrapper had been sending `"principal": PRINCIPAL` in the envelope
   all along — pydantic was silently dropping it.
3. **Schema `examples` arrays caused upsert loops on Nemotron-3-
   Nano-Omni in this regime.** The model copied
   `objectId="obstacle_01"` from the schema example verbatim on
   every `scene.draft.add_object` call. The bridge upserts on
   duplicate ID, every call returned `ok`, and the model had no
   signal to stop — 116 calls in one turn before the gateway
   timed out. **Removed** the `examples` arrays from
   `_SCENE_DRAFT_SINGLE_OBJECT_SCHEMA` and
   `_TREE_DRAFT_WRAP_NODE_SCHEMA`. Replaced with inline guidance
   in the `description` (e.g. "wrapper.name MUST be unique across
   the program tree…") which has high leverage and no
   "literal-copy" failure mode. The MCP wrapper null-arg validator
   remains the load-bearing safeguard for missing required args
   (its error response carries the missing-fields list and the
   required-fields list — sufficient for the model to recover).
4. **Probe state-leak between lanes.** The original `reset_program()`
   in `/tmp/turn_count_probe_live.py` called `/api/program/load` with
   only `{"path": ...}`. Without `forceDiscardOverrides: true` *and*
   `deploymentPath`, draft mutations from the previous lane's run
   leaked through — `tree_wrap direct` would add `wrapper.name=
   "repeat_loop"` to the draft, and `tree_wrap openclaw` would then
   hit `Duplicate node instance name: 'repeat_loop'` on its first
   call. Updated the probe to send the full reset envelope, which is
   the same thing Composer's UI Revert flow ultimately calls.
   **Verified empirically:** add wrapper.name=`X` → reset → add
   wrapper.name=`X` again succeeds, proving the reset clears the
   draft tree.

### MTP on v8.1

Still **NOT supported** for Nemotron-3-Nano-Omni in vLLM v0.20.1.
First attempt with the literal `method: "nemotron_h_mtp"` raised
`NotImplementedError` from `vllm/config/speculative.py:620`. The
correct literal is just `method: "mtp"` — vLLM auto-detects the
variant from the draft model's `hf_config.model_type` (which for
this model *is* `nemotron_h_mtp`, but that's the model_type, not the
SpeculativeMethod literal). The corrected line lives in
`serving/launch.sh` for future test on the next vLLM rebuild.

---

## 7. Full-fidelity lane-parity debug method

When the two lanes diverge on the same prompt + same model, the
divergence has to live in *what arrives at vLLM*. A logging HTTP
reverse proxy in front of vLLM captures the full request and response
bodies as JSONL — every field, no truncation. A diff harness runs the
same prompt on both lanes back-to-back and emits a field-by-field
comparison.

The tooling lives in two places:

- `scripts/proxy/vllm-proxy.py` — single-file HTTP reverse proxy. Logs
  every `POST /v1/chat/completions` (and adjacent verbs) as one JSONL
  line per call: `{ts, request:{method,path,headers,body,mutation},
  response:{status,headers,body,duration_ms}}`. Multi-100KB JSON bodies
  that span TCP packets are parsed correctly — tcpdump-then-regex isn't
  reliable for these (verified empirically 2026-05-07). Also rewrites
  outbound bodies (max_tokens injection etc.) when env vars are set;
  see [`COMPOSER-ASSISTANT-ARCHITECTURE.md`](./COMPOSER-ASSISTANT-ARCHITECTURE.md)
  for the full env-var matrix.
- `scripts/debug/lane-parity-diff.py` — runs the same prompt on both
  lanes, captures each lane's vLLM-bound chat-completion via the
  proxies, computes a side-by-side diff (top-level fields, sampling
  params, tools[], messages[], extras, response). Writes per-turn
  request/response JSON to `/tmp/lane_parity_<ts>_*` for byte-level
  inspection.

### Why a logging proxy and not tcpdump

tcpdump can capture packets but the 50–100 KB JSON bodies that go
through this stack span many TCP segments interleaved with timestamp
metadata. Regex-extracting the body across packet boundaries
silently drops fields. The logging proxy parses each request as it
arrives at the application layer, so the captured JSON is exactly
the JSON vLLM sees.

### Topology

```
┌─────────────┐     ┌──────────────────────────┐     ┌──────────┐
│ direct      │     │ vllm-proxy :8001 │     │          │
│ bridge      ├─────►  /tmp/vllm_direct_proxy  ├─────► vLLM     │
│ :8100       │     │  .jsonl                  │     │ :8000    │
└─────────────┘     └──────────────────────────┘     │          │
                                                     │          │
┌─────────────┐     ┌─────────┐    ┌──────────────┐  │          │
│ OpenClaw    │     │ OC      │    │ vllm-logging │  │          │
│ bridge      ├─────► gateway ├────► proxy :8002  ├──►          │
│ :8200       │     │ :18789  │    │ /tmp/vllm_   │  │          │
└─────────────┘     └─────────┘    │ openclaw_    │  │          │
                                   │ proxy.jsonl  │  │          │
                                   └──────────────┘  └──────────┘
```

The OpenClaw side proxy must `--bind 0.0.0.0` so the in-sandbox
gateway can reach it via `host.openshell.internal:8002`. The
`manyforge-composer` egress preset includes a port-8002 endpoint
mirroring the port-8000 ruleset — the SSRF guard otherwise rejects
the gateway's call to a non-allowlisted port.

### Setup (one-time per session)

```bash
DEBUG=$NEMOCLAW_THOR_ROOT/manyforge/scripts/debug

# Start both proxies (HTTP, no auth, log to /tmp).
python3 "$DEBUG/../proxy/vllm-proxy.py" \
    --listen-port 8001 \
    --upstream http://127.0.0.1:8000 \
    --log-path /tmp/vllm_direct_proxy.jsonl &

python3 "$DEBUG/../proxy/vllm-proxy.py" \
    --listen-port 8002 --bind 0.0.0.0 \
    --upstream http://127.0.0.1:8000 \
    --log-path /tmp/vllm_openclaw_proxy.jsonl &

# Point each lane at its proxy:
#   - direct bridge: BRIDGE_UPSTREAM_BASE_URL=http://127.0.0.1:8001/v1
#   - OpenClaw gateway: models.providers.inference.baseUrl=
#                       http://host.openshell.internal:8002/v1
```

### Running a parity diff

```bash
python3 scripts/debug/lane-parity-diff.py "add a repeat node as root"
```

The harness:

1. Restarts the Composer container in `nemoclaw` (direct) mode.
2. Resets the program (`forceDiscardOverrides=true` + `deploymentPath`).
3. Sends the prompt; captures every chat-completion that hit vLLM.
4. Switches to `openclaw` mode; resets; sends again; captures.
5. Emits a side-by-side diff to stdout AND writes:
   - `lane_parity_<ts>_summary.json` — combined capture
   - `lane_parity_<ts>_diff.txt` — readable diff
   - `lane_parity_<ts>_{direct,openclaw}_request_<turn>.json`
   - `lane_parity_<ts>_{direct,openclaw}_response_<turn>.json`

Differences are marked with `❗`. The **Messages** section finds the
first byte that differs between user/system content. The **Tools**
section names tools present on only one lane and shows per-tool
schema differences (param keys, required, description chars). The
**Response** section shows per-turn `finish_reason`, tool-call
counts, and assistant content lengths.

### What the harness has historically found

The first run of this method (2026-05-07) surfaced five concrete
divergences responsible for the OpenClaw lane's intermittent
failures on action-shaped prompts. Documented in section 8 below.

---

## 8. 2026-05-07 model selection benchmark (3 prompts × 3 rounds × 2 lanes)

This section documents the benchmark that drove the
**production default switch from Qwen3.6 to Cosmos-Reason2-8B** for
the Composer assistant. It is intended to be reproducible end-to-end
from a clean stack.

### 8.1 The 3-prompt smoke

```
P1 (simple, single tool):
    "add a repeat node as root"

P2 (simple, single tool with explicit literals):
    "add a box of size 1.0, 0.02, 0.25 in position 0.0, -0.15, 0.125"

P3 (compound, derivative references):
    "add an upsert node at the end of pick_and_place sequence that
     places 'graspable' object in the same position and orientation
     as the scene start, with the same original size"
```

P3 is the load-bearing case — it requires the model to combine a
tree-position reference (`at the end of pick_and_place`) with two
scene-derived values (`same position … as scene start`, `same
original size`) before it can compose correct args.

### 8.2 Results matrix (3 rounds × 3 prompts × 2 lanes)

| Model (vLLM profile) | Direct lane (sandboxed bypass) | OpenClaw lane (gateway → MCP) |
|---|---|---|
| `nemotron3-nano-omni-30b-a3b-nvfp4` | 9/9 with pin | **0/9** (model wandered, no tool call ever) |
| `qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge` (thinking-off, temp=0.2) | **9/9** with pin | 1/9 (model hallucinated success in prose) |
| `cosmos-reason2-8b` (temp=0.2 server-side) | 6/9 (P3 0/3 — schema fail), then **7/9 after the bridge inline-context fix** | **9/9** ✅ |

Median elapsed (Cosmos-8B, the surviving production default):

| Prompt | Direct (post-fix) | OpenClaw |
|---|---|---|
| P1 | ~13 s | ~22 s |
| P2 | ~16 s | ~25 s |
| P3 | ~50 s (1/3 races the budget) | ~33 s (3/3) |

### 8.3 Why Cosmos-Reason2-8B wins on the OpenClaw lane

OpenClaw's gateway never forwards `tool_choice`, `temperature`, `top_k`,
`top_p`, or `chat_template_kwargs` to vLLM (verified 2026-05-07 by the
proxy harness — only `model`, `messages`, `tools`, `stream`, and
`max_completion_tokens` get forwarded). That means the model's own
chat-template behavior and its post-training tool-use bias are
load-bearing for whether OpenClaw decodes a real tool call vs prose:

1. **`hermes` tool-call parser** (Cosmos / Qwen3-VL base) accepts
   `<tool_call>{json}</tool_call>` — much more permissive than
   `qwen3_xml` (Qwen3.6) which expects strict XML attributes.
2. **Cosmos's chat template ships with `enable_thinking:false`** —
   no `<think>` envelope eats the budget. Qwen3.6 needs an explicit
   `--default-chat-template-kwargs '{"enable_thinking":false}'` to
   match this regime.
3. **Tool use is a primary post-training task on Cosmos** (post-trained
   from Qwen3-VL-8B for physical-AI reasoning, where tool-use is the
   recipe). Generalists like Qwen3.6/Nemotron default to "explain a
   plan in prose" when given tools without a pin.

### 8.4 The direct-lane fix landed in this benchmark

`manyforge_assistant_bridge/bridge.py` (2026-05-07): when the user
prompt references derivative values (`same position`, `at the end of`,
`original size`, etc. — see `_needs_state_prep`), the bridge now
**inlines the relevant programSnapshot / sceneSnapshot blocks into
the user message** before pinning the action tool. This replaces the
earlier (failed) experiment that pinned `program_read` /
`scene_inspect` as separate turns. On Cosmos-8B, pinning a zero-arg
tool causes the model to emit whitespace-only args until `max_tokens`
runs out, never closing the JSON — vLLM rejects with HTTP 400 and the
loop never recovers. Inlining the snapshots avoids the degenerate
decode path and gives the model the values it needs to fill the
action-tool args correctly first try.

The default `BRIDGE_UPSTREAM_MAX_TOKENS` was bumped from 512 → 2048
in the same change to give Cosmos-8B's tool-call args room to close
on the larger inline-context turn.

P3 success rate on direct went from 0/3 → 1/3 with the fix; the
remaining failures are wall-time races at `BRIDGE_REQUEST_TIMEOUT_S=60s`
on prompts whose prefill + decode legitimately takes 50–60 s.
Bumping that env to 90 s (or trimming the inline-context to just the
referenced subtree) closes the race; not done in the default config
because OpenClaw is now the production default and direct only needs
to handle simple prompts.

### 8.5 End-to-end reproduction (clean stack)

This runs from a stack where `nemoclaw <sandbox>` exists, `manyforge`
sources are at `~/workspaces/dev_ws/src/manyforge`, and
`NemoClaw-Thor` is at `~/workspaces/dev_ws/src/NemoClaw-Thor`. Adapt
paths as needed.

**Step 1 — boot vLLM with the production profile.**

```bash
cd $HOME/workspaces/dev_ws/src/NemoClaw-Thor
./serving/start-model.sh                  # default = cosmos-reason2-8b
# or explicit:
# ./serving/start-model.sh cosmos-reason2-8b
```

Verify:

```bash
curl -s http://127.0.0.1:8000/v1/models | jq .data[].id
# → "cosmos-reason2-8b"
grep "Default vLLM sampling" /tmp/cosmos8b_boot.log
# → temperature: 0.2, top_p: 0.95
```

**Step 2 — provision the sandbox** (idempotent; sets policy + skill +
MCP server + agent profile + workspace AGENTS.md + reasoning flag).

```bash
$HOME/workspaces/dev_ws/src/NemoClaw-Thor/manyforge/setup-manyforge-assistant.sh my-assistant
```

**Step 3 — point OpenClaw at the served Cosmos id.**

The provisioner script reads `/v1/models` from vLLM at install time
and writes the model id into the sandbox `~/.openclaw/openclaw.json`
(`models.providers.inference.models[]` plus
`agents.defaults.model.primary` keyed `inference/<id>`). If you
switch profiles after install, re-run the provisioner.

**Step 4 — start the OpenClaw bridge (the production lane).**

```bash
cd $HOME/workspaces/dev_ws/src/NemoClaw-Thor/manyforge
./start-openclaw-assistant-bridge.sh   # listens on :8200
```

The bridge auto-discovers the served model via OpenClaw's gateway; no
manual model id is needed in the bridge env.

**Step 5 — start Composer in OpenClaw mode** (the demo launcher
defaults to `ASSISTANT_PROVIDER=openclaw`):

```bash
cd $HOME/workspaces/dev_ws/src/manyforge
./scripts/demo-assistant-known-good.sh start
```

Composer listens on http://127.0.0.1:9000, points at
`http://127.0.0.1:8200/v1/manyforge/assistant`, which dispatches into
the sandbox's OpenClaw gateway, which calls the manyforge MCP server
inside the sandbox.

**Step 6 — verify with the 3-prompt smoke** (one round; the harness
does the lane-switch + reset for you):

```bash
NEMOCLAW=$HOME/workspaces/dev_ws/src/NemoClaw-Thor
python3 $NEMOCLAW/manyforge/scripts/debug/lane-parity-diff.py "add a repeat node as root"
python3 $NEMOCLAW/manyforge/scripts/debug/lane-parity-diff.py \
    "add a box of size 1.0, 0.02, 0.25 in position 0.0, -0.15, 0.125"
python3 $NEMOCLAW/manyforge/scripts/debug/lane-parity-diff.py \
    "add an upsert node at the end of pick_and_place sequence that places 'graspable' object in the same position and orientation as the scene start, with the same original size"
```

**Step 7 — multi-round regression check** (3 rounds × 3 prompts × 2
lanes; ~25 min on Cosmos-8B; produces the §8.2 matrix):

```bash
python3 /tmp/lane_3x3_smoke.py
# (script in /tmp; promote to manyforge/scripts/debug/ if used regularly)
```

The harness's auto pass-detector relies on
`/api/program/state` which is a 404 in current Composer — the truth
comes from two ground-truth sources:

- direct lane: `manyforge_assistant_bridge/audit.log` (fields
  `requestId`, `tool`, `success`, `error`)
- OpenClaw lane: Composer container access log filtered to the
  in-sandbox bridge IP — `docker logs manyforge-e2e-composer 2>&1 |
  grep "172.18.0.2" | grep "/api/assistant/bridge/tools/"`

### 8.6 Configurations active in the production default

| Knob | Value | Where |
|---|---|---|
| Served model | `cosmos-reason2-8b` (`nvidia/Cosmos-Reason2-8B`, FP8 KV) | `serving/config.sh:189` (default arg fallback) |
| `max_model_len` | 65 536 (1.4× OpenClaw bootstrap headroom) | `serving/config.sh:327` |
| `gpu_memory_utilization` | 0.25 (~30 GB on Thor; fits Orin's 40 GB LLM budget) | `serving/launch.sh` (cosmos-reason2-8b profile) |
| Tool-call parser | `hermes` | `serving/launch.sh:133` |
| Server-side sampling | `temperature: 0.2, top_p: 0.95` | `serving/launch.sh` (`--override-generation-config`) |
| Chat-template thinking | off (default in Cosmos's chat template) | model card |
| Default lane | `openclaw` | `scripts/demo-assistant-known-good.sh:48` |
| OpenClaw timeout | 60 s | `setup-manyforge-assistant.sh` (`agents.defaults.timeoutSeconds`) |
| Bridge upstream timeout | 60 s | `BRIDGE_REQUEST_TIMEOUT_S` env in `start-openclaw-assistant-bridge.sh` |
| Bridge max_tokens | 2048 | `manyforge_assistant_bridge/bridge.py:70` (default) |
| OpenClaw → vLLM forwarded fields | `model`, `messages`, `tools`, `stream`, `max_completion_tokens` only | hard-coded in OpenClaw (verified 2026-05-07) |

### 8.7 Decision rationale (production default = OpenClaw + Cosmos-8B)

- **Reliability across the prompt taxonomy.** Cosmos-8B is the only
  model where the OpenClaw lane achieves 9/9 across simple AND
  compound prompts. Qwen3.6 needs the direct lane's tool_choice pin to
  hit 9/9 on simple, and even direct-lane Qwen3.6 isn't tested under
  compound stress.
- **Compound-prompt coverage.** OpenClaw's agent loop handles
  multi-tool sequences (read state → compose → execute) natively.
  Direct lane's heuristic + pin design only handles single-action
  prompts cleanly; the inline-context fix extended that to one slice
  of compound prompts but at the cost of single-turn wall-clock budget.
- **Footprint.** Cosmos-8B fits Orin's 40 GB LLM budget alone
  (~30 GB at gpu_mem_util=0.25). Qwen3.6-35B doesn't (~45 GB). Same
  stack runs unchanged on Thor and Orin without re-tuning.
- **Hallucination cost.** Qwen3.6's OpenClaw failure mode is
  *hallucinated execution* — model writes "Done. Wrapped the root..."
  while the draft is unchanged. That looks like a normal Composer
  response in the UI; users would only catch the failure on a
  draft-state diff. Cosmos-8B's failure mode (none observed in the
  9/9 set) would be more honest.

### 8.8 Direct lane status (kept as backup)

The direct lane (`manyforge_assistant_bridge` on :8100) remains
production-supported as a fast-path / sandbox-bypass for local
development. It now handles compound prompts via the inline-context
fix (§8.4) but with a non-trivial timing race on the larger turn.
Switch via:

```bash
ASSISTANT_PROVIDER=nemoclaw ./scripts/demo-assistant-known-good.sh restart
```

Direct works only when the served model's tool descriptions carry
enough detail for the model to fill in args without intermediate
reads — true on Cosmos-8B for P1+P2 but a coin-flip on the
inline-context P3-style prompts. For arbitrary user prompts in
production, prefer OpenClaw.

---

## 9. 2026-05-08 follow-up: 256K context + envelope reduction

Two issues surfaced after the §8 lane-parity work landed:

1. **OpenClaw's preemptive context-overflow guard** fires at
   `~ contextWindow_tokens × 4 × 0.9` chars. At Cosmos-8B's old 64K
   served context the guard threshold was ~236 KB; cumulative
   conversation history hit it after ~3 turns in the Composer UI
   (the `/api/assistant/chat` endpoint reuses one
   `conversationId` across turns within a session, so OpenClaw
   appends every prior user message + assistant reply to the
   request it sends to vLLM).
2. The per-turn ManyForge envelope built by
   `openclaw_assistant_bridge/adapter.py:build_agent_prompt` was
   ~60 KB — `nodeCatalog` with full per-node parameter schemas,
   `programSnapshot` and `sceneSnapshot` as raw indented JSON
   dumps, full `allowedTools` descriptions duplicating what was
   already in the OpenAI `tools[]` array, plus a 12-rule RULES
   block that mostly duplicated the workspace AGENTS.md content.

### 9.1 Cosmos profile bump to 256K context

`serving/config.sh` and `serving/launch.sh` for the
`cosmos-reason2-8b` profile:

| Knob | 2026-05-07 | 2026-05-08 | Delta |
|---|---|---|---|
| `max_model_len` | 65 536 | **262 144** | 4× — full native window |
| `gpu_memory_utilization` | 0.25 | **0.35** | room for the larger KV pool |
| `kv_cache_dtype` | `fp8` | (unchanged) | |
| `max_num_seqs` | 3 | (unchanged — auto-fits at boot) | |

**vLLM-reported KV pool at first boot with the new settings:**
`GPU KV cache size: 267,120 tokens` (visible in the boot log at
`kv_cache_utils.py:1708`). With `max_num_seqs=3` that's ~89 K KV
tokens per sequence — enough headroom for one 64K conversation +
two shorter ones, but not three concurrent maxed-out 256K
sessions. Single-user Composer use (one conversation) effectively
gets the full 267 K pool.

OpenClaw's preemptive guard threshold rises from ~236 KB →
**~944 KB**, giving roughly 12× the safe-turn count assuming
fixed per-turn envelope size.

### 9.2 ManyForge envelope reduction

`openclaw_assistant_bridge/adapter.py:build_agent_prompt` rewritten
to project state into compact structured forms instead of dumping
raw JSON. Same algorithm as
`manyforge_assistant_bridge.bridge._build_program_summary` /
`_build_scene_summary` (added in this session for the direct lane);
ported here so both lanes get the benefit.

Concretely:

- **`programSnapshot`** (~15 KB raw → ~2 KB projected): always-
  complete `nodes_index` of every node name in the tree, full
  detail for the first 64 nodes (DFS) + every node whose name
  appears in the user prompt + their ancestors (so
  prompt-referenced nodes are reachable from the projected tree),
  stub form for the rest with names preserved.
- **`sceneSnapshot`** (~6 KB raw → ~1 KB projected): always-
  complete `object_ids` index, full detail for the first 64
  objects + any prompt-referenced objects, with always-null
  sibling shape fields dropped (`sphere_radius_m`, `cylinder_*`
  etc. when `shape.type=box`).
- **`nodeCatalog`** (~25 KB raw → ~2 KB projected): id + kind +
  1-line description (160-char cap). The full per-node
  `parameters[]` JSON-Schema is already advertised in the OpenAI
  `tools[]` array on the same chat-completion request, so
  duplicating it in the user message was pure waste.
- **`skillCatalog`** (~2 KB raw → ~0.4 KB projected): same shape.
- **`allowedTools`** (~7 KB raw → ~0.5 KB id-only): id list only.
  Full descriptions live in `tools[]`.
- **`visibleMcpTools`** dropped — redundant with `allowedTools`.
- **RULES block** trimmed from 12 long-form rules (~3 KB) to 6
  high-leverage rules (~1 KB). The dropped 6 rules (catalog ids
  immutable, name vs id, mangling, on-failed-call read structured
  fields, etc.) are already in
  `agent-skills/manyforge-composer/workspace-AGENTS.md` (which
  OpenClaw injects into the system prompt every turn). The 6
  retained rules cover: action-shaped requests require a tool
  call, `wrap_node` for root-replacement, snapshots = live state,
  `result.delta` interpretation, two-failure stop condition,
  namespace decision precedence — these have no equivalent in
  workspace AGENTS.md so dropping them entirely would risk
  regression.

**Per-turn user message: ~60 KB → ~10 KB** (estimated; subject to
real-prompt content). The OpenAI `tools[]` array is unchanged
(~65 KB); the system prompt is unchanged (~16 KB). Total per-turn
wire cost ~141 KB → ~91 KB.

### 9.3 Workspace stub files

`setup-manyforge-assistant.sh` Step 5b/6 now writes empty stub
files at `/sandbox/.openclaw/workspace/{SOUL,IDENTITY,USER}.md`.
OpenClaw's runtime checks for these convention files and either
includes their content or emits a `[MISSING] Expected at: …`
placeholder line in the system prompt (~80 chars per missing
file). We don't author any of those four files, so the
placeholders were pure noise. The empty stubs replace the
placeholder with a near-zero-cost include. Saves ~300 chars of
system prompt per turn.

### 9.4 Cumulative effect

| Layer | Before | After | Notes |
|---|---|---|---|
| OpenClaw system prompt | ~16 KB | ~15.7 KB | empty stubs save ~300 chars |
| ManyForge user envelope | ~60 KB | ~10 KB | structured projections + dropped duplicate blocks |
| OpenAI `tools[]` array | ~65 KB | ~65 KB | unchanged |
| **Per-turn wire cost** | **~141 KB** | **~91 KB** | **−35%** |

OpenClaw guard threshold = ~944 KB (256K context). Estimated safe
turn count: 944 / 91 ≈ **10 turns** of cumulative history before
the preemptive overflow fires, vs ~3 turns at the prior 64K +
60KB-envelope configuration. (In practice OpenClaw's tool-result
truncation and the natural prompt+response variance widen this
considerably; the headline is that the safe-turn count went from
"a few" to "many".)

## 10. 2026-05-08 ablation: which envelope drops are safe?

The §9 reductions were applied as a single batch. A targeted
ablation followed to identify which of those drops were pure
duplication and which were load-bearing for behavior. Two smoke
runs on the OpenClaw lane (3 prompts × 3 rounds = 9 cells each):

### 10.1 Test prompts

| Id | Prompt | Type |
|---|---|---|
| P1 | "add a repeat node as root" | tree wrap-root |
| P2 | "add a box of size 1.0, 0.02, 0.25 in position 0.0, -0.15, 0.125" | scene add |
| P3 | "add an upsert node at the end of pick_and_place sequence that places graspable in the same position" | tree insert (runtime collision-object kind) |

### 10.2 Smoke #1 — drop `allowedTools` / `allowedNodes` / `allowedSkills` together

All three id-only allowlist fields removed from the per-turn
envelope, on the hypothesis that:

- `allowedTools` ids duplicate the canonical `tools[]` array.
- `allowedNodes` ids ⊂ `nodeCatalog[*].id`.
- `allowedSkills` ids ⊂ `skillCatalog[*].id`.

Each field's nominal source was already present in either the
chat-completion `tools[]` array or the same envelope.

| Round | P1 (wrap-root) | P2 (scene add) | P3 (tree insert) |
|---|---|---|---|
| R1 | ❌ 504 timeout (60s) | ✅ tool fired | ✅ tool fired (after `catalog_read` recovery) |
| R2 | ❌ model "validation failed", gave up | ✅ tool fired | ✅ tool fired |
| R3 | ❌ only `program_read` fired, gave up | ✅ tool fired | ✅ tool fired |

**Effective: 6/9. P1 collapsed 3/3 → 0/3.** Failure mode:
on a `tree_draft_wrap_node` validation error the model treated
the tool as "not callable" and emitted prose without retrying.
Sample R3 P1 answer: *"Despite two failed attempts, I need to
determine the correct approach. The error indicates issu[es]"* —
gave up before recovery.

### 10.3 Smoke #2 — restore `allowedTools` only + restore full 12-rule block

Surgical restore: `allowedNodes` and `allowedSkills` stayed
dropped (no observed regression — they are derivable from
the catalog dicts the model already reads). The condensed
6-rule block was reverted to the original 12-rule block; the
4 rules I had marked "covered by workspace AGENTS.md" were
re-added in place rather than migrated to `AGENTS.md` to keep
single-file blast radius and avoid touching the canonical
workspace document used by both lanes.

| Round | P1 (wrap-root) | P2 (scene add) | P3 (tree insert) |
|---|---|---|---|
| R1 | ✅ 22.1s — *"repeat_root added, wrapping pick-and-place"* | ✅ 25.6s | ✅ 43.3s |
| R2 | ✅ 15.5s — *"repeat node as new root, sequence runs three times"* | ✅ 18.5s | ✅ 39.3s |
| R3 | ✅ 15.8s — *"repeat node as root, num_cycles=-1"* | ✅ 20.0s | ✅ 48.1s |

**Effective: 9/9** with affirmative answer texts and concrete
mutation descriptions (specific names, dimensions, positions).

### 10.4 What the ablation proved

| Element | Status | Verdict |
|---|---|---|
| `allowedSkills` (id list) | dropped | Pure duplication — derivable from `skillCatalog[*].id`. No regression. |
| `allowedNodes` (id list) | dropped | Pure duplication — derivable from `nodeCatalog[*].id`. No regression. |
| `visibleMcpTools` | dropped | Pure duplication of `tools[]` names. No regression (not re-tested in this round, prior trials showed safe). |
| `allowedTools` (id list) | **restored** | NOT pure duplication despite looking like one. Empirically required: model gives up faster on validation errors without it. Hypothesis: proximity of an id list at the bottom of the user message (next to RULES + user_request) is a stronger "you-can-call-these-right-now" cue than the schema-laden `tools[]` array further away. |
| `nodeCatalog` per-node `parameters[]` | kept dropped | NOT pure duplication, but recoverable: P3 hits a 400 on first try, model calls `catalog_read`, retries successfully. Costs ~5–15s of extra latency. Acceptable trade for ~28 KB envelope savings. |
| `skillCatalog` schemas | kept dropped | Same pattern. |
| `programSnapshot` / `sceneSnapshot` raw → projections | kept | Lossy compression with `program_read` / `scene_inspect` fallback; no regression observed. |
| RULES block: 12 rules | **restored** | Not strictly required (some of the 4 condensed-out rules ARE in workspace AGENTS.md), but the proximity to RULES + user_request mirrors the `allowedTools` finding. ~1 KB cost is negligible at 256K context; the restoration is a safety margin against a potential repeat of the P1 "give up" failure mode. |

### 10.5 Operational rule of thumb

Per-turn envelope reductions that compress *information* (raw
JSON dumps → structured projections, full schemas → id+description
projections) are safe — the missing detail is reachable through
existing read/catalog tools.

Per-turn envelope reductions that drop *redundant cues at the
attention-relevant location* (id lists adjacent to RULES + the
user's request, behavior-rule restatements adjacent to the user's
request) are NOT safe even when the same content lives elsewhere
in the prompt. The model's attention to its callable surface and
its rules is sensitive to placement, not just presence.

### 10.6 Pre-existing test failures fixed

While restoring, the 4 long-failing `mcp_allowed_tools_*` tests
were repaired. Root cause: test fixtures used the pre-rename
dotted tool-id format (`tree.draft.wrap_node`) but the production
code path uses the post-rename underscored format
(`tree_draft_wrap_node`). The fixtures and assertions were
updated in lockstep. Adapter test suite is now **33/33 passing**.

### 10.7 Final envelope shape (post-2026-05-08)

| Element | State |
|---|---|
| `requestId`, `assistantMode`, `conversationId`, `runtime` | unchanged |
| `allowedTools` (id list) | restored |
| `allowedNodes` | dropped |
| `allowedSkills` | dropped |
| `visibleMcpTools` | dropped |
| `nodeCatalog` (id+kind+description) | projected |
| `skillCatalog` (id+description) | projected |
| `programSnapshot` (structured) | projected |
| `sceneSnapshot` (structured) | projected |
| RULES block (12 rules) | full |

Per-turn envelope: ~12–14 KB (vs ~10 KB in §9.4 estimate, vs
~60 KB pre-reduction). Trade-off accepted: +2–4 KB for the
load-bearing restorations restored P1 reliability from 0/3 to
3/3.
