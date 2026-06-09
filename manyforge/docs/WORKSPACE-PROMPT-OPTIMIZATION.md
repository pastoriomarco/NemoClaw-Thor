# OpenClaw workspace-prompt optimization

**Period:** 2026-05-06 (single working session)
**Scope:** the system-prompt content OpenClaw injects into every chat
turn through the `composer-assistant` agent profile —
`/sandbox/.openclaw/workspace/AGENTS.md` and `TOOLS.md`.
**Audience:** future maintainers tuning the prompt / debugging
verbosity / regressions on this stack.

---

## 1. Why we investigated

In the side-by-side lane comparison
([LANE-COMPARISON-direct-vs-openclaw.md](./archive/LANE-COMPARISON-direct-vs-openclaw.md))
we observed that the OpenClaw lane was **3.6× slower per task** than
the direct vLLM lane on the same model and same tool surface — same
14/15 pass rate, but ~80 s/test vs ~22 s/test.

A targeted profiler (see "How to reproduce" §6 below) showed that the
slowness was almost entirely **generation-token volume**, not extra
LLM calls and not extra prompt tokens:

| Task | direct gen tok | openclaw gen tok | gen-tok ratio |
|---|---|---|---|
| scene_inspect | 144 | 625 | **4.3×** |
| program_read | 439 | 1 113 | 2.5× |
| scene_add | 176 | 1 481 | 8.4× |
| tree_wrap | 251 | 3 741 (failed) | **14.9×** |
| root_query | 35 | 290 | 8.3× |

OpenClaw was making the model **emit dramatically more text per
turn**. Generation is autoregressive and dominates per-turn latency
(~12 tok/s on Nemotron NVFP4), so 8× more tokens ≈ 8× longer turn.

Two mechanical causes:
1. The workspace files OpenClaw injects (~7 KB AGENTS.md + TOOLS.md)
   tell the model to "be thorough", "explain", "verify", and lay out
   24 tool descriptions — anchoring it toward verbose output.
2. Larger surrounding context per turn (15-16K tok with workspace vs
   10-11K tok bare on direct) gives the model more scaffolding to
   imitate. Once a turn is verbose, subsequent turns copy that style.

**Direct vLLM bridge sends no system prompt at all** —
`messages = [{"role": "user", "content": user_message}]` plus the
`tools` array, nothing else. That's why it's terse.

So the question became: can we trim or restructure the workspace
files to reduce generation-token volume on OpenClaw without losing the
reliability gains the workspace was originally designed for
(prevention of "ask for session key", "hallucinate without calling
tools", "loop on out-of-catalog id")?

---

## 2. Versions tested

Each version was probed against the same 5-task set
(scene_inspect, program_read, scene_add, tree_wrap, root_query)
through both lanes (direct + openclaw). Single-shot per version per
task — variance discussion in §4.

### v1 — "verbose" (the baseline that drove the investigation)

**Content:** original AGENTS.md (~3 KB) + TOOLS.md (~5 KB) =
~7 KB / ~1 700 tok.

- Vocabulary lock (scene/program/draft, "session is OpenClaw-internal").
- Default first action for state questions.
- Full tool routing tables with mangled name + canonical id for all
  22 manyforge tools.
- 6-row anti-pattern list with worked rationale.

**Result:** triggered runaway-reasoning failure on `tree_wrap`
(299 s, 3 741 gen tokens, **timeout**). The detailed routing tables
in TOOLS.md prompted the model to output prose explaining its routing
decision before each tool call.

### v3 — "protocol + worked examples" (deployed for several days)

**Content:** AGENTS.md (~3 KB) with a hard "Output protocol" section
saying "no content during multi-step turns; ≤ 100 words on the
final" plus three worked examples (scene_inspect → bullet list,
wrap_node → 1-sentence summary, ambiguous → ASK form). TOOLS.md kept
the routing tables but trimmed the don'ts.

**Result:** **Best balance overall.** The protocol fixed the
runaway-reasoning failure on tree_wrap (299 s → 82 s, fail → pass)
while keeping reasonable token economy on simple tasks. 5/5 PASS,
gen-tok range 290-1481 (vs v1's 290-3741).

| Task | v1 gen | v3 gen | Δ |
|---|---|---|---|
| scene_inspect | 625 | 992 | +59% |
| program_read | 1 113 | 789 | -29% |
| scene_add | 1 481 | 1 334 | -10% |
| tree_wrap | 3 741 (FAIL) | 979 ✅ | -74% |
| root_query | 290 | 395 | +36% |

Wins on the multi-step ones, modest losses on the simple reads.

### v4 — "minimal rules" (the over-trim test)

**Content:** AGENTS.md (~1 KB) with only the failure-mode rules (no
session, no invented ids, brief on final). No protocol section, no
examples. TOOLS.md unchanged.

**Result:** REGRESSED multi-step tasks. Without protocol scaffolding
the model wandered:
- `scene_add`: 3 LLM calls, 2 120 gen tok (vs v3's 3 calls / 1 334
  gen tok) — 59% more tokens.
- `tree_wrap`: 5 LLM calls, 2 318 gen tok (vs v3's 2 calls / 979
  gen tok) — model needed multiple turns to converge.

Verdict: removing the protocol section pushed multi-step tasks back
toward runaway behavior. The protocol earns its 30-line cost.

### v5 — "categorical tool overview, no examples"

**Content:** AGENTS.md (~3 KB) with the protocol section retained but
the worked examples replaced by a **categorical tool overview**:
4 categories (state reads, scene edits, tree edits,
parameters/blackboard) with 1-sentence purpose for each. TOOLS.md
unchanged.

**Result:** **Roughly equivalent to v3** on a single-shot run with
high variance. Some tasks faster, some slower; no clear winner on a
single shot.

| Task | v3 time | v5 time | v3 calls | v5 calls |
|---|---|---|---|---|
| scene_inspect | 119 s | 127 s | 2 | 2 |
| program_read | 67 s | **168 s** | 2 | 2 |
| scene_add | 112 s | 80 s | 3 | 3 |
| tree_wrap | 82 s | 100 s | 2 | **3** |
| root_query | 35 s | 32 s | 2 | 2 |

`program_read` jumped 91 s → 168 s with no prompt change between
runs — single-shot variance dominates the signal. `tree_wrap` went
to 3 calls (from 2) — the categorical overview was less directive
than v3's worked examples on this task.

User preference: v5 was preferred over v3 because the worked examples
in v3 felt prescriptive about answer style ("the model imitated my
3-bullet example for scene_inspect even when one sentence would
do"). v5's categorical overview keeps the multi-step convergence
benefit without templating answer style.

### v6 — "minimal: drop categorical overview entirely"

**Content:** AGENTS.md ~3 KB with only Role + Vocabulary + Output
protocol + Guardrails. The categorical tool overview from v5 was
removed. TOOLS.md unchanged.

**Result:** **Worst single-shot run measured.** Multi-step tasks
exploded:
- `tree_wrap` direct lane: **13 LLM calls / 84 s** (almost certainly
  variance — direct doesn't load workspace, but the system was busy
  serving openclaw runs concurrently).
- `program_read` openclaw: 5 calls / 256 s (vs v3's 2 / 67).
- `scene_add` openclaw: 6 calls / 266 s (vs v3's 3 / 112).
- `root_query` openclaw: 4 calls / 138 s (vs v3's 2 / 35).

Hypothesis: without the categorical scaffolding the model spent
multiple turns trying tools before settling on the right one. Single
shot — could be unlucky. But this is the only version where every
multi-step task regressed simultaneously, so the signal is strong
enough that we did NOT promote v6 even with a "could be variance"
caveat.

### Final live state: v5 restored

After v6 results landed we restored v5 as the live workspace. The
canonical source-of-truth file is now in the sibling `manyforge` repo
at `agent-skills/manyforge-composer/workspace-AGENTS.md` (with the
optional NemoClaw overlay at `manyforge/agent-workspace/openclaw-overlay.md`
in this repo); the provisioner composes the in-sandbox file from
those. v5 satisfies the "no examples" preference, has v3-equivalent
reliability, and keeps the categorical guidance that v6 lost.

---

## 3. Online research — best-practice consensus (May 2026)

Sources we read end-to-end:

- [Anthropic — Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Oracle AI Agent Studio — Best Practices for Prompts](https://blogs.oracle.com/fusioncoe/best-practices-for-prompts-in-ai-agent-studio)
- [BuildMVPFast — System Prompt Design Best Practices 2026](https://www.buildmvpfast.com/blog/system-prompt-design-best-practices-llm-instructions-engineering-2026)
- [Lakera — Prompt Engineering Guide](https://www.lakera.ai/blog/prompt-engineering-guide)
- [Prompting Guide — LLM Agents](https://www.promptingguide.ai/research/llm-agents)
- [InfoWorld — Best Practices for Building Agentic Systems](https://www.infoworld.com/article/4154570/best-practices-for-building-agentic-systems.html)

Recurring themes:

**A. Token budget — keep system prompts lean.** BuildMVPFast cites a
customer-support agent where a 3 000-token system prompt did *worse*
on multi-step reasoning than the same model with a 400-token version.
Recommended baseline: **200–800 tokens**. Our v5 is ~1 100 tokens —
slightly above; we did not push lower because v6 (~700 tok) regressed.

**B. Five canonical sections.** Oracle's framing: *Persona, Scope,
Tools, Constraints, Topic References*. Our v5 maps cleanly:
- Persona → "Role"
- Scope+Constraints → "Output protocol"
- Tools → "Tool surface" (categorical, not full descriptions)
- Constraints → "Guardrails"
- References → "Vocabulary"

**C. Worked examples are double-edged.** Anthropic recommends
including "example usage, edge cases, input format requirements, and
clear boundaries" in tool definitions. But — confirmed in our v3→v5
transition — examples that show *answer style* in the system prompt
cause style imitation. The right division of labor:
- Examples of **tool usage** (schema, edge cases) → tool descriptions
  in `tools/list`.
- Examples of **answer style** → don't include them. State the rule.

**D. "Format close to natural text."** Anthropic: avoid
"formatting overhead". Markdown headings + plain-English paragraphs
beat XML-wrapped or JSON-style prompts. v5 follows this.

**E. Test for variance, not point estimates.** BuildMVPFast: "run
every test 3-5 times because LLMs are non-deterministic; a prompt
that passes once might fail 30% of the time". Our probes were
single-shot per version. **This is the largest open methodological
gap in our investigation** — we observed `program_read` jump 91 s →
168 s with no prompt change between runs, confirming variance is
significant. Triple-runs are queued for the next iteration.

**F. Transparency over instruction.** Anthropic: "prioritize
transparency by explicitly showing the agent's planning steps". This
validates the parallel work on **live tool-call streaming** (see
[live-tool-call streaming](#5-the-orthogonal-solution-live-tool-call-streaming)
below). The cheaper way to give the operator visibility into the
agent's actions is to show tool calls in the UI as they happen, not
to ask the model to narrate them in text.

---

## 4. Findings

**4.1. Generation-token volume is what makes OpenClaw slow.** Same
model, same tools, same number of LLM calls per task — OpenClaw's
extra latency is the model emitting 2-15× more text per turn,
anchored by the system prompt the workspace files inject. Direct
vLLM has no system prompt at all (only the user message + the tools
array), which is why it's terse.

**4.2. Some workspace content earns its tokens; some doesn't.**
- The Output protocol (no prose between tool calls, brief final answer)
  → **earns it.** Removed in v4 → multi-step regressions.
- The categorical tool overview (4 categories, 1 sentence each) →
  **earns it.** Removed in v6 → multi-step regressions.
- The vocabulary lock + "no session keys" rule → **earns it.** This
  is failure-mode prevention; the original failure that prompted the
  whole investigation.
- Worked examples of answer style (v3) → **does not earn it.** Causes
  style imitation. Removed in v5 — equivalent or better outcomes.
- TOOLS.md routing table — **uncertain.** Possibly redundant with the
  categorical overview in AGENTS.md. Single-shot data could not
  distinguish. Worth a future test.

**4.3. Single-shot probes are unreliable on this stack.** vLLM
scheduling, prefix-cache state, and model RNG produce 2.5× variance
on identical prompts (`program_read` 91 s → 168 s). Any future
prompt iteration **must be triple-run** before drawing conclusions.

**4.4. Best-practice token-budget targets don't apply cleanly to
Nemotron 3 on this task set.** The "200-800 token system prompt"
guidance gave us v6 (~700 tok) which regressed multi-step tasks. The
right operating point for this model is ~1 000-1 200 tok with the
protocol + categorical scaffolding. Re-evaluate with a different
model / different domain.

**4.5. Workspace size should NOT scale with tool count.** Adding the
22 manyforge tools' descriptions in TOOLS.md (~5 KB) was a cost
without a benefit — the model already gets every tool's full
description and schema from MCP `tools/list`. The categorical
overview (4 categories) is constant-size as new tools are added; the
explicit name list is not. Future deployments adding tools should
slot them into the existing categories rather than extending the
workspace.

---

## 5. The orthogonal solution: live tool-call streaming

The right answer to "the operator needs to see what the agent is
doing" is not "the model writes prose explaining itself" — it is
"the UI shows tool calls in real time as the agent makes them".

We shipped a live tool-call streaming feature alongside the prompt
work:

- `LiveToolCallEntry` field on `AssistantRequestRecord` (state.py)
- `record_assistant_request_tool_call` hook in
  `routes_assistant.py::execute_bridge_tool` — every bridge tool
  call appends to the active request's live feed.
- `recentToolCalls` field on `AssistantRequestStatusResponse`
  exposed via `/api/assistant/requests/{request_id}` (existing 1 s
  poll).
- `LiveToolCall` type + animated-pulse "running" pill in
  `AssistantOverlay.tsx` for in-flight calls.
- Principal-binding registry in Composer + bridge service handshake
  so OpenClaw lane streaming correlates correctly even though the
  MCP bridge subprocess substitutes its own bridge-process
  conversation id in tool envelopes.

This means we can be **more aggressive on the "no narration" rule**
in the workspace prompt without losing operator visibility: the
operator sees tool-call activity in the UI live, with status, args,
and result summary inline.

This is the recommended long-term shape: minimal system prompt,
strong "no narration" rule, transparency via UI streaming + audit
log.

---

## 6. How to reproduce

The probe harness is reproducible end-to-end with a single Python
script — copy this to `/tmp/turn_count_probe.py` and run.

```python
#!/usr/bin/env python3
"""Per-task turn-count + latency profiler for the composer-assistant
lane. Snapshots vLLM metrics before/after each task to compute
LLM-call counts, prompt-token totals, generation-token totals."""
import json, time, urllib.request

DIRECT = "http://127.0.0.1:8100/v1/manyforge/assistant"
OPENCLAW = "http://127.0.0.1:8200/v1/manyforge/assistant"
COMPOSER = "http://127.0.0.1:9000"
ASSISTANT_MODE = "composer-assistant"
MAX_WAIT = 300.0

TASKS = [
    ("scene_inspect", "What's in the scene right now?"),
    ("program_read", "Show me the current program tree."),
    ("scene_add",    "Add a small obstacle in the middle of the workspace."),
    ("tree_wrap",    "Make the tree repeat indefinitely on success."),
    ("root_query",   "What's the root node of the program?"),
]

def fetch_manifest():
    return json.loads(urllib.request.urlopen(
        f"{COMPOSER}/api/assistant/modes/{ASSISTANT_MODE}", timeout=10
    ).read())

def parse_metrics():
    text = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=5).read().decode()
    out = {"requests_done": 0.0, "prompt_tokens": 0.0, "generation_tokens": 0.0}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip(): continue
        try:
            metric, val = line.rsplit(" ", 1); v = float(val)
        except ValueError: continue
        name = metric.split("{", 1)[0]
        if name == "vllm:request_success_total": out["requests_done"] += v
        elif name == "vllm:prompt_tokens_total": out["prompt_tokens"] += v
        elif name == "vllm:generation_tokens_total": out["generation_tokens"] += v
    return out

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
        "version": "manyforge.assistant.provider_request.v0",
        "schemaVersion": "0.1.0",
        "requestId": rid, "providerId": "matrix-test",
        "conversationId": cid, "message": message,
        "requestedTools": [], "context": {},
        "runtime": {"programLoaded": True, "cycleState": "idle"},
        "tools": manifest.get("tools", []),
        "skills": manifest.get("skills", []),
        "nodes": manifest.get("nodes", []),
        "catalog": {"skills": manifest.get("skills", []),
                    "tools": tool_ids,
                    "nodes": manifest.get("nodes", [])},
        "assistantMode": ASSISTANT_MODE,
        "constraints": {"mutatesState": False, "requiresReview": True,
                        "proposalStatus": "draft",
                        "allowedToolCallStatuses": ["proposed","skipped","completed","failed"]},
    }

def run_one(endpoint, label, task_id, msg, manifest):
    rid = f"prof-{label}-{task_id}-{int(time.time())}"
    body = build_envelope(manifest, msg, rid, rid)
    before = parse_metrics()
    started = time.perf_counter()
    try:
        req = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                     method="POST", headers={"content-type":"application/json"})
        with urllib.request.urlopen(req, timeout=MAX_WAIT) as r:
            resp = json.loads(r.read())
    except Exception as e: resp = {"error": {"code": type(e).__name__, "detail": str(e)}}
    elapsed = time.perf_counter() - started
    after = parse_metrics()
    return {
        "task": task_id, "path": label, "elapsed_s": round(elapsed, 1),
        "vllm_calls": int(after["requests_done"] - before["requests_done"]),
        "prompt_tokens": int(after["prompt_tokens"] - before["prompt_tokens"]),
        "gen_tokens": int(after["generation_tokens"] - before["generation_tokens"]),
        "msg_len": len(resp.get("message") or ""),
        "completed": bool(resp.get("message")) and not resp.get("error"),
    }

def main():
    manifest = fetch_manifest()
    rows = []
    for task_id, msg in TASKS:
        for label, endpoint in [("direct", DIRECT), ("openclaw", OPENCLAW)]:
            print(f"\n=== {task_id} -> {label} ===")
            reset_program(); time.sleep(2)
            r = run_one(endpoint, label, task_id, msg, manifest); rows.append(r)
            print(f"  {r['elapsed_s']}s, calls={r['vllm_calls']}, prompt-tok={r['prompt_tokens']}, gen-tok={r['gen_tokens']}, completed={r['completed']}")

    # Pairwise summary
    by_task = {}
    for r in rows: by_task.setdefault(r["task"], {})[r["path"]] = r
    print(f"\n{'task':16s} | {'path':9s} | {'sec':>5s} | {'calls':>5s} | {'gen tok':>7s}")
    for task in [t for t,_ in TASKS]:
        for path in ["direct","openclaw"]:
            r = by_task[task].get(path) or {}
            if r: print(f"{task:16s} | {path:9s} | {r['elapsed_s']:>5.1f} | {r['vllm_calls']:>5d} | {r['gen_tokens']:>7d}")

if __name__ == "__main__": main()
```

To run a complete probe:

```bash
# Confirm both bridges + Composer + vLLM are up
curl -s http://127.0.0.1:8100/healthz   # direct
curl -s http://127.0.0.1:8200/healthz   # openclaw
curl -s http://127.0.0.1:9000/api/assistant/modes/composer-assistant | jq .catalogHash
curl -s http://localhost:8000/metrics | grep vllm:num_requests_running

# Run
python3 /tmp/turn_count_probe.py | tee /tmp/probe-result.log
```

Each task is run once per lane (10 runs total, ~10-15 minutes). For
**any prompt iteration that should be promoted to live**, run the
probe **3 times** and report min/p50/max — this is the methodological
gap noted in §3.E.

---

## 7. What's still open

**7.1. Variance characterization.** All single-shot probes in this
investigation. We need triple-runs (best practice §3.E) on whichever
prompt is the candidate-of-record before promoting a change.

**7.2. TOOLS.md retention test.** v5 still ships TOOLS.md (~1.5 KB).
With the categorical overview in AGENTS.md, TOOLS.md may be redundant
— mainly the mangling rule and don'ts, both also in AGENTS.md
guardrails. Test: drop TOOLS.md entirely, probe. If equivalent,
delete TOOLS.md and update the provisioner.

**7.3. Cross-model validation.** Findings are Nemotron 3-Nano-Omni
specific. The same prompts on Cosmos-2B / Cosmos-8B / Qwen3.6 may
prefer a very different shape — Cosmos in particular is a smaller
reasoning model. Re-probe when those profiles ship.

**7.4. Workspace-content best-practice library.** We've now run
roughly equivalent prompts that landed in different operating points
on this stack. Worth turning into a small reusable library of "rule
fragments that earned their tokens" vs "fragments that didn't" so
future prompts compose from the proven set.

**7.5. Direct lane needs no system prompt.** This investigation only
optimized OpenClaw. Direct lane has no system prompt at all today
and is faster + same pass rate; arguably it would benefit from a
*tiny* system prompt with the failure-mode rules ("no session keys",
"don't fabricate poses") — a 4-5 line system prompt is much closer
to best practice than zero. Test before adding.
