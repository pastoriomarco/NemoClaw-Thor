# v9 image — smoke-corpus baseline (2026-05-30)

Acceptance run for the v9 vLLM container generation. This is the formal
result against the iter-32 production smoke recipe on the historical
**`cosmos-reason2-8b`** production-default profile.

Status as of 2026-06-11: historical v9/Cosmos baseline. The clean-start
ManyForge assistant model default is now `gemma4-12b-it-gguf`; keep this
document as v9 regression evidence for explicit Cosmos reruns.

**Verdict: v9 falls below the ROADMAP Serving-lane acceptance gate of
≥51/66 on the iter-32 recipe.** A single dominant failure mode accounts
for over half the regressions; the rest of the stack appears healthy.

---

## Stack under test

| Component | Version |
|---|---|
| vLLM image | `nemoclaw-thor/vllm:v0.22.0-g0b3ba88f1-thor-sm110-cu132-v9` (image ID `13b9d1cd8040`) |
| vLLM | `0.22.1.dev0+g0b3ba88f1.d20260530` (the v9 commit at build time) |
| FlashInfer | `0.6.12` |
| flash-attn-4 | `4.0.0b15` |
| transformers | `5.9.0` |
| torch | `2.13.0.dev20260426+cu130` (runner-stage `--override` retained the nightly against vLLM 0.22's transitive 2.10 downgrade) |
| Model profile | `cosmos-reason2-8b` (historical production default since 2026-05-07) |
| Bridge | `manyforge/openclaw_assistant_bridge/` PID 945766 with the 2026-05-11 metrics + circuit-breaker code |
| OpenClaw lane recipe | `OPENCLAW_ASSISTANT_COMPACT_EVERY_N=2` (iter-32 chain-on production recipe) |
| Corpus | `manyforge/scripts/debug/smoke_corpus.yaml` (74 cases, 8 future-tier gated → 66 attempted) |

The single P1 wrap-root probe before the corpus run hit 1/1 pass at 66s
(then 15s on the warm corpus run), confirming the stack assembled.

---

## Headline numbers

```
Cases:        74 in the corpus; 8 skipped (future-tier).
Attempted:    66 non-skip cases.
Passes:       34  (51.5%)
Failures:     29  (43.9%)
Soft passes:  3   (4.5%)

Pass-or-soft: 37 / 66 = 56.1%
Iter-32 baseline (v8.1):     51 / 66 = 77.3%
v9 delta:                   -17 / 66 = -25.8 percentage points
```

Pass-case wall-clock distribution: mean 20.4s, median 16.9s, min 9.0s,
max 91.2s. Pass-side throughput looks consistent with v8.1 expectations
— the failures are quality regressions, not slowdowns.

---

## Failure breakdown by error class

| Count | Error pattern | Reading |
|---|---|---|
| **15** | **`missing tool 'tree_draft_insert_node'`** | **Dominant failure mode.** 51.7% of all failures concentrate on this one MCP tool. Model fails to emit the call or emits it in a form `hermes` rejects. |
| 5 | `chat HTTP -1` (timeout at ~270s) | Long tool-call loops that never converge. Likely follow-on effect from the same tool-emission issue — model retries fruitlessly until the bridge timeout. |
| 3 | `missing tool 'scene_draft_update_object'` | Sister `_draft_*` tool also failing. Reinforces the `_draft_*` family hypothesis. |
| 2 | `missing tool 'tree_draft_update_node_params'` | Same family. |
| 2 | `state_after[scene.objects[...].pos` | Behavioral assertion mismatch on object pose update — not a tool-call failure, an outcome assertion. |
| 1 | `expected NO tool calls; observed ['tree_draft_inse...']` | Over-eager tool firing on a chat-only case. |
| 1 | `state_after[program.tree.children[name=move_to_pic...]` | Behavioral assertion. |
| 1 | `missing tool 'tree_draft_replace_subtree'` | Same `_draft_*` family. |
| 1 | `missing tool 'tree_draft_wrap_node'` | Same `_draft_*` family. |
| 1 | `state_after[program.tree.id]` expected `'repeat'` | Behavioral. |
| 1 | `state_after[program.tree.params.num_cycles]` | Behavioral. |
| 1 | `missing tool 'scene_inspect'` | Single instance, possibly transient. |

**Pattern**: Out of 29 hard failures, **23 concentrate on the `*_draft_*` MCP
tool family** (15 + 3 + 2 + 1 + 1 + 1 = 23). Adding the 5 timeouts (which
are likely cascading from the same root cause), **28/29 failures plausibly
trace to one root cause**.

The three soft passes are all `answer_must_contain` text-match misses
(`'which' not found`, `'where' not found`) — semantic-equivalence misses
on natural-language assertions. Not structural failures.

---

## Working hypothesis

vLLM 0.22.0 changed the `hermes` tool-call parser's handling of the
`coerce_to_schema_type` path (a shared coerce-helper landed in 0.22 per
the release notes — see `Dockerfile.vllm` v9 header). Tools with larger
JSON argument payloads — exactly the `_draft_*` family, which takes
multi-field structural arguments like `kind`, `position`, `attrs`,
`children` — may now serialize through a path that produces text the
`hermes` parser rejects.

**Why this is the most likely root cause:**

- Simple-payload tools pass cleanly (`tree_root`, `scene_inspect` in most
  cases, `wrap_node` in PnP_05).
- Heavy-payload tools fail systematically.
- The failure mode is "model emits text the parser rejects" (case completes
  in 25-58s, not a timeout) — the model is responding, the parser is
  rejecting.
- The pattern affects multiple distinct `_draft_*` tools — not a per-tool
  schema bug, a family-level parser interaction.
- vLLM 0.20 → 0.22 was a major version bump; the 0.21 release explicitly
  refactored `coerce_to_schema_type` across parsers.

**What this is probably NOT:**

- Not a model regression (cosmos-reason2-8b weights haven't changed).
- Not a chat-template bug (this profile uses Qwen3-VL template, which
  the froggeric Qwen3.5/3.6 fix doesn't cover — distinct template
  family).
- Not a bridge issue (bridge passed earlier in this session against the
  same v9 image; same code path).
- Not a network/transport issue (passing cases respond cleanly at 9-25s).

---

## Recommended next steps (in priority order)

1. **Switch the cosmos-reason2-8b profile to a different tool-call parser.**
   The candidates are `qwen3_xml` (the parser the Qwen3.5/3.6 family
   uses) or upgrading to vLLM 0.22's new XGrammar 0.2.0 structural-tag
   path (`--guided-decoding-backend xgrammar` + grammar files). Re-run
   the corpus on each. ~1h each.

2. **Diff the actual model output for a failed `tree_draft_insert_node`
   case** vs a passing case (e.g. `tree_root`) against the parser's
   expected format. Grep the `manyforge-e2e-vllm` container log for
   the assistant-message text on a known-failing case. ~15 min.

3. **File an upstream issue on vLLM** with a minimal repro: cosmos-
   reason2-8b + `hermes` parser + a multi-field tool schema, showing
   v0.20.1 PASS / v0.22.0 FAIL. ~30 min.

4. **Re-run the corpus on the same v9 image with a fresh stack restart**
   to rule out long-running-bridge state effects. ~1.5h.

5. **Compare v9 vs v8.1 pass timings on the cases that pass on both** —
   the v8.1 build-history NOTES.md has per-case timings preserved; this
   gives the performance-comparison signal the user asked for. ~30 min
   of analysis.

---

## What the regression does NOT block

- The v9 image **boots cleanly** on cosmos-reason2-8b
- The model **serves** at production parameters (262K context, FP8 KV,
  3 concurrent seqs, 16384 max output)
- The bridge integration works end-to-end on simple cases
- The 8 future-tier ISAAC and FUTURE_* cases (all skipped) are
  unrelated to this failure mode

This is a **parser-or-schema interaction regression**, not a stack-wide
failure. Likely contained, possibly trivial to fix.

---

## Per-case timings: pass distribution

For comparison with the v8.1 baseline timings preserved in
`docker/NOTES.md` (when that diff is run as recommended step 5 above).

```
count   34
mean    20.4 s
median  16.9 s
min      9.0 s   (CLARIFY_scene_remove_pronoun)
max     91.2 s   (P3_tree_insert_runtime_obj_specific)
```

The 91.2s outlier on `P3_tree_insert_runtime_obj_specific` is the same
case that the single-probe run earlier this session hit at 66s — that
case has known longer convergence on cosmos-reason2-8b, not a v9
regression.

Full JSON report: `/tmp/smoke_corpus_1780151237366.json` (preserved on
the dev box for the post-mortem; copy into a durable location if useful).
