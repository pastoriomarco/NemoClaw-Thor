# gemma-4-12b-it (GGUF) — composer-assistant hardening: cascade, tool-id ABI, timeouts, and a fail-open history budget

**Date:** 2026-06-05 → 2026-06-06
**Model:** `unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL` + `unsloth/gemma-4-E2B-it-GGUF` speculative draft, llama.cpp (`ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-thor`) on Jetson Thor. **128K context** (`-c 131072`), q8_0 KV cache, flash-attn on. Served id `gemma4-12b-it-gguf` on `:8050`; vllm-proxy mutator on `:8000`.
**Lane:** openclaw / tools surface (composer `:9000` → openclaw bridge `:8200` → in-sandbox OpenClaw 2026.5.22 gateway → proxy `:8000` → model `:8050`).
**Harness:** `manyforge/scripts/debug/smoke_corpus_runner.py` (66 active cases; PnP build chain = 19 active). Stack brought up by `manyforge/scripts/launch.sh`.

This is the umbrella write-up for the multi-day hardening pass. Two earlier evidence dirs cover the first findings:
`../2026-06-05-gemma4-12b-it-gguf/` (3-lane matrix) and `../2026-06-05-gemma4-12b-128k-fix1-pnp/` (bridge fix #1 + tool-id root cause).

---

## Headline results (openclaw / tools, gemma-4-12b)

| Run | Config | Result |
|---|---|---|
| Morning 3-lane matrix | 64K, no proxy | openclaw-tools **37/66 (56.1%)** |
| **Run 2 (best)** | 128K, proxy, dedupe + 285s, **no history guard** | **48/66 (72.7%)** |
| Run 3 | 128K, **history guard @200k HARD-FAIL (413)** | 44/66 (66.7%) |
| PnP @200k fail-open (rungs 1-3) | guard too tight; couldn't shed user-context | PnP **8/19 (42.1%)** |
| PnP @auto-guard fail-open (rungs 1-4) | **guard auto-sized 90% ctx (471,859), never 413** | PnP **10/19 (52.6%)** |
| PnP reference (run 2, no guard) | — | PnP **12/19** |

**Bottom line: no configuration beats run 2's 72.7%.** The residual failures are gemma-bound (deep-chain reliability, bounded-autonomy, arg-shape), not infrastructural. The history-budget work's value is **graceful degradation under context pressure**, not a pass-rate gain.

---

## Fixes landed this pass (all committed)

| Commit | Repo | What |
|---|---|---|
| `7a14274` | NT | **fix #1** — bridge clears loop-history on successful turn (stops chained-conversation cascade) |
| `6a21f8b` | NT | **fix #2** — proxy normalizes streamed nested MCP tool ids (SSE-aware) |
| `1d8a020` | NT | **fix #3** — proxy configurable upstream timeout (default 300s) + initial oversized-history guard |
| `3ade005` | NT | bridge quarantines poisoned sessions |
| `f2d29fb` | NT | **fail-open history trim ladder (never 413)** + conversationId tagging |
| `205240f` | MF | composer dedupes repeated bridge read results (catalog/state) |
| `a0a7e8a` | MF | launcher passes proxy upstream timeout |
| `f8f4c3d` | MF | **launcher auto-sizes history budget to model context (off/auto/specific)** |

---

## Findings (in the order they were uncovered)

### 1. Chained-conversation cascade (bridge) — fix #1
The openclaw bridge's loop detector keyed history on `(conversationId, assistantMode)` and persisted it across turns. A flow that reuses one conversationId across distinct prompts (the PnP build chain) legitimately repeats a tool family; fingerprints accumulated and tripped `bridge_loop_detected_stop` mid-chain, after which every later turn short-circuited (409 → composer 502) **without invoking the model** — a deterministic 16-case cascade. **Fix:** clear the conversation's loop-history after a turn that reaches a successful (200) answer; failing turns never reach that path, so genuine stuck-loop detection is preserved. Cascade went to **0**.

### 2. Tool-call id ABI: gemma emits malformed ids, and the normalizer was SSE-blind — fix #2
With the cascade gone, the PnP block still failed `tree_draft_insert_node not observed`. The model's self-explanation ("I omitted the id") was a **hallucination** — the raw gateway log showed the `id` was **present but mis-formatted**: kebab-cased (`manyforge__scene-draft-add-object`) or unprefixed (`tree_draft_insert_node`). A proxy normalizer existed but **never fired**: OpenClaw streams `tool_calls[*].function.arguments` **one token per SSE `data:` event**, so the contiguous `"id":"…"` the text-regex needed never appears in the body. **Fix:** an SSE-aware normalizer that reassembles per-tool-call arguments across chunks, canonicalizes the id, and re-emits — plus flat-tool dashed patterns (`program-read`, etc.). Confirmed firing live (`dash_to_underscore`).

### 3. Proxy upstream timeout was hardcoded below the case budget — fix #3
The proxy's per-call socket timeout was hardcoded `200.0s`, **below** `ASSISTANT_TIMEOUT_S=300`. On the gemma path a single prefill-heavy turn (large accumulated context; observed ~446s model-side) exceeded 200s, so the proxy aborted (`chat HTTP -1`) before the case budget was spent. **Fix:** `MANYFORGE_PROXY_UPSTREAM_TIMEOUT_S`, default 300, launcher sets it ~15s **below** the case budget so the proxy fails first and frees the single llama.cpp slot deterministically; defensive parse + explicit `proxy_upstream_timeout` event + deterministic socket close.

### 4. launch.sh model selection is env-driven; `cosmos` is an alias label
`launch.sh` does not prompt for model/lane/mode — they are env vars (`MODEL_PROFILE`, `ASSISTANT_PROVIDER`, `OPENCLAW_ASSISTANT_TOOL_SURFACE`), defaults `cosmos-reason2-8b` / `openclaw` / `tools`. A `MOODEL_PROFILE` typo (and underscore-vs-hyphen in the value) silently fell back to cosmos. Also: the **served id** reflects the real engine (`gemma4-12b-it-gguf` via llama.cpp `--alias`), but the **OpenClaw sandbox registry** labels every model `cosmos-reason2-8b` (bakeoff convention) — harmless because llama.cpp ignores the request `model` field. **Always verify the model at the container level** (`docker inspect … .Args`), not the served id.
Reproducer: `MODEL_PROFILE=gemma4-12b-it-gguf bash scripts/launch.sh restart --lane manyforge-only --assistant on --scenario ur10e-scene-authoring --non-interactive --yes`.

### 5. The history-budget guard journey (the core of this pass)
- A history-budget guard at **200k chars HARD-FAILED (413)** over-budget requests → wiped the PnP block (cases that passed without it now 413'd). 44/66 → and 8/19 on PnP.
- **Why 200k was wrong:** the deep PnP requests are **~420k chars ≈ 105k tokens — 80% of the 128k context. They fit.** They only tripped an artificially tight guard (~50k tokens, 38% of context).
- **What the bloat actually is:** not the catalog (a single `catalog_read` dump was 109k in one early sample, but that was unrepresentative). The deep-chain bloat is **accumulated `user`-message context** — the `<nemoclaw-runtime>` preamble + periodic `MANYFORGE_STATE_CONTEXT` state dumps (≈127k across ~9 user msgs; assistant reasoning was a trivial ~1k). Trimming only reads + reasoning can't shed it.
- **The fix is two-fold:** (a) **auto-size the guard to ~90% of context** so context-fitting requests forward untouched; (b) make the proxy **fail-open** — shed re-fetchable content, then hard-trim oldest, then forward anyway + warn. **Never 413.**

### 6. conversationId IS propagated (correcting an earlier wrong conclusion)
The model request is a raw OpenAI body with **no top-level conversationId**, and OpenClaw doesn't inject a `user`/header (its inference config is empty; session UUIDs stay internal). **But** the composer injects a per-request **manifest** (a list-style `user` message) that contains `"conversationId": "…"` — present and extractable in **100% (69/69)** of requests. So the proxy can read it (no OpenClaw patching, no time-window heuristic) and tag trim/truncate events for exact per-conversation correlation of the user warning.

---

## The fail-open history trim ladder (current behavior)

Proxy `vllm-proxy.py`, gated on `MANYFORGE_PROXY_MAX_REQUEST_CHARS > 0`:

1. **Rung 1 (always):** stub OLD re-fetchable read results (`catalog_read`, `program_read`, `scene_inspect`, …), keep the latest of each kind.
2. **Rung 2 (if over):** stub ALL re-fetchable reads.
3. **Rung 3 (if over):** stub OLD assistant reasoning (keep `tool_calls` + latest reasoning).
4. **Rung 4 (if over):** hard-trim OLDEST messages until under budget, repairing orphan tool results. Any residual over-budget single message is **forwarded anyway** with a `userWarning`. **Never 413.**

Safety: only re-fetchable READ results are elided (mutation/state never); envelopes (`role` + `tool_call_id`) preserved → OpenAI tool_call↔result pairing intact. Every shed logged (`proxy_history_trimmed` / `proxy_history_truncated` with `rungs`, `chars_shed`, `conversationId`, `userWarning`).

**Budget selector** (launcher `assistant.sh`, `MANYFORGE_PROXY_HISTORY_BUDGET`):
- `auto` (default) → `THOR_TARGET_MAX_MODEL_LEN × 4 chars/token × PCT%` (default 90) — scales per profile (gemma 128k → 471,859; 256k → 943,718; 32k → 117,964).
- `off` → disabled. `<integer>` → exact chars. Explicit `MANYFORGE_PROXY_MAX_REQUEST_CHARS` wins. Graceful fallback to off if context size unavailable.

**Validation (PnP subset, auto-guard 471,859, fail-open):** the wipe is gone — PnP_01–10 pass with **real durations** (PnP_08 @92s, not the prior 4.6s guard-reject), **0×413**, **11 graceful rung-4 truncations** on the deep tail (dropping 1→11 oldest turns as the chain grew). The gap to run 2 (10 vs 12) is gemma variance: the two diverging cases (PnP_11, PnP_13) failed at real durations with **zero trim firing on them** — PnP_11 genuinely missed an insert, breaking the chained state for PnP_12–15 downstream. The trimmed deep cases (16/17/18/20) match exactly the classes run 2 also failed (300s case-budget timeouts + gateway 502).

---

## Failure taxonomy (what's left, all gemma-bound)
- **bounded-autonomy** — gemma acts (emits tool calls) when an underspecified `_generic` prompt expects it to ask. Dominant non-PnP class.
- **arg-shape** — flat snake_case vs nested objects (`shape.box_dims`); largely masked in openclaw lanes by tool-call coercion, surfaces in direct lane.
- **deep-chain reliability / semantic cascade** — one genuine insert miss breaks state for dependent chained cases.
- **case-budget timeouts (300s)** — deepest PnP/CUR cases are simply slow at large context (many turns), independent of the proxy timeout.

None of these are fixed by the proxy/guard; they are model-capability limits.

## Remaining work
- **Composer warning UI + reset button (Phase B):** proxy already emits `proxy_history_truncated` with `conversationId` + `userWarning`. Remaining: composer/bridge correlate by conversationId and surface the banner in `AssistantOverlay`; a reset button (cancel in-flight via the existing `cancel_request` path, rotate conversationId, no summary port).

## Reproduce
Bring the stack up on gemma (auto-guard is the default):
```
MODEL_PROFILE=gemma4-12b-it-gguf bash manyforge/scripts/launch.sh restart \
  --lane manyforge-only --assistant on --scenario ur10e-scene-authoring --non-interactive --yes
python3 NemoClaw-Thor/manyforge/scripts/debug/smoke_corpus_runner.py --report <out>.json   # full 66
python3 …/smoke_corpus_runner.py --filter '^PnP_' --report <out>.json                       # PnP subset
```
Artifacts here: `run2-full-corpus-*` (72.7% best), `run3-guard200k-hardfail-stdout.txt`, `pnp-failopen-200k-stdout.txt`, `pnp-autoguard-failopen-*` (the fail-open recovery).
