# Phase 4 — Hermes lane bring-up + longitudinal design

**Status:** implementation landed; pipeline fixes validated; clean full-corpus
empirical gate pending an operator-driven run.
**Anchor model:** cosmos-reason2-8b (parity); production candidate gemma4-12b-it-gguf (QAT).
**Companion docs:** [THREE-LANE-MIGRATION-PLAN.md §Phase 4](./THREE-LANE-MIGRATION-PLAN.md#phase-4--hermes-lane-bring-up-5-7-days), [PHASE-0.5-HERMES-SPIKE.md](./archive/PHASE-0.5-HERMES-SPIKE.md).

## What landed (code-complete, unit-verified)

The Hermes lane is implemented and behind the `HERMES_LANE_PHASE4_ENABLED`
opt-in gate. Components:

| Piece | Path | Notes |
|---|---|---|
| Session dispatcher | [`lanes/hermes/session_dispatcher.py`](../lanes/hermes/session_dispatcher.py) | Async client for Hermes' native `/v1/runs` API: start → run_id, `/events` SSE stream, status, `/stop`, `/approval`. Wire-shape assumptions centralised in `HermesRunsContract` for one-line reconciliation with live `/v1/capabilities`. |
| Progress observer | [`lanes/hermes/progress_observer.py`](../lanes/hermes/progress_observer.py) | Maps SSE lifecycle events → universal audit `toolsObserved[]` (prefix-stripped via `common.tool_calls.strip_mcp_prefix`) + `hermes-session-events.jsonl` (skill/memory/cron/delegation). Best-effort: unknown events dropped. |
| Transport | [`lanes/hermes/transport.py`](../lanes/hermes/transport.py) | `HermesTransport(AssistantTransport)`. `dispatch` submits the run and consumes the event stream to completion (Hermes owns the agent loop); `normalize_tool_calls` is `[]` by design (bridge never sees per-turn tool calls). |
| Turn engine | [`lanes/hermes/engine.py`](../lanes/hermes/engine.py) | Dependency-light per-turn logic (envelope→prompt→run→observe→envelope); circuit breaker; cancellation registry. Unit-tested without httpx/fastapi. |
| Bridge service | [`lanes/hermes/service.py`](../lanes/hermes/service.py) | Thin FastAPI on `:8300`: `POST /v1/manyforge/assistant` (+ `/{id}/cancel`, `/healthz`). Persists audit + session-events. |
| Bring-up | [`../../scripts/setup-hermes.sh`](https://github.com/pastoriomarco/manyforge/blob/main/scripts/setup-hermes.sh) (dev_ws) | The spike's strict order-of-ops: onboard → policy-add → cp bridge → inject `mcp_servers`+`base_url`+`API_SERVER_KEY` **before** gateway → `recover` (not `rebuild`) → verify subprocess + catalog. |
| Launcher | `dev_ws/manyforge/scripts/lib/assistant.sh::start_bridge_hermes` | Starts the `:8300` bridge host-side; adopts an externally-started one via `/healthz`. |
| Composer wiring | `dev_ws/.../assistant_provider.py` | `LANE_REGISTRY["hermes"].inert=False`; `HermesAssistantProvider` (same envelope, longer agent-loop timeout) routed in `build_assistant_provider`. |
| Longitudinal harness | [`../scripts/debug/longitudinal_hermes.py`](../scripts/debug/longitudinal_hermes.py) | This document's measurement tool (§ below). |

Resolved open questions (from the spike): **Q4 session API = `/v1/runs`** (the
only surface emitting structured lifecycle events; `/api/sessions/{id}/chat`
does not exist in 0.14.0). **Q6 `API_SERVER_KEY`** = env-var bearer key, injected
by `setup-hermes.sh`. **Q3 (`--tool-call-parser hermes` on vLLM for cosmos)**
remains a live probe — it does not block the lane code (the bridge sends the
prompt to Hermes' runs API; the parser is a vLLM-side serving detail), but the
5-case probe should run before declaring the cosmos-on-Hermes wire shape sound.

## Longitudinal corpus design

The per-turn smoke corpus judges every lane on stateless per-case pass rate —
the wrong yardstick for Hermes, whose distinctive value is memory + skills +
cron + todo + delegation compounding across a session sequence (plan principle
#3). The longitudinal harness ([`longitudinal_hermes.py`](../scripts/debug/longitudinal_hermes.py),
corpus [`longitudinal_corpus.yaml`](../scripts/debug/longitudinal_corpus.yaml))
measures the metric Hermes is *allowed* to win on.

**Shape.** N sessions × up to M turns. Every session attempts the **same
repeated pattern** (default: "wrap the root in a `repeat` node `loop_root`").
A preference is stated **once** on session 1 turn 1, then never repeated. Each
session is a distinct `conversationId`; within a session, turns share it so
Hermes memory accumulates. Completion = the expected manyforge tool firing
(`tree_draft_wrap_node`), observed in `hermes-session-events.jsonl` and
attributed by `conversationId`.

**Metrics** (the harness computes + reports):

- **Turns-to-completion trend** — avg turns first-half vs second-half of the
  session sequence. A downward trend = compounding (`compoundingObserved`).
- **Memory hit-rate** — fraction of sessions completing in **1 turn** (pattern
  recalled, not re-derived).
- **Skill emergences** — `skill_created` events across the run.
- **Memory writes / cron fires / delegations** — the other distinctive signals.

**Determinism hazard.** Hermes memory persists in `/sandbox/.hermes/`. Run with
`--reset-hermes-state` for a clean baseline; omit it to test on top of existing
state. The per-turn smoke parity run (below) is always reset for fairness.

## 2026-06-08 pipeline validation status

The per-turn Hermes path has been debugged to the point where the next useful
signal is a clean full-corpus run, not more insert-family micro-tuning. Four
pipeline fixes are validated:

1. **Native-MCP dispatch primer.** Hermes now receives guidance for its own MCP
   surface instead of an OpenClaw-shaped tool-search flow.
2. **Run-lifecycle dispatcher.** `SessionDispatcher` treats only `run.*`
   lifecycle events as terminal. Tool events such as `tool.completed` remain
   observable but no longer stop recovery loops.
3. **Lean catalog Rule 5.** `nodeCatalog` is the lean node-kind chooser. Full
   parameter schemas come from `catalog_read`; the prompt no longer claims that
   stripped catalog entries contain the same payload as `catalog_read`.
4. **Catalog-read loop breaker.** Repeated identical read-only catalog fetches
   are interrupted so Hermes acts instead of spending the turn on successful
   but redundant discovery.

The insert-family probe after these fixes settled at **2/3 effective**:
`P3_tree_insert_runtime_obj_specific` soft-passed,
`TREE_insert_runtime_medium` passed, and
`INSERT_position_first_specific` failed by choosing the wrong node kind
(`command_gripper` instead of `wait_for_signal_bool`). That remaining failure
is classified as model comprehension unless another lane/model passes it under
the same lean-catalog contract.

Do not switch the baseline back to full inline node params. Inline params also
reached 2/3 on the probe, but with larger prompts and slower cases. The
lean-catalog + loop-breaker path gives the same quality signal with lower
prompt cost and keeps the catalog contract truthful.

## Clean full-corpus protocol

Use this sequence for the next publishable comparison:

1. Stop any contaminated exploratory smoke. Reset Hermes session/memory state
   for the per-turn parity run.
2. Run the **75-case smoke corpus** through Composer with
   `ASSISTANT_PROVIDER=hermes`, `HERMES_LANE_PHASE4_ENABLED=true`, and
   self-heal enabled. Monitor live for MCP breaker trips, repeated validation
   400s, stuck read-only loops, and orphaned runs.
3. Capture a Hermes taxonomy, not only a scalar score: first-try rate,
   effective rate, pass / soft-pass / fail, latency distribution, heal count,
   tool retry count, `catalog_read` loop-breaker count, MCP breaker events, and
   failure buckets (`act-vs-clarify`, node-kind choice, param fill,
   assertion/text-only, infrastructure).
4. Restore OpenClaw to its production tools-mode configuration after the Hermes
   run.
5. Run same-day OpenClaw and Direct baselines on the same corpus, model/profile,
   proxy profile, self-heal policy, and host. Compare only clean same-day
   reports; keep the contaminated Hermes traces as diagnostic evidence.

Before publishing lane claims, close these review items:

- **`toolsObserved` telemetry parity.** Hermes must populate
  `toolsObserved[]` from Composer bridge-tool callbacks, matching Direct and
  OpenClaw. Hermes progress events are useful augmentation, not the hard source
  of mutation truth.
- **MCP breaker tuning.** Repeated validation 400s should back off or
  quarantine the run so a bad insert-family case does not create cross-case
  breaker contamination.
- **Contract probe.** Add a pre-run probe that checks the live assistant mode
  and MCP surface: catalog hash, expected tool IDs, lean `nodeCatalog` shape,
  `catalog_read` availability, and representative parameterized node schemas.

## Gate (plan §Phase 4) and how to run it

The gate has two halves; both need a live Thor host with a warm model and a
provisioned Hermes sandbox (operator-driven — the user drives all loads):

1. **Per-turn smoke ≥ 40/66 with memory disabled** (sanity floor only — Hermes
   is not optimised for stateless turns). Run the existing
   `smoke_corpus_runner.py` with Composer pointed at the Hermes lane
   (`ASSISTANT_PROVIDER=hermes`, `HERMES_LANE_PHASE4_ENABLED=true`), Hermes
   memory off / reset.
2. **Longitudinal harness shows measurable session-over-session improvement OR
   an explicit "no improvement" finding** with diagnosis:
   ```
   python3 scripts/debug/longitudinal_hermes.py --sessions 10 --turns-per-session 8 \
       --reset-hermes-state --report /tmp/hermes-longitudinal.json
   ```

Bring-up first:
```
export HERMES_LANE_PHASE4_ENABLED=true
scripts/setup-hermes.sh hermes-assistant          # provisions the sandbox
export API_SERVER_KEY=...                          # printed by setup-hermes.sh
ASSISTANT_PROVIDER=hermes <launch.sh restart>      # starts the :8300 bridge
```

## Results — TBD (live run)

> Pending an operator-driven run on the Thor host. Record the
> `(OpenClaw, Hermes, NemoClaw, vLLM, model, proxy-profile)` tuple (plan §10)
> with the numbers.

| Metric | Value | Notes |
|---|---|---|
| Per-turn smoke (memory off) | TBD / 66 | sanity floor ≥ 40/66 |
| Q3 `--tool-call-parser hermes` (cosmos, 5-case) | TBD | structured `tool_calls[]` rate vs default parser |
| Sessions completed | TBD / N | |
| Avg turns-to-completion (first → second half) | TBD → TBD | downward = compounding |
| Memory hit-rate | TBD | sessions completing in 1 turn |
| Skill emergences | TBD | `skill_created` events |
| Compounding observed | TBD | the Hermes-wins signal, or documented "no" |

## Known caveats to validate live

- **MCP registration is gateway-startup-only** (spike probe 3 online finding):
  the `mcp_servers` watcher runs under the interactive CLI, not the gateway. So
  `mcp_servers` must be present **before** gateway start, and changes need a
  `recover`. `setup-hermes.sh` enforces this order.
- **`/v1/runs` request-body + event-type names** were not live-enumerable in the
  spike. They are centralised in `HermesRunsContract` (dispatcher) and
  `HermesEventTaxonomy` (observer); if `/v1/capabilities` on the live gateway
  differs, reconcile those two classes — no logic changes needed. The observer
  is best-effort, so a mismatch degrades audit richness without breaking turns.
- **Effect tracking** is via Composer's `/api/assistant/bridge/tools/{toolId}`
  callback log, NOT Hermes-visible tool names — the hard correctness source
  (plan §6). The progress observer is augmentation.
