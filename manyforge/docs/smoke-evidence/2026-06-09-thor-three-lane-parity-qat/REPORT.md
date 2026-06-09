# Three-lane parity + scorer-strictness sweep — direct / openclaw / hermes (2026-06-09)

> Companion to [`2026-06-07-thor-orin-smoke-sweep-qat`](../2026-06-07-thor-orin-smoke-sweep-qat/REPORT.md)
> (which swept **models** on one lane). This run fixes the **model** (gemma-QAT) and
> sweeps the **lane**: direct / openclaw / hermes — asking *are the lanes
> comparable?* The answer surfaced a scorer-strictness artifact, not a model gap.

**Hardware:** Jetson AGX Thor. **Model:** `gemma4-12b-it-gguf` (QAT), temp=1.0,
single llama.cpp slot, spec-decode, 131072 ctx.
**Method (identical across lanes):** full 75-case `smoke_corpus.yaml` (9 future-tier
skipped → 66 scored), `--self-heal` ON, recovery-turn OFF. Lane selected by
`ASSISTANT_PROVIDER`; vLLM + proxy shared across lanes (no model reload between
lanes — surgical composer/bridge re-point, see
[`docs/operations/LANE_BRINGUP.md`](../../operations/LANE_BRINGUP.md)).
**Evidence (this dir):** `three-lane-comparison.txt` (per-case cross-lane matrix
+ corrected re-score). Per-lane run JSONs (`{hermes,openclaw,direct-baseline,
direct-fixed-scorer}.json`) were the ephemeral `/tmp` smoke outputs, not retained
here (matching the REPORT-only convention of prior sweeps).

## Scoreboard

| Lane | first-try | effective | soft | fail | per-case median | full 66-case run |
|---|---|---|---|---|---|---|
| **hermes** (native MCP) | 78.8% (52/66) | **81.8%** (54/66) | 2 | 12 | 67.5s | ~80 min |
| **openclaw** (gateway discovery) | 66.7% (44/66) | 77.3% (51/66) | 7 | 15 | 36.8s | ~53 min |
| **direct** — *original scorer* | 50.0% (33/66) | 57.6% (38/66) | 5 | 28 | **11.4s** | **~30 min** |
| **direct** — *corrected scorer* | **59.1%** (39/66) | **71.2%** (47/66) | 8 | 19 | **11.4s** | **~30 min** |

(OpenClaw landed slightly under its 2026-06-07 baseline of 71.2/78.8 — attributable
to late-PnP context bloat: the 20-step chain grows to ~118k tokens near the 131k
ctx and induces tail timeouts.)

## Headline 1 — Direct's low score was scorer strictness, not the model

The raw Direct number (57.6%) is **depressed by a Direct-lane-only scoring
artifact**, not a capability gap.

- Direct exposes the **acceptance surface too literally**: the scene/tree tool
  schemas advertise flat aliases (`shape_type`, `dimensions`, `box_dimensions_m`,
  `radius`, …) and require only `objectId`. Composer **accepts and normalises**
  these. In the direct lane the schema sits directly in OpenAI `tools[]`, so the
  model rationally emits the **short flat form**; the gateway lanes (driving an
  MCP agent) emit the **nested** `shape.type` / `shape.box_dimensions_m` form.
- The corpus `args_contain` golden credited only the nested form. Verified from
  `direct-baseline.json` + bridge audit: `scene_draft_add_object` returns
  **200/completed** with correct dimensions and `state_after` **passes**, yet the
  case failed on `args_contain` alone.
- A semantic-effect-first re-score of the existing logs (`/tmp/rescore.py`,
  applied to all 3 lanes) re-credited **3 state-proven Direct cases and 0 gateway
  cases**, and flagged **12 Direct vs 0 gateway** needs-evidence — confirming the
  false-negatives are **entirely Direct-lane**.

## Headline 2 — Direct is the fast lane (~6× Hermes, ~3× OpenClaw)

Median per-case latency: **Direct 11.4s · OpenClaw 36.8s · Hermes 67.5s**; full
run 30 / 53 / 80 min. Direct runs the tool loop **in-process** (bridge → vLLM, no
hops); the gateway lanes route every turn through a sandbox gateway agent (extra
hops + an agent loop + MCP round-trips). That same in-process rapid-fire is why
Direct alone trips the proxy loop-stop / heavy-generation upstream errors — it
hammers the single vLLM slot back-to-back while the gateways pace themselves.

## Fixes applied (this run validated them; 0 regressions)

Scorer contract (`scripts/debug/smoke_corpus_runner.py`) + corpus + composer +
bridge — full detail and rationale in
[`docs/operations/THREE-LANE-SCORER-NOTE.md`](../../operations/THREE-LANE-SCORER-NOTE.md):

1. **Accepted-form aliases** — credit top-level `box_dimensions_m` / `shape_type`
   / `sphere_radius_m` and nested `shape.shape_type`.
2. **Semantic-effect-first** — if a case's `state_after` proves the effect, demote
   residual `args_contain` arg-phrasing mismatches to *soft*. Over-acts /
   wrong-tool / wrong-state stay hard.
3. **Type-inferred-from-dims** — a box specified by dims IS a box even if
   `shape.type` is omitted.
4. **List-of-dict subset matching** — blackboard `keys` may carry extra
   Composer-accepted fields.
5. **Corpus**: INSERT "after X" verifies `afterName` placement (not `parentName`);
   distinguishing `state_after` on `CUR_scene_add_static_fixture`.
6. **Failure attribution** (reporting only): bridge proxy-409 loop-stop →
   `upstream_loop_stop` (not 502); composer maps `upstream_loop_stop` /
   `stuck_loop` / `max_turns_exceeded` → 409; runner surfaces the reason in its
   label. Loops no longer read as `chat HTTP 502` transport faults.
7. **Parity cleanup** (`assistant_tool_schemas.py`): direct provider envelope uses
   the same curated descriptor source as the gateway manifest. *Not* a score fix
   (the ops already completed under the reduced schema) — single-source-of-truth.

**Result:** Direct 50.0/57.6 → **59.1/71.2** (effective **+13.6 pts**) — now in
the gateway band, and by far the fastest lane.

## Residual fails — all genuine (corrected-scorer Direct run, 19 fails)

| # | category | nature |
|---|---|---|
| 7 | clarification over-act | genuine; **fails on all three lanes** (model mutates when it should ask) — NOT a Direct-specific gap |
| 5 | wrong arg value / id | genuine (e.g. `PnP_08` gripper 1.0≠0.66; `BB_modify` id `grip_force_key`) |
| 4 | model-loop / heavy-gen upstream error | **Direct-specific** (in-process rapid-fire → proxy loop-stop / vLLM 502 on huge gens) |
| 3 | wrong / missing tool | genuine (`REPLACE_simple`, `SCENE_update_pose`, `CUR_runtime_remove`) |

No scorer false-negatives remain. The wrong-value/over-act cases held as fails
throughout (demotion correctly scoped to accepted-alias-with-state-proof).

## Verdict

**With a fair scorer the three lanes are functionally comparable** (direct 71.2 ≈
openclaw 77.3, approaching hermes 81.8), and **direct is ~3–6× faster**. The lane
trade-off: **direct = fastest + comparable quality (in-process, no sandbox);
gateways = slower but bounded-autonomy / sandboxed**. The remaining true
Direct-specific deficit is the ~4 loop/heavy-gen cases (in-process slot
contention), partially mitigated by the bridge's graceful 409 handling.

## Recommendations

1. Land the scorer-contract changes (semantic-effect-first + accepted-form
   aliases + afterName→placement). They make all-lane scoring measure effect, not
   arg phrasing.
2. Longer term: split the **accepted** runtime schema from a canonical
   **model-facing** schema, so model output is consistent across lanes (one way
   to say "box dims").
3. Direct-lane robustness: the in-process loop should pace / cap rapid-fire turns
   to avoid proxy loop-stop / vLLM 502 on heavy generations.

## Follow-ups

- Extend `state_after` coverage to the remaining distinguishable needs-evidence
  cases (PnP_20 grip_force, runtime CUR cases). Generic cases (`SCENE_add_generic`)
  need an object-count-delta primitive — a box already exists, so type==box passes
  vacuously.
- Re-run hermes with the local catalog_read serve + corrected scorer for a fully
  apples-to-apples 3-lane sweep on one corpus revision (the hermes.json here
  predates the local-serve fix; catalog_read is counted in it).
- Attribution fix goes live on the next composer restart (code is in; this run's
  502 labels predate it).
