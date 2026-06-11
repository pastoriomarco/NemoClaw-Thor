# Phase 5 — Production Decision (Interim)

This is the Phase 5 decision document per [THREE-LANE-MIGRATION-PLAN.md](./THREE-LANE-MIGRATION-PLAN.md) §8 Phase 5. **Marked interim** because the empirical Phase 3 measurement that would inform a confident production routing decision requires foundation triage that exceeded the autonomous run's scope.

## Update — 2026-06-07: foundation triage RESOLVED; OpenClaw lane validated end-to-end

The interim blocker below (proxy-in-path race + sandbox→`host.openshell.internal:8000` L7 403) is **resolved** — fixed by the §4.6 rev.5 merged-policy work (single `policies/manyforge-composer.merged.yaml` carrying both endpoints AND the *resolved* `/usr/bin/python3.13` binary subject; the MCP bridge routes through the OpenShell proxy at `10.200.0.1:3128`). The OpenClaw native-discovery lane now runs full smoke corpora end-to-end with the proxy in path.

**Evidence — 2026-06-07 7-model sweep (OpenClaw lane, 120W, `--self-heal`, 66 attempted of 75):**

| Model on the OpenClaw lane | effective | vs Phase-3 gate (≥46/66) |
|---|---|---|
| gemma4-12b-it-gguf **(QAT)** | 52/66 (78.8%) | ✅ |
| qwen3.6-35b-a3b-nvfp4 | 51/66 (77.3%) | ✅ |
| gemma4-12b-it-gguf (plain) | 47/66 (71.2%) | ✅ |
| cosmos-reason2-8b (vLLM) | 39/66 (59.1%) | ❌ (below gate + below iter-32 51/66) |
| nemotron3-nano-4b-gguf | 38/66 (57.6%) | ❌ |
| cosmos-reason2-8b-gguf | 36/66 (54.5%) | ❌ |
| nemotron-omni-30b (think-off) | 28/66 (42.4%) | ❌ |

Full report: [`smoke-evidence/2026-06-07-thor-7model-sweep-qat/REPORT.md`](./smoke-evidence/2026-06-07-thor-7model-sweep-qat/REPORT.md).

**Phase-3 read (honest):** the native-discovery lane clears the ≥46/66 gate comfortably on the stronger models (gemma / qwen / gemma-QAT, 71–79%) — so the lane architecture is empirically sound, not just historically. **But on the production-anchor cosmos-reason2-8b it scored 39/66 this session — below the gate and below the iter-32 51/66 baseline.** The comparison is NOT apples-to-apples: the corpus grew and hardened since iter-32 (2026-05-10); these runs used `--self-heal`; and the **`P2_scene_add` false-fail counted against every model here** and has since been fixed (commit `7571da2`). A clean cosmos re-run with the P2 fix in is required before declaring the anchor pass/fail. Separately, **gemma-QAT (78.8%) is now the strongest model on this lane** and a candidate to re-anchor the production *model* choice — orthogonal to the lane-routing decision this doc owns.

**Net correction:** the lane default (`openclaw`) is now empirically validated end-to-end, not just historical; foundation triage and the Phase-0 O-probes are unblocked. This 2026-06-07 note is superseded by the 2026-06-09 three-lane head-to-head below for Hermes status and model-default evidence.

## Update — 2026-06-09: first real three-lane head-to-head (direct / openclaw / hermes)

[`smoke-evidence/2026-06-09-thor-three-lane-parity-qat/`](./smoke-evidence/2026-06-09-thor-three-lane-parity-qat/REPORT.md)
fixes the **model** (gemma-QAT) and sweeps the **lane** — the comparison this doc
was waiting on. Same 66-case corpus, `--self-heal` ON, shared vLLM+proxy, lane
selected by `ASSISTANT_PROVIDER`.

| Lane | effective | first-try | median latency | full run |
|---|---|---|---|---|
| **hermes** (native MCP) | **81.8%** (54/66) | 78.8% | 67.5s | ~80 min |
| openclaw (gateway discovery) | 77.3% (51/66) | 66.7% | 36.8s | ~53 min |
| direct — corrected scorer | 71.2% (47/66) | 59.1% | **11.4s** | **~30 min** |
| direct — original scorer | 57.6% (38/66) | 50.0% | 11.4s | ~30 min |

**Three findings that reshape this decision:**

1. **Hermes tops per-turn quality (81.8%)** — above OpenClaw's best-ever (78.8 on
   06-07) — and uniquely passed the two genuinely-hard runtime cases
   (`CUR_runtime_remove_then_restore`, `CUR_runtime_update_pose`) that OpenClaw
   failed. This is a *per-turn* win; the longitudinal metric Hermes is designed
   for is still unmeasured.
2. **Direct's old 57.6% was a scorer artifact, not a model gap.** Direct emits the
   flat schema form Composer accepts; the golden credited only the nested form. A
   semantic-effect-first re-score (state-proven) lifts Direct to **71.2%** — into
   the gateway band — and it is **3–6× faster** than the gateways (in-process, no
   sandbox hops).
3. **With a fair scorer the three lanes are functionally comparable (71–82%).** The
   differentiator is no longer pass-rate — it is **latency vs autonomy/sandboxing.**

**Caveats before this becomes final:** (a) the Hermes run is *not* apples-to-apples
— its `hermes.json` predates the local `catalog_read` serve fix (report follow-up
#2 calls for a re-run); (b) OpenClaw landed slightly below its own baseline here
(late-PnP context bloat near the 131k ctx), so the hermes↔openclaw gap is closer to
a tie than 4.5 pts; (c) Hermes is the slowest lane (67.5s median, ~80 min/run) — a
real interactive-product cost.

**Net production read (revised):** Do *not* flip the single default to Hermes on a
per-turn win plus an un-clean measurement. Production runs **one lane at a time,
chosen at startup** via `ASSISTANT_PROVIDER` (the launcher starts only that lane's
bridge — no second sandbox or agent loop is up concurrently). So the decision this
doc owns is simply **which single lane is the default**, and on current evidence
that stays **openclaw** until the two gating items below land. The per-lane
strengths still inform *operator choice* of which lane to boot for a given session:

- **direct** → latency-bound / simple known workflows (11s, in-process)
- **openclaw** → balanced default (sandboxed, moderate latency) — **current default**
- **hermes** → hard multi-step / long-running where autonomy + memory compound
  (the cases the others fail), latency-tolerant

**Gating items before changing the default:** (1) the apples-to-apples Hermes
re-run; (2) the Phase-4 longitudinal gate (still TBD) — the metric Hermes is
*allowed* to win on.

**Explicit non-goal (2026-06-09):** *concurrent* multi-lane serving (a per-request
router picking among live lanes) is **not** a current goal. It would require
multiple sandboxes + agent loops running at once — a large step up in complexity and
resource use for no demonstrated need. A per-request router was prototyped against
the `_resolve_provider` seam and **reverted** to keep the single-lane-at-startup
model. `lane_routing.yaml` therefore remains aspirational: only its `default_lane`
concept is live today, and it is expressed through `ASSISTANT_PROVIDER`, not a
router. Revisit only if a concrete need for simultaneous lanes appears.

## Empirical evidence collected

| Run | Lane | Proxy in path | Effective rate | Notes |
|---|---|---|---|---|
| Iter-32 baseline (memory, 2026-05-10) | OpenClaw (plugin attempt) | YES | 51/66 (77.3%) | Recorded in `project_smoke_corpus_iter32_winner` memory |
| Pre-rebuild bake-off (2026-06-01) | OpenClaw on Qwen3.6-35B | YES | 56/66 (84.8%) | `archive/SMOKE-BAKEOFF-2026-06-01-3model.md` |
| **Phase 0 D-1 (this branch)** | Direct (cosmos-reason2-8b) | YES (vLLM:8050) | 28/66 (42.4%) | Sanity floor 40/66 NOT met — model-quality issue (`<MISSING>` args) |
| **Phase 0 O-1 (this branch)** | OpenClaw (cosmos-reason2-8b) | NO (vLLM:8000, no proxy) | 14/66 (21.2%) | NOT a measure of the lane; without-proxy config |
| **Phase 0 O-1 retry (this branch)** | OpenClaw (cosmos-reason2-8b) | YES (banner shows `compat`) but L7 policy 403 | 14/66 (21.2%) | OpenClaw → inference.local → not actually reaching the proxy (zero `/chat/completions` entries in proxy log) |
| Phase 3 OpenClaw native + skill addendum | OpenClaw (gemma-QAT) | YES | **77.3/66 (06-09)** ; gemma-QAT 78.8 (06-07) | Gate ≥46/66 cleared on strong models |
| Phase 4 Hermes per-turn (06-09 three-lane) | Hermes (gemma-QAT) | YES | **81.8% (54/66)** — tops the per-turn board | NOT apples-to-apples (predates catalog_read local-serve fix); re-run pending |
| Phase 4 Hermes longitudinal | Hermes | YES | **pending** | The metric Hermes is designed to win on; harness exists, run TBD |

## Production default — interim

**Recommend `openclaw` as the production default lane** and
`gemma4-12b-it-gguf` as the clean-start model default. The lane decision is
still interim pending the Hermes apples-to-apples rerun and longitudinal gate,
but the current launcher/serving default is aligned to Gemma QAT because it led
the 2026-06-07 model sweep and remained strong in the 2026-06-09 head-to-head.

Caveats:

1. Hermes needs an apples-to-apples rerun after the local `catalog_read` serve
   fix and corrected scorer.
2. Phase 4 Hermes longitudinal numbers do not exist yet.
3. Cosmos remains a historical anchor profile, but it is no longer the
   clean-start default.

## Lane routing (`lane_routing.yaml`)

> **⚠️ Config-only — by design (2026-06-09).** Nothing in Composer reads
> `lane_routing.yaml`, and that is intentional under the single-lane-at-startup
> model. The active lane is whatever the launcher sets via `ASSISTANT_PROVIDER`
> (one provider for the whole process; only that lane's bridge runs); a per-request
> `provider_id` can only *match* the configured lane (`_resolve_provider` in
> `routes_assistant.py` 404s otherwise). Per-request routing across concurrently
> live lanes is an explicit non-goal (see above) — a prototype router was reverted.
> Of this file, only `default_lane` is operative today, and it is expressed through
> `ASSISTANT_PROVIDER`. The `overrides`/`rollback_force` fields are aspirational.

The lane-routing config (aspirational beyond `default_lane`) lives at [`manyforge/lanes/lane_routing.yaml`](../lanes/lane_routing.yaml):

```yaml
default_lane: openclaw  # based on the iter-32 historical evidence

overrides:
  - match:
      request_shape: known_workflow_simple
      time_budget_ms_lt: 5000
    lane: direct
  - match:
      request_shape: long_running
      conversation_turns_gt: 3
    lane: hermes
    requires: HERMES_LANE_PHASE4_ENABLED

rollback_force: ""  # emergency lever — set to one of: openclaw|direct|hermes
```

## What followed-up cycles must do

1. **Foundation triage** — resolve the `start-model.sh` vs launcher proxy race. Possible fixes:
   - Make `start-model.sh` honor `MANYFORGE_PROXY_LISTEN_PORT` so vLLM and proxy ports come from one source of truth.
   - Have the launcher's proxy step detect an already-managed proxy from `start-model.sh` (matching PID file) and reuse instead of erroring.
   - Add a `setup-openclaw.sh` (Phase 3 task) that takes a clean machine to a working OpenClaw lane with proxy in path, idempotent.
2. **L7 policy investigation** — the sandbox is on `172.18.0.2` which IS in the `manyforge_composer` policy's `allowed_ips` (`172.18.0.0/16`), yet the in-sandbox curl to `host.openshell.internal:8000/v1/models` returns 403. Either:
   - Policy applied to the wrong network policy entry.
   - L7 path filter is denying `/v1/models` GET (but the policy explicitly allows it).
   - The sandbox is hitting a different proxy than the host's `:8000`.
3. **Re-run O-1..O-5 probes** once foundation triage lands. The full results populate this doc and gate the Phase 3 decision.
4. **Phase 3 empirical** — append the discovery-protocol skill addendum to the OpenClaw lane's system prompt and re-run the smoke corpus. Pass criteria: ≥46/66.
5. **Hermes follow-up** — rerun Hermes apples-to-apples after the local
   `catalog_read` serve fix, then run the longitudinal harness.
6. **Phase 5 final decision** — once the Hermes follow-ups exist, update this
   document with the chosen default, the startup lane rule, and the rollback
   playbook.

## Sign-off

- [x] Universal core (`common/`, `assistant_session/`), lane registry, merged policies, OpenClaw skill addendum + `lane_routing.yaml` landed (Phases 1 + 3 code)
- [x] **Foundation triage RESOLVED** (proxy-in-path race + L7 403) — §4.6 rev.5 merged policy; OpenClaw lane runs full corpora end-to-end (2026-06-07)
- [x] Empirical Phase 0 D-1 (28/66) + O-1 root-caused (14/66, foundation issue — now fixed)
- [x] Phase 3 empirical: ≥46/66 met on gemma/qwen/gemma-QAT (71–79%) via the 2026-06-07 sweep
- [~] **Sign-off corrections (tree does NOT match the original "all code-complete" claim):** `lanes/direct/` NOT created; `setup-{direct,openclaw,hermes}.sh` absent; longitudinal harness absent
- [ ] Clean cosmos-reason2-8b anchor re-run with the P2 fix in (current 39/66 not apples-to-apples vs iter-32 51/66)
- [ ] Phase 2 direct-lane formalization (`lanes/direct/`) + Q1 decision (move vs cross-repo import)
- [x] Phase 0.5/Hermes per-turn implementation has live 2026-06-09 evidence
- [ ] Phase 4 Hermes apples-to-apples rerun + longitudinal harness
- [x] **First real three-lane head-to-head (2026-06-09)** — hermes 81.8 / openclaw 77.3 / direct 71.2 (corrected); lanes functionally comparable, differentiator is latency vs autonomy
- [ ] Apples-to-apples Hermes re-run (local `catalog_read` serve + corrected scorer)
- [x] **Per-request lane routing prototyped and reverted (2026-06-09)** — out of scope under the single-lane-at-startup model; concurrent multi-lane is an explicit non-goal (needs multiple sandboxes + agent loops live at once)
- [ ] **Phase 5 final decision** (this document, once the longitudinal gate + apples-to-apples Hermes re-run exist) — the decision is *which single default lane*, selected at startup via `ASSISTANT_PROVIDER`

## Lane selection model (settled 2026-06-09)

Production serves **one lane at a time, chosen at startup** by `ASSISTANT_PROVIDER`
(`direct` | `openclaw` | `hermes`; hermes additionally gated by
`HERMES_LANE_PHASE4_ENABLED`). The launcher starts only the selected lane's bridge;
no second sandbox or agent loop runs concurrently. This is the whole mechanism —
there is no per-request router and `lane_routing.yaml` is not consulted at runtime.

A feature-flagged per-request router (`lane_router.py` + a `_resolve_provider` hook)
was prototyped on 2026-06-09 and **reverted the same day**: concurrent multi-lane
serving would require multiple sandboxes / agent loops alive at once — a large
complexity and resource step with no demonstrated need. If that need ever appears,
the `_resolve_provider` seam is the integration point and `lane_routing.yaml`'s
`overrides` / `rollback_force` schema is the intended policy source; until then both
remain aspirational.
