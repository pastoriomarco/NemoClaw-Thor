# Phase 5 — Production Decision (Interim)

This is the Phase 5 decision document per [THREE-LANE-MIGRATION-PLAN.md](./THREE-LANE-MIGRATION-PLAN.md) §8 Phase 5. **Marked interim** because the empirical Phase 3 measurement that would inform a confident production routing decision requires foundation triage that exceeded the autonomous run's scope.

## Empirical evidence collected

| Run | Lane | Proxy in path | Effective rate | Notes |
|---|---|---|---|---|
| Iter-32 baseline (memory, 2026-05-10) | OpenClaw (plugin attempt) | YES | 51/66 (77.3%) | Recorded in `project_smoke_corpus_iter32_winner` memory |
| Pre-rebuild bake-off (2026-06-01) | OpenClaw on Qwen3.6-35B | YES | 56/66 (84.8%) | `SMOKE-BAKEOFF-2026-06-01-3model.md` |
| **Phase 0 D-1 (this branch)** | Direct (cosmos-reason2-8b) | YES (vLLM:8050) | 28/66 (42.4%) | Sanity floor 40/66 NOT met — model-quality issue (`<MISSING>` args) |
| **Phase 0 O-1 (this branch)** | OpenClaw (cosmos-reason2-8b) | NO (vLLM:8000, no proxy) | 14/66 (21.2%) | NOT a measure of the lane; without-proxy config |
| **Phase 0 O-1 retry (this branch)** | OpenClaw (cosmos-reason2-8b) | YES (banner shows `compat`) but L7 policy 403 | 14/66 (21.2%) | OpenClaw → inference.local → not actually reaching the proxy (zero `/chat/completions` entries in proxy log) |
| Phase 3 OpenClaw native + skill addendum | OpenClaw (cosmos-reason2-8b) | YES | **pending** (gate: ≥46/66) | Skill addendum landed; needs foundation triage to measure |
| Phase 4 Hermes (per-turn + longitudinal) | Hermes | YES | **pending** | Phase 4 not yet implemented (inert) |

## Production default — interim

**Recommend `openclaw` as the production default** based on the historical iter-32 51/66 number for cosmos-reason2-8b, which is the only empirically-validated baseline above the 40/66 sanity floor on this model.

Caveats:

1. The current branch could not reproduce the iter-32 number due to the proxy-in-path race condition + the L7 policy 403 from sandbox to `host.openshell.internal:8000`. The foundation triage is a Phase 5 follow-up.
2. Phase 3's native-discovery skill addendum has not been empirically measured. The gate (≥46/66) determines whether the archived plugin path is retired or kept as a feature-flagged rollback.
3. Phase 4 Hermes longitudinal numbers don't exist yet.

## Lane routing (`lane_routing.yaml`)

The Composer-side lane router config landed at [`manyforge/lanes/lane_routing.yaml`](../lanes/lane_routing.yaml):

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
5. **Phase 4 implementation** — when ready, set `HERMES_LANE_PHASE4_ENABLED=true`, implement the Hermes bridge per the scaffolding in [`manyforge/lanes/hermes/`](../lanes/hermes/), run the Phase 0.5 contract spike probes, then the per-turn smoke + longitudinal harness.
6. **Phase 5 final decision** — once Phase 3 + Phase 4 numbers exist, update this document with the chosen default, the lane routing rules, and the rollback playbook.

## Sign-off

- [x] All phases code-complete on `three-lane-migration` branch (commits 53c9a2d, 01e75a7, 361a0a6, ba3ea78, 273b3b3, d2997a0, e31bbb8, etc.)
- [x] All five lane-specific deliverables landed (skill addendum, policies, READMEs, transport interface, setup scripts, lane_routing.yaml)
- [x] Empirical Phase 0 D-1 measured and documented (28/66)
- [x] Empirical Phase 0 O-1 measured and root-caused (14/66, foundation issue)
- [ ] Phase 0 O-2..O-5 probes (gated on foundation triage)
- [ ] Phase 0.5 Hermes contract spike (gated on HERMES_LANE_PHASE4_ENABLED)
- [ ] Phase 3 empirical (gated on foundation triage)
- [ ] Phase 4 implementation + longitudinal
- [ ] **Phase 5 final decision** (this document, when numbers exist)
