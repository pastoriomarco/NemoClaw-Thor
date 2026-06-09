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

**Net correction:** the lane default (`openclaw`) is now empirically validated end-to-end, not just historical; foundation triage and the Phase-0 O-probes are unblocked. The genuinely-remaining work is **Phase 4 (Hermes lane — still unbuilt), Phase 2 (direct-lane formalization under `lanes/direct/`), the bake-off harnesses (`compare_lanes.py` / `longitudinal_hermes.py`), and a clean cosmos-anchor re-run** before this doc goes final. The interim record below is preserved as-was.

## Empirical evidence collected

| Run | Lane | Proxy in path | Effective rate | Notes |
|---|---|---|---|---|
| Iter-32 baseline (memory, 2026-05-10) | OpenClaw (plugin attempt) | YES | 51/66 (77.3%) | Recorded in `project_smoke_corpus_iter32_winner` memory |
| Pre-rebuild bake-off (2026-06-01) | OpenClaw on Qwen3.6-35B | YES | 56/66 (84.8%) | `archive/SMOKE-BAKEOFF-2026-06-01-3model.md` |
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

- [x] Universal core (`common/`, `assistant_session/`), lane registry, merged policies, OpenClaw skill addendum + `lane_routing.yaml` landed (Phases 1 + 3 code)
- [x] **Foundation triage RESOLVED** (proxy-in-path race + L7 403) — §4.6 rev.5 merged policy; OpenClaw lane runs full corpora end-to-end (2026-06-07)
- [x] Empirical Phase 0 D-1 (28/66) + O-1 root-caused (14/66, foundation issue — now fixed)
- [x] Phase 3 empirical: ≥46/66 met on gemma/qwen/gemma-QAT (71–79%) via the 2026-06-07 sweep
- [~] **Sign-off corrections (tree does NOT match the original "all code-complete" claim):** `lanes/direct/` NOT created; Hermes lane code (transport/service/dispatcher/observer) NOT built; `setup-{direct,openclaw,hermes}.sh` absent; `compare_lanes.py` / `longitudinal_hermes.py` absent
- [ ] Clean cosmos-reason2-8b anchor re-run with the P2 fix in (current 39/66 not apples-to-apples vs iter-32 51/66)
- [ ] Phase 2 direct-lane formalization (`lanes/direct/`) + Q1 decision (move vs cross-repo import)
- [ ] Phase 0.5 Hermes contract spike run (doc exists; gated on `HERMES_LANE_PHASE4_ENABLED`)
- [ ] Phase 4 Hermes implementation + longitudinal harness
- [ ] **Phase 5 final decision** (this document, once Phase 4 numbers + the cosmos re-run exist)
