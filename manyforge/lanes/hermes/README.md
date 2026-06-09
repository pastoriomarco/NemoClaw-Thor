# `manyforge/lanes/hermes/`

Lane-specific artifacts for the Hermes Agents lane (Phase 4 of the THREE-LANE-MIGRATION-PLAN).

## Status — Phase 4 LANDED (opt-in)

The Hermes lane is **implemented and unit-verified**, gated behind the
`HERMES_LANE_PHASE4_ENABLED` opt-in flag until its live bake-off validates it as
a production default. See [PHASE-4-HERMES-LONGITUDINAL.md](../../docs/PHASE-4-HERMES-LONGITUDINAL.md)
for the component map, the longitudinal design, and the gate (the live
smoke/longitudinal numbers are operator-driven and still TBD).

- The Composer-side `LANE_REGISTRY` carries the `hermes` entry with `inert=False`; `build_assistant_provider` routes it to `HermesAssistantProvider`.
- The launcher (`assistant.sh::start_bridge_hermes`) starts the `:8300` bridge when `ASSISTANT_PROVIDER=hermes` **and** `HERMES_LANE_PHASE4_ENABLED=true`.
- `setup-hermes.sh` (in `dev_ws/src/manyforge/scripts/`) implements the seven bring-up steps in the spike's strict order-of-ops.

## Contents

| File | Purpose |
|---|---|
| `session_dispatcher.py` / `progress_observer.py` / `transport.py` / `engine.py` / `service.py` | The lane implementation. See the component table in [PHASE-4-HERMES-LONGITUDINAL.md](../../docs/PHASE-4-HERMES-LONGITUDINAL.md). |
| `tests/` | 30 unit tests (dispatcher SSE/lifecycle, observer event→audit mapping, transport conformance, engine envelope round-trip), dependency-free. |
| [`mcp_servers_config.yaml`](./mcp_servers_config.yaml) | The `mcp_servers.manyforge` block emitted into Hermes' `/sandbox/.hermes/config.yaml`. Hermes' native MCP runtime (verified against the 0.14.0 wheel — `cli.py:2691, 9314+`, `tools/mcp_tool.py`) registers manyforge tools at startup with the `mcp_manyforge_` prefix. |
| [`policy.yaml`](./policy.yaml) | `SessionPolicy` config for this lane. Compaction OFF (Hermes owns its session lifecycle); memory/skills/cron/todo/delegation all ENABLED; `preferred_session_api: runs` (resolved by the Phase 0.5 spike). |

## Why this configuration

### Hermes natively supports MCP

The first rev of the THREE-LANE plan assumed Hermes had no MCP transport and proposed a ~400-line MCP-to-Hermes-tool wrapper. Verifying against the Hermes 0.14.0 wheel showed this was wrong: Hermes' `mcp_servers` config field is natively supported. The wrapper was dropped from the plan in rev 2; the entire wrapper code we'd have written becomes ~30 lines of YAML emission (this directory).

**Important correction (Phase 0.5 probe-3 online finding):** the `mcp_servers`
auto-reload watcher (`cli.py:9314+`) only runs under Hermes' **interactive
CLI**, NOT the gateway-only deployment NemoClaw provisions. So the gateway does
**not** pick up `mcp_servers` edits at runtime — the block must be present
**before the gateway starts**, and any later change requires a gateway restart
(`nemoclaw <name> recover`, never `rebuild` which resets the config). The
durable fix is to make the sandbox *born with* the block via the NemoClaw
Hermes blueprint (see [PHASE-4-HERMES-LONGITUDINAL.md](../../docs/PHASE-4-HERMES-LONGITUDINAL.md));
the direct-config-write in `setup-hermes.sh` is a diagnostic fallback for an
existing sandbox, not the primary bring-up path.

### Hermes owns the agent loop

Per principle #1 ("each lane works as upstream intends"), the Hermes lane uses Hermes' native session API (`/api/sessions/{id}/chat` or `/v1/runs`, decided by the Phase 4 probe). The bridge submits and observes; it does NOT shuttle per-turn tool calls. Memory, skills, cron, todo, and delegation are all enabled — the lane gives up some determinism in exchange for leveraging Hermes' distinctive functionality.

### NO_PROXY guidance

`NO_PROXY` must NOT include `host.openshell.internal`. The lane-neutral MCP bridge documents this at [`manyforge-mcp-bridge.py:75-81`](file:///home/tndlux/workspaces/dev_ws/src/manyforge/scripts/manyforge-mcp-bridge.py#L75-L81): the only network path to Composer from inside the sandbox runs through the proxy at `10.200.0.1:3128`, and bypassing it yields "Connection refused". `NO_PROXY` should cover only loopback hosts.

## What Phase 4 landed vs what the live run still owes

**Landed (code-complete, unit-verified):**

1. ✅ `setup-hermes.sh` — the seven-step bring-up (spike order-of-ops) behind `HERMES_LANE_PHASE4_ENABLED`.
2. ✅ The `mcp_servers.manyforge` emission path — `setup-hermes.sh` renders `mcp_servers_config.yaml` (env-substituted) into the sandbox config **before** gateway start (the `hermes-config.ts` overlay is one option for Q7; the direct-config-write path is what the spike validated and what `setup-hermes.sh` uses).
3. ✅ The `:8300` FastAPI bridge implementing `AssistantTransport` against Hermes' `/v1/runs` API (`service.py` + `transport.py` + `session_dispatcher.py` + `progress_observer.py` + `engine.py`).
4. ✅ Session API decision (Q4 = `/v1/runs`) and `API_SERVER_KEY` wiring (Q6).
5. ✅ `PHASE-4-HERMES-LONGITUDINAL.md` + the `longitudinal_hermes.py` harness + `longitudinal_corpus.yaml`.

**Owed by the live, operator-driven run (the empirical gate):**

- The Q3 probe (`--tool-call-parser hermes` on vLLM for cosmos-reason2-8b, 5 cases) — does not block the lane code; needs a vLLM restart with the parser flag.
- Per-turn smoke (memory off) ≥ 40/66 through the Hermes lane.
- The longitudinal numbers (skill emergences / memory hit-rate / turns-to-completion trend) → fill the TBD table in `PHASE-4-HERMES-LONGITUDINAL.md`.
- The live head-to-head session-API validation (now a validation, not a decision — Q4 is resolved to `/v1/runs`).

## What's NOT here

- Hermes daemon installation: lives in NemoClaw's blueprint at `agents/hermes/`.
- The lane-neutral MCP bridge: it's at `dev_ws/src/manyforge/scripts/manyforge-mcp-bridge.py` and is already lane-neutralized in Phase 1 (verified for `MANYFORGE_LANE=hermes` resolution).

## Operational & related docs

- **Bring-up + live-monitoring (operational):**
  [`manyforge/docs/operations/LANE_BRINGUP.md`](/home/tndlux/workspaces/dev_ws/src/manyforge/docs/operations/LANE_BRINGUP.md)
  — the `hermes` section (bridge on `:8300`; forwards to the Hermes gateway
  sandbox; health `curl http://127.0.0.1:8300/healthz`). Requires
  `HERMES_LANE_PHASE4_ENABLED=true`.
- **Longitudinal design + gate:** [`../../docs/PHASE-4-HERMES-LONGITUDINAL.md`](../../docs/PHASE-4-HERMES-LONGITUDINAL.md).
- **Benchmarks + scorer analysis:** [`../../docs/LANE-COMPARISON.md`](../../docs/LANE-COMPARISON.md).
