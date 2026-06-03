# `manyforge/lanes/hermes/`

Lane-specific artifacts for the Hermes Agents lane (Phase 4 of the THREE-LANE-MIGRATION-PLAN).

## Status — Phase 4 SCAFFOLDING

The Hermes lane is **inert until Phase 4 lands**. This directory ships the configuration that Phase 4 implementation work will consume:

- The Composer-side `LANE_REGISTRY` carries the `hermes` entry with `inert=True`.
- The launcher rejects `ASSISTANT_PROVIDER=hermes` unless `HERMES_LANE_PHASE4_ENABLED=true` is also set.
- `setup-hermes.sh` ships in `dev_ws/manyforge/scripts/` as a stub that documents the seven bring-up steps.

## Contents

| File | Purpose |
|---|---|
| [`mcp_servers_config.yaml`](./mcp_servers_config.yaml) | The `mcp_servers.manyforge` block emitted into Hermes' `/sandbox/.hermes/config.yaml`. Hermes' native MCP runtime (verified against the 0.14.0 wheel — `cli.py:2691, 9314+`, `tools/mcp_tool.py`) registers manyforge tools at startup with the `mcp_manyforge_` prefix. |
| [`policy.yaml`](./policy.yaml) | `SessionPolicy` config for this lane. Compaction OFF (Hermes owns its session lifecycle); synthetic short-circuits OFF (opt-in after bake-off); memory/skills/cron/todo/delegation all ENABLED per the explicit user direction. |

## Why this configuration

### Hermes natively supports MCP

The first rev of the THREE-LANE plan assumed Hermes had no MCP transport and proposed a ~400-line MCP-to-Hermes-tool wrapper. Verifying against the Hermes 0.14.0 wheel showed this was wrong: Hermes' `mcp_servers` config field is natively supported and auto-reloads on config-file mtime changes. The wrapper was dropped from the plan in rev 2; the entire wrapper code we'd have written becomes ~30 lines of YAML emission (this directory).

### Hermes owns the agent loop

Per principle #1 ("each lane works as upstream intends"), the Hermes lane uses Hermes' native session API (`/api/sessions/{id}/chat` or `/v1/runs`, decided by the Phase 4 probe). The bridge submits and observes; it does NOT shuttle per-turn tool calls. Memory, skills, cron, todo, and delegation are all enabled — the lane gives up some determinism in exchange for leveraging Hermes' distinctive functionality.

### NO_PROXY guidance

`NO_PROXY` must NOT include `host.openshell.internal`. The lane-neutral MCP bridge documents this at [`manyforge-mcp-bridge.py:75-81`](file:///home/tndlux/workspaces/dev_ws/src/manyforge/scripts/manyforge-mcp-bridge.py#L75-L81): the only network path to Composer from inside the sandbox runs through the proxy at `10.200.0.1:3128`, and bypassing it yields "Connection refused". `NO_PROXY` should cover only loopback hosts.

## What Phase 4 still needs

1. **Implement `setup-hermes.sh`** behind the `HERMES_LANE_PHASE4_ENABLED` feature gate.
2. **NemoClaw `hermes-config.ts` overlay** that emits the `mcp_servers.manyforge` block (the YAML in this directory is the source of truth).
3. **`manyforge_hermes_bridge`** — a thin FastAPI service on `:8300` that implements the `AssistantTransport` Protocol against Hermes' session API.
4. **Phase 4 probes**:
   - `--tool-call-parser hermes` on vLLM 0.x for cosmos-reason2-8b (5 cases).
   - sessions_chat vs `/v1/runs` head-to-head (3 cases).
5. **`PHASE-4-HERMES-LONGITUDINAL.md`** documenting the longitudinal corpus design + measured skill emergences / memory hit-rate / turns-to-completion.

## What's NOT here

- Hermes daemon installation: lives in NemoClaw's blueprint at `agents/hermes/`.
- The lane-neutral MCP bridge: it's at `dev_ws/manyforge/scripts/manyforge-mcp-bridge.py` and is already lane-neutralized in Phase 1 (verified for `MANYFORGE_LANE=hermes` resolution).
