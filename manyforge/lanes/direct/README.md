# `manyforge/lanes/direct/`

Dev/analysis pointer for the **Direct vLLM** assistant lane.

> **Implementation lives in the `manyforge` deployment repo, not here.**
> The Direct-lane bridge is
> [`manyforge_assistant_bridge/`](/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_assistant_bridge/)
> in the `manyforge` repo (host venv, serves `:8100`, started with
> `ASSISTANT_PROVIDER=direct`). This directory carries only the lane's
> dev/analysis cross-links — there are no lane-specific artifacts to ship
> (unlike `openclaw/` and `hermes/`), because the Direct lane puts the tool
> schema **directly in the OpenAI `tools[]`** with no discovery primer or
> session policy.

NemoClaw-Thor documents and benchmarks the Direct lane for parity/support; the
canonical id is `direct` (routing `direct-vllm`) in the Composer
`LANE_REGISTRY`.

## Architecture

The Direct lane runs the tool loop **in-process** (bridge → vLLM) with **no
gateway hops** — the model sees the tool schema in the request envelope and
emits tool calls that the bridge dispatches straight to the Composer. This
makes it by far the **fastest** lane (typical case ~6× faster than Hermes, ~3×
faster than OpenClaw on the same model). The same in-process rapid-fire is also
why Direct alone tends to hit the proxy loop-stop / heavy-generation upstream
errors — it hammers the single vLLM slot back-to-back while the gateway lanes
pace themselves. See the comparison doc for the numbers.

## Operational & related docs

- **Bring-up + live-monitoring (operational):**
  [`manyforge/docs/operations/LANE_BRINGUP.md`](/home/tndlux/workspaces/dev_ws/src/manyforge/docs/operations/LANE_BRINGUP.md)
  — the `direct` section (bridge on `:8100`, host venv, no sandbox; health
  `curl http://127.0.0.1:8100/healthz`).
- **Benchmarks + scorer analysis:** [`../../docs/LANE-COMPARISON.md`](../../docs/LANE-COMPARISON.md).
- **Architecture hub:** [`../../docs/THREE-LANE-MIGRATION-PLAN.md`](../../docs/THREE-LANE-MIGRATION-PLAN.md).
- **Pipeline internals:** [`../../docs/COMPOSER-ASSISTANT-ARCHITECTURE.md`](../../docs/COMPOSER-ASSISTANT-ARCHITECTURE.md).

## What's NOT here

- The bridge implementation, venv provisioning, and `:8100` supervisor — all in
  the `manyforge` repo (`manyforge_assistant_bridge/`,
  `scripts/lib/assistant.sh`).
- Provider registration: the `LANE_REGISTRY` `direct` entry is in
  [`assistant_provider.py`](/home/tndlux/workspaces/dev_ws/src/manyforge/manyforge_composer/backend/assistant_provider.py).
