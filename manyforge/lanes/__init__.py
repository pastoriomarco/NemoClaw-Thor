"""ManyForge assistant lanes.

Each lane implements the shared :class:`common.transport.AssistantTransport`
and ships a thin FastAPI bridge service. Per THREE-LANE-MIGRATION-PLAN.md:

- ``lanes.hermes`` — Hermes Agents lane (Phase 4): native ``/v1/runs`` sessions
  API + ``mcp_servers``, memory/skills/cron/todo/delegation enabled. Bridge :8300.
- ``lanes.openclaw`` — OpenClaw gateway lane (config-only here; the transport
  still lives in ``openclaw_assistant_bridge`` pending the Phase 3 extraction).
- ``lanes.direct`` — Direct vLLM lane (Phase 2; not yet created here).
"""
