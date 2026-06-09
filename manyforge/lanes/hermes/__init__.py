"""Hermes Agents lane (Phase 4 of THREE-LANE-MIGRATION-PLAN.md).

Modules:

- :mod:`lanes.hermes.session_dispatcher` — async client for Hermes' native
  ``/v1/runs`` API (start / stream events / status / stop / approval).
- :mod:`lanes.hermes.progress_observer` — translates the run's SSE lifecycle
  events into the universal audit + ``hermes-session-events`` log.
- :mod:`lanes.hermes.transport` — :class:`HermesTransport`, the lane's
  :class:`common.transport.AssistantTransport` implementation.
- :mod:`lanes.hermes.service` — the FastAPI bridge on :8300 (``app``).

Config in this directory (consumed at bring-up by ``setup-hermes.sh``):
``policy.yaml`` (SessionPolicy: compaction off, memory/skills/etc on) and
``mcp_servers_config.yaml`` (the ``mcp_servers.manyforge`` block emitted into
Hermes' ``config.yaml``).
"""
