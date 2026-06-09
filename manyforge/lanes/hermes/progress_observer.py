"""Hermes progress observer — SSE lifecycle events → universal audit.

Per THREE-LANE-MIGRATION-PLAN.md §5.3(4,6) and §4.7. The Hermes lane does not
shuttle per-turn tool calls (Hermes owns its agent loop); instead the bridge
*observes* the ``/v1/runs/{run_id}/events`` stream and translates each
observable event into:

1. the **universal audit** ``toolsObserved[]`` (§4.7) — tool names with the
   ``mcp_manyforge_`` prefix stripped so cross-lane reports compare apples to
   apples; and
2. a **per-event session log** (``hermes-session-events.jsonl``) capturing the
   distinctive Hermes signals: skill creations, memory writes, cron fires, and
   delegation calls (§5.3(6)) — the inputs the longitudinal harness scores.

**Progress observation is best-effort augmentation, not correctness-bearing**
(plan §5.3(4)): the Composer ``/api/assistant/bridge/tools/{toolId}`` callback
log is the hard source of truth for what tools ran. So an unknown, missing, or
malformed event is dropped from the audit and the lane continues. The
event-type taxonomy was not live-enumerable in the Phase 0.5 spike, so
classification is substring-based against the centralised
:class:`HermesEventTaxonomy` — reconcile that one class with live
``/v1/capabilities`` output if names differ; no logic changes needed.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from common.tool_calls import strip_mcp_prefix

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HermesEventTaxonomy:
    """Substring patterns that classify a Hermes SSE event's type.

    Matched case-insensitively against the event ``type`` string. Generous by
    design — the spike could not enumerate the exact names, and the observer is
    best-effort, so over-matching merely enriches the audit while a miss simply
    omits an event.
    """

    tool_call: tuple[str, ...] = ("tool_call", "tool.call", "tool_invoc", "tool.invoked", "toolcall")
    memory_write: tuple[str, ...] = ("memory", "remember", "memory_write", "memory.write")
    skill_created: tuple[str, ...] = ("skill", "skill_created", "skill.created", "skill_emerg")
    cron_fire: tuple[str, ...] = ("cron", "schedule", "cron_fire", "cron.fired")
    delegation: tuple[str, ...] = ("deleg", "subagent", "spawn", "delegation")
    # Keys under event ``data`` that may carry a tool name (checked in order).
    tool_name_keys: tuple[str, ...] = ("tool", "tool_name", "toolName", "name", "id")


DEFAULT_TAXONOMY = HermesEventTaxonomy()

# Session-event kinds emitted into hermes-session-events.jsonl.
KIND_TOOL_CALL = "tool_call"
KIND_MEMORY_WRITE = "memory_write"
KIND_SKILL_CREATED = "skill_created"
KIND_CRON_FIRE = "cron_fire"
KIND_DELEGATION = "delegation"


@dataclass
class Observation:
    """Aggregated result of observing one run's event stream."""

    tools_observed: list[str] = field(default_factory=list)  # prefix-stripped, in order
    session_events: list[dict[str, Any]] = field(default_factory=list)  # for the jsonl
    raw_event_count: int = 0

    # Distinctive-Hermes counters (the longitudinal harness reads these).
    memory_writes: int = 0
    skill_creations: int = 0
    cron_fires: int = 0
    delegations: int = 0


class HermesProgressObserver:
    """Translate Hermes lifecycle events into the universal audit + session log.

    Pure aggregation (no I/O) in :meth:`observe` for testability; the optional
    :meth:`write_session_events` helper appends the per-event records to the
    lane's ``hermes-session-events.jsonl`` when the service wants them
    persisted.
    """

    def __init__(self, *, taxonomy: HermesEventTaxonomy = DEFAULT_TAXONOMY) -> None:
        self._tax = taxonomy

    def observe(self, events: list[dict[str, Any]], *, conversation_id: str = "") -> Observation:
        obs = Observation(raw_event_count=len(events))
        for event in events:
            if not isinstance(event, dict):
                continue
            etype = str(event.get("event") or "").lower()
            data = event.get("data") if isinstance(event.get("data"), dict) else {}
            if not etype:
                continue
            try:
                self._classify(etype, data, obs, conversation_id)
            except Exception as exc:  # noqa: BLE001 — best-effort; never break the lane
                logger.debug("observer skipped malformed event %r: %s", etype, exc)
        return obs

    def _classify(
        self, etype: str, data: dict[str, Any], obs: Observation, conversation_id: str
    ) -> None:
        # Tool calls first — the only signal that feeds the universal audit's
        # toolsObserved[] (the cross-lane comparison surface).
        if _matches(etype, self._tax.tool_call):
            name = self._tool_name(data)
            if name:
                bare = strip_mcp_prefix(name)
                obs.tools_observed.append(bare)
                obs.session_events.append(
                    _record(KIND_TOOL_CALL, conversation_id, tool=bare, raw_name=name)
                )
            return
        if _matches(etype, self._tax.memory_write):
            obs.memory_writes += 1
            obs.session_events.append(
                _record(KIND_MEMORY_WRITE, conversation_id, summary=_short(data))
            )
            return
        if _matches(etype, self._tax.skill_created):
            obs.skill_creations += 1
            obs.session_events.append(
                _record(
                    KIND_SKILL_CREATED,
                    conversation_id,
                    skill=str(data.get("name") or data.get("skill") or "").strip(),
                    summary=_short(data),
                )
            )
            return
        if _matches(etype, self._tax.cron_fire):
            obs.cron_fires += 1
            obs.session_events.append(_record(KIND_CRON_FIRE, conversation_id, summary=_short(data)))
            return
        if _matches(etype, self._tax.delegation):
            obs.delegations += 1
            obs.session_events.append(_record(KIND_DELEGATION, conversation_id, summary=_short(data)))
            return
        # Unknown event type: dropped from the audit (best-effort).

    def _tool_name(self, data: dict[str, Any]) -> str:
        for key in self._tax.tool_name_keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            # Some Hermes builds nest as {tool: {name: ...}}.
            if isinstance(value, dict):
                nested = value.get("name") or value.get("id")
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
        return ""

    @staticmethod
    def write_session_events(records: list[dict[str, Any]], *, path: str | None = None) -> None:
        """Append per-event session records to ``hermes-session-events.jsonl``.

        Best-effort: a write failure is logged, not raised — the session log is
        an analysis aid, not a correctness path."""
        if not records:
            return
        target = path or os.environ.get(
            "MANYFORGE_HERMES_SESSION_EVENTS_PATH",
            "/tmp/manyforge-assistant-e2e/hermes-session-events.jsonl",
        )
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "a", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except OSError as exc:
            logger.warning("could not write hermes session events to %s: %s", target, exc)


# ---- module helpers ----------------------------------------------------------


def _matches(etype: str, patterns: tuple[str, ...]) -> bool:
    return any(p in etype for p in patterns)


def _record(kind: str, conversation_id: str, **fields: Any) -> dict[str, Any]:
    rec = {"kind": kind, "conversationId": conversation_id}
    rec.update({k: v for k, v in fields.items() if v not in (None, "")})
    return rec


def _short(data: dict[str, Any], limit: int = 200) -> str:
    try:
        text = json.dumps(data, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        text = str(data)
    return text[:limit]


__all__ = [
    "HermesProgressObserver",
    "HermesEventTaxonomy",
    "DEFAULT_TAXONOMY",
    "Observation",
    "KIND_TOOL_CALL",
    "KIND_MEMORY_WRITE",
    "KIND_SKILL_CREATED",
    "KIND_CRON_FIRE",
    "KIND_DELEGATION",
]
