# `manyforge.assistant_session` — per-session orchestration policy

Per-session policy that sits ABOVE the transport layer but BELOW the
lane-specific glue. Each policy is per-lane opt-in via a typed dataclass;
defaults preserve the iter-32 OpenClaw production recipe while Hermes
and Direct lanes can configure independently.

## What lives here

| Module | Responsibility | Lane defaults |
|---|---|---|
| [`compaction.py`](./compaction.py) | Bridge-fired `/compact` policy. Counter + threshold + per-lane action verb. | OpenClaw: every 2 (iter-32). Hermes: OFF (owns its lifecycle). Direct: every 4 with `truncate` action. |
| [`synthetic_short_circuits.py`](./synthetic_short_circuits.py) | Cosmos-specific "bypass clarification" + retry-loop detector. | OpenClaw: ON (proven). Direct: OFF (opt-in, needs benchmark). Hermes: OFF (opt-in). |
| [`circuit_breaker.py`](./circuit_breaker.py) | Per-transport circuit breaker; opens after N consecutive failures for a cooldown window. | Lane-agnostic; always on. |
| [`session_key.py`](./session_key.py) | Stable session key derived from conversationId + assistantMode + catalogHash + revision. | Lane-agnostic; always on. |

## Why these aren't transport code

The OpenClaw bridge's `service.py` historically interleaved transport
selection (`if cfg.use_gateway`) with these orchestration policies. That
left the Hermes lane with no path to inherit them and the Direct lane
with the projection-mirror anti-pattern.

Lifting these into a separate package means:

- Each lane adapter consumes the same `CompactionPolicy` /
  `SyntheticPolicy` dataclass surface.
- Per-lane defaults are explicit and documented (the constants at the
  bottom of each module).
- A `SessionPolicy` config (loaded from `manyforge/lanes/<lane>/policy.yaml`)
  composes the four policies and the session orchestrator reads it at
  startup. (Phase 2 lands the YAML loader.)

## Phase 1 status (behavior-preserving)

- `compaction.py` re-exports the iter-32 bookkeeping helpers from
  `openclaw_assistant_bridge.service`. The dataclass is new (pure
  config).
- `synthetic_short_circuits.py` is dataclass-only for now — the actual
  detector code stays in `service.py` until Phase 2 moves it.
- `circuit_breaker.py` is a star-re-export of the existing module.
- `session_key.py` re-exports `derive_gateway_session_key` (now also
  exposed via `manyforge.common.envelope.derive_session_key`).

No runtime behavior changes. The existing OpenClaw lane continues to
import the same symbols from the same locations; new code can import via
this package.
