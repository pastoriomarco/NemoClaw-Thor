#!/usr/bin/env bash
# start-openclaw-assistant-bridge.sh — run the OpenClaw-lane assistant
# provider adapter on :8200 (production default since 2026-05-07).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_DIR="${SCRIPT_DIR}/openclaw_assistant_bridge"
VENV="${BRIDGE_DIR}/.venv"

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
  "${VENV}/bin/pip" install -r "${BRIDGE_DIR}/requirements.txt"
fi

# Production-aligned defaults (see scripts/lib/assistant.sh):
#   LOCAL=false          → bridge dispatches to the OpenShell sandbox-hosted
#                           agent runner via `nemoclaw exec`, NOT a local
#                           in-process model. The legacy `LOCAL=true` mode
#                           is dev-only (boots a CPU stub) and produces
#                           non-production behavior; do not use for smoke,
#                           probe, or composer-driven runs.
#   USE_GATEWAY=false    → CLI shell-out transport. OpenClaw 2026.5.22 does
#                           not expose /v1/chat/completions on the gateway
#                           HTTP server; the gateway_http transport returns
#                           404 for every call. Set to "true" only if you
#                           are running an OpenClaw build that has the
#                           /v1/chat/completions endpoint, or are testing
#                           the legacy path explicitly.
#   LOOP_TOOL_THRESHOLD=5 / LOOP_ARGS_THRESHOLD=2 → bridge-side fail-fast
#                           detectors (see FIX 5 in service.py). 0 disables.
#   TOOL_SURFACE=tools   → Production default since Phase 3 (2026-06-03).
#                           Requires sandbox openclaw.json to set
#                           tools.toolSearch = {enabled: true, mode: "tools"}.
#                           tools[] contains the three discrete control verbs
#                           (tool_search / tool_describe / tool_call). The
#                           bridge prompt teaches direct tool_call dispatch.
#                           Measured 58.3% effective rate on cosmos-reason2-8b
#                           vs 29.0% for code mode (Phase 3 LANE-COMPARISON).
#                           Set to "code" only for the multi-model bake-off
#                           or to test the alternative surface.
#                           The vLLM proxy (see scripts/proxy/vllm-proxy.py)
#                           cross-checks the observed tools[] against this
#                           env and emits ``tool_surface_mismatch`` if the
#                           sandbox config and this env disagree.
export OPENCLAW_ASSISTANT_LOCAL="${OPENCLAW_ASSISTANT_LOCAL:-false}"
export OPENCLAW_ASSISTANT_USE_GATEWAY="${OPENCLAW_ASSISTANT_USE_GATEWAY:-false}"
export OPENCLAW_ASSISTANT_AGENT="${OPENCLAW_ASSISTANT_AGENT:-manyforge-composer}"
export OPENCLAW_ASSISTANT_LOOP_TOOL_THRESHOLD="${OPENCLAW_ASSISTANT_LOOP_TOOL_THRESHOLD:-5}"
export OPENCLAW_ASSISTANT_LOOP_ARGS_THRESHOLD="${OPENCLAW_ASSISTANT_LOOP_ARGS_THRESHOLD:-2}"
export OPENCLAW_ASSISTANT_TOOL_SURFACE="${OPENCLAW_ASSISTANT_TOOL_SURFACE:-tools}"
export MANYFORGE_STATE_CONTEXT="${MANYFORGE_STATE_CONTEXT:-first_then_every_n}"
export MANYFORGE_STATE_CONTEXT_EVERY_N="${MANYFORGE_STATE_CONTEXT_EVERY_N:-3}"
export OPENCLAW_ASSISTANT_STATE_CONTEXT="${OPENCLAW_ASSISTANT_STATE_CONTEXT:-${MANYFORGE_STATE_CONTEXT}}"
export OPENCLAW_ASSISTANT_STATE_CONTEXT_EVERY_N="${OPENCLAW_ASSISTANT_STATE_CONTEXT_EVERY_N:-${MANYFORGE_STATE_CONTEXT_EVERY_N}}"
#   MANYFORGE_STATE_CONTEXT={off|always|every_n|first_then_every_n}
#                        Controls the neutral compact state capsule in
#                        the request prompt. This is a fixed cadence,
#                        not a prompt-keyword router. Default includes
#                        state on the first request of a session and
#                        then every Nth request.
#   MANYFORGE_STATE_CONTEXT_EVERY_N=3
#                        N for every_n / first_then_every_n. Lane-local
#                        OPENCLAW_ASSISTANT_STATE_CONTEXT* overrides are
#                        accepted but not required.
#   COMPACT_EVERY_N=2  → Production iter-32 chain-on recipe (2026-05-10).
#                        Fires `/compact` to the gateway before every 2nd
#                        user prompt on the same conversation. The
#                        iter-32 baseline of 51/66 (77.3%) on
#                        cosmos-reason2-8b was measured WITH this
#                        cadence. Without it, context bloats across
#                        long sessions and the smoke result regresses
#                        sharply (observed 17/66 / 25.8% during the
#                        Phase 0 baseline pass on 2026-06-03 when this
#                        env was unset by default).
#                        Set to 0 to disable compaction entirely for
#                        debugging or for the chain-session-OFF iter-28
#                        production recipe (which also reaches ~51/66
#                        via different mechanics).
#   COMPACT_TIMEOUT_S=120 → Per-compaction request timeout. 120s leaves
#                          enough headroom for the gateway's compaction
#                          path to complete on a fully loaded context.
export OPENCLAW_ASSISTANT_COMPACT_EVERY_N="${OPENCLAW_ASSISTANT_COMPACT_EVERY_N:-2}"
export OPENCLAW_ASSISTANT_COMPACT_TIMEOUT_S="${OPENCLAW_ASSISTANT_COMPACT_TIMEOUT_S:-120}"
#   TIMEOUT_S=300  → Production iter-32 chain-on recipe (2026-05-10).
#                    Per-request timeout the bridge applies to the
#                    underlying `openclaw agent` invocation (or
#                    gateway POST when USE_GATEWAY=true). The
#                    bridge's internal default is 120s, which is too
#                    tight for compacted chain-on conversations where
#                    the model often spends 60-90s thinking on hard
#                    cases before emitting the first tool call. With
#                    the default 120s, 5 cases in the 2026-06-03
#                    Phase 0 OpenClaw smoke timed out at ~126s with
#                    `chat HTTP 504` — the iter-32 51/66 baseline
#                    was measured with the 300s value. Strictly
#                    additive: longer timeout only changes behavior
#                    when a request would otherwise time out, so
#                    raising the default cannot regress cases that
#                    were passing under the old 120s ceiling.
export OPENCLAW_ASSISTANT_TIMEOUT_S="${OPENCLAW_ASSISTANT_TIMEOUT_S:-300}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
exec "${VENV}/bin/python" -m openclaw_assistant_bridge.service
