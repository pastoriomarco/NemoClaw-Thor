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
#   TOOL_SURFACE=code    → OpenClaw 2026.5.6+ default. tools[] contains only
#                           ``tool_search_code``; the bridge prompt teaches
#                           the model to wrap real calls inside its body.
#                           Set to "tools" if the sandbox-side OpenClaw
#                           config exposes the three discrete control verbs
#                           (tool_search / tool_describe / tool_call) — the
#                           prompt then switches to the tools-mode primer.
#                           The vLLM proxy (see scripts/proxy/vllm-proxy.py)
#                           cross-checks the observed tools[] against this
#                           env and emits ``tool_surface_mismatch`` if the
#                           operator misconfigured. Default "code" matches
#                           current production; unset is logged at INFO.
export OPENCLAW_ASSISTANT_LOCAL="${OPENCLAW_ASSISTANT_LOCAL:-false}"
export OPENCLAW_ASSISTANT_USE_GATEWAY="${OPENCLAW_ASSISTANT_USE_GATEWAY:-false}"
export OPENCLAW_ASSISTANT_AGENT="${OPENCLAW_ASSISTANT_AGENT:-manyforge-composer}"
export OPENCLAW_ASSISTANT_LOOP_TOOL_THRESHOLD="${OPENCLAW_ASSISTANT_LOOP_TOOL_THRESHOLD:-5}"
export OPENCLAW_ASSISTANT_LOOP_ARGS_THRESHOLD="${OPENCLAW_ASSISTANT_LOOP_ARGS_THRESHOLD:-2}"
export OPENCLAW_ASSISTANT_TOOL_SURFACE="${OPENCLAW_ASSISTANT_TOOL_SURFACE:-code}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
exec "${VENV}/bin/python" -m openclaw_assistant_bridge.service
