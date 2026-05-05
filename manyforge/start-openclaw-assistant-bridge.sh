#!/usr/bin/env bash
# start-openclaw-assistant-bridge.sh — run the experimental Phase 2 adapter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_DIR="${SCRIPT_DIR}/openclaw_assistant_bridge"
VENV="${BRIDGE_DIR}/.venv"

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
  "${VENV}/bin/pip" install -r "${BRIDGE_DIR}/requirements.txt"
fi

export OPENCLAW_ASSISTANT_LOCAL="${OPENCLAW_ASSISTANT_LOCAL:-true}"
export OPENCLAW_ASSISTANT_AGENT="${OPENCLAW_ASSISTANT_AGENT:-manyforge-composer}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
exec "${VENV}/bin/python" -m openclaw_assistant_bridge.service
