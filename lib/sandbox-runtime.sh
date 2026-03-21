#!/usr/bin/env bash
# lib/sandbox-runtime.sh — Helpers for syncing NemoClaw runtime config inside a sandbox
#
# Source this file; do not execute it directly.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This file is meant to be sourced, not executed directly." >&2
    exit 1
fi

active_openshell_gateway_name() {
    command -v openshell &>/dev/null || return 1

    openshell gateway info 2>/dev/null \
        | sed 's/\x1b\[[0-9;]*m//g' \
        | awk -F': ' '/Gateway:/ {print $2; exit}'
}

active_openshell_cluster_container_name() {
    command -v docker &>/dev/null || return 1

    local gateway_name="${1:-}"
    if [[ -z "${gateway_name}" ]]; then
        gateway_name=$(active_openshell_gateway_name 2>/dev/null || true)
    fi

    [[ -n "${gateway_name}" ]] || return 1

    local container_name="openshell-cluster-${gateway_name}"
    if docker ps --format '{{.Names}}' | grep -Fx "${container_name}" >/dev/null 2>&1; then
        printf '%s\n' "${container_name}"
        return 0
    fi

    return 1
}

sandbox_runtime_config_summary_json() {
    local sandbox_name="${1:-}"
    local cluster_container=""

    [[ -n "${sandbox_name}" ]] || return 1
    cluster_container=$(active_openshell_cluster_container_name 2>/dev/null || true)
    [[ -n "${cluster_container}" ]] || return 1

    docker exec \
        -i \
        -e THOR_SANDBOX_NAME="${sandbox_name}" \
        "${cluster_container}" \
        sh <<'SH'
set -euo pipefail

kubectl -n openshell exec -i "${THOR_SANDBOX_NAME}" -- python3 - <<'PY'
import json
import os
from pathlib import Path

out = {}

onboard_path = Path("/sandbox/.nemoclaw/config.json")
openclaw_path = Path("/sandbox/.openclaw/openclaw.json")

out["onboard_exists"] = onboard_path.exists()
out["openclaw_exists"] = openclaw_path.exists()

if onboard_path.exists():
    with onboard_path.open(encoding="utf-8") as f:
        onboard_cfg = json.load(f)
    out["onboard_model"] = onboard_cfg.get("model")
    out["onboard_provider"] = onboard_cfg.get("provider")
else:
    out["onboard_model"] = None
    out["onboard_provider"] = None

if openclaw_path.exists():
    with openclaw_path.open(encoding="utf-8") as f:
        openclaw_cfg = json.load(f)
    out["openclaw_primary_model"] = (
        openclaw_cfg.get("agents", {})
        .get("defaults", {})
        .get("model", {})
        .get("primary")
    )
    inference_cfg = (
        openclaw_cfg.get("models", {})
        .get("providers", {})
        .get("inference", {})
    )
    models = inference_cfg.get("models") or []
    out["openclaw_inference_model_ids"] = [m.get("id") for m in models if isinstance(m, dict)]
else:
    out["openclaw_primary_model"] = None
    out["openclaw_inference_model_ids"] = []

print(json.dumps(out))
PY
SH
}

sync_sandbox_runtime_config() {
    local sandbox_name="${1:-}"
    local cluster_container=""

    if [[ -z "${sandbox_name}" ]]; then
        sandbox_name=$(resolve_thor_sandbox_name 2>/dev/null || true)
    fi

    if [[ -z "${sandbox_name}" ]]; then
        fail "Could not determine the sandbox name for runtime config sync"
        fix "Set THOR_MANAGED_SANDBOX_NAME in ${THOR_CONFIG_FILE}, or keep only one sandbox present."
        return 1
    fi

    cluster_container=$(active_openshell_cluster_container_name 2>/dev/null || true)
    if [[ -z "${cluster_container}" ]]; then
        fail "Could not determine the active OpenShell cluster container"
        fix "Check: openshell gateway info"
        fix "Check: docker ps --format '{{.Names}}'"
        return 1
    fi

    docker exec \
        -i \
        -e THOR_SANDBOX_NAME="${sandbox_name}" \
        -e THOR_MODEL_ID="${THOR_MODEL_ID}" \
        -e THOR_LOCAL_PROVIDER_NAME="${THOR_LOCAL_PROVIDER_NAME}" \
        "${cluster_container}" \
        sh <<'SH'
set -euo pipefail

kubectl -n openshell exec "${THOR_SANDBOX_NAME}" -- env \
    THOR_MODEL_ID="${THOR_MODEL_ID}" \
    THOR_LOCAL_PROVIDER_NAME="${THOR_LOCAL_PROVIDER_NAME}" \
    sh -lc '
python3 - <<'"'"'PY'"'"'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

model_id = os.environ["THOR_MODEL_ID"]
provider_name = os.environ["THOR_LOCAL_PROVIDER_NAME"]
provider_label = "Local vLLM" if provider_name == "vllm-local" else provider_name
primary_model = f"inference/{model_id}"

onboard_path = Path("/sandbox/.nemoclaw/config.json")
openclaw_path = Path("/sandbox/.openclaw/openclaw.json")

if onboard_path.exists():
    with onboard_path.open(encoding="utf-8") as f:
        onboard_cfg = json.load(f)
else:
    onboard_cfg = {}

onboard_cfg.update({
    "endpointType": "custom",
    "endpointUrl": "https://inference.local/v1",
    "ncpPartner": None,
    "model": model_id,
    "profile": "inference-local",
    "credentialEnv": "OPENAI_API_KEY",
    "provider": provider_name,
    "providerLabel": provider_label,
    "onboardedAt": datetime.now(timezone.utc).isoformat(),
})

onboard_path.parent.mkdir(parents=True, exist_ok=True)
with onboard_path.open("w", encoding="utf-8") as f:
    json.dump(onboard_cfg, f, indent=2)
    f.write("\n")
os.chmod(onboard_path, 0o600)

with openclaw_path.open(encoding="utf-8") as f:
    openclaw_cfg = json.load(f)

openclaw_cfg.setdefault("agents", {}).setdefault("defaults", {}).setdefault("model", {})["primary"] = primary_model
models_cfg = openclaw_cfg.setdefault("models", {})
models_cfg["mode"] = "merge"
providers_cfg = models_cfg.setdefault("providers", {})
providers_cfg["inference"] = {
    "baseUrl": "https://inference.local/v1",
    "apiKey": "unused",
    "api": "openai-completions",
    "models": [
        {
            "id": model_id,
            "name": model_id,
            "reasoning": False,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 131072,
            "maxTokens": 4096,
        }
    ],
}

with openclaw_path.open("w", encoding="utf-8") as f:
    json.dump(openclaw_cfg, f, indent=2)
    f.write("\n")
os.chmod(openclaw_path, 0o444)
PY
'
SH
}
