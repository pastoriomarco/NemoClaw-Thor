#!/usr/bin/env bash
# lib/sandbox-runtime.sh — Helpers for syncing NemoClaw runtime config inside a sandbox
#
# Source this file; do not execute it directly.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This file is meant to be sourced, not executed directly." >&2
    exit 1
fi

thor_openclaw_tools_deny_json() {
    python3 - <<'PYEOF'
import json

print(json.dumps([
    "cron",
    "web_fetch",
    "web_search",
]))
PYEOF
}

thor_openshell_gateway_name() {
    printf '%s\n' "${THOR_OPENSHELL_GATEWAY_NAME:-nemoclaw}"
}

thor_openshell_cluster_container_name() {
    command -v docker &>/dev/null || return 1

    local container_name="openshell-cluster-$(thor_openshell_gateway_name)"
    if docker ps -a --format '{{.Names}}' | grep -Fx "${container_name}" >/dev/null 2>&1; then
        printf '%s\n' "${container_name}"
        return 0
    fi

    return 1
}

gateway_ssh_handshake_secret() {
    local cluster_container=""
    cluster_container=$(thor_openshell_cluster_container_name 2>/dev/null || true)
    [[ -n "${cluster_container}" ]] || return 1

    # Read from the running openshell-0 pod, not the statefulset spec.
    # After a reboot the gateway can rotate its handshake secret during init,
    # and the statefulset spec may lag behind the running process.
    local live_secret=""
    live_secret=$(docker exec "${cluster_container}" \
        kubectl -n openshell exec openshell-0 -- printenv OPENSHELL_SSH_HANDSHAKE_SECRET 2>/dev/null || true)

    if [[ -n "${live_secret}" ]]; then
        printf '%s' "${live_secret}"
        return 0
    fi

    # Fallback to statefulset spec if the pod is not yet exec-ready.
    docker exec \
        "${cluster_container}" \
        sh -lc "kubectl -n openshell get statefulset openshell -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name==\"OPENSHELL_SSH_HANDSHAKE_SECRET\")].value}'"
}

sandbox_ssh_handshake_secret() {
    local sandbox_name="${1:-}"
    local cluster_container=""

    [[ -n "${sandbox_name}" ]] || return 1
    cluster_container=$(thor_openshell_cluster_container_name 2>/dev/null || true)
    [[ -n "${cluster_container}" ]] || return 1

    # Read from the running sandbox pod first, matching the gateway-side approach.
    local live_secret=""
    live_secret=$(docker exec "${cluster_container}" \
        kubectl -n openshell exec "${sandbox_name}" -- printenv OPENSHELL_SSH_HANDSHAKE_SECRET 2>/dev/null || true)

    if [[ -n "${live_secret}" ]]; then
        printf '%s' "${live_secret}"
        return 0
    fi

    # Fallback to sandbox CRD spec if the pod is not yet exec-ready.
    docker exec \
        "${cluster_container}" \
        sh -lc "kubectl -n openshell get sandbox \"${sandbox_name}\" -o jsonpath='{.spec.podTemplate.spec.containers[0].env[?(@.name==\"OPENSHELL_SSH_HANDSHAKE_SECRET\")].value}'"
}

wait_for_sandbox_pod_ready() {
    local sandbox_name="${1:-}"
    local cluster_container=""
    local attempts="${2:-60}"

    [[ -n "${sandbox_name}" ]] || return 1
    cluster_container=$(thor_openshell_cluster_container_name 2>/dev/null || true)
    [[ -n "${cluster_container}" ]] || return 1

    local i phase ready
    for i in $(seq 1 "${attempts}"); do
        phase=$(docker exec "${cluster_container}" sh -lc "kubectl -n openshell get pod \"${sandbox_name}\" -o jsonpath='{.status.phase}'" 2>/dev/null || true)
        ready=$(docker exec "${cluster_container}" sh -lc "kubectl -n openshell get pod \"${sandbox_name}\" -o jsonpath='{.status.containerStatuses[0].ready}'" 2>/dev/null || true)
        if [[ "${phase}" == "Running" && "${ready}" == "true" ]]; then
            return 0
        fi
        sleep 2
    done

    return 1
}

reconcile_sandbox_ssh_handshake_secret() {
    local sandbox_name="${1:-}"
    local cluster_container=""
    local gateway_secret=""
    local sandbox_secret=""
    local patch_json=""

    [[ -n "${sandbox_name}" ]] || return 1
    cluster_container=$(thor_openshell_cluster_container_name 2>/dev/null || true)
    [[ -n "${cluster_container}" ]] || return 1

    gateway_secret=$(gateway_ssh_handshake_secret 2>/dev/null || true)
    sandbox_secret=$(sandbox_ssh_handshake_secret "${sandbox_name}" 2>/dev/null || true)

    if [[ -z "${gateway_secret}" || -z "${sandbox_secret}" ]]; then
        fail "Could not inspect OpenShell SSH handshake state for '${sandbox_name}'"
        fix "Check: ./start-gateway.sh"
        fix "Check: openshell sandbox get ${sandbox_name}"
        return 1
    fi

    if [[ "${gateway_secret}" == "${sandbox_secret}" ]]; then
        pass "Sandbox SSH handshake secret matches the gateway"
        return 0
    fi

    warn "Sandbox SSH handshake secret drifted from the gateway"
    info "Updating sandbox '${sandbox_name}' to the current gateway handshake secret..."

    patch_json=$(
        python3 -c '
import json
import sys

gateway_secret = sys.argv[1]
data = json.load(sys.stdin)
env = (
    data.get("spec", {})
    .get("podTemplate", {})
    .get("spec", {})
    .get("containers", [{}])[0]
    .get("env", [])
)

for idx, item in enumerate(env):
    if item.get("name") == "OPENSHELL_SSH_HANDSHAKE_SECRET":
        if item.get("value") == gateway_secret:
            print("")
        else:
            print(json.dumps([{
                "op": "replace",
                "path": f"/spec/podTemplate/spec/containers/0/env/{idx}/value",
                "value": gateway_secret,
            }]))
        break
else:
    print(json.dumps([{
        "op": "add",
        "path": "/spec/podTemplate/spec/containers/0/env/-",
        "value": {
            "name": "OPENSHELL_SSH_HANDSHAKE_SECRET",
            "value": gateway_secret,
        },
    }]))
' "${gateway_secret}" \
        < <(docker exec "${cluster_container}" sh -lc "kubectl -n openshell get sandbox \"${sandbox_name}\" -o json")
    )

    if [[ -n "${patch_json}" ]]; then
        docker exec "${cluster_container}" \
            sh -lc "kubectl -n openshell patch sandbox \"${sandbox_name}\" --type='json' -p='${patch_json}'" >/dev/null
        docker exec "${cluster_container}" \
            sh -lc "kubectl -n openshell delete pod \"${sandbox_name}\" --wait=false" >/dev/null
    fi

    if ! wait_for_sandbox_pod_ready "${sandbox_name}" 60; then
        fail "Sandbox '${sandbox_name}' did not return to Ready after SSH handshake reconciliation"
        fix "Check: openshell sandbox get ${sandbox_name}"
        fix "Check: docker logs $(thor_openshell_cluster_container_name) | tail -n 80"
        return 1
    fi

    pass "Sandbox SSH handshake secret matches the gateway"
}

sandbox_runtime_config_summary_json() {
    local sandbox_name="${1:-}"
    local cluster_container=""

    [[ -n "${sandbox_name}" ]] || return 1
    cluster_container=$(thor_openshell_cluster_container_name 2>/dev/null || true)
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
    primary_model = (
        openclaw_cfg.get("agents", {})
        .get("defaults", {})
        .get("model", {})
        .get("primary")
    )
    out["openclaw_primary_model"] = primary_model
    inference_cfg = (
        openclaw_cfg.get("models", {})
        .get("providers", {})
        .get("inference", {})
    )
    models = inference_cfg.get("models") or []
    out["openclaw_inference_model_ids"] = [m.get("id") for m in models if isinstance(m, dict)]
    out["openclaw_inference_api"] = inference_cfg.get("api")
    matched_model = None
    if primary_model and "/" in primary_model:
        primary_model_id = primary_model.split("/", 1)[1]
        for model in models:
            if isinstance(model, dict) and model.get("id") == primary_model_id:
                matched_model = model
                break
    if matched_model is None and models:
        first_model = models[0]
        if isinstance(first_model, dict):
            matched_model = first_model
    out["openclaw_inference_context_window"] = (
        matched_model.get("contextWindow") if isinstance(matched_model, dict) else None
    )
    out["openclaw_tools_profile"] = (
        openclaw_cfg.get("tools", {})
        .get("profile")
    )
    out["openclaw_tools_deny"] = (
        openclaw_cfg.get("tools", {})
        .get("deny") or []
    )
    out["openclaw_parallel_tool_calls"] = (
        openclaw_cfg.get("agents", {})
        .get("defaults", {})
        .get("models", {})
        .get(primary_model or "", {})
        .get("params", {})
        .get("parallel_tool_calls")
    )
    out["openclaw_main_max_concurrent"] = (
        openclaw_cfg.get("agents", {})
        .get("defaults", {})
        .get("maxConcurrent")
    )
    out["openclaw_subagents_max_concurrent"] = (
        openclaw_cfg.get("agents", {})
        .get("defaults", {})
        .get("subagents", {})
        .get("maxConcurrent")
    )
    out["openclaw_subagents_max_children"] = (
        openclaw_cfg.get("agents", {})
        .get("defaults", {})
        .get("subagents", {})
        .get("maxChildrenPerAgent")
    )
    out["openclaw_subagents_max_spawn_depth"] = (
        openclaw_cfg.get("agents", {})
        .get("defaults", {})
        .get("subagents", {})
        .get("maxSpawnDepth")
    )
    out["openclaw_temperature"] = (
        openclaw_cfg.get("agents", {})
        .get("defaults", {})
        .get("models", {})
        .get(primary_model or "", {})
        .get("params", {})
        .get("temperature")
    )
else:
    out["openclaw_primary_model"] = None
    out["openclaw_inference_model_ids"] = []
    out["openclaw_inference_api"] = None
    out["openclaw_inference_context_window"] = None
    out["openclaw_tools_profile"] = None
    out["openclaw_tools_deny"] = []
    out["openclaw_parallel_tool_calls"] = None
    out["openclaw_main_max_concurrent"] = None
    out["openclaw_subagents_max_concurrent"] = None
    out["openclaw_subagents_max_children"] = None
    out["openclaw_subagents_max_spawn_depth"] = None
    out["openclaw_temperature"] = None

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

    cluster_container=$(thor_openshell_cluster_container_name 2>/dev/null || true)
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
        -e THOR_TARGET_MAX_MODEL_LEN="${THOR_TARGET_MAX_MODEL_LEN}" \
        -e THOR_EFFECTIVE_OPENCLAW_MAIN_MAX_CONCURRENT="${THOR_EFFECTIVE_OPENCLAW_MAIN_MAX_CONCURRENT}" \
        -e THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CONCURRENT="${THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CONCURRENT}" \
        -e THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CHILDREN="${THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CHILDREN}" \
        -e THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_SPAWN_DEPTH="${THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_SPAWN_DEPTH}" \
        -e THOR_OPENCLAW_TOOLS_DENY_JSON="$(thor_openclaw_tools_deny_json)" \
        "${cluster_container}" \
        sh <<'SH'
set -euo pipefail

kubectl -n openshell exec "${THOR_SANDBOX_NAME}" -- env \
    THOR_MODEL_ID="${THOR_MODEL_ID}" \
    THOR_LOCAL_PROVIDER_NAME="${THOR_LOCAL_PROVIDER_NAME}" \
    THOR_TARGET_MAX_MODEL_LEN="${THOR_TARGET_MAX_MODEL_LEN}" \
    THOR_EFFECTIVE_OPENCLAW_MAIN_MAX_CONCURRENT="${THOR_EFFECTIVE_OPENCLAW_MAIN_MAX_CONCURRENT}" \
    THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CONCURRENT="${THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CONCURRENT}" \
    THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CHILDREN="${THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CHILDREN}" \
    THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_SPAWN_DEPTH="${THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_SPAWN_DEPTH}" \
    THOR_OPENCLAW_TOOLS_DENY_JSON="${THOR_OPENCLAW_TOOLS_DENY_JSON}" \
    sh -lc '
python3 - <<'"'"'PY'"'"'
import json
import os
import pwd
from datetime import datetime, timezone
from pathlib import Path

model_id = os.environ["THOR_MODEL_ID"]
provider_name = os.environ["THOR_LOCAL_PROVIDER_NAME"]
provider_label = "Local vLLM" if provider_name == "vllm-local" else provider_name
primary_model = f"inference/{model_id}"
context_window = int(os.environ.get("THOR_TARGET_MAX_MODEL_LEN") or "65536")
main_max_concurrent = int(os.environ.get("THOR_EFFECTIVE_OPENCLAW_MAIN_MAX_CONCURRENT") or "1")
subagents_max_concurrent = int(os.environ.get("THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CONCURRENT") or "1")
subagents_max_children = int(os.environ.get("THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CHILDREN") or "1")
subagents_max_spawn_depth = int(os.environ.get("THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_SPAWN_DEPTH") or "1")
tools_deny = json.loads(os.environ["THOR_OPENCLAW_TOOLS_DENY_JSON"])

onboard_path = Path("/sandbox/.nemoclaw/config.json")
openclaw_path = Path("/sandbox/.openclaw/openclaw.json")
openclaw_identity_dir = openclaw_path.parent / "identity"
openclaw_device_json = openclaw_identity_dir / "device.json"
openclaw_device_auth_json = openclaw_identity_dir / "device-auth.json"
openclaw_devices_dir = openclaw_path.parent / "devices"
openclaw_devices_pending_json = openclaw_devices_dir / "pending.json"
openclaw_devices_paired_json = openclaw_devices_dir / "paired.json"

try:
    sandbox_user = pwd.getpwnam("sandbox")
    sandbox_uid = sandbox_user.pw_uid
    sandbox_gid = sandbox_user.pw_gid
except KeyError:
    sandbox_uid = os.getuid()
    sandbox_gid = os.getgid()

def chown_path(path: Path) -> None:
    try:
        os.chown(path, sandbox_uid, sandbox_gid)
    except FileNotFoundError:
        pass

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
chown_path(onboard_path.parent)
chown_path(onboard_path)
os.chmod(onboard_path, 0o600)

with openclaw_path.open(encoding="utf-8") as f:
    openclaw_cfg = json.load(f)

agents_defaults = openclaw_cfg.setdefault("agents", {}).setdefault("defaults", {})
agents_defaults.setdefault("model", {})["primary"] = primary_model
agents_defaults["maxConcurrent"] = main_max_concurrent
agents_defaults.setdefault("models", {})[primary_model] = {
    "params": {
        "parallel_tool_calls": False,
        "temperature": 0,
    }
}
subagents_cfg = agents_defaults.setdefault("subagents", {})
subagents_cfg["maxConcurrent"] = subagents_max_concurrent
subagents_cfg["maxChildrenPerAgent"] = subagents_max_children
subagents_cfg["maxSpawnDepth"] = subagents_max_spawn_depth

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
            "contextWindow": context_window,
            "maxTokens": 4096,
        }
    ],
}

gateway_cfg = openclaw_cfg.setdefault("gateway", {})
gateway_cfg["mode"] = "local"
control_ui_cfg = gateway_cfg.setdefault("controlUi", {})
control_ui_cfg["allowInsecureAuth"] = True
control_ui_cfg["dangerouslyDisableDeviceAuth"] = True
control_ui_cfg["allowedOrigins"] = [
    "http://127.0.0.1:18789",
    "http://localhost:18789",
    "http://[::1]:18789",
]
gateway_cfg["trustedProxies"] = ["127.0.0.1", "::1"]

tools_cfg = openclaw_cfg.setdefault("tools", {})
tools_cfg["profile"] = "coding"
tools_cfg["deny"] = tools_deny
tools_cfg.setdefault("sandbox", {}).setdefault("tools", {})["allow"] = [
    "group:fs",
    "group:runtime",
]
tools_cfg["sandbox"]["tools"]["deny"] = []

openclaw_path.parent.mkdir(parents=True, exist_ok=True)
openclaw_identity_dir.mkdir(parents=True, exist_ok=True)
openclaw_devices_dir.mkdir(parents=True, exist_ok=True)
chown_path(openclaw_identity_dir)
chown_path(openclaw_devices_dir)
os.chmod(openclaw_identity_dir, 0o700)
os.chmod(openclaw_devices_dir, 0o700)
for writable_path in (
    openclaw_device_json,
    openclaw_device_auth_json,
    openclaw_devices_pending_json,
    openclaw_devices_paired_json,
):
    if writable_path.exists():
        chown_path(writable_path)
        os.chmod(writable_path, 0o600)

with openclaw_path.open("w", encoding="utf-8") as f:
    json.dump(openclaw_cfg, f, indent=2)
    f.write("\n")
os.chmod(openclaw_path, 0o444)
PY
'
SH
}
