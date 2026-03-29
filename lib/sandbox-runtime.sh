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
        # Record the current pod UID so we can detect when the replacement pod
        # is up — without this, wait_for_sandbox_pod_ready can mistake the
        # still-running old pod as the new one (--wait=false returns immediately).
        local old_uid=""
        old_uid=$(docker exec "${cluster_container}" \
            sh -lc "kubectl -n openshell get pod \"${sandbox_name}\" -o jsonpath='{.metadata.uid}'" 2>/dev/null || true)

        docker exec "${cluster_container}" \
            sh -lc "kubectl -n openshell patch sandbox \"${sandbox_name}\" --type='json' -p='${patch_json}'" >/dev/null
        docker exec "${cluster_container}" \
            sh -lc "kubectl -n openshell delete pod \"${sandbox_name}\" --wait=false" >/dev/null

        # Wait until the pod UID changes (old pod gone, new pod created).
        if [[ -n "${old_uid}" ]]; then
            local wait_i new_uid
            for wait_i in $(seq 1 30); do
                new_uid=$(docker exec "${cluster_container}" \
                    sh -lc "kubectl -n openshell get pod \"${sandbox_name}\" -o jsonpath='{.metadata.uid}'" 2>/dev/null || true)
                if [[ -n "${new_uid}" && "${new_uid}" != "${old_uid}" ]]; then
                    break
                fi
                sleep 2
            done
        fi
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

sandbox_ssh_command() {
    local sandbox_name="${1:-}"
    shift
    local cmd="$*"

    [[ -n "${sandbox_name}" ]] || return 1
    [[ -n "${cmd}" ]] || return 1

    local openshell_bin
    openshell_bin=$(command -v openshell 2>/dev/null || echo "")
    [[ -n "${openshell_bin}" ]] || return 1

    local gateway_name
    gateway_name=$(thor_openshell_gateway_name)

    ssh -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR \
        -o "ProxyCommand=${openshell_bin} ssh-proxy --gateway-name ${gateway_name} --name ${sandbox_name}" \
        "sandbox@openshell-${sandbox_name}" "${cmd}"
}

ensure_sandbox_gateway_running() {
    local sandbox_name="${1:-}"
    [[ -n "${sandbox_name}" ]] || return 1

    # Ensure the host forward is active.
    local forward_running=""
    forward_running=$(openshell forward list 2>/dev/null \
        | sed 's/\x1b\[[0-9;]*m//g' \
        | awk -v name="${sandbox_name}" '$1 == name && $3 == "18789" && /running/ {found=1} END {print found+0}')

    if [[ "${forward_running}" != "1" ]]; then
        openshell forward stop 18789 "${sandbox_name}" 2>/dev/null || true
        openshell forward start 18789 "${sandbox_name}" --background >/dev/null 2>&1 || true
    fi

    # Everything below runs in a single SSH session inside the sandbox.
    # This is critical: the gateway must run in the SSH network namespace
    # (not the pod root namespace) for the host forward to reach it.
    #
    # The script:
    #   1. Kills any existing gateway (stale process from a previous run)
    #   2. Pre-creates a device identity + paired.json so the TUI doesn't
    #      need manual device approval (identities are lost on pod restart)
    #   3. Starts the gateway with --auth none
    #   4. Reports the result
    info "Starting OpenClaw gateway inside sandbox '${sandbox_name}'..."
    local gw_result=""
    gw_result=$(sandbox_ssh_command "${sandbox_name}" '
#!/bin/sh
set -e
HOME=/sandbox
export HOME

OPENCLAW_DIR="$HOME/.openclaw"
IDENTITY_DIR="$OPENCLAW_DIR/identity"
DEVICES_DIR="$OPENCLAW_DIR/devices"
CRON_DIR="$OPENCLAW_DIR/cron"
GW_LOG="$HOME/openclaw-gateway.log"

# 1. Kill any existing gateway process.
for pid_dir in /proc/[0-9]*; do
    pid="${pid_dir##*/}"
    comm=$(cat "$pid_dir/comm" 2>/dev/null || true)
    case "$comm" in openclaw*) kill "$pid" 2>/dev/null || true ;; esac
done
sleep 1

# 2. Ensure device identity exists and is pre-paired.
python3 - <<'"'"'PY'"'"'
import json, os, re, base64
from pathlib import Path
from datetime import datetime, timezone

identity_dir = Path("/sandbox/.openclaw/identity")
devices_dir = Path("/sandbox/.openclaw/devices")
device_path = identity_dir / "device.json"
paired_path = devices_dir / "paired.json"
pending_path = devices_dir / "pending.json"
cron_dir = Path("/sandbox/.openclaw/cron")
cron_jobs = cron_dir / "jobs.json"

# Ensure directories exist and are writable by sandbox user.
for d in [identity_dir, devices_dir, cron_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Ensure cron/jobs.json exists (gateway fails to start without it).
if not cron_jobs.exists():
    cron_jobs.write_text("{}\n")

# If no device identity exists, let openclaw generate one later.
if not device_path.exists():
    print("no-identity")
    raise SystemExit(0)

with device_path.open() as f:
    device = json.load(f)

device_id = device["deviceId"]
public_key_pem = device.get("publicKeyPem", "")

# Extract the raw Ed25519 public key from the SPKI-encoded PEM.
# The PEM contains a DER-encoded SPKI structure; for Ed25519 the
# raw 32-byte key starts at byte offset 12 (after the ASN.1 header).
# Convert to URL-safe base64 without padding to match what OpenClaw
# devices approve produces.
match = re.search(
    r"-----BEGIN PUBLIC KEY-----\s*(.+?)\s*-----END PUBLIC KEY-----",
    public_key_pem, re.DOTALL)
if match:
    spki_b64 = match.group(1).replace("\n", "")
    spki_bytes = base64.b64decode(spki_b64)
    # Ed25519 SPKI is 44 bytes: 12-byte header + 32-byte raw key
    raw_key = spki_bytes[12:] if len(spki_bytes) == 44 else spki_bytes
    public_key = base64.urlsafe_b64encode(raw_key).rstrip(b"=").decode()
else:
    public_key = ""

now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
paired = {
    device_id: {
        "deviceId": device_id,
        "publicKey": public_key,
        "displayName": "openclaw-tui",
        "platform": "linux",
        "clientId": "cli",
        "clientMode": "cli",
        "role": "operator",
        "roles": ["operator"],
        "scopes": [
            "operator.admin", "operator.read", "operator.write",
            "operator.approvals", "operator.pairing"
        ],
        "approvedScopes": [
            "operator.admin", "operator.read", "operator.write",
            "operator.approvals", "operator.pairing"
        ],
        "approvedAt": now_ms,
        "lastSeenAt": now_ms,
    }
}

paired_path.write_text(json.dumps(paired, indent=2) + "\n")
pending_path.write_text("[]\n")
print("pre-paired")
PY

# 3. Start the gateway.
nohup openclaw gateway run --auth none > "$GW_LOG" 2>&1 &
disown
GW_PID=$!
sleep 3

# 4. Check if it is alive.
if kill -0 "$GW_PID" 2>/dev/null; then
    echo "started"
else
    echo "failed"
    cat "$GW_LOG" 2>/dev/null | tail -5 >&2
fi
' 2>&1) || true

    if ! echo "${gw_result}" | grep -qE "started|already-running"; then
        warn "OpenClaw gateway may not have started correctly"
        info "${gw_result}"
        fix "Inside the sandbox, run: HOME=/sandbox openclaw gateway run &"
        return 2
    fi

    # Verify the gateway is actually serving by probing from the host side.
    # The PID check inside the SSH session can pass even if the gateway
    # crashes shortly after startup (e.g. config reload triggered by a
    # recent openclaw.json write from sync_sandbox_runtime_config).
    local probe_ok=false
    local probe_attempt
    for probe_attempt in 1 2 3 4 5 6; do
        sleep 2
        local probe_result=""
        probe_result=$(python3 -c "
import socket
try:
    s = socket.create_connection(('127.0.0.1', 18789), timeout=5)
    s.sendall(b'GET / HTTP/1.1\r\nHost: 127.0.0.1:18789\r\n\r\n')
    s.settimeout(5)
    data = s.recv(1024)
    s.close()
    print('ok' if data and b'HTTP/' in data else 'empty')
except ConnectionResetError:
    print('reset')
except Exception:
    print('error')
" 2>/dev/null || echo "error")

        if [[ "${probe_result}" == "ok" ]]; then
            probe_ok=true
            break
        fi
    done

    if [[ "${probe_ok}" == "true" ]]; then
        pass "OpenClaw gateway started successfully"
        return 0
    fi

    # Gateway didn't stabilize — try one full restart (kills + starts again).
    # This handles the case where a config file watcher triggered a crash
    # shortly after the first start.
    info "Gateway did not stabilize — retrying..."
    sandbox_ssh_command "${sandbox_name}" '
#!/bin/sh
HOME=/sandbox; export HOME
for pid_dir in /proc/[0-9]*; do
    pid="${pid_dir##*/}"
    comm=$(cat "$pid_dir/comm" 2>/dev/null || true)
    case "$comm" in openclaw*) kill "$pid" 2>/dev/null || true ;; esac
done
sleep 2
nohup openclaw gateway run --auth none > /sandbox/openclaw-gateway.log 2>&1 &
disown
sleep 3
kill -0 $! 2>/dev/null && echo "restarted" || echo "failed"
' >/dev/null 2>&1 || true

    for probe_attempt in 1 2 3 4 5 6; do
        sleep 2
        local retry_result=""
        retry_result=$(python3 -c "
import socket
try:
    s = socket.create_connection(('127.0.0.1', 18789), timeout=5)
    s.sendall(b'GET / HTTP/1.1\r\nHost: 127.0.0.1:18789\r\n\r\n')
    s.settimeout(5)
    data = s.recv(1024)
    s.close()
    print('ok' if data and b'HTTP/' in data else 'empty')
except ConnectionResetError:
    print('reset')
except Exception:
    print('error')
" 2>/dev/null || echo "error")

        if [[ "${retry_result}" == "ok" ]]; then
            pass "OpenClaw gateway started successfully (after retry)"
            return 0
        fi
    done

    warn "OpenClaw gateway did not stabilize after retry"
    fix "Inside the sandbox, run: HOME=/sandbox openclaw gateway run &"
    return 2
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
        -e THOR_TARGET_MODEL_REASONING="${THOR_TARGET_MODEL_REASONING}" \
        -e THOR_TARGET_MAX_TOKENS="${THOR_TARGET_MAX_TOKENS}" \
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
    THOR_TARGET_MODEL_REASONING="${THOR_TARGET_MODEL_REASONING}" \
    THOR_TARGET_MAX_TOKENS="${THOR_TARGET_MAX_TOKENS}" \
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
            "reasoning": os.environ.get("THOR_TARGET_MODEL_REASONING", "false").lower() == "true",
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": context_window,
            "maxTokens": int(os.environ.get("THOR_TARGET_MAX_TOKENS") or "16384"),
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

openclaw_cron_dir = openclaw_path.parent / "cron"
openclaw_cron_jobs = openclaw_cron_dir / "jobs.json"

openclaw_path.parent.mkdir(parents=True, exist_ok=True)
openclaw_identity_dir.mkdir(parents=True, exist_ok=True)
openclaw_devices_dir.mkdir(parents=True, exist_ok=True)
openclaw_cron_dir.mkdir(parents=True, exist_ok=True)
chown_path(openclaw_identity_dir)
chown_path(openclaw_devices_dir)
chown_path(openclaw_cron_dir)
os.chmod(openclaw_identity_dir, 0o700)
os.chmod(openclaw_devices_dir, 0o700)
os.chmod(openclaw_cron_dir, 0o755)
if not openclaw_cron_jobs.exists():
    openclaw_cron_jobs.write_text("{}\n")
chown_path(openclaw_cron_jobs)
os.chmod(openclaw_cron_jobs, 0o644)
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

# ---------------------------------------------------------------------------
# Patch OpenClaw pi-ai provider: redirect inference.local to local
# re-streaming proxy that buffers tool-call argument fragments.
#
# The qwen3_coder streaming tool parser in vLLM fragments large tool-call
# arguments across many SSE chunks. OpenClaw (or the parser) silently drops
# content.  The proxy buffers tool-call chunks and re-emits them intact.
# ---------------------------------------------------------------------------
oc_provider_js = Path(
    "/usr/local/lib/node_modules/openclaw/node_modules/"
    "@mariozechner/pi-ai/dist/providers/openai-completions.js"
)
if oc_provider_js.exists():
    import re as _re
    src = oc_provider_js.read_text(encoding="utf-8")
    q = chr(34)  # JS double quote — avoids single quotes that break sh -lc

    # --- Step 0: Strip ALL previous NemoClaw patches ---
    # Remove any "// NemoClaw: ..." blocks (redirect, shim, comments)
    src = _re.sub(r"\n\s*// NemoClaw:.*?(?=\n\s*(?:return |function |const |var |let |export |\}))", "", src, flags=_re.DOTALL)
    # Remove standalone NemoClaw comment lines that might remain
    src = _re.sub(r"\n\s*//\s*--\s*NemoClaw.*", "", src)
    # Undo "const client = new OpenAI" -> "return new OpenAI"
    src = src.replace("const client = new OpenAI({", "return new OpenAI({", 1)
    # Undo "let openaiStream" -> "const openaiStream"
    src = src.replace("let openaiStream = await client.chat.completions.create",
                      "const openaiStream = await client.chat.completions.create", 1)
    # Remove timeout addition if present
    src = src.replace("        timeout: 1800000,\n", "")
    # Undo stream:false if present
    src = src.replace("stream: false,", "stream: true,", 1)
    # Remove stale "return client;" lines
    src = src.replace("\n    return client;\n}", "\n}")
    src = src.replace("\n    return client;\n    }\n}", "\n}")
    # Clean up orphan closing braces from removed if-blocks
    src = src.replace("    });\n    }\n}", "    });\n}")

    # --- Step 1: Apply clean patches ---
    # (a) Add timeout to new OpenAI()
    src = src.replace(
        "defaultHeaders: headers,\n    });",
        "defaultHeaders: headers,\n        timeout: 1800000,\n    });",
        1
    )
    # (b) createClient: capture in variable, add redirect, return
    src = src.replace("return new OpenAI({", "const client = new OpenAI({", 1)
    anchor_pos = src.find("timeout: 1800000")
    if anchor_pos >= 0:
        close_pos = src.find("});", anchor_pos)
        if close_pos >= 0:
            ins = close_pos + len("});")
            redirect = (
                "\n    // NemoClaw: redirect to local re-streaming proxy\n"
                "    if (client.baseURL && client.baseURL.includes("
                + q + "inference.local" + q + ")) {\n"
                "        client.baseURL = "
                + q + "http://127.0.0.1:8001/v1" + q + ";\n"
                "    }\n"
                "    return client;"
            )
            src = src[:ins] + redirect + src[ins:]

    oc_provider_js.chmod(0o644)
    oc_provider_js.write_text(src, encoding="utf-8")
    oc_provider_js.chmod(0o444)
    # Remove stale backup if any
    backup = oc_provider_js.with_suffix(".js.orig")
    if backup.exists():
        backup.unlink()

# ---------------------------------------------------------------------------
# Deploy re-streaming proxy (buffers tool-call argument fragments).
# ---------------------------------------------------------------------------
import base64
_proxy_b64 = "IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiJSZS1zdHJlYW1pbmcgcHJveHkgZm9yIE9wZW5DbGF3IDwtPiB2TExNLgoKU2l0cyBiZXR3ZWVuIE9wZW5DbGF3IChzdHJlYW06dHJ1ZSkgYW5kIHZMTE0gKHN0cmVhbTp0cnVlKS4KQnVmZmVycyB0b29sLWNhbGwgYXJndW1lbnQgZnJhZ21lbnRzIGFuZCByZS1lbWl0cyB0aGVtIGFzIGNvbXBsZXRlIGNodW5rcy4KRm9yd2FyZHMgY29udGVudC9yZWFzb25pbmcgZGVsdGFzIGltbWVkaWF0ZWx5IHRvIGtlZXAgdGhlIGNvbm5lY3Rpb24gYWxpdmUuCgpVc2FnZToKICAgIHB5dGhvbjMgcmVzdHJlYW0tcHJveHkucHkgWy0tcG9ydCA4MDAxXSBbLS11cHN0cmVhbSBodHRwczovL2luZmVyZW5jZS5sb2NhbF0KClRoZSBwcm94eSBsaXN0ZW5zIG9uIEhUVFAgKG5vIFRMUykgYW5kIHNwZWFrcyBIVFRQUyB0byB0aGUgdXBzdHJlYW0uCiIiIgoKaW1wb3J0IGFyZ3BhcnNlCmltcG9ydCBqc29uCmltcG9ydCBzc2wKaW1wb3J0IHN5cwppbXBvcnQgdGltZQppbXBvcnQgdGhyZWFkaW5nCmZyb20gaHR0cC5zZXJ2ZXIgaW1wb3J0IEhUVFBTZXJ2ZXIsIEJhc2VIVFRQUmVxdWVzdEhhbmRsZXIKZnJvbSB1cmxsaWIucmVxdWVzdCBpbXBvcnQgUmVxdWVzdCwgdXJsb3Blbgpmcm9tIHVybGxpYi5lcnJvciBpbXBvcnQgVVJMRXJyb3IKCgpMT0dfRklMRSA9IE5vbmUKCgpkZWYgbG9nKG1zZyk6CiAgICB0cyA9IHRpbWUuc3RyZnRpbWUoIiVIOiVNOiVTIikKICAgIGxpbmUgPSBmIlt7dHN9XSB7bXNnfSIKICAgIHByaW50KGxpbmUsIGZpbGU9c3lzLnN0ZGVyciwgZmx1c2g9VHJ1ZSkKICAgIGlmIExPR19GSUxFOgogICAgICAgIHRyeToKICAgICAgICAgICAgd2l0aCBvcGVuKExPR19GSUxFLCAiYSIpIGFzIGY6CiAgICAgICAgICAgICAgICBmLndyaXRlKGxpbmUgKyAiXG4iKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHBhc3MKCgpkZWYgbWFrZV9zc2xfY3R4KCk6CiAgICBjdHggPSBzc2wuY3JlYXRlX2RlZmF1bHRfY29udGV4dCgpCiAgICBjdHguY2hlY2tfaG9zdG5hbWUgPSBGYWxzZQogICAgY3R4LnZlcmlmeV9tb2RlID0gc3NsLkNFUlRfTk9ORQogICAgcmV0dXJuIGN0eAoKClNTTF9DVFggPSBtYWtlX3NzbF9jdHgoKQoKCmNsYXNzIFByb3h5SGFuZGxlcihCYXNlSFRUUFJlcXVlc3RIYW5kbGVyKToKICAgIHVwc3RyZWFtOiBzdHIgPSAiaHR0cHM6Ly9pbmZlcmVuY2UubG9jYWwiCgogICAgZGVmIGxvZ19tZXNzYWdlKHNlbGYsIGZtdCwgKmFyZ3MpOgogICAgICAgIHBhc3MgICMgc3VwcHJlc3MgZGVmYXVsdCBhY2Nlc3MgbG9nCgogICAgZGVmIGRvX1BPU1Qoc2VsZik6CiAgICAgICAgY29udGVudF9sZW4gPSBpbnQoc2VsZi5oZWFkZXJzLmdldCgiQ29udGVudC1MZW5ndGgiLCAwKSkKICAgICAgICBib2R5ID0gc2VsZi5yZmlsZS5yZWFkKGNvbnRlbnRfbGVuKQoKICAgICAgICBpZiAiL2NoYXQvY29tcGxldGlvbnMiIG5vdCBpbiBzZWxmLnBhdGg6CiAgICAgICAgICAgIHNlbGYuX3Bhc3N0aHJvdWdoKGJvZHkpCiAgICAgICAgICAgIHJldHVybgoKICAgICAgICB0cnk6CiAgICAgICAgICAgIHJlcV9qc29uID0ganNvbi5sb2Fkcyhib2R5KQogICAgICAgIGV4Y2VwdCBqc29uLkpTT05EZWNvZGVFcnJvcjoKICAgICAgICAgICAgc2VsZi5fcGFzc3Rocm91Z2goYm9keSkKICAgICAgICAgICAgcmV0dXJuCgogICAgICAgIGlzX3N0cmVhbWluZyA9IHJlcV9qc29uLmdldCgic3RyZWFtIiwgRmFsc2UpCiAgICAgICAgaWYgbm90IGlzX3N0cmVhbWluZzoKICAgICAgICAgICAgc2VsZi5fcGFzc3Rocm91Z2goYm9keSkKICAgICAgICAgICAgcmV0dXJuCgogICAgICAgIHNlbGYuX3Jlc3RyZWFtKGJvZHksIHJlcV9qc29uKQoKICAgIGRlZiBkb19HRVQoc2VsZik6CiAgICAgICAgc2VsZi5fcGFzc3Rocm91Z2goYiIiKQoKICAgIGRlZiBfcGFzc3Rocm91Z2goc2VsZiwgYm9keSk6CiAgICAgICAgdXJsID0gc2VsZi51cHN0cmVhbSArIHNlbGYucGF0aAogICAgICAgIGhlYWRlcnMgPSB7CiAgICAgICAgICAgIGs6IHYgZm9yIGssIHYgaW4gc2VsZi5oZWFkZXJzLml0ZW1zKCkKICAgICAgICAgICAgaWYgay5sb3dlcigpIG5vdCBpbiAoImhvc3QiLCAidHJhbnNmZXItZW5jb2RpbmciKQogICAgICAgIH0KICAgICAgICByZXEgPSBSZXF1ZXN0KHVybCwgZGF0YT1ib2R5IGlmIHNlbGYuY29tbWFuZCA9PSAiUE9TVCIgZWxzZSBOb25lLAogICAgICAgICAgICAgICAgICAgICAgaGVhZGVycz1oZWFkZXJzLCBtZXRob2Q9c2VsZi5jb21tYW5kKQogICAgICAgIHRyeToKICAgICAgICAgICAgcmVzcCA9IHVybG9wZW4ocmVxLCBjb250ZXh0PVNTTF9DVFgsIHRpbWVvdXQ9MTgwMCkKICAgICAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKHJlc3Auc3RhdHVzKQogICAgICAgICAgICBmb3IgaywgdiBpbiByZXNwLmdldGhlYWRlcnMoKToKICAgICAgICAgICAgICAgIGlmIGsubG93ZXIoKSBub3QgaW4gKCJ0cmFuc2Zlci1lbmNvZGluZyIsKToKICAgICAgICAgICAgICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKGssIHYpCiAgICAgICAgICAgIHNlbGYuZW5kX2hlYWRlcnMoKQogICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKHJlc3AucmVhZCgpKQogICAgICAgIGV4Y2VwdCBVUkxFcnJvciBhcyBlOgogICAgICAgICAgICBsb2coZiJwYXNzdGhyb3VnaCBlcnJvcjoge2V9IikKICAgICAgICAgICAgc2VsZi5zZW5kX2Vycm9yKDUwMiwgc3RyKGUpKQoKICAgIGRlZiBfcmVzdHJlYW0oc2VsZiwgYm9keSwgcmVxX2pzb24pOgogICAgICAgIHVybCA9IHNlbGYudXBzdHJlYW0gKyBzZWxmLnBhdGgKICAgICAgICBsb2coZiJyZXN0cmVhbTogUE9TVCB7dXJsfSBtb2RlbD17cmVxX2pzb24uZ2V0KCdtb2RlbCcsJz8nKX0iKQogICAgICAgIGhlYWRlcnMgPSB7CiAgICAgICAgICAgIGs6IHYgZm9yIGssIHYgaW4gc2VsZi5oZWFkZXJzLml0ZW1zKCkKICAgICAgICAgICAgaWYgay5sb3dlcigpIG5vdCBpbiAoImhvc3QiLCAidHJhbnNmZXItZW5jb2RpbmciKQogICAgICAgIH0KICAgICAgICBoZWFkZXJzWyJBY2NlcHQiXSA9ICJ0ZXh0L2V2ZW50LXN0cmVhbSIKICAgICAgICByZXEgPSBSZXF1ZXN0KHVybCwgZGF0YT1ib2R5LCBoZWFkZXJzPWhlYWRlcnMsIG1ldGhvZD0iUE9TVCIpCgogICAgICAgIHRyeToKICAgICAgICAgICAgcmVzcCA9IHVybG9wZW4ocmVxLCBjb250ZXh0PVNTTF9DVFgsIHRpbWVvdXQ9MTgwMCkKICAgICAgICBleGNlcHQgVVJMRXJyb3IgYXMgZToKICAgICAgICAgICAgbG9nKGYicmVzdHJlYW0gdXBzdHJlYW0gZXJyb3I6IHtlfSIpCiAgICAgICAgICAgIHNlbGYuc2VuZF9lcnJvcig1MDIsIHN0cihlKSkKICAgICAgICAgICAgcmV0dXJuCgogICAgICAgIGxvZyhmInJlc3RyZWFtOiB1cHN0cmVhbSBjb25uZWN0ZWQsIHN0YXR1cz17cmVzcC5zdGF0dXN9IikKCiAgICAgICAgc2VsZi5zZW5kX3Jlc3BvbnNlKDIwMCkKICAgICAgICBzZWxmLnNlbmRfaGVhZGVyKCJDb250ZW50LVR5cGUiLCAidGV4dC9ldmVudC1zdHJlYW0iKQogICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNhY2hlLUNvbnRyb2wiLCAibm8tY2FjaGUiKQogICAgICAgIHNlbGYuc2VuZF9oZWFkZXIoIkNvbm5lY3Rpb24iLCAia2VlcC1hbGl2ZSIpCiAgICAgICAgc2VsZi5zZW5kX2hlYWRlcigiWC1BY2NlbC1CdWZmZXJpbmciLCAibm8iKQogICAgICAgIHNlbGYuZW5kX2hlYWRlcnMoKQoKICAgICAgICB0b29sX2NhbGxzID0ge30KICAgICAgICBmaW5pc2hfcmVhc29uID0gTm9uZQogICAgICAgIGxhc3RfY2h1bmtfYmFzZSA9IE5vbmUKICAgICAgICBldmVudHNfZm9yd2FyZGVkID0gMAogICAgICAgIGV2ZW50c19idWZmZXJlZCA9IDAKCiAgICAgICAgdHJ5OgogICAgICAgICAgICBidWYgPSBiIiIKICAgICAgICAgICAgZm9yIGxpbmVfYnl0ZXMgaW4gcmVzcDoKICAgICAgICAgICAgICAgIGJ1ZiArPSBsaW5lX2J5dGVzCiAgICAgICAgICAgICAgICB3aGlsZSBiIlxuXG4iIGluIGJ1ZjoKICAgICAgICAgICAgICAgICAgICBldmVudF9ieXRlcywgYnVmID0gYnVmLnNwbGl0KGIiXG5cbiIsIDEpCiAgICAgICAgICAgICAgICAgICAgZXZlbnRfc3RyID0gZXZlbnRfYnl0ZXMuZGVjb2RlKCJ1dGYtOCIsIGVycm9ycz0icmVwbGFjZSIpLnN0cmlwKCkKCiAgICAgICAgICAgICAgICAgICAgaWYgbm90IGV2ZW50X3N0cjoKICAgICAgICAgICAgICAgICAgICAgICAgY29udGludWUKCiAgICAgICAgICAgICAgICAgICAgaWYgZXZlbnRfc3RyID09ICJkYXRhOiBbRE9ORV0iOgogICAgICAgICAgICAgICAgICAgICAgICBpZiB0b29sX2NhbGxzOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgc2VsZi5fZmx1c2hfdG9vbF9jYWxscyh0b29sX2NhbGxzLCBmaW5pc2hfcmVhc29uLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBsYXN0X2NodW5rX2Jhc2UpCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBsb2coZiJyZXN0cmVhbTogZmx1c2hlZCB7bGVuKHRvb2xfY2FsbHMpfSB0b29sIGNhbGxzIikKICAgICAgICAgICAgICAgICAgICAgICAgc2VsZi53ZmlsZS53cml0ZShiImRhdGE6IFtET05FXVxuXG4iKQogICAgICAgICAgICAgICAgICAgICAgICBzZWxmLndmaWxlLmZsdXNoKCkKICAgICAgICAgICAgICAgICAgICAgICAgbG9nKGYicmVzdHJlYW06IGRvbmUuIGZvcndhcmRlZD17ZXZlbnRzX2ZvcndhcmRlZH0gIgogICAgICAgICAgICAgICAgICAgICAgICAgICAgZiJidWZmZXJlZD17ZXZlbnRzX2J1ZmZlcmVkfSIpCiAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybgoKICAgICAgICAgICAgICAgICAgICBpZiBub3QgZXZlbnRfc3RyLnN0YXJ0c3dpdGgoImRhdGE6ICIpOgogICAgICAgICAgICAgICAgICAgICAgICBzZWxmLndmaWxlLndyaXRlKGV2ZW50X2J5dGVzICsgYiJcblxuIikKICAgICAgICAgICAgICAgICAgICAgICAgc2VsZi53ZmlsZS5mbHVzaCgpCiAgICAgICAgICAgICAgICAgICAgICAgIGNvbnRpbnVlCgogICAgICAgICAgICAgICAgICAgIGRhdGFfc3RyID0gZXZlbnRfc3RyWzY6XQogICAgICAgICAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgICAgICAgICAgY2h1bmsgPSBqc29uLmxvYWRzKGRhdGFfc3RyKQogICAgICAgICAgICAgICAgICAgIGV4Y2VwdCBqc29uLkpTT05EZWNvZGVFcnJvcjoKICAgICAgICAgICAgICAgICAgICAgICAgc2VsZi53ZmlsZS53cml0ZShldmVudF9ieXRlcyArIGIiXG5cbiIpCiAgICAgICAgICAgICAgICAgICAgICAgIHNlbGYud2ZpbGUuZmx1c2goKQogICAgICAgICAgICAgICAgICAgICAgICBldmVudHNfZm9yd2FyZGVkICs9IDEKICAgICAgICAgICAgICAgICAgICAgICAgY29udGludWUKCiAgICAgICAgICAgICAgICAgICAgY2hvaWNlID0gKGNodW5rLmdldCgiY2hvaWNlcyIpIG9yIFt7fV0pWzBdCiAgICAgICAgICAgICAgICAgICAgZGVsdGEgPSBjaG9pY2UuZ2V0KCJkZWx0YSIsIHt9KQogICAgICAgICAgICAgICAgICAgIGZyID0gY2hvaWNlLmdldCgiZmluaXNoX3JlYXNvbiIpCiAgICAgICAgICAgICAgICAgICAgaWYgZnI6CiAgICAgICAgICAgICAgICAgICAgICAgIGZpbmlzaF9yZWFzb24gPSBmcgoKICAgICAgICAgICAgICAgICAgICB0Y19kZWx0YXMgPSBkZWx0YS5nZXQoInRvb2xfY2FsbHMiKQogICAgICAgICAgICAgICAgICAgIGhhc19jb250ZW50ID0gImNvbnRlbnQiIGluIGRlbHRhCiAgICAgICAgICAgICAgICAgICAgaGFzX3JlYXNvbmluZyA9ICJyZWFzb25pbmdfY29udGVudCIgaW4gZGVsdGEKCiAgICAgICAgICAgICAgICAgICAgaWYgdGNfZGVsdGFzOgogICAgICAgICAgICAgICAgICAgICAgICBmb3IgdGMgaW4gdGNfZGVsdGFzOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgaWR4ID0gdGMuZ2V0KCJpbmRleCIsIDApCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiBpZHggbm90IGluIHRvb2xfY2FsbHM6CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdG9vbF9jYWxsc1tpZHhdID0gewogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiaWQiOiB0Yy5nZXQoImlkIiwgIiIpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAidHlwZSI6IHRjLmdldCgidHlwZSIsICJmdW5jdGlvbiIpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiZnVuY3Rpb24iOiB7CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAibmFtZSI6IHRjLmdldCgiZnVuY3Rpb24iLCB7fSkuZ2V0KCJuYW1lIiwgIiIpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgImFyZ3VtZW50cyI6ICIiLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9LAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0KICAgICAgICAgICAgICAgICAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgdGMuZ2V0KCJpZCIpOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0b29sX2NhbGxzW2lkeF1bImlkIl0gPSB0Y1siaWQiXQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZuID0gdGMuZ2V0KCJmdW5jdGlvbiIsIHt9KQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIGZuLmdldCgibmFtZSIpOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0b29sX2NhbGxzW2lkeF1bImZ1bmN0aW9uIl1bIm5hbWUiXSA9IGZuWyJuYW1lIl0KICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFyZ3NfZnJhZyA9IHRjLmdldCgiZnVuY3Rpb24iLCB7fSkuZ2V0KCJhcmd1bWVudHMiLCAiIikKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIGFyZ3NfZnJhZzoKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0b29sX2NhbGxzW2lkeF1bImZ1bmN0aW9uIl1bImFyZ3VtZW50cyJdICs9IGFyZ3NfZnJhZwogICAgICAgICAgICAgICAgICAgICAgICBldmVudHNfYnVmZmVyZWQgKz0gMQogICAgICAgICAgICAgICAgICAgICAgICBsYXN0X2NodW5rX2Jhc2UgPSBjaHVuawoKICAgICAgICAgICAgICAgICAgICAgICAgaWYgaGFzX2NvbnRlbnQgb3IgaGFzX3JlYXNvbmluZzoKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZ3ZF9kZWx0YSA9IHt9CiAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiBoYXNfY29udGVudDoKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBmd2RfZGVsdGFbImNvbnRlbnQiXSA9IGRlbHRhWyJjb250ZW50Il0KICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIGhhc19yZWFzb25pbmc6CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZndkX2RlbHRhWyJyZWFzb25pbmdfY29udGVudCJdID0gZGVsdGFbInJlYXNvbmluZ19jb250ZW50Il0KICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmICJyb2xlIiBpbiBkZWx0YToKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBmd2RfZGVsdGFbInJvbGUiXSA9IGRlbHRhWyJyb2xlIl0KICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZ3ZF9jaHVuayA9IGRpY3QoY2h1bmspCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBmd2RfY2hvaWNlID0gZGljdChjaG9pY2UpCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBmd2RfY2hvaWNlWyJkZWx0YSJdID0gZndkX2RlbHRhCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBmd2RfY2hvaWNlLnBvcCgiZmluaXNoX3JlYXNvbiIsIE5vbmUpCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBmd2RfY2h1bmtbImNob2ljZXMiXSA9IFtmd2RfY2hvaWNlXQogICAgICAgICAgICAgICAgICAgICAgICAgICAgc2VsZi5fc2VuZF9zc2UoZndkX2NodW5rKQogICAgICAgICAgICAgICAgICAgICAgICAgICAgZXZlbnRzX2ZvcndhcmRlZCArPSAxCiAgICAgICAgICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICAgICAgc2VsZi53ZmlsZS53cml0ZShldmVudF9ieXRlcyArIGIiXG5cbiIpCiAgICAgICAgICAgICAgICAgICAgICAgIHNlbGYud2ZpbGUuZmx1c2goKQogICAgICAgICAgICAgICAgICAgICAgICBldmVudHNfZm9yd2FyZGVkICs9IDEKCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgICAgICBsb2coZiJyZXN0cmVhbSBlcnJvcjoge3R5cGUoZSkuX19uYW1lX199OiB7ZX0iKQoKICAgICAgICBpZiB0b29sX2NhbGxzOgogICAgICAgICAgICBzZWxmLl9mbHVzaF90b29sX2NhbGxzKHRvb2xfY2FsbHMsIGZpbmlzaF9yZWFzb24sIGxhc3RfY2h1bmtfYmFzZSkKICAgICAgICBzZWxmLndmaWxlLndyaXRlKGIiZGF0YTogW0RPTkVdXG5cbiIpCiAgICAgICAgc2VsZi53ZmlsZS5mbHVzaCgpCiAgICAgICAgbG9nKGYicmVzdHJlYW06IGVuZGVkIChsb29wIGV4aXQpLiBmb3J3YXJkZWQ9e2V2ZW50c19mb3J3YXJkZWR9ICIKICAgICAgICAgICAgZiJidWZmZXJlZD17ZXZlbnRzX2J1ZmZlcmVkfSIpCgogICAgZGVmIF9mbHVzaF90b29sX2NhbGxzKHNlbGYsIHRvb2xfY2FsbHMsIGZpbmlzaF9yZWFzb24sIGJhc2VfY2h1bmspOgogICAgICAgIHRjX2xpc3QgPSBbXQogICAgICAgIGZvciBpZHggaW4gc29ydGVkKHRvb2xfY2FsbHMua2V5cygpKToKICAgICAgICAgICAgdGMgPSB0b29sX2NhbGxzW2lkeF0KICAgICAgICAgICAgdGNfbGlzdC5hcHBlbmQoewogICAgICAgICAgICAgICAgImluZGV4IjogaWR4LAogICAgICAgICAgICAgICAgImlkIjogdGNbImlkIl0sCiAgICAgICAgICAgICAgICAidHlwZSI6IHRjWyJ0eXBlIl0sCiAgICAgICAgICAgICAgICAiZnVuY3Rpb24iOiB7CiAgICAgICAgICAgICAgICAgICAgIm5hbWUiOiB0Y1siZnVuY3Rpb24iXVsibmFtZSJdLAogICAgICAgICAgICAgICAgICAgICJhcmd1bWVudHMiOiB0Y1siZnVuY3Rpb24iXVsiYXJndW1lbnRzIl0sCiAgICAgICAgICAgICAgICB9LAogICAgICAgICAgICB9KQoKICAgICAgICBjaHVuayA9IHsKICAgICAgICAgICAgImlkIjogYmFzZV9jaHVuay5nZXQoImlkIiwgIiIpIGlmIGJhc2VfY2h1bmsgZWxzZSAiIiwKICAgICAgICAgICAgIm9iamVjdCI6ICJjaGF0LmNvbXBsZXRpb24uY2h1bmsiLAogICAgICAgICAgICAiY3JlYXRlZCI6IGJhc2VfY2h1bmsuZ2V0KCJjcmVhdGVkIiwgMCkgaWYgYmFzZV9jaHVuayBlbHNlIDAsCiAgICAgICAgICAgICJtb2RlbCI6IGJhc2VfY2h1bmsuZ2V0KCJtb2RlbCIsICIiKSBpZiBiYXNlX2NodW5rIGVsc2UgIiIsCiAgICAgICAgICAgICJjaG9pY2VzIjogW3sKICAgICAgICAgICAgICAgICJpbmRleCI6IDAsCiAgICAgICAgICAgICAgICAiZGVsdGEiOiB7CiAgICAgICAgICAgICAgICAgICAgInRvb2xfY2FsbHMiOiB0Y19saXN0LAogICAgICAgICAgICAgICAgfSwKICAgICAgICAgICAgICAgICJmaW5pc2hfcmVhc29uIjogZmluaXNoX3JlYXNvbiBvciAidG9vbF9jYWxscyIsCiAgICAgICAgICAgIH1dLAogICAgICAgIH0KICAgICAgICBpZiBiYXNlX2NodW5rIGFuZCAidXNhZ2UiIGluIGJhc2VfY2h1bms6CiAgICAgICAgICAgIGNodW5rWyJ1c2FnZSJdID0gYmFzZV9jaHVua1sidXNhZ2UiXQoKICAgICAgICBmb3IgdGMgaW4gdGNfbGlzdDoKICAgICAgICAgICAgYXJnc19sZW4gPSBsZW4odGNbImZ1bmN0aW9uIl1bImFyZ3VtZW50cyJdKQogICAgICAgICAgICBsb2coZiIgIHRvb2xfY2FsbDoge3RjWydmdW5jdGlvbiddWyduYW1lJ119IGFyZ3NfbGVuPXthcmdzX2xlbn0iKQoKICAgICAgICBzZWxmLl9zZW5kX3NzZShjaHVuaykKCiAgICBkZWYgX3NlbmRfc3NlKHNlbGYsIGNodW5rKToKICAgICAgICBkYXRhID0ganNvbi5kdW1wcyhjaHVuaywgc2VwYXJhdG9ycz0oIiwiLCAiOiIpKQogICAgICAgIHNlbGYud2ZpbGUud3JpdGUoZiJkYXRhOiB7ZGF0YX1cblxuIi5lbmNvZGUoKSkKICAgICAgICBzZWxmLndmaWxlLmZsdXNoKCkKCgpkZWYgbWFpbigpOgogICAgZ2xvYmFsIExPR19GSUxFCiAgICBwYXJzZXIgPSBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcihkZXNjcmlwdGlvbj0iUmUtc3RyZWFtaW5nIHByb3h5IikKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tcG9ydCIsIHR5cGU9aW50LCBkZWZhdWx0PTgwMDEpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXVwc3RyZWFtIiwgZGVmYXVsdD0iaHR0cHM6Ly9pbmZlcmVuY2UubG9jYWwiKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1sb2ciLCBkZWZhdWx0PSIvc2FuZGJveC8ubmVtb2NsYXcvcmVzdHJlYW0tcHJveHkubG9nIikKICAgIGFyZ3MgPSBwYXJzZXIucGFyc2VfYXJncygpCgogICAgTE9HX0ZJTEUgPSBhcmdzLmxvZwogICAgUHJveHlIYW5kbGVyLnVwc3RyZWFtID0gYXJncy51cHN0cmVhbS5yc3RyaXAoIi8iKQoKICAgIHNlcnZlciA9IEhUVFBTZXJ2ZXIoKCIxMjcuMC4wLjEiLCBhcmdzLnBvcnQpLCBQcm94eUhhbmRsZXIpCiAgICBsb2coZiJsaXN0ZW5pbmcgb24gMTI3LjAuMC4xOnthcmdzLnBvcnR9IC0+IHthcmdzLnVwc3RyZWFtfSIpCgogICAgdGhyZWFkID0gdGhyZWFkaW5nLlRocmVhZCh0YXJnZXQ9c2VydmVyLnNlcnZlX2ZvcmV2ZXIsIGRhZW1vbj1UcnVlKQogICAgdGhyZWFkLnN0YXJ0KCkKICAgIHRyeToKICAgICAgICB0aHJlYWQuam9pbigpCiAgICBleGNlcHQgS2V5Ym9hcmRJbnRlcnJ1cHQ6CiAgICAgICAgc2VydmVyLnNodXRkb3duKCkKCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgbWFpbigpCg=="
_proxy_path = Path("/sandbox/.nemoclaw/restream-proxy.py")
_proxy_path.parent.mkdir(parents=True, exist_ok=True)
_proxy_path.write_bytes(base64.b64decode(_proxy_b64))
_proxy_path.chmod(0o755)

# Kill any existing proxy and start fresh
import subprocess
subprocess.run(["pkill", "-f", "restream-proxy"], capture_output=True)
subprocess.Popen(
    ["python3", str(_proxy_path), "--port", "8001",
     "--upstream", "https://inference.local"],
    stdout=open("/sandbox/.nemoclaw/restream-proxy.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
import time
time.sleep(0.5)
# Verify proxy started
import socket
_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    _sock.connect(("127.0.0.1", 8001))
    print("restream-proxy: listening on :8001", flush=True)
except ConnectionRefusedError:
    print("WARNING: restream-proxy failed to start on :8001", flush=True)
finally:
    _sock.close()
PY
'
SH
}
