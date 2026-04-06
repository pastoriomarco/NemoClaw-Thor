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

kubectl -n openshell exec -i "${THOR_SANDBOX_NAME}" -- env \
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
    python3 - <<'PY'
import json
import os
from pathlib import Path

model_id = os.environ["THOR_MODEL_ID"]
context_window = int(os.environ.get("THOR_TARGET_MAX_MODEL_LEN") or "65536")

openclaw_path = Path("/sandbox/.openclaw/openclaw.json")

with openclaw_path.open(encoding="utf-8") as f:
    openclaw_cfg = json.load(f)

# --- Proven necessary by hand (v4 session) ---
# Fix provider API: onboard bakes openai-responses into the sandbox image,
# which bypasses vLLM's --tool-call-parser and breaks tool calling.
inference = (
    openclaw_cfg.setdefault("models", {})
    .setdefault("providers", {})
    .setdefault("inference", {})
)
inference["api"] = "openai-completions"
inference["baseUrl"] = "https://inference.local/v1"

for model in inference.get("models", []):
    if isinstance(model, dict):
        model["reasoning"] = os.environ.get("THOR_TARGET_MODEL_REASONING", "false").lower() == "true"
        model["maxTokens"] = int(os.environ.get("THOR_TARGET_MAX_TOKENS") or "16384")
        model["contextWindow"] = context_window

# --- Not yet proven necessary with v4 stack ---
# Uncomment individually as testing reveals they're needed.

# # --- Onboard config rewrite (/sandbox/.nemoclaw/config.json) ---
# # What: overwrites NemoClaw's own config file inside the sandbox with our
# #   provider name, model id, endpoint URL, and a fresh onboardedAt timestamp.
# # Why it existed: when switching models (e.g. qwopus → gemma4), the onboard
# #   config still referenced the old model. NemoClaw reads this on startup to
# #   determine which provider/model to use for `nemoclaw agent` dispatch.
# # Why it may not be needed: we only use one model profile at a time now, and
# #   onboard already wrote the correct values. Only matters if model switching
# #   without re-onboarding is needed.
# #
# import pwd
# from datetime import datetime, timezone
# provider_name = os.environ["THOR_LOCAL_PROVIDER_NAME"]
# provider_label = "Local vLLM" if provider_name == "vllm-local" else provider_name
# onboard_path = Path("/sandbox/.nemoclaw/config.json")
# if onboard_path.exists():
#     with onboard_path.open(encoding="utf-8") as f:
#         onboard_cfg = json.load(f)
# else:
#     onboard_cfg = {}
# onboard_cfg.update({
#     "endpointType": "custom",
#     "endpointUrl": "https://inference.local/v1",
#     "ncpPartner": None,
#     "model": model_id,
#     "profile": "inference-local",
#     "credentialEnv": "OPENAI_API_KEY",
#     "provider": provider_name,
#     "providerLabel": provider_label,
#     "onboardedAt": datetime.now(timezone.utc).isoformat(),
# })
# onboard_path.parent.mkdir(parents=True, exist_ok=True)
# with onboard_path.open("w", encoding="utf-8") as f:
#     json.dump(onboard_cfg, f, indent=2)
#     f.write("\n")

# # --- Agent behavior (primary model, temperature, parallel tool calls, timeout) ---
# # What: sets the primary model reference (inference/<model_id>), forces
# #   temperature=0 for deterministic output, disables parallel_tool_calls,
# #   sets maxConcurrent=1 for the main agent, and timeoutSeconds=1800.
# # Why it existed: local models produce garbage when attempting multiple tool
# #   calls in one turn (parallel_tool_calls). temperature=0 prevents random
# #   variation in code generation. timeoutSeconds=1800 (30min) prevents long
# #   reasoning sessions from being killed by a short default timeout.
# #   maxConcurrent=1 ensures only one main agent turn runs at a time.
# # Why it may not be needed: OpenClaw may have sane defaults for these now.
# #   The onboard wizard may set them. Test: run an agent session and check
# #   if tool calls work, if output is deterministic, and if long tasks timeout.
# #
# primary_model = f"inference/{model_id}"
# agents_defaults = openclaw_cfg.setdefault("agents", {}).setdefault("defaults", {})
# agents_defaults.setdefault("model", {})["primary"] = primary_model
# agents_defaults["maxConcurrent"] = int(os.environ.get("THOR_EFFECTIVE_OPENCLAW_MAIN_MAX_CONCURRENT") or "1")
# agents_defaults["timeoutSeconds"] = 1800
# agents_defaults.setdefault("models", {})[primary_model] = {
#     "params": {
#         "parallel_tool_calls": False,
#         "temperature": 0,
#     }
# }

# # --- Subagent concurrency ---
# # What: configures how many subagents OpenClaw can spawn concurrently, how
# #   many children each agent can have, and max spawn depth. Values are
# #   calculated from max_num_seqs minus 1 (reserve 1 slot for main agent).
# #   For qwopus3.5-27b-nvfp4 (max_num_seqs=4): 3 subagent slots.
# # Why it existed: without this, OpenClaw uses its own defaults which may
# #   spawn more concurrent agents than vLLM can serve, causing queuing and
# #   timeouts. Each active agent consumes a vLLM sequence slot.
# # Why it may not be needed: if only running single-agent sessions (no
# #   subagent spawning), this has no effect. Only matters for multi-agent
# #   ManyForge orchestration sessions.
# #
# subagents_cfg = agents_defaults.setdefault("subagents", {})
# subagents_cfg["maxConcurrent"] = int(os.environ.get("THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CONCURRENT") or "1")
# subagents_cfg["maxChildrenPerAgent"] = int(os.environ.get("THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_CHILDREN") or "1")
# subagents_cfg["maxSpawnDepth"] = int(os.environ.get("THOR_EFFECTIVE_OPENCLAW_SUBAGENTS_MAX_SPAWN_DEPTH") or "1")

# # --- Gateway auth bypass ---
# # What: sets gateway.mode=local, disables device auth, allows insecure auth,
# #   whitelists localhost origins for the control UI, and trusts localhost as
# #   a proxy. This means the TUI connects without approval prompts.
# # Why it existed: the OpenClaw gateway requires device pairing by default —
# #   a new device must be approved on first connect. On v3, pod restarts
# #   wiped the identity/device files (stored in pod tmpfs), so every restart
# #   triggered a new approval prompt. These flags bypassed that entirely.
# # Why it may not be needed: with sandbox survival (0.0.22), identity files
# #   persist. Also NemoClaw v0.0.6 may handle gateway auth defaults during
# #   onboard. Test: `nemoclaw connect` — if the TUI connects without prompts,
# #   this is unnecessary.
# #
# gateway_cfg = openclaw_cfg.setdefault("gateway", {})
# gateway_cfg["mode"] = "local"
# control_ui_cfg = gateway_cfg.setdefault("controlUi", {})
# control_ui_cfg["allowInsecureAuth"] = True
# control_ui_cfg["dangerouslyDisableDeviceAuth"] = True
# control_ui_cfg["allowedOrigins"] = [
#     "http://127.0.0.1:18789",
#     "http://localhost:18789",
#     "http://[::1]:18789",
# ]
# gateway_cfg["trustedProxies"] = ["127.0.0.1", "::1"]

# # --- Tool deny list ---
# # What: sets tools.profile to "coding", denies cron/web_fetch/web_search,
# #   and restricts sandbox tools to fs and runtime groups only.
# # Why it existed: without a deny list, the agent has access to web_fetch
# #   and web_search which would fail silently against the egress firewall
# #   (wasting tokens on retries), and cron which has no use case. The
# #   sandbox tool restrictions prevent the agent from using tools meant for
# #   the host environment.
# # Why it may not be needed: if the egress firewall is off and you want
# #   the agent to have full tool access, this is unnecessary. If strict-local
# #   policy is active, these tools would fail anyway — but denying them
# #   prevents the agent from wasting tokens trying.
# #
# tools_deny = json.loads(os.environ["THOR_OPENCLAW_TOOLS_DENY_JSON"])
# tools_cfg = openclaw_cfg.setdefault("tools", {})
# tools_cfg["profile"] = "coding"
# tools_cfg["deny"] = tools_deny
# tools_cfg.setdefault("sandbox", {}).setdefault("tools", {})["allow"] = [
#     "group:fs",
#     "group:runtime",
# ]
# tools_cfg["sandbox"]["tools"]["deny"] = []

# # --- Directory/permission setup ---
# # What: creates .openclaw/identity, .openclaw/devices, .openclaw/cron dirs,
# #   chowns them to the sandbox user, sets restrictive permissions, and
# #   creates cron/jobs.json if missing.
# # Why it existed: the openclaw gateway crashes on startup if cron/jobs.json
# #   doesn't exist. The identity and devices dirs need to be writable by the
# #   sandbox user for device pairing to work. Without chown, these dirs are
# #   root-owned (created by kubectl exec running as root) and the sandbox
# #   user can't write device identity files.
# # Why it may not be needed: onboard may create these dirs with correct
# #   ownership. Sandbox survival means they persist. Only needed if the
# #   gateway crashes with "cron/jobs.json not found" or if device pairing
# #   fails with permission errors.
# #
# try:
#     sandbox_user = pwd.getpwnam("sandbox")
#     sandbox_uid = sandbox_user.pw_uid
#     sandbox_gid = sandbox_user.pw_gid
# except KeyError:
#     sandbox_uid = os.getuid()
#     sandbox_gid = os.getgid()
# def chown_path(path):
#     try: os.chown(path, sandbox_uid, sandbox_gid)
#     except FileNotFoundError: pass
# for d in [openclaw_path.parent / "identity", openclaw_path.parent / "devices", openclaw_path.parent / "cron"]:
#     d.mkdir(parents=True, exist_ok=True)
#     chown_path(d)
# cron_jobs = openclaw_path.parent / "cron" / "jobs.json"
# if not cron_jobs.exists():
#     cron_jobs.write_text("{}\n")

# --- Write openclaw.json ---
os.chmod(openclaw_path, 0o644)
with openclaw_path.open("w", encoding="utf-8") as f:
    json.dump(openclaw_cfg, f, indent=2)
    f.write("\n")
os.chmod(openclaw_path, 0o444)
PY
SH
}
