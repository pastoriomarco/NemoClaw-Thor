#!/usr/bin/env bash
# setup/checks.sh — Prerequisite check functions for NemoClaw-Thor
#
# Source this file; do not execute it directly.
# Each check_* function prints its own result line and returns:
#   0 — pass
#   1 — fail
#   2 — warning (condition is unusual but not necessarily blocking)
#
# Usage in calling scripts within setup/:
#   source "$(dirname "${BASH_SOURCE[0]}")/checks.sh"
#
# Usage from a script in a sibling scope (e.g. serving/start-model.sh):
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/setup/checks.sh"

# Guard against direct execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This file is meant to be sourced, not executed directly." >&2
    exit 1
fi

# ── Terminal colors ────────────────────────────────────────────────────────────
# Only use colors when stdout is a real terminal
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

# ── Output helpers ─────────────────────────────────────────────────────────────
pass()   { echo -e "${GREEN}  ✓${NC}  $*"; }
fail()   { echo -e "${RED}  ✗${NC}  $*"; }
warn()   { echo -e "${YELLOW}  !${NC}  $*"; }
info()   { echo -e "       $*"; }
header() { echo -e "\n${BOLD}$*${NC}"; }
fix()    { echo -e "       ${YELLOW}→${NC} $*"; }

# ── OpenShell checks ───────────────────────────────────────────────────────────

check_openshell_installed() {
    if command -v openshell &>/dev/null; then
        local ver
        ver=$(openshell --version 2>/dev/null || echo "version unknown")
        pass "openshell command found ($ver)"
        return 0
    else
        fail "openshell command not found"
        fix "Install OpenShell before continuing."
        fix "Then run ./apply-host-fixes.sh or see https://github.com/JetsonHacks/openshell-thor"
        return 1
    fi
}

check_openshell_gateway() {
    local gateway_name="${THOR_OPENSHELL_GATEWAY_NAME:-nemoclaw}"

    if openshell status -g "${gateway_name}" &>/dev/null; then
        pass "OpenShell gateway '${gateway_name}' is running"
        return 0
    else
        fail "OpenShell gateway '${gateway_name}' is not reachable"
        fix "Run: ./start-gateway.sh"
        return 1
    fi
}

ensure_openshell_gateway_running() {
    local gateway_name="${THOR_OPENSHELL_GATEWAY_NAME:-nemoclaw}"
    local sandbox_name="${THOR_OPENSHELL_SANDBOX:-${OPENCLAW_ASSISTANT_SANDBOX:-my-assistant}}"
    local attempts="${1:-60}"

    if ! command -v openshell &>/dev/null; then
        fail "openshell command not found"
        fix "Install OpenShell before continuing."
        return 1
    fi

    # Gateway runtime marker (docker driver): records the version + pid of the
    # daemon actually running.
    local gw_runtime="${NEMOCLAW_OPENSHELL_GATEWAY_STATE_DIR:-${HOME}/.local/state/nemoclaw/openshell-docker-gateway}/runtime.json"

    if openshell status -g "${gateway_name}" &>/dev/null; then
        # The gateway responds — but an in-place OpenShell upgrade swaps the
        # binary on disk WITHOUT restarting the running daemon, so a stale
        # gateway keeps serving the old wire protocol. Newly-created sandboxes
        # then fail (e.g. "no sandbox token source available" once the host
        # moved to 0.0.71). Compare the running version (recorded in the gateway
        # runtime marker) with the installed binary and force a restart on drift.
        local installed_ver running_ver
        installed_ver="$(openshell --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
        running_ver="$(grep -oE '"openshellVersion"[[:space:]]*:[[:space:]]*"[0-9.]+"' "${gw_runtime}" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
        if [[ -z "${installed_ver}" || -z "${running_ver}" || "${installed_ver}" == "${running_ver}" ]]; then
            pass "OpenShell gateway '${gateway_name}' is running (${running_ver:-version n/a})"
            return 0
        fi
        warn "OpenShell gateway is running ${running_ver} but ${installed_ver} is installed (stale after an in-place upgrade)"
        info "Stopping the stale gateway so it relaunches on ${installed_ver}..."
        local stale_pid w
        stale_pid="$(grep -oE '"pid"[[:space:]]*:[[:space:]]*[0-9]+' "${gw_runtime}" 2>/dev/null | grep -oE '[0-9]+' | head -1)"
        if [[ -n "${stale_pid}" ]]; then
            kill "${stale_pid}" 2>/dev/null || true
            for w in $(seq 1 20); do kill -0 "${stale_pid}" 2>/dev/null || break; sleep 0.5; done
            kill -0 "${stale_pid}" 2>/dev/null && kill -9 "${stale_pid}" 2>/dev/null || true
        fi
        # Fall through to the recover path below, which relaunches the
        # docker-driver gateway on the installed binary.
    fi

    # OpenShell 0.0.44: the gateway runs as a host "docker-driver" process
    # (openshell-gateway on :8080), NOT the old `openshell-cluster-*` container,
    # and `openshell gateway start` was removed (the gateway CLI is
    # registration-only now: add/remove/select/info/list). The 0.0.44 way to
    # (re)start the local gateway is the NemoClaw sandbox recovery, which
    # re-spawns the docker-driver gateway + the dashboard port-forward. This is
    # what keeps launch.sh self-healing across reboots, where the gateway host
    # process has no auto-restart (verified 2026-06-06: a reboot dropped the
    # gateway and the old `gateway start` path errored "unrecognized
    # subcommand 'start'").
    if ! command -v nemoclaw &>/dev/null; then
        fail "nemoclaw command not found; cannot (re)start the OpenShell gateway"
        fix "Install NemoClaw, or start the gateway manually: nemoclaw <sandbox> recover"
        return 1
    fi
    info "Recovering OpenShell gateway via 'nemoclaw ${sandbox_name} recover'..."
    # Exit code is not authoritative (recover runs a multi-step flow that can
    # emit non-fatal warnings); the status poll below is the real success signal.
    nemoclaw "${sandbox_name}" recover >/dev/null 2>&1 || true

    local i
    for i in $(seq 1 "${attempts}"); do
        if openshell status -g "${gateway_name}" &>/dev/null; then
            pass "OpenShell gateway '${gateway_name}' is running"
            return 0
        fi
        sleep 2
    done

    fail "OpenShell gateway '${gateway_name}' did not become reachable"
    fix "Diagnose: nemoclaw ${sandbox_name} doctor"
    fix "Gateway log: ~/.local/state/nemoclaw/openshell-docker-gateway/openshell-gateway.log"
    return 1
}

# ── openshell-thor fix verification ───────────────────────────────────────────
# These check that the five JetPack 7.1-specific fixes from openshell-thor
# are in place. NemoClaw will fail in non-obvious ways without them.

check_fix_iptable_raw() {
    if grep -q '^iptable_raw' /proc/modules 2>/dev/null; then
        pass "iptable_raw kernel module is loaded"
        return 0
    else
        fail "iptable_raw kernel module is not loaded"
        info "Required for OpenShell network isolation policy."
        info "CONFIG_IP_NF_RAW=n in the stock Thor kernel; must be built out-of-tree."
        fix "Run ./apply-host-fixes.sh to build and install it with a host backup."
        fix "Or apply the upstream fix manually: https://github.com/JetsonHacks/openshell-thor"
        return 1
    fi
}

check_fix_iptables_legacy() {
    local current
    current=$(update-alternatives --query iptables 2>/dev/null \
              | grep '^Value:' | awk '{print $2}')
    if echo "${current}" | grep -q 'legacy'; then
        pass "iptables is set to legacy backend (${current})"
        return 0
    else
        fail "iptables is not using the legacy backend (found: ${current:-unknown})"
        info "JetPack 7.1 defaults to iptables-nft, which breaks K3s service routing"
        info "and pod DNS resolution inside OpenShell sandboxes."
        fix "Run ./apply-host-fixes.sh to switch the Thor host with a backup first."
        fix "Run: sudo update-alternatives --set iptables /usr/sbin/iptables-legacy"
        fix "     sudo update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy"
        return 1
    fi
}

check_fix_br_netfilter() {
    if grep -q '^br_netfilter' /proc/modules 2>/dev/null; then
        pass "br_netfilter kernel module is loaded"
        return 0
    else
        fail "br_netfilter kernel module is not loaded"
        info "Required for K3s flannel CNI pod networking inside OpenShell."
        fix "Run ./apply-host-fixes.sh to apply the complete Thor network setup."
        fix "Run: sudo modprobe br_netfilter"
        fix "To persist across reboots:"
        fix "  echo 'br_netfilter' | sudo tee /etc/modules-load.d/br_netfilter.conf"
        return 1
    fi
}

check_fix_docker_ipv6() {
    local daemon_json="/etc/docker/daemon.json"

    if [[ ! -f "${daemon_json}" ]]; then
        fail "Docker daemon.json not found (${daemon_json})"
        info "Without this file, Docker may prefer IPv6 for registry pulls and time out."
        fix "Run ./apply-host-fixes.sh to set the Thor Docker defaults with a backup."
        fix "Create ${daemon_json} with content: {\"ipv6\": false}"
        fix "Then restart Docker: sudo systemctl restart docker"
        return 1
    fi

    if python3 -c "
import json, sys
with open('${daemon_json}') as f:
    d = json.load(f)
sys.exit(0 if d.get('ipv6') == False else 1)
" 2>/dev/null; then
        pass "Docker IPv6 is disabled in ${daemon_json}"
        return 0
    else
        fail "Docker IPv6 is not set to false in ${daemon_json}"
        info "The OpenShell gateway container has no IPv6 routing. When Docker prefers"
        info "IPv6 on dual-stack DNS results, registry pulls from docker.io and"
        info "registry.k8s.io will time out."
        fix "Run ./apply-host-fixes.sh to update Docker with a host backup."
        fix "Add or set \"ipv6\": false in ${daemon_json}"
        fix "Then restart Docker: sudo systemctl restart docker"
        return 1
    fi
}

check_fix_cgroupns() {
    local daemon_json="/etc/docker/daemon.json"

    if [[ ! -f "${daemon_json}" ]]; then
        fail "Docker daemon.json not found (${daemon_json})"
        info "Required to set default-cgroupns-mode=host for K3s inside Docker."
        fix "Run ./apply-host-fixes.sh to set the Thor Docker defaults with a backup."
        fix "Create ${daemon_json} with content: {\"default-cgroupns-mode\": \"host\"}"
        fix "Then restart Docker: sudo systemctl restart docker"
        return 1
    fi

    if python3 -c "
import json, sys
with open('${daemon_json}') as f:
    d = json.load(f)
sys.exit(0 if d.get('default-cgroupns-mode') == 'host' else 1)
" 2>/dev/null; then
        pass "Docker default-cgroupns-mode is set to host in ${daemon_json}"
        return 0
    else
        fail "Docker default-cgroupns-mode is not set to host in ${daemon_json}"
        info "JetPack 7.1 uses cgroup v2. Without this setting, K3s inside the"
        info "OpenShell gateway container will fail with:"
        info "  openat2 /sys/fs/cgroup/kubepods/pids.max: no such file or directory"
        fix "Run ./apply-host-fixes.sh to update Docker with a host backup."
        fix "Run the following to add the setting and restart Docker:"
        fix "  sudo python3 -c \""
        fix "import json"
        fix "path = '/etc/docker/daemon.json'"
        fix "d = json.load(open(path))"
        fix "d['default-cgroupns-mode'] = 'host'"
        fix "json.dump(d, open(path, 'w'), indent=4)"
        fix "  \""
        fix "  sudo systemctl restart docker"
        return 1
    fi
}

# ── Docker checks ──────────────────────────────────────────────────────────────

check_docker_installed() {
    if command -v docker &>/dev/null; then
        pass "docker found ($(docker --version 2>/dev/null))"
        return 0
    else
        fail "docker command not found"
        fix "Install Docker Engine: https://docs.docker.com/engine/install/ubuntu/"
        return 1
    fi
}

check_docker_running() {
    if docker info &>/dev/null; then
        pass "Docker daemon is running"
        return 0
    else
        fail "Docker daemon is not running"
        fix "Run: sudo systemctl start docker"
        fix "To start on boot: sudo systemctl enable docker"
        return 1
    fi
}

check_docker_nvidia_runtime() {
    if docker info 2>/dev/null | grep -qi 'nvidia'; then
        pass "NVIDIA container runtime is available in Docker"
        return 0
    else
        fail "NVIDIA container runtime not found in Docker"
        info "Required to give the vLLM container access to the Thor GPU."
        fix "Install nvidia-container-toolkit:"
        fix "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        return 1
    fi
}

# ── Node.js / nvm checks ───────────────────────────────────────────────────────

check_nvm_installed() {
    local nvm_dir="${NVM_DIR:-$HOME/.nvm}"
    if [[ -d "${nvm_dir}" ]] && [[ -s "${nvm_dir}/nvm.sh" ]]; then
        pass "nvm is installed (${nvm_dir})"
        return 0
    else
        warn "nvm is not installed"
        info "Current upstream NemoClaw can install nvm and Node.js automatically"
        info "during its own installer flow when Node.js is missing."
        fix "Optional: run ./install-node.sh to preinstall nvm and Node.js 22"
        fix "Or continue and let the upstream installer bootstrap them"
        return 2
    fi
}

check_node_version() {
    if ! command -v node &>/dev/null; then
        warn "node command not found"
        info "Current upstream NemoClaw can install Node.js 22 automatically"
        info "when no Node.js runtime is present."
        fix "Optional: run ./install-node.sh first for a prevalidated Node 22 setup"
        return 2
    fi

    local version major
    version=$(node --version)           # e.g. v22.4.0
    major=$(echo "${version}" | cut -d. -f1 | tr -d 'v')

    if ! [[ "${major}" =~ ^[0-9]+$ ]]; then
        fail "Could not determine the Node.js major version from ${version}"
        fix "Run ./install-node.sh to install a known-good Node.js 22 runtime"
        return 1
    fi

    if [[ "${major}" == "22" ]]; then
        pass "Node.js ${version} (preferred major version 22)"
        return 0
    fi

    if (( major < 20 )); then
        fail "Node.js ${version} is too old for current upstream NemoClaw"
        info "Do not use the Ubuntu apt default Node.js 18 for this workflow."
        fix "Install Node 22 via nvm:"
        fix "  nvm install 22 && nvm alias default 22 && nvm use 22"
        return 1
    fi

    warn "Node.js ${version} is supported upstream, but 22.x is the preferred target for this fork"
    fix "Optional: switch to Node 22 with nvm install 22 && nvm alias default 22 && nvm use 22"
    return 2
}

# ── Build tooling (needed for OpenClaw's npm install inside the sandbox) ───────

check_build_tools() {
    local missing=()
    command -v python3  &>/dev/null || missing+=("python3")
    command -v make     &>/dev/null || missing+=("make")
    command -v gcc      &>/dev/null || missing+=("gcc (build-essential)")

    if [[ ${#missing[@]} -eq 0 ]]; then
        pass "Build tools present (python3, make, gcc)"
        return 0
    else
        fail "Missing build tools: ${missing[*]}"
        info "OpenClaw's npm install may build the 'sharp' image library from source"
        info "on aarch64 if a prebuilt binary is unavailable."
        fix "Run: sudo apt-get install -y build-essential python3"
        return 1
    fi
}

# ── HuggingFace token ──────────────────────────────────────────────────────────

check_hf_token() {
    if [[ -z "${HF_TOKEN:-}" ]]; then
        fail "HF_TOKEN is not set"
        info "Required to pull the NVFP4 model weights from HuggingFace."
        fix "Export your token: export HF_TOKEN=hf_..."
        fix "Get a token at: https://huggingface.co/settings/tokens"
        fix "You must also accept the model license at:"
        fix "  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
        return 1
    fi

    pass "HF_TOKEN is set"
    return 0
}

check_hf_token_optional() {
    if [[ -z "${HF_TOKEN:-}" ]]; then
        warn "HF_TOKEN is not set"
        info "This is fine if your selected Qwen model is already downloaded or does"
        info "not require an authenticated Hugging Face pull."
        fix "Optional: export HF_TOKEN=hf_..."
        return 2
    fi

    pass "HF_TOKEN is set"
    return 0
}

# ── Disk space ─────────────────────────────────────────────────────────────────

check_disk_space() {
    # This keeps the install check intentionally conservative.
    # The exact local-serving footprint depends on which Qwen model you run,
    # whether the weights are already cached, and whether the Docker image is
    # already present on disk.
    local required_gb=10
    local docker_root
    docker_root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null \
                  || echo "/var/lib/docker")

    # Walk up to the nearest existing directory
    local check_dir="${docker_root}"
    while [[ ! -d "${check_dir}" ]]; do
        check_dir="$(dirname "${check_dir}")"
    done

    local available_gb
    available_gb=$(df -BG "${check_dir}" | awk 'NR==2 {gsub("G","",$4); print $4}')

    if [[ "${available_gb}" -ge "${required_gb}" ]]; then
        pass "Disk space: ${available_gb}GB available"
        return 0
    else
        fail "Disk space: only ${available_gb}GB available (need at least ${required_gb}GB)"
        info "Local vLLM with Qwen models typically needs tens of GB for the image,"
        info "model weights, and runtime caches."
        fix "Free up space on the filesystem containing ${check_dir}"
        return 1
    fi
}

# ── Network connectivity ───────────────────────────────────────────────────────

check_connectivity_huggingface() {
    if curl -fsSL --max-time 10 --head https://huggingface.co &>/dev/null; then
        pass "huggingface.co is reachable"
        return 0
    else
        fail "Cannot reach huggingface.co"
        info "Required to download NVFP4 model weights and the reasoning parser."
        fix "Check network connectivity and any proxy settings."
        return 1
    fi
}

check_connectivity_huggingface_optional() {
    if curl -fsSL --max-time 10 --head https://huggingface.co &>/dev/null; then
        pass "huggingface.co is reachable"
        return 0
    else
        warn "Cannot reach huggingface.co"
        info "This is fine if your local model weights are already cached."
        fix "Required later only if you still need to download model weights."
        return 2
    fi
}

check_connectivity_github() {
    if curl -fsSL --max-time 10 --head https://github.com &>/dev/null; then
        pass "github.com is reachable"
        return 0
    else
        fail "Cannot reach github.com"
        info "Required to clone and install NemoClaw (package is not on the npm registry)."
        fix "Check network connectivity and any proxy settings."
        return 1
    fi
}
