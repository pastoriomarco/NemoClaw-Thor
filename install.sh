#!/usr/bin/env bash
# install.sh — Install NemoClaw on Jetson AGX Thor and point it at a local Qwen server
#
# Usage:
#   ./install.sh [model-profile] [--policy-profile <profile>] [--apply-host-fixes]
#
# Supported model profiles:
#   qwen3.5-122b-a10b-nvfp4-resharded
#   qwen3.5-27b-fp8
#   qwen3.5-35b-a3b-fp8
#   qwen3.5-35b-a3b-nvfp4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/checks.sh"
source "${SCRIPT_DIR}/lib/config.sh"
source "${SCRIPT_DIR}/lib/policy.sh"

MODEL_PROFILE_ARG=""
POLICY_PROFILE_ARG=""
APPLY_HOST_FIXES="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: ./install.sh [model-profile] [--policy-profile <profile>] [--apply-host-fixes]"
            echo ""
            print_supported_model_profiles
            echo ""
            print_supported_policy_profiles
            exit 0
            ;;
        --policy-profile)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --policy-profile" >&2
                exit 1
            fi
            POLICY_PROFILE_ARG="$2"
            shift 2
            ;;
        --apply-host-fixes)
            APPLY_HOST_FIXES="1"
            shift
            ;;
        *)
            if [[ -n "${MODEL_PROFILE_ARG}" ]]; then
                echo "Unexpected argument: $1" >&2
                exit 1
            fi
            MODEL_PROFILE_ARG="$1"
            shift
            ;;
    esac
done

load_thor_runtime_config "${MODEL_PROFILE_ARG}"
resolve_policy_profile "${POLICY_PROFILE_ARG:-${THOR_POLICY_PROFILE:-}}"

NEMOCLAW_DIR="${HOME}/NemoClaw"
NEMOCLAW_REPO="https://github.com/NVIDIA/NemoClaw.git"
NVM_DIR="${NVM_DIR:-$HOME/.nvm}"

activate_preferred_node_runtime() {
    if [[ -s "${NVM_DIR}/nvm.sh" ]]; then
        # shellcheck source=/dev/null
        source "${NVM_DIR}/nvm.sh"
        if nvm use 22 &>/dev/null; then
            pass "Node.js $(node --version) active via nvm"
            return 0
        fi
        warn "nvm is installed, but Node.js 22 is not active in this shell"
        info "The upstream NemoClaw installer can install Node.js 22 if needed."
        return 0
    fi

    if command -v node &>/dev/null; then
        pass "Node.js $(node --version) already available"
    else
        warn "Node.js is not yet available in this shell"
        info "The upstream NemoClaw installer can bootstrap nvm and Node.js 22."
    fi
}

refresh_nemoclaw_path() {
    local npm_prefix=""
    local npm_bin=""

    export PATH="${HOME}/.local/bin:${PATH}"

    if [[ -s "${NVM_DIR}/nvm.sh" ]]; then
        # shellcheck source=/dev/null
        source "${NVM_DIR}/nvm.sh"
        nvm use 22 &>/dev/null || true
    fi

    if command -v npm &>/dev/null; then
        npm_prefix=$(npm config get prefix 2>/dev/null || echo "")
        npm_bin="${npm_prefix}/bin"
        if [[ -n "${npm_prefix}" && -d "${npm_bin}" ]]; then
            export PATH="${npm_bin}:${PATH}"
        fi
    fi
}

echo ""
echo -e "${BOLD}NemoClaw-Thor Installer${NC}"
echo "Repo: ${SCRIPT_DIR}"
echo ""
echo "This script installs NemoClaw on Jetson AGX Thor, applies a hardened"
echo "sandbox policy baseline, and configures NemoClaw to use your local"
echo "Qwen vLLM endpoint after onboarding completes."
echo ""
print_thor_runtime_config
echo ""
echo "Estimated time: 10-20 minutes depending on network speed."
echo ""

if [[ "${APPLY_HOST_FIXES}" == "1" ]]; then
    header "Step 0: Backing up host state and applying Thor fixes"
    echo ""
    "${SCRIPT_DIR}/apply-host-fixes.sh"
fi

header "Step 1: Prerequisite checks"
echo ""

CHECKS_FAILED=0

run_check() {
    local ret
    "$@" && ret=0 || ret=$?
    if [[ "${ret}" -ne 0 && "${ret}" -ne 2 ]]; then
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
    fi
    return 0
}

run_check check_openshell_installed
run_check check_fix_iptable_raw
run_check check_fix_iptables_legacy
run_check check_fix_br_netfilter
run_check check_fix_docker_ipv6
run_check check_fix_cgroupns
run_check check_docker_installed
run_check check_docker_running
run_check check_docker_nvidia_runtime
run_check check_nvm_installed
run_check check_node_version
run_check check_build_tools
run_check check_disk_space
run_check check_connectivity_github

if [[ "${CHECKS_FAILED}" -gt 0 ]]; then
    echo ""
    echo -e "${RED}${BOLD}  ${CHECKS_FAILED} prerequisite check(s) failed.${NC}"
    echo ""
    echo "  Run ./check-prerequisites.sh ${THOR_MODEL_PROFILE} for details."
    echo "  Address all failures before running install.sh again."
    echo ""
    exit 1
fi

echo ""
echo -e "${GREEN}${BOLD}  All blocking prerequisite checks passed.${NC}"

header "Step 2: Docker cgroupns configuration"
echo ""

DAEMON_JSON="/etc/docker/daemon.json"

if python3 -c "
import json, sys
with open('${DAEMON_JSON}') as f:
    d = json.load(f)
sys.exit(0 if d.get('default-cgroupns-mode') == 'host' else 1)
" 2>/dev/null; then
    pass "Docker cgroupns already configured — skipping"
else
    info "Applying cgroupns fix to ${DAEMON_JSON}..."
    info "This requires sudo."
    echo ""
    sudo python3 - << 'PYEOF'
import json

path = "/etc/docker/daemon.json"
with open(path) as f:
    data = json.load(f)

data["default-cgroupns-mode"] = "host"

with open(path, "w") as f:
    json.dump(data, f, indent=4)
    f.write("\n")
PYEOF

    info "Restarting Docker..."
    sudo systemctl restart docker

    retries=5
    i=0
    while [[ "${i}" -lt "${retries}" ]]; do
        if docker info &>/dev/null; then
            break
        fi
        sleep 2
        i=$((i + 1))
    done

    if ! docker info &>/dev/null; then
        fail "Docker did not restart cleanly after cgroupns fix"
        fix "Check: sudo systemctl status docker"
        fix "Check: sudo journalctl -u docker --no-pager | tail -20"
        exit 1
    fi

    pass "cgroupns fix applied and Docker restarted successfully"
fi

header "Step 3: Checking for existing NemoClaw installation"
echo ""

if [[ -d "${NEMOCLAW_DIR}" ]]; then
    fail "${NEMOCLAW_DIR} already exists"
    info "Use ./configure-local-provider.sh ${THOR_MODEL_PROFILE} if NemoClaw is"
    info "already installed and you only need to switch the local model profile."
    fix "Run ./apply-policy-profile.sh ${THOR_POLICY_PROFILE} to update the static sandbox policy."
    fix "Run ./uninstall.sh to remove the existing install if you want a clean reinstall."
    exit 1
fi

pass "${NEMOCLAW_DIR} does not exist — ready to install"

header "Step 4: Preparing Node.js runtime"
echo ""
activate_preferred_node_runtime

header "Step 5: Cloning NemoClaw"
echo ""
info "Cloning ${NEMOCLAW_REPO} to ${NEMOCLAW_DIR}..."
echo ""

git clone "${NEMOCLAW_REPO}" "${NEMOCLAW_DIR}"
pass "NemoClaw cloned to ${NEMOCLAW_DIR}"

header "Step 6: Installing hardened sandbox policy"
echo ""
install_static_policy_profile "${NEMOCLAW_DIR}" "${THOR_POLICY_PROFILE}"
save_thor_runtime_config
echo ""

header "Step 7: Running NemoClaw installer"
echo ""
echo "  The upstream NemoClaw installer and onboarding wizard will now run."
echo "  Depending on the upstream version and your environment, you may be prompted for:"
echo "    • Inference endpoint"
echo "    • NVIDIA API key"
echo "    • Primary model"
echo ""
echo "  For this fork, do not enable external chat channels."
echo "  The sandbox policy baseline has already been replaced with:"
echo "    ${THOR_POLICY_PROFILE}"
echo "  This script will reapply the Thor local-provider configuration afterward."
echo ""
echo "  Press Enter to continue..."
read -r

cd "${NEMOCLAW_DIR}"
./install.sh

SANDBOX_NAME=$(openshell sandbox list 2>/dev/null \
    | sed 's/\x1b\[[0-9;]*m//g' \
    | awk 'NR>1 && $1 != "" {print $1; exit}' || echo "")

if [[ -n "${SANDBOX_NAME}" ]]; then
    openshell forward stop 18789 "${SANDBOX_NAME}" 2>/dev/null || true
    openshell forward start 18789 "${SANDBOX_NAME}" --background
fi

disown -a 2>/dev/null || true

header "Step 8: Verifying nemoclaw installation"
echo ""
refresh_nemoclaw_path

if ! command -v nemoclaw &>/dev/null; then
    fail "nemoclaw command not found after installation"
    info "Upstream NemoClaw may have installed the command under ~/.local/bin or"
    info "the active npm prefix without those paths being visible in this shell."
    fix "Open a new shell or source ~/.bashrc, then run: command -v nemoclaw"
    fix "If needed, inspect: ${HOME}/.local/bin and \$(npm config get prefix 2>/dev/null)/bin"
    exit 1
fi

pass "nemoclaw found at $(command -v nemoclaw)"

header "Step 9: Configuring local inference"
echo ""
"${SCRIPT_DIR}/configure-local-provider.sh" "${THOR_MODEL_PROFILE}"

SANDBOX_NAME="${SANDBOX_NAME:-<sandbox-name>}"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}${BOLD}  NemoClaw installation complete.${NC}"
echo ""
echo "  Sandbox:   ${SANDBOX_NAME}"
echo "  Provider:  ${THOR_LOCAL_PROVIDER_NAME}"
echo "  Model:     ${THOR_MODEL_ID}"
echo ""
echo -e "${BOLD}  Next steps:${NC}"
echo ""
echo "  1. Start your selected local model server:"
echo "       ./start-model.sh ${THOR_MODEL_PROFILE}"
echo "  2. Run: ./status.sh ${THOR_MODEL_PROFILE}"
echo "  3. If the agent later needs temporary internet, either:"
echo "       ./apply-policy-additions.sh research-lite"
echo "     or use: openshell term"
echo "  4. Connect to the sandbox and run a one-message smoke test."
echo ""
