#!/usr/bin/env bash
# uninstall.sh — Remove NemoClaw from Jetson AGX Thor
#
# Removes:
#   - The OpenShell sandbox created by NemoClaw
#   - The vllm-local inference provider
#   - The nvidia-nim inference provider
#   - The nemoclaw CLI
#   - The ~/NemoClaw directory
#   - Optionally: ~/.nemoclaw (contains NVIDIA API key and sandbox metadata)
#   - The NemoClaw-Thor runtime config file
#
# Does NOT remove:
#   - OpenShell itself
#   - The openshell-thor kernel/networking fixes
#     (use ./restore-host-state.sh if you want to revert the saved host snapshot)
#   - Docker or the NVIDIA container runtime
#   - nvm or Node.js (use ./uninstall-node.sh for that)
#   - The vLLM container image or local model weights
#
# Usage:
#   ./uninstall.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/checks.sh"
source "${SCRIPT_DIR}/lib/config.sh"

NEMOCLAW_DIR="${HOME}/NemoClaw"
NEMOCLAW_CONFIG_DIR="${HOME}/.nemoclaw"
NEMOCLAW_THOR_CONFIG_FILE="$(thor_config_file)"
NVM_DIR="${NVM_DIR:-$HOME/.nvm}"

# ── Header ─────────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}NemoClaw-Thor Uninstaller${NC}"
echo -e "JetsonHacks — https://github.com/JetsonHacks/NemoClaw-Thor"
echo ""

# ── Ensure nvm/node are active ─────────────────────────────────────────────────

if [[ -s "${NVM_DIR}/nvm.sh" ]]; then
    # shellcheck source=/dev/null
    source "${NVM_DIR}/nvm.sh"
    nvm use 22 &>/dev/null || true
fi

# ── Discover what's installed ──────────────────────────────────────────────────

sandboxes=()
providers=()
nemoclaw_bin=""
nemoclaw_dir_exists=false
nemoclaw_config_exists=false
nemoclaw_thor_config_exists=false

# Sandboxes
if command -v openshell &>/dev/null; then
    while IFS= read -r name; do
        [[ -n "${name}" ]] && sandboxes+=("${name}")
    done < <(openshell sandbox list 2>/dev/null \
        | sed 's/\x1b\[[0-9;]*m//g' \
        | awk 'NR>1 && $1 != "" {print $1}' || true)

    # Providers
    while IFS= read -r name; do
        [[ -n "${name}" ]] && providers+=("${name}")
    done < <(openshell provider list 2>/dev/null \
        | sed 's/\x1b\[[0-9;]*m//g' \
        | awk 'NR>1 && $1 != "" {print $1}' || true)
fi

# nemoclaw binary — check PATH first, then search nvm prefix directly
if command -v nemoclaw &>/dev/null; then
    nemoclaw_bin=$(command -v nemoclaw)
elif [[ -d "${NVM_DIR}" ]]; then
    # nvm is installed but PATH may not include the active node bin
    nemoclaw_bin=$(find "${NVM_DIR}/versions" -name "nemoclaw" 2>/dev/null | head -1)
fi

# Directories
[[ -d "${NEMOCLAW_DIR}" ]] && nemoclaw_dir_exists=true
[[ -d "${NEMOCLAW_CONFIG_DIR}" ]] && nemoclaw_config_exists=true
[[ -f "${NEMOCLAW_THOR_CONFIG_FILE}" ]] && nemoclaw_thor_config_exists=true

# ── Check there is anything to remove ─────────────────────────────────────────

if [[ ${#sandboxes[@]} -eq 0 ]] && \
   [[ ${#providers[@]} -eq 0 ]] && \
   [[ -z "${nemoclaw_bin}" ]] && \
   [[ "${nemoclaw_dir_exists}" == false ]] && \
   [[ "${nemoclaw_config_exists}" == false ]] && \
   [[ "${nemoclaw_thor_config_exists}" == false ]]; then
    echo "Nothing to remove — NemoClaw does not appear to be installed."
    echo ""
    exit 0
fi

# ── Show current sandboxes ─────────────────────────────────────────────────────

echo -e "${BOLD}Current OpenShell sandboxes:${NC}"
echo ""
if command -v openshell &>/dev/null; then
    openshell sandbox list 2>/dev/null || echo "  (none)"
else
    echo "  (openshell not found)"
fi
echo ""

# ── Show what will be removed ──────────────────────────────────────────────────

echo -e "${BOLD}The following will be removed:${NC}"
echo ""

if [[ ${#sandboxes[@]} -gt 0 ]]; then
    echo "  • OpenShell sandboxes:"
    for s in "${sandboxes[@]}"; do
        echo "      ${s}"
    done
    echo ""
fi

if [[ ${#providers[@]} -gt 0 ]]; then
    echo "  • OpenShell inference providers:"
    for p in "${providers[@]}"; do
        echo "      ${p}"
    done
    echo ""
fi

if [[ -n "${nemoclaw_bin}" ]]; then
    echo "  • nemoclaw CLI: ${nemoclaw_bin}"
    echo ""
fi

if [[ "${nemoclaw_dir_exists}" == true ]]; then
    echo "  • NemoClaw directory: ${NEMOCLAW_DIR}"
    echo ""
fi

if [[ "${nemoclaw_thor_config_exists}" == true ]]; then
    echo "  • NemoClaw-Thor config: ${NEMOCLAW_THOR_CONFIG_FILE}"
    echo ""
fi

echo "  This does NOT remove OpenShell, nvm, Node.js, Docker,"
echo "  the vLLM container image, or any local model weights."
echo "  Use ./restore-host-state.sh separately if you want to revert"
echo "  the saved pre-OpenShell host snapshot."
echo ""

# ── Ask about ~/.nemoclaw ──────────────────────────────────────────────────────

remove_config=false
if [[ "${nemoclaw_config_exists}" == true ]]; then
    echo -e "${BOLD}NemoClaw configuration directory:${NC}"
    echo ""
    echo "  ${NEMOCLAW_CONFIG_DIR} contains:"
    ls "${NEMOCLAW_CONFIG_DIR}" 2>/dev/null | sed 's/^/    /'
    echo ""
    echo "  This includes your NVIDIA API key and sandbox metadata."
    echo "  Keeping it means a reinstall can reuse your credentials."
    echo ""
    echo -e "${YELLOW}${BOLD}  Remove ~/.nemoclaw? [y/N]${NC} " && read -r config_response
    echo ""
    if [[ "${config_response}" =~ ^[Yy]$ ]]; then
        remove_config=true
        echo "  Will remove ${NEMOCLAW_CONFIG_DIR}"
    else
        echo "  Will keep ${NEMOCLAW_CONFIG_DIR}"
    fi
    echo ""
fi

# ── Confirm ────────────────────────────────────────────────────────────────────

echo -e "${YELLOW}${BOLD}  Proceed with removal? [y/N]${NC} " && read -r response
echo ""

if [[ ! "${response}" =~ ^[Yy]$ ]]; then
    echo "Cancelled — nothing was changed."
    echo ""
    exit 0
fi

# ── Remove sandboxes ───────────────────────────────────────────────────────────

if [[ ${#sandboxes[@]} -gt 0 ]]; then
    header "Removing sandboxes"
    echo ""
    for sandbox in "${sandboxes[@]}"; do
        info "Removing sandbox: ${sandbox}..."
        if openshell sandbox delete "${sandbox}" 2>/dev/null; then
            pass "Sandbox ${sandbox} removed"
        else
            warn "Could not remove sandbox ${sandbox} — may need manual removal"
            fix "Run: openshell sandbox delete ${sandbox}"
        fi
    done
    echo ""
fi

# ── Clear inference route ──────────────────────────────────────────────────────

header "Clearing inference route"
echo ""

current_provider=$(openshell inference get 2>/dev/null \
    | grep 'Provider:' | awk '{print $2}' || echo "")

if [[ -n "${current_provider}" ]]; then
    info "Current inference provider: ${current_provider}"
    # No direct 'unset' command — we note it will be cleared when provider is deleted
    info "Inference route will be cleared when providers are removed"
else
    info "No active inference route"
fi
echo ""

# ── Remove providers ───────────────────────────────────────────────────────────

if [[ ${#providers[@]} -gt 0 ]]; then
    header "Removing inference providers"
    echo ""
    for provider in "${providers[@]}"; do
        info "Removing provider: ${provider}..."
        if openshell provider delete "${provider}" 2>/dev/null; then
            pass "Provider ${provider} removed"
        else
            warn "Could not remove provider ${provider} — may need manual removal"
            fix "Run: openshell provider delete ${provider}"
        fi
    done
    echo ""
fi

# ── Remove nemoclaw CLI ────────────────────────────────────────────────────────

if [[ -n "${nemoclaw_bin}" ]]; then
    header "Removing nemoclaw CLI"
    echo ""
    info "Removing nemoclaw from npm..."

    if [[ -d "${NEMOCLAW_DIR}" ]]; then
        # Uninstall from the source directory
        cd "${NEMOCLAW_DIR}"
        npm uninstall -g nemoclaw 2>/dev/null && \
            pass "nemoclaw CLI removed" || {
            # Fall back to removing the binary directly
            rm -f "${nemoclaw_bin}" && \
                pass "nemoclaw binary removed directly" || \
                warn "Could not remove nemoclaw — may need manual removal"
        }
        cd - &>/dev/null
    else
        npm uninstall -g nemoclaw 2>/dev/null && \
            pass "nemoclaw CLI removed" || {
            rm -f "${nemoclaw_bin}" && \
                pass "nemoclaw binary removed directly" || \
                warn "Could not remove nemoclaw — may need manual removal"
        }
    fi
    echo ""
fi

# ── Remove ~/NemoClaw directory ────────────────────────────────────────────────

if [[ "${nemoclaw_dir_exists}" == true ]]; then
    header "Removing NemoClaw directory"
    echo ""
    info "Removing ${NEMOCLAW_DIR}..."
    rm -rf "${NEMOCLAW_DIR}"
    if [[ ! -d "${NEMOCLAW_DIR}" ]]; then
        pass "${NEMOCLAW_DIR} removed"
    else
        fail "Could not remove ${NEMOCLAW_DIR}"
        fix "Try: rm -rf ${NEMOCLAW_DIR}"
    fi
    echo ""
fi

# ── Remove ~/.nemoclaw ────────────────────────────────────────────────────────

if [[ "${remove_config}" == true ]]; then
    header "Removing NemoClaw configuration"
    echo ""
    info "Removing ${NEMOCLAW_CONFIG_DIR}..."
    rm -rf "${NEMOCLAW_CONFIG_DIR}"
    if [[ ! -d "${NEMOCLAW_CONFIG_DIR}" ]]; then
        pass "${NEMOCLAW_CONFIG_DIR} removed"
    else
        fail "Could not remove ${NEMOCLAW_CONFIG_DIR}"
        fix "Try: rm -rf ${NEMOCLAW_CONFIG_DIR}"
    fi
    echo ""
fi

if [[ "${nemoclaw_thor_config_exists}" == true ]]; then
    header "Removing NemoClaw-Thor runtime config"
    echo ""
    info "Removing ${NEMOCLAW_THOR_CONFIG_FILE}..."
    rm -f "${NEMOCLAW_THOR_CONFIG_FILE}"
    if [[ ! -f "${NEMOCLAW_THOR_CONFIG_FILE}" ]]; then
        pass "${NEMOCLAW_THOR_CONFIG_FILE} removed"
    else
        fail "Could not remove ${NEMOCLAW_THOR_CONFIG_FILE}"
        fix "Try: rm -f ${NEMOCLAW_THOR_CONFIG_FILE}"
    fi
    echo ""
fi

# ── Done ───────────────────────────────────────────────────────────────────────

echo "══════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}${BOLD}  NemoClaw has been removed.${NC}"
echo ""

if [[ "${nemoclaw_config_exists}" == true ]] && \
   [[ "${remove_config}" == false ]]; then
    echo "  Your NVIDIA API key and configuration have been kept at:"
    echo "  ${NEMOCLAW_CONFIG_DIR}"
    echo "  A reinstall will reuse these credentials."
    echo ""
fi

echo "  To reinstall, run ./install.sh"
echo ""
