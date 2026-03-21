#!/usr/bin/env bash
# lib/host-state.sh — Host backup/restore helpers for Thor OpenShell fixes
#
# Source this file; do not execute it directly.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This file is meant to be sourced, not executed directly." >&2
    exit 1
fi

thor_state_dir() {
    local state_home
    state_home="${XDG_STATE_HOME:-$HOME/.local/state}"
    echo "${state_home}/nemoclaw-thor"
}

thor_host_backup_root() {
    echo "$(thor_state_dir)/host-backups"
}

thor_latest_host_backup_file() {
    echo "$(thor_state_dir)/latest-host-backup"
}

thor_default_openshell_thor_dir() {
    echo "${THOR_OPENSHELL_THOR_DIR:-${HOME}/OpenShell-Thor}"
}

thor_host_backup_dir_from_name() {
    local label="${1:-pre-openshell}"
    local ts
    ts="$(date -u +%Y%m%dT%H%M%SZ)"
    echo "$(thor_host_backup_root)/${ts}-${label}"
}

record_latest_host_backup_dir() {
    local backup_dir="$1"
    mkdir -p "$(thor_state_dir)"
    printf '%s\n' "${backup_dir}" > "$(thor_latest_host_backup_file)"
}

load_latest_host_backup_dir() {
    local latest_file
    latest_file="$(thor_latest_host_backup_file)"
    if [[ -f "${latest_file}" ]]; then
        cat "${latest_file}"
        return 0
    fi
    return 1
}

_capture_file_if_present() {
    local src="$1"
    local dest="$2"
    if [[ -f "${src}" ]]; then
        mkdir -p "$(dirname "${dest}")"
        sudo cat "${src}" > "${dest}"
        return 0
    fi
    return 1
}

create_host_backup() {
    local label="${1:-pre-openshell}"
    local backup_dir
    local daemon_json="/etc/docker/daemon.json"
    local modules_conf="/etc/modules-load.d/openshell-k3s.conf"
    local sysctl_conf="/etc/sysctl.d/99-openshell-k3s.conf"
    local pre_iptables_path=""
    local pre_ip6tables_path=""
    local pre_docker_active="0"
    local pre_docker_enabled="0"
    local pre_daemon_json_exists="0"
    local pre_modules_conf_exists="0"
    local pre_sysctl_conf_exists="0"
    local pre_iptable_raw_loaded="0"
    local pre_br_netfilter_loaded="0"
    local pre_iptable_raw_module_present="0"
    local pre_iptable_raw_module_path=""
    local pre_bridge_nf_call_iptables=""
    local pre_bridge_nf_call_ip6tables=""
    local pre_jetson_kernel_dir_exists="0"
    local pre_jetson_kernel_tarball_exists="0"

    backup_dir="$(thor_host_backup_dir_from_name "${label}")"
    mkdir -p "${backup_dir}"

    pre_iptables_path=$(update-alternatives --query iptables 2>/dev/null | awk '/^Value:/ {print $2}')
    pre_ip6tables_path=$(update-alternatives --query ip6tables 2>/dev/null | awk '/^Value:/ {print $2}')

    if systemctl is-active --quiet docker 2>/dev/null; then
        pre_docker_active="1"
    fi
    if systemctl is-enabled --quiet docker 2>/dev/null; then
        pre_docker_enabled="1"
    fi

    [[ -f "${daemon_json}" ]] && pre_daemon_json_exists="1"
    [[ -f "${modules_conf}" ]] && pre_modules_conf_exists="1"
    [[ -f "${sysctl_conf}" ]] && pre_sysctl_conf_exists="1"
    grep -q '^iptable_raw ' /proc/modules 2>/dev/null && pre_iptable_raw_loaded="1"
    grep -q '^br_netfilter ' /proc/modules 2>/dev/null && pre_br_netfilter_loaded="1"
    if modinfo iptable_raw &>/dev/null; then
        pre_iptable_raw_module_present="1"
        pre_iptable_raw_module_path=$(modinfo -n iptable_raw 2>/dev/null || true)
    fi
    pre_bridge_nf_call_iptables=$(sysctl -n net.bridge.bridge-nf-call-iptables 2>/dev/null || true)
    pre_bridge_nf_call_ip6tables=$(sysctl -n net.bridge.bridge-nf-call-ip6tables 2>/dev/null || true)
    [[ -d /usr/src/jetson-kernel ]] && pre_jetson_kernel_dir_exists="1"
    [[ -f /usr/src/jetson-kernel/public_sources.tbz2 ]] && pre_jetson_kernel_tarball_exists="1"

    _capture_file_if_present "${daemon_json}" "${backup_dir}/etc/docker/daemon.json" || true
    _capture_file_if_present "${modules_conf}" "${backup_dir}/etc/modules-load.d/openshell-k3s.conf" || true
    _capture_file_if_present "${sysctl_conf}" "${backup_dir}/etc/sysctl.d/99-openshell-k3s.conf" || true

    sudo iptables-save > "${backup_dir}/iptables.rules.v4"
    sudo ip6tables-save > "${backup_dir}/ip6tables.rules.v6"
    if command -v iptables-nft-save &>/dev/null; then
        sudo iptables-nft-save > "${backup_dir}/iptables-nft.rules.v4" 2>/dev/null || true
    fi
    if command -v ip6tables-nft-save &>/dev/null; then
        sudo ip6tables-nft-save > "${backup_dir}/ip6tables-nft.rules.v6" 2>/dev/null || true
    fi
    if command -v nft &>/dev/null; then
        sudo nft list ruleset > "${backup_dir}/nft.ruleset" 2>/dev/null || true
    fi

    if command -v docker &>/dev/null; then
        docker ps -a > "${backup_dir}/docker-ps-a.txt" 2>/dev/null || true
        docker network ls > "${backup_dir}/docker-network-ls.txt" 2>/dev/null || true
    fi

    {
        printf 'BACKUP_CREATED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'BACKUP_HOSTNAME=%q\n' "$(hostname)"
        printf 'PRE_IPTABLES_PATH=%q\n' "${pre_iptables_path}"
        printf 'PRE_IP6TABLES_PATH=%q\n' "${pre_ip6tables_path}"
        printf 'PRE_DOCKER_ACTIVE=%q\n' "${pre_docker_active}"
        printf 'PRE_DOCKER_ENABLED=%q\n' "${pre_docker_enabled}"
        printf 'PRE_DAEMON_JSON_EXISTS=%q\n' "${pre_daemon_json_exists}"
        printf 'PRE_MODULES_CONF_EXISTS=%q\n' "${pre_modules_conf_exists}"
        printf 'PRE_SYSCTL_CONF_EXISTS=%q\n' "${pre_sysctl_conf_exists}"
        printf 'PRE_IPTABLE_RAW_LOADED=%q\n' "${pre_iptable_raw_loaded}"
        printf 'PRE_BR_NETFILTER_LOADED=%q\n' "${pre_br_netfilter_loaded}"
        printf 'PRE_IPTABLE_RAW_MODULE_PRESENT=%q\n' "${pre_iptable_raw_module_present}"
        printf 'PRE_IPTABLE_RAW_MODULE_PATH=%q\n' "${pre_iptable_raw_module_path}"
        printf 'PRE_BRIDGE_NF_CALL_IPTABLES=%q\n' "${pre_bridge_nf_call_iptables}"
        printf 'PRE_BRIDGE_NF_CALL_IP6TABLES=%q\n' "${pre_bridge_nf_call_ip6tables}"
        printf 'PRE_JETSON_KERNEL_DIR_EXISTS=%q\n' "${pre_jetson_kernel_dir_exists}"
        printf 'PRE_JETSON_KERNEL_TARBALL_EXISTS=%q\n' "${pre_jetson_kernel_tarball_exists}"
    } > "${backup_dir}/metadata.env"

    record_latest_host_backup_dir "${backup_dir}"
    THOR_HOST_BACKUP_DIR_CREATED="${backup_dir}"
}

resolve_host_backup_dir() {
    local requested="${1:-}"
    if [[ -n "${requested}" ]]; then
        echo "${requested}"
        return 0
    fi
    load_latest_host_backup_dir
}

restore_host_backup_dir() {
    local backup_dir="$1"
    local restore_firewall="${2:-1}"
    local metadata_file="${backup_dir}/metadata.env"
    local daemon_json="/etc/docker/daemon.json"
    local modules_conf="/etc/modules-load.d/openshell-k3s.conf"
    local sysctl_conf="/etc/sysctl.d/99-openshell-k3s.conf"
    local docker_available="0"

    if [[ ! -f "${metadata_file}" ]]; then
        echo "Backup metadata not found: ${metadata_file}" >&2
        return 1
    fi

    # shellcheck source=/dev/null
    source "${metadata_file}"

    if command -v openshell &>/dev/null; then
        openshell gateway stop 2>/dev/null || true
    fi

    if command -v docker &>/dev/null; then
        docker_available="1"
        docker ps -aq --filter name=openshell 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1 || true
    fi

    if systemctl list-unit-files docker.service >/dev/null 2>&1; then
        sudo systemctl stop docker >/dev/null 2>&1 || true
    fi

    if [[ -n "${PRE_IPTABLES_PATH:-}" && -x "${PRE_IPTABLES_PATH}" ]]; then
        sudo update-alternatives --set iptables "${PRE_IPTABLES_PATH}"
    fi
    if [[ -n "${PRE_IP6TABLES_PATH:-}" && -x "${PRE_IP6TABLES_PATH}" ]]; then
        sudo update-alternatives --set ip6tables "${PRE_IP6TABLES_PATH}"
    fi

    if [[ "${PRE_MODULES_CONF_EXISTS:-0}" == "1" ]]; then
        sudo install -D -m 644 "${backup_dir}/etc/modules-load.d/openshell-k3s.conf" "${modules_conf}"
    else
        sudo rm -f "${modules_conf}"
    fi

    if [[ "${PRE_SYSCTL_CONF_EXISTS:-0}" == "1" ]]; then
        sudo install -D -m 644 "${backup_dir}/etc/sysctl.d/99-openshell-k3s.conf" "${sysctl_conf}"
    else
        sudo rm -f "${sysctl_conf}"
    fi

    if [[ "${PRE_DAEMON_JSON_EXISTS:-0}" == "1" ]]; then
        sudo install -D -m 644 "${backup_dir}/etc/docker/daemon.json" "${daemon_json}"
    else
        sudo rm -f "${daemon_json}"
    fi

    if [[ -n "${PRE_BRIDGE_NF_CALL_IPTABLES:-}" ]]; then
        sudo sysctl -w "net.bridge.bridge-nf-call-iptables=${PRE_BRIDGE_NF_CALL_IPTABLES}" >/dev/null
    fi
    if [[ -n "${PRE_BRIDGE_NF_CALL_IP6TABLES:-}" ]]; then
        sudo sysctl -w "net.bridge.bridge-nf-call-ip6tables=${PRE_BRIDGE_NF_CALL_IP6TABLES}" >/dev/null
    fi

    if [[ "${restore_firewall}" == "1" ]]; then
        sudo iptables-restore < "${backup_dir}/iptables.rules.v4"
        sudo ip6tables-restore < "${backup_dir}/ip6tables.rules.v6"
    fi

    if systemctl list-unit-files docker.service >/dev/null 2>&1; then
        if [[ "${PRE_DOCKER_ACTIVE:-0}" == "1" || "${PRE_DOCKER_ENABLED:-0}" == "1" ]]; then
            sudo systemctl start docker
        fi
    fi

    if [[ "${PRE_BR_NETFILTER_LOADED:-0}" == "1" ]]; then
        sudo modprobe br_netfilter >/dev/null 2>&1 || true
    else
        sudo modprobe -r br_netfilter >/dev/null 2>&1 || true
    fi

    if [[ "${PRE_IPTABLE_RAW_LOADED:-0}" == "1" ]]; then
        sudo modprobe iptable_raw >/dev/null 2>&1 || true
    else
        sudo modprobe -r iptable_raw >/dev/null 2>&1 || true
    fi

    if [[ "${PRE_IPTABLE_RAW_MODULE_PRESENT:-0}" != "1" ]]; then
        local built_module
        built_module=$(modinfo -n iptable_raw 2>/dev/null || true)
        if [[ -n "${built_module}" && -f "${built_module}" ]]; then
            sudo rm -f "${built_module}"
            sudo depmod -a
        fi
    fi

    if [[ "${PRE_JETSON_KERNEL_DIR_EXISTS:-0}" != "1" && -d /usr/src/jetson-kernel ]]; then
        sudo rm -rf /usr/src/jetson-kernel
    elif [[ "${PRE_JETSON_KERNEL_TARBALL_EXISTS:-0}" != "1" && -f /usr/src/jetson-kernel/public_sources.tbz2 ]]; then
        sudo rm -f /usr/src/jetson-kernel/public_sources.tbz2
    fi

    if [[ "${docker_available}" == "1" ]]; then
        docker network ls > "${backup_dir}/docker-network-ls.after-restore.txt" 2>/dev/null || true
    fi
}

ensure_openshell_thor_repo() {
    local repo_dir="${1:-$(thor_default_openshell_thor_dir)}"
    local repo_url="${2:-https://github.com/jetsonhacks/OpenShell-Thor.git}"

    if [[ -d "${repo_dir}/.git" ]]; then
        return 0
    fi

    mkdir -p "$(dirname "${repo_dir}")"
    git clone "${repo_url}" "${repo_dir}"
}

clear_stale_nft_compat_firewall() {
    local changed=0
    local table=""

    if command -v iptables-nft-save &>/dev/null; then
        if sudo iptables-nft-save 2>/dev/null | grep -Eq '(^-A DOCKER|^-A FORWARD|^-A OUTPUT|^-A PREROUTING|^-A POSTROUTING|^\*nat|^\*filter)'; then
            changed=1
        fi
    fi
    if command -v ip6tables-nft-save &>/dev/null; then
        if sudo ip6tables-nft-save 2>/dev/null | grep -Eq '(^-A DOCKER|^-A FORWARD|^-A OUTPUT|^-A PREROUTING|^-A POSTROUTING|^\*nat|^\*filter)'; then
            changed=1
        fi
    fi

    if [[ "${changed}" == "0" ]]; then
        return 1
    fi

    if command -v iptables-nft &>/dev/null; then
        sudo iptables-nft -P INPUT ACCEPT >/dev/null 2>&1 || true
        sudo iptables-nft -P FORWARD ACCEPT >/dev/null 2>&1 || true
        sudo iptables-nft -P OUTPUT ACCEPT >/dev/null 2>&1 || true
        for table in raw mangle nat filter security; do
            sudo iptables-nft -t "${table}" -F >/dev/null 2>&1 || true
            sudo iptables-nft -t "${table}" -X >/dev/null 2>&1 || true
        done
    fi

    if command -v ip6tables-nft &>/dev/null; then
        sudo ip6tables-nft -P INPUT ACCEPT >/dev/null 2>&1 || true
        sudo ip6tables-nft -P FORWARD ACCEPT >/dev/null 2>&1 || true
        sudo ip6tables-nft -P OUTPUT ACCEPT >/dev/null 2>&1 || true
        for table in raw mangle nat filter security; do
            sudo ip6tables-nft -t "${table}" -F >/dev/null 2>&1 || true
            sudo ip6tables-nft -t "${table}" -X >/dev/null 2>&1 || true
        done
    fi

    return 0
}
