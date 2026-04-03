#!/usr/bin/env bash
# lib/egress-firewall.sh — iptables-based egress firewall for sandbox pods
#
# Enforces network isolation at the k3s node level via FORWARD chain rules.
# This is necessary because:
#   1. Kernel 6.8 doesn't support Landlock network filtering (needs 6.10+)
#   2. K3s default Flannel CNI + kube-router stub don't enforce NetworkPolicy
#   3. OpenShell policy enforcement is application-level, not kernel-level
#
# Allowed traffic from sandbox pod:
#   - Loopback (127.0.0.1) — implicit, not routed through FORWARD
#   - DNS to CoreDNS
#   - OpenShell gateway (pod and service IPs)
#   - Host vLLM via Docker bridge (172.17.0.1:8000)
#
# Everything else is DROPped.
#
# Source this file; do not execute it directly.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This file is meant to be sourced, not executed directly." >&2
    exit 1
fi

EGRESS_FW_CHAIN="SANDBOX-EGRESS"

_egress_fw_resolve_sandbox_ip() {
    local cluster_container="$1"
    local sandbox_name="$2"

    docker exec "${cluster_container}" \
        kubectl -n openshell get pod "${sandbox_name}" \
        -o jsonpath='{.status.podIP}' 2>/dev/null
}

_egress_fw_resolve_dns_ip() {
    local cluster_container="$1"

    docker exec "${cluster_container}" \
        kubectl -n kube-system get svc kube-dns \
        -o jsonpath='{.spec.clusterIP}' 2>/dev/null
}

_egress_fw_resolve_gateway_ips() {
    local cluster_container="$1"

    # Pod IP
    docker exec "${cluster_container}" \
        kubectl -n openshell get pod openshell-0 \
        -o jsonpath='{.status.podIP}' 2>/dev/null

    echo ""

    # Service ClusterIP
    docker exec "${cluster_container}" \
        kubectl -n openshell get svc openshell \
        -o jsonpath='{.spec.clusterIP}' 2>/dev/null
}

_egress_fw_resolve_host_bridge_ip() {
    local cluster_container="$1"
    local sandbox_name="$2"

    docker exec "${cluster_container}" \
        kubectl -n openshell exec -i "${sandbox_name}" -- \
        getent hosts host.openshell.internal 2>/dev/null \
        | awk '{print $1; exit}'
}

enforce_sandbox_egress_firewall() {
    local cluster_container="$1"
    local sandbox_name="${2:-thor-assistant}"
    local host_vllm_port="${3:-8000}"

    if [[ -z "${cluster_container}" ]]; then
        echo "Usage: enforce_sandbox_egress_firewall <cluster-container> [sandbox-name] [vllm-port]" >&2
        return 1
    fi

    # Resolve dynamic IPs
    local pod_ip dns_ip gw_pod_ip gw_svc_ip host_bridge_ip

    pod_ip=$(_egress_fw_resolve_sandbox_ip "${cluster_container}" "${sandbox_name}")
    if [[ -z "${pod_ip}" ]]; then
        echo "ERROR: Could not resolve pod IP for ${sandbox_name}" >&2
        return 1
    fi

    dns_ip=$(_egress_fw_resolve_dns_ip "${cluster_container}")
    if [[ -z "${dns_ip}" ]]; then
        echo "ERROR: Could not resolve CoreDNS service IP" >&2
        return 1
    fi

    local gw_ips
    gw_ips=$(_egress_fw_resolve_gateway_ips "${cluster_container}")
    gw_pod_ip=$(echo "${gw_ips}" | head -1)
    gw_svc_ip=$(echo "${gw_ips}" | tail -1)

    host_bridge_ip=$(_egress_fw_resolve_host_bridge_ip "${cluster_container}" "${sandbox_name}")
    host_bridge_ip="${host_bridge_ip:-172.17.0.1}"

    echo "Enforcing egress firewall for sandbox '${sandbox_name}':"
    echo "  Pod IP:         ${pod_ip}"
    echo "  DNS IP:         ${dns_ip}"
    echo "  Gateway pod:    ${gw_pod_ip}"
    echo "  Gateway svc:    ${gw_svc_ip}"
    echo "  Host bridge:    ${host_bridge_ip}:${host_vllm_port}"

    # Create chain (ignore error if exists)
    docker exec "${cluster_container}" iptables -N "${EGRESS_FW_CHAIN}" 2>/dev/null || true

    # Flush existing rules
    docker exec "${cluster_container}" iptables -F "${EGRESS_FW_CHAIN}"

    # Allow established/related
    docker exec "${cluster_container}" \
        iptables -A "${EGRESS_FW_CHAIN}" -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN

    # Allow DNS
    docker exec "${cluster_container}" \
        iptables -A "${EGRESS_FW_CHAIN}" -d "${dns_ip}/32" -p udp --dport 53 -j RETURN
    docker exec "${cluster_container}" \
        iptables -A "${EGRESS_FW_CHAIN}" -d "${dns_ip}/32" -p tcp --dport 53 -j RETURN

    # Allow OpenShell gateway
    if [[ -n "${gw_pod_ip}" ]]; then
        docker exec "${cluster_container}" \
            iptables -A "${EGRESS_FW_CHAIN}" -d "${gw_pod_ip}/32" -p tcp --dport 8080 -j RETURN
    fi
    if [[ -n "${gw_svc_ip}" ]]; then
        docker exec "${cluster_container}" \
            iptables -A "${EGRESS_FW_CHAIN}" -d "${gw_svc_ip}/32" -p tcp --dport 8080 -j RETURN
    fi

    # Allow host vLLM via Docker bridge
    docker exec "${cluster_container}" \
        iptables -A "${EGRESS_FW_CHAIN}" -d "${host_bridge_ip}/32" -p tcp --dport "${host_vllm_port}" -j RETURN

    # DROP everything else
    docker exec "${cluster_container}" \
        iptables -A "${EGRESS_FW_CHAIN}" -j DROP

    # Insert into FORWARD chain (idempotent)
    docker exec "${cluster_container}" \
        iptables -C FORWARD -s "${pod_ip}/32" -j "${EGRESS_FW_CHAIN}" 2>/dev/null || \
    docker exec "${cluster_container}" \
        iptables -I FORWARD 1 -s "${pod_ip}/32" -j "${EGRESS_FW_CHAIN}"

    echo "  Egress firewall active."
}

remove_sandbox_egress_firewall() {
    local cluster_container="$1"
    local sandbox_name="${2:-thor-assistant}"

    if [[ -z "${cluster_container}" ]]; then
        echo "Usage: remove_sandbox_egress_firewall <cluster-container> [sandbox-name]" >&2
        return 1
    fi

    local pod_ip
    pod_ip=$(_egress_fw_resolve_sandbox_ip "${cluster_container}" "${sandbox_name}")

    if [[ -n "${pod_ip}" ]]; then
        docker exec "${cluster_container}" \
            iptables -D FORWARD -s "${pod_ip}/32" -j "${EGRESS_FW_CHAIN}" 2>/dev/null || true
    fi

    docker exec "${cluster_container}" iptables -F "${EGRESS_FW_CHAIN}" 2>/dev/null || true
    docker exec "${cluster_container}" iptables -X "${EGRESS_FW_CHAIN}" 2>/dev/null || true

    echo "Egress firewall removed for sandbox '${sandbox_name}'."
}

check_sandbox_egress_firewall() {
    local cluster_container="$1"

    if [[ -z "${cluster_container}" ]]; then
        echo "Usage: check_sandbox_egress_firewall <cluster-container>" >&2
        return 1
    fi

    echo "SANDBOX-EGRESS chain rules:"
    docker exec "${cluster_container}" iptables -L "${EGRESS_FW_CHAIN}" -n -v 2>&1

    echo ""
    echo "FORWARD chain reference:"
    docker exec "${cluster_container}" iptables -L FORWARD -n -v 2>&1 | grep -E "${EGRESS_FW_CHAIN}|Chain"
}
