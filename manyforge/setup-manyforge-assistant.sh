#!/usr/bin/env bash
# setup-manyforge-assistant.sh — provision the OpenClaw sandbox to act as a
# ManyForge composer-assistant.
#
# Idempotent. Re-runnable. Source files live in versioned repos; this script
# stages them and applies official NemoClaw / OpenClaw routes.
#
# What this does (officially-supported routes only):
#   1. Apply the `manyforge-composer` egress preset (host:9000) via
#      `nemoclaw <sandbox> policy-add --from-file <preset>`. Companion to
#      the built-in `local-inference` preset.
#   2. Install the `manyforge-composer` skill (SKILL.md + bundled MCP stdio
#      bridge) via `nemoclaw <sandbox> skill install <staging-dir>`. The
#      staging dir is created from the versioned repo files; no /tmp paths
#      end up in any persistent location.
#   3. Register the `manyforge` MCP server in the sandbox's openclaw.json
#      via `openclaw mcp set manyforge '<json>'` (run inside the sandbox
#      via `kubectl exec`). The MCP command points at the bundled bridge.
#   4. Install a dedicated `manyforge-composer` OpenClaw agent profile that
#      points at the ManyForge skill and keeps generic OpenClaw tools minimal.
#
# After this runs successfully, the OpenClaw agent in the sandbox can call
# ManyForge tools via the `manyforge.*` MCP namespace.
#
# Bind requirement: the ManyForge composer must listen on a sandbox-reachable
# address. The default bind (`127.0.0.1`) is NOT reachable from the sandbox;
# the composer launch script must use `--host 0.0.0.0` (or another interface
# that answers on host.openshell.internal:9000). This script verifies the
# listener and prints a remediation hint if it isn't reachable, but does not
# modify the composer's launch.

set -euo pipefail

SANDBOX="${1:-my-assistant}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMOCLAW_THOR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANYFORGE_ROOT="${MANYFORGE_ROOT:-${HOME}/workspaces/dev_ws/src/manyforge}"

# Preset is co-located with this script under manyforge/policies/.
PRESET_PATH="${SCRIPT_DIR}/policies/manyforge-composer.preset.yaml"
SKILL_SRC="${MANYFORGE_ROOT}/agent-skills/manyforge-composer"
# Workspace files versioned in this repo, injected into the agent's
# prompt at every run via OpenClaw's standard workspace-file slots.
# Without these, the agent's prompt contains only OpenClaw's built-in
# session_status tool — meaning the model has no awareness of the
# ManyForge MCP tools and either asks for "session keys" or
# hallucinates plausible-sounding but wrong answers.
#
# Canonical source: the workspace AGENTS.md lives in `manyforge` (the
# deployment repo) alongside the skill, so ManyForge owns the agent's
# semantic guidance (role / vocabulary / tool routing / guardrails).
# This repo (NemoClaw-Thor) only ships the OpenClaw-mechanical overlay
# (currently empty — when needed, append OpenClaw-specific bits like
# the NO_REPLY rule or /sandbox/ paths via WORKSPACE_OVERLAY below).
WORKSPACE_CANONICAL="${MANYFORGE_ROOT}/agent-skills/manyforge-composer/workspace-AGENTS.md"
WORKSPACE_OVERLAY="${SCRIPT_DIR}/agent-workspace/openclaw-overlay.md"

step() {
  printf '\n==> %s\n' "$*"
}

ok() {
  printf '  ✓ %s\n' "$*"
}

fail() {
  printf '  ✗ %s\n' "$*" >&2
  exit 1
}

SANDBOX_EXEC_ATTEMPTS="${MANYFORGE_SANDBOX_EXEC_ATTEMPTS:-8}"
SANDBOX_EXEC_MAX_BACKOFF_S="${MANYFORGE_SANDBOX_EXEC_MAX_BACKOFF_S:-20}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found in PATH: $1"
}

sandbox_exec_is_retryable_error() {
  local text="$1"
  printf '%s' "${text}" | grep -qiE \
    'phase:[[:space:]]*Provisioning|sandbox .*not ready|not ready \(phase:|status:[[:space:]]*Unavailable|DeadlineExceeded|context deadline|connection refused|ECONNREFUSED|socket hang up|transport is closing|temporarily unavailable|no healthy upstream'
}

sandbox_exec_retry() {
  local label="$1"
  shift
  local attempt rc backoff out
  for ((attempt = 1; attempt <= SANDBOX_EXEC_ATTEMPTS; attempt++)); do
    set +e
    out="$(nemoclaw "${SANDBOX}" exec --no-tty -- "$@" 2>&1)"
    rc=$?
    set -e
    if [[ ${rc} -eq 0 ]]; then
      printf '%s\n' "${out}"
      return 0
    fi
    if [[ "${SANDBOX_EXEC_FORCE_RETRY:-0}" != "1" ]] && ! sandbox_exec_is_retryable_error "${out}"; then
      printf '  ✗ %s failed with a non-retryable error (rc=%d)\n' "${label}" "${rc}" >&2
      printf '%s\n' "${out}" | tail -10 | sed 's/^/    /' >&2
      return "${rc}"
    fi
    if [[ ${attempt} -eq ${SANDBOX_EXEC_ATTEMPTS} ]]; then
      printf '  ✗ %s failed after %d attempts (rc=%d)\n' "${label}" "${attempt}" "${rc}" >&2
      printf '%s\n' "${out}" | tail -10 | sed 's/^/    /' >&2
      return "${rc}"
    fi
    backoff=$((2 ** attempt))
    if (( backoff > SANDBOX_EXEC_MAX_BACKOFF_S )); then
      backoff="${SANDBOX_EXEC_MAX_BACKOFF_S}"
    fi
    printf '    %s attempt %d/%d failed (rc=%d); retrying in %ds...\n' "${label}" "${attempt}" "${SANDBOX_EXEC_ATTEMPTS}" "${rc}" "${backoff}" >&2
    sleep "${backoff}"
  done
}

need_cmd nemoclaw
need_cmd docker

if [[ ! -f "${PRESET_PATH}" ]]; then
  fail "Preset not found at ${PRESET_PATH}"
fi
if [[ ! -d "${SKILL_SRC}" ]]; then
  fail "Skill source not found at ${SKILL_SRC}"
fi
if [[ ! -f "${SKILL_SRC}/SKILL.md" ]]; then
  fail "SKILL.md not found in ${SKILL_SRC}"
fi

step "Sandbox check: ${SANDBOX}"
# Use `exec -- true` as the health probe. `nemoclaw status` is the obvious
# choice but NemoClaw 0.0.56 throws "TypeError: shields.getShieldsPosture
# is not a function" in `status` regardless of actual sandbox health,
# blocking the gate even when the sandbox is fully up. The exec probe
# actually connects to the OpenShell exec endpoint and runs a command,
# which is a stronger liveness signal than `status` anyway.
if ! sandbox_exec_retry "initial sandbox liveness probe" true >/dev/null; then
  fail "Sandbox '${SANDBOX}' not found or unhealthy. Run 'nemoclaw list' to inspect."
fi
ok "sandbox ${SANDBOX} is up"

# Skill-vs-runtime compatibility gate: the SKILL.md's contract block
# declares the assistant mode it expects. Refuse to install if the
# running Composer either isn't reachable or doesn't expose that mode.
# This catches the "stale skill" failure mode at install time rather
# than at first MCP call.
# Precheck runs on the HOST (validating the composer is up), while the
# sandbox-stored MCP config uses host.openshell.internal:9000 from inside
# the sandbox container. These are two different namespaces:
#   - On the host, host.openshell.internal does NOT resolve.
#   - In the sandbox, localhost is the sandbox's own loopback (nothing
#     listening on :9000 there).
# Use MANYFORGE_COMPOSER_PRECHECK_BASE for the host-side validation URL.
# It defaults to http://localhost:9000 which is correct on the host.
# MANYFORGE_COMPOSER_BASE (used farther down for the MCP env in the
# sandbox) stays at host.openshell.internal:9000 unless explicitly
# overridden for a non-default routing setup.
PRECHECK_COMPOSER_BASE="${MANYFORGE_COMPOSER_PRECHECK_BASE:-http://localhost:9000}"
PRECHECK_MODE="${MANYFORGE_ASSISTANT_MODE:-composer-assistant}"
PRECHECK_URL="${PRECHECK_COMPOSER_BASE}/api/assistant/modes/${PRECHECK_MODE}"
step "Runtime compatibility check: ${PRECHECK_URL}"
PRECHECK_BODY="$(curl -fsS --max-time 5 "${PRECHECK_URL}" 2>&1 || true)"
if [[ -z "${PRECHECK_BODY}" || "${PRECHECK_BODY}" == *"<html"* ]]; then
  fail "Composer mode endpoint did not answer at ${PRECHECK_URL}.

  Either the composer is not running, it is bound to 127.0.0.1 only, or no
  deployment with assistant_modes['${PRECHECK_MODE}'] is loaded. Start with:

      cd \${MANYFORGE_ROOT} && \\
        COMPOSER_BIND_HOST=0.0.0.0 ./scripts/demo-assistant-known-good.sh"
fi
PRECHECK_HASH="$(printf '%s' "${PRECHECK_BODY}" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("catalogHash",""))' 2>/dev/null || true)"
if [[ -z "${PRECHECK_HASH}" ]]; then
  fail "Composer responded but did not return a catalogHash. Body head: ${PRECHECK_BODY:0:200}"
fi
ok "composer mode '${PRECHECK_MODE}' reachable (catalogHash: ${PRECHECK_HASH:0:16}…)"

step "Step 1/5: apply lane presets (local-inference + manyforge-composer merged)"
# REVISED 2026-06-03 per THREE-LANE-MIGRATION-PLAN route fix: keep
# 'local-inference' active alongside the manyforge stack. Prior versions
# of this script removed 'local-inference' on the assumption that the
# manyforge preset was a strict superset for the inference path.
# Empirical evidence in the Phase 0 O-1 baseline (docs/archive/PHASE-0-LANE-BASELINE.md)
# shows it is NOT — without local-inference active, the OpenShell network
# proxy denies sandbox→host:8000 with `policy_denied` for chat-completion
# POSTs.
#
# 2026-06-03 (LATE-DAY): the split-file approach (shared FIRST, overlay
# SECOND) is BROKEN by current `nemoclaw policy-add` semantics. Applying
# the overlay does NOT additively merge the `network_policies.manyforge_composer`
# block — it REPLACES the block, dropping the endpoints from the shared
# preset and leaving only `binaries`. Result: the MCP bridge's proxy
# request to host.openshell.internal:9000 returns HTTP 403 ("not permitted
# by policy") and `bundle-mcp` logs "Request timed out" because the
# tools/list handler can never fetch the mode manifest. Verified
# end-to-end 2026-06-03 by reading `openshell policy get my-assistant
# --full` after each apply step — endpoints present at v18 (shared only),
# gone at v19 (overlay applied on top), present again at v20 with the
# merged file. We now apply a SINGLE merged file that contains both the
# endpoints AND the binary subject whitelist (including the versioned
# Python interpreter `/usr/bin/python3.13` that `readlink /proc/<pid>/exe`
# resolves to on the OpenShell `my-assistant` base image — the un-versioned
# `/usr/bin/python3` symlink alone does NOT satisfy the policy enforcer).
# The split files remain in `policies/` for documentation and Hermes-lane
# composition; bring-up uses the merged file as the source of truth.
MERGED_PRESET_PATH="${SCRIPT_DIR}/policies/manyforge-composer.merged.yaml"
# ALWAYS reapply (no policy-list check). The daemon detects unchanged
# content and emits "Policy unchanged" if at the latest version, otherwise
# it loads the new version. Both outcomes are safe. Skipping based on
# `policy-list` output is actively harmful — we observed today that the
# live sandbox sometimes shows the preset as active (●) while the actual
# loaded policy has stale or partial rules; only a fresh apply guarantees
# the merge order is correct.
if nemoclaw "${SANDBOX}" policy-list 2>&1 | grep -qE "● .*local-inference"; then
  ok "preset 'local-inference' already applied"
else
  nemoclaw "${SANDBOX}" policy-add local-inference --yes
  ok "preset 'local-inference' applied"
fi
nemoclaw "${SANDBOX}" policy-add --from-file "${MERGED_PRESET_PATH}" --yes 2>&1 | tail -2 | sed 's/^/    /'
ok "preset 'manyforge-composer-merged' applied (or already at latest version)"
# Verify the merged policy actually carries both port-9000 endpoints and
# the python3.13 binary. If either is missing the merge failed silently
# and the bridge will fail to register its catalog; fail loudly here
# rather than leave the operator chasing a "Unknown tool id" error.
ACTIVE_POLICY="$(openshell policy get "${SANDBOX}" --full 2>/dev/null || true)"
if ! echo "${ACTIVE_POLICY}" | grep -q "port: 9000"; then
  echo "    ERROR: active policy missing port:9000 endpoint after merged-preset apply"
  echo "    Run 'openshell policy get ${SANDBOX} --full' to inspect"
  exit 1
fi
if ! echo "${ACTIVE_POLICY}" | grep -q "/usr/bin/python3.13"; then
  echo "    ERROR: active policy missing /usr/bin/python3.13 binary subject"
  echo "    Bridge subprocess exe will not match policy and proxy will return 403"
  exit 1
fi
ok "active policy verified: port:9000 endpoints + /usr/bin/python3.13 subject present"
# Remove the legacy split overlay if it's still applied — its presence
# would re-trigger the overwrite-bug on the next bring-up because the
# CLI replays applied presets in the order they were added.
#
# REGEX SAFETY: match at end-of-name (` ` or end-of-line) to avoid
# matching the merged preset by prefix — `manyforge-composer\b` would
# match `manyforge-composer-merged` because `\b` triggers at the `r-`
# boundary. That bug silently removed the freshly-applied merged
# preset on every launcher restart, reverting the policy to
# local-inference only and breaking the OpenClaw lane. Fixed by
# anchoring the match to an explicit end (space, dash to em-dash, or
# EOL) per preset name.
for legacy in manyforge-egress-shared manyforge-openclaw-overlay manyforge-composer; do
  if nemoclaw "${SANDBOX}" policy-list 2>&1 \
       | grep -qE "● ${legacy}( |$|—)"; then
    nemoclaw "${SANDBOX}" policy-remove "${legacy}" --yes 2>/dev/null \
      && ok "legacy preset '${legacy}' removed (merged preset now authoritative)" \
      || echo "    WARN: legacy preset '${legacy}' still present; remove manually"
  fi
done

step "Step 1b: patch openclaw.json inference provider (baseUrl + timeoutSeconds)"
# REVISED 2026-06-03 per THREE-LANE-MIGRATION-PLAN route fix. The
# default OpenClaw config sets models.providers.inference.baseUrl to
# https://inference.local/v1 — a TLS-terminating route routed via
# OpenShell's network proxy. On OpenShell 0.0.44 this route does not
# reliably reach our :8000 vllm-proxy (the proxy receives 0 POSTs).
#
# Bypass the inference.local hop by setting baseUrl directly to
# http://host.openshell.internal:8000/v1 — which the proxy at
# 10.200.0.1:3128 forwards via our :8000 proxy (with local-inference
# preset above allowing the route). All four proxy mutations
# (UNWRAP_TOOL_CALL_ARGS, PROMOTE_REASONING_TO_CONTENT,
# NORMALIZE_TOOL_NAMES, TOOL_ERROR_REWRITE) are then applied to every
# /chat/completions request.
#
# REVISED 2026-06-04: also set models.providers.inference.timeoutSeconds.
# OpenClaw has a model-stream idle watchdog (separate from the runtime
# timeoutMs the bridge sends in the request body). Default is 120s. On
# long tool-search chains and reasoning-heavy turns the model can go
# >120s between stream chunks, which fires EmbeddedAttemptSessionTakeoverError
# with returncode 1 (the bridge then translates this to HTTP 502). 300s
# matches OPENCLAW_ASSISTANT_TIMEOUT_S and the smoke runner default.
cat > /tmp/manyforge-patch-openclaw-baseurl.py <<'PY'
import json, hashlib, pathlib
p = pathlib.Path("/sandbox/.openclaw/openclaw.json")
d = json.loads(p.read_text())
inf = d.setdefault("models", {}).setdefault("providers", {}).setdefault("inference", {})
changes = []
target_url = "http://host.openshell.internal:8000/v1"
old_url = inf.get("baseUrl")
if old_url != target_url:
    inf["baseUrl"] = target_url
    changes.append(f"baseUrl: {old_url} -> {target_url}")
target_timeout = 300
old_timeout = inf.get("timeoutSeconds")
if old_timeout != target_timeout:
    inf["timeoutSeconds"] = target_timeout
    changes.append(f"timeoutSeconds: {old_timeout} -> {target_timeout}")
if not changes:
    print("models.providers.inference already correct — no change")
else:
    p.write_text(json.dumps(d, indent=2))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    pathlib.Path("/sandbox/.openclaw/.config-hash").write_text(f"{h}  openclaw.json\n")
    print("; ".join(changes) + "; hash refreshed")
PY
openshell sandbox upload "${SANDBOX}" /tmp/manyforge-patch-openclaw-baseurl.py /tmp/ >/dev/null
sandbox_exec_retry "openclaw.json baseUrl patch" python3 /tmp/manyforge-patch-openclaw-baseurl.py | tail -1
ok "openclaw.json baseUrl + timeoutSeconds patched (will be picked up by gateway hot-reload or next recover)"

step "Step 1c: set tools.toolSearch.mode=tools (Phase 3 production default)"
# Tools mode is the production default for the OpenClaw lane since
# Phase 3 (2026-06-03). Measured 58.3% effective rate on cosmos-
# reason2-8b vs 29.0% for code mode (see
# docs/LANE-COMPARISON.md). The flip is a single field
# in openclaw.json: the bare boolean ``tools.toolSearch: true`` gets
# replaced with the dict form ``tools.toolSearch: {enabled: true,
# mode: "tools"}`` which signals the OpenClaw runtime
# (resolveToolSearchConfig in pi-tools-iVT6BGHc.js) to expose
# tool_search / tool_describe / tool_call in the model's tools[]
# instead of tool_search_code.
cat > /tmp/manyforge-patch-openclaw-toolsearch.py <<'PY'
import json, hashlib, pathlib
p = pathlib.Path("/sandbox/.openclaw/openclaw.json")
d = json.loads(p.read_text())
tools = d.setdefault("tools", {})
current = tools.get("toolSearch")
target = {"enabled": True, "mode": "tools"}
if isinstance(current, dict) and current.get("mode") == "tools" and current.get("enabled", True):
    print("toolSearch already tools mode — no change")
else:
    tools["toolSearch"] = target
    p.write_text(json.dumps(d, indent=2))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    pathlib.Path("/sandbox/.openclaw/.config-hash").write_text(f"{h}  openclaw.json\n")
    print(f"toolSearch: {current!r} -> {target!r}; hash refreshed")
PY
openshell sandbox upload "${SANDBOX}" /tmp/manyforge-patch-openclaw-toolsearch.py /tmp/ >/dev/null
sandbox_exec_retry "openclaw.json toolSearch patch" python3 /tmp/manyforge-patch-openclaw-toolsearch.py | tail -1
ok "openclaw.json tools.toolSearch.mode patched (gateway must be restarted to take effect — see RUNBOOK)"

step "Step 2/5: stage skill (resolves repo symlinks; no /tmp left in persistent state)"
STAGING_DIR="$(mktemp -d -t manyforge-skill-XXXX)"
trap 'rm -rf "${STAGING_DIR}"' EXIT

cp -L "${SKILL_SRC}/SKILL.md" "${STAGING_DIR}/SKILL.md"
# Copy any non-dot files, dereferencing symlinks (the MCP bridge is
# symlinked from manyforge/scripts/manyforge-mcp-bridge.py to keep a single
# source of truth in the repo).
shopt -s nullglob
for f in "${SKILL_SRC}"/*; do
  base="$(basename "$f")"
  case "${base}" in
    SKILL.md|.*|__pycache__|*.pyc|node_modules) continue ;;
  esac
  # Skip source-tree artifacts (pycache, venv, etc.) — only flat
  # companion files belong in the staged bundle.
  [[ -d "$f" && ! -L "$f" ]] && continue
  cp -L "$f" "${STAGING_DIR}/${base}"
done
shopt -u nullglob

ok "staged $(ls "${STAGING_DIR}" | wc -l) file(s) under ${STAGING_DIR}"

step "Step 3/5: install skill 'manyforge-composer'"
# Idempotency probe: compare the staged content hash against the
# in-sandbox copy. When they match, skip the upload entirely — this
# works around a nemoclaw idempotency regression observed 2026-05-08
# where `skill install` against a sandbox that already has the skill
# fails with "Failed to upload N file(s)" without surfacing a useful
# reason. The skill content is what matters; if hashes match, the
# in-sandbox copy is correct and we can move on.
# OpenShell 0.0.44+ docker driver: no k3s/kubectl. `nemoclaw exec`
# runs as the sandbox user (uid 998, HOME=/sandbox) via the OpenShell
# exec endpoint, so we don't need `su sandbox -c`.
KEX_USER=(nemoclaw "${SANDBOX}" exec --no-tty -- bash -c)

sandbox_bash_retry() {
  local label="$1"
  local script="$2"
  sandbox_exec_retry "${label}" bash -c "${script}"
}

# Like sandbox_bash_retry, but retries on ANY non-zero exit — not only the
# infra-transient signatures matched by sandbox_exec_is_retryable_error. Use
# ONLY for IDEMPOTENT commands. Motivating case: the openclaw.json writers
# below (`openclaw mcp set`, and the python read-modify-write edits for the
# agent profile / model.reasoning / compaction route) race the OpenClaw
# gateway's hot-reload of openclaw.json triggered by the Step 1b/1c patches.
# That race surfaces as a transient rc=1 whose text matches no infra pattern,
# so plain sandbox_bash_retry treats it as non-retryable and aborts the whole
# launch; re-running the same idempotent write a moment later succeeds. A
# genuinely broken command still fails after SANDBOX_EXEC_ATTEMPTS (surfacing
# its rc) — just more slowly.
sandbox_bash_retry_idempotent() {
  local label="$1"
  local script="$2"
  local rc
  local prev="${SANDBOX_EXEC_FORCE_RETRY:-0}"
  SANDBOX_EXEC_FORCE_RETRY=1
  set +e
  sandbox_exec_retry "${label}" bash -c "${script}"
  rc=$?
  set -e
  SANDBOX_EXEC_FORCE_RETRY="${prev}"
  return "${rc}"
}

sandbox_bash_optional() {
  local label="$1"
  local script="$2"
  local out rc
  set +e
  out="$(sandbox_bash_retry "${label}" "${script}" 2>&1)"
  rc=$?
  set -e
  if [[ ${rc} -eq 0 ]]; then
    printf '%s\n' "${out}"
    return 0
  fi
  printf '  ! optional diagnostic skipped: %s\n' "${label}" >&2
  printf '%s\n' "${out}" | tail -8 | sed 's/^/    /' >&2
  return 0
}

sandbox_settle() {
  local label="${1:-sandbox settle}"
  sandbox_bash_retry "${label}" "true" >/dev/null
}

SKILL_REMOTE_DIR="/sandbox/.openclaw/skills/manyforge-composer"
# Concatenated sha256 of every staged file. Order-stable (sorted by
# filename) so we don't trip on directory-iteration nondeterminism.
staged_hash() {
  ( cd "${STAGING_DIR}" && \
      find . -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort | \
      xargs -I{} sha256sum {} ) 2>/dev/null | sha256sum | cut -d' ' -f1
}
remote_hash() {
  # Wrap the whole pipe in a subshell with `|| true` so that a missing
  # skill directory on a fresh sandbox (test -d fails) doesn't propagate
  # through pipefail and trip the outer `set -e`. Empty stdout from an
  # absent directory is the correct signal ("no remote skill yet"); we
  # don't want it to be a fatal script error.
  ( "${KEX_USER[@]}" "test -d ${SKILL_REMOTE_DIR} && cd ${SKILL_REMOTE_DIR} && find . -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort | xargs -I{} sha256sum {} | sha256sum | cut -d' ' -f1" 2>/dev/null \
      | tr -d '[:space:]' | tail -c 64 ) || true
}
STAGED_HASH="$(staged_hash)"
SANDBOX_HASH="$(remote_hash)"
if [[ -n "${SANDBOX_HASH}" && "${STAGED_HASH}" == "${SANDBOX_HASH}" ]]; then
  ok "skill 'manyforge-composer' already installed (content sha256 ${STAGED_HASH:0:12}…); skipping upload"
else
  if nemoclaw "${SANDBOX}" skill install "${STAGING_DIR}"; then
    ok "skill installed"
  else
    install_rc=$?
    # If nemoclaw failed but a copy already exists in-sandbox, the
    # most likely cause is the upstream idempotency bug. Re-check the
    # remote hash; if it now equals the staged hash, treat as success
    # (race: maybe nemoclaw partial-uploaded then refused). Otherwise
    # fall through to fail() so the operator sees the real problem.
    SANDBOX_HASH_AFTER="$(remote_hash)"
    if [[ -n "${SANDBOX_HASH_AFTER}" && "${STAGED_HASH}" == "${SANDBOX_HASH_AFTER}" ]]; then
      ok "skill 'manyforge-composer' already in-sandbox at the expected version (sha256 ${STAGED_HASH:0:12}…); continuing despite nemoclaw exit ${install_rc}"
    elif [[ -n "${SANDBOX_HASH_AFTER}" ]]; then
      printf '  ! nemoclaw skill install failed (exit %d) and the in-sandbox copy differs from staged content.\n' "${install_rc}" >&2
      printf '  ! staged sha256:  %s\n' "${STAGED_HASH}" >&2
      printf '  ! sandbox sha256: %s\n' "${SANDBOX_HASH_AFTER}" >&2
      fail "skill install failed and in-sandbox copy is stale; resolve manually"
    else
      fail "skill install failed (exit ${install_rc}) and no skill found in sandbox"
    fi
  fi
fi

step "Step 4/5: register 'manyforge' MCP server in sandbox openclaw.json"
MCP_BRIDGE_PATH="/sandbox/.openclaw/skills/manyforge-composer/manyforge-mcp-bridge.py"
COMPOSER_BASE="${MANYFORGE_COMPOSER_BASE:-http://host.openshell.internal:9000}"
ASSISTANT_MODE="${MANYFORGE_ASSISTANT_MODE:-composer-assistant}"
MCP_PRINCIPAL="${MANYFORGE_PRINCIPAL:-openclaw-${SANDBOX}}"
# Mode-scoped path: the bridge script translates MCP JSON-RPC into calls
# to /api/assistant/bridge/tools/{toolId} with a full bounded-autonomy
# envelope (assistantMode + catalogHash + requestId + conversationId +
# principal). Server-side enforcement is the source of truth; the bridge
# only narrows the visible surface.
#
# Proxy envs: OpenClaw spawns MCP servers with a SCRUBBED environment
# (HOME, PATH, USER, SHELL, MANYFORGE_*, plus the keys listed here only).
# host.openshell.internal:9000 is reachable from the sandbox; the
# behavior of HTTP_PROXY for THAT route is the tricky bit:
#
#   - For inference traffic to host.openshell.internal:8000 (vllm-proxy)
#     we WANT to go through OpenShell's egress proxy 10.200.0.1:3128 —
#     that proxy enforces the local-inference policy preset and the
#     manyforge-egress-shared preset, which together allow the route.
#   - For MCP traffic to host.openshell.internal:9000 (Composer's
#     mode-scoped /api/assistant/...) the bridge MUST route through
#     the same proxy. The OpenClaw gateway spawns the MCP bridge in
#     an isolated network namespace; direct TCP to 172.18.0.1 (which
#     host.openshell.internal resolves to) times out because that
#     gateway IP is unreachable from the bridge's netns. The only
#     network path back to Composer is via the OpenShell egress
#     proxy at 10.200.0.1:3128. That proxy will allow the route iff
#     the `manyforge_composer` policy block carries the port-9000
#     endpoints AND the bridge's exe path (/usr/bin/python3.13 on
#     the current my-assistant base image) is in its binaries list.
#     Both are enforced by the merged preset applied in Step 1.
#
# Earlier setup-script revisions DEFENSIVELY appended
# host.openshell.internal to NO_PROXY on the assumption that a direct
# connection was available. That assumption is WRONG for the OpenClaw
# MCP child — adding it makes urllib bypass the proxy, then the
# bridge hangs on `tools/list` for the full REQUEST_TIMEOUT because
# the TCP connect to 172.18.0.1 silently times out, after which the
# gateway logs the misleading `MCP error -32001: Request timed out`.
# Verified end-to-end 2026-06-03 from the bridge's own trace file
# (MANYFORGE_MCP_TRACE) on three back-to-back boots.
HTTP_PROXY_VAL="${HTTP_PROXY:-${http_proxy:-http://10.200.0.1:3128}}"
# Loopback hosts ONLY in NO_PROXY. The bridge must reach Composer at
# host.openshell.internal:9000 via the OpenShell egress proxy — the
# isolated netns has no other route.
NO_PROXY_VAL="${NO_PROXY:-${no_proxy:-127.0.0.1,localhost,::1}}"
# Defensive: STRIP host.openshell.internal if an inherited env added
# it. Operators carrying a stale NO_PROXY from the old setup would
# otherwise break the bridge (see comment block above).
NO_PROXY_VAL="$(echo "${NO_PROXY_VAL}" | sed -E 's/,?host\.openshell\.internal//g; s/^,//; s/,$//')"
MCP_CONFIG_JSON=$(cat <<JSON
{"command":"python3","args":["${MCP_BRIDGE_PATH}"],"env":{"MANYFORGE_COMPOSER_BASE":"${COMPOSER_BASE}","MANYFORGE_ASSISTANT_MODE":"${ASSISTANT_MODE}","MANYFORGE_PRINCIPAL":"${MCP_PRINCIPAL}","HTTP_PROXY":"${HTTP_PROXY_VAL}","HTTPS_PROXY":"${HTTP_PROXY_VAL}","NO_PROXY":"${NO_PROXY_VAL}","http_proxy":"${HTTP_PROXY_VAL}","https_proxy":"${HTTP_PROXY_VAL}","no_proxy":"${NO_PROXY_VAL}"}}
JSON
)
sandbox_bash_retry_idempotent "register manyforge MCP server" "openclaw mcp set manyforge '${MCP_CONFIG_JSON}'" >/dev/null
ok "MCP server 'manyforge' registered (mode: ${ASSISTANT_MODE}; principal: ${MCP_PRINCIPAL})"
sandbox_bash_optional "show manyforge MCP server" "openclaw mcp show manyforge" 2>&1 | sed 's/^/    /'

step "Step 5/6: install dedicated OpenClaw agent profile 'manyforge-composer'"

# Note (2026-05-06, post lane-parity probe):
# Per-model sampling params (temperature, top_k, top_p,
# chat_template_kwargs.enable_thinking) are NOT settable via OpenClaw
# config — there are no schema fields for them anywhere in
# /usr/local/lib/node_modules/openclaw/dist/runtime-schema-cADw9D2m.js,
# and OpenClaw never forwards them on the wire (verified by tcpdump).
# The current source of truth for sampling defaults is vLLM's own
# server-side flags, baked into the matching profile in
# nemoclaw-thor/serving/launch.sh:
#   --override-generation-config '{"temperature":0.6,"top_p":0.95}'
#   --default-chat-template-kwargs '{"enable_thinking":false}'
# Step 6/6 below sets `models.providers.inference.models[].reasoning`
# to true on the active model — this flag is consumed *internally* by
# OpenClaw's loop runner and gives noticeably better tool-error
# recovery on multi-step planning tasks (verified by lane-parity
# probe 2026-05-06: tree_wrap dropped from 97-turn timeout to
# 12 turns / 20.6 s). It does not change anything on the wire.

PROFILE_SCRIPT_B64="$(cat <<'PY' | base64 -w0
import json
import os

path = os.path.expanduser("~/.openclaw/openclaw.json")
try:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
except FileNotFoundError:
    data = {}

agents = data.setdefault("agents", {})
entries = agents.get("list")
if not isinstance(entries, list):
    entries = []
    agents["list"] = entries

profile = {
    "id": "manyforge-composer",
    "name": "ManyForge Composer Assistant",
    "skills": ["manyforge-composer"],
    "thinkingDefault": "off",
    # The "minimal" CORE_TOOL_PROFILES allow list does NOT include
    # "bundle-mcp" (only "coding"/"messaging"/"full" do). With profile
    # alone, manyforge MCP tools register and advertise to the model,
    # the model emits tool_calls correctly, but execution is policy-
    # blocked → "Manyforge Scene-inspect failed" → retry-until-timeout.
    # Verified 2026-05-05 by reading
    # /usr/local/lib/node_modules/openclaw/dist/tool-policy-DArLXMH2.js.
    # alsoAllow keeps the minimal core surface but unblocks bundle-mcp.
    "tools": {"profile": "minimal", "alsoAllow": ["bundle-mcp"]},
    "skillsLimits": {"maxSkillsPromptChars": 24000},
    # Tool-result size budget: catalog.read returns ~66 KB on the
    # current ur10e_robotiq deployment (34 entries with full param
    # metadata). The previous 20 KB cap truncated the JSON tail,
    # the model then looped on an invalid result, and the run timed
    # out at 240s (verified 2026-05-06 against R5 in the comparison
    # matrix). 100 KB gives 1.5x headroom over the worst observed
    # tool result and matches what direct-vLLM lane handles cleanly.
    # postCompactionMaxChars also raised so the compacted prompt can
    # still carry the full result on multi-turn conversations.
    # OpenClaw 2026.5.22 caps postCompactionMaxChars at 50000 (was
    # uncapped in 2026.4.24); larger values fail validation and the
    # config is refused. 50000 is enough to carry catalog_read output
    # through one compaction, but tight: if multi-turn compactions
    # start dropping tool history, request OpenClaw to raise the cap.
    "contextLimits": {
        "toolResultMaxChars": 100000,
        "postCompactionMaxChars": 50000,
    },
}

entries[:] = [
    entry
    for entry in entries
    if not (isinstance(entry, dict) and entry.get("id") == "manyforge-composer")
]
entries.append(profile)

tmp = f"{path}.tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp, path)
print("manyforge-composer")
PY
)"
sandbox_bash_retry_idempotent "install manyforge-composer agent profile" "printf %s '${PROFILE_SCRIPT_B64}' | base64 -d | python3 -" >/dev/null
ok "agent profile 'manyforge-composer' installed"
sandbox_settle "sandbox settle after agent profile install"
sandbox_bash_optional "list OpenClaw agents" "openclaw agents list --json 2>/dev/null | head -c 1200 || openclaw agents list" 2>&1 | sed 's/^/    /'

step "Step 5b/6: install workspace guidance file (AGENTS.md)"
# AGENTS.md is injected into every agent run via OpenClaw's standard
# workspace-file slot. It carries: vocabulary lock (no session keys),
# the output protocol, the categorical tool surface (which mirrors
# tools/list, never replaces it), and the guardrails (mangling rule,
# don't invent ids, etc.).
#
# Composition pattern (since 2026-05-08): the canonical content lives
# in `manyforge` (deployment repo, alongside the skill), and this
# provisioner *composes* the in-sandbox file by concatenating:
#   (a) the canonical manyforge file (semantic guidance), then
#   (b) any OpenClaw-mechanical overlay this repo ships (currently
#       optional — empty by default; append OpenClaw-specific bits
#       like NO_REPLY rules or /sandbox/ paths to it as needed).
# This eliminates the prior drift hazard where ManyForge-semantic
# content lived in two places (manyforge skill AND NemoClaw-Thor
# workspace) and could diverge.
#
# v7 (2026-05-06): TOOLS.md was folded into the AGENTS.md guardrails
# section and the standalone file is no longer installed. If the
# sandbox has a stale TOOLS.md from a prior provisioner run, we
# delete it explicitly so the agent stops reading two-source-of-truth
# content.
if [[ ! -f "${WORKSPACE_CANONICAL}" ]]; then
  fail "workspace canonical source not found at ${WORKSPACE_CANONICAL} \
(expected in manyforge repo at agent-skills/manyforge-composer/workspace-AGENTS.md)"
fi
WORKSPACE_DIR_REMOTE="/sandbox/.openclaw/workspace"
WORKSPACE_TMP="$(mktemp -t manyforge-workspace-AGENTS-XXXXXX.md)"
trap 'rm -f "${WORKSPACE_TMP}"' EXIT
cat "${WORKSPACE_CANONICAL}" > "${WORKSPACE_TMP}"
if [[ -f "${WORKSPACE_OVERLAY}" ]]; then
  printf '\n\n' >> "${WORKSPACE_TMP}"
  cat "${WORKSPACE_OVERLAY}" >> "${WORKSPACE_TMP}"
  ok "  composed canonical (${WORKSPACE_CANONICAL}) + overlay (${WORKSPACE_OVERLAY})"
else
  ok "  using canonical only (no platform overlay at ${WORKSPACE_OVERLAY})"
fi
sandbox_bash_retry "create OpenClaw workspace dir" "mkdir -p ${WORKSPACE_DIR_REMOTE}" >/dev/null
WS_B64="$(base64 -w0 < "${WORKSPACE_TMP}")"
sandbox_bash_retry "install OpenClaw workspace AGENTS.md" "printf %s '${WS_B64}' | base64 -d > ${WORKSPACE_DIR_REMOTE}/AGENTS.md" >/dev/null
sandbox_bash_retry "remove stale OpenClaw workspace TOOLS.md" "rm -f ${WORKSPACE_DIR_REMOTE}/TOOLS.md" >/dev/null
# Empty workspace stub files (2026-05-08): suppress OpenClaw's
# "[MISSING] Expected at: ..." placeholder lines for SOUL.md /
# IDENTITY.md / USER.md. OpenClaw's runtime checks for these
# convention files in the workspace and either includes their
# content in the system prompt or emits a MISSING placeholder
# (~80 chars per missing file). We don't author any of those four
# files, so the placeholders are pure prompt noise. An empty stub
# (single "<!-- intentionally empty -->" line) is enough to
# replace the MISSING placeholder with a near-zero-cost include.
# Saves ~300 chars of system prompt per turn × every turn.
for stub in SOUL.md IDENTITY.md USER.md; do
  sandbox_bash_retry "ensure OpenClaw workspace ${stub} stub" "test -e ${WORKSPACE_DIR_REMOTE}/${stub} || echo '<!-- intentionally empty: stub by setup-manyforge-assistant.sh -->' > ${WORKSPACE_DIR_REMOTE}/${stub}" >/dev/null
done
ok "installed AGENTS.md into ${WORKSPACE_DIR_REMOTE} (+ empty SOUL/IDENTITY/USER stubs; removed any stale TOOLS.md)"
sandbox_bash_optional "list OpenClaw workspace dir" "ls -la ${WORKSPACE_DIR_REMOTE}/" 2>&1 | sed 's/^/    /'

step "Step 6/6: enable OpenClaw internal reasoning loop on the active model"
# Sets `models.providers.inference.models[].reasoning = true` on the
# entry matching the active served-model name (auto-detected from
# vLLM's /v1/models). The flag stays inside OpenClaw — never reaches
# vLLM — and improves the loop's tool-error recovery on multi-step
# planning tasks. See the comment block above Step 5/6 for context.
REASONING_SCRIPT_B64="$(cat <<'PY' | base64 -w0
import json
import os
import sys
import urllib.request

target_id = os.environ.get("MANYFORGE_MODEL_NAME") or ""
if not target_id:
    try:
        with urllib.request.urlopen(
            "http://host.openshell.internal:8000/v1/models", timeout=5
        ) as resp:
            data = json.load(resp)
            entries = data.get("data") or []
            if entries:
                target_id = entries[0].get("id") or ""
    except Exception:
        pass

if not target_id:
    print("WARN: could not determine active model name; skipping reasoning flip", file=sys.stderr)
    sys.exit(0)

path = os.path.expanduser("~/.openclaw/openclaw.json")
with open(path, "r", encoding="utf-8") as handle:
    cfg = json.load(handle)

provider = cfg.get("models", {}).get("providers", {}).get("inference", {})
models = provider.get("models", [])
hit = False
for entry in models:
    if isinstance(entry, dict) and entry.get("id") == target_id:
        before = entry.get("reasoning")
        entry["reasoning"] = True
        print(f"reasoning: {before} -> True ({target_id})")
        hit = True
        break

if not hit:
    print(
        f"WARN: model {target_id!r} not found in inference.models; available: "
        f"{[m.get('id') for m in models if isinstance(m, dict)]}",
        file=sys.stderr,
    )
    sys.exit(0)

tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(cfg, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp, path)
PY
)"
sandbox_bash_retry_idempotent "enable OpenClaw model.reasoning" "printf %s '${REASONING_SCRIPT_B64}' | base64 -d | python3 -" 2>&1 | sed 's/^/    /'
ok "OpenClaw model.reasoning=true ensured on active inference model"

# 2026-05-10 (iter 32): route OpenClaw's compaction calls through the
# same local inference model. The default is `openai/gpt-5.5` which is
# unreachable from the sandbox; without this fix, every `/compact` call
# (bridge-fired via OPENCLAW_ASSISTANT_COMPACT_EVERY_N) silently fails
# and the chain-session-ON PnP cascade returns. Idempotent: re-runs
# leave the field at the same value.
step "Step 6b/6: route OpenClaw compaction through the active inference model"
COMPACTION_SCRIPT_B64="$(cat <<'PY' | base64 -w0
import json
import os
import sys
import urllib.request

target_id = os.environ.get("MANYFORGE_MODEL_NAME") or ""
if not target_id:
    try:
        with urllib.request.urlopen(
            "http://host.openshell.internal:8000/v1/models", timeout=5
        ) as resp:
            data = json.load(resp)
            entries = data.get("data") or []
            if entries:
                target_id = entries[0].get("id") or ""
    except Exception:
        pass

if not target_id:
    print("WARN: could not determine active model name; skipping compaction route", file=sys.stderr)
    sys.exit(0)

compact_model = f"inference/{target_id}"
path = os.path.expanduser("~/.openclaw/openclaw.json")
with open(path, "r", encoding="utf-8") as handle:
    cfg = json.load(handle)

agents = cfg.setdefault("agents", {})
defaults = agents.setdefault("defaults", {})
compaction = defaults.setdefault("compaction", {})
before = compaction.get("model")
compaction["model"] = compact_model
print(f"agents.defaults.compaction.model: {before} -> {compact_model}")

tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(cfg, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp, path)
PY
)"
sandbox_bash_retry_idempotent "route OpenClaw compaction model" "printf %s '${COMPACTION_SCRIPT_B64}' | base64 -d | python3 -" 2>&1 | sed 's/^/    /'
ok "OpenClaw agents.defaults.compaction.model routed to the active inference model"

step "Composer reachability check (mode-scoped manifest)"
# Host-side curl uses PRECHECK_COMPOSER_BASE (localhost:9000 by default)
# because the COMPOSER_BASE value stored in the MCP config (e.g.
# host.openshell.internal:9000) is only resolvable from inside the
# sandbox.
HOST_MODE_URL="${PRECHECK_COMPOSER_BASE}/api/assistant/modes/${ASSISTANT_MODE}"
# Stored URL is what the in-sandbox MCP bridge will use; logged for
# verification but not curled from the host.
MODE_URL="${COMPOSER_BASE}/api/assistant/modes/${ASSISTANT_MODE}"
if curl -fsS -o /dev/null --max-time 3 "${HOST_MODE_URL}" 2>/dev/null; then
  ok "composer ${HOST_MODE_URL} reachable from this host (sandbox MCP will use ${MODE_URL})"
else
  printf '  ! composer mode endpoint %s did not answer from this host (curl).\n' "${MODE_URL}"
  printf '  ! If the composer is bound to 127.0.0.1 only, it will not reach. Confirm with:\n'
  printf '  !   ss -tlnp | grep :9000\n'
  printf '  ! and rebind via demo-assistant-known-good.sh with COMPOSER_BIND_HOST=0.0.0.0.\n'
  printf '  ! Also confirm a deployment with assistant_modes is loaded — without one,\n'
  printf '  ! /api/assistant/modes/${ASSISTANT_MODE} returns 404.\n'
fi

step "Sandbox-side reachability probe (manyforge-composer policy)"
# 2026-06-03 (REVISED): the probe MUST use the same proxy env that the
# MCP bridge will see — HTTP_PROXY/NO_PROXY etc. — or it can pass while
# the MCP bridge silently fails with 403. The other LLM's review
# identified this exact masking failure: my previous "retry then
# continue" version probed with default env, succeeded against direct
# 127.0.0.1 access, and let the launcher proceed while the MCP bridge
# in the gateway then failed to register the manyforge catalog at all.
# We now (a) inject the EXACT MCP env, (b) keep the retry-with-backoff
# for transient policy-reload windows, and (c) FAIL HARD on persistent
# 403 because there is no recovery for it past this point — the model
# will see an empty catalog and the smoke is invalid.
set +e
PROBE_PY="
import os, urllib.request, sys
os.environ['HTTP_PROXY']=\"${HTTP_PROXY_VAL}\"
os.environ['HTTPS_PROXY']=\"${HTTP_PROXY_VAL}\"
os.environ['NO_PROXY']=\"${NO_PROXY_VAL}\"
os.environ['http_proxy']=\"${HTTP_PROXY_VAL}\"
os.environ['https_proxy']=\"${HTTP_PROXY_VAL}\"
os.environ['no_proxy']=\"${NO_PROXY_VAL}\"
try:
    r = urllib.request.urlopen(\"${MODE_URL}\", timeout=5)
    print(r.read(400).decode('utf-8', 'replace'))
    sys.exit(0)
except Exception as e:
    print(f'PROBE FAILED: {type(e).__name__}: {e}', file=sys.stderr)
    sys.exit(2)
"
PROBE_PY_B64="$(printf '%s' "${PROBE_PY}" | base64 | tr -d '\n')"
for attempt in 1 2 3 4 5; do
    probe_out="$("${KEX_USER[@]}" "printf %s '${PROBE_PY_B64}' | base64 -d | python3 -" 2>&1)"
    probe_rc=$?
    if [[ ${probe_rc} -eq 0 ]]; then
        echo "${probe_out}" | sed 's/^/    /'
        break
    fi
    if [[ ${attempt} -eq 5 ]]; then
        echo "  ✗ sandbox reachability probe failed after 5 attempts (using MCP env)." >&2
        echo "${probe_out}" | tail -10 | sed 's/^/    /' >&2
        echo "    This means the manyforge MCP bridge will not register its catalog."
        echo "    Smoke results past this point WILL be invalid (model sees only OpenClaw core tools)."
        echo "    Most common cause: HTTP_PROXY forces traffic through OpenShell's egress proxy"
        echo "    which denies host.openshell.internal:9000 by policy. Keep host.openshell.internal"
        echo "    OUT of NO_PROXY; the MCP bridge must use the OpenShell egress proxy."
        echo "    Verify the active policy contains port 9000 and the bridge Python path"
        echo "    (/usr/bin/python3.13 on the current sandbox image). Current values:"
        echo "      HTTP_PROXY=${HTTP_PROXY_VAL}"
        echo "      NO_PROXY=${NO_PROXY_VAL}"
        exit 1
    fi
    backoff=$((2 ** attempt))
    echo "    attempt ${attempt}/5 failed (rc=${probe_rc}); retrying in ${backoff}s..."
    sleep ${backoff}
done
set -e

cat <<EOF

Setup complete.

Next steps:
  - Verify the agent sees the manyforge MCP tools:
      nemoclaw ${SANDBOX} exec --no-tty -- \\
        openclaw agent --agent manyforge-composer --message 'List the manyforge MCP tools you can call. Reply with a JSON array of tool names.' --json --timeout 120
  - Composer is now wired to use the openclaw lane by default
    (demo-assistant-known-good.sh ASSISTANT_PROVIDER=openclaw). Run the
    launcher's 'start' or 'restart' to bring the openclaw bridge
    on :8200 up against this provisioned sandbox.
EOF
