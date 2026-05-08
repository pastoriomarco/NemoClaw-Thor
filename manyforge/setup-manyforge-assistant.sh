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
WORKSPACE_SRC="${SCRIPT_DIR}/agent-workspace"

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

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found in PATH: $1"
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
if ! nemoclaw "${SANDBOX}" status >/dev/null 2>&1; then
  fail "Sandbox '${SANDBOX}' not found or unhealthy. Run 'nemoclaw list' to inspect."
fi
ok "sandbox ${SANDBOX} is up"

# Skill-vs-runtime compatibility gate: the SKILL.md's contract block
# declares the assistant mode it expects. Refuse to install if the
# running Composer either isn't reachable or doesn't expose that mode.
# This catches the "stale skill" failure mode at install time rather
# than at first MCP call.
PRECHECK_COMPOSER_BASE="${MANYFORGE_COMPOSER_BASE:-http://host.openshell.internal:9000}"
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

step "Step 1/5: apply egress preset 'manyforge-composer' (replaces 'local-inference')"
# Why we remove 'local-inference': OpenShell's SSRF guard rejects the
# private-IP resolution of host.openshell.internal (172.17.0.1) by default.
# The canonical workaround per OpenShell policy schema is the per-endpoint
# `allowed_ips` field. The built-in 'local-inference' preset does not set
# that field, and the SSRF engine appears to honor the first matching
# endpoint rather than the union — so leaving 'local-inference' active
# causes the persistent gateway lane (/v1/chat/completions) to fail with
# `internal error` even when our preset DOES include `allowed_ips`. Our
# 'manyforge-composer' preset is a strict superset of 'local-inference'
# (same vLLM endpoint, plus the Composer endpoint, plus `allowed_ips` on
# both), so removing 'local-inference' in favor of it loses no
# functionality. This is the configure-only fix; no openshell or nemoclaw
# upstream patches are required.
if nemoclaw "${SANDBOX}" policy-list 2>&1 | grep -qE "● .*local-inference"; then
  nemoclaw "${SANDBOX}" policy-remove local-inference --yes
  ok "removed built-in 'local-inference' preset (superseded by manyforge-composer)"
fi
if nemoclaw "${SANDBOX}" policy-list 2>&1 | grep -qE "● .*manyforge-composer"; then
  ok "preset 'manyforge-composer' already applied"
else
  nemoclaw "${SANDBOX}" policy-add --from-file "${PRESET_PATH}" --yes
  ok "preset applied"
fi

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
KEX_USER=(docker exec openshell-cluster-nemoclaw kubectl exec -n openshell "${SANDBOX}" -c agent -- su sandbox -c)
SKILL_REMOTE_DIR="/sandbox/.openclaw/skills/manyforge-composer"
# Concatenated sha256 of every staged file. Order-stable (sorted by
# filename) so we don't trip on directory-iteration nondeterminism.
staged_hash() {
  ( cd "${STAGING_DIR}" && \
      find . -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort | \
      xargs -I{} sha256sum {} ) 2>/dev/null | sha256sum | cut -d' ' -f1
}
remote_hash() {
  "${KEX_USER[@]}" "test -d ${SKILL_REMOTE_DIR} && cd ${SKILL_REMOTE_DIR} && find . -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort | xargs -I{} sha256sum {} | sha256sum | cut -d' ' -f1" 2>/dev/null \
    | tr -d '[:space:]' | tail -c 64
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
# host.openshell.internal:9000 is reachable only via OpenShell's egress
# proxy at 10.200.0.1:3128, so we MUST forward the proxy envs explicitly
# — without them urllib tries direct-connect to 172.17.0.1:9000 and
# fails with [Errno 111] Connection refused (verified 2026-05-05).
HTTP_PROXY_VAL="${HTTP_PROXY:-${http_proxy:-http://10.200.0.1:3128}}"
NO_PROXY_VAL="${NO_PROXY:-${no_proxy:-127.0.0.1,localhost,::1}}"
MCP_CONFIG_JSON=$(cat <<JSON
{"command":"python3","args":["${MCP_BRIDGE_PATH}"],"env":{"MANYFORGE_COMPOSER_BASE":"${COMPOSER_BASE}","MANYFORGE_ASSISTANT_MODE":"${ASSISTANT_MODE}","MANYFORGE_PRINCIPAL":"${MCP_PRINCIPAL}","HTTP_PROXY":"${HTTP_PROXY_VAL}","HTTPS_PROXY":"${HTTP_PROXY_VAL}","NO_PROXY":"${NO_PROXY_VAL}","http_proxy":"${HTTP_PROXY_VAL}","https_proxy":"${HTTP_PROXY_VAL}","no_proxy":"${NO_PROXY_VAL}"}}
JSON
)
"${KEX_USER[@]}" "openclaw mcp set manyforge '${MCP_CONFIG_JSON}'" >/dev/null
ok "MCP server 'manyforge' registered (mode: ${ASSISTANT_MODE}; principal: ${MCP_PRINCIPAL})"
"${KEX_USER[@]}" "openclaw mcp show manyforge" 2>&1 | sed 's/^/    /'

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
    "contextLimits": {
        "toolResultMaxChars": 100000,
        "postCompactionMaxChars": 80000,
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
"${KEX_USER[@]}" "printf %s '${PROFILE_SCRIPT_B64}' | base64 -d | python3 -" >/dev/null
ok "agent profile 'manyforge-composer' installed"
"${KEX_USER[@]}" "openclaw agents list --json 2>/dev/null | head -c 1200 || openclaw agents list" 2>&1 | sed 's/^/    /'

step "Step 5b/6: install workspace guidance file (AGENTS.md)"
# AGENTS.md is injected into every agent run via OpenClaw's standard
# workspace-file slot. It carries: vocabulary lock (no session keys),
# the output protocol, the categorical tool surface (which mirrors
# tools/list, never replaces it), and the guardrails (mangling rule,
# don't invent ids, etc.).
#
# v7 (2026-05-06): TOOLS.md was folded into AGENTS.md guardrails and
# the file is no longer installed. If the sandbox has a stale TOOLS.md
# from a prior provisioner run, we delete it explicitly so the agent
# stops reading two-source-of-truth content.
if [[ ! -d "${WORKSPACE_SRC}" ]]; then
  fail "workspace source not found at ${WORKSPACE_SRC}"
fi
WORKSPACE_DIR_REMOTE="/sandbox/.openclaw/workspace"
"${KEX_USER[@]}" "mkdir -p ${WORKSPACE_DIR_REMOTE}" >/dev/null
WS_B64="$(base64 -w0 < "${WORKSPACE_SRC}/AGENTS.md")"
"${KEX_USER[@]}" "printf %s '${WS_B64}' | base64 -d > ${WORKSPACE_DIR_REMOTE}/AGENTS.md" >/dev/null
"${KEX_USER[@]}" "rm -f ${WORKSPACE_DIR_REMOTE}/TOOLS.md" >/dev/null
ok "installed AGENTS.md into ${WORKSPACE_DIR_REMOTE} (and removed any stale TOOLS.md)"
"${KEX_USER[@]}" "ls -la ${WORKSPACE_DIR_REMOTE}/" 2>&1 | sed 's/^/    /'

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
"${KEX_USER[@]}" "printf %s '${REASONING_SCRIPT_B64}' | base64 -d | python3 -" 2>&1 | sed 's/^/    /'
ok "OpenClaw model.reasoning=true ensured on active inference model"

step "Composer reachability check (mode-scoped manifest)"
MODE_URL="${COMPOSER_BASE}/api/assistant/modes/${ASSISTANT_MODE}"
if curl -fsS -o /dev/null --max-time 3 "${MODE_URL}" 2>/dev/null; then
  ok "composer ${MODE_URL} reachable from this host"
else
  printf '  ! composer mode endpoint %s did not answer from this host (curl).\n' "${MODE_URL}"
  printf '  ! If the composer is bound to 127.0.0.1 only, it will not reach. Confirm with:\n'
  printf '  !   ss -tlnp | grep :9000\n'
  printf '  ! and rebind via demo-assistant-known-good.sh with COMPOSER_BIND_HOST=0.0.0.0.\n'
  printf '  ! Also confirm a deployment with assistant_modes is loaded — without one,\n'
  printf '  ! /api/assistant/modes/${ASSISTANT_MODE} returns 404.\n'
fi

step "Sandbox-side reachability probe (manyforge-composer policy)"
"${KEX_USER[@]}" "python3 -c 'import urllib.request; print(urllib.request.urlopen(\"${MODE_URL}\", timeout=5).read(400).decode(\"utf-8\", \"replace\"))'" 2>&1 | sed 's/^/    /'

cat <<EOF

Setup complete.

Next steps:
  - Verify the agent sees the manyforge MCP tools:
      kubectl exec -n openshell ${SANDBOX} -c agent -- su sandbox -c \\
        "openclaw agent --agent manyforge-composer --message 'List the manyforge MCP tools you can call. Reply with a JSON array of tool names.' --json --timeout 60"
  - Composer is now wired to use the openclaw lane by default
    (demo-assistant-known-good.sh ASSISTANT_PROVIDER=openclaw). Run the
    launcher's 'start' or 'restart-bridge' to bring the openclaw bridge
    on :8200 up against this provisioned sandbox.
EOF
