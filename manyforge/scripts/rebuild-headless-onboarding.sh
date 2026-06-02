#!/usr/bin/env bash
# rebuild-headless-onboarding.sh
#
# Reproduce the manyforge stack from scratch on a host that already has:
#   - Docker + NVIDIA runtime
#   - NemoClaw `lkg` host CLI (= v0.0.55; NVIDIA's "last known good" alias).
#     Install: cd ~/NemoClaw && git fetch --tags && git checkout lkg && npm install -g
#     `nemoclaw --version` → 0.0.55. v0.0.56 also works (same OpenClaw 2026.5.22
#     + same sandbox digest), but lkg is the public installer default.
#   - OpenShell 0.0.44 host CLI (`openshell --version` → 0.0.44, auto-installed
#     by NemoClaw above)
#   - vLLM image `nemoclaw-thor/vllm:latest` (≥ v9.1)
#   - The manyforge composer container `manyforge-e2e-composer` running on :9000
#
# This is the headless equivalent of:
#   1. `nemoclaw onboard` (interactive wizard)
#   2. `./manyforge/setup-manyforge-assistant.sh my-assistant`
#   3. The handful of openclaw.json edits OpenClaw 2026.5.22 requires.
#
# Exit codes:
#   0  ok, ready for smoke
#   1  prerequisites missing
#   2  onboarding failed (the wizard would have done the same)
#   3  policy / skill / config patch failed
#
# Usage:
#   ./rebuild-headless-onboarding.sh [SANDBOX]
#     SANDBOX  defaults to my-assistant
#
# Env overrides (rarely needed):
#   NEMOCLAW_MODEL                 default cosmos-reason2-8b
#   THOR_VLLM_PORT                 default 8050 (proxy listens on 8000)
#   MANYFORGE_COMPOSER_PRECHECK_BASE  default http://localhost:9000
#   MANYFORGE_COMPOSER_BASE        default http://host.openshell.internal:9000
#
# After this completes:
#   - export OPENCLAW_ASSISTANT_AGENT=manyforge-composer
#   - export PYTHONPATH=<repo>/manyforge
#   - nohup .../bridge.service &
#   - python3 manyforge/scripts/debug/smoke_corpus_runner.py --filter P1_wrap_root_specific

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
info() { echo -e "${GREEN}[onboard]${NC} $1"; }
warn() { echo -e "${YELLOW}[onboard]${NC} $1"; }
fail() { echo -e "${RED}[onboard]${NC} $1" >&2; exit "${2:-1}"; }

SANDBOX="${1:-my-assistant}"
MODEL="${NEMOCLAW_MODEL:-cosmos-reason2-8b}"
THOR_VLLM_PORT="${THOR_VLLM_PORT:-8050}"
PRECHECK_BASE="${MANYFORGE_COMPOSER_PRECHECK_BASE:-http://localhost:9000}"
RUNTIME_BASE="${MANYFORGE_COMPOSER_BASE:-http://host.openshell.internal:9000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SETUP_SCRIPT="${REPO_ROOT}/manyforge/setup-manyforge-assistant.sh"

# ────────────────────────────────────────────────────────────────────────
# 1. Prerequisites
# ────────────────────────────────────────────────────────────────────────
info "Step 1/7  prerequisites"
command -v nemoclaw >/dev/null || fail "nemoclaw not installed; run NemoClaw install first" 1
command -v openshell >/dev/null || fail "openshell not installed; NemoClaw should have installed it" 1
[[ -x "${SETUP_SCRIPT}" ]] || fail "manyforge setup script missing at ${SETUP_SCRIPT}" 1

NEMOCLAW_VERSION="$(nemoclaw --version 2>&1 | head -1 | awk '{print $NF}')"
OPENSHELL_VERSION="$(openshell --version 2>&1 | awk '{print $NF}')"
info "  NemoClaw=${NEMOCLAW_VERSION}  OpenShell=${OPENSHELL_VERSION}"
# Warn (don't fail) if NemoClaw is past the lkg / v0.0.55 target: v0.0.56
# was tagged the same day with the same OpenClaw pin so it's functionally
# equivalent; newer untested versions may drift the sandbox image digest
# or change the OpenClaw inside. Set NEMOCLAW_SKIP_VERSION_WARN=1 to silence.
case "${NEMOCLAW_VERSION}" in
  v0.0.55|0.0.55|v0.0.56|0.0.56) ;;
  *)
    if [[ -z "${NEMOCLAW_SKIP_VERSION_WARN:-}" ]]; then
      warn "  NemoClaw ${NEMOCLAW_VERSION} differs from the tested baseline (v0.0.55=lkg / v0.0.56)."
      warn "  This script was validated on those versions. Newer tags may need adjustment."
    fi
    ;;
esac

# vLLM reachable on the host-side proxy port (8000)?
if ! curl -fsS -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  warn "  Proxy on :8000 not reachable; start vLLM+proxy first via:"
  warn "      THOR_VLLM_PORT=${THOR_VLLM_PORT} ./serving/start-model.sh ${MODEL}"
  fail "  Aborting." 1
fi
info "  ✓ proxy + vLLM responding on :8000"

# Composer reachable?
if ! curl -fsS -m 3 "${PRECHECK_BASE}/api/assistant/modes/composer-assistant" >/dev/null 2>&1; then
  fail "  Composer at ${PRECHECK_BASE} not reachable. Start the composer first." 1
fi
info "  ✓ composer responding at ${PRECHECK_BASE}"

# ────────────────────────────────────────────────────────────────────────
# 2. Onboarding — non-interactive
# ────────────────────────────────────────────────────────────────────────
info "Step 2/7  nemoclaw onboard (non-interactive)"

# Clear any half-written session from a previous attempt.
rm -f ~/.nemoclaw/onboard-session.json
# Reset sandboxes.json to empty so re-run is clean.
mkdir -p ~/.nemoclaw
printf '{\n  "sandboxes": {},\n  "defaultSandbox": null\n}\n' > ~/.nemoclaw/sandboxes.json

# nemoclaw onboard --non-interactive recognizes provider/model/endpoint via env.
ONBOARD_LOG="/tmp/headless-onboard.${SANDBOX}.log"
set +e
NEMOCLAW_PROVIDER=compatible-endpoint \
NEMOCLAW_ENDPOINT_URL="http://127.0.0.1:8000/v1" \
NEMOCLAW_MODEL="${MODEL}" \
nemoclaw onboard \
  --non-interactive \
  --yes \
  --fresh \
  --name "${SANDBOX}" \
  --yes-i-accept-third-party-software \
  > "${ONBOARD_LOG}" 2>&1
ONBOARD_RC=$?
set -e

# Onboarding step 4 (Setting up inference provider) often dies with
# "transport error: received corrupt message of type InvalidContentType"
# because NemoClaw registers the gateway as https+mTLS but the gateway
# is plaintext http. Detect + repair + retry.
if grep -q "InvalidContentType\|received corrupt message" "${ONBOARD_LOG}"; then
  warn "  Detected NemoClaw https-vs-http mismatch on gateway registration."
  warn "  Re-registering gateway as http://127.0.0.1:8080 (plaintext)…"
  openshell gateway remove nemoclaw >/dev/null 2>&1 || true
  openshell gateway add http://127.0.0.1:8080 --local --name nemoclaw >/dev/null
  info "  Retrying onboarding…"
  rm -f ~/.nemoclaw/onboard-session.json
  set +e
  NEMOCLAW_PROVIDER=compatible-endpoint \
  NEMOCLAW_ENDPOINT_URL="http://127.0.0.1:8000/v1" \
  NEMOCLAW_MODEL="${MODEL}" \
  nemoclaw onboard \
    --non-interactive \
    --yes \
    --fresh \
    --name "${SANDBOX}" \
    --yes-i-accept-third-party-software \
    >> "${ONBOARD_LOG}" 2>&1
  ONBOARD_RC=$?
  set -e
fi

if [[ ${ONBOARD_RC} -ne 0 ]]; then
  warn "  Onboarding exit ${ONBOARD_RC}; full log: ${ONBOARD_LOG}"
  fail "  Onboarding failed. Inspect the log and apply manual interventions per docs/REBUILD-*.md." 2
fi
info "  ✓ sandbox '${SANDBOX}' onboarded"

# ────────────────────────────────────────────────────────────────────────
# 3. setup-manyforge-assistant.sh
# ────────────────────────────────────────────────────────────────────────
info "Step 3/7  setup-manyforge-assistant.sh"
SETUP_LOG="/tmp/headless-setup.${SANDBOX}.log"
set +e
MANYFORGE_COMPOSER_PRECHECK_BASE="${PRECHECK_BASE}" \
"${SETUP_SCRIPT}" "${SANDBOX}" > "${SETUP_LOG}" 2>&1
SETUP_RC=$?
set -e
if [[ ${SETUP_RC} -ne 0 ]]; then
  warn "  Setup exit ${SETUP_RC}; full log: ${SETUP_LOG}"
  fail "  Setup script failed." 3
fi
info "  ✓ manyforge policy + skill + MCP server + agent profile installed"

# ────────────────────────────────────────────────────────────────────────
# 4. Force-reapply policy (idempotency by name, not content)
# ────────────────────────────────────────────────────────────────────────
info "Step 4/7  force-reapply policy (content may have changed)"
nemoclaw "${SANDBOX}" policy-add \
  --from-file "${REPO_ROOT}/manyforge/policies/manyforge-composer.preset.yaml" \
  --force >/dev/null
info "  ✓ policy re-applied"

# ────────────────────────────────────────────────────────────────────────
# 5. openclaw.json patches required by OpenClaw 2026.5.22
# ────────────────────────────────────────────────────────────────────────
info "Step 5/7  openclaw.json patches"
PATCH_SCRIPT="/tmp/headless-patch-openclaw.py"
cat > "${PATCH_SCRIPT}" <<'PY'
import json, pathlib, sys
p = pathlib.Path("/sandbox/.openclaw/openclaw.json")
d = json.loads(p.read_text())

# (a) agents.list[manyforge-composer].tools.profile = "full"; drop bundle-mcp.
for ag in d.get("agents", {}).get("list", []):
    if ag.get("id") == "manyforge-composer":
        ag["tools"] = {"profile": "full"}

# (b) inference baseUrl: pin to the host-side proxy alias from the sandbox.
inf = d.get("models", {}).get("providers", {}).get("inference")
if isinstance(inf, dict):
    inf["baseUrl"] = "http://host.openshell.internal:8000/v1"

p.write_text(json.dumps(d, indent=2))
print("patched")
PY
openshell sandbox upload "${SANDBOX}" "${PATCH_SCRIPT}" /tmp/ >/dev/null 2>&1
nemoclaw "${SANDBOX}" exec --no-tty -- python3 "${PATCH_SCRIPT}" >/dev/null
info "  ✓ openclaw.json patched (tools.profile=full, baseUrl=host.openshell.internal:8000)"

# ────────────────────────────────────────────────────────────────────────
# 6. openclaw doctor --fix  (caps postCompactionMaxChars ≤ 50000 if needed)
# ────────────────────────────────────────────────────────────────────────
info "Step 6/7  openclaw doctor --fix"
nemoclaw "${SANDBOX}" exec --no-tty -- openclaw doctor --fix >/dev/null 2>&1 || true
nemoclaw "${SANDBOX}" exec --no-tty -- openclaw config validate >/dev/null
info "  ✓ openclaw config valid"

# ────────────────────────────────────────────────────────────────────────
# 7. Smoke: in-sandbox reachability probe
# ────────────────────────────────────────────────────────────────────────
info "Step 7/7  in-sandbox reachability probe"
if nemoclaw "${SANDBOX}" exec --no-tty -- \
    python3 -c "import urllib.request; urllib.request.urlopen('${RUNTIME_BASE}/api/assistant/modes/composer-assistant', timeout=5).read(50)" \
    >/dev/null 2>&1; then
  info "  ✓ sandbox can reach composer at ${RUNTIME_BASE}"
else
  warn "  ! sandbox-side composer probe failed. Check that the manyforge-composer policy"
  warn "    is active and includes both 172.17.0.0/16 and 172.18.0.0/16 in allowed_ips."
fi

if nemoclaw "${SANDBOX}" exec --no-tty -- \
    python3 -c "import urllib.request,json; json.loads(urllib.request.urlopen('http://host.openshell.internal:8000/v1/models', timeout=5).read())" \
    >/dev/null 2>&1; then
  info "  ✓ sandbox can reach proxy/vLLM at host.openshell.internal:8000"
else
  warn "  ! sandbox-side vLLM probe failed. Same policy issue likely."
fi

cat <<EOF

═══════════════════════════════════════════════════════════════════════
Headless onboarding complete: ${SANDBOX}
═══════════════════════════════════════════════════════════════════════

Next steps:

  # Start bridge (env vars required)
  export PYTHONPATH=${REPO_ROOT}/manyforge
  export OPENCLAW_ASSISTANT_AGENT=manyforge-composer
  nohup ${REPO_ROOT}/manyforge/openclaw_assistant_bridge/.venv/bin/python \\
        -m openclaw_assistant_bridge.service > /tmp/bridge.log 2>&1 &

  # Verify with one smoke case
  python3 ${REPO_ROOT}/manyforge/scripts/debug/smoke_corpus_runner.py \\
          --filter P1_wrap_root_specific

  # Or full corpus
  python3 ${REPO_ROOT}/manyforge/scripts/debug/smoke_corpus_runner.py \\
          --report /tmp/smoke.${SANDBOX}.json

Notes:
  - The proxy must already be running with OPENCLAW_PROXY_FORCE_ENABLE_THINKING=on
    AND OPENCLAW_PROXY_PROMOTE_REASONING_TO_CONTENT=1 — start-model.sh handles
    enable_thinking; promote_reasoning_to_content is on by default.
  - Logs:
      onboard:     ${ONBOARD_LOG}
      setup:       ${SETUP_LOG}
      proxy:       /tmp/manyforge-assistant-e2e/vllm-proxy.jsonl
      bridge:      /tmp/bridge.log
═══════════════════════════════════════════════════════════════════════
EOF
