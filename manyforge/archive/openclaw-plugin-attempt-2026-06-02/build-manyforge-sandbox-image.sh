#!/usr/bin/env bash
# build-manyforge-sandbox-image.sh
#
# Build a custom OpenClaw sandbox image with the ManyForge Direct Tools
# plugin baked in. The plugin replaces OpenClaw 2026.5.22's tool-search
# compaction surface with the full ManyForge MCP catalog for the
# managed-inference provider — see plugin source for the why.
#
# What this script does:
#   1. Validates that the host's manyforge composer is reachable
#      and serving the assistant-mode manifest.
#   2. Fetches the catalog via
#        GET http://localhost:9000/api/assistant/modes/composer-assistant
#      and writes it as catalog.json into a temp build context.
#   3. Stages the plugin source files (index.js, openclaw.plugin.json)
#      into the build context.
#   4. Runs docker build, producing the image
#        manyforge/openclaw-sandbox:plugin-direct-tools
#   5. Prints the next-step `nemoclaw onboard --from <Dockerfile>` command
#      (the script does NOT onboard — that's a separate, sandbox-state-
#      destroying step).
#
# Exit codes:
#   0  ok
#   1  prerequisites missing (docker, composer)
#   2  catalog fetch failed
#   3  docker build failed
#
# Usage:
#   ./build-manyforge-sandbox-image.sh [--tag TAG] [--mode MODE_NAME]
#     --tag        image tag (default: manyforge/openclaw-sandbox:plugin-direct-tools)
#     --mode       composer mode whose catalog to bake in
#                  (default: composer-assistant)

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
info() { echo -e "${GREEN}[build]${NC} $1"; }
warn() { echo -e "${YELLOW}[build]${NC} $1"; }
fail() { echo -e "${RED}[build]${NC} $1" >&2; exit "${2:-1}"; }

# ── argv ────────────────────────────────────────────────────────
IMAGE_TAG="manyforge/openclaw-sandbox:plugin-direct-tools"
COMPOSER_MODE="composer-assistant"
COMPOSER_BASE="${MANYFORGE_COMPOSER_BASE:-http://localhost:9000}"
# The post-NemoClaw image tag we FROM (the upstream openclaw image
# alone lacks `nemoclaw-start` and the plugin infrastructure that
# `nemoclaw onboard`'s own build step installs). Auto-detected as the
# newest `openshell/sandbox-from:<timestamp>` cached locally, override
# with --base-image.
BASE_IMAGE_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)        IMAGE_TAG="$2"; shift 2 ;;
    --mode)       COMPOSER_MODE="$2"; shift 2 ;;
    --composer)   COMPOSER_BASE="$2"; shift 2 ;;
    --base-image) BASE_IMAGE_OVERRIDE="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,35p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) fail "unknown arg: $1" ;;
  esac
done

# Pick the newest cached openshell/sandbox-from tag unless overridden.
if [[ -n "${BASE_IMAGE_OVERRIDE}" ]]; then
  BASE_IMAGE="${BASE_IMAGE_OVERRIDE}"
else
  BASE_IMAGE="$(docker images openshell/sandbox-from --format '{{.Repository}}:{{.Tag}}' | head -1)"
  [[ -n "${BASE_IMAGE}" ]] || fail \
    "no openshell/sandbox-from:<tag> image cached locally. Run a baseline 'nemoclaw onboard' (without --from) once to produce one, then re-run this script." 1
fi
info "Base image: ${BASE_IMAGE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLUGIN_SRC_DIR="${REPO_ROOT}/openclaw-plugins/manyforge-direct-tools"
DOCKERFILE="${REPO_ROOT}/openclaw-plugins/Dockerfile.manyforge-sandbox"

[[ -d "${PLUGIN_SRC_DIR}" ]] || fail "plugin source missing: ${PLUGIN_SRC_DIR}" 1
[[ -f "${DOCKERFILE}" ]] || fail "Dockerfile missing: ${DOCKERFILE}" 1

# ── 1. prereqs ─────────────────────────────────────────────────
info "Step 1/4  prerequisites"
command -v docker >/dev/null || fail "docker not installed" 1
docker info >/dev/null 2>&1 || fail "docker daemon not running" 1
info "  ✓ docker available"

CATALOG_URL="${COMPOSER_BASE}/api/assistant/modes/${COMPOSER_MODE}"
if ! curl -fsS -m 5 "${CATALOG_URL}" >/dev/null 2>&1; then
  fail "composer not serving ${CATALOG_URL}. Start the composer first." 1
fi
info "  ✓ composer mode '${COMPOSER_MODE}' reachable at ${CATALOG_URL}"

# ── 2. fetch catalog ────────────────────────────────────────────
info "Step 2/4  fetching catalog from composer"
STAGE_DIR="$(mktemp -d -t manyforge-sandbox-build-XXXX)"
trap 'rm -rf "${STAGE_DIR}"' EXIT
mkdir -p "${STAGE_DIR}/manyforge-direct-tools"

if ! curl -fsS -m 10 "${CATALOG_URL}" > "${STAGE_DIR}/manyforge-direct-tools/catalog.json"; then
  fail "catalog fetch failed; composer returned non-2xx" 2
fi

# Sanity check the body. Composer's mode manifest is JSON like
# {mode, catalogHash, tools: [...]}. We need at least one tool.
TOOL_COUNT="$(python3 -c "
import json, sys, pathlib
try:
    d = json.loads(pathlib.Path('${STAGE_DIR}/manyforge-direct-tools/catalog.json').read_text())
    tools = d.get('tools') if isinstance(d, dict) else (d if isinstance(d, list) else None)
    if not isinstance(tools, list) or not tools:
        print('0'); sys.exit(0)
    print(len(tools))
except Exception as e:
    print('0'); sys.exit(0)
")"
[[ "${TOOL_COUNT}" -gt 0 ]] || fail "fetched catalog has no tools" 2
info "  ✓ catalog captured (${TOOL_COUNT} tools)"

# ── 3. stage plugin sources ─────────────────────────────────────
info "Step 3/4  staging plugin sources"
cp "${PLUGIN_SRC_DIR}/index.js"             "${STAGE_DIR}/manyforge-direct-tools/index.js"
cp "${PLUGIN_SRC_DIR}/openclaw.plugin.json" "${STAGE_DIR}/manyforge-direct-tools/openclaw.plugin.json"
cp "${DOCKERFILE}"                          "${STAGE_DIR}/Dockerfile"
info "  ✓ staged 3 files + Dockerfile under ${STAGE_DIR}"

# ── 4. docker build ─────────────────────────────────────────────
info "Step 4/4  docker build → ${IMAGE_TAG}"
BUILD_LOG="/tmp/build-manyforge-sandbox-image.log"
if ! docker build \
        --tag "${IMAGE_TAG}" \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        --file "${STAGE_DIR}/Dockerfile" \
        "${STAGE_DIR}" \
        > "${BUILD_LOG}" 2>&1; then
  warn "  build failed; tail of ${BUILD_LOG}:"
  tail -25 "${BUILD_LOG}" >&2
  fail "  docker build exit non-zero" 3
fi
tail -3 "${BUILD_LOG}" | sed 's/^/    /'
info "  ✓ image built: ${IMAGE_TAG}"

cat <<EOF

═══════════════════════════════════════════════════════════════════════
Image built: ${IMAGE_TAG}  (${TOOL_COUNT} ManyForge tools baked in)
═══════════════════════════════════════════════════════════════════════

Next step — onboard a fresh sandbox with this image (this destroys the
current sandbox state — re-run setup-manyforge-assistant.sh +
apply-openclaw-overrides.sh afterward):

  # 1. Tag the image so nemoclaw onboard --from will pick it up. nemoclaw
  #    expects a Dockerfile path, not an image tag, so we re-export the
  #    Dockerfile path pointing at the same content we just built.
  cd ${REPO_ROOT}
  NEMOCLAW_PROVIDER=custom \\
  NEMOCLAW_ENDPOINT_URL=http://127.0.0.1:8000/v1 \\
  NEMOCLAW_MODEL=cosmos-reason2-8b \\
  NEMOCLAW_PROVIDER_KEY=dummy \\
  NEMOCLAW_POLICY_TIER=restricted \\
  NEMOCLAW_POLICY_PRESETS=local-inference \\
  nemoclaw onboard \\
    --non-interactive --yes --fresh \\
    --name my-assistant \\
    --recreate-sandbox \\
    --from "${REPO_ROOT}/manyforge/openclaw-plugins/Dockerfile.manyforge-sandbox-prebuilt" \\
    --yes-i-accept-third-party-software

  # 2. Re-apply the manyforge layer (skill, MCP server, policy)
  ./manyforge/setup-manyforge-assistant.sh my-assistant

  # 3. Apply the model-specific-setup overrides (including the new
  #    plugin entry — see openclaw-overrides/*.json)
  ./manyforge/scripts/apply-openclaw-overrides.sh my-assistant

  # 4. Restart the gateway so OpenClaw loads the plugin
  nemoclaw my-assistant recover

  # 5. Smoke through the OpenClaw lane (port 8200)
  cd ~/workspaces/dev_ws/src/manyforge
  ASSISTANT_PROVIDER=openclaw \\
  OPENCLAW_ASSISTANT_AGENT=manyforge-composer \\
  START_VLLM_PROXY=false \\
  ./scripts/launch.sh restart --lane manyforge-only --assistant on \\
    --scenario ur10e-scene-authoring --non-interactive --yes

  # 6. Verify tools[] sent to vLLM contains real manyforge tools, not the shim:
  python3 -u manyforge/scripts/debug/smoke_corpus_runner.py --filter P1_wrap_root_specific
  tail -1 /tmp/manyforge-assistant-e2e/vllm-proxy.jsonl | python3 -c "
import sys, json
b = json.loads(sys.stdin.read())['request'].get('body', {})
names = [t.get('function',{}).get('name','?') for t in b.get('tools',[])]
print('first tool names:', names[:8])
"

═══════════════════════════════════════════════════════════════════════
EOF
