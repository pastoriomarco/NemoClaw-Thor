#!/usr/bin/env bash
# apply-openclaw-overrides.sh
#
# Apply NemoClaw-style "model-specific-setup" overrides to an already-onboarded
# OpenClaw sandbox WITHOUT forking NemoClaw or rebuilding the sandbox image.
#
# Why this exists: OpenClaw 2026.5.22 routes every chat-completion through a
# `tool_search_code` discovery shim that hides MCP tools behind a code-mode
# bridge. For manyforge's small, schema-rich catalog this kills accuracy.
# NemoClaw's official escape hatch is a JSON file in
# `nemoclaw-blueprint/model-specific-setup/openclaw/<id>.json` with
# `effects.openclawTools.toolSearch: false` (see PR #4120, the Kimi K2.6 case).
# At sandbox image-build time NVIDIA's generator bakes the matching override
# into `openclaw.json`'s `tools.toolSearch: false`.
#
# We can't change the upstream image. But we CAN reach the same end state at
# runtime: the in-sandbox `openclaw.json` is sandbox-mutable by default
# (NemoClaw PR #2227 — "shields-down" state), guarded only by a hash file
# the gateway re-verifies at startup. This script:
#
#   1. Reads the JSON manifest(s) under manyforge/openclaw-overrides/
#      that match the active sandbox's model.
#   2. Patches `/sandbox/.openclaw/openclaw.json` with the resolved
#      `tools.toolSearch` value (and any `openclawCompat` extras).
#   3. Refreshes `/sandbox/.openclaw/.config-hash` so the gateway's
#      integrity check passes on next start.
#   4. Restarts the OpenClaw gateway in-sandbox so it re-reads the
#      patched config.
#
# Idempotent. Re-runnable. Verifies + exits cleanly when the config is
# already at the desired state.
#
# Usage:
#   ./apply-openclaw-overrides.sh [SANDBOX] [--dry-run] [--no-restart]
#     SANDBOX       default my-assistant
#     --dry-run     print the planned patch but don't apply
#     --no-restart  skip the gateway restart (you'll restart manually)
#
# After applying:
#   - Re-run a smoke case and confirm vLLM's request body shows the actual
#     manyforge tool catalog, not just `tool_search_code`.

set -euo pipefail

# ── argv ────────────────────────────────────────────────────────
SANDBOX="my-assistant"
DRY_RUN=0
RESTART=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)    DRY_RUN=1; shift ;;
    --no-restart) RESTART=0; shift ;;
    -h|--help)
      sed -n '1,40p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)            SANDBOX="$1"; shift ;;
  esac
done

# ── paths ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OVERRIDES_DIR="${REPO_ROOT}/openclaw-overrides"
[[ -d "${OVERRIDES_DIR}" ]] || { echo "missing overrides dir: ${OVERRIDES_DIR}" >&2; exit 1; }

OVERRIDE_FILES=( "${OVERRIDES_DIR}"/*.json )
[[ -e "${OVERRIDE_FILES[0]}" ]] || { echo "no override JSONs in ${OVERRIDES_DIR}" >&2; exit 1; }

# Stage all override JSONs into a tarball so we ship them to the sandbox
# in one upload.
STAGE_DIR="$(mktemp -d -t openclaw-overrides-XXXX)"
trap 'rm -rf "${STAGE_DIR}"' EXIT
mkdir -p "${STAGE_DIR}/openclaw-overrides"
cp "${OVERRIDE_FILES[@]}" "${STAGE_DIR}/openclaw-overrides/"

# ── in-sandbox patcher (uploaded + executed) ────────────────────
# Reads the staged JSONs, finds the one matching the sandbox's current
# inference config, applies `effects.openclawTools.toolSearch` and
# `effects.openclawCompat` to /sandbox/.openclaw/openclaw.json,
# refreshes the integrity hash.
cat > "${STAGE_DIR}/patch-in-sandbox.py" <<'PY'
#!/usr/bin/env python3
"""Apply the matching model-specific-setup override to openclaw.json."""
from __future__ import annotations
import hashlib, json, os, pathlib, sys

OVERRIDES_DIR = pathlib.Path("/tmp/openclaw-overrides")
OPENCLAW_JSON = pathlib.Path("/sandbox/.openclaw/openclaw.json")
HASH_FILE = pathlib.Path("/sandbox/.openclaw/.config-hash")

def load_active_model_context(config: dict) -> dict:
    """Pull the fields generate-openclaw-config.mts uses to match a setup."""
    providers = config.get("models", {}).get("providers", {})
    primary_key = config.get("models", {}).get("primary") or "inference"
    provider = providers.get(primary_key, {})
    models = provider.get("models") or []
    model_id = ""
    if models and isinstance(models, list):
        first = models[0]
        if isinstance(first, dict):
            model_id = first.get("id", "")
    return {
        "model": model_id,
        "providerKey": primary_key,
        "baseUrl": provider.get("baseUrl", ""),
        "inferenceApi": provider.get("api", ""),
    }

def manifest_matches(manifest: dict, ctx: dict) -> bool:
    match = manifest.get("match", {})
    model_ids = match.get("modelIds")
    if model_ids and ctx["model"].lower() not in {m.lower() for m in model_ids}:
        return False
    for key in ("providerKey", "baseUrl", "inferenceApi"):
        wanted = match.get(key)
        if wanted and str(ctx.get(key, "")).rstrip("/") != str(wanted).rstrip("/"):
            return False
    return True

def apply_effects(config: dict, manifest: dict) -> bool:
    """Apply effects.openclawTools + openclawCompat + openclawPlugins. Returns True if changed."""
    effects = manifest.get("effects", {})
    changed = False

    tools_effects = effects.get("openclawTools") or {}
    if "toolSearch" in tools_effects:
        tools = config.setdefault("tools", {})
        new_val = bool(tools_effects["toolSearch"])
        if tools.get("toolSearch") != new_val:
            tools["toolSearch"] = new_val
            changed = True

    compat_effects = effects.get("openclawCompat") or {}
    if compat_effects:
        models = config.setdefault("models", {})
        providers = models.setdefault("providers", {})
        primary = models.get("primary") or "inference"
        provider = providers.setdefault(primary, {})
        compat = provider.setdefault("compat", {})
        for k, v in compat_effects.items():
            if compat.get(k) != v:
                compat[k] = v
                changed = True

    # 2026-06-02: openclawPlugins entries.
    # Matches the shape NemoClaw's generate-openclaw-config.mts writes
    # at sandbox-image build time (see PR #4120 for the Kimi precedent).
    # NemoClaw's runtime normalization reads:
    #   plugins.entries[<plugin-id>] = {enabled: <bool>}
    #   plugins.load.paths = [<absolute install paths>]
    # Both must be set or the plugin won't load even if its source is on
    # disk. The install path is the manifest's `loadPath` field; the
    # sandbox image must have the actual `.js` files there
    # (see manyforge/openclaw-plugins/Dockerfile.manyforge-sandbox).
    plugin_effects = effects.get("openclawPlugins") or []
    if plugin_effects:
        plugins = config.setdefault("plugins", {})
        entries = plugins.setdefault("entries", {})
        load = plugins.setdefault("load", {})
        paths = load.setdefault("paths", [])
        for plug in plugin_effects:
            if not isinstance(plug, dict):
                continue
            plug_id = plug.get("id")
            plug_load = plug.get("loadPath")
            if not (isinstance(plug_id, str) and plug_id.strip()):
                continue
            if not (isinstance(plug_load, str) and plug_load.strip()):
                continue
            # entries
            cur = entries.get(plug_id)
            if not (isinstance(cur, dict) and cur.get("enabled") is True):
                entries[plug_id] = {"enabled": True}
                changed = True
            # load.paths
            if plug_load not in paths:
                paths.append(plug_load)
                changed = True
    return changed

def main() -> int:
    if not OPENCLAW_JSON.exists():
        print(f"error: {OPENCLAW_JSON} not found", file=sys.stderr)
        return 2
    config = json.loads(OPENCLAW_JSON.read_text())
    ctx = load_active_model_context(config)
    print(f"active model context: {ctx}")

    manifests = []
    for p in sorted(OVERRIDES_DIR.glob("*.json")):
        try:
            m = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"  ! skipping malformed {p.name}: {e}", file=sys.stderr)
            continue
        if m.get("agent") != "openclaw":
            continue
        if manifest_matches(m, ctx):
            manifests.append((p.name, m))

    if not manifests:
        print("no matching override found for this sandbox — nothing to do")
        return 0

    if len(manifests) > 1:
        names = ", ".join(name for name, _ in manifests)
        print(f"error: multiple overrides matched: {names}", file=sys.stderr)
        return 3

    name, manifest = manifests[0]
    print(f"applying override: {name}")
    changed = apply_effects(config, manifest)
    if not changed:
        print("config already matches override — no write needed")
        return 0

    if os.environ.get("DRY_RUN") == "1":
        print("--- DRY RUN: would write the following ---")
        print(json.dumps(config.get("tools", {}), indent=2))
        return 0

    # Write back. Indent matches generate-openclaw-config.mts (2-space).
    OPENCLAW_JSON.write_text(json.dumps(config, indent=2))
    # Refresh hash. Format mirrors `sha256sum openclaw.json`:
    #   "<hex>  openclaw.json\n"
    h = hashlib.sha256(OPENCLAW_JSON.read_bytes()).hexdigest()
    HASH_FILE.write_text(f"{h}  openclaw.json\n")
    print("config patched + hash refreshed")
    return 0

sys.exit(main())
PY

# Pack into a tarball (single upload roundtrip).
TARBALL="${STAGE_DIR}/payload.tar"
tar -C "${STAGE_DIR}" -cf "${TARBALL}" openclaw-overrides patch-in-sandbox.py

# ── upload + run ────────────────────────────────────────────────
echo "[1/4] uploading ${OVERRIDES_DIR##*/} ($(ls "${OVERRIDES_DIR}" | wc -l) files) to sandbox '${SANDBOX}'…"
openshell sandbox upload "${SANDBOX}" "${TARBALL}" /tmp/ >/dev/null

# OpenShell's exec endpoint rejects argv with newlines, so we run each step
# as a single command rather than a multi-line bash -c. (Same constraint
# the bridge adapter base64-wraps around.)
TARBASENAME="$(basename "${TARBALL}")"
echo "[2/4] unpacking + applying patch (dry-run=${DRY_RUN})…"
PATCH_LOG="/tmp/apply-openclaw-overrides.${SANDBOX}.log"
nemoclaw "${SANDBOX}" exec --no-tty -- mkdir -p /tmp/openclaw-overrides >/dev/null
nemoclaw "${SANDBOX}" exec --no-tty -- tar -xf "/tmp/${TARBASENAME}" -C /tmp/ >/dev/null
nemoclaw "${SANDBOX}" exec --no-tty -- rm -f "/tmp/${TARBASENAME}" >/dev/null
nemoclaw "${SANDBOX}" exec --no-tty -- env "DRY_RUN=${DRY_RUN}" python3 /tmp/patch-in-sandbox.py 2>&1 | tee "${PATCH_LOG}"

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "[dry-run] no changes applied. Log: ${PATCH_LOG}"
  exit 0
fi

# ── restart gateway ─────────────────────────────────────────────
if [[ ${RESTART} -eq 0 ]]; then
  echo "[3/4] --no-restart: skipping gateway restart"
  echo "  -> restart manually with: nemoclaw ${SANDBOX} recover"
  exit 0
fi

echo "[3/4] restarting OpenClaw gateway in '${SANDBOX}' so the new config is picked up…"
nemoclaw "${SANDBOX}" recover 2>&1 | tail -15

# ── verify ──────────────────────────────────────────────────────
echo "[4/4] verifying tools.toolSearch in the live config…"
# OpenShell exec rejects newlines, so we use jq inline (single line).
nemoclaw "${SANDBOX}" exec --no-tty -- jq -r '.tools.toolSearch' /sandbox/.openclaw/openclaw.json > /tmp/_ts.${SANDBOX}.txt
TS_VALUE="$(tr -d '[:space:]' < /tmp/_ts.${SANDBOX}.txt)"
rm -f /tmp/_ts.${SANDBOX}.txt
echo "  tools.toolSearch = ${TS_VALUE}"
if [[ "${TS_VALUE}" != "false" ]]; then
  echo "WARN: tools.toolSearch is not False; the patch may not have applied" >&2
  exit 4
fi

echo
echo "✓ done. To confirm direct tool exposure, run a smoke case and inspect"
echo "  /tmp/manyforge-assistant-e2e/vllm-proxy.jsonl — the latest request body"
echo "  should contain the manyforge MCP tools by name, NOT just 'tool_search_code'."
