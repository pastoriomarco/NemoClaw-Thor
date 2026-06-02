// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// NemoClaw ManyForge Direct Tools — OpenClaw provider plugin.
//
// OpenClaw 2026.5.22 unconditionally wraps every chat-completion's tool
// catalog behind a discovery surface (`tool_search_code` in code mode,
// `tool_search`/`tool_describe`/`tool_call` in tools mode). For
// schema-rich, small catalogs like ManyForge's (~25 manyforge MCP tools
// the model already knows by name from the skill prompt), the discovery
// shim destroys accuracy: the model has to navigate two indirections
// before reaching the actual tool and rarely picks the right argument
// shape from the description alone.
//
// This plugin mirrors the pattern NVIDIA shipped in PR #4120 for Kimi K2.6
// (the `kimi-inference-compat` plugin). It hooks the provider's
// `normalizeToolSchemas` callback and — when the request is on our
// managed-inference provider AND the model-facing tools[] currently
// contains the discovery shim — REPLACES the shim entries with the
// real ManyForge catalog.
//
// The catalog itself is captured at sandbox-image build time (the
// companion `Dockerfile.manyforge-sandbox` extends the upstream
// openclaw sandbox image and COPYs a `catalog.json` next to this file).
// Baked-in means no runtime network dependency, deterministic behavior,
// and graceful no-op if the catalog file is missing (we fall back to
// OpenClaw's stock behavior).
//
// Loading: the plugin is referenced by every model-specific-setup
// manifest under `manyforge/openclaw-overrides/*.json` via
// `effects.openclawPlugins`. NemoClaw's generate-openclaw-config.mts
// resolves the manifests at sandbox image build time and writes a
// matching `plugins.entries[<id>] = {enabled: true}` block plus a
// loadPath into the generated openclaw.json.
//
// Disable: set `effects.openclawPlugins` to `[]` in the matching
// model-specific-setup JSON, or remove the JSON, then rebuild.

const fs = require("fs");
const path = require("path");

const PLUGIN_ID = "nemoclaw-manyforge-direct-tools";

// Catalog baked at image build time. Single source: composer's
// `/api/assistant/modes/composer-assistant` response captured during
// the Dockerfile build step (see scripts/build-manyforge-sandbox-image.sh).
// If the file is missing or malformed, the plugin no-ops and the
// upstream OpenClaw discovery surface is used unchanged.
const CATALOG_PATH = path.join(__dirname, "catalog.json");

let CATALOG_TOOLS = null;
let CATALOG_LOAD_ERROR = null;
try {
  const raw = fs.readFileSync(CATALOG_PATH, "utf-8");
  const doc = JSON.parse(raw);
  // Accept either {tools: [...]} (composer's mode-manifest shape) or a
  // bare array of tool descriptors.
  const tools = Array.isArray(doc) ? doc : doc && Array.isArray(doc.tools) ? doc.tools : null;
  if (!Array.isArray(tools) || tools.length === 0) {
    throw new Error("catalog.json missing tools[] array");
  }
  CATALOG_TOOLS = tools;
} catch (err) {
  CATALOG_LOAD_ERROR = err && err.message ? err.message : String(err);
  // Continue without a catalog — the plugin will no-op and OpenClaw's
  // upstream behavior is preserved.
}

// Compaction-surface control tool names that OpenClaw injects in
// place of the real catalog when toolSearch is on. We strip these
// before adding our own entries.
const COMPACTION_SHIM_NAMES = new Set([
  "tool_search_code",
  "tool_search",
  "tool_describe",
  "tool_call",
]);

// Convert a manyforge MCP tool descriptor (as composer's mode-manifest
// emits it) into an OpenAI-compatible function-tool schema.
// Composer descriptor shape (`api/assistant/modes/composer-assistant`):
//   {id, description, effect, inputSchema: { ... JSON Schema ... }}
function toFunctionSchema(toolDesc) {
  if (!toolDesc || typeof toolDesc !== "object") return null;
  const id = String(toolDesc.id || "").trim();
  if (!id) return null;
  const description = typeof toolDesc.description === "string" ? toolDesc.description : "";
  const inputSchema =
    toolDesc.inputSchema && typeof toolDesc.inputSchema === "object"
      ? toolDesc.inputSchema
      : { type: "object", properties: {}, additionalProperties: false };
  return {
    type: "function",
    function: {
      // OpenClaw's MCP runtime resolves bare ManyForge tool ids back to
      // the manyforge MCP server. We emit bare ids (no namespace
      // prefix); the proxy's `normalize_tool_names` mutation handles
      // any required `manyforge__` rewriting on the response side.
      name: id,
      description,
      parameters: inputSchema,
    },
  };
}

function isCompactionShim(tool) {
  const name = tool && tool.function && tool.function.name;
  return typeof name === "string" && COMPACTION_SHIM_NAMES.has(name);
}

function isManagedInference(ctx) {
  if (!ctx || typeof ctx !== "object") return false;
  const provider = typeof ctx.provider === "string" ? ctx.provider.trim().toLowerCase() : "";
  // Match the same providerKey we use in our model-specific-setup
  // manifests. NemoClaw's onboard puts our managed inference route
  // under the "inference" provider id.
  return provider === "inference";
}

// Build the catalog as function-tool schemas exactly once, at module load.
const CATALOG_FUNCTION_TOOLS = CATALOG_TOOLS
  ? CATALOG_TOOLS.map(toFunctionSchema).filter(Boolean)
  : [];

module.exports = {
  id: PLUGIN_ID,
  name: "NemoClaw ManyForge Direct Tools",
  version: "0.1.0",
  description:
    "Replaces OpenClaw's tool-search compaction surface with the full ManyForge MCP catalog for the managed-inference provider.",
  register(api) {
    api.registerProvider({
      id: "inference",
      label: "NemoClaw managed inference",
      auth: [],
      normalizeToolSchemas(ctx) {
        // Bail out if not our provider or catalog isn't loaded.
        if (!isManagedInference(ctx)) return null;
        if (!CATALOG_FUNCTION_TOOLS.length) return null;

        const tools = Array.isArray(ctx.tools) ? ctx.tools : [];
        // Only act when the compaction shim is present. If OpenClaw
        // already passed direct tools (e.g. future fix that respects
        // tools.toolSearch = false), no-op.
        if (!tools.some(isCompactionShim)) return null;

        // Strip shim entries and append the catalog. Preserve any
        // non-shim tools (e.g. agent-runtime control tools that
        // OpenClaw may have added alongside).
        const passthrough = tools.filter((t) => !isCompactionShim(t));
        // De-dupe by function.name: if any catalog tool happens to be
        // in passthrough already, prefer the passthrough version
        // (OpenClaw's annotations may differ from ours).
        const seen = new Set();
        const out = [];
        for (const t of passthrough) {
          const n = t && t.function && t.function.name;
          if (typeof n === "string" && !seen.has(n)) {
            seen.add(n);
            out.push(t);
          }
        }
        for (const t of CATALOG_FUNCTION_TOOLS) {
          const n = t && t.function && t.function.name;
          if (typeof n === "string" && !seen.has(n)) {
            seen.add(n);
            out.push(t);
          }
        }
        return out;
      },
      // No stream wrapping needed: the model is producing direct tool
      // calls that OpenClaw's MCP runtime already knows how to
      // dispatch. The proxy's `normalize_tool_names` and
      // `unwrap_tool_call_args` mutations handle remaining shape fixups
      // on the response side.
      inspectToolSchemas() {
        // Optional hook used by `openclaw doctor` to verify the
        // schemas reaching the model. We surface a small marker so the
        // operator can confirm the plugin ran.
        return [
          {
            severity: "info",
            code: "manyforge-direct-tools.active",
            message: CATALOG_TOOLS
              ? `manyforge-direct-tools: substituting ${CATALOG_FUNCTION_TOOLS.length} tools for the compaction shim`
              : `manyforge-direct-tools: catalog not loaded (${CATALOG_LOAD_ERROR || "missing"}); plugin no-op`,
          },
        ];
      },
    });
  },
  __testing: {
    PLUGIN_ID,
    COMPACTION_SHIM_NAMES,
    toFunctionSchema,
    isCompactionShim,
    isManagedInference,
    CATALOG_TOOLS,
    CATALOG_FUNCTION_TOOLS,
    CATALOG_LOAD_ERROR,
  },
};
