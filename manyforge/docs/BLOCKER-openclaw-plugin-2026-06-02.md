# Blocker: OpenClaw direct-tools plugin path abandoned (2026-06-02)

## What we tried
Built an OpenClaw provider plugin (`nemoclaw-manyforge-direct-tools`) intended
to override the OpenClaw 2026.5.6+ tool-search compaction shim by registering
the manyforge MCP catalog directly under the `inference` provider via
`api.registerProvider({id:"inference", normalizeToolSchemas})`.

## Why it didn't work
The bundled NemoClaw `nemoclaw` plugin
([extensions/nemoclaw/dist/index.js:178](file:///sandbox/.openclaw/extensions/nemoclaw/dist/index.js#L178))
unconditionally registers `id:"inference"` with the real provider config
(model, baseUrl, credentialEnv). Our plugin's second `registerProvider` is
rejected by OpenClaw with:

    [plugins] provider already registered: inference (nemoclaw)
    (plugin=nemoclaw-manyforge-direct-tools, ...)

OpenClaw 2026.5.22 exposes no `api.extendProvider(id, hooks)` or equivalent.
The `kimi-inference-compat` precedent uses the same `registerProvider` pattern
and would face the identical collision; we found no NemoClaw-side gating that
suppresses the bundled plugin when a compat plugin is enabled. The plugin path
is therefore a dead end on the current OpenClaw API surface.

## The pivot
[THREE-LANE-MIGRATION-PLAN.md](./THREE-LANE-MIGRATION-PLAN.md) §5.2 and §8
Phase 3 instead embrace the OpenClaw discovery surface as intended: the
manyforge skill is rewritten to teach the model the
`tool_search → tool_describe → tool_call` protocol. The shim's overhead
(~3 LLM round-trips per first-use tool, collapsing to 2 when the model
already knows the tool name) is accepted as the cost of using OpenClaw the
way it expects to be used.

## Archived artifacts
Located at `manyforge/archive/openclaw-plugin-attempt-2026-06-02/`:

- `openclaw-overrides/` — per-model JSON manifests that drove `apply-openclaw-overrides.sh`
- `openclaw-plugins/manyforge-direct-tools/` — the plugin source (`index.js`, `openclaw.plugin.json`)
- `openclaw-plugins/Dockerfile.manyforge-sandbox{,-prebuilt}` — sandbox image build that baked the plugin in
- `apply-openclaw-overrides.sh` — runtime patcher for `/sandbox/.openclaw/openclaw.json`
- `build-manyforge-sandbox-image.sh` — image build orchestrator

These remain reachable for forensics and as a feature-flagged rollback path
(`OPENCLAW_LANE_MODE=plugin|native`) if Phase 3's native-discovery result
falls short of the iter-32 baseline by more than ~5 cases. Final deletion
is gated by Phase 5 of the THREE-LANE plan.

## Upstream paths that would obviate this
1. OpenClaw exposes `api.extendProvider(id, hooks)` (filed as a feature
   request would resolve every per-model overlay including Kimi's).
2. NemoClaw's `generate-openclaw-config.mts` emits
   `plugins.entries.nemoclaw = {enabled: false}` when an
   `effects.openclawPlugins[*]` entry declares `providers: ["inference"]`.
3. OpenClaw's `tools.toolSearch.enabled = false` config knob actually
   takes effect (currently silently ignored on 2026.5.6+; the relevant
   code path is `applyToolSearchCatalog` in `dist/selection-hR-AeOeU.js`).

Any of the three lets a future cycle revisit the plugin path with no
collision. Until then, the discovery surface is what we work with.
