# Changelog

All notable changes to NemoClaw-Thor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project is versioned with [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-1.0 releases (0.x.y) signal that the public surfaces — the
`serving/start-model.sh` profile names, the `setup/configure-local-provider.sh`
flags, the OpenClaw bridge wire contract — may change between minor versions.

For the per-component pinned versions (vLLM container, FlashInfer, NemoClaw
CLI, OpenShell, OpenClaw, model profiles), see [`VERSIONS.md`](VERSIONS.md).
That table moves on external upstream releases, not on NemoClaw-Thor's
own SemVer cadence.

## [Unreleased]

Work in progress on the `dev` branch — see [`ROADMAP.md`](ROADMAP.md)
for direction.

### Added

- **Rebuild documentation suite** for the OpenShell 0.0.36 → 0.0.44 +
  OpenClaw 2026.4.24 → 2026.5.22 + NemoClaw v0.0.31 → `lkg` (=v0.0.55)
  upgrade. Consolidated report:
  - [`manyforge/docs/archive/REBUILD-2026-06-02.md`](manyforge/docs/archive/REBUILD-2026-06-02.md)
    — exact rebuild procedure (12 fixes catalogued); architectural findings +
    upstream issues to file (OpenClaw
    [#80909](https://github.com/openclaw/openclaw/issues/80909), NemoClaw
    https↔http gateway-registration mismatch); and smoke results + decision
    rationale.
- [`manyforge/scripts/rebuild-headless-onboarding.sh`](manyforge/scripts/rebuild-headless-onboarding.sh)
  — headless reproduction of the rebuild (detects + auto-fixes the
  NemoClaw `https`-vs-`http` gateway-registration mismatch that breaks
  the onboard step 4/8 "Setting up inference provider").
- [`manyforge/scripts/debug/run-smoke-all-models.sh`](manyforge/scripts/debug/run-smoke-all-models.sh)
  — multi-model bake-off runner across cosmos / omni / 4B-on /
  4B-off / 35B with profile + thinking-flag tuning per run.
- **Proxy: `reasoning → content` SSE/JSON response mutation**
  (`OPENCLAW_PROXY_PROMOTE_REASONING_TO_CONTENT`). OpenClaw 2026.5.22
  treats `content==null` as `code=incomplete_result`; vLLM profiles
  launched with `--reasoning-parser` (cosmos `qwen3`, nemotron3
  `nano_v3`) route output into `reasoning`. The proxy now mirrors it
  into `content` so OpenClaw's stricter contract is satisfied while
  any reasoning-only consumer downstream still sees the field.
  Default on; `=0` to opt out.

### Changed

- **NemoClaw / OpenShell / OpenClaw pinning bumped** in
  [`VERSIONS.md`](VERSIONS.md): NemoClaw `lkg` (v0.0.55), OpenShell
  CLI 0.0.44, OpenClaw 2026.5.22, OpenShell driver `docker` (no k3s).
  Prior audit row kept for diff context.
- **Bridge adapter** (`manyforge/openclaw_assistant_bridge/adapter.py`):
  k3s `docker exec ... kubectl exec` exec wrapper replaced with
  `nemoclaw <sandbox> exec --no-tty -- bash -c`. Multi-line shell
  commands base64-wrapped to bypass the new OpenShell exec gRPC
  guard ("argv contains newline or carriage return characters").
- **Bridge service** logs `openclaw_request_exit_nonzero` with
  stderr/stdout excerpts on the 502 path so silent OpenClaw exits
  are diagnosable from `/tmp/bridge.log`.
- **`setup-manyforge-assistant.sh`** — 5 patches: exec wrapper for
  the new docker driver; health probe via `exec true` (instead of
  the broken `nemoclaw status`); `remote_hash` empty-dir guard for
  fresh sandboxes; split precheck-vs-runtime composer base URL;
  `postCompactionMaxChars` 80000 → 50000 (OpenClaw 2026.5.22
  validation cap).
- **`policies/manyforge-composer.preset.yaml`** — `allowed_ips`
  now includes both `172.17.0.0/16` (old k3s bridge) and
  `172.18.0.0/16` (new docker-driver bridge).

### Manyforge MCP fix (composer-side; tracked here for context)

- [`/api/assistant/bridge/tools/{toolId}` validator path](https://github.com/pastoriomarco/manyforge/blob/dev/manyforge_composer/backend/routes_assistant.py)
  returns HTTP 200 + `{success:false, error:..., result:envelope}`
  envelope instead of 4xx, so OpenClaw's hardcoded
  `erroredAssistantResultPolicy:"drop"` doesn't silently strip
  errored turns + tool results from chat history. (Composer repo is
  upstream of NemoClaw-Thor; this entry exists for traceability.)

### Known regression (deferred, upstream)

- **OpenClaw 2026.5.22 hides MCP catalogs behind `tool_search_code`**
  regardless of `tools.toolSearch.enabled=false` or `mode="tools"`
  config. Cosmos hits 0/3 on a smoke subset because the JS-discovery
  pattern (`tools.search → tools.describe → tools.call`) is not what
  the manyforge skill/bridge prompts/smoke assertions were designed
  for. Pivot to Hermes agent (NemoClaw natively supports
  `NEMOCLAW_AGENT=hermes`) under evaluation; Hermes exposes MCP
  tools directly per its
  [architecture docs](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture).

## [0.1.0] — 2026-05-14

First publishable release. Retrospective entry covering the phase
milestones already tracked in [`VERSIONS.md` §C](VERSIONS.md#c-manyforge-integration).

### Added

- **vLLM serving (`serving/`)**
  - Thor-hardened vLLM container (`v8` shipped on 2026-04-29) built on
    vLLM 0.18 with triton_attn and FlashInfer CUTLASS GEMM/MoE
    optimized for SM110 (Blackwell). `v8.1` staged with vLLM 0.20.1 +
    FlashInfer 0.6.10 + flash-attn-4 4.0.0b12 for `gpt-5.5` support.
  - Cosmos-Reason2-8B profile validated as production default
    (2026-05-07): 9/9 on the OpenClaw lane vs Qwen3.6 1/9 and
    Nemotron-3 0/9 on the same evaluation set.
  - Multiple model profiles: Qwen3.5 variants, Qwen3.6 variants
    (FP4, FP8KV, MTP-2), Gemma 4 31B Turbo NVFP4, Cosmos-Reason2-8B.
    See `serving/config.sh` for the registry.
  - TurboQuant SM110 port for KV-cache 2.5/3.5-bit compression
    (`fix-pr39931-turboquant` runtime mod, auto-applied per profile).

- **Sandbox + control plane (`setup/`)**
  - `configure-local-provider.sh` for non-interactive OpenShell
    provider setup pointing at the local vLLM endpoint.
  - `sandbox-runtime.sh` (~33 KB of helpers) handles Landlock +
    seccomp + netns wiring for the OpenClaw sandbox.
  - iptables egress firewall guidance and presets
    (`setup/policies/`) for fail-closed network isolation.
  - HF_TOKEN handling baked into `start-duo.sh` so gated-repo
    vLLM model loads pick up the token without manual fiddling.

- **ManyForge integration (`manyforge/`)**
  - Phase 1 (2026-05-04): OpenClaw assistant bridge on `:8200`,
    persistent gateway, mode-scoped MCP wrapper. Direct model
    bridge on `:8100` (`direct` lane, `nemoclaw` alias) preserved as backup.
  - Phase 2 (2026-05-07): OpenClaw lane promoted to production
    default. Lane-parity probe verified all 5 tasks within 1.3× of
    the direct lane; OpenClaw is faster on `scene_inspect` and
    `scene_add`.
  - Iter-32 production recipe (2026-05-10): chain-session ON with
    bridge-fired `/compact` every 2 prompts cancels the cascade and
    matches iter-28 chain-off scores (51/66 = 77.3%).
  - vLLM mutator proxy (`manyforge/scripts/proxy/vllm-proxy.py`) on
    `:8000` between OpenClaw and vLLM (`:8050`) injects
    `max_tokens=2048` + bounds the thinking budget — required to
    prevent runaway tool-loop chat-completions.
  - manyforge-composer skill + sandbox provisioner
    (`setup-manyforge-assistant.sh`) installs the agent profile,
    egress preset, and `manyforge` MCP server into the OpenClaw
    sandbox.
  - Smoke harnesses for reliability + lane comparison:
    `smoke-openclaw-assistant-reliability.py`,
    `ab-direct-vs-openclaw.py`.

### Repository hygiene (publication prep)

- Added `VERSION` file (`0.1.0`) as the repository SemVer source of
  truth — distinct from the per-component pins in `VERSIONS.md`.
- Added CHANGELOG.md (this file), ROADMAP.md, CONTRIBUTING.md,
  SECURITY.md.
- Added GitHub PR + issue templates and CI workflows (Python tests,
  bash syntax).
- AGENTS.md updated with the new `dev`-branch + PR-required workflow
  so LLM agents follow the same rules as human contributors.

[Unreleased]: https://github.com/pastoriomarco/NemoClaw-Thor/compare/v0.1.0...dev
[0.1.0]: https://github.com/pastoriomarco/NemoClaw-Thor/releases/tag/v0.1.0
