# Control-plane upgrade — 2026-07-08 (Thor)

Bumped the NemoClaw / OpenShell control plane and re-onboarded the Hermes lane
on Jetson AGX Thor. This is the companion procedure record for the Scope A pin
move in [`VERSIONS.md`](../../VERSIONS.md); it is the analogue of the earlier
[`REBUILD-2026-06-02.md`](archive/REBUILD-2026-06-02.md) for this upgrade.

## What changed

| Component | From | To | Rebuild forced? |
|---|---|---|---|
| NemoClaw CLI (host) | `lkg` = v0.0.55 (running v0.0.56) | `lkg` = **v0.0.73** | no (npm rebuild of `~/NemoClaw`) |
| OpenShell CLI / gateway / sandbox | 0.0.44 | **0.0.71** | no (binary swap) — **but requires a host-gateway restart** |
| Hermes agent (Hermes lane) | v0.14.0 (`v2026.5.16`) | **v0.17.0** (`v2026.6.19`) | yes — Hermes sandbox image rebuilt from the v0.17 base |
| OpenClaw sandbox image | `sha256:b3d832b596…` | **unchanged** | no — digest + `min_openclaw 2026.3.11` identical in both blueprints |
| vLLM serving image / model | (unchanged) | `qwen3.6-27b-nvfp4` on `:8050`, proxy `:8000` | no |

Upstream `lkg` had advanced 18 patch releases (v0.0.55 → v0.0.73; latest tag
v0.0.76). We track `lkg`, not latest — the v0.0.73→v0.0.76 tail is `dcode`,
messaging (WhatsApp/Teams), and test refactors with nothing this stack needs.

## Correct upgrade procedure (any lane)

Run from a quiesced state with the **served model left running** (onboard/create
probes the inference endpoint). The steps are host-level and lane-independent
except where noted.

1. **Back up** `~/.nemoclaw` (registry + provider state) — the rollback anchor.
2. **Bump the NemoClaw CLI** on the read-only `~/NemoClaw` checkout:
   ```bash
   cd ~/NemoClaw && git fetch --tags && git checkout v0.0.73 && npm install
   nemoclaw --version   # v0.0.73
   ```
3. **Upgrade OpenShell** (coupled — the v0.0.73 blueprint pins `min==max==0.0.71`):
   ```bash
   NEMOCLAW_NON_INTERACTIVE=1 bash ~/NemoClaw/scripts/install-openshell.sh
   openshell --version  # 0.0.71  (also openshell-gateway / openshell-sandbox)
   ```
4. **Restart the host gateway — do not skip this.** The in-place install swaps
   the on-disk binaries but leaves the **old 0.0.44 gateway daemon running in
   memory** (`runtime.json` still reports `openshellVersion: 0.0.44`, and
   `/proc/<pid>/exe` shows the deleted old binary). A 0.0.71 sandbox run under
   that stale gateway crash-loops on `no sandbox token source available` — the
   0.0.71 gateway↔sandbox protocol injects a per-sandbox auth token the old
   daemon never provided. Restart it the supported way — this is what
   `ensure_openshell_gateway_running` (in [`setup/checks.sh`](../../setup/checks.sh))
   and `launch.sh` already use to self-heal after a reboot:
   ```bash
   nemoclaw <sandbox> recover      # respawns the docker-driver gateway on 0.0.71
   openshell status -g nemoclaw    # healthy; runtime.json now shows 0.0.71
   ```
   `openshell gateway start` does **not** exist (removed); a plain `kill` of the
   daemon does **not** auto-relaunch it. `recover` is the only supported path.
5. **Re-onboard sandboxes.** A sandbox created under the old gateway carries the
   old token wiring and cannot be repaired in place (the token is injected at
   create time) — `destroy` and recreate it under the 0.0.71 gateway.

## Hermes-lane re-baseline (v0.14 → v0.17)

The Hermes lane derives its sandbox Dockerfile from the pinned NemoClaw's
`agents/hermes/Dockerfile`, so bumping the CLI moved Hermes to v0.17.0 and
surfaced three drifts that `manyforge/scripts/setup-hermes.sh` had to absorb:

1. **Dockerfile COPY-source staging.** v0.17 added 10 new COPY sources
   (`src/lib/messaging/`, `scripts/gateway-control.sh`,
   `scripts/lib/gateway-supervisor.sh`, `scripts/lib/sandbox-rlimits.sh`,
   `scripts/managed-gateway-control.py`, `scripts/state-dir-guard.py`, and four
   `agents/hermes/*.py` helpers). The hand-maintained staging list in
   `setup-hermes.sh` step 1 must include them or the build fails at the COPY.
   This list re-lists a subset of the upstream Dockerfile's COPY set, so it
   drifts on **every** Hermes bump.
2. **Dashboard port collision.** v0.17 `agents/hermes/start.sh` hardcodes the
   Hermes API `INTERNAL_PORT=18642` and rejects a dashboard port equal to it.
   `setup-hermes.sh` was passing `NEMOCLAW_DASHBOARD_PORT=18642` → moved to
   `18643`.
3. **Secret-boundary validator.** v0.17 ships
   `validate-env-secret-boundary.py`, whose `SECRET_KEY_RE` matches the `API`
   word-token and refuses startup for any such env var carrying a raw value.
   `setup-hermes.sh` was passing `HERMES_API_TIMEOUT` (a numeric timeout) →
   dropped. It is not on the validator's allowlist and cannot be passed; Hermes
   uses its built-in default. This loses nothing on Thor — `run_agent.py` only
   applies `HERMES_API_TIMEOUT` when `is_local_endpoint(base_url)` is true, and
   that helper does not recognize the `.openshell.internal` host.

## Lane relevance

- **Host layer (NemoClaw v0.0.73, OpenShell 0.0.71, the gateway restart in
  step 4): applies to every lane** (direct / openclaw / hermes). The gateway
  restart is the single most important cross-lane step — the `no sandbox token`
  crash-loop is a gateway-protocol failure, not a Hermes one.
- **The three Hermes-lane fixes above: Hermes-only.** The OpenClaw lane uses a
  prebuilt sandbox image (digest unchanged) and a different provisioner
  (`setup-manyforge-assistant.sh`); it never touches the Hermes Dockerfile
  derivation, so none of the messaging-staging / dashboard-port /
  `HERMES_API_TIMEOUT` issues apply.
- **OpenClaw lane: onboards cleanly, but an in-sandbox agent turn is blocked by
  an OpenShell 0.0.71 regression — needs its own re-baseline.** The re-onboard
  itself is clean: `nemoclaw onboard` (managed flow, native token injection)
  built `my-assistant` to **Ready/healthy** on the unchanged OpenClaw image, the
  provider `compatible-endpoint`→`custom` rename was a non-issue (onboard reused
  the provider and its inference smoke passed), and the sandbox reaches the host
  model directly (`host.openshell.internal:8000` → 200 from inside). What is
  broken: the agent routes model calls through `https://inference.local/v1`, and
  that **`inference.local` indirection does not resolve under 0.0.71** (direct
  host reachability works, the indirection fails) → `ECONNREFUSED`. Also new:
  the in-sandbox OpenClaw gateway now requires **device pairing / role approval**
  (`pairing required: device is not approved yet`) and its pairing-file write
  fails (`PermissionError: /sandbox/.openclaw/devices/pending.json`). These are
  OpenClaw-lane config issues, separate from this control-plane upgrade; the
  correct smoke is the production `composer → openclaw bridge` path
  (`scripts/debug/run-cell-openclaw.sh`), not a bare `openclaw agent` dispatch.

## Validated / deferred

- Validated: `nemoclaw --version` v0.0.73; `openshell*` 0.0.71; gateway
  `runtime.json` 0.0.71; `hermes-assistant` phase **Ready**; Hermes `/health`
  → `{"status":"ok","version":"0.17.0"}`; **authenticated Hermes round-trip**
  through the bridge (`/healthz` `apiKeyConfigured:true`; a real turn returned a
  model completion, HTTP 200); OpenClaw `my-assistant` onboards to Ready +
  inference smoke passes.
- Deferred (OpenClaw-lane re-baseline, separate task): fix `inference.local`
  resolution under 0.0.71 (or point the agent at `host.openshell.internal:8000`),
  the device-pairing/role-approval flow, and the pairing-file permission; then
  run the `run-cell-openclaw.sh` production smoke.

## Rollback

- NemoClaw CLI: `cd ~/NemoClaw && git checkout v0.0.56 && npm install`.
- OpenShell: the v0.0.56 checkout's `install-openshell.sh` pins 0.0.44 —
  re-run it, then `nemoclaw <sandbox> recover` to relaunch the 0.0.44 gateway.
- Registry: restore the `~/.nemoclaw` backup from step 1.

> **Orin note.** [`ORIN-SETUP.md`](../../ORIN-SETUP.md) is a separate device
> (gguf models, `/mnt/nova_ssd`, bind-mounted `~/.nemoclaw`) and is out of scope
> for this Thor upgrade.
