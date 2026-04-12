# NemoClaw-Thor v5 Transition Plan

**Created**: 2026-04-12
**Status**: COMPLETE (transition executed 2026-04-12)
**Prerequisite**: No active agentic sessions running on Thor

---

## 1. Context — Where v4 Stands

v4 (2026-04-04 to 2026-04-12) brought the full stack to working condition:
vLLM 0.19, FlashInfer CUTLASS GEMM/MoE on SM110, 8 model profiles, KV cache
budget tuning, tool call streaming fix (qwen3_xml), and the OpenClaw agentic
workflow validated end-to-end with Gemma 4 and Qwen 3.5 models.

**v4 unresolved items carried into v5:**

| Item | Issue | v5 Resolution |
|------|-------|---------------|
| `inference.local` timeout | Node.js 22 `NODE_USE_ENV_PROXY` unreliable; OpenClaw can't reach vLLM through OpenShell TLS proxy | Fixed in v4 session (baseUrl → `http://host.openshell.internal:8000/v1`). Need to verify NemoClaw v0.0.13 onboard doesn't regress this. |
| Sandbox survival | Never validated (v4 Phase B.3) | Test in v5 Phase B |
| Model switching cycle | Never validated full Qwen→Gemma→Qwen cycle (v4 Phase F.6) | Test in v5 Phase D |
| Long session stability | Never ran 30+ min session (v4 Phase F.4) | Defer — orthogonal to upgrade |
| OpenClaw upgrade | Deferred in v4 (2026.3.11 still pinned) | Still deferred — NemoClaw main pins 2026.3.11 |
| Egress firewall | Commented out in configure-local-provider.sh | Removed — OpenShell sandbox policy handles network |
| TurboQuant SM110 port | Researched, not built (v4 image plan) | Defer — independent of NemoClaw upgrade |

---

## 2. What Changes — v4 → v5

### Version Matrix

| Component | v4 (current) | v5 (target) | Source |
|-----------|-------------|-------------|--------|
| NemoClaw | v0.0.6-7-g2dd32859 | v0.0.13-1-g1ba57abf (**DONE**) | `git pull` on ~/NemoClaw |
| OpenShell | 0.0.22 | 0.0.26 (**DONE**) | Auto-upgraded by NemoClaw installer |
| OpenClaw | 2026.3.11 | 2026.3.11 (unchanged) | Still pinned in Dockerfile.base |
| vLLM image | 0.19.1rc1.dev195 (ge281cb721) | Same (unchanged) | No rebuild needed |
| Sandbox image | `ghcr.io/nvidia/nemoclaw/sandbox-base:latest` (Mar 30 pull) | Rebuilt during onboard from new Dockerfile.base | NemoClaw builds locally |
| Node.js | 22.22.1 | 22.22.1 (unchanged) | Already meets `>=22.16.0` requirement |

### New NemoClaw runtime dependencies

| Dependency | Type | Notes |
|-----------|------|-------|
| `js-yaml ^4.1.1` | npm (new) | Added to package.json. `npm install` pulls it. |
| OpenShell >= 0.0.24 | binary | Hard gate in onboard preflight + `install-openshell.sh` |
| OpenClaw >= 2026.3.0 | in-sandbox | Enforced by `blueprint.yaml` `min_openclaw_version` |

No new system-level dependencies. No Python dependency changes. No CUDA/driver changes.

### What does NOT change

- vLLM container image (no rebuild)
- Model weights on disk
- Host CUDA/driver/JetPack
- KV cache budget math and model profiles
- Thor-specific vLLM launch arguments

---

## 3. New NemoClaw Features — Impact Assessment

### HIGH impact on our workflow

**A. `--dangerously-skip-permissions` (#1708)**

Applies maximally permissive sandbox policy. Replaces our:
- Commented-out egress firewall in `configure-local-provider.sh`
- Manual policy yaml patching
- The entire `enforce-egress-firewall.sh` script (for permissive mode)

Use at onboard: `nemoclaw onboard --dangerously-skip-permissions`
Use on running sandbox: `nemoclaw thor-v5 connect --dangerously-skip-permissions`

**NemoClaw-Thor change**: Remove egress firewall code from `configure-local-provider.sh`
and document the `--dangerously-skip-permissions` approach. Keep
`enforce-egress-firewall.sh` for when we want to lock down.

**B. Runtime model override via env vars (#1633)**

`NEMOCLAW_MODEL_OVERRIDE` patches `openclaw.json` at sandbox boot.
Our `sync_sandbox_runtime_config()` does the same thing but also patches:
- `baseUrl` (inference.local → host.openshell.internal:8000)
- `api` (openai-responses → openai-completions)
- concurrency settings (maxConcurrent, subagents)
- timeoutSeconds
- onboard config.json

The env var only overrides model name — NOT url, api, concurrency, or timeout.

**NemoClaw-Thor change**: Keep `sync_sandbox_runtime_config()` but use
`NEMOCLAW_MODEL_OVERRIDE` as a complementary mechanism for simpler model
switches. The full sync is still needed for the baseUrl fix and concurrency tuning.

**C. Inference timeout 180s for local providers (#1620)**

Default local provider timeout bumped from 60s to 180s. Configurable via
`NEMOCLAW_LOCAL_INFERENCE_TIMEOUT`.

**NemoClaw-Thor change**: Our `timeoutSeconds=1800` in openclaw.json is the
agent-level timeout (how long the agent waits for a single LLM call). The
180s is the OpenShell inference gateway timeout (how long the gateway holds
the connection open). Both are needed — 180s gateway + 1800s agent. No
script changes needed, but verify the onboard default is sufficient.

**D. Jetson host setup in installer (#1702)**

Auto-detects Thor (JP7), applies `br_netfilter` and `bridge-nf-call-iptables`.
For JP7 specifically: no iptables or Docker daemon.json changes needed (unlike
JP6/Orin which requires iptables-legacy).

**NemoClaw-Thor change**: Our `apply-host-fixes.sh` was deleted in v4 E.3.
The NemoClaw installer now handles this. Document that `nemoclaw` installer
handles host prereqs on Thor.

### MEDIUM impact

**E. `--from <Dockerfile>` custom sandbox images (#1059)**

Build sandbox from custom Dockerfile with NemoClaw args injected. Useful if
we need custom packages (build-essential, Python deps for ManyForge dev).

**NemoClaw-Thor change**: No immediate change. Document as the path for
custom sandbox images if needed in future. Replaces our v3 Dockerfile patching.

**F. Onboard presets TUI (#1463)**

Interactive policy preset selector. We'd skip this with `--dangerously-skip-permissions`.

**NemoClaw-Thor change**: None. Skip with the permissive flag.

**G. OpenClaw version as build arg (#1712)**

`ARG OPENCLAW_VERSION=2026.3.11` in Dockerfile.base, overridable via `--build-arg`.

**NemoClaw-Thor change**: None now. Useful later for testing OpenClaw 2026.4.x
without waiting for NemoClaw to bump the pin. Document the override mechanism.

### LOW impact

**H. Multi-agent Hermes (#1618)** — We use OpenClaw, not Hermes. Onboard flow
now has an agent selection step (defaults to OpenClaw). No impact if we use
`--agent openclaw` or accept the default.

**I. Podman support (#1359)** — We use Docker. No impact.

**J. `nemoclaw credentials` command (#1597)** — Nice to have for key management.
No workflow change.

**K. Landlock read-only /sandbox (#1121)** — Agent can only write to
`/sandbox/.openclaw-data` and `/sandbox/.nemoclaw`. Our workspace is under
`/sandbox/workspace` which symlinks to `/sandbox/.openclaw-data/workspace` — 
should be writable. **VERIFY during onboard.**

**L. SSH host-key TOFU (#691)** — Replaces `StrictHostKeyChecking=no` with
trust-on-first-use. May affect our `sandbox_ssh_command()` if it relied on
the old behavior. **VERIFY during testing.**

### Security improvements (passive, no action needed)

- SSRF blocklist with all IANA-reserved ranges (#1557)
- Secret redaction across all modules (#1761)
- Sandbox image pinned by digest in blueprint.yaml (#1438)
- GitHub access gated behind opt-in preset (#1660)
- Config integrity verification before env var overrides (#1633)

---

## 4. Phases

### Phase A — Pre-flight & Backup

**Goal**: Snapshot current state so we can roll back if anything breaks.

#### A.1: Save current sandbox state

```bash
# Dump current openclaw.json for reference
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell thor-v4 -c agent \
    -- cat /sandbox/.openclaw/openclaw.json > /tmp/thor-v4-openclaw.json

# Dump current nemoclaw config
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell thor-v4 -c agent \
    -- cat /sandbox/.nemoclaw/config.json > /tmp/thor-v4-nemoclaw-config.json

# Save workspace file list (if anything valuable)
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell thor-v4 -c agent \
    -- find /sandbox/workspace -type f 2>/dev/null > /tmp/thor-v4-workspace-files.txt
```

#### A.2: Record current versions

```bash
openshell --version                          # 0.0.22
nemoclaw --version                           # v0.0.6-7
# OpenClaw version from sandbox
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell thor-v4 -c agent \
    -- openclaw --version                    # 2026.3.11
```

#### A.3: Verify vLLM is stopped or note running model

```bash
docker ps --filter name=sweet_chaplygin --format '{{.Names}} {{.Status}}'
```

If vLLM is running, note the profile. It can stay running — the upgrade
doesn't touch the vLLM container.

---

### Phase B — NemoClaw + OpenShell Upgrade

**Goal**: Update NemoClaw to latest main, let it auto-upgrade OpenShell.

#### B.1: Update NemoClaw source

```bash
cd ~/NemoClaw
git fetch origin
git log --oneline HEAD..origin/main | wc -l    # expect ~154 commits
git pull origin main
npm install                                      # pulls js-yaml
```

Verify: `nemoclaw help` shows new commands (`credentials`, new flags).
Verify: `nemoclaw --version` shows new version.

#### B.2: Destroy old sandbox

The v4 sandbox was created under NemoClaw v0.0.6 with OpenShell 0.0.22 and
a stale sandbox image (Mar 30). Clean slate is safer than in-place upgrade.

```bash
nemoclaw thor-v4 destroy --yes
```

**Why destroy instead of re-onboard in-place**: The v4 sandbox has:
- Stale `NEMOCLAW_*` env vars baked into the pod spec (from v0.0.6 onboard)
- Old sandbox image without Landlock read-only, without runtime model override
  entrypoint, without proxy-env.sh in /tmp
- Accumulated state from debugging sessions (test files, stale sessions)

A fresh onboard builds a new sandbox image from the updated Dockerfile.base
and gets all the v0.0.13 entrypoint improvements.

#### B.3: Re-onboard with default (secure) policy

**EXECUTED 2026-04-12** — Used default sandbox policy (not `--dangerously-skip-permissions`)
with github/pypi/npm presets. This reproduces the correct production setting.

```bash
nemoclaw onboard
```

During onboard:
- **Agent selection**: Accept default (OpenClaw) or specify `--agent openclaw`
- **Inference**: Choose "Custom OpenAI-compatible" → enter our vLLM endpoint
  - URL: `http://host.openshell.internal:8000/v1` (note: HTTP not HTTPS)
  - API key: `unused` (or whatever THOR_LOCAL_VLLM_API_KEY is set to)
  - Model: whichever model is currently running in vLLM
- **Sandbox name**: `thor-v5` (new name to avoid confusion with v4)
- **OpenShell upgrade**: The installer will detect 0.0.22 < 0.0.24 and
  auto-upgrade. Confirm when prompted.

**CRITICAL CHECK after onboard**: Verify the sandbox can reach vLLM directly:
```bash
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell thor-v5 -c agent \
    -- curl -s --max-time 5 http://host.openshell.internal:8000/v1/models
```

If this fails (likely — onboard writes `https://inference.local/v1` as baseUrl),
proceed to Phase C which fixes it.

#### B.4: Verify OpenShell version

```bash
openshell --version    # should be >= 0.0.24
```

#### B.5: Verify sandbox survival

```bash
nemoclaw stop
sleep 5
nemoclaw start
# Wait for gateway to come up
nemoclaw list                    # thor-v5 should appear
nemoclaw thor-v5 connect         # should work without re-onboard
```

This was never tested in v4 (Phase B.3 was skipped). OpenShell 0.0.24
specifically fixed sandbox survival — this is the right time to validate.

---

### Phase C — NemoClaw-Thor v5 Script Updates

**Goal**: Update Thor scripts for the new NemoClaw version and clean up v4 debt.

#### C.1: Update sandbox-runtime.sh — baseUrl fix

The `inference.local` → `host.openshell.internal:8000` fix was already applied
in v4 (this session). Verify it's still correct after re-onboard.

**Test**: Run `configure-local-provider.sh <profile>`, then verify:
```bash
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell thor-v5 -c agent \
    -- python3 -c "import json; d=json.load(open('/sandbox/.openclaw/openclaw.json')); print(d['models']['providers']['inference']['baseUrl'])"
# Expected: http://host.openshell.internal:8000/v1
```

If onboard overwrites this to `https://inference.local/v1`, our sync function
will fix it — but we should also check if the new NemoClaw entrypoint
(runtime model override) interferes.

#### C.2: Update config.sh — sandbox name

Change default/fallback sandbox name references from `thor-v4` to `thor-v5`
where hardcoded (if any). The resolve function uses registry auto-detection
so this may not be needed.

**Check**: `grep -r "thor-v4" src/NemoClaw-Thor/`

#### C.3: Update config.sh — new env vars

Add support for NemoClaw v0.0.13 env vars in `load_thor_runtime_config`:

```bash
# Runtime model override (NemoClaw >= 0.0.13)
# Complementary to sync_sandbox_runtime_config — patches model name at boot
THOR_NEMOCLAW_MODEL_OVERRIDE="${NEMOCLAW_MODEL_OVERRIDE:-}"

# Inference timeout (NemoClaw >= 0.0.13)
# Default 180s for local providers
THOR_NEMOCLAW_INFERENCE_TIMEOUT="${NEMOCLAW_LOCAL_INFERENCE_TIMEOUT:-180}"
```

#### C.4: Clean up configure-local-provider.sh

Items to address:

1. **Egress firewall block** (lines 186-205): Currently commented out. 
   Replace the entire block with a note that `--dangerously-skip-permissions`
   handles this. Remove the conditional call to `enforce_sandbox_egress_firewall`.

2. **SSH handshake reconciliation** (lines 71-83): Still commented out. 
   With OpenShell 0.0.24 sandbox survival, SSH secrets persist. Delete
   the commented block entirely.

3. **NemoClaw registry update** (lines 90-105): Cosmetic only. Delete
   the commented block.

4. **Gateway lifecycle** (lines 114-131): Was uncommented in v4 to ensure
   the gateway runs. Verify it's still needed with the new NemoClaw onboard.
   The new onboard may handle gateway startup natively.

#### C.5: Update PLAN file and documentation

- Rename `PLAN-v4-reboot.md` status to COMPLETE
- This file (`PLAN-v5-transition.md`) is the active plan
- Update `KV-CACHE-BUDGET.md` if agent allocation changes
- Update `README.md` with v5 instructions

#### C.6: Verify Landlock compatibility

The new sandbox restricts `/sandbox` to read-only. Verify:
```bash
# Check workspace is writable (should symlink to .openclaw-data/workspace)
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell thor-v5 -c agent \
    -- ls -la /sandbox/workspace
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell thor-v5 -c agent \
    -- touch /sandbox/workspace/test-landlock && echo "OK" || echo "BLOCKED"

# Check .nemoclaw is writable (for our config.json writes)
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell thor-v5 -c agent \
    -- touch /sandbox/.nemoclaw/test-landlock && echo "OK" || echo "BLOCKED"

# Check .openclaw is read-only (expected)
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell thor-v5 -c agent \
    -- touch /sandbox/.openclaw/test-landlock && echo "WRITABLE (unexpected)" || echo "READ-ONLY (expected)"
```

If Landlock blocks our `sync_sandbox_runtime_config` from writing openclaw.json
(which is in `/sandbox/.openclaw/`), we have a problem. The v4 script does
`chmod 644 → write → chmod 444` on this file. With Landlock, even root can't
write to read-only paths.

**Mitigation options if blocked**:
1. Use `NEMOCLAW_MODEL_OVERRIDE` env var instead (only covers model name)
2. Patch the file before Landlock locks it (in the entrypoint, before the
   sandbox user takes over)
3. Use `openshell sandbox exec --privileged` if available
4. Use `--dangerously-skip-permissions` which sets `include_workdir: true`

This is the **highest-risk item** in the entire upgrade.

---

### Phase D — End-to-End Validation

**Goal**: Verify the full agentic workflow works on the new stack.

#### D.1: Configure provider and test basic connectivity

```bash
./configure-local-provider.sh <running-model-profile>
```

Then test:
```bash
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell thor-v5 -c agent \
    -- su sandbox -c 'openclaw agent --local --agent main --session-id v5-test-1 \
       --message "Reply with exactly: V5_WORKING" --timeout 60 --json'
```

#### D.2: Test tool calling with file write

```bash
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell thor-v5 -c agent \
    -- su sandbox -c 'openclaw agent --local --agent main --session-id v5-write-test \
       --message "Create /sandbox/workspace/hello.py with a 100-line Python program" \
       --timeout 300 --json'
```

Verify file exists, is complete, and runs.

#### D.3: Model switching cycle

```bash
# Start with model A (e.g., qwen3.5-27b)
./start-model.sh qwopus3.5-27b-nvfp4
./configure-local-provider.sh qwopus3.5-27b-nvfp4
# Test agent → should work

# Switch to model B (e.g., gemma4-26b)
# Stop model A, start model B
./start-model.sh gemma4-26b-a4b-it
./configure-local-provider.sh gemma4-26b-a4b-it
# Test agent → should work with new model

# Switch back to model A
./start-model.sh qwopus3.5-27b-nvfp4
./configure-local-provider.sh qwopus3.5-27b-nvfp4
# Test agent → should work, no stale state
```

#### D.4: Sandbox survival under model switch

```bash
# With agent working, restart the gateway
nemoclaw stop && nemoclaw start
# Re-configure provider (gateway restart clears provider route)
./configure-local-provider.sh <current-profile>
# Test agent → should work without re-onboard
```

---

## 5. NemoClaw-Thor v5 File Changes Summary

| File | Action | What |
|------|--------|------|
| `PLAN-v5-transition.md` | NEW | This plan |
| `PLAN-v4-reboot.md` | DELETED | Superseded by this plan |
| `AGENT_NOTES-v4.md` | DELETED | Outdated v4 notes |
| `dashboard_tcp_proxy.py` | DELETED | Unused proxy script |
| `enforce-egress-firewall.sh` | DELETED | Replaced by OpenShell sandbox policy |
| `lib/egress-firewall.sh` | DELETED | Replaced by OpenShell sandbox policy |
| `templates/qwen3-tool-call-compat.jinja` | DELETED | Replaced by vLLM native parser |
| `configure-local-provider.sh` | UPDATE | Removed dead egress-firewall source, commented-out blocks |
| `lib/launch.sh` | UPDATE | Added TORCHINDUCTOR_CACHE_DIR for persistent cubin cache |
| `README.md` | REWRITTEN | Complete v5 rewrite with architecture, model profiles, config explanation |
| `USER_QUICKSTART_MANUAL.md` | REWRITTEN | Updated all versions, sandbox name, model profiles |

---

## 6. Risk Assessment (Updated 2026-04-12)

| Risk | Likelihood | Impact | Mitigation | Status |
|------|-----------|--------|------------|--------|
| Landlock blocks openclaw.json writes | ~~HIGH~~ **NONE** | N/A | kubectl exec bypasses Landlock (new process); chattr fails (no capability) | **RESOLVED** — verified from code |
| Config hash mismatch after sync | MEDIUM | HIGH — entrypoint refuses to start | Add hash recomputation to sync function (Section 10, C.7) | **Mitigated** |
| Model name format (`inference/` prefix) | MEDIUM | MEDIUM — agent routing broken | Verify during Phase D (C.8); upstream sets plain name, we set prefixed | **To verify** |
| OpenShell 0.0.26 upgrade breaks gateway | LOW | HIGH — sandbox unreachable | Can downgrade: `npm install -g openshell@0.0.22` | **RESOLVED** — 0.0.26 works |
| New onboard flow breaks our provider setup | MEDIUM | MEDIUM — need to debug new flow | Saved v4 config as reference; can re-onboard | **RESOLVED** — configure-local-provider.sh fixes all |
| Sandbox survival still broken | LOW | LOW — just re-onboard | OpenShell 0.0.26 supports pod persistence | Open (not tested yet) |
| Hermes agent selection confuses onboard | LOW | LOW — just specify `--agent openclaw` | Default is OpenClaw anyway | **RESOLVED** — default is OpenClaw |

---

## 7. Rollback Plan

If v5 is broken and we need to go back:

```bash
# 1. Destroy v5 sandbox
nemoclaw thor-v5 destroy --yes

# 2. Revert NemoClaw
cd ~/NemoClaw && git checkout v0.0.6-7-g2dd32859
npm install

# 3. Downgrade OpenShell if needed
npm install -g openshell@0.0.22

# 4. Re-onboard with old settings
nemoclaw onboard    # creates thor-v4 equivalent

# 5. Apply v4 configure
cd ~/workspaces/nemoclaw/src/NemoClaw-Thor
./configure-local-provider.sh <profile>
```

vLLM image and model weights are untouched by the upgrade — no risk there.

---

## 8. Open Questions — RESOLVED

### Q1: Landlock + openclaw.json writes — RESOLVED

**Answer**: kubectl exec processes bypass Landlock entirely (kernel-level
per-process restriction, not inherited by new processes started via exec).
chattr +i fails because the container lacks `CAP_LINUX_IMMUTABLE` (not in
bounding set, not in the entrypoint's drop list either — just never granted
by OpenShell's container runtime). The entrypoint falls back gracefully
(`chattr +i ... 2>/dev/null || true`).

**Result**: Our `sync_sandbox_runtime_config()` via kubectl exec still works
under v0.0.13. DAC restriction (`root:root chmod 444`) is the only barrier —
our script already handles this (`chmod 644` → write → `chmod 444`).

### Q2: baseUrl rewrite — RESOLVED

**Answer**: The entrypoint's `apply_model_override()` (PR #1633) does NOT
touch `baseUrl`. It only patches: model id, model name, primary model ref,
context window, max tokens, reasoning, and API type. The baseUrl is baked
at build time via `NEMOCLAW_INFERENCE_BASE_URL` ARG and never modified at
runtime.

**Result**: Set `NEMOCLAW_INFERENCE_BASE_URL` to
`http://host.openshell.internal:8000/v1` at build time (pre-onboard
Dockerfile patch). The entrypoint won't touch it afterward.

### Q3: Gateway lifecycle — RESOLVED

**Answer**: The NemoClaw entrypoint starts the OpenClaw gateway natively as
PID 1's child process (either via `gosu gateway openclaw gateway run` in
root mode, or directly in non-root mode). Gateway startup is handled by the
entrypoint, not by external scripts.

**Result**: `ensure_sandbox_gateway_running()` in `configure-local-provider.sh`
is redundant for v5. The gateway runs automatically. Keep as a diagnostic
fallback only.

### Q4: Non-interactive onboard — PARTIALLY RESOLVED

**Answer**: `patchStagedDockerfile()` in onboard.js rewrites the staged
Dockerfile's ARG values based on interactive prompts. There is no
`--build-arg` passthrough for arbitrary values. However, we can patch
`~/NemoClaw/Dockerfile` BEFORE running onboard — the onboard copies the
Dockerfile to a temp dir and patches specific ARGs, preserving our additions.

**Result**: Pre-onboard Dockerfile patch is the mechanism. See Section 9 below.

---

## 9. Configuration Architecture — kubectl exec Runtime Patching

### Key Discovery (2026-04-12)

**OpenShell strips all Docker ENV vars.** The sandbox entrypoint (PID 1) is
`openshell-sandbox`, NOT the Docker ENTRYPOINT. It replaces the entire process
environment with a sanitized set (proxy vars, TLS certs, PATH). Docker `ENV`
directives — including `NEMOCLAW_MODEL_OVERRIDE`, `NEMOCLAW_CONTEXT_WINDOW`,
etc. — are NOT propagated to the entrypoint process (`nemoclaw-start`).

This means NemoClaw v0.0.13's `apply_model_override()` is dead code under
OpenShell. The 4-layer configuration architecture originally planned (build-time
→ boot-time → runtime → production) collapsed to a single layer:

**All runtime config patching happens via `kubectl exec` in
`sync_sandbox_runtime_config()`.** This is the only mechanism that works.

### Why `configure-local-provider.sh` Is the Only Path

| Mechanism | Works? | Why |
|-----------|--------|-----|
| Docker ENV vars | NO | OpenShell strips them from the process environment |
| `apply_model_override()` | NO | Reads stripped ENV vars — always sees empty |
| Build-time ARGs in Dockerfile | PARTIAL | `baseUrl` baked correctly, but `contextWindow`/`maxTokens` hardcoded in Python |
| Patching upstream Dockerfile | NO | It's NemoClaw's repo, not ours |
| `kubectl exec` runtime patching | **YES** | Bypasses Landlock (new process) and DAC (`chmod 644→write→chmod 444`) |

### What `sync_sandbox_runtime_config()` Patches

| Setting | Onboard default | What we set |
|---------|----------------|-------------|
| `baseUrl` | `https://inference.local/v1` | `http://host.openshell.internal:8000/v1` |
| `api` | `openai-completions` | `openai-completions` |
| `contextWindow` | 131072 | 262144 |
| `maxTokens` | 4096 | 16384 |
| `timeoutSeconds` | (unset) | 1800 |
| `maxConcurrent` | (unset) | Per-profile (1-6) |
| `subagents` | (unset) | Per-profile |
| `/sandbox/.nemoclaw/config.json` | Onboard state | Updated model ref |

### Why It Works Under v0.0.13

- `kubectl exec` bypasses Landlock (kernel-level per-process, not inherited)
- `chattr +i` fails silently (container lacks `CAP_LINUX_IMMUTABLE`)
- DAC (`root:root chmod 444`) is handled by our `chmod 644→write→chmod 444` sequence
- Config hash: recomputed by our sync function after writing

### Upstream Bug: `inference.local` Failure

OpenClaw's `globalThis.fetch` (Node.js 22 undici) does NOT honor
`HTTP_PROXY`/`HTTPS_PROXY` env vars. Bug: openclaw/openclaw#62181.
Fix PR (openclaw/openclaw#43919) has been open 5+ weeks, unmerged.

This is why the default `https://inference.local/v1` baseUrl doesn't work —
the request never reaches the OpenShell L7 proxy. Our direct
`http://host.openshell.internal:8000/v1` URL bypasses the proxy entirely.

---

## 10. Transition Execution Log (2026-04-12)

1. Updated NemoClaw: `git pull origin main && npm install` → v0.0.13-1-g1ba57abf
2. OpenShell auto-upgraded to 0.0.26
3. Onboarded with default (secure) policy + github/pypi/npm presets → sandbox `thor-v5`
4. Discovered OpenShell ENV stripping → abandoned Dockerfile patching approach
5. Confirmed `kubectl exec` bypasses Landlock and `chattr +i` fails gracefully
6. Ran `configure-local-provider.sh` → all settings patched successfully
7. Agent functional test: 378-line Python file, 14 tests, all passing
8. Repository cleanup: deleted 7 dead files, rewrote README.md and USER_QUICKSTART_MANUAL.md

---

## 11. Future (post-v5, not in scope)

- **OpenClaw 2026.4.x**: Wait for NemoClaw to bump the pin. Timeout/failover
  fixes could help, but not validated by upstream yet.
- **TurboQuant SM110 port**: KV cache compression (2.5/3.5 bits). Independent
  of NemoClaw upgrade. Pure Triton, needs 2-file SM110 guard relaxation.
- **FlashInfer CUTLASS autotuning**: Both Triton and FlashInfer MoE paths are
  untuned on Thor. High-value optimization, independent of NemoClaw.
- **Upstream fix for OpenShell ENV stripping**: If OpenShell propagates Docker
  ENV vars in a future version, NemoClaw's `apply_model_override()` would work
  and we could reduce reliance on `kubectl exec` patching.
- **Upstream fix for inference.local**: openclaw/openclaw#62181 / #43919. Once
  merged, `https://inference.local/v1` would work through the L7 proxy and
  our `baseUrl` override would no longer be needed.
