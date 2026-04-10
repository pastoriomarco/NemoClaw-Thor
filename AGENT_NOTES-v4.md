# Agent Notes (v4)

This file is a session handoff and operational note for future work on this
repository. It is intentionally more explicit and more candid than the README.

For the operator-facing usage guide, see `USER_QUICKSTART_MANUAL.md`.

## v4 Changes From v3

- **vLLM 0.19** — native Gemma 4 support, streaming tool-call fix (PR #35615),
  9.9% MoE throughput improvement, async speculative decoding
- **Restream proxy eliminated** — vLLM 0.19 fixes the qwen3_coder streaming
  parser bug that required the restream proxy. OpenClaw now connects directly
  to vLLM via the OpenShell provider path
- **OpenClaw v2026.4.1** — tool-call repair, subagent validation, legacy
  config aliases removed
- **OpenShell v0.0.22** — 10 CVEs fixed, seccomp hardening
- **Gemma 4 model profiles** — `gemma4-31b-it-nvfp4` and `gemma4-26b-a4b-it`
  added alongside existing Qwen profiles
- **Codex review integration** — quality gate now includes an optional
  independent review from OpenAI Codex via the `codex-plugin-cc` Claude Code
  plugin, providing a second-opinion pass before accepting agent output
- **transformers 5.5** — required for Gemma 4 model class support

## What This Repo Is

This repository is a Thor-specific integration and hardening layer around:

- NVIDIA OpenShell
- NVIDIA NemoClaw
- a local OpenAI-compatible vLLM server
- local Qwen model serving based on the sibling `thor_llm` repo

It is not the agent runtime itself, and it is not the place where the actual
multi-agent orchestrator/coder/tester logic is implemented. Its job is to make
the base stack safer, more local-first, and more reversible on Jetson Thor.

## CRITICAL: Upstream Evolution Warning

NemoClaw, OpenClaw, and OpenShell are all rapidly evolving NVIDIA projects in
active alpha/beta development. APIs, config formats, CLI flags, sandbox image
internals, and onboarding flows change frequently between releases.

**The suggested install path for a new machine is: have a frontier-class coding
model (Claude, GPT-4, etc.) follow the install instructions interactively on
the target machine and fix any issues that arise in real time.** Do not blindly
run scripts expecting them to work unchanged against a newer upstream. Treat
this repo's scripts as a reference implementation and adaptation guide, not as
a frozen installer.

Specific upstream areas that have broken or changed between versions:

- OpenClaw device pairing protocol and auth modes
- Sandbox image filesystem layout and permission model for `/sandbox/.openclaw/`
- `openclaw.json` config schema (new keys, changed nesting)
- OpenShell gateway SSH handshake secret rotation behavior on reboot
- OpenShell sandbox CRD spec format
- NemoClaw onboarding wizard prompts and step ordering
- NemoClaw CLI subcommands (`nemoclaw <sandbox> exec` was removed)
- Network namespace isolation between SSH sessions and pod root namespace

When updating, change one component at a time and verify with `./status.sh`
after each change.

## ★ ManyForge Development — Quick Navigation

If you're here to run ManyForge agent sessions, go directly to these sections:

1. **[Programmatic Agent Access From Host](#programmatic-agent-access-from-host)** — kubectl exec, session cleanup, SSH
2. **[ManyForge Development Rules](#manyforge-development-rules)** — supervisor/agent behavioral contracts, contract freeze
3. **[ManyForge Agent Orchestration Workflow](#manyforge-agent-orchestration-workflow-complete-reference)** — full 9-step workflow with exact commands
4. **[Codex Review Integration](#codex-review-integration)** — dual-reviewer quality gate with OpenAI Codex
5. **[Sandbox Environment Requirements](#sandbox-environment-requirements-for-manyforge-v00)** — complete toolchain needed

## Sandbox Environment Requirements for ManyForge v0.0

The sandbox must contain **all** toolchain and library dependencies for the
entire v0.0 walking skeleton. Running different components in different
environments makes end-to-end testing impossible — which is a primary v0.0 goal.

**Principle: one sandbox image, one build/test environment for everything.**

### Required toolchain

| Category | Package | Why |
|----------|---------|-----|
| **C++ build** | `build-essential`, `cmake`, `libgtest-dev` | manyforge_core (C++ DTOs, kernel, SM, policies, cache) |
| **C++ extras** | `libfmt-dev` (if used), `clang-format` | Formatting, diagnostics |
| **Python** | `python3`, `python3-pip`, `python3-venv` | manyforge_behavior, manyforge_planning |
| **Python packages** | `pytest`, `numpy`, `py_trees`, `pybind11` | Testing, behavior trees, C++/Python bridge |
| **Python bindings** | `nanobind`, `scikit-build-core` | RoboPlan's binding system (nanobind, not pybind11) |
| **Pinocchio** | `pip install pin==3.7.0` (pre-built aarch64 wheel) | Robot kinematics/dynamics/collision (RoboPlan dep) |
| **RoboPlan** | Built from source (`github.com/open-planning/roboplan`) | Planning backend (IK, RRT, TOPPRA, collision) |
| **Viser** | `pip install viser>=1.0.1` | Visualization (HMI/Composer reference, v0.0 UI) |
| **General** | `curl`, `git`, `ca-certificates` | Tooling |

### Not required in v0.0 sandbox

| Package | Why deferred |
|---------|-------------|
| ROS 2 Jazzy | ROS wrapper nodes are post-v0.0 |
| MoveIt 2 | Alternative planner backend, not reference |
| Node.js/React | HMI uses Viser-only in v0.0; React shell deferred to v0.1+ |
| Docker-in-Docker | No container orchestration in sandbox |

### How to rebuild the sandbox image

RoboPlan is not on PyPI — it must be built from source inside the Dockerfile.
Pinocchio has pre-built aarch64 wheels. See `MANYFORGE_STARTUP.md` for the
full step-by-step process.

**Key gotchas:**
1. RoboPlan must be pre-cloned on the host (`git clone --recursive` fails inside Docker)
2. Strip `.git` directories from the clone before copying to build context
   (OpenShell tar has a 100-char path limit)
3. Copy as `rp/` (short name) — the Dockerfile `COPY rp/ /tmp/roboplan/` restores
   the canonical name inside the container
4. Pinocchio (cmeel package) installs cmake files and shared libs under
   `/usr/local/lib/python3.11/dist-packages/cmeel.prefix/` — the Dockerfile
   discovers this at build time and sets `CMAKE_PREFIX_PATH` and `LD_LIBRARY_PATH`
5. RoboPlan's tests require GMock — install `libgmock-dev` and build from
   `/usr/src/googletest` (not `/usr/src/gtest`)

After rebuilding, restore provider config:

```bash
cd ~/workspaces/nemoclaw/src/NemoClaw-Thor
./configure-local-provider.sh qwen3.5-27b-fp8
./status.sh qwen3.5-27b-fp8
```

**Important:** Rebuilding the sandbox destroys all state. Download any in-progress
work before rebuilding. Verify all packages are installed before dispatching agents.

### Verification after rebuild

```bash
CLUSTER=$(docker ps --format '{{.Names}}' | grep openshell-cluster)
docker exec "$CLUSTER" kubectl -n openshell exec thor-assistant -- sh -c '
  echo "=== C++ ===" && g++ --version && cmake --version &&
  echo "=== Python ===" && python3 --version &&
  echo "=== Python packages ===" && pip3 list 2>/dev/null | grep -iE "pytest|numpy|py.trees|pybind11|viser|roboplan" &&
  echo "=== Node.js ===" && node --version && npm --version
'
```

## Validated State

### 2026-03-23 — qwen3.5-35b-a3b-fp8 (multimodal)

Validated on this host:

- Jetson AGX Thor
- JetPack 7.1 / L4T 38.4
- OpenShell `0.0.12`
- OpenClaw `2026.3.11` (29dc654) inside sandbox
- NemoClaw `be7ec09` from upstream repo cloned at `~/NemoClaw`
- local vLLM serving `Qwen3.5-35B-A3B-FP8` (multimodal: text + image + video)
- vLLM image: `ghcr.io/nvidia-ai-iot/vllm:0.16.0-g15d76f74e-r38.2-arm64-sbsa-cu130-24.04`
- OpenShell provider `vllm-local`
- sandbox `thor-assistant`
- policy baseline `strict-local`

Validated runtime parameters:

- `--max-model-len 65536`
- `--kv-cache-dtype fp8`
- `--max-num-seqs 8`
- `--tensor-parallel-size 1`
- auto tool choice with `qwen3_coder` parser
- speculative decoding with `qwen3_next_mtp` (2 speculative tokens)
- **no** `--language-model-only` flag (full multimodal with vision encoder)

What was proven:

- Full reboot recovery cycle: reboot → `configure-local-provider.sh` → all 19
  status checks pass → TUI connects → agent responds to inference
- SSH handshake secret reconciliation after gateway secret rotation on reboot
- OpenClaw gateway auto-start in the correct network namespace
- Device identity pre-pairing for TUI access without manual approval
- Model warmup request during configure to avoid cold-start latency
- Programmatic agent invocation via `openclaw agent --json` from host SSH

### 2026-03-21 — qwen3.5-27b-fp8 (text-only)

Previous validation on the same host with `Qwen3.5-27B-FP8`:

- `--max-num-seqs 16` (higher than 35B due to smaller model)
- tool-use smoke test passed end-to-end
- `openclaw agent --agent main --local` worked without gateway

Known operator limitations:

- The browser dashboard can connect but browser-originated prompts can fail
  with `LLM request timed out.` even when TUI works. This is a Control UI /
  gateway session-state issue, not vLLM throughput.
- Avoid driving the same session from both TUI and dashboard concurrently.
- First inference after model load is slow (30-60s); the warmup request in
  `configure-local-provider.sh` mitigates this.

## Bugs Fixed In This Fork (2026-03-23)

These are the non-obvious issues discovered during live Thor operation. Future
agents should be aware of these because upstream changes may reintroduce them
or change the conditions under which they appear.

### 1. SSH handshake secret drift after reboot

**Symptom:** `nemoclaw thor-assistant connect` fails with "Connection reset by
peer". Sandbox logs show "SSH connection: handshake verification failed".

**Root cause:** After a reboot, the OpenShell gateway pod (`openshell-0`)
regenerates its SSH handshake secret during init. The old
`gateway_ssh_handshake_secret()` function read from the **statefulset spec**,
which lags behind the running pod's actual secret. This caused
`reconcile_sandbox_ssh_handshake_secret()` to falsely report secrets as
matching when they had diverged.

**Fix in `lib/sandbox-runtime.sh`:** Both `gateway_ssh_handshake_secret()` and
`sandbox_ssh_handshake_secret()` now read from the running pod's environment
via `kubectl exec ... printenv OPENSHELL_SSH_HANDSHAKE_SECRET` first, with
fallback to the spec if the pod is not yet exec-ready.

**How to detect:** `./status.sh` checks this under "Native Connect Path". If
it reports a mismatch, run `./configure-local-provider.sh` which reconciles
automatically.

### 2. OpenClaw gateway runs in wrong network namespace

**Symptom:** `openclaw tui` shows "gateway disconnected: closed" even though
`./status.sh` shows the gateway check passing.

**Root cause:** The sandbox has two network namespaces:
- **Pod root namespace** — where `kubectl exec` runs
- **SSH session namespace** — where `nemoclaw connect` / SSH sessions land

The OpenClaw gateway must run in the SSH session namespace because the
`openshell forward` tunnel (host:18789 → sandbox:18789) routes through the SSH
proxy and lands in the SSH namespace. Starting the gateway via `kubectl exec`
puts it in the wrong namespace — the port is listening but unreachable through
the tunnel.

**Fix in `lib/sandbox-runtime.sh`:** `ensure_sandbox_gateway_running()` starts
the gateway via SSH (using `openshell ssh-proxy` as ProxyCommand), not via
`kubectl exec`. This places the gateway in the SSH namespace where the tunnel
can reach it.

**How to detect:** The `status.sh` gateway probe now sends an actual HTTP
request through the tunnel instead of just checking TCP connectivity. A TCP-
only check was a false positive because the SSH tunnel accepts connections even
when nothing is behind it.

### 3. OpenClaw device pairing required after every pod restart

**Symptom:** `openclaw tui` shows "Pairing required. Run openclaw devices list,
approve your request ID, then reconnect."

**Root cause:** OpenClaw's device pairing system requires each TUI client to be
registered in `~/.openclaw/devices/paired.json`. After a pod restart, this file
is lost (ephemeral pod storage). The device identity
(`~/.openclaw/identity/device.json`) persists because it was created during
initial install, but the pairing record does not.

Neither `--auth none` nor `dangerouslyDisableDeviceAuth` in the config disable
the WebSocket-level device pairing check (they only affect transport auth and
the control UI respectively).

**Fix in `lib/sandbox-runtime.sh`:** `ensure_sandbox_gateway_running()` pre-
populates `paired.json` by reading the existing device identity from
`device.json`, extracting the raw Ed25519 public key (stripping the SPKI ASN.1
header, converting to URL-safe base64 without padding), and writing a pre-
approved paired entry before starting the gateway.

**Critical detail:** The public key format in `paired.json` must be the raw
32-byte Ed25519 key in URL-safe base64 without padding (e.g.,
`1xH7ArHVjG1Hn4YO...`), NOT the full SPKI-encoded PEM key (e.g.,
`MCowBQYDK2VwAyEA1xH7...`). Getting this wrong causes the gateway to reject
the device as "not-paired" even though the deviceId matches.

### 4. status.sh gateway check was a false positive

**Symptom:** `./status.sh` reports "OpenClaw gateway is reachable" but TUI
cannot connect.

**Root cause:** The original check used a TCP connect probe to port 18789.
The SSH tunnel itself accepts TCP connections and then forwards them. If nothing
is listening on 18789 inside the sandbox, the tunnel accepts the connection but
the remote end resets or returns empty — the TCP connect still succeeds.

**Fix in `status.sh`:** The probe now sends an HTTP request and checks for an
HTTP response. An empty response, connection reset, or timeout all correctly
report as gateway-not-running.

### 5. Cron directory permissions inside sandbox

**Symptom:** Gateway log shows `EACCES: permission denied, open
'/sandbox/.openclaw/cron/jobs.json'`.

**Root cause:** The `/sandbox/.openclaw/cron/` directory is created by the
sandbox image as root-owned. The gateway runs as the `sandbox` user and cannot
create `jobs.json`.

**Fix in `lib/sandbox-runtime.sh`:** Both `sync_sandbox_runtime_config()` (via
`kubectl exec` as root) and `ensure_sandbox_gateway_running()` (via SSH)
ensure the cron directory exists and is writable, and create an empty
`jobs.json` if missing.

## Current Supported Operator Path

Use this repo's wrapper flow, not raw upstream onboarding:

- `./install.sh`
- `./configure-local-provider.sh`
- `./status.sh`

Do not treat raw upstream `nemoclaw onboard` as the supported Thor workflow.
It still has a Thor-specific failure mode described below.

## What This Repo Is Expected To Do

Expected responsibilities:

- prepare a Jetson Thor host for OpenShell
- keep those host-level Thor fixes reversible with a saved pre-change snapshot
- install NemoClaw
- swap NemoClaw/OpenShell inference to a local provider after upstream onboarding
- launch supported local Qwen profiles with Thor-oriented defaults
- harden sandbox network policy before onboarding
- provide simple checks for install and runtime status

Explicit non-goals:

- implementing the higher-level autonomous coding workflow
- replacing the upstream NemoClaw onboarding UX
- claiming final benchmarked throughput numbers
- exposing external chat channels or personal-account integrations

## Concepts That Drove The Changes

These are the core design constraints behind the fork. If a future change
conflicts with them, the change should be questioned.

### 1. Local-first inference

The original JetsonHacks fork assumes local Nemotron. This fork instead treats
the sibling `thor_llm` workflow as the source of truth for model serving and
targets local Qwen profiles.

Reason:

- the user already has Qwen serving working on Thor
- changing the agent runtime and the model stack at the same time adds noise
- a local OpenAI-compatible endpoint is enough for OpenShell/NemoClaw routing

### 2. Sandbox-first development

The repo is meant for local development inside an OpenShell sandbox, not for
general assistant use.

Reason:

- the user wants repo work to happen in disposable workspaces
- the user does not want broad access to personal accounts or host data
- internet should be denied by default and only widened deliberately

### 3. Reversible host changes

OpenShell on Jetson Thor needs host-level changes. Those changes are not
confined to NemoClaw and can affect other Docker workloads.

Reason:

- Thor requires OpenShell-specific networking and Docker fixes
- those changes are system-wide
- the repo therefore adds host backup/apply/restore scripts instead of treating
  the upstream helper scripts as harmless one-way setup

### 4. Prefer narrow, explicit policy changes

The baseline should start strict and only open up when needed.

Reason:

- the default NemoClaw policy is too broad for the user's intended workflow
- a local coding sandbox should not start with pre-approved broad egress
- temporary research access should be explicit and scoped

### 5. Keep upstream assumptions visible

The repo does not pretend upstream NemoClaw is already local-first.

Reason:

- onboarding is still interactive
- onboarding is still cloud-first
- onboarding still asks for an NVIDIA API key
- this repo works around that by switching inference after onboarding, not by
  fully replacing upstream behavior

## Repo Map

The most important files are:

- `README.md`
  - public-facing current scope and status
- `install.sh`
  - main install entrypoint
- `check-prerequisites.sh`
  - readiness checks
- `status.sh`
  - runtime health check
- `configure-local-provider.sh`
  - creates or reuses the local OpenShell provider and sets inference
- `start-model.sh`
  - main local vLLM launcher
- `lib/config.sh`
  - supported model profiles and saved runtime config
- `lib/launch.sh`
  - per-model vLLM args and defaults
- `lib/sandbox-runtime.sh`
  - SSH handshake reconciliation, sandbox config sync, gateway lifecycle,
    device pre-pairing — this is where most of the non-obvious fixes live
- `lib/policy.sh`
  - static and dynamic policy helpers
- `lib/host-state.sh`
  - backup and restore helpers for Thor host changes
- `apply-host-fixes.sh`
  - wraps JetsonHacks OpenShell-Thor setup with a host snapshot
- `restore-host-state.sh`
  - restores the saved pre-change host state
- `apply-policy-profile.sh`
  - installs static policy into a cloned NemoClaw repo
- `apply-policy-additions.sh`
  - applies dynamic session-scoped policy additions

Upstream leftovers still present:

- `nemotron3-thor.sh`
- `nemotron3-thor-no-thinking.sh`

These are not the preferred path for this fork.

## External Path Assumptions

The fork assumes a sibling model-serving repo relative to this checkout:

- `../thor_llm`

In this workspace, that currently resolves to:

- `/home/tndlux/workspaces/nemoclaw/src/thor_llm`
- `/home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor`

The 122B profile also assumes a pre-resharded local model path under:

- `$HOME/thor-hf-cache/hub/qwen-3.5-122b-a10b-nvfp4-resharded/resharded`

If these assumptions change, update the docs and any helper messages that point
at the sibling `thor_llm` model READMEs.

## Current Runtime Defaults

From `lib/config.sh` and `lib/launch.sh`:

- default model profile: `qwen3.5-27b-fp8`
- supported profiles:
  - `qwen3.5-122b-a10b-nvfp4-resharded`
  - `qwen3.5-27b-claude-distilled-nvfp4`
  - `qwopus3.5-27b-nvfp4`
  - `qwen3.5-27b-fp8`
  - `qwen3.5-35b-a3b-fp8`
  - `qwen3.5-35b-a3b-nvfp4`
  - `gemma4-31b-it-nvfp4`
  - `gemma4-26b-a4b-it`
- default policy profile: `strict-local`
- default provider name: `vllm-local`
- default local provider URL inside OpenShell:
  - `http://host.openshell.internal:8000/v1`
- default host models URL for health checks:
  - `http://127.0.0.1:8000/v1/models`
- default local vLLM API key:
  - `dummy`

Serving defaults across profiles:

- max context: `65536`
- KV cache dtype: `fp8`
- max sequences: `8`

These are conservative startup defaults, not finished throughput claims.

On this validated host, the saved runtime config was overridden to:

- `THOR_TARGET_MAX_MODEL_LEN=65536`
- `THOR_TARGET_KV_CACHE_DTYPE=fp8`
- `THOR_TARGET_MAX_NUM_SEQS=16`

Those values live in `~/.config/nemoclaw-thor/config.env` and are not the repo
defaults for a brand-new machine unless they are exported or saved first.

## Current Known Upstream Thor Caveat

Current upstream NemoClaw still tries to rewrite `~/.openclaw/openclaw.json`
from inside the sandbox during onboarding step 6.

That conflicts with the current sandbox image design, which intentionally makes
`/sandbox/.openclaw/openclaw.json` root-owned and read-only.

Observed behavior on Thor:

- upstream onboarding creates the sandbox successfully
- upstream onboarding creates the provider successfully
- step 6 fails when it tries to rewrite the immutable file

This repo now works around that by:

- allowing upstream onboarding to get as far as sandbox creation
- recovering the sandbox name from `~/.nemoclaw/sandboxes.json`
- syncing the internal NemoClaw/OpenClaw runtime config from the host side in
  `configure-local-provider.sh`
- updating the host-side registry so `nemoclaw <name> status` shows the real
  model id

Operational rule:

- use the Thor wrapper scripts
- do not debug this by repeatedly rerunning raw `nemoclaw onboard`

## Policy Model

Static baseline profiles:

- `strict-local`
  - deny by default, no broad pre-approved external endpoints
- `local-hardened`
  - small compatibility-oriented allowlist, still much tighter than upstream

Dynamic additions:

- `research-lite`
  - session-scoped troubleshooting and research access
  - intended for docs, package registries, and source lookup

Important operational point:

- prefer `strict-local` first
- only widen access if the actual workflow demands it
- prefer one-off approvals with `openshell term` before expanding baseline

## Host-Fix Model

The Thor-specific OpenShell fixes are host-wide and come from the JetsonHacks
OpenShell-Thor helper repo. The important host changes include:

- switching `iptables` to legacy
- loading `br_netfilter`
- writing bridge sysctls
- setting Docker `ipv6` to `false`
- setting Docker `default-cgroupns-mode` to `host`
- building and loading `iptable_raw`

These changes can affect other Docker workloads on the same Thor.

Known example concern:

- Isaac ROS dev containers are less likely than bridge-networked containers to
  break because they run with `--network host`, `--ipc=host`, `--privileged`,
  and `--runtime nvidia`
- but the changes are still system-wide and should not be treated as isolated

## SSH And Remote-Control Reality Check

Do not assume a future session is actually on the Thor just because VS Code is
open or Remote Explorer is configured.

Always verify at the start:

```bash
hostname
uname -a
echo "SSH_CONNECTION=${SSH_CONNECTION:-}"
pwd
```

If the session is still on the laptop, direct SSH access may or may not work.
At one point this tooling context still could not do:

```bash
ssh 192.168.1.136
```

So a future session must verify remote reachability again before promising live
Thor actions.

## Before Touching The Thor Host

If you are about to apply the Thor host fixes, inspect the current firewall and
Docker state first.

Run:

```bash
sudo iptables -S
sudo ip6tables -S
sudo ufw status verbose || true
sudo nft list ruleset || true
sudo cat /etc/docker/daemon.json || true
docker info | sed -n '1,120p'
```

What to watch for:

- `-P INPUT DROP`
- `-P INPUT REJECT`
- active `ufw`
- custom `iptables` rules that appear to protect SSH access
- existing Docker config that should not be silently overwritten

Reason:

- the upstream JetsonHacks setup script flushes `iptables` rules and restarts
  Docker
- that can drop SSH if the host relies on current firewall rules

## Safe Remote Sequence

If running over SSH, treat these differently:

- `./backup-host-state.sh`
  - generally safe
- `./apply-host-fixes.sh`
  - not guaranteed safe in a single SSH session

If firewall state is unknown or restrictive, prefer one of:

- local console on the Thor
- serial console
- a second management path
- `tmux` plus an automatic rollback timer

Do not casually run `./install.sh ... --apply-host-fixes` over a single remote
session without first checking the firewall state.

## Temporary Passwordless sudo For Agent-Driven Install

If a remote coding agent is expected to execute the privileged steps itself,
interactive `sudo` is often not sufficient because the prompt can land in a
background PTY that the user cannot reach.

The clean workaround is a temporary sudoers entry for the local user running
the session.

Create it:

```bash
sudo visudo -f /etc/sudoers.d/nemoclaw-codex
```

Add:

```text
tndlux ALL=(ALL) NOPASSWD:ALL
```

Verify from the agent session:

```bash
sudo -n true
```

Revert it immediately after the install/debug window:

```bash
sudo rm /etc/sudoers.d/nemoclaw-codex
sudo -k
```

This should be treated as temporary privileged access, not a permanent host
configuration.

## What To Monitor During A Real Thor Install

When live access to the Thor is available, monitor in phases.

### Phase 1: Host sanity

Check:

- OS and kernel match expectations for Thor / JetPack 7.1
- Docker is installed and healthy
- NVIDIA runtime is present in Docker
- OpenShell is installed
- `nvm` and Node 22 are available

Useful commands:

```bash
./check-prerequisites.sh qwen3.5-35b-a3b-fp8
docker info
openshell --version
node --version
```

### Phase 2: Host backup and Thor fixes

If applying host fixes:

```bash
./backup-host-state.sh
./apply-host-fixes.sh
```

Watch for:

- `build_ip_table_raw.sh` failures due to missing headers or changed BSP paths
- `setup-openshell-network.sh` failures on `update-alternatives`
- Docker restart failures after daemon config changes
- loss of network reachability

Immediately after:

```bash
update-alternatives --query iptables
lsmod | grep -E 'iptable_raw|br_netfilter'
sudo cat /etc/docker/daemon.json
docker info
```

### Phase 3: NemoClaw install

Run:

```bash
./install.sh qwen3.5-35b-a3b-fp8 --policy-profile strict-local
```

Or, if you already accepted the host-fix risk and want one command:

```bash
./install.sh qwen3.5-35b-a3b-fp8 --policy-profile strict-local --apply-host-fixes
```

Watch for:

- failed prerequisite checks
- Docker restart issues in install step 2
- Node version mismatch after upstream install
- clone or npm install failures
- upstream onboarding prompts that try to push toward cloud or external channels

Important install behavior:

- the install flow replaces the static sandbox policy in the cloned NemoClaw
  repo before upstream onboarding runs
- after onboarding, it switches inference to the local provider

### Phase 4: Local model serve

Start the selected model:

```bash
./start-model.sh qwen3.5-35b-a3b-fp8
```

Watch for:

- image pull failures
- missing Hugging Face credentials for profiles that need them
- wrong served model name
- model OOM or unstable concurrency
- wrong cache paths

Useful checks:

```bash
curl -s -H "Authorization: Bearer dummy" http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

The served model id must match the runtime config expected by
`configure-local-provider.sh` and `status.sh`.

If vLLM is stopped and a different model needs to be loaded, Thor may retain
page cache aggressively. The working recovery command on this host was:

```bash
sudo sync
sudo sysctl -w vm.drop_caches=3
```

Run that before starting the next vLLM server if memory does not come back.

### Phase 5: OpenShell and NemoClaw runtime

Check:

```bash
openshell gateway info
openshell sandbox list
openshell inference get
./status.sh qwen3.5-35b-a3b-fp8
```

Watch for:

- sandbox not reaching `Ready`
- local provider not created
- inference route still pointing at a non-local provider
- host vLLM reachable but serving the wrong model id

## Fresh Host Install Guidance

The validated clean sequence on a fresh Thor is staged:

1. apply host fixes
2. start vLLM
3. run the NemoClaw-Thor installer

Recommended commands:

```bash
export THOR_TARGET_MAX_MODEL_LEN=65536
export THOR_TARGET_KV_CACHE_DTYPE=fp8
export THOR_TARGET_MAX_NUM_SEQS=16
export THOR_OPENCLAW_MAIN_MAX_CONCURRENT=1

./apply-host-fixes.sh
./start-model.sh qwen3.5-35b-a3b-fp8
./install.sh qwen3.5-35b-a3b-fp8 --policy-profile strict-local
```

`install.sh` now ensures the `nemoclaw` OpenShell gateway is running before the
upstream NemoClaw install/onboard step.

OpenClaw concurrency is now written explicitly into the sandbox config during
`configure-local-provider.sh` / `install.sh` sync:

- main lane concurrency defaults to `THOR_OPENCLAW_MAIN_MAX_CONCURRENT=1`
- subagent concurrency is derived as `THOR_TARGET_MAX_NUM_SEQS - effective_main`
- `maxChildrenPerAgent` follows the same derived value
- `maxSpawnDepth` is pinned to `1`
- `./status.sh` now verifies those values inside `/sandbox/.openclaw/openclaw.json`

Why not the single-command `--apply-host-fixes` path for this case:

- `apply-host-fixes.sh` restarts Docker
- restarting Docker stops a vLLM container started earlier
- upstream onboarding only offers local vLLM when the server is already up

The one-command path still exists, but it is not the preferred local-vLLM path
for a clean host.

## How To Use The Current Install

If the current services are still up, use:

```bash
source "$HOME/.nvm/nvm.sh"
nvm use 22
export PATH="$HOME/.local/bin:$PATH"
nemoclaw thor-assistant connect
```

or:

```bash
export PATH="$HOME/.local/bin:$PATH"
openshell sandbox connect thor-assistant
```

If services were stopped or the host rebooted:

```bash
source "$HOME/.nvm/nvm.sh"
nvm use 22
export PATH="$HOME/.local/bin:$PATH"
./start-model.sh qwen3.5-35b-a3b-fp8
./configure-local-provider.sh qwen3.5-35b-a3b-fp8
./status.sh qwen3.5-35b-a3b-fp8
nemoclaw thor-assistant connect
```

`configure-local-provider.sh` is now the normal restart/reboot recovery entry:
it revives the `nemoclaw` gateway if needed, rebinds the local inference route,
and reconciles the sandbox SSH handshake secret so native
`nemoclaw thor-assistant connect` works again.

If the sandbox is missing after restart, rerun:

```bash
./install.sh qwen3.5-35b-a3b-fp8 --policy-profile strict-local
```

If the install is present and only the provider binding needs refresh:

```bash
./configure-local-provider.sh qwen3.5-35b-a3b-fp8
```

## Suggested Tests For Real Use

Use the following progression.

### 1. Stack health

```bash
source "$HOME/.nvm/nvm.sh"
nvm use 22
./status.sh qwen3.5-35b-a3b-fp8
nemoclaw thor-assistant status
```

### 2. Minimal inference

Inside the sandbox:

```bash
openclaw agent --agent main --local \
  -m "Reply with one word: working" --session-id test
```

### 3. Tool-use smoke test

Inside the sandbox:

```bash
openclaw agent --agent main --local --thinking off \
  -m "Run uname -a and python3 --version, write both to /sandbox/smoke.txt, then reply done." \
  --session-id smoke-tools
```

Current reality:

- this now passes on the validated `qwen3.5-27b-fp8` stack
- the working Thor-side config uses `openai-completions`, `parallel_tool_calls=false`, and a reduced tool surface of `read`, `edit`, `write`, `exec`, and `process`
- the working Thor-side config also pins `temperature=0` for the local 27B OpenClaw model override
- if a prior broken run poisoned the embedded local session history, run `./reset-sandbox-session-state.sh qwen3.5-27b-fp8` once before retrying

If you want a direct non-agent fallback inside the sandbox:

```bash
uname -a
python3 --version
```

### 4. Real repo task

Inside the sandbox:

- clone or copy a disposable repo into `/sandbox`
- ask the agent to inspect it
- ask for one tiny code change
- run the repo test command
- inspect the diff and logs

This is a better first real test than broad internet tasks because the default
policy is intentionally local-only.

## What To Verify About Policy

There are two separate concerns:

### Static policy was installed into NemoClaw before onboarding

Verify the cloned NemoClaw repo contains the intended file:

```bash
cd ~/NemoClaw
ls nemoclaw-blueprint/policies
sed -n '1,220p' nemoclaw-blueprint/policies/openclaw-sandbox.yaml
```

What to look for:

- `strict-local` or `local-hardened` content is present
- the upstream file was saved as `openclaw-sandbox.upstream.yaml`

### Runtime policy behaves as expected

Verify behavior through the sandbox:

- no broad outbound access by default under `strict-local`
- `research-lite` only when explicitly applied
- one-off approvals via `openshell term` still work when desired

If the live OpenShell CLI provides policy inspection commands, use them, but do
not assume their output format without checking the installed version first.

## Common Failure Modes

### 1. Session is not actually on the Thor

Symptom:

- commands report laptop hostname or `x86_64`

Action:

- stop assuming remote control
- re-establish actual Thor access before proceeding

### 2. SSH gets dropped during host-fix apply

Cause:

- firewall flush or Docker restart while host depends on current rules

Action:

- reconnect through console or alternate path
- use `./restore-host-state.sh` if needed

### 3. Docker daemon does not come back

Cause:

- conflicting or malformed `/etc/docker/daemon.json`

Action:

```bash
sudo systemctl status docker --no-pager
sudo journalctl -u docker --no-pager | tail -50
```

### 4. `iptable_raw` build breaks

Cause:

- kernel headers path or BSP source layout changed

Action:

- inspect the helper repo scripts
- do not patch blindly; confirm current JetPack and kernel paths first

### 5. NemoClaw installs but `nemoclaw` command is missing

Cause:

- Node version collision or PATH mismatch after upstream install

Action:

- source `nvm`
- confirm `nvm use 22`
- reopen shell if necessary

### 6. Local provider exists but status still warns

Cause:

- vLLM is serving a different model id than the config expects

Action:

- align `--served-model-name`
- or override `THOR_MODEL_ID`

### 7. Sandbox is up but the agent cannot browse

Cause:

- expected behavior under `strict-local`

Action:

- use `openshell term` for one-off approvals
- or apply `./apply-policy-additions.sh research-lite`

## Things To Be Careful Not To Regress

- do not loosen the default policy baseline casually
- do not make host-fix apply look SSH-safe if it is not
- do not reintroduce reliance on external chat bridges
- do not hide the cloud-first upstream onboarding constraint
- do not let the local provider model id drift from the served vLLM model id
- do not treat the Nemotron scripts as the recommended path
- do not silently drop the host backup/restore path from the install story

## Useful Commands

Quick health:

```bash
./check-prerequisites.sh qwen3.5-35b-a3b-fp8
./status.sh qwen3.5-35b-a3b-fp8
```

Host backup and restore:

```bash
./backup-host-state.sh
./apply-host-fixes.sh
./restore-host-state.sh
./restore-host-state.sh --skip-firewall-restore
```

Model serve:

```bash
./start-model.sh qwen3.5-35b-a3b-fp8
./start-model.sh qwen3.5-27b-fp8
./start-model.sh qwen3.5-122b-a10b-nvfp4-resharded
```

Policy operations:

```bash
./apply-policy-profile.sh strict-local
./apply-policy-profile.sh local-hardened
./apply-policy-additions.sh research-lite
```

## If Starting Fresh In A New Session

Recommended order:

1. Read `README.md`.
2. Read this file.
3. Check `git status`.
4. Confirm whether the session is actually on the Thor.
5. If working on install/debug, inspect `install.sh`, `status.sh`,
   `configure-local-provider.sh`, `lib/config.sh`, `lib/launch.sh`,
   `lib/policy.sh`, and `lib/host-state.sh`.
6. If live Thor access is available, start with firewall and Docker inspection
   before applying host fixes.
7. Change one variable at a time and keep the install path observable.

## Programmatic Agent Access From Host

### Preferred method: kubectl exec (validated 2026-03-30)

kubectl exec runs in the pod's root network namespace, where the restream
proxy (port 8199) and vLLM (host.openshell.internal:8000) are both reachable.
This avoids the network namespace isolation problems that affect SSH.

```bash
# Get the cluster container name
CLUSTER=$(docker ps --format '{{.Names}}' | grep openshell-cluster)

# Run a single agent task
docker exec -i "$CLUSTER" \
  kubectl -n openshell exec thor-assistant -- \
    env HOME=/sandbox \
    openclaw agent --agent main --local --json \
      --session-id my-task \
      -m "your prompt here" \
      --timeout 1800
```

### Session cleanup between tasks

OpenClaw maps all `--session-id` values to a single internal key. Previous
session state contaminates new tasks. **Clear between tasks:**

```bash
docker exec -i "$CLUSTER" \
  kubectl -n openshell exec thor-assistant -- \
    sh -c 'rm -f /sandbox/.openclaw/agents/main/sessions/*.jsonl && \
           echo "{}" > /sandbox/.openclaw/agents/main/sessions/sessions.json'
```

### Workspace file cleanup

OpenClaw auto-generates SOUL.md, TOOLS.md, etc. in the working directory on
each run, inflating the system prompt. **Delete before each agent run:**

```bash
docker exec -i "$CLUSTER" \
  kubectl -n openshell exec thor-assistant -- \
    sh -c 'cd /sandbox/work/<task-dir> && rm -f SOUL.md TOOLS.md AGENT.md AGENTS.md 2>/dev/null; true'
```

### Legacy method: SSH

SSH still works for the TUI and interactive use, but runs in a separate network
namespace. The restream proxy on port 8199 is **not reachable from SSH** — SSH
sessions can only reach the gateway on port 18789. For programmatic agent
invocation, always prefer kubectl exec.

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    -o "ProxyCommand=openshell ssh-proxy --gateway-name nemoclaw --name thor-assistant" \
    sandbox@openshell-thor-assistant \
    'HOME=/sandbox openclaw agent --agent main --local --json --session-id my-task \
      -m "your prompt here" --timeout 120'
```

Key points:

- `--session-id` maintains conversation state across calls
- `--json` returns structured output for parsing
- `--local` runs the embedded agent (no gateway needed for agent CLI)
- Max concurrency: 1 main agent + 7 subagents (configured in openclaw.json)
- The `sandbox_ssh_command` helper in `lib/sandbox-runtime.sh` wraps the SSH
  proxy setup

## Re-Streaming Proxy (deprecated in v4)

**v4 status:** The restream proxy is no longer needed. vLLM 0.19 fixes the
qwen3_coder streaming parser bug (PR #35615) that required the proxy.
Additionally, OpenClaw v2026.4.1 has built-in tool-call repair that
reconstructs partial tool arguments on streamed calls.

The v4 inference path is: OpenClaw → inference.local → OpenShell provider → vLLM.
No proxy in the data path.

The proxy code (`lib/restream-proxy.py`) is retained in the repo for reference.

### Timeout chain (v4)

With the proxy removed, the timeout chain simplifies:

| Layer | Value | Controls |
|-------|-------|----------|
| `--timeout` (agent dispatch) | 1800s | Total session wall-clock time |
| `timeoutSeconds` (openclaw.json) | 1800s | Per-LLM-request timeout (SSE keeps alive) |

Set both to at least 1800s for coding tasks.

### Historical context (v3)

In v3, the restream proxy (`lib/restream-proxy.py`) sat inside the sandbox pod
on `127.0.0.1:8199` and buffered tool-call argument fragments to work around
vLLM's streaming parser bug. Before that, the nonstream proxy used
`stream: false` which caused timeouts and suppressed thinking mode. Both
approaches are superseded by vLLM 0.19's native fix.

## ManyForge Development Rules

### Supervisor (Claude Code / Opus) rules

**ALLOWED:**
- Assess agent output quality (diff review, test results, file boundaries)
- Run build/test commands on host for final verification
- Write and upload TASK.md prompts
- Delete stale files from sandbox before new sessions
- Call `/codex:review` for independent second opinion on agent output
- Direct intervention (write code) ONLY after 2 failed agent rounds

**NOT ALLOWED:**
- Write implementation code in rounds 1-2 (send errors back to the agent)
- Modify agent-produced code (send errors back, never edit model output)
- Manage individual coding tasks (describe what to fix, not how)
- Skip quality gates to save time

### Agent rules

**MUST:**
- Work only within declared file boundaries (listed in TASK.md)
- Run build + test and iterate until all tests pass
- Report "TASK COMPLETE" with RESULT.md, FILES.txt, TEST_LOG.txt
- Report "STUCK: <description>" after 3 failed attempts on the same error

**MUST NOT:**
- Modify frozen contract headers (types/, ports/, time/iclock.hpp, time/ischeduler.hpp)
- Invent new interfaces, DTOs, or port abstractions not in the specs
- Include ROS headers in manyforge_core

### Contract freeze rules

Once frozen contracts are committed (Phase 0.5):
- No agent may modify any file under `include/manyforge_core/types/`,
  `include/manyforge_core/ports/`, or the IClock/IScheduler interfaces
- If a contract bug is found, supervisor pauses all affected tasks, fixes
  the contract on host, re-uploads to all active workers, then resumes
- Contract headers are uploaded to `/sandbox/contracts/` AND included in
  each worker's repo copy

### Pre-clean before sessions

Before dispatching any agent session, delete stale files from previous sessions
to prevent agents from reading old code:

```bash
docker exec "$CLUSTER" kubectl -n openshell exec thor-assistant -- \
  rm -rf /sandbox/work/<task-id>/
```

## ManyForge Agent Orchestration Workflow (Complete Reference)

This is the end-to-end workflow for dispatching coding/testing tasks to the
local Qwen agent on the NemoClaw sandbox. Claude Code (Opus) acts as supervisor:
sets up the workspace, writes the prompt, dispatches, monitors, quality-gates,
and merges. The agent does the implementation.

**Validated 2026-03-31:** Session 5 round 2 used this exact workflow to fix 2
bugs and expand test coverage from 16 to 43 tests in ~14 minutes, fully
autonomous (main agent + 2 subagents).

### Prerequisites

```bash
# Store the cluster container name (used in all commands below)
CLUSTER=$(docker ps --format '{{.Names}}' | grep openshell-cluster)

# Verify the sandbox is running
docker exec "$CLUSTER" kubectl -n openshell get pods

# Verify vLLM is healthy
docker exec "$CLUSTER" kubectl -n openshell exec thor-assistant -- \
  curl -s http://host.openshell.internal:8000/health

# Verify OpenClaw agent is reachable
docker exec "$CLUSTER" kubectl -n openshell exec thor-assistant -- \
  openclaw version
```

### Step 1: Prepare the workspace

Create the task workspace in the sandbox. For Python tasks, upload the
dependency packages alongside the task package.

```bash
# Create workspace directory
docker exec -i "$CLUSTER" kubectl -n openshell exec -i thor-assistant -- \
  mkdir -p /sandbox/work/<task-id>/

# Upload dependency packages (e.g. manyforge_behavior for Session 5)
# Use openshell sandbox upload OR pipe through kubectl exec:
docker exec -i "$CLUSTER" kubectl -n openshell exec -i thor-assistant -- \
  sh -c 'cat > /tmp/upload.tar' < /tmp/local-archive.tar
docker exec -i "$CLUSTER" kubectl -n openshell exec -i thor-assistant -- \
  sh -c 'cd /sandbox/work/<task-id>/ && tar xf /tmp/upload.tar'

# Alternative: openshell sandbox upload (simpler for directories)
openshell sandbox upload thor-assistant ./local-dir/ /sandbox/work/<task-id>/dir/
```

### Step 2: Write and upload the task prompt

Write a TASK.md file locally, then upload it. The prompt should contain:

- **Task description**: What the agent should build/fix
- **Interface/spec excerpts**: Frozen contracts the code must implement
- **Build & test commands**: Exact commands including PYTHONPATH or cmake flags
- **File boundaries**: Which files may be created/modified
- **Conventions**: Naming, error handling, test organization
- **Subagent hints** (optional): How to split work across subagents
- **Output requirements**: RESULT.md, FILES.txt, TEST_LOG.txt

**Example TASK.md structure** (from Session 5 round 2):

```markdown
# Task: Fix bugs and expand test coverage for manyforge_planning

## Workspace
All code is in `/sandbox/work/s5-planning/manyforge_planning/`.

## Bugs to fix
### Bug 1: <title>
<Description of the bug, where it is, what the fix should achieve>

### Bug 2: <title>
<Description>

## Missing test coverage
### <Category>
- <Test scenario description> → <expected behavior>

## How to organize work
You have access to subagents. Consider splitting:
- One subagent for bug fixes
- Others for new tests

## Build and test
\```bash
cd /sandbox/work/s5-planning
PYTHONPATH="..." python -m pytest manyforge_planning/tests/ -v --tb=short
\```

Iterate until ALL tests pass. Target: at least 30 tests.

## Rules
- Do not modify <frozen files>
- Keep test files organized: <list>
```

Upload the prompt:

```bash
# Write TASK.md locally first, then upload
docker exec -i "$CLUSTER" kubectl -n openshell exec -i thor-assistant -- \
  sh -c 'cat > /sandbox/work/<task-id>/TASK.md' < /tmp/my-task-prompt.md
```

### Step 3: Clean session state

**Always clean between tasks.** OpenClaw reuses session state even with
different `--session-id` values.

```bash
docker exec -i "$CLUSTER" kubectl -n openshell exec -i thor-assistant -- \
  sh -c 'rm -f /sandbox/.openclaw/agents/main/sessions/*.jsonl && \
         echo "{}" > /sandbox/.openclaw/agents/main/sessions/sessions.json'

# Also clean auto-generated files from workspace
docker exec -i "$CLUSTER" kubectl -n openshell exec -i thor-assistant -- \
  sh -c 'cd /sandbox/work/<task-id> && rm -f SOUL.md TOOLS.md AGENT.md AGENTS.md 2>/dev/null; true'
```

### Step 4: Dispatch the agent

```bash
docker exec -i "$CLUSTER" kubectl -n openshell exec -i thor-assistant -- \
  env HOME=/sandbox \
  openclaw agent --agent main --local --json \
    --session-id <task-id> \
    --timeout 1800 \
    -m "Read /sandbox/work/<task-id>/TASK.md and complete the task. \
You should use subagents to parallelize the work. \
Coordinate them, verify their work, and run the final test suite. \
Report TASK COMPLETE when done."
```

Key flags:
- `--timeout 1800` — 30 min session wall-clock (sufficient for most tasks)
- `--json` — structured output (useful for automated parsing)
- `--local` — runs the embedded agent, not through the gateway
- Short `-m` message referencing TASK.md avoids shell quoting issues

For background dispatch from Claude Code, pipe to tee:

```bash
docker exec -i "$CLUSTER" kubectl -n openshell exec -i thor-assistant -- \
  env HOME=/sandbox \
  openclaw agent --agent main --local --json \
    --session-id <task-id> \
    --timeout 1800 \
    -m "Read /sandbox/work/<task-id>/TASK.md and complete the task." \
  2>&1 | tee /tmp/<task-id>-output.json
```

### Step 5: Monitor progress

Monitor every 2-3 minutes while the agent runs. Check these signals:

**Agent process** — verify it's still running:
```bash
docker exec "$CLUSTER" kubectl -n openshell exec thor-assistant -- \
  ps aux | grep openclaw-agent
```

**Run tests from outside** — check current state without disturbing agent:
```bash
# For Python packages:
docker exec "$CLUSTER" kubectl -n openshell exec thor-assistant -- \
  sh -c 'cd /sandbox/work/<task-id> && \
  PYTHONPATH="/sandbox/work/<task-id>/dep1:/sandbox/work/<task-id>/dep2:$PYTHONPATH" \
  /sandbox/pyenv/bin/python -m pytest <package>/tests/ -v --tb=short 2>&1 | tail -50'

# For C++ packages:
docker exec "$CLUSTER" kubectl -n openshell exec thor-assistant -- \
  sh -c 'cd /sandbox/work/<task-id>/build && ctest --output-on-failure 2>&1'
```

**Session file size** — rough progress indicator (~10KB/min):
```bash
docker exec "$CLUSTER" kubectl -n openshell exec thor-assistant -- \
  ls -la /sandbox/.openclaw/agents/main/sessions/<task-id>.jsonl
```

**Modified files** — what has the agent changed:
```bash
docker exec "$CLUSTER" kubectl -n openshell exec thor-assistant -- \
  find /sandbox/work/<task-id>/<package>/ -name "*.py" -newer /sandbox/work/<task-id>/TASK.md -type f
```

**Stuck detection heuristics:**
- No proxy log entries for >5 minutes → check if openclaw-agent is alive
- openclaw-agent gone but no output → session completed or crashed
- Session file not growing → agent likely idle or stuck in a loop
- Same `tool_call: exec` repeated 3+ times → agent looping on a build error

### Step 6: Download results

After the agent process exits, download the modified files:

```bash
# Download the entire workspace
mkdir -p /tmp/<task-id>-results/
for f in <list of files to download>; do
  docker exec "$CLUSTER" kubectl -n openshell exec thor-assistant -- \
    cat /sandbox/work/<task-id>/$f > /tmp/<task-id>-results/$(basename $f)
done

# Or download the entire package directory (tar method):
docker exec "$CLUSTER" kubectl -n openshell exec thor-assistant -- \
  sh -c 'cd /sandbox/work/<task-id> && tar cf - <package>/' | \
  tar xf - -C /tmp/<task-id>-results/

# Also grab the agent's report files:
for f in RESULT.md FILES.txt TEST_LOG.txt; do
  docker exec "$CLUSTER" kubectl -n openshell exec thor-assistant -- \
    cat /sandbox/work/<task-id>/$f > /tmp/<task-id>-results/$f 2>/dev/null
done
```

### Step 7: Quality gate

Review the output before merging. This is the most important step.

**7a. Diff against baseline:**
```bash
# If you saved a baseline before dispatch:
diff -r /tmp/<task-id>-baseline/ /tmp/<task-id>-results/

# Or diff individual files:
diff /tmp/<task-id>-baseline/file.py /tmp/<task-id>-results/file.py
```

**7b. Review the diff for:**
- Bug fixes are correct and minimal (no unrelated changes)
- New code follows project conventions
- Tests cover the right scenarios
- No security issues (hardcoded secrets, injection vectors)
- Frozen interfaces untouched
- Mock changes minimal (if any)

**7c. Run tests independently** (from sandbox or host):
```bash
# Python:
docker exec "$CLUSTER" kubectl -n openshell exec thor-assistant -- \
  sh -c 'cd /sandbox/work/<task-id> && \
  PYTHONPATH="..." /sandbox/pyenv/bin/python -m pytest <package>/tests/ -v --tb=short'

# C++:
cd /path/to/host/repo && mkdir -p build && cd build && \
  cmake .. -DBUILD_TESTING=ON && make -j$(nproc) && ctest --output-on-failure
```

**7d. Check file scope** — verify only expected files were modified:
```bash
docker exec "$CLUSTER" kubectl -n openshell exec thor-assistant -- \
  cat /sandbox/work/<task-id>/FILES.txt
```

**7e. Codex review (optional but recommended):**

Request an independent second opinion from OpenAI Codex on the agent's output.
This catches issues that the supervisor's own review may miss — different model,
different biases, different blind spots.

```
/codex:review
```

Codex reviews the current working tree diff and returns structured feedback.
Use this after steps 7a–7d pass — Codex review is a second-opinion layer, not
a replacement for the supervisor's own diff review and test verification.

If Codex flags issues: treat them as quality gate failures and proceed to Step 9
(revision round) with the Codex feedback included in the fix prompt.

**Quality gate checklist:**
- [ ] Diff reviewed — changes match task description
- [ ] File scope respected — no unexpected files modified
- [ ] Frozen interfaces untouched
- [ ] All tests pass
- [ ] No forbidden patterns (ROS headers in core, hardcoded secrets, etc.)
- [ ] Codex review passed (if run) — no blocking issues flagged

### Step 8: Merge to host repo

After quality gate passes, copy the results to the host repo and commit.

```bash
# Copy package to host repo
cp -r /tmp/<task-id>-results/<package>/ ~/workspaces/dev_ws/src/manyforge/<package>/

# Verify on host
cd ~/workspaces/dev_ws/src/manyforge/
# For Python: run pytest with appropriate PYTHONPATH
# For C++: cmake build + ctest

# Commit
cd ~/workspaces/dev_ws/src/manyforge/
git add <package>/
git commit -m "Session <N>: <description>

<N> tests passing. Agent-produced, supervisor quality-gated.

Co-authored-by: Qwen3.5-27B-Claude-Distilled <local@thor>"
```

### Step 9: If quality gate fails — revision round

If the quality gate finds issues, start a new agent session with fix instructions:

```bash
# Write a fix prompt referencing specific errors
# Include Codex feedback if /codex:review was run in Step 7e
cat > /tmp/<task-id>-fix-prompt.md << 'EOF'
# Task: Fix issues found in quality gate

## Issues to fix
1. <specific error with file:line reference>
2. <specific error>
3. <Codex review finding, if applicable>

## Build and test
<same commands as original>

## Rules
<same rules as original>
EOF

# Upload and dispatch (reuse workspace, new session-id)
docker exec -i "$CLUSTER" kubectl -n openshell exec -i thor-assistant -- \
  sh -c 'cat > /sandbox/work/<task-id>/TASK.md' < /tmp/<task-id>-fix-prompt.md

# Clean session state
docker exec -i "$CLUSTER" kubectl -n openshell exec -i thor-assistant -- \
  sh -c 'rm -f /sandbox/.openclaw/agents/main/sessions/*.jsonl && \
         echo "{}" > /sandbox/.openclaw/agents/main/sessions/sessions.json'

# Dispatch
docker exec -i "$CLUSTER" kubectl -n openshell exec -i thor-assistant -- \
  env HOME=/sandbox \
  openclaw agent --agent main --local --json \
    --session-id <task-id>-fix \
    --timeout 1800 \
    -m "Read /sandbox/work/<task-id>/TASK.md and fix the issues."
```

Max 2 revision rounds. If still failing after round 2, supervisor edits code
directly on host (direct intervention).

### Worked Example: Session 5 Round 2 (2026-03-31)

**Task:** Fix 2 bugs in manyforge_planning, expand tests from 16 to 30+.

**Workspace setup:**
- `/sandbox/work/s5-planning/manyforge_behavior/` — dependency (KernelClient interface)
- `/sandbox/work/s5-planning/manyforge_planning/` — the package under development
- `/sandbox/work/s5-planning/TASK.md` — 93-line prompt describing bugs and test gaps

**Prompt style:** Hint-driven. Described what each bug was and what test scenarios
to cover, without providing implementation code. The agent figured out the fixes
and test implementations autonomously.

**Timeline:**
- 0:00 — Agent starts, reads 5 files (roboplan_client.py, test files, conftest.py)
- 1:30 — Runs existing tests (16 pass)
- 3:00 — Spawns 2 subagents via `sessions_spawn`
- 5:00 — First edit to roboplan_client.py (3683 bytes — bug fixes)
- 6:00 — Second edit to roboplan_client.py (3110 bytes — quaternion conversion)
- 7:00 — Tests: 37 collected, 36 pass, 1 fail (invalid frame test)
- 9:00 — Edits roboplan_mock.py (2-line fix for frame validation)
- 10:00 — Creates test_validation.py (10KB exec — 18 new tests)
- 12:00 — Tests: 43 collected, 43 pass
- 14:00 — Session complete. 138KB session file.

**Quality gate result:** PASS. Both bugs correctly fixed (quaternion extraction,
"universe" parent frame). 27 new tests covering all requested scenarios. Mock
change was minimal (2 lines). No frozen interfaces touched.

## Codex Review Integration

The v4 workflow adds an optional dual-reviewer quality gate using the
`codex-plugin-cc` Claude Code plugin. This provides an independent second
opinion from OpenAI Codex on agent-produced code.

### How it works

1. **Local agent produces code** — Qwen model on NemoClaw sandbox, dispatched
   via kubectl exec as described in the orchestration workflow above.
2. **Supervisor reviews** — Claude Code (Opus) reviews the diff, runs tests,
   checks file scope (Steps 7a–7d).
3. **Codex reviews** — Supervisor calls `/codex:review` to get an independent
   assessment from a different model with different training biases.
4. **Supervisor synthesizes** — Claude Code evaluates both its own review and
   Codex's feedback, decides whether the quality gate passes or fails.
5. **Iterate if needed** — If issues are found, the supervisor writes a fix
   prompt incorporating feedback from both reviews and dispatches a revision
   round (Step 9).

### Available commands

| Command | Purpose | When to use |
|---------|---------|-------------|
| `/codex:review` | Review working tree changes | Per-task quality gate (Step 7e) |
| `/codex:adversarial-review` | Red-team review: challenges design choices, assumptions, failure modes | Once at the end of a major implementation or refactor phase |
| `/codex:rescue` | Delegate investigation or fix to Codex (write-capable) | Emergency only — when neither agents nor supervisor can solve it |
| `/codex:task <prompt>` | Run an arbitrary Codex task | Ad-hoc deeper analysis |
| `/codex:status` | Check status of running Codex tasks | — |
| `/codex:cancel <id>` | Cancel a running Codex task | — |

### When to use each review level

- **`/codex:review`** (standard) — per-task second opinion during Step 7e.
  Recommended for tasks touching frozen interfaces, security-sensitive code, or
  complex logic. Optional for straightforward bug fixes or test additions.
- **`/codex:adversarial-review`** — full red-team pass that tries to break
  confidence in the implementation. Run once at the end of a major phase (e.g.,
  after completing all sessions in a ManyForge phase), not on individual tasks.
  Focuses on auth/trust boundaries, data loss, race conditions, rollback safety,
  migration hazards, and observability gaps.
- **`/codex:rescue`** — last resort. Only use when the local agent has failed
  its revision rounds AND the supervisor cannot diagnose or fix the issue. Codex
  gets write access and can edit files directly. Do not use as a convenience
  shortcut for work that agents or the supervisor should handle.

### When NOT to use Codex

- Do not use any Codex command as a replacement for the supervisor's own diff
  review and test runs. Codex is a second-opinion layer, not a substitute for
  Steps 7a–7d.
- Do not call `/codex:review` on every minor iteration within a revision round —
  reserve it for the final output of each round.
- Do not call `/codex:adversarial-review` on individual tasks — save it for
  phase-level validation.
- Do not reach for `/codex:rescue` before exhausting the agent's 2 revision
  rounds and the supervisor's own direct intervention option.

### Requirements

- `codex-plugin-cc` installed in Claude Code (`npm install -g @openai/codex`)
- Codex authenticated (`codex login` via ChatGPT or API key)
- Claude Code running as CLI or desktop app (not VSCode extension — the extension
  does not support plugin hooks)

### Review gate (optional)

The plugin supports a stop-review gate that can block session stops until Codex
approves. This is disabled by default and is **not recommended** for the
ManyForge workflow — it fires on every session stop, which is too frequent for
iterative agent dispatching. Instead, call `/codex:review` manually at the
supervisor's discretion during Step 7e.

To enable if desired: `/codex:setup --enable-review-gate`

## Validated Agent Run (Run 4, 2026-03-30)

Model: `Qwen3.5-27B-Claude-Distilled-NVFP4` on Jetson AGX Thor

Infrastructure: kubectl exec + session cleanup between tasks
(Run 4 used nonstream proxy; runs after 2026-03-31 used restream proxy;
v4 connects directly to vLLM — proxy eliminated)

Results (manyforge_core Phase 1, 5 tasks):
- 450 implementation lines, 45 Phase 1 test cases, 52 total tests
- 1 bug (use-after-erase UB in trajectory_cache.cpp)
- 0 fix cycles — each task completed first try
- ~5 min per task wall time
- ~$2-5 Claude Opus API cost (supervisor dispatching only)

This validates the full autonomous agent pipeline: write task prompt, kubectl
exec, download results, verify on host, commit. Zero human intervention needed
per task.

## Version Management Strategy

This repo does not pin upstream versions. Instead:

1. **Document what was validated** — the "Validated State" section records exact
   versions, commits, and image tags that were tested together.
2. **Document what broke and why** — the "Bugs Fixed" section explains the root
   causes so future agents can recognize when upstream changes reintroduce
   similar issues.
3. **Treat scripts as adaptation guides** — the install and configure scripts
   encode the integration logic, but a new install on a newer upstream may need
   the scripts adjusted. A frontier coding model working interactively is the
   recommended install path.
4. **Change one component at a time** — update NemoClaw, OpenClaw, or OpenShell
   separately and verify with `./status.sh` after each.

Components that can update independently:

| Component | Update trigger | Risk |
|-----------|---------------|------|
| NemoClaw (`~/NemoClaw`) | `git pull` | Install scripts, onboarding flow |
| OpenClaw (in sandbox) | Sandbox image rebuild or `openclaw update` | Config format, pairing, gateway API |
| OpenShell (host binary) | Manual binary update | SSH proxy, sandbox CRD, K3s internals |
| vLLM image | Changed in `lib/launch.sh` | Model loading, GPU memory, API compat |

## Current Honest Summary

This repo is validated end-to-end on Jetson AGX Thor with `qwen3.5-27b-fp8`,
`qwen3.5-27b-claude-distilled-nvfp4`, and `qwen3.5-35b-a3b-fp8` profiles.
Gemma 4 profiles (`gemma4-31b-it-nvfp4`, `gemma4-26b-a4b-it`) are added but
not yet validated in full agent orchestration sessions.

The full reboot recovery cycle works: reboot the device, run
`./configure-local-provider.sh`, and the TUI connects with working inference.

The integration layer handles several non-obvious issues that arise from
running NemoClaw/OpenShell on Jetson Thor with local inference: SSH handshake
secret rotation, network namespace isolation for the OpenClaw gateway, device
pairing persistence across pod restarts, and accurate health checking. The v3
restream proxy (which worked around vLLM streaming tool_call_parser bugs) has
been eliminated — vLLM 0.19 fixes the underlying issue natively.

What is ready for use:

- local-first inference with hardened sandbox policy
- automated reboot recovery
- programmatic agent access from the host (kubectl exec preferred, SSH for TUI)
- direct vLLM tool calls with thinking mode (no proxy needed in v4)
- autonomous agent tasks via kubectl exec with zero-touch per task
- multimodal inference (Qwen3.5 vision encoder active on 35B profile)
- dual-reviewer quality gate via Codex plugin (optional)

What is not yet built:

- automated upstream update testing
- persistent workspace volumes across pod restarts
- Gemma 4 profile validation in full agent sessions
