# Agent Notes

This file is a session handoff and operational note for future work on this
repository. It is intentionally more explicit and more candid than the README.

For the operator-facing usage guide, see `USER_QUICKSTART_MANUAL.md`.

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
  - `qwen3.5-27b-fp8`
  - `qwen3.5-35b-a3b-fp8`
  - `qwen3.5-35b-a3b-nvfp4`
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

kubectl exec runs in the pod's root network namespace, where the nonstream
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
      --timeout 600
```

### Session cleanup between tasks

OpenClaw maps all `--session-id` values to a single internal key. Previous
session state contaminates new tasks. **Clear between tasks:**

```bash
docker exec -i "$CLUSTER" \
  kubectl -n openshell exec thor-assistant -- \
    sh -c 'rm -f /sandbox/.openclaw-data/agents/main/sessions/*.jsonl && \
           echo "{}" > /sandbox/.openclaw-data/agents/main/sessions/sessions.json'
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
namespace. The nonstream proxy on port 8199 is **not reachable from SSH** — SSH
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

## Non-Streaming-to-SSE Proxy

The nonstream proxy (`lib/nonstream-proxy.py`) is the key enabler for reliable
tool calls with vLLM's qwen3_coder tool_call_parser.

**Problem:** vLLM's streaming `qwen3_coder` parser has an IndexError bug that
corrupts large tool-call arguments across SSE chunks. This causes empty
`arguments` fields and broken file writes — the dominant failure mode in agent
runs.

**Solution:** The proxy sits inside the sandbox pod on `127.0.0.1:8199`:
1. Receives streaming requests from OpenClaw
2. Sends `stream: false` + `chat_template_kwargs: {enable_thinking: false}` to vLLM
3. Waits for the complete response
4. Converts it to chunked SSE format that OpenClaw expects

Tool calls arrive intact regardless of argument size. No JS patching of
OpenClaw's provider code is needed — the `baseUrl` in openclaw.json points
directly to `http://127.0.0.1:8199/v1`.

**Deployed by:** `sync_sandbox_runtime_config()` in `lib/sandbox-runtime.sh`.
The proxy source is written inline and started as a background process.

## Validated Agent Run (Run 4, 2026-03-30)

Model: `Qwen3.5-27B-Claude-Distilled-NVFP4` on Jetson AGX Thor

Infrastructure: kubectl exec + nonstream proxy + session cleanup between tasks

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
The full reboot recovery cycle works: reboot the device, run
`./configure-local-provider.sh`, and the TUI connects with working inference.

The integration layer handles several non-obvious issues that arise from
running NemoClaw/OpenShell on Jetson Thor with local inference: SSH handshake
secret rotation, network namespace isolation for the OpenClaw gateway, device
pairing persistence across pod restarts, accurate health checking, and vLLM
streaming tool_call_parser bugs (bypassed by the nonstream proxy).

What is ready for use:

- local-first inference with hardened sandbox policy
- automated reboot recovery
- programmatic agent access from the host (kubectl exec preferred, SSH for TUI)
- non-streaming-to-SSE proxy for reliable tool calls
- autonomous agent tasks via kubectl exec with zero-touch per task
- multimodal inference (Qwen3.5 vision encoder active on 35B profile)

What is not yet built:

- automated upstream update testing
- persistent workspace volumes across pod restarts
