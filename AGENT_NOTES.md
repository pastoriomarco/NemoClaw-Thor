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

## Validated State As Of 2026-03-21

The repo is no longer just a draft for the `qwen3.5-27b-fp8` coding path. A
full Thor run was completed and verified on the target machine.

Validated on this host:

- Jetson AGX Thor
- JetPack 7.1 / L4T 38.4
- OpenShell `0.0.12`
- NemoClaw installed from the upstream repo cloned at `~/NemoClaw`
- local vLLM serving `Qwen3.5-27B-FP8`
- OpenShell provider `vllm-local`
- sandbox `thor-assistant`
- policy baseline `strict-local`

Validated runtime parameters for that run:

- `--max-model-len 65536`
- `--kv-cache-dtype fp8`
- `--max-num-seqs 16`
- auto tool choice enabled with `qwen3_coder`

What was actually proven:

- Thor host fixes applied successfully
- OpenShell gateway started successfully
- sandbox image built and sandbox reached `Ready`
- local provider route set to `Qwen3.5-27B-FP8`
- `./status.sh qwen3.5-27b-fp8` passed all checks
- inside the sandbox, `openclaw agent --agent main --local --thinking off -m "Reply with one word: working"` returned `working`
- inside the sandbox, the tool-use smoke test now completes end-to-end when the embedded session store is clean

What is not currently proven:

- every other model profile from this repo on this host
- stable tool-use behavior on the `qwen3.5-35b-a3b-fp8` path

Known operator limitation as of this session:

- `openclaw tui` requires the OpenClaw gateway inside the sandbox. The gateway
  normally auto-starts. If the TUI shows "gateway disconnected: closed | idle",
  start it manually: `HOME=/sandbox openclaw gateway run &`
- `openclaw agent --agent main --local ...` works without the gateway.
- The browser dashboard can connect and is useful for visibility, but
  browser-originated prompts can still fail with `LLM request timed out.` even
  when the same prompt works in TUI.
- The current evidence points to a Control UI / gateway session-state issue,
  not a raw vLLM throughput problem.
- Avoid driving the same session from both TUI and dashboard concurrently.

This does not mean every model profile is now validated. It means the
`qwen3.5-27b-fp8` flow is working on this Thor with the Thor-side fixes in this
repo.

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

## Current Honest Summary

This repo is no longer just an outline. It contains the major integration and
hardening pieces needed for a local-first Thor setup around NemoClaw/OpenShell
and local Qwen serving.

What is still missing is real end-to-end validation on the actual Thor target,
plus any higher-level workflow layer that would turn this foundation into the
full orchestrator/coder/tester development system the user ultimately wants.
