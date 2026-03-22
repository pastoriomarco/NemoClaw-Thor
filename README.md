# NemoClaw-Thor

This fork adapts the original [JetsonHacks/NemoClaw-Thor](https://github.com/JetsonHacks/NemoClaw-Thor) flow for a stricter local-first setup on Jetson AGX Thor.

The upstream repository assumes local Nemotron 3 Nano inference. This fork now targets locally served Qwen models from [`../thor_llm/models`](../thor_llm/models), a hardened sandbox policy baseline, and a more reversible host-setup flow for repository development on Thor.

## Status

This repository has now been validated end-to-end on a Jetson AGX Thor host for
the `qwen3.5-27b-fp8` coding path.

For a more operational handoff note aimed at future coding sessions, see
[`AGENT_NOTES.md`](AGENT_NOTES.md).

For day-to-day operator usage, see
[`USER_QUICKSTART_MANUAL.md`](USER_QUICKSTART_MANUAL.md).

Implemented now:

- `install.sh`, `check-prerequisites.sh`, `status.sh`, and `configure-local-provider.sh` follow the Qwen-based local-provider flow.
- `start-model.sh` and the Qwen convenience wrappers provide local vLLM launch paths for the supported model profiles.
- Static and dynamic sandbox policy tooling is in place for `strict-local`, `local-hardened`, and `research-lite`.
- Host backup/apply/restore scripts are in place for the Thor-specific OpenShell fixes.
- The Thor wrapper now recovers from the current upstream NemoClaw onboarding failure on Thor where step 6 tries to rewrite an immutable `~/.openclaw/openclaw.json` inside the sandbox.
- Sandbox, provider, and internal OpenClaw/NemoClaw runtime state are now synced to exact tracked names and model ids instead of list-order guesses.

Validated on this host:

- Jetson AGX Thor / JetPack 7.1 / L4T 38.4
- OpenShell gateway creation and sandbox creation
- local vLLM provider creation and `inference.local` routing
- sandbox policy baseline installation with `strict-local`
- internal NemoClaw/OpenClaw config sync to `Qwen3.5-27B-FP8`
- end-to-end reply from inside the sandbox via:
  - `openclaw agent --agent main --local -m "Reply with one word: working" --session-id test`
- end-to-end tool execution from inside the sandbox via:
  - `openclaw agent --agent main --local --thinking off -m "Run uname -a and python3 --version, write both to /sandbox/smoke.txt, then reply done." --session-id smoke-tools`

Current validated runtime on this Thor:

- sandbox name: `thor-assistant`
- provider name: `vllm-local`
- served model id: `Qwen3.5-27B-FP8`
- validated runtime targets:
  - `--max-model-len 65536`
  - `--kv-cache-dtype fp8`
  - `--max-num-seqs 16`
  - auto tool choice enabled with `qwen3_coder`
  - `parallel_tool_calls=false` in sandbox OpenClaw config
  - `temperature=0` in sandbox OpenClaw config
  - sandbox tool surface reduced to `read`, `edit`, `write`, `exec`, `process`

Still incomplete or only partially validated:

- The upstream `NemoClaw` onboarding path is still interactive and cloud-first.
- The old `nemotron3-thor*.sh` launchers are still upstream leftovers and should not be treated as the preferred path for this fork.
- This repository is a setup and hardening layer for NemoClaw/OpenShell on Thor. It does not itself implement the higher-level orchestrator/coder/tester workflow logic.
- Other model profiles are not yet validated end-to-end on this host from this fork.
- The `qwen3.5-35b-a3b-fp8` path remains text-capable, but tool-use reliability on that stack is still not considered validated.

Important operational rule:

- use this repo's `install.sh` / `configure-local-provider.sh`
- do not treat raw upstream `nemoclaw onboard` as the supported Thor path

## Current Scope

This repository currently covers:

- preparing the Thor host for OpenShell with a reversible backup path
- installing `NemoClaw` and switching it to a local OpenAI-compatible provider
- launching supported local Qwen models with conservative Thor-oriented defaults
- applying a stricter sandbox network policy baseline for local repo work
- checking prerequisites and basic install/runtime status

This repository does not currently try to be:

- a finished autonomous coding agent product
- a replacement for the upstream `NemoClaw` onboarding wizard
- a benchmark suite for final model throughput claims

## Goal

Run `NemoClaw` and `OpenClaw` locally on Jetson AGX Thor with:

- local inference only
- local control only
- a sandboxed execution environment
- no WhatsApp, Telegram, or similar external channels
- disposable repo workspaces inside the sandbox
- a clear path to multi-agent coding workflows

The initial target is a local coding setup with separate orchestrator, coder, tester, and optional researcher agents running against a repo clone inside the sandbox.

## Requirements For This Fork

These are the intended requirements for the Qwen-based flow.

### Hardware and OS

- Jetson AGX Thor
- JetPack 7.1
- L4T 38.4
- Ubuntu 24.04

### Host software

- [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) installed
- All fixes from [JetsonHacks/openshell-thor](https://github.com/JetsonHacks/openshell-thor) applied
  - either manually
  - or through `./apply-host-fixes.sh`, which also snapshots the current host state first
- Docker installed and running
- NVIDIA container runtime enabled in Docker
- `nvm` with Node.js 22 for `NemoClaw`

### Local model serving

- A working local OpenAI-compatible vLLM endpoint on the Thor host
- Model-specific caches under the `thor_llm` workflow
- Hugging Face access only for the model weights you actually choose to run
- The launcher scripts in this repo assume a local vLLM API key and save it in the NemoClaw-Thor runtime config so the OpenShell provider and the local server agree on the same credential

Relevant model docs in this workspace:

- [`../thor_llm/models/qwen3.5-122b-a10b-nvfp4-resharded/README.md`](../thor_llm/models/qwen3.5-122b-a10b-nvfp4-resharded/README.md)
- [`../thor_llm/models/qwen3.5-27b-fp8/README.md`](../thor_llm/models/qwen3.5-27b-fp8/README.md)
- [`../thor_llm/models/qwen3.5-35b-a3b-fp8/README.md`](../thor_llm/models/qwen3.5-35b-a3b-fp8/README.md)
- [`../thor_llm/models/qwen3.5-35b-a3b-nvfp4/README.md`](../thor_llm/models/qwen3.5-35b-a3b-nvfp4/README.md)

### Current upstream constraint that still applies

The upstream `NemoClaw` onboarding flow is still cloud-first and still has one
Thor-specific failure mode that this repo works around afterward.

- Today, the onboarding wizard still asks for an NVIDIA API key.
- This fork intends to switch the actual inference route to a local provider after onboarding.
- On Thor, the upstream step that rewrites `~/.openclaw/openclaw.json` can fail because the sandbox image now makes that file immutable by design.
- `install.sh` and `configure-local-provider.sh` in this repo now recover from that state and finish the local-provider sync from the host side.

## Safety Requirements

This fork is being shaped around these requirements:

- `OpenClaw` should be controlled only from the local machine.
- Agent work should happen only inside the sandbox.
- The sandbox should use disposable repo clones instead of the real working tree.
- Personal accounts and personal tokens should not be mounted into the sandbox.
- Outbound network access should default to deny.
- If internet access is later enabled, it should be limited to an explicit allowlist and ideally to a dedicated researcher agent.
- The default `NemoClaw` policy presets should not be accepted blindly for this fork.

## Host Backup And Restore

The Thor-specific OpenShell fixes are host-wide. This fork now includes a
backup and restore layer for those changes.

Available scripts:

- `./backup-host-state.sh`
- `./apply-host-fixes.sh`
- `./restore-host-state.sh`

What gets backed up before the Thor fixes are applied:

- `/etc/docker/daemon.json`
- `/etc/modules-load.d/openshell-k3s.conf`
- `/etc/sysctl.d/99-openshell-k3s.conf`
- the active `iptables` and `ip6tables` alternatives
- `iptables-save` and `ip6tables-save` output
- a small Docker inventory snapshot for reference

What `./restore-host-state.sh` does:

- stops the OpenShell gateway if present
- restores the saved Docker config and persistence files
- restores the saved `iptables` backend choice
- optionally restores the saved firewall snapshots
- removes the out-of-tree `iptable_raw` module and downloaded kernel source if they were not present before the fix was applied

This is intended to restore the real pre-change host state, not just JetPack defaults.

Limits still apply:

- exact firewall restore is most reliable on a quiet host
- if Docker workloads changed substantially after the backup, restoring the old firewall snapshot may no longer be an exact fit
- `./apply-host-fixes.sh` is not guaranteed safe over a single SSH session
  - the upstream Thor helper switches `iptables` backend, flushes current firewall rules, and restarts Docker
  - if the host currently depends on custom firewall rules to keep SSH reachable, you can lose the session
  - `./backup-host-state.sh` is the safe remote step; the apply step should be treated as console-preferred unless the current firewall state is known

## Temporary Passwordless sudo For LLM-Assisted Install

If you want a coding agent or remote LLM session to run the privileged install
steps directly, interactive `sudo` prompts are usually not enough because the
prompt may appear in a background PTY the user cannot access.

The practical workaround is a temporary `NOPASSWD` sudoers entry for the local
user running the session.

Create it with:

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

After the install/debug session is finished, remove it:

```bash
sudo rm /etc/sudoers.d/nemoclaw-codex
sudo -k
```

This should be treated as temporary access for the install/debug window only.

## Sandbox Policy Hardening

This fork now hardens the sandbox policy before upstream onboarding runs.

The default static policy profile is:

- `strict-local`

That profile removes all pre-allowed external endpoints from the sandbox baseline. It is the closest match to a local-only development setup.

A second static profile is available:

- `local-hardened`

That profile keeps only the small OpenClaw endpoint set that may be needed for compatibility, while still removing cloud inference, messaging, GitHub, npm, and other broader egress.

For temporary, session-only troubleshooting access, this fork also provides:

- `research-lite`

That dynamic additions file allows a narrow set of developer-oriented endpoints such as GitHub, `docs.openclaw.ai`, `docs.nvidia.com`, PyPI, and the npm registry. It must be applied explicitly and resets when the sandbox stops.

If you want one-off approvals instead of a preset, use:

```bash
openshell term
```

and approve only the blocked requests you actually need.

## What This Fork Changes From The Original JetsonHacks Instructions

Compared with the original JetsonHacks flow, this fork changes the setup in the following ways:

1. Replace local Nemotron 3 Nano inference with locally served Qwen models.
2. Reuse the existing `thor_llm` model-serving flow instead of bundling a single fixed Nemotron launcher.
3. Document three model profiles instead of one.
4. Standardize the initial tuning target around:
   - `--max-model-len 65536`
   - `--kv-cache-dtype fp8`
   - higher `--max-num-seqs` than the current max-context configs
5. Rewrite the installation instructions around local-only development rather than generic assistant usage.
6. Tighten the security posture:
   - local dashboard only
   - no chat bridges
   - no personal browser/account access
   - stricter sandbox and network policy guidance
7. Rewrite health checks and verification steps so they no longer assume `nvidia/nemotron-3-nano-30b-a3b`.

## Target Model Profiles

This fork is being prepared to support these three model choices:

| Model | Role | Current source doc |
|---|---|---|
| `qwen3.5-122b-a10b-nvfp4-resharded` | Most capable | [`../thor_llm/models/qwen3.5-122b-a10b-nvfp4-resharded/README.md`](../thor_llm/models/qwen3.5-122b-a10b-nvfp4-resharded/README.md) |
| `qwen3.5-27b-fp8` | Capable but slower | [`../thor_llm/models/qwen3.5-27b-fp8/README.md`](../thor_llm/models/qwen3.5-27b-fp8/README.md) |
| `qwen3.5-35b-a3b-fp8` or `qwen3.5-35b-a3b-nvfp4` | Fastest, lower ceiling | [`../thor_llm/models/qwen3.5-35b-a3b-fp8/README.md`](../thor_llm/models/qwen3.5-35b-a3b-fp8/README.md) and [`../thor_llm/models/qwen3.5-35b-a3b-nvfp4/README.md`](../thor_llm/models/qwen3.5-35b-a3b-nvfp4/README.md) |

## Initial Serving Targets

These are planning targets for the rewritten instructions, not final benchmarked guarantees.

| Model | Planned context | Planned KV cache | Initial `max-num-seqs` target | Notes |
|---|---|---|---|---|
| `qwen3.5-122b-a10b-nvfp4-resharded` | `65536` | `fp8` | `8` | Strongest model, likely the practical upper bound for coding agents on Thor |
| `qwen3.5-27b-fp8` | `65536` | `fp8` | `8` to `12` | More headroom than the current max-context profile, but throughput still needs validation |
| `qwen3.5-35b-a3b-fp8` | `65536` | `fp8` | `8` to `12` | Fastest path from the existing Qwen docs |
| `qwen3.5-35b-a3b-nvfp4` | `65536` | `fp8` | `8` to `12` | Expected to have at least similar memory headroom to the FP8 profile |

### Why these numbers

These targets come from the current `thor_llm` docs in this workspace:

- `qwen3.5-122b-a10b-nvfp4-resharded` already uses `--kv-cache-dtype fp8` and `--max-num-seqs 2`. Lowering the working context target to `65536` should make `8` the first concurrency target worth validating.
- `qwen3.5-27b-fp8` is currently documented with `--max-model-len 262144` and `--max-num-seqs 1`. Moving to `65536` and `fp8` KV cache should permit materially more concurrency, but the exact stable point still needs measurement.
- `qwen3.5-35b-a3b-fp8` is already documented at `262144` with `fp8` KV cache and `--max-num-seqs 2`. At `65536`, `8` is the first reasonable target, with `12` as a stretch target to test.

These numbers should be treated as startup targets for the rewritten instructions, not as finished performance claims.

## Recommended Concurrency Interpretation

For this repository, "concurrent requests" should mean "simultaneous active vLLM sequences that keep the agent stack responsive enough for development."

That means:

- latency matters more than peak queue depth
- long context plus long tool-use generations will reduce practical throughput
- an agentic coding workflow should prefer a stable `max-num-seqs` over an aggressive one

For the first documented version of this fork, the conservative baseline should be:

- `8` for `qwen3.5-122b-a10b-nvfp4-resharded`
- `8` for `qwen3.5-27b-fp8`
- `8` for `qwen3.5-35b-a3b-fp8`
- `8` for `qwen3.5-35b-a3b-nvfp4`

After that baseline is validated, the smaller models can be pushed further.

## Planned Setup Shape

The intended final setup is:

1. Back up the Thor host state and apply the Thor-specific OpenShell fixes.
2. Start one selected Qwen model locally through the `thor_llm` flow.
3. Point `OpenShell` local inference at that host endpoint.
4. Switch `NemoClaw` from cloud inference to the local provider.
5. Use a sandboxed repo clone for agent work.
6. Keep network access off by default.
7. Enable a separate research-capable agent only if explicit allowlist rules are added later.

## Install Sequence

For a fresh Thor or fresh JetPack reinstall, the validated sequence is staged,
not the single-command path.

Recommended sequence on a fresh host:

1. Apply the host fixes first.
2. Start the local vLLM server in one terminal.
3. Run the NemoClaw-Thor installer in a second terminal.

If you want the currently validated runtime targets on a fresh host, export:

```bash
export THOR_TARGET_MAX_MODEL_LEN=65536
export THOR_TARGET_KV_CACHE_DTYPE=fp8
export THOR_TARGET_MAX_NUM_SEQS=16
```

Then run:

```bash
./apply-host-fixes.sh
./start-model.sh qwen3.5-35b-a3b-fp8
./install.sh qwen3.5-35b-a3b-fp8 --policy-profile strict-local
```

This is the path that matches the validated Thor run.

`install.sh` now ensures the `nemoclaw` OpenShell gateway is running before it
hands off to upstream NemoClaw.

The one-command shortcut still exists:

```bash
./install.sh qwen3.5-35b-a3b-fp8 --policy-profile strict-local --apply-host-fixes
```

But it is not the preferred local-vLLM onboarding path on a clean host because
the host-fix step restarts Docker, which also stops any already-running vLLM
container. Use the staged sequence above when you want the local provider path
to be available during onboarding.

`--apply-host-fixes` runs `./apply-host-fixes.sh` first, which:

1. saves a restore snapshot of the current host state
2. clones `jetsonhacks/OpenShell-Thor` if needed
3. builds `iptable_raw`
4. applies the Thor Docker, iptables, module, and sysctl fixes

Do not assume `--apply-host-fixes` is SSH-safe on a remotely administered Thor. Check the current firewall policy first, or use console access or a rollback timer.

If you prefer to do that step explicitly, use:

```bash
./apply-host-fixes.sh
./install.sh qwen3.5-35b-a3b-fp8 --policy-profile strict-local
```

To revert the host-level changes back to the saved pre-change state:

```bash
./restore-host-state.sh
```

## Policy Tooling

Static baseline installation into the cloned NemoClaw repo:

```bash
./apply-policy-profile.sh strict-local
./apply-policy-profile.sh local-hardened
```

Session-scoped dynamic additions for troubleshooting:

```bash
./apply-policy-additions.sh research-lite
```

Recommended sequence for the most conservative setup:

1. Install with `strict-local`.
2. Try the local-only workflow first.
3. If the sandboxed agent needs limited outbound access, prefer `openshell term` for one-off approvals.
4. If repeated troubleshooting access is needed for a session, apply `research-lite`.

## Launcher Scripts

The preferred launcher entrypoint is:

```bash
./start-model.sh <model-profile>
```

Convenience wrappers are also provided:

- `./start-qwen3.5-122b-a10b-nvfp4-resharded.sh`
- `./start-qwen3.5-27b-fp8.sh`
- `./start-qwen3.5-35b-a3b-fp8.sh`
- `./start-qwen3.5-35b-a3b-nvfp4.sh`

All launchers target:

- `--max-model-len 65536`
- `--kv-cache-dtype fp8`
- `--max-num-seqs 8` by default

The smaller models can still be pushed further with environment overrides after validation.

For the currently validated Thor runtime, this host has:

- `THOR_TARGET_MAX_MODEL_LEN=65536`
- `THOR_TARGET_KV_CACHE_DTYPE=fp8`
- `THOR_TARGET_MAX_NUM_SEQS=16`

If you stop vLLM and want to reclaim memory before starting another model, run:

```bash
sudo sync
sudo sysctl -w vm.drop_caches=3
```

before launching the next server.

## Using An Installed Stack

If the current install is still running, you can use it immediately.

For `nemoclaw` commands, activate the Node 22 runtime first:

```bash
source "$HOME/.nvm/nvm.sh"
nvm use 22
export PATH="$HOME/.local/bin:$PATH"
nemoclaw thor-assistant connect
```

For a direct OpenShell shell, only the `openshell` path is required:

```bash
export PATH="$HOME/.local/bin:$PATH"
openshell sandbox connect thor-assistant
```

If you want to bring the stack back after a reboot or after stopping services,
the clean path is:

```bash
source "$HOME/.nvm/nvm.sh"
nvm use 22
export PATH="$HOME/.local/bin:$PATH"
./start-model.sh qwen3.5-35b-a3b-fp8
./configure-local-provider.sh qwen3.5-35b-a3b-fp8
./status.sh qwen3.5-35b-a3b-fp8
nemoclaw thor-assistant connect
```

`configure-local-provider.sh` now revives the `nemoclaw` gateway if needed and
repairs native `nemoclaw thor-assistant connect` if the sandbox SSH handshake
state drifted from the gateway.

If `./status.sh` reports that the sandbox is missing, rerun:

```bash
./install.sh qwen3.5-35b-a3b-fp8 --policy-profile strict-local
```

If `~/NemoClaw` already exists and you only need to refresh the local-provider
binding after a model change, use:

```bash
./configure-local-provider.sh qwen3.5-35b-a3b-fp8
```

## Suggested Tests

Start with small operator checks:

```bash
source "$HOME/.nvm/nvm.sh"
nvm use 22
./status.sh qwen3.5-27b-fp8
nemoclaw thor-assistant status
```

Then do the validated inference smoke test from inside the sandbox:

```bash
openclaw agent --agent main --local --thinking off \
  -m "Reply with one word: working" --session-id test
```

The validated tool-use smoke test on the current 27B stack is:

```bash
openclaw agent --agent main --local --thinking off \
  -m "Run uname -a and python3 --version, write both to /sandbox/smoke.txt, then reply done." \
  --session-id smoke-tools
```

If a previous broken model/tool round leaves the embedded agent replaying stale
history, reset the sandbox-local session store and retry:

```bash
./reset-sandbox-session-state.sh qwen3.5-27b-fp8
```

After that, the next useful test is a real repo task in a disposable sandbox
workspace under `/sandbox`, for example:

- inspect a local repo clone
- make a tiny code change
- run the repo test command
- report what passed or failed

## Not Yet Updated

The following pieces still need to be rewritten in this fork:

- the Nemotron launch scripts

The main install, provider configuration, status flow, local Qwen launchers, and sandbox policy hardening flow are now in place.

## Upstream References

- [JetsonHacks/NemoClaw-Thor](https://github.com/JetsonHacks/NemoClaw-Thor)
- [JetsonHacks/openshell-thor](https://github.com/JetsonHacks/openshell-thor)
- [NVIDIA/NemoClaw](https://github.com/NVIDIA/NemoClaw)
- [NVIDIA/OpenShell](https://github.com/NVIDIA/OpenShell)
- [openclaw/openclaw](https://github.com/openclaw/openclaw)

## License

This project remains under the MIT License. See [LICENSE](LICENSE).
