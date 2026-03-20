# NemoClaw-Thor

This fork adapts the original [JetsonHacks/NemoClaw-Thor](https://github.com/JetsonHacks/NemoClaw-Thor) flow for a stricter local-first setup on Jetson AGX Thor.

The upstream repository assumes local Nemotron 3 Nano inference. This fork is being rewritten to use locally served Qwen models from [`../thor_llm/models`](../thor_llm/models) and to document a tighter sandbox policy for repository development.

## Status

This repository is a work in progress.

- `install.sh`, `check-prerequisites.sh`, `status.sh`, and `configure-local-provider.sh` now follow the Qwen-based local-provider flow.
- The README describes the target setup and the planned changes from upstream.
- The old `nemotron3-thor*.sh` launchers are still upstream leftovers and should not be treated as the preferred path for this fork.

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

The upstream `NemoClaw` onboarding flow is still cloud-first.

- Today, the onboarding wizard still asks for an NVIDIA API key.
- This fork intends to switch the actual inference route to a local provider after onboarding.
- Until the installer is rewritten, that upstream onboarding constraint remains in force.

## Safety Requirements

This fork is being shaped around these requirements:

- `OpenClaw` should be controlled only from the local machine.
- Agent work should happen only inside the sandbox.
- The sandbox should use disposable repo clones instead of the real working tree.
- Personal accounts and personal tokens should not be mounted into the sandbox.
- Outbound network access should default to deny.
- If internet access is later enabled, it should be limited to an explicit allowlist and ideally to a dedicated researcher agent.
- The default `NemoClaw` policy presets should not be accepted blindly for this fork.

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

## What Will Change From The Original JetsonHacks Instructions

Compared with the original JetsonHacks flow, this fork is intended to change the setup in the following ways:

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

1. Start one selected Qwen model locally through the `thor_llm` flow.
2. Point `OpenShell` local inference at that host endpoint.
3. Switch `NemoClaw` from cloud inference to the local provider.
4. Use a sandboxed repo clone for agent work.
5. Keep network access off by default.
6. Enable a separate research-capable agent only if explicit allowlist rules are added later.

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
