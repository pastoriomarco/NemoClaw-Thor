# NemoClaw-Thor

Local-first NemoClaw/OpenShell integration for Jetson AGX Thor (SM110a / Blackwell).

> **📖 Operator manual**: this README is a landing page / quickstart.
> For the full step-by-step procedure — swap setup, image rebuild, JIT
> compile expectations, sandbox workflows, cleanup procedure,
> troubleshooting — see
> [**USER_QUICKSTART_MANUAL.md**](USER_QUICKSTART_MANUAL.md).
> Additional deep-dive docs: [KV-CACHE-BUDGET.md](KV-CACHE-BUDGET.md),
> [DFLASH-INVESTIGATION.md](DFLASH-INVESTIGATION.md),
> [docker/NOTES.md](docker/NOTES.md).

## Quick start

From scratch, using the validated v6-pinned image and default profile
(NVFP4 + DFlash-15 — **45.7 tok/s single, 192.5 tok/s @ 8-concurrent**):

```bash
# Terminal 1: start the fastest model
cd ~/workspaces/nemoclaw/src/NemoClaw-Thor
./start-model.sh

# Terminal 2: wire the sandbox + sanity-check
./configure-local-provider.sh
./status.sh
nemoclaw my-assistant connect          # inside sandbox: `openclaw tui`
```

`./start-model.sh` with no args picks up the default profile
`qwen3.6-35b-a3b-nvfp4-dflash` (best for low-latency single-user work).

For **many concurrent sequences or huge context**, use the TQ-MTP variant:

```bash
./start-model.sh qwen3.6-35b-a3b-nvfp4-tq-mtp
./configure-local-provider.sh qwen3.6-35b-a3b-nvfp4-tq-mtp
```

Trade-off: 28.6 tok/s single (vs 45.7) but **2.22M KV tokens**, 29x
concurrency at 256K context, and 154.7 tok/s aggregate at 8-concurrent.
Requires the `fix-pr39931-turboquant` runtime mod (auto-applied). See
[Model profiles](#model-profiles) below for the full comparison.

Prerequisites: 32 GiB swap active, HF token at `~/.cache/huggingface/token`
(for the gated DFlash drafter), NemoClaw+OpenShell installed (see
[Usage](#usage) for install commands).

## Stack

| Component | Version | Notes |
|-----------|---------|-------|
| NemoClaw | v0.0.18-10-g946c52b7 (2026-04-17 validated) | `~/NemoClaw` (origin/main) — **not pinned** |
| OpenShell | 0.0.31 (2026-04-17 validated) | `curl ... install.sh` — **not pinned** |
| OpenClaw | 2026.4.2 | Pinned upstream in NemoClaw's Dockerfile.base |
| vLLM | v6 (dev338 pinned, commit `9965f501a`) | Custom SM110 image, fully pinned — see docker/NOTES.md |
| Sandbox | `my-assistant` (or `thor-v5`) | Landlock + seccomp + netns |
| Provider | `vllm-local` | Direct HTTP to host vLLM (`:8000`) or ManyForge mux mode (`:8888`) |

**Authoritative version references**:
- **vLLM image pins** (CUDA base, vLLM/FlashInfer commits, every pip package) — see
  [docker/NOTES.md → Pinned versions](docker/NOTES.md#pinned-versions)
- **NemoClaw pipeline versions + install commands to reproduce** — see
  [USER_QUICKSTART_MANUAL.md → Validated baseline](USER_QUICKSTART_MANUAL.md#user-quickstart-manual--nemoclaw-thor-v6)
  and section 3 for the exact `git checkout` / `OPENSHELL_VERSION=` commands

## Scripts

| Script | Purpose |
|--------|---------|
| `start-model.sh <profile>` | Launch vLLM with a model profile |
| `configure-local-provider.sh [OPTIONS] [profile]` | Wire OpenShell provider + patch sandbox config |
| `status.sh [profile]` | System health checks |

## Usage

### First time (after fresh `nemoclaw onboard`)

```bash
cd ~/workspaces/nemoclaw/src/NemoClaw-Thor

# Terminal 1: start vLLM with the default (fastest) profile
./start-model.sh                       # loads qwen3.6-35b-a3b-nvfp4-dflash

# Terminal 2: configure and verify
./configure-local-provider.sh          # picks up the same default
./status.sh
nemoclaw my-assistant connect
```

Pass a profile name to either script to pick a non-default (e.g.
`./start-model.sh qwen3.6-35b-a3b-nvfp4-tq-mtp` for max context).

### ManyForge-integrated mode

If the OpenClaw main agent must reach ManyForge tools through the verified
workspace-plugin path, switch the provider to the muxed route first:

```bash
./configure-local-provider.sh --with-manyforge-mux qwen3.6-35b-a3b-nvfp4-dflash
./status.sh
```

This keeps the OpenClaw-side provider name the same (`vllm-local`) but points
the OpenShell provider target at `http://host.openshell.internal:8888/v1`,
while the sandbox/OpenClaw client continues to use `https://inference.local/v1`.
In this mode the ManyForge mux forwards normal inference to vLLM and
`x_manyforge` traffic to ManyForge.

To restore the default direct-vLLM path:

```bash
./configure-local-provider.sh --without-manyforge-mux
```

### After reboot

Same sequence: `start-model.sh`, then `configure-local-provider.sh`, then
`status.sh`.

### Switch model

Stop vLLM (Ctrl-C), drop caches, start new model, reconfigure:

```bash
sudo sync && sudo sysctl -w vm.drop_caches=3
./start-model.sh <new-profile>
./configure-local-provider.sh <new-profile>
```

Always drop caches between model switches — Thor's unified memory is not
automatically freed.

## Model profiles

### Qwen3.6 (v6 container, production)

| Profile | Tok/s | KV Tokens | Seqs | Spec Method | Notes |
|---------|-------|-----------|------|-------------|-------|
| `qwen3.6-35b-a3b-nvfp4-dflash` | **45.7 / 192.5@8** | 678K | 5 | DFlash-15 | FASTEST, 256K ctx |
| `qwen3.6-35b-a3b-fp8-dflash` | **47.6** | ~700K | 4 | DFlash-15 | Best FP8 |
| `qwen3.6-35b-a3b-nvfp4-tq-mtp` | 28.6 | 2.22M | 8 | MTP N=4 | MAX CONTEXT, 153 tok/s @ 8-conc |
| `qwen3.6-35b-a3b-fp8-mtp-fp8kv` | 25.7 | 1.44M | 8 | MTP N=4 | FP8+FP8 KV |
| `qwen3.6-35b-a3b-fp8-turboquant` | 26.2 | 1.89M | 6 | MTP N=4 | FP8+TQ KV |

### Legacy / other

| Profile | Model | Seqs | Notes |
|---------|-------|------|-------|
| `qwen3.5-122b-a10b-nvfp4` | 122B MoE | 3 | Most capable |
| `qwen3.5-27b-claude-distilled-v2-nvfp4` | 27B DeltaNet | 9 | Claude v2 distilled (hermes tool parser) |
| `qwen3.5-9b-claude-distilled-nvfp4` | 9B VLM | 8 | Multimodal |
| `gemma4-e4b-it` | 8B MoE | 12 | Vision+text+audio |
| `gemma4-31b-it-nvfp4` | 31B dense | 6 | Vision+text, NVFP4 |
| `gemma4-26b-a4b-it` | 26B MoE | 17 | Vision+text, BF16 |

**Default profile**: `qwen3.6-35b-a3b-nvfp4-dflash` — what `./start-model.sh`
(no args) loads. FASTEST in single-req and 8-concurrent benchmarks. Requires
HF token for the gated drafter model.

If you can't use NVFP4 (e.g. no HF token, or prefer FP8 weights), run:
`./start-model.sh qwen3.6-35b-a3b-fp8-dflash`.

## Architecture

```
Host (Jetson AGX Thor)
├── vLLM v6 container (Docker, --network host, port 8000)
│   └── Model serving: Qwen3.6 DFlash/MTP + flash_attn/FlashInfer
├── OpenShell gateway (K3s, openshell-cluster-nemoclaw)
│   ├── L7 proxy (10.200.0.1:3128) — TLS termination, policy enforcement
│   └── Inference route:
│       • direct mode      → vllm-local → host:8000
│       • ManyForge mode   → vllm-local → host:8888 (mux)
└── Sandbox pod (thor-v5)
    ├── OpenClaw gateway (port 18789) — agent orchestration
    ├── OpenClaw agent — LLM-powered task execution
    └── Workspace (/sandbox/workspace → /sandbox/.openclaw-data/workspace)
```

### Why configure-local-provider.sh is needed

`nemoclaw onboard` bakes defaults that don't work for our local runtime modes:

| Setting | Onboard default | What we need | Why |
|---------|----------------|--------------|-----|
| `baseUrl` | `https://inference.local/v1` | Keep `https://inference.local/v1` in the sandbox, but repoint the OpenShell provider target to `http://host.openshell.internal:8000/v1` by default or `http://host.openshell.internal:8888/v1` in ManyForge mode | In this build the sandbox/OpenClaw client works through the proxy route; the provider target decides whether inference goes straight to vLLM or through the ManyForge mux |
| `contextWindow` | 131072 | 262144 | Models support 256K context |
| `maxTokens` | 4096 | 16384 | Agent needs long outputs for code generation |
| `timeoutSeconds` | (unset) | 1800 | Long reasoning sessions need 30min timeout |
| Concurrency | (unset) | Per-profile | Matches vLLM max_num_seqs budget |

The `configure-local-provider.sh` script patches these via `kubectl exec` into
the sandbox. This bypasses Landlock (kubectl exec starts a new process, not a
child of the sandbox entrypoint) and DAC restrictions (runs as root).

When ManyForge integration is enabled, the same script also persists the mux
state in `~/.config/nemoclaw-thor/config.env` so `status.sh` and later runs
stay consistent.

## Key files

```
NemoClaw-Thor/
├── start-model.sh              # vLLM launcher (picks profile, mounts caches)
├── configure-local-provider.sh # OpenShell provider + sandbox config sync
├── status.sh                   # Health checks
├── lib/
│   ├── config.sh               # Model profiles, concurrency math, runtime config
│   ├── launch.sh               # Docker run logic, cache mounts, env vars
│   ├── sandbox-runtime.sh      # sync_sandbox_runtime_config(), sandbox helpers
│   └── checks.sh               # Diagnostic checks for status.sh
├── docker/
│   ├── Dockerfile              # Multi-stage vLLM build for SM110
│   ├── build.sh                # Build orchestration
│   └── NOTES.md                # SM110 compatibility map, build history
└── KV-CACHE-BUDGET.md          # Memory planning reference
```

## References

- [NemoClaw](https://github.com/NVIDIA/NemoClaw) — sandbox framework
- [OpenShell](https://github.com/NVIDIA/OpenShell) — container orchestration
- [OpenClaw](https://github.com/openclaw/openclaw) — agent runtime
- [PLAN-v5-transition.md](PLAN-v5-transition.md) — v5 upgrade plan (historical)
- [DFLASH-INVESTIGATION.md](DFLASH-INVESTIGATION.md) — DFlash speculative decoding investigation and results
