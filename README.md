# NemoClaw-Thor

Local-first NemoClaw/OpenShell integration for Jetson AGX Thor (SM110a / Blackwell).

## Stack

| Component | Version | Notes |
|-----------|---------|-------|
| NemoClaw | v0.0.13 | `~/NemoClaw` (origin/main) |
| OpenShell | 0.0.26 | Auto-upgraded by NemoClaw installer |
| OpenClaw | 2026.3.11 | Pinned in NemoClaw's Dockerfile.base |
| vLLM | 0.19.1rc1 | Custom SM110 image with FlashInfer CUTLASS |
| Sandbox | `thor-v5` | Landlock + seccomp + netns |
| Provider | `vllm-local` | Direct HTTP to host vLLM (`:8000`) or ManyForge mux mode (`:8888`) |

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

# Terminal 1: start vLLM
./start-model.sh qwen3.5-27b-claude-distilled-v2-nvfp4

# Terminal 2: configure and verify
./configure-local-provider.sh qwen3.5-27b-claude-distilled-v2-nvfp4
./status.sh
nemoclaw thor-v5 connect
```

### ManyForge-integrated mode

If the OpenClaw main agent must reach ManyForge tools through the verified
workspace-plugin path, switch the provider to the muxed route first:

```bash
./configure-local-provider.sh --with-manyforge-mux qwen3.5-27b-claude-distilled-v2-nvfp4
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

| Profile | Model | Seqs | Agents | Notes |
|---------|-------|------|--------|-------|
| `qwen3.5-122b-a10b-nvfp4-resharded` | 122B MoE | 4 | 1 | Most capable, local resharded weights |
| `qwen3.5-27b-claude-distilled-v2-nvfp4` | 27B DeltaNet | 9 | 3 | Claude v2 distilled, best for coding |
| `qwen3.5-27b-claude-distilled-nvfp4` | 27B DeltaNet | 9 | 3 | Claude v1 distilled |
| `qwopus3.5-27b-nvfp4` | 27B DeltaNet | 9 | 3 | Opus-distilled, NVFP4 |
| `qwen3.5-27b-fp8` | 27B dense | 8 | 2 | FP8 quantized |
| `qwen3.5-35b-a3b-fp8` | 35B MoE | 22 | 5 | FP8, highest concurrency |
| `qwen3.5-35b-a3b-nvfp4` | 35B MoE | 26 | 6 | NVFP4, highest concurrency |
| `gemma4-31b-it-nvfp4` | 31B dense | 6 | 6 | Vision+text, NVFP4 |
| `gemma4-26b-a4b-it` | 26B MoE | 17 | 4 | Vision+text, BF16 |

**Seqs** = max concurrent sequences in vLLM. **Agents** = max concurrent
OpenClaw main agents (subagents fill remaining slots automatically).

## Architecture

```
Host (Jetson AGX Thor)
├── vLLM container (Docker, --network host, port 8000)
│   └── Model serving with SM110 NVFP4 + FlashInfer CUTLASS
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
- [PLAN-v5-transition.md](PLAN-v5-transition.md) — upgrade plan and architecture notes
