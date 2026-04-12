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
| Provider | `vllm-local` | Direct HTTP to host vLLM |

## Scripts

| Script | Purpose |
|--------|---------|
| `start-model.sh <profile>` | Launch vLLM with a model profile |
| `configure-local-provider.sh [profile]` | Wire OpenShell provider + patch sandbox config |
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
│   └── Inference route: inference.local → vllm-local → host:8000
└── Sandbox pod (thor-v5)
    ├── OpenClaw gateway (port 18789) — agent orchestration
    ├── OpenClaw agent — LLM-powered task execution
    └── Workspace (/sandbox/workspace → /sandbox/.openclaw-data/workspace)
```

### Why configure-local-provider.sh is needed

`nemoclaw onboard` bakes defaults that don't work for local vLLM inference:

| Setting | Onboard default | What we need | Why |
|---------|----------------|--------------|-----|
| `baseUrl` | `https://inference.local/v1` | `http://host.openshell.internal:8000/v1` | OpenClaw's fetch doesn't honor HTTP_PROXY (upstream bug openclaw/openclaw#62181) |
| `contextWindow` | 131072 | 262144 | Models support 256K context |
| `maxTokens` | 4096 | 16384 | Agent needs long outputs for code generation |
| `timeoutSeconds` | (unset) | 1800 | Long reasoning sessions need 30min timeout |
| Concurrency | (unset) | Per-profile | Matches vLLM max_num_seqs budget |

The `configure-local-provider.sh` script patches these via `kubectl exec` into
the sandbox. This bypasses Landlock (kubectl exec starts a new process, not a
child of the sandbox entrypoint) and DAC restrictions (runs as root).

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
