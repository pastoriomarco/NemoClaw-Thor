# NemoClaw-Thor v4

Local-first NemoClaw/OpenShell integration for Jetson AGX Thor.

## Current state (2026-04-06)

### Stack versions

| Component | v3 (previous) | v4 (current) |
|-----------|--------------|-------------|
| vLLM | 0.18.1rc1, NVIDIA jetson image | 0.19.0, custom SM110 image |
| NemoClaw | `be7ec09` (2026-03-20) | v0.0.6+ (2026-04-04) |
| OpenShell | 0.0.12 | 0.0.22 |
| OpenClaw | 2026.3.11 (in sandbox image) | 2026.3.11 (not yet upgraded) |
| Sandbox | `thor-assistant` | `thor-v4` |
| Provider | `vllm-local` | `vllm-local` |

### What changed from v3

**Proxy removed.** v3 routed inference through a restream proxy (port 8199)
inside the sandbox pod to buffer vLLM's broken streaming tool-call arguments.
v4 eliminates this entirely — OpenClaw talks to `https://inference.local/v1`
via the OpenShell provider route. vLLM 0.19 includes the streaming fix
(PR #35615).

**Scripts stripped.** 25 files deleted (legacy launchers, Nemotron scripts,
proxy code, install/uninstall tooling, host backup/restore, policy tooling).
4 scripts + 5 libs remain — only what was proven working in the v4 session.

**Sandbox survival.** OpenShell 0.0.22 persists sandboxes across gateway
restarts. Not yet fully tested — the SSH reconciliation, gateway lifecycle,
and device pre-pairing code is commented out pending verification.

**Onboard is now the entry point.** v3 used a custom `install.sh`. v4 uses
`nemoclaw onboard` directly, then `configure-local-provider.sh` to fix the
4 settings onboard gets wrong for local models.

### What's tested

| Feature | Status |
|---------|--------|
| vLLM v4 image build (SM110, NVFP4, FlashInfer) | Verified |
| NemoClaw v0.0.6 install + onboard | Verified |
| OpenShell 0.0.22 CLI install | Verified |
| Provider create/update, inference route | Verified |
| Text responses (reasoning + content) | Verified |
| Tool calling through vLLM (qwen3_coder parser) | Verified |
| vLLM warmup with reasoning models | Verified |
| openclaw.json patching (api, reasoning, maxTokens, baseUrl) | Verified |
| `docker exec → kubectl exec` chain to sandbox | Verified |

### What's NOT tested yet

| Feature | Risk |
|---------|------|
| Sandbox survival across gateway restart | May obsolete SSH reconciliation |
| `nemoclaw connect` device pairing | May obsolete gateway auth bypass |
| End-to-end agent session through TUI | Only direct vLLM tested |
| Subagent concurrency | Config commented out |
| Large tool call args without proxy | vLLM 0.19 fix should handle it |
| Egress firewall with OpenShell 0.0.22 | iptables rules not applied |
| OpenClaw 2026.4.5 upgrade | Still on 2026.3.11 |
| Gateway container version | May still be 0.0.12 |

## Scripts

| Script | Purpose |
|--------|---------|
| `start-model.sh <profile>` | Launch vLLM with a model profile |
| `configure-local-provider.sh [profile]` | Wire OpenShell provider + patch openclaw.json |
| `status.sh [profile]` | System health checks |
| `enforce-egress-firewall.sh` | iptables sandbox lockdown (standalone) |

## Usage

### First time (after fresh `nemoclaw onboard`)

```bash
cd ~/workspaces/nemoclaw/src/NemoClaw-Thor

# Terminal 1: start vLLM
./start-model.sh qwopus3.5-27b-nvfp4

# Terminal 2: configure and verify
./configure-local-provider.sh
./status.sh
nemoclaw thor-v4 connect
```

### After reboot

Same sequence: `start-model.sh`, then `configure-local-provider.sh`, then
`status.sh`, then `nemoclaw connect`.

### Switch model

Stop vLLM (Ctrl-C), drop caches, start new model, reconfigure:

```bash
sudo sync && sudo sysctl -w vm.drop_caches=3
./start-model.sh qwen3.5-35b-a3b-nvfp4
./configure-local-provider.sh qwen3.5-35b-a3b-nvfp4
```

## Model profiles

| Profile | Model | Attention | Notes |
|---------|-------|-----------|-------|
| `qwen3.5-122b-a10b-nvfp4-resharded` | 122B MoE | FlashInfer | Most capable, local resharded weights |
| `qwopus3.5-27b-nvfp4` | 27B DeltaNet | FlashInfer | Opus-distilled, NVFP4 |
| `qwen3.5-27b-claude-distilled-nvfp4` | 27B DeltaNet | FlashInfer | Claude-distilled, NVFP4 |
| `qwen3.5-27b-fp8` | 27B dense | FlashInfer | MTP speculative decoding |
| `qwen3.5-35b-a3b-fp8` | 35B MoE | FlashInfer | Fastest Qwen, MTP spec |
| `qwen3.5-35b-a3b-nvfp4` | 35B MoE | FlashInfer | NVFP4 quantized |
| `gemma4-31b-it-nvfp4` | 31B dense | triton_attn | Vision+text, NVFP4 |
| `gemma4-26b-a4b-it` | 26B MoE | triton_attn | Vision+text, BF16 |

## Key v3→v4 architectural differences

| Aspect | v3 | v4 |
|--------|----|----|
| Inference path | OpenClaw → restream proxy (8199) → host vLLM (8000) | OpenClaw → inference.local → OpenShell provider → host vLLM |
| Tool call fix | Restream proxy buffered fragments | vLLM 0.19 native fix (PR #35615) |
| Sandbox setup | Custom `install.sh` → nemoclaw setup | `nemoclaw onboard` → `configure-local-provider.sh` |
| Config patching | Full `sync_sandbox_runtime_config` (20+ settings) | Minimal 4-field patch (api, reasoning, maxTokens, baseUrl) |
| Script count | 25+ scripts | 4 scripts + 5 libs |
| Sandbox name | `thor-assistant` | `thor-v4` |
| OpenShell | 0.0.12 | 0.0.22 (sandbox survival) |

## References

- [PLAN-v4-reboot.md](PLAN-v4-reboot.md) — full upgrade plan with phase status
- [AGENT_NOTES-v3.md](AGENT_NOTES-v3.md) — v3 operational notes (historical reference)
- [NVIDIA/NemoClaw](https://github.com/NVIDIA/NemoClaw)
- [NVIDIA/OpenShell](https://github.com/NVIDIA/OpenShell)
- [openclaw/openclaw](https://github.com/openclaw/openclaw)
