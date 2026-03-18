# NemoClaw-Thor

Scripts and fixes for running [NVIDIA NemoClaw](https://github.com/NVIDIA/NemoClaw) on the Jetson AGX Thor (JetPack 7.1 / L4T 38.4, Ubuntu 24.04 Noble).

NemoClaw combines [OpenClaw](https://github.com/openclaw/openclaw) with [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) to run AI agents in a secure, policy-enforced sandbox. This repository provides the Thor-specific fixes, launch scripts, and installer needed to get the full stack running with local Nemotron 3 Nano inference.

## Prerequisites

**Hardware:** Jetson AGX Thor with JetPack 7.1 installed.

**Software — must be in place before starting:**
- OpenShell installed with all five openshell-thor fixes applied
  See: [JetsonHacks/openshell-thor](https://github.com/JetsonHacks/openshell-thor)
- Docker installed and running with the NVIDIA container runtime
- A HuggingFace account with access to the Nemotron 3 Nano NVFP4 model
  Accept the license at: [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4)
- An NVIDIA API key from [build.nvidia.com](https://build.nvidia.com/settings/api-keys)
  Required during the NemoClaw onboarding wizard

Run `./check-prerequisites.sh` to verify your system before proceeding.

## First-Run Download Requirements

The first run of `nemotron3-thor-no-thinking.sh` or `nemotron3-thor.sh` will download:

| Item | Size | Location |
|---|---|---|
| vLLM container image | ~35GB | Docker image cache |
| Nemotron 3 Nano NVFP4 weights | ~19GB | `~/.cache/huggingface` |

Both are cached after the first pull. Ensure you have a stable network connection and sufficient disk space before starting.

## Repository Contents

| Script | Description |
|---|---|
| `check-prerequisites.sh` | Verify system prerequisites before installing |
| `install-node.sh` | Install nvm and Node.js 22 (required by NemoClaw) |
| `uninstall-node.sh` | Remove nvm and Node.js installed by install-node.sh |
| `install.sh` | Install NemoClaw and configure local vLLM inference |
| `uninstall.sh` | Remove NemoClaw, sandbox, and inference providers |
| `nemotron3-thor-no-thinking.sh` | Start the inference server — Fast mode |
| `nemotron3-thor.sh` | Start the inference server — Thinking mode |
| `status.sh` | Check system health and print end-to-end test instructions |

## Installation

**1. Check prerequisites:**
```bash
./check-prerequisites.sh
```
Address any failures before continuing.

**2. Install Node.js 22 via nvm:**
```bash
./install-node.sh
```
Open a new terminal after this completes.

**3. Set your HuggingFace token:**
```bash
export HF_TOKEN=hf_...
```

**4. Install NemoClaw:**
```bash
./install.sh
```
The NemoClaw onboarding wizard runs interactively. You will be prompted for a sandbox name, your NVIDIA API key, and policy presets. Accept the suggested defaults for policy presets.

## Starting the Inference Server

Open a new terminal and start one of the inference servers:

```bash
./nemotron3-thor-no-thinking.sh   # Fast — recommended for most use
./nemotron3-thor.sh               # Thinking — better accuracy on complex tasks
```

Wait until you see `Application startup complete.` before proceeding. The first startup downloads the model weights (~19GB) if not already cached and may take several minutes.

## Verifying the Installation

Check system health:
```bash
./status.sh
```

When all checks pass, connect to the sandbox and send a test message:
```bash
nemoclaw <sandbox-name> connect
```
Then inside the sandbox:
```bash
openclaw agent --agent main --local \
  -m "Reply with one word: working" --session-id test
```
Expected response: a single word reply from Nemotron 3 Nano.

The first response may be slow while the model warms up. If you see `No reply from agent`, wait a moment and try again.

## Web Interface

The OpenClaw Gateway Dashboard is available at `http://127.0.0.1:18789/` when the port forward is running. It provides a browser-based interface for monitoring sessions, agents, channels, usage, and logs.

The port forward is started automatically during installation. To start it manually:
```bash
openshell forward start 18789 <sandbox-name> --background
```

To stop it:
```bash
openshell forward stop 18789 <sandbox-name>
```

To find your sandbox name: `openshell sandbox list`

**Authentication (as of March 2026):** Due to a known bug in the current OpenClaw
version, the dashboard requires the gateway token passed directly in the URL.

The gateway token is a secret string generated during NemoClaw onboarding that
authenticates the browser to the OpenClaw gateway WebSocket. It is stored in
`openclaw.json` inside the sandbox and is unique to your sandbox instance.

To retrieve it, connect to the sandbox:
```bash
nemoclaw <sandbox-name> connect
```
Then inside the sandbox:
```bash
openclaw config get gateway.auth.token
```
Open the dashboard with the token in the URL:
```
http://127.0.0.1:18789/?token=<token-value>
```
The token persists across sandbox restarts — you only need to retrieve it once.
This workaround will not be needed once the OpenClaw authentication bug is fixed.

## Inference Modes

Two launch scripts are provided for the inference server:

| Script | Mode | Latency | Use when |
|---|---|---|---|
| `nemotron3-thor-no-thinking.sh` | Fast | ~5 seconds | Most agentic tasks, chat |
| `nemotron3-thor.sh` | Thinking | ~60 seconds | Hard reasoning, math, code |

Fast mode disables Nemotron 3 Nano's internal reasoning trace. Thinking mode enables it, producing better results on difficult tasks at the cost of significantly higher latency.

## Memory Tuning

`--gpu-memory-utilization` controls the fraction of Thor's 128GB unified memory allocated to vLLM. Edit the value in the launch script to adjust:

| Utilization | Context | vLLM pool | Weights | KV cache | OS headroom |
|---|---|---|---|---|---|
| 0.35 | 8192 | ~44.8 GB | ~19 GB | ~25 GB | ~83 GB |
| 0.45 | 32768 | ~57.6 GB | ~19 GB | ~38 GB | ~70 GB (default) |
| 0.55 | 65536 | ~70.4 GB | ~19 GB | ~51 GB | ~57 GB |

## Uninstalling

Remove NemoClaw (sandbox, inference providers, CLI, and NemoClaw directory):
```bash
./uninstall.sh
```

Remove nvm and Node.js:
```bash
./uninstall-node.sh
```

## Known Limitations

**Plugin banner** — The NemoClaw plugin inside the sandbox always displays `Endpoint: build.nvidia.com`. This is cosmetic. Inference is routed locally through the OpenShell gateway to the vLLM server.

**Web dashboard authentication (as of March 2026)** — The Control UI at `http://127.0.0.1:18789/` requires the gateway token in the URL due to a known OpenClaw authentication bug. See the Web Interface section above for the workaround.

**First-call latency** — The first inference request after starting the server may time out while the model loads into memory. This is expected — wait a moment and retry.

**Response latency** — In Fast mode, simple prompts take approximately 7-8 seconds end-to-end (3 seconds OpenClaw startup, 4-5 seconds model inference). Thinking mode adds significant latency due to the reasoning trace.

## Related Repositories

- [JetsonHacks/openshell-thor](https://github.com/JetsonHacks/openshell-thor) — OpenShell fixes for JetPack 7.1
- [NVIDIA/NemoClaw](https://github.com/NVIDIA/NemoClaw) — NemoClaw upstream
- [NVIDIA/OpenShell](https://github.com/NVIDIA/OpenShell) — OpenShell upstream
- [openclaw/openclaw](https://github.com/openclaw/openclaw) — OpenClaw upstream

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.