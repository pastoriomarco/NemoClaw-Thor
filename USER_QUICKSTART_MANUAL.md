# User Quickstart Manual — NemoClaw-Thor v4

Operator manual for `NemoClaw-Thor` on a Jetson AGX Thor with NemoClaw v0.0.6+
and OpenShell v0.0.22.

Validated baseline:

- NemoClaw: v0.0.6+
- OpenShell: v0.0.22
- vLLM: v0.19.0 (custom SM110 image)
- Gateway: `nemoclaw`
- Sandbox: `thor-v4`
- Provider: `vllm-local`

Important rule:

- `./start-model.sh <profile>` only starts vLLM.
- `./configure-local-provider.sh [profile]` binds OpenShell to the running
  model, patches openclaw.json inside the sandbox, and sends a warmup request.

## 1. Shell Prep

From a fresh host shell:

```bash
cd /home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor
source "$HOME/.nvm/nvm.sh"
nvm use 22
export PATH="$HOME/.local/bin:$PATH"
```

## 2. Swap Configuration

Jetson AGX Thor uses unified memory (128 GiB shared between CPU and GPU).
Large models need swap space to survive transient memory spikes during
startup — particularly FlashInfer CUTLASS kernel compilation and CUDA graph
capture. Without swap, these spikes cause hard system crashes (device
reboots, no kernel OOM log).

**Requirement: 32 GiB swap file.** This is needed for models ≥70 GiB
(the 122B profile), and recommended for all profiles as a safety margin.

### Create swap (one-time)

```bash
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make persistent across reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Verify

```bash
swapon --show
# Should show: /swapfile file 32G 0B -2
free -h
# Swap line should show 32Gi total
```

### Why 32 GiB

The 122B NVFP4 model loads 70.47 GiB of weights. On first launch, vLLM
also JIT-compiles FlashInfer CUTLASS MoE GEMM kernels for SM110a (266
build targets via ninja), which temporarily requires significant additional
memory for nvcc processes running in parallel. The model weights + KV cache
allocation + compiler processes exceed 128 GiB physical RAM. Swap absorbs
the overflow — once compilation finishes and caches, memory drops back to
normal operating levels.

Smaller models (27B, 35B) may not strictly need swap, but a 32 GiB swap
file costs nothing at rest and prevents hard crashes if memory spikes occur
during torch.compile or CUDA graph capture.

## 3. First-Time Setup (After Fresh NemoClaw Install)

Use this on a fresh Thor or after a full reinstall.

1. Install NemoClaw and OpenShell upstream (if not done):

```bash
cd ~/NemoClaw && git pull origin main && npm install && npm link
```

2. Run the NemoClaw onboard wizard:

```bash
nemoclaw onboard
```

This creates the sandbox (e.g. `thor-v4`), the OpenShell gateway, and bakes a
base config into the sandbox image. Some onboard defaults are wrong for local
models (see "Known onboard issues" below) — `configure-local-provider.sh`
fixes them.

3. Start the model server and leave it running in that terminal:

```bash
./start-model.sh qwopus3.5-27b-nvfp4
```

4. In a second terminal, configure and verify:

```bash
./configure-local-provider.sh
./status.sh
nemoclaw thor-v4 connect
```

### Known onboard issues

`nemoclaw onboard` bakes these defaults that break local model inference:

| Setting | Onboard default | Required value | Fixed by |
|---------|----------------|----------------|----------|
| `inference.api` | `openai-responses` | `openai-completions` | `configure-local-provider.sh` |
| `reasoning` | `false` | `true` | `configure-local-provider.sh` |
| `maxTokens` | `4096` | `16384` | `configure-local-provider.sh` |
| `baseUrl` | (cloud endpoint) | `https://inference.local/v1` | `configure-local-provider.sh` |

The `openai-responses` API bypasses vLLM's `--tool-call-parser` and breaks tool
calling for all local models. See NemoClaw issue #976.

## 4. Start After Reboot

Same sequence every time — no special reboot handling needed:

1. Start the model server:

```bash
./start-model.sh qwopus3.5-27b-nvfp4
```

2. Rebind the provider and patch the sandbox:

```bash
./configure-local-provider.sh
```

3. Verify and connect:

```bash
./status.sh
nemoclaw thor-v4 connect
```

If `./status.sh` says the sandbox is missing, re-run `nemoclaw onboard`.

## 5. Switch Model

Stop vLLM (Ctrl-C in the model terminal), drop caches, start the new model,
reconfigure:

```bash
sudo sync && sudo sysctl -w vm.drop_caches=3
./start-model.sh qwen3.5-35b-a3b-nvfp4
./configure-local-provider.sh qwen3.5-35b-a3b-nvfp4
```

Always drop caches between model switches — Thor's unified memory is not
automatically freed.

## 6. Model Profiles

| Profile | Model | Notes |
|---------|-------|-------|
| `qwen3.5-122b-a10b-nvfp4-resharded` | 122B MoE | Most capable, local resharded weights |
| `qwopus3.5-27b-nvfp4` | 27B DeltaNet | Opus-distilled, NVFP4 |
| `qwen3.5-27b-claude-distilled-nvfp4` | 27B DeltaNet | Claude-distilled, NVFP4 |
| `qwen3.5-27b-fp8` | 27B dense | MTP speculative decoding |
| `qwen3.5-35b-a3b-fp8` | 35B MoE | Fastest Qwen, MTP spec |
| `qwen3.5-35b-a3b-nvfp4` | 35B MoE | NVFP4 quantized |
| `gemma4-31b-it-nvfp4` | 31B dense | Vision+text, NVFP4 |
| `gemma4-26b-a4b-it` | 26B MoE | Vision+text, BF16 |

### Launcher overrides

The most relevant per-run overrides for `./start-model.sh`:

- `THOR_MAX_MODEL_LEN`
- `THOR_KV_CACHE_DTYPE`
- `THOR_MAX_NUM_SEQS`
- `THOR_GPU_MEMORY_UTILIZATION`
- `THOR_MAX_NUM_BATCHED_TOKENS`

Example:

```bash
THOR_MAX_MODEL_LEN=65536 \
THOR_KV_CACHE_DTYPE=fp8 \
THOR_MAX_NUM_SEQS=20 \
THOR_GPU_MEMORY_UTILIZATION=0.80 \
./start-model.sh qwen3.5-35b-a3b-nvfp4
```

Persistent defaults are saved in:

```text
~/.config/nemoclaw-thor/config.env
```

## 7. First Launch — FlashInfer JIT Compilation

The first time a model profile is launched on a fresh install, vLLM
JIT-compiles FlashInfer CUTLASS kernels for SM110a (Blackwell/Thor).
This is a one-time process — compiled kernels are cached at
`/root/.cache/flashinfer/` inside the vLLM container.

**What to expect on first launch:**

| Phase | Duration (approx) | Notes |
|-------|-------------------|-------|
| Weight loading | 2-3 min | 8 safetensor shards for 122B |
| torch.compile | 2-10 min | Cached after first run |
| FlashInfer CUTLASS MoE GEMM compilation | 15-45 min | 266 ninja build targets, `nvcc -j14` |
| CUDA graph capture | 2-5 min | Batch sizes [1, 2, 4] |
| **Total first launch** | **20-60 min** | Depends on model size |
| **Subsequent launches** | **3-8 min** | All compilation cached |

During FlashInfer compilation, you will see no new vLLM log output for
an extended period — the process is running nvcc in the background. Check
with:

```bash
docker top <container> | grep nvcc
```

**Cache volumes must persist** for compilation results to survive across
container recreations. The `start-model.sh` script bind-mounts host
directories into the container:

| Container path | Host path | Contents |
|----------------|-----------|----------|
| `/root/.cache/flashinfer` | `~/thor-flashinfer-cache` | CUTLASS MoE GEMM `.so` files |
| `/root/.cache/vllm` | `~/thor-vllm-cache` | torch.compile AOT artifacts |
| `/root/.cache/torch` | `~/thor-torch-cache` | Triton kernel cache |

If any of these host directories are deleted, the next launch repeats
the corresponding compilation step.

Memory usage during compilation peaks significantly above normal operating
levels. This is the primary reason swap is required (see Section 2).

## 8. Use The Sandbox

Connect:

```bash
nemoclaw thor-v4 connect
```

Inside the sandbox, the main interactive UI is:

```bash
openclaw tui
```

The OpenClaw gateway normally auto-starts inside the sandbox. If the TUI shows
"gateway disconnected", start the gateway manually:

```bash
HOME=/sandbox openclaw gateway run &
openclaw tui
```

### Work on a repo copy inside the sandbox

Upload a repo into `/sandbox`:

```bash
openshell sandbox upload thor-v4 /path/to/repo /sandbox/myrepo
```

Then connect and work in the copy:

```bash
nemoclaw thor-v4 connect
cd /sandbox/myrepo
openclaw tui
```

Important:

- `openshell sandbox upload` copies files into the sandbox.
- Agent edits stay in `/sandbox/myrepo`.
- Your host repo is not changed automatically.

Copy the repo back out:

```bash
openshell sandbox download thor-v4 /sandbox/myrepo /path/to/output-copy
```

Or copy out only a patch:

```bash
# inside the sandbox
cd /sandbox/myrepo
git diff > /sandbox/myrepo.patch
```

Then on the host:

```bash
openshell sandbox download thor-v4 /sandbox/myrepo.patch .
```

## 9. Stop Without Uninstalling

### Leave the sandbox shell

```bash
exit
```

This does not stop the sandbox, gateway, or model server.

### Stop the model server

If `./start-model.sh` is running in the current terminal, press `Ctrl-C`.

If the vLLM container is running elsewhere:

```bash
docker ps --format '{{.ID}}\t{{.Command}}' | grep 'vllm serve'
docker stop <container-id>
```

After stopping vLLM, always reclaim memory before starting another model:

```bash
sudo sync && sudo sysctl -w vm.drop_caches=3
```

### Stop the OpenShell gateway

```bash
nemoclaw stop
```

Or directly:

```bash
openshell gateway stop
```

## 10. Practical Rules

- **Swap must be active** before starting any model (see Section 2).
  Verify with `swapon --show` before launching `start-model.sh`.
- Use `nemoclaw thor-v4 connect` as the normal shell entrypoint.
- Use `openclaw tui` as the normal prompt UI.
- If the TUI shows "gateway disconnected": `HOME=/sandbox openclaw gateway run &`
- After any model change, run `./configure-local-provider.sh <profile>`.
- After a reboot: verify swap (`swapon --show`), then `./start-model.sh`,
  `./configure-local-provider.sh`, `./status.sh`, `nemoclaw thor-v4 connect`.
- If you stop vLLM, always run `sudo sync && sudo sysctl -w vm.drop_caches=3`
  before loading another model.
- First launch of a new model profile takes 20-60 min for kernel compilation
  (see Section 7). Subsequent launches take 3-8 min.

## Key Differences From v3

| Aspect | v3 | v4 |
|--------|----|----|
| Setup | `./install.sh` + `./apply-host-fixes.sh` | `nemoclaw onboard` + `configure-local-provider.sh` |
| Sandbox | `thor-assistant` | `thor-v4` |
| Inference | Restream proxy (port 8199) → host vLLM | OpenShell provider route → host vLLM |
| Tool streaming | Proxy buffered tool-call fragments | vLLM 0.19 native fix (PR #35615) |
| Scripts | 25+ scripts | 4 scripts + 5 libs |
| Teardown | `./uninstall.sh` + `./restore-host-state.sh` | `nemoclaw stop` + remove sandbox manually |
