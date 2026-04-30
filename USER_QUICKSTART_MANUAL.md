# User Quickstart Manual — NemoClaw-Thor

Operator manual for `NemoClaw-Thor` on a Jetson AGX Thor.

The currently verified versions of NemoClaw, OpenShell (CLI + cluster
image), and OpenClaw live in [AGENTS.md](AGENTS.md) — single source of
truth, updated on each tested upgrade. The vLLM image is owned by this
repo and fully pinned in `docker/Dockerfile.vllm` (see `docker/NOTES.md`
for build details).

**Operating shape:**

- Gateway: `nemoclaw`
- Sandbox: created during `nemoclaw onboard` (typically `my-assistant`)
- Provider route: `vllm-local` → `host.openshell.internal:8000` (direct
  to vLLM); a muxed mode is available for ManyForge integration when
  the bridge service is in play.

**Reproducibility note:** the vLLM container is fully pinned by this repo's
build; NemoClaw and OpenShell install from their respective upstream
release channels via the install script in NemoClaw's `scripts/`
directory. To reproduce a specific tested baseline on a new host, pin
to the versions in AGENTS.md using the commands in section 3.

Important rule:

- `./start-model.sh <profile>` only starts vLLM.
- `./configure-local-provider.sh [profile]` binds OpenShell to the running
  model, patches openclaw.json inside the sandbox, and sends a warmup request.
- `./configure-local-provider.sh --with-manyforge-mux [profile]` switches the
  provider to `http://host.openshell.internal:8888/v1` so the embedded
  OpenClaw agent can use the verified ManyForge workspace-plugin path.

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
# Latest main (what works for us today)
cd ~/NemoClaw && git pull origin main && npm install && npm link
```

**For reproducing a specific verified baseline:**

Replace `<NEMOCLAW_REF>` and `<OPENSHELL_VERSION>` below with the
values from the AGENTS.md verified-versions table.

```bash
# Pin NemoClaw to the verified ref (commit hash or tag like v0.0.31)
cd ~/NemoClaw
git fetch origin
git checkout <NEMOCLAW_REF>
npm install && npm link

# Pin OpenShell CLI to the verified version
bash ~/NemoClaw/scripts/install-openshell.sh
# (the installer reads min/max version pins from NemoClaw's blueprint
#  and downloads the appropriate release; falls within AGENTS.md range)

# OpenClaw is pinned automatically by NemoClaw's Dockerfile.base —
# it gets installed into the sandbox image during `nemoclaw onboard`.
```

Verify:
```bash
nemoclaw --version
openshell --version
# After onboard, check OpenClaw inside sandbox:
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell my-assistant -c agent -- openclaw --version
```

Cross-check the printed versions against AGENTS.md.

2. Run the NemoClaw onboard wizard:

```bash
nemoclaw onboard
```

This creates the sandbox (e.g. `thor-v5`), the OpenShell gateway, and bakes a
base config into the sandbox image. Some onboard defaults are wrong for local
vLLM inference — `configure-local-provider.sh` fixes them (see Section 11).

3. Start the model server and leave it running in that terminal:

```bash
./start-model.sh qwen3.6-35b-a3b-prismaquant-dflash
```

4. In a second terminal, configure and verify:

```bash
./configure-local-provider.sh
./status.sh
nemoclaw my-assistant connect
```

If you are enabling ManyForge tool access for the embedded OpenClaw agent, use:

```bash
./configure-local-provider.sh --with-manyforge-mux
./status.sh
```

## 4. Start After Reboot

Same sequence every time — no special reboot handling needed:

1. Start the model server:

```bash
./start-model.sh qwen3.6-35b-a3b-prismaquant-dflash
```

2. Rebind the provider and patch the sandbox:

```bash
./configure-local-provider.sh
```

To restore direct local inference after using ManyForge mode:

```bash
./configure-local-provider.sh --without-manyforge-mux
```

3. Verify and connect:

```bash
./status.sh
nemoclaw my-assistant connect
```

If `./status.sh` says the sandbox is missing, re-run `nemoclaw onboard`.

## 5. Switch Model

Stop vLLM (Ctrl-C in the model terminal), drop caches, start the new model,
reconfigure:

```bash
sudo sync && sudo sysctl -w vm.drop_caches=3
./start-model.sh qwen3.6-35b-a3b-prismaquant-dflash
./configure-local-provider.sh qwen3.6-35b-a3b-prismaquant-dflash
```

Always drop caches between model switches — Thor's unified memory is not
automatically freed.

## 6. Model Profiles

### Qwen3.6 profiles (v6 container, production)

Tok/s = **single peak / aggregate@N-concurrent** under matched methodology
(coding prompts, `enable_thinking: false`, temp 0.2, 1200-tok outputs, 2026-04-22
drafter main). The drafter is unpinned — numbers shift with upstream z-lab
releases.

| Profile | Tok/s | KV Tokens | Seqs | KV dtype | Spec Method | Notes |
|---------|-------|-----------|------|----------|-------------|-------|
| `qwen3.6-35b-a3b-prismaquant-dflash` | **50.7 / 142.4@5** | 938K | 5 | BF16 | DFlash-15 | **★★ DEFAULT**. Mixed-precision 4.75 bpp, claimed −0.56 pp vs BF16 |
| `qwen3.6-35b-a3b-nvfp4-dflash` | 44.6 / 140.2@5 | 678K | 5 | BF16 | DFlash-15 | Uniform NVFP4 fallback. Lighter weights (~19 GB) |
| `qwen3.6-35b-a3b-fp8-dflash` | **47.6** | ~700K | 4 | BF16 | DFlash-15 | Best FP8 throughput (historical — re-measure) |
| `qwen3.6-35b-a3b-nvfp4-tq-mtp` | 28.6 | 2.22M | 8 | TQ K8V4 | MTP N=4 | MAX CONTEXT, 153 tok/s @ 8-conc. Requires PR #39931 mod |
| `qwen3.6-35b-a3b-fp8-mtp-fp8kv` | 25.7 | 1.44M | 8 | FP8 | MTP N=4 | FP8 weights + FP8 KV |
| `qwen3.6-35b-a3b-fp8-turboquant` | 26.2 | 1.89M | 6 | TQ K8V4 | MTP N=4 | FP8 weights + TQ KV |

All Qwen3.6 profiles use `--attention-backend flash_attn` (DFlash) or FlashInfer
(MTP/TQ), `--enable-auto-tool-choice --tool-call-parser qwen3_xml`, and
`--enforce-eager`. DFlash profiles require HF token for the gated drafter model
(z-lab/Qwen3.6-35B-A3B-DFlash).

### Legacy Qwen3.5 profile

| Profile | Model | Seqs | Agents | Notes |
|---------|-------|------|--------|-------|
| `qwen3.5-9b-claude-distilled-nvfp4` | 9B VLM | 8 | 2 | Claude distilled, multimodal, 0.4 GPU mem |

### Gemma 4 profiles (all containers)

| Profile | Model | Seqs | Agents | Notes |
|---------|-------|------|--------|-------|
| `gemma4-e4b-it` | 8B MoE (4B active) | 12 | 3 | Vision+text+audio, BF16, 0.4 GPU mem |
| `gemma4-31b-it-nvfp4` | 31B dense | 6 | 6 | Vision+text, NVFP4 |
| `gemma4-26b-a4b-it` | 26B MoE | 17 | 4 | Vision+text, BF16 |

**Seqs** = max concurrent sequences in vLLM. **Agents** = max concurrent
OpenClaw main agents (subagents fill remaining slots automatically).

**Default profile**: `qwen3.6-35b-a3b-fp8-dflash` (best overall for agentic work).

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
./start-model.sh qwen3.6-35b-a3b-nvfp4-mtp-fp8kv
```

Persistent defaults are saved in:

```text
~/.config/nemoclaw-thor/config.env
```

### DFlash profiles (speculative decoding)

Profiles ending in `-dflash` use block diffusion speculative decoding with
a ~0.5B drafter model that generates 15 tokens in a single diffusion step.
The target model then verifies the draft in one forward pass.

**Key architecture advantage**: Qwen3.6-35B-A3B has `head_dim=128` (unlike
Qwen3.5 which had `head_dim=256`). FA2 works natively on SM110 for
head_dim<=128, so DFlash runs with `--attention-backend flash_attn` without
any runtime mods — matching z-lab's tested configuration exactly.

DFlash profile requirements:
- **flash_attn attention backend** — works natively on SM110 with head_dim=128
- **BF16 KV cache** — FP8 KV is incompatible with DFlash non-causal verification
- **HF token** — z-lab/Qwen3.6-35B-A3B-DFlash is gated (mount `~/.cache/huggingface`)
- **Reduced max_num_seqs** — BF16 KV + drafter model uses more memory (4 seqs)

DFlash profiles have fewer concurrent sequences but 4-5x higher per-request
throughput (54.7 tok/s vs 11.6 baseline).

### MTP + KV compression profiles

Profiles with `-mtp` use built-in Multi-Token Prediction heads (zero drafter
overhead). Combined with FP8 KV or TurboQuant K8V4 for KV cache compression:

- **MTP N=4**: 4 speculative tokens per step, ~76% acceptance
- **FP8 KV**: ~2x KV compression (1.44-1.68M tokens)
- **TurboQuant K8V4**: ~2.6x KV compression (1.89-2.22M tokens), requires
  vLLM with PR #39931 (baked in v6 image)

MTP profiles maximize context capacity at moderate throughput (25-28 tok/s).

Gemma 4 models do NOT have DFlash or MTP profiles — no DFlash drafters exist,
and head_dim=256/512 is incompatible with flash_attn on SM110.

## 7. Image Builds

### Full rebuild (FlashInfer + vLLM from source)

```bash
cd docker/
./build-vllm.sh --skip-flashinfer --skip-vllm   # reuse cached wheels
./build-vllm.sh                                  # full rebuild from main
./build-vllm.sh --vllm-ref v0.8.5               # pin vLLM version
```

This produces `nemoclaw-thor/vllm:latest` via a multi-stage Dockerfile.
See `./build-vllm.sh --help` for all options.

### Overlay build (add packages without rebuilding)

To add Python packages on top of the existing image without rebuilding
FlashInfer/vLLM from source:

```bash
cd ~/workspaces/nemoclaw/src/NemoClaw-Thor

# Preserve current image as a backup tag
docker tag nemoclaw-thor/vllm:latest nemoclaw-thor/vllm:v4-base

# Build overlay (seconds, not hours)
docker build -f docker/Dockerfile.overlay -t nemoclaw-thor/vllm:latest docker/
```

Edit `docker/Dockerfile.overlay` to add more packages. The overlay
inherits everything from the base image and just adds a thin layer.

## 8. First Launch — FlashInfer JIT Compilation

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
| `/root/.cache/torch` | `~/thor-torch-cache` | Triton kernel cache, inductor cubins |

If any of these host directories are deleted, the next launch repeats
the corresponding compilation step.

Memory usage during compilation peaks significantly above normal operating
levels. This is the primary reason swap is required (see Section 2).

## 9. Use The Sandbox

Connect:

```bash
nemoclaw thor-v5 connect
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
openshell sandbox upload thor-v5 /path/to/repo /sandbox/myrepo
```

Then connect and work in the copy:

```bash
nemoclaw thor-v5 connect
cd /sandbox/myrepo
openclaw tui
```

Important:

- `openshell sandbox upload` copies files into the sandbox.
- Agent edits stay in `/sandbox/myrepo`.
- Your host repo is not changed automatically.

Copy the repo back out:

```bash
openshell sandbox download thor-v5 /sandbox/myrepo /path/to/output-copy
```

Or copy out only a patch:

```bash
# inside the sandbox
cd /sandbox/myrepo
git diff > /sandbox/myrepo.patch
```

Then on the host:

```bash
openshell sandbox download thor-v5 /sandbox/myrepo.patch .
```

## 10. Stop Without Uninstalling

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

## 11. Practical Rules

- **Swap must be active** before starting any model (see Section 2).
  Verify with `swapon --show` before launching `start-model.sh`.
- Use `nemoclaw thor-v5 connect` as the normal shell entrypoint.
- Use `openclaw tui` as the normal prompt UI.
- If the TUI shows "gateway disconnected": `HOME=/sandbox openclaw gateway run &`
- After any model change, run `./configure-local-provider.sh <profile>`.
- After a reboot: verify swap (`swapon --show`), then `./start-model.sh`,
  `./configure-local-provider.sh`, `./status.sh`, `nemoclaw thor-v5 connect`.
- If you stop vLLM, always run `sudo sync && sudo sysctl -w vm.drop_caches=3`
  before loading another model.
- First launch of a new model profile takes 20-60 min for kernel compilation
  (see Section 7). Subsequent launches take 3-8 min.

## 12. Why configure-local-provider.sh Is Needed

`nemoclaw onboard` bakes defaults that don't work for local vLLM inference:

| Setting | Onboard default | What we need | Why |
|---------|----------------|--------------|-----|
| `baseUrl` | `https://inference.local/v1` | Keep `https://inference.local/v1` in the sandbox and repoint the OpenShell provider target to `http://host.openshell.internal:8000/v1` or `http://host.openshell.internal:8888/v1` in ManyForge mode | In the current build the embedded OpenClaw client works through `inference.local`; the provider target controls whether requests go direct to vLLM or through the ManyForge mux |
| `contextWindow` | 131072 | 262144 | Models support 256K context |
| `maxTokens` | 4096 | 16384 | Agent needs long outputs for code generation |
| `timeoutSeconds` | (unset) | 1800 | Long reasoning sessions need 30min timeout |
| Concurrency | (unset) | Per-profile | Matches vLLM max_num_seqs budget |

The script patches these via `kubectl exec` into the sandbox. This bypasses
Landlock (kubectl exec starts a new process, not a child of the sandbox
entrypoint) and DAC restrictions (runs as root).

## 13. Key Differences From v5

| Aspect | v5 | v6 |
|--------|----|----|
| vLLM image | 0.19.1rc1.dev195 (e281cb721) | dev356 + PR #39931 (TurboQuant hybrid) |
| Default model | `qwen3.5-27b-claude-distilled-v2-nvfp4` | `qwen3.6-35b-a3b-fp8-dflash` |
| Best throughput | ~12 tok/s (27B NVFP4, MTP N=1) | **54.7 tok/s** (35B NVFP4, DFlash-15) |
| DFlash backend | FlashInfer + 35 runtime mods | flash_attn native (head_dim=128, no mods) |
| Runtime mods | 35 mods in docker/mods/ | **All deleted** — clean install |
| KV compression | FP8 only | FP8, TurboQuant K8V4 (2.6x), BF16 |
| MTP tokens | N=1-2 (`qwen3_next_mtp`) | N=4 (`mtp` method) |
| Tool parser | `qwen3_xml` + `--reasoning-parser qwen3` | `qwen3_xml` only |
| Chat template | `qwen3-tool-call-compat.jinja` (all Qwen) | Removed for Qwen3.6 (native support) |
| HF token mount | Not present | Added for gated DFlash drafter |
| VLLM_DISABLED_KERNELS | 2 kernels | 3 kernels (+CutlassFp8BlockScaledMMKernel) |
| transformers | ==5.5.0 | >=5.5.4 (Qwen3.6 support) |
