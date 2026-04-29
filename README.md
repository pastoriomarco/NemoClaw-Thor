# NemoClaw-Thor

Local-first NemoClaw/OpenShell integration for Jetson AGX Thor (SM110a / Blackwell).

> **📖 Operator manual**: this README is a landing page / quickstart.
> For the full step-by-step procedure — swap setup, image rebuild, JIT
> compile expectations, sandbox workflows, cleanup procedure,
> troubleshooting — see
> [**USER_QUICKSTART_MANUAL.md**](USER_QUICKSTART_MANUAL.md).
> Additional deep-dive docs: [KV-CACHE-BUDGET.md](KV-CACHE-BUDGET.md),
> [DFLASH-INVESTIGATION.md](DFLASH-INVESTIGATION.md),
> [TOOL-EVAL-BENCH-THOR.md](TOOL-EVAL-BENCH-THOR.md),
> [docker/NOTES.md](docker/NOTES.md).

> **⚠ Benchmark methodology**: DFlash throughput numbers in this repo
> were all measured with **coding prompts**, **`enable_thinking: false`**,
> **temperature ≈ 0.2**, and **~1200-token outputs**. Running naive prompts
> with default thinking mode produces ~30 tps on a profile that can do 45+
> tps under the intended workload. Always record the drafter SHA
> (`z-lab/Qwen3.6-35B-A3B-DFlash` is unpinned, so numbers drift with
> upstream).

## Quick start

From scratch, using the validated v6-pinned image and default profile
(PrismaQuant + DFlash-15 — **50.7 tok/s single peak, 142.4 tok/s aggregate @ 5-concurrent**,
matched methodology: coding prompts + `enable_thinking: false` + temp 0.2):

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
`qwen3.6-35b-a3b-prismaquant-dflash` (mixed-precision 4.75 bpp, claimed
quality within −0.56 pp of BF16 vs uniform NVFP4's −2.21 pp).

For **lower weight memory** or **max-context / low-latency-critical** paths, the
uniform NVFP4 variant is a close fallback:

```bash
./start-model.sh qwen3.6-35b-a3b-nvfp4-dflash
./configure-local-provider.sh qwen3.6-35b-a3b-nvfp4-dflash
```

Numbers (matched methodology): 44.6 tok/s single peak, 140.2 @ 5-concurrent.
~11–16% behind PrismaQuant at low-to-mid concurrency; tied at saturation.

For **many concurrent sequences or huge context**, use the TQ-MTP variant:

```bash
./start-model.sh qwen3.6-35b-a3b-nvfp4-tq-mtp
./configure-local-provider.sh qwen3.6-35b-a3b-nvfp4-tq-mtp
```

Trade-off: 28.6 tok/s single but **2.22M KV tokens**, 29× concurrency at 256K
context, and 154.7 tok/s aggregate at 8-concurrent. Requires the
`fix-pr39931-turboquant` runtime mod (auto-applied). See
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

# Terminal 1: start vLLM with the default profile
./start-model.sh                       # loads qwen3.6-35b-a3b-prismaquant-dflash

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
./configure-local-provider.sh --with-manyforge-mux qwen3.6-35b-a3b-prismaquant-dflash
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

Tok/s columns are **single peak / aggregate at N-conc** under matched
methodology (coding prompts, `enable_thinking: false`, temp 0.2, 1200-tok
outputs). Numbers are against the 2026-04-22 drafter main — drafter is
unpinned so these shift with upstream z-lab releases; always re-measure
after a version bump.

| Profile | Tok/s | KV Tokens | Seqs | Spec Method | Notes |
|---------|-------|-----------|------|-------------|-------|
| `qwen3.6-35b-a3b-prismaquant-dflash` | **50.7 / 142.4@5** | 938K | 5 | DFlash-15 | **★★ DEFAULT** — mixed-precision 4.75 bpp, best on every axis, −0.56 pp vs BF16 claimed |
| `qwen3.6-35b-a3b-nvfp4-dflash` | 44.6 / 140.2@5 | 678K | 5 | DFlash-15 | Uniform NVFP4, fallback (lighter weights, max concurrent seqs can stretch to 8) |
| `qwen3.6-35b-a3b-fp8-dflash` | **47.6** | ~700K | 4 | DFlash-15 | Best FP8 (historical — re-measure) |
| `qwen3.6-35b-a3b-nvfp4-tq-mtp` | 28.6 | 2.22M | 8 | MTP N=4 | MAX CONTEXT, 153 tok/s @ 8-conc |
| `qwen3.6-35b-a3b-fp8-mtp-fp8kv` | 25.7 | 1.44M | 8 | MTP N=4 | FP8+FP8 KV |
| `qwen3.6-35b-a3b-fp8-turboquant` | 26.2 | 1.89M | 6 | MTP N=4 | FP8+TQ KV |

### Legacy / other

| Profile | Model | Seqs | Notes |
|---------|-------|------|-------|
| `qwen3.5-9b-claude-distilled-nvfp4` | 9B VLM | 8 | Multimodal, Claude-distilled |
| `gemma4-e4b-it` | 8B MoE | 12 | Vision+text+audio |
| `gemma4-31b-it-nvfp4` | 31B dense | 6 | Vision+text, NVFP4 |
| `gemma4-26b-a4b-it` | 26B MoE | 17 | Vision+text, BF16 |

**Default profile**: `qwen3.6-35b-a3b-prismaquant-dflash` — what
`./start-model.sh` (no args) loads. Beats the uniform NVFP4 variant on
single-stream and all tested concurrency levels (matched methodology, today's
drafter). ~22 GB weights + the DFlash drafter (gated — HF token required).

If you need max context or fewer weight GB (e.g. when running other services
alongside vLLM), fall back to the uniform-NVFP4 variant:

```bash
./start-model.sh qwen3.6-35b-a3b-nvfp4-dflash
```

If you can't use NVFP4 at all (no HF token, or prefer FP8 weights), run:
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

## Building images

The repo produces two independent runtime images (vLLM and TRT-Edge-LLM) plus
a production bundle of the vLLM image with baked-in JIT caches.

| Goal | Command | Dockerfile | Output image |
|---|---|---|---|
| Build/rebuild vLLM | `cd docker && ./build-vllm.sh` | `Dockerfile.vllm` | `nemoclaw-thor/vllm:<tag>` + `:latest` |
| Build/rebuild TRT-Edge-LLM | `cd docker && ./build-trt.sh` | `Dockerfile.trt` | `nemoclaw-thor/trt-edge-llm:<tag>` + `:latest` |
| Build vLLM production bundle | `cd docker && ./bundle.sh` | `Dockerfile.bundle` | `nemoclaw-thor/vllm:<tag>-bundled` |
| Add a package without full rebuild | `cd docker && docker build -f Dockerfile.overlay -t nemoclaw-thor/vllm:latest .` | `Dockerfile.overlay` | overrides `:latest` |

Each `build-*.sh` accepts `--help` for arg reference. Both vLLM and TRT
builds share apt cache (`id=apt-cache-thor`) and pip cache mounts so package
downloads done by either build are reused by the other on subsequent runs.

vLLM and TRT-Edge-LLM images are independent — no inheritance — so you can
delete or rebuild either without affecting the other. They co-exist on disk
fine; the host filesystem deduplicates layers where possible.

For the **runtime tradeoff** between vLLM and TRT-Edge-LLM (memory, throughput,
which model classes work best with which runtime), see `PERFORMANCE-V7.md`.

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
│   ├── Dockerfile.vllm         # Multi-stage vLLM build for SM110
│   ├── Dockerfile.trt          # TensorRT-Edge-LLM standalone build for SM110
│   ├── Dockerfile.bundle       # vLLM bundled with baked-in JIT caches (production)
│   ├── Dockerfile.overlay      # Quick add-package overlay (dev convenience)
│   ├── build-vllm.sh           # vLLM build orchestration (multi-phase)
│   ├── build-trt.sh            # TRT-Edge-LLM build orchestration (single-stage)
│   ├── bundle.sh               # vLLM production bundle wrapper
│   ├── patches/                # Build-time patches (FlashInfer, TRT-Edge-LLM)
│   └── NOTES.md                # SM110 compatibility map, build history
└── KV-CACHE-BUDGET.md          # Memory planning reference
```

## References

- [NemoClaw](https://github.com/NVIDIA/NemoClaw) — sandbox framework
- [OpenShell](https://github.com/NVIDIA/OpenShell) — container orchestration
- [OpenClaw](https://github.com/openclaw/openclaw) — agent runtime
- [PLAN-v5-transition.md](PLAN-v5-transition.md) — v5 upgrade plan (historical)
- [DFLASH-INVESTIGATION.md](DFLASH-INVESTIGATION.md) — DFlash speculative decoding investigation and results
