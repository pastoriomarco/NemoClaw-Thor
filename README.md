# NemoClaw-Thor

Local-first NemoClaw/OpenShell integration for Jetson AGX Thor (SM110a / Blackwell).

> **For LLMs** (Claude, ChatGPT, agentic IDEs): read
> [`AGENTS.md`](AGENTS.md) before making any change. It tells you what
> this repo is, what it does NOT own (specs live in `manyforge_specs`;
> orchestration lives in `manyforge`), and the branch + commit workflow
> that applies to LLM contributors. Full contributor reference is
> [`CONTRIBUTING.md`](CONTRIBUTING.md). For the full **doc map** see
> [`INDEX.md`](INDEX.md); for **plan/report status** (live + archived) see
> [`PLANS_INDEX.md`](PLANS_INDEX.md).

> **📖 Operator manual**: this README is a landing page / quickstart.
> For the full step-by-step procedure — swap setup, image rebuild, JIT
> compile expectations, sandbox workflows, cleanup procedure,
> troubleshooting — see
> [**USER_QUICKSTART_MANUAL.md**](USER_QUICKSTART_MANUAL.md).
> Additional deep-dive docs: [serving/docs/KV-CACHE-BUDGET.md](serving/docs/KV-CACHE-BUDGET.md),
> [serving/docs/DFLASH-INVESTIGATION.md](serving/docs/DFLASH-INVESTIGATION.md),
> [serving/docs/TOOL-EVAL-BENCH-THOR.md](serving/docs/TOOL-EVAL-BENCH-THOR.md),
> [serving/docker/NOTES.md](serving/docker/NOTES.md).

> **⚠ Benchmark methodology**: DFlash throughput numbers in this repo
> were all measured with **coding prompts**, **`enable_thinking: false`**,
> **temperature ≈ 0.2**, and **~1200-token outputs**. Running naive prompts
> with default thinking mode produces ~30 tps on a profile that can do 45+
> tps under the intended workload. Always record the drafter SHA
> (`z-lab/Qwen3.6-35B-A3B-DFlash` is unpinned, so numbers drift with
> upstream).

## Quick start

From scratch, using the default ManyForge assistant profile
(`gemma4-12b-it-gguf`, Gemma 4 12B GGUF served by llama.cpp):

```bash
# Terminal 1: start the default model
cd ~/workspaces/dev_ws/src/NemoClaw-Thor
./serving/start-model.sh

# Terminal 2: wire the sandbox + sanity-check
./setup/configure-local-provider.sh
./setup/status.sh
nemoclaw my-assistant connect          # inside sandbox: `openclaw tui`
```

`./serving/start-model.sh` with no args picks up the default profile
`gemma4-12b-it-gguf`. This matches the ManyForge launcher default and the
current OpenClaw-lane evidence: 52/66 on the 2026-06-07 model sweep and 51/66
in the 2026-06-09 three-lane head-to-head (75 total / 66 scored). See
[`manyforge/docs/PHASE-5-PRODUCTION-DECISION.md`](manyforge/docs/PHASE-5-PRODUCTION-DECISION.md)
for the decision record.

For **higher single-stream throughput** at the cost of a less-reliable
OpenClaw lane, the 35B Qwen variant remains the documented alternative from
the PrismaQuant + DFlash-15 benchmark set:

```bash
./serving/start-model.sh qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge
./setup/configure-local-provider.sh qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge
```

The Qwen profile is faster on direct/single-prompt workloads, but default
selection is based on the ManyForge assistant smoke corpus, not raw token
throughput.

For **many concurrent sequences or huge context**, use the TQ-MTP variant:

```bash
./serving/start-model.sh qwen3.6-35b-a3b-nvfp4-tq-mtp
./setup/configure-local-provider.sh qwen3.6-35b-a3b-nvfp4-tq-mtp
```

Trade-off: 28.6 tok/s single but **2.22M KV tokens**, 29× concurrency at 256K
context, and 154.7 tok/s aggregate at 8-concurrent. Requires the
`fix-pr39931-turboquant` runtime mod (auto-applied). See
[Model profiles](#model-profiles) below for the full comparison.

Prerequisites: 32 GiB swap active, HF token at `~/.cache/huggingface/token`
(for the gated DFlash drafter), NemoClaw+OpenShell installed (see
[Usage](#usage) for install commands).

## Stack

| Component | Pin source |
|-----------|------------|
| NemoClaw, OpenShell CLI, OpenShell cluster, OpenClaw | See [VERSIONS.md § A](VERSIONS.md#a-setup--control-plane). |
| vLLM image | Custom SM110 build owned by this repo. See [VERSIONS.md § B](VERSIONS.md#b-model-serving-container) for the canonical image generation and per-pin status; `serving/docker/Dockerfile.vllm` header carries the per-pin rationale. |
| Sandbox | Created via `nemoclaw onboard`. Landlock + seccomp + netns. |
| Provider | `vllm-local` route on the OpenShell gateway. Direct HTTP to host vLLM (`:8000`); see `setup/configure-local-provider.sh`. |
| ManyForge integration | See [VERSIONS.md § C](VERSIONS.md#c-manyforge--nemoclaw-integration) for the active phase and artifacts. |

**Authoritative version reference**: [VERSIONS.md](VERSIONS.md) is the
single source of truth across all three scopes. Reproduction commands for
the control-plane bring-up live in
[USER_QUICKSTART_MANUAL.md](USER_QUICKSTART_MANUAL.md).

## Scripts

| Script | Purpose |
|--------|---------|
| `serving/start-model.sh <profile>` | Launch vLLM with a model profile |
| `setup/configure-local-provider.sh [OPTIONS] [profile]` | Wire OpenShell provider + patch sandbox config |
| `setup/status.sh [profile]` | System health checks |

## Usage

### First time (after fresh `nemoclaw onboard`)

```bash
cd ~/workspaces/dev_ws/src/NemoClaw-Thor

# Terminal 1: start the default local model
./serving/start-model.sh               # loads gemma4-12b-it-gguf

# Terminal 2: configure and verify
./setup/configure-local-provider.sh    # picks up the same default
./setup/status.sh
nemoclaw my-assistant connect
```

Pass a profile name to either script to pick a non-default (e.g.
`./serving/start-model.sh qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge` for the
35B Qwen profile — slower OpenClaw lane on this model, see comparison doc).

### ManyForge Composer-assistant (production default)

The Composer-assistant lane (`openclaw` provider, in-sandbox gateway,
manyforge MCP bridge) is the supported user-facing path. Bring it up
end-to-end after the local model server is running:

```bash
# 1. Provision the sandbox (idempotent — policy + skill + MCP register +
#    agent profile + workspace AGENTS.md). Uses the model that's currently
#    being served by vLLM.
./manyforge/setup-manyforge-assistant.sh my-assistant

# 2. Start the OpenClaw assistant bridge on :8200
./manyforge/start-openclaw-assistant-bridge.sh &

# 3. Start Composer in OpenClaw mode (defaults to ASSISTANT_PROVIDER=openclaw)
cd ~/workspaces/dev_ws/src/manyforge
./scripts/demo-assistant-known-good.sh start
```

**Production default = OpenClaw + gemma4-12b-it-gguf.** The decision and
benchmark data are in
[manyforge/docs/PHASE-5-PRODUCTION-DECISION.md](manyforge/docs/PHASE-5-PRODUCTION-DECISION.md);
the operational runbook is
[manyforge/docs/COMPOSER-ASSISTANT-RUNBOOK.md](manyforge/docs/COMPOSER-ASSISTANT-RUNBOOK.md);
the lane parity debug tooling lives at
[manyforge/scripts/debug/](manyforge/scripts/debug/).

To swap to the direct lane (fast-path for simple prompts; sandboxed
bypass — bridge runs its own loop with a tool_choice pin):

```bash
# manyforge's unified launcher tears the assistant stack down
# and brings it back up with the new provider.
ASSISTANT_PROVIDER=direct ./scripts/demo-assistant-known-good.sh restart
```

### After reboot

Same sequence: `serving/start-model.sh`, then `setup/configure-local-provider.sh`,
then `setup/status.sh`. To restore the Composer-assistant lane on top of
that, re-run the three steps in the previous block (the provisioner and
bridge are idempotent; Composer's container is recreated by the launcher).

### Switch model

Stop vLLM (Ctrl-C), drop caches, start new model, reconfigure:

```bash
sudo sync && sudo sysctl -w vm.drop_caches=3
./serving/start-model.sh <new-profile>
./setup/configure-local-provider.sh <new-profile>
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
| `qwen3.6-35b-a3b-prismaquant-dflash` | 50.7 / 142.4@5 | 938K | 5 | DFlash-15 | Best raw throughput; mixed-precision 4.75 bpp, claimed −0.56 pp vs BF16 |
| `qwen3.6-35b-a3b-nvfp4-dflash` | 44.6 / 140.2@5 | 678K | 5 | DFlash-15 | Uniform NVFP4 (lighter weights, max concurrent seqs can stretch to 8) |
| `qwen3.6-35b-a3b-fp8-dflash` | **47.6** | ~700K | 4 | DFlash-15 | Best FP8 (historical — re-measure) |
| `qwen3.6-35b-a3b-nvfp4-tq-mtp` | 28.6 | 2.22M | 8 | MTP N=4 | MAX CONTEXT, 153 tok/s @ 8-conc |
| `qwen3.6-35b-a3b-fp8-mtp-fp8kv` | 25.7 | 1.44M | 8 | MTP N=4 | FP8+FP8 KV |
| `qwen3.6-35b-a3b-fp8-turboquant` | 26.2 | 1.89M | 6 | MTP N=4 | FP8+TQ KV |

### Legacy / other

| Profile | Model | Seqs | Notes |
|---------|-------|------|-------|
| `qwen3.5-9b-claude-distilled-nvfp4` | 9B VLM | 8 | Multimodal, Claude-distilled |
| `gemma4-e4b-it` | 8B MoE | 12 | Vision+text+audio |
| `gemma4-26b-a4b-it` | 26B MoE | 17 | Vision+text, BF16 |

**Default profile**: `gemma4-12b-it-gguf` — what
`./serving/start-model.sh` (no args) loads on a clean config. Gemma 4 12B GGUF
is served by llama.cpp with the E2B speculative draft and is the current
ManyForge assistant default. Footprint is much smaller than the historical
Cosmos/vLLM anchor; persisted `THOR_MODEL_PROFILE` or an explicit CLI profile
still overrides the clean-start default.

If you need higher single-stream throughput (and don't depend on the OpenClaw
lane's reliability — e.g. direct-bridge usage), fall back to the 35B Qwen
profile:

```bash
./serving/start-model.sh qwen3.6-35b-a3b-nvfp4-dflash
```

If you can't use NVFP4 at all (no HF token, or prefer FP8 weights), run:
`./serving/start-model.sh qwen3.6-35b-a3b-fp8-dflash`.

### Containerized DeepSeek-V4-Flash-0731 / DS4

DS4 is an alternative, isolated Docker service rather than a vLLM model
profile. It preserves the active vLLM/ManyForge service on `:8000` and serves
the matching 0731 base + DSpark drafter on `127.0.0.1:8050` by default:

```bash
./serving/start-ds4.sh start
./serving/start-ds4.sh logs
```

For a LAN client, bind only Thor's intended LAN address:

```bash
DS4_BIND_ADDRESS=192.168.1.136 ./serving/start-ds4.sh start
```

The first start builds the `sm_110` image inside Docker and resumes the model
download into `~/thor-hf-cache/ds4/`; later starts reuse the persistent weights
and KV cache. The current tested settings, container files, operations, and
Thor-specific safety limits are in
[`serving/docs/DS4-ON-THOR.md`](serving/docs/DS4-ON-THOR.md).

## Architecture

```
Host (Jetson AGX Thor)
├── Local model server (Docker, --network host, port 8000)
│   └── Model serving: Gemma GGUF default, plus vLLM profiles for bake-offs
├── OpenShell gateway (K3s, openshell-cluster-nemoclaw)
│   ├── L7 proxy (10.200.0.1:3128) — TLS termination, policy enforcement
│   └── Inference route:
│       • direct mode      → vllm-local → host:8000
│       • ManyForge default → vllm-local → host:8000; Composer talks to
│         openclaw_assistant_bridge(:8200) or direct bridge(:8100)
│       • legacy mux diagnostics → vllm-local → host:8888
└── Sandbox pod (thor-v5)
    ├── OpenClaw gateway (port 18789) — agent orchestration
    ├── OpenClaw agent — LLM-powered task execution
    └── Workspace (/sandbox/workspace → /sandbox/.openclaw-data/workspace)
```

### Why configure-local-provider.sh is needed

`nemoclaw onboard` bakes defaults that don't work for our local runtime modes:

| Setting | Onboard default | What we need | Why |
|---------|----------------|--------------|-----|
| `baseUrl` | `https://inference.local/v1` | Keep `https://inference.local/v1` in the sandbox, but repoint the OpenShell provider target to `http://host.openshell.internal:8000/v1` by default | In this build the sandbox/OpenClaw client works through the proxy route; Composer reaches ManyForge through the assistant-provider bridge, not the old mux |
| `contextWindow` | 131072 | 262144 | Models support 256K context |
| `maxTokens` | 4096 | 16384 | Agent needs long outputs for code generation |
| `timeoutSeconds` | (unset) | 1800 | Long reasoning sessions need 30min timeout |
| Concurrency | (unset) | Per-profile | Matches vLLM max_num_seqs budget |

The `setup/configure-local-provider.sh` script patches these via `kubectl exec` into
the sandbox. This bypasses Landlock (kubectl exec starts a new process, not a
child of the sandbox entrypoint) and DAC restrictions (runs as root).

Legacy mux diagnostics can still persist mux state in
`~/.config/nemoclaw-thor/config.env`, but the current ManyForge assistant path
does not require mux.

## Building images

The repo produces two independent runtime images (vLLM and TRT-Edge-LLM) plus
a production bundle of the vLLM image with baked-in JIT caches.

| Goal | Command | Dockerfile | Output image |
|---|---|---|---|
| Build/rebuild vLLM | `cd serving/docker && ./build-vllm.sh` | `Dockerfile.vllm` | `nemoclaw-thor/vllm:<tag>` + `:latest` |
| Build/rebuild TRT-Edge-LLM | `cd serving/docker && ./build-trt.sh` | `Dockerfile.trt` | `nemoclaw-thor/trt-edge-llm:<tag>` + `:latest` |
| Build vLLM production bundle | `cd serving/docker && ./bundle.sh` | `Dockerfile.bundle` | `nemoclaw-thor/vllm:<tag>-bundled` |
| Add a package without full rebuild | `cd serving/docker && docker build -f Dockerfile.overlay -t nemoclaw-thor/vllm:latest .` | `Dockerfile.overlay` | overrides `:latest` |
| Build DS4 alternative service | `./serving/start-ds4.sh build` | `serving/docker/Dockerfile.ds4` | `nemoclaw-thor/ds4:v0.5.6.2-sm110-thor` |

Each `build-*.sh` accepts `--help` for arg reference. Both vLLM and TRT
builds share apt cache (`id=apt-cache-thor`) and pip cache mounts so package
downloads done by either build are reused by the other on subsequent runs.

vLLM and TRT-Edge-LLM images are independent — no inheritance — so you can
delete or rebuild either without affecting the other. They co-exist on disk
fine; the host filesystem deduplicates layers where possible.

For the **runtime tradeoff** between vLLM and TRT-Edge-LLM (memory, throughput,
which model classes work best with which runtime), see [`serving/docs/PERFORMANCE-V7.md`](serving/docs/PERFORMANCE-V7.md).

## Key files

```
NemoClaw-Thor/
├── README.md                       # This file (entrypoint)
├── AGENTS.md                       # Agent/LLM instructions for working in this repo
├── VERSIONS.md                     # Single source of truth for current versions across all scopes
├── USER_QUICKSTART_MANUAL.md       # Operator quickstart
│
├── setup/                          # Scope A: NemoClaw / OpenShell / OpenClaw control plane
│   ├── configure-local-provider.sh #   OpenShell provider + sandbox config sync
│   ├── status.sh                   #   System health checks
│   ├── checks.sh                   #   Diagnostic check functions (sourced)
│   ├── sandbox-runtime.sh          #   sync_sandbox_runtime_config(), sandbox helpers
│   ├── policies/                   #   Egress policy presets (dynamic/, static/)
│   └── NEMOCLAW-OPENCLAW-WORKFLOW.md
│
├── serving/                        # Scope B: vLLM model serving + benchmarks
│   ├── start-model.sh              #   vLLM launcher (picks profile, mounts caches)
│   ├── start-duo.sh                #   Two-model co-serving (Qwen3.6 + Cosmos-2B)
│   ├── config.sh                   #   Model profiles, concurrency math, runtime config
│   ├── launch.sh                   #   Docker run logic, cache mounts, env vars
│   ├── docker/                     #   Image builds: vLLM, TRT-Edge-LLM, bundle, overlay
│   ├── benchmarks/                 #   Loose perf probes (bench-throughput.py, etc.)
│   ├── agentic-bench/              #   lm-eval-harness wrappers + tool-call proxy
│   ├── templates/                  #   Tokenizer chat templates (qwen3 tool-call jinjas)
│   ├── scripts/                    #   Model-serving prep scripts (patch-minimax-w4a16-config.sh)
│   └── docs/                       #   KV-CACHE-BUDGET, PERFORMANCE-V*, TOOL-EVAL-BENCH-THOR,
│                                   #   *-INVESTIGATION.md, COSMOS-REASON2-32B-QUANTIZATION
│
└── manyforge/                      # Scope C: ManyForge ↔ NemoClaw integration
    ├── setup-manyforge-assistant.sh#   Idempotent provisioner (skill + preset + MCP register)
    ├── policies/                   #   manyforge-composer.preset.yaml (egress preset)
    ├── bridge/                     #   Audit-log mount point (bridge service lives in manyforge repo)
    └── docs/                       #   MANYFORGE-MCP-INTEGRATION, -ASSISTANT-DEPLOYMENT-PLAN,
                                    #   -PROFILE-CALIBRATION
```

## References

- [NemoClaw](https://github.com/NVIDIA/NemoClaw) — sandbox framework
- [OpenShell](https://github.com/NVIDIA/OpenShell) — container orchestration
- [OpenClaw](https://github.com/openclaw/openclaw) — agent runtime
- Historical v5 transition notes now live in git history.
- [serving/docs/DFLASH-INVESTIGATION.md](serving/docs/DFLASH-INVESTIGATION.md) — DFlash speculative decoding investigation and results

## Versioning, releases, security

- **Current version**: see [`VERSION`](VERSION) (single line, SemVer).
  Per-component pins (vLLM container, FlashInfer, NemoClaw CLI,
  OpenClaw, model profiles) live in [`VERSIONS.md`](VERSIONS.md) and
  move on external upstream releases, independent of the repo SemVer.
- **What changed**: [`CHANGELOG.md`](CHANGELOG.md) (Keep-a-Changelog
  format).
- **What's next**: [`ROADMAP.md`](ROADMAP.md).
- **License**: MIT — see [`LICENSE`](LICENSE). Upstream NemoClaw has
  its own license; NemoClaw-Thor is a sibling project, not a fork.
- **Reporting a security vulnerability**: see [`SECURITY.md`](SECURITY.md).
  Do NOT use public issues for security topics.

## Contributing

Both human contributors and LLM agents are first-class. The branch
workflow (`main` = released, `dev` = integration where work lands,
squash-merge `dev → main` on release) and the LLM-specific commit/push
rules are documented in [`CONTRIBUTING.md`](CONTRIBUTING.md) and
reinforced in [`AGENTS.md`](AGENTS.md). Special rule for this repo:
model-profile changes in `serving/config.sh` require operator approval
and a smoke-corpus retest before merge.

Compatible with **manyforge 0.1.x** for the assistant lane — see the
companion repo at <https://github.com/pastoriomarco/manyforge>.

## Acknowledgements

Originally forked from [`jetsonhacks/NemoClaw-Thor`](https://github.com/jetsonhacks/NemoClaw-Thor)
in March 2026. That project has since been archived (2026-04-17), and this
repository's scope has grown well beyond the original installer-script set —
adding a vLLM container build pipeline, an agentic evaluation harness, and the
ManyForge integration layer. As of May 2026 the fork relationship has been
detached on GitHub; this is now an independent project. Credit to the
original maintainer for the initial Thor onboarding scaffolding that made the
first weeks easier.
