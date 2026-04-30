# NemoClaw / OpenClaw Workflow on Thor

End-to-end recipe for running the ManyForge production agent stack on this
Thor host: a single `cosmos-reason2-8b` vLLM endpoint served on port 8000,
consumed by OpenClaw inside the `my-assistant` sandbox through an OpenShell
`vllm-local` provider route.

## Versions verified (2026-04-30)

| Component | Version |
|---|---|
| NemoClaw CLI (host) | `v0.0.31` |
| OpenShell CLI (host) | `0.0.36` |
| OpenShell cluster image | `0.0.36` |
| OpenClaw (in-sandbox agent) | `v2026.4.24` |
| vLLM image | `nemoclaw-thor/vllm:latest` (v8 container) |
| Reference model used in this doc | `nvidia/Cosmos-Reason2-8B` (text+vision, 32K→262K context capable). Other profiles in `lib/config.sh`. |
| Host | NVIDIA Thor (SM110) |

Update this table whenever a tested upgrade lands. The recipes below
were last validated against the 2026-04-30 versions; small command
shifts can occur between NemoClaw monthly releases and the workflow may
need touch-ups.

## One-time prerequisites

1. **Accept the Cosmos-Reason2-8B license** at https://huggingface.co/nvidia/Cosmos-Reason2-8B (gated repo).
2. **Save your HF token** to `~/.cache/huggingface/token`:
   ```bash
   huggingface-cli login    # interactive; stores ~/.cache/huggingface/token
   # or paste the token directly:
   # echo -n "hf_xxx" > ~/.cache/huggingface/token
   ```
3. **Download the model weights**:
   ```bash
   export HF_TOKEN="$(tr -d '[:space:]' < ~/.cache/huggingface/token)"
   HF_HUB_CACHE=~/thor-hf-cache/hub \
       hf download nvidia/Cosmos-Reason2-8B
   ```
   Weights land in `~/thor-hf-cache/hub/models--nvidia--Cosmos-Reason2-8B/` (~17 GB).
4. **Ensure the OpenShell cluster container and the `my-assistant` sandbox are up**:
   ```bash
   docker ps --format '{{.Names}}' | grep openshell-cluster-nemoclaw   # should print
   nemoclaw list                                                         # my-assistant should be listed
   ```

## First-time NemoClaw onboard recipe (v0.0.31+)

For a clean Thor host (no existing `my-assistant` sandbox), this is the
end-to-end onboarding the wizard expects. Validated 2026-04-30 against
NemoClaw `v0.0.31`, OpenShell CLI `0.0.36`, OpenShell cluster image
`0.0.36`, OpenClaw `v2026.4.24`.

### Order of operations matters

The NemoClaw wizard probes the inference endpoint as part of step 3/8.
**Start the vLLM container before running `nemoclaw onboard`** — the
wizard's bounded probe (PR #2714, fixed in v0.0.31) will fail-fast and
prompt re-entry if the endpoint is unreachable, which is annoying mid-
onboard.

```bash
# 1. Start the model first
cd /home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor
./start-model.sh <profile-slug>           # e.g. nemotron3-nano-omni-30b-a3b-nvfp4

# Wait for /v1/models to respond
curl -s http://127.0.0.1:8000/v1/models | head -c 300

# 2. Run the onboard wizard
nemoclaw onboard
```

### Wizard answers

The wizard walks through 8 steps. Answers that produced a working
`my-assistant` sandbox on Thor:

| Step | Prompt | Answer |
|---|---|---|
| 1/8 | Preflight checks | (automatic) |
| 2/8 | Starting OpenShell gateway | (automatic; first run pulls `ghcr.io/nvidia/openshell/cluster:0.0.36`, ~60-180s) |
| 3/8 | Inference options (1–7) | **`3`** — Other OpenAI-compatible endpoint |
| 3/8 | OpenAI-compatible base URL | **`http://127.0.0.1:8000/v1`** (host loopback; the wizard probes from the host. `host.openshell.internal:8000` is the sandbox-internal name and won't resolve here) |
| 3/8 | Endpoint model | **the served-model-name** (== profile slug per the alignment in `lib/config.sh`, e.g. `nemotron3-nano-omni-30b-a3b-nvfp4`) |
| 3/8 | Sandbox name | `my-assistant` (default; press Enter) |
| 3/8 | Apply this configuration? | `Y` |
| 4/8 | Setting up inference provider | (automatic; creates `compatible-endpoint` provider + route) |
| 4/8 | Enable Brave Web Search? | `N` |
| 5/8 | Messaging channels | (Enter — skip all) |
| 6/8 | Creating sandbox | (automatic; ~6 min on first run for the sandbox image build + upload to gateway) |
| 7/8 | Setting up OpenClaw inside sandbox | (automatic; gateway launched on port 18789) |
| 8/8 | Policy tier | **Balanced** (or Restricted; see notes below) |
| 8/8 | Policy presets | **`local-inference` only** (untick all Balanced defaults if you want minimum egress) |

### Policy preset choice — minimum egress

The sandbox image is fully built with all NemoClaw/OpenClaw deps baked in
during step 6/8. Runtime egress only needs to cover what the agent will
*do* at runtime. For ManyForge-assistant work the minimum surface is:

- ✅ **`local-inference`** — opens host gateway endpoints `host.openshell.internal:11434` (Ollama default), `:11435` (Ollama auth proxy, added in v0.0.21 by PR #2114), and **`:8000` (vLLM)**. Allowed binaries: `openclaw`, `claude`, `node`, `curl`, `python3`. Without this, the in-sandbox agent cannot reach the vLLM endpoint.
- All other presets OFF unless you have a specific need.

Add presets later with `nemoclaw <sandbox> policy-add <preset>` — no
re-onboard required (the v0.0.31 fix made this reliable).

### After onboard completes

The wizard prints a tokenized OpenClaw UI URL once:

```
Dashboard: http://127.0.0.1:18789/#token=<long-hex>
```

**Save this URL securely.** It contains an auth token treated as a
password and is not printed again. To reveal it later:

```bash
nemoclaw <sandbox> gateway-token              # prints just the token
```

**Run `configure-local-provider.sh` after onboard finishes.** This
step is required, not optional. Onboard creates the OpenShell-side
inference provider with the URL you typed at the wizard prompt
(typically `http://127.0.0.1:8000/v1`), but the **in-sandbox
`/sandbox/.openclaw/openclaw.json`** that the embedded agent runner
actually reads keeps the OpenShell-blueprint default
`https://inference.local/v1` — which OpenClaw's undici fetch cannot
reach (the L7-proxy issue documented in the appendix below).
`configure-local-provider.sh` performs the kubectl-exec patch that
overrides the in-sandbox `baseUrl` to
`http://host.openshell.internal:8000/v1`, plus the warmup, and is
the step that makes the assistant actually able to call the model.

```bash
./configure-local-provider.sh <profile-slug>
```

Without this step the agent dispatch fails with
`LLM request failed: network connection error` even though
`nemoclaw list` shows the sandbox correctly bound to the model.

The `configure-local-provider.sh` script also remains the tool for
**switching the active model on an existing sandbox** without
re-onboarding (same script, different `<profile-slug>`).

## Starting the vLLM endpoint

```bash
cd /home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor
./start-model.sh cosmos-reason2-8b
```

What this does (see [lib/launch.sh](lib/launch.sh) for the full profile):
- Pulls the `nemoclaw-thor/vllm:latest` container.
- Serves `nvidia/Cosmos-Reason2-8B` on `0.0.0.0:8000` (override with `THOR_VLLM_PORT=...`).
- Uses `flashinfer` attention backend (required for FP8 KV on SM110).
- Uses `TORCH_SDPA` for the ViT encoder (SM110 workaround, vllm #38411).
- Enables tool-calling with the **`hermes`** parser (not `qwen3_xml` — see caveat below).
- max_model_len = **65536** (see "Why 64K context matters" below).
- `gpu_memory_utilization=0.25` → ~30 GB total footprint: 6.4 GiB KV (93,696 tokens), ~16 GB weights, ~1.5 GB ViT, ~6 GB activations.

Readiness check:
```bash
curl -s http://127.0.0.1:8000/v1/models | head -c 300
```

### Running detached (non-interactive)

Use the same env hooks [start-duo.sh](start-duo.sh) uses:
```bash
export HF_TOKEN="$(tr -d '[:space:]' < ~/.cache/huggingface/token)"
THOR_DETACH=1 THOR_CONTAINER_NAME="nemoclaw-cosmos-reason2-8b" THOR_VLLM_PORT=8000 \
    ./start-model.sh cosmos-reason2-8b
```

## Wiring OpenShell to route OpenClaw to the local endpoint

> **When to use this script:** for **switching the active model on an
> existing sandbox** without re-onboarding. The first-time onboard
> wizard (above) already creates the inference provider, route, sandbox
> runtime config, in-sandbox gateway, and `local-inference` policy
> preset. `configure-local-provider.sh` remains the tool for changing
> the bound model on a sandbox you already have.

```bash
./configure-local-provider.sh cosmos-reason2-8b
```

What this does:
- Creates or updates the `vllm-local` provider to `http://host.openshell.internal:8000/v1`.
- Sets the OpenShell gateway inference route to `vllm-local / Cosmos-Reason2-8B`.
- Syncs the `my-assistant` sandbox runtime config (`~/.nemoclaw/sandboxes.json`).
- Applies the `local-inference` policy preset to the sandbox.
- Starts the in-sandbox OpenClaw gateway on port 18789 and sends a warmup request.

Expected tail of successful output:
```
  ✓  Inference route set to vllm-local / Cosmos-Reason2-8B (180s)
  ✓  Sandbox runtime config synced to Cosmos-Reason2-8B
  ✓  OpenClaw gateway started successfully
  ✓  Host vLLM endpoint is serving Cosmos-Reason2-8B
  ✓  Model warmup complete
```

## Running an OpenClaw agent task

There are two paths. Use whichever fits your need.

### A — Interactive (recommended for day-to-day coding)

```bash
nemoclaw my-assistant connect
```
Opens the OpenClaw TUI inside the sandbox. All inference goes through the gateway you just configured.

### B — Scripted / non-interactive

> **Important — gateway lifecycle.** `configure-local-provider.sh` starts the
> in-sandbox gateway via `nohup … &` + `disown`. That background process is
> reaped when the host-side SSH/`kubectl exec` session that started it exits.
> So by the time you open a *new* `kubectl exec` to run `openclaw agent`, the
> gateway is already gone and the agent falls back to the "embedded" path,
> which then looks for an Anthropic API key and fails.
>
> **Workaround:** start the gateway and dispatch the agent in the SAME
> `kubectl exec` invocation.

Working pattern:
```bash
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell my-assistant -- bash -c '
# Kill any stale openclaw/node processes (best-effort, idempotent)
for d in /proc/[0-9]*; do
    p="${d##*/}"
    c=$(cat "$d/comm" 2>/dev/null || true)
    case "$c" in openclaw*|node*) kill "$p" 2>/dev/null || true ;; esac
done
sleep 2
# Fresh gateway pinned to this session
HOME=/sandbox nohup openclaw gateway run --auth none --port 18789 \
    > /sandbox/openclaw-gateway.log 2>&1 &
for i in 1 2 3 4 5 6 7 8; do
    sleep 2
    ss -tlnp 2>/dev/null | grep -q ":18789" && break
done
# Dispatch the task. --session-id is required; pick any string.
HOME=/sandbox openclaw agent \
    --session-id <descriptive-id> \
    --json \
    --timeout 600 \
    -m "<your prompt here>"
'
```

Notes:
- `--session-id <anything>` avoids the "Pass --to <E.164>, --session-id, or --agent to choose a session" error.
- The agent writes human-readable replies to `/sandbox/openclaw-gateway.log`.
- The JSON dispatch result contains `status`, `summary`, `durationMs`, and `meta.agentMeta.model` confirming which model served the turn.

## Verified end-to-end sanity check

The following task was run against this exact recipe on 2026-04-20 and succeeded:

**Prompt:**
> Create `/tmp/cosmos_agent_test/fib.py` with function `fib(n)` that returns
> the nth Fibonacci number (fib(0)=0, fib(1)=1). Use whatever write/shell
> tools are available. After writing, cat the file back and confirm with a
> one-sentence summary. Keep responses short.

**Result:** 33 s round-trip; agent wrote a correct recursive implementation;
`fib(0)=0, fib(1)=1, fib(10)=55, fib(15)=610` all verified.

If this test fails, check — in order:
1. `curl http://127.0.0.1:8000/v1/models` — vLLM responsive?
2. `openshell inference get` — provider `vllm-local`, model `Cosmos-Reason2-8B`?
3. Gateway log inside the sandbox: `/sandbox/openclaw-gateway.log` — look for `[gateway] agent model: inference/Cosmos-Reason2-8B` and any `error=Context overflow`.
4. Run with `THOR_NO_RM=1 THOR_DETACH=1 THOR_CONTAINER_NAME=debug-cosmos …` to preserve the container so `docker logs` survives a crash.

## Stopping and freeing memory

```bash
docker stop nemoclaw-cosmos-reason2-8b
sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
free -g    # should show ~110+ GB available on Thor
```

The `drop_caches` step is part of the Thor memory-leak protocol — vLLM can
leave unified-memory pages pinned after a crash; dropping caches reclaims them.

## Caveats (why the recipe is the way it is)

### Why `hermes`, not `qwen3_xml`, as the tool-call parser
Cosmos-Reason-2 inherits the Qwen3-VL base chat template, which emits
tool calls as `<tool_call>{"name":"...","arguments":{...}}</tool_call>` —
a **hermes**-format JSON payload, NOT Qwen3.6's XML-attribute format.
Using `qwen3_xml` here silently fails: the model emits correct calls,
but the parser leaves `tool_calls: []` and dumps the raw text into
`content`. Verified empirically on both 2B and 8B.

### Why max_model_len = 65536, not 32768
OpenClaw's bootstrap injects a large system prompt (AGENTS.md, SOUL.md,
TOOLS.md, skills registry, tool schemas) that totals ~16 K tokens. The
default output budget is 16 K more. At `max_model_len=32768` this
overflows on turn 1 with:
```
This model's maximum context length is 32768 tokens. However, you requested
16384 output tokens and your prompt contains at least 16385 input tokens,
for a total of at least 32769 tokens.
```
Raising to 64 K gives comfortable headroom. Cosmos-Reason-2-8B natively
supports up to 262 144 tokens (`text_config.max_position_embeddings`), so
64 K is well inside the architected range.

### KV budget tradeoff at 64 K vs 32 K
At `gpu_memory_utilization=0.25`:
- 32 K context → 22.8 GiB KV = 332,640 tokens (9 × headroom at 3-conc)
- 64 K context → 6.4 GiB KV = 93,696 tokens (1.4 × a single 64 K session)

vLLM reserves more activation memory for a larger attention window, which
comes out of the same gpu_mem_util budget. For single-agent OpenClaw use
this is fine. If you need 3 concurrent agents at 64 K, raise
`gpu_memory_utilization` to 0.35–0.40 (env override:
`THOR_GPU_MEMORY_UTILIZATION=0.35 ./start-model.sh cosmos-reason2-8b`).

### HF_TOKEN must be in env, not just the cache file
Cosmos-Reason-2 is a gated repo. Even with the model weights on disk,
vLLM re-fetches `processor_config.json` at cold start, and Transformers'
gated-repo check specifically wants `HF_TOKEN` in the environment (the
mounted `~/.cache/huggingface/token` file is **not** consulted by that
code path). [start-duo.sh](start-duo.sh) handles this automatically;
manual launches must `export HF_TOKEN=…` before `./start-model.sh`.

### Cosmos-2B (not 8B) has fragile zero-arg tool calls
An earlier test run showed `Cosmos-Reason2-2B` emitting malformed JSON
for tools with no parameters: `{"name": "scene_get_objects", ""}` —
unparseable, `tool_calls: []`. The 8B variant does **not** have this
failure; it handles zero-arg tools cleanly. Stay on 8B for production
OpenClaw.

### No MTP / DFlash for Cosmos
No public speculative-decoding drafter exists for Cosmos-Reason-2.
The profile runs standard autoregressive decoding at ~14–15 tok/s. If
latency becomes a blocker, the alternative is to go back to the
Qwen3.6-manyforge profile (Thor-only, doesn't fit Orin's 40 GB budget).

## Orin AGX deployment notes

This recipe targets Orin AGX (64 GB) with 24 GB reserved for Isaac ROS
(Jetpack 7.2 + SBSA). Measured footprint of `cosmos-reason2-8b` at
`gpu_memory_utilization=0.25` is ~30 GB, which fits inside the 40 GB LLM
budget with ~10 GB of Isaac headroom.

Tweaks that may be worth empirical verification on Orin:
- If Orin enforces tighter memory pressure, drop `max_num_seqs` to 2 or
  lower `gpu_memory_utilization` to 0.22 (costs KV headroom).
- If 64 K overflow recurs, trim OpenClaw's bootstrap workspace files
  (AGENTS.md is ~8 K chars, largest single contributor).
- If only a single-user interactive agent session is expected,
  `max_num_seqs=1` frees ~2 GB of KV/activation reservation.

## Appendix: Why the wiring is the way it is

`configure-local-provider.sh` does something that looks convoluted at first
glance — it patches `/sandbox/.openclaw/openclaw.json` inside the running
sandbox pod via `kubectl exec` from the host, rather than using the more
obvious mechanisms (Docker `ENV`, NemoClaw's runtime `apply_model_override()`,
build-time ARGs). Every one of those obvious mechanisms is broken or
insufficient. This appendix explains why, so anyone touching the script
later doesn't have to rediscover the traps.

### OpenShell strips Docker ENV

The sandbox pod's PID 1 is `openshell-sandbox`, not the Docker `ENTRYPOINT`
you'd read from `docker inspect`. It replaces the process environment with
a sanitized set (proxy vars, TLS certs, PATH). Docker `ENV` directives —
including `NEMOCLAW_MODEL_OVERRIDE`, `NEMOCLAW_CONTEXT_WINDOW`,
`NEMOCLAW_LOCAL_INFERENCE_TIMEOUT`, etc. — are **not** propagated to the
NemoClaw entrypoint process. NemoClaw's `apply_model_override()` reads
the (stripped) environment and always sees empty.

### Runtime config patching is the only path that works

| Mechanism | Works? | Why |
|---|---|---|
| Docker `ENV` directives | NO | Stripped by OpenShell's PID-1 sanitizer |
| `NEMOCLAW_*` override env vars | NO | Reads the same stripped environment |
| Build-time ARGs in `Dockerfile.base` | PARTIAL | `baseUrl` can be baked, but `contextWindow`, `maxTokens`, `timeoutSeconds` are hardcoded in Python sources |
| Patching NemoClaw's upstream Dockerfile | NO | Not our repo; would break on every `git pull` |
| **`kubectl exec` runtime patching** | **YES** | Bypasses Landlock (fresh process); DAC handled by `chmod 644 → write → chmod 444`; config hash recomputed after write |

### What `sync_sandbox_runtime_config()` overwrites

| Setting | Onboard default | Our override |
|---|---|---|
| `models.providers.inference.baseUrl` | `https://inference.local/v1` | `http://host.openshell.internal:8000/v1` |
| `models.providers.inference.api` | `openai-completions` | `openai-completions` (unchanged; sanity-checked) |
| `models.providers.inference.contextWindow` | 131072 | profile-defined (e.g. 65536 for `cosmos-reason2-8b`) |
| `models.providers.inference.maxTokens` | 4096 | profile-defined (e.g. 16384) |
| `models.providers.inference.timeoutSeconds` | unset | 1800 |
| `models.providers.inference.maxConcurrent` | unset | profile-defined |
| `models.providers.inference.subagents` | unset | profile-defined |
| `/sandbox/.nemoclaw/config.json` (onboard state) | original model ref | updated model ref |

### Why Landlock's read-only `/sandbox/.openclaw` doesn't block us

Landlock is applied to a process tree and inherited by its descendants, but
**not by processes started fresh via `kubectl exec`**. Those get a clean
Landlock state. `chattr +i` would bypass even that, but the sandbox container
lacks `CAP_LINUX_IMMUTABLE`, so those calls fail silently (the NemoClaw
entrypoint handles this with `|| true`). DAC (`root:root chmod 444`) is the
only real barrier left, and the sync function handles it by temporarily
flipping to `chmod 644` around the write.

### Why `baseUrl` must be direct, not `https://inference.local/v1`

OpenClaw's `globalThis.fetch` runs on Node.js 22's undici, which does **not**
honor `HTTP_PROXY` / `HTTPS_PROXY` environment variables even when set. The
default onboard baseUrl `https://inference.local/v1` relies on OpenShell's
L7 proxy to translate the virtual hostname to the local vLLM endpoint;
since fetch never talks to the proxy, the request never reaches vLLM and
times out. Our `http://host.openshell.internal:8000/v1` override skips the
proxy entirely.

Upstream tracking: openclaw/openclaw#62181 (bug), openclaw/openclaw#43919
(fix PR, open for weeks, unmerged at the time this doc was written).
If those land and `inference.local` starts working, the `baseUrl` override
can be dropped.

## Reference: files touched by this workflow

| File | Role |
|---|---|
| [lib/launch.sh](lib/launch.sh) | `cosmos-reason2-8b` profile — vLLM args + env vars |
| [lib/config.sh](lib/config.sh) | `cosmos-reason2-8b` runtime config — `max_model_len=65536`, `max_num_seqs=3` |
| [start-model.sh](start-model.sh) | Model launcher (honors `THOR_DETACH`, `THOR_CONTAINER_NAME`, `THOR_NO_RM`) |
| [start-duo.sh](start-duo.sh) | Dual-serve launcher (Qwen3.6 + Cosmos) — kept for benchmark scenarios |
| [configure-local-provider.sh](configure-local-provider.sh) | Wires OpenShell `vllm-local` provider + inference route + sandbox sync + gateway start |
| `~/.cache/huggingface/token` | HF token for gated-repo access |
| `~/.nemoclaw/sandboxes.json` | Sandbox→provider/model binding (autoupdated by configure-local-provider.sh) |
| `/sandbox/openclaw-gateway.log` | In-sandbox gateway log — first place to look when dispatch fails |

## Quick-start (TL;DR)

```bash
export HF_TOKEN="$(tr -d '[:space:]' < ~/.cache/huggingface/token)"

# 1. Start the model
cd /home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor
THOR_DETACH=1 THOR_CONTAINER_NAME="nemoclaw-cosmos-reason2-8b" \
    ./start-model.sh cosmos-reason2-8b

# 2. Route OpenClaw to it
./configure-local-provider.sh cosmos-reason2-8b

# 3. Use it (interactive)
nemoclaw my-assistant connect

# ... or use it (scripted — see "Scripted / non-interactive" above)

# 4. Shut down when done
docker stop nemoclaw-cosmos-reason2-8b
sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```
