# NemoClaw-Thor on Jetson Orin AGX (64 GB)

This repo is named for Jetson **Thor**, but it also runs on **Jetson Orin AGX
Dev Kit (64 GB)**. This document is the Orin-specific delta for **serving a
model** and **installing + onboarding the NemoClaw / OpenShell / OpenClaw
toolchain**. For the Thor flow see [`USER_QUICKSTART_MANUAL.md`](USER_QUICKSTART_MANUAL.md)
and [`AGENTS.md`](AGENTS.md) §A.

> **Whole-stack master index** (host → serving → toolchain → onboard →
> composer → run → smoke): see
> [`manyforge/docs/operations/ORIN_STACK_SETUP.md`](../manyforge/docs/operations/ORIN_STACK_SETUP.md)
> in the **manyforge** repo. Start there if you're bringing up the full stack.

**Platform:** Orin AGX = aarch64, CC **8.7** (Ampere), **64 GB** unified memory.
Deltas vs Thor (SM110/Blackwell, 128 GB): a different llama.cpp image, a smaller
memory budget, **slower prefill** (≈240 tok/s on 24k-token prompts → multi-turn
agent cases run near the 170 s wall), and a **fully NVMe-staged install** (the
~30 GB eMMC must stay free; everything below lives on `/mnt/nova_ssd`).

## Repository availability

Prefer the sibling checkout layout under `${HOME}/workspaces/dev_ws/src/`.
If a referenced repo is missing locally, inspect or clone the matching GitHub
repo under `https://github.com/pastoriomarco/` before treating the path as
stale.

| Repo | Preferred local path | Accepted alternate local path | Remote |
|---|---|---|---|
| `NemoClaw-Thor` | `${HOME}/workspaces/dev_ws/src/NemoClaw-Thor` | `${HOME}/workspaces/nemoclaw/src/NemoClaw-Thor` | `https://github.com/pastoriomarco/NemoClaw-Thor` |
| `manyforge` | `${HOME}/workspaces/dev_ws/src/manyforge` | - | `https://github.com/pastoriomarco/manyforge` |
| `isaac_ros_custom_bringup` | `${HOME}/workspaces/dev_ws/src/isaac_ros_custom_bringup` | `${HOME}/workspaces/isaac_ros-dev/src/isaac_ros_custom_bringup` | `https://github.com/pastoriomarco/isaac_ros_custom_bringup` |
| `manyforge_specs` | `${HOME}/workspaces/dev_ws/src/manyforge_specs` | - | `https://github.com/pastoriomarco/manyforge_specs` |

---

## 0. Host storage and runtime prerequisites

The canonical, command-heavy NVMe host-storage recipe lives in
[`isaac_ros_custom_bringup/jetson_orin_storage/README.md`](../isaac_ros_custom_bringup/jetson_orin_storage/README.md)
(or the GitHub fallback above if the sibling checkout is absent). This section
is the compact contract that NemoClaw/OpenShell/OpenClaw depends on.

Required host contract:

- `/mnt/nova_ssd` exists, is persistent in `/etc/fstab`, and has enough free
  space for Docker images, GGUF/model caches, OpenClaw/NemoClaw state, and temp
  files.
- Docker uses the NVMe: `"data-root": "/mnt/nova_ssd/docker"` in
  `/etc/docker/daemon.json`, NVIDIA is the default runtime, and the user can run
  Docker **without sudo**.
- Large user/toolchain paths are NVMe-backed using bind mounts, not symlinks:
  at minimum `~/.cache`, `~/.local`, `/usr/local`, `/tmp`, and `/var/tmp`.
- Model/cache directories are on the NVMe and owned by the user:
  `/mnt/nova_ssd/hf-cache-orin`, `/mnt/nova_ssd/llama-cpp-cache`, and
  `/mnt/nova_ssd/opt`.
- `~/.nemoclaw` is a real directory or bind mount backed by NVMe, never a
  symlink. NemoClaw v0.0.59 rejects a symlinked config directory as a possible
  symlink attack.
- Passwordless `sudo` is available for the setup flow (`apt`, one-time bind
  mounts, and `drop_caches`).

Quick verification:

```bash
findmnt /mnt/nova_ssd
findmnt -T "$HOME/.cache"
findmnt -T "$HOME/.local"
findmnt -T "$HOME/.nemoclaw" || true
test ! -L "$HOME/.nemoclaw"

docker info | grep -E 'Docker Root Dir|Default Runtime|Runtimes|Storage Driver'
df -h / /mnt/nova_ssd
ls -ld /mnt/nova_ssd \
       /mnt/nova_ssd/hf-cache-orin \
       /mnt/nova_ssd/llama-cpp-cache \
       /mnt/nova_ssd/opt \
       /mnt/nova_ssd/nemoclaw-state/nemoclaw
```

Expected checks:

- `findmnt /mnt/nova_ssd` shows the NVMe mount.
- `docker info` shows `Docker Root Dir: /mnt/nova_ssd/docker` and
  `Default Runtime: nvidia`.
- `test ! -L "$HOME/.nemoclaw"` exits `0`.
- The NVMe cache/state directories are writable by the current user.

---

## 1. Serve the model — profile `gemma4-12b-it-gguf-orin`

Defined in [`serving/config.sh`](serving/config.sh) +
[`serving/launch.sh`](serving/launch.sh). It is the Orin twin of the Thor
`gemma4-12b-it-gguf` profile — **same** model (unsloth `gemma-4-12b-it-GGUF:UD-Q4_K_XL`)
+ E2B speculative draft, 128k context, q8_0 KV (model **and** draft),
`--flash-attn on`. **Only the image and cache mounts differ:**

| | value |
|---|---|
| image | `ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin` |
| `/mnt/nova_ssd/hf-cache-orin` | → `/data/models/huggingface` **and** `/root/.cache/huggingface` |
| `/mnt/nova_ssd/llama-cpp-cache` | → `/root/.cache/llama.cpp` |

**The mounts are load-bearing** — they keep the ~11 GB of GGUFs on the NVMe and
out of the eMMC. Override the host dirs with `THOR_HF_CACHE_DIR` /
`THOR_LLAMACPP_CACHE_DIR_HOST` if your NVMe is elsewhere.

### Stage the weights once
Pre-download into `/mnt/nova_ssd/hf-cache-orin` (the operator's tested standalone
`docker run … llama-server -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL
--spec-draft-hf unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL …` does this on first run).

### Cache-first serving (do **not** re-download)
llama.cpp `-hf` re-resolves the repo's `main` revision every launch and
**re-downloads when upstream moves** even if a complete copy is cached. The
profile defaults `THOR_LLAMACPP_OFFLINE=auto` → it serves `--offline` (cached
weights, no network) when the repo is staged, and **warns** when a newer
upstream revision exists. To deliberately pull an update:
`THOR_LLAMACPP_OFFLINE=0 ./serving/start-model.sh gemma4-12b-it-gguf-orin`.

### Start (standalone)
```bash
THOR_DETACH=1 THOR_RESTART_PROXY=0 THOR_CONTAINER_NAME=manyforge-e2e-vllm \
THOR_VLLM_PORT=8000 ./serving/start-model.sh gemma4-12b-it-gguf-orin
```
Verify: `curl :8000/v1/models` → `gemma4-12b-it-gguf-orin`; a tool-calling
completion returns clean OpenAI `tool_calls`. Cached load ≈ 15 s.
(Through the manyforge launcher the port/proxy/container name are managed for
you — the launcher puts the model on `:8050` behind the vllm-proxy on `:8000`.)

---

## 2. Install the toolchain (node / nemoclaw / openshell / openclaw) — all on the NVMe

A fresh Orin has none of these. `nemoclaw onboard` does **not** install them.

### 2a. Node.js → `/mnt/nova_ssd/opt/node`
```bash
mkdir -p /mnt/nova_ssd/opt && cd /mnt/nova_ssd/opt
curl -fsSL -o node.tar.xz https://nodejs.org/dist/v22.22.3/node-v22.22.3-linux-arm64.tar.xz
tar -xJf node.tar.xz && mv node-v22.22.3-linux-arm64 node && rm node.tar.xz
mkdir -p ~/.local/bin                      # ~/.local is bind-mounted to the NVMe + on PATH
ln -sf /mnt/nova_ssd/opt/node/bin/node ~/.local/bin/node
ln -sf /mnt/nova_ssd/opt/node/bin/npm  ~/.local/bin/npm
ln -sf /mnt/nova_ssd/opt/node/bin/npx  ~/.local/bin/npx
npm config set prefix "$HOME/.local"       # global npm installs land on the NVMe + on PATH
```

### 2b. NemoClaw CLI → `/mnt/nova_ssd/NemoClaw`
NVIDIA/NemoClaw publishes **no GitHub Releases** — install from the latest
**tag**. As of this writing the newest is **v0.0.59** (`lkg` = v0.0.55 is the
doc-verified default; either works). Its pins (from
`nemoclaw-blueprint/blueprint.yaml`): **OpenShell 0.0.44**, **OpenClaw
2026.5.22** (sandbox image digest `b3d832b596…`).
```bash
git clone https://github.com/NVIDIA/NemoClaw /mnt/nova_ssd/NemoClaw
git -C /mnt/nova_ssd/NemoClaw checkout v0.0.59
cd /mnt/nova_ssd/NemoClaw && npm install && npm link     # → ~/.local/bin/nemoclaw
nemoclaw --version   # v0.0.59
```

### 2c. OpenShell (NemoClaw's installer) → `~/.local/bin`
```bash
NEMOCLAW_NON_INTERACTIVE=1 bash /mnt/nova_ssd/NemoClaw/scripts/install-openshell.sh
# /usr/local/bin is not writable → non-interactively it falls back to ~/.local/bin (NVMe).
# Installs openshell + openshell-gateway + openshell-sandbox 0.0.44.
openshell --version  # 0.0.44
```

### 2d. Bridge venvs (host lacks `ensurepip`)
```bash
sudo apt-get install -y python3.12-venv
python3 -m venv  <manyforge>/manyforge_assistant_bridge/.venv
<…>/.venv/bin/pip install -r <manyforge>/manyforge_assistant_bridge/requirements.txt          # direct/nemoclaw bridge
python3 -m venv  manyforge/openclaw_assistant_bridge/.venv
<…>/.venv/bin/pip install -r manyforge/openclaw_assistant_bridge/requirements.txt             # openclaw bridge
```

### 2e. `~/.nemoclaw` on the NVMe — **bind mount, not symlink**
NemoClaw v0.0.59 **refuses a symlinked config dir** (`~/.nemoclaw is a symbolic
link … may indicate a symlink attack`). Use a bind mount (the
`isaac_ros_custom_bringup` pattern):
```bash
mkdir -p ~/.nemoclaw /mnt/nova_ssd/nemoclaw-state/nemoclaw
sudo mount --bind /mnt/nova_ssd/nemoclaw-state/nemoclaw ~/.nemoclaw
echo "/mnt/nova_ssd/nemoclaw-state/nemoclaw $HOME/.nemoclaw none bind,nofail,x-gvfs-hide,x-systemd.requires-mounts-for=/mnt/nova_ssd 0 0" | sudo tee -a /etc/fstab
```
(The openshell docker-driver gateway state under `~/.local/state/nemoclaw` is
already on the NVMe via the `~/.local` bind mount.)

---

## 3. Onboard the OpenClaw sandbox

**Start the model first** (onboard probes the inference endpoint). Then:
```bash
NEMOCLAW_NON_INTERACTIVE=1 \
NEMOCLAW_PROVIDER=custom \                       # v0.0.59 renamed "compatible-endpoint" → "custom"
NEMOCLAW_ENDPOINT_URL=http://127.0.0.1:8000/v1 \
NEMOCLAW_MODEL=gemma4-12b-it-gguf-orin \
NEMOCLAW_PROVIDER_KEY=dummy-local-key \          # any non-empty value; NOT checked for local serving
nemoclaw onboard --non-interactive --yes --fresh --name my-assistant --yes-i-accept-third-party-software
```
First run builds the sandbox image (~12 min, on the NVMe via docker data-root)
and registers the `my-assistant` sandbox. `nemoclaw list` should show it.

### Required post-onboard
```bash
./setup/configure-local-provider.sh gemma4-12b-it-gguf-orin   # in-sandbox baseUrl + local-inference egress + model warmup
```

### Start the in-sandbox OpenClaw gateway (loopback + auth)
OpenClaw 2026.5.x **refuses to bind `0.0.0.0` without auth**. Start it on
loopback with a token (the manyforge launcher's `start_bridge_openclaw` does
this for you):
```bash
cid=$(docker ps -q --filter name=openshell-my-assistant | head -1)
docker exec -d "$cid" bash -c 'cd /sandbox; HOME=/sandbox exec openclaw gateway --allow-unconfigured --bind loopback --auth token >/sandbox/openclaw-gateway.log 2>&1'
```

### Wire the manyforge integration
```bash
manyforge/setup-manyforge-assistant.sh my-assistant   # skill + MCP server + manyforge egress preset + agent profile
```

> **Egress is subject-scoped — don't be fooled by the reachability probe.** The
> `manyforge-composer` policy whitelists egress to `host.openshell.internal:{9000,8000}`
> **by calling binary** (openclaw / node / `/usr/bin/python3.13`), not by host
> alone. So a probe (or an ad-hoc `docker exec … curl`/`python3`) that uses a
> non-whitelisted subject gets **403 — a false negative**. The real agent path
> (node/openclaw) is allowed and works: verified that every openclaw chat
> completion goes through the `:8000` `vllm-proxy.py` (mutations applied:
> `max_completion_tokens 4096→2048`, `thinking_token_budget →512`) then to
> llama.cpp on `:8050` — the **same model path as the direct lane**. Do **not**
> add `host.openshell.internal` to `NO_PROXY` (that would bypass the restricted
> egress). If `setup-manyforge-assistant.sh` reports the probe failing, confirm
> with the whitelisted subject before treating it as real:
> `nemoclaw my-assistant exec --no-tty -- /usr/bin/python3.13 -c "import urllib.request as u; print(u.urlopen('http://host.openshell.internal:9000/api/assistant/modes/composer-assistant',timeout=8).status)"`
> Tip: tail the pipeline live with
> [`manyforge/scripts/debug/pipeline_message_monitor.py`](manyforge/scripts/debug/pipeline_message_monitor.py).

---

## Orin gotchas (all hit + fixed this session)
| symptom | fix |
|---|---|
| composer image build fails: `ImportError: liburdfdom_sensor.so.4.0` | pin `cmeel-urdfdom==4.0.1` (done in manyforge `Dockerfile`) |
| model re-downloads on every launch | cache-first `--offline` (profile default `THOR_LLAMACPP_OFFLINE=auto`) |
| `~/.nemoclaw is a symbolic link … symlink attack` | bind-mount `~/.nemoclaw` (§2e), don't symlink |
| onboard: `Unsupported NEMOCLAW_PROVIDER: compatible-endpoint` | use `NEMOCLAW_PROVIDER=custom` (v0.0.59) |
| `Refusing to bind gateway to auto without auth` | start gateway `--bind loopback --auth token` |
| `ensurepip is not available` when making venvs | `sudo apt-get install -y python3.12-venv` |

## Pinned versions on this Orin
NemoClaw **v0.0.59** · OpenShell **0.0.44** · OpenClaw **2026.5.22** (sandbox
digest `b3d832b596…`) · Node **22.22.3** · llama.cpp image `…:latest-jetson-orin`.
