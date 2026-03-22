# User Quickstart Manual

This manual is the operator path for `NemoClaw-Thor` on a Thor that already has
OpenShell installed and the Thor host fixes applied.

Validated baseline:

- gateway: `nemoclaw`
- sandbox: `thor-assistant`
- provider: `vllm-local`
- policy: `strict-local`

Important rule:

- `./start-model.sh <profile>` only starts vLLM.
- `./configure-local-provider.sh <profile>` binds NemoClaw/OpenShell to the
  running model, ensures the gateway is up, and repairs native
  `nemoclaw thor-assistant connect` if the OpenShell SSH handshake state drifted.

## 1. Shell Prep

From a fresh host shell:

```bash
cd /home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor
source "$HOME/.nvm/nvm.sh"
nvm use 22
export PATH="$HOME/.local/bin:$PATH"
```

## 2. Clean Install From Scratch

Use this on a fresh Thor install or after a full uninstall.

1. Apply the Thor host fixes once:

```bash
./apply-host-fixes.sh
```

2. Start the model server and leave it running in that terminal:

```bash
./start-model.sh qwen3.5-35b-a3b-fp8
```

3. In a second terminal, run the installer:

```bash
./install.sh qwen3.5-35b-a3b-fp8 --policy-profile strict-local
```

4. Verify and connect:

```bash
./status.sh qwen3.5-35b-a3b-fp8
nemoclaw thor-assistant connect
```

Notes:

- `install.sh` now ensures the `nemoclaw` OpenShell gateway is running before it
  continues.
- If `~/NemoClaw` already exists, this is no longer a clean install. Either use
  `./uninstall.sh` first or switch to the reconfigure/start paths below.

## 3. Start An Existing Install

Use this when the repo is already installed and you only need to bring the stack
 back up.

1. Start the model server and leave it running:

```bash
./start-model.sh qwen3.5-35b-a3b-fp8
```

2. Rebind the local provider and repair the sandbox runtime state if needed:

```bash
./configure-local-provider.sh qwen3.5-35b-a3b-fp8
```

3. Verify and connect:

```bash
./status.sh qwen3.5-35b-a3b-fp8
nemoclaw thor-assistant connect
```

This is also the clean reboot path. After a reboot, do the same 3 steps above.

If `./status.sh` still says the sandbox is missing, recreate it with:

```bash
./install.sh qwen3.5-35b-a3b-fp8 --policy-profile strict-local
```

## 4. Reconfigure An Existing Install

### Switch NemoClaw to a different running model

Start the new model first, then rebind the provider:

```bash
./configure-local-provider.sh <profile>
./status.sh <profile>
nemoclaw thor-assistant connect
```

Examples:

```bash
./configure-local-provider.sh qwen3.5-27b-fp8
./configure-local-provider.sh qwen3.5-35b-a3b-fp8
./configure-local-provider.sh qwen3.5-122b-a10b-nvfp4-resharded
```

### Reset stale local session history

If a prior local run left the embedded session store in a bad state:

```bash
./reset-sandbox-session-state.sh <profile>
```

### Launcher knobs

The most relevant per-run overrides for `./start-model.sh` are:

- `THOR_MAX_MODEL_LEN`
- `THOR_KV_CACHE_DTYPE`
- `THOR_MAX_NUM_SEQS`
- `THOR_OPENCLAW_MAIN_MAX_CONCURRENT`
- `THOR_GPU_MEMORY_UTILIZATION`
- `THOR_MAX_NUM_BATCHED_TOKENS`

Example for `35B-A3B-FP8`:

```bash
THOR_MAX_MODEL_LEN=65536 \
THOR_KV_CACHE_DTYPE=fp8 \
THOR_MAX_NUM_SEQS=20 \
THOR_OPENCLAW_MAIN_MAX_CONCURRENT=1 \
THOR_GPU_MEMORY_UTILIZATION=0.80 \
THOR_MAX_NUM_BATCHED_TOKENS=8192 \
./start-model.sh qwen3.5-35b-a3b-fp8
```

Example for `122B-A10B-NVFP4-resharded`:

```bash
THOR_MAX_MODEL_LEN=65536 \
THOR_KV_CACHE_DTYPE=fp8 \
THOR_MAX_NUM_SEQS=6 \
THOR_OPENCLAW_MAIN_MAX_CONCURRENT=1 \
THOR_GPU_MEMORY_UTILIZATION=0.80 \
THOR_MAX_NUM_BATCHED_TOKENS=8192 \
./start-model.sh qwen3.5-122b-a10b-nvfp4-resharded
```

Persistent defaults are saved in:

```text
~/.config/nemoclaw-thor/config.env
```

## 5. Use The Installed Sandbox

Connect:

```bash
nemoclaw thor-assistant connect
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

Quick smoke test inside the sandbox:

```bash
openclaw agent --agent main --local --thinking off \
  -m "Reply with one word: working" --session-id test
```

Tool-use smoke test inside the sandbox:

```bash
openclaw agent --agent main --local --thinking off \
  -m "Run uname -a and python3 --version, write both to /sandbox/smoke.txt, then reply done." \
  --session-id smoke-tools
cat /sandbox/smoke.txt
```

### Work on a repo copy inside the sandbox

Upload a repo into `/sandbox`:

```bash
openshell sandbox upload thor-assistant /path/to/repo /sandbox/myrepo
```

Then connect and work in the copy:

```bash
nemoclaw thor-assistant connect
cd /sandbox/myrepo
openclaw tui
```

Important:

- `openshell sandbox upload` copies files into the sandbox.
- Agent edits stay in `/sandbox/myrepo`.
- Your host repo is not changed automatically.

Copy the repo back out:

```bash
openshell sandbox download thor-assistant /sandbox/myrepo /path/to/output-copy
```

Or copy out only a patch:

```bash
# inside the sandbox
cd /sandbox/myrepo
git diff > /sandbox/myrepo.patch
```

Then on the host:

```bash
openshell sandbox download thor-assistant /sandbox/myrepo.patch .
```

### Dashboard

The dashboard is optional.

- Use `openclaw tui` for real prompting.
- The dashboard is still useful for visibility, but browser-originated prompts
  are not fully reliable on this Thor setup.

If you still want the dashboard:

1. Inside the sandbox, make sure the OpenClaw gateway is running (see above).

2. On the host, start the access helper:

```bash
./start-dashboard-access.sh
```

3. Open the tokenized URL printed by the helper in Firefox.

To print the token URL again inside the sandbox:

```bash
HOME=/sandbox openclaw dashboard --no-open
```

## 6. Stop Without Uninstalling

### Leave only the current shell

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

After stopping vLLM, reclaim memory before starting another model:

```bash
sudo sync
sudo sysctl -w vm.drop_caches=3
```

### Stop the OpenShell gateway

```bash
openshell gateway stop
```

### Stop the dashboard path

If you started the browser dashboard helper:

```bash
./stop-dashboard-access.sh
```

Then stop the sandbox-side `openclaw gateway run` process with `Ctrl-C`.

## 7. Uninstall Cleanly

1. Stop the model server and drop caches:

```bash
docker ps --format '{{.ID}}\t{{.Command}}' | grep 'vllm serve'
docker stop <container-id>
sudo sync
sudo sysctl -w vm.drop_caches=3
```

2. Stop the OpenShell gateway:

```bash
openshell gateway stop
```

3. Remove the NemoClaw-Thor tracked install:

```bash
./uninstall.sh
```

4. If you also want to revert the saved Thor host changes:

```bash
./restore-host-state.sh
```

Notes:

- `./uninstall.sh` removes the tracked sandbox, tracked providers, the
  `nemoclaw` CLI, `~/NemoClaw`, and the Thor runtime config.
- It does not remove OpenShell itself, Docker, Node.js, or local model weights.
- It asks whether to remove `~/.nemoclaw`.

## 8. Practical Rules

- Use `nemoclaw thor-assistant connect` as the normal shell entrypoint.
- Use `openclaw tui` as the normal prompt UI.
- If the TUI shows "gateway disconnected": `HOME=/sandbox openclaw gateway run &`
- After any model change, run `./configure-local-provider.sh <profile>`.
- After a reboot, use the same path as any normal restart:
  `./start-model.sh`, then `./configure-local-provider.sh`, then `./status.sh`.
- If you stop vLLM, always run `sudo sysctl -w vm.drop_caches=3` before loading
  another model.
