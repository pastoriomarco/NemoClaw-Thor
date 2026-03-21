# User Quickstart Manual

This manual is for using the current `NemoClaw-Thor` install on a Thor that has
already been set up with this repo.

Current validated stack:

- sandbox: `thor-assistant`
- provider: `vllm-local`
- model: `Qwen3.5-27B-FP8`
- policy baseline: `strict-local`

## 1. Shell Prep

From a fresh host shell, prepare the environment first:

```bash
cd /home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor
source "$HOME/.nvm/nvm.sh"
nvm use 22
export PATH="$HOME/.local/bin:$PATH"
```

`nemoclaw` needs the Node 22 runtime. `openshell` lives in `~/.local/bin`.

## 2. Start Or Reconnect

### If the stack is already running

Check health:

```bash
./status.sh qwen3.5-27b-fp8
nemoclaw thor-assistant status
```

Connect:

```bash
nemoclaw thor-assistant connect
```

If you want a direct OpenShell shell instead:

```bash
openshell sandbox connect thor-assistant
```

### If the host rebooted or services were stopped

Start the model server in one terminal:

```bash
./start-model.sh qwen3.5-27b-fp8
```

Start the OpenShell gateway in another:

```bash
openshell gateway start --name nemoclaw
```

Then verify and connect:

```bash
./status.sh qwen3.5-27b-fp8
nemoclaw thor-assistant connect
```

If `./status.sh` says the sandbox is missing, recreate it:

```bash
./install.sh qwen3.5-27b-fp8 --policy-profile strict-local
```

If the install is present and only the provider/model binding needs refresh:

```bash
./configure-local-provider.sh qwen3.5-27b-fp8
```

## 3. Quick Overview

What you are connecting to:

- `nemoclaw thor-assistant connect` opens a shell inside the sandbox.
- The main writable workspace is `/sandbox`.
- `strict-local` blocks broad outbound internet by default.
- Local inference is routed through OpenShell to the host vLLM server.

Typical workflow:

1. Connect to the sandbox.
2. Work inside `/sandbox`.
3. Ask the agent to inspect files, edit code, and run commands.
4. Check results, logs, and diffs from the sandbox shell.

Useful commands:

```bash
nemoclaw thor-assistant status
nemoclaw thor-assistant logs --follow
openshell sandbox list
openshell inference get
```

Minimal inference test from inside the sandbox:

```bash
openclaw agent --agent main --local --thinking off \
  -m "Reply with one word: working" --session-id test
```

Tool-use status on the current validated stack:

```text
Plain text inference works.
Tool execution works end-to-end on the validated 27B stack.
```

```bash
openclaw agent --agent main --local --thinking off \
  -m "Run uname -a and python3 --version, write both to /sandbox/smoke.txt, then reply done." \
  --session-id smoke-tools
```

If a previous broken model/tool round leaves the embedded agent replaying stale
history, clear the sandbox-local session store and retry:

```bash
./reset-sandbox-session-state.sh qwen3.5-27b-fp8
```

Direct shell commands are still useful for quick runtime checks:

```bash
uname -a
python3 --version
```

For real work, clone or copy a disposable repo into `/sandbox` and operate on
that copy rather than on the host working tree.

## 4. Closing A Session

To leave a connected sandbox shell:

```bash
exit
```

That closes only your current terminal session. It does not stop the sandbox,
gateway, or model server.

## 5. Stopping Things

There are four different shutdown levels.

### A. Stop only your current shell session

Use:

```bash
exit
```

### B. Stop the model server

If `./start-model.sh` is running in the current terminal, press `Ctrl-C`.

If the vLLM container is running elsewhere:

```bash
docker ps --format '{{.ID}}\t{{.Command}}' | grep 'vllm serve'
docker stop <container-id>
```

If you want to reclaim memory before starting another model:

```bash
sudo sync
sudo sysctl -w vm.drop_caches=3
```

### C. Stop the OpenShell gateway but keep state

```bash
openshell gateway stop
```

This pauses the gateway. It does not delete the sandbox metadata or the runtime
config tracked by this repo.

### D. Delete a sandbox

List sandboxes first:

```bash
openshell sandbox list
nemoclaw list
```

Delete one by name:

```bash
openshell sandbox delete thor-assistant
```

NemoClaw also supports:

```bash
nemoclaw thor-assistant destroy
```

For this Thor setup, prefer explicit OpenShell deletion when you want precise
control over which sandbox is being removed.

## 6. Full Cleanup

To remove the `NemoClaw-Thor` install artifacts tracked by this repo:

```bash
./uninstall.sh
```

That removes the tracked sandbox and provider, the local `nemoclaw` CLI, the
`~/NemoClaw` checkout, and the Thor runtime config. It does not remove
OpenShell itself, Docker, model weights, or the Thor host fixes.

If you also want to revert the saved host-level OpenShell-Thor changes:

```bash
./restore-host-state.sh
```

Use that only when you intentionally want to roll the Thor host back toward its
pre-install state.

## 7. Important Notes

- `nemoclaw stop` is usually not the command you want on this Thor setup. It
  targets old auxiliary services such as Telegram/cloudflared, not the local
  vLLM server or the OpenShell sandbox/gateway stack.
- The currently validated path is one tracked sandbox, `thor-assistant`.
  OpenShell can support multiple sandboxes, but that is not the main validated
  operator path for this repo.
- The validated coding/tool-use stack on this Thor is currently
  `Qwen3.5-27B-FP8`.
- The `Qwen3.5-35B-A3B-FP8` path may still be useful for plain inference, but
  it is not the validated tool-use profile.
- If you switch model profiles, re-run `./configure-local-provider.sh <profile>`
  after starting the new vLLM server so the inference route and sandbox runtime
  config stay aligned.
