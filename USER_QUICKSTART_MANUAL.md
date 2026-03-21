# User Quickstart Manual

This manual is for daily use of `NemoClaw-Thor` on a Thor that has already been
set up with this repo.

Validated operator path:

- sandbox: `thor-assistant`
- provider: `vllm-local`
- policy baseline: `strict-local`
- validated model profiles:
  - `qwen3.5-27b-fp8`
  - `qwen3.5-35b-a3b-fp8`
  - `qwen3.5-122b-a10b-nvfp4-resharded`

Important rule:

- Starting a model with `./start-model.sh <profile>` does not by itself switch
  NemoClaw to that model.
- After any model change, run `./configure-local-provider.sh <profile>`.

## 1. Shell Prep

From a fresh host shell:

```bash
cd /home/tndlux/workspaces/nemoclaw/src/NemoClaw-Thor
source "$HOME/.nvm/nvm.sh"
nvm use 22
export PATH="$HOME/.local/bin:$PATH"
```

Notes:

- `nemoclaw` needs Node 22.
- `openshell` lives in `~/.local/bin`.

## 2. Common Use Cases

### Reconnect to a running stack

If the model server, gateway, and sandbox are already up:

```bash
./status.sh <profile>
nemoclaw thor-assistant connect
```

If you want a direct shell instead:

```bash
openshell sandbox connect thor-assistant
```

If you are not sure what model the host is serving:

```bash
curl -sf http://127.0.0.1:8000/v1/models
```

### Start the stack after reboot or after services were stopped

In terminal 1, start the model server:

```bash
./start-model.sh <profile>
```

In terminal 2, start the OpenShell gateway:

```bash
openshell gateway start --name nemoclaw
```

Then bind NemoClaw to the running model, verify, and connect:

```bash
./configure-local-provider.sh <profile>
./status.sh <profile>
nemoclaw thor-assistant connect
```

If `./status.sh` says the sandbox is missing:

```bash
./install.sh <profile> --policy-profile strict-local
```

### Switch NemoClaw to a different already-running model

This is the common case when vLLM is serving one model, but the sandbox runtime
was last configured for another.

Example: `35B` is running, but NemoClaw was last bound to `122B`.

```bash
./configure-local-provider.sh qwen3.5-35b-a3b-fp8
./status.sh qwen3.5-35b-a3b-fp8
nemoclaw thor-assistant connect
```

If the previous model left stale embedded session history behind, reset it first:

```bash
./reset-sandbox-session-state.sh qwen3.5-35b-a3b-fp8
```

### Run quick smoke tests

From inside the sandbox, plain inference:

```bash
openclaw agent --agent main --local --thinking off \
  -m "Reply with one word: working" --session-id test
```

From inside the sandbox, tool-use smoke:

```bash
openclaw agent --agent main --local --thinking off \
  -m "Run uname -a and python3 --version, write both to /sandbox/smoke.txt, then reply done." \
  --session-id smoke-tools
cat /sandbox/smoke.txt
```

If the first plain run says `No reply from agent`, retry once. After model
switches, `./reset-sandbox-session-state.sh <profile>` can also help.

## 3. How To Use It

What you are connecting to:

- `nemoclaw thor-assistant connect` opens a shell inside the sandbox.
- The main writable workspace is `/sandbox`.
- Local inference is routed through OpenShell to the host vLLM server.
- `strict-local` blocks broad outbound internet by default.

Typical workflow:

1. Connect to the sandbox.
2. Change into `/sandbox`.
3. Copy or clone a disposable repo there.
4. Ask the agent to inspect files, edit code, and run commands.
5. Review results, logs, tests, and diffs from the sandbox shell.

Useful host-side commands:

```bash
nemoclaw thor-assistant status
nemoclaw thor-assistant logs --follow
openshell sandbox list
openshell inference get
```

For real work, operate on a repo copy inside `/sandbox`, not on the host working
tree directly.

## 4. Optional Launch Knobs

These are one-run overrides for `./start-model.sh`:

- `THOR_MAX_MODEL_LEN`: context window
- `THOR_KV_CACHE_DTYPE`: usually `fp8`
- `THOR_MAX_NUM_SEQS`: concurrent sequences
- `THOR_GPU_MEMORY_UTILIZATION`: vLLM GPU memory target
- `THOR_MAX_NUM_BATCHED_TOKENS`: prefill batching limit

Example for `35B`:

```bash
THOR_MAX_MODEL_LEN=65536 \
THOR_KV_CACHE_DTYPE=fp8 \
THOR_MAX_NUM_SEQS=20 \
THOR_GPU_MEMORY_UTILIZATION=0.80 \
THOR_MAX_NUM_BATCHED_TOKENS=8192 \
./start-model.sh qwen3.5-35b-a3b-fp8
```

Example for `122B`:

```bash
THOR_MAX_MODEL_LEN=65536 \
THOR_KV_CACHE_DTYPE=fp8 \
THOR_MAX_NUM_SEQS=6 \
THOR_GPU_MEMORY_UTILIZATION=0.80 \
THOR_MAX_NUM_BATCHED_TOKENS=8192 \
./start-model.sh qwen3.5-122b-a10b-nvfp4-resharded
```

Persistent defaults live in:

```text
~/.config/nemoclaw-thor/config.env
```

The persistent keys are:

- `THOR_TARGET_MAX_MODEL_LEN`
- `THOR_TARGET_KV_CACHE_DTYPE`
- `THOR_TARGET_MAX_NUM_SEQS`

## 5. Stop Or Shutdown

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

### Delete the sandbox

List first:

```bash
openshell sandbox list
nemoclaw list
```

Delete by name:

```bash
openshell sandbox delete thor-assistant
```

Alternative:

```bash
nemoclaw thor-assistant destroy
```

### Full cleanup

Remove the install artifacts tracked by this repo:

```bash
./uninstall.sh
```

If you also want to revert the saved Thor host changes:

```bash
./restore-host-state.sh
```

## 6. Important Notes

- `nemoclaw stop` is usually not the command you want here. It does not manage
  the local vLLM server in this Thor setup.
- The main validated path is one tracked sandbox, `thor-assistant`.
- After switching model profiles, always run
  `./configure-local-provider.sh <profile>`.
- A model can be running correctly while the sandbox is still bound to a
  different profile. `./status.sh <profile>` is the quickest sanity check.
