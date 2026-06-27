# Smoke-corpus iter setup runbook

Steps to bring the Composer-assistant stack up cold and run a smoke iter.
Captures the order of operations + namespace gotchas that are easy to
miss because the old gateway typically lives long enough that they
become invisible.

## ⭐ Autonomous run — direct lane (recommended; no gateway/cluster needed)

The fastest, most reliable way to run the corpus, and the path an LLM/agent
should use. The **direct lane** is an in-process bridge → vLLM with **no
OpenClaw gateway, sandbox, or K3s cluster** — it does NOT need
`openshell-cluster-nemoclaw` or a provisioned sandbox. (The cold-start sections
below are only for testing the gateway/openclaw lane itself.)

### 0. Pre-flight
- **Runner + corpus live in THIS repo** (`NemoClaw-Thor/manyforge`), NOT the
  standalone `manyforge` checkout:
  - `src/NemoClaw-Thor/manyforge/scripts/debug/smoke_corpus_runner.py`
  - `src/NemoClaw-Thor/manyforge/scripts/debug/smoke_corpus.yaml`
- Free ports/GPU: `ss -tlnp | grep -E ':8000|:8050|:8100|:9000'` and
  `sudo docker ps --filter ancestor=vllm/vllm-openai:nightly-aarch64`. Stop any
  stale model container / `vllm-proxy.py` first.
- On the direct lane, **ignore** "OpenClaw gateway may not have started / sandbox
  not found" — direct bypasses the gateway entirely.

### 1. Bring up the whole stack — one command
Brings up the model (vLLM `:8050` + vllm-proxy `:8000`), Composer (`:9000`,
auto-loads the scenario's deployment + program), and the in-process bridge
(`:8100`):
```bash
bash src/manyforge/scripts/launch.sh --lane manyforge-only --assistant on \
  --provider direct --platform thor --model-profile <PROFILE> \
  --scenario ur10e-scene-authoring --non-interactive --yes
```
`<PROFILE>` = a thor vLLM recipe (`qwen3.6-35b-a3b-nvfp4-nvidia`,
`qwen3.6-27b-nvfp4`, …). Port map: **proxy `:8000` (what Composer/bridge call) →
vLLM `:8050`**.

### 2. Verify ready (gate before running)
```bash
curl -s 127.0.0.1:8000/v1/models | python3 -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])'
curl -s 127.0.0.1:9000/api/assistant/modes/composer-assistant | python3 -c 'import sys,json;print("tools:",len(json.load(sys.stdin)["tools"]))'  # expect 25
curl -s 127.0.0.1:8100/healthz   # {"status":"ok","model":"<served>",...}
```
A fresh NVFP4 27B/35B boot (weights + ViT + CUDA-graph capture) takes ~3–6 min —
wait for `:8000/v1/models` before running.

### 3. Run the corpus (from THIS repo's manyforge dir)
```bash
cd src/NemoClaw-Thor/manyforge
python3 scripts/debug/smoke_corpus_runner.py \
  --corpus scripts/debug/smoke_corpus.yaml --enable-recovery-turn \
  --report /tmp/smoke_<tag>.json > /tmp/smoke_<tag>.log 2>&1 &
```
Use the SAME `<tag>` in `.json` and `.log`. Verdicts stream line-buffered.

### 4. Monitor + score
```bash
tail -f /tmp/smoke_<tag>.log     # ✅pass  🛟recovered-pass  🟡soft-pass  ❌fail  [SKIP]future
```
**Effective rate = (pass + recovered-pass + soft-pass) / scored** (skips not
counted); pass gate ≥ 51% (iter-32 prod 77.3%; qwen3.6-35b-a3b 84.8%). The
per-case JSON is written to `--report` on completion.

### Public-image serving gotchas (when a recipe targets a stock image)
- Stock `vllm/vllm-openai:*` bake `ENTRYPOINT=["vllm","serve"]`; the recipe must
  set `THOR_VLLM_ENTRYPOINT=""` (blank) or the command doubles →
  `unrecognized arguments: serve <model>`.
- vLLM binds `THOR_VLLM_PORT` (default 8000); set `THOR_VLLM_PORT=8050` so the
  proxy (8000) fronts it, else `Address already in use`.
- `THOR_VLLM_IMAGE` is assigned near the top of `launch.sh` *before* the
  per-recipe `case`; a per-recipe `${THOR_VLLM_IMAGE:-…}` fallback is a no-op —
  assign it unconditionally (see the `qwen3.6-27b-nvfp4` recipe).

---

## Stack components

| Component | Where it runs | Port | Restart trigger |
|-----------|---------------|------|-----------------|
| vLLM | host | :8050 | rare (model swap, CUDA OOM) |
| vllm-proxy | host (Python) | :8000 | iter mutator change (max_tokens, thinking_budget, log_path) |
| Composer | docker container `manyforge-e2e-composer` | :9000 | any backend code change OR deployment YAML change |
| openclaw_assistant_bridge | host (uvicorn in venv) | :8200 | any bridge code change OR env var change |
| OpenClaw gateway | inside sandbox `my-assistant`, **SSH namespace** | :18789 | when MCP tool catalog changes (Composer YAML ⇒ new tools) |
| MCP wrapper subprocess | spawned by gateway | stdio | dies with gateway |

## Required-restart matrix per iter type

| Change | Composer | Bridge | Gateway+MCP |
|--------|----------|--------|-------------|
| Bridge env (compact_every_n, etc.) | — | ✓ | — |
| Bridge code (service.py, adapter.py) | — | ✓ | — |
| Composer Python (routes_assistant.py, schemas.py) | ✓ | — | ✓ ⚠ |
| Deployment YAML (new tool, mode allowlist) | ✓ + reload | — | ✓ ⚠ |
| node_catalog.yaml | ✓ + reload | — | — |
| smoke_corpus.yaml | — | — | — |
| smoke_corpus_runner.py | — | — | — |
| vllm-proxy mutators | — | — | — (proxy only) |
| vLLM/serving config | (vLLM) | ✓ | ✓ |

⚠ Gateway+MCP must be restarted after Composer rotates the tool
catalog. The MCP wrapper caches the manifest at startup; it only
refreshes on a 409 catalog-hash mismatch from a `tools/call` (not
`tools/list`). New tools won't appear in the model's tool list
until the wrapper restarts. Easiest path: bounce the gateway, which
respawns the wrapper.

## Cold-start sequence

Assume vLLM and vllm-proxy are already running (these have 10-hour
uptimes on a normal day). If not, bring them up first via the
`serving/launch.sh` profile.

### 1. Composer (Docker)

```bash
docker restart manyforge-e2e-composer
# wait ~5s, then verify
curl -s http://127.0.0.1:9000/api/deployment | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print(d.get('name') or 'EMPTY')"
```

After restart Composer has no deployment loaded. Re-load it:

```bash
curl -fsS -X POST http://127.0.0.1:9000/api/deployment/load \
  -H "Content-Type: application/json" \
  -d '{"path":"/workspace/examples/assistant_modes_scene_authoring.deployment.yaml"}'

curl -fsS -X POST http://127.0.0.1:9000/api/program/load \
  -H "Content-Type: application/json" \
  -d '{"path":"/workspace/examples/pick_and_place_ur10e_robotiq.program.yaml",
       "deploymentPath":"/workspace/examples/assistant_modes_scene_authoring.deployment.yaml",
       "forceDiscardOverrides":true}'
```

Verify the tool list is what you expect:

```bash
curl -s http://127.0.0.1:9000/api/assistant/modes/composer-assistant \
  | python3 -c "import json,sys; d=json.load(sys.stdin); ids=[t['id'] for t in d['tools']]; \
     print('count:', len(ids)); print('change_node_kind:', 'tree_draft_change_node_kind' in ids)"
```

Iter 32 production has **25 tools**; verify that count after deployment load.

### 2. Bridge

Stop the current bridge:

```bash
pkill -f openclaw_assistant_bridge.service
```

Start with the iter-32 production env:

```bash
cd /home/tndlux/workspaces/dev_ws/src/NemoClaw-Thor

PYTHONPATH=$(pwd)/manyforge:${PYTHONPATH:-} \
OPENCLAW_ASSISTANT_USE_GATEWAY=false \
OPENCLAW_ASSISTANT_BRIDGE_HOST=127.0.0.1 \
OPENCLAW_ASSISTANT_BRIDGE_PORT=8200 \
OPENCLAW_ASSISTANT_AGENT=manyforge-composer \
OPENCLAW_ASSISTANT_LOCAL=false \
OPENCLAW_ASSISTANT_TIMEOUT_S=300 \
OPENCLAW_ASSISTANT_COMPACT_EVERY_N=2 \
OPENCLAW_ASSISTANT_COMPACT_TIMEOUT_S=120 \
nohup ./manyforge/openclaw_assistant_bridge/.venv/bin/python \
  -m openclaw_assistant_bridge.service \
  > /tmp/iterN_bridge.log 2>&1 &

sleep 3 && curl -s http://127.0.0.1:8200/healthz
```

### 3. OpenClaw gateway (SSH namespace ⚠)

The gateway **must run in the openshell SSH session's network
namespace**, not the pod root namespace. Running it via plain `kubectl
exec` puts it in the wrong namespace and the host SSH forward to
:18789 will hit nothing — symptom: bridge requests come back with
`"OpenClaw agent exited with code 52: curl: (52) Empty reply from
server"`.

Re-provision (re-registers MCP server in openclaw.json so new tools
are present in the wrapper after gateway restart):

```bash
cd /home/tndlux/workspaces/dev_ws/src/NemoClaw-Thor
./manyforge/setup-manyforge-assistant.sh my-assistant
```

Kill any gateway in the wrong namespace, then start in the SSH
namespace via the helper:

```bash
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell my-assistant -c agent -- \
  sh -c 'pkill -f openclaw 2>/dev/null'

bash -c 'source setup/sandbox-runtime.sh; ensure_sandbox_gateway_running my-assistant' \
  2>&1 | grep -v "command not found"
# (`info` and `pass` shell helpers aren't sourced in plain bash — the warnings
#  are harmless; the gateway still starts.)

sleep 8

# Verify gateway is reachable through the host port-forward.
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell my-assistant -c agent -- \
  pgrep -af openclaw

# Smoke probe end-to-end (~20-40s on first call due to vLLM warmup):
curl -s -X POST http://127.0.0.1:9000/api/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"add a parallel","mode":"provider",
       "conversationId":"probe","requestId":"probe",
       "assistantMode":"composer-assistant","timeoutSeconds":300}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('msg:', (d.get('message') or '')[:200])"
```

If the probe returns an `OpenClaw agent exited with code 52` error,
the gateway is in the wrong namespace — kill `openclaw*` in the
sandbox and re-run the `ensure_sandbox_gateway_running` step above.

### 4. Run the smoke iter

```bash
cd /home/tndlux/workspaces/dev_ws/src/NemoClaw-Thor/manyforge

nohup python3 scripts/debug/smoke_corpus_runner.py \
  --corpus scripts/debug/smoke_corpus.yaml \
  --enable-recovery-turn \
  --report /tmp/smoke_corpus_iterN.json \
  > /tmp/iterN_runner.log 2>&1 &
```

Add `--no-chain-session` if comparing against the chain-off baseline.

**`--enable-recovery-turn` is now default-on for all iters from iter 33 onward.** The flag instructs the runner to send ONE generic follow-up message in the same conversation when a case fails its initial asserts (and the chat returned 200). Two trigger paths inside `_build_recovery_message`:
- *4xx recovery*: if any tool call hit a 4xx, send "the previous `<tool>` call failed: `<err>` — re-read structured recovery fields (`validParentNames`, `allowedNodeKinds`, …) and retry with corrected arguments."
- *No-tool-fired*: if zero successful tools fired but the case expected one, send "the previous turn produced no tool call — please call the appropriate tool now."

Cases that pass on the second turn are scored `recovered-pass` (counts toward effective rate). Iter 33 measured +10 cases salvaged this way (vs running with the flag off). Always include `--enable-recovery-turn` unless deliberately measuring the no-recovery baseline.

**Per-case verdicts stream realtime.** The runner explicitly line-buffers stdout at startup ([`smoke_corpus_runner.py:48`](../scripts/debug/smoke_corpus_runner.py) `sys.stdout.reconfigure(line_buffering=True)`), so `tail -f /tmp/iterN_runner.log` shows ✅/❌/🛟 status lines as cases complete. This lets you halt a run early when it's clearly off-rails — iter 33 lost ~80 min wallclock because the buffering bug deferred all verdicts to end-of-run.

### 5. Monitor

```bash
tail -f /tmp/iter33_runner.log
# bridge events (compact telemetry):
tail -f /tmp/iterN_bridge.log | grep -E "compact|request_complete"
```

Iter typical wall-clock: 75 min for chain-on with COMPACT_EVERY_N=2;
~41 min for chain-off.

## Failure recovery cheat-sheet

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Composer 502 from bridge | gateway in wrong namespace | re-run step 3 |
| Bridge healthz fails | bridge crashed | check `/tmp/iterN_bridge.log` and restart |
| Composer returns `tools count: 0` | no deployment loaded | re-run step 1 (deployment + program load) |
| Model never sees a newly-added tool | MCP wrapper cached old manifest | bounce gateway (step 3) |
| `inference.local` unreachable | gateway in pod root NS, not SSH NS | re-run step 3 — DO NOT start gateway via `kubectl exec` directly |
| `model_fallback_decision: timeout` in gateway log | same as above | same |
| `Empty reply from server` | port-forward stale OR gateway down | `openshell forward stop/start 18789` then step 3 |
