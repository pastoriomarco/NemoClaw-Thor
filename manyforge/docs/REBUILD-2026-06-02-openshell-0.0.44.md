# Rebuild — OpenShell 0.0.36 → 0.0.44 + OpenClaw 2026.4.24 → 2026.5.22

**Date**: 2026-06-02
**Trigger**: NemoClaw `lkg` (v0.0.55, "last known good") pinned new versions; OpenShell v0.0.37 introduced a breaking change (gateway protocol overhaul; k3s cluster replaced by docker driver). The HEAD tag at session time was v0.0.56, which only adds [PR #4613 (defaulting public installs to lkg)](https://github.com/NVIDIA/NemoClaw/pull/4613) — same OpenClaw 2026.5.22 pin, same sandbox image digest — so this doc targets `lkg` for reproducibility.
**Outcome**: stack working end-to-end with patches. Cosmos full smoke launched.

## TL;DR

Twelve breaking changes (some load-bearing, some cosmetic) had to be fixed across the manyforge stack to restore composer → bridge → openclaw → proxy → vLLM operation. The complete diff lives in:

- [`scripts/proxy/vllm-proxy.py`](../scripts/proxy/vllm-proxy.py) — new `reasoning→content` response mutation
- [`openclaw_assistant_bridge/adapter.py`](../openclaw_assistant_bridge/adapter.py) — exec wrapper rewritten for docker driver + base64-wrap newline guard
- [`openclaw_assistant_bridge/service.py`](../openclaw_assistant_bridge/service.py) — stderr capture on 502 path
- [`setup-manyforge-assistant.sh`](../setup-manyforge-assistant.sh) — five patches
- [`policies/manyforge-composer.preset.yaml`](../policies/manyforge-composer.preset.yaml) — IP allowlist expanded for new docker bridge subnet

A headless reproduction is at [`scripts/rebuild-headless-onboarding.sh`](../scripts/rebuild-headless-onboarding.sh).

## Architecture changes upstream

### OpenShell 0.0.37+ — driver swap

The v0.0.36 cluster container `ghcr.io/nvidia/openshell/cluster:0.0.36` ran a single-node k3s cluster, with the gateway + sandboxes as pods inside it. We accessed sandboxes via:

```
docker exec openshell-cluster-nemoclaw kubectl exec -n openshell <sandbox> -c agent -- su sandbox -c "<cmd>"
```

v0.0.44 replaces this entirely: the gateway is a host-side binary (`openshell-gateway`) backed by Docker directly (`driver: "docker"`). Sandboxes are bare Docker containers managed by the gateway. The k3s layer is gone. The new exec entrypoint:

```
nemoclaw <sandbox> exec --no-tty -- <cmd>
```

The `sandbox` user inside (uid 998, HOME=/sandbox) is the default — no `su sandbox -c` needed.

### OpenClaw 2026.5.22 — stricter response contract + new tool model

Three relevant changes:

1. **Stricter terminal-response check**. 2026.4.24 accepted assistant messages whose `content` was null as long as `reasoning` was populated. 2026.5.22 rejects them with `code=incomplete_result`. Affects any vLLM profile launched with `--reasoning-parser` (e.g. cosmos with `qwen3`).

2. **Tool search compaction**. New "tool search compact prompt surface" pattern that hides MCP tools behind a single `tool_search_code` discovery tool when `agents.<id>.tools.profile != "full"`. We must set `profile: "full"` explicitly.

3. **Validation cap on `postCompactionMaxChars`**. New `≤50000` hard cap. The manyforge agent profile previously used 80000.

4. **Gateway pairing/scope upgrade approval flow**. When the bridge uses `cli_shell_out` transport (the default), the agent starts in embedded mode after the gateway scope-upgrade request remains "pending approval". This is the path we use; OpenClaw fallback is automatic and acceptable.

### NemoClaw 0.0.56 — gateway registration mismatch

NemoClaw's onboard command registers the OpenShell gateway with `https://...` + mTLS, but the gateway it actually starts is plaintext HTTP on `127.0.0.1:8080`. Onboarding step 4 (provider register) then fails with `transport error: received corrupt message of type InvalidContentType` — the OpenShell CLI sends TLS ClientHello bytes (`\x16\x03...`) to the plaintext gateway and the parse error rebounds.

Workaround until NemoClaw fixes the registration: remove + re-add the gateway with `http://` scheme before onboarding completes:

```bash
openshell gateway remove nemoclaw
openshell gateway add http://127.0.0.1:8080 --local --name nemoclaw
```

This restores `openshell provider list`, `openshell sandbox list`, and the rest of `nemoclaw onboard`.

## Twelve fixes applied in this stack

| # | Subject | File | Why |
|---|---|---|---|
| 1 | host CLIs + sandbox image | (already pinned by NemoClaw blueprint) | OpenShell 0.0.44 + OpenClaw 2026.5.22 |
| 2 | gateway re-registration `https`→`http` | (manual `openshell gateway add`) | NemoClaw onboard mis-registers as mTLS |
| 3 | bridge adapter — exec wrapper rewrite | `openclaw_assistant_bridge/adapter.py` | k3s `docker exec ... kubectl exec` removed |
| 4 | bridge adapter — base64-wrap shell command | `openclaw_assistant_bridge/adapter.py` | OpenShell exec gRPC rejects newlines in argv |
| 5 | bridge service — stderr capture on 502 | `openclaw_assistant_bridge/service.py` | otherwise the failure is silent |
| 6 | setup script — exec wrapper rewrite | `setup-manyforge-assistant.sh:176` | same as #3 |
| 7 | setup script — health probe via `exec true` | `setup-manyforge-assistant.sh:Sandbox check` | `nemoclaw status` has a TypeError in 0.0.56 |
| 8 | setup script — `remote_hash` empty-dir guard | `setup-manyforge-assistant.sh:remote_hash` | fresh sandbox tripped `set -euo pipefail` |
| 9 | setup script — split precheck vs runtime base URL | `setup-manyforge-assistant.sh:PRECHECK_*` | host can't resolve `host.openshell.internal` |
| 10 | setup script — `postCompactionMaxChars` 80000→50000 | `setup-manyforge-assistant.sh` | OpenClaw 2026.5.22 validation cap |
| 11 | policy — IP allowlist for docker bridge 172.18 | `policies/manyforge-composer.preset.yaml` | new docker-driver gateway uses `172.18.0.0/16` (was `172.17`) |
| 12 | proxy — `reasoning`→`content` SSE+JSON mutation | `scripts/proxy/vllm-proxy.py` | OpenClaw 2026.5.22 rejects null-content responses |
| 13 | openclaw config — `tools.profile: "full"` + drop `bundle-mcp` | `openclaw.json` (in-sandbox) | hides MCP tools behind compaction otherwise |
| 14 | openclaw config — `models.providers.inference.baseUrl` pinned to `host.openshell.internal:8000` | `openclaw.json` (in-sandbox) | the "managed inference route" via `inference.local` has a 5s lane timeout that breaks cosmos + thinking |
| 15 | bridge env — `OPENCLAW_ASSISTANT_AGENT=manyforge-composer` | shell env | the new OpenClaw doesn't provide a default `main` agent |

(One row over the headline; the fourteenth & fifteenth are config, not code changes, hence the "12 fixes" framing.)

## Recovery procedure (steps in order)

These are the exact steps that worked end-to-end on this Thor host. The headless script reproduces them all.

### Pre-flight

1. **Stop** dependent services: bridge, proxy, vLLM container, smoke runners.
2. **Snapshot** `~/.nemoclaw/sandboxes.json`, `~/.nemoclaw/onboard-session.json` to `/tmp/` so the previous config can be re-applied if anything regresses.
3. **Confirm** the composer container stays up — it owns the loaded program and the assistant API.

### Tear-down (destructive)

4. `docker stop && docker rm openshell-cluster-nemoclaw` (the k3s cluster container)
5. `docker volume rm openshell-cluster-nemoclaw` (k3s state — sandboxes baked here)
6. `docker network rm openshell-cluster-nemoclaw` (custom bridge network)
7. Kill any orphan `openshell ssh-proxy` and forwarded `ssh -L 18789` processes from the previous cluster
8. `rm ~/.nemoclaw/onboard-session.json` and reset `~/.nemoclaw/sandboxes.json` to `{"sandboxes": {}, "defaultSandbox": null}`

### Upgrade host CLIs (one-time)

9. Install NemoClaw at the `lkg` tag (last known good = v0.0.55):
   ```bash
   cd ~/NemoClaw && git fetch --tags && git checkout lkg && npm install -g
   ```
   (Equivalent to `git checkout v0.0.55` — `lkg` is the floating alias NVIDIA bumps on each LKG release. v0.0.56 also works in practice, but v0.0.55=lkg is the version NVIDIA's installer defaults to for fresh public installs.)
10. The chosen NemoClaw revision installs OpenShell 0.0.44 automatically via `install-openshell.sh`, and the sandbox image (built on first onboard) bakes in OpenClaw 2026.5.22.

### Onboard

11. **Start vLLM + proxy first** (`THOR_VLLM_PORT=8050 ./serving/start-model.sh cosmos-reason2-8b`). Onboarding's connectivity check fails otherwise.
12. Run `nemoclaw onboard` interactively (the wizard prompts cleanly when the gateway issue from §2 is patched). Choices:
    - Provider: 3) Other OpenAI-compatible endpoint
    - Base URL: `http://127.0.0.1:8000/v1` (the proxy)
    - API key: any non-empty (vLLM ignores it)
    - Model: `cosmos-reason2-8b`
    - Sandbox name: `my-assistant`
    - Resource profile: 6 (No profile / OpenShell defaults)
    - Policy tier: Balanced
    - Policy presets: **only `local-inference`** (disable npm/pypi/huggingface/brew/brave/openclaw-pricing)
13. As soon as onboarding starts step 4 (provider register), watch for the gRPC `InvalidContentType` error. If it appears: in another shell, run `openshell gateway remove nemoclaw && openshell gateway add http://127.0.0.1:8080 --local --name nemoclaw`, then re-pick option 3 in the wizard. (Documented in §"Architecture changes upstream/NemoClaw 0.0.56".)

### Re-apply manyforge layer

14. `./manyforge/setup-manyforge-assistant.sh my-assistant` — applies policy, installs skill, registers MCP server, installs agent profile.
15. Force-reapply the policy if step 14 reports "already applied" (script's idempotency by name, not content): `nemoclaw my-assistant policy-add --from-file <preset> --force`.

### In-sandbox config patches

16. Edit `/sandbox/.openclaw/openclaw.json`:
    - `agents.list[manyforge-composer].tools` → `{"profile": "full"}` (drop `bundle-mcp` from `alsoAllow`).
    - `models.providers.inference.baseUrl` → `http://host.openshell.internal:8000/v1`.
17. `nemoclaw my-assistant exec --no-tty -- openclaw doctor --fix` — repairs any oversize `postCompactionMaxChars` left from prior runs.

### Restart bridge

18. With env `OPENCLAW_ASSISTANT_AGENT=manyforge-composer` and `PYTHONPATH=<repo>/manyforge`:
    ```
    nohup ./manyforge/openclaw_assistant_bridge/.venv/bin/python -m openclaw_assistant_bridge.service &
    ```

### Validate

19. Single-case smoke: `python3 -u manyforge/scripts/debug/smoke_corpus_runner.py --filter P1_wrap_root_specific` — should return chat HTTP 200 (the case may still fail on model accuracy; that's separate).

## Why thinking must stay ON for cosmos despite the OpenClaw contract change

When the proxy forces `enable_thinking: false` on cosmos, the model produces narration-mode prose instead of `<tool_call>` structured output:

```
> Call program_read.
< Tool call completed. The program_read tool has been successfully executed.
```

(`tool_calls: []`, no actual call.) Cosmos was post-trained from Qwen3-VL with long-CoT reasoning assumed — thinking-off is OOD and corrupts tool-call format. This was documented in `serving/launch.sh:146-152` from iter-17 (2026-05-09).

With thinking-on, cosmos emits `<think>` blocks that vLLM's `qwen3` reasoning parser routes to the `reasoning` field, leaving `content` empty. OpenClaw 2026.5.22 then rejects the response. The proxy's new `reasoning→content` mutation makes both ends happy: cosmos keeps emitting reasoning correctly, OpenClaw sees content populated. The original `reasoning` field also stays populated for any downstream consumer.

## Open questions / deferred items

- **NemoClaw plugin still logs "Endpoint: Managed Inference Route (inference.local)"** even after we pin `baseUrl` to `host.openshell.internal:8000`. The plugin appears to log the configured route name regardless of the actual provider URL. In testing the actual call DOES go to `host.openshell.internal:8000` (confirmed in proxy log `Via: host-side` not `Via: 1.1 openshell-sandbox`). Cosmetic; no functional impact, but worth checking that the route name doesn't reactivate after a sandbox reboot.

- **`bundle-mcp` allowlist entry** was the recommended way in 2026.4.24 to expose MCP tools while using `profile: "minimal"`. In 2026.5.22 it's no longer recognized. `profile: "full"` exposes MCP tools but also exposes every other catalog tool, which is wider than we want. The proper fix is an OpenClaw config update to add an MCP-tools-only profile or restore `bundle-mcp` recognition.

- **OpenClaw managed-route 5-second lane timeout** is hard-coded and breaks any model whose first-token-latency exceeds 5s. The workaround is to pin a direct provider URL. Worth filing upstream.

- **NemoClaw onboard register-as-https-but-spawn-http** mismatch needs a tracking bug upstream. Until then the manual gateway re-registration step is required on every fresh onboard.
