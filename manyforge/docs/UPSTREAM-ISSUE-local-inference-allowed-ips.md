# Upstream issue draft — `local-inference` preset blocks gateway-embedded inference on Docker-bridge deployments

Target repo: NVIDIA/NemoClaw
Suggested labels: bug, network-policy, openshell-integration

> Save and file at https://github.com/NVIDIA/NemoClaw/issues/new — the body
> below is ready to paste. The validation log referenced as a "live
> reproduction" is in `dev_ws/src/NemoClaw-Thor/manyforge/docs/MANYFORGE-MCP-INTEGRATION.md`
> (validation log "Phase 2 latency leap — 2026-05-05 evening (canonical fix)").

---

## Title

`local-inference` policy preset is missing `allowed_ips`, blocking the
OpenClaw gateway lane from reaching vLLM on Docker-bridge sandboxes

## Summary

The built-in `local-inference` policy preset declares
`host.openshell.internal:8000` for vLLM but does **not** include the
canonical `allowed_ips` field required by OpenShell's SSRF guard. On
deployments where `host.openshell.internal` resolves to a private/RFC
1918 address (the default Docker-bridge `172.17.0.1`), this means the
OpenClaw gateway lane (`POST /v1/chat/completions`) cannot reach vLLM
even when `local-inference` is the active preset. The OpenClaw CLI
shell-out path (`openclaw agent --local`) is unaffected because it
takes a different code path.

The symptom from the operator side:

- Persistent gateway returns `{"error":{"message":"internal error","type":"api_error"}}` on every `/v1/chat/completions` call.
- The gateway log shows `lane task error ... error="FailoverError: LLM request failed: network connection error."`
- The OpenShell sandbox log shows repeated `NET:OPEN [MED] DENIED /usr/local/bin/node(<gw_pid>) -> host.openshell.internal:8000 [policy:- engine:ssrf] [reason:host.openshell.internal resolves to internal address 172.17.0.1, connection rejected]`.

## Reproduction (validated 2026-05-05 on Thor)

Software:

- NemoClaw CLI v0.0.31
- OpenShell CLI 0.0.36
- OpenShell cluster image 0.0.36
- OpenClaw v2026.4.24
- Local vLLM serving `nemotron3-nano-omni-30b-a3b-nvfp4` on
  `127.0.0.1:8000` (host.openshell.internal:8000 inside sandbox)

Steps:

1. `nemoclaw onboard` with `local-inference` preset, OpenAI-compatible
   provider pointing at `http://127.0.0.1:8000/v1`.
2. `openclaw config set gateway.http.endpoints.chatCompletions.enabled true`.
3. `nemoclaw <sandbox> connect` (ensures host SSH port-forward to
   `127.0.0.1:18789`).
4. `./setup/configure-local-provider.sh` (runs `ensure_sandbox_gateway_running`
   which spawns the gateway via `nohup openclaw gateway run --auth none --port 18789 &` inside an openshell SSH session).
5. From host: `curl -X POST http://127.0.0.1:18789/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"openclaw/default","messages":[{"role":"user","content":"reply OK"}],"max_tokens":32}'`

Expected: `{"choices":[{"message":{"content":"OK"}}]}` (or similar).

Actual: `{"error":{"message":"internal error","type":"api_error"}}` after
~20-50 s; sandbox log shows SSRF DENY for `host.openshell.internal:8000`.

## Diagnosis

`host.openshell.internal` resolves to `172.17.0.1` (the Docker bridge
gateway) in this deployment. Per the OpenShell policy schema, RFC 1918
addresses are blocked by default. The documented opt-in is the
per-endpoint `allowed_ips` CIDR allowlist (see
`docs.nvidia.com/openshell/latest/reference/policy-schema.html` and
`OpenShell/examples/private-ip-routing`). The `local-inference` preset
has the right OPA rules for the vLLM endpoint but omits `allowed_ips`,
so the SSRF guard rejects the resolved private IP.

Empirically, applying a second preset that DOES declare the same
endpoint with `allowed_ips: ["172.17.0.0/16"]` does not unblock the
call: the SSRF engine appears to honor the **first matching endpoint**
rather than the union of `allowed_ips` across applied presets.
Removing `local-inference` and applying only the augmented preset
unblocks immediately.

The OpenClaw CLI shell-out path (`openclaw agent --local`) is not
affected: it goes through a different network code path that doesn't
trip the SSRF guard. That's why this is invisible to most users —
they typically test through the dashboard or CLI, not through the
gateway's `/v1/chat/completions` endpoint.

## Proposed fix

Either (preferred) update the built-in `local-inference` preset to
include `allowed_ips: ["172.17.0.0/16"]` (or a deployment-configurable
CIDR — the OpenShell deployment knows its own bridge subnet) on the
vLLM endpoint, OR change the SSRF engine to union `allowed_ips` across
all matching preset endpoints rather than honor the first match. The
former is a one-line YAML change to the preset; the latter is
behavioral but more general.

## Workaround currently shipped (NemoClaw-Thor)

The `manyforge-composer` preset in NemoClaw-Thor is now a strict
superset of `local-inference` (same trusted binaries, same vLLM
endpoint, plus the Composer endpoint, both with
`allowed_ips: ["172.17.0.0/16"]`); the provisioner removes
`local-inference` before applying it. Validated 2026-05-05: warm
chat-completions calls 5–8 s through the full canonical stack
(host SSH tunnel → SSH-session gateway → SSRF-guarded vLLM call).

Files:

- [`manyforge/policies/manyforge-composer.preset.yaml`](https://github.com/<your-fork>/NemoClaw-Thor/blob/main/manyforge/policies/manyforge-composer.preset.yaml)
- [`manyforge/setup-manyforge-assistant.sh`](https://github.com/<your-fork>/NemoClaw-Thor/blob/main/manyforge/setup-manyforge-assistant.sh)
- Validation log: [`manyforge/docs/MANYFORGE-MCP-INTEGRATION.md`](https://github.com/<your-fork>/NemoClaw-Thor/blob/main/manyforge/docs/MANYFORGE-MCP-INTEGRATION.md) → "Phase 2 latency leap — 2026-05-05 evening (canonical fix)"

## Why this is worth fixing in the built-in preset

The OpenClaw gateway's persistent `/v1/chat/completions` lane is the
right path for any deployment that wants warm-prefix-cache latency
benefits or wants the gateway to manage the agent loop. Today, anyone
who follows the documented `nemoclaw onboard` → `local-inference` flow
on a default Docker-bridge deployment will silently get a broken
gateway lane and not know why; the failure mode is a generic "internal
error" with no operator-visible hint about SSRF. The single-line
preset fix removes that footgun.
