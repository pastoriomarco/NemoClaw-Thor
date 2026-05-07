# Composer-assistant lane debugging tools

Operator-facing tools for capturing the *exact* HTTP traffic that
reaches vLLM on the direct vs. OpenClaw lanes, and diffing every field
side-by-side. Use these when one lane works and the other doesn't and
you need to find the structural difference at the wire.

These scripts are durable, not transient: they assume the standard
ManyForge stack from `setup-manyforge-assistant.sh` and the bridges
launched by `start-openclaw-assistant-bridge.sh` /
`start-bridge.sh`.

## Files

- `vllm-logging-proxy.py` — single-file HTTP reverse proxy that logs
  every request/response body as JSONL. Multi-100KB JSON bodies that
  span many TCP packets are parsed correctly (tcpdump-then-regex
  isn't reliable for these — verified empirically).
- `lane-parity-diff.py` — runs the same prompt on both lanes
  back-to-back, captures each lane's vLLM-bound chat-completion via
  the proxies, and emits a field-by-field diff (top-level scalars,
  sampling params, tools[], messages[], extras, response). Writes per-turn
  request/response JSON to disk for byte-level inspection.
- `lane-3x3-smoke.py` — multi-round reliability smoke. Runs 3 prompts
  × 3 rounds × 2 lanes (~25 min on Cosmos-8B) with one Composer
  container restart per lane, captures per-run results to JSON, and
  prints a pass-rate table. The auto pass-detector relies on
  `/api/program/state` (currently 404 in Composer); ground truth comes
  from `manyforge_assistant_bridge/audit.log` (direct lane) and
  `docker logs manyforge-e2e-composer` filtered to the in-sandbox
  bridge IP `172.18.0.2` (OpenClaw lane). See
  [LANE-COMPARISON-direct-vs-openclaw.md §8.2](../../docs/LANE-COMPARISON-direct-vs-openclaw.md)
  for the production matrix this script generates.

## When to reach for this

- Direct lane works but OpenClaw lane fails (or vice versa) on the
  same prompt and same model.
- You suspect the request that arrives at vLLM differs between lanes
  and want to confirm exactly *which* fields differ.
- You need to debug per-turn agent-loop behavior on the OpenClaw side
  (the gateway runs its own loop and emits multiple chat completions
  per Composer request).
- You're verifying that a config or code change actually reaches the
  wire (gateway/bridge layers can drop fields silently).

## Setup (once per session)

The proxies sit between the OpenAI clients (the direct bridge and the
OpenClaw gateway) and vLLM. They listen on two ports and forward to
`vLLM:8000`.

```bash
DEBUG=/path/to/manyforge/scripts/debug

# Direct-lane proxy (bridge :8100 → :8001 → vLLM :8000)
python3 "$DEBUG/vllm-logging-proxy.py" \
    --listen-port 8001 \
    --upstream http://127.0.0.1:8000 \
    --log-path /tmp/vllm_direct_proxy.jsonl &

# OpenClaw-lane proxy (gateway :18789 → :8002 → vLLM :8000).
# bind 0.0.0.0 so the in-sandbox gateway can reach it via the docker
# bridge IP (host.openshell.internal:8002).
python3 "$DEBUG/vllm-logging-proxy.py" \
    --listen-port 8002 --bind 0.0.0.0 \
    --upstream http://127.0.0.1:8000 \
    --log-path /tmp/vllm_openclaw_proxy.jsonl &
```

Then point each lane at its proxy:

- **Direct bridge:** start with `BRIDGE_UPSTREAM_BASE_URL=http://127.0.0.1:8001/v1`.
- **OpenClaw gateway:** override the in-sandbox config so
  `models.providers.inference.baseUrl=http://host.openshell.internal:8002/v1`.
  The `manyforge-composer` egress preset already allows port 8002 with
  the same rules as port 8000 (debug-only entry; safe to leave even
  when the proxy isn't running).

## Running a parity diff

After both lanes are up and pointing at their proxies:

```bash
python3 scripts/debug/lane-parity-diff.py "add a repeat node as root"
```

This will:
1. Restart the Composer container in `nemoclaw` (direct) mode.
2. Reset the program (forceDiscardOverrides + deploymentPath).
3. Send the prompt, capture every chat-completion that hits vLLM.
4. Switch to `openclaw` mode; reset; send again; capture.
5. Emit a side-by-side diff to stdout AND write artifacts to `/tmp`:
   - `lane_parity_<ts>_summary.json` — combined capture
   - `lane_parity_<ts>_diff.txt` — readable diff
   - `lane_parity_<ts>_{direct,openclaw}_request_<turn>.json`
   - `lane_parity_<ts>_{direct,openclaw}_response_<turn>.json`

The diff highlights every divergence with `❗`. The **Messages**
section finds the first byte that differs between the user/system
content; the **Tools** section names tools present on only one lane.

## What the diff has historically found

- OpenClaw drops `tool_choice`, `temperature`, `top_p`, `top_k`, and
  `chat_template_kwargs` on the way from gateway to vLLM (verified
  2026-05-07). Only `model`, `messages`, `tools`, `stream`, and
  `max_completion_tokens` are forwarded.
- OpenClaw injects ~16 KB of "personal-assistant" boilerplate as a
  system prompt before the bridge's payload, regardless of skill.
- OpenClaw prefixes MCP tool names with the server name
  (`manyforge__tree_draft_wrap_node` instead of canonical
  `tree_draft_wrap_node`). The model trained on canonical names can
  drift to prose-mode when the names are unfamiliar AND no tool_choice
  is pinned.
- OpenClaw exposes its internal `session_status` tool to the model as
  a sibling of the ManyForge tools — a distractor on Composer tasks.

## Re-arming during longer sessions

`vllm-logging-proxy.py` truncates its log on startup, so each session
starts fresh. The harness reads with byte-offsets-from-baseline so
multiple runs in one session don't conflict, but for clean traces
restart the proxies between major test campaigns.

If a proxy dies (e.g. the sandbox tears down its docker bridge), you
can relaunch it with the same command — `allow_reuse_address=True` so
the listen port comes back immediately.
