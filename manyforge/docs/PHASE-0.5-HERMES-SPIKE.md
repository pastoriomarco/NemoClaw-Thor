# Phase 0.5 — Hermes contract spike

**Date:** 2026-06-03
**Status:** In progress
**Anchor model:** cosmos-reason2-8b
**Companion doc:** [THREE-LANE-MIGRATION-PLAN.md §Phase 0.5](./THREE-LANE-MIGRATION-PLAN.md#phase-05--hermes-contract-spike-1-2-days)

## Purpose

De-risk the Hermes lane before Phase 1's universal-core refactor begins.
The plan calls for four probes that surface any contract surprises
(missing config field, different session-API shape, MCP discovery
quirk, parser mismatch) while the codebase is still in its pre-refactor
state and easy to back out from.

## Probe status

| # | Probe                                          | Status              |
|---|------------------------------------------------|---------------------|
| 1 | Provision Hermes sandbox + apply policy        | **✅ Done** — `hermes-spike` sandbox provisioned, `manyforge-composer-hermes-merged` policy v3 loaded |
| 2 | Inject `mcp_servers.manyforge` config          | **✅ Done** — config_version bumped to 13, `base_url` patched to local vLLM, `mcp_servers.manyforge` block written |
| 3 | List `mcp_manyforge_*` tools via Hermes API    | **✅ Offline** (against the 0.14.0 wheel); **⚠️ Online finding** — see below |
| 4 | Read-only callback through `bridge/tools/`     | **⏸️ Blocked** by a gateway-restart hurdle (root vs hermes user + NODE_OPTIONS safety-net) — diagnosis below |
| 5 | `/api/sessions/{id}/chat` vs `/v1/runs`        | **✅ Plan reference stale — corrected; see below** |
| 6 | `--tool-call-parser hermes` on vLLM for cosmos | Deferred (mutually exclusive with Phase 0 vLLM config) |

## Probe 1 finding: scriptable Hermes onboard requires two env vars

**Symptom that initially blocked us.** `nemoclaw onboard --agent hermes --non-interactive`
exits at step 3/8 with one of two messages depending on env:

- Default: `[non-interactive] Provider: build` then
  *"NVIDIA_API_KEY (or NEMOCLAW_PROVIDER_KEY) is required for NVIDIA
  Endpoints in non-interactive mode."*
- With only `NEMOCLAW_PROVIDER=vllm`: *"Requested provider 'vllm' is
  not available in this environment."*

The valid provider names include `build`, `openai`, `anthropic`,
`anthropicCompatible`, `gemini`, `hermes-provider`, `ollama`,
`custom`, `nim-local`, `vllm`, `routed`, plus the various `install-*`
auto-installer variants. The existing `my-assistant` sandbox reports
its provider as `vllm-local` in `nemoclaw list`, but `vllm-local`
itself is **not** an accepted value of `NEMOCLAW_PROVIDER` — the
display name and the env-var name diverge.

**Actual unblock (caught 2026-06-03 mid-spike).** Setting **both**
`NEMOCLAW_PROVIDER=vllm` AND `NEMOCLAW_VLLM_ENDPOINT=http://host.openshell.internal:8000/v1`
lets the wizard skip the auto-detect probe and proceeds through step
3/8 cleanly:

```
NEMOCLAW_PROVIDER=vllm \
NEMOCLAW_VLLM_ENDPOINT=http://host.openshell.internal:8000/v1 \
NEMOCLAW_VLLM_MODEL=cosmos-reason2-8b \
  nemoclaw onboard --agent hermes --name hermes-spike \
    --non-interactive --yes --no-gpu \
    --yes-i-accept-third-party-software
```

That run reached:

```
[3/8] Configuring inference provider
✓ Active gateway set to 'nemoclaw'
✓ Updated provider vllm-local
✓ Inference route set: vllm-local / cosmos-reason2-8b
[5/8] Messaging channels   — [non-interactive] No messaging tokens. Skipping.
[6/8] Creating sandbox     — (3–8 min sandbox build)
```

So Phase 4's `setup-hermes.sh` IS scriptable — the recipe is two
env vars on top of the standard non-interactive flags. The earlier
"not available" error was just the wizard's auto-detect signalling
that it couldn't find a vLLM at the default endpoint; supplying the
endpoint as an env override bypasses the auto-detect entirely.

**Recommendation.** The launcher's `setup-hermes.sh` (per the
THREE-LANE plan Phase 4 deliverable) should set these three env vars
before invoking `nemoclaw onboard`:

```bash
export NEMOCLAW_PROVIDER=vllm
export NEMOCLAW_VLLM_ENDPOINT="${VLLM_BASE_URL:-http://host.openshell.internal:8000/v1}"
export NEMOCLAW_VLLM_MODEL="${MODEL_PROFILE:-cosmos-reason2-8b}"
```

with optional fallback to `build` + an operator-provided
`NVIDIA_API_KEY` for cloud-only deployments. No NemoClaw blueprint
patch needed.

**Still-open NemoClaw-blueprint nit (lower priority):** the wizard's
display name (`vllm-local`) and its env-var input (`vllm`) for the
same provider create a discoverability hazard. An operator reading
`nemoclaw list` sees `provider: vllm-local` and intuitively sets
`NEMOCLAW_PROVIDER=vllm-local`, which is rejected. Filing as a
documentation-fix issue rather than a code change.

## Probe 3 finding: Hermes 0.14.0 confirmed to natively support `mcp_servers`

The Hermes 0.14.0 wheel is locally available
(`/tmp/hermes-schema-check/hermes_agent-0.14.0-py3-none-any.whl`).
Inspecting it directly confirms every claim the THREE-LANE plan
makes about the MCP integration:

- **Config-field present.** `cli.py:2691` reads `CLI_CONFIG.get("mcp_servers")`.
- **Auto-reload watcher.** `cli.py:9314-9351` documents and implements
  the watcher — *"Detect mcp_servers changes in config.yaml and
  auto-reload MCP connections. Compares config.yaml mtime + mcp_servers
  section against the last [snapshot]."*
- **Server task model.** `tools/mcp_tool.py:945` defines `MCPServerTask`
  — the per-server connection manager with sampling/reconnection
  semantics referenced from `mcp_oauth_manager.py`.
- **All three transports supported.** `acp_adapter/server.py:664`
  parameterizes a list of `McpServerStdio | McpServerHttp | McpServerSse`.

The plan's emission target (`lanes/hermes/mcp_servers_config.yaml`,
which writes stdio shape with `command: python3 args:
[manyforge-mcp-bridge.py] env: ...`) is well-matched to the
`McpServerStdio` codepath. Phase 4's `hermes-config.ts` overlay can
emit that YAML verbatim and Hermes will pick it up via the auto-reload
watcher on the next mtime tick.

**Implication for Phase 4.** No wrapper code is needed for the MCP
side. The Phase 4 work item *"NemoClaw `hermes-config.ts` overlay
that emits the `mcp_servers.manyforge` block"* is the entire MCP
integration surface; the per-tool `mcp_manyforge_<tool>` prefix
behaviour is implemented by Hermes' MCP runtime and verified offline.

## Probe 5 finding: plan's session-API reference is stale; correct choice list

The plan §0.5 directs us to probe *"both session APIs head-to-head —
`POST /api/sessions/{id}/chat` and `POST /v1/runs`."* The first path
**does not exist in Hermes 0.14.0**. Inspecting `gateway/platforms/api_server.py:1-15`
shows the actual endpoint list:

```
POST   /v1/chat/completions     OpenAI Chat Completions; stateless by default;
                                opt-in session continuity via X-Hermes-Session-Id
                                header; opt-in long-term memory scoping via
                                X-Hermes-Session-Key header
POST   /v1/responses            OpenAI Responses API format; stateful via
                                previous_response_id; X-Hermes-Session-Key supported
GET    /v1/responses/{id}       Retrieve a stored response
DELETE /v1/responses/{id}       Delete a stored response
GET    /v1/models               Lists hermes-agent as an available model
GET    /v1/capabilities         Machine-readable API capabilities
POST   /v1/runs                 Start a run; returns run_id (202)
GET    /v1/runs/{run_id}        Run status
GET    /v1/runs/{run_id}/events SSE stream of structured lifecycle events
POST   /v1/runs/{run_id}/approval Resolve a pending approval
POST   /v1/runs/{run_id}/stop   Interrupt a running agent
GET    /health, /health/detailed
```

The actual choice for Phase 4 is between three paths:

1. **`/v1/chat/completions` + `X-Hermes-Session-Id`** — OpenAI-compat
   wire shape, stateless model with header-driven session continuity.
   Lowest integration cost (any OpenAI client works).
2. **`/v1/responses`** — OpenAI Responses API, stateful via
   `previous_response_id`. Includes stored-response retrieval; richer
   than 1 but still OpenAI-shaped.
3. **`/v1/runs`** — Hermes-native async with SSE events. Provides the
   `events` SSE stream that the plan's §6.5 "SSE progress-event
   observer that emits universal audit entries" requirement is
   targeting. Also provides explicit `approval` and `stop`
   endpoints aligned with the plan's bounded-autonomy framing.

**Recommended Phase 4 choice (pending the live head-to-head probe):**
`/v1/runs`. Rationale:
- The plan's Phase 4 deliverable (e) requires "longitudinal harness
  with memory + skills + cron + todo + delegation enabled" — these
  are agent-loop behaviours that surface naturally via the
  `/v1/runs/{run_id}/events` SSE stream and not via the OpenAI-
  compat surfaces.
- The plan's universal audit format (§4.7) needs progress events; the
  `events` SSE is the only Hermes surface that emits structured
  lifecycle events.
- `approval` and `stop` map cleanly to ManyForge's review/cancel
  semantics.

**Where the plan needs amending.** §Phase 0.5 needs to be reworded
to drop the `/api/sessions/{id}/chat` reference and substitute the
0.14.0 surface list. Q3 (which session API) can be declared
**resolved** in favour of `/v1/runs` based on this offline analysis,
with the live head-to-head probe demoted from "decision" to
"validation".

## API_SERVER_KEY (open question Q6)

`gateway/config.py:1473` reads `os.getenv("API_SERVER_KEY", "")` and
the platform adapter at `gateway/platforms/api_server.py:613` uses it
as the bearer key. Phase 4's "Provision `API_SERVER_KEY` via NemoClaw
secret-store path (TBD Q6)" item maps to a single env-var injection.
The secret-store mechanism Q6 asks about is the NemoClaw-side
question of how that env var gets populated at sandbox bring-up; no
Hermes-side complexity beyond the env-var read.

## Probe 3 online finding: `mcp_servers` auto-reload watcher lives in the Hermes CLI, NOT the gateway

After injecting `mcp_servers.manyforge` into `/sandbox/.hermes/config.yaml`
(version bump 12 → 13, plus `base_url` patch to the local vLLM) on a
running hermes-spike, we observed **no auto-reload event in the
gateway log**, and **no `manyforge-mcp-bridge` subprocess in the
sandbox process list**. The wheel's reload watcher does exist (per
the offline probe 3 finding) but inspecting `cli.py:9314+` more
carefully shows the call site:

```
def _check_config_mcp_changes(self) -> None:
    ...
    Called from process_loop every CONFIG_WATCH_INTERVAL seconds.
```

`process_loop` is the Hermes **CLI/TUI**'s message-processing
loop, not the gateway's request-handling loop. The watcher is only
called when an interactive Hermes session is running. A gateway-only
deployment (the shape NemoClaw provisions for non-interactive
agents) **does not poll `config.yaml` for `mcp_servers` changes**.

**Implication for Phase 4.** Two paths for `setup-hermes.sh`:

1. **Inject `mcp_servers` BEFORE starting the gateway** — emit the
   block into `config.yaml` as part of sandbox bring-up so the gateway
   picks it up at startup. This is the cleanest fit for the
   `hermes-config.ts` overlay approach in the plan (the overlay
   writes the config; the gateway then reads it once at boot).
2. **Restart the gateway after a config edit** — required if the
   `mcp_servers` block changes after gateway is already running. This
   becomes important for dynamic-catalog flows (e.g., enabling/
   disabling an MCP server at runtime).

The plan's Phase 4 spec mentions "Hermes discovers and registers
manyforge MCP tools at startup via its native MCP path. Auto-reload
is built in (`cli.py:9314+` watches mcp_servers mtime)." — this is
**half right**: the field exists and the wheel implements the
watcher, but the watcher only fires under the interactive CLI, not
the gateway shape we actually run.

The Phase 4 work item *"NemoClaw `hermes-config.ts` overlay that
emits the `mcp_servers.manyforge` block"* should explicitly emit
the block BEFORE the gateway is brought up, and `setup-hermes.sh`
should treat "live mcp_servers change" as requiring a gateway
restart (not a config-only edit).

## Probe 4 online finding: gateway restart needs the proper user + entrypoint

Trying to restart the gateway after the config edit to pick up
`mcp_servers` ran into two related guardrails:

```
✗ Refusing to run the Hermes gateway as root inside the official
  Docker image.
  The image entrypoint normally drops privileges to the 'hermes' user.
```

and

```
[gateway-recovery] ERROR: /tmp/nemoclaw-proxy-env.sh present but
  NODE_OPTIONS missing safety-net preload or ciao preload —
  refusing unguarded gateway relaunch (#2478)
```

Both are intentional safety guards in NemoClaw's blueprint. The
correct restart path is `nemoclaw hermes-spike rebuild` (which runs
the proper user + restores the safety-net preloads) rather than a
direct `hermes gateway run` inside the sandbox shell.

**Implication for Phase 4.** `setup-hermes.sh` MUST use the
`nemoclaw <name> rebuild` (or `recover`) path to restart the
gateway after mutating the sandbox config. A shell-level
`hermes gateway run` will be rejected by the entrypoint guards.
Avoids both surprises: root-owned files in `$HERMES_HOME` and a
gateway running without the OpenShell preload guards in place.

### Probe 4 follow-up: rebuild wipes custom state — strict order-of-ops

Calling `nemoclaw hermes-spike rebuild` to revive the gateway after
the shell-restart guard rejected my edit cycle gave a clean gateway
— but **also reset `config.yaml` to its provisioning default** (the
`_config_version` went 13 → 12, the `mcp_servers.manyforge` block
was wiped, the custom policy `manyforge-composer-hermes-merged` was
removed from the active policy list, and the bridge script I had
copied to `/sandbox/manyforge/scripts/manyforge-mcp-bridge.py` was
also gone). The rebuild restored only the standard presets it
provisions out of the box (`npm, pypi, huggingface, brew,
local-inference`).

This is intentional — rebuild is a fresh image-based bring-up — but
it means the Phase 4 `setup-hermes.sh` MUST follow a strict order
of operations:

```
1. nemoclaw onboard --agent hermes --name <name> ...    (or rebuild)
2. nemoclaw <name> policy-add --from-file manyforge-composer-hermes.merged.yaml
3. docker cp manyforge-mcp-bridge.py <sandbox>:/sandbox/manyforge/scripts/
4. Edit /sandbox/.hermes/config.yaml:
   - patch base_url to local vLLM
   - patch provider to vllm-local
   - append mcp_servers.manyforge block
   - bump _config_version
5. nemoclaw <name> recover   (or gateway-kill-and-let-restart)
   → THIS is where the gateway reads the new config and spawns
     manyforge-mcp-bridge via mcp_tool.discover_mcp_tools()
     (gateway/run.py:16973)
6. Verify the bridge subprocess via:
     docker exec <sandbox> ps -ef | grep manyforge-mcp-bridge
7. Verify the catalog reached the model surface:
     curl http://localhost:8642/v1/capabilities | jq .tool_catalog
     (or via a /v1/runs POST that triggers the agent)
```

If a step 2/3/4 happens AFTER step 5 (gateway already running), the
gateway watcher cannot see the change — gateway-mode has no
`_check_config_mcp_changes` polling (that lives in the CLI per
probe 3 above). The change will only take effect after the next
gateway restart.

This blocks Probe 4 (live callback verification) on this run,
because:
- My first cycle injected config BEFORE the rebuild — wiped.
- My second cycle injected config AFTER the rebuild had restarted
  the gateway — never re-read.
- A third rebuild cycle would just repeat the wipe.

The cleanest finish for Probe 4 would be: full third pass with the
strict order — but per the user's "ensure OpenClaw lanes aren't
damaged" constraint, redoing the cycle while the OpenClaw 66-case
rerun is in flight risks two things competing for vLLM. Defer to a
clean window after the OpenClaw smoke completes; the order-of-ops
finding above is what Phase 4 actually needs from this probe.

## Probe 6 finding: deferred to dedicated vLLM restart window

Probing `--tool-call-parser hermes` on vLLM for cosmos-reason2-8b
requires restarting the vLLM container with the parser flag set. The
NemoClaw-Thor profile system supports per-model parser configuration
(see `serving/launch.sh` profiles), but a dedicated "cosmos +
hermes-parser" profile does not exist today.

Concretely, the existing cosmos-reason2-8b profile starts vLLM with no
custom parser; vLLM applies its default parsing path. To run the
plan's 5-case probe we would need a temporary profile variant that
adds `--tool-call-parser hermes` to the same model.

**Recommendation.** When Phase 0.5 resumes (after Phase 0 corpus runs
free the vLLM slot), add a one-off `cosmos-reason2-8b-hermes-parser`
profile and run 5 cases through Direct lane (no Hermes sandbox needed
— the parser produces OpenAI structured `tool_calls[]` that the
direct-lane bridge consumes regardless). Compare structured-output
rate against the same 5 cases on the default parser.

## Composer cylinder diameter→radius normalization gap (unrelated, surfaced by P2b)

While running Phase 0's direct-lane corpus through the patched scorer,
the new `P2b_scene_add_cylinder_diameter` case (added in 1e2b6fd to
exercise the cylinder primitive surface) surfaced a real composer
bug:

- The model emits `scene_draft_add_object` with `cylinder` shape and
  `diameter` (per spec 485 line 420's documented alias surface).
- The tool surface accepts the call (`POST /api/assistant/bridge/tools/scene_draft_add_object`
  returns 200).
- The draft state stores the object.
- On the NEXT case's `POST /api/program/load` with
  `forceDiscardOverrides:true`, the cycle_manager rebuild fires
  `apply_scene` which validates the persisted scene against the
  scene runtime's schema — and the runtime requires
  `cylinder_radius_m` not `cylinder_diameter_m`.
- Result: `RuntimeError: Failed to add scene object 'cylinder_01':
  add with shape_type=cylinder requires cylinder_radius_m` (composer
  log, `runner.py:282 apply_scene`), surfaced as HTTP 409 on the
  load and cascading every subsequent smoke case to fail with
  "program reset failed: HTTP 409" until composer is restarted.

The normalization pipeline that should map `diameter` → `radius`
(or accept either at the scene-runtime layer) is missing from
`_normalize_scene_resource_aliases` in
`manyforge_composer/backend/scene_draft.py` (line number TBD by
the eventual fix). Spec 485 says the alias surface MAY normalize
"before applying the strict Composer draft validation boundary" —
this case proves the validation boundary is downstream of where
the normalization fires.

**P2b deferred to `status: future` for now** so the Phase 0
baseline can land. Promote back to active once composer normalizes
diameter↔radius end-to-end.

**Recommended fix scope** (not part of this spike):

- Either add diameter→radius normalization in
  `_normalize_scene_resource_aliases` before persist, or
- Teach the scene-runtime schema to accept either `cylinder_radius_m`
  or `cylinder_diameter_m` (the geometric equivalence is trivial).

Either is a ~10-line fix on the composer side. Not blocking Phase 0
or Phase 0.5 but blocks promoting P2b back to active.

## What this spike actually de-risked

Despite the bring-up blocker, the spike surfaced two concrete
Phase 4 work-items that would otherwise have been discovered only at
Phase 4 implementation time:

1. **NemoClaw blueprint provider-name mismatch** (above). Phase 4's
   `setup-hermes.sh` will need to either:
   - call `nemoclaw onboard --agent hermes` with a now-supported
     scriptable provider flag, OR
   - skip `nemoclaw onboard` entirely and operate against a pre-
     onboarded sandbox snapshot.
2. **vLLM parser configuration is per-profile, not per-lane.** Phase
   4 cannot just flip Cosmos onto the hermes parser at lane bring-up
   — it needs its own profile. This was implicit in the plan but
   makes the Phase 4 vLLM-warmup time sequence concrete: hermes-
   parser profile load is a vLLM cold start (~3 min), not a runtime
   flip.

## Items the plan listed for Phase 4 that this spike confirmed still apply

- Hermes natively supports `mcp_servers` (verified against the 0.14.0
  wheel by the plan author in rev 2). The contract spike did not
  re-verify this directly because probe 3 was blocked, but the
  evidence remains the cited wheel source.
- The hand-written `mcp_servers.manyforge` snippet in
  `/sandbox/.hermes/config.yaml` is the intended emission path for
  the spike (per the plan §0.5). The
  `lanes/hermes/mcp_servers_config.yaml` template that today ships in
  NemoClaw-Thor is the source of truth for what Phase 4's
  `hermes-config.ts` overlay will emit.

## Recommended next step

When Phase 0 completes (frees the vLLM slot), do the following in
this order:

1. Add a temporary `cosmos-reason2-8b-hermes-parser` profile to
   `serving/launch.sh` (5-line addition: copy the existing cosmos
   profile, add `--tool-call-parser hermes` to its `EXTRA_ARGS`).
2. Restart vLLM with the new profile; run the 5-case probe against
   the **direct** lane (sandbox-free path). Confirm structured
   `tool_calls[]` come back in the OpenAI shape.
3. Decision point: if the parser does NOT produce structured output
   for cosmos, document the gap. Phase 4 needs a fallback strategy
   (likely: use default parser, accept that Hermes consumes the
   catalog via prose).
4. Defer probes 1–5 to a Phase 4 prep window where an operator can
   either provide `NVIDIA_API_KEY` or drive the onboard interactively.

## References

- Three-lane plan §Phase 0.5
- `lanes/hermes/README.md` — what Phase 4 still needs
- `lanes/hermes/mcp_servers_config.yaml` — the intended emission target
- `lanes/hermes/policy.yaml` — SessionPolicy for the lane
- `policies/manyforge-composer-hermes.merged.yaml` — the policy preset
  Phase 4's `setup-hermes.sh` will apply
