# Runtime mods

This directory holds runtime patches (shell scripts) that the container
entrypoint applies to the installed vLLM/FlashInfer Python packages
before exec-ing into the requested command.

A mod is a subdirectory containing an executable `run.sh`. To activate
one for a launch, set:

```
-e VLLM_MODS=mod-name-1,mod-name-2
```

If `VLLM_MODS` is unset, the entrypoint skips this dispatch entirely.

## Currently shipped: 1 mod (dormant by default)

### `fix-pr39931-turboquant/`

**Status:** **obsolete on v9 (vLLM 0.22.0)** — PR #39931 merged upstream on
2026-05-05 (commit `4f2af1a7`) and is included in vLLM v0.20.2 / v0.21.0 /
v0.22.0. The hybrid-rejection guard the mod patched around is no longer
present in v9 vLLM source.

**Status (v7-v8.1 images):** restored 2026-04-27 after v7 image testing
revealed PR #39931 **did NOT** actually merge into vLLM v0.20.0 (despite
earlier research claiming it did). The hybrid-rejection guard is still
present in v0.20.0 / v0.20.1 source, blocking TurboQuant on Qwen3.6 hybrid
(linear+full attention) profiles. Keep the mod in the directory for back-
compat with anyone still running v7/v8/v8.1 images.

**For v9 launch.sh wiring:** the `THOR_DOCKER_ENV_ARGS+=("-e"
"VLLM_MODS=fix-pr39931-turboquant")` lines on the 3 TurboQuant profiles
(`fp8-turboquant`, `nvfp4-tq-mtp`, `nvfp4-tq-mtp-manyforge`) become no-ops
on v9 since the entrypoint will detect the patches are already applied
upstream (marker-based idempotency). Safe to leave in place; cleaner to
drop once the v8.x lineage is fully retired.

**What it does:** replays PR #39931's 6 source-level hunks across 4
files (`vllm/engine/arg_utils.py`, `vllm/model_executor/layers/quantization/turboquant/config.py`,
`vllm/platforms/interface.py`, `vllm/v1/attention/backends/turboquant_attn.py`).
Idempotent (uses marker-based detection to skip if already applied).

**Affected profiles:** the 3 TurboQuant profiles (`fp8-turboquant`,
`nvfp4-tq-mtp`, `nvfp4-tq-mtp-manyforge`) need this mod to boot on v7.

**To activate:** the v7 image needs to be either:
1. **Rebuilt** (so `COPY mods/` picks up the restored files), OR
2. Bind-mount the host's `serving/docker/mods/` over `/workspace/mods/` at
   container start (one-line edit in `start-model.sh`'s docker run args)

Then re-add `THOR_DOCKER_ENV_ARGS+=("-e" "VLLM_MODS=fix-pr39931-turboquant")`
to the 3 TurboQuant profiles in `../../launch.sh`.

**Until activated:** TurboQuant profiles are blocked on v7. The
ManyForge production profile (`qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge`)
in particular cannot run on v7 yet. Use v6 (`nemoclaw-thor/vllm:v6-pinned-2026-04-17`)
for ManyForge until this is wired up.

### Removed in this lineage:
- `fix-nvfp4-moe-scale-merge/` — for MiniMax NVFP4 split-scale checkpoints.
  Profile removed 2026-04-23. No active caller.

If a future need for runtime patching arises, drop a `mod-name/run.sh`
here and reference it via `VLLM_MODS=...` in the relevant profile in
`../../launch.sh`. Use `git log -- serving/docker/mods/` to see prior structures.
