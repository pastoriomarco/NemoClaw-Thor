# MANYFORGE-PROFILE-CALIBRATION.md — sizing guide for ManyForge-pipeline profiles

> Reusable methodology for calibrating `max_model_len`, `max_num_seqs`,
> and `gpu_memory_utilization` on Thor profiles intended to serve the
> ManyForge assistant pipeline (composer-assistant, error-recovery,
> query modes per `manyforge_specs/docs/spec/480-...md`). Concept-
> focused; specific per-profile numbers live in `lib/config.sh` and
> `lib/launch.sh`.

---

## When to use this doc

Read this before:

- adding a new ManyForge-pipeline-targeted profile to `lib/config.sh` /
  `lib/launch.sh`,
- changing `THOR_TARGET_MAX_MODEL_LEN`, `THOR_TARGET_MAX_NUM_SEQS`, or
  `THOR_LAUNCH_GPU_MEMORY_UTILIZATION` for any existing
  ManyForge-pipeline profile,
- evaluating whether an existing profile (e.g. `qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge`,
  `cosmos-reason2-8b`) is correctly calibrated for the bridge service
  workload after a model swap or hardware change.

---

## Three knobs, in order

### 1. `max_model_len` — minimum context for the agent loop

The OpenClaw in-sandbox agent injects a bootstrap (AGENTS.md, SOUL.md,
TOOLS.md, skills registry, tool schemas) totaling **~16K tokens**. The
default output budget is another **16K tokens**. So a single turn
needs at least 32K of context just to fit the bootstrap and a full
response.

For ManyForge's hybrid agentic workflow (per `manyforge_specs` 480),
agent loops accumulate context across turns:

- multi-step error-recovery sessions inspect scene state, propose
  actions, observe results, iterate,
- composer-assistant edit sessions accumulate proposal/feedback
  history,
- multi-turn `query_model` workloads carry tool-result history.

**Calibration rule:** target the model's native maximum context unless
memory pressure forces a step down. Bigger context removes
context-overflow as a recurring failure mode in multi-step loops, which
is exactly what the bridge / hybrid-autonomy work depends on.

For existing profiles:

- Nemotron-Omni-30B-A3B native max: 256K. Use `262144`.
- Cosmos-Reason2-8B native max: 256K. Currently set to 64K — fine for
  single-turn coding agent, may need bumping for ManyForge multi-turn
  loops once empirically tested.
- Qwen3.6 family native max: 256K. The `*-manyforge` variant currently
  uses 64K (for 3×64K concurrent context); revisit if multi-turn
  agentic loops accumulate beyond that.

### 2. `max_num_seqs` — concurrency budget for the ManyForge load shape

`max_num_seqs` is vLLM's continuous-batching scheduler limit: the
maximum number of sequences (requests) processed in parallel. For
ManyForge, plan for these concurrent sources, all originating inside
one ManyForge composer process:

| Source | Per-source concurrency | Notes |
|---|---|---|
| GUI assistant (composer-assistant / query) | 1 | single user typing |
| Multi-step bridge agent loop | 1 (logical, multi-turn) | bridge runs serial within one request |
| Behavior-tree `query_model` nodes | open-ended | spec 470 §4.1 — non-blocking semantics; parallel BT branches each fire a request |
| Error-recovery sessions | 1 at a time | only at STOPPED, single supervisor |

**Calibration rule:** size for "GUI assistant + N parallel BT nodes +
optional fan-out into subagents", with N depending on tree complexity.
For an unknown future BT shape, a generous `max_num_seqs` future-proofs
without significant cost — vLLM's continuous batching makes the
marginal scheduling overhead immaterial up to the dozens, and unused
slots cost nothing.

Reference points already in this repo:

- `qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge`: `max_num_seqs=3`,
  `OPENCLAW_MAIN_MAX_CONCURRENT=2`. Empirically calibrated for the
  Qwen-manyforge live-test runs; assumes ~2 main agents.
- `nemotron3-nano-omni-30b-a3b-nvfp4`: `max_num_seqs=16`,
  `OPENCLAW_MAIN_MAX_CONCURRENT=1`. Future-proofed against
  unanticipated BT-parallelism (4 main × 3 subagents + headroom).
  Calibration rationale: while we don't yet know if multiple main
  agents are needed, the marginal cost of the higher slot count is
  negligible on Thor's memory budget. Revise downward only if Orin
  deployment forces it (Orin's 64 GB unified memory shared with Isaac
  ROS leaves a tighter envelope).

The OpenClaw side has its own concurrency knob
(`OPENCLAW_MAIN_MAX_CONCURRENT`) that bounds main-agent fan-out. Set
this independently per profile based on agent topology, not vLLM
scheduler capacity.

### 3. `gpu_memory_utilization` — derived from the two above

This knob is computed from the first two, not chosen first. Method:

1. **Start vLLM with a generous `gpu_memory_utilization`** (e.g.
   `0.65`) at the target `max_model_len` and `max_num_seqs`.
2. **Read the load log** for two values:
   - `Available KV cache memory: <X> GiB`
   - `Maximum concurrency for <max_model_len> tokens per request: <Y>x`
3. **Compute fixed costs:** `total_budget − KV_pool` where
   `total_budget = gpu_mem_util × total_unified_memory`. These costs
   (weights, activation buffer, framework overhead) are roughly
   invariant when you change `gpu_memory_utilization`.
4. **Compute the per-slot KV cost at full context:** `KV_pool / Y`.
5. **Decide on headroom over your target `max_num_seqs`:**
   - `1.0×` — bare minimum. Avoid; KV pressure under any load spike.
   - `1.5×` — minimum acceptable. Some buffer for irregular usage.
   - `2.0×` — recommended. Comfortable multi-step loop budget.
   - `3.0×` — paranoid. Useful if you're memory-rich and want to
     eliminate KV pressure as a failure mode.
6. **Solve for `gpu_memory_utilization`:**
   `total_budget = fixed_costs + (target_num_seqs × headroom × per_slot_KV)`
   `gpu_memory_utilization = total_budget / total_unified_memory`
7. **Empirical sanity check:** restart vLLM with the new value and
   confirm the new `Maximum concurrency` log line is at or above
   `target_num_seqs × headroom`. If lower, bump
   `gpu_memory_utilization` up by `0.03–0.05` and retry.

#### Worked example: Nemotron Omni at 256K / 16-seq / Thor (2026-04-30)

Anchor measurement at `gpu_memory_utilization=0.65`:

```
Available KV cache memory: 37.32 GiB
Maximum concurrency for 262,144 tokens per request: 47.88x
```

```
Total budget   = 0.65 × 128 GB ≈ 83.2 GB
Fixed costs    = 83.2 − 37.3   ≈ 45.9 GB
Per-slot KV    = 37.3 / 47.88  ≈ 0.78 GB at 256K full context
```

Targeting `max_num_seqs=16` with `2.0× headroom` (32× supportable):

```
KV pool needed = 32 × 0.78 GB        ≈ 25.0 GB
Total budget   = 45.9 + 25.0         ≈ 71 GB
gpu_mem_util   = 71 / 128            ≈ 0.55
```

Round down to **`0.50`** for a slightly tighter fit (~24× supportable
concurrency at 256K). Marco's call 2026-04-30: the small headroom
margin is acceptable — we'll bump if measurements show pressure, drop
further if `max_num_seqs` is later reduced.

The hybrid Mamba-Transformer architecture is what makes the per-slot
KV cost so low (Mamba layers have fixed-size state per sequence; only
attention layers scale with context). Pure-Transformer models will
have a much higher per-slot KV cost at the same context length;
recalculate per profile.

---

## Pending profile updates

These existing profiles are calibrated for non-ManyForge-pipeline
workloads. Revisit when ManyForge live-test data arrives:

- **`qwen3.6-35b-a3b-nvfp4-tq-mtp-manyforge`** — currently
  `max_num_seqs=3`, 64K context. May need bump if ManyForge BT shapes
  produce >3 parallel `query_model` nodes, or if multi-step recovery
  loops accumulate beyond 64K.
- **`cosmos-reason2-8b`** — currently `max_num_seqs=3`, 64K context.
  Designated Orin profile per `MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md`;
  Orin's tighter memory envelope (64 GB total minus Isaac ROS) likely
  forces its own calibration distinct from the Thor numbers. Consider
  a future NVFP4 quantization of Cosmos-8B (~halves weight footprint)
  if Orin memory pressure is real.

---

## Hardware-specific notes

- **Thor (128 GB unified memory).** Generous; can support
  `max_num_seqs=16` at 256K for hybrid models without forcing
  concurrency drops. Use the Worked Example above as the template.
- **Orin AGX (64 GB unified memory).** Constrained; system + Isaac
  ROS likely consume 20–30 GB, leaving ~35–40 GB for the LLM stack.
  ManyForge profiles intended for Orin deployment will need their own
  calibration (smaller model, lower `max_num_seqs`, possibly reduced
  `max_model_len`). Don't reuse Thor numbers.

---

## What this doc does *not* cover

- Tool-call parser selection (`hermes` vs `qwen3_xml` vs `qwen3_coder`)
  — see auto-memory `reference_vllm_tool_parsers` and per-profile
  comments in `lib/launch.sh`.
- Speculative decoding (DFlash / MTP) tuning — see
  `DFLASH-INVESTIGATION.md` and the per-profile comments in
  `lib/config.sh` / `lib/launch.sh`.
- Quality benchmark methodology — see `agentic-bench/README.md`.
- Profile selection for a given deployment target — see
  `MANYFORGE-ASSISTANT-DEPLOYMENT-PLAN.md`.

This doc is strictly about sizing the three knobs above for the
ManyForge bridge / agentic-pipeline workload.
