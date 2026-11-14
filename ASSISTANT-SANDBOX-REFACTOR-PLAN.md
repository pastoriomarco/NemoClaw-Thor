# ManyForge Assistant: Sandboxed Component Plan

Status: **Draft** (design only; nothing executed).
- Rev. 2, 2026-11-08. Rev. 1 was written 2026-11-05.
- Rev. 2 covers: the assistant as a v2 component, the model server in a hardened container, portability, bounded AI motion authority, and a first bring-up on ManyForge 0.2.1 as it ships.

Scope:
- the ManyForge AI assistant (bridge plus model server) and its containment;
- model serving on Jetson AGX Thor, AGX Orin (both JetPack 7.2.1) and AMD64 with an NVIDIA dGPU;
- what this repo becomes.

## 1. Goals and constraints

1. **Safety and containment.** The model may reach and do only what its job needs:
   - help build the Program tree;
   - explain ManyForge;
   - answer runtime model queries (`assistant.query_model`);
   - later, act through VLA policy nodes.
2. **Simple setup.** The assistant is a component like any other: added in the Deployment, prepared and enabled from the GUI. Machine setup is one installer command at most.
3. **Portable mechanisms only.** Docker, the NVIDIA Container Toolkit, Linux cgroups and OpenShell (Linux x86_64 and aarch64).
   - Nothing depends on a specific board.
   - No MIG (Thor-only, and recorded as a serving non-goal in `serving/docs/THOR-ACCELERATOR-ROI.md`).
   - No mode gating as a requirement.
4. **AI may move the robot.** VLAs are planned, so motion authority is **bounded**, not forbidden (§5.4).
5. **Less machinery.** Drop NemoClaw, OpenClaw and Hermes.

## 2. Current state

### ManyForge 0.2.0 (released 2026-11-03) and the 0.2.1 integration tree

0.2.1 was released on 2026-11-10 (manyforge commit `34bb5f4a`, `CHANGELOG.md` "0.2.1"); the integration tree was merged into `src/manyforge` before the release.

The assistant's structure is the same in 0.2.0 and 0.2.1:
- **Bridge.** `manyforge_assistant_bridge`: FastAPI on 127.0.0.1:8100, owns the agent loop (16 turns, 180 s).
  - `scripts/dev-workbench.py` always spawns it, inside the workbench container (`--network host`, repo mounted read-write, full environment). The operator orchestrator spawns it too, as a `ProcessSpec`.
  - Composer gets `--assistant-bridge-url`. Its "configured" flag only means the URL was passed: the bridge's `/health` always says ready, and nothing checks liveness.
- **Tools.** 24 tools: 6 reads and 18 Draft edits. Schemas are generated from the node catalog. The bridge validates arguments, then Composer validates them again. Edits land only in the Draft; the operator Applies.
- **`ai_model` component** (`ai_model_v0`: `base_url`, `model`, `api_key_env`):
  - real lifecycle;
  - one `endpoint` stream fed by `ai_model_probe` (GET `/models` every 2 s), where staleness is shown but is not a fault;
  - the probe process exiting is a fault (`ai_model_exited`).
- **Runtime queries.** `assistant.query_model` is called by the program runner directly, in JSON mode or with one forced tool. Outputs are typed and validated.
- **Security gaps that are not about sandboxing:**
  - Composer routes have no auth, and Composer binds 0.0.0.0 in operator-runtime mode;
  - the opt-in `--mcp-http` adapter exposes control commands and program Apply;
  - the ROS graph is not limited to localhost by default.

0.2.1 changes relevant to the assistant (diff against 0.2.0):
- **Roles deletion (OD-28).** `query_model` frame references are now `{component_id, frame_id}` instead of target and frame roles, and the tool registry is regenerated.
- **`program_validate` route** now runs real validation (it is not in the assistant's tool list).
- **Removed:** `setup-direct.sh`, the OpenClaw smoke script and the i9 snapshot probe.
- **Renamed:** H0 read operations → `ASSISTANT_READ_OPERATIONS`.
- **Unchanged:** bridge spawning and the `ai_model` component.

### This repo (NemoClaw-Thor, last commit 2026-09-16)

- **Pins:** NemoClaw v0.0.73 / OpenShell 0.0.71, from July.
- **Sandbox:** it only ever held OpenClaw and Hermes; the direct bridge never ran in it.
- **Policies:** only the egress presets in `manyforge/policies/` are enforced. `setup/policies/static/*` is never applied, and `SECURITY.md` claims protections the code does not implement.
- **Serving** (`serving/`) is self-contained (Docker plus an HF token). Its defaults are open:
  - model server and proxy bind 0.0.0.0, with no API key;
  - both default to port 8000;
  - most vLLM profiles pass `--trust-remote-code`.

## 3. Upstream status (verified 2026-11-05; unchanged on 2026-11-08)

| Component | Latest | Notes |
|---|---|---|
| **NemoClaw** | v0.0.141 (2026-11-03); v1.0.0-rc1 prerelease | Alpha; operation telemetry on by default. Manages NVIDIA's agent packages only; no custom-agent registry. Pins **OpenShell 0.0.116** (2026-08-28), before OpenShell's 0.1.0 break. Jetson: the Thor express installer is an open epic (#8500); the IGX Orin GPU proof still fails (#12806). |
| **OpenShell** | v0.1.7 (2026-11-03) | 0.1.0 (2026-09-25) was breaking: removed `inference.local` and managed inference; `--from` no longer builds images. 0.1.6 requires gateway auth. Features: custom image as a long-lived service (`--from`, `--restart-policy`, `--expose` / `forward`); egress allowed per host, port, method and path, everything else denied; credential placeholders; offline policy prover; Python SDK (`SandboxClient`: create, wait_ready, health, delete). Requires Linux ≥ 6.2 (Landlock ABI 3), which all target machines meet. |
| **OpenClaw**, **Hermes Agent** | v2026.11.1 / v0.22.7 | Not needed. |

**Conclusion:** drop NemoClaw, OpenClaw and Hermes, and use OpenShell directly, pinned, for the bridge only (§5).

## 4. Threat model

| Component | What it runs | Containment |
|---|---|---|
| **Model server** (llama.cpp / vLLM; later VLA policy servers) | Third-party engine. vLLM can run Python from the model repo (`--trust-remote-code`), fetch media URLs, run built-in tool servers and load LoRA adapters at runtime. llama.cpp has a smaller surface (Jinja, GGUF). | Hardened container with no outbound network (§5.3) |
| **Bridge** | Our typed code; Draft-only tools | OpenShell sandbox (§5.2): contains code execution and keeps credentials out. Essential once the assistant gains wider tools. |
| **ManyForge APIs** (Composer HTTP, headless API, ROS graph) | Can command the robot | Local only (§5.1). A sandbox can't fix these, because it limits only what runs inside it. |
| **Robot controller API** (e.g. xArm SDK and Modbus ports) | Unauthenticated on its network | Dedicated link from the ManyForge machine (§5.1) |

Physical access to the box gives access to everything; that is by design.

## 5. Target design

### 5.1 Host containment (portable)

- **Local-only APIs.** Composer, the headless API and every internal service bind to loopback.
  - Ports published by Docker use `127.0.0.1:` (Docker's iptables rules bypass `ufw`).
  - The host firewall denies inbound connections by default.
  - Remote GUI access goes through an SSH or VPN tunnel; later, through one authenticated TLS entry point if several remote users need it.
- **ROS graph local only:** `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` (or shared-memory transport) in all ROS containers.
- **Robot controller** on a dedicated link or VLAN, reachable only from the ManyForge machine.
- **Performance:**
  - cgroup CPU sets, priorities and memory limits keep AI and perception containers from starving or out-of-memory-killing the control path;
  - a per-machine acceptance test measures control jitter, perception latency, VLA action rate and LLM tok/s under combined load.

### 5.2 The assistant as a v2 component (proposed: OD-95 … OD-99; renumbered 2026-11-10 from OD-60 … OD-64, which were already in use)

| Component | Kind | Adapter family | Prepare / Enable / Disable / Recover | Health |
|---|---|---|---|---|
| **AI model** (exists) | `ai_model` | `endpoint` (today). Later, `managed_server`, which owns the hardened model container (OD-100). | Lifecycle with the endpoint probe | `endpoint` stream; stale is shown, not a fault |
| **Assistant** (new) | `assistant` | `openshell` | **Prepare:** create the sandbox from the pinned bridge image, with policy and provider generated from the Deployment. **Enable:** start the bridge and the forward. **Disable:** stop both. **Recover:** recreate the sandbox. | `bridge` stream from a real `/health`. Sandbox or process exit → `fault_process_exited`. |

- **Deployment row:** `runtime_kind: assistant`, `model_component` (a reference to an `ai_model` component, the same mechanism as OD-28) and `assistant_profile`. Nothing else to fill in.
- **Generated policy** (OD-98), enforce mode, everything else denied. The adapter derives it from the Deployment:
  - Composer: POST on the bridge-tools callback route only;
  - headless API: GET on the read operations of `HeadlessReadClient`;
  - the referenced model's `base_url`: POST `/v1/chat/completions` (and GET `/v1/models`);
  - the model's `api_key_env` becomes the OpenShell provider credential, and the sandbox only sees a placeholder;
  - filesystem: read-only image, plus `/tmp` and a capped audit-log directory.

  After a model change, Apply recreates the sandbox like any other unit. The prover check runs in CI.
- **Process:** the adapter launches a small **sandbox host** under the execution host's supervised launch, the same pattern as `ai_model_probe` and `pointcloud_source`.
  - It drives OpenShell through the SDK or CLI and reports lifecycle, health and activity: `busy` during a turn, `idle` otherwise, as the plan already says for AI models.
  - The core sees a component with declared capabilities, never OpenShell (rule RP-1b).
- **Composer:**
  - the AI tab is enabled from the assistant component's state, and the bridge URL comes from the component's published endpoint (OD-97);
  - the workbench and the operator orchestrator stop spawning the bridge, and the fixed port 8100 goes away;
  - this implements the publication TODO decision "the chat bridge runs only when the Deployment has an active AI component".
- **Core rules:** an assistant fault latches the cell like any component fault (OD-96, under the "component fault is catastrophic" rule). The existing `ai_model_exited` already does this. R2, R3, R4 and R5 are unchanged.
- **Packaging** (OD-99): the new `manyforge-assistant` repo ships a normal component package:
  - `component.yaml` (`manyforge.component_package.v1`);
  - the sandbox host and adapter registration;
  - the bridge image build and the policy template;
  - the OpenShell install step.

  Step 1 builds the image from ManyForge's `manyforge_assistant_bridge`. Step 2 moves the bridge code into the new repo.

### 5.3 Model servers (portable, no OpenShell for now)

- **Default backend:** llama.cpp + GGUF on every machine. The model file contains no executable code.
- **vLLM only for the advanced recipes:**
  - media restricted to data URLs;
  - no tool server, no runtime LoRA loading;
  - `--trust-remote-code` only where the model needs it.
- **Hardened container:**
  - its own Docker network with no outbound access and no route to Composer;
  - port published on loopback only, with an API key;
  - read-only weights, non-root, no capabilities;
  - no HF token while serving: weights are downloaded in a separate one-off step;
  - cgroup limits.

  The exact Docker network shape (an internal network vs. a `DOCKER-USER` egress drop) is settled in the spike.
- **Clients:** the bridge sandbox (through its policy) and the program runner (`query_model`). Future VLA executors use the same pattern.
- **Later:** an `ai_model` adapter `managed_server` that owns this container, so the model is also prepared and enabled from the GUI (OD-100).

#### 5.3.1 First reference recipe: Nemotron 3.5 Lightning on vLLM v0.33.0 (Thor, 2026-11-08)

Brought up on Thor and serving: fast in interactive use (Cline), with no image builds. This is the reference for the performance comparison with Qwen3.8-27B (next).

- **Image:** the official `vllm/vllm-openai:v0.33.0`, pinned by digest. It was pulled on Thor and checked:
  - CUDA 13.0 on driver 595.78 (JetPack 7.2.1);
  - the GPU is seen as "NVIDIA Thor", compute capability (11, 0);
  - native kernels compiled for `sm_87` (Orin) and `sm_110` (Thor);
  - FP4 CUTLASS support reported for Thor: `cutlass_scaled_mm_supports_fp4(110) = True`.
- **Weights:** `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (snapshot `bee7596…`) and the drafter `…-NVFP4-DSpark` (snapshot `6d127b5…`), 22 GB in `~/thor-hf-cache`, served offline.
- **Test-time exposure:** listens on Thor's LAN address, without an API key, so it can be used from the laptop. This is **not** the production containment of §5.1 and §5.3, which require loopback, an API key and no outbound network.

```bash
docker run --rm -it --name lightning-happy --runtime nvidia --network host --ipc=host \
  -e HF_HUB_OFFLINE=1 \
  -v $HOME/thor-hf-cache:/root/.cache/huggingface \
  -v $HOME/vllm-cache:/root/.cache/vllm \
  -v $HOME/flashinfer-cache:/root/.cache/flashinfer \
  vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --served-model-name nemotron35-lightning \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.30 --max-model-len 262144 \
  --kv-cache-dtype bfloat16 \
  --enable-prefix-caching --max-num-batched-tokens 16384 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --mamba-backend flashinfer --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5 \
  --disable-flashinfer-autotune \
  --speculative-config '{"method":"dspark","model":"nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark","num_speculative_tokens":5}'
```

Findings from the bring-up:
- **Kernels.** The checkpoint is a precision mix:
  - experts (5,888), shared experts and `lm_head` are `W4A16_NVFP4` (weight-only FP4), so vLLM uses **Marlin**, not the FP4 tensor cores (`tcgen05`);
  - the Mamba input/output projections (46) are FP8 (FlashInfer FP8 GEMM);
  - attention uses FlashInfer.

  No W4A4 Lightning checkpoint exists on Hugging Face. Using `tcgen05` FP4 would need our own ModelOpt W4A4 quantization (Phase 4 option). vLLM 0.33 has no W4A8 kernel for Thor: CUTLASS W4A8 is Hopper-only, and `w4a8_nvfp4` exists only in the SM12x b12x backend.
- **KV cache:** `--kv-cache-dtype auto` follows the checkpoint's FP8 KV, so BF16 must be set explicitly. Lightning's KV cache is small (few attention layers): about 1.6M tokens of FP8 KV fit at a 0.30 memory share, about half that in BF16, so 256K context is affordable.
- **Autotune:** FlashInfer's autotune stalled for more than 10 minutes on `fp8_gemm` (single-core CPU-bound), so it is disabled here. It matters little for Lightning, where Marlin carries most of the bytes.
  - Finer controls: `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=fp8_gemm`, or a smaller `--max-num-batched-tokens` to cap the buckets.
  - Keep `~/flashinfer-cache` mounted so JIT modules and tuned tactics persist across restarts.
- **Startup:** the first start took about 15–20 minutes, from single-threaded preparation before the shards load, Mamba warmup, compilation and FlashInfer JIT. Later starts reuse the host caches.
- **Not applicable here:** `--mamba-cache-mode` (no effect in 0.33) and `draft_lm_head_quantization` (EAGLE and MTP only, so not DSpark).
- **Next reference:** `nvidia/Qwen3.8-27B-NVFP4`, which is W4A4 NVFP4 on its MLP and `lm_head`, FP8 on attention and linear-attention layers, with no KV quantization, plus a built-in MTP head. This is the `tcgen05` FP4 test. Start with built-in MTP and `draft_lm_head_quantization`; DSpark comes second. Autotune runs once with the cache mounted.
- **OpenShell GPU sandboxes for model servers** are deferred. Background in §7.

### 5.4 AI motion authority (VLAs): bounded

- A VLA is a **programmer-placed tree node** backed by a policy-server component (model). It has an envelope:
  - workspace bounds;
  - speed and force caps;
  - a timeout;
  - exit and failure behaviour.

  It acts only while the program is inside that node.
- **Its actions go through the same execution path as all motion:**
  - joint, workspace and collision checks;
  - the vendor TCP speed limit;
  - supervision, and Stop.

  Leaving the envelope means a Stop. The client limits the action horizon, so a slow or stale policy can't run ahead.
- **The integrator's safety controller stays the safety layer;** ManyForge's limits are operational.
- **The assistant** keeps its Draft + Apply behaviour. Runtime `query_model` keeps typed outputs and failure policies.

## 6. Repositories

| ManyForge keeps | New `manyforge-assistant` repo | NemoClaw-Thor (this repo) |
|---|---|---|
| Tool-schema generator, Composer callback API, Draft/Apply, `query_model`, the `ai_model` kind | Assistant component package (§5.2): sandbox host, image, policy template, OpenShell installer step | `serving/` as the "advanced serving" add-on (Qwen3.8-27B, Flash-Next, DS4, accelerator ROI plan) |
| **New:** the `assistant` kind and its state in the GUI; Composer reads the bridge endpoint from the component; workbench and operator no longer spawn the bridge | Slim llama.cpp model-server profiles and the hardened container recipe, for Thor, Orin and AMD64 | The `manyforge/` subtree retires (§9) |
| Local-only binding, ROS discovery limited to localhost | Smoke corpus and runner, ported to the direct bridge; later the bridge code itself | |

**Contract between repos:** the catalog snapshot and the callback API (ManyForge assistant specs 480/485). The bridge checks the catalog hash on every request, and the new repo declares a compatible ManyForge version range.

## 7. Background: OpenShell GPU on Jetson (deferred)

This matters only if a model server is ever moved into an OpenShell sandbox.

How it works: the Docker driver requests the GPU through CDI, then OpenShell runs the workload as non-root under Landlock and seccomp. Plain Docker CDI works for non-root on Thor and on IGX Orin; OpenShell's confinement is what breaks it.

- **Thor** (NemoClaw evidence, OpenShell #2846, 2026-09-15). `cuInit` fails with 801 without `/sys` access and succeeds with `/sys` read-only. Also needed:
  - a raised policy path limit (301 paths against a limit of 256);
  - `/usr/sbin` on `PATH`.

  The issue tracking the minimal sysfs paths (#3348) was **closed as stale, not fixed**.
- **Orin:** only IGX Orin on JetPack 6.2 has been tested, and it still fails (NemoClaw #12806). AGX Orin on JetPack 7.2.1 is untested.
- **In flight:**
  - OpenShell PR #4204 (derives sandbox access from CDI specs; draft, stacked on #4461 and #4529; untested on Orin or Spark hardware; does not cover sysfs);
  - PR #2846 (`PATH`);
  - #4837 (a non-CDI route; open).
- **Our own patch path if ever needed:** a narrow sysfs grant traced with `strace`, the path-limit change, and #4204 plus #2846. Every patch would also be proposed upstream.

## 8. Phases

Every build, test or live run on Thor, Orin or the laptop needs maintainer approval before it starts.

### Phase A: assistant working on ManyForge 0.2.1 as it ships (no sandbox yet)

1. **Model on Thor.** Use the §5.3.1 Lightning recipe; it is the first reference. Point the Scenario's `ai_model` component at it (`model: nemotron35-lightning`, `base_url` set to the server).

   Fallback: the earlier Gemma 4 12B QAT llama.cpp profile (`serving/start-model.sh gemma4-12b-it-gguf`, `THOR_RESTART_PROXY=0`), which serves the `gemma4-12b-it-gguf` name the example Scenarios ship with.
2. **Workbench.** Start the r23 workbench. It spawns the bridge itself.
3. **Scenario.** Load `mock_ur10e_assistant_scene_authoring`, whose `ai_model` points at `http://127.0.0.1:8000/v1`, then Prepare and Enable the AI model component.
4. **Smokes.** Run `scripts/smoke-assistant-provider-bridge.py`, then `scripts/smoke-assistant-e2e.py --confirm-draft-only`.
5. **GUI.** The maintainer runs a first pass in the AI tab.
6. **Runtime queries.** Re-check the vision Scenarios (`mock_ur5_vision_*`, `mock_ur5_semantic_object_pick`) against 0.2.1's new `{component_id, frame_id}` frame references in `query_model`.

**Exit:** the smokes pass, the GUI pass is OK and the results are recorded. This becomes the baseline for later phases.

### Phase 0: OpenShell spike on Thor (OpenShell v0.1.7; no repo changes)

1. Install OpenShell, start the gateway with auth, record its footprint, and confirm the kernel probes pass.
2. Run the bridge image in a sandbox (`--from`, `forward`): Composer → bridge works, including streaming responses. Measure the added latency.
3. **Policy:** only the allowed paths succeed. Settle whether the proxy reaches host-loopback services through `host.openshell.internal`, or whether Composer and the headless API must listen on the Docker bridge address.
4. **Credentials:** the provider placeholder is substituted into the `Authorization` header.
5. **Reboot:** the sandbox and the forward come back.
6. **Dev loop:** how long an image rebuild plus sandbox recreate takes.
7. **Model container:** the hardened model-server Docker network shape (§5.3).

**Exit:** a spike report, which then feeds the contracts.

### Phase 1: contracts (ManyForge, under its approval rules)

Write the contracts for:
- the `assistant` kind and its states, streams and activity;
- the `model_component` reference;
- the generated policy;
- how Composer finds the bridge and gates the AI tab;
- removing bridge spawning;
- local-only binding and ROS discovery limited to localhost.

### Phase 2: implementation

- **New repo:** component package, sandbox host, image, policy template and installer.
- **ManyForge:** the kind, Composer, workbench and operator changes, and the removals from §9.
- **Model:** the hardened model-server recipe.

### Phase 3: quality

- Port the smoke corpus to the direct bridge, re-baseline Gemma 4 12B QAT and add constrained tool decoding.
- Evaluate model candidates:
  - **Nemotron 3.5 Bolt E4B:** about 2–3× faster, weaker on composite edits;
  - **Bolt 74B-A9B** via llama.cpp;
  - **Qwen3.8-Flash-Next.**
- Measure the real tool payload before choosing a model with a 64K context: each large tree tool's schema is about 56K characters.

### Phase 4: extensions

- `managed_server` `ai_model` adapter (OD-100).
- A read-only `docs_read` tool.
- Optional LiquidAI d1-3B decision model for `query_model`. It is user-selected only, because its LFM 1.0 license removes commercial rights above $10M annual revenue.
- VLA policy-server components and envelope nodes (§5.4).

### Phase 5: cleanup

- Retire this repo's `manyforge/` subtree.
- Rewrite `SECURITY.md` so it describes only what is enforced.
- Update `PLANS_INDEX.md` and `INDEX.md`.

## 9. Removal proposals (maintainer decides)

**ManyForge:**
- `provider_kind: nemoclaw`, which is identical to `direct`;
- the profiles pointing at port 8200, where nothing listens;
- `examples/model_endpoint_profiles.yaml`, which nothing reads;
- OpenClaw and Hermes stubs: `setup-hermes.sh`, the OpenClaw plugin builder, `manyforge-mcp-bridge.py`, `scripts/hermes/*`, the policy-fragment templates;
- the `openshell` upload in `setup-worker.sh`, and `dev-agent-session-start.sh`;
- the non-activatable `agent-skills` copy;
- the `--mcp-http` adapter and the `mcp` pin in four Dockerfiles, whose consumers were the retired MCP lanes;
- in specs: the NemoClaw/OpenClaw sections of the backend contract, the ADR-0021 carve-out (to be replaced by §5.2), and the stale "implementation pending" headers.

Some of these overlap the 0.2.1 "leftover stub modules" decision in the night report.

**This repo:**
- `manyforge/openclaw_assistant_bridge/`, `manyforge/lanes/hermes/`, `manyforge/assistant_session/`, `manyforge/policies/`, the NemoClaw `setup/` scripts, `archive/openclaw-plugin-attempt-*`;
- `manyforge/common/`, after moving out what is still used, since it imports the OpenClaw adapter;
- `vllm-proxy.py`, unless the advanced recipes still want its logging.

The smoke corpus moves to the new repo, and the evidence folders stay as history.

## 10. Open decisions

| Code | Decision | Recommendation |
|---|---|---|
| **OD-95** | New component kind `assistant`, with adapter family `openshell` | Yes |
| **OD-96** | An assistant fault latches the cell like any component | Yes; revisit only with live evidence |
| **OD-97** | Composer gates the AI tab on the component state and gets the bridge endpoint from the component; workbench and operator stop spawning the bridge | Yes |
| **OD-98** | Sandbox policy and provider generated from the Deployment | Yes |
| **OD-99** | Packaging: component package in `manyforge-assistant`; bridge code moves in a second step | Yes |
| **OD-100** | `ai_model` adapter `managed_server` owning the hardened model container | Phase 4 |
| — | Removals in §9 | Per item |
| — | This repo afterwards: `serving/` as advanced add-on; the rest retires | Yes |

The codes are proposed. They get recorded in the decision table once the maintainer confirms the list.

## 11. Risks

- **OpenShell churn:** a breaking 0.1.0 and weekly releases. Pin it, and use the Phase 0 checks as a regression suite on upgrade.
- **Proxy and forward latency** on every model call and tool callback; measured in Phase 0.
- **An assistant fault latching the cell** (OD-96) can stop production work until Recover. This is accepted by rule; revisit with live evidence.
- **Context size:** the tool payload may exceed small models' context windows.

## 12. Sources

**Upstream projects:**
- NemoClaw: <https://github.com/NVIDIA/NemoClaw>
- OpenShell: <https://github.com/NVIDIA/OpenShell>
- OpenShell docs: <https://docs.nvidia.com/openshell/latest/>
- OpenShell Python SDK: <https://docs.nvidia.com/openshell/v0.1.7/sdk/python.md>

**OpenShell issues and PRs:**
- #3348: <https://github.com/NVIDIA/OpenShell/issues/3348>
- #4837: <https://github.com/NVIDIA/OpenShell/issues/4837>
- #4204: <https://github.com/NVIDIA/OpenShell/pull/4204>
- #2846: <https://github.com/NVIDIA/OpenShell/pull/2846>

**NemoClaw issues:**
- #12806: <https://github.com/NVIDIA/NemoClaw/issues/12806>
- #8500: <https://github.com/NVIDIA/NemoClaw/issues/8500>

**Policy-server references:**
- Isaac GR00T: <https://github.com/NVIDIA/Isaac-GR00T>
- openpi: <https://github.com/Physical-Intelligence/openpi>

**Models:**
- LiquidAI d1-3B: <https://huggingface.co/LiquidAI/d1-3B>
- Nemotron 3.5 Bolt E4B: <https://research.nvidia.com/labs/nemotron/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB/>
