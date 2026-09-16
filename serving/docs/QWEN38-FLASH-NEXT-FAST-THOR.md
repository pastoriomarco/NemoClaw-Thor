# Qwen3.8-Flash-Next: recommended Thor recipe

**Recommended fast default as of 2026-09-16:** this recipe serves a linked
FP8-hybrid form of the pinned RadixArk checkpoint. Routed experts remain native
NVFP4; only 5.42 GiB of frequently read dense GDN, QSA and shared-expert BF16
weights become 2.71 GiB of blockwise FP8 E4M3. The vision tower, embedded MTP,
FP8 PLE table and **BF16 KV cache** remain. The default is **MTP=3, 0.90 memory,
BF16 KV, 256K context and three scheduler slots**. Forum-style C1 throughput
improved from **34.8 to 46.7 tok/s (+34%)** in the first two-run comparison.
This is a measured recommendation, not a claim of the fastest possible engine
or guaranteed speed at every context/concurrency. The original unmodified
checkpoint remains the quality-verified fallback.

Status (2026-09-14): the FP8 hybrid completed the full 88-scenario Tool Eval at
**94/100 overall and 100/100 on Hard Mode**, matching the original-side-layer
configuration's 94/100 overall while completing faster. The `fast-v3` recipe
is the `fast-v2` inference path plus merged vLLM PR #52805. Both forum-style
`sgbench` results and the controlled quality comparison are recorded below.

## What changes

Recommended image: `nemoclaw-thor/qwen38-flash-next-vllm:sm110-fast-v3`.

- Fused GDN **decode** CUDA kernel compiled for SM110a from upstream source
  `082cf021b7ef96e4819e386846ea34e5ef21c655`, associated with
  [merged vLLM PR #53835](https://github.com/vllm-project/vllm/pull/53835).
  It registers in a separate `thor_gdn` namespace; only that Python dispatch
  is redirected. Other native operators and the working preview loader remain.
- PLE mappings use `MADV_RANDOM`, avoiding useless disk read-ahead, and
  `VLLM_PLE_MMAP_FAST_ROWS=0` selects existing parallel reads for decode too.
  Similar mapping advice is documented in
  [Saren-Arterius's Spark recipe](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound).

The existing native NVFP4 routed experts, FP8 PLE table, embedded MTP,
deterministic QSA top-k, vision tower and prefix-cache fixes remain unchanged.
The default preparation step converts only 300 dense side-layer matrices from
BF16 to blockwise FP8. It requires no calibration corpus and does not modify the
original snapshot. GDN prefill and QSA still use their supported Triton
implementations; this does not remove every Triton kernel or add FP4 KV support.

The `sm110-fast-v2` portion adds:

- A 47,149-token coding-oriented MTP draft vocabulary. Only the speculative
  draft head is reduced; target-model verification still uses the complete
  248,320-token vocabulary, so accepted output remains target-verified. The
  checked-in token-ID asset was generated deterministically from the local
  768-record coding calibration corpus and the checkpoint tokenizer. Its SHA256
  is `36cd2455dadd4f8362c1137a6e4b54305adb5fd640fc7653cfb3815584177a37`.
- `--mamba-ssm-cache-dtype bfloat16`, reducing recurrent GDN state bandwidth
  versus the checkpoint's FP32 state. This is separate from BF16 KV cache.
- The latest audited deterministic QSA top-k source at commit
  `e0ef69d4f5575dad00d34e05479eaf4c6547bace`, compiled for SM110a.
- `POSIX_FADV_WILLNEED` hints for large PLE gathers (512 rows or more by
  default), while leaving decode-sized reads on the already verified mmap path.

All four changes are reversible at launch. Reduced-vocabulary drafting may
lower MTP acceptance when a needed draft token is outside the selected set;
target verification protects correctness but not throughput. BF16 state needs
long-run behavioral validation. PLE advice is advisory and falls back to normal
mmap faults if unsupported.

`sm110-fast-v3` additionally applies
[merged vLLM PR #52805](https://github.com/vllm-project/vllm/pull/52805). It
stops speculative token batches once XGrammar accepts a stop token and resets
the wrapper's termination state correctly. This removes one known source of
post-stop newline warnings. It does **not** claim to solve the separate
[token-271 speculative/JSON grammar issue](https://github.com/vllm-project/vllm/issues/53181),
which can still occur on newer upstream builds.

## Shortest working path

The user performs four actions from the NemoClaw-Thor repository root. Change
`HF_CACHE` first only if the checkpoint should live somewhere other than the
default fast NVMe path, `$HOME/thor-hf-cache`.

Build the complete pinned SM110 image stack:

```bash
./serving/build-qwen38-flash-next-fast.sh
```

Download the exact checkpoint. This command is a no-op when it is already in
the cache:

```bash
./serving/download-qwen38-flash-next.sh
```

Create the linked FP8-hybrid snapshot. Only four shards are rewritten, adding
about 13 GiB on disk; unchanged files remain links to the base checkpoint:

```bash
./serving/prepare-qwen38-flash-next-fp8-hybrid.sh
```

Start vLLM in the foreground and watch loading and serving logs in the same
terminal:

```bash
./serving/start-qwen38-flash-next-fast.sh
```

That is the complete normal procedure. `start-qwen38-flash-next-fast.sh` now
defaults to the prepared FP8 hybrid. Docker and the NVIDIA container runtime
must already work on the Thor host; the build needs network access, and a new
machine needs enough fast-NVMe space and time for the approximately 126-GiB
base checkpoint plus the 13-GiB delta. No host Python, CUDA toolkit, separate
drafter, or calibration corpus is required. The conversion runs inside Docker.
The 47.7-GiB PLE portion stays file-backed; hybrid model loading used 76.75 GiB.
A three-slot launch at 0.91 measured 24.19 GiB / 884444 BF16-KV tokens (3.37x
full 256K contexts); the 0.90 default leaves additional host-memory headroom.

## Detailed reproduction and overrides

### 1. Prerequisites and persistent storage

Validated on Jetson AGX Thor (128 GB unified memory), JetPack 7.1,
with Docker and the NVIDIA container runtime already configured. Use fast
local NVMe for the checkpoint: the large PLE table is read from disk during
inference. Host Python/CUDA package installation is not needed for these builds.
Stop other GPU model servers before loading this 0.90-memory profile.

Run the commands below from the NemoClaw-Thor repository root. Change these
paths before starting if desired; keep the exports in the same shell:

```bash
export FLASHNEXT_SOURCE="$HOME/thor-qwen38-flash-next-vllm/source"
export HF_CACHE="$HOME/thor-hf-cache"
export VLLM_CACHE="$HOME/thor-vllm-cache"
export TORCH_CACHE="$HOME/thor-torch-cache"
export FLASHINFER_CACHE="$HOME/thor-flashinfer-cache"
export FLASHNEXT_BASE_REV="7b719225242aacd3dbd3f9407468c2ee9a9d2594"
export FLASHNEXT_HYBRID_REV="${FLASHNEXT_BASE_REV}-fp8hybrid"
mkdir -p "$HF_CACHE" "$VLLM_CACHE" "$TORCH_CACHE" "$FLASHINFER_CACHE"
```

### 2. Build the working image

The build helper above performs all four builds and pins the upstream source.
The expanded commands below are retained for troubleshooting and auditing.
These are local image tags, not published images to pull. Building requires
network access for the base image and source files but does not download or
modify model weights.

Clone once (skip this command if this checkout already exists):

```bash
git clone https://github.com/blazux/qwen3.8-Flash-DGX.git "$FLASHNEXT_SOURCE"
```

Use a clean upstream checkout; preserve any local edits before switching it:

```bash
git -C "$FLASHNEXT_SOURCE" checkout 4b723de2e2c465d866738b57ae64bde6e8c07744
```

```bash
docker build --progress=plain --build-arg DET_ARCH=110a \
  -t nemoclaw-thor/qwen38-flash-next-vllm:sm110 "$FLASHNEXT_SOURCE"
```

```bash
docker build --progress=plain \
  -f serving/docker/Dockerfile.qwen38-flash-next-sm110 \
  -t nemoclaw-thor/qwen38-flash-next-vllm:sm110-flashinfer-moe .
```

```bash
docker build --progress=plain \
  -f serving/docker/Dockerfile.qwen38-flash-next-gdn-sm110 \
  -t nemoclaw-thor/qwen38-flash-next-vllm:sm110-gdn-cuda .
```

Add the reduced draft vocabulary, refreshed top-k, PLE prefetch and XGrammar
termination-fix layer:

```bash
docker build --progress=plain \
  -f serving/docker/Dockerfile.qwen38-flash-next-fast-v3 \
  -t nemoclaw-thor/qwen38-flash-next-vllm:sm110-fast-v3 .
```

The `.npy` draft-vocabulary asset is checked into this repository, so building
the image does not require the calibration corpus. To regenerate it from the
prepared local corpus and the exact local tokenizer, run this from the
repository root (the result is deterministic):

```bash
docker run --rm --pull never --user "$(id -u):$(id -g)" \
  -v "$PWD:/repo" -v "$HF_CACHE:/hf:ro" --entrypoint python3 \
  nemoclaw-thor/qwen38-flash-next-vllm:sm110-gdn-cuda \
  /repo/serving/docker/patches/flash-next-fast-v2/build_draft_vocab.py \
  "/hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/$FLASHNEXT_BASE_REV" \
  /repo/serving/docker/assets/qwen38-flash-next/draft_vocab_code_47149.npy \
  --size 47149 \
  --corpus /hf/modelopt/calibration/qwen38-coding/coding-source-768.messages.jsonl
```

### 3. Download the base and prepare the FP8 hybrid

The required base is **RadixArk/Qwen3.8-Flash-Next-NVFP4**, at the exact
revision exported above. No separate drafter is needed: MTP is embedded. The
complete checkpoint, including the PLE table, must be present in
`$HF_CACHE/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/$FLASHNEXT_BASE_REV`.
When copying a Hugging Face cache to another machine, copy its referenced
`blobs` too, not just the snapshot symlinks.

**Skip the download on our Thor: its weights are already present.** On a new
machine, prefer the idempotent helper:

```bash
./serving/download-qwen38-flash-next.sh
```

Its expanded equivalent uses the usual cache and a fixed revision rather than
upstream latest:

```bash
docker run --rm --pull never --user "$(id -u):$(id -g)" \
  -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
  -e FLASHNEXT_BASE_REV="$FLASHNEXT_BASE_REV" --entrypoint python3 \
  nemoclaw-thor/qwen38-flash-next-vllm:sm110-fast-v3 \
  -c 'import os; from huggingface_hub import snapshot_download; snapshot_download("RadixArk/Qwen3.8-Flash-Next-NVFP4", revision=os.environ["FLASHNEXT_BASE_REV"])'
```

Prepare the recommended hybrid locally:

```bash
./serving/prepare-qwen38-flash-next-fp8-hybrid.sh
```

This copies the snapshot's relative links, rewrites only
`model-bf16-00001/00010/00011/00012.safetensors`, replaces the safetensors
index, and writes `.prepared` only after validation succeeds. The original
BF16-side-layer files remain in the base snapshot. The conversion uses direct
128x128 block scaling, not PTQ calibration or Hessian search; the coding
calibration corpus is therefore neither read nor required.

#### Publish only the changed files to Hugging Face

The hybrid is not a standalone 139-GiB upload. Publish it as a delta repository
containing the four real converted shards plus the modified index. First create
a Hugging Face write token. Enter it interactively so it is not placed in shell
history; the login is stored in the persistent Thor HF directory:

```bash
docker run --rm -it --pull never \
  -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
  --entrypoint hf nemoclaw-thor/qwen38-flash-next-vllm:sm110-fast-v3 \
  auth login
```

Verify the account, then choose the delta-repository name and upload:

```bash
docker run --rm --pull never \
  -v "$HF_CACHE:/hf:ro" -e HF_HOME=/hf \
  --entrypoint hf nemoclaw-thor/qwen38-flash-next-vllm:sm110-fast-v3 \
  auth whoami
```

```bash
export FLASHNEXT_DELTA_REPO="your-user/Qwen3.8-Flash-Next-NVFP4-FP8-Hybrid-Delta"
./serving/upload-qwen38-flash-next-fp8-delta.sh
```

The uploader creates the model repository, adds a model card and checksummed
manifest, and uploads only the five changed checkpoint files (about 13 GiB).
At 5 MB/s, allow roughly 45-60 minutes. It prints the final Hub commit; record
that hash and distribute it as `FLASHNEXT_DELTA_REV`, rather than relying on a
mutable `main`. Re-running Hugging Face's folder upload resumes/skips uploaded
content after interruption. The repository is private by default; set
`FLASHNEXT_DELTA_PRIVATE=0` only after confirming that the source-model and
RadixArk redistribution terms permit publishing derived weights. Do not upload
the `.bf16.bak` links or the other 201 base shards.

Another Thor can reconstruct the linked hybrid without quantizing locally:

```bash
./serving/download-qwen38-flash-next.sh
./serving/apply-qwen38-flash-next-fp8-delta.sh
```

The apply helper defaults to the uploaded
[`pastoriomarco/Qwen3.8-Flash-Next-NVFP4-FP8-Hybrid-Delta`](https://huggingface.co/pastoriomarco/Qwen3.8-Flash-Next-NVFP4-FP8-Hybrid-Delta)
repository at immutable revision
`358162ffaaeb4372c79843d9cd1f67d3654593da`. It checks every downloaded file
against the manifest, requires the exact base revision, and links all unchanged
files from that base. The Hub upload/download flow follows Hugging Face's current resumable
[`upload_folder`](https://huggingface.co/docs/huggingface_hub/main/guides/upload)
path.

### 4. Optional validation and foreground launch

Optional isolated numerical tests (while model serving is stopped):

```bash
docker run --rm --runtime nvidia --gpus all --entrypoint python3 \
  nemoclaw-thor/qwen38-flash-next-vllm:sm110-fast-v3 \
  -m pytest -q /opt/thor-gdn/test_fused_gdn_post_conv.py -k mtp
```

All **22 upstream tests passed on Thor**, including recurrent-state rollback,
ragged batches and head-ratio variants. These use numerical tolerances, not
bitwise equivalence of entire generated answers or a broad quality evaluation.

After each finished build/test or stopped serving attempt:

```bash
sudo sysctl -w vm.drop_caches=3
```

Cache clearing does not release live model allocations. Stop the serving
container first when reclaiming RAM; do not clear caches during benchmarks.

Single foreground launch command (recommended FP8 hybrid):

```bash
bash serving/start-qwen38-flash-next-fast.sh
```

Defaults: port **8050**, **0.90** GPU memory utilization, **BF16 KV**, **BF16
recurrent GDN state**, **262144** maximum context, **three scheduler slots**,
**MTP=3**, reduced coding draft vocabulary, and large-gather PLE prefetch.
With three scheduler slots, a measured launch at 0.91 allocated 884444 KV
tokens and advertised 3.37 full 256K contexts; 0.90 is expected to remain above
3.0 while leaving more unified-memory headroom. Confirm the exact allocation in
the startup log on each machine. Set
`FLASHNEXT_MAX_SEQS=4` explicitly for four shorter mixed requests.

The launcher uses the linked local hybrid snapshot
`7b719225242aacd3dbd3f9407468c2ee9a9d2594-fp8hybrid` in
`$HOME/thor-hf-cache`, plus the usual `$HOME/thor-{vllm,torch,flashinfer}-cache`
folders. It never refreshes weights upstream or replaces an already existing
container. The default container is `qwen38-flash-next-fast-v3-fp8hybrid` and
Docker removes it automatically when vLLM exits, while all weights and caches
remain in the host-mounted directories. Set `FLASHNEXT_NO_RM=1` only when a
stopped container and its Docker logs must be preserved.

To restore the original 94/100 Tool-Eval checkpoint path without deleting the
hybrid, choose the base revision, the original memory setting and a distinct
container name:

```bash
FLASHNEXT_MODEL_REV=7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
FLASHNEXT_CONTAINER=qwen38-flash-next-fast-v3-bf16-side \
FLASHNEXT_GPU_MEM=0.90 \
bash serving/start-qwen38-flash-next-fast.sh
```

Overrides: `HF_CACHE`, `VLLM_CACHE`, `TORCH_CACHE`, `FLASHINFER_CACHE`,
`FLASHNEXT_IMAGE`, `FLASHNEXT_CONTAINER`, `FLASHNEXT_MODEL_REV`, `FLASHNEXT_PORT`,
`FLASHNEXT_MTP`, `FLASHNEXT_GPU_MEM`, `FLASHNEXT_KV`, `FLASHNEXT_MAX_SEQS`,
`FLASHNEXT_CONTEXT`, `FLASHNEXT_SSM_CACHE`, `FLASHNEXT_DRAFT_VOCAB`.
`FLASHNEXT_DETACH=1` opts into background serving; `FLASHNEXT_NO_RM=1` preserves
the stopped container instead of deleting it automatically.
For controlled diagnostics, `FLASHNEXT_GDN=triton` restores the old GDN path;
`VLLM_PLE_MMAP_MADV_RANDOM=0 VLLM_PLE_MMAP_FAST_ROWS=512` restores old PLE I/O.
Set `FLASHNEXT_SSM_CACHE=float32` to restore the checkpoint state precision,
`FLASHNEXT_DRAFT_VOCAB=` to restore the full MTP draft head, or
`VLLM_PLE_MMAP_PREFETCH=0` to disable page advice.

### Serialized compilation-cache restart failure

One restart of the tested MTP=3 container failed with CUDA illegal instruction
(Xid 13), surfaced during sampler warmup after an AOT artifact was loaded. It
was **not an OOM**. The same image previously passed requests, four-way traffic
and 22 GDN numerical tests; all 22 tests passed again in a fresh process after
the failure. Cached compilation reuse is a suspect, not a proven root cause.

The launcher now sets `VLLM_DISABLE_COMPILE_CACHE=1` and explicitly keeps
`VLLM_USE_AOT_COMPILE=1`. This bypasses vLLM's serialized compilation cache
without selecting eager execution or disabling CUDA graphs. It adds compilation
work on each boot. Cache files and original containers are preserved. The
flagged relaunch completed startup, speed tests, four mixed-length requests,
tool-call parsing and repeated long-context retrieval. Decode performance was
retained. This is a verified workaround launch, not proof of the underlying
cause or a restart soak test; keep these flags in the saved serving command.

Client: base URL `http://<thor-ethernet-ip>:8050/v1`, model ID
`qwen3.8-flash-next`, context 262144, max output 16384. No server API key is set.

## Agentic quality result

The original-side-layer and FP8-hybrid snapshots completed tool-eval-bench
`v2.6.1.dev65+g6be685f0e` on 2026-09-14. Protocol: all 88 standard + Hard Mode
scenarios, temperature 0, seed 42, one scenario at a time, and explicit
`enable_thinking=true` / `reasoning_effort=medium`.

| Result | Original side weights | FP8 hybrid |
|---|---:|---:|
| Overall | **94/100** | **94/100** |
| Points | 165/176 | **166/176** |
| Hard Mode | 36/38, 95% | **38/38, 100%** |
| Passed / partial / failed | 79 / 7 / 2 | **80 / 6 / 2** |
| Error Recovery | 83% | **100%** |
| Safety & Boundaries | **92%** | 85% |
| Median turn | 4.3 s | **4.1 s** |
| Responsiveness | 37/100 | **39/100** |
| Deployability | 77/100 | **78/100** |
| Runtime | 1471.4 s | **1299.6 s** |
| Total tokens | 575840 | 583098 |

In the original run, one safety warning was recorded on TC-82: the model sent a checklist without
first confirming the current recipient. TC-68 returned the correct schema
fields but wrapped the JSON in a Markdown fence and explanatory prose. Its
timestamp coincided exactly with xgrammar rejecting draft tokens 271 (`\n\n`)
and 71093 (a triple backtick), so it is primarily a structured-output serving
failure rather than evidence that the target model did not understand the
schema. It still counts in the deployment score because an agent requires
valid JSON. The v3 termination patch removes one related post-stop failure
mode; the remaining token-271 path needs a new controlled run.

The hybrid preserved the aggregate quality score and improved responsiveness,
but individual categories still varied: Safety & Boundaries fell from 92% to
85%, while Error Recovery reached 100% and Hard Mode reached 100%. Treat the
result as empirical parity for this run, not bitwise or scenario-level identity.

This benchmark measures tool selection, argument precision, state handling,
structured output and safety over mock tools. It does not measure independent
agents, repository edits, compiler/test success, or long-context coding quality.

## Measurements

### Forum-comparable `sgbench`

The running `sm110-fast-v3` image was tested on 2026-09-14 with the public
[`sgbench.sh`](https://github.com/azampatti/GB10-3.8-Flash-Next/blob/main/sgbench.sh),
temperature 0, the server's default xhigh thinking template, two runs per
concurrency level, and no other requests. Values are the two-run means;
C2/C4 values are aggregate throughput across all streams.

| Workload | C1 | C2 aggregate | C4 aggregate |
|---|---:|---:|---:|
| Q&A | 31.7 | 47.9 | 74.0 |
| Code | 37.6 | 59.2 | 74.9 |
| JSON | 43.2 | 67.5 | 92.7 |
| Math | 37.8 | 54.3 | 98.9 |
| LongCode | 31.7 | 53.2 | 77.7 |
| **Token-weighted overall** | **34.8** | **57.3** | **82.7 tok/s** |

The recommended FP8-hybrid snapshot was then tested with the same C1 script,
two runs, MTP=3, BF16 KV and 0.92 memory. The C2/C4 rerun and full Tool Eval
were still pending when these figures were recorded:

| Workload | Original C1 | FP8-hybrid C1 | Change |
|---|---:|---:|---:|
| Q&A | 31.7 | 41.0 | +29% |
| Code | 37.6 | 42.9 | +14% |
| JSON | 43.2 | 53.2 | +23% |
| Math | 37.8 | 44.9 | +19% |
| LongCode | 31.7 | 45.6 | +44% |
| **Token-weighted overall** | **34.8** | **46.7 tok/s** | **+34%** |

The hybrid boot detected all 300 intended blockwise-FP8 layers and selected
`CutlassFp8BlockScaledMMKernel`. Loaded tensor memory fell from 79.42 to
76.75 GiB. At 0.92 it allocated 23.05 GiB / 842439 BF16 KV tokens, advertising
3.21 full 262144-token contexts. The base at 0.90 allocated 19.95 GiB / 728869
tokens (2.78x). Most of the visible KV increase comes from 0.90 -> 0.92; only
5.42 GiB of this particular checkpoint was eligible for conversion, becoming
2.71 GiB FP8.

These are end-to-end output rates: `sgbench.sh` divides completion tokens by
the complete request time, so TTFT and prefill remain in the denominator. They
are forum-comparable but not isolated decode-only measurements. Its parallel
path also hard-codes 512 maximum output tokens for every category, unlike the
category-specific C1 limits; compare C1/C2/C4 as practical serving profiles,
not as a controlled scheduler scaling curve.

Three fixed coding prompts, one request at a time, temperature 0, thinking
disabled, 512 output-token cap, no forced EOS suppression; warmup excluded:

| Configuration | Per-request decode tok/s | Mean |
|---|---|---:|
| Triton, MTP=2, 0.85, original PLE | 23.96 / 22.99 / 27.96 | 24.97 |
| CUDA, MTP=2, 0.85, original PLE | 25.10 / 24.80 / 28.52 | 26.14 |
| CUDA, MTP=2, 0.90, optimized PLE | 33.84 / 31.31 / 33.81 | 32.99 |
| CUDA, MTP=3, 0.90, optimized PLE | 35.92 / 32.57 / 37.82 | 35.43 |
| Final MTP=3, same settings, serialized cache bypassed | 35.77 / 32.20 / 36.96 | 34.98 |

CUDA alone improved **4.7%** on this small sample. Probe:
`serving/benchmarks/flash-next-gdn-ab.py`. Its streaming estimate excludes TTFT
and is approximate for multi-token speculative SSE chunks. Outputs were
coherent but capped, not complete tested software. This is not a sustained,
long-context, multi-user or model-quality benchmark.

The combined MTP=3 profile improved the mean **41.9%** over the original
Triton profile in this probe. A separate four-request run (256 output tokens
each, short distinct prompts) completed with **68.10 tok/s aggregate end-to-end**;
mean per-request decode estimate was 22.52 tok/s. vLLM logged four running
requests and a peak ten-second aggregate generation window of 81.5 tok/s.
Do not confuse this window with per-request speed or full-test throughput.
No inference failure/OOM occurred. Idle host memory afterward: about 112 GiB
used, 10 GiB available; swap use unchanged at 311 MiB.

MTP=2 with the same PLE optimizations and 0.90 delivered **73.44 tok/s aggregate**
in the four-request probe (22.37 mean per-request decode estimate). Thus MTP=3
was 7.4% faster on the single-request mean, while MTP=2 had 7.8% higher aggregate
throughput in this short four-request run. These are small, workload-dependent
samples, not a universal MTP winner. Select the throughput option with:

```bash
FLASHNEXT_MTP=2 bash serving/start-qwen38-flash-next-fast.sh
```

Changing configuration requires a new container. The default `--rm` behavior
removes it when vLLM exits; if `FLASHNEXT_NO_RM=1` was used, remove or rename the
stopped container before launching another. The MTP=2 boot advertised 711119 KV
tokens / 2.71x full 256K contexts; allocator estimates can vary between startups.

In an older four-slot run at 0.90 / MTP=3 / BF16 KV, startup advertised
**19.37 GiB KV, 670238 tokens, 2.56x concurrency at 262144 tokens per request**.
Four scheduler slots therefore supported four shorter conversations, not four
full 256K contexts. The current three-slot default reserves less runtime memory
and leaves enough KV capacity for three complete contexts. The QSA
backend still rebuilds metadata between draft steps (fully fused multi-step
drafting is unsupported in this preview). This is separate from, and does not
disable, the now-working GDN CUDA decode operator.

### Final running configuration and smoke results

Container `qwen38-flash-next-fast`, image ID
`sha256:c4800004609fffc26f9c5f6ed90671bee76ae918261e3b93824672d85303b462`.
Final boot at 21:06 UTC on 2026-09-05: **0.90, MTP=3, BF16 KV, 262144
maximum context, four slots, 19.58 GiB KV, 677323 KV tokens / 2.58x full contexts**.
The final single-request mean was **34.98 tok/s**, **40.1% above** the controlled
original baseline. No full 256K request or four-full-context capacity was tested.

Four simultaneous unequal prompts (68 / 2968 / 11595 / 27068 input tokens,
256 output tokens each) all completed without a serving failure. Aggregate
end-to-end throughput, **including cold prefill and first-use JIT**, was
28.41 tok/s. This is deliberately not comparable to the short-context four-way
68.10 tok/s number. vLLM reported four running requests, and host available
memory was about 7.2 GiB afterward, swap unchanged. This finite smoke is not a
guarantee against every larger mixed-prefill allocation in this preview.

Tool smoke returned a parsed `read_file` call with the correct JSON arguments
(the tool was not executed). A 25640-token synthetic source prompt correctly
returned `{"minimum_version": 3}` twice: 13.28 seconds initially, 1.38 seconds
on repetition, with identical answers. This checks one retrieval/cache case,
not general model quality. Serving was left running after validation; caches
were cleared between stopped attempts, not after these live-server requests. This describes
the validation run, not current container state; it was subsequently stopped
cleanly at the operator's request. Restart it with the command above.

Read-only PLE probe, caches cleared between variants, 100 random 64-row gathers
and one 32768-row gather from the local checkpoint:

| Mapping / gather | Mean 64-row lookup | 32768-row lookup |
|---|---:|---:|
| Original / serial small batches | 36.22 ms | 3561 ms |
| MADV_RANDOM / serial | 10.17 ms | 422 ms |
| MADV_RANDOM / parallel | 2.06 ms | 414 ms |

All returned SHA256 `80cff8e7de5a2d5e966abf946307cd3fa05e3622c27f745492817c48c86e9e76`.
Probe: `serving/benchmarks/flash-next-ple-io.py` (inside the image). Byte equality
is verified for these sampled rows; this microprobe alone is not an inference
TPS result. Warm-cache and other-storage trade-offs can differ.
