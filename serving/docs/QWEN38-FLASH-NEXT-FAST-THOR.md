# Qwen3.8-Flash-Next: current best verified Thor recipe

**Recommended as of 2026-09-06:** this is the best verified recipe in this
repository for Qwen3.8-Flash-Next on Jetson AGX Thor. Use this setup instead
of the historical Triton fallback. It prioritizes single-request decode speed
without further quantizing the downloaded weights: fused SM110a GDN CUDA,
native NVFP4 MoE, optimized PLE reads, **MTP=3, 0.90 memory, BF16 KV,
256K context and four scheduler slots**. The final controlled short-context
mean was about **35 tok/s versus 25 tok/s** for the original setup.
This is a measured recommendation, not a claim of the fastest possible engine
or guaranteed speed at every context/concurrency; MTP=2 won the small
four-request aggregate-throughput comparison below.

Status (2026-09-05): fused CUDA + parallel random PLE reads verified at 0.90
with MTP=2 and MTP=3, including four concurrent requests. Default: MTP=3
for individual response speed; MTP=2 is the measured aggregate-throughput option.

## What changes

Image: `nemoclaw-thor/qwen38-flash-next-vllm:sm110-gdn-cuda`.

- Fused GDN **decode** CUDA kernel compiled for SM110a from upstream source
  `082cf021b7ef96e4819e386846ea34e5ef21c655`, associated with
  [merged vLLM PR #53835](https://github.com/vllm-project/vllm/pull/53835).
  It registers in a separate `thor_gdn` namespace; only that Python dispatch
  is redirected. Other native operators and the working preview loader remain.
- PLE mappings use `MADV_RANDOM`, avoiding useless disk read-ahead, and
  `VLLM_PLE_MMAP_FAST_ROWS=0` selects existing parallel reads for decode too.
  Similar mapping advice is documented in
  [Saren-Arterius's Spark recipe](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound).

The existing native NVFP4 routed experts, BF16 protected layers, FP8 PLE table,
embedded MTP, deterministic QSA top-k, and prefix-cache fixes remain unchanged.
**No weights downloaded or requantized.** GDN prefill and QSA still use their
supported Triton implementations; this does not remove every Triton kernel or
add FP4 KV support.

## Reproduce

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
export FLASHNEXT_MODEL_REV="7b719225242aacd3dbd3f9407468c2ee9a9d2594"
mkdir -p "$HF_CACHE" "$VLLM_CACHE" "$TORCH_CACHE" "$FLASHINFER_CACHE"
```

### 2. Build the working image

These are local image tags, not published images to pull. On a new machine,
build all three layers in order. If they already exist from the verified
setup, skip rebuilding and proceed to launch. The upstream Dockerfile pins
its official base by digest; source and GDN kernel revisions are pinned too.
Building requires network access for the base image and source files, but
does not download or modify model weights.

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

### 3. Use the existing checkpoint (or download once on a new machine)

The required model is **RadixArk/Qwen3.8-Flash-Next-NVFP4**, at the exact
revision exported above. No separate drafter or conversion is needed: MTP is
embedded. The complete checkpoint, including the PLE table, must be present
in `$HF_CACHE/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/$FLASHNEXT_MODEL_REV`.
When copying a Hugging Face cache to another machine, copy its referenced
`blobs` too, not just the snapshot symlinks.

**Skip the following command on our Thor: its weights are already downloaded.**
On a new machine only, this optional one-time download uses the built image,
the usual cache, and a fixed revision (not upstream latest). It can be a large
download; allow enough disk space for the complete checkpoint and Docker images.

```bash
docker run --rm --pull never --user "$(id -u):$(id -g)" \
  -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
  -e FLASHNEXT_MODEL_REV="$FLASHNEXT_MODEL_REV" --entrypoint python3 \
  nemoclaw-thor/qwen38-flash-next-vllm:sm110-gdn-cuda \
  -c 'import os; from huggingface_hub import snapshot_download; snapshot_download("RadixArk/Qwen3.8-Flash-Next-NVFP4", revision=os.environ["FLASHNEXT_MODEL_REV"])'
```

### 4. Optional validation and foreground launch

Optional isolated numerical tests (while model serving is stopped):

```bash
docker run --rm --runtime nvidia --gpus all --entrypoint python3 \
  nemoclaw-thor/qwen38-flash-next-vllm:sm110-gdn-cuda \
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

Single foreground launch command:

```bash
bash serving/start-qwen38-flash-next-fast.sh
```

Defaults: port **8050**, **0.90** GPU memory utilization, **BF16 KV**, **262144**
maximum context, **four scheduler slots**, **MTP=3**.
Four slots do not guarantee four fully resident 256K contexts. Check the
actual KV capacity in the boot log.

The launcher uses only the existing local RadixArk snapshot
`7b719225242aacd3dbd3f9407468c2ee9a9d2594` in `$HOME/thor-hf-cache`, and the usual
`$HOME/thor-{vllm,torch,flashinfer}-cache` folders. It never refreshes weights
upstream and never deletes/replaces a container. For a stopped existing
instance use `docker start -a qwen38-flash-next-fast`.

Overrides: `HF_CACHE`, `VLLM_CACHE`, `TORCH_CACHE`, `FLASHINFER_CACHE`,
`FLASHNEXT_IMAGE`, `FLASHNEXT_CONTAINER`, `FLASHNEXT_MODEL_REV`, `FLASHNEXT_PORT`,
`FLASHNEXT_MTP`, `FLASHNEXT_GPU_MEM`, `FLASHNEXT_KV`, `FLASHNEXT_MAX_SEQS`,
`FLASHNEXT_CONTEXT`. `FLASHNEXT_DETACH=1` opts into background serving.
For controlled diagnostics, `FLASHNEXT_GDN=triton` restores the old GDN path;
`VLLM_PLE_MMAP_MADV_RANDOM=0 VLLM_PLE_MMAP_FAST_ROWS=512` restores old PLE I/O.

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

## Measurements

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

Changing configuration requires a new container; `docker start` reuses the old
arguments. Stop and rename an existing container if preserving its logs before
launching another. The MTP=2 boot advertised 711119 KV tokens / 2.71x full 256K
contexts; allocator estimates can vary between startups.

At 0.90 / MTP=3 / BF16 KV, startup advertised **19.37 GiB KV, 670238 tokens,
2.56x concurrency at 262144 tokens per request**. Four scheduler slots therefore
support four shorter conversations, not four full 256K contexts. The QSA
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
