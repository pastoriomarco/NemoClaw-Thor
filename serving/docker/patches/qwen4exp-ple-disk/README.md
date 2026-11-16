# qwen4exp-ple-disk: serve the Qwen4Exp PLE table from disk (vLLM 0.33.0)

The Qwen3.8-Flash-Next ("Qwen4Exp") FP4-Local checkpoint carries one PLE
n-gram table (layer 1, `split_ngram_parts=128`, 16 n-gram heads x 160 dims,
about 320 M rows). In NVFP4 a row is 80 B of packed E2M1 values plus 10 B of
E4M3 block scales, so the table is about 29 GB. Stock 0.33 keeps it in pinned
host memory (`Qwen4ExpPLEPinnedHostEmbedding`, Engram `cpu_offload=True`) or
on the GPU. On Thor both come out of the same 122 GiB pool, and the model
cannot start.

A token reads 16 rows of the table. With `VLLM_PLE_DISK=1` the table stays in
the checkpoint's safetensors files. Each lookup reads only the rows it needs
through `np.memmap` views of the tensors' byte ranges, so the page cache backs
them, as llama.cpp does with a GGUF mmap. With the flag unset, the patched
file falls back to the stock embedding classes. The only other difference is
that it imports the small NumPy-only `ple_disk_table` module.

This is a port of the earlier `vllm_ple_mmap.py` recipe (preview fork, FP8
table in `model-plefp8-*` shards). It moves that recipe to 0.33's embedding
classes and to the NVFP4 table format.

## Files

Paths mirror the package, relative to `vllm/`:

| File | Kind | What it does |
|---|---|---|
| `vllm/models/qwen4_exp/common/ple_disk_table.py` | new, NumPy only | Parses safetensors headers. Maps every `layers.<L>.ple.ple_embedding.ngram_embedding.shard_<i>.{weight,weight_scale}` tensor as an `np.memmap`, with `MADV_RANDOM` by default. Checks the shard layout against the model. Host half of a lookup: CPU dedup (`np.unique`), unique rows gathered straight into the staging buffer (inline for decode-sized batches, a thread pool for larger ones), inverse indices in front. `stage_async` runs the same staging on a one-thread prefetch executor after a wait callable (the CUDA event behind the ids' D2H copy). Timing counters for the stats line. |
| `vllm/models/qwen4_exp/common/ple_disk.py` | new | `Qwen4ExpPLEDiskEmbedding` (subclass of 0.33's `Qwen4ExpPLEEmbedding`): the 0.33 prefetch interface (`start_prefetch` / `finalize_prefetch`, default) and the synchronous custom op `vllm::qwen4_exp_ple_disk_lookup` (`VLLM_PLE_DISK_PREFETCH=0`). `PLEEarlyFetch` and the hooks `ple_early_begin` / `ple_early_token` for the [early row fetch](#early-row-fetch-vllm_ple_disk_early). |
| `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` | modified full copy of 0.33 | Three hunks marked `[ple-disk]` (diff below). |
| `vllm/models/qwen4_exp/common/ple_disk_cache.py` | new, NumPy only | Slot-pool cache (`VLLM_PLE_DISK_CACHE_GB`, off by default): CLOCK slot index, read planner, `pread` readers, cached staging. `ple_disk.py` imports it only when a cache switch is set, so only cache runs need the mount. See [Slot-pool cache](#slot-pool-cache-vllm_ple_disk_cache_gb). |
| `tools/ple_repack.py` | offline tool, not mounted | Writes the PLE table as one file with each row's values and scales contiguous (one page per row), for `VLLM_PLE_DISK_PACKED`. |
| `tests/test_ple_disk_table.py` | CPU self-test, not mounted | Synthetic NVFP4-shaped checkpoint. Compares the mmap gather with direct indexing. |
| `tests/test_ple_disk_cache.py` | CPU self-test, not mounted | CLOCK against a port of Edge-LLM's `PleCache`, the read planner, and the cached lookup against today's staging. |
| `tests/test_ple_disk_early.py` | CPU self-test, not mounted | Early row fetch: staging unchanged by warming, warm reads, prefetch-thread order; with the image's vLLM, early ids against the target's verify-batch ids. |

No other 0.33 file is touched. In particular the CSF patch's `modelopt.py` and
`nvfp4_utils.py` are not changed. The AMD Qwen4Exp path is unchanged.

### How `Qwen4ExpPLEDiskEmbedding` works

- **Weights.** `allocate_embedding_weight` returns zero-row CPU placeholders,
  so the 0.33 NVFP4 method's `create_weights` still registers `weight` (uint8)
  and `weight_scale` (E4M3) with the right width and dtype, but allocates
  nothing. `weight_scale_2` (one fp32 value) is created and loaded as usual.
- **Loader.** The table shards reach `parameter.weight_loader(...,
  checkpoint_start=...)` from the unmodified `Qwen4ExpNGramEmbedding.load_weights`.
  The class overrides `weight_loader` so that every call carrying
  `checkpoint_start` returns without copying. The shape and dtype checks in
  `load_weights` still run, because the placeholders keep their row width.
  The shards count as loaded, and no missing or unexpected parameter is
  reported. Every other tensor loads normally: `weight_scale_2`, the hash
  buffers, and the conv, norm and proj weights.
- **Table mapping.** The tables are opened in `__init__`, from
  `--model` or from `VLLM_PLE_DISK_DIR`, through
  `model.safetensors.index.json` (bare file names are joined with the model
  directory). The layout is checked before any weight loads.
- **Prefetched lookup (default).** `supports_prefetch` is `True`, so 0.33's
  own prefetch wiring drives the lookup; no other 0.33 file changes:
  1. `Qwen4ExpModel.forward` calls `_start_layer_ple_prefetch` for layer
     `i + 1` before it runs layer `i` (0.33 `nvidia/model.py:494`, calls at
     `:548` and `:559`). For the PLE layer (layer 1) that is before layer 0.
  2. `Qwen4ExpNGramEmbedding.start_prefetch` (0.33
     `nvidia/ngram_embedding.py:400`) hashes the ids on the GPU into its
     persistent `_prefetch_ids` buffer (allocated at `:239` because
     `supports_prefetch` is true) and calls our `start_prefetch`, an eager
     break. It enqueues a non-blocking D2H copy of the ids into a pinned id
     buffer and records an event, then hands `stage` to the prefetch thread.
     The main thread returns at once and launches layer 0.
  3. The prefetch thread waits for the event (`cudaEventSynchronize`, GIL
     released), dedups, and gathers the rows into the pinned staging buffer
     while the GPU runs the embedding and layer 0.
  4. `Qwen4ExpNGramEmbedding.forward` (`:388`) calls `finalize_prefetch`. It
     allocates the output in the graph segment; its eager half
     `_finalize_prefetch` joins the staging job, enqueues the H2D copy and the
     0.33 `_lookup_nvfp4_ple_embedding_kernel` on the current stream. With
     ETP=1 the kernel writes the output directly. Otherwise it decodes into a
     temporary, then does the pinned class's ETP reduce and DP selection.
  The rows, inverse indices, H2D copy and kernel are the same as the
  synchronous path, so the output is bit-identical.
- **Synchronous lookup (`VLLM_PLE_DISK_PREFETCH=0`).** Today's path, unchanged.
  `supports_prefetch` is `False`, so `start_prefetch` returns early (`:408`)
  and `forward` takes the `embedding(ngram_ids)` branch (`:395`):
  1. Allocate the output in the graph segment.
  2. Call the op, which is an eager break of the breakable CUDA graph and opaque to `torch.compile`.
  3. Inside the op: a blocking D2H copy of the ids, CPU dedup and range mask, gather into one persistent pinned staging buffer, and one async H2D copy into a persistent device buffer.
  4. Run 0.33's `_lookup_nvfp4_ple_embedding_kernel` on the unique rows. The inverse indices act as local ids in `[0, U)`, and `-1` marks rows another tensor-parallel (TP) rank owns.

  The kernel expands and decodes on the GPU in one pass. The output has the
  same dtype (`params_dtype`) and the same values as the device and pinned
  paths. TP and ETP: ids outside `shard_indices.org_vocab_[start,end)` give
  zero rows, like the 0.33 kernels. The ETP all-reduce is the pinned class's
  `_reduce_etp_embeddings`. FP8 and unquantized tables use the 0.33 raw-row
  kernel instead.
- **Staging size.** The staging buffer is fixed at
  `max_num_batched_tokens x ETP-DP x 16 x (8 + 90)` bytes, about 12.8 MB
  pinned plus 12.8 MB device memory at 8192 batched tokens. Prefetch adds a
  pinned id buffer of `max_num_batched_tokens x ETP-DP x 16 x 8` bytes (1 MB)
  and one thread. Batches larger than that are refused; the bound never grows.

### Which lookups are prefetched (MTP)

- Only the target model looks up the table. The MTP drafter passes
  `ngram_context=None` (0.33 `nvidia/mtp.py:350`) and has no PLE layer.
- With prefetch on, every target forward goes through `start_prefetch` /
  `finalize_prefetch`: prefill chunks, decode, the MTP verify step (last
  token + K drafts), mixed batches, and the profile and capture runs. None
  uses the synchronous op.
- The batch's ids cannot be known any earlier. The runner writes the last
  sampled token and the drafts into `input_ids` on the GPU (0.33
  `v1/worker/gpu/input_batch.py:457-469`), and the drafts come from the end of
  the previous step. The host never has them before the forward, so the n-gram
  hash cannot run on the CPU for decode or verify. Each single token's ids are
  final earlier, though; the [early row fetch](#early-row-fetch-vllm_ple_disk_early)
  uses that. It could only do so for
  pure prefill chunks, where the overlap with layer 0 already hides the
  gather. The D2H wait still exists. It runs on the prefetch thread, not in
  the forward.
- What stays exposed per step: dedup + gather minus the GPU time of the
  embedding and layer 0. At decode, layer 0 is about 1 ms, so most of the
  ~4.7 ms gather stays on the critical path. At prefill, layer 0 of an
  8192-token chunk should cover most of the ~45 ms lookup. The stats line
  shows the split: `prefetch wait` is the time the main thread still waits.

## Early row fetch (`VLLM_PLE_DISK_EARLY`)

Off by default (step A2 of `docs/FLASH-NEXT-SPEED-PLAN.md`). With
`VLLM_PLE_DISK_EARLY=1`, the rows of every token of the next MTP verify batch
but the last are read while that batch is still being drafted. Only the last
draft's 16 rows are left for the batch's own staging to read from the disk
while layer 0 runs. Needs the prefetched lookup (the default), speculative
decoding, and the hooks in the `qwen4exp-spec-step` overlay. Works with and
without the slot-pool cache, for any K.

### What happens per step (one sequence, K drafts)

A verify batch is `t0, d1, ..., dK`: `t0` is the token sampled by the
previous verify, `d<j>` comes from draft pass `j - 1`. A token's 16 n-gram ids
depend only on that token and its `ngram_size - 1` predecessors, so they are
final as soon as the token exists on the GPU:

| When (main thread, no host sync) | Token | Hook |
|---|---|---|
| after `postprocess_sampled` in `sample_tokens` | `t0` | `ple_early_begin` (runner) |
| after draft pass 0 (draft prefill), if K ≥ 2 | `d1` | `ple_early_token` (speculator `propose`) |
| after draft pass `s` for `s = 1 .. K-2` | `d<s+1>` | `ple_early_token` (`_multi_step_decode`) |
| target forward, `start_prefetch` | whole batch | today's path, disarms the early fetch |

At each hook, `PLEEarlyFetch` (`ple_disk.py`):

1. `begin` only: builds the predecessors of `t0` exactly as
   `Qwen4ExpModelState._prepare_ngram_context` will for the next step (from
   `all_token_ids` at `num_computed_tokens - (ngram_size - 1) ..`, EOS before
   the start) and takes `t0` from `last_sampled_tokens`, as
   `combine_sampled_and_draft_tokens` will. A draft is copied from
   `draft_tokens[0, j]`. All into one persistent int32 token buffer.
2. Hashes that one token with the stock `compute_ngram_ids` (the same Triton
   kernel), its predecessors given as a one-row `ngram_context` view of the
   token buffer, into a persistent id buffer.
3. On a side stream: a non-blocking D2H copy of the 16 ids into pinned memory
   and an event, then `lookup.warm_async` on the prefetch thread.
4. Prefetch thread: waits for the event, then `warm` reads the token's owned
   rows and drops them. Uncached path: through the mmap gather. Cached path:
   only the rows the slot pool lacks (`row_slot == 0`), through the same read
   planner and `pread` readers. No lookup state changes: no reservation, no
   pin, no CLOCK step.

The prefetch thread runs these jobs in order with the staging jobs, so the
target's staging job starts after them. It stages exactly as without the
switch; the rows it misses are now in the page cache, so its reads return
without waiting for the disk. Hits, slots, staging bytes, uploads and the
decode are unchanged, so the output is bit-identical. A guess that does not
match the next batch (another request scheduled, fewer drafts, a prefill
chunk) only costs reads nobody uses.

Not covered: with more than one running request the hooks do nothing; the
fused draft path (`_fused_multi_step_decode`, not used by this model) only
fetches `t0` and `d1`.

Cost per hook: about 3 small kernels (10 for `begin`), a D2H copy of 128 B and
an event on the side stream, a few tens of µs of host time while the host runs
ahead of the GPU. The buffers (K x 16 ids, pinned and on the GPU, K events)
are reused one batch later; by then that batch's staging job, which runs after
these reads, has been joined.

### How to verify in the log

- **At init:** `PLE disk: early row fetch on for <layer>: the bonus token and
  drafts 1..<K-1> of each verify batch`. Without prefetch or speculative
  decoding: a warning `... early row fetch off`.
- **Every `VLLM_PLE_DISK_STATS_SEC`:** the `PLE disk stats` line ends with
  `; early: <n> tokens, <r> rows read (<d> reads waited for the disk), <x>
  ms/token`. Expect about K tokens per verify lookup. With the cache on, `r`
  counts only rows the pool lacked. The cache counts `preads` and `waited for
  the disk` stay those of the staging jobs: with the switch on, `waited for
  the disk` should drop to about one token's share.
- **Effect:** `prefetch wait` (ms/lookup) should drop. The plan's gate is
  `prefetch wait - id sync < 0.3 ms` or target GPU time -1 ms
  (`VLLM_SPEC_STEP_TIMING`). If `early: 0 tokens`, the hooks do not fire
  (check that the spec-step overlay is mounted and the switch reaches both
  files).
- **Exactness:** at temperature 0, same prompts and seed, compare the output
  text with a run without `VLLM_PLE_DISK_EARLY` (same other settings). It must
  be identical, and so must the spec-decode acceptance counters.

## Slot-pool cache (`VLLM_PLE_DISK_CACHE_GB`)

Off by default; with the variable unset, every path above is unchanged.
Modelled on TensorRT Edge-LLM 0.12 `--ple-residency cached`
(`cpp/runtime/embedding/pleCache`, `pleReadPlan`, `pleReadExecutor`). The GPU
keeps a small working set of table rows, so most lookups need no disk read
and no row upload, and the ~27 GB page-cache reservation for the table is
no longer needed. The output stays exact: every miss is read from the table
on disk, nothing is dropped, and the GPU decodes with the same 0.33 kernel.

### Data flow per step

The prefetch flow, the side stream and the staging during graph capture are
the same as above. Only what is staged and uploaded changes.

1. `start_prefetch` (eager break) copies the ids to the host on the side
   stream and hands staging to the prefetch thread, as today.
2. Prefetch thread (`PLECachedLookup.stage`):
   - Waits for the ids and dedups them (`np.unique`).
   - Reserves the step's distinct rows in the host index
     (`ClockSlotCache.reserve`). First it pins every hit, so a miss of the
     same step never evicts a row the step uses. Then it claims CLOCK slots
     for the misses: the hand skips pinned slots, takes a free or
     unreferenced slot, and clears reference bits. Slot 0 is reserved: it is
     never a row and holds zeros (unowned rows map to it).
   - Reads only the missed rows:
     - Up to `VLLM_PLE_DISK_PREAD_ROWS` misses (decode and MTP verify): the
       read planner widens each row to whole pages, sorts the pages by
       (file, offset) and merges overlapping or adjacent ones. A first
       `preadv(RWF_NOWAIT)` pass on this thread takes what the page cache
       holds. The reads that must wait for the disk go to up to
       `VLLM_PLE_DISK_READERS` threads, so their NVMe latencies overlap
       instead of queueing one fault at a time as in the mmap gather.
     - More misses (prefill chunks): the threaded mmap gather of the uncached
       path, for the missed rows only. With many cached pages, one
       Python-level syscall per page costs more than the page faults.
   - Writes into the pinned staging buffer: the per-id slot map (int32), then
     the missed rows' slots (int64), then their packed values and scales.
3. `_finalize_prefetch` (eager break) waits for that job, then:
   - Enqueues one H2D copy of the used part of the staging buffer. On a step
     with no misses that is the slot map alone (4 B per id).
   - Runs `index_copy_` per table to write the missed rows into their pool
     slots.
   - Runs `_lookup_nvfp4_ple_embedding_kernel` with the pools as the table and
     the slot map as the ids.
   - Releases the step's pins.

   Releasing at that point is safe. A later step can overwrite an evicted
   slot only through its own `index_copy_`, and that runs after this step's
   decode on the same stream.

Hits cost the index lookup and 4 B of upload. Misses cost the disk read, and
90 B of upload plus 8 B for the slot.

### Memory

- **GPU pool:** `slots x 90 B`, allocated in the layer's `__init__`, which
  runs inside vLLM's `load_model`. vLLM counts it as model memory before it
  sizes the KV cache, like any other weight. 1 GiB holds 11.9 M rows. The
  replay of this policy on captured agent traffic stayed under 0.25 GiB per
  session.
- **Host index:**
  - 4 B per table row (`row -> slot`, int32), about 1.2 GiB for the
    ~320 M-row table. It is touched at init, so it is resident from the
    start.
  - 13 B per slot (`slot -> row`, reference bit, pin count).
- **Staging:** pinned and device staging grow by `4 x max_rows` bytes (the
  slot map), about 0.5 MB at 8192 batched tokens. Reads use a 64 MiB host
  read buffer, allocated on the first read and enlarged only if one merged
  read is larger than that.
- **Page cache:** misses are still buffered reads, so the page cache keeps
  recently read pages while memory is free. Nothing depends on it any more:
  reclaiming those pages costs only extra disk reads on later misses. The KV
  share (`--gpu-memory-utilization`) can therefore grow by about the old
  page-cache reservation, minus the pool and the host index.

### CUDA graphs

Same as the uncached path. The upload, the scatter and the decode run in
the `_finalize_prefetch` eager break. The pools are static tensors, and the
output buffer is allocated in the graph segment. No new host sync is on the
graph path: the prefetch thread still waits for the side-stream event, and
during capture the step is still staged on the main thread. FULL graphs are
still refused.

### Repacked table (optional, `VLLM_PLE_DISK_PACKED`)

FP4-Local stores a row's 80 B of values and 10 B of scales in separate
tensors, so a miss reads two pages. `tools/ple_repack.py` writes one file in
which each row's 90 B are contiguous and whole rows fill 4 KiB blocks (45
rows and 46 B of padding per block). A row then never spans two pages, so a
miss reads exactly one page.

- The output is a safetensors file: one U8 tensor `ple_packed`, data starting
  at a 4 KiB boundary, and metadata the loader checks (format, row count, row
  layout, block size).
- The file is about 29 GB.
- The tool reads the whole table once and checks 4096 random rows of the
  output against the checkpoint.
- Run it offline, when the NVMe is not serving. It needs only NumPy (any
  container that has it):

```bash
P=${HOME}/workspaces/dev_ws/src/NemoClaw-Thor/serving/docker/patches/qwen4exp-ple-disk
MODEL=<host checkpoint directory>          # the directory --model points at
OUT=${HOME}/fn-bench/ple                   # needs ~29 GB free
mkdir -p "$OUT"
docker run --rm --entrypoint python3 \
  -v "$P":/p:ro -v "$MODEL":/model:ro -v "$OUT":/out \
  <the vLLM image start.sh uses> \
  -I /p/tools/ple_repack.py --model /model --layer 1 --out /out/ple_packed_l1.safetensors
```

The file only applies with the cache on. The uncached path keeps reading
the checkpoint.

### Not supported

- ETP/TP > 1: refused at init.
- DP with ETP=1 should work, with one pool and index per rank, but it is
  untested.
- FP8 or unquantized tables use the same pool path with the raw-row kernel,
  but they are untested: this checkpoint is NVFP4.

### How to verify in the log

- **At init:** the `PLE disk: ... reads ... rows` line ends with
  `slot-pool cache <N> slots (<x> GiB GPU, <y> GiB host index), <r> readers,
  checkpoint rows` (or `packed <path>`). With the cache off it ends with
  `no slot-pool cache`.
- **Every `VLLM_PLE_DISK_STATS_SEC`:** the `PLE disk stats` line gains
  `; cache: <hit rows>, <miss rows> (<hit %>), <evictions>, <preads> (<n>
  waited for the disk), <MiB read>, <rows via mmap>, <used>/<usable> slots
  used`. The counts are per interval, except `slots used`.
  - Expect a decode hit rate of about 60–70 %.
  - `prefetch wait` (ms/lookup) and `gather` should drop well below the
    uncached run.
  - `slots used` should stay far below the pool size; if it reaches the
    pool size, evictions start.
- **Exactness:** at temperature 0, compare the output with a run without
  `VLLM_PLE_DISK_CACHE_GB` (same seed and prompts).

### Diff of `nvidia/ngram_embedding.py` against the 0.33.0 original

```diff
@@ -29,6 +29,7 @@
     Qwen4ExpPLEUnquantizedEmbeddingMethod,
 )
 from ..common.ops.ple import ple_ngram_ids
+from ..common.ple_disk_table import ple_disk_enabled  # [ple-disk]
 
 logger = init_logger(__name__)
 
@@ -219,6 +220,10 @@
             if engram_config is not None and engram_config.cpu_offload
             else Qwen4ExpPLEDeviceEmbedding
         )
+        if ple_disk_enabled():  # [ple-disk] serve the table from the checkpoint
+            from ..common.ple_disk import Qwen4ExpPLEDiskEmbedding
+
+            embedding_cls = Qwen4ExpPLEDiskEmbedding
         # A TP reduce-scatter covers all vocabulary owners only when ETP=TP.
         self.use_reduce_scatter = (
             self.use_sequence_parallel
@@ -236,6 +241,8 @@
             data_parallel_rank=data_parallel_rank,
             reduce_results=not self.use_reduce_scatter,
         )
+        if ple_disk_enabled():  # [ple-disk] early row fetch reuses this hash
+            self.ngram_embedding.attach_early_fetch(self)
         if self.ngram_embedding.supports_prefetch:
             # The side-stream lookup outlives eager-break args, whose
             # graph-pool storage later segments may reuse.
```

## Docker mounts

The container package root is `/usr/local/lib/python3.12/dist-packages/vllm`.
`P` is this directory:

```bash
P=${HOME}/workspaces/dev_ws/src/NemoClaw-Thor/serving/docker/patches/qwen4exp-ple-disk
V=/usr/local/lib/python3.12/dist-packages/vllm
  -v $P/vllm/models/qwen4_exp/common/ple_disk.py:$V/models/qwen4_exp/common/ple_disk.py:ro \
  -v $P/vllm/models/qwen4_exp/common/ple_disk_table.py:$V/models/qwen4_exp/common/ple_disk_table.py:ro \
  -v $P/vllm/models/qwen4_exp/nvidia/ngram_embedding.py:$V/models/qwen4_exp/nvidia/ngram_embedding.py:ro \
```

The slot-pool cache needs one more mount. `ple_disk.py` imports the module
only when `VLLM_PLE_DISK_CACHE_GB` or `_SLOTS` is set, so the lines above
still suffice with the cache off:

```bash
  -v $P/vllm/models/qwen4_exp/common/ple_disk_cache.py:$V/models/qwen4_exp/common/ple_disk_cache.py:ro \
```

With `VLLM_PLE_DISK_PACKED`, the repacked file needs a mount too, for
example:

```bash
  -v ${HOME}/fn-bench/ple:/opt/qwen4exp-ple:ro \
  -e VLLM_PLE_DISK_PACKED=/opt/qwen4exp-ple/ple_packed_l1.safetensors \
```

None of the new files exist in the image; Docker creates their mount points.

The early row fetch (`VLLM_PLE_DISK_EARLY=1`) needs no extra file of this
overlay, but it needs the three `qwen4exp-spec-step` mounts (their README),
whose runner and speculator call the hooks.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_PLE_DISK` | `0` | `1` enables the disk table. |
| `VLLM_PLE_DISK_DIR` | `--model` | Directory holding the checkpoint index and shards, if it differs from `--model`. |
| `VLLM_PLE_DISK_WORKERS` | `14` | Gather threads for batches above `FAST_ROWS` (page faults overlap across threads). |
| `VLLM_PLE_DISK_FAST_ROWS` | `512` | Unique-row count up to which the gather runs inline, without the thread pool. |
| `VLLM_PLE_DISK_CHUNK` | `2048` | Rows per thread-pool task. |
| `VLLM_PLE_DISK_MADV_RANDOM` | `1` | `madvise(MADV_RANDOM)` on every mapping, which disables readahead. |
| `VLLM_PLE_DISK_PREFETCH` | `1` | `0` restores the synchronous lookup exactly (A/B). |
| `VLLM_PLE_DISK_STATS_SEC` | `30` | Log interval for the `PLE disk stats` line (`0` turns the log off). |
| `VLLM_PLE_DISK_CACHE_GB` | `0` (off) | GPU slot pool size in GiB (`slots = GiB x 2^30 / 90`); turns the slot-pool cache on. It must hold at least one step's distinct rows (`max_num_batched_tokens x 16`, about 11 MB at 8192 tokens), or init refuses. |
| `VLLM_PLE_DISK_CACHE_SLOTS` | unset | Slot count instead of `CACHE_GB` (slot 0 included); wins over `CACHE_GB`. |
| `VLLM_PLE_DISK_READERS` | `4` | `pread` threads for misses that wait for the disk (1..32; Edge-LLM's default is 4). |
| `VLLM_PLE_DISK_PREAD_ROWS` | `4096` | Misses per step up to which the planned `pread` path runs; larger miss sets (prefill) use the mmap gather. |
| `VLLM_PLE_DISK_PACKED` | unset | Repacked table file from `tools/ple_repack.py` (cache path only). |
| `VLLM_PLE_DISK_EARLY` | `0` | `1` reads each verify-batch token's rows while the batch is drafted ([early row fetch](#early-row-fetch-vllm_ple_disk_early)). Read by this overlay and by `qwen4exp-spec-step`; needs prefetch and MTP. |

With the cache on, `VLLM_PLE_DISK_MADV_RANDOM=1` also sets
`POSIX_FADV_RANDOM` on the `pread` file descriptors (no readahead).
`VLLM_PLE_DISK_WORKERS`, `_FAST_ROWS` and `_CHUNK` still drive the mmap
gather of large miss sets.

The stats line reports lookups split into `sync`, `prefetch hit` (staging
finished before the PLE layer needed it) and `miss`, row counts, and per-lookup
averages in ms: `host op` (main-thread time in the lookup ops),
`prefetch wait` (main thread waiting for the staging job), `id sync` (wait for
the ids' D2H copy; on the prefetch thread when prefetching, so it overlaps),
`dedup`, `gather`, `H2D` (enqueue) and `decode launch`.

Tuning hint: at decode the gather of ~80 unique rows runs inline
(`VLLM_PLE_DISK_FAST_ROWS=512`). If `gather` stays in the milliseconds, the rows
miss the page cache and every fault waits for one NVMe read in turn.
`VLLM_PLE_DISK_FAST_ROWS=0` sends them to the thread pool, so the faults
overlap. Try it A/B with the stats line.

## vLLM flags

- **`-cc.cudagraph_mode=PIECEWISE`** is required (or `--enforce-eager`), with
  or without prefetch. The lookup runs on the host on every step, so it cannot
  sit inside a FULL CUDA graph. `start_prefetch` and `_finalize_prefetch` are
  eager breaks, as in 0.33's pinned-host class, and refuse stream capture
  with the same error. FULL graphs would need the GPU to wait on a host-written
  flag, for example `cuStreamWaitValue32` on pinned memory fed by the
  prefetch thread. That is not implemented: the GPU would block on a host
  thread, and would hang if that thread stalls. 0.33 auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH=1` for
  `Qwen4ExpForConditionalGeneration`. In breakable PIECEWISE mode the op ends
  the current graph segment, runs eagerly, and resumes; the same happens on
  every replay. With the default `FULL_AND_PIECEWISE`, decode batches would be
  captured FULL, where `eager_break_during_capture` runs the function inside
  the capture (`compilation/breakable_cudagraph.py:108`). `lookup_into`
  detects stream capture and refuses with an error naming this flag, so it
  never replays stale rows.
- **No `splitting_ops` entry is needed.** With breakable CUDA graphs on, 0.33
  forces `compilation_config.mode = NONE` (`config/vllm.py:869`), so there is
  no torch.compile and no FX graph splitting. The lookup is still registered
  as the opaque custom op `vllm::qwen4_exp_ple_disk_lookup`
  (`torch.ops.vllm.qwen4_exp_ple_disk_lookup`). That is the name to add to
  `splitting_ops` only if someone forces `-cc.mode=VLLM_COMPILE` with
  piecewise graphs. Do not set `VLLM_USE_BREAKABLE_CUDAGRAPH=0`. This model has
  no compiled submodule, so without breakable graphs 0.33 drops piecewise
  capture (`config/compilation.py` "neither a compiled submodule nor
  breakable CUDA graphs"). Only FULL capture is left, which the lookup
  refuses; use `--enforce-eager` in that case.
- **Leave memory for the page cache.** Pinned host memory, GPU memory and the
  page cache all come out of the same unified pool. Every byte that
  `--gpu-memory-utilization` / KV cache claims is a byte the hot part of the
  29 GB table cannot stay cached in. Size the KV cache so that several GB stay
  free. Watch the `PLE disk stats` log line: the gather time per lookup rises
  sharply when rows miss the cache.

## Self-test

```bash
python3 -I tests/test_ple_disk_table.py
python3 -I tests/test_ple_disk_cache.py
python3 -I tests/test_ple_disk_early.py
```

The early test's host half needs only NumPy. Its id half needs the image's
vLLM with this overlay's files mounted at their package paths (CPU container,
no GPU), and is skipped otherwise:

```bash
IMG=<the vLLM 0.33 image>; P=<copy of this directory>
V=/usr/local/lib/python3.12/dist-packages/vllm
docker run --rm --runtime runc -e NVIDIA_VISIBLE_DEVICES=void --network none \
  -v $P:/p:ro -w /tmp \
  -v $P/vllm/models/qwen4_exp/common/ple_disk.py:$V/models/qwen4_exp/common/ple_disk.py:ro \
  -v $P/vllm/models/qwen4_exp/common/ple_disk_table.py:$V/models/qwen4_exp/common/ple_disk_table.py:ro \
  -v $P/vllm/models/qwen4_exp/nvidia/ngram_embedding.py:$V/models/qwen4_exp/nvidia/ngram_embedding.py:ro \
  --entrypoint python3 $IMG -I /p/tests/test_ple_disk_early.py
```

Early row fetch (`test_ple_disk_early.py`):

- **Staging unchanged:** 36 MTP-shaped verify batches and prefill chunks for
  K = 1, 3, 5. One lookup warms every token but the last of each verify batch
  first (through `warm_async`, ids set after submission), the other does not.
  Both must stage the same bytes and slots and keep the same CLOCK state
  (row/slot maps, reference bits, pins, hand, evictions). Uncached, a TP
  range, a pool holding the table, a pool of one step (evicts every step),
  with and without `RWF_NOWAIT`.
- **Reads:** warming reads exactly each token's owned rows the pool lacks,
  and its reads do not change the staging jobs' read counters.
- **Order:** a warm job waits for its ids, and a staging job submitted after
  it runs only after it.
- **Ids (image only):** `PLEEarlyFetch` on CPU tensors, driven like the
  runner and speculator hooks, must give the ids the target computes for the
  whole batch (stock `_prepare_ngram_context` + `compute_ngram_ids`), for
  K = 1..5, n-gram sizes 2, 3 and 5, short histories (EOS padding) and EOS
  tokens in context and drafts. Hooks beyond K-1 drafts and after the
  disarm do nothing.

The cache test needs only NumPy. Torch is optional: with it, the test also
compares the NVFP4 decode. It checks:

- **CLOCK:** the vectorized index matches a sequential Python port of
  Edge-LLM's `PleCache` in slots, evictions and hand position, over random
  steps with hits, misses, repeats and padding, for 3 to 2000 slots. It also
  checks that pinned rows of an unreleased step survive later steps, that a
  claim refuses when every slot is pinned, that slot 0 is never handed out,
  the capacity refusal, and rollback.
- **Read planner:** 300 random plans against brute force. Reads are
  page-aligned (or end at the file end), sorted, disjoint and never
  adjacent. No gap page is read and no page is read twice. Every span's
  bytes sit at its planned position.
- **Cached lookup against today's staging:** a synthetic 20,000-row
  NVFP4-shaped checkpoint, run through 41 decode, verify and prefill steps,
  synchronous and prefetched. The GPU half is emulated in NumPy: one upload,
  scatter into the pools, gather through the slot map. Every step must give
  the same rows (and BF16 decode) as the uncached path. This is checked for:
  - a pool that holds the table
  - a pool of one step's size (evicts every step)
  - 1 and 8 readers without `RWF_NOWAIT`
  - the repacked table
  - the mmap gather for large miss sets, plain and repacked
  - 12 KiB read waves
  - a TP vocabulary range (unowned rows map to slot 0)

  Also: the repacked table reads one page per missed row (3 `pread`s against
  6), a mismatched repacked file is refused, and a pool smaller than one
  step is refused.
- **Env:** the cache is off by default; GiB, slots, readers, `PREAD_ROWS`
  and MADV reach the lookup; reader counts outside 1..32 are refused.

The test needs NumPy; torch is optional and adds an NVFP4 decode comparison.
It builds a synthetic two-file checkpoint: 7 shards, weight and scale shards
interleaved across files, plus distractor MTP and other-layer tensors. It then
checks that the staged and expanded rows are byte-equal to direct indexing for:

- decode-sized batches (inline path)
- prefill-sized batches (threaded path)
- repeated ids
- a TP-rank vocabulary range with unowned rows (zeros)
- refusal of a batch larger than the staging buffer

- prefetch staging (`stage_async`): waits for its ids, then writes the same
  staging bytes as `stage`; an oversize batch raises to the caller
- the timing counters are filled

The GPU half (D2H/H2D copies, event, the 0.33 Triton kernel, eager breaks) is
only exercised in the container.

## Open risks

Early row fetch:

- **Untested on the GPU.** On Thor only: the hooks inside the real runner and
  speculator, the one-token Triton hash on views of the token buffer, the
  side-stream copies and events between draft FULL-graph replays.
- **Gain depends on the disk.** It removes waits for the disk, not the host
  work of staging (dedup, CLOCK, `pread` from the page cache, upload). With
  rows already in the page cache, little changes. Under memory pressure the
  page cache can drop warmed pages before the batch stages them.
- **Errors in a warm job are not raised.** Nobody joins its future; a
  failing warm shows up as `early: 0 tokens`, and the batch's own staging
  raises the same read error.
- **Extra reads:** rejected drafts' rows are read but not used (about
  K x 16 rows per step at most).

Slot-pool cache:

- **Untested on the GPU.** The CPU tests cover the host half and an
  emulation of the GPU half. What runs only on Thor is `index_copy_` into
  the pools and the 0.33 kernel with int32 ids (a new Triton
  specialization).
- **Decode wait.** A step's misses wait for one round of parallel disk reads
  plus thread hand-off. Python thread wake-ups add tens to hundreds of µs;
  `VLLM_PLE_DISK_READERS` trades parallelism against that. Tune it with the
  stats line.
- **Host index:** 1.2 GiB of host memory regardless of pool size.
- **Pin leak.** If a forward raises between `start_prefetch` and
  `finalize_prefetch`, that step's pins are never released and its slots
  stay out of CLOCK. This is the same unguarded case as below.
- **Page size.** Reads align to the system page size. A 64 KiB-page kernel
  would read 64 KiB per missed row; the JetPack kernel uses 4 KiB pages.
- **`RWF_NOWAIT`.** If the filesystem refuses `RWF_NOWAIT`, the reader
  turns it off at once and every miss goes through the threads.
- **Reads are not split.** Unlike Edge-LLM, merged reads have no size cap.
  Every page read holds a requested row, so this only matters for long runs
  of adjacent rows, which hashed n-grams do not produce.

Uncached path:

- Untested on the GPU so far. Check on Thor: the output matches the
  synchronous path (`VLLM_PLE_DISK_PREFETCH=0`) at temperature 0, the
  `PLE disk stats` line shows `0 sync`, and decode tok/s.
- The decode gain is bounded by the GPU time of layer 0 (about 1 ms). The
  gather itself stays on the critical path; see the `FAST_ROWS` hint.
- If a forward raises between `start_prefetch` and `finalize_prefetch`, the
  staging job can still be running when the next step starts. Nothing guards
  that case; the engine normally stops on such an error anyway.
- `start_prefetch` and `_finalize_prefetch` are plain eager-break methods,
  like 0.33's pinned class, not custom ops. They rely on 0.33 forcing
  `compilation_config.mode = NONE` with breakable CUDA graphs. A forced
  `-cc.mode=VLLM_COMPILE` would trace into them. Use `VLLM_PLE_DISK_PREFETCH=0`
  in that case.
