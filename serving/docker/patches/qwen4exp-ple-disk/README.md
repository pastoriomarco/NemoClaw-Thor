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
| `vllm/models/qwen4_exp/common/ple_disk.py` | new | `Qwen4ExpPLEDiskEmbedding` (subclass of 0.33's `Qwen4ExpPLEEmbedding`): the 0.33 prefetch interface (`start_prefetch` / `finalize_prefetch`, default) and the synchronous custom op `vllm::qwen4_exp_ple_disk_lookup` (`VLLM_PLE_DISK_PREFETCH=0`). |
| `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` | modified full copy of 0.33 | Two hunks marked `[ple-disk]` (diff below). |
| `tests/test_ple_disk_table.py` | CPU self-test, not mounted | Synthetic NVFP4-shaped checkpoint. Compares the mmap gather with direct indexing. |

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
- The ids cannot be known any earlier. The runner writes the last sampled
  token and the drafts into `input_ids` on the GPU (0.33
  `v1/worker/gpu/input_batch.py:457-469`), and the drafts come from the end of
  the previous step. The host never has them before the forward, so the n-gram
  hash cannot run on the CPU for decode or verify. It could only do so for
  pure prefill chunks, where the overlap with layer 0 already hides the
  gather. The D2H wait still exists. It runs on the prefetch thread, not in
  the forward.
- What stays exposed per step: dedup + gather minus the GPU time of the
  embedding and layer 0. At decode, layer 0 is about 1 ms, so most of the
  ~4.7 ms gather stays on the critical path. At prefill, layer 0 of an
  8192-token chunk should cover most of the ~45 ms lookup. The stats line
  shows the split: `prefetch wait` is the time the main thread still waits.

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

The two new files don't exist in the image. Docker creates their mount points.
The prefetch change adds no mount: these three lines are unchanged.

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
```

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
