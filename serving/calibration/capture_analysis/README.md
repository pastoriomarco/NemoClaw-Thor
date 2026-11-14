# Capture analysis: PLE row usage and MTP draft vocabulary

Offline analysis of agent traffic captured in front of the Qwen3.8-Flash-Next
FP4-Local server (vLLM 0.33, Thor). It answers two questions:

- **PLE:** which rows of the layer-2 PLE n-gram table (320,001,536 rows x 90 B,
  NVFP4) are looked up, and what share of the lookups the N most frequent rows
  serve. This decides whether a small hot subset could stay resident in GPU
  memory with the rest served (exactly) from disk.
- **Draft vocabulary:** how well the reduced MTP draft vocabulary
  (`draft_vocab_code_47149.npy`) covers the generated tokens, what is missing,
  and replacement candidates (47,232 and 65,536 ids).

Everything runs on CPU in the official vLLM 0.33 image (`docker run --network
none`, read-only mounts, no GPU, no weights loaded).

## Run

```bash
# laptop: copy the scripts (+ build_draft_vocab.py) to Thor ~/fn-capture/scripts/
serving/calibration/capture_analysis/deploy_to_thor.sh
# optional, once: also copy the coding corpus (see "Coding corpus")
CORPUS=/path/to/coding-source-768.messages.jsonl serving/calibration/capture_analysis/deploy_to_thor.sh

# Thor: whole capture -> ~/fn-capture/analysis/REPORT.md
~/fn-capture/scripts/run_all.sh
# other inputs/outputs
~/fn-capture/scripts/run_all.sh ~/edgellm/dumps ~/fn-capture/analysis/preview-cline
```

`run_all.sh` runs `verify_ple.py` (id verification, below), `analyze.py`
(which also writes the lookup stream to a temporary spool), `learning_curve.py`
and `ple_cache_sim.py`, and appends the last two sections to REPORT.md. Each
dump folder is its own request stream (default `~/fn-capture/dumps` = effort
xhigh, `~/fn-capture/dumps-low` = effort low; file numbers may overlap, tasks
never cross folders); reasoning effort (`chat_template_kwargs.reasoning_effort`,
template default xhigh, `none` when thinking is off) is reported as a dimension.

Quick learning curve on a partial capture (about a minute):
`~/fn-capture/scripts/learning_curve.sh` (prints the table and a rising/flat verdict per pool).

- `learning_curve.py`: tasks by first request; per pool (all, agent, reasoning
  family, per effort) the last 25% of tasks are held out and the hot set is
  built from the first k = 1, 2, 4, 6, 8, 12, ... others: ceiling (rows seen) and
  1/2/4 GiB budgets for prefill/decode/combined. Verdict: slope of the combined
  ceiling over the last three points per doubling of k; rising if >= 1 pp.
- `ple_cache_sim.py`: replay of TensorRT Edge-LLM 0.12 `--ple-residency cached`
  (PleCache CLOCK, slot 0 reserved, per-step pinning of the union; planPleReads
  4 KiB-aligned merged ranges, 1 MiB reads; PleRowStore counters). Pools 0.25-8 GiB
  for the FP4-Local table (90 B slots, values + scales read as two spans; Edge-LLM
  0.12 has no NVFP4 source type) and the NVIDIA FP8 table (160 B rows, offsets from
  its safetensors header). Variants: cold pool over all requests in arrival order;
  warm pool from the top rows of half A, replaying half B. Prefill in 8192-token
  chunks, one step per decode token, MTP drafts not modelled. At most 2e8 lookup
  positions are replayed (a prefix in arrival order; stated in the report). Env: `CPUS` (default 4), `MEMORY` (default `8g` container limit),
`ANALYZE_ARGS` (extra flags, e.g. `--verify-every 1`, `--limit 100`).
Runtime: about 20 s per 100 MB of dumps plus ~1 min fixed (corpus
tokenisation, coverage passes); 8 GB of dumps should take about 30-40 min. Peak memory about 4 GB: four dense
uint16 counters over the 320 M rows plus the no-cache counter.

## Files

| File | What it does |
|---|---|
| `ple_hash.py` | NumPy re-implementation of the vLLM 0.33 n-gram row ids (`Qwen4ExpNGramEmbedding` layout + `compute_ngram_ids`). |
| `verify_ple.py` | Checks `ple_hash` against vLLM and the checkpoint (below). |
| `render.py` | Rebuilds prompt and reply token sequences from the dumps. |
| `analyze.py` | Streams the dumps, counts PLE rows and generated tokens, writes the report. |
| `run_all.sh`, `deploy_to_thor.sh` | Container wrapper on Thor; copy to Thor. |

## Outputs (`~/fn-capture/analysis/`)

| File | Content |
|---|---|
| `REPORT.md` | All results and the approximations. |
| `summary.json` | The same numbers, machine-readable. |
| `requests.csv` | Per request: conversation, hold-out half, family, prompt tokens, server vs modelled cached tokens, computed prefill, generated tokens (rebuilt vs server). |
| `ple_coverage_curves.csv` | Coverage vs top-N rows (log grid) for prefill, decode, combined, no-cache and the hold-outs. |
| `ple_hot_rows_top.npy` | uint32 row ids by descending combined lookup count, at most 4 GiB of table (47.7 M ids, 191 MB). Row id r is checkpoint shard `r // 2,500,012`, local row `r % 2,500,012`. |
| `ple_verification.json` | Result of `verify_ple.py`. |
| `draft_vocab_missing_tokens.csv` | Generated tokens outside the current draft vocabulary, by count, split by reasoning / content / tool call. |
| `draft_vocab_{capture,mixed_50_50}_{47232,65536}.npy` | Candidate vocabularies (sorted int32 ids, same format as `build_draft_vocab.py`; multiples of 128, so no runtime top-up). |

## How the PLE ids are verified

`verify_ple.py`, CPU only:

1. The multipliers, per-head prime sizes and offsets that `ple_hash` derives
   from `config.json` equal those of vLLM's own
   `Qwen4ExpNGramEmbedding._make_layer_multipliers` / `_make_vocab_layout`.
2. They also equal the persistent buffers stored in the checkpoint
   (`layers.1.ple.ple_embedding.{layer_multipliers,ngram_heads_vocab_sizes,ngram_heads_offsets}`,
   a few bytes read straight from the safetensors file).
3. The table layout from the safetensors headers: 128 shards of
   ceil(320,001,536 / 128) = 2,500,012 rows, 80 B E2M1 + 10 B E4M3 scales per
   row. All 16 heads (8 bigram, 8 trigram) index this one concatenated table
   through their offsets, and the `qwen4exp-ple-disk` overlay maps id r to
   shard `r // shard_size` — so the ids index the same rows the disk overlay
   reads.
4. Ids equal vLLM's CPU `compute_ngram_ids` (called unbound on the real
   class) for batched, chunked steps with `ngram_context` built like
   `Qwen4ExpModelState` (real text, random ids, EOS-segmented input).
5. Ids equal the Triton kernel `_ple_ngram_ids_kernel` (the GPU path), run in
   the Triton CPU interpreter on the same batches.

Ids are a pure function of the token sequence: position p reads 16 rows from
tokens p, p-1, p-2 (predecessors behind an `<|endoftext|>` or before the start
become `<|endoftext|>`). The MTP drafter has no PLE layer.

## Method and approximations

- **Prompt tokens** are exact: vLLM 0.33's own render path is called offline
  (`ChatCompletionRequest` validation, `build_chat_params`,
  `parse_chat_messages`, `safe_apply_chat_template` with the model's tokenizer
  and template; tools `model_dump()`-ed as the server does). On the test dumps
  the length equals `usage.prompt_tokens` for every request. Long prompts are
  tokenised incrementally: the tokens up to the last `<|im_start|>` inside the
  text shared with an earlier prompt are reused (added tokens split the text
  before BPE, so this is exact); every 25th request is also tokenised in full
  and compared (`incremental ... mismatches` in the report).
- **Reply tokens** are rebuilt: the SSE `reasoning` + `</think>` + `content`
  + tool calls in the template's XML format + `<|im_end|>`, tokenised in one
  piece. The token count matched `usage.completion_tokens` exactly on the
  Cline test dumps (13,891 of 13,891) and to 2 tokens on the first capture
  dumps. Arguments are re-serialised from the streamed JSON. Labels: up to
  `</think>` = reasoning (so one more token than the server's
  `reasoning_tokens`), `<tool_call>`..`</tool_call>` = tool call, rest =
  content. Requests without a reply (non-streaming, aborted) get their output
  from the next request of the same conversation (the assistant turn in its
  history, up to `<|im_end|>`); reasoning is missing there if the client drops
  it. The last such request of a conversation has unknown output (counted).
- **Conversations** = same tools + system prompt + first user message (the
  task); a drop in the message count starts a new conversation. The runner's
  `[family:x]` tag in the first user message gives the family; untagged
  requests are `untagged`.
- **Prefix caching:** only the uncached prompt suffix is looked up. When the
  reply carries `usage.prompt_tokens_details.cached_tokens`, that is used.
  Otherwise the prefix model: on this server (hybrid Mamba model, 832-token
  blocks) a request hits exactly at the cache point
  `c = floor(len(Q) / 832) * 832` of an earlier prompt Q it shares at least
  c tokens with; generated tokens are never reused. This matched the server on
  18 of 19 Cline requests and 66 of 67 capture requests (the misses: no
  earlier prompt in the dumps; one cache entry the server no longer had).
  KV eviction is not modelled. The no-cache upper bound (every prompt token)
  is reported as token totals and as a curve over a request sample.
- **Decode:** one lookup position per fed generated token (all but the last).
  MTP verification also looks up rejected draft tokens (K=3 drafts per step);
  not modelled, so real decode lookups are higher, and rejected-draft rows
  are mostly outside any hot set.
- **Coverage curves:** rows ranked by lookup count; ties at the cut are split
  proportionally (expected value under random tie-breaking). GiB = 2^30 B of
  table at 90 B/row.
- **Hold-out:** tasks are split into two halves of similar dump size
  (greedy); the hot set is ranked on one half and measured on the other,
  both directions. With fewer than 4 distinct tasks the split is by time
  (first half of the requests), which measures drift within one session, not
  generalisation. `ceiling` is the share of the other half's lookups that hit
  any row seen in the ranking half; no hot set from that half can exceed it.
- **Per family:** every 8th lookup position is sampled; coverage of the
  global and the hold-out hot sets on that sample.
- **Draft vocabulary:** coverage = share of generated tokens whose id is in
  the vocabulary (after the runtime top-up to 47,232), a proxy for what the
  drafter can propose. Candidates keep the byte, special and added ids, then
  rank by `capture` frequency (corpus only fills), by the coding corpus alone
  (`corpus_only`, the existing builder's rule), or by the mean of normalised
  capture and corpus frequencies (`mixed_50_50`); remaining slots go by id
  order. Hold-out: built on one half (+ corpus), measured on the other. The
  `.npy` artifacts are built on all capture + corpus; their in-sample coverage
  is optimistic.
- Hot sets are row-granular; page-cache behaviour (4 KiB pages, ~45 rows) is
  not modelled.

## Coding corpus

`../build_coding_corpus.py` builds it from local committed files of
NemoClaw-Thor, manyforge and manyforge_specs (stdlib + git). The vLLM image has
no git, so run it in any container that has both, e.g. on the laptop:

```bash
SRC=$HOME/workspaces/dev_ws/src
docker run --rm --network none --user $(id -u):$(id -g) -e HOME=/tmp \
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0='*' \
  -v $SRC/NemoClaw-Thor:/r/NemoClaw-Thor:ro -v $SRC/manyforge:/r/manyforge:ro \
  -v $SRC/manyforge_specs:/r/manyforge_specs:ro -v /tmp/corpus:/out \
  --entrypoint python3 manyforge-dev:latest -I /r/NemoClaw-Thor/serving/calibration/build_coding_corpus.py \
  --nemoclaw-thor /r/NemoClaw-Thor --manyforge /r/manyforge --manyforge-specs /r/manyforge_specs --output-dir /out
```

`run_all.sh` uses `~/fn-capture/analysis/corpus/coding-source-768.messages.jsonl`
when present and skips the corpus-based candidates otherwise. It is not
necessarily the corpus the current 47,149-id asset was built from (that came
from an older fork); the report shows the current asset's coverage of it.
