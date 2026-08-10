# ManyForge local-model smoke evaluation — 2026-08-06

Point-in-time evaluation of Cosmos3 Nano, DeepSeek-V4-Flash-0731 (DS4) and
Cosmos3 Edge on one Jetson AGX Thor T5000 with 128 GiB unified memory.
ManyForge was on clean `main` at `8ae05265a7da`; NemoClaw-Thor was on `main`
at `7a418e7da9f3`.

These results measure compatibility with the ManyForge direct-lane Composer
tool protocol. They do not measure general model intelligence, coding quality,
vision quality or physical-reasoning quality.

## Summary

| Model/profile | Attempted | Pass | Soft | Fail | Effective | Median case | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| Cosmos3 Nano, no forced thinking | 66 | 26 | 12 | 28 | 38/66 (57.6%) | 8.9 s | Full run completed |
| Cosmos3 Nano, forced `<think>` format | 16 | 3 | 4 | 9 | 7/16 (43.8%) | 39.9 s | Stopped early |
| DS4, `reasoning_effort=off` | 2 | 0 | 1 | 1 | 1/2 soft only | 278.6 s | Stopped after two poor cases |
| Cosmos3 Edge, unconstrained auto tools | 2 | 0 | 0 | 2 | 0/2 (0%) | 169.1 s | Stopped after two failed cases |

Nine future-tier cases were skipped in the full Cosmos3 Nano run. A soft-pass
means the main state/tool assertion was acceptable but one or more secondary
assertions failed; it should not be read as a clean success. The DS4 and Edge
samples are intentionally too small to estimate full-corpus pass rates.

## Cosmos3 Nano

The full run used NVIDIA's 7.04 GiB ModelOpt NVFP4 checkpoint with vLLM
0.25.1, 65,536-token context, FP8 KV cache, 8,192-token chunked prefill and
two sequences. It did well on simple parameter and blackboard edits, but was
unreliable on composite tree operations, schema-sensitive calls and parts of
the pick-and-place chain. The vLLM log recorded 41 Hermes tool-parser JSON
errors. Rolling decode during sustained generation was normally about
18-26 tok/s; vLLM's rolling prompt-throughput samples are prefix-cache
dependent and are not comparable to DS4's per-request timing fields.

The baseline took 1,611.5 seconds across 66 attempted cases: 24.4 seconds
average, 8.9 seconds median, 136.1 seconds P95 and 183.2 seconds maximum.
Only one of nine requested chain self-heals succeeded; the other eight were
also affected by a stale golden replay payload (`motion_type` validation), so
the later pick-and-place failures are not a clean model-only measurement.

A second run forced the model-card-style
`<think>Your reasoning.</think>` response format and allowed 4,096 output
tokens. It was stopped after 16 attempted cases: seven cases took 144.7-162.9
seconds, the partial prefix scored 43.8% effective, and 25 upstream responses
generated 32,361 completion tokens. This partial run is not a full-corpus
head-to-head, but it was sufficient to show that forced reasoning greatly
increased latency without an encouraging reliability signal.

### Cosmos3 Nano conclusion

Cosmos3 Nano is fast enough for many simple calls but is not reliable enough
to replace the current ManyForge Composer assistant without parser/tool-call
work. Forced long reasoning is worse for this workload.

## DeepSeek-V4-Flash-0731 / DS4

DS4 v0.5.5 used the Thor `sm_110` image, the matching 0731 base and DSpark
drafter, 524,288-token context, two continuous banks, 4,096-token prefill
chunks, a 12 GiB VMM budget and an 8 GiB live-memory floor. The proxy injected
`reasoning_effort=off` into every request. The direct bridge was bounded to
285 seconds, below the corpus client's 300-second limit, so a timed-out case
could not keep mutating state during the next case.

The two-case run was intentionally stopped. Both canaries were poor enough
that a multi-hour continuation was not justified:

- `P1_wrap_root_specific` failed after 285.1 seconds. The model read state,
  repeatedly called `tree_draft_wrap_node`, then exhausted the bridge budget.
- `P2_scene_add_specific` soft-passed after 272.0 seconds. It repeated
  `scene_draft_add_object`, switched to the batch upsert tool, and omitted the
  expected shape/pose arguments from the scored call.

Across the ten completed DS4 model calls, server-reported prefill averaged
342.7 tok/s (363.5 token-weighted; 432.2 maximum). Decode averaged 16.3 tok/s
per call (11.6 token-weighted; 21.4 maximum). Cached follow-up TTFT was often
1.5-7.5 seconds, while cold/mostly-uncached TTFT reached 181.8 seconds. The
DS4 container itself remained healthy: no restart, OOM, Xid or serial fallback
was observed. The failure was tool-loop quality and end-to-end latency, not
serving stability.

### DS4 conclusion

DS4 remains useful as a long-context coding/reasoning service, but the
reasoning-off profile is a poor fit for the current ManyForge tool harness.
Do not run the full corpus again unchanged; first test a bounded-reasoning
profile or a DS4-specific tool-schema adaptation on a small canary set.

## Cosmos3 Edge

### Runtime

- Checkpoint: `nvidia/Cosmos3-Edge`, Hugging Face revision
  `2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2`.
- Weights: BF16, 7.19 GiB across three reasoner shards.
- Image: NVIDIA/model-card-recommended `vllm/vllm-openai:cosmos3`, digest
  `sha256:db0bb920b0b54e82ea96a98659bbd21921f87d0dcfc86feffdafa2db3f08be55`.
- vLLM: `0.23.1rc1.dev1306+gbfee6a802`.
- Served name and port: `cosmos3-edge`, `8050`.
- Context: 131,072 tokens; BF16 KV; `--gpu-memory-utilization 0.65`;
  `--max-num-seqs 2`; 8,192-token chunked prefill; prefix caching enabled.
- Parsers tested: `qwen3_xml` and `step3p5`, both with the `qwen3`
  reasoning parser.

From a clean memory state, startup and inference were stable. vLLM allocated
4.67 GiB for weights and 69.5 GiB for KV cache (650,672-token capacity), with
no Xid or inference OOM. One parser-only container recreation was attempted
before the old unified-memory allocation had been reclaimed; vLLM saw only
38.05 GiB free versus its 79.84 GiB target and entered a restart loop.
Stopping that loop, dropping page cache and relaunching from 115 GiB available
resolved it. No model files were affected.

### Tool-template verification

The server did not receive an explicit `--chat-template` flag because vLLM
automatically resolved the template bundled with the mounted checkpoint. A
post-run tokenizer check proved that the selected template was byte-for-byte
the Edge template:

| Source | Length | SHA-256 |
|---|---:|---|
| tokenizer-resolved `chat_template` | 12,158 bytes | `7120ee6666468d4e9b2dc11e133ac5c2fa765fa5907706bf0f906270aa5510c8` |
| `/model/chat_template.jinja` | 12,158 bytes | same |
| `/model/text_tokenizer/chat_template.jinja` | 12,158 bytes | same |

The resolved template contains the expected named-parameter protocol:

```xml
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
</tool_call>
```

The failure below was therefore not caused by a missing or incorrect chat
template.

### Direct canaries

A plain text canary (`Reply with exactly: READY`) passed in about 1.6 seconds.
The `qwen3` reasoning parser separated 88 reasoning tokens from the final
`READY` content, confirming that ordinary text generation and reasoning
extraction worked.

The tool canary asked Edge to call one `configure_scene` function with two
required nested objects:

- `shape = {"type":"box", "box_dims":[1.0,0.02,0.25]}`
- `pose = {"position":[0.0,-0.15,0.125]}`

| Tool parser | Thinking | Result |
|---|---|---|
| `qwen3_xml` | on | Correct function name, `arguments: "{}"`; exhausted the 1,024-token output budget |
| `step3p5` | on | Correct function name, `arguments: "{}"`; 978 completion tokens dominated by repetitive schema reasoning |
| `step3p5` | off | Correct function name, `arguments: "{}"`; exhausted the 1,024-token output budget |

In the thinking-on response, the model described malformed unnamed parameter
blocks such as `<parameter><value>...</value></parameter>` instead of the
named `<parameter=shape>...</parameter>` form. vLLM logged repeated XML
`mismatched tag` parse warnings. Both parsers could recover the function name,
but neither could infer missing parameter names or reconstruct the nested
arguments safely.

### Composer smoke canaries

The direct bridge used `tool_choice="auto"`, thinking enabled, temperature
zero, a 4,096-token response cap, 16 turns and a 285-second wall limit. The
run was deliberately stopped after the first two representative cases failed.

| Case | Time | Result | Evidence |
|---|---:|---|---|
| `P1_wrap_root_specific` | 173.3 s | fail | No `tree_draft_wrap_node`; no draft mutation; no final text |
| `P2_scene_add_specific` | 164.9 s | fail | No `scene_draft_add_object`; no scene mutation; no final text |

The bridge completed one upstream turn in each case, but received no parsed
tool calls, proposals, warnings or draft mutation. The raw scorer reports and
bridge response-shape audit are stored beside this report with
`cosmos3-edge-` filename prefixes.

This was not an inference-throughput failure. During the real ManyForge
prompt, vLLM reported a rolling prompt-throughput sample of about 4,885 tok/s
and sustained decode samples around 24.6-25.5 tok/s. The 165-173 second case
times came from generating close to the 4,096-token cap without reaching a
valid executable call.

### Root cause and corrected conclusion

The tested requests used unconstrained `tool_choice="auto"`. The ManyForge
direct bridge emits no OpenAI `function.strict` flag, so Edge was free to
generate arbitrary XML and vLLM only attempted to parse it afterward. The
model selected the intended function in the synthetic canary but violated its
own checkpoint template's parameter syntax.

The measured conclusion is:

> Stock Edge fails this workload through unconstrained automatic tool
> generation. Its schema-constrained tool path was not activated and remains
> to be tested.

### Unverified fix path

Inspection of the exact `cosmos3` image found that its `qwen3_xml` parser is
registered with vLLM's `qwen_3_coder` structural-tag grammar. The image also
has strict tool calling enabled by default. Generating the structural grammar
offline for the nested canary produced the correct trigger, named function,
nested JSON schemas and Qwen XML argument style.

Test one change at a time:

1. Serve with `--tool-call-parser qwen3_xml --reasoning-parser qwen3`.
2. Add `"strict": true` to each function descriptor while retaining
   `tool_choice="auto"`; also send `parallel_tool_calls=false`.
3. Repeat the nested direct canary, then `P1` and `P2` with thinking unchanged.
4. If calls become valid but reasoning remains unbounded, repeat with thinking
   disabled.
5. If automatic mode still fails to start a call, use
   `tool_choice="required"` until the first successful mutation, then return
   to `auto` so the agent can finish with text.
6. If format is fixed but tool selection remains poor, expose a smaller
   intent-relevant tool subset before considering parser heuristics or
   fine-tuning.

A custom post-hoc parser is not the preferred fix: it cannot reliably recover
arguments whose names were never emitted. Schema-constrained generation acts
during decoding and prevents the malformed region instead.

### Edge teardown

Edge and the direct bridge were stopped after the two-case result. Page cache
was reclaimed with `vm.drop_caches=3`; Thor returned to approximately 115 GiB
available RAM. Model weights remain in the external Hugging Face cache.

## Overall decision

- Cosmos3 Nano is the only model in this set with a full-corpus result. It is
  responsive on simple operations but not reliable enough for the default
  Composer assistant.
- DS4 remains the strongest long-context coding/reasoning option here, but its
  current reasoning-off ManyForge tool loop is too slow and unreliable.
- Cosmos3 Edge's unconstrained tool path failed, but the checkpoint template
  was correct and the exact serving image contains an untested strict
  structural grammar. Do not reject Edge for structured physical-AI tasks
  until that constrained path has been tested on the same two canaries.
