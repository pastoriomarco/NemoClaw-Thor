# Why the proxy nested-id normalizer never fired: SSE fragmentation

**Source:** proxy mutation log `/tmp/manyforge-assistant-e2e/vllm-proxy.jsonl`
(105 MB, this run). Proxy: `vllm-proxy.py`, profile `compat`, `:8000 → :8050`.

## Decisive measurement

```
contiguous 'scene-draft-add-object':  in RESPONSE bodies = 0 | in REQUEST excerpts = 0
```

The mangled id is **never present as a contiguous string** in any response body
the regex runs over. (It only shows up contiguously in assembled *conversation
history* on the request side — and even there it is outside the logged excerpt.)

## Mechanism: `function.arguments` streams one token per SSE event

Reassembling `tool_calls[*].function.arguments` deltas from one response
(`data:` events parsed individually):

```
#arguments-fragments in body: 396
longest single arguments-fragment = 33 chars   (a canonical id needs ~38 contiguous)
first fragments: ['{', '"code"', '":"', '"', '#', ' The', ' user', ' wants',
                  ' to', ' add', ' a', ' repeat', ' node', ' as', ' root', '.', …]
```

Each token arrives in its own SSE event wrapped as
`data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"<frag>"}}]}}],"created":…}`.
So a 38-char id is spread across ~10+ events, with full chunk boilerplate
(`"}}]}}],"created":…\n\ndata: {…"arguments":"`) between every token. A
text-level regex over the concatenated body — even the escaped-form patterns —
**cannot** match `"id":"manyforge__…"` because that substring exists nowhere in
the byte stream.

## Conclusion

`_NORMALIZE_NESTED_MCP_IDS` is a **text-regex over the response body** and is
**not SSE-aware**. OpenClaw streams responses, so `function.arguments` (and
content) are fragmented token-by-token; the normalizer therefore matches
nothing (0 rewrite-rule hits across 865 mutation records) and every mangled id
reaches OpenClaw's exact-match dispatcher unchanged → `Unknown tool id`.

This is the **primary** tools-mode defect — independent of the pattern gaps
(flat dashed tools like `manyforge__program-read`) and the code-mode JS-literal
gap. Fixing patterns without fixing streaming would not move the tools-mode
numbers.

## Required shape of the fix (streaming-aware)

1. In the proxy's SSE response path, accumulate per-choice, per-`tool_calls[index]`
   `function.arguments` delta fragments (and assistant `content` deltas, since
   gemma sometimes emits the call there) into a buffer.
2. When a tool call's arguments JSON is complete (balanced braces /
   `finish_reason`), parse it, canonicalize the nested ManyForge `id`
   (dash→underscore on the tool segment, prepend `manyforge__` for bare names,
   strip `mcp:…:` locator), **only if it resolves to exactly one known tool.**
3. Re-emit corrected SSE chunk(s) downstream (or re-chunk the assembled call).
   Log each repair `original→rewritten`.
4. Extend the id patterns to flat/read tools' dashed forms
   (`program-read`, `scene-inspect`, `catalog-read`, …).
5. Code-mode JS-string ids (`openclaw.tools.call('…')`) = separate, lower
   priority (or document code mode as unviable for gemma: code lane scored
   1/19 here).

Simpler alternative to evaluate: request the openclaw lane **non-streaming**
(if the gateway allows), so the proxy buffers a complete body the existing
regex can match — far smaller change, at a latency/UX cost.
