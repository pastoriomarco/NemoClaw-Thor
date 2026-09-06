#!/usr/bin/env python3
"""Small fixed-prompt HTTP A/B, not a quality benchmark or long-context sweep."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.request

PROMPTS = [
    'Write a complete Python asyncio bounded worker pool with graceful shutdown, '
    'exception propagation, timeouts, and comprehensive pytest tests. Explain the '
    'race conditions it avoids. Include all code, not just an outline.',
    'Review this cache design: a dict maps keys to (value, expiry), reads delete '
    'expired keys, writes evict the first key when capacity is full, and multiple '
    'threads access it. Explain correctness bugs, implement a thread-safe LRU with '
    'TTL in Python, and provide comprehensive deterministic tests.',
    'Design a transactional job queue backed by PostgreSQL for four workers. '
    'Provide schema, claim and retry SQL, crash recovery, idempotency handling, '
    'Python worker pseudocode, and tests for the failure modes. Be thorough.',
]


def source_context(count, offset=0):
    source = '\n'.join(f'def validate_record_{i}(record):\n'
                       f'    """Validate module {i}; empty records are rejected."""\n'
                       f'    return bool(record) and record.get("version", 0) >= {i % 7}\n'
                       for i in range(offset, offset+count))
    return 'Repository source for reference:\n```python\n' + source + '\n```\n'


def request(base, prompt, limit):
    payload = dict(model='qwen3.8-flash-next', messages=[dict(role='user', content=prompt)],
                   temperature=0, seed=42, max_tokens=limit, stream=True,
                   stream_options={'include_usage': True},
                   chat_template_kwargs={'enable_thinking': False})
    req = urllib.request.Request(base + '/chat/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    first = last = None
    output = ''
    usage = None
    finish = None
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            event = json.loads(data)
            if event.get('error'):
                raise RuntimeError(event['error'])
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                fragment = delta.get('content') or delta.get('reasoning') or delta.get('reasoning_content') or ''
                if fragment:
                    last = time.perf_counter()
                    if first is None:
                        first = last
                    output += fragment
                finish = choice.get('finish_reason') or finish
    elapsed = time.perf_counter() - start
    if not usage or first is None or last <= first:
        raise RuntimeError(f'Invalid stream: usage={usage}, output={output!r}')
    tokens = usage['completion_tokens']
    return dict(prompt_tokens=usage['prompt_tokens'], completion_tokens=tokens,
                prompt_tokens_details=usage.get('prompt_tokens_details'),
                ttft_s=first-start, total_s=elapsed,
                decode_tps=(tokens-1)/(last-first), end_to_end_tps=tokens/elapsed,
                finish_reason=finish, output=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8050/v1')
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--parallel', type=int, default=0)
    parser.add_argument('--long-context', action='store_true',
                        help='Prepend synthetic source to measure cold longer-context requests')
    parser.add_argument('--mixed-context', action='store_true',
                        help='Use unequal synthetic prompt lengths with --parallel')
    args = parser.parse_args()
    results = []
    warmup = request(args.base_url, 'Explain the Python GIL in three sentences.', 64)
    print('warmup', json.dumps({k:v for k,v in warmup.items() if k!='output'}), flush=True)
    if args.parallel:
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(request, args.base_url,
                                   (source_context([0, 64, 256, 600][i % 4], i*1000)
                                    if args.mixed_context else '') + PROMPTS[i % len(PROMPTS)] +
                                   f' Use example module worker_{i}.', 256)
                       for i in range(args.parallel)]
            results = [f.result() for f in futures]
        aggregate = sum(r['completion_tokens'] for r in results)/(time.perf_counter()-started)
        print('aggregate_completion_tps', aggregate, flush=True)
    else:
        for prompt in PROMPTS:
            if args.long_context:
                prompt = source_context(600) + prompt
            result = request(args.base_url, prompt, 512)
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='output'}), flush=True)
    report = dict(label=args.label, timestamp=time.time(), warmup=warmup, results=results,
                  method='(completion_tokens - 1) / (last text event - first text event); '
                         'approximate for multi-token speculative SSE chunks; warmup excluded')
    if args.parallel:
        report['aggregate_completion_tps'] = aggregate
    Path(args.output).write_text(json.dumps(report, indent=2))
    print('mean_decode_tps', sum(r['decode_tps'] for r in results)/len(results), flush=True)


if __name__ == '__main__':
    main()
