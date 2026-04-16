#!/usr/bin/env python3
"""Benchmark concurrent requests to test batch throughput scaling on Thor."""

import json, time, sys, concurrent.futures
import urllib.request

API = "http://localhost:8000/v1/chat/completions"

PROMPTS = [
    "Count from 1 to 20. Just list the numbers separated by commas.",
    "Write a short poem about the ocean in exactly 4 lines.",
    "What is the capital of France? Explain in 2 sentences.",
    "List the first 10 prime numbers.",
    "Translate 'hello world' to Spanish, French, German, and Japanese.",
    "Write a Python function to compute fibonacci numbers. Keep it under 10 lines.",
    "Explain what a neural network is in simple terms. 3 sentences max.",
    "What are the planets in our solar system? List them in order.",
]

def send_request(prompt, max_tokens=100):
    data = json.dumps({
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    elapsed = time.monotonic() - t0
    usage = result["usage"]
    return usage["completion_tokens"], elapsed

def bench_concurrent(concurrency, num_requests=8, max_tokens=100):
    """Send num_requests with given concurrency level."""
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]

    t0 = time.monotonic()
    total_tokens = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, p, max_tokens) for p in prompts]
        for f in concurrent.futures.as_completed(futures):
            try:
                tokens, elapsed = f.result()
                total_tokens += tokens
            except Exception as e:
                print(f"  Error: {e}")

    total_elapsed = time.monotonic() - t0
    throughput = total_tokens / total_elapsed
    return total_tokens, total_elapsed, throughput

def main():
    # Warmup
    print("Warmup...", end=" ", flush=True)
    send_request("Say hello", 10)
    print("done\n")

    print(f"{'Concurrency':>12} | {'Tokens':>8} | {'Time':>8} | {'Tok/s':>8} | {'Speedup':>8}")
    print("-" * 60)

    base_throughput = None
    for c in [1, 2, 3, 4, 6, 8]:
        tokens, elapsed, throughput = bench_concurrent(c, num_requests=8, max_tokens=100)
        if base_throughput is None:
            base_throughput = throughput
        speedup = throughput / base_throughput
        print(f"{c:>12} | {tokens:>8} | {elapsed:>7.1f}s | {throughput:>7.1f} | {speedup:>7.2f}x")

if __name__ == "__main__":
    main()
