#!/usr/bin/env python3
"""Benchmark vLLM throughput with various prompts.
Sends sequential requests and measures tokens/sec, acceptance rate, latency."""

import json, time, sys, argparse
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

def send_request(prompt, max_tokens=100, temperature=0):
    data = json.dumps({
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    elapsed = time.monotonic() - t0
    usage = result["usage"]
    content = result["choices"][0]["message"]["content"]
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "elapsed": elapsed,
        "tok_per_sec": usage["completion_tokens"] / elapsed if elapsed > 0 else 0,
        "content_preview": content[:80],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--label", type=str, default="benchmark")
    args = parser.parse_args()

    print(f"=== {args.label} ===")
    print(f"Rounds: {args.rounds}, Max tokens: {args.max_tokens}")
    print()

    # Warmup
    print("Warmup...", end=" ", flush=True)
    send_request("Say hello", max_tokens=10)
    print("done")

    total_tokens = 0
    total_time = 0
    results = []

    for r in range(args.rounds):
        for i, prompt in enumerate(PROMPTS):
            try:
                res = send_request(prompt, max_tokens=args.max_tokens)
                results.append(res)
                total_tokens += res["completion_tokens"]
                total_time += res["elapsed"]
                print(f"  [{r+1}/{args.rounds}][{i+1}/{len(PROMPTS)}] "
                      f"{res['completion_tokens']:3d} tok in {res['elapsed']:.2f}s "
                      f"({res['tok_per_sec']:.1f} tok/s) | {res['content_preview'][:50]}...")
            except Exception as e:
                print(f"  [{r+1}/{args.rounds}][{i+1}/{len(PROMPTS)}] ERROR: {e}")

    print()
    print(f"=== RESULTS: {args.label} ===")
    print(f"Total: {total_tokens} tokens in {total_time:.1f}s")
    print(f"Average throughput: {total_tokens/total_time:.2f} tok/s")
    avg_tps = sum(r["tok_per_sec"] for r in results) / len(results) if results else 0
    print(f"Average per-request: {avg_tps:.2f} tok/s")
    print(f"Requests: {len(results)} successful")

if __name__ == "__main__":
    main()
