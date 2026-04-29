"""Long-output sustained-tps probe over OpenAI-compatible HTTP.
Same 3 prompts as omni_trt_long_tps.py (TRT-direct version), but goes through
HTTP so it works against vLLM, TRT-Edge-LLM api_server, or anything else
that speaks OpenAI chat-completions.

Usage:
    BASE_URL=http://127.0.0.1:8000 MODEL=nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
        python long_tps_oai.py
"""
import json
import os
import sys
import time
import urllib.request

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4")
ENDPOINT = f"{BASE_URL}/v1/chat/completions"


def chat(prompt, max_tokens, temperature=0.6):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    elapsed = time.time() - t0
    text = resp["choices"][0]["message"]["content"] or ""
    n_tokens = resp.get("usage", {}).get("completion_tokens", 0)
    if n_tokens == 0:
        # vLLM/TRT use different schemas occasionally; fall back to rough word count
        n_tokens = max(1, len(text.split()))
    tps = n_tokens / elapsed if elapsed > 0 else 0
    return n_tokens, elapsed, tps, text


def main():
    print(f"Probing {BASE_URL}\n")
    # Warmup (discard) — caps autotune + warm caches
    print("[warmup] discard run")
    chat("Count from 1 to 50, one number per line.", max_tokens=80)

    prompts = [
        ("L1: 600t open-ended writeup",
         "Write a detailed 600-word essay explaining how transformers work, "
         "covering attention, multi-head attention, positional encoding, layer "
         "normalization, and feed-forward networks in detail.",
         650),
        ("L2: 600t reasoning chain",
         "Solve step by step, showing all reasoning: A factory produces 12 widgets "
         "per hour. After running for 8 hours, 15% are defective. The non-defective "
         "ones go to two warehouses A and B with A getting 60% and B getting 40%. "
         "How many widgets in each warehouse? Then answer 5 follow-up questions "
         "about scaling this to 24 hours, 18% defective rate, and 70/30 split.",
         650),
        ("L3: 800t code+explanation",
         "Write a complete Python implementation of a binary search tree with "
         "insert, delete, search, and in-order traversal methods. Include detailed "
         "docstrings, type hints, and example usage at the end. Also explain the "
         "time complexity of each operation.",
         800),
    ]

    results = []
    for name, prompt, mt in prompts:
        n, e, tps, _ = chat(prompt, max_tokens=mt)
        print(f"=== {name}: {e:.2f}s, {n} tokens, {tps:.2f} tps ===")
        results.append((name, n, e, tps))

    print()
    print("=" * 60)
    print("Sustained tps summary:")
    for name, n, e, tps in results:
        print(f"  {name:42s}  {tps:6.2f} tps  ({n:4d}t / {e:5.1f}s)")
    total_tokens = sum(r[1] for r in results)
    total_secs = sum(r[2] for r in results)
    total_tps = total_tokens / total_secs if total_secs > 0 else 0
    print(f"  {'TOTAL':42s}  {total_tps:6.2f} tps  ({total_tokens}t / {total_secs:.1f}s)")


if __name__ == "__main__":
    main()
