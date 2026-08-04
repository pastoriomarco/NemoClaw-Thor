#!/usr/bin/env python3
"""Repeated fixed-corpus concurrency probe for the DS4 OpenAI endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request


def make_prompt(target_bytes: int, key: str) -> str:
    parts = [
        f"Concurrency nonce {hashlib.sha256(key.encode()).hexdigest()}.\n",
        "Read this deterministic archive, then output consecutive decimal integers "
        "starting at 1000001, one per line, without explanation.\n",
    ]
    index = 0
    while sum(len(part) for part in parts) < target_bytes:
        parts.append(
            f"record_{index:06d}: state=verified dependency=runtime "
            f"checksum={hashlib.sha256(f'{key}:{index}'.encode()).hexdigest()[:16]}\n"
        )
        index += 1
    return "".join(parts)


def request(endpoint: str, model: str, prompt: str, output: int, timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": output,
        "reasoning_effort": "none",
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.load(response)
    return {
        "wall_seconds": time.monotonic() - started,
        "prompt_tokens": int(body.get("usage", {}).get("prompt_tokens", 0)),
        "completion_tokens": int(body.get("usage", {}).get("completion_tokens", 0)),
        "prefill_tok_s": float(body.get("timings", {}).get("prefill_tok_s", 0)),
        "decode_tok_s": float(body.get("timings", {}).get("decode_tok_s", 0)),
        "content_sha256": hashlib.sha256(
            body.get("choices", [{}])[0].get("message", {}).get("content", "").encode()
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8050/v1")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--prompt-kib", type=int, default=72)
    parser.add_argument("--output", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    all_rows = []

    for wave in range(1, args.waves + 1):
        prompts = [
            make_prompt(args.prompt_kib * 1024, f"{args.label}:{wave}:{slot}")
            for slot in range(args.concurrency)
        ]
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(request, endpoint, args.model, prompt, args.output, args.timeout)
                for prompt in prompts
            ]
            rows = [future.result() for future in futures]
        wall = time.monotonic() - started
        total_output = sum(row["completion_tokens"] for row in rows)
        aggregate = total_output / wall
        all_rows.extend(rows)
        print(
            f"wave={wave} concurrency={args.concurrency} wall={wall:.2f}s "
            f"output={total_output} aggregate_decode={aggregate:.2f} tok/s "
            f"request_decode={[row['decode_tok_s'] for row in rows]}",
            flush=True,
        )

    print(json.dumps({"label": args.label, "runs": all_rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
