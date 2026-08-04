#!/usr/bin/env python3
"""Long-context retrieval and throughput gate for the DS4 OpenAI endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request


def build_archive(target_kib: int, label: str) -> tuple[str, list[str]]:
    target_bytes = target_kib * 1024
    secrets = [
        f"THOR-{hashlib.sha256(f'{label}:early'.encode()).hexdigest()[:16]}",
        f"THOR-{hashlib.sha256(f'{label}:middle'.encode()).hexdigest()[:16]}",
        f"THOR-{hashlib.sha256(f'{label}:late'.encode()).hexdigest()[:16]}",
    ]
    header = (
        "This is an immutable diagnostic archive. Most records are distractors. "
        "Remember every value on a line beginning EXACT_NEEDLE.\n"
    )
    footer = (
        "\nReturn JSON only, with exactly this schema and no extra keys: "
        '{"needles":["early value","middle value","late value"]}. '
        "Keep archive order.\n"
    )
    needle_offsets = [target_bytes // 8, target_bytes // 2, target_bytes * 7 // 8]
    parts = [header]
    size = len(header.encode())
    needle_index = 0
    record_index = 0
    while size + len(footer.encode()) < target_bytes:
        if needle_index < len(secrets) and size >= needle_offsets[needle_index]:
            line = f"EXACT_NEEDLE_{needle_index + 1}: {secrets[needle_index]}\n"
            needle_index += 1
        else:
            digest = hashlib.sha256(
                f"{label}:{record_index}:ds4-thor-depth".encode()
            ).hexdigest()[:24]
            line = (
                f"archive_record_{record_index:07d}: kind=distractor "
                f"state=verified digest={digest}\n"
            )
            record_index += 1
        parts.append(line)
        size += len(line.encode())
    while needle_index < len(secrets):
        parts.append(f"EXACT_NEEDLE_{needle_index + 1}: {secrets[needle_index]}\n")
        needle_index += 1
    parts.append(footer)
    return "".join(parts), secrets


def post(url: str, payload: dict, timeout: int) -> tuple[dict, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response), time.monotonic() - started
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8050/v1")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--label", required=True)
    parser.add_argument("--target-kib", type=int, action="append")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    sizes = args.target_kib or [288]
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    failures = 0

    for target_kib in sizes:
        prompt, expected = build_archive(target_kib, f"{args.label}:{target_kib}")
        body, wall = post(
            endpoint,
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "reasoning_effort": "none",
            },
            args.timeout,
        )
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = body.get("usage", {})
        timings = body.get("timings", {})
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
        passed = parsed == {"needles": expected}
        failures += 0 if passed else 1
        print(
            f"depth={target_kib}KiB prompt={usage.get('prompt_tokens', 0)} "
            f"output={usage.get('completion_tokens', 0)} "
            f"ttft={timings.get('ttft_ms', 0)}ms "
            f"prefill={timings.get('prefill_tok_s', 0)} "
            f"decode={timings.get('decode_tok_s', 0)} tok/s "
            f"wall={wall:.2f}s retrieval={'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        if not passed:
            print(f"expected={json.dumps({'needles': expected})}")
            print(f"received={content}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
