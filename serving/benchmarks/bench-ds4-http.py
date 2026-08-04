#!/usr/bin/env python3
"""Controlled OpenAI-path throughput probe for Entrpi DS4.

The generated prompt starts with a label-derived fixed-width nonce. This keeps
candidate runs out of DS4's warm-prefix cache while retaining deterministic
prompt construction within a named run. The script uses only the Python
standard library and trusts the server's own prefill/decode timing fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request


def prompt_of_size(target_bytes: int, run_key: str) -> str:
    nonce = hashlib.sha256(run_key.encode("utf-8")).hexdigest()
    header = (
        f"Benchmark nonce {nonce}. Ignore the nonce.\n"
        "The following is an immutable synthetic source archive. Read it, then "
        "continue the integer sequence requested at the end.\n"
    )
    line_template = (
        "record_{index:06d}: component=planner state=verified "
        "dependency=runtime invariant=deterministic checksum={checksum}\n"
    )
    parts = [header]
    size = len(header.encode("utf-8"))
    index = 0
    while size < target_bytes:
        checksum = hashlib.sha256(f"{index}:ds4-thor".encode("utf-8")).hexdigest()[:16]
        line = line_template.format(index=index, checksum=checksum)
        parts.append(line)
        size += len(line.encode("utf-8"))
        index += 1
    parts.append(
        "\nReturn only consecutive decimal integers, one per line, beginning at "
        "1000001. Continue without explanation until the response limit."
    )
    return "".join(parts)


def request_json(url: str, payload: dict, timeout: int) -> tuple[dict, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    return body, time.monotonic() - started


def parse_case(value: str) -> tuple[int, int]:
    try:
        prompt_kib_text, output_text = value.split(":", 1)
        prompt_kib = int(prompt_kib_text)
        output_tokens = int(output_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("case must be PROMPT_KIB:OUTPUT_TOKENS") from error
    if prompt_kib < 1 or output_tokens < 1:
        raise argparse.ArgumentTypeError("case values must be positive")
    return prompt_kib, output_tokens


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8050/v1")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        help="PROMPT_KIB:OUTPUT_TOKENS; repeat for a matrix",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    cases = args.case or [(8, 128), (8, 512), (72, 128)]
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    models_url = f"{args.base_url.rstrip('/')}/models"
    with urllib.request.urlopen(models_url, timeout=10) as response:
        models = json.load(response)
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    print(f"DS4 HTTP benchmark: {args.label}")
    print(f"endpoint={endpoint} model={args.model} repeats={args.repeats}")
    print(f"advertised_models={','.join(item['id'] for item in models.get('data', []))}")

    warm_payload = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": prompt_of_size(8 * 1024, f"{args.label}:warmup"),
        }],
        "temperature": 0,
        "max_tokens": 16,
        "reasoning_effort": "none",
    }
    warm_body, warm_wall = request_json(endpoint, warm_payload, args.timeout)
    warm_usage = warm_body.get("usage", {})
    warm_timings = warm_body.get("timings", {})
    print(
        f"warmup prompt={warm_usage.get('prompt_tokens', 0)} "
        f"output={warm_usage.get('completion_tokens', 0)} "
        f"prefill={warm_timings.get('prefill_tok_s', 0)} "
        f"decode={warm_timings.get('decode_tok_s', 0)} tok/s "
        f"wall={warm_wall:.2f}s",
        flush=True,
    )

    report = {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "repeats": args.repeats,
        "cases": [],
    }
    for prompt_kib, output_limit in cases:
        rows = []
        for repeat in range(1, args.repeats + 1):
            run_key = f"{args.label}:{prompt_kib}:{output_limit}:{repeat}"
            prompt = prompt_of_size(prompt_kib * 1024, run_key)
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": output_limit,
                "reasoning_effort": "none",
            }
            body, wall_seconds = request_json(endpoint, payload, args.timeout)
            usage = body.get("usage", {})
            timings = body.get("timings", {})
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            row = {
                "repeat": repeat,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "ttft_ms": float(timings.get("ttft_ms", 0.0)),
                "prefill_tok_s": float(timings.get("prefill_tok_s", 0.0)),
                "decode_tok_s": float(timings.get("decode_tok_s", 0.0)),
                "wall_seconds": wall_seconds,
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
            rows.append(row)
            print(
                f"case={prompt_kib}KiB:{output_limit} repeat={repeat} "
                f"prompt={row['prompt_tokens']} output={row['completion_tokens']} "
                f"ttft={row['ttft_ms']:.1f}ms prefill={row['prefill_tok_s']:.2f} "
                f"decode={row['decode_tok_s']:.2f} tok/s wall={wall_seconds:.2f}s",
                flush=True,
            )
        summary = {
            "prompt_kib": prompt_kib,
            "output_limit": output_limit,
            "median_prompt_tokens": median([row["prompt_tokens"] for row in rows]),
            "median_completion_tokens": median([row["completion_tokens"] for row in rows]),
            "median_ttft_ms": median([row["ttft_ms"] for row in rows]),
            "median_prefill_tok_s": median([row["prefill_tok_s"] for row in rows]),
            "median_decode_tok_s": median([row["decode_tok_s"] for row in rows]),
            "min_prefill_tok_s": min(row["prefill_tok_s"] for row in rows),
            "max_prefill_tok_s": max(row["prefill_tok_s"] for row in rows),
            "min_decode_tok_s": min(row["decode_tok_s"] for row in rows),
            "max_decode_tok_s": max(row["decode_tok_s"] for row in rows),
            "runs": rows,
        }
        report["cases"].append(summary)
        print(
            f"median={prompt_kib}KiB:{output_limit} "
            f"prompt={summary['median_prompt_tokens']:.0f} "
            f"prefill={summary['median_prefill_tok_s']:.2f} "
            f"decode={summary['median_decode_tok_s']:.2f} tok/s"
        )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as output:
            output.write(rendered)
            output.write("\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
