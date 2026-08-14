#!/usr/bin/env python3
"""Render a messages JSONL with a model's real Hugging Face chat template."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Local model/checkpoint directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Optional per-record cap; 0 preserves the complete rendered record (default)",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face network access. Off by default so local weights are authoritative.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_tokens < 0:
        raise SystemExit("--max-tokens must be zero or positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=not args.allow_network,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0
    token_counts: list[int] = []
    with args.source.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            rendered = tokenizer.apply_chat_template(
                record["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            token_ids = tokenizer.encode(rendered, add_special_tokens=False)
            original_tokens = len(token_ids)
            if args.max_tokens and original_tokens > args.max_tokens:
                token_ids = token_ids[: args.max_tokens]
                rendered = tokenizer.decode(token_ids, skip_special_tokens=False)
            digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            if digest in seen:
                raise RuntimeError(f"Duplicate rendered content at source line {line_number}")
            seen.add(digest)
            token_counts.append(len(token_ids))
            output.write(json.dumps({"text": rendered}, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1

    if count == 0:
        raise RuntimeError("No records rendered")
    summary = {
        "records": count,
        "model": str(Path(args.model).resolve()),
        "local_files_only": not args.allow_network,
        "max_tokens": args.max_tokens or None,
        "tokens": {
            "minimum": min(token_counts),
            "maximum": max(token_counts),
            "mean": round(sum(token_counts) / len(token_counts), 2),
            "total": sum(token_counts),
        },
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
