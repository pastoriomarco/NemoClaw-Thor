#!/usr/bin/env python3
"""Build a domain-specific reduced vocabulary for Flash-Next MTP drafting.

The target model still verifies proposed tokens with its complete vocabulary.
Restricting the draft projection therefore affects acceptance/speed, not the
target distribution.  Corpus-frequency ranking is followed by special tokens,
byte fallbacks and tokenizer-id order as a deterministic frequency proxy.
"""

import argparse
import collections
import glob
import json
import os

import numpy as np
from transformers import AutoTokenizer


def texts_from(path: str):
    if os.path.isdir(path):
        for filename in glob.glob(os.path.join(path, "**", "*"), recursive=True):
            if (
                os.path.isfile(filename)
                and os.path.getsize(filename) < 20_000_000
                and filename.rsplit(".", 1)[-1]
                in {"json", "jsonl", "md", "txt", "py", "log"}
            ):
                yield from texts_from(filename)
        return

    try:
        raw = open(path, "rb").read().decode("utf-8", errors="ignore")
    except OSError:
        return

    if path.endswith((".json", ".jsonl")):
        records = []
        try:
            if path.endswith(".jsonl"):
                records = [json.loads(line) for line in raw.splitlines() if line.strip()]
            else:
                records = [json.loads(raw)]

            def walk(value):
                if isinstance(value, str):
                    yield value
                elif isinstance(value, dict):
                    for child in value.values():
                        yield from walk(child)
                elif isinstance(value, list):
                    for child in value:
                        yield from walk(child)

            for record in records:
                yield from walk(record)
            return
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    yield raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_dir")
    parser.add_argument("out")
    parser.add_argument("--size", type=int, default=47_149)
    parser.add_argument("--corpus", nargs="+", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, trust_remote_code=True, local_files_only=True
    )
    vocab_size = len(tokenizer)
    if not 256 < args.size < vocab_size:
        raise ValueError(f"size must be between 257 and {vocab_size - 1}")

    counts = collections.Counter()
    chars = 0
    for corpus_path in args.corpus:
        for text in texts_from(corpus_path):
            if len(text) < 20:
                continue
            chars += len(text)
            for offset in range(0, len(text), 200_000):
                counts.update(
                    tokenizer.encode(
                        text[offset : offset + 200_000], add_special_tokens=False
                    )
                )

    required = set(range(min(256, vocab_size)))
    required.update(int(token_id) for token_id in tokenizer.all_special_ids)
    required.update(int(token_id) for token_id in tokenizer.added_tokens_decoder)
    chosen = set(required)
    for token_id, _count in counts.most_common():
        if len(chosen) >= args.size:
            break
        chosen.add(int(token_id))
    for token_id in range(vocab_size):
        if len(chosen) >= args.size:
            break
        chosen.add(token_id)

    ids = np.asarray(sorted(chosen), dtype=np.int32)
    if ids.size != args.size:
        raise RuntimeError(f"built {ids.size} ids, expected {args.size}")
    np.save(args.out, ids)

    total = sum(counts.values())
    covered = sum(count for token_id, count in counts.items() if token_id in chosen)
    coverage = 100.0 * covered / max(1, total)
    print(f"tokenizer vocabulary: {vocab_size}")
    print(f"corpus: {chars / 1e6:.2f} M chars, {total / 1e6:.2f} M tokens")
    print(f"draft vocabulary: {ids.size} ids, corpus coverage {coverage:.4f}%")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
