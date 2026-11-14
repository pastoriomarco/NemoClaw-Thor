#!/usr/bin/env python3
"""Build or verify a reduced MTP draft vocabulary for Qwen4Exp (QWEN4EXP_DRAFT_VOCAB).

The target model verifies every proposed token with its complete vocabulary, so
the draft vocabulary affects acceptance and speed only, never the output.
Ranking: corpus token frequency, after byte tokens (ids < 256), special and
added tokens; any shortfall is filled in tokenizer-id order (BPE merge order is
a frequency proxy). The result is deterministic for a given corpus and tokenizer.

Build (e.g. a ManyForge vocabulary):
    build_draft_vocab.py TOKENIZER_DIR OUT.npy --size 47232 --corpus DIR_OR_FILE...
Verify an existing asset:
    build_draft_vocab.py TOKENIZER_DIR ASSET.npy --verify [--corpus ...]

A size that is a multiple of 128 avoids the runtime top-up to the next multiple.
"""

import argparse
import collections
import glob
import json
import os

import numpy as np
from transformers import AutoTokenizer

TEXT_SUFFIXES = {"json", "jsonl", "md", "txt", "py", "log", "rs", "ts", "js", "sh"}


def texts_from(path: str):
    if os.path.isdir(path):
        for filename in sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True)):
            if (
                os.path.isfile(filename)
                and os.path.getsize(filename) < 20_000_000
                and filename.rsplit(".", 1)[-1] in TEXT_SUFFIXES
            ):
                yield from texts_from(filename)
        return

    raw = open(path, "rb").read().decode("utf-8", errors="ignore")
    if path.endswith((".json", ".jsonl")):
        try:
            if path.endswith(".jsonl"):
                records = [json.loads(line) for line in raw.splitlines() if line.strip()]
            else:
                records = [json.loads(raw)]
        except json.JSONDecodeError:
            yield raw
            return

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
    yield raw


def count_tokens(tokenizer, corpus: list[str]) -> collections.Counter:
    counts = collections.Counter()
    chars = 0
    for corpus_path in corpus:
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
    total = sum(counts.values())
    print(f"corpus: {chars / 1e6:.2f} M chars, {total / 1e6:.2f} M tokens")
    return counts


def required_ids(tokenizer, vocab_size: int) -> set[int]:
    required = set(range(min(256, vocab_size)))
    required.update(int(token_id) for token_id in tokenizer.all_special_ids)
    required.update(int(token_id) for token_id in tokenizer.added_tokens_decoder)
    return required


def verify(tokenizer, ids: np.ndarray, counts: collections.Counter | None) -> None:
    vocab_size = len(tokenizer)
    unique = np.unique(ids)
    print(f"tokenizer vocabulary: {vocab_size}")
    print(f"asset: {ids.size} ids ({unique.size} unique), range {ids.min()}..{ids.max()}")
    if unique.size != ids.size or ids.min() < 0 or ids.max() >= vocab_size:
        raise SystemExit("invalid: duplicate or out-of-range ids")
    kept = set(unique.tolist())
    missing = required_ids(tokenizer, vocab_size) - kept
    if missing:
        raise SystemExit(f"invalid: {len(missing)} byte/special/added ids missing")
    for word in ("def", " return", " self", "import", "\n", "<think>", "<tool_call>"):
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        print(f"  {word!r}: {token_ids} kept: {set(token_ids) <= kept}")
    if counts:
        covered = sum(count for token_id, count in counts.items() if token_id in kept)
        print(f"corpus coverage {100.0 * covered / max(1, sum(counts.values())):.4f}%")
    print("valid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_dir")
    parser.add_argument("path", help="output .npy (build) or asset (--verify)")
    parser.add_argument("--size", type=int, default=47_232)
    parser.add_argument("--corpus", nargs="+", default=[])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    counts = count_tokens(tokenizer, args.corpus) if args.corpus else None
    if args.verify:
        verify(tokenizer, np.load(args.path), counts)
        return
    if counts is None:
        raise SystemExit("--corpus is required to build a vocabulary")

    vocab_size = len(tokenizer)
    if not 256 < args.size < vocab_size:
        raise SystemExit(f"--size must be between 257 and {vocab_size - 1}")
    chosen = required_ids(tokenizer, vocab_size)
    for token_id, _count in counts.most_common():
        if len(chosen) >= args.size:
            break
        chosen.add(int(token_id))
    for token_id in range(vocab_size):
        if len(chosen) >= args.size:
            break
        chosen.add(token_id)

    ids = np.asarray(sorted(chosen), dtype=np.int32)
    np.save(args.path, ids)
    print(f"saved {ids.size} ids to {args.path}")
    verify(tokenizer, ids, counts)


if __name__ == "__main__":
    main()
