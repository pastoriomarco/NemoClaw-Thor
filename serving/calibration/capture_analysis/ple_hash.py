"""NumPy re-implementation of the Qwen4Exp PLE n-gram row ids (vLLM 0.33).

Mirrors ``vllm/models/qwen4_exp/nvidia/ngram_embedding.py``
(``Qwen4ExpNGramEmbedding._make_layer_multipliers``, ``_make_vocab_layout``,
``compute_ngram_ids``) and the Triton kernel ``_ple_ngram_ids_kernel`` in
``common/ops/ple.py``. Row ids are a pure function of the token sequence:

    head h of n-gram order k (k = 2..ngram_size, heads_per_ngram heads each):
        mixed_k = xor_{s=0..k-1} token[p - s] * multiplier[s]
        id      = offset[h] + mixed_k % size[h]

Predecessors before the sequence start, and every predecessor behind the
first ``eos_token_id`` met walking back from ``p - 1``, are replaced by
``eos_token_id``. All heads index one concatenated table of
``round_up(sum(size), make_ngram_vocab_size_divisible_by)`` rows; the
checkpoint splits that table row-wise into ``split_ngram_parts`` shards of
``ceil(rows / split_ngram_parts)`` rows (shard = id // shard_size).
"""

from __future__ import annotations

import numpy as np

MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
PLE_LAYER_PRIME = 10007


def splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent, shifts = value - 1, 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


def round_up(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor * divisor


class PLEHasher:
    """Row-id computation for one PLE layer, derived from the text config."""

    def __init__(self, text_config: dict, ple_dense_layer_id: int = 0):
        self.ngram_size = int(text_config["ngram_size"])
        self.heads_per_ngram = int(text_config["heads_per_ngram"])
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.eos_token_id = int(text_config["eos_token_id"])
        self.unigram_vocab_size = int(text_config["vocab_size"])
        self.split_ngram_parts = int(text_config.get("split_ngram_parts", 512))
        self.embed_dim = int(text_config["ple_embed_dim"])
        self.head_dim = self.embed_dim // self.ngram_heads
        seed = int(text_config.get("seed", 1234))

        max_multiplier = ((1 << 63) - 1) // self.unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = seed + PLE_LAYER_PRIME * ple_dense_layer_id
        self.multipliers = [
            2 * (splitmix64(base_seed + SPLITMIX_GAMMA * (i + 1)) % half_bound) + 1
            for i in range(self.ngram_size)
        ]

        sizes, offsets, offset = [], [], 0
        base = int(text_config["ngram_vocab_size_base"])
        for local_head in range(self.ngram_heads):
            global_head = ple_dense_layer_id * self.ngram_heads + local_head
            size = nth_prime_after(base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        self.sizes, self.offsets = sizes, offsets
        self.total_rows = offset
        self.padded_rows = round_up(offset, int(text_config["make_ngram_vocab_size_divisible_by"]))
        self.shard_size = (self.padded_rows + self.split_ngram_parts - 1) // self.split_ngram_parts
        # Products token * multiplier fit in int63 by construction (max_multiplier).
        self._mult = np.asarray(self.multipliers, dtype=np.int64)
        self._sizes = np.asarray(sizes, dtype=np.int64)
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._order = np.repeat(np.arange(2, self.ngram_size + 1), self.heads_per_ngram)
        assert self.padded_rows < 2**32, "row ids no longer fit in uint32"

    def ids(self, tokens: np.ndarray, positions: np.ndarray | None = None) -> np.ndarray:
        """Row ids, shape (len(positions), ngram_heads), uint32.

        ``tokens`` is the request's full token sequence (all_token_ids), so the
        n-gram context of position p is tokens[p-1], tokens[p-2], ...
        """
        tokens = np.asarray(tokens, dtype=np.int64)
        if positions is None:
            positions = np.arange(tokens.shape[0], dtype=np.int64)
        positions = np.asarray(positions, dtype=np.int64)
        eos = self.eos_token_id
        current = tokens[positions]
        mixed_by_order = {}
        mixed = current * self._mult[0]
        crossed = np.zeros(positions.shape[0], dtype=bool)
        for shift in range(1, self.ngram_size):
            source = positions - shift
            candidate = np.where(source >= 0, tokens[np.maximum(source, 0)], eos)
            candidate = np.where(crossed, eos, candidate)
            crossed |= candidate == eos
            mixed = np.bitwise_xor(mixed, candidate * self._mult[shift])
            mixed_by_order[shift + 1] = mixed
        out = np.empty((positions.shape[0], self.ngram_heads), dtype=np.uint32)
        for head in range(self.ngram_heads):
            m = mixed_by_order[int(self._order[head])]
            out[:, head] = np.remainder(m, self._sizes[head]) + self._offsets[head]
        return out

    def describe(self) -> dict:
        return {
            "ngram_size": self.ngram_size,
            "heads_per_ngram": self.heads_per_ngram,
            "ngram_heads": self.ngram_heads,
            "head_dim": self.head_dim,
            "eos_token_id": self.eos_token_id,
            "multipliers": [int(m) for m in self.multipliers],
            "head_sizes": [int(s) for s in self.sizes],
            "head_offsets": [int(o) for o in self.offsets],
            "total_rows": self.total_rows,
            "padded_rows": self.padded_rows,
            "split_ngram_parts": self.split_ngram_parts,
            "shard_size": self.shard_size,
        }
