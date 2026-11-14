#!/usr/bin/env python3
"""Offline analysis of captured agent traffic: PLE n-gram row usage and MTP draft-vocab coverage.

Streams ``NNNNNN.request.json`` (+ optional ``NNNNNN.reply.sse``) dumps in file
order, rebuilds prompt and reply token sequences (render.py), derives the PLE
row ids of every looked-up position (ple_hash.py) and counts them in dense
per-row counters, and collects generated tokens for the vocabulary analysis.
Writes REPORT.md, summary.json, CSVs and small artifacts to --out.

See README.md for the method and its approximations.
"""

from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ple_hash import PLEHasher  # noqa: E402
import render  # noqa: E402

GIB = 2**30
GIB_POINTS = [0.25, 0.5, 1, 2, 4, 8, 16]
ROW_POINTS = [10_000, 100_000, 1_000_000, 10_000_000]
FRACTIONS = [0.5, 0.8, 0.9, 0.95, 0.99]
DRAFT_ALIGN = 128
SLICE = 8 * 2**20

# ----------------------------------------------------------------------------
# Dense per-row counters (uint16 + sparse overflow), buffered.


class RowCounter:
    def __init__(self, rows: int, flush_ids: int = 24 * 2**20):
        self.counts = np.zeros(rows, dtype=np.uint16)  # calloc: untouched pages stay unmapped
        self.ov_ids = np.empty(0, dtype=np.int64)
        self.ov_cnt = np.empty(0, dtype=np.int64)
        self.buffer: list[np.ndarray] = []
        self.buffered = 0
        self.flush_ids = flush_ids
        self.total = 0

    def add(self, ids: np.ndarray) -> None:
        if ids.size == 0:
            return
        self.buffer.append(ids.reshape(-1))
        self.buffered += ids.size
        self.total += ids.size
        if self.buffered >= self.flush_ids:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        ids, cnt = np.unique(np.concatenate(self.buffer), return_counts=True)
        self.buffer, self.buffered = [], 0
        cur = self.counts[ids].astype(np.int64) + cnt
        over = cur > 65535
        if over.any():
            self.ov_ids = np.concatenate([self.ov_ids, ids[over].astype(np.int64)])
            self.ov_cnt = np.concatenate([self.ov_cnt, cur[over]])
            self.ov_ids, inv = np.unique(self.ov_ids, return_inverse=True)
            self.ov_cnt = np.bincount(inv, weights=self.ov_cnt).astype(np.int64)
            cur[over] = 0
        self.counts[ids] = cur.astype(np.uint16)

    def slice(self, lo: int, hi: int) -> np.ndarray:
        out = self.counts[lo:hi].astype(np.int64)
        if self.ov_ids.size:
            a, b = np.searchsorted(self.ov_ids, [lo, hi])
            out[self.ov_ids[a:b] - lo] += self.ov_cnt[a:b]
        return out

    def get(self, ids: np.ndarray) -> np.ndarray:
        out = self.counts[ids].astype(np.int64)
        if self.ov_ids.size:
            pos = np.searchsorted(self.ov_ids, ids)
            pos = np.minimum(pos, self.ov_ids.size - 1)
            hit = self.ov_ids[pos] == ids
            out[hit] += self.ov_cnt[pos[hit]]
        return out


def summed(counters):
    return lambda lo, hi: sum(c.slice(lo, hi) for c in counters)


def summed_get(counters):
    return lambda ids: sum(c.get(ids) for c in counters)


# ----------------------------------------------------------------------------
# Coverage curves from count-value groups (exact for ties: expected value under
# random tie-breaking).


class Ranking:
    """Rows ranked by one count array; evaluated against one or more count arrays."""

    def __init__(self, rows: int, rank_fn, eval_fns: dict):
        groups: dict[int, list] = {}
        self.eval_total = {k: 0 for k in eval_fns}
        self.eval_unique = {k: 0 for k in eval_fns}
        self.rank_unique = 0
        self.rank_total = 0
        for lo in range(0, rows, SLICE):
            hi = min(rows, lo + SLICE)
            r = rank_fn(lo, hi)
            evals = {k: (r if fn is rank_fn else fn(lo, hi)) for k, fn in eval_fns.items()}
            for k, e in evals.items():
                self.eval_total[k] += int(e.sum())
                self.eval_unique[k] += int(np.count_nonzero(e))
            nz = np.flatnonzero(r)
            if nz.size == 0:
                continue
            vals = r[nz]
            self.rank_unique += nz.size
            self.rank_total += int(vals.sum())
            u, inv = np.unique(vals, return_inverse=True)
            nrows = np.bincount(inv)
            sums = {k: np.bincount(inv, weights=e[nz]) for k, e in evals.items()}
            for i, v in enumerate(u.tolist()):
                g = groups.setdefault(v, [0] + [0.0] * len(eval_fns))
                g[0] += int(nrows[i])
                for j, k in enumerate(eval_fns):
                    g[j + 1] += float(sums[k][i])
        order = sorted(groups, reverse=True)
        self.values = np.asarray(order, dtype=np.int64)
        self.rows = np.asarray([groups[v][0] for v in order], dtype=np.float64)
        self.cum_rows = np.cumsum(self.rows)
        self.keys = list(eval_fns)
        self.sums = {k: np.asarray([groups[v][j + 1] for v in order]) for j, k in enumerate(self.keys)}
        self.cum = {k: np.cumsum(s) for k, s in self.sums.items()}

    def coverage(self, key: str, n_rows: float) -> float:
        total = self.eval_total[key]
        if total == 0 or self.values.size == 0:
            return float("nan")
        if n_rows >= self.cum_rows[-1]:
            return self.cum[key][-1] / total
        i = int(np.searchsorted(self.cum_rows, n_rows, side="left"))
        before_rows = self.cum_rows[i - 1] if i else 0.0
        before = self.cum[key][i - 1] if i else 0.0
        frac = (n_rows - before_rows) / self.rows[i]
        return (before + frac * self.sums[key][i]) / total

    def rows_for(self, key: str, fraction: float) -> float:
        total = self.eval_total[key]
        if total == 0 or self.values.size == 0:
            return float("nan")
        target = fraction * total
        cum = self.cum[key]
        if target > cum[-1]:
            return float("inf")  # not reachable with rows seen in the ranking set
        i = int(np.searchsorted(cum, target, side="left"))
        before_rows = self.cum_rows[i - 1] if i else 0.0
        before = cum[i - 1] if i else 0.0
        if self.sums[key][i] <= 0:
            return before_rows
        return before_rows + (target - before) / self.sums[key][i] * self.rows[i]

    def threshold(self, n_rows: float) -> tuple[int, float]:
        """(v, frac): rows with count > v are in the top-N, rows == v with probability frac."""
        if self.values.size == 0:
            return (1 << 62, 0.0)
        if n_rows >= self.cum_rows[-1]:
            return (0, 1.0)  # every ranked row (count >= 1); count 0 rows are not in
        i = int(np.searchsorted(self.cum_rows, n_rows, side="left"))
        before_rows = self.cum_rows[i - 1] if i else 0.0
        return (int(self.values[i]), (n_rows - before_rows) / self.rows[i])


def membership(counts: np.ndarray, threshold: tuple[int, float]) -> np.ndarray:
    v, frac = threshold
    if v == 0:
        return (counts > 0).astype(np.float64)
    return np.where(counts > v, 1.0, np.where(counts == v, frac, 0.0))


def rows_of_gib(gib: float) -> int:
    return int(gib * GIB / 90)


# ----------------------------------------------------------------------------
# Loading one request (+ reply), tokenised incrementally per conversation.


def load_body(path: str):
    try:
        body = json.load(open(path, "rb"))
    except Exception as exc:
        return None, f"json: {exc!r}"[:200]
    if not isinstance(body, dict) or "messages" not in body:
        return None, "not a chat/completions request"
    return body, None


def load_request(R: render.Renderer, path: str, body: dict, ref: dict | None, full_check: bool, stats) -> dict:
    name = os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path).split(".")[0]
    out = {"path": path, "name": name, "effort": effort_of(body),
           "family": render.family_of(body), "n_messages": len(body.get("messages") or []),
           "stream": bool(body.get("stream"))}
    try:
        text = R.prompt_text(body)
        prompt, state = R.encode_incremental(text, ref)
        if full_check:
            exact = R.prompt_ids(body)
            stats["incremental_checked"] += 1
            if not np.array_equal(exact, prompt):
                stats["incremental_mismatch"] += 1
                prompt, state = R.encode_incremental(text, None)
    except Exception as exc:
        out["error"] = f"render: {exc!r}"[:300]
        return out
    out["prompt"], out["enc_state"] = prompt, state
    out["thinking"] = R.thinking_open(prompt)
    reply_path = path[: -len(".request.json")] + ".reply.sse"
    reply = render.parse_sse(reply_path) if os.path.exists(reply_path) else None
    if reply is not None:
        gen = R.reply_ids(reply, out["thinking"])
        out["gen"] = gen
        out["labels"] = R.label(gen, out["thinking"])
        out["usage"] = reply["usage"]
        out["finish"] = reply["finish_reason"]
    return out


EFFORTS = ["xhigh", "medium", "low", "none"]


def effort_of(body: dict) -> str:
    """Reasoning effort the template sees: chat_template_kwargs, then the request field; template default xhigh."""
    kwargs = body.get("chat_template_kwargs") or {}
    if kwargs.get("enable_thinking") is False or body.get("reasoning_effort") == "none":
        return "none"
    effort = kwargs.get("reasoning_effort") or body.get("reasoning_effort") or "xhigh"
    return effort if effort in EFFORTS else "xhigh"


def stream_key(stream: int, body: dict) -> str:
    """Task key; dump folders are separate request streams, so tasks never cross them."""
    return f"{stream}:{render.conversation_key(body)}"


def assign_halves(files: list[tuple[int, str]], min_tasks: int) -> tuple[dict, str]:
    """Split tasks (conversation keys) into two halves of similar dump size, greedily.

    With fewer than ``min_tasks`` distinct tasks the split is by time instead
    (first half of the requests = A), and the returned map is empty.
    """
    weight = collections.Counter()
    for stream, path in files:
        body, err = load_body(path)
        if body is not None:
            weight[stream_key(stream, body)] += os.path.getsize(path)
    if len(weight) < min_tasks:
        return {}, "time"
    halves, load = {}, [0, 0]
    for key, w in sorted(weight.items(), key=lambda kv: (-kv[1], kv[0])):
        h = 0 if load[0] <= load[1] else 1
        halves[key] = h
        load[h] += w
    return halves, "task"


# ----------------------------------------------------------------------------


class Spool:
    """Lookup stream for learning_curve.py / ple_cache_sim.py, in arrival order.

    tokens.bin: int32 tokens; segments.npy: one record per prefill suffix or
    decode run. Lookup positions of a segment are its tokens [lead:], the
    ``lead`` (<= 2) tokens before them are only n-gram context; a segment at
    the sequence start has lead < 2 (EOS context, as in ple_hash).
    """

    DTYPE = np.dtype([("off", np.int64), ("n", np.int32), ("lead", np.int8), ("kind", np.uint8),
                      ("req", np.int32), ("task", np.int32), ("half", np.uint8), ("fam", np.int16),
                      ("eff", np.uint8)])

    def __init__(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.handle = open(os.path.join(path, "tokens.bin"), "wb")
        self.records = []
        self.tasks: dict[str, int] = {}
        self.task_first: list[int] = []
        self.task_fam: list[int] = []
        self.off = 0
        self.n_segments = 0
        self.n_tokens = 0

    def add(self, kind, seq, start, stop, reqno, key, half, fam, eff=0):
        if stop <= start:
            return
        lo = max(0, start - 2)
        chunk = np.ascontiguousarray(seq[lo:stop], dtype=np.int32)
        self.handle.write(chunk.tobytes())
        if key not in self.tasks:
            self.tasks[key] = len(self.tasks)
            self.task_first.append(reqno)
            self.task_fam.append(fam)
        self.records.append((self.off, chunk.size, start - lo, kind, reqno, self.tasks[key], half, fam, eff))
        self.off += chunk.size
        self.n_segments += 1
        self.n_tokens += stop - start

    def close(self, meta: dict):
        self.handle.close()
        np.save(os.path.join(self.path, "segments.npy"), np.asarray(self.records, dtype=self.DTYPE))
        meta = dict(meta, tasks=list(self.tasks), task_first_request=self.task_first, task_family=self.task_fam)
        with open(os.path.join(self.path, "meta.json"), "w") as handle:
            json.dump(meta, handle)


def lcp(a: np.ndarray, b: np.ndarray) -> int:
    m = min(a.shape[0], b.shape[0])
    if m == 0:
        return 0
    diff = np.flatnonzero(a[:m] != b[:m])
    return int(diff[0]) if diff.size else m


def pad_draft_vocab(ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """Same as the qwen4exp-draft-head runtime: unique ids topped up with the lowest unused ids."""
    ids = np.unique(ids.astype(np.int64))
    num_pad = -len(ids) % DRAFT_ALIGN
    if num_pad:
        unused = np.setdiff1d(np.arange(vocab_size), ids, assume_unique=True)
        ids = np.union1d(ids, unused[:num_pad])
    return ids


def load_builder(path: str):
    spec = importlib.util.spec_from_file_location("build_draft_vocab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_vocab(size: int, required: np.ndarray, score: np.ndarray, vocab_size: int) -> np.ndarray:
    """Required ids, then ids by descending score (ties by id), then id order."""
    chosen = np.zeros(vocab_size, dtype=bool)
    chosen[required] = True
    need = size - int(chosen.sum())
    order = np.lexsort((np.arange(vocab_size), -score))
    order = order[(score[order] > 0) & ~chosen[order]]
    take = order[:need]
    chosen[take] = True
    need = size - int(chosen.sum())
    if need > 0:
        rest = np.flatnonzero(~chosen)[:need]
        chosen[rest] = True
    return np.flatnonzero(chosen).astype(np.int32)


def fmt_pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.2f}%"


def fmt_rows(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if math.isinf(x):
        return "not reached"
    size = x * 90
    return f"{x / 1e6:.3f} M ({size / GIB:.2f} GiB)" if size >= GIB else f"{x / 1e6:.3f} M ({size / 2**20:.1f} MiB)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--model", required=True, help="model dir (tokenizer, chat template, config.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draft-vocab", required=True, help="current draft vocabulary .npy")
    ap.add_argument("--draft-builder", required=True, help="path of build_draft_vocab.py")
    ap.add_argument("--corpus", nargs="*", default=[], help="coding corpus files/dirs (build_draft_vocab format)")
    ap.add_argument("--verify-every", type=int, default=25,
                    help="every k-th request is also tokenised in full and compared with the incremental tokens")
    ap.add_argument("--min-tasks-for-task-split", type=int, default=4)
    ap.add_argument("--cache-block", type=int, default=832, help="prefix-cache block of the server (prefix model)")
    ap.add_argument("--cache-prompts", type=int, default=256, help="recent prompts the prefix model can hit")
    ap.add_argument("--nocache-target-tokens", type=float, default=1e8,
                    help="prompt tokens sampled (every k-th request) for the no-cache curve")
    ap.add_argument("--family-sample-every", type=int, default=8)
    ap.add_argument("--hot-rows-max-gib", type=float, default=4.0)
    ap.add_argument("--vocab-sizes", type=int, nargs="+", default=[47232, 65536])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--spool", default="", help="write the lookup stream (tokens per prefill/decode segment) here")
    ap.add_argument("--spool-only", action="store_true", help="only tokenise and write the spool (fast)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    config = json.load(open(os.path.join(args.model, "config.json")))
    tc = config.get("text_config", config)
    hasher = PLEHasher(tc)
    R = hasher.padded_rows
    H = hasher.ngram_heads

    RR = render.Renderer(args.model)
    tokenizer = RR.tokenizer
    im_end = RR.ids["<|im_end|>"]
    vocab_size = int(tc["vocab_size"])

    files = []  # (stream, path): each dump folder is its own request stream, in folder then file order
    for stream, d in enumerate(args.dumps):
        files.extend((stream, f) for f in render.request_files(d))
    if args.limit:
        files = files[: args.limit]
    dump_bytes = sum(os.path.getsize(f) for _, f in files)
    est_tokens = dump_bytes / 3.6
    nocache_every = max(1, int(math.ceil(est_tokens / args.nocache_target_tokens)))
    halves, split = assign_halves(files, args.min_tasks_for_task_split)
    print(f"{len(files)} requests, {dump_bytes / 1e9:.2f} GB; split by {split} ({len(halves)} tasks); "
          f"no-cache sample every {nocache_every} request(s)", flush=True)

    pA, pB, dA, dB = (RowCounter(R) for _ in range(4))
    nocache = RowCounter(R)
    prefill = {0: pA, 1: pB}
    decode = {0: dA, 1: dB}

    convs: dict[str, dict] = {}
    conv_ids = collections.Counter()
    recent = collections.deque(maxlen=args.cache_prompts)  # (prompt, cache point) of recent requests
    global_enc = None
    families: dict[str | None, int] = {}
    fam_samples = {"ids": [], "fam": [], "half": [], "phase": [], "eff": []}
    gen_store = {"tok": [], "lab": [], "half": [], "fam": [], "eff": []}
    cur_eff = [0]  # effort index of the request whose lookups are being added
    records = []
    errors = []
    totals = collections.Counter()
    observed_cached = []

    spool = Spool(args.spool) if args.spool else None

    def add_decode(half, seq, start, end, fam_idx, reqno, key):
        """Lookups of the fed generated tokens seq[start : end-1] (the last one is never fed)."""
        if end - start < 2:
            return 0
        if spool:
            spool.add(1, seq, start, end - 1, reqno, key, half, fam_idx, cur_eff[0])
        if args.spool_only:
            return end - 1 - start
        positions = np.arange(start, end - 1, dtype=np.int64)
        ids = hasher.ids(seq, positions)
        decode[half].add(ids)
        sample(ids, positions, fam_idx, half, 1, reqno)
        return positions.size

    def sample(ids, positions, fam_idx, half, phase, reqno):
        sel = ((positions + reqno) % args.family_sample_every) == 0
        if sel.any():
            k = int(sel.sum())
            fam_samples["eff"].append(np.full(k, cur_eff[0], np.uint8))
            fam_samples["ids"].append(ids[sel])
            fam_samples["fam"].append(np.full(k, fam_idx, np.uint16))
            fam_samples["half"].append(np.full(k, half, np.uint8))
            fam_samples["phase"].append(np.full(k, phase, np.uint8))

    def add_gen(tokens, labels, half, fam_idx):
        gen_store["tok"].append(tokens.astype(np.int32))
        gen_store["lab"].append(labels.astype(np.uint8))
        gen_store["half"].append(np.full(tokens.shape[0], half, np.uint8))
        gen_store["fam"].append(np.full(tokens.shape[0], fam_idx, np.uint16))
        gen_store["eff"].append(np.full(tokens.shape[0], cur_eff[0], np.uint8))

    for reqno, (stream, path) in enumerate(files):
        body, err = load_body(path)
        if err:
            errors.append((path, err))
            continue
        key = stream_key(stream, body)
        n_messages = len(body.get("messages") or [])
        conv = convs.get(key)
        if conv is None or n_messages < conv["n_messages"]:
            conv_ids[key] += 1
            conv = {"id": f"{key}-{conv_ids[key]}", "key": key, "enc": None,
                    "pending": None, "n_messages": 0, "requests": 0}
            convs[key] = conv
            totals["conversations"] += 1
        ref = conv["enc"] if conv["enc"] is not None else global_enc
        res = load_request(RR, path, body, ref, reqno % args.verify_every == 0, totals)
        del body
        if "error" in res:
            errors.append((res["name"], res["error"]))
            continue
        prompt = res["prompt"]
        P = int(prompt.shape[0])
        fam_idx = families.setdefault(res["family"], len(families))
        eff_idx = EFFORTS.index(res["effort"])
        totals[f"requests_effort_{res['effort']}"] += 1
        h = halves.get(key, 0) if split == "task" else int(reqno >= len(files) // 2)
        conv["n_messages"] = n_messages
        conv["requests"] += 1
        conv["seen"] = reqno
        cur_eff[0] = eff_idx
        rec = {"request": res["name"], "conversation": conv["id"], "half": "AB"[h], "effort": res["effort"],
               "family": res["family"] or "", "prompt_tokens": P, "stream": res["stream"]}

        # A previous request of this conversation without a reply: recover its
        # generated tokens from this prompt's history.
        pend = conv["pending"]
        if pend is not None:
            Pp = pend["P"]
            common = lcp(prompt, pend["prompt"])
            if common >= Pp - 2:
                start = min(common, Pp)
                ends = np.flatnonzero(prompt[start:] == im_end)
                if ends.size:
                    end = start + int(ends[0]) + 1
                    gen = prompt[start:end]
                    labels = RR.label(gen, pend["thinking"])
                    if pend["thinking"] and (gen[:2] == RR.ids["</think>"]).any():
                        totals["recovered_without_reasoning"] += 1
                    cur_eff[0] = pend["eff"]
                    n_dec = add_decode(pend["half"], prompt, start, end, pend["fam"], pend["reqno"], key)
                    add_gen(gen, labels, pend["half"], pend["fam"])
                    totals["decode_tokens"] += n_dec
                    totals["gen_tokens"] += gen.shape[0]
                    totals["gen_recovered_requests"] += 1
                    pend["rec"]["gen_tokens"] = int(gen.shape[0])
                    pend["rec"]["gen_source"] = "next request history"
            conv["pending"] = None
        cur_eff[0] = eff_idx

        # Prefix cache: server-reported cached tokens, else the prefix model.
        # Observed on this server (hybrid Mamba model, block 832): a request
        # hits exactly at c = floor(len(Q) / block) * block of an earlier
        # prompt Q that it shares at least c tokens with; generated tokens and
        # other block boundaries are never reused (see README).
        cached_lcp = 0
        for q, c in recent:
            if cached_lcp < c <= P - 1 and np.array_equal(prompt[:c], q[:c]):
                cached_lcp = c
        usage = res.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cached_srv = details.get("cached_tokens") if usage else None
        if usage.get("prompt_tokens") is not None:
            totals["usage_requests"] += 1
            totals["prompt_tokens_mismatch"] += int(usage["prompt_tokens"] != P)
        if cached_srv is not None:
            cached = min(int(cached_srv), P - 1)
            observed_cached.append(int(cached_srv))
            totals["cache_server"] += 1
            totals["prefix_model_tokens_on_server_requests"] += P - cached_lcp
            totals["server_tokens_on_server_requests"] += P - cached
            totals["prefix_model_exact_requests"] += int(cached_lcp == cached)
        else:
            cached = cached_lcp
            totals["cache_prefix_model"] += 1
        positions = np.arange(cached, P, dtype=np.int64)
        if spool:
            spool.add(0, prompt, cached, P, reqno, key, h, fam_idx, cur_eff[0])
        if not args.spool_only:
            ids = hasher.ids(prompt, positions)
            prefill[h].add(ids)
            sample(ids, positions, fam_idx, h, 0, reqno)
        totals["prompt_tokens"] += P
        totals["prefill_tokens"] += positions.size
        totals["prefill_tokens_prefix_model"] += P - cached_lcp
        rec.update(cached_server=cached_srv, cached_prefix_model=cached_lcp, prefill_computed=int(positions.size))

        if reqno % nocache_every == 0 and not args.spool_only:
            nocache.add(hasher.ids(prompt))
            totals["nocache_sampled_tokens"] += P

        if "gen" in res:
            gen, labels = res["gen"], res["labels"]
            seq = np.concatenate([prompt, gen])
            n_dec = add_decode(h, seq, P, seq.shape[0], fam_idx, reqno, key)
            add_gen(gen, labels, h, fam_idx)
            totals["decode_tokens"] += n_dec
            totals["gen_tokens"] += gen.shape[0]
            totals["gen_reply_requests"] += 1
            ct = usage.get("completion_tokens")
            rec.update(gen_tokens=int(gen.shape[0]), gen_source="reply", completion_tokens=ct,
                       finish=res.get("finish"),
                       reasoning_tokens_server=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                       reasoning_tokens_rebuilt=int((labels == render.LABEL_REASONING).sum()))
            if ct is not None:
                totals["completion_tokens_server"] += ct
                totals["completion_tokens_rebuilt"] += int(gen.shape[0])
                totals["completion_abs_diff"] += abs(int(gen.shape[0]) - ct)
        else:
            conv["pending"] = {"P": P, "prompt": prompt, "thinking": res["thinking"], "fam": fam_idx, "eff": eff_idx,
                               "half": h, "reqno": reqno, "rec": rec}
            rec["gen_source"] = "none"
        recent.append((prompt, (P // args.cache_block) * args.cache_block))
        conv["enc"] = res["enc_state"]
        global_enc = res["enc_state"]
        records.append(rec)
        totals["requests"] += 1
        # Keep token state for the most recent conversations only (bounded memory).
        live = [c for c in convs.values() if c["enc"] is not None]
        if len(live) > 128:
            for c in sorted(live, key=lambda c: c["seen"])[: len(live) - 128]:
                c["enc"], c["pending"] = None, None
        if reqno % 200 == 0:
            print(f"  {reqno}/{len(files)} requests, {time.time() - t0:.0f}s", flush=True)
    for c in (pA, pB, dA, dB, nocache):
        c.flush()
    totals["requests_without_gen"] = sum(1 for r in records if r.get("gen_source") == "none")
    print(f"tokenised in {time.time() - t0:.0f}s", flush=True)
    if spool:
        spool.close({"families": {str(i): (f if f is not None else "untagged") for f, i in families.items()},
                     "efforts": EFFORTS,
                     "split": split, "requests": len(files)})
        print(f"spool: {spool.n_segments} segments, {spool.n_tokens:,} tokens -> {args.spool}", flush=True)
    if args.spool_only:
        return

    summary = {"totals": dict(totals), "errors": errors[:50], "n_errors": len(errors),
               "hasher": hasher.describe(), "ngram_heads": H, "nocache_every": nocache_every,
               "split": split, "tasks": len(halves)}
    if observed_cached:
        g = 0
        for v in observed_cached:
            g = math.gcd(g, v)
        summary["observed_cache_gcd"] = g

    with open(os.path.join(args.out, "requests.csv"), "w", newline="") as handle:
        keys = ["request", "conversation", "half", "family", "effort", "stream", "prompt_tokens", "cached_server",
                "cached_prefix_model", "prefill_computed", "gen_source", "gen_tokens", "completion_tokens",
                "reasoning_tokens_server", "reasoning_tokens_rebuilt", "finish"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in keys})

    # ------------------------------------------------------------------ PLE
    print("PLE coverage...", flush=True)
    ple = {"lookups": {"prefill": pA.total + pB.total, "decode": dA.total + dB.total,
                       "prefill_A": pA.total, "prefill_B": pB.total, "decode_A": dA.total, "decode_B": dB.total,
                       "nocache_sampled": nocache.total}}
    pf, dc, cb = summed([pA, pB]), summed([dA, dB]), summed([pA, pB, dA, dB])
    cA, cB = summed([pA, dA]), summed([pB, dB])
    rankings = {
        "prefill": Ranking(R, pf, {"self": pf}),
        "decode": Ranking(R, dc, {"self": dc}),
        "combined": Ranking(R, cb, {"self": cb}),
    }
    if nocache.total:
        nc = summed([nocache])
        rankings["nocache_sampled"] = Ranking(R, nc, {"self": nc})
    holdout = {
        "A->B": Ranking(R, cA, {"combined": cB, "prefill": summed([pB]), "decode": summed([dB])}),
        "B->A": Ranking(R, cB, {"combined": cA, "prefill": summed([pA]), "decode": summed([dA])}),
    }
    holdout_phase = {
        "prefill A->B": Ranking(R, summed([pA]), {"prefill": summed([pB])}),
        "prefill B->A": Ranking(R, summed([pB]), {"prefill": summed([pA])}),
        "decode A->B": Ranking(R, summed([dA]), {"decode": summed([dB])}),
        "decode B->A": Ranking(R, summed([dB]), {"decode": summed([dA])}),
    }
    curve_points = [("rows", n) for n in ROW_POINTS] + [("GiB", g) for g in GIB_POINTS]

    def npoints(n_or_g):
        kind, v = n_or_g
        return v if kind == "rows" else rows_of_gib(v)

    insample = {}
    for name, rk in rankings.items():
        insample[name] = {
            "unique_rows": rk.rank_unique, "lookups": rk.rank_total,
            "coverage": {f"{v} {kind}": rk.coverage("self", npoints((kind, v))) for kind, v in curve_points},
            "rows_for": {f"{int(f * 100)}%": rk.rows_for("self", f) for f in FRACTIONS},
        }
    ho = {}
    for name, rk in holdout.items():
        ho[name] = {"rank_unique_rows": rk.rank_unique}
        for key in rk.keys:
            ho[name][key] = {
                "lookups": rk.eval_total[key], "unique_rows": rk.eval_unique[key],
                "coverage": {f"{v} {kind}": rk.coverage(key, npoints((kind, v))) for kind, v in curve_points},
                "rows_for": {f"{int(f * 100)}%": rk.rows_for(key, f) for f in FRACTIONS},
                "ceiling": rk.coverage(key, float(R)),
            }
    for name, rk in holdout_phase.items():
        key = rk.keys[0]
        ho[name] = {key: {
            "lookups": rk.eval_total[key],
            "coverage": {f"{v} {kind}": rk.coverage(key, npoints((kind, v))) for kind, v in curve_points},
            "rows_for": {f"{int(f * 100)}%": rk.rows_for(key, f) for f in FRACTIONS},
            "ceiling": rk.coverage(key, float(R)),
        }}
    ple["insample"] = insample
    ple["holdout"] = ho

    # Dense curve CSV (log-spaced N).
    grid = sorted(set(np.unique(np.round(np.logspace(3, math.log10(R), 60))).astype(np.int64).tolist()))
    with open(os.path.join(args.out, "ple_coverage_curves.csv"), "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["top_N_rows", "GiB", "prefill", "decode", "combined", "nocache_sampled",
                    "holdout_A->B_combined", "holdout_B->A_combined"])
        for n in grid:
            w.writerow([n, f"{n * 90 / GIB:.4f}"]
                       + [f"{rankings[k].coverage('self', n):.6f}" if k in rankings else ""
                          for k in ("prefill", "decode", "combined", "nocache_sampled")]
                       + [f"{holdout['A->B'].coverage('combined', n):.6f}",
                          f"{holdout['B->A'].coverage('combined', n):.6f}"])

    # Hot-row id file: top rows of the combined ranking.
    comb = rankings["combined"]
    k_max = min(rows_of_gib(args.hot_rows_max_gib), comb.rank_unique)
    if k_max > 0:
        v, frac = comb.threshold(k_max)
        need_eq = int(round(frac * (comb.rows[np.searchsorted(-comb.values, -v)] if v else 0)))
        ids_hi, cnt_hi, ids_eq = [], [], []
        for lo in range(0, R, SLICE):
            c = cb(lo, min(R, lo + SLICE))
            sel = np.flatnonzero(c > v) if v else np.flatnonzero(c > 0)
            ids_hi.append((sel + lo).astype(np.uint32))
            cnt_hi.append(c[sel])
            if v and need_eq > sum(x.size for x in ids_eq):
                ids_eq.append((np.flatnonzero(c == v) + lo).astype(np.uint32))
        ids_hi = np.concatenate(ids_hi)
        cnt_hi = np.concatenate(cnt_hi)
        order = np.argsort(-cnt_hi, kind="stable")
        hot = ids_hi[order]
        if ids_eq:
            hot = np.concatenate([hot, np.concatenate(ids_eq)[:need_eq]])
        hot = hot[:k_max]
        np.save(os.path.join(args.out, "ple_hot_rows_top.npy"), hot)
        ple["hot_rows_file"] = {"rows": int(hot.size), "GiB_of_table": hot.size * 90 / GIB,
                                "coverage_combined_in_sample": comb.coverage("self", hot.size)}
        shard = (hot // hasher.shard_size).astype(np.int64)
        ple["hot_rows_file"]["per_shard_min_max"] = [int(np.bincount(shard, minlength=hasher.split_ngram_parts).min()),
                                                     int(np.bincount(shard, minlength=hasher.split_ngram_parts).max())]

    # Per-head share of lookups served by the combined top 1 GiB (heads 0-7 bigram, 8-15 trigram).
    # Per family (sampled positions).
    fam_names = {i: (f if f is not None else "untagged") for f, i in families.items()}
    fam_report = {}
    if fam_samples["ids"]:
        s_ids = np.concatenate(fam_samples["ids"])  # (n, H)
        s_fam = np.concatenate(fam_samples["fam"])
        s_half = np.concatenate(fam_samples["half"])
        s_phase = np.concatenate(fam_samples["phase"])
        s_eff = np.concatenate(fam_samples["eff"])
        flat = s_ids.reshape(-1).astype(np.int64)
        c_all = summed_get([pA, pB, dA, dB])(flat).reshape(s_ids.shape)
        c_A = summed_get([pA, dA])(flat).reshape(s_ids.shape)
        c_B = summed_get([pB, dB])(flat).reshape(s_ids.shape)
        head_share = {}
        for g in GIB_POINTS:
            n = rows_of_gib(g)
            m_in = membership(c_all, comb.threshold(n))
            m_ho = np.where(s_half[:, None] == 1, membership(c_A, holdout["A->B"].threshold(n)),
                            membership(c_B, holdout["B->A"].threshold(n)))
            if g == 1:
                head_share = {"in_sample": m_in.mean(axis=0).round(4).tolist(),
                              "holdout": m_ho.mean(axis=0).round(4).tolist()}
            groups = [(fname, s_fam == fi) for fi, fname in fam_names.items()]
            groups += [(f"effort {e}", s_eff == ei) for ei, e in enumerate(EFFORTS) if (s_eff == ei).any()]
            for fname, gmask in groups:
                for phase_name, pmask in (("combined", None), ("prefill", 0), ("decode", 1)):
                    mask = gmask.copy()
                    if pmask is not None:
                        mask &= s_phase == pmask
                    if not mask.any():
                        continue
                    entry = fam_report.setdefault(fname, {}).setdefault(phase_name, {"sampled_positions": int(mask.sum())})
                    entry[f"{g} GiB in-sample"] = float(m_in[mask].mean())
                    entry[f"{g} GiB holdout"] = float(m_ho[mask].mean())
        ple["head_share_top_1GiB"] = head_share
    ple["families"] = fam_report
    summary["ple"] = ple
    del pA, pB, dA, dB, nocache, prefill, decode

    # ------------------------------------------------------------------ vocab
    print("draft vocabulary...", flush=True)
    vocab = {}
    if gen_store["tok"]:
        tok = np.concatenate(gen_store["tok"]).astype(np.int64)
        lab = np.concatenate(gen_store["lab"])
        half = np.concatenate(gen_store["half"])
        gfam = np.concatenate(gen_store["fam"])
        geff = np.concatenate(gen_store["eff"])
    else:
        tok = np.empty(0, np.int64)
        lab = half = gfam = geff = np.empty(0, np.uint8)
    builder = load_builder(args.draft_builder)
    required = np.asarray(sorted(builder.required_ids(tokenizer, len(tokenizer))), dtype=np.int64)
    current_raw = np.load(args.draft_vocab)
    current = pad_draft_vocab(current_raw, vocab_size)
    in_vocab = np.zeros(vocab_size, dtype=bool)
    in_vocab[current] = True

    def coverage_by_label(member: np.ndarray, mask: np.ndarray | None = None) -> dict:
        out = {}
        m = np.ones(tok.shape[0], bool) if mask is None else mask
        for li, lname in enumerate(render.LABEL_NAMES):
            sel = m & (lab == li)
            out[lname] = {"tokens": int(sel.sum()), "coverage": float(member[tok[sel]].mean()) if sel.any() else float("nan")}
        sel = m & (lab != render.LABEL_END)
        out["all_generated_excl_end"] = {"tokens": int(sel.sum()),
                                         "coverage": float(member[tok[sel]].mean()) if sel.any() else float("nan")}
        return out

    vocab["current"] = {"asset_ids": int(current_raw.size), "loaded_ids": int(current.size),
                        "coverage": coverage_by_label(in_vocab)}
    vocab["current_by_family"] = {fam_names[fi]: coverage_by_label(in_vocab, gfam == fi)["all_generated_excl_end"]
                                  for fi in fam_names if (gfam == fi).any()}
    vocab["by_effort"] = {}
    for ei, e in enumerate(EFFORTS):
        m = geff == ei
        if not m.any():
            continue
        cov = coverage_by_label(in_vocab, m)
        gen_excl = cov["all_generated_excl_end"]["tokens"]
        vocab["by_effort"][e] = {"coverage": cov, "reasoning_share": cov["reasoning"]["tokens"] / max(1, gen_excl),
                                 "requests": int(totals.get(f"requests_effort_{e}", 0))}
    # Top missing tokens.
    miss = ~in_vocab[tok]
    missing = []
    if miss.any():
        counts = np.bincount(tok[miss], minlength=vocab_size)
        by_label = {li: np.bincount(tok[miss & (lab == li)], minlength=vocab_size) for li in range(4)}
        top = np.argsort(-counts, kind="stable")[:200]
        total_gen = int((lab != render.LABEL_END).sum())
        for t in top:
            if counts[t] == 0:
                break
            missing.append({"id": int(t), "count": int(counts[t]), "share_of_generated": counts[t] / max(1, total_gen),
                            "reasoning": int(by_label[0][t]), "content": int(by_label[1][t]),
                            "tool_call": int(by_label[2][t]), "token": tokenizer.convert_ids_to_tokens(int(t)),
                            "text": tokenizer.decode([int(t)])})
        vocab["missing_unique_ids"] = int(np.count_nonzero(counts))
        vocab["missing_tokens"] = int(miss.sum())
    with open(os.path.join(args.out, "draft_vocab_missing_tokens.csv"), "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["id", "count", "share_of_generated", "reasoning", "content", "tool_call", "token", "text"])
        for m in missing:
            w.writerow([m["id"], m["count"], f"{m['share_of_generated']:.6f}", m["reasoning"], m["content"],
                        m["tool_call"], m["token"], repr(m["text"])])
    vocab["top_missing"] = missing[:60]

    # Corpus counts (same tokenisation as build_draft_vocab).
    corpus_counts = np.zeros(vocab_size, dtype=np.float64)
    if args.corpus:
        cc = builder.count_tokens(tokenizer, args.corpus)
        for t, c in cc.items():
            if t < vocab_size:
                corpus_counts[t] = c
        vocab["corpus_tokens"] = int(corpus_counts.sum())
        vocab["current_coverage_on_corpus"] = float(corpus_counts[in_vocab].sum() / max(1, corpus_counts.sum()))

    def capture_counts(mask):
        return np.bincount(tok[mask & (lab != render.LABEL_END)], minlength=vocab_size).astype(np.float64)

    def scores(cap):
        out = {"capture": cap + (corpus_counts / max(1.0, corpus_counts.sum())) * 1e-3}  # corpus breaks ties/fills
        if corpus_counts.sum():
            out["corpus_only"] = corpus_counts.copy()
            out["mixed_50_50"] = 0.5 * cap / max(1.0, cap.sum()) + 0.5 * corpus_counts / corpus_counts.sum()
        return out

    # Hold-out: build on one half (+ corpus), measure on the other.
    cand_eval = {}
    have_halves = tok.size and (half == 0).any() and (half == 1).any()
    for size in args.vocab_sizes:
        for train, test in ((0, 1), (1, 0)):
            if not have_halves:
                break
            sc = scores(capture_counts(half == train))
            for name, score in sc.items():
                ids = build_vocab(size, required, score, vocab_size)
                member = np.zeros(vocab_size, bool)
                member[ids] = True
                cov = coverage_by_label(member, half == test)
                e = cand_eval.setdefault(f"{name} {size}", {})
                for lname, v in cov.items():
                    agg = e.setdefault(lname, [0, 0.0])
                    agg[0] += v["tokens"]
                    agg[1] += v["coverage"] * v["tokens"] if v["tokens"] else 0.0
    vocab["candidates_holdout"] = {k: {l: (v[1] / v[0] if v[0] else float("nan")) for l, v in e.items()}
                                   for k, e in cand_eval.items()}
    if have_halves:
        cur_ho = {}
        for test in (0, 1):
            for lname, v in coverage_by_label(in_vocab, half == test).items():
                agg = cur_ho.setdefault(lname, [0, 0.0])
                agg[0] += v["tokens"]
                agg[1] += v["coverage"] * v["tokens"] if v["tokens"] else 0.0
        vocab["candidates_holdout"]["current (padded) 47232"] = {l: (v[1] / v[0] if v[0] else float("nan"))
                                                                for l, v in cur_ho.items()}

    # Final artifacts on all capture (+ corpus).
    artifacts = {}
    if tok.size:
        sc = scores(capture_counts(np.ones(tok.shape[0], bool)))
        for size in args.vocab_sizes:
            for name in ("capture", "mixed_50_50"):
                if name not in sc:
                    continue
                ids = build_vocab(size, required, sc[name], vocab_size)
                path = os.path.join(args.out, f"draft_vocab_{name}_{size}.npy")
                np.save(path, ids)
                member = np.zeros(vocab_size, bool)
                member[ids] = True
                ok = (np.unique(ids).size == ids.size and ids.min() >= 0 and ids.max() < vocab_size
                      and set(required.tolist()) <= set(ids.tolist()) and ids.size % DRAFT_ALIGN == 0)
                artifacts[os.path.basename(path)] = {
                    "ids": int(ids.size), "valid": bool(ok),
                    "in_sample_coverage": coverage_by_label(member)["all_generated_excl_end"]["coverage"],
                    "corpus_coverage": (float(corpus_counts[member].sum() / corpus_counts.sum())
                                        if corpus_counts.sum() else None),
                    "overlap_with_current": int(np.intersect1d(ids, current).size)}
    vocab["artifacts"] = artifacts
    summary["vocab"] = vocab
    summary["elapsed_s"] = time.time() - t0

    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_report(args, summary, fam_names)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}/REPORT.md", flush=True)


# ----------------------------------------------------------------------------


def write_report(args, s, fam_names):
    t = s["totals"]
    ple = s["ple"]
    v = s["vocab"]
    H = s["ngram_heads"]
    L = []
    a = L.append
    a("# Capture analysis: PLE row usage and MTP draft vocabulary\n")
    a(f"Dumps (separate request streams, in this order): {', '.join(args.dumps)}. Generated by `capture_analysis/analyze.py` in {s['elapsed_s']:.0f} s.\n")
    a("## Traffic\n")
    a("| | value |\n|---|---|")
    rows = [
        ("requests analysed", t.get("requests", 0)),
        ("requests skipped (errors)", s["n_errors"]),
        ("conversations", t.get("conversations", 0)),
        ("families", ", ".join(sorted(fam_names.values()))),
        ("prompt tokens, no prefix cache (upper bound)", f"{t.get('prompt_tokens', 0):,}"),
        ("prefill tokens computed (used model)", f"{t.get('prefill_tokens', 0):,}"),
        ("  of which requests with server-reported cached_tokens", t.get("cache_server", 0)),
        ("  of which requests on the prefix model", t.get("cache_prefix_model", 0)),
        ("prefill tokens computed, prefix model for all requests", f"{t.get('prefill_tokens_prefix_model', 0):,}"),
        ("prefix model vs server on server-reported requests (tokens)",
         f"{t.get('prefix_model_tokens_on_server_requests', 0):,} vs {t.get('server_tokens_on_server_requests', 0):,}"),
        ("generated tokens (reply SSE / recovered from next request)",
         f"{t.get('gen_tokens', 0):,} ({t.get('gen_reply_requests', 0)} / {t.get('gen_recovered_requests', 0)} requests)"),
        ("requests with unknown output (no reply, not recoverable)", t.get("requests_without_gen", 0)),
        ("decode lookup positions", f"{t.get('decode_tokens', 0):,}"),
        ("rendered prompt length != usage.prompt_tokens", f"{t.get('prompt_tokens_mismatch', 0)} of {t.get('usage_requests', 0)}"),
        ("rebuilt vs server completion tokens", f"{t.get('completion_tokens_rebuilt', 0):,} vs {t.get('completion_tokens_server', 0):,}"
         f" (sum of per-request abs diff {t.get('completion_abs_diff', 0):,})"),
        ("requests by reasoning effort", ", ".join(f"{e}: {t[f'requests_effort_{e}']}" for e in EFFORTS
                                                   if t.get(f"requests_effort_{e}"))),
        ("observed cached_tokens granularity (gcd)", s.get("observed_cache_gcd", "n/a")),
        ("prefix model == server cached_tokens", f"{t.get('prefix_model_exact_requests', 0)} of {t.get('cache_server', 0)} requests"),
        ("incremental tokenisation checked against full / mismatches",
         f"{t.get('incremental_checked', 0)} / {t.get('incremental_mismatch', 0)}"),
        ("hold-out split", f"by {s['split']}" + (f" ({s['tasks']} tasks)" if s["split"] == "task" else
                                                 " (too few distinct tasks: first half of requests = A)")),
    ]
    for k, val in rows:
        a(f"| {k} | {val} |")
    if s["errors"]:
        a("\nSkipped: " + "; ".join(f"{n}: {e}" for n, e in s["errors"][:10]))
    a("")

    a("## PLE n-gram table: which rows are looked up\n")
    a(f"Table: {s['hasher']['padded_rows']:,} rows x 90 B = {s['hasher']['padded_rows'] * 90 / GIB:.1f} GiB; "
      f"{H} lookups (heads) per position (heads 0-7 bigram, 8-15 trigram). "
      f"Lookups: prefill {ple['lookups']['prefill']:,}, decode {ple['lookups']['decode']:,}.\n")
    a("### Coverage in-sample: share of lookups served by the top-N rows (ranked on the same traffic)\n")
    keys = [f"{g} GiB" for g in GIB_POINTS] + [f"{n} rows" for n in ROW_POINTS]
    names = [k for k in ("prefill", "decode", "combined", "nocache_sampled") if k in ple["insample"]]
    a("| top-N | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for k in keys:
        n = rows_of_gib(float(k.split()[0])) if k.endswith("GiB") else int(k.split()[0])
        label = f"{k} ({n / 1e6:.2f} M rows)" if k.endswith("GiB") else f"{n:,} rows ({n * 90 / GIB * 1024:.1f} MiB)"
        a(f"| {label} | " + " | ".join(fmt_pct(ple["insample"][nm]["coverage"][k]) for nm in names) + " |")
    a("| unique rows | " + " | ".join(f"{ple['insample'][nm]['unique_rows']:,}" for nm in names) + " |")
    a("| lookups | " + " | ".join(f"{ple['insample'][nm]['lookups']:,}" for nm in names) + " |")
    for f in FRACTIONS:
        fk = f"{int(f * 100)}%"
        a(f"| rows for {fk} | " + " | ".join(fmt_rows(ple["insample"][nm]["rows_for"][fk]) for nm in names) + " |")
    a("\n`nocache_sampled`: all prompt tokens of every request with index divisible by "
      f"{s['nocache_every']} (no prefix cache: upper bound of prefill lookups).\n")

    a("### Hold-out (the number that matters): hot set built from one half of the conversations, measured on the other\n")
    if s["split"] == "task":
        a("Halves split by task (tools + system prompt + first user message), balanced by dump size. ")
    else:
        a("Too few distinct tasks for a task split: halves split by time (A = first half of the requests). "
          "This measures drift within a session, not generalisation to new sessions. ")
    a("`ceiling` = share of the other half's lookups that hit any row seen in the ranking half at all.\n")
    a("| top-N (hot set from combined lookups) | A->B combined | B->A combined | A->B prefill | B->A prefill | A->B decode | B->A decode |")
    a("|---|---|---|---|---|---|---|")
    for k in keys:
        cells = [ple["holdout"][d][ph]["coverage"][k] for ph in ("combined", "prefill", "decode") for d in ("A->B", "B->A")]
        a(f"| {k} | " + " | ".join(fmt_pct(c) for c in cells) + " |")
    a("| ceiling | " + " | ".join(fmt_pct(ple["holdout"][d][ph]["ceiling"])
                                   for ph in ("combined", "prefill", "decode") for d in ("A->B", "B->A")) + " |")
    for f in FRACTIONS:
        fk = f"{int(f * 100)}%"
        a(f"| rows for {fk} | " + " | ".join(fmt_rows(ple["holdout"][d][ph]["rows_for"][fk])
                                            for ph in ("combined", "prefill", "decode") for d in ("A->B", "B->A")) + " |")
    a("\nPhase-specific hot sets (ranked on the same phase of the other half):\n")
    a("| top-N | prefill A->B | prefill B->A | decode A->B | decode B->A |\n|---|---|---|---|---|")
    for k in keys:
        cells = [ple["holdout"][n][n.split()[0]]["coverage"][k] for n in
                 ("prefill A->B", "prefill B->A", "decode A->B", "decode B->A")]
        a(f"| {k} | " + " | ".join(fmt_pct(c) for c in cells) + " |")
    a("")
    if ple.get("head_share_top_1GiB"):
        hs = ple["head_share_top_1GiB"]
        a("Per-head hit rate of the 1 GiB hot set (sampled positions; heads 0-7 bigram, 8-15 trigram): in-sample "
          + ", ".join(f"{x:.2f}" for x in hs["in_sample"]) + "; hold-out " + ", ".join(f"{x:.2f}" for x in hs["holdout"]) + "\n")
    if ple.get("families"):
        a("### Per family and per reasoning effort (sampled positions, hot set built from all traffic / hold-out)\n")
        a("| family | phase | sampled positions | " + " | ".join(f"{g} GiB in / hold-out" for g in (0.5, 1, 2, 4, 8)) + " |")
        a("|---|---|---|" + "---|" * 5)
        for fam, phases in sorted(ple["families"].items()):
            for ph, e in phases.items():
                a(f"| {fam} | {ph} | {e['sampled_positions']:,} | " + " | ".join(
                    f"{fmt_pct(e.get(f'{g} GiB in-sample'))} / {fmt_pct(e.get(f'{g} GiB holdout'))}" for g in (0.5, 1, 2, 4, 8)) + " |")
        a("")
    if ple.get("hot_rows_file"):
        h = ple["hot_rows_file"]
        a(f"Hot-row file `ple_hot_rows_top.npy`: {h['rows']:,} uint32 row ids ({h['GiB_of_table']:.2f} GiB of table), "
          f"descending lookup count, in-sample combined coverage {fmt_pct(h['coverage_combined_in_sample'])}; "
          f"rows per checkpoint shard min/max {h['per_shard_min_max']}.\n")

    a("## MTP draft vocabulary\n")
    cur = v["current"]
    a(f"Current asset: {cur['asset_ids']:,} ids, {cur['loaded_ids']:,} after the runtime top-up.\n")
    a("| generated part | tokens | covered by current vocab |\n|---|---|---|")
    for lname, e in cur["coverage"].items():
        a(f"| {lname} | {e['tokens']:,} | {fmt_pct(e['coverage'])} |")
    if v.get("current_by_family"):
        a("\nPer family (all generated, excl. end marker): " + "; ".join(
            f"{f}: {fmt_pct(e['coverage'])} of {e['tokens']:,}" for f, e in sorted(v["current_by_family"].items())))
    if v.get("by_effort"):
        a("\nBy reasoning effort (chat_template_kwargs.reasoning_effort; template default xhigh):\n")
        a("| effort | requests | generated tokens | reasoning share | coverage reasoning / content / tool / all |\n|---|---|---|---|---|")
        for e, d in v["by_effort"].items():
            c = d["coverage"]
            a(f"| {e} | {d['requests']} | {c['all_generated_excl_end']['tokens']:,} | {100 * d['reasoning_share']:.1f}% | "
              f"{fmt_pct(c['reasoning']['coverage'])} / {fmt_pct(c['content']['coverage'])} / "
              f"{fmt_pct(c['tool_call']['coverage'])} / {fmt_pct(c['all_generated_excl_end']['coverage'])} |")
    if "current_coverage_on_corpus" in v:
        a(f"\nCoding corpus ({v['corpus_tokens']:,} tokens) coverage by the current vocab: {fmt_pct(v['current_coverage_on_corpus'])}.")
    if v.get("top_missing"):
        a(f"\n### Top missing tokens ({v.get('missing_unique_ids', 0):,} distinct ids, {v.get('missing_tokens', 0):,} occurrences; full list in `draft_vocab_missing_tokens.csv`)\n")
        a("| id | count | share | reasoning / content / tool | token | text |\n|---|---|---|---|---|---|")
        for m in v["top_missing"][:40]:
            text = repr(m["text"]).replace("|", "\\|")
            token = str(m["token"]).replace("|", "\\|")
            a(f"| {m['id']} | {m['count']:,} | {100 * m['share_of_generated']:.3f}% | {m['reasoning']} / {m['content']} / {m['tool_call']} | `{token}` | `{text}` |")
    if v.get("candidates_holdout"):
        a("\n### Candidate vocabularies, hold-out coverage (built on one half + corpus, measured on the other; averaged)\n")
        labels = ["reasoning", "content", "tool_call", "all_generated_excl_end"]
        a("| candidate | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels))
        for name, e in sorted(v["candidates_holdout"].items()):
            a(f"| {name} | " + " | ".join(fmt_pct(e.get(l)) for l in labels) + " |")
        a("\n`capture`: capture frequency first (corpus frequency only fills/ties). `corpus_only`: the existing builder "
          "on the coding corpus alone. `mixed_50_50`: mean of the normalised capture and corpus frequencies. "
          "All keep byte, special and added ids first and fill by id order (build_draft_vocab rules).")
    if v.get("artifacts"):
        a("\n### Artifacts (built on all captured conversations + corpus)\n")
        a("| file | ids | valid | in-sample coverage (optimistic) | corpus coverage | overlap with current |\n|---|---|---|---|---|---|")
        for f, e in v["artifacts"].items():
            a(f"| `{f}` | {e['ids']:,} | {e['valid']} | {fmt_pct(e['in_sample_coverage'])} | "
              f"{fmt_pct(e['corpus_coverage']) if e['corpus_coverage'] is not None else 'n/a'} | {e['overlap_with_current']:,} |")
    a("\n## Approximations\n")
    a("- Prompt tokens: vLLM 0.33's own render path offline (exact; checked against usage.prompt_tokens above).")
    a("- Reply tokens: SSE text re-assembled in the template format and re-tokenised; tool-call arguments re-serialised. "
      "Compare rebuilt vs server completion tokens above. Replies without SSE are recovered from the next request's "
      "history when the conversation continues (reasoning missing there if the client drops it).")
    a("- Prefill lookups: only the uncached suffix. Server-reported `cached_tokens` when the reply has usage, otherwise "
      f"the prefix model: the largest cache point floor(len(Q)/{args.cache_block})*{args.cache_block} of the last "
      f"{args.cache_prompts} prompts Q that the request shares (KV eviction not modelled).")
    a("- Decode lookups: one position per fed generated token. MTP verification also looks up the rejected draft "
      "tokens (K=3 drafts per step); that is not modelled, so real decode lookups are higher, and the rejected rows "
      "are mostly not in any hot set.")
    a("- Hot sets are row-granular; the page cache works in 4 KiB pages (~45 rows), not modelled.")
    with open(os.path.join(args.out, "REPORT.md"), "w") as handle:
        handle.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
