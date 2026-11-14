#!/usr/bin/env python3
"""Replay the captured PLE lookup stream through TensorRT Edge-LLM 0.12's ``--ple-residency cached``.

Mirrors ``cpp/runtime/embedding`` of TensorRT-Edge-LLM 0.12.0:

- ``PleCache::reserveBatch`` / ``claimSlot``: one batch (one request, batch 1)
  per step; the step's distinct rows are the union; hits are pinned and get
  the CLOCK reference bit, then each miss (in row order) claims a slot with
  CLOCK (hand over slots 1..S-1, slot 0 reserved for padding; skip pinned
  slots; take a free slot; clear a set reference bit and move on; otherwise
  evict). New rows are pinned and referenced. Pins last until the next step
  (``releaseLeases`` at the next ``embed``), so only the current step's rows
  are pinned when a slot is claimed.
- ``PleRowStore`` counters per step: uniqueRows, misses,
  usefulBytes = misses x row bytes, readBytes and readCalls of the read plan.
- ``planPleReads``: per source tensor ("shard"), each row span is widened to
  4 KiB pages (end clipped at the tensor end), spans sorted by (shard, offset),
  overlapping/adjacent ranges merged, ranges split at 1 MiB (``readBytes``
  default). Up to 32 readers only parallelise these reads; bytes do not change.

Tables (row ids are identical, same hash and 320,001,536 rows):
- ``fp4local``: our FP4-Local checkpoint, 90 B slots. Edge-LLM 0.12 cannot
  read NVFP4 sources (PleTable has FP8/FP16/BF16/FP32 only); simulated as if
  it read the row's 80 B of packed values and 10 B of scales from their two
  checkpoint tensors (two spans per row).
- ``fp8``: the NVIDIA NVFP4 checkpoint's FP8 table, 160 B/row, one span per
  row, offsets from its safetensors header; 160 B slots (FP8 pool).

Steps: a prefill suffix is split into chunks of ``--prefill-chunk`` tokens
(one batch each); every fed decode token is one step (MTP verify/draft
lookups not modelled, batch 1, no staging-wave split of large fills).

Variants: (a) cold pool, all requests in arrival order across conversations
(one server lifetime); metrics also split by hold-out half. (b) warm: pool
pre-filled with the most frequent rows of half A (by count, then id; reference
bits clear), then only half B replayed. Compare (b) with (a) on half B.

Baseline "current": our vLLM disk overlay gathers every distinct row of the
step on the host (page cache) and copies all of them: unique rows x 90 B per
step, and the 4 KiB pages those rows touch.
"""

import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
import numba  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import RowCounter  # noqa: E402
from learning_curve import load_spool  # noqa: E402
from ple_hash import PLEHasher  # noqa: E402

GIB = 2**30
PAGE = 4096
MAX_READ = 1 << 20
SIZES = [0.25, 0.5, 1, 2, 4, 8]
NPAGES = 160  # histogram bins for pages per decode step (max 16 rows x 2 spans x 2 pages)


def header(path):
    with open(path, "rb") as handle:
        (n,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(n)), 8 + n


def tensor_layout(model_dir, names):
    """Absolute file offset (begin, end) and file index of each tensor, from headers only."""
    weight_map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    files, cache, out = {}, {}, []
    for name in names:
        fname = weight_map[name]
        if fname not in cache:
            cache[fname] = header(os.path.join(model_dir, fname))
        hd, start = cache[fname]
        b, e = hd[name]["data_offsets"]
        out.append((files.setdefault(fname, len(files)), start + b, start + e, hd[name]["shape"], hd[name]["dtype"]))
    return out


def find_prefix(model_dir):
    weight_map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    for key in weight_map:
        if key.endswith("ple_embedding.ngram_embedding.shard_0.weight"):
            return key[: -len("shard_0.weight")]
    raise SystemExit(f"no PLE shards in {model_dir}")


@numba.njit(cache=True)
def plan_one(rows, n, shard_size, base, end, row_bytes, out):
    """planPleReads for one span per row (rows sorted). out[0] += readBytes, out[1] += calls, out[2] += pages."""
    cur_sh, cur_s, cur_e = -1, 0, 0
    for i in range(n):
        r = np.int64(rows[i])
        sh = r // shard_size
        s = base[sh] + (r - sh * shard_size) * row_bytes
        e = s + row_bytes
        a_s = s - s % PAGE
        pad = 0 if e % PAGE == 0 else PAGE - e % PAGE
        a_e = e + min(pad, end[sh] - e)
        if sh == cur_sh and a_s <= cur_e:
            if a_e > cur_e:
                cur_e = a_e
        else:
            if cur_sh >= 0:
                out[0] += cur_e - cur_s
                out[1] += (cur_e - cur_s + MAX_READ - 1) // MAX_READ
            cur_sh, cur_s, cur_e = sh, a_s, a_e
    if cur_sh >= 0:
        out[0] += cur_e - cur_s
        out[1] += (cur_e - cur_s + MAX_READ - 1) // MAX_READ


@numba.njit(cache=True)
def plan_rows(rows, n, layout, lay, shard_size, out):
    """Read plan of the rows' spans for table ``layout`` (0 fp4local: values + scales, 1 fp8)."""
    tmp = np.zeros(2, np.int64)
    if layout == 0:
        plan_one(rows, n, shard_size, lay[0], lay[1], 80, tmp)
        plan_one(rows, n, shard_size, lay[2], lay[3], 10, tmp)
    else:
        plan_one(rows, n, shard_size, lay[4], lay[5], 160, tmp)
    out[0] += tmp[0]
    out[1] += tmp[1]
    return tmp[0]


@numba.njit(cache=True)
def run_steps(ids, step_start, step_end, step_kind, step_half, active, H,
              slot_row, cfg_off, hand, cfg_layout, row_bytes, resident, refbits, pin,
              lay, shard_size, sums, miss_hist, page_hist, base_sums, base_uniq_hist, base_page_hist, failures):
    """sums[c, half, kind, :] = (unique, misses, usefulBytes, readBytes, readCalls, steps)."""
    n_cfg = cfg_off.shape[0] - 1
    buf = np.empty(0, np.uint32)
    miss = np.empty(0, np.uint32)
    planout = np.zeros(2, np.int64)
    for st in range(step_start.shape[0]):
        if not active[st]:
            continue
        a = step_start[st] * H
        b = step_end[st] * H
        flat = np.sort(ids[a:b])
        if buf.shape[0] < flat.shape[0]:
            buf = np.empty(flat.shape[0], np.uint32)
            miss = np.empty(flat.shape[0], np.uint32)
        u = 0
        for i in range(flat.shape[0]):
            if i == 0 or flat[i] != flat[i - 1]:
                buf[u] = flat[i]
                u += 1
        kind = step_kind[st]
        half = step_half[st]
        for i in range(u):
            pin[buf[i]] = 1
        # Baseline: every distinct row gathered (pages touched in the FP4-Local layout).
        planout[0] = 0
        planout[1] = 0
        pb = plan_rows(buf, u, 0, lay, shard_size, planout)
        base_sums[half, kind, 0] += u
        base_sums[half, kind, 1] += pb
        base_sums[half, kind, 2] += 1
        if kind == 1:
            base_uniq_hist[half, min(u, base_uniq_hist.shape[1] - 1)] += 1
            base_page_hist[half, min(pb // PAGE, NPAGES - 1)] += 1
        for c in range(n_cfg):
            bit = np.uint16(1 << c)
            nbit = np.uint16(~bit & 0xFFFF)
            off = cfg_off[c]
            usable = cfg_off[c + 1] - off - 1
            m = 0
            for i in range(u):
                r = buf[i]
                if resident[r] & bit:
                    refbits[r] |= bit
                else:
                    miss[m] = r
                    m += 1
            for i in range(m):
                r = miss[i]
                got = -1
                for _ in range(2 * usable):
                    sid = hand[c]
                    hand[c] = 1 if sid == usable else sid + 1
                    old = slot_row[off + sid]
                    if old < 0:
                        got = sid
                        break
                    if pin[old]:
                        continue
                    if refbits[old] & bit:
                        refbits[old] &= nbit
                        continue
                    resident[old] &= nbit
                    refbits[old] &= nbit
                    got = sid
                    break
                if got < 0:
                    failures[c] += 1
                    continue
                slot_row[off + got] = r
                resident[r] |= bit
                refbits[r] |= bit
            planout[0] = 0
            planout[1] = 0
            rb = plan_rows(miss, m, cfg_layout[c], lay, shard_size, planout)
            sums[c, half, kind, 0] += u
            sums[c, half, kind, 1] += m
            sums[c, half, kind, 2] += m * row_bytes[c]
            sums[c, half, kind, 3] += rb
            sums[c, half, kind, 4] += planout[1]
            sums[c, half, kind, 5] += 1
            if kind == 1:
                miss_hist[c, half, min(m, miss_hist.shape[2] - 1)] += 1
                page_hist[c, half, min(rb // PAGE, NPAGES - 1)] += 1
        for i in range(u):
            pin[buf[i]] = 0


def quantile(hist, q, unit=1.0):
    total = hist.sum()
    if total == 0:
        return float("nan")
    cum = np.cumsum(hist)
    return float(np.searchsorted(cum, q * total, side="left")) * unit


def mean_of(hist, unit=1.0):
    total = hist.sum()
    return float((hist * np.arange(hist.shape[0])).sum() / total) * unit if total else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spool", required=True)
    ap.add_argument("--model", required=True, help="FP4-Local model dir (config, index, headers)")
    ap.add_argument("--fp8-model", default="", help="NVIDIA NVFP4 snapshot dir with the FP8 PLE table (headers only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes-gib", type=float, nargs="+", default=SIZES)
    ap.add_argument("--prefill-chunk", type=int, default=8192)
    ap.add_argument("--block-positions", type=int, default=2_000_000)
    ap.add_argument("--max-stream-tokens", type=float, default=2e8,
                    help="replay at most this many lookup positions (prefix in arrival order)")
    args = ap.parse_args()
    t0 = time.time()
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    hasher = PLEHasher(cfg.get("text_config", cfg))
    R, H, SS = hasher.padded_rows, hasher.ngram_heads, hasher.shard_size
    parts = hasher.split_ngram_parts

    # Layouts (headers only).
    p4 = find_prefix(args.model)
    wl = tensor_layout(args.model, [f"{p4}shard_{i}.weight" for i in range(parts)])
    sl = tensor_layout(args.model, [f"{p4}shard_{i}.weight_scale" for i in range(parts)])
    lay = np.zeros((6, parts), np.int64)
    lay[0] = [x[1] for x in wl]
    lay[1] = [x[2] for x in wl]
    lay[2] = [x[1] for x in sl]
    lay[3] = [x[2] for x in sl]
    tables = [("fp4local", 0, 90)]
    fp8_note = "FP8 table not given"
    if args.fp8_model:
        p8 = find_prefix(args.fp8_model)
        fl = tensor_layout(args.fp8_model, [f"{p8}shard_{i}.weight" for i in range(parts)])
        rows8 = sum(x[3][0] for x in fl)
        width8 = fl[0][3][1]
        assert rows8 == R and fl[0][4] == "F8_E4M3", (rows8, fl[0][4])
        lay[4] = [x[1] for x in fl]
        lay[5] = [x[2] for x in fl]
        tables.append(("fp8", 1, width8))
        fp8_note = (f"NVIDIA FP8 table: {rows8:,} rows x {width8} B = {rows8 * width8 / GIB:.1f} GiB "
                    f"({len(fl)} shard tensors in {len({x[0] for x in fl})} file(s))")
    configs = [(t, layout, rb, g, int(g * GIB // rb)) for t, layout, rb in tables for g in args.sizes_gib]
    assert len(configs) <= 16
    n_cfg = len(configs)

    seg, tokens, meta = load_spool(args.spool)
    total_tokens = int((seg["n"] - seg["lead"]).sum())
    truncated = ""
    if total_tokens > args.max_stream_tokens:
        keep = int(np.searchsorted(np.cumsum(seg["n"] - seg["lead"]), args.max_stream_tokens, side="right"))
        seg = seg[:keep]
        truncated = (f" Stream truncated to the first {keep:,} of the segments ({args.max_stream_tokens:,.0f} of "
                     f"{total_tokens:,} lookup positions, a contiguous prefix in arrival order) to bound the run time.")
    efforts = meta.get("efforts", ["xhigh"])
    G = 2 * len(efforts)  # step group = effort * 2 + hold-out half
    ALL = list(range(G))
    HALF_B = [g for g in ALL if g % 2 == 1]
    present = sorted({int(e) for e in np.unique(seg["eff"])}) if "eff" in seg.dtype.names else [0]
    # Steps per block of segments.
    def blocks():
        i = 0
        while i < seg.shape[0]:
            ids_list, st_s, st_e, st_k, st_h = [], [], [], [], []
            pos = 0
            while i < seg.shape[0] and pos < args.block_positions:
                rec = seg[i]
                seq = np.asarray(tokens[rec["off"] : rec["off"] + rec["n"]], dtype=np.int64)
                ids = hasher.ids(seq, np.arange(int(rec["lead"]), seq.shape[0], dtype=np.int64))
                n = ids.shape[0]
                ids_list.append(ids.reshape(-1))
                if rec["kind"] == 0:
                    starts = np.arange(0, n, args.prefill_chunk)
                    ends = np.minimum(starts + args.prefill_chunk, n)
                else:
                    starts = np.arange(n)
                    ends = starts + 1
                st_s.append(starts + pos)
                st_e.append(ends + pos)
                st_k.append(np.full(starts.shape[0], rec["kind"], np.uint8))
                st_h.append(np.full(starts.shape[0], int(rec["eff"]) * 2 + int(rec["half"]), np.uint8))
                pos += n
                i += 1
            yield (np.concatenate(ids_list).astype(np.uint32), np.concatenate(st_s).astype(np.int64),
                   np.concatenate(st_e).astype(np.int64), np.concatenate(st_k), np.concatenate(st_h))

    def new_state():
        sizes = np.asarray([c[4] for c in configs], np.int64)
        cfg_off = np.zeros(n_cfg + 1, np.int64)
        cfg_off[1:] = np.cumsum(sizes)
        return {"slot_row": np.full(int(cfg_off[-1]), -1, np.int32), "cfg_off": cfg_off,
                "hand": np.ones(n_cfg, np.int64), "resident": np.zeros(R, np.uint16),
                "refbits": np.zeros(R, np.uint16)}

    pin = np.zeros(R, np.uint8)
    cfg_layout = np.asarray([c[1] for c in configs], np.int64)
    row_bytes = np.asarray([c[2] for c in configs], np.int64)

    def replay(state, only_half=None):
        sums = np.zeros((n_cfg, G, 2, 6), np.int64)
        miss_hist = np.zeros((n_cfg, G, H + 2), np.int64)
        page_hist = np.zeros((n_cfg, G, NPAGES), np.int64)
        base_sums = np.zeros((G, 2, 3), np.int64)
        base_uniq = np.zeros((G, H + 2), np.int64)
        base_page = np.zeros((G, NPAGES), np.int64)
        failures = np.zeros(n_cfg, np.int64)
        for ids, ss, se, sk, sh in blocks():
            active = np.ones(ss.shape[0], np.bool_) if only_half is None else (sh % 2 == only_half)
            run_steps(ids, ss, se, sk, sh, active, H, state["slot_row"], state["cfg_off"], state["hand"],
                      cfg_layout, row_bytes, state["resident"], state["refbits"], pin, lay, SS,
                      sums, miss_hist, page_hist, base_sums, base_uniq, base_page, failures)
            print(f"    block done ({time.time() - t0:.0f}s)", flush=True)
        return {"sums": sums, "miss_hist": miss_hist, "page_hist": page_hist, "base_sums": base_sums,
                "base_uniq": base_uniq, "base_page": base_page, "failures": failures}

    print(f"{seg.shape[0]} segments; configs: " + ", ".join(f"{c[0]} {c[3]} GiB={c[4]:,} slots" for c in configs), flush=True)
    print("variant (a): cold pool, all requests", flush=True)
    res_a = replay(new_state())

    print("variant (b): warm pool from half A, replay half B", flush=True)
    counter = RowCounter(R)
    for rec in seg[seg["half"] == 0]:
        seq = np.asarray(tokens[rec["off"] : rec["off"] + rec["n"]], dtype=np.int64)
        counter.add(hasher.ids(seq, np.arange(int(rec["lead"]), seq.shape[0], dtype=np.int64)))
    counter.flush()
    state = new_state()
    have_b = bool((seg["half"] == 1).any()) and counter.total > 0
    res_b = None
    if have_b:
        nz, cnts = [], []
        for lo in range(0, R, 8 * 2**20):
            c = counter.slice(lo, min(R, lo + 8 * 2**20))
            idx = np.flatnonzero(c)
            nz.append((idx + lo).astype(np.uint32))
            cnts.append(c[idx].astype(np.int64))
        nz = np.concatenate(nz)
        cnts = np.concatenate(cnts)
        del counter
        order = np.lexsort((nz, -cnts))
        top = nz[order]
        del order, cnts, nz
        for c, conf in enumerate(configs):
            k = min(conf[4] - 1, top.shape[0])
            off = state["cfg_off"][c]
            state["slot_row"][off + 1 : off + 1 + k] = top[:k].astype(np.int32)
            state["resident"][top[:k]] |= np.uint16(1 << c)
        warm_rows = int(top.shape[0])
        del top
        res_b = replay(state, only_half=1)
    else:
        warm_rows = 0
    del state

    # ---- report
    def stats(res, c, halves):
        s = res["sums"][c][halves].sum(axis=0)  # (kind, 6)
        mh = res["miss_hist"][c][halves].sum(axis=0)
        ph = res["page_hist"][c][halves].sum(axis=0)
        out = {}
        for k, name in ((0, "prefill"), (1, "decode")):
            uniq, miss, useful, readb, calls, steps = (int(x) for x in s[k])
            out[name] = {"unique": uniq, "misses": miss, "hit_rate": 1 - miss / uniq if uniq else float("nan"),
                         "useful_bytes": useful, "read_bytes": readb, "read_calls": calls, "steps": steps,
                         "amplification": readb / useful if useful else float("nan")}
        out["decode"].update({
            "misses_mean": mean_of(mh), "misses_p50": quantile(mh, .5), "misses_p95": quantile(mh, .95),
            "misses_p99": quantile(mh, .99), "zero_miss_share": float(mh[0] / mh.sum()) if mh.sum() else float("nan"),
            "read_kib_mean": out["decode"]["read_bytes"] / max(1, out["decode"]["steps"]) / 1024,
            "read_kib_p50": quantile(ph, .5, 4), "read_kib_p95": quantile(ph, .95, 4), "read_kib_p99": quantile(ph, .99, 4),
            "calls_mean": out["decode"]["read_calls"] / max(1, out["decode"]["steps"])})
        return out

    def base_stats(res, halves):
        bs = res["base_sums"][halves].sum(axis=0)
        bu = res["base_uniq"][halves].sum(axis=0)
        bp = res["base_page"][halves].sum(axis=0)
        return {"decode_unique_mean": mean_of(bu), "decode_unique_p50": quantile(bu, .5),
                "decode_unique_p95": quantile(bu, .95), "decode_unique_p99": quantile(bu, .99),
                "decode_copy_bytes_mean": mean_of(bu) * 90,
                "decode_pages_kib_mean": bs[1][1] / max(1, bs[1][2]) / 1024,
                "decode_pages_kib_p50": quantile(bp, .5, 4), "decode_pages_kib_p95": quantile(bp, .95, 4),
                "decode_pages_kib_p99": quantile(bp, .99, 4),
                "prefill_unique_per_chunk": bs[0][0] / max(1, bs[0][2]),
                "prefill_pages_kib_per_chunk": bs[0][1] / max(1, bs[0][2]) / 1024}

    summary = {"configs": [{"table": c[0], "gib": c[3], "slots": c[4], "slot_bytes": c[2]} for c in configs],
               "fp8_note": fp8_note, "warm_rows_available": warm_rows,
               "a_all": [stats(res_a, i, ALL) for i in range(n_cfg)],
               "a_halfB": [stats(res_a, i, HALF_B) for i in range(n_cfg)],
               "b_halfB": [stats(res_b, i, HALF_B) for i in range(n_cfg)] if res_b else None,
               "a_by_effort": {efforts[e]: [stats(res_a, i, [2 * e, 2 * e + 1]) for i in range(n_cfg)] for e in present},
               "baseline_all": base_stats(res_a, ALL), "baseline_halfB": base_stats(res_a, HALF_B),
               "baseline_by_effort": {efforts[e]: base_stats(res_a, [2 * e, 2 * e + 1]) for e in present},
               "clock_failures": res_a["failures"].tolist()}

    L = ["## Edge-LLM 0.12 `--ple-residency cached` replay\n",
         "CLOCK pool replayed on the captured lookup stream (prefill uncached suffix in "
         f"{args.prefill_chunk}-token chunks, one step per decode token, batch 1, MTP drafts not modelled). "
         "Hit rate = 1 - misses / distinct rows per step (Edge-LLM counters). Read bytes: page-aligned (4 KiB), "
         "sorted, merged ranges per source tensor, 1 MiB max per read; amplification = readBytes / usefulBytes. "
         f"fp4local = 90 B slots, two spans per row (values + scales; Edge-LLM 0.12 itself has no NVFP4 source). "
         f"{fp8_note}; slots for an FP16 pool of that table would be half the FP8 counts.{truncated}\n"]
    fp8_slots = ", ".join(f"{g} GiB = {int(g * GIB // 160):,} (FP8) / {int(g * GIB // 320):,} (FP16)" for g in args.sizes_gib)
    L.append(f"Slot counts for the FP8 table: {fp8_slots}.\n")
    b_all = summary["baseline_all"]
    L.append("**Current scheme (vLLM disk overlay, every distinct row gathered on the host and copied):** decode "
             f"{b_all['decode_unique_mean']:.1f} rows/step (p50 {b_all['decode_unique_p50']:.0f}, p95 "
             f"{b_all['decode_unique_p95']:.0f}, p99 {b_all['decode_unique_p99']:.0f}) = "
             f"{b_all['decode_copy_bytes_mean'] / 1024:.2f} KiB copied/step, touching "
             f"{b_all['decode_pages_kib_mean']:.0f} KiB of 4 KiB pages/step (p50 {b_all['decode_pages_kib_p50']:.0f}, "
             f"p95 {b_all['decode_pages_kib_p95']:.0f}, p99 {b_all['decode_pages_kib_p99']:.0f}); prefill "
             f"{b_all['prefill_unique_per_chunk']:,.0f} distinct rows and {b_all['prefill_pages_kib_per_chunk'] / 1024:.1f} MiB "
             "of pages per chunk.\n")

    def table(title, rows):
        L.append(f"### {title}\n")
        L.append("| table | pool | slots | prefill hit | decode hit | decode misses/step mean / p50 / p95 / p99 | "
                 "zero-miss steps | decode read KiB/step mean / p50 / p95 / p99 | read calls/step | amplification decode / prefill |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for conf, st in zip(configs, rows):
            d, p = st["decode"], st["prefill"]
            L.append(f"| {conf[0]} | {conf[3]} GiB | {conf[4]:,} | {100 * p['hit_rate']:.2f}% | {100 * d['hit_rate']:.2f}% | "
                     f"{d['misses_mean']:.2f} / {d['misses_p50']:.0f} / {d['misses_p95']:.0f} / {d['misses_p99']:.0f} | "
                     f"{100 * d['zero_miss_share']:.1f}% | {d['read_kib_mean']:.1f} / {d['read_kib_p50']:.0f} / "
                     f"{d['read_kib_p95']:.0f} / {d['read_kib_p99']:.0f} | {d['calls_mean']:.2f} | "
                     f"{d['amplification']:.1f}x / {p['amplification']:.1f}x |")
        L.append("")

    table("(a) cold pool, one server lifetime, all requests in arrival order", summary["a_all"])
    table("(a) cold pool, measured on half B only (same run)", summary["a_halfB"])
    if summary["b_halfB"]:
        table(f"(b) warm pool (top rows of half A, {warm_rows:,} distinct rows available), half B replayed", summary["b_halfB"])
    else:
        L.append("(b) skipped: no rows in half A or no requests in half B.\n")
    if len(present) > 1:
        L.append("### (a) cold pool, split by reasoning effort of the request (same run, shared pool)\n")
        L.append("| effort | table | pool | prefill hit | decode hit | decode misses/step mean / p95 | zero-miss steps | "
                 "decode read KiB/step mean / p95 | current scheme rows/step mean |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for e, rows in summary["a_by_effort"].items():
            base = summary["baseline_by_effort"][e]
            for conf, st in zip(configs, rows):
                d, p = st["decode"], st["prefill"]
                L.append(f"| {e} | {conf[0]} | {conf[3]} GiB | {100 * p['hit_rate']:.2f}% | {100 * d['hit_rate']:.2f}% | "
                         f"{d['misses_mean']:.2f} / {d['misses_p95']:.0f} | {100 * d['zero_miss_share']:.1f}% | "
                         f"{d['read_kib_mean']:.1f} / {d['read_kib_p95']:.0f} | {base['decode_unique_mean']:.1f} |")
        L.append("")
    if any(summary["clock_failures"]):
        L.append(f"CLOCK failures (no unpinned slot): {summary['clock_failures']}\n")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ple_cache_sim.json"), "w") as handle:
        json.dump(summary, handle, indent=1, default=float)
    with open(os.path.join(args.out, "ple_cache_sim.md"), "w") as handle:
        handle.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"cache simulation done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
