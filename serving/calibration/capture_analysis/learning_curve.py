#!/usr/bin/env python3
"""Learning curve of a static PLE hot set: does more capture still raise the held-out ceiling?

Reads the lookup stream written by ``analyze.py --spool``. Tasks (tools + system
prompt + first user message) are ordered by their first request. Per pool
(all tasks; agent tasks; reasoning-family tasks) the last ``--eval-share`` of
the tasks (at least 2) are held out; the hot set is built from the first k of
the other tasks, k = 1, 2, 4, 6, 8, 12, 16, 24, ... and measured on the
held-out tasks:

- ceiling: share of held-out lookups whose row occurs in the k training tasks
  at all (the best any hot set built from them can do);
- 1/2/4 GiB: share served by the top rows of the training counts (90 B/row).

Prefill (uncached suffix), decode and combined, as in analyze.py. Verdict on the
combined ceiling: slope over the last three points in percentage points per
doubling of k; "rising" if >= --rising-pp, else "flat".
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import Ranking, RowCounter, membership, rows_of_gib  # noqa: E402
from ple_hash import PLEHasher  # noqa: E402

BUDGETS = [1, 2, 4]
PHASES = ["prefill", "decode", "combined"]


def load_spool(path):
    seg = np.load(os.path.join(path, "segments.npy"))
    tokens = np.memmap(os.path.join(path, "tokens.bin"), dtype=np.int32, mode="r")
    meta = json.load(open(os.path.join(path, "meta.json")))
    return seg, tokens, meta


def segment_ids(hasher, tokens, rec):
    seq = np.asarray(tokens[rec["off"] : rec["off"] + rec["n"]], dtype=np.int64)
    lead = int(rec["lead"])
    # Positions < 2 in this slice are true sequence starts only when lead < 2 (EOS context).
    return hasher.ids(seq, np.arange(lead, seq.shape[0], dtype=np.int64))


def ks_for(n):
    ks, k = [], 1
    for k in [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]:
        if k < n:
            ks.append(k)
    ks.append(n)
    return sorted(set(ks))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spool", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="directory for learning_curve.md / .json")
    ap.add_argument("--eval-share", type=float, default=0.25)
    ap.add_argument("--eval-max-positions", type=int, default=4_000_000, help="held-out positions sampled (stride)")
    ap.add_argument("--reasoning-family", default="reasoning")
    ap.add_argument("--rising-pp", type=float, default=1.0)
    args = ap.parse_args()
    t0 = time.time()
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    hasher = PLEHasher(cfg.get("text_config", cfg))
    R = hasher.padded_rows
    seg, tokens, meta = load_spool(args.spool)
    fam_names = meta["families"]
    n_tasks = len(meta["tasks"])
    order = np.argsort(np.asarray(meta["task_first_request"]), kind="stable")
    task_is_reasoning = np.asarray([fam_names.get(str(f)) == args.reasoning_family for f in meta["task_family"]])
    by_task = [np.flatnonzero(seg["task"] == t) for t in range(n_tasks)]
    efforts = meta.get("efforts", ["xhigh"])
    task_eff = np.zeros(n_tasks, np.int64)
    if "eff" in seg.dtype.names:
        for t in range(n_tasks):
            if by_task[t].size:
                task_eff[t] = int(seg["eff"][by_task[t][0]])
    pools = {"all": [int(t) for t in order],
             "agent": [int(t) for t in order if not task_is_reasoning[t]],
             "reasoning": [int(t) for t in order if task_is_reasoning[t]]}
    if len(set(task_eff.tolist())) > 1:
        for e in sorted(set(task_eff.tolist())):
            pools[f"effort {efforts[e]}"] = [int(t) for t in order if task_eff[t] == e]

    results = {}
    lines = ["## Learning curve of a static hot set (held-out tasks)\n",
             f"Spool: {meta['requests']} requests, {n_tasks} tasks. Tasks ordered by first request; the last "
             f"{int(args.eval_share * 100)}% of each pool's tasks (min 2) are held out, the hot set is built from the "
             "first k of the others. `ceil` = held-out lookups whose row the k tasks touched at all; "
             "1/2/4 GiB = top rows of the k tasks' counts. Prefill = uncached suffix, decode = generated tokens.\n"]
    for pool, tasks in pools.items():
        if len(tasks) < 3:
            lines.append(f"- pool `{pool}`: {len(tasks)} tasks, too few for a curve.\n")
            continue
        n_eval = max(2, int(round(len(tasks) * args.eval_share)))
        train, held = tasks[:-n_eval], tasks[-n_eval:]
        # Held-out sample (stride so that at most eval_max_positions positions are kept).
        held_segs = np.concatenate([by_task[t] for t in held])
        total_pos = int((seg["n"][held_segs] - seg["lead"][held_segs]).sum())
        stride = max(1, math.ceil(total_pos / args.eval_max_positions))
        ev_ids, ev_phase = [], []
        for i in held_segs:
            ids = segment_ids(hasher, tokens, seg[i])[::stride]
            ev_ids.append(ids)
            ev_phase.append(np.full(ids.shape[0], seg["kind"][i], np.uint8))
        ev_ids = np.concatenate(ev_ids).astype(np.int64)
        ev_phase = np.concatenate(ev_phase)
        counter = RowCounter(R)
        rows, done = [], 0
        for k in ks_for(len(train)):
            for t in train[done:k]:
                for i in by_task[t]:
                    counter.add(segment_ids(hasher, tokens, seg[i]))
            done = k
            counter.flush()
            rank = Ranking(R, counter.slice, {})
            counts = counter.get(ev_ids.reshape(-1)).reshape(ev_ids.shape)
            row = {"k": k, "train_lookups": counter.total, "train_unique_rows": rank.rank_unique}
            members = {"ceil": (counts > 0).astype(np.float64)}
            for g in BUDGETS:
                members[f"{g} GiB"] = membership(counts, rank.threshold(rows_of_gib(g)))
            for phase in PHASES:
                mask = slice(None) if phase == "combined" else (ev_phase == (0 if phase == "prefill" else 1))
                for name, m in members.items():
                    sel = m[mask]
                    row[f"{phase} {name}"] = float(sel.mean()) if sel.size else float("nan")
            rows.append(row)
            print(f"  {pool} k={k}: combined ceil {100 * row['combined ceil']:.2f}% ({time.time() - t0:.0f}s)", flush=True)
        del counter
        # Verdict: slope of the combined ceiling over the last three points, per doubling of k.
        pts = [(math.log2(r["k"]), 100 * r["combined ceil"]) for r in rows[-3:]]
        slope = float(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0]) if len(pts) >= 2 else float("nan")
        per_task = ((rows[-1]["combined ceil"] - rows[-2]["combined ceil"]) * 100 / (rows[-1]["k"] - rows[-2]["k"])
                    if len(rows) >= 2 else float("nan"))
        verdict = "rising" if slope >= args.rising_pp else "flat"
        results[pool] = {"train_tasks": len(train), "held_out_tasks": len(held), "eval_stride": stride,
                         "held_out_positions_sampled": int(ev_ids.shape[0]), "rows": rows,
                         "slope_pp_per_doubling": slope, "slope_pp_per_task_last": per_task, "verdict": verdict}
        lines.append(f"### Pool `{pool}`: {len(train)} training tasks, {len(held)} held out "
                     f"({ev_ids.shape[0]:,} held-out positions, stride {stride})\n")
        head = ["k", "train lookups"] + [f"{p} {n}" for p in PHASES for n in ["ceil"] + [f"{g} GiB" for g in BUDGETS]]
        lines.append("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        for r in rows:
            cells = [str(r["k"]), f"{r['train_lookups']:,}"] + [
                f"{100 * r[f'{p} {n}']:.1f}" for p in PHASES for n in ["ceil"] + [f"{g} GiB" for g in BUDGETS]]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append(f"\n**Verdict `{pool}`: {verdict}** — combined ceiling slope {slope:+.2f} pp per doubling of "
                     f"training tasks over the last {len(pts)} points ({per_task:+.2f} pp per added task at the end; "
                     f"threshold {args.rising_pp} pp/doubling).\n")
        print(f"VERDICT {pool}: {verdict} ({slope:+.2f} pp per doubling, last k={rows[-1]['k']}, "
              f"combined ceiling {100 * rows[-1]['combined ceil']:.1f}%)", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "learning_curve.json"), "w") as handle:
        json.dump(results, handle, indent=1)
    with open(os.path.join(args.out, "learning_curve.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"learning curve done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
