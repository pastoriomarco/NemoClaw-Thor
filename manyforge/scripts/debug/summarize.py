#!/usr/bin/env python3
"""summarize.py — pretty-print a smoke_corpus_runner report JSON.

Usage:
    python3 summarize.py <report.json>

Where <report.json> is a file written by `smoke_corpus_runner.py --report`.
Prints, from the structured report (no model/composer needed):
  - status counts + first-try / effective rates
  - self-heal activity (healed steps + any heal failures, from the healed/
    heal_detail fields)
  - every FAIL and SOFT-PASS *with its recorded reason*
  - the PnP chain step-by-step (pass/fail/heal)
  - the slowest 5 cases

Pure stdlib; safe to run on any historical report JSON for re-analysis or
cross-run diffing. Companion to tps.sh (decode/gen throughput from container
logs).
"""
import json
import sys
import collections

rep = json.load(open(sys.argv[1]))
results = rep.get("results", [])
flags = rep.get("runtime_flags", [])
name = sys.argv[1].split("/")[-1]

counts = collections.Counter(r["status"] for r in results)
attempted = [r for r in results if r["status"] not in ("skipped", "contaminated")]
eff = [r for r in attempted if r["status"] in ("pass", "soft-pass", "recovered-pass")]

print("=" * 70)
print(f"REPORT: {name}   flags={flags}")
print("=" * 70)
order = ["pass", "soft-pass", "recovered-pass", "fail", "skipped", "contaminated"]
print("counts: " + "  ".join(f"{k}={counts[k]}" for k in order if counts.get(k)) +
      f"   (total {len(results)})")
if attempted:
    print(f"first-try: {counts.get('pass',0)}/{len(attempted)} "
          f"({100*counts.get('pass',0)/len(attempted):.1f}%)   "
          f"effective: {len(eff)}/{len(attempted)} "
          f"({100*len(eff)/len(attempted):.1f}%)")

healed = [r["case_id"] for r in results if r.get("healed")]
healf = [(r["case_id"], r.get("heal_detail", "")) for r in results
         if r.get("heal_detail") and not r.get("healed")]
if healed:
    print(f"self-healed: {len(healed)}  {healed}")
if healf:
    print(f"self-heal FAILED: {len(healf)}  {[c for c, _ in healf]}")
    for c, d in healf:
        print(f"    {c}: {d[:160]}")

fails = [r for r in results if r["status"] == "fail"]
if fails:
    print(f"\n-- FAILS ({len(fails)}) --")
    for r in fails:
        fr = "; ".join(r.get("failures") or []) or "(no reason recorded)"
        h = "  [HEALED for downstream]" if r.get("healed") else ""
        print(f"  FAIL {r['case_id']}  ({r.get('elapsed_s',0):.1f}s){h}")
        print(f"       {fr[:320]}")

softs = [r for r in results if r["status"] == "soft-pass"]
if softs:
    print(f"\n-- SOFT-PASSES ({len(softs)}) --")
    for r in softs:
        sf = "; ".join(r.get("soft_failures") or []) or "(none)"
        print(f"  SOFT {r['case_id']}  ({r.get('elapsed_s',0):.1f}s): {sf[:240]}")

chain = [r for r in results if r["case_id"].startswith("PnP_")]
if chain:
    okc = sum(1 for r in chain if r["status"] in ("pass", "soft-pass", "recovered-pass"))
    print(f"\n-- PnP CHAIN ({len(chain)} steps, {okc} ok) --")
    for r in chain:
        h = f"  heal:{r.get('heal_detail','')[:48]}" if r.get("healed") else ""
        print(f"  {r['status']:>12}  {r['case_id']}{h}")

slow = sorted(results, key=lambda r: -(r.get("elapsed_s") or 0))[:5]
print("\n-- SLOWEST 5 --")
for r in slow:
    print(f"  {r.get('elapsed_s',0):6.1f}s  {r['case_id']}  ({r['status']})")
