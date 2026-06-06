#!/usr/bin/env python3
"""golden_trajectory.py — TEST-ONLY (smoke harness).

Replay a chained corpus family's per-step *golden changes* from the base program,
applying each via the real composer bridge tools (bypassing the model), and
assertion-gating each step against the corpus's `expected.state_after` where one is
declared. Surfaces corpus incoherence (a golden change that doesn't satisfy its own
step's assertion) and produces the validated golden trajectory that the runner
self-heal consumes (state replay-from-base + transcript splice).

This NEVER runs in production — it only POSTs to a live test composer's bridge tool
endpoints, exactly like `apply_fixtures` in the runner.

Usage:
  python3 golden_trajectory.py --chain pnp_build
  python3 golden_trajectory.py --chain pnp_build --steps 5        # first N steps only
"""
from __future__ import annotations
import argparse, json, os, sys, time
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_corpus_runner import (  # reuse the runner's primitives — single source of truth
    _post_json, capture_state, assert_state, reset_program, fetch_catalog_hash,
)

DEFAULT_GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_trajectories.yaml")
DEFAULT_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_corpus.yaml")


def _count_tree_nodes(state: dict) -> int:
    """Total nodes in the program tree (recursive) — used to verify that an
    insert actually LANDED (the bridge can return HTTP 200 yet add nothing if a
    node spec is invalid)."""
    tree = ((state or {}).get("program") or {}).get("tree") or {}

    def walk(n) -> int:
        if not isinstance(n, dict):
            return 0
        return 1 + sum(walk(k) for k in (n.get("children") or []))

    return walk(tree)


def apply_change(composer: str, tool: str, args: dict, catalog_hash: str, tag: str):
    """POST one golden change to the bridge tool endpoint (same path the assistant
    flow and apply_fixtures use). Returns (http_code, response)."""
    envelope = {
        "requestId": f"golden-{tag}-{int(time.time()*1000)}",
        "assistantMode": "composer-assistant",
        "catalogHash": catalog_hash,
        "arguments": args,
    }
    return _post_json(f"{composer}/api/assistant/bridge/tools/{tool}", envelope, timeout=20.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True, help="chain_id, e.g. pnp_build")
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--composer", default="http://127.0.0.1:9000")
    ap.add_argument("--steps", type=int, default=0, help="limit to first N steps (0 = all)")
    ap.add_argument("--dump-final", action="store_true", help="print final program+scene state")
    a = ap.parse_args()

    corpus = yaml.safe_load(open(a.corpus))
    cases = {c["id"]: c for c in (corpus.get("cases") or [])}
    default_pre = corpus.get("default_precondition") or {}
    spec = yaml.safe_load(open(a.golden))
    if a.chain not in spec:
        print(f"chain {a.chain!r} not in {a.golden}; have: {list(spec)}"); return 2
    chain = spec[a.chain]
    base = chain.get("base") or {}
    dep = base.get("deployment_path", default_pre.get("deployment_path"))
    emptyp = base.get("empty_program_path",
                      "/workspace/examples/empty_pick_and_place_ur10e_robotiq.program.yaml")

    print(f"== golden-trajectory: chain={a.chain}  composer={a.composer} ==")
    code, _ = reset_program(a.composer, dep, emptyp)
    print(f"reset-to-base: HTTP {code}  ({os.path.basename(emptyp)})")
    if code != 200:
        print("  cannot reach base; abort"); return 1
    catalog_hash = fetch_catalog_hash(a.composer)

    steps = chain.get("steps") or []
    if a.steps:
        steps = steps[: a.steps]
    prev_nodes = _count_tree_nodes(capture_state(a.composer))
    coherent = True
    for step in steps:
        cid = step["id"]
        case = cases.get(cid)
        if case is None:
            print(f"  ⚠ {cid}: not in corpus (skipping assertion)")
        changes = step.get("changes") or []
        n_inserts = sum(1 for ch in changes if ch.get("tool") == "tree_draft_insert_node")
        applied = 0
        broke = False
        for ch in changes:
            code, resp = apply_change(a.composer, ch["tool"], ch.get("args") or {}, catalog_hash, cid)
            # The bridge wraps tool-level failures as {success:false, result:{message}}
            # inside an HTTP 200 — gate on success, not just the HTTP code.
            tool_ok = isinstance(resp, dict) and resp.get("success") is not False
            if code != 200 or not tool_ok:
                msg = ((resp.get("result") or {}).get("message") if isinstance(resp, dict) else None) or str(resp)
                print(f"  ❌ {cid}: {ch['tool']} -> HTTP {code} success={isinstance(resp, dict) and resp.get('success')}: {msg[:260]}")
                coherent = False; broke = True; break
            applied += 1
        if broke:
            continue
        failures: list[str] = []
        state = capture_state(a.composer)
        # node-landed gate: an insert returning 200 but adding no node is a silent reject
        new_nodes = _count_tree_nodes(state)
        if n_inserts and (new_nodes - prev_nodes) < n_inserts:
            failures.append(f"insert did not land (tree {prev_nodes}->{new_nodes}, expected +{n_inserts})")
        prev_nodes = new_nodes
        # state_after gate where the corpus declares one
        if case and (case.get("expected") or {}).get("state_after"):
            assert_state(case, state, failures)
        mark = "✅" if not failures else "❌"
        if failures:
            coherent = False
        print(f"  {mark} {cid}: applied {applied}, tree={new_nodes}" + (f"  FAIL: {failures}" if failures else ""))

    print(f"\nGOLDEN TRAJECTORY [{a.chain}]: " + ("COHERENT ✅" if coherent else "HAS GAPS ❌"))
    if a.dump_final:
        print("--- final state ---")
        print(json.dumps(capture_state(a.composer), indent=2)[:4000])
    return 0 if coherent else 1


if __name__ == "__main__":
    raise SystemExit(main())
