#!/usr/bin/env python3
"""A/B reliability + latency probe: direct vLLM vs canonical OpenClaw gateway.

Hits the same prompt suite through both inference paths and prints a
comparison table. The point is reproducible evidence, not a pass/fail
replacement for the test suite.

Path A — direct vLLM:
    POST http://127.0.0.1:8000/v1/chat/completions
    model = "<vLLM served-model-name>"
    chat_template_kwargs forced to enable_thinking=false +
    force_nonempty_content=true so Nemotron-3 returns visible content
    rather than reasoning-only blocks (openclaw#71847).

Path B — canonical OpenClaw gateway:
    POST http://127.0.0.1:18789/v1/chat/completions
    model = "openclaw/manyforge-composer"
    Auth via gateway-token if the gateway is in --auth token mode.
    (force_nonempty_content is now applied via openclaw config set
    agents.defaults.models[<id>].params.chat_template_kwargs.)

Usage:
    ./ab-direct-vs-openclaw.py            # default: 3 prompts x N=3 each
    ./ab-direct-vs-openclaw.py --runs 5
    ./ab-direct-vs-openclaw.py --json /tmp/ab.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


VLLM_BASE = "http://127.0.0.1:8000"
GATEWAY_BASE = "http://127.0.0.1:18789"
VLLM_MODEL = "nemotron3-nano-omni-30b-a3b-nvfp4"
GATEWAY_MODEL = "openclaw/manyforge-composer"


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str
    max_tokens: int = 64


PROMPTS: list[Prompt] = [
    Prompt("trivial_ok", "Reply with just OK", max_tokens=16),
    Prompt(
        "short_factual",
        "What does the abbreviation MCP stand for in the OpenClaw context? Answer in one short sentence.",
        max_tokens=80,
    ),
    Prompt(
        "describe_repeat",
        "In one sentence, what does a behavior-tree 'repeat' decorator node do?",
        max_tokens=80,
    ),
]


@dataclass
class Sample:
    path: str
    prompt: str
    wall_s: float
    success: bool
    content_len: int
    content_preview: str
    error: str | None = None


def get_gateway_token() -> str | None:
    try:
        out = subprocess.run(
            ["nemoclaw", "my-assistant", "gateway-token"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        first_line = (out.stdout or "").splitlines()
        return first_line[0].strip() if first_line else None
    except Exception:
        return None


def post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float) -> tuple[dict[str, Any] | None, str | None]:
    """One POST, return (parsed_json, error_string)."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout_s)
        raw = resp.read()
        try:
            return json.loads(raw), None
        except (ValueError, TypeError) as exc:
            return None, f"parse-error: {exc} (head={raw[:200]!r})"
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return None, f"HTTP {exc.code}: {body[:240]}"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def run_direct_vllm(prompt: Prompt, timeout_s: float = 120.0) -> Sample:
    body = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": prompt.text}],
        "max_tokens": prompt.max_tokens,
        "stream": False,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "force_nonempty_content": True,
        },
    }
    started = time.perf_counter()
    parsed, err = post_json(
        f"{VLLM_BASE}/v1/chat/completions",
        body,
        headers={},
        timeout_s=timeout_s,
    )
    wall = time.perf_counter() - started
    if err is not None:
        return Sample("direct_vllm", prompt.name, wall, False, 0, "", err)
    content = ""
    try:
        content = parsed["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        pass
    return Sample(
        "direct_vllm",
        prompt.name,
        wall,
        bool(content.strip()),
        len(content),
        content[:80].replace("\n", " "),
        None,
    )


def run_gateway(prompt: Prompt, token: str | None, timeout_s: float = 120.0) -> Sample:
    body = {
        "model": GATEWAY_MODEL,
        "messages": [{"role": "user", "content": prompt.text}],
        "max_tokens": prompt.max_tokens,
        "stream": False,
    }
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    started = time.perf_counter()
    parsed, err = post_json(
        f"{GATEWAY_BASE}/v1/chat/completions",
        body,
        headers=headers,
        timeout_s=timeout_s,
    )
    wall = time.perf_counter() - started
    if err is not None:
        return Sample("openclaw_gw", prompt.name, wall, False, 0, "", err)
    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
        return Sample(
            "openclaw_gw",
            prompt.name,
            wall,
            False,
            0,
            "",
            parsed["error"].get("message") or "gateway error",
        )
    content = ""
    try:
        content = parsed["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        pass
    return Sample(
        "openclaw_gw",
        prompt.name,
        wall,
        bool(content.strip()),
        len(content),
        content[:80].replace("\n", " "),
        None,
    )


def summarize(samples: list[Sample]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_path: dict[str, list[Sample]] = {"direct_vllm": [], "openclaw_gw": []}
    for s in samples:
        by_path.setdefault(s.path, []).append(s)
    for path, group in by_path.items():
        successes = [s for s in group if s.success]
        wall_times = [s.wall_s for s in group]
        out[path] = {
            "runs": len(group),
            "success": len(successes),
            "success_rate": (len(successes) / len(group)) if group else 0.0,
            "p50_s": statistics.median(wall_times) if wall_times else 0.0,
            "p95_s": (
                statistics.quantiles(wall_times, n=20)[-1]
                if len(wall_times) >= 2
                else (wall_times[0] if wall_times else 0.0)
            ),
            "min_s": min(wall_times) if wall_times else 0.0,
            "max_s": max(wall_times) if wall_times else 0.0,
        }
    return out


def print_table(samples: list[Sample]) -> None:
    print()
    print(f"{'path':<14} {'prompt':<22} {'#':>2} {'wall':>7} {'ok':>3} {'len':>5}  preview")
    print("-" * 100)
    for s in samples:
        ok = "✓" if s.success else "✗"
        print(
            f"{s.path:<14} {s.prompt:<22} {1:>2} "
            f"{s.wall_s:>6.2f}s {ok:>3} {s.content_len:>5}  {s.content_preview}"
        )
    print()
    print("Aggregates (P50 / P95 / success rate):")
    summary = summarize(samples)
    for path, agg in summary.items():
        print(
            f"  {path:<14} runs={agg['runs']} succ={agg['success']}/{agg['runs']} "
            f"({agg['success_rate']*100:.0f}%)  "
            f"P50={agg['p50_s']:.2f}s  P95={agg['p95_s']:.2f}s  "
            f"min={agg['min_s']:.2f}s  max={agg['max_s']:.2f}s"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="Runs per prompt per path (default: 3)")
    parser.add_argument("--json", help="Optional path to write full results as JSON")
    parser.add_argument(
        "--prompt", action="append",
        help="Run only the named prompt(s) (repeatable). Default: all.",
    )
    parser.add_argument(
        "--paths", default="direct_vllm,openclaw_gw",
        help="Comma-separated subset of paths to run.",
    )
    args = parser.parse_args()
    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    selected_prompts = PROMPTS
    if args.prompt:
        selected_prompts = [p for p in PROMPTS if p.name in args.prompt]
    if not selected_prompts:
        print("no prompts selected", file=sys.stderr)
        return 2

    token = get_gateway_token() if "openclaw_gw" in paths else None

    all_samples: list[Sample] = []
    for prompt in selected_prompts:
        for run_idx in range(args.runs):
            if "direct_vllm" in paths:
                s = run_direct_vllm(prompt)
                all_samples.append(s)
                ok = "✓" if s.success else "✗"
                print(f"  direct_vllm  {prompt.name:<22} run {run_idx+1}/{args.runs}: {s.wall_s:.2f}s {ok}", flush=True)
            if "openclaw_gw" in paths:
                s = run_gateway(prompt, token)
                all_samples.append(s)
                ok = "✓" if s.success else "✗"
                print(f"  openclaw_gw  {prompt.name:<22} run {run_idx+1}/{args.runs}: {s.wall_s:.2f}s {ok}", flush=True)

    print_table(all_samples)

    if args.json:
        out = {
            "samples": [s.__dict__ for s in all_samples],
            "summary": summarize(all_samples),
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nFull results written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
