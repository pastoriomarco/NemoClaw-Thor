#!/usr/bin/env python3
"""Probe the OpenClaw-routed ManyForge assistant path.

The goal is repeatable evidence, not a pass/fail replacement for the Composer
test suite. By default the script runs only read-only prompts. Mutating prompts
are opt-in because they edit the loaded Composer draft.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Case:
    name: str
    message: str
    requested_tools: tuple[str, ...] = ()
    mutating: bool = False
    timeout_s: float = 180.0


CASES: dict[str, Case] = {
    "catalog_read": Case(
        name="catalog_read",
        message="Read the catalog entry for repeat and report its node kind in one sentence.",
        requested_tools=("catalog.read",),
        timeout_s=120.0,
    ),
    "root_wrap": Case(
        name="root_wrap",
        message="Add a repeat node as root node, and make the current root node its child.",
        mutating=True,
        timeout_s=240.0,
    ),
    "runtime_reset_object": Case(
        name="runtime_reset_object",
        message=(
            "At the end of the cycle, remove the graspable object and re-add it "
            "at its current scene pose using behavior-tree nodes."
        ),
        mutating=True,
        timeout_s=240.0,
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composer-url", default="http://127.0.0.1:9000")
    parser.add_argument("--adapter-url", default="http://127.0.0.1:8200/v1/manyforge/assistant")
    parser.add_argument("--mode", default="composer-assistant")
    parser.add_argument(
        "--conversation-id",
        default=f"openclaw-smoke-{int(time.time())}",
        help="Conversation/session prefix. Defaults to a fresh id per script run.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        choices=sorted(CASES),
        help="Case to run. May be repeated. Default: catalog_read only.",
    )
    parser.add_argument(
        "--include-mutating",
        action="store_true",
        help="Allow cases that edit the loaded Composer draft.",
    )
    parser.add_argument(
        "--jsonl",
        default="",
        help="Optional path to append one JSON record per attempt.",
    )
    args = parser.parse_args()

    selected = [CASES[name] for name in (args.cases or ["catalog_read"])]
    skipped = [case.name for case in selected if case.mutating and not args.include_mutating]
    selected = [case for case in selected if not case.mutating or args.include_mutating]
    if skipped:
        print(
            "Skipping mutating case(s) without --include-mutating: "
            + ", ".join(skipped),
            file=sys.stderr,
        )
    if not selected:
        print("No cases selected.", file=sys.stderr)
        return 2

    mode_manifest = _fetch_json(
        f"{args.composer_url.rstrip('/')}/api/assistant/modes/{args.mode}",
        timeout_s=10,
    )
    tools = mode_manifest.get("tools")
    if not isinstance(tools, list) or not tools:
        raise SystemExit("Composer mode manifest did not include a non-empty tools list")

    results: list[dict[str, Any]] = []
    for repeat_index in range(args.repeat):
        for case in selected:
            request_id = f"openclaw-smoke-{case.name}-{int(time.time() * 1000)}"
            payload = {
                "version": "manyforge.assistant.provider_request.v0",
                "schemaVersion": "0.1.0",
                "requestId": request_id,
                "conversationId": f"{args.conversation_id}-{case.name}",
                "assistantMode": args.mode,
                "catalogHash": mode_manifest.get("catalogHash"),
                "message": case.message,
                "tools": tools,
                # Bare-id `nodes` / `skills` were dropped from the
                # envelope contract; consumers derive ids from
                # nodeCatalog/skillCatalog. The manifest still has
                # those rich payloads for back-compat smoke tests.
                "nodeCatalog": mode_manifest.get("nodeCatalog") or [],
                "skillCatalog": mode_manifest.get("skillCatalog") or [],
                "runtime": {"programLoaded": True, "cycleState": "idle"},
                "context": {"source": "smoke-openclaw-assistant-reliability"},
                "timeoutSeconds": case.timeout_s,
            }
            if case.requested_tools:
                payload["requestedTools"] = list(case.requested_tools)

            started = time.perf_counter()
            print(
                f"[{repeat_index + 1}/{args.repeat}] {case.name}: dispatching "
                f"(timeout {case.timeout_s:.0f}s)",
                flush=True,
            )
            try:
                body, status = _post_json(
                    args.adapter_url,
                    payload,
                    timeout_s=case.timeout_s + 30.0,
                )
            except Exception as exc:  # noqa: BLE001
                body = {"error": {"code": type(exc).__name__, "detail": str(exc)}}
                status = 0
            duration_ms = (time.perf_counter() - started) * 1000.0
            record = _summarize(case, repeat_index, status, duration_ms, body)
            results.append(record)
            print(_format_record(record), flush=True)
            if args.jsonl:
                with open(args.jsonl, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")

    failures = [
        item
        for item in results
        if item["httpStatus"] >= 400 or item["errorCode"] or item["toolFailureCount"]
    ]
    print(
        json.dumps(
            {
                "attempts": len(results),
                "failures": len(failures),
                "durationsMs": [item["durationMs"] for item in results],
            },
            sort_keys=True,
        )
    )
    return 1 if failures else 0


def _fetch_json(url: str, *, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        data = response.read()
    parsed = json.loads(data.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{url} did not return a JSON object")
    return parsed


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> tuple[dict[str, Any], int]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        data = exc.read()
        status = exc.code
    parsed = json.loads(data.decode("utf-8")) if data else {}
    if not isinstance(parsed, dict):
        parsed = {"raw": parsed}
    return parsed, status


def _summarize(
    case: Case,
    repeat_index: int,
    status: int,
    duration_ms: float,
    body: dict[str, Any],
) -> dict[str, Any]:
    tool_calls = body.get("toolCalls") if isinstance(body.get("toolCalls"), list) else []
    openclaw = body.get("openclaw") if isinstance(body.get("openclaw"), dict) else {}
    timings = openclaw.get("timings") if isinstance(openclaw.get("timings"), dict) else {}
    failed_tools = [
        call
        for call in tool_calls
        if isinstance(call, dict) and str(call.get("status")) != "completed"
    ]
    error = body.get("error") if isinstance(body.get("error"), dict) else {}
    return {
        "case": case.name,
        "repeatIndex": repeat_index,
        "httpStatus": status,
        "durationMs": round(duration_ms, 1),
        "adapterDurationMs": openclaw.get("durationMs"),
        "agentRunMs": timings.get("agentRunMs"),
        "promptChars": openclaw.get("promptChars"),
        "allowedMcpTools": openclaw.get("allowedMcpTools") or [],
        "toolCallCount": len(tool_calls),
        "toolFailureCount": len(failed_tools),
        "draftMutated": bool(body.get("draftMutated")),
        "errorCode": error.get("code"),
        "errorDetail": error.get("detail"),
        "messageHead": str(body.get("message") or "")[:240],
    }


def _format_record(record: dict[str, Any]) -> str:
    return (
        f"  status={record['httpStatus']} duration={record['durationMs']}ms "
        f"agent={record.get('agentRunMs')}ms tools={record['toolCallCount']} "
        f"failedTools={record['toolFailureCount']} error={record.get('errorCode') or '-'} "
        f"allowed={record.get('allowedMcpTools')}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
