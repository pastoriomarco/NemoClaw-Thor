#!/usr/bin/env python3
"""Gate honest max_tokens cutoffs inside DS4 continuous tool calls."""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request


PROMPT = "Look up the current weather in Paris with the get_weather tool. You must call the tool."
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


def log_counts(container: str) -> tuple[int, int]:
    result = subprocess.run(
        ["docker", "logs", container], capture_output=True, text=True, check=True
    )
    text = result.stdout + result.stderr
    return (
        text.count("tool call cut by token budget"),
        text.count("tool-error continuation appended"),
    )


def request(url: str, model: str, limit: int, stream: bool, timeout: int):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": TOOLS,
        "temperature": 0,
        "reasoning_effort": "off",
        "max_tokens": limit,
        "stream": stream,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode()
    if not stream:
        return json.loads(raw)
    return [
        json.loads(line[6:])
        for line in raw.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8050/v1")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--container", default="nemoclaw-ds4-ds4-1")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"

    control = request(endpoint, args.model, 600, False, args.timeout)
    choice = control["choices"][0]
    completion = int(control["usage"]["completion_tokens"])
    if choice["finish_reason"] != "tool_calls" or not choice["message"].get("tool_calls"):
        raise RuntimeError("control did not emit a complete tool call")
    print(f"control finish=tool_calls completion={completion}", flush=True)

    cut = completion - 6
    for attempt in range(1, 4):
        before = log_counts(args.container)
        body = request(endpoint, args.model, cut, False, args.timeout)
        after = log_counts(args.container)
        choice = body["choices"][0]
        message = choice["message"]
        engagement = after[0] - before[0]
        recovery = after[1] - before[1]
        print(
            f"buffered attempt={attempt} limit={cut} finish={choice['finish_reason']} "
            f"completion={body['usage']['completion_tokens']} "
            f"content_len={len(message.get('content') or '')} "
            f"engagement={engagement} recovery={recovery}",
            flush=True,
        )
        if engagement:
            if choice["finish_reason"] != "length":
                raise RuntimeError("buffered cutoff did not report length")
            if int(body["usage"]["completion_tokens"]) != cut:
                raise RuntimeError("buffered cutoff exceeded max_tokens")
            if message.get("tool_calls") or not message.get("content"):
                raise RuntimeError("buffered cutoff did not preserve partial assistant text")
            if recovery:
                raise RuntimeError("buffered cutoff ran tool-error recovery")
            break
        if choice["finish_reason"] != "tool_calls":
            raise RuntimeError("buffered cutoff missed both complete and partial tool states")
        cut = int(body["usage"]["completion_tokens"]) - 6
    else:
        raise RuntimeError("buffered cutoff never landed inside the tool call")

    for attempt in range(1, 3):
        before = log_counts(args.container)
        chunks = request(endpoint, args.model, cut, True, args.timeout)
        after = log_counts(args.container)
        finishes = [
            chunk["choices"][0].get("finish_reason")
            for chunk in chunks
            if chunk.get("choices") and chunk["choices"][0].get("finish_reason")
        ]
        engagement = after[0] - before[0]
        recovery = after[1] - before[1]
        finish = finishes[-1] if finishes else None
        print(
            f"stream attempt={attempt} limit={cut} finish={finish} "
            f"engagement={engagement} recovery={recovery}",
            flush=True,
        )
        if engagement:
            if finish != "length" or recovery:
                raise RuntimeError("stream cutoff was not an honest length stop")
            print("TOOL_BUDGET_GATE=PASS")
            return 0
        cut -= 4

    raise RuntimeError("stream cutoff never landed inside the tool call")


if __name__ == "__main__":
    raise SystemExit(main())
