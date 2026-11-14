#!/usr/bin/env python3
"""OpenAI-compatible pass-through proxy that prints live throughput per request.

    python tps_proxy.py --listen 8001 --upstream http://127.0.0.1:8000

Streaming chat/completions requests get stream_options.include_usage so the
final line reports exact token counts; while a response streams, a progress
line is printed every --interval seconds (output chunks counted as tokens,
an undercount when one chunk carries several speculative tokens).
"""
import argparse
import itertools
import json
import time

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

ap = argparse.ArgumentParser()
ap.add_argument("--listen", type=int, default=8001)
ap.add_argument("--upstream", default="http://127.0.0.1:8000")
ap.add_argument("--interval", type=float, default=2.0)
ap.add_argument("--dump-dir", default=None,
                help="keep the last --dump-keep request bodies and streamed replies here")
ap.add_argument("--dump-keep", type=int, default=20)
ap.add_argument("--dump-max-mb", type=int, default=0,
                help="if > 0, keep dumps up to this total size (oldest removed first) instead of --dump-keep")
ap.add_argument("--force-reasoning-effort", default=None, choices=["low", "medium", "xhigh"],
                help="set chat_template_kwargs.reasoning_effort on every chat request, replacing the client's")
ap.add_argument("--model-override", default=None,
                help="replace the 'model' field of every JSON request (client asks for another served name)")
ap.add_argument("--thinking-effort", default=None, choices=["low", "medium", "xhigh"],
                help="turn thinking on at this effort for chat requests that do not set "
                     "chat_template_kwargs.enable_thinking (Edge-LLM defaults to non-thinking)")
args = ap.parse_args()


def dump(n, kind, data: bytes):
    if not args.dump_dir:
        return
    import os
    os.makedirs(args.dump_dir, exist_ok=True)
    with open(f"{args.dump_dir}/{n:06d}.{kind}", "wb") as f:
        f.write(data)
    names = sorted(os.listdir(args.dump_dir))
    if args.dump_max_mb > 0:
        sizes = [os.path.getsize(f"{args.dump_dir}/{name}") for name in names]
        total, cap = sum(sizes), args.dump_max_mb << 20
        stale = []
        for name, size in zip(names, sizes):
            if total <= cap:
                break
            stale.append(name)
            total -= size
    else:
        stale = names[:-2 * args.dump_keep]
    for name in stale:
        os.remove(f"{args.dump_dir}/{name}")

client = httpx.AsyncClient(base_url=args.upstream, timeout=None)
ids = itertools.count(1)
HOP = {"host", "content-length", "transfer-encoding", "connection"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def summary(n, t0, t_first, t_end, prompt, completion, cached, chunks):
    ttft = (t_first - t0) if t_first else None
    decode = (t_end - t_first) if t_first else None
    tokens = completion if completion is not None else chunks
    rate = (tokens - 1) / decode if decode and tokens > 1 else 0.0
    exact = "" if completion is not None else " (chunk count)"
    # decode= assumes tokens stream as generated; a server that sends reasoning or tool calls
    # in late bursts inflates it, so e2e= (output / whole request time) is the safe figure.
    log(f"[#{n}] done: prompt={prompt} cached={cached} out={tokens}{exact} "
        f"ttft={ttft:.2f}s decode={rate:.1f} tok/s e2e={tokens / (t_end - t0):.1f} tok/s "
        f"total={t_end - t0:.1f}s chunks={chunks}"
        if ttft is not None else f"[#{n}] done: no output, total={t_end - t0:.1f}s")


async def proxy(request: Request):
    n = next(ids)
    path = request.url.path
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    payload = None
    if request.method == "POST" and body:
        try:
            payload = json.loads(body)
        except ValueError:
            payload = None
    streaming = bool(payload and payload.get("stream"))
    if payload and args.model_override and "model" in payload:
        payload["model"] = args.model_override
        body = json.dumps(payload).encode()
    if payload and path.endswith("/chat/completions"):
        if streaming:
            payload.setdefault("stream_options", {}).setdefault("include_usage", True)
        kwargs = payload.setdefault("chat_template_kwargs", {})
        if args.thinking_effort and "enable_thinking" not in kwargs and "enable_thinking" not in payload:
            kwargs["enable_thinking"] = True
            kwargs.setdefault("reasoning_effort", args.thinking_effort)
        if args.force_reasoning_effort:
            kwargs["reasoning_effort"] = args.force_reasoning_effort
        body = json.dumps(payload).encode()
    if payload:
        dump(n, "request.json", body)
        log(f"[#{n}] {path} model={payload.get('model')} stream={streaming} "
            f"messages={len(payload.get('messages') or [])} tools={len(payload.get('tools') or [])}")

    upstream_req = client.build_request(request.method, path, params=request.query_params,
                                        headers=headers, content=body)
    t0 = time.monotonic()
    upstream = await client.send(upstream_req, stream=True)
    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP}

    if not streaming:
        data = await upstream.aread()
        await upstream.aclose()
        t_end = time.monotonic()
        if payload:
            try:
                usage = json.loads(data).get("usage") or {}
            except ValueError:
                usage = {}
            out = usage.get("completion_tokens")
            rate = f" ~{out / (t_end - t0):.1f} tok/s incl. prefill" if out else ""
            log(f"[#{n}] done: status={upstream.status_code} prompt={usage.get('prompt_tokens')} "
                f"out={out} total={t_end - t0:.1f}s{rate}")
        return Response(data, status_code=upstream.status_code, headers=out_headers)

    async def relay():
        t_first = None
        last_print = t0
        chunks = 0
        usage = {}
        buffer = b""
        replay = b""
        try:
            async for raw in upstream.aiter_raw():
                yield raw
                replay += raw
                buffer += raw
                *lines, buffer = buffer.split(b"\n")
                for line in lines:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    try:
                        event = json.loads(line[6:])
                    except ValueError:
                        continue
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        # vLLM >= 0.33 streams reasoning as "reasoning"; older servers use "reasoning_content".
                        if (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
                                or delta.get("tool_calls")):
                            chunks += 1
                            if t_first is None:
                                t_first = time.monotonic()
                                log(f"[#{n}] first token after {t_first - t0:.2f}s")
                now = time.monotonic()
                if t_first and now - last_print >= args.interval:
                    last_print = now
                    log(f"[#{n}] streaming: {chunks} chunks, {chunks / (now - t_first):.1f} chunks/s")
        finally:
            await upstream.aclose()
            dump(n, "reply.sse", replay)
            details = usage.get("prompt_tokens_details") or {}
            summary(n, t0, t_first, time.monotonic(), usage.get("prompt_tokens"),
                    usage.get("completion_tokens"), details.get("cached_tokens"), chunks)

    return StreamingResponse(relay(), status_code=upstream.status_code, headers=out_headers)


app = Starlette(routes=[Route("/{path:path}", proxy, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])])

if __name__ == "__main__":
    log(f"tps proxy :{args.listen} -> {args.upstream}")
    uvicorn.run(app, host="0.0.0.0", port=args.listen, log_level="warning")
