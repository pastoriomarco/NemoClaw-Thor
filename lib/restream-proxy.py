#!/usr/bin/env python3
"""Re-streaming proxy for OpenClaw <-> vLLM.

Sits between OpenClaw (stream:true) and vLLM (stream:true).
Buffers tool-call argument fragments and re-emits them as complete chunks.
Forwards content/reasoning deltas immediately to keep the connection alive.

Usage:
    python3 restream-proxy.py [--port 8199] [--upstream http://host.openshell.internal:8000]

The proxy listens on HTTP (no TLS) and speaks HTTP or HTTPS to the upstream.
"""

import argparse
import json
import ssl
import sys
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import URLError


LOG_FILE = None


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, file=sys.stderr, flush=True)
    if LOG_FILE:
        try:
            with open(LOG_FILE, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass


def make_ssl_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


SSL_CTX = make_ssl_ctx()


class ProxyHandler(BaseHTTPRequestHandler):
    upstream: str = "https://inference.local"

    def log_message(self, fmt, *args):
        pass  # suppress default access log

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)

        if "/chat/completions" not in self.path:
            self._passthrough(body)
            return

        try:
            req_json = json.loads(body)
        except json.JSONDecodeError:
            self._passthrough(body)
            return

        is_streaming = req_json.get("stream", False)
        if not is_streaming:
            self._passthrough(body)
            return

        self._restream(body, req_json)

    def do_GET(self):
        self._passthrough(b"")

    def _passthrough(self, body):
        url = self.upstream + self.path
        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ("host", "transfer-encoding")
        }
        req = Request(url, data=body if self.command == "POST" else None,
                      headers=headers, method=self.command)
        try:
            resp = urlopen(req, context=SSL_CTX, timeout=3600)
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding",):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(resp.read())
        except URLError as e:
            log(f"passthrough error: {e}")
            self.send_error(502, str(e))

    def _restream(self, body, req_json):
        url = self.upstream + self.path
        log(f"restream: POST {url} model={req_json.get('model','?')}")
        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ("host", "transfer-encoding")
        }
        headers["Accept"] = "text/event-stream"
        req = Request(url, data=body, headers=headers, method="POST")

        try:
            resp = urlopen(req, context=SSL_CTX, timeout=3600)
        except URLError as e:
            log(f"restream upstream error: {e}")
            self.send_error(502, str(e))
            return

        log(f"restream: upstream connected, status={resp.status}")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True

        tool_calls = {}
        finish_reason = None
        last_chunk_base = None
        events_forwarded = 0
        events_buffered = 0

        try:
            buf = b""
            for line_bytes in resp:
                buf += line_bytes
                while b"\n\n" in buf:
                    event_bytes, buf = buf.split(b"\n\n", 1)
                    event_str = event_bytes.decode("utf-8", errors="replace").strip()

                    if not event_str:
                        continue

                    if event_str == "data: [DONE]":
                        if tool_calls:
                            self._flush_tool_calls(tool_calls, finish_reason,
                                                   last_chunk_base)
                            log(f"restream: flushed {len(tool_calls)} tool calls")
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        log(f"restream: done. forwarded={events_forwarded} "
                            f"buffered={events_buffered}")
                        return

                    if not event_str.startswith("data: "):
                        self.wfile.write(event_bytes + b"\n\n")
                        self.wfile.flush()
                        continue

                    data_str = event_str[6:]
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        self.wfile.write(event_bytes + b"\n\n")
                        self.wfile.flush()
                        events_forwarded += 1
                        continue

                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta", {})
                    fr = choice.get("finish_reason")
                    if fr:
                        finish_reason = fr

                    tc_deltas = delta.get("tool_calls")
                    has_content = "content" in delta
                    has_reasoning = "reasoning_content" in delta

                    if tc_deltas:
                        for tc in tc_deltas:
                            idx = tc.get("index", 0)
                            if idx not in tool_calls:
                                tool_calls[idx] = {
                                    "id": tc.get("id", ""),
                                    "type": tc.get("type", "function"),
                                    "function": {
                                        "name": tc.get("function", {}).get("name", ""),
                                        "arguments": "",
                                    },
                                }
                            else:
                                if tc.get("id"):
                                    tool_calls[idx]["id"] = tc["id"]
                                fn = tc.get("function", {})
                                if fn.get("name"):
                                    tool_calls[idx]["function"]["name"] = fn["name"]
                            args_frag = tc.get("function", {}).get("arguments", "")
                            if args_frag:
                                tool_calls[idx]["function"]["arguments"] += args_frag
                        events_buffered += 1
                        last_chunk_base = chunk

                        if has_content or has_reasoning:
                            fwd_delta = {}
                            if has_content:
                                fwd_delta["content"] = delta["content"]
                            if has_reasoning:
                                fwd_delta["reasoning_content"] = delta["reasoning_content"]
                            if "role" in delta:
                                fwd_delta["role"] = delta["role"]
                            fwd_chunk = dict(chunk)
                            fwd_choice = dict(choice)
                            fwd_choice["delta"] = fwd_delta
                            fwd_choice.pop("finish_reason", None)
                            fwd_chunk["choices"] = [fwd_choice]
                            self._send_sse(fwd_chunk)
                            events_forwarded += 1
                    else:
                        self.wfile.write(event_bytes + b"\n\n")
                        self.wfile.flush()
                        events_forwarded += 1

        except Exception as e:
            log(f"restream error: {type(e).__name__}: {e}")

        if tool_calls:
            self._flush_tool_calls(tool_calls, finish_reason, last_chunk_base)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        log(f"restream: ended (loop exit). forwarded={events_forwarded} "
            f"buffered={events_buffered}")

    def _flush_tool_calls(self, tool_calls, finish_reason, base_chunk):
        """Emit buffered tool calls in incremental format matching the OpenAI
        streaming protocol:
          1. One chunk per tool call with id/type/name (arguments="")
          2. One chunk per tool call with the complete arguments string
          3. One final chunk with finish_reason and empty delta
        This avoids issues where SDKs finalize on finish_reason before
        processing the delta, or skip arguments in the initial chunk."""
        base_id = base_chunk.get("id", "") if base_chunk else ""
        base_created = base_chunk.get("created", 0) if base_chunk else 0
        base_model = base_chunk.get("model", "") if base_chunk else ""

        def _make_chunk(tc_deltas, fr=None):
            c = {
                "id": base_id,
                "object": "chat.completion.chunk",
                "created": base_created,
                "model": base_model,
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": tc_deltas} if tc_deltas else {},
                    "finish_reason": fr,
                }],
            }
            return c

        # Step 1: emit initial chunk for each tool call (id, type, name)
        init_deltas = []
        for idx in sorted(tool_calls.keys()):
            tc = tool_calls[idx]
            init_deltas.append({
                "index": idx,
                "id": tc["id"],
                "type": tc["type"],
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": "",
                },
            })
        self._send_sse(_make_chunk(init_deltas))

        # Step 2: emit arguments for each tool call
        for idx in sorted(tool_calls.keys()):
            tc = tool_calls[idx]
            args = tc["function"]["arguments"]
            if args:
                self._send_sse(_make_chunk([{
                    "index": idx,
                    "function": {"arguments": args},
                }]))
                log(f"  tool_call: {tc['function']['name']} args_len={len(args)}")

        # Step 3: emit finish_reason chunk (empty delta)
        finish_chunk = _make_chunk(None, fr=finish_reason or "tool_calls")
        if base_chunk and "usage" in base_chunk:
            finish_chunk["usage"] = base_chunk["usage"]
        self._send_sse(finish_chunk)

    def _send_sse(self, chunk):
        data = json.dumps(chunk, separators=(",", ":"))
        self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.flush()


def main():
    global LOG_FILE
    parser = argparse.ArgumentParser(description="Re-streaming proxy")
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--upstream", default="http://host.openshell.internal:8000")
    parser.add_argument("--log", default="/sandbox/.nemoclaw/restream-proxy.log")
    args = parser.parse_args()

    LOG_FILE = args.log
    ProxyHandler.upstream = args.upstream.rstrip("/")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), ProxyHandler)
    log(f"listening on 127.0.0.1:{args.port} -> {args.upstream}")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        thread.join()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
