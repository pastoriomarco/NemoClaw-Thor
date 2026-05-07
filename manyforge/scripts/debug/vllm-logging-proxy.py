#!/usr/bin/env python3
"""HTTP reverse proxy that logs every request/response body.

Sits between the openclaw_assistant_bridge and the OpenClaw gateway,
so we can capture the exact JSON the bridge sends to the gateway and
the exact JSON the gateway returns — including the `tools[]` array,
`tool_choice` value, `messages` / `input` content, and the
`choices[]` / `output[]` reply. tcpdump can't reliably parse our
multi-100KB preamble bodies; this proxy parses them for free.

Usage:
  python3 openclaw_logging_proxy.py [--listen-port 18790]
                                    [--upstream http://127.0.0.1:18789]
                                    [--log-path /tmp/openclaw_proxy.jsonl]

Environment variables (override flags):
  OPENCLAW_PROXY_LISTEN_PORT
  OPENCLAW_PROXY_UPSTREAM
  OPENCLAW_PROXY_LOG_PATH

Then point the bridge at the proxy by setting
  OPENCLAW_ASSISTANT_GATEWAY_PORT=<listen-port>
on the bridge service before starting it.

Each upstream call is appended to the JSONL log as one record:
  {
    "ts": <unix_ms>,
    "request": {
      "method": "POST",
      "path": "/v1/chat/completions",
      "headers": {...},
      "body": <parsed json or raw string>
    },
    "response": {
      "status": 200,
      "headers": {...},
      "body": <parsed json or raw string>,
      "duration_ms": 123.4
    }
  }

The harness reads this JSONL the same way it reads the wrapper
delta-log — by byte offset diff per request — so the per-request
view stays scoped.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import socketserver
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler


_LOG_LOCK = threading.Lock()
_LOG_PATH = "/tmp/openclaw_proxy.jsonl"
_UPSTREAM_HOST = "127.0.0.1"
_UPSTREAM_PORT = 18789
_UPSTREAM_SCHEME = "http"
_LISTEN_PORT = 18790


def _try_parse_json(blob: bytes) -> tuple[object | None, str]:
    """Return (parsed, raw_text). parsed is None when not valid JSON."""
    try:
        text = blob.decode("utf-8", errors="replace")
    except Exception:
        text = repr(blob)[:2048]
    try:
        return json.loads(text), text
    except Exception:
        return None, text


def _truncate(text: str, limit: int = 8192) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"…<truncated {len(text) - limit} chars>"


def _append_log(record: dict) -> None:
    line = json.dumps(record, sort_keys=False, default=str)
    with _LOG_LOCK:
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


class ProxyHandler(BaseHTTPRequestHandler):
    """One handler per request. Reads the full body, forwards
    upstream, captures the full response, logs both."""

    # We never serve our own pages — only forward.
    server_version = "OpenClawProxy/0.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        # Quiet the default access log; we have our own JSONL.
        return

    def _read_request_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _forward(self, method: str) -> None:
        path = self.path
        body = self._read_request_body()
        # Strip hop-by-hop headers per RFC 7230 §6.1.
        hop_headers = {
            "connection", "keep-alive", "proxy-authenticate",
            "proxy-authorization", "te", "trailer", "transfer-encoding",
            "upgrade", "host", "content-length",
        }
        forward_headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in hop_headers
        }
        # Force Content-Length on forwarded request.
        forward_headers["Content-Length"] = str(len(body))

        ts_in = time.time()
        request_record = {
            "method": method,
            "path": path,
            "headers": dict(forward_headers),
            "body_chars": len(body),
        }
        body_json, body_raw = _try_parse_json(body)
        if body_json is not None:
            request_record["body"] = body_json
        else:
            request_record["body_raw_excerpt"] = _truncate(body_raw, 4096)

        # Forward to upstream.
        try:
            conn = http.client.HTTPConnection(
                _UPSTREAM_HOST, _UPSTREAM_PORT, timeout=600.0,
            )
            conn.request(method, path, body=body, headers=forward_headers)
            up_resp = conn.getresponse()
            resp_body = up_resp.read()
            resp_status = up_resp.status
            resp_headers = dict(up_resp.getheaders())
            conn.close()
        except Exception as exc:
            # Surface the proxy failure as a 502 to the bridge so it's
            # not confused with a gateway error; log the failure too.
            err_text = f"proxy upstream error: {type(exc).__name__}: {exc}"
            _append_log({
                "ts": int(ts_in * 1000),
                "request": request_record,
                "response": {
                    "status": -1,
                    "error": err_text,
                    "duration_ms": round((time.time() - ts_in) * 1000.0, 1),
                },
            })
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(err_text)))
            self.end_headers()
            self.wfile.write(err_text.encode())
            return

        duration_ms = (time.time() - ts_in) * 1000.0
        resp_body_json, resp_body_raw = _try_parse_json(resp_body)
        response_record: dict = {
            "status": resp_status,
            "headers": resp_headers,
            "duration_ms": round(duration_ms, 1),
            "body_chars": len(resp_body),
        }
        if resp_body_json is not None:
            response_record["body"] = resp_body_json
        else:
            response_record["body_raw_excerpt"] = _truncate(resp_body_raw, 4096)

        _append_log({
            "ts": int(ts_in * 1000),
            "request": request_record,
            "response": response_record,
        })

        # Forward response to the bridge. Strip hop-by-hop again on
        # the way back; preserve Content-Type and any custom headers.
        self.send_response(resp_status)
        for hk, hv in resp_headers.items():
            if hk.lower() in hop_headers:
                continue
            self.send_header(hk, hv)
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def do_POST(self) -> None:  # noqa: N802
        self._forward("POST")

    def do_GET(self) -> None:  # noqa: N802
        self._forward("GET")

    def do_PUT(self) -> None:  # noqa: N802
        self._forward("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self._forward("DELETE")


class ThreadedProxyServer(socketserver.ThreadingMixIn,
                          socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def _resolve_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-port", type=int, default=int(
        os.environ.get("OPENCLAW_PROXY_LISTEN_PORT", "18790")))
    parser.add_argument("--bind", type=str, default=os.environ.get(
        "OPENCLAW_PROXY_BIND", "127.0.0.1"),
        help="Bind address (use 0.0.0.0 to be reachable from "
             "containerised callers via the docker bridge IP).")
    parser.add_argument("--upstream", type=str, default=os.environ.get(
        "OPENCLAW_PROXY_UPSTREAM", "http://127.0.0.1:18789"))
    parser.add_argument("--log-path", type=str, default=os.environ.get(
        "OPENCLAW_PROXY_LOG_PATH", "/tmp/openclaw_proxy.jsonl"))
    return parser.parse_args()


def main() -> None:
    global _UPSTREAM_HOST, _UPSTREAM_PORT, _UPSTREAM_SCHEME
    global _LISTEN_PORT, _LOG_PATH

    cfg = _resolve_config()
    parsed = urllib.parse.urlparse(cfg.upstream)
    _UPSTREAM_SCHEME = parsed.scheme or "http"
    _UPSTREAM_HOST = parsed.hostname or "127.0.0.1"
    _UPSTREAM_PORT = parsed.port or (443 if _UPSTREAM_SCHEME == "https" else 18789)
    _LISTEN_PORT = cfg.listen_port
    _LOG_PATH = cfg.log_path

    # Truncate prior log so each session starts fresh; harness handles
    # offset-from-baseline anyway, but a clean file on launch is
    # convenient for ad-hoc cat'ing.
    open(_LOG_PATH, "w").close()

    server = ThreadedProxyServer((cfg.bind, _LISTEN_PORT), ProxyHandler)
    print(
        f"openclaw-logging-proxy listening on {cfg.bind}:{_LISTEN_PORT} "
        f"-> {_UPSTREAM_SCHEME}://{_UPSTREAM_HOST}:{_UPSTREAM_PORT} "
        f"(log: {_LOG_PATH})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutdown", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
