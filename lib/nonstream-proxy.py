#!/usr/bin/env python3
"""Non-streaming-to-SSE proxy for vLLM. Bypasses streaming tool_call_parser bug.
Also injects chat_template_kwargs to disable thinking mode.

The qwen3_coder streaming tool_call_parser in vLLM corrupts large tool-call
arguments across SSE chunks (IndexError / empty arguments).  This proxy sends
stream:false to vLLM, waits for the complete response, then converts it to
chunked SSE that OpenClaw expects.  Tool calls arrive intact regardless of size.

Usage (inside sandbox pod):
    python3 nonstream-proxy.py                     # defaults: port 8199, upstream host.openshell.internal:8000
    python3 nonstream-proxy.py --port 8199 --upstream http://host.openshell.internal:8000
"""
import http.server, json, urllib.request, urllib.error, sys, uuid, time

UPSTREAM = "http://host.openshell.internal:8000"  # Direct to vLLM, no HTTPS
PORT = 8199

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        # Proxy GET requests (e.g. /v1/models) directly
        url = UPSTREAM + self.path
        try:
            resp = urllib.request.urlopen(url, timeout=30)
            body = resp.read()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding",):
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_response(502)
            err = str(e).encode()
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(cl)
        url = UPSTREAM + self.path
        d = json.loads(body)
        d["chat_template_kwargs"] = {"enable_thinking": False}
        d["stream"] = False  # Force non-streaming
        body = json.dumps(d).encode()
        req = urllib.request.Request(url, data=body,
            headers={"Content-Type": "application/json",
                     "Content-Length": str(len(body))},
            method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=600)
            full = json.loads(resp.read())
            # Convert to SSE format
            cid = full.get("id", "chatcmpl-" + uuid.uuid4().hex[:16])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            # Emit chunks...
            for choice in full.get("choices", []):
                msg = choice.get("message", {})
                # Role chunk
                chunk = {"id": cid, "object": "chat.completion.chunk",
                         "choices": [{"index": 0,
                                      "delta": {"role": msg.get("role", "assistant")},
                                      "finish_reason": None}]}
                self._send_sse(chunk)
                # Content chunk (if any)
                if msg.get("content"):
                    chunk["choices"][0]["delta"] = {"content": msg["content"]}
                    self._send_sse(chunk)
                # Tool calls (if any)
                for tc in msg.get("tool_calls", []):
                    chunk["choices"][0]["delta"] = {"tool_calls": [tc]}
                    self._send_sse(chunk)
                # Finish chunk
                chunk["choices"][0] = {"index": 0, "delta": {},
                    "finish_reason": choice.get("finish_reason", "stop")}
                if "usage" in full: chunk["usage"] = full["usage"]
                self._send_sse(chunk)
            self._send_chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except urllib.error.HTTPError as e:
            rb = e.read()
            self.send_response(e.code)
            self.send_header("Content-Length", str(len(rb)))
            self.end_headers()
            self.wfile.write(rb)

    def _send_sse(self, obj):
        self._send_chunk(("data: " + json.dumps(obj) + "\n\n").encode())
    def _send_chunk(self, data):
        self.wfile.write(f"{len(data):x}\r\n".encode())
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[proxy] {fmt % args}\n"); sys.stderr.flush()

if __name__ == "__main__":
    s = http.server.HTTPServer(("127.0.0.1", PORT), H)
    print(f"nothink-proxy on 127.0.0.1:{PORT}", flush=True)
    s.serve_forever()
