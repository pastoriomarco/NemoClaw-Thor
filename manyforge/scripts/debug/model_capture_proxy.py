#!/usr/bin/env python3
"""Evidence proxy: forwards OpenAI-compatible requests to a model server and
records every chat-completion request body (system prompt, messages, tools)
and its response to a JSONL file. Size-capped. Changes payloads only with
--sampling (an experiment override; null removes a key so the server default applies).

    model_capture_proxy.py --listen 8001 --upstream http://127.0.0.1:8000 \
        --out ~/assist-phaseA/runs/<run>.model-requests.jsonl
"""
import argparse, hashlib, json, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--listen", type=int, default=8001)
ap.add_argument("--upstream", default="http://127.0.0.1:8000")
ap.add_argument("--out", required=True)
ap.add_argument("--max-bytes", type=int, default=2 * 1024 * 1024 * 1024)
ap.add_argument("--case-file", required=True,
                help="JSON {case_id, turn} the corpus runner writes before each submit")
ap.add_argument("--sampling", default=None,
                help='JSON merged into every chat-completion request, e.g. {"temperature":0.6}; null removes a key')
args = ap.parse_args()
sampling = json.loads(args.sampling) if args.sampling else None
lock = threading.Lock()
sink = open(args.out, "a", encoding="utf-8")
written = sink.tell()  # the cap bounds the whole file, also when appending to an earlier run
capped = False
last_tools: dict = {"case": None, "digest": None}
call_index: dict = {}


def current_case():
    try:
        with open(args.case_file) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"case_id": None, "turn": None}


def record(row):
    global written, capped
    with lock:
        if capped:
            return
        case = current_case()
        key = (case.get("case_id"), case.get("turn"))
        call_index[key] = call_index.get(key, 0) + 1
        row = {"case_id": key[0], "corpus_turn": key[1], "model_call": call_index[key], **row}
        request = row.get("request")
        if isinstance(request, dict) and "tools" in request:
            digest = hashlib.sha256(json.dumps(request["tools"], sort_keys=True).encode()).hexdigest()
            if last_tools == {"case": key[0], "digest": digest}:
                request = dict(request)
                request["tools"] = {"deduplicated": True, "sha256": digest,
                                    "same_as": "first call of this case"}
                row["request"] = request
            else:
                last_tools.update(case=key[0], digest=digest)
                row["tools_sha256"] = digest
        line = json.dumps(row, ensure_ascii=False) + "\n"
        if written + len(line.encode()) > args.max_bytes:
            sink.write(json.dumps({"marker": "capture_cap_reached", "max_bytes": args.max_bytes,
                                   "t": time.time()}) + "\n")
            sink.flush()
            capped = True
            return
        written += len(line.encode())
        sink.write(line)
        sink.flush()


class Proxy(BaseHTTPRequestHandler):
    def _forward(self, body):
        req = urllib.request.Request(args.upstream + self.path, data=body, method=self.command,
                                     headers={k: v for k, v in self.headers.items()
                                              if k.lower() not in ("host", "content-length")})
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def _reply(self, status, headers, data):
        self.send_response(status)
        for k, v in headers.items():
            if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._reply(*self._forward(None))

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if sampling and self.path.endswith("/chat/completions"):
            payload = json.loads(body)
            for key, value in sampling.items():
                if value is None:
                    payload.pop(key, None)
                else:
                    payload[key] = value
            body = json.dumps(payload).encode()
        t0 = time.time()
        status, headers, data = self._forward(body)
        if self.path.endswith("/chat/completions"):
            try:
                request, response = json.loads(body), json.loads(data)
            except ValueError:
                request, response = body.decode(errors="replace"), data.decode(errors="replace")
            record({"t": t0, "elapsed_s": round(time.time() - t0, 3), "status": status,
                    "sampling_override": sampling, "request": request, "response": response})
        self._reply(status, headers, data)

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", args.listen), Proxy).serve_forever()
