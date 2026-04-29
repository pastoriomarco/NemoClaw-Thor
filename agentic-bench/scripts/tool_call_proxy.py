"""
OpenAI-compatible tool-call parsing proxy for TensorRT-Edge-LLM (Nemotron Omni).

Why this exists:
  TRT-Edge-LLM v0.7.0's experimental/server/api_server.py has zero tool-call
  parsing — it returns the model's raw text in the `content` field. Nemotron
  Omni, however, emits tool calls as JSON-in-content like:
      {"tool": "get_weather", "arguments": {"city": "Tokyo"}}
  Standard tool-calling clients (BFCL "FC" handlers, OpenAI SDK with tools=,
  vLLM benches) expect the OpenAI structured `tool_calls` field. This proxy
  bridges the gap without touching TRT-Edge-LLM upstream.

Usage:
  TRT_BASE_URL=http://127.0.0.1:8000 PROXY_PORT=8001 \
      python tool_call_proxy.py

  Then point clients at http://127.0.0.1:8001/v1/chat/completions instead.

Design:
  - Forwards all paths transparently
  - For /v1/chat/completions: parses Omni's text response, hoists matched
    tool-calls into structured `tool_calls`, sets `finish_reason='tool_calls'`,
    leaves content as null per OpenAI spec
  - Streaming is passed through unchanged for now (tool_calls in streamed
    responses requires re-tokenization that we don't need for our benches)
  - Logs every request to a JSONL file at PROXY_LOG_DIR for diagnosis
"""
import ast
import os
import re
import json
import uuid
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

TRT_BASE_URL = os.environ.get("TRT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8001"))
PROXY_LOG_DIR = os.environ.get("PROXY_LOG_DIR", "/home/tndlux/agentic-bench/logs")
PROXY_LOG_PATH = os.path.join(PROXY_LOG_DIR, "tool_call_proxy.jsonl")

os.makedirs(PROXY_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("tool_call_proxy")

app = FastAPI(title="TRT-Edge-LLM Tool-Call Proxy", version="0.1.0")
client = httpx.AsyncClient(base_url=TRT_BASE_URL, timeout=httpx.Timeout(600.0))


# Tool-call formats observed across model families:
#
#   A. Nemotron-3-Nano native (the format we get from Omni when tools= is set
#      properly in the request — the chat template wraps in <TOOLCALL> tags):
#         <TOOLCALL>[get_weather(city="Rome")]</TOOLCALL>
#         <TOOLCALL>[fn1(a=1), fn2(b="x")]</TOOLCALL>     # parallel calls
#      Pythonic function-call syntax. Args use Python literal syntax
#      (strings, numbers, lists, dicts).
#
#   B. JSON-in-content (older Omni outputs without proper tools= or some
#      Qwen-distill outputs):
#         {"tool": "name", "arguments": { ... }}
#         {"name": "name", "arguments": { ... }}
#
#   C. Code-fenced JSON:
#         ```json\n{...}\n```
#
#   D. XML-tagged JSON (some Qwen variants):
#         <tool_call>{...}</tool_call>
#
# Try in order: A is the canonical Omni format when tools= is set (most of
# our BFCL traffic), B-D are fallbacks for prose-style outputs.

_TOOLCALL_TAG_RE = re.compile(r"<TOOLCALL>\s*\[(.*?)\]\s*</TOOLCALL>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(\{.*?\})\s*```", re.DOTALL)
_XML_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_BARE_JSON_RE = re.compile(
    r'\{\s*"(?:tool|name)"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}',
    re.DOTALL,
)


def _parse_pythonic_calls(body: str) -> List[Dict[str, Any]]:
    """Parse `fn1(a=1), fn2(b="x")` (the inside of a <TOOLCALL>[...]</TOOLCALL>)
    into a list of {name, arguments-dict} entries using ast.parse.

    Returns [] on any parse failure (malformed Python). Caller decides whether
    to fall back to other patterns.
    """
    # Wrap in a fake list literal so ast can parse multiple calls together.
    wrapped = f"[{body}]"
    try:
        tree = ast.parse(wrapped, mode="eval")
    except SyntaxError:
        return []
    if not isinstance(tree.body, ast.List):
        return []
    calls = []
    for node in tree.body.elts:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        args: Dict[str, Any] = {}
        # Only kwargs are valid in OpenAI tool-calling (positional args don't map).
        for kw in node.keywords:
            try:
                value = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                # Fall back to the raw source for unsupported expressions.
                value = ast.unparse(kw.value)
            if kw.arg is not None:
                args[kw.arg] = value
        calls.append({"name": name, "arguments": args})
    return calls


def _normalize_call(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert {tool/name, arguments} → OpenAI structured tool_call dict."""
    name = obj.get("tool") or obj.get("name")
    arguments = obj.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, (dict, str)):
        return None
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, separators=(",", ":"))
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _extract_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Return list of OpenAI-format tool_calls extracted from content, or []."""
    tool_calls: List[Dict[str, Any]] = []

    # Format A: <TOOLCALL>[fn(a=1), fn2(b="x")]</TOOLCALL>  (Nemotron native)
    for m in _TOOLCALL_TAG_RE.finditer(content):
        for parsed in _parse_pythonic_calls(m.group(1)):
            normalized = _normalize_call(parsed)
            if normalized is not None:
                tool_calls.append(normalized)
    if tool_calls:
        return tool_calls

    # Formats B/C/D: JSON in various wrappings
    candidates: List[str] = []
    candidates.extend(m.group(1) for m in _FENCE_RE.finditer(content))
    candidates.extend(m.group(1) for m in _XML_TAG_RE.finditer(content))
    if not candidates:
        candidates.extend(m.group(0) for m in _BARE_JSON_RE.finditer(content))

    for raw in candidates:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                normalized = _normalize_call(obj)
                if normalized is not None:
                    tool_calls.append(normalized)
        except json.JSONDecodeError:
            continue
    return tool_calls


def _log_event(event: Dict[str, Any]):
    try:
        with open(PROXY_LOG_PATH, "a") as fh:
            fh.write(json.dumps(event) + "\n")
    except OSError as e:
        log.warning("log write failed: %s", e)


@app.get("/v1/models")
async def models():
    r = await client.get("/v1/models")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/health")
async def health():
    r = await client.get("/health")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    is_streaming = bool(body.get("stream"))

    # Streaming pass-through for now — bench traffic doesn't need streamed tool_calls.
    if is_streaming:
        async def streamer():
            async with client.stream("POST", "/v1/chat/completions", json=body) as r:
                async for chunk in r.aiter_raw():
                    yield chunk
        return StreamingResponse(streamer(), media_type="text/event-stream")

    r = await client.post("/v1/chat/completions", json=body)
    if r.status_code != 200:
        return JSONResponse(r.json(), status_code=r.status_code)

    response = r.json()
    has_tools_in_request = bool(body.get("tools") or body.get("functions"))

    # Only attempt tool-call extraction when caller actually requested tools.
    # Otherwise leave the response untouched (lm-eval IFEval/GSM8K traffic
    # doesn't request tools — we don't want to mangle prose responses).
    if has_tools_in_request:
        for choice in response.get("choices", []):
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            tool_calls = _extract_tool_calls(content)
            if tool_calls:
                msg["content"] = None
                msg["tool_calls"] = tool_calls
                choice["finish_reason"] = "tool_calls"
                _log_event({
                    "event": "tool_call_extracted",
                    "n_calls": len(tool_calls),
                    "raw_content_len": len(content),
                })

    return JSONResponse(response)


# Optional: completions endpoint passthrough so lm-eval text-completion mode works
@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    r = await client.post("/v1/completions", json=body)
    return JSONResponse(r.json(), status_code=r.status_code)


if __name__ == "__main__":
    import uvicorn
    log.info(f"Starting tool-call proxy on :{PROXY_PORT} -> {TRT_BASE_URL}")
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="info")
