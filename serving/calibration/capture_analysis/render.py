"""Rebuild the token sequences of captured chat/completions traffic.

Prompt: vLLM 0.33's own rendering path, run offline on CPU: the request body is
validated by ``ChatCompletionRequest`` (renames ``reasoning_content`` to
``reasoning``, maps ``reasoning_effort`` to ``enable_thinking``), its
``build_chat_params`` gives the template kwargs, ``parse_chat_messages``
normalises the messages (content parts, tool-call arguments to dicts), tools
are ``model_dump()``-ed like the server does, and ``safe_apply_chat_template``
renders and tokenises with the model's tokenizer and chat template. On the
Cline test dumps the result equals the server's ``usage.prompt_tokens`` for
every request.

Reply: the SSE deltas are concatenated per field and re-assembled into the text
the model generated after the generation prompt (``...<think>\\n``):
``reasoning + "</think>" + content + tool calls + "<|im_end|>"``, tool calls in
the template's XML format (``<tool_call>\\n<function=NAME>\\n<parameter=K>\\nV\\n
</parameter>\\n</function>\\n</tool_call>``). That text is re-tokenised in one
piece; this is an approximation of the sampled ids (a different but equivalent
segmentation is possible, argument JSON is re-serialised). Each token is
labelled by scanning the special tokens: up to and including ``</think>`` =
reasoning, ``<tool_call>``..``</tool_call>`` = tool call, the final
``<|im_end|>`` = end marker, everything else = content.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import types

import numpy as np

LABEL_REASONING, LABEL_CONTENT, LABEL_TOOL, LABEL_END = 0, 1, 2, 3
LABEL_NAMES = ["reasoning", "content", "tool_call", "end_marker"]
FAMILY_RE = re.compile(r"\[family:([A-Za-z0-9_.\-]+)\]")


def message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def conversation_key(body: dict) -> str:
    """Same tools + system prompt + first user message = same task/conversation."""
    messages = body.get("messages") or []
    system = [m.get("content") for m in messages[:1] if m.get("role") in ("system", "developer")]
    first_user = next((message_text(m.get("content")) for m in messages if m.get("role") == "user"), "")
    blob = json.dumps([body.get("tools"), system, first_user], sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def family_of(body: dict) -> str | None:
    for message in body.get("messages") or []:
        if message.get("role") == "user":
            match = FAMILY_RE.search(message_text(message.get("content")))
            return match.group(1) if match else None
    return None


def parse_sse(path: str) -> dict | None:
    """Concatenate the streamed deltas of choice 0. None if the file is empty."""
    reasoning, content, calls = [], [], {}
    usage = finish = None
    seen = False
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            seen = True
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                if choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta") or {}
                text = delta.get("reasoning") or delta.get("reasoning_content")
                if text:
                    reasoning.append(text)
                if delta.get("content"):
                    content.append(delta["content"])
                for call in delta.get("tool_calls") or []:
                    entry = calls.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                    function = call.get("function") or {}
                    if function.get("name"):
                        entry["name"] += function["name"]
                    if function.get("arguments"):
                        entry["arguments"] += function["arguments"]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    if not seen:
        return None
    return {
        "reasoning": "".join(reasoning),
        "content": "".join(content),
        "tool_calls": [calls[i] for i in sorted(calls)],
        "finish_reason": finish,
        "usage": usage,
    }


def tool_call_text(call: dict) -> str:
    text = "<tool_call>\n<function=" + call["name"] + ">\n"
    try:
        arguments = json.loads(call["arguments"]) if call["arguments"] else {}
    except json.JSONDecodeError:
        arguments = None
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            text += "<parameter=" + key + ">\n" + value + "\n</parameter>\n"
    elif call["arguments"]:  # unparsable: keep the raw text as one parameter body
        text += call["arguments"] + "\n"
    return text + "</function>\n</tool_call>"


class Renderer:
    def __init__(self, model_dir: str):
        from transformers import AutoTokenizer
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.renderers.hf import _detect_content_format, parse_chat_messages, safe_apply_chat_template

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.template = self.tokenizer.chat_template
        self.content_format = _detect_content_format(self.template, default="string")
        self._request_cls = ChatCompletionRequest
        self._parse = parse_chat_messages
        self._apply = safe_apply_chat_template
        # Text-only stand-in for ModelConfig: only these attributes are read.
        self._model_config = types.SimpleNamespace(
            hf_config=types.SimpleNamespace(), is_multimodal_model=False,
            multimodal_config=None, enable_prompt_embeds=False,
        )
        vocab = self.tokenizer.get_vocab()
        self.ids = {name: vocab[name] for name in ("</think>", "<tool_call>", "</tool_call>", "<|im_end|>")}

        self.boundary = "<|im_start|>"
        self.boundary_id = vocab[self.boundary]

    def _render(self, body: dict, tokenize: bool):
        request = self._request_cls.model_validate(body)
        params = request.build_chat_params(None, "auto")
        kwargs = params.get_apply_chat_template_kwargs()
        kwargs.pop("chat_template", None)
        # Server default: exclude_tools_when_tool_choice_none is off.
        tools = [tool.model_dump() for tool in request.tools] if request.tools is not None else None
        conversation, _, _ = self._parse(request.messages, self._model_config, content_format=self.content_format)
        return self._apply(self._model_config, self.tokenizer, conversation, tools=tools,
                           chat_template=self.template, tokenize=tokenize, **kwargs)

    def prompt_ids(self, body: dict) -> np.ndarray:
        """Exactly the server's prompt token ids (full tokenisation, slow for long prompts)."""
        return np.asarray(self._render(body, True), dtype=np.int32)

    def prompt_text(self, body: dict) -> str:
        return self._render(body, False)

    def encode_incremental(self, text: str, ref: dict | None) -> tuple[np.ndarray, dict]:
        """Tokenise ``text`` reusing the tokens of a previous prompt ``ref``.

        Added tokens are split out before BPE, so tokenising ``text[:c]`` and
        ``text[c:]`` separately is exact when ``text[c:]`` starts with the
        added token ``<|im_start|>``. The shared prefix up to the last such
        boundary inside the common character prefix is taken from ``ref``.
        """
        start_char, start_tok, prefix = 0, 0, np.empty(0, dtype=np.int32)
        if ref is not None:
            common = common_prefix_len(text, ref["text"])
            usable = np.flatnonzero(ref["bchars"] + len(self.boundary) <= common)
            if usable.size:
                i = int(usable[-1])
                start_char, start_tok = int(ref["bchars"][i]), int(ref["btoks"][i])
                prefix = ref["ids"][:start_tok]
        new = np.asarray(self.tokenizer.encode(text[start_char:], add_special_tokens=False), dtype=np.int32)
        ids = np.concatenate([prefix, new])
        chars = [start_char + m.start() for m in re.finditer(re.escape(self.boundary), text[start_char:])]
        toks = (np.flatnonzero(new == self.boundary_id) + start_tok).tolist()
        if len(chars) != len(toks):  # a boundary string not tokenised as the added token: no reuse later
            chars, toks = [], []
        bchars = np.asarray((ref["bchars"][: np.searchsorted(ref["bchars"], start_char)].tolist() if start_tok else []) + chars,
                            dtype=np.int64)
        btoks = np.asarray((ref["btoks"][: bchars.size - len(chars)].tolist() if start_tok else []) + toks, dtype=np.int64)
        return ids, {"text": text, "ids": ids, "bchars": bchars, "btoks": btoks}

    def thinking_open(self, prompt: np.ndarray) -> bool:
        """True if the generation prompt leaves <think> open (thinking enabled)."""
        return not bool((prompt[-3:] == self.ids["</think>"]).any())

    def reply_text(self, reply: dict, thinking: bool = True) -> str:
        # A reply cut by max_tokens inside the reasoning never generated </think>.
        closed = bool(reply["content"] or reply["tool_calls"] or reply["finish_reason"] != "length")
        text = (reply["reasoning"] + ("</think>" if closed else "") if thinking else "") + reply["content"]
        if reply["tool_calls"]:
            content = reply["content"]
            if not content.strip():
                sep = "" if content.endswith("\n\n") else "\n\n"
            else:
                sep = "" if content.endswith("\n\n") else ("\n" if content.endswith("\n") else "\n\n")
            text += sep + "\n".join(tool_call_text(call) for call in reply["tool_calls"])
        if reply["finish_reason"] in ("stop", "tool_calls"):
            text += "<|im_end|>"
        return text

    def reply_ids(self, reply: dict, thinking: bool = True) -> np.ndarray:
        text = self.reply_text(reply, thinking)
        return np.asarray(self.tokenizer.encode(text, add_special_tokens=False), dtype=np.int32)

    def label(self, ids: np.ndarray, reasoning_open: bool = True) -> np.ndarray:
        """Per-token label of a generated sequence (see module docstring)."""
        labels = np.full(ids.shape[0], LABEL_CONTENT, dtype=np.uint8)
        think_end = self.ids["</think>"]
        hits = np.flatnonzero(ids == think_end)
        if reasoning_open and hits.size:
            labels[: hits[0] + 1] = LABEL_REASONING
        starts = np.flatnonzero(ids == self.ids["<tool_call>"])
        ends = np.flatnonzero(ids == self.ids["</tool_call>"])
        for start in starts:
            after = ends[ends >= start]
            labels[start : (after[0] + 1 if after.size else ids.shape[0])] = LABEL_TOOL
        if ids.shape[0] and ids[-1] == self.ids["<|im_end|>"]:
            labels[-1] = LABEL_END
        return labels


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n  # a[:lo] == b[:lo], a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def request_files(dump_dir: str) -> list[str]:
    names = [n for n in os.listdir(dump_dir) if n.endswith(".request.json")]
    return [os.path.join(dump_dir, n) for n in sorted(names)]
