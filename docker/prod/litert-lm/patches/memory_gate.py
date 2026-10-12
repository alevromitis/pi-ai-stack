"""Answers LibreChat's memory agent without the engine on ordinary turns.

Not part of upstream LiteRT-LM: copied into litert_lm_cli/commands/serve/ by
the Dockerfile and wired in by apply_memory_gate.py.

With `memory.agent.enabled` in librechat.yaml, LibreChat sends a second
request with every chat message: the last few messages plus two tools,
set_memory and delete_memory, and instructions to call them only when the
user asks to remember or forget something. `serve` is single-threaded, so
that request queues next to the reply, and it takes one of the prefix
cache's few conversation slots (prefix_cache.py; until 2026-10-12 the only
one, so every reply re-prefilled the whole chat).

When a request's tools are exactly those two and the newest user turn names
no memory action, this answers "no tool call" at once, without touching the
engine or the cache. Only turns like "remember that I..." reach the model.

Env:
  LITERT_MEMORY_GATE=0   disable (every memory request reaches the model)
"""

from __future__ import annotations

import datetime
import os
import re
from typing import Any

import click

from litert_lm_cli.commands.serve import openai_common

_ENABLED = os.environ.get("LITERT_MEMORY_GATE", "1") != "0"
_MEMORY_TOOLS = frozenset({"set_memory", "delete_memory"})

# The phrases LibreChat's default memory instructions list ("remember",
# "don't forget", "store this", "forget", "delete the memory"), plus "keep in
# mind" and memorise/memory, in English and Greek (θυμ-, ξεχ-, μνήμ-). A false
# match only costs one model call; a missed one means nothing is saved.
_MEMORY_WORDS = re.compile(
    r"\bremember|\bforget|\bforgot|\bmemor[iy]|\bkeep in mind\b"
    r"|\bstore (?:this|that)\b"
    r"|θυμ|ξ[εέ]χ|μν[ηή]μ",
    re.IGNORECASE,
)


def _log(text: str) -> None:
  click.echo(click.style(f"memory-gate: {text}", fg="magenta"))


def _text(content: Any) -> str:
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    return "".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )
  return ""


def _latest_user_turn(body: dict[str, Any]) -> str | None:
  """The newest user message inside LibreChat's "# Current Chat:" buffer.

  The memory agent gets the chat as one user message, rendered by
  LangChain's getBufferString ("Human: ...\nAI: ..."), before the reply to
  the newest message exists, so the text after the last "Human: " is it.
  """
  messages = body.get("messages")
  if not isinstance(messages, list):
    return None
  for msg in reversed(messages):
    if isinstance(msg, dict) and msg.get("role") == "user":
      return _text(msg.get("content")).rsplit("Human: ", 1)[-1]
  return None


def _is_memory_request(body: dict[str, Any]) -> bool:
  tools = body.get("tools")
  if not isinstance(tools, list):
    return False
  names = {
      (t.get("function") or {}).get("name")
      for t in tools
      if isinstance(t, dict)
  }
  return names == _MEMORY_TOOLS


def answer(handler: Any, body: dict[str, Any]) -> bool:
  """Replies "no tool call" to a memory request with no memory action.

  Returns True if it replied (the caller must stop), False to serve the
  request normally.
  """
  if not _ENABLED or not _is_memory_request(body):
    return False
  turn = _latest_user_turn(body)
  if turn is None:
    return False
  match = _MEMORY_WORDS.search(turn)
  if match:
    _log(f"pass: {match.group(0)!r} in the newest user turn")
    return False
  _log("skip: no memory action in the newest user turn")

  now = datetime.datetime.now(datetime.timezone.utc)
  completion_id = f"chatcmpl_{now.strftime('%Y%m%d%H%M%S%f')}"
  created = int(now.timestamp())
  model = body.get("model")
  handler.headers_sent = True
  handler.send_response(200)
  if body.get("stream"):
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    for choice in (
        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None},
        {"index": 0, "delta": {}, "finish_reason": "stop"},
    ):
      handler.wfile.write(
          openai_common.sse_data(
              openai_common.dump_json({**chunk, "choices": [choice]})
          )
      )
    handler.wfile.write(openai_common.format_sse_final())
  else:
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(
        (
            openai_common.dump_json({
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            })
            + "\n"
        ).encode("utf-8")
    )
  handler.wfile.flush()
  return True
