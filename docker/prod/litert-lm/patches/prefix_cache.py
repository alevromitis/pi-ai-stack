"""Prefix (KV) cache for `litert-lm serve`'s chat completions endpoint.

Not part of upstream LiteRT-LM: copied into litert_lm_cli/commands/serve/ by
the Dockerfile and wired in by apply_prefix_cache.py.

Upstream builds a fresh Conversation for every request, so the system
prompt, tool schemas and the whole history are prefilled again on every
turn - and on every tool round of an agent turn. Here the last Conversation
stays alive. When a request's messages are that conversation's history plus
the reply it generated, followed only by new user/tool messages, just those
new messages are sent and only they are prefilled.

Anything else (edited or regenerated messages, other chats, title requests,
changed tools or sampler settings) is a miss: the cached conversation is
closed first, so the engine never holds two sessions, and the request is
served exactly like upstream.

Env:
  LITERT_PREFIX_CACHE=0   disable (pure upstream behaviour)
"""

from __future__ import annotations

import collections.abc
import contextlib
import hashlib
import json
import os
from typing import Any

import click

import litert_lm
from litert_lm import conversation as _conversation_module

_ENABLED = os.environ.get("LITERT_PREFIX_CACHE", "1") != "0"
_REUSABLE_ROLES = ("user", "tool")

_original_normalize_message = _conversation_module.normalize_message


def _normalize_message(message: Any) -> Any:
  # Several tool results arrive together after parallel tool calls. The C API
  # takes a JSON list of messages (upstream's own automatic tool calling sends
  # one), but normalize_message rejects lists.
  if isinstance(message, list):
    return message
  return _original_normalize_message(message)


_conversation_module.normalize_message = _normalize_message


def _log(text: str) -> None:
  click.echo(click.style(f"prefix-cache: {text}", fg="magenta"))


def _digest(data: str) -> str:
  return hashlib.sha1(data.encode("utf-8")).hexdigest()[:16]


def _canonical_args(args: Any) -> str:
  if isinstance(args, str):
    try:
      args = json.loads(args)
    except json.JSONDecodeError:
      return args
  return json.dumps(args, sort_keys=True, ensure_ascii=False)


def _canonical_content(content: Any) -> Any:
  """Text is compared verbatim (trimmed); media by digest."""
  if content is None:
    return ""
  if isinstance(content, str):
    return content.strip()
  if not isinstance(content, list):
    return json.dumps(content, sort_keys=True)
  parts = []
  for part in content:
    if not isinstance(part, dict):
      parts.append(str(part))
    elif part.get("type") == "text":
      parts.append(part.get("text", ""))
    elif part.get("type") == "image_url":
      parts.append("img:" + _digest(str(part.get("image_url", {}).get("url"))))
    elif part.get("type") == "input_audio":
      parts.append("aud:" + _digest(str(part.get("input_audio", {}).get("data"))))
    else:
      parts.append(json.dumps(part, sort_keys=True))
  return "".join(parts).strip()


def _canonical_message(
    msg: collections.abc.Mapping[str, Any],
    name_by_tool_call_id: collections.abc.Mapping[str, str],
) -> tuple[Any, ...]:
  """Comparable form of an OpenAI message.

  Tool call ids are dropped (clients may re-id calls); tool results are keyed
  by the called function's name instead.
  """
  role = msg.get("role")
  if role == "tool":
    name = name_by_tool_call_id.get(msg.get("tool_call_id", ""), "")
    return ("tool", name, _canonical_content(msg.get("content")))
  calls = tuple(
      (
          (tc.get("function") or {}).get("name"),
          _canonical_args((tc.get("function") or {}).get("arguments", {})),
      )
      for tc in (msg.get("tool_calls") or [])
  )
  return (role, _canonical_content(msg.get("content")), calls)


def _config_key(body: collections.abc.Mapping[str, Any]) -> str:
  """Everything fixed at conversation creation must match to reuse it."""
  keys = (
      "model", "tools", "temperature", "top_p", "top_k", "seed",
      "reasoning_effort", "response_format",
  )
  return json.dumps({k: body.get(k) for k in keys}, sort_keys=True)


class _RecordingConversation:
  """Proxies a Conversation and records the reply it generates."""

  def __init__(self, conv: litert_lm.Conversation):
    self._conv = conv
    self.text_parts: list[str] = []
    self.tool_calls: list[litert_lm.ToolCall] = []
    self.completed = False
    self.cancelled = False

  def __getattr__(self, name: str) -> Any:
    return getattr(self._conv, name)

  def cancel_process(self) -> None:
    self.cancelled = True
    self._conv.cancel_process()

  def send_message_async(self, *args, **kwargs):
    # Chunks are litert_lm.Message; str() is its text, as upstream sends it.
    for chunk in self._conv.send_message_async(*args, **kwargs):
      self.text_parts.append(str(chunk))
      self.tool_calls.extend(chunk.tool_calls)
      yield chunk
    self.completed = True

  def reply(self) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "".join(self.text_parts),
        "tool_calls": [
            {"function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in self.tool_calls
        ],
    }


class _Entry:

  def __init__(self, engine, conv, key: str, history: list[tuple[Any, ...]]):
    self.engine = engine
    self.conv = conv
    self.key = key
    self.history = history


_entry: _Entry | None = None


def _drop() -> None:
  global _entry
  if _entry is not None:
    _entry.conv.close()
    _entry = None


def _match(
    engine,
    key: str,
    canonical: list[tuple[Any, ...]],
    messages: list[Any],
) -> int | None:
  """Returns how many leading messages the cached conversation holds."""
  e = _entry
  if e is None:
    return None
  if e.engine is not engine:
    _log("miss: engine was re-initialized")
    return None
  if e.key != key:
    _log("miss: tools/model/sampler settings changed")
    return None
  n = len(e.history)
  if len(canonical) <= n or canonical[:n] != e.history:
    diverge = next(
        (i for i, (a, b) in enumerate(zip(canonical, e.history)) if a != b),
        min(len(canonical), n),
    )
    _log(f"miss: history differs at message {diverge} (cached {n})")
    return None
  new_roles = {m.get("role") for m in messages[n:]}
  if not new_roles <= set(_REUSABLE_ROLES) or (
      "user" in new_roles and len(messages) - n > 1
  ):
    _log(f"miss: new messages have roles {sorted(new_roles)}")
    return None
  return n


@contextlib.contextmanager
def conversation(
    engine,
    *,
    body: collections.abc.Mapping[str, Any],
    translated_messages: list[Any],
    prompt: Any,
    name_by_tool_call_id: collections.abc.Mapping[str, str] | None,
    **create_kwargs,
):
  """Yields (conversation, prompt to send) for a chat completions request.

  Drop-in for `engine.create_conversation(messages=translated[:-1], ...)`.
  """
  global _entry
  messages = body.get("messages")
  if not _ENABLED or not isinstance(messages, list) or not messages:
    with engine.create_conversation(
        messages=translated_messages[:-1] if translated_messages else [],
        **create_kwargs,
    ) as conv:
      yield conv, prompt
    return

  name_map = name_by_tool_call_id or {}
  key = _config_key(body)
  canonical = [_canonical_message(m, name_map) for m in messages]

  n = _match(engine, key, canonical, messages)
  if n is not None:
    conv = _entry.conv
    new = translated_messages[n:]
    send = new[0] if len(new) == 1 else new
    _log(f"hit: reusing {n} messages, prefilling {len(new)} new")
  else:
    _drop()
    conv = engine.create_conversation(
        messages=translated_messages[:-1], **create_kwargs
    )
    send = prompt
  _entry = None  # Owned by this request until it completes cleanly.

  recorder = _RecordingConversation(conv)
  try:
    yield recorder, send
  except BaseException:
    conv.close()
    raise

  # _stream_response swallows errors, so only a generator that ran to its
  # end without cancellation is known to have a KV cache matching `history`.
  # A reply cut off at the token limit also ends "normally"; the next send on
  # it fails and the entry is dropped then.
  if recorder.completed and not recorder.cancelled:
    reply = _canonical_message(recorder.reply(), {})
    _entry = _Entry(engine, conv, key, canonical + [reply])
  else:
    conv.close()
