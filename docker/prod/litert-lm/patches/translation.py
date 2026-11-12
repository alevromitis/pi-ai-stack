"""Translates chats in other languages to English and the reply back.

Not part of upstream LiteRT-LM: copied into litert_lm_cli/commands/serve/ by
the Dockerfile and wired in by apply_translation.py.

gemma4-e2b reasons, searches and cites best in English, and Wikipedia is
English-only. With mcp-translate offered as a tool (2026-10-13) the model
had to decide to call it on the way in and again on the way out, and it
often didn't. So the server does it for every chat, tools or not:

1. Each user message is sent to mcp-translate's /api/translate; one
   detected as another language (confidence >= LITERT_TRANSLATE_MIN_CONF)
   is replaced by its English translation. Code, links and URLs stay as
   they are. Translations are cached by text, so a chat's history is
   translated once, and the same English every time, which the prefix
   cache's history comparison needs. Assistant turns are left as the
   client sends them: the prefix cache matches them against what was
   streamed (prefix_cache.recording), i.e. the translated reply.
2. When the newest user message was translated, the reply is held back
   and, if it is in English, translated into that message's language
   when it ends (the thoughts still stream). A reply already in another
   language ("translate this into French") passes as it is. Code blocks,
   links and the Sources line (citations.py, which works on the English
   reply first) stay as they are.

Runs first on a request, before citations.collect, so the search-result
matching, the memory gate and the reminder all see English.

If mcp-translate can't be reached, messages and replies pass untouched.

Env:
  LITERT_TRANSLATE=0              disable
  LITERT_TRANSLATE_URL            mcp-translate's /api/translate
  LITERT_TRANSLATE_MIN_CONF=0.8   detection confidence needed to translate
                                  a message ("ok", "hi" score ~0.03)
"""

from __future__ import annotations

import collections
import json
import os
import urllib.request
from typing import Any

import click

_ENABLED = os.environ.get("LITERT_TRANSLATE", "1") != "0"
_URL = os.environ.get("LITERT_TRANSLATE_URL", "http://mcp-translate-1:8000/api/translate")
_MIN_CONF = float(os.environ.get("LITERT_TRANSLATE_MIN_CONF", "0.8"))
_NATIVE_INSTAGRAM = "[instagram-personal:native]"
# A long reply takes ~1 s a sentence on the Pi.
_TIMEOUT_S = 120
_CACHE_SIZE = 512

# User message text -> (its English, or None if it stays; its language).
_cache: collections.OrderedDict[str, tuple[str | None, str]] = collections.OrderedDict()


def _log(text: str) -> None:
  click.echo(click.style(f"translation: {text}", fg="cyan"))


def _call(**request: Any) -> dict[str, Any] | None:
  req = urllib.request.Request(
      _URL, data=json.dumps(request).encode(),
      headers={"Content-Type": "application/json"},
  )
  try:
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
      return json.load(r)
  except Exception as e:  # pylint: disable=broad-exception-caught
    _log(f"mcp-translate failed, passing through: {e}")
    return None


def _to_english(text: str) -> tuple[str | None, str]:
  """(English, language) for a message in another language, else (None, "")."""
  if text in _cache:
    _cache.move_to_end(text)
    return _cache[text]
  r = _call(text=text, target="en", min_confidence=_MIN_CONF)
  if r is None:
    return None, ""  # not cached: retried on the next request
  found = (r["text"], r["source"]) if r.get("translated") else (None, "")
  _cache[text] = found
  while len(_cache) > _CACHE_SIZE:
    _cache.popitem(last=False)
  return found


def _native_instagram(messages: list[Any]) -> bool:
  # Only trusted system/developer instructions can select this route. A user
  # message, tool result or retrieved page containing the marker cannot.
  for message in messages:
    if not isinstance(message, dict) or message.get("role") not in ("system", "developer"):
      continue
    content = message.get("content")
    if isinstance(content, list):
      content = "".join(part.get("text", "") for part in content
                        if isinstance(part, dict) and part.get("type") == "text")
    if isinstance(content, str) and _NATIVE_INSTAGRAM in content:
      return True
  return False


def _bound_instagram_search(messages: list[Any]) -> None:
  # Search result prose is expensive to prefill on the Pi. Keep a bounded
  # excerpt; leave calculator results, user text and other chats untouched.
  names = {"searxng_web_search_mcp_searxng"}
  call_ids = {call.get("id") for message in messages if isinstance(message, dict)
              for call in (message.get("tool_calls") or [])
              if isinstance(call, dict) and call.get("id") is not None
              and (call.get("function") or {}).get("name") in names}
  for message in messages:
    if not isinstance(message, dict) or message.get("role") != "tool":
      continue
    if message.get("name") not in names and message.get("tool_call_id") not in call_ids:
      continue
    content = message.get("content")
    if isinstance(content, list) and all(isinstance(part, dict) and part.get("type") == "text"
                                        and isinstance(part.get("text"), str) for part in content):
      content = "\n".join(part["text"] for part in content)
    if not isinstance(content, str) or len(content.encode("utf-8")) <= 1200:
      continue
    excerpt = content.encode("utf-8")[:1200].decode("utf-8", "ignore")
    # Avoid a half URL/line when the result contains line breaks.
    if "\n" in excerpt:
      excerpt = excerpt.rsplit("\n", 1)[0]
    message["content"] = excerpt + "\n[search output shortened]"
    _log("Instagram search result bounded to 1200 bytes")


def apply(handler: Any, body: dict[str, Any]) -> None:
  """Translates the user messages in place; remembers the reply's language."""
  handler.translate_back = None
  messages = body.get("messages")
  if not _ENABLED or not isinstance(messages, list):
    return
  if _native_instagram(messages):
    _bound_instagram_search(messages)
    _log("Instagram native-language route: translation bypassed")
    return
  language = ""
  for msg in messages:
    if not isinstance(msg, dict) or msg.get("role") != "user":
      continue
    language = ""
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
      english, language = _to_english(content)
      if english is not None:
        msg["content"] = english
    elif isinstance(content, list):
      for part in content:
        if isinstance(part, dict) and part.get("type") == "text" and str(part.get("text") or "").strip():
          english, lang = _to_english(part["text"])
          if english is not None:
            part["text"], language = english, lang
  if language:
    _log(f"newest message in {language}, sent in English")
    handler.translate_back = language


class Back:
  """Holds the reply and returns it in the user's language at the end."""

  def __init__(self, language: str):
    self._language = language
    self._parts: list[str] = []

  def feed(self, text: str) -> str:
    self._parts.append(text or "")
    return ""

  def finish(self) -> str:
    text = "".join(self._parts)
    if not text.strip():
      return text
    r = _call(text=text, target=self._language, only_from="en")
    if r is None:
      return text
    if r.get("translated"):
      _log(f"reply translated into {self._language} ({len(text)} chars)")
    return r["text"] if r.get("translated") else text


def stream_filter(handler: Any) -> Back | None:
  """The Back for this request, or None (one use)."""
  language = handler.__dict__.pop("translate_back", None)
  return Back(language) if language else None


def back(handler: Any, text: str | None) -> str | None:
  """Non-streamed replies."""
  filt = stream_filter(handler)
  if filt is None or not text:
    return text
  filt.feed(text)
  return filt.finish()
