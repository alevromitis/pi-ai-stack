"""Answers LibreChat's chat-title request without the engine.

Not part of upstream LiteRT-LM: copied into litert_lm_cli/commands/serve/ by
the Dockerfile and wired in by apply_title_gate.py.

With `titleConvo: true`, LibreChat sends one more request after the first
reply of every chat: a single user message asking for a 5-word title of
"User: <first message>\nAI: <reply>". The prefix cache (prefix_cache.py)
holds one conversation, and that request has other settings (no tools), so
it evicted the chat: the second message of every chat then re-prefilled
the system prompt and the whole first exchange (about 20-35 s on the Pi).

This answers it at once with a title cut from the user's first message,
without touching the engine or the cache, so the second message is a
prefix-cache hit. Only LibreChat's default completion title prompt is
recognised (`titlePrompt` unset in librechat.yaml); anything else is served
normally.

Env:
  LITERT_TITLE_GATE=0   disable (the model writes titles, evicting the cache)
"""

from __future__ import annotations

import datetime
import os
import re
from typing import Any

import click

from litert_lm_cli.commands.serve import openai_common

_ENABLED = os.environ.get("LITERT_TITLE_GATE", "1") != "0"

# @librechat/agents' defaultCompletionPrompt (src/utils/title.ts), filled
# with the default titlePromptTemplate "User: {input}\nAI: {output}".
_PROMPT = (
    "Provide a concise, 5-word-or-less title for the conversation, using"
    " title case conventions. Only return the title itself.\n\n"
    "Conversation:\nUser: "
)
_MAX_WORDS = 6
_FALLBACK = "New Chat"

# Openers that say nothing about the topic, dropped from the front (several
# may stack: "hi, can you tell me about ..."). A title is cosmetic, so the
# list stays short; an unlisted opener only makes a clumsier title.
_FILLER = re.compile(
    r"^(?:please|hey|hi|hello|thanks|ok(?:ay)?|so|"
    r"can you|could you|would you|will you|"
    r"i (?:want|need|would like) to|i['’]d like to|"
    r"help me(?: to)?|tell me(?: about)?|give me|show me|explain|"
    r"write|create|make|generate|draft|compose|"
    r"what(?: is|['’]s| are| was| were)|who(?: is|['’]s| was| were)|"
    r"do you know)\b[\s,:]*",
    re.IGNORECASE,
)
_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_SMALL_WORDS = frozenset(
    "a an the and or nor but of in on at to for with by from as vs via".split()
)
# Not left dangling at the end of a title cut at _MAX_WORDS.
_TRAILING = _SMALL_WORDS | frozenset(
    "that which who about is are was were this these those its their my your"
    " our".split()
)


def _log(text: str) -> None:
  click.echo(click.style(f"title-gate: {text}", fg="magenta"))


def _first_message(body: dict[str, Any]) -> str | None:
  """The user's first message, if this is LibreChat's title request."""
  if body.get("tools") or body.get("response_format"):
    return None
  messages = body.get("messages")
  if not isinstance(messages, list) or len(messages) != 1:
    return None
  msg = messages[0]
  if not isinstance(msg, dict) or msg.get("role") != "user":
    return None
  content = msg.get("content")
  if isinstance(content, list):
    content = "".join(
        p.get("text", "")
        for p in content
        if isinstance(p, dict) and p.get("type") == "text"
    )
  if not isinstance(content, str) or not content.startswith(_PROMPT):
    return None
  # The reply follows the last "\nAI: " the template adds; a user message
  # containing "\nAI: " itself only loses its tail.
  return content[len(_PROMPT):].split("\nAI: ", 1)[0]


def _title_case(word: str, first: bool) -> str:
  if any(c.isupper() for c in word):  # Names, acronyms: as written.
    return word
  if not first and word.lower() in _SMALL_WORDS:
    return word.lower()
  return word[:1].upper() + word[1:]


def _topic(sentence: str) -> list[str]:
  """The words of a sentence, openers and markup removed."""
  sentence = re.sub(r"https?://\S+", " ", sentence)
  sentence = re.sub(r"[`*_#>\[\]()\"“”]", " ", sentence)
  sentence = " ".join(sentence.split()).strip(" .,;:!?-")
  previous = None
  while sentence != previous:
    previous = sentence
    sentence = _FILLER.sub("", sentence).strip(" .,;:!?-")
  sentence = _ARTICLE.sub("", sentence)
  words = (w.strip(".,;:!?'‘’") for w in sentence.split())
  return [w for w in words if w]


def make_title(text: str) -> str:
  """Up to _MAX_WORDS words naming the topic of the first message.

  Taken from its first sentence with a topic ("Hi! Can you ..." skips the
  greeting); "e.g." or "3.5" don't end a sentence (no space after the dot).
  """
  for line in text.splitlines():
    for sentence in re.split(r"(?<=[.?!])\s", line):
      words = _topic(sentence)
      if len(words) > _MAX_WORDS:
        words = words[:_MAX_WORDS]
        while len(words) > 1 and words[-1].lower() in _TRAILING:
          words.pop()
      if words:
        return " ".join(_title_case(w, i == 0) for i, w in enumerate(words))
  return _FALLBACK


def _reply(handler: Any, body: dict[str, Any], text: str) -> None:
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
        {"index": 0, "delta": {"content": text}, "finish_reason": None},
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
                    "message": {"role": "assistant", "content": text},
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


def answer(handler: Any, body: dict[str, Any]) -> bool:
  """Replies to LibreChat's title request with a title from its first message.

  Returns True if it replied (the caller must stop), False to serve the
  request normally.
  """
  if not _ENABLED:
    return False
  first = _first_message(body)
  if first is None:
    return False
  title = make_title(first)
  _log(f"answered: {title!r}")
  _reply(handler, body, title)
  return True
