"""Repeats the system prompt's "Before answering:" paragraph in every user message.

Not part of upstream LiteRT-LM: copied into litert_lm_cli/commands/serve/ by
the Dockerfile and wired in by apply_reminder.py.

gemma4-e2b follows the newest turn far more than a system prompt thousands
of tokens back: asked in the system prompt to reply in English, it mostly
didn't with thinking off; the same request appended to the user's message
was followed 8 times out of 8 (2026-10-10). So instead of rules for
kinds of questions, the system prompt ends with one general paragraph on how
to work out any answer (what is asked, where each fact and number comes
from, what to cite), and every user message gets a copy of it in
parentheses. The text comes from the system prompt itself, so it is written
in one place (librechat.yaml), and requests whose system prompt has no such
paragraph (LibreChat's memory agent, titles) pass through untouched. Every
user message in the history gets the same copy, so the prefix cache (which
compares whole histories) still matches.

Runs after citations.collect and presearch, which read the user's own words.

Env:
  LITERT_REMINDER=0   disable (messages pass through untouched)
"""

from __future__ import annotations

import os
import re
from typing import Any

_ENABLED = os.environ.get("LITERT_REMINDER", "1") != "0"
# The paragraph: from the heading to the next blank line or the end.
_PARAGRAPH = re.compile(r"^Before answering:.*(?:\n(?![ \t]*\n).*)*", re.MULTILINE)


def _text(content: Any) -> str:
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    return "".join(
        p.get("text", "") for p in content
        if isinstance(p, dict) and p.get("type") == "text"
    )
  return ""


def _note(messages: list[Any]) -> str:
  for msg in messages:
    if isinstance(msg, dict) and msg.get("role") in ("system", "developer"):
      found = _PARAGRAPH.search(_text(msg.get("content")))
      if found:
        return "(" + " ".join(found.group(0).split()) + ")"
  return ""


def apply(body: dict[str, Any]) -> None:
  """Appends the note, in place, to each user message that lacks it."""
  messages = body.get("messages")
  if not _ENABLED or not isinstance(messages, list):
    return
  note = _note(messages)
  if not note:
    return
  for msg in messages:
    if not isinstance(msg, dict) or msg.get("role") != "user":
      continue
    content = msg.get("content")
    if note in _text(content):
      continue
    if isinstance(content, str):
      msg["content"] = f"{content}\n\n{note}"
    elif isinstance(content, list):
      content.append({"type": "text", "text": f"\n\n{note}"})
