"""Asks for English replies to messages in other scripts (Greek, ...).

Not part of upstream LiteRT-LM: copied into litert_lm_cli/commands/serve/ by
the Dockerfile and wired in by apply_english.py.

gemma4-e2b's Greek garbles names and letters: "Νόμο (Tokyo)",
"ΜουνάκOlympus", "29ης মে 1453" (Bengali), "ホμερ" (Japanese), at
temperature 1.0 and 0.5 alike, while its facts were usually right. A
system-prompt rule to reply in English was mostly ignored with thinking
off. So every user message written mostly in a non-Latin script gets a
note appended, "(Please reply in English, and search in English.)", which
the model follows far more reliably ("[Answer in English; ...]" made it
start replies with "Answer:"). Every such message in the history gets the
same note, so the prefix cache (which compares whole histories) still
matches.
Requests for LibreChat's memory agent (only set_memory/delete_memory) are
left alone.

Env:
  LITERT_ENGLISH=0   disable (messages pass through untouched)
"""

from __future__ import annotations

import os
import re
from typing import Any

_ENABLED = os.environ.get("LITERT_ENGLISH", "1") != "0"
_NOTE = "(Please reply in English, and search in English.)"
_LETTER = re.compile(r"[^\W\d_]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
_MEMORY_TOOLS = frozenset({"set_memory", "delete_memory"})


def _foreign(text: str) -> bool:
  """Mostly letters outside the Latin script."""
  letters = _LETTER.findall(text)
  return len(letters) >= 3 and len(_LATIN.findall(text)) < 0.5 * len(letters)


def apply(body: dict[str, Any]) -> None:
  """Appends the note, in place, to each user message in a non-Latin script."""
  messages = body.get("messages")
  if not _ENABLED or not isinstance(messages, list):
    return
  names = {
      (t.get("function") or {}).get("name")
      for t in body.get("tools") or []
      if isinstance(t, dict)
  }
  if names and names <= _MEMORY_TOOLS:
    return
  for msg in messages:
    if not isinstance(msg, dict) or msg.get("role") != "user":
      continue
    content = msg.get("content")
    if isinstance(content, str):
      if _foreign(content) and _NOTE not in content:
        msg["content"] = f"{content}\n\n{_NOTE}"
    elif isinstance(content, list):
      text = "".join(
          p.get("text", "") for p in content
          if isinstance(p, dict) and p.get("type") == "text"
      )
      if _foreign(text) and _NOTE not in text:
        content.append({"type": "text", "text": f"\n\n{_NOTE}"})
