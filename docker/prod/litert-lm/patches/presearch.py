"""Searches before the model runs, by answering a factual question with a tool call.

Not part of upstream LiteRT-LM: copied into litert_lm_cli/commands/serve/ by
the Dockerfile and wired in by apply_presearch.py.

gemma4-e2b decides badly whether to search: told to search for every fact,
it still answers some from memory. So when a request ends with a new user
message that reads as a plain factual question ("When was the Eiffel Tower
completed?", "tell me about Kariba Dam") and the request offers a search
tool, the server replies at once, without the engine, with a call to that
tool, the question as its query. LibreChat runs it and sends the results
back; the model then only has to read and answer (and may search again).
An arithmetic question ("what is 17*23") becomes a `calculate` call the
same way when the Calculator tools are on, and "what day of the week was
20 July 1969?" a `date_add` call with no days added (its answer names the
weekday). Math in words ("square root of 7921", "12 times 7") is left to
the model, which picks the calculator itself.

Which tool: wikipedia_search, or searxng_web_search for questions about now
("latest", "current", "today", prices, weather) or when Wikipedia is off.
LibreChat appends "_mcp_<server>" to tool names, so names match by prefix.

Anything else goes to the model as before: messages that mention "you",
"I" or "my", or (later in a chat) "he", "it", "that"..., since they depend
on context the query wouldn't carry; requests for advice or how-tos; text
with URLs or code; long or multi-line messages; non-Latin text.

The prefix cache (prefix_cache.py) accepts the server's tool-call turn, so
the follow-up request still prefills only the new messages.

Env:
  LITERT_PRESEARCH=0   disable (every request reaches the model)
"""

from __future__ import annotations

import datetime
import json
import os
import re
from typing import Any

import click

from litert_lm_cli.commands.serve import openai_common

_ENABLED = os.environ.get("LITERT_PRESEARCH", "1") != "0"
_MAX_CHARS = 200

# Stripped before deciding, repeatedly: "hey, can you tell me who...".
_FILLER = re.compile(
    r"^(?:hey|hi|hello|ok|okay|so|um|please|pls|quick question)\b[\s,!:.-]*",
    re.IGNORECASE,
)
# A request to look something up: what follows is the query, question or not.
_ASK = re.compile(
    r"^(?:(?:can|could|would|will) you (?:please )?(?:tell me|find out|look up|check)"
    r"|(?:do|does) (?:you|anyone) know"
    r"|i (?:wonder|want to know|would like to know|'d like to know)"
    r"|tell me|find out|look up|lookup|search(?: for)?)\b\s*(?:about\b\s*)?",
    re.IGNORECASE,
)
_QUESTION = re.compile(
    r"^(?:who|whom|whose|what|what's|whats|when|where|which|who's"
    r"|how (?:many|much|old|long|tall|big|high|far|deep|large|wide|fast|heavy|hot|cold)"
    r"|(?:in|on|at|from|for|by) (?:what|which|whom))\b",
    re.IGNORECASE,
)
# Yes/no questions count only with a question mark ("Have a nice day").
_YES_NO = re.compile(r"^(?:is|are|was|were|did|does|has|have|had)\b.*\?$", re.IGNORECASE)
# "us" only in lowercase ("the US president" is a lookup), and "I" not
# after a capitalised word ("World War I").
_ALWAYS_PERSONAL = re.compile(
    r"\b(?:(?i:you|your|yours|yourself|i'm|i've|me|my|mine|myself|we|our|ours)|i|us)\b"
    r"|(?:^|\b[a-z]+\W+)I\b|\bI\s+(?!of\b)[a-z]"
)
# Point back into the chat; only fine in its first message, and there only
# next to a name ("How tall is the Eiffel Tower and when was it built?", not
# "How long is it?"). Later in a chat a question must name its subject
# itself: "What's the best time to visit?" after "Tell me about Santorini"
# searched for "best time to visit".
_REFERS_BACK = re.compile(
    r"\b(?:he|she|it|they|him|her|his|hers|its|their|theirs|them"
    r"|this|that|these|those|there|above|previous|earlier)\b",
    re.IGNORECASE,
)
_NOT_LOOKUP = re.compile(
    r"\b(?:should|best way|recommend|opinion|think|feel|why|how to|how do|how can"
    r"|how should|write|rewrite|translate|summari[sz]e|explain|compare|difference"
    r"|meaning of life|joke|poem|story|code|function|script|regex|bug|error"
    r"|what (?:time|day|date)|percent"
    # Calculator questions: dates, counting letters, math in words.
    r"|(?:days|weeks|months|years) (?:until|till|since|between|ago|from|after|before)"
    r"|letters?|syllables?|characters?|words? (?:are )?in|sentence|the word|spell|[a-z]'s"
    r"|square root|cube root|sqrt|squared|cubed|to the power|factorial"
    r"|logarithm|log of|divided by|multiplied by|modulo"
    r"|\d+(?:\.\d+)?\s*(?:plus|minus|times|over)\s*\d)\b|%",
    re.IGNORECASE,
)
_RECENT = re.compile(
    r"\b(?:latest|current|currently|today|tonight|now|right now|recent|recently"
    r"|this (?:week|month|year)|news|price|prices|cost|stock|weather|forecast"
    r"|score|standings|live)\b",
    re.IGNORECASE,
)
# Comparing numbers is for the calculator or the model: "9.11 or 9.9?".
_NUMBERS = re.compile(r"\d.*\b(?:or|than|vs|versus)\b.*\d", re.IGNORECASE)
_SKIP = re.compile(r"https?://|www\.|```|`")
_ARITHMETIC = re.compile(r"^[\d\s.,+\-*/x×÷^%()]+$")
_OPERATOR = re.compile(r"\d\s*[+\-*/x×÷^%]\s*\(?\s*\d")
_MATH_PREFIX = re.compile(
    r"^(?:what(?:'s| is)|calculate|compute|how much is|solve|evaluate)\b\s*",
    re.IGNORECASE,
)
# "What day of the week was 20 July 1969?". Only dates date_add reads:
# YYYY-MM-DD or a day, month name and year ("Christmas 2026" is not one).
_MONTH = (
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
)
_WEEKDAY = re.compile(
    r"^(?:what|which) (?:day of the week|day|weekday)"
    r" (?:was|is|will be|did|does|will)\s+(?:the\s+)?(?P<date>"
    r"\d{4}-\d{1,2}-\d{1,2}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)? (?:of )?{_MONTH},? \d{{1,4}}"
    rf"|{_MONTH} \d{{1,2}}(?:st|nd|rd|th)?,? \d{{1,4}})"
    r"(?:\s+(?:fall on|fall|land on|be))?\s*\??$",
    re.IGNORECASE,
)
# A capitalised word after the first: "the Eiffel Tower", "World War I".
_NAME = re.compile(r"(?<=\S)\s+[A-ZÀ-Þ]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
_LETTER = re.compile(r"[^\W\d_]")


def _log(text: str) -> None:
  click.echo(click.style(f"presearch: {text}", fg="green"))


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


def _tool(body: dict[str, Any], prefix: str, param: str) -> str | None:
  """The offered tool whose name starts with `prefix` and takes `param`."""
  for tool in body.get("tools") or []:
    fn = (tool or {}).get("function") or {}
    name = fn.get("name") or ""
    props = (fn.get("parameters") or {}).get("properties") or {}
    if name.startswith(prefix) and param in props:
      return name
  return None


def _strip(text: str) -> tuple[str, bool]:
  """The text without fillers and ask-phrases, and whether it had an ask-phrase."""
  text = " ".join(text.split())
  asked = False
  while True:
    stripped = _FILLER.sub("", text, count=1)
    unasked = _ASK.sub("", stripped, count=1)
    asked = asked or unasked != stripped
    if unasked == text:
      return text, asked
    text = unasked


def plan(body: dict[str, Any]) -> tuple[str, dict[str, str], str] | None:
  """(tool name, arguments, reason) to call before the model, or None."""
  if not _ENABLED or body.get("tool_choice") == "none":
    return None
  messages = body.get("messages")
  if not isinstance(messages, list) or not messages:
    return None
  last = messages[-1]
  if not isinstance(last, dict) or last.get("role") != "user":
    return None
  raw = _text(last.get("content")).strip()
  if not raw or "\n" in raw or len(raw) > _MAX_CHARS or _SKIP.search(raw):
    return None
  letters = _LETTER.findall(raw)
  if letters and len(_LATIN.findall(raw)) < 0.8 * len(letters):
    return None
  text, asked = _strip(raw)

  math = _MATH_PREFIX.sub("", text).rstrip(" ?=.")
  if _ARITHMETIC.match(math) and _OPERATOR.search(math):
    calc = _tool(body, "calculate", "expression")
    return (calc, {"expression": math}, "arithmetic") if calc else None

  weekday = _WEEKDAY.match(text)
  if weekday:
    add = _tool(body, "date_add", "date")
    if add:
      date = re.sub(r"(?<=\d) of ", " ", weekday.group("date"))
      return add, {"date": date.strip(" ,.")}, "weekday"

  later = any(
      isinstance(m, dict) and m.get("role") == "assistant" for m in messages[:-1]
  )
  if not (asked or _QUESTION.match(text) or _YES_NO.match(text)):
    return None
  if _ALWAYS_PERSONAL.search(text) or _NOT_LOOKUP.search(text):
    return None
  if _OPERATOR.search(text) or _NUMBERS.search(text):
    return None
  if later and not _NAME.search(text):
    return None
  if _REFERS_BACK.search(text) and (later or not _NAME.search(text)):
    return None
  query = text.rstrip(" ?.!")
  if len(query) < 3:
    return None

  wiki = _tool(body, "wikipedia_search", "query")
  web = _tool(body, "searxng_web_search", "query")
  if web and (_RECENT.search(text) or not wiki):
    return web, {"query": query}, "web question"
  if wiki:
    return wiki, {"query": query}, "question"
  return None


def answer(handler: Any, body: dict[str, Any]) -> bool:
  """Replies with a search tool call for a factual question.

  Returns True if it replied (the caller must stop), False to serve the
  request normally.
  """
  planned = plan(body)
  if planned is None:
    return False
  name, args, reason = planned
  # Not the query: it may be personal.
  _log(f"{name} ({reason})")

  now = datetime.datetime.now(datetime.timezone.utc)
  now_str = now.strftime("%Y%m%d%H%M%S%f")
  created = int(now.timestamp())
  call = {
      "id": f"call_{now_str}_presearch",
      "type": "function",
      "function": {"name": name, "arguments": json.dumps(args)},
  }
  base = {
      "id": f"chatcmpl_{now_str}",
      "created": created,
      "model": body.get("model"),
  }
  handler.headers_sent = True
  handler.send_response(200)
  if body.get("stream"):
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    for choice in (
        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None},
        {
            "index": 0,
            "delta": {"tool_calls": [{"index": 0, **call}]},
            "finish_reason": None,
        },
        {"index": 0, "delta": {}, "finish_reason": "tool_calls"},
    ):
      handler.wfile.write(
          openai_common.sse_data(
              openai_common.dump_json({
                  **base,
                  "object": "chat.completion.chunk",
                  "choices": [choice],
              })
          )
      )
    handler.wfile.write(openai_common.format_sse_final())
  else:
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(
        (
            openai_common.dump_json({
                **base,
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [call],
                    },
                    "finish_reason": "tool_calls",
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
