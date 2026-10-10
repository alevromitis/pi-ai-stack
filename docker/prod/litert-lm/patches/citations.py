"""Ends answers built on search results with a Sources line the server writes.

Not part of upstream LiteRT-LM: copied into litert_lm_cli/commands/serve/ by
the Dockerfile and wired in by apply_citations.py.

gemma4-e2b cites unpredictably: it skips or reformats the Sources line,
drops it under a length limit ("in two sentences"), and does worse with
SearXNG results, which carry no ready-made line. So when a request's
messages since the last user turn hold results from the search tools
(wikipedia_search, searxng_web_search, web_url_read - LibreChat appends
"_mcp_<server>" to the names) and the reply calls no tool, the server:

1. collects the sources those results offer: the Sources line of a
   Wikipedia result, each Title/URL of a SearXNG result, and the URL of a
   page read (titled by its first heading);
2. holds back any Sources block the model starts writing (a line like
   "Sources:" or "**Sources**:" and the link lines after it) instead of
   streaming it;
3. ends the reply with one "Sources: [Title](URL), ..." line of the
   offered sources that back the answer. What backs it is the answer's
   own words, not the question's: every result shares the question's
   words ("Greek", "god", "sea"), only a source holding "Poseidon" backs
   "Poseidon". So a source counts if it holds one of the answer's names
   or long numbers that the question and queries don't (capitalised
   words, numbers of 3+ digits or with a decimal point), or 30% (at least
   two) of its other new words; among those, the ones scoring at least
   half the best (a source the model linked in its text needs only to
   back it). What the model listed in its own Sources block counts for
   nothing: it listed "Eiffel Tower (Paris, Tennessee)" for the Paris
   tower's height. Links the model made up never reach the line, and with
   nothing backing the answer ("I don't know", or an answer from memory)
   there is no line.
   Listing every result, as before, cited "The Greek" and "The God
   Beneath the Sea" for Poseidon.

When no search ran since the user's message, a Sources block passes
untouched only if every link in it came from a tool result earlier in the
chat (a follow-up citing the last turn's article). Otherwise it is dropped:
"Sources: []", "[Odyssey]([URL not provided])" copied from the prompt's
citing rule, or "[Calculation result](https://example.com/...)" after a
date calculation.

Text streams as before; only a line that might start a Sources block waits
for its newline.

Env:
  LITERT_CITATIONS=0   disable (replies pass through untouched)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import click

_ENABLED = os.environ.get("LITERT_CITATIONS", "1") != "0"

_URL = re.compile(r"https?://[^\s)\]>\"'<]+")
# Wikipedia paths hold balanced parentheses: [Titan (moon)](.../Titan_(moon)).
_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://(?:[^\s()]|\([^\s()]*\))+)\)")
# A Sources block header: "Sources:", "**Sources:**", "### Sources", "Source:".
_HEADER = re.compile(r"^[\s*_#>-]*sources?[\s*_]*(?::|\s*$)", re.IGNORECASE)
_LEAD = re.compile(r"^[\s*_#>-]+")
# The same header inside a line: "... is "Example Domain". Sources: [...](URL)".
_INLINE = re.compile(r"(?<=\s)[*_]*Sources?[*_]*:")
# What the end of a streamed piece may be the start of: "S", "Sour", "**Sources".
_INLINE_START = re.compile(r"[*_]*S(?:o(?:u(?:r(?:c(?:e(?:s?[*_]*)?)?)?)?)?)?")
_SEARX_URL = re.compile(r"^URL: (\S+)$", re.MULTILINE)
_SEARX_TITLE = re.compile(r"^Title: (.*)$", re.MULTILINE)
_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_PASSAGE = re.compile(r"^\[\d+\] (.+?)(?: › .*)?$")
_WORD = re.compile(r"[^\W_]+(?:[.,]\d+)*")
# Common in answers, useless for telling sources apart.
_STOP = frozenset("""
about above after again also another answer around based because been before
being below between both cannot could does doing during each either every
found from have having here however information into just know known like
many more most much must only other over page provide provided result
results search searched should since some source sources such than that
their them then there these they this those through under until very were
what when where which while will with would your today according sure
certainly called located known also
""".split())
# Names in an answer: "Poseidon", "Au", "NaCl"; not a capitalised "The".
_CAPITALISED = re.compile(r"\b[A-ZÀ-ÞΑ-Ω][^\W_]+")
_SHORT = frozenset("""
the and but for nor yet its his her our you she him they them who how why
was are has had not can may all any few one two six ten yes per via in on
at to of is as by or if it we he me my so no do be an up us am
""".split())


def _log(text: str) -> None:
  click.echo(click.style(f"citations: {text}", fg="cyan"))


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


def _words(text: str) -> set[str]:
  """Content words and numbers ("319,045" -> "319045")."""
  out = set()
  for word in _WORD.findall(text.lower()):
    word = word.replace(",", "")
    if word[0].isdigit() and len(word) >= 2 or len(word) >= 4 and word not in _STOP:
      out.add(word)
  return out


# A source: (title, url, the text the result gave for it).
Source = tuple[str, str, str]


class Offered(list):
  """Sources per tool result; .asked is the user's message and the queries,
  .known every URL any tool result in the chat gave."""

  asked = ""
  known: frozenset[str] = frozenset()


def _wikipedia(output: str) -> list[Source]:
  texts: dict[str, list[str]] = {}
  title = None
  links: list[tuple[str, str]] = []
  for line in output.splitlines():
    passage = _PASSAGE.match(line)
    if passage:
      title = passage.group(1)
      texts.setdefault(title, [])
    elif line.startswith("Sources: "):
      links = _MD_LINK.findall(line)
    elif title is not None:
      texts[title].append(line)
  return [(t, u, t + " " + " ".join(texts.get(t, []))) for t, u in links]


def _searxng(output: str) -> list[Source]:
  # Results are blank-line separated blocks of "Key: value" lines.
  found = []
  for block in output.split("\n\n"):
    url = _SEARX_URL.search(block)
    if url:
      title = _SEARX_TITLE.search(block)
      found.append((title.group(1).strip() if title else url.group(1), url.group(1), block))
  return found


def _page(url: str, output: str) -> list[Source]:
  if not url or output.lstrip().startswith(("Error", "🔧", "❌")):
    return []
  heading = _HEADING.search(output)
  return [(heading.group(1).strip() if heading else url, url, output[:8000])]


def collect(body: dict[str, Any]) -> Offered | None:
  """Per tool result since the last user message, its sources.

  Empty when there are none (the reply is still filtered), None when
  disabled.
  """
  messages = body.get("messages")
  if not _ENABLED or not isinstance(messages, list):
    return None
  start = 0
  for i, msg in enumerate(messages):
    if isinstance(msg, dict) and msg.get("role") == "user":
      start = i + 1
  calls: dict[str, tuple[str, dict[str, Any]]] = {}
  groups = Offered()
  # Links in tool results, and pages the model read (web_url_read's url).
  groups.known = frozenset(
      url for msg in messages if isinstance(msg, dict)
      for text in (
          [_text(msg.get("content"))] if msg.get("role") == "tool" else
          [str((c.get("function") or {}).get("arguments") or "") for c in msg.get("tool_calls") or [] if isinstance(c, dict)]
      )
      for url in _URL.findall(text)
  )
  asked = [_text(messages[start - 1].get("content"))] if start and isinstance(messages[start - 1], dict) else []
  for msg in messages[start:]:
    if not isinstance(msg, dict):
      continue
    for call in msg.get("tool_calls") or []:
      fn = call.get("function") or {}
      try:
        args = json.loads(fn.get("arguments") or "{}")
      except (TypeError, ValueError):
        args = {}
      calls[call.get("id")] = (fn.get("name") or "", args if isinstance(args, dict) else {})
      if isinstance(args, dict):
        asked.extend(str(v) for v in args.values())
    if msg.get("role") != "tool":
      continue
    name, args = calls.get(msg.get("tool_call_id"), ("", {}))
    output = _text(msg.get("content"))
    if "wikipedia_search" in name:
      found = _wikipedia(output)
    elif "searxng_web_search" in name:
      found = _searxng(output)
    elif "web_url_read" in name:
      found = _page(str(args.get("url") or ""), output)
    else:
      continue
    if found:
      groups.append(found)
  groups.asked = "\n".join(asked)
  return groups


def _line(groups: list[list[Source]], prose: str, asked: str) -> str:
  """The Sources line for the offered sources that back the answer, or ""."""
  # Link text stays, URLs go: "as [War and Peace](...) says".
  answer = _URL.sub(" ", _MD_LINK.sub(r"\1", prose))
  asked_tokens = {t.lower() for t in _WORD.findall(asked)}
  new = _words(answer) - _words(asked)
  strong = {
      n.lower() for n in _CAPITALISED.findall(answer)
      if len(n) >= 2 and n.lower() not in _STOP | _SHORT
      and n.lower() not in asked_tokens
  } | {w for w in new if w[0].isdigit() and (len(w) >= 3 or not w.isdigit())}
  # Without a name or number, a source needs a fair share of the answer's
  # words: an explanation of range() shares "sequence" and "integers"
  # with "Lazy evaluation".
  need = max(2, -(-3 * len(new - strong) // 10))
  score: dict[str, int] = {}
  for g in groups:
    for _, u, text in g:
      words = _words(text)
      tokens = {t.lower().replace(",", "") for t in _WORD.findall(text)}
      hits, other = len(strong & tokens), len((new - strong) & words)
      score[u] = 2 * hits + other if hits or other >= need else 0
  best = max(score.values(), default=0)
  used: dict[str, str] = {}
  for g in groups:
    for t, u, _ in g:
      # Substring, not a URL regex: Wikipedia paths hold parentheses.
      if score[u] and (score[u] * 2 >= best or u in prose):
        used.setdefault(u, t)
  if not used:
    return ""
  return "Sources: " + ", ".join(f"[{t}]({u})" for u, t in used.items())


class Filter:
  """Streams a reply, swapping the model's Sources block for the server's."""

  def __init__(self, sources: list[list[Source]]):
    self._sources = sources
    self._asked = getattr(sources, "asked", "")
    self._known = getattr(sources, "known", frozenset())
    self._partial = ""       # current line, held while it may be a header
    self._passing = False    # current line already (partly) streamed
    self._hold = ""          # end of a streaming line that may start "Sources:"
    self._block: list[str] | None = None  # held Sources block lines
    self._seen = ""          # everything the model wrote, for its links
    self._tail = ""          # last characters streamed
    self._wrote = False      # streamed anything but whitespace

  def _emit(self, parts: list[str]) -> str:
    text = "".join(parts)
    self._tail = (self._tail + text)[-2:]
    self._wrote = self._wrote or bool(text.strip())
    return text

  def feed(self, text: str) -> str:
    """Returns the part of `text` that can be streamed now."""
    self._seen += text
    out = []
    for piece in re.split(r"(?<=\n)", text):
      if not piece:
        continue
      if self._block is not None:
        self._block.append(piece)
      elif self._passing:
        out.append(self._continue(piece))
      else:
        self._partial += piece
        if self._partial.endswith("\n"):
          if _HEADER.match(self._partial):
            self._block = [self._partial]
          else:
            out.append(self._continue(self._partial))
          self._partial = ""
        elif not self._maybe_header(self._partial):
          partial, self._partial = self._partial, ""
          out.append(self._continue(partial))
    return self._emit(out)

  def _continue(self, piece: str) -> str:
    """The streamable part of a piece of a line already streaming. A
    Sources header inside the line starts the held block there; a last word
    that may begin one waits for the next piece."""
    text, self._hold = self._hold + piece, ""
    # One character before it, so a header right at its start is seen.
    probe = (self._tail[-1:] or "\n") + text
    found = _INLINE.search(probe)
    if found:
      self._block = [probe[found.start():]]
      self._passing = False
      return probe[1:found.start()]
    self._passing = not text.endswith("\n")
    cut = max(probe.rfind(" "), probe.rfind("\t"), probe.rfind("\n")) + 1
    if self._passing and cut and _INLINE_START.fullmatch(probe[cut:]):
      self._hold = probe[cut:]
      return probe[1:cut]
    return text

  @staticmethod
  def _maybe_header(partial: str) -> bool:
    word = _LEAD.sub("", partial).lower()
    return "sources".startswith(word) or word.startswith("source")

  def finish(self, has_tool_calls: bool) -> str:
    """Returns the rest of the reply, ending with the Sources line."""
    held, block = self._hold + self._partial, self._block
    if block is None and held and _HEADER.match(held):
      held, block = "", [held]
    if has_tool_calls:
      return self._emit([held, *(block or [])])
    links = _URL.findall("".join(block or []))
    if not self._sources and (block is None or links and set(links) <= self._known):
      return self._emit([held, *(block or [])])
    # Prose the model wrote after its Sources header is kept; link lines go.
    rest = [
        line for line in "".join(block or []).splitlines()[1:]
        if _LEAD.sub("", line).strip() and not _URL.search(line)
    ]
    kept = "\n".join(rest)
    if kept and not (self._tail + held).endswith("\n"):
      kept = "\n" + kept
    text = self._emit([held + kept])
    if not self._sources:
      _log("dropped a Sources block with no known links")
      return text
    listed = "".join(block or [])
    prose = self._seen.replace(listed, "") if listed else self._seen
    line = _line(self._sources, prose, self._asked)
    _log(f"{sum(map(len, self._sources))} offered; {line[:160] or 'none backs the answer'}")
    if not line:
      return text
    if not self._wrote:
      return text + line
    sep = "" if self._tail.endswith("\n\n") else "\n" if self._tail.endswith("\n") else "\n\n"
    return text + sep + line


# Gemma's control tokens: "<|channel>", "<channel|>", "<|tool_call>", "<|"|>".
_MARKUP = re.compile(r"<\|[^<>|\s]{1,20}\|?>|<[^<>|\s]{1,20}\|>")


class Scrub:
  """Drops Gemma's control tokens that leak into the reply text ("...
  <channel|>Ana is 14 years old now."), and raw tool calls with them: with
  no tool offered, the model may call its trained-in search, and the reply
  would be "<|tool_call>call:google_search{...}<tool_call|>". Each token
  arrives whole in one chunk; a call spans many, so its state carries over.
  Runs whether citations are on or off."""

  def __init__(self) -> None:
    self._in_call = False

  def feed(self, text: str) -> str:
    out, pos = [], 0
    for token in _MARKUP.finditer(text):
      if not self._in_call:
        out.append(text[pos:token.start()])
      if token.group(0) == "<|tool_call>":
        self._in_call = True
      elif token.group(0) == "<tool_call|>":
        self._in_call = False
      pos = token.end()
    if not self._in_call:
      out.append(text[pos:])
    return "".join(out)


def stream_filter(handler: Any) -> Filter | None:
  """The Filter for this request, unless disabled (one use)."""
  sources = handler.__dict__.pop("citation_sources", None)
  return Filter(sources) if sources is not None else None


def apply(handler: Any, text: str, has_tool_calls: bool) -> str:
  """Non-streamed replies: the whole text through a Filter."""
  filt = stream_filter(handler)
  if filt is None:
    return text
  return filt.feed(text) + filt.finish(has_tool_calls)
