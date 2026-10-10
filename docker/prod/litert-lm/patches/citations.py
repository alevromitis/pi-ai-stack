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
3. ends the reply with one "Sources: [Title](URL), ..." line. If the model
   linked none of the offered sources, it lists them all. Otherwise it
   lists the ones it linked, plus the top result of each search it linked
   nothing from (gemma4-e2b often links only one of two parallel
   searches). Links the model made up never reach the line.

When no search ran since the user's message, a Sources block with no link
in it ("Sources: []", "[Odyssey]([URL not provided])", copied from the
prompt's citing rule) is dropped; one with real links passes untouched
(it may cite an earlier turn's results).

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
_SEARX_URL = re.compile(r"^URL: (\S+)$", re.MULTILINE)
_SEARX_TITLE = re.compile(r"^Title: (.*)$", re.MULTILINE)
_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


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


def _wikipedia(output: str) -> list[tuple[str, str]]:
  for line in reversed(output.splitlines()):
    if line.startswith("Sources: "):
      return [(t, u) for t, u in _MD_LINK.findall(line)]
  return []


def _searxng(output: str) -> list[tuple[str, str]]:
  # Results are blank-line separated blocks of "Key: value" lines.
  found = []
  for block in output.split("\n\n"):
    url = _SEARX_URL.search(block)
    if url:
      title = _SEARX_TITLE.search(block)
      found.append((title.group(1).strip() if title else url.group(1), url.group(1)))
  return found


def _page(url: str, output: str) -> list[tuple[str, str]]:
  if not url or output.lstrip().startswith(("Error", "🔧", "❌")):
    return []
  heading = _HEADING.search(output)
  return [(heading.group(1).strip() if heading else url, url)]


def collect(body: dict[str, Any]) -> list[list[tuple[str, str]]] | None:
  """Per tool result since the last user message, its (title, url) sources.

  [] when there are none (the reply is still filtered), None when disabled.
  """
  messages = body.get("messages")
  if not _ENABLED or not isinstance(messages, list):
    return None
  start = 0
  for i, msg in enumerate(messages):
    if isinstance(msg, dict) and msg.get("role") == "user":
      start = i + 1
  calls: dict[str, tuple[str, dict[str, Any]]] = {}
  groups: list[list[tuple[str, str]]] = []
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
  return groups


def _line(groups: list[list[tuple[str, str]]], reply: str) -> str:
  # Substring, not a URL regex: Wikipedia paths hold parentheses.
  linked = [[(t, u) for t, u in g if u in reply] for g in groups]
  if any(linked):
    picked = [l or g[:1] for l, g in zip(linked, groups)]
  else:
    picked = groups
  used = dict((u, t) for g in picked for t, u in g)
  return "Sources: " + ", ".join(f"[{t}]({u})" for u, t in used.items())


class Filter:
  """Streams a reply, swapping the model's Sources block for the server's."""

  def __init__(self, sources: list[list[tuple[str, str]]]):
    self._sources = sources
    self._partial = ""       # current line, held while it may be a header
    self._passing = False    # current line already (partly) streamed
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
        out.append(piece)
        self._passing = not piece.endswith("\n")
      else:
        self._partial += piece
        if self._partial.endswith("\n"):
          if _HEADER.match(self._partial):
            self._block = [self._partial]
          else:
            out.append(self._partial)
          self._partial = ""
        elif not self._maybe_header(self._partial):
          out.append(self._partial)
          self._partial = ""
          self._passing = True
    return self._emit(out)

  @staticmethod
  def _maybe_header(partial: str) -> bool:
    word = _LEAD.sub("", partial).lower()
    return "sources".startswith(word) or word.startswith("source")

  def finish(self, has_tool_calls: bool) -> str:
    """Returns the rest of the reply, ending with the Sources line."""
    held, block = self._partial, self._block
    if block is None and held and _HEADER.match(held):
      held, block = "", [held]
    if has_tool_calls:
      return self._emit([held, *(block or [])])
    if not self._sources and (block is None or _URL.search("".join(block))):
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
      _log("dropped a Sources block with no links")
      return text
    line = _line(self._sources, self._seen)
    _log(f"{sum(map(len, self._sources))} offered; {line[:160]}")
    if not self._wrote:
      return text + line
    sep = "" if self._tail.endswith("\n\n") else "\n" if self._tail.endswith("\n") else "\n\n"
    return text + sep + line


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
