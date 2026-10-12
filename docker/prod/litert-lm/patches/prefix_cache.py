"""Prefix (KV) cache for `litert-lm serve`'s chat completions endpoint.

Not part of upstream LiteRT-LM: copied into litert_lm_cli/commands/serve/ by
the Dockerfile and wired in by apply_prefix_cache.py.

Upstream builds a fresh Conversation for every request, so the system
prompt, tool schemas and the whole history are prefilled again on every
turn - and on every tool round of an agent turn. Here the last few
Conversations stay alive. When a request's messages are one of them's
history plus the reply it generated, followed only by new user/tool
messages, just those new messages are sent and only they are prefilled.

Anything else (edited or regenerated messages, a chat not cached, changed
tools or sampler settings) is a miss, served exactly like upstream; the
cached conversations stay for the requests that do extend them. Several
live conversations alternate exactly on this engine, MTP on or off
(probes/clone_probe.py --switching, 2026-10-12). Each holds its KV cache in
RAM (~250-430 MB for a short chat on the Pi), so the least recently used
ones are closed past LITERT_PREFIX_CACHE_SIZE conversations, and while the
container's memory is over LITERT_PREFIX_CACHE_MAX_MB. The conversation in
use is never closed for memory.

New chats (system message(s) then one user message) don't prefill the
system prompt: they clone a snapshot that holds only it, prefilled when the
snapshot was created (patches/prefill_preface_option.py) and never sent to.
LITERT_PREFIX_CACHE_SNAPSHOT snapshots are kept, one per system prompt,
tools and settings, least recently used closed first; each saves a full
preface prefill when switching back (2026-10-12, Pi: 17 s with no tools,
29 s with Wikipedia, web search and calculator, 94 s with Playwright). The
default is 1: with several, clones of snapshots with different tools
faulted the MTP drafter more often, and the memory pushed the Pi into swap.
Clones of it continue exactly like a fresh conversation, with or without
other clones alive (probes/clone_probe.py --snapshot-control, 2026-10-12);
first token ~2-3 s instead of ~13 s.

A restart loses them all, and the first request then also waits for the
engine to load and its first decode to warm up: 36 s to the first token
with search and calculator on (2026-10-12). So each snapshot's request
(system messages, tools, settings) is saved to LITERT_PREFIX_CACHE_WARM_FILE,
and when the server starts it sends itself one-token requests for those
saved the same UTC day, most recently used first, the rest only while no
request has come for _WARM_IDLE_S (each holds the engine for 12-65 s) (LibreChat's {{current_date}} is UTC too; an older
system prompt would not be asked for again). Real requests queue behind
them, as they would behind the work they replace. The file holds the
system prompts, saved memories included.

With MTP on, a clone (or a chat
continuing one) sometimes faults in the drafter before its first token; the
request is then retried on a new clone or a fresh conversation
(_ENGINE_FAULTS).

Env:
  LITERT_PREFIX_CACHE=0            disable (pure upstream behaviour)
  LITERT_PREFIX_CACHE_SIZE=3       most conversations kept alive
  LITERT_PREFIX_CACHE_MAX_MB=5120  close idle ones above this (cgroup memory)
  LITERT_PREFIX_CACHE_SNAPSHOT=1   snapshots kept; 0 = new chats prefill
                                   their system prompt
  LITERT_PREFIX_CACHE_WARM_FILE=   where snapshot requests are saved for
                                   the next start; empty = not saved
"""

from __future__ import annotations

import collections.abc
import contextlib
import copy
import ctypes
import datetime
import hashlib
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from typing import Any

import click

import litert_lm
from litert_lm import conversation as _conversation_module

_ENABLED = os.environ.get("LITERT_PREFIX_CACHE", "1") != "0"
_MAX_ENTRIES = max(1, int(os.environ.get("LITERT_PREFIX_CACHE_SIZE", "3")))
_MAX_MB = int(os.environ.get("LITERT_PREFIX_CACHE_MAX_MB", "5120"))
_MAX_SNAPSHOTS = max(
    0, int(os.environ.get("LITERT_PREFIX_CACHE_SNAPSHOT", "1")))
_WARM_FILE = os.environ.get(
    "LITERT_PREFIX_CACHE_WARM_FILE", "/root/.litert-lm/prefix-snapshots.json")
_WARM_PORT = 9379  # entrypoint.sh's `serve --port`.
# Marks the server's own warm-up requests (an extra body field).
_WARM_FIELD = "litert_prefix_cache_warm"
_WARM_IDLE_S = 30
# Monotonic time the last request other than a warm-up ended (at first, the
# start), 0 while one runs.
_last_request = time.monotonic()
# The container's own cgroup (v2, private cgroup namespace).
_MEMORY_CURRENT = "/sys/fs/cgroup/memory.current"
_REUSABLE_ROLES = ("user", "tool")
# With MTP on, a conversation cloned from a snapshot, or one continuing such
# a clone, sometimes fails in the drafter before its first token: "Cannot
# auto-resize tensor kv_cache_k_N: no dims_signature exists", "Failed to
# allocate tensors", then "Task failed with state: 7" on that conversation.
# It happened with chats alone, 2 of 8 requests in one run
# (probes/mtp_interleave_probe.py, 2026-10-12). The same request on a fresh
# conversation went through every time, so it is retried there.
_ENGINE_FAULTS = (
    "Cannot auto-resize tensor",
    "Failed to allocate tensors",
    "Task failed with state",
)

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


_CONFIG_KEYS = (
    "model", "tools", "temperature", "top_p", "top_k", "seed",
    "reasoning_effort", "response_format",
)


def _config_key(body: collections.abc.Mapping[str, Any]) -> str:
  """Everything fixed at conversation creation must match to reuse it."""
  return json.dumps({k: body.get(k) for k in _CONFIG_KEYS}, sort_keys=True)


def engine_fault(e: BaseException) -> bool:
  """An error from the engine's buffers, not from the request."""
  return any(sign in str(e) for sign in _ENGINE_FAULTS)


class _RecordingConversation:
  """Proxies a Conversation and records the reply it generates.

  `fallbacks` make fresh conversations, each with the message to send it,
  for an engine fault before the first token (see _ENGINE_FAULTS).
  """

  def __init__(self, conv: litert_lm.Conversation, fallbacks=()):
    self._conv = conv
    self._fallbacks = list(fallbacks)
    self.text_parts: list[str] = []
    self.sent_parts: list[str] = []
    self.reported = False
    self.tool_calls: list[litert_lm.ToolCall] = []
    self.completed = False
    self.cancelled = False
    self.retried = False

  def __getattr__(self, name: str) -> Any:
    return getattr(self._conv, name)

  def cancel_process(self) -> None:
    self.cancelled = True
    self._conv.cancel_process()

  def send_message_async(self, message, *args, **kwargs):
    while True:
      started = False
      try:
        # Chunks are litert_lm.Message; str() is its text, as upstream
        # sends it.
        for chunk in self._conv.send_message_async(message, *args, **kwargs):
          started = True
          self.text_parts.append(str(chunk))
          self.tool_calls.extend(chunk.tool_calls)
          yield chunk
        break
      except Exception as e:  # pylint: disable=broad-exception-caught
        if started or self.cancelled or not self._fallbacks or not engine_fault(e):
          raise
        why = str(e).strip().splitlines()[-1].lstrip("└ ")[:80]
        self._conv.close()
        while True:
          try:
            self._conv, message, how = self._fallbacks.pop(0)()
            break
          except Exception:  # pylint: disable=broad-exception-caught
            if not self._fallbacks:
              raise
        self.retried = True
        _log(f"engine fault before the first token ({why}); retrying on {how}")
    self.completed = True

  def reply(self) -> dict[str, Any]:
    # The client sends back the reply it received, which the handlers
    # rewrite (citations' Sources line, leaked control tokens scrubbed, the
    # length note): matched against the raw text, every follow-up to a
    # search-backed answer was a miss. The KV keeps the raw generation.
    parts = self.sent_parts if self.reported else self.text_parts
    return {
        "role": "assistant",
        "content": "".join(parts),
        "tool_calls": [
            {"function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in self.tool_calls
        ],
    }


def sent(conv: Any, text: str | None) -> str | None:
  """Records the non-streamed reply as the client gets it; returns it."""
  if isinstance(conv, _RecordingConversation):
    conv.reported = True
    conv.sent_parts.append(text or "")
  return text


def recording(conv: Any, formatter: Any) -> Any:
  """Makes the stream formatter record each content delta it sends."""
  if isinstance(conv, _RecordingConversation):
    conv.reported = True
    format_delta = formatter.format_delta

    def recorded_delta(text: str) -> bytes:
      conv.sent_parts.append(text)
      return format_delta(text)

    formatter.format_delta = recorded_delta
  return formatter


class _Entry:

  def __init__(self, engine, conv, key: str, history: list[tuple[Any, ...]]):
    self.engine = engine
    self.conv = conv
    self.key = key
    self.history = history

# Cached conversations, least recently used first. A request owns the one it
# continues (or creates) until it completes cleanly.
_entries: list[_Entry] = []


def _memory_mb() -> int | None:
  try:
    with open(_MEMORY_CURRENT) as f:
      return int(f.read()) // 2**20
  except (OSError, ValueError):
    return None


def _close_oldest(why: str) -> None:
  old = _entries.pop(0)
  _log(f"evict: a {len(old.history)}-message conversation ({why})")
  old.conv.close()


def _trim() -> None:
  """Hands freed heap back to the kernel after conversations are closed.

  glibc keeps it otherwise: closing conversations and snapshots left the
  container at 5.4 GB of anonymous memory with one chat and two snapshots
  alive (2026-10-12), so the memory cap closed everything else and the Pi
  swapped.
  """
  try:
    before = _memory_mb()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    _log(f"trim: memory {before} -> {_memory_mb()} MB")
  except (OSError, AttributeError):
    pass


def _over_memory() -> int | None:
  mb = _memory_mb()
  return mb if mb is not None and mb > _MAX_MB else None


def _evict(max_entries: int) -> None:
  """Closes least recently used conversations past the caps.

  The memory cap closes snapshots first, keeping the most recent one: one
  costs 12-65 s to rebuild, a chat's history can cost minutes. It never
  closes the most recent chat: the one in use, or the one last used when a
  miss is about to create a new conversation.
  """
  while len(_entries) > max_entries:
    _close_oldest(f"over {max_entries} cached")
  if _over_memory():
    _trim()
  while len(_snapshots) > 1 and (mb := _over_memory()):
    _close_snapshot(f"memory {mb} MB > {_MAX_MB} MB")
    _trim()
  while len(_entries) > 1 and (mb := _over_memory()):
    _close_oldest(f"memory {mb} MB > {_MAX_MB} MB")
    _trim()


def _extends(
    e: _Entry,
    key: str,
    canonical: list[tuple[Any, ...]],
    messages: list[Any],
) -> tuple[int | None, str]:
  """(messages it holds, "") if the request continues e, else (None, why)."""
  if e.key != key:
    return None, "tools/model/sampler settings changed"
  n = len(e.history)
  if len(canonical) <= n or canonical[:n] != e.history:
    diverge = next(
        (i for i, (a, b) in enumerate(zip(canonical, e.history)) if a != b),
        min(len(canonical), n),
    )
    return None, f"history differs at message {diverge} (cached {n})"
  new_roles = {m.get("role") for m in messages[n:]}
  if not new_roles <= set(_REUSABLE_ROLES) or (
      "user" in new_roles and len(messages) - n > 1
  ):
    return None, f"new messages have roles {sorted(new_roles)}"
  return n, ""


def _match(
    engine,
    key: str,
    canonical: list[tuple[Any, ...]],
    messages: list[Any],
) -> _Entry | None:
  """The cached conversation the request continues, taken out of the cache.

  If several do, the one holding the most messages.
  """
  stale = [e for e in _entries if e.engine is not engine]
  if stale:
    _log(f"drop: engine was re-initialized ({len(stale)} cached)")
    for e in stale:
      _entries.remove(e)
      e.conv.close()
  best, reasons = None, []
  for e in reversed(_entries):  # Most recent first.
    n, why = _extends(e, key, canonical, messages)
    if n is None:
      reasons.append((e.key != key, why))  # Same settings sort first.
    elif best is None or n > len(best.history):
      best = e
  if best is None:
    if reasons:
      others = f" (+{len(reasons) - 1} other cached)" if len(reasons) > 1 else ""
      _log(f"miss: {sorted(reasons, key=lambda r: r[0])[0][1]}{others}")
    return None
  _entries.remove(best)
  return best


class _Snapshot:

  def __init__(self, engine, key: str, system: list[tuple[Any, ...]], conv,
               recipe: dict[str, Any]):
    self.engine = engine
    self.key = key
    self.system = system
    self.conv = conv
    self.recipe = recipe

# System-prompt snapshots, least recently used first.
_snapshots: list[_Snapshot] = []


def _close_snapshot(why: str) -> None:
  old = _snapshots.pop(0)
  _log(f"snapshot: closed one ({why})")
  old.conv.close()


def _clone(conv):
  """litert_lm_conversation_clone: in the C API, not wrapped in Python."""
  lib = conv._lib
  ptr_type = lib.litert_lm_conversation_create.restype
  lib.litert_lm_conversation_clone.restype = ptr_type
  lib.litert_lm_conversation_clone.argtypes = [ptr_type]
  ptr = lib.litert_lm_conversation_clone(conv._ptr)
  if not ptr:
    raise RuntimeError("litert_lm_conversation_clone failed")
  twin = copy.copy(conv)  # Same Python-side settings, own native handle.
  twin._ptr = ptr
  twin._current_callback = None
  return twin


def _today() -> str:
  return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def _load_warm() -> list[dict[str, Any]]:
  try:
    with open(_WARM_FILE) as f:
      recipes = json.load(f)
    return recipes if isinstance(recipes, list) else []
  except (OSError, ValueError):
    return []


def _recipe(body: collections.abc.Mapping[str, Any]) -> dict[str, Any]:
  """A new chat's request without its user message."""
  recipe = {k: body[k] for k in _CONFIG_KEYS if body.get(k) is not None}
  recipe["messages"] = body["messages"][:-1]
  return recipe


def _save_warm() -> None:
  """Records the snapshots kept, least recently used first."""
  if not _WARM_FILE:
    return
  recipes = [s.recipe for s in _snapshots]
  try:
    tmp = _WARM_FILE + ".tmp"
    with open(tmp, "w") as f:
      json.dump(recipes, f, ensure_ascii=False)
    os.replace(tmp, _WARM_FILE)
  except OSError as e:
    _log(f"warm: not saved ({e})")


def _warm() -> None:
  """Replays today's saved snapshot requests once the server listens."""
  recipes = [r for r in _load_warm() if r.pop("saved", None) == _today()]
  if not recipes:
    return
  deadline = time.monotonic() + 300
  while True:
    try:
      socket.create_connection(("127.0.0.1", _WARM_PORT), timeout=1).close()
      break
    except OSError:
      if time.monotonic() > deadline:
        return
      time.sleep(0.5)
  for i, recipe in enumerate(reversed(recipes)):  # Most recently used first.
    while i and not (
        _last_request and time.monotonic() - _last_request > _WARM_IDLE_S):
      time.sleep(1)
    body = dict(recipe, stream=False, max_tokens=16, **{_WARM_FIELD: True})
    body["messages"] = recipe["messages"] + [{"role": "user", "content": "Hi."}]
    started = time.monotonic()
    try:
      urllib.request.urlopen(urllib.request.Request(
          f"http://127.0.0.1:{_WARM_PORT}/v1/chat/completions",
          json.dumps(body).encode(), {"Content-Type": "application/json"},
      ), timeout=600).read()
      _log(f"warm: snapshot ready in {time.monotonic() - started:.1f} s")
    except Exception as e:  # pylint: disable=broad-exception-caught
      _log(f"warm: failed ({e})")


def _new_chat(engine, key, canonical, messages, translated_messages,
              create_kwargs, body):
  """The snapshot for a new chat's system prompt, tools and settings, or None.

  Most recently used last; created if none matches.
  """
  if (not _MAX_SNAPSHOTS or len(messages) < 2
      or messages[-1].get("role") != "user"
      or any(m.get("role") != "system" for m in messages[:-1])):
    return None
  for s in [s for s in _snapshots if s.engine is not engine]:
    _snapshots.remove(s)
    s.conv.close()
  system = canonical[:-1]
  snap = next(
      (s for s in _snapshots if s.key == key and s.system == system), None)
  if snap is not None:
    _snapshots.remove(snap)
  else:
    while len(_snapshots) >= _MAX_SNAPSHOTS:
      _close_snapshot(f"over {_MAX_SNAPSHOTS} kept")
    # With prefill_preface_option.py this argument means "prefill the
    # preface on creation"; the filter keeps its default.
    started = time.monotonic()
    conv = engine.create_conversation(
        messages=translated_messages[:-1],
        filter_channel_content_from_kv_cache=True, **create_kwargs)
    snap = _Snapshot(engine, key, system, conv, _recipe(body))
    _log(f"snapshot: created in {time.monotonic() - started:.1f} s "
         f"({len(_snapshots) + 1} kept)")
  if body.get(_WARM_FIELD):
    # Replayed most recent first: each goes before those already rebuilt.
    # The file already lists them all.
    _snapshots.insert(0, snap)
    return snap
  snap.recipe["saved"] = _today()
  _snapshots.append(snap)
  _save_warm()
  return snap


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
  messages = body.get("messages")
  if not _ENABLED or not isinstance(messages, list) or not messages:
    with engine.create_conversation(
        messages=translated_messages[:-1] if translated_messages else [],
        **create_kwargs,
    ) as conv:
      yield conv, prompt
    return

  global _last_request
  warm = bool(body.get(_WARM_FIELD))
  if not warm:
    _last_request = 0.0
  name_map = name_by_tool_call_id or {}
  key = _config_key(body)
  canonical = [_canonical_message(m, name_map) for m in messages]

  def fresh():
    conv = engine.create_conversation(
        messages=translated_messages[:-1], **create_kwargs)
    return conv, prompt, "a fresh conversation"

  snap = None

  def reclone():
    return _clone(snap.conv), prompt, "a new clone of the snapshot"

  fallbacks = [fresh]
  entry = _match(engine, key, canonical, messages)
  if entry is not None:
    conv = entry.conv
    n = len(entry.history)
    new = translated_messages[n:]
    send = new[0] if len(new) == 1 else new
    _log(f"hit: reusing {n} messages, prefilling {len(new)} new")
  else:
    _evict(_MAX_ENTRIES - 1)  # Room for the new conversation's session.
    conv = None
    try:
      snap = _new_chat(engine, key, canonical, messages, translated_messages,
                       create_kwargs, body)
      if snap is not None:
        conv = _clone(snap.conv)
    except Exception as e:  # pylint: disable=broad-exception-caught
      _log(f"snapshot: failed ({e}); prefilling the system prompt")
    if conv is not None:
      _log("new chat: cloned the system-prompt snapshot")
      fallbacks = [reclone, fresh]
    else:
      conv = engine.create_conversation(
          messages=translated_messages[:-1], **create_kwargs
      )
    send = prompt

  recorder = _RecordingConversation(conv, fallbacks)
  try:
    yield recorder, send
  except BaseException:
    recorder._conv.close()
    raise
  finally:
    if not warm:
      _last_request = time.monotonic()

  # _stream_response swallows errors, so only a generator that ran to its
  # end without cancellation is known to have a KV cache matching `history`.
  # A reply cut off at the token limit also ends "normally"; the next send on
  # it fails and the conversation is closed then.
  # A retry replaced the conversation the request started with.
  conv = recorder._conv
  if recorder.completed and not recorder.cancelled and not warm:
    reply = _canonical_message(recorder.reply(), {})
    _entries.append(_Entry(engine, conv, key, canonical + [reply]))
    _evict(_MAX_ENTRIES)
    _log(f"cached: {len(_entries)}, memory {_memory_mb()} MB")
  else:
    conv.close()


# Only in the server: the CLI imports this module for every command.
if _ENABLED and _MAX_SNAPSHOTS and _WARM_FILE and "serve" in sys.argv[1:]:
  threading.Thread(target=_warm, name="prefix-cache-warm", daemon=True).start()
