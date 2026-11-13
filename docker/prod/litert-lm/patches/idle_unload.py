"""Frees LiteRT-LM's engines after a stretch without requests (on-demand loading).

Upstream loads an engine on the first request that needs it and keeps it
until the server exits: litert-lm-1 held ~3.4 GB around the clock. With
LITERT_IDLE_UNLOAD_S > 0, once no request has used an engine for that many
seconds, the chat engine and the embedding engine are closed, along with the
prefix cache's conversations and snapshots (they belong to the engine), and
the freed heap is handed back to the kernel. The next request loads the
engine again through upstream's get_or_initialize_* (E2B: ~36 s to the first
token on the Pi; a snapshot costs another 12-65 s to rebuild).

  LITERT_IDLE_UNLOAD_S=600   seconds idle before unloading; 0 or unset = never

No race with requests: LiteRTLMServer is a single-threaded
http.server.HTTPServer, and serve_forever() calls service_actions() between
requests (every poll_interval, 0.5 s), which is where the unload runs.
Requests that never touch an engine (title and memory gates, /v1/models,
health checks) don't count as use, so polling can't keep a model loaded.
"""

from __future__ import annotations

import ctypes
import gc
import os
import sys
import time

_MEMORY_CURRENT = "/sys/fs/cgroup/memory.current"
_PREFIX_CACHE = "litert_lm_cli.commands.serve.prefix_cache"


def idle_seconds() -> float:
  try:
    return max(0.0, float(os.environ.get("LITERT_IDLE_UNLOAD_S", "0") or 0))
  except ValueError:
    return 0.0


def _log(text: str) -> None:
  print(f"idle-unload: {text}", file=sys.stderr, flush=True)


def _memory_mb() -> int | None:
  try:
    with open(_MEMORY_CURRENT) as f:
      return int(f.read()) // 2**20
  except (OSError, ValueError):
    return None


def _close_prefix_cache() -> int:
  """Closes the prefix cache's conversations and snapshots; returns how many."""
  pc = sys.modules.get(_PREFIX_CACHE)
  if pc is None:
    return 0
  closed = 0
  while pc._entries:  # pylint: disable=protected-access
    pc._entries.pop().conv.close()  # pylint: disable=protected-access
    closed += 1
  while pc._snapshots:  # pylint: disable=protected-access
    pc._close_snapshot("idle unload")  # pylint: disable=protected-access
    closed += 1
  return closed


def unload(server, why: str) -> bool:
  """Closes every loaded engine on `server`. True if anything was loaded."""
  if server.litert_lm_engine is None and server.litert_lm_embedding_engine is None:
    return False
  before = _memory_mb()
  closed = _close_prefix_cache()
  if server.litert_lm_engine is not None:
    server.litert_lm_engine.__exit__(None, None, None)
    server.litert_lm_engine = None
    server.model_id = None
    server.backend = None
    server.max_num_tokens = None
    server.vision_backend = None
    server.audio_backend = None
    server.activation_data_type = None
  if server.litert_lm_embedding_engine is not None:
    server.litert_lm_embedding_engine.close()
    server.litert_lm_embedding_engine = None
    server.embedding_model_id = None
    server.embedding_backend = None
    server.embedding_vision_backend = None
    server.embedding_audio_backend = None
  gc.collect()
  try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
  except (OSError, AttributeError):
    pass
  _log(f"engine unloaded ({why}; {closed} cached conversations closed); "
       f"memory {before} -> {_memory_mb()} MB")
  return True


def install(util) -> None:
  """Wires the idle unload into litert_lm_cli.commands.serve.util."""
  limit = idle_seconds()
  if limit <= 0:
    return
  get_engine = util.get_or_initialize_server_engine
  get_embedding = util.get_or_initialize_server_embedding_engine

  # The handlers call these through the util module, so rebinding the names
  # catches every request that needs an engine.
  def get_or_initialize_server_engine(server, *args, **kwargs):
    server.idle_unload_used = True
    return get_engine(server, *args, **kwargs)

  def get_or_initialize_server_embedding_engine(server, *args, **kwargs):
    server.idle_unload_used = True
    return get_embedding(server, *args, **kwargs)

  util.get_or_initialize_server_engine = get_or_initialize_server_engine
  util.get_or_initialize_server_embedding_engine = (
      get_or_initialize_server_embedding_engine)

  cls = util.LiteRTLMServer
  process_request = cls.process_request

  def patched_process_request(self, request, client_address):
    try:
      process_request(self, request, client_address)
    finally:
      # Counted from the end of the request: a long generation is use.
      if getattr(self, "idle_unload_used", False):
        self.idle_unload_used = False
        self.idle_unload_last = time.monotonic()

  def service_actions(self):
    last = getattr(self, "idle_unload_last", None)
    if last is not None and time.monotonic() - last >= limit:
      self.idle_unload_last = None
      unload(self, f"idle {limit:.0f} s")

  cls.process_request = patched_process_request
  cls.service_actions = service_actions
  _log(f"engines unload after {limit:.0f} s without use")
