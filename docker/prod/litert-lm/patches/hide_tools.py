"""Drops tools the model never needs from a request.

Not part of upstream LiteRT-LM: copied into litert_lm_cli/commands/serve/ by
the Dockerfile and wired in by apply_hide_tools.py.

Every tool schema is prefilled with the system prompt, and a new chat whose
tools differ from the kept snapshot's rebuilds it (prefix_cache.py, ~24 s),
so useless tools cost time on every rebuild. mcp-searxng and LibreChat can't
hide single tools; searxng's search_suggestions and instance_info only
distract a small model.

Env:
  LITERT_HIDE_TOOLS   comma-separated tool names as LibreChat sends them
                      (empty: hide none)
"""

from __future__ import annotations

import os
from typing import Any

_HIDDEN = frozenset(
    name.strip()
    for name in os.environ.get(
        "LITERT_HIDE_TOOLS",
        "searxng_search_suggestions_mcp_searxng,"
        "searxng_instance_info_mcp_searxng",
    ).split(",")
    if name.strip()
)


def apply(body: dict[str, Any]) -> None:
  tools = body.get("tools")
  if not _HIDDEN or not isinstance(tools, list):
    return
  kept = [
      t for t in tools
      if not (isinstance(t, dict)
              and (t.get("function") or {}).get("name") in _HIDDEN)
  ]
  if len(kept) != len(tools):
    body["tools"] = kept
