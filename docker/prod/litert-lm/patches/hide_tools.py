"""Drops tools, and tool descriptions, the model never needs from a request.

Not part of upstream LiteRT-LM: copied into litert_lm_cli/commands/serve/ by
the Dockerfile and wired in by apply_hide_tools.py.

Every tool schema is prefilled with the system prompt, and a new chat whose
tools differ from the kept snapshot's rebuilds it (prefix_cache.py, ~24 s),
so useless tools cost time on every rebuild. mcp-searxng and LibreChat can't
hide single tools; searxng's search_suggestions and instance_info only
distract a small model. web_url_read's description (~70 tokens, mostly
about FlareSolverr/Byparr, which aren't run here; quoted in the README)
says nothing its name and `url` parameter don't.

Env:
  LITERT_HIDE_TOOLS   comma-separated tool names as LibreChat sends them
                      (empty: hide none)
  LITERT_HIDE_TOOL_DESCRIPTIONS
                      the same, for tools sent without their description
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

_NO_DESCRIPTION = frozenset(
    name.strip()
    for name in os.environ.get(
        "LITERT_HIDE_TOOL_DESCRIPTIONS", "web_url_read_mcp_searxng",
    ).split(",")
    if name.strip()
)


def _name(tool: Any) -> str | None:
  return (tool.get("function") or {}).get("name") if isinstance(tool, dict) else None


def apply(body: dict[str, Any]) -> None:
  tools = body.get("tools")
  if not isinstance(tools, list):
    return
  kept = []
  for t in tools:
    name = _name(t)
    if name in _HIDDEN:
      continue
    if name in _NO_DESCRIPTION and "description" in t["function"]:
      t = dict(t, function={k: v for k, v in t["function"].items()
                            if k != "description"})
    kept.append(t)
  body["tools"] = kept
