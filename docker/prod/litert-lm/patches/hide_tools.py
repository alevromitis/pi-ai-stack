"""Drops tools the model never needs from a request and shortens descriptions.

Not part of upstream LiteRT-LM: copied into litert_lm_cli/commands/serve/ by
the Dockerfile and wired in by apply_hide_tools.py.

Every tool schema is prefilled with the system prompt, and a new chat whose
tools differ from the kept snapshot's rebuilds it (prefix_cache.py, ~24 s),
so useless tools cost time on every rebuild. mcp-searxng and LibreChat can't
hide single tools; searxng's search_suggestions and instance_info only
distract a small model. web_url_read's description (~70 tokens, mostly
about FlareSolverr/Byparr, which aren't run here; quoted in agent-tools/README.md)
is replaced with one short line.

Env:
  LITERT_HIDE_TOOLS   comma-separated tool names as LibreChat sends them
                      (empty: hide none)
  LITERT_TOOL_DESCRIPTIONS
                      JSON {tool name: description} replacing the tool's
                      own ("" drops it; {} changes none)
"""

from __future__ import annotations

import json
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

_DESCRIPTIONS: dict[str, str] = json.loads(os.environ.get(
    "LITERT_TOOL_DESCRIPTIONS",
    '{"web_url_read_mcp_searxng": "Read a web page or file by URL, as markdown."}',
) or "{}")


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
    if name in _DESCRIPTIONS:
      function = {k: v for k, v in t["function"].items() if k != "description"}
      if _DESCRIPTIONS[name]:
        function["description"] = _DESCRIPTIONS[name]
      t = dict(t, function=function)
    kept.append(t)
  body["tools"] = kept
