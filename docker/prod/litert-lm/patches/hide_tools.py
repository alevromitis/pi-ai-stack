"""Drops tools the model never needs from a request and shortens tool text.

Not part of upstream LiteRT-LM: copied into litert_lm_cli/commands/serve/ by
the Dockerfile and wired in by apply_hide_tools.py (and bind-mounted over the
image's copy by docker-compose.litert-lm.yml, so a change needs no rebuild).

Every tool schema is prefilled with the system prompt, and a new chat whose
tools differ from the kept snapshot's rebuilds it (prefix_cache.py, ~24 s),
so tool text costs time on every rebuild, and a small model follows a short,
direct description better than a manual. mcp-searxng and LibreChat can't hide
single tools or change their text; searxng's search_suggestions and
instance_info only distract a small model.

_TEXT replaces the descriptions of tools whose text we don't write:
LibreChat's built-ins (file_search, memory, the skill tools) and mcp-searxng.
The originals, measured 2026-11-13 on the NTUA Instagram agent, were 8.0k of
the request's 11.7k characters of tool schemas: the four skill tools alone
6.3k, mostly about features off here (code execution, renaming skills) and
SKILL.md style advice. web_url_read's (~70 tokens about FlareSolverr/Byparr,
not run here) is quoted in agent-tools/README.md. Only text changes: names,
parameters, types and required lists stay as LibreChat sends them, so a call
the model makes is still one LibreChat accepts.

Env:
  LITERT_HIDE_TOOLS   comma-separated tool names as LibreChat sends them
                      (empty: hide none)
  LITERT_TOOL_DESCRIPTIONS
                      JSON replacing _TEXT: {tool name: description} or
                      {tool name: {"description": ..., "params": {param:
                      description}}}; "" drops a description, {} changes none
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

# Each tool: what it does in one line, and a parameter's description only
# where its name and type don't already say it ("" drops one).
_DEFAULT_TEXT: dict[str, Any] = {
    "web_url_read_mcp_searxng": {
        "description": "Read a web page or file by URL, as markdown.",
        "params": {"url": ""},
    },
    "searxng_web_search_mcp_searxng": {
        "description": "Web search. Returns titles, URLs and snippets.",
        "params": {"query": ""},
    },
    "file_search": {
        "description": "Search the attached documents. Returns matching passages and their file names.",
        "params": {"query": "Keywords or a question"},
    },
    "set_memory": {
        "description": "Save a fact for later chats. Saving under an existing key replaces it.",
        "params": {"key": "Short name, e.g. exam_date", "value": "The fact, as one full sentence"},
    },
    "delete_memory": {
        "description": "Forget a saved fact.",
        "params": {"key": "Its key"},
    },
    "skill": {
        "description": "Load a skill's full instructions. Only a name listed under Available Skills or "
                       "created in this chat, and not one already active.",
        "params": {"skillName": "Skill name, kebab-case", "args": "Optional arguments"},
    },
    "read_file": {
        "description": "Read a skill file, with line numbers.",
        "params": {"path": "skills/{skill}/{file}, e.g. skills/{skill}/SKILL.md"},
    },
    "create_file": {
        "description": "Create a skill file. A skill's SKILL.md holds its instructions and starts with "
                       "YAML frontmatter whose name is the skill's name.",
        "params": {
            "path": "skills/{skill}/{file}",
            "content": "The whole file",
            "overwrite": "true to replace an existing file",
        },
    },
    "edit_file": {
        "description": "Replace text in an existing skill file. Returns a diff.",
        "params": {
            "path": "skills/{skill}/{file}",
            "old_text": "Exact text; must match one place unless replace_all",
            "new_text": "",
            "replace_all": "Replace every match",
            "edits": "Several replacements at once",
        },
    },
}

_TEXT: dict[str, Any] = (
    json.loads(os.environ["LITERT_TOOL_DESCRIPTIONS"] or "{}")
    if "LITERT_TOOL_DESCRIPTIONS" in os.environ else _DEFAULT_TEXT
)


def _name(tool: Any) -> str | None:
  return (tool.get("function") or {}).get("name") if isinstance(tool, dict) else None


def _rewrite(function: dict[str, Any], text: Any) -> dict[str, Any]:
  if isinstance(text, str):
    text = {"description": text}
  function = dict(function)
  if "description" in text:
    function.pop("description", None)
    if text["description"]:
      function["description"] = text["description"]
  params = text.get("params") or {}
  properties = (function.get("parameters") or {}).get("properties")
  if params and isinstance(properties, dict):
    properties = dict(properties)
    for param, description in params.items():
      if not isinstance(properties.get(param), dict):
        continue  # LibreChat no longer sends it: nothing to change
      prop = {k: v for k, v in properties[param].items() if k != "description"}
      if description:
        prop["description"] = description
      properties[param] = prop
    function["parameters"] = dict(function["parameters"], properties=properties)
  return function


def apply(body: dict[str, Any]) -> None:
  tools = body.get("tools")
  if not isinstance(tools, list):
    return
  kept = []
  for t in tools:
    name = _name(t)
    if name in _HIDDEN:
      continue
    if name in _TEXT:
      t = dict(t, function=_rewrite(t["function"], _TEXT[name]))
    kept.append(t)
  body["tools"] = kept
