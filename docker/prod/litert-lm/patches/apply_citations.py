"""Wires citations.py into litert-lm 0.18.0's serve package.

Usage: apply_citations.py <path to litert_lm_cli/commands/serve>

Run at image build, after apply_reasoning.py (it anchors on that patch's
streaming code). Exits non-zero if any anchor is missing, so an upstream
change fails the build instead of silently dropping the patch.
"""

import pathlib
import sys

serve = pathlib.Path(sys.argv[1])

replacements = {
    "openai_chat_completions.py": [
        (
            "from litert_lm_cli.commands.serve import openai_common\n",
            "from litert_lm_cli.commands.serve import citations\n"
            "from litert_lm_cli.commands.serve import openai_common\n",
        ),
        # Set on every chat request (None without search results), so a
        # kept-alive connection never reuses the previous request's sources.
        (
            "  body = handler.parse_request_body()\n"
            "  if body is None:\n"
            "    return\n",
            "  body = handler.parse_request_body()\n"
            "  if body is None:\n"
            "    return\n"
            "  handler.citation_sources = citations.collect(body)\n",
        ),
        # Non-streaming reply.
        (
            "    text_output = \"\".join(text_parts)\n",
            "    text_output = citations.apply(\n"
            "        handler, \"\".join(text_parts), bool(tool_calls)\n"
            "    )\n",
        ),
    ],
    "openai_handler.py": [
        (
            "from litert_lm_cli.commands.serve import openai_chat_completions\n",
            "from litert_lm_cli.commands.serve import citations\n"
            "from litert_lm_cli.commands.serve import openai_chat_completions\n",
        ),
        (
            "      has_tool_calls = False\n"
            "      reasoning_tokens = 0\n",
            "      has_tool_calls = False\n"
            "      reasoning_tokens = 0\n"
            "      cite = citations.stream_filter(self)\n",
        ),
        (
            "        text_output = str(chunk)\n"
            "        if text_output:\n",
            "        text_output = str(chunk)\n"
            "        if cite is not None:\n"
            "          text_output = cite.feed(text_output)\n"
            "        if text_output:\n",
        ),
        (
            "      finish_reason = \"tool_calls\" if has_tool_calls else \"stop\"\n",
            "      if cite is not None:\n"
            "        tail = cite.finish(has_tool_calls)\n"
            "        if tail:\n"
            "          self.wfile.write(formatter.format_delta(tail))\n"
            "          self.wfile.flush()\n"
            "      finish_reason = \"tool_calls\" if has_tool_calls else \"stop\"\n",
        ),
    ],
}

for name, file_replacements in replacements.items():
  path = serve / name
  src = path.read_text()
  # The anchors survive the patch, so a second run would wire it in twice.
  if "citations" in src:
    sys.exit(f"apply_citations: {path} is already patched")
  for old, new in file_replacements:
    if src.count(old) != 1:
      sys.exit(f"apply_citations: anchor not found exactly once in {name}:\n{old}")
    src = src.replace(old, new)
  path.write_text(src)
  print(f"apply_citations: patched {path}")
