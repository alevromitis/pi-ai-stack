"""Wires translation.py into litert-lm 0.18.0's serve package.

Usage: apply_translation.py <path to litert_lm_cli/commands/serve>

Run at image build, after apply_citations.py (it anchors on that patch's
code). Exits non-zero if any anchor is missing, so an upstream change fails
the build instead of silently dropping the patch.
"""

import pathlib
import sys

serve = pathlib.Path(sys.argv[1])

replacements = {
    "openai_chat_completions.py": [
        (
            "from litert_lm_cli.commands.serve import openai_common\n",
            "from litert_lm_cli.commands.serve import openai_common\n"
            "from litert_lm_cli.commands.serve import translation\n",
        ),
        # First, so everything after it reads English.
        (
            "  handler.citation_sources = citations.collect(body)\n",
            "  translation.apply(handler, body)\n"
            "  handler.citation_sources = citations.collect(body)\n",
        ),
        # Non-streaming reply, after its Sources line.
        (
            "        bool(tool_calls),\n"
            "    )\n",
            "        bool(tool_calls),\n"
            "    )\n"
            "    text_output = translation.back(handler, text_output)\n",
        ),
    ],
    "openai_handler.py": [
        (
            "from litert_lm_cli.commands.serve import openai_chat_completions\n",
            "from litert_lm_cli.commands.serve import openai_chat_completions\n"
            "from litert_lm_cli.commands.serve import translation\n",
        ),
        (
            "      scrub = citations.Scrub()\n",
            "      scrub = citations.Scrub()\n"
            "      back = translation.stream_filter(self)\n",
        ),
        (
            "          text_output = cite.feed(text_output)\n",
            "          text_output = cite.feed(text_output)\n"
            "        if back is not None:\n"
            "          text_output = back.feed(text_output)\n",
        ),
        # The held reply, Sources line included, goes out translated.
        (
            "      if cite is not None:\n"
            "        tail = cite.finish(has_tool_calls)\n"
            "        if tail:\n"
            "          self.wfile.write(formatter.format_delta(tail))\n"
            "          self.wfile.flush()\n",
            "      tail = cite.finish(has_tool_calls) if cite is not None else \"\"\n"
            "      if back is not None:\n"
            "        tail = back.feed(tail) + back.finish()\n"
            "      if tail:\n"
            "        self.wfile.write(formatter.format_delta(tail))\n"
            "        self.wfile.flush()\n",
        ),
    ],
}

for name, file_replacements in replacements.items():
  path = serve / name
  src = path.read_text()
  # The anchors survive the patch, so a second run would wire it in twice.
  if "import translation" in src:
    sys.exit(f"apply_translation: {path} is already patched")
  for old, new in file_replacements:
    if src.count(old) != 1:
      sys.exit(f"apply_translation: anchor not found exactly once in {name}:\n{old}")
    src = src.replace(old, new)
  path.write_text(src)
  print(f"apply_translation: patched {path}")
