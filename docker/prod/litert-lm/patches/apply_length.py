"""Wires length.py into litert-lm 0.18.0's serve package.

Usage: apply_length.py <path to litert_lm_cli/commands/serve>

Run at image build, after apply_citations.py (its Sources line comes before
the cut-off note). Exits non-zero if any anchor is missing, so an upstream
change fails the build instead of silently dropping the patch.
"""

import pathlib
import sys

serve = pathlib.Path(sys.argv[1])

_IMPORT = (
    "from litert_lm_cli.commands.serve import openai_common\n",
    "from litert_lm_cli.commands.serve import length\n"
    "from litert_lm_cli.commands.serve import openai_common\n",
)

replacements = {
    "openai_chat_completions.py": [
        _IMPORT,
        # Non-streaming reply.
        (
            "    resp_body = {\n",
            "    finish_reason = \"tool_calls\" if openai_tool_calls else \"stop\"\n"
            "    if not openai_tool_calls and length.cut(conv, max_completion_tokens):\n"
            "      finish_reason = \"length\"\n"
            "      text_output = (\n"
            "          (text_output or \"\") + length.note(max_completion_tokens)\n"
            "      ).strip()\n"
            "    resp_body = {\n",
        ),
        (
            "            \"finish_reason\": \"tool_calls\" if openai_tool_calls else \"stop\",\n",
            "            \"finish_reason\": finish_reason,\n",
        ),
    ],
    "openai_handler.py": [
        _IMPORT,
        # Streaming reply.
        (
            "      finish_reason = \"tool_calls\" if has_tool_calls else \"stop\"\n",
            "      finish_reason = \"tool_calls\" if has_tool_calls else \"stop\"\n"
            "      if not has_tool_calls and length.cut(conv, max_completion_tokens):\n"
            "        finish_reason = \"length\"\n"
            "        self.wfile.write(\n"
            "            formatter.format_delta(length.note(max_completion_tokens))\n"
            "        )\n"
            "        self.wfile.flush()\n",
        ),
    ],
}

for name, file_replacements in replacements.items():
  path = serve / name
  src = path.read_text()
  # The anchors survive the patch, so a second run would wire it in twice.
  if "import length" in src:
    sys.exit(f"apply_length: {path} is already patched")
  for old, new in file_replacements:
    if src.count(old) != 1:
      sys.exit(f"apply_length: anchor not found exactly once in {name}:\n{old}")
    src = src.replace(old, new)
  path.write_text(src)
  print(f"apply_length: patched {path}")
