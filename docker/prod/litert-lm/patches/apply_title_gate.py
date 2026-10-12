"""Wires title_gate.py into litert-lm 0.18.0's serve package.

Usage: apply_title_gate.py <path to litert_lm_cli/commands/serve>

Run at image build. Exits non-zero if any anchor is missing, so an upstream
change fails the build instead of silently dropping the patch.
"""

import pathlib
import sys

handler = pathlib.Path(sys.argv[1]) / "openai_chat_completions.py"
src = handler.read_text()
# The anchors survive the patch, so a second run would wire it in twice.
if "title_gate" in src:
  sys.exit(f"apply_title_gate: {handler} is already patched")

replacements = [
    (
        "from litert_lm_cli.commands.serve import openai_common\n",
        "from litert_lm_cli.commands.serve import openai_common\n"
        "from litert_lm_cli.commands.serve import title_gate\n",
    ),
    # Before the engine is looked up, so a title request never touches it
    # or the prefix cache.
    (
        "  body = handler.parse_request_body()\n"
        "  if body is None:\n"
        "    return\n",
        "  body = handler.parse_request_body()\n"
        "  if body is None:\n"
        "    return\n"
        "  if title_gate.answer(handler, body):\n"
        "    return\n",
    ),
]

for old, new in replacements:
  if src.count(old) != 1:
    sys.exit(f"apply_title_gate: anchor not found exactly once:\n{old}")
  src = src.replace(old, new)

handler.write_text(src)
print(f"apply_title_gate: patched {handler}")
