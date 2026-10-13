"""Wires hide_tools.py into litert-lm 0.18.0's serve package.

Usage: apply_hide_tools.py <path to litert_lm_cli/commands/serve>

Run at image build, after apply_reminder.py (it anchors on that patch's
call). Exits non-zero if any anchor is missing, so an upstream change fails
the build instead of silently dropping the patch.
"""

import pathlib
import sys

handler = pathlib.Path(sys.argv[1]) / "openai_chat_completions.py"
src = handler.read_text()
# The anchors survive the patch, so a second run would wire it in twice.
if "hide_tools" in src:
  sys.exit(f"apply_hide_tools: {handler} is already patched")

replacements = [
    (
        "from litert_lm_cli.commands.serve import openai_common\n",
        "from litert_lm_cli.commands.serve import openai_common\n"
        "from litert_lm_cli.commands.serve import hide_tools\n",
    ),
    # Before the prefix cache keys the request on its tools.
    (
        "  reminder.apply(body)\n",
        "  reminder.apply(body)\n"
        "  hide_tools.apply(body)\n",
    ),
]

for old, new in replacements:
  if src.count(old) != 1:
    sys.exit(f"apply_hide_tools: anchor not found exactly once:\n{old}")
  src = src.replace(old, new)

handler.write_text(src)
print(f"apply_hide_tools: patched {handler}")
