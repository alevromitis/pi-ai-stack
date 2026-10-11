"""Wires reminder.py into litert-lm 0.18.0's serve package.

Usage: apply_reminder.py <path to litert_lm_cli/commands/serve>

Run at image build, after apply_memory_gate.py (it anchors on that patch's
call). Exits non-zero if any anchor is missing, so an upstream change fails
the build instead of silently dropping the patch.
"""

import pathlib
import sys

handler = pathlib.Path(sys.argv[1]) / "openai_chat_completions.py"
src = handler.read_text()
# The anchors survive the patch, so a second run would wire it in twice.
if "reminder" in src:
  sys.exit(f"apply_reminder: {handler} is already patched")

replacements = [
    (
        "from litert_lm_cli.commands.serve import openai_common\n",
        "from litert_lm_cli.commands.serve import openai_common\n"
        "from litert_lm_cli.commands.serve import reminder\n",
    ),
    # After the memory gate (its requests carry no reminder), before the
    # messages are translated and compared by the prefix cache, so every
    # request carries the same notes.
    (
        "  if memory_gate.answer(handler, body):\n"
        "    return\n",
        "  if memory_gate.answer(handler, body):\n"
        "    return\n"
        "  reminder.apply(body)\n",
    ),
]

for old, new in replacements:
  if src.count(old) != 1:
    sys.exit(f"apply_reminder: anchor not found exactly once:\n{old}")
  src = src.replace(old, new)

handler.write_text(src)
print(f"apply_reminder: patched {handler}")
