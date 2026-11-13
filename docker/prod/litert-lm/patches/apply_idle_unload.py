"""Wires idle_unload.py into litert-lm 0.18.0's serve package.

Usage: apply_idle_unload.py <path to litert_lm_cli/commands/serve>

Run at image build. Exits non-zero if an anchor is missing, so an upstream
change fails the build instead of silently dropping the patch.
"""

import pathlib
import sys

util = pathlib.Path(sys.argv[1]) / "util.py"
src = util.read_text()
if "idle_unload" in src:
  sys.exit(f"apply_idle_unload: {util} is already patched")

# idle_unload rebinds these and wraps these server methods by name.
for anchor in (
    "class LiteRTLMServer(http.server.HTTPServer):\n",
    "def get_or_initialize_server_engine(\n",
    "def get_or_initialize_server_embedding_engine(\n",
    "    self.litert_lm_embedding_engine: litert_lm.EmbeddingEngine | None = None\n",
):
  if src.count(anchor) != 1:
    sys.exit(f"apply_idle_unload: anchor not found exactly once:\n{anchor}")

src += (
    "\n\n# On-demand loading: close the engines after LITERT_IDLE_UNLOAD_S idle"
    " (patches/idle_unload.py).\n"
    "from litert_lm_cli.commands.serve import idle_unload as _idle_unload"
    "  # pylint: disable=g-import-not-at-top\n"
    "_idle_unload.install(sys.modules[__name__])\n"
)
if "\nimport sys\n" not in src:
  src = src.replace("import socket\n", "import socket\nimport sys\n", 1)
util.write_text(src)
print(f"apply_idle_unload: patched {util}")
