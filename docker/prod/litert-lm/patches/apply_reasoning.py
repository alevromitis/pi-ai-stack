"""Returns Gemma's thinking as reasoning_content in litert-lm 0.17.1's
openai_handler.py.

Usage: apply_reasoning.py <path to openai_handler.py>

With reasoning_effort set, upstream generates the thought tokens (chunk
"channels") but only counts them, so thinking costs time and shows nothing.
This streams them as delta.reasoning_content (chat completions only) and adds
message.reasoning_content to non-streamed replies - the DeepSeek-style field
LibreChat renders as a collapsible "Thoughts" block.

Run at image build. Exits non-zero if any anchor is missing, so an upstream
change fails the build instead of silently dropping the patch.
"""

import pathlib
import sys

handler = pathlib.Path(sys.argv[1])
src = handler.read_text()

replacements = [
    # Chat-completions formatter: a delta carrying reasoning text. The
    # responses formatter has no such method, so _stream_response skips it.
    (
        "  def format_tool_call_delta(\n"
        "      self, tool_calls: Sequence[Mapping[str, Any]]\n"
        "  ) -> bytes:\n",
        "  def format_reasoning_delta(self, reasoning: str) -> bytes:\n"
        "    \"\"\"Formats a delta chunk with reasoning (thinking) text.\"\"\"\n"
        "    return _sse_data(\n"
        "        _dump_json(\n"
        "            self._make_payload([{\n"
        "                \"index\": 0,\n"
        "                \"delta\": {\"reasoning_content\": reasoning},\n"
        "                \"finish_reason\": None,\n"
        "            }])\n"
        "        )\n"
        "    )\n"
        "\n"
        "  def format_tool_call_delta(\n"
        "      self, tool_calls: Sequence[Mapping[str, Any]]\n"
        "  ) -> bytes:\n",
    ),
    # Streaming: emit each chunk's channel text before its content.
    (
        "        if chunk.get(\"channels\"):\n"
        "          reasoning_tokens += 1\n"
        "        text_output = \"\".join(\n",
        "        if chunk.get(\"channels\"):\n"
        "          reasoning_tokens += 1\n"
        "          reasoning = \"\".join(chunk[\"channels\"].values())\n"
        "          if reasoning and hasattr(formatter, \"format_reasoning_delta\"):\n"
        "            self.wfile.write(formatter.format_reasoning_delta(reasoning))\n"
        "            self.wfile.flush()\n"
        "        text_output = \"\".join(\n",
    ),
    # Non-streaming: collect channel text alongside the content.
    (
        "        if chunk.get(\"channels\"):\n"
        "          reasoning_tokens += 1\n"
        "        for item in chunk.get(\"content\", []):\n",
        "        if chunk.get(\"channels\"):\n"
        "          reasoning_tokens += 1\n"
        "          reasoning_parts.extend(chunk[\"channels\"].values())\n"
        "        for item in chunk.get(\"content\", []):\n",
    ),
    (
        "      text_parts = []\n"
        "      tool_calls = []\n"
        "      reasoning_tokens = 0\n",
        "      text_parts = []\n"
        "      reasoning_parts = []\n"
        "      tool_calls = []\n"
        "      reasoning_tokens = 0\n",
    ),
    (
        "                  \"content\": text_output or None,\n",
        "                  \"content\": text_output or None,\n"
        "                  **(\n"
        "                      {\"reasoning_content\": \"\".join(reasoning_parts)}\n"
        "                      if reasoning_parts\n"
        "                      else {}\n"
        "                  ),\n",
    ),
]

for old, new in replacements:
  if src.count(old) != 1:
    sys.exit(f"apply_reasoning: anchor not found exactly once:\n{old}")
  src = src.replace(old, new)

handler.write_text(src)
print(f"apply_reasoning: patched {handler}")
