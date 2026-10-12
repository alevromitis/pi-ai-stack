"""Wires prefix_cache.py into litert-lm 0.18.0's serve package.

Usage: apply_prefix_cache.py <path to litert_lm_cli/commands/serve>

Run at image build. Exits non-zero if any anchor is missing, so an upstream
change fails the build instead of silently dropping the patch.
"""

import pathlib
import sys

handler = pathlib.Path(sys.argv[1]) / "openai_chat_completions.py"
src = handler.read_text()

replacements = [
    (
        "from litert_lm_cli.commands.serve import openai_common\n",
        "from litert_lm_cli.commands.serve import openai_common\n"
        "from litert_lm_cli.commands.serve import prefix_cache\n",
    ),
    # LibreChat (like most OpenAI clients) sends max_tokens; upstream only
    # reads max_completion_tokens, so generation was never capped.
    (
        '  max_completion_tokens = body.get("max_completion_tokens")\n',
        '  max_completion_tokens = body.get(\n'
        '      "max_completion_tokens", body.get("max_tokens")\n'
        '  )\n',
    ),
    (
        "    with engine.create_conversation(\n"
        "        messages=context_messages,\n"
        "        tools=tools or None,\n"
        "        automatic_tool_calling=False,\n"
        "        sampler_config=sampler_config,\n"
        "        thinking_config=thinking_config,\n"
        "        constrained_decoding_config=constrained_decoding_config,\n"
        "    ) as conv:\n",
        "    with prefix_cache.conversation(\n"
        "        engine,\n"
        "        body=body,\n"
        "        translated_messages=translated_messages,\n"
        "        prompt=prompt,\n"
        "        name_by_tool_call_id=name_by_tool_call_id,\n"
        "        tools=tools or None,\n"
        "        automatic_tool_calling=False,\n"
        "        sampler_config=sampler_config,\n"
        "        thinking_config=thinking_config,\n"
        "        constrained_decoding_config=constrained_decoding_config,\n"
        "    ) as (conv, prompt):\n",
    ),
    # Record the reply as the client gets it, after every rewrite (Sources
    # line, scrubbed tokens, length note), so its follow-up matches.
    (
        "    handler.headers_sent = True\n",
        '    prefix_cache.sent(conv, resp_body["choices"][0]["message"]["content"])\n'
        "    handler.headers_sent = True\n",
    ),
    (
        "  formatter = _OpenAIChatCompletionsFormatter(\n"
        "      now_str, created_ts, model_id, include_usage=include_usage\n"
        "  )\n",
        "  formatter = prefix_cache.recording(conv, _OpenAIChatCompletionsFormatter(\n"
        "      now_str, created_ts, model_id, include_usage=include_usage\n"
        "  ))\n",
    ),
]

for old, new in replacements:
  if src.count(old) != 1:
    sys.exit(f"apply_prefix_cache: anchor not found exactly once:\n{old}")
  src = src.replace(old, new)

handler.write_text(src)
print(f"apply_prefix_cache: patched {handler}")
