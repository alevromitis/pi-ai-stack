"""Says when a reply stopped at max_tokens.

litert-lm 0.18.0 ends every reply with finish_reason "stop", also one cut off
at max_output_tokens, and thought tokens count toward that limit. gemma4-e2b,
asked without tools how many ways make 50 cents, enumerated the combinations
for all 4096 tokens: the thought stopped mid-sentence, the reply was empty,
and the client got "stop", an empty answer that looked finished. A reply
that used the whole limit gets finish_reason "length" and a closing line
that says so.
"""

from typing import Any


def cut(conv: Any, max_tokens: int | None) -> bool:
  """Whether the reply just generated used all of max_tokens."""
  if not max_tokens:
    return False
  try:
    return conv.get_benchmark_info().last_decode_token_count >= max_tokens
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def note(max_tokens: int) -> str:
  return (
      f"\n\n[Cut off: this reply reached its limit of {max_tokens} tokens,"
      " thinking included, before it was finished. Raise max tokens in the"
      " parameters and ask again.]"
  )
