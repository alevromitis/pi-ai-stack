"""Probe: which conversation between snapshot-cloned chats breaks MTP's drafter?

Runs inside a throwaway container from litert-lm-rpi5:local, with the same
engine as `serve` (YNNPACK, MTP, 16k context), while litert-lm-1 is stopped.
Use run-mtp-interleave-probe.sh.

On 2026-10-12 a test server that also ran LibreChat's memory requests (their
own conversations, other tools, thinking off) failed on about 1 in 3 chat
requests that came right after one, in the drafter:
"llm_litert_mtp_drafter.cc:741 ... Cannot auto-resize tensor kv_cache_k_N:
no dims_signature exists". The chats were clones of a system-prompt snapshot
(prefix_cache.py) with thinking on, like pi-chat's.

Each scenario runs CYCLES cycles of: a new chat cloned from the snapshot, an
"in-between" conversation, the chat's second message, and another new chat.
Scenarios differ only in the in-between conversation, so the error counts
show what triggers it. After an error, a fresh clone is tried at once (would
a retry recover?).

Prints one line per step and a final JSON summary line.
"""

from __future__ import annotations

import copy
import json
import sys
import time
import types

import litert_lm
from litert_lm_cli.commands.serve import openai_chat_completions as occ
from litert_lm_cli.commands.serve import util

MODEL = "gemma4-e2b"
CYCLES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None
SYSTEM = open("/probe/system.txt").read()
CHAT_TOOL = {"type": "function", "function": {
    "name": "wikipedia_search", "description": "Search offline English Wikipedia.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                   "required": ["query"]}}}
MEM_TOOLS = [
    {"type": "function", "function": {
        "name": "set_memory", "description": "Saves important information about the user into memory.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                       "required": ["key", "value"]}}},
    {"type": "function", "function": {
        "name": "delete_memory", "description": "Deletes specific memory data about the user using the provided key.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}}]
MEM_SYSTEM = (
    "Use the `set_memory` tool to save important information about the user, but ONLY when the user"
    " has requested you to remember something.\n\nNEVER store information just because the user"
    " mentioned it in conversation. If the user doesn't ask you to remember or forget something, DO"
    " NOT use any memory tools.\n\n# Memory Status:\nCurrent memory usage: 0 tokens\nToken limit:"
    " 1000 tokens\n\n# Existing memory:\nNo existing memories")
TASKS = ["Write a haiku about rain.", "Reverse the word 'stone' letter by letter.",
         "Give me a two-line toast for a friend's birthday.",
         "Rewrite this more politely: give me the report now.",
         "Write a limerick about a cat who hates Mondays.", "Name a colour that rhymes with 'bed'."]
CHAT_OUT = 48  # The error came at the first drafting step.
MEM_OUT = 8

summary: dict = {}
server = types.SimpleNamespace(litert_lm_engine=None)
t0 = time.time()
engine = util.get_or_initialize_server_engine(server, model_id=MODEL)
print(f"engine ready in {time.time() - t0:.1f} s", flush=True)

SAMPLER = occ.parse_sampler_config({"top_k": 64})
THINK_ON = litert_lm.ThinkingConfig(enable_thinking=True, thinking_token_budget=-1)
THINK_OFF = litert_lm.ThinkingConfig(enable_thinking=False, thinking_token_budget=0)
CDC = litert_lm.ConstrainedDecodingConfig(enable=True, provider=None)

lib = engine._lib
ptr_type = lib.litert_lm_conversation_create.restype
lib.litert_lm_conversation_clone.restype = ptr_type
lib.litert_lm_conversation_clone.argtypes = [ptr_type]


def clone(conv):
  ptr = lib.litert_lm_conversation_clone(conv._ptr)
  if not ptr:
    raise RuntimeError("clone returned NULL")
  twin = copy.copy(conv)
  twin._ptr = ptr
  twin._current_callback = None
  return twin


def create(system, tools, thinking, snapshot=False, mtp=None):
  extra = {}
  if snapshot:  # patches/prefill_preface_option.py: prefill the preface now.
    extra["filter_channel_content_from_kv_cache"] = True
  if mtp is not None:
    extra["enable_speculative_decoding"] = mtp
  return engine.create_conversation(
      messages=[{"role": "system", "content": system}],
      tools=[occ._ProxyTool(t) for t in tools], automatic_tool_calling=False,
      sampler_config=SAMPLER, thinking_config=thinking,
      constrained_decoding_config=CDC, **extra)


def send(conv, text, max_out, cancel_after=None):
  """'ok' or the error's last line."""
  try:
    n = 0
    for chunk in conv.send_message_async(
        {"role": "user", "content": [{"type": "text", "text": text}]},
        max_output_tokens=max_out):
      n += 1
      if cancel_after is not None and n >= cancel_after:
        conv.cancel_process()
    return "ok"
  except Exception as e:  # pylint: disable=broad-exception-caught
    msg = str(e).strip().splitlines()
    return "ERR " + (msg[-1] if msg else repr(e))[:90]


# The in-between conversation of each scenario: () -> (conversation, text, cancel_after).
def mem_fresh(thinking=THINK_OFF, mtp=None, tools=MEM_TOOLS, system=MEM_SYSTEM):
  return lambda: (create(system, tools, thinking, mtp=mtp),
                  "# Current Chat:\n\nHuman: Explain how rainbows form.", 3)


mem_snapshot = {}


def mem_from_snapshot(mtp=False):
  def make():
    if "s" not in mem_snapshot:
      mem_snapshot["s"] = create(MEM_SYSTEM, MEM_TOOLS, THINK_OFF, snapshot=True, mtp=mtp)
    return clone(mem_snapshot["s"]), "# Current Chat:\n\nHuman: Explain how rainbows form.", 3
  return make


SCENARIOS = {
    # name: (in-between factory or None, close it right after?, chat thinking)
    "control": (None, True, THINK_ON),
    "mem-mtp-on": (mem_fresh(), True, THINK_ON),
    "mem-mtp-off": (mem_fresh(mtp=False), True, THINK_ON),
    "mem-kept-alive": (mem_fresh(), False, THINK_ON),
    "chat-tools-think-off": (mem_fresh(tools=[CHAT_TOOL], system=SYSTEM), True, THINK_ON),
    "mem-tools-think-on": (mem_fresh(thinking=THINK_ON), True, THINK_ON),
    "mem-snapshot-mtp-off": (mem_from_snapshot(), True, THINK_ON),
    "chats-think-off": (mem_fresh(), True, THINK_OFF),
}

for name, (between, close_now, chat_thinking) in SCENARIOS.items():
  if ONLY and name not in ONLY:
    continue
  t = time.time()
  snap = create(SYSTEM, [CHAT_TOOL], chat_thinking, snapshot=True)
  mem_snapshot.clear()
  errors, retries_ok, steps, kept = 0, 0, 0, []

  def step(label, conv, text, max_out, cancel_after=None):
    global errors, retries_ok, steps
    res = send(conv, text, max_out, cancel_after)
    steps += 1
    line = f"{name:22s} {label:28s} {res}"
    if res != "ok" and "rainbow" not in label:
      errors += 1
      retry = clone(snap)
      r2 = send(retry, text, max_out)
      retry.close()
      retries_ok += r2 == "ok"
      line += f" | retry on a fresh clone: {r2}"
    print(line, flush=True)
    return res

  for i in range(CYCLES):
    chat = clone(snap)
    step(f"c{i} new chat", chat, TASKS[i % len(TASKS)], CHAT_OUT)
    if between is not None:
      conv, text, cancel_after = between()
      step(f"c{i} in-between (rainbow)", conv, text, MEM_OUT, cancel_after)
      if close_now:
        conv.close()
      else:
        kept.append(conv)
    step(f"c{i} chat turn 2", chat, "Make it shorter.", CHAT_OUT)
    if between is not None:
      conv, text, cancel_after = between()
      step(f"c{i} in-between 2 (rainbow)", conv, text, MEM_OUT, cancel_after)
      if close_now:
        conv.close()
      else:
        kept.append(conv)
    chat.close()
  for conv in kept:
    conv.close()
  if "s" in mem_snapshot:
    mem_snapshot["s"].close()
  snap.close()
  summary[name] = {"chat_errors": errors, "retries_ok": retries_ok,
                   "steps": steps, "s": round(time.time() - t)}
  print(f"== {name}: {summary[name]}", flush=True)

print("SUMMARY " + json.dumps(summary), flush=True)
