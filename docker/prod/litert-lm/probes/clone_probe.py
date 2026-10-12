"""Probe: can the prefix cache clone conversations and keep several alive?

Runs inside a throwaway container from litert-lm-rpi5:local, with the same
engine as `serve` (its own get_or_initialize_server_engine: YNNPACK, MTP and
max_num_tokens from config.json), while litert-lm-1 is stopped (two engines
don't fit in RAM). Use run-clone-probe.sh.

Questions (PLAN-prefix-clone.md):
  1. Does litert_lm_conversation_clone (C API, not wrapped in Python) work
     with this engine (CPU, MTP, constrained decoding), and does a clone
     continue exactly like the original?
  2. Can two conversations stay alive and alternate, and what does each
     cost in RAM and in time when switching back?
  3. Is the system prompt prefilled when a conversation is created? If not,
     is there any way to get a "system prompt only" state to clone for new
     chats?

Sampling is greedy (top_k 1), so equal histories must give equal text.
Prints one line per step and a final JSON summary line.
"""

from __future__ import annotations

import copy
import ctypes
import json
import time
import types

import litert_lm
from litert_lm_cli.commands.serve import openai_chat_completions as occ
from litert_lm_cli.commands.serve import openai_common
from litert_lm_cli.commands.serve import util

MODEL = "gemma4-e2b"
SYSTEM = open("/probe/system.txt").read()
TOOL = {"type": "function", "function": {
    "name": "web_search", "description": "Search the web.",
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string"}},
                   "required": ["query"]}}}
U1 = "Write a short haiku about autumn leaves."
U2 = "Now write one about spring rain."
U3 = "And one about winter snow."
UB = "Write a limerick about a cat who hates Mondays."
MAX_OUT = 120

summary: dict = {}


def log(step: str, **kv) -> None:
  print(f"STEP {step} " + json.dumps(kv, ensure_ascii=False), flush=True)
  summary[step] = kv


def rss_mb() -> dict:
  out = {}
  for line in open("/proc/self/status"):
    if line.startswith(("VmRSS:", "RssAnon:")):
      k, v = line.split(":")
      out[k] = int(v.split()[0]) // 1024
  return out


server = types.SimpleNamespace(litert_lm_engine=None)
t0 = time.time()
engine = util.get_or_initialize_server_engine(server, model_id=MODEL)
log("engine", init_s=round(time.time() - t0, 1), rss=rss_mb(),
    speculative=getattr(server, "speculative_decoding", None))

SAMPLER = occ.parse_sampler_config({"top_k": 1, "temperature": 1.0, "top_p": 1.0, "seed": 0})
THINKING = openai_common.parse_thinking_config({"reasoning_effort": "none"}, model_id=MODEL)
CDC = litert_lm.ConstrainedDecodingConfig(enable=True, provider=None)


def new(messages: list[dict]) -> litert_lm.Conversation:
  return engine.create_conversation(
      messages=messages, tools=[occ._ProxyTool(TOOL)],
      automatic_tool_calling=False, sampler_config=SAMPLER,
      thinking_config=THINKING, constrained_decoding_config=CDC)


def tokens(conv) -> int:
  """Tokens in the conversation's KV cache (a property in this wrapper)."""
  count = conv.token_count
  return count() if callable(count) else count


def send(conv, message, max_out: int = MAX_OUT) -> dict:
  t = time.time()
  first = None
  parts = []
  for chunk in conv.send_message_async(message, max_output_tokens=max_out):
    text = str(chunk)
    if text and first is None:
      first = time.time() - t
    parts.append(text)
  return {"text": "".join(parts).strip(), "ttft_s": round(first or -1, 2),
          "total_s": round(time.time() - t, 2), "tokens": tokens(conv)}


lib = engine._lib
ptr_type = lib.litert_lm_conversation_create.restype
lib.litert_lm_conversation_clone.restype = ptr_type
lib.litert_lm_conversation_clone.argtypes = [ptr_type]


def clone(conv):
  t = time.time()
  ptr = lib.litert_lm_conversation_clone(conv._ptr)
  if not ptr:
    raise RuntimeError("litert_lm_conversation_clone returned NULL")
  twin = copy.copy(conv)  # Same Python-side settings, own native handle.
  twin._ptr = ptr
  twin._current_callback = None
  return twin, round(time.time() - t, 3)


def user(text: str) -> dict:
  return {"role": "user", "content": text}


def assistant(text: str) -> dict:
  return {"role": "assistant", "content": text}


sys_msg = {"role": "system", "content": SYSTEM}

# --- References: fresh conversations, one per history, as `serve` misses do.
ref = {}
c = new([sys_msg]); ref["1"] = send(c, user(U1)); c.close()
c = new([sys_msg, user(U1), assistant(ref["1"]["text"])]); ref["2"] = send(c, user(U2)); c.close()
c = new([sys_msg, user(U1), assistant(ref["1"]["text"]), user(U2),
         assistant(ref["2"]["text"])]); ref["3"] = send(c, user(U3)); c.close()
c = new([sys_msg]); ref["b"] = send(c, user(UB)); c.close()
for k, v in ref.items():
  log(f"ref{k}", ttft_s=v["ttft_s"], tokens=v["tokens"], text=v["text"][:60])

# --- Q3a: is the system prompt prefilled at creation?
c = new([sys_msg])
log("lazy_preface", tokens_after_create=tokens(c),
    note="0 = the system prompt is prefilled only with the first message")
c.close()

# --- Q1: continue, clone, and compare with the references.
a = new([sys_msg])
a1 = send(a, user(U1))
log("a1", same_as_ref=a1["text"] == ref["1"]["text"], ttft_s=a1["ttft_s"], tokens=a1["tokens"])
before = rss_mb()
try:
  a2c, clone_s = clone(a)
except Exception as e:  # The answer to Q1 may simply be "no".
  log("clone", ok=False, error=repr(e))
  print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
  raise SystemExit(0)
log("clone", ok=True, clone_s=clone_s, tokens=tokens(a2c),
    rss_before=before, rss_after=rss_mb())
r = send(a2c, user(U2))
log("clone_continues", same_as_ref=r["text"] == ref["2"]["text"],
    ttft_s=r["ttft_s"], ref_ttft_s=ref["2"]["ttft_s"], tokens=r["tokens"])
r = send(a, user(U2))
log("original_after_clone", same_as_ref=r["text"] == ref["2"]["text"],
    ttft_s=r["ttft_s"], tokens=r["tokens"])

# --- Q2: a second live conversation, then back to the clone.
b = new([sys_msg])
rb = send(b, user(UB))
log("second_live", same_as_ref=rb["text"] == ref["b"]["text"],
    ttft_s=rb["ttft_s"], rss=rss_mb())
r = send(a2c, user(U3))
log("switch_back", same_as_ref=r["text"] == ref["3"]["text"],
    ttft_s=r["ttft_s"], ref_ttft_s=ref["3"]["ttft_s"], tokens=r["tokens"])
extra = []
for _ in range(3):
  extra.append(clone(a2c)[0])
log("three_more_clones", rss=rss_mb())
for x in extra:
  x.close()

# --- Q3b: a "system prompt only" state via a 1-token reply to the system
# message alone. Exact reuse needs the clone to continue like [system, user].
p = new([])
pr = send(p, [sys_msg], max_out=1)
pc, _ = clone(p)
r = send(pc, user(U1))
log("system_only_priming", tokens_after_priming=pr["tokens"],
    primed_reply=pr["text"][:40], same_as_ref=r["text"] == ref["1"]["text"],
    ttft_s=r["ttft_s"], ref_ttft_s=ref["1"]["ttft_s"], text=r["text"][:60])

for x in (a, a2c, b, p, pc):
  x.close()
print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
