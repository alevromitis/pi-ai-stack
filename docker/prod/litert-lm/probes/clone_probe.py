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

--switching: only question 2, twice (MTP on, then off): two conversations
alternating, clones of a snapshot that is never sent to again, and an
original continued after its clone ran. Errors are logged, not fatal.

--snapshot: new chats cloned from a system-prompt-only snapshot, twice (MTP
on, then off). Needs the image's patches/prefill_preface_option.py, which
makes create_conversation(filter_channel_content_from_kv_cache=True) prefill
the preface at creation. Every chat is compared with a fresh conversation
built the way serve builds a new chat.

--snapshot-control: MTP on only. Separates the two causes the --snapshot run
left open for clones that differed from a fresh conversation: the preface
and the message prefilled in two steps instead of one (numerics), or other
live clones of the same snapshot. For two prompts it compares a fresh
conversation, one created with its preface prefilled and sent to directly,
a clone with no other clone alive, and a clone while two others are alive.
"""

from __future__ import annotations

import copy
import ctypes
import json
import sys
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
UT = "What is the latest news about the James Webb Space Telescope?"
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


def new(messages: list[dict], prefill_preface: bool = False) -> litert_lm.Conversation:
  # With patches/prefill_preface_option.py, this argument sets
  # prefill_preface_on_init instead.
  extra = {"filter_channel_content_from_kv_cache": True} if prefill_preface else {}
  return engine.create_conversation(
      messages=messages, tools=[occ._ProxyTool(TOOL)],
      automatic_tool_calling=False, sampler_config=SAMPLER,
      thinking_config=THINKING, constrained_decoding_config=CDC, **extra)


def tokens(conv) -> int:
  """Tokens in the conversation's KV cache (a property in this wrapper)."""
  count = conv.token_count
  return count() if callable(count) else count


def send(conv, message, max_out: int = MAX_OUT) -> dict:
  t = time.time()
  first = None
  parts = []
  calls = []
  for chunk in conv.send_message_async(message, max_output_tokens=max_out):
    text = str(chunk)
    if (text or chunk.tool_calls) and first is None:
      first = time.time() - t
    parts.append(text)
    calls.extend(f"{tc.name}({json.dumps(tc.arguments, sort_keys=True)})"
                 for tc in chunk.tool_calls)
  return {"text": "".join(parts).strip(), "calls": calls,
          "ttft_s": round(first or -1, 2),
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


def first_difference(x: str, y: str) -> dict:
  i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
  return {"at": i, "here": x[max(0, i - 20):i + 40], "there": y[max(0, i - 20):i + 40]}


def switching(label: str) -> None:
  """Question 2 alone, against one conversation that runs U1, U2, U3 with
  nothing in between. Each case is logged, errors included, and the next
  case runs regardless."""
  c = new([sys_msg])
  seq = [send(c, user(u))["text"] for u in (U1, U2, U3)]
  c.close()

  def check(case: str, conv, turn: int) -> None:
    try:
      text = send(conv, user((U1, U2, U3)[turn - 1]))["text"]
    except Exception as e:  # pylint: disable=broad-exception-caught
      log(f"{label}_{case}_turn{turn}", error=str(e).splitlines()[-1][:200])
      return
    same = text == seq[turn - 1]
    log(f"{label}_{case}_turn{turn}", same_as_sequential=same,
        **({} if same else first_difference(text, seq[turn - 1])))

  # (a) Two conversations alternate; no clones.
  a = new([sys_msg])
  send(a, user(U1))
  b = new([sys_msg])
  send(b, user(UB))
  check("alternating", a, 2)
  send(b, user("Now one about a dog."))
  check("alternating", a, 3)
  for conv in (a, b):
    conv.close()

  # (b) Clones of a base that is never sent to again: what a snapshot
  # cache would do.
  base = new([sys_msg])
  send(base, user(U1))
  c1, _ = clone(base)
  check("snapshot_clone1", c1, 2)
  check("snapshot_clone1", c1, 3)
  c2, _ = clone(base)
  check("snapshot_clone2", c2, 2)

  # (c) The original continued after its clone ran (failed with MTP on in
  # the third run).
  check("original_after_clone", base, 2)
  for conv in (base, c1, c2):
    conv.close()


def snapshot(label: str) -> None:
  """New chats cloned from one system-prompt snapshot that is never sent to,
  against fresh conversations (the preface prefilled with the first
  message, as serve's misses do). Errors are logged, not fatal."""
  c = new([sys_msg])
  seq = [send(c, user(u)) for u in (U1, U2, U3)]
  c.close()
  ref = {}
  for k, u in (("b", UB), ("t", UT)):
    c = new([sys_msg])
    ref[k] = send(c, user(u))
    c.close()
  log(f"{label}_refs", ttft_s=[seq[0]["ttft_s"], ref["b"]["ttft_s"], ref["t"]["ttft_s"]],
      tokens=[s["tokens"] for s in seq], t_calls=ref["t"]["calls"],
      t_text=ref["t"]["text"][:60])

  def check(case: str, conv, message: str, expected: dict) -> None:
    try:
      r = send(conv, user(message))
    except Exception as e:  # pylint: disable=broad-exception-caught
      log(f"{label}_{case}", error=str(e).splitlines()[-1][:200])
      return
    same = r["text"] == expected["text"] and r["calls"] == expected["calls"]
    log(f"{label}_{case}", same=same, tokens=r["tokens"],
        ref_tokens=expected["tokens"], ttft_s=r["ttft_s"],
        ref_ttft_s=expected["ttft_s"],
        **({} if same else {"calls": r["calls"], "ref_calls": expected["calls"],
                            **first_difference(r["text"], expected["text"])}))

  # Prefilled at creation, sent to directly: does the split (preface, then
  # the message) tokenize and decode like one prefill?
  t = time.time()
  d = new([sys_msg], prefill_preface=True)
  log(f"{label}_direct_create", create_s=round(time.time() - t, 2), tokens=tokens(d))
  check("direct_turn1", d, U1, seq[0])
  d.close()

  t = time.time()
  snap = new([sys_msg], prefill_preface=True)
  log(f"{label}_snapshot", create_s=round(time.time() - t, 2),
      tokens=tokens(snap), rss=rss_mb())
  c1, s1 = clone(snap)
  log(f"{label}_clone1", clone_s=s1, tokens=tokens(c1), rss=rss_mb())
  check("clone1_turn1", c1, U1, seq[0])
  c2, _ = clone(snap)
  check("clone2_turn1", c2, UB, ref["b"])
  check("clone1_turn2", c1, U2, seq[1])
  c3, _ = clone(snap)
  check("clone3_tool_turn1", c3, UT, ref["t"])
  check("clone1_turn3", c1, U3, seq[2])
  # The snapshot after its clones grew and closed: still a clean start?
  for conv in (c1, c2, c3):
    conv.close()
  c4, _ = clone(snap)
  check("clone4_after_close_turn1", c4, U1, seq[0])
  c4.close()
  snap.close()
  log(f"{label}_end", rss=rss_mb())


def snapshot_control() -> None:
  out: dict = {}

  def run(case: str, conv, message: str) -> None:
    try:
      out[case] = send(conv, user(message))
    except Exception as e:  # pylint: disable=broad-exception-caught
      out[case] = {"text": "", "calls": [], "error": str(e).splitlines()[-1][:200]}
    r = out[case]
    log(case, text=r["text"][:70], calls=r["calls"], ttft_s=r.get("ttft_s"),
        **({"error": r["error"]} if "error" in r else {}))

  prompts = (("b", UB), ("t", UT))
  for k, u in prompts:
    c = new([sys_msg])
    run(f"fresh_{k}", c, u)
    c.close()
    c = new([sys_msg], prefill_preface=True)
    run(f"direct_{k}", c, u)
    c.close()
  snap = new([sys_msg], prefill_preface=True)
  for k, u in prompts:
    c, _ = clone(snap)
    run(f"clone_alone_{k}", c, u)
    c.close()
  alive = []
  c, _ = clone(snap)
  run("sibling_u1", c, U1)
  alive.append(c)
  for k, u in prompts:
    c, _ = clone(snap)
    run(f"clone_with_siblings_{k}", c, u)
    alive.append(c)
  for c in alive + [snap]:
    c.close()

  def same(x: str, y: str) -> bool:
    return (out[x]["text"], out[x]["calls"]) == (out[y]["text"], out[y]["calls"])

  for k, _ in prompts:
    log(f"verdict_{k}",
        direct_vs_fresh=same(f"direct_{k}", f"fresh_{k}"),
        clone_alone_vs_direct=same(f"clone_alone_{k}", f"direct_{k}"),
        siblings_vs_clone_alone=same(f"clone_with_siblings_{k}", f"clone_alone_{k}"))


def restart_without_mtp() -> None:
  global engine, lib
  engine.close()
  server.litert_lm_engine = None
  _resolve = util.model.resolve_config_option
  util.model.resolve_config_option = lambda value, m, key, *args, **kwargs: (
      False if key == "speculative_decoding"
      else _resolve(value, m, key, *args, **kwargs))
  engine = util.get_or_initialize_server_engine(server, model_id=MODEL)
  log("engine_mtp_off", speculative=getattr(engine, "enable_speculative_decoding", None))
  lib = engine._lib
  lib.litert_lm_conversation_clone.restype = ptr_type
  lib.litert_lm_conversation_clone.argtypes = [ptr_type]


if "--snapshot-control" in sys.argv:
  snapshot_control()
  print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
  raise SystemExit(0)

for flag, check_fn in (("--switching", switching), ("--snapshot", snapshot)):
  if flag in sys.argv:
    check_fn("mtp_on")
    restart_without_mtp()
    check_fn("mtp_off")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    raise SystemExit(0)

# --- References: fresh conversations, one per history, as `serve` misses do.
ref = {}
c = new([sys_msg]); ref["1"] = send(c, user(U1)); c.close()
c = new([sys_msg, user(U1), assistant(ref["1"]["text"])]); ref["2"] = send(c, user(U2)); c.close()
c = new([sys_msg, user(U1), assistant(ref["1"]["text"]), user(U2),
         assistant(ref["2"]["text"])]); ref["3"] = send(c, user(U3)); c.close()
c = new([sys_msg]); ref["b"] = send(c, user(UB)); c.close()
for k, v in ref.items():
  log(f"ref{k}", ttft_s=v["ttft_s"], tokens=v["tokens"], text=v["text"][:60])
# The references rebuild each history from text, and a reply re-tokenizes
# slightly differently from how it was generated. So a conversation that
# continues its own KV can drift from them after a few turns; this control
# is one conversation sending U1, U2, U3 with nothing else in between.
c = new([sys_msg])
seq = [send(c, user(u)) for u in (U1, U2, U3)]
c.close()
log("sequential_control", same_as_refs=[s["text"] == ref[k]["text"]
                                        for s, k in zip(seq, "123")],
    tokens=[s["tokens"] for s in seq])

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
    same_as_sequential=r["text"] == seq[2]["text"],
    ttft_s=r["ttft_s"], ref_ttft_s=ref["3"]["ttft_s"], tokens=r["tokens"])
r = send(a, user(U3))
log("original_switch_back", same_as_sequential=r["text"] == seq[2]["text"],
    ttft_s=r["ttft_s"], tokens=r["tokens"])
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
