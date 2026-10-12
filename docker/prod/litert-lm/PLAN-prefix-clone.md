# Plan: cloned conversations for the prefix cache

Status: **probe drafted, 2026-10-12.** Results below once it has run.

## Why

Time to first token on the Pi is almost all prefill, at ~75 tok/s (16k
context, MTP doesn't help prefill). `patches/prefix_cache.py` keeps **one**
conversation, so:

| Situation | Prefilled | First token |
|---|---|---|
| New chat | system prompt + first message, ~1,000 tokens | 13–15 s |
| Second message of a chat | everything, because LibreChat's title request evicted the chat | ~17–35 s; **fixed by `patches/title_gate.py`** (92dff19): now ~3.5 s |
| Any other side request with other settings, another chat, an edit or a regeneration | everything again | 15 s and up |

Two things would help:

- **Several live conversations** (a small LRU instead of one entry): side
  requests and switching between chats stop evicting each other.
- **Cloning**: a new chat could continue from a snapshot holding only the
  system prompt (its ~834 tokens are identical in every pi-chat), and an
  edit or regeneration could continue from the turn before it.

The 0.18.0 Python package wraps neither. The C library exports
`litert_lm_conversation_clone(conversation) -> conversation` ("duplicating
its prefilled state"; C++ `Conversation::Clone` → `SessionAdvanced::Clone`),
and the Python `Conversation` is a thin ctypes wrapper (`_lib`, `_ptr`), so
the cache can call it directly.

**Known blocker for the new-chat case:** a conversation's system prompt and
history (its "preface") are prefilled together with the first message
sent, not at creation. The C++ API has `prefill_preface_on_init` and
prefill-only "pending" messages, but 0.18.0's C API exposes neither (checked
in `c/conversation.cc` and `runtime/conversation/conversation.cc`). So there
is no clean "system prompt only" state to clone.

## The probe

`probes/clone_probe.py`, run by `probes/run-clone-probe.sh` on rpi. It uses
a throwaway container from the serving image, created through serve's own
`get_or_initialize_server_engine` (YNNPACK, MTP, 16k context), with
litert-lm-1 stopped for about 6 minutes. Sampling is greedy, so equal
histories must give byte-identical text; every check is against a fresh
conversation built the way a cache miss builds it.

1. **Clone:** does it work on this engine (CPU, MTP, constrained decoding
   for tools)? Does a clone continue exactly like a fresh conversation with
   the same history, and is the original unaffected? How long does it
   take, and what does it cost in RAM?
2. **Several live conversations:** a second chat runs, then the first
   continues. Is the output still exact, and does switching back cost a
   re-prefill?
3. **"System prompt only":**
   - Is anything prefilled at creation? (`token_count()` right after
     `create_conversation`.)
   - Does priming a conversation with the system message alone and a
     1-token reply give a clone that continues like `[system, user]`? It
     probably doesn't, since the extra turn changes the context.

## What the results decide

- **1 and 2 work:** replace the single entry with an LRU of a few
  conversations, capped by RAM, and use clones for edits and regenerations.
  That's Python only, in `prefix_cache.py`.
- **3 doesn't work:** the new-chat case needs either a one-instruction
  binary patch that turns on `prefill_preface_on_init` in
  `litert_lm_conversation_create` (like `patches/fix_mtp_rollback.py`), or
  an upstream release whose C API exposes it.
- **1 fails:** stay with the title gate. Upstream may wrap clone in Python
  later.

## Results

Not run yet.
