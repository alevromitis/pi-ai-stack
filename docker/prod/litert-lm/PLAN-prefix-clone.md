# Plan: cloned conversations for the prefix cache

Status: **step 1 deployed 2026-10-12 (7f3ffa7, 44fbd57); step 2 open.**
Live conversations alternate exactly, MTP on or off, and the prefix cache
now keeps several. Clones are exact and fast, as long as the conversation
they were cloned from is treated as a frozen snapshot (see Results).

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

## Results (2026-10-12)

**Setup:** rpi, the serving image (0.18.0 wheel with the #3439 fix), CPU,
YNNPACK, MTP, 16k context, pi-chat's system prompt (~890 tokens) with one
tool, greedy sampling. Raw logs are in `~/litert-bench/ab/clone-probe*.log`
on rpi.

| Check | Result |
|---|---|
| Engine | init 7–10 s; RSS 2.26 GB (1.25 GB anon) |
| Fresh conversation, system prompt + first message | first token after 11.7–12.7 s |
| Is the system prompt prefilled at creation? | **No.** `token_count` is 0 until the first message |
| `litert_lm_conversation_clone` | **Works.** 1.1 s; about +95–110 MB RAM per clone at ~920 tokens |
| Clone continued | **Byte-identical** to a fresh conversation with the same history; first token after **1.8 s instead of 12.0 s** |
| Original continued right after cloning | identical; first token after 0.6 s |
| A second conversation alive at the same time | works and is exact; +~360 MB including its own history |
| The clone, continued after the second chat ran | identical to an uninterrupted conversation (2 of 2 runs) |
| **The original, continued after its clone and the second chat ran** | **Broken with MTP on.** One run gave different text. The other failed in the draft model: `llm_litert_mtp_drafter.cc:741 … Cannot auto-resize tensor kv_cache_v_3: no dims_signature exists` |
| "System prompt only" by priming with a 1-token reply | **Not exact.** The one token opened a tool call, and the reply differs from `[system, user]` |

**Not a clone effect: re-tokenization.** A conversation that keeps
continuing its own KV drifts from one rebuilt from the same text by its
third turn (993 vs 995 tokens, different reply). The generated tokens
aren't always what re-tokenizing the reply gives. Today's cache misses
rebuild from text, so they have this too; it isn't a regression.

**What this means:**
- Cloning itself is sound and turns a ~12 s prefill into ~2 s.
- **With MTP on, a conversation and its clones share something in the
  draft model** (its KV cache, by the error), so using one breaks the
  other. A cache built on clones must never send to a conversation after
  cloning it, if that pattern is safe at all.
- **New chats are still blocked,** as expected: there is no
  system-prompt-only state to clone.

### Fourth run: `--switching`

Each case is compared with one conversation that sends the same three
messages with nothing in between.

| Case | MTP on | MTP off |
|---|---|---|
| (a) Two conversations alternating, no clones, to turn 3 | exact | exact |
| (b) Two clones of a snapshot that is never sent to again, to turn 3 | exact | exact |
| (c) The original sent one more message after its clones ran | exact | exact |

**Reading runs 2–4 together:**
- **Alternation is safe.**
- **Clones of a frozen snapshot are safe.**
- **The only failure** was an original that kept going after its clone had
  run. The draft model broke when the original grew to ~990 tokens (turn
  3). One message (turn 2, ~960 tokens) was fine.
- **Likely cause:** that fits the draft model's KV buffers moving to the
  clone, so the original can't grow them. Upstream issue material:
  `Conversation::Clone` with MTP.

## Decision

1. **Done: several live conversations** in `prefix_cache.py`, no clones
   (case a).
   - **Caps:** up to `LITERT_PREFIX_CACHE_SIZE` (3); idle ones are closed,
     oldest first, while the container is over `LITERT_PREFIX_CACHE_MAX_MB`
     (5120).
   - **Live test** (three interleaved chats, pi-chat prompt and a tool):
     going back to a chat after another one ran is a hit, first token after
     **4.0 s and 3.8 s instead of ~17 s**.
   - **Memory:** container 2.3 GB with the engine, 3.9 GB with one chat,
     4.8–5.0 GB with three. Closing a ~1.2k-token chat frees ~145 MB; the
     rest is buffers the engine keeps. With a 4,000 MB test cap it held at
     3.86–3.98 GB.
2. **Later, the big one: new chats from a system-prompt snapshot** (case b).
   That needs the preface prefilled at creation (`prefill_preface_on_init`),
   which 0.18.0's C API can't set: a binary patch like
   `fix_mtp_rollback.py`, or a newer release. It would cut a new chat's
   first token from ~13 s to ~3 s.
