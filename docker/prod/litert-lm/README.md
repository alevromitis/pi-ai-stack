# litert-lm

[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)'s OpenAI-compatible
server on the Raspberry Pi 5's CPU, built locally as `litert-lm-rpi5:local`
(arm64 only). Two containers: `litert-lm-1` serves the chat model
(`gemma4-e2b`) on the published port `9379` (`APP_PORT`, no
authentication), and `litert-lm-2` serves the embedding model for
[LibreChat's file search](../librechat/README.md#file-search-rag),
unpublished. The stack owns the `litert-lm-external` network that
[LibreChat](../librechat/README.md) joins; how LibreChat is configured for
this endpoint (fixed model list, context window, `dropParams`) is in
[its README](../librechat/README.md#the-litert-lm-endpoint).

The `PLAN-*.md` files are the research notes behind the larger changes;
`probes/` holds the scripts they ran.

## The server

LiteRT-LM's `serve` command (0.18) listens on `0.0.0.0:9379` and implements
`GET /v1/models`, `POST /v1/chat/completions` (streaming, and OpenAI-style
`tools` / `tool_calls`, streamed too) and `POST /v1/embeddings`. Verified on
`rpi` on 2026-10-06 with `gemma4-e2b` (0.17). 0.18 changed `serve`'s default
host to `127.0.0.1`, which LibreChat's container can't reach; the entrypoint
passes `--host 0.0.0.0`. The upgrade (from 0.17.1, 2026-10-09) also moved
the server into `litert_lm_cli/commands/serve/` and made response chunks
`litert_lm.Message` objects, so the image's patches were ported; with the
same seed 0.18 returns the same tokens, reasoning and tool calls as 0.17.1
at the same speed, and the prefix cache still hits. Rollback: retag
`litert-lm-rpi5:0.17.1` as `:local` and `make up`.

## What the image changes

Upstream `serve` is patched at build time (`Dockerfile`, `patches/`); every
patch fails the build if upstream drifts from the code it anchors on. The
ones that change behaviour you see in a chat:

* **Prefix (KV) cache.** Upstream `serve` builds a fresh conversation per
  request, so every turn — and every tool round of an agent turn —
  re-prefills the system prompt, tool schemas and whole history. The image
  patches in `patches/prefix_cache.py`: the last few conversations stay
  alive, and a request that extends one (same history + the reply it
  generated, then only new user/tool messages) prefills just the new
  messages. Anything else (edits, regenerations, a chat not cached, changed
  tools) is a full prefill, and the cached conversations stay for the
  requests that do extend them.
  - **How many:** up to `LITERT_PREFIX_CACHE_SIZE` (3) conversations, least
    recently used closed first. Idle ones are also closed while the
    container uses more than `LITERT_PREFIX_CACHE_MAX_MB` (5120); the chat
    in use never is.
  - **Memory:** each keeps its KV cache in RAM. The engine is ~2.3 GB, a
    first chat brings the container to ~3.9 GB, and each further short chat
    adds ~250–430 MB.
  - **Why several:** until 2026-10-12 only one was kept, so any other
    request (another chat, a side request with other settings) evicted the
    chat in use. Live conversations alternate exactly on this engine, MTP on
    or off (`probes/clone_probe.py`, `PLAN-prefix-clone.md`).
  - **Engine faults:** with MTP on, a new chat cloned from the system-prompt
    snapshot, or the next message of such a chat, sometimes fails in the
    draft model before its first token ("Cannot auto-resize tensor
    kv_cache_k_N", 2 of 8 chat requests in one run of
    `probes/mtp_interleave_probe.py`). The request is retried on a new clone
    or a fresh conversation, which has always gone through; logged as
    `prefix-cache: engine fault before the first token (…); retrying on …`.
    Once, under heavy churn, a clone answered wrongly without an error
    (`PLAN-prefix-clone.md`).

  Each request logs `prefix-cache: hit/miss/evict: <reason>` in
  `make logs`; `LITERT_PREFIX_CACHE=0` turns it off. The same patch makes
  the server honour `max_tokens` (upstream read only
  `max_completion_tokens`, so the spec's output cap was ignored). The reply
  is matched as the client got it (after the Sources rewrite and the other
  reply patches), not as generated: until 2026-10-12 every follow-up to a
  search-backed answer missed and re-prefilled the whole chat.
* **Speculative decoding (MTP) is on** (`LITERT_SPECULATIVE_DECODING`,
  2026-10-12). The model file's drafter proposes tokens and the model checks
  several per step: decode +37–73% in real chats on the Pi (7.2–7.4 →
  10.1–12.5 tok/s), prefill unchanged. Its drafter and verifier are greedy,
  so `temperature`/`top_p`/`top_k` no longer shape chat replies. Upstream
  0.18.0 drops a turn's stop token from the KV cache whenever the drafter
  got past it ([#3439](https://github.com/google-ai-edge/LiteRT-LM/issues/3439)),
  which matters here because the prefix cache continues conversations;
  `patches/fix_mtp_rollback.py` fixes it in the wheel's own library, one
  instruction (source form: `patches/litert-lm-v0.18.0-mtp-rollback.patch`).
  A source build of the library was tried and dropped: it can't use
  Google's prebuilt tool-call constraint provider (built against their
  libc++), which traps on the first tool call.
* **A cut-off reply says so** (`patches/length.py`). Thought tokens count
  toward `max_tokens`, and upstream ends every reply with `finish_reason:
  "stop"`: a no-tools reply that spent all 4096 tokens thinking came back
  empty and looked finished. A reply that used the whole limit now ends
  with `finish_reason: "length"` and a closing line saying it was cut off.
* **The prompt's checklist in every message** (`patches/reminder.py`,
  `LITERT_REMINDER=0` turns it off). `gemma4-e2b` follows the newest turn
  far more than a system prompt thousands of tokens back: a rule there was
  mostly ignored with thinking off, while the same rule appended to the
  user's message was followed 8 times out of 8.
  So pi-chat's prompt ends with one general paragraph, "Before answering:
  work out exactly what is asked …", on how to reach any answer (what is
  asked and in what form, a tool result for each fact, a tool for each
  calculation or count, only what those support, cite only what was used),
  and the server appends a copy of it, in parentheses, to every user
  message. It is read from the system prompt, so it is written once in
  `librechat.yaml`; requests without the paragraph (memory agent, titles)
  pass through. Every user message in the history gets the same copy, so
  the prefix cache still matches. General rules replace rules for kinds of
  questions: fixes for single failed test prompts were dropped
  (2026-10-10).
* **Titles come from the first message, not the model**
  (`patches/title_gate.py`, `LITERT_TITLE_GATE=0` turns it off,
  2026-10-12).
  - **Why:** LibreChat asks for a title after the first reply of every chat
    (`titleModel: 'current_model'`, `titleTiming: 'final'`). That request has
    no tools, so it evicted the chat from the prefix cache, and every chat's
    second message re-prefilled the system prompt and the first exchange.
  - **What it does:** the server answers the request at once with the first
    message's topic, up to 6 words ("What is the capital of France?" →
    "Capital of France"), without the model.
  - **Limits:** it recognises only LibreChat's default completion title
    prompt, so leave `titlePrompt` and `titleMethod` unset. Logged as
    `title-gate: answered: '<title>'`.

Documented with the tools they work on, in
[agent-tools](../agent-tools/README.md):

* **Hidden tools and shorter descriptions** (`patches/hide_tools.py`,
  `LITERT_HIDE_TOOLS`, `LITERT_TOOL_DESCRIPTIONS`).
* **A Sources line written by the server** (`patches/citations.py`,
  `LITERT_CITATIONS`), under [Wikipedia](../agent-tools/README.md#wikipedia-mcp-wikipedia).
* **Automatic translation** (`patches/translation.py`, `LITERT_TRANSLATE`),
  under [Translation](../agent-tools/README.md#translation-mcp-translate).

Also: `patches/apply_reasoning.py` streams Gemma's thinking as
`reasoning_content` (upstream generates and discards it), and
`patches/memory_gate.py` answers LibreChat's memory agent without the model
unless the user asked to remember or forget something (`LITERT_MEMORY_GATE`).
