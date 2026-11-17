# Plan: one engine at a time, chosen by the request (Qwen3.5-2B default, E2B for audio)

Status: **SHELVED 2026-11-14** by the user after step 0 (Qwen too slow on
the Pi; logic never tested). Nothing built. Original status: draft, 2026-11-14. Gated on the Qwen3.5-2B benchmark
running in another session (litert-lm-1 is currently started with
`LITERT_MODEL_ID=qwen35-2b`, 16k, MTP off; do not restart or redeploy
litert-lm-1 until that session is done).

## Step 0 result (2026-11-14): Qwen3.5-2B fails on speed

`litert-community/Qwen3.5-2B` has only `Qwen3.5-2B_int8.litertlm`
(plus a VL variant). On the Pi 5 CPU:

- **YNNPACK (what `serve` uses in this image): does not load.** Its
  1024-token prefill graph is 25.7k nodes; after "YNNPack delegated 25632 of
  25703 nodes" the engine sat on one core for 32 min (RSS creeping to
  1.65 GB) and was stopped.
- **XNNPACK (`litert-lm benchmark`, 1024 prefill / 256 decode):**

  | ctx | init | first token | prefill | decode |
  |---|---|---|---|---|
  | 4096 | 146 s | 37 s | 27.7 tok/s | 2.95 tok/s |
  | 16384 | 146 s | 40 s | 25.8 tok/s | 2.92 tok/s |

  E2B on the same Pi: ~75 prefill, 10–12.5 decode (MTP). Qwen is about
  3× slower to read and 3–4× slower to write, with a 2.5 min cold load
  (every idle unload). It leaves a 1.9 GB `.xnnpack_cache` next to the model.

Conclusion: with this file, Qwen as the default model would make every
reply 3–4× slower than now. The router below is on hold unless a faster
Qwen build appears (int4, or a YNNPACK fix in LiteRT 0.19.x).

## Goal

- Requests that carry audio (`input_audio` content parts) run on `gemma4-e2b`.
- Every other request runs on `qwen35-2b`.
- Only one of them is in RAM at a time; `idle_unload.py` still frees it after
  600 s idle.
- Reasoning is on for both models on every request that reaches a model.

## What already exists (checked on the Pi today)

- Both models are in litert-lm-1's registry: `gemma4-e2b` 2.4 GB,
  `qwen35-2b` 2.0 GB (imported 2026-11-14 by the benchmark session).
- Upstream `get_or_initialize_server_engine(server, model_id=…)` already
  switches engines: if the request's model id differs from the loaded one it
  calls `__exit__` on the old engine and loads the new one (the
  "b/513076049" TODO). So "one at a time" is free; what is missing is
  (a) choosing the model id and (b) cleaning up the state that belongs to the
  old engine.
- `config.json` is rendered from `config/config.json.template` with one
  `models` entry (`LITERT_MODEL_ID`), so a second model gets defaults
  (no 16k context, no MTP settings).
- `patches/idle_unload.py` closes the engine, prefix-cache conversations and
  snapshots after `LITERT_IDLE_UNLOAD_S`.
- `patches/apply_reasoning.py` turns Gemma's thought "channels" into
  `reasoning_content`, only when the request sets `reasoning_effort`.
- Audio callers: only instagram-personal's `transcribe_audio`
  (`instagram-personal/workflow.py`, `model: gemma4-e2b`, `max_tokens: 384`),
  straight to `litert-lm-1:9379`. LibreChat does not send audio parts.

## Design

### 1. Config: both models, each with its own settings

`config.json.template` gets two entries instead of `${LITERT_MODEL_ID}`:

- `gemma4-e2b`: cpu, 16384 tokens, `speculative_decoding: true` (MTP).
- `qwen35-2b`: cpu, max tokens from the benchmark (16384 unless it says
  otherwise), `speculative_decoding: false` (no drafter in the file).

New env: `LITERT_DEFAULT_MODEL=qwen35-2b`, `LITERT_AUDIO_MODEL=gemma4-e2b`;
the entrypoint imports both if missing (loop over the two ids).

### 2. New patch `patches/model_router.py` (+ `apply_model_router.py`, tests)

Runs first in chat-completions handling, before memory/title gates and the
prefix cache, and rewrites `body["model"]`:

- any message with an `input_audio` part (current or history) →
  `LITERT_AUDIO_MODEL`; Qwen cannot read it, so history counts too;
- otherwise → `LITERT_DEFAULT_MODEL`, whatever the client asked for;
- `LITERT_ROUTER=0` passes the client's model through (rollback/bench).
- logs `router: <model> (<reason>)`.

Embeddings (`/v1/embeddings`) are untouched; litert-lm-2 stays separate.

`/v1/models` should list one public id (e.g. `pi-chat`) plus the two real
ids, so LibreChat and the Instagram bot can stop naming a model at all.

### 3. Switch hygiene (the part upstream doesn't do)

When the router's choice differs from `server.model_id` and an engine is
loaded:

- close every prefix-cache conversation and snapshot first (they hold the
  old engine's KV; reusing one after `__exit__` would crash or misanswer).
  Reuse `idle_unload._close_prefix_cache()` + `malloc_trim`, log
  `router: switching e2b -> qwen (n conversations closed)`.
- prefix-snapshot warm file: key entries by model id so a restart rebuilds
  only the default model's snapshot, not E2B's.
- memory: unload before load is already upstream's order, so the peak is
  one model (~2.3–3.9 GB), within `mem_limit` 7g. Verify RSS returns to
  ~270 MB between engines (same check as idle_unload).

### 4. Reasoning always on

- The router sets `reasoning_effort` (e.g. `"medium"`) on every request that
  reaches a model, overriding the client. Title and memory gates answer
  without a model, so they're unaffected.
- E2B: works today through `apply_reasoning.py`.
- Qwen3.5 via LiteRT's `qwen3` model type: **open question for the
  benchmark** — does `reasoning_effort` turn on `<think>` in its template,
  and do thoughts come out as channels (→ `reasoning_content`) or as
  `<think>…</think>` text in `content`? If the latter, extend
  `apply_reasoning.py` to split them, or the Instagram bridge's
  markup checks will reject replies.
- `length.py` already reports thought-heavy cut-offs; budgets must grow:
  `transcribe_audio`'s `max_tokens: 384` will run out with thinking on.
  Raise it (≈1536) and measure; this is the one caller-side change needed.

### 5. Callers

- instagram-personal: `transcribe_audio` keeps working (it sends audio →
  E2B). The E2B profile's answering request is text → now runs on Qwen; rename
  the profile/`setup_agent.py` model to the public id and re-provision the
  agent. Check `E2B_MODEL_PARAMETERS` (context, max tokens) against Qwen.
- LibreChat `librechat.yaml` LiteRT endpoint: model list → public id;
  context window per the benchmark.
- README sections (server, on-demand loading) and compose comments.

## Cost to accept or tune

A voice message in Instagram now does: transcribe on E2B → switch → answer
on Qwen. Each switch is a cold load (11–13 s from page cache, ~36 s from
disk) plus a snapshot rebuild for Qwen's tool preface (12–65 s). With 8 GB
RAM and swap already at 3.7 GB, page cache may not keep both files warm.
Options if that is too slow (decide after measuring): answer voice turns on
E2B too (whole Instagram turn on the audio model), or keep a short
"stay on E2B" grace period. Neither is in the base plan because the rule
is "audio → E2B, everything else → Qwen".

## Steps

0. **Wait for the benchmark.** Inputs needed: Qwen tool-call validity,
   thinking terminates, reasoning field format, tok/s at 16k, RSS. If Qwen
   fails those (2026-11-13 desk research flagged it as verbose), stop here.
1. Template + entrypoint for two models (build-time only, no behaviour change
   with `LITERT_ROUTER=0`).
2. `model_router.py` + switch hygiene + tests in `tests/` (routing rules,
   reasoning forced, prefix cache closed on switch, rollback flag).
3. Reasoning for Qwen in `apply_reasoning.py` if step 0 says it's needed.
4. Deploy on the Pi when pi-loop is idle; test: text chat → Qwen, audio
   transcription → E2B → next text → Qwen; RSS after each switch and after
   idle unload; `make logs` shows router lines; Instagram voice-message
   end-to-end time.
5. Callers: Instagram `max_tokens`, model id, re-provision; LibreChat yaml.
6. README/compose docs, commit, push.

## Rollback

`LITERT_ROUTER=0` and `LITERT_MODEL_ID=gemma4-e2b` restore today's
single-model behaviour without a rebuild.
