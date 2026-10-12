# Plan: MTP's side effects

Status: **draft, 2026-10-12. Nothing in this plan has been changed.**

MTP is on (88c7205), with upstream #3439 fixed in the wheel's library
(`patches/fix_mtp_rollback.py`). Decode is +37–73% in real chats on the Pi,
7.2–7.4 → 10.1–12.5 tok/s; prefill is unchanged.

## The side effects

| | Effect | Evidence |
|---|---|---|
| S1 | **The request's sampling is ignored.** `temperature`, `top_p`, `top_k` and `seed` act only on the first token after each prefill; every later token is greedy. The same question always gets the same answer. | Byte-identical output from greedy and temperature-1.0 requests (real_bench, 2026-10-12). Upstream [#3017](https://github.com/google-ai-edge/LiteRT-LM/issues/3017), open. |
| S2 | **Greedy changes how the model works.** It writes the answer *before* calling the tool, then says nothing after the tool result. It skips tools it calls when sampling. | Rent chat: "**35.08%** … Calculation:" + `calculate`, then an empty reply; MTP off gives the same empty reply after that history. Letter count: 3 e's from memory (MTP off called `count_text`: 4). Same in both MTP rounds. |
| S3 | **Greedy loops are possible** in long thinking (4096-token cap). | Not observed yet. Upstream's loop truncation (58fcc1ca) covers ASR only. |
| S4 | **Our measurement misreads S2.** run4.py grades only the text after the last tool result. LibreChat also shows the text before the tool call. | The "empty" rent answer contained 35.08%. |
| S5 | **Upgrades are pinned.** The #3439 patch checks the 0.18.0 wheel's sha256 and fails the image build on any other version. | By design. |

## Root cause of S1–S3: one sampler

`runtime/executor/llm_litert_mtp_drafter.cc` creates both the drafter's and
the verifier's samplers with `CreateGreedySampler` (top-k 1).

The verifier step itself is **sample-and-match**:
- The verifier picks a token at every draft position.
- Each draft is kept while it equals the verifier's pick.
- At the first mismatch the verifier's own pick is emitted, and the step ends.

The NPU executor does the same (`PerformRejectionSampling`). Every emitted
token is therefore the verifier's own draw given the tokens before it. So:

- **The fix is exact:** with the request's sampler on the verifier (temperature, top-k, top-p, seed), MTP samples from exactly the distribution MTP off does.
- **The drafter can stay greedy.** It only decides how many drafts get accepted, not what is emitted.
- **Cost:** acceptance falls from "the drafter's top pick is the model's top pick" to "the model's random draw equals the drafter's pick", so the speed-up shrinks. Unmeasured; expect less than the greedy +37–73%.

## Options

**A. Fix it in C++ (the right end state).**
- **The change:** the drafter's verifier samples with the session's sampler parameters (per conversation), plus a test. About 50 lines. It would also carry #3439 in source and upstream's drafter re-prime (deaaa643).
- **Blocker: it needs a source build of the library.** That build works (lib/build.sh in f6b66df, ~1.5 h on the Pi with the other containers stopped, same speed as the wheel). But it can't use Google's prebuilt tool-call constraint provider: it is built against Google's libc++ (`std::__u`) and traps on the first tool call (FstConstraint::ComputeMask). Two ways round it:
  - **A1.** Constrain tool calls with the open-source **llguidance** provider instead; it is already in the build (`llg_fc_tool_calls.cc`). serve would pass `provider=LL_GUIDANCE`. Needs testing: tool-call reliability on Gemma 4, and grammar build time per request against the FST's 0.6 s.
  - **A2.** Build against a libc++ with Google's ABI (namespace `__u`, plus their hardening and ABI flags). Unknown flags; risky.
- **Effort:** about 1–2 days, mostly validating A1.

**B. Binary-patch the verifier sampler (no source build).**
- **The change:** patch `CreateGreedySampler`'s constants in the wheel library (top-k 1 → 64, temperature 1.0), the way fix_mtp_rollback.py patches the rollback. All requests would get Gemma's recommended sampling, which is what every LiteRT-LM spec and preset sends anyway; a request's own values would still be ignored.
- **Feasibility is unverified:**
  - If the function is shared rather than inlined twice, the drafter samples too. That is still exact, with lower acceptance.
  - top-p 0.95 is a float constant with no room for it, so it would likely stay 0. CPU sampler semantics for p=0 still need checking.
- **Effort:** a few hours if the code allows it at all. Fragile.

**C. Turn MTP on per request (Python only).**
- **The change:** create the engine with MTP off. The prefix-cache patch passes `enable_speculative_decoding` per conversation (the C API has it per session; the drafter loads lazily). Rule: MTP only for greedy requests (`top_k` 1 or `temperature` 0), or for a model id or preset that opts in.
- **Result:**
  - Presets choose, for example Fast = greedy + MTP and Thinker = sampled without MTP.
  - No silent override; cheap and reversible.
  - Sampled presets lose the speed-up.
- **Effort:** about an hour plus a check.

**D. Upstream (public, needs your OK and GitHub account).**
- **#3017:** the sample-and-match analysis: the CPU drafter only needs the session sampler on its verifier. Offer the patch.
- **#3439:** the root cause (rollback by `sequence_length - step` drops the stop token, which the executor keeps as its pending token) and the one-line fix, with the token-count probe: 21 39 68 88 / 20 37 65 84 / fixed 21 39 68 88.

**E. Measure properly (needed whatever we pick).**
- **Grading:** run4.py grades the whole turn's text (pre-tool text included).
- **A bigger round:** at least 24 fresh chats, MTP off vs MTP on (and vs the chosen fix). At 11 turns, 9/11 vs 8/11 is noise.
- **S3:** log thinking length and cut-offs to catch greedy loops.

## Suggested order

1. E, then D: cheap, and upstream may fix #3017 for us.
2. If the measurement shows greedy hurts answers: C now (honest, an hour), A1 when there's a day for it.
3. B only if A is off the table and C's slower sampled presets aren't acceptable.
