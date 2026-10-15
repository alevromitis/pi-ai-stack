# agent-tools

Sandboxed [MCP](https://modelcontextprotocol.io) servers that give LibreChat
agents web search, offline Wikipedia, a calculator, a shell, file access and a browser. Lives in
`docker/prod/agent-tools/`. LibreChat reaches them on `agent-tools-internal`;
none is published on the host. `kiwix-serve-1`, the web view of the same
Wikipedia, is (`http://rpi.<tailnet>.ts.net:8090`), so the links in
`mcp-wikipedia`'s results open.

| Container | Server | Network | Can touch |
|---|---|---|---|
| `mcp-shell-1` (disabled 2026-10-13, Compose profile `disabled`) | [`mcp-shell-server`](https://github.com/tumf/mcp-shell-server) 1.1.12 via `supergateway` | internal only | `/workspace`; commands in `SHELL_ALLOW_COMMANDS` |
| `mcp-filesystem-1` (disabled 2026-10-13) | `@modelcontextprotocol/server-filesystem` via `supergateway` | internal only | `/workspace` |
| `mcp-searxng-1` | [`mcp-searxng`](https://github.com/ihor-sokoliuk/mcp-searxng) 2.5.1 (upstream image) | internal + `agent-tools-egress` + `searxng-external` | `searxng-1:8080`; the internet (`web_url_read`) |
| `mcp-playwright-1` | [Playwright MCP](https://github.com/microsoft/playwright-mcp) 0.0.82, headless Chromium | internal + `agent-tools-egress` | the internet (and, via Docker NAT, the LAN); writes to `/workspace/playwright` |
| `mcp-wikipedia-1` | `wikipedia/server.py` (this repo; libzim 3.13, mcp 2.3) | internal only | `${WIKIPEDIA_ZIM_DIR}`, read-only |
| `mcp-calc-1` | `calc/server.py` (this repo; mcp 2.3) | internal only | nothing |
| `mcp-translate-1` | `translate/server.py` (this repo; CTranslate2 + Argos models) | internal only | `${TRANSLATE_MODELS_DIR}` (`/mnt/external/argos-models`), read-only |
| `kiwix-serve-1` | [`kiwix-serve`](https://github.com/kiwix/kiwix-tools) 3.8.2 (upstream image) | `agent-tools-web`, published `8090` | `${WIKIPEDIA_ZIM_DIR}`, read-only |

**Threat model.** A model that reads web pages can be prompt-injected.
`mcp-shell-server` refuses interpreters (`python`, `sh`, `node`), `env`,
`xargs`, `find -exec` and shell operators inside arguments whatever the
allowlist says, so it is a text/file tool, not a code runner — but its
README is explicit that this hardening is best-effort. So the containers
are the boundary, as `mcp-shell-server`'s own README
demands: uid 1000, `cap_drop: ALL`, `no-new-privileges`, read-only root,
CPU/memory/pid limits, no credentials anywhere in the stack, and the only
host paths are `${AGENT_WORKSPACE_DIR}` and, read-only and for
`mcp-wikipedia` and `kiwix-serve` alone, `${WIKIPEDIA_ZIM_DIR}`. Shell and filesystem sit on an
`internal: true` network — no internet, no LAN, no host ports — so code they
run cannot exfiltrate or pivot. Playwright needs egress; its browser *can*
reach LAN services (router, the host's published ports). Its
`--blocked-origins` is documented upstream as not a security boundary, so
the real fix, if needed, is a `DOCKER-USER` iptables rule dropping
`agent-tools-egress` → `192.168.0.0/16`.

## First run

```bash
mkdir -p ~/docker/prod/agent-tools/workspace/playwright
sudo chown -R 1000:1000 ~/docker/prod/agent-tools/workspace
# The Wikipedia ZIM (52.7 GB; ~45 min at the ~20 MB/s mirror.accum.se gives rpi).
mkdir -p /mnt/external/wikipedia && cd /mnt/external/wikipedia
z=wikipedia_en_all_nopic_2026-06.zim
curl -fLO "https://download.kiwix.org/zim/wikipedia/$z.sha256"
curl -fL -C - -o "$z" "https://download.kiwix.org/zim/wikipedia/$z" && sha256sum -c "$z.sha256"
cd - && make decrypt && make up    # builds agent-tools-mcp:local, agent-tools-wikipedia:local and agent-tools-calc:local (arm64) first
```

The stack must be up before `librechat` (it owns `agent-tools-internal`)
and after `searxng` (it joins `searxng-external`). `mcp-wikipedia` fails to
start, rather than serving an empty directory, if `/mnt/external` isn't
mounted (`create_host_path: false`).

**`mcp-searxng` tuning for an 8–32k-token CPU model** (env in the compose
file): `SEARXNG_LITE_TOOLS` (query/url-only schemas), 5 results of ≤300
chars, `web_url_read` capped at 6000 chars (~1.5k tokens), English default.
Every token the tools return is prefilled on the Pi's CPU, so these caps
are the main latency lever.

**What `gemma4-e2b` doesn't see** (`patches/hide_tools.py` in litert-lm):
every tool schema is prefilled into the system-prompt snapshot, and a new
chat whose preset or tools differ rebuilds it (~22 s). mcp-searxng can't
hide single tools, so litert-lm drops `searxng_search_suggestions` and
`searxng_instance_info` (`LITERT_HIDE_TOOLS`) and replaces `web_url_read`'s
description with "Read a web page or file by URL, as markdown."
(`LITERT_TOOL_DESCRIPTIONS`). Qwen still gets all of them unchanged. The
original description (mcp-searxng 2.5.1):

> Fetch URL. Converts HTML to markdown; returns explicit JSON, plain text,
> YAML, TOML, and XML as readable markdown; supports bounded PDF text
> extraction; other binary/media/archive downloads are rejected. When
> browser solvers are configured, mcp-searxng attempts FlareSolverr first
> and then Byparr only after a busy or transient-unavailable acquisition;
> after a final busy or unavailable provider it uses one uncached direct
> read.

## Wikipedia (`mcp-wikipedia`)

The pi models are told their memory of facts is unreliable and to search
first; `mcp-wikipedia` gives them a local source for that. It serves one
tool, `wikipedia_search(query)` (~320 characters of schema), over a Kiwix
ZIM of English Wikipedia without images
(`wikipedia_en_all_nopic_2026-06.zim`: snapshot 2026-06-17, 52.7 GB, with
libzim's title and Xapian full-text indexes). Per search:

1. **Candidates.** The article titled with the first name the query gives
   (a capitalised run like "Eiffel Tower", or a quoted phrase; "the Night
   Watch" tries `The_Night_Watch` first; numbers next to a name belong to
   it, "2010 FIFA World Cup" or "Apollo 11", tried before the name without
   them), then libzim's title suggestions,
   then a full-text search on every content word, topped up with any word.
2. **Passages.** The top 3 articles become lead / section paragraphs
   (infobox rows kept as one line; references, see-also and the like
   dropped), ranked by BM25 with a nudge towards leads and better articles.
   The named article's opening paragraph goes first, its first prose
   paragraph then its infobox: "Rembrandt birth date" shares no word with
   "Born: 15 July 1606", and HTTP's infobox never says "Hypertext Transfer
   Protocol". Without the year, "Which country won the 2010 FIFA World Cup?"
   got the general article, whose infobox says "Current champions:
   Argentina", and the model answered Argentina. With both changes, answers
   in the output went from 38/43 to 39/43 for the model's own queries (38/44
   verbatim, unchanged). Portal and disambiguation pages are skipped
   unless the query names one ("Portal:Solar System" had crowded out the
   answer to "largest planet"), and lists and outlines rank below an
   article about one thing ("Gold" over "Period 6 element"); recall
   unchanged. When the query names nothing, the output ends with a hint
   to search again with the subject's name first ("Oxygen atomic number",
   not "what is the atomic number of oxygen"): the model had searched the
   latter three times and given up, and hints in tool results are
   followed where ones in tool descriptions weren't.
3. **Answer.** The best 5, each cut to ≤500 characters at a sentence, with
   its section, then `End your answer with this line:` and a ready-made
   `Sources: [Title](link), …` line for their articles: ~2.1k characters,
   ~550 tokens. A link (`WIKIPEDIA_LINK_BASE` + the article's ZIM path)
   opens the article on `kiwix-serve-1`, so it shows the same snapshot the
   passage came from. With pi-chat's rule "End your answer with the Sources
   line the search gave you", `gemma4-e2b` cited 11 of 12 answers
   (2026-10-10, fixed searches, local copy of the model); a link under each
   passage got 0 of 12, and the line without the prompt rule 9 of 12. It
   copies the whole line rather than only the articles it used.
   Since 2026-10-10 litert-lm writes the final Sources line itself
   (`patches/citations.py`, `LITERT_CITATIONS`): from the Wikipedia,
   SearXNG and `web_url_read` results since the user's message, keeping
   only those that back the answer: they hold a name or long number from
   the answer that the question and queries don't (or two of its other new
   words); one the model linked in its text needs only to back it. With
   nothing backing it ("I don't know", an answer from memory) there is no
   line; before, every result was listed and "Poseidon" cited "The Greek".
   The model's own Sources block is held back while streaming and
   replaced, so made-up links never show, and what it listed counts for
   nothing: it listed "Eiffel Tower (Paris, Tennessee)" and "Statue of
   Liberty (Finland)" next to the real ones. On 9 citation cases
   (web-only, mixed, page reads, "in two sentences") every answer ended
   with valid sources, with thinking on and off. With no search since the
   user's message, a Sources block with no link in it ("Sources: []",
   copied from the prompt's rule) is dropped, and so is one whose links no
   tool result in the chat gave ("[Calculation result](https://example.com/
   …)" after a date calculation); a follow-up citing an earlier turn's
   article keeps its block.
   A Sources block that starts mid-line ('... "Example Domain". Sources:
   [Example Domain](URL)') is held back too. The same filter drops
   Gemma's control tokens that leak into replies (`<channel|>`) and a raw
   `<|tool_call>…<tool_call|>` the model writes when no tool is offered,
   whether citations are on or off.
   The header names the snapshot month, read from the ZIM's `Date`
   metadata.
4. **Wrong names.** If no shown passage names anything the query names,
   the passages are left out: the output says nothing mentions "Tao Lung",
   lists the closest article titles and says to search again or say "I
   don't know". With the passages shown under a warning, `gemma4-e2b`
   still answered "Tao Lung" from Tan Tao-liang's films. If only some
   names are missing, the passages stay with a note. It fired for none of
   103 real questions and Gemma queries (2026-10-10).

Measured 2026-10-09/10 with 44 checked questions (named things, long-tail
facts, descriptions, 2025 events, Greece, numbers, two-step questions):

| | |
|---|---|
| answer in the tool output | 38/44 as worded; 38/43 for the queries Gemma itself wrote (35 and 36 before the named-article step, 2026-10-10) |
| right article in the top 3 | 24/44; 27/43 for Gemma's queries |
| latency on `rpi` (1 CPU) | 0.21 s median, 0.48 s worst with a cold page cache; 0.13 s warm |
| memory | ~340 MB RSS (opening the archive: ~1 s, 252 MB) |
| end to end, 12 questions, `gemma4-e2b` with thinking | 8 right, 3 "couldn't verify", 1 wrong; ~60 s median a question |

Keyword search is the weak step: it misses articles named differently from
the query (Eurovision 2025, the 2025 NBA Finals) and descriptions of a
thing rather than its name. A dense index of article leads
(EmbeddingGemma 2) found the right article in the top 3 for 43/43 of
Gemma's queries in a 78k-article test; it is the candidate next step, not
built. With thinking off, Gemma skipped searching for 13/44 questions it
thought it knew, and no answer in the end-to-end run
cited its link.

**Updating the snapshot.** Kiwix publishes new `nopic` builds every few
months. Download the new file and its `.sha256` next to the old one, check
it, set `WIKIPEDIA_ZIM_FILE` in `.env` (and `make encrypt`), `make up`,
then delete the old file. Links already in chats keep working: they use
`kiwix-serve`'s dateless alias (`wikipedia_en_all_nopic`), which follows
the file. Two copies need ~105 GB on `/mnt/external`,
which also holds the restic repository: delete first if space is short.
The ZIM is not backed up — it is re-downloadable.

## Calculator (`mcp-calc`)

`gemma4-e2b` miscounts letters and gets arithmetic and date sums wrong even
when it thinks step by step. `mcp-calc` (`calc/calc.py`, pure Python, no
network or files) gives it exact answers through four tools, ~1.3k
characters of schema together:

* `calculate(expression)`: `+ - * / ^ %`, "15% of 80", `sqrt`, `log`,
  trig, `factorial`, `pi`. Evaluated on a whitelisted Python AST, never
  `eval`; exponents and results are capped.
* `date_diff(start, end)` and `date_add(date, days, weeks, months,
  years)`: dates as `YYYY-MM-DD`, "today" or "3 March 2024", answers with
  weekdays. "today" is in the container's `TZ` (`TZ` in `.env`, required).
* `count_text(text, find)`: characters, letters, words and how often
  `find` appears. Without `find` it lists every letter's count, since the
  model often leaves it out and then counts the letters itself.

Each answer restates its input ("17*23 = 391"). In a local test with the
model ("How many r letters are in strawberry?", "What date is 90 days after
2026-10-10?") it picked the right tool and answered correctly. Before the
per-letter tally, a call without `find` still ended in "2".

## Translation (`mcp-translate`)

Added 2026-10-13, reversing the 2026-10-12 English-only decision. The
models work best in English, and Wikipedia is English-only. So for a message
in another language the model translates it into English, thinks and
searches in English, then translates its answer back. Both prompts
(qwen-chat, pi-chat) say so. qwen-chat has the tool on; for pi-chat add it
from the input bar.

* One tool, `translate(text, target="en", source="")`. When `source` is
  empty, the language is detected (py3langid, limited to the installed
  languages). The reply starts with "Greek -> English:", which tells the
  model the user's language.
* Engine: [Argos Translate](https://github.com/argosopentech/argos-translate)
  models run by CTranslate2 (int8), English <-> 30 languages (`ar bg ca cs
  da de el es fa fi fr he hi hu id it ja ko nl pl pt ro ru sk sv th tr uk
  vi zh`). Other pairs pivot through English. The `argostranslate` package
  is not used, because it needs stanza and torch (~1 GB) only to split
  sentences; a regex splits them here. The older packages (es, cs, da, fi,
  hu, id, pl, ro, uk, one or both directions) use Moses + BPE instead of
  SentencePiece, handled as argostranslate does.
* On the Pi: 0.6-1 s for a sentence, ~2 s for a pivot. RSS levels off at
  ~500 MB with 2 pairs loaded (`TRANSLATE_KEEP_LOADED`; evicted models are
  unloaded and `malloc_trim`'d, `MALLOC_ARENA_MAX=2`). Without that it
  climbed to ~740 MB.
* Models: 60 pairs, 6 GB. They are not in the image; fetch them once on the
  host (stdlib only, re-runs skip what's there):

```bash
python3 docker/prod/agent-tools/translate/fetch_models.py /mnt/external/argos-models
```

**Automatic translation in litert-lm (deployed 2026-10-14).** With only
the tool, gemma4-e2b often didn't call it, on the way in or on the way
out. So litert-lm-1 (`patches/translation.py`, `LITERT_TRANSLATE`) now does
it for every chat, tools or not. It calls `POST /api/translate` on
`mcp-translate-1` over `agent-tools-translate` (internal, those two only):

* Each user message detected as another language with confidence of at
  least `LITERT_TRANSLATE_MIN_CONF` (0.8; "ok" and "hi" score about 0.03)
  reaches the model in English. Translations are cached, so the history
  gives the same English every time and the prefix cache still matches.
  Assistant turns pass as the client sends them, because the prefix cache
  matches them against what was streamed.
* When the newest message was translated, the reply is held back (the
  thoughts still stream) and translated into that language at the end.
  This happens only if the reply is in English, so "translate this into
  French" stays French. Code blocks, inline code, links and the Sources
  line (built by `citations.py` from the English reply) stay as they are.
* It runs before everything else on a request, so citations, the memory
  gate and the reminder all see English. If `mcp-translate` is down,
  messages and replies pass through untouched.
* The cost: the reply arrives all at once instead of streaming, plus
  roughly 1 s per sentence on the Pi.

Qwen (llama-server) is not behind litert-lm and keeps using the tool.

**Alternatives considered (not built):**

* *NLLB-200-distilled-600M (CTranslate2 int8) on the Pi.* Better quality and
  200 languages, but ~1 GB RAM and 1-3 s a sentence. That competes with
  litert-lm for memory: tests OOM-rebooted the Pi on 2026-10-12.
* *NLLB (or a larger model) on the workstation GPU*, reached over the
  tailnet. Fast, but the Pi's chats would then depend on the workstation
  being on.
* *A second litert-lm instance running a small LLM as translator.* Not
  enough free RAM next to litert-lm-1, tens of seconds per paragraph, and
  less literal than a dedicated translation model.

