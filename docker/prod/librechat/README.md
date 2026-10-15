# LibreChat

[LibreChat](https://www.librechat.ai) fronting the **litert-lm** stack as an
OpenAI-compatible custom endpoint, with web search through **searxng** via
an MCP server. Lives in `docker/prod/librechat/`. Built from [`template-docker`](../../../template-docker/README.md); read
that README first for the `.env` rule and the conventions every service
here shares.

```bash
docker compose -f docker-compose.librechat.yml up -d
```

## Layout

```
.
├── docker-compose.librechat.yml   # librechat + mongodb + meilisearch + rag_api + vectordb
├── .env.example                   # ports, host paths, image tags (interpolated)
├── .env.librechat.example         # app config + secrets (container-only)
├── rag-api/
│   └── run_with_prefixes.py       # rag_api entrypoint wrapper: EmbeddingGemma prefixes
└── librechat-config/
    └── librechat.yaml             # endpoint, MCP server, model spec; mounted read-only
```

## Topology

```
   host :${APP_PORT}
        │
        ▼
  ┌───────────────┐   librechat-internal (internal: true, no route off host)
  │  librechat-1  │──────────► librechat-mongodb-1:27017
  │   (uid 1000)  │──────────► librechat-meilisearch-1:7700
  └───────────────┘──────────► librechat-rag-api-1:8000 ──► librechat-vectordb-1:5432
                                    │  litert-lm-external
                                    └──► litert-lm-2:9379  POST /v1/embeddings
     │         │  agent-tools-internal  (owned by the agent-tools stack)
     │         └──► mcp-searxng-1:8080 ──► searxng-1:8080
     │              mcp-shell-1, mcp-filesystem-1, mcp-playwright-1
     │  agent-connectors-internal  (owned by the agent-connectors stack)
     │         └──► mcp-freshrss-1:8000 ──► freshrss-1
     │              mcp-nextcloud-1:8000 ──► nextcloud-aio-apache:11000
     │  agent-sandbox-internal  (owned by the agent-sandbox stack)
     │         └──► agent-sandbox-gateway-1:8000 ──► one Debian sandbox per chat
     │  litert-lm-external  (owned by the litert-lm stack)
     ▼
   litert-lm-1:9379  ──  GET /v1/models, POST /v1/chat/completions
```

`librechat` is on six networks: `librechat-external` (its own bridge, the
default route and the address the published port is forwarded to — that is
what the `priority: 100` is for), `librechat-internal` for the two data
stores and `rag_api`, `litert-lm-external` for the provider, `agent-tools-internal`
for the MCP tool servers (search included), `agent-connectors-internal`
for the FreshRSS and Nextcloud MCP servers and `agent-sandbox-internal`
for the sandbox gateway.

## Prerequisites

1. **The litert-lm stack must be running first.** Its compose file creates
   `litert-lm-external`; this stack declares it `external: true` and will
   refuse to start with `network litert-lm-external not found` otherwise.
   `docker/prod/litert-lm/docker-compose.litert-lm.yml` creates it and runs
   on `rpi`. If litert-lm ends up naming its network something else, set
   `LITERT_NETWORK` in `.env`.
2. **The agent-tools stack must be running too**, for
   `agent-tools-internal` (`AGENT_TOOLS_NETWORK` in `.env`), and therefore
   searxng, whose JSON API must be enabled — it
   returns `403` otherwise. `settings.yml` is host-only (it holds
   `secret_key`), so add this to
   `${DATA_ROOT}/searxng/config/settings.yml` and
   `make restart` the searxng stack:

   ```yaml
   search:
     formats:
       - html
       - json
   ```
3. **The agent-connectors stack must be running too**, for
   `agent-connectors-internal` (`AGENT_CONNECTORS_NETWORK` in `.env`); see
   [agent-connectors](../agent-connectors/README.md).
4. **The agent-sandbox stack must be running too**, for
   `agent-sandbox-internal` (`AGENT_SANDBOX_NETWORK` in `.env`); see
   [agent-sandbox](../agent-sandbox/README.md).
5. **Host directories**, owned by uid/gid 1000 — the image runs as `node`
   (uid 1000) and every data mount is a bind mount:

   ```bash
   sudo mkdir -p ~/docker/prod/librechat/{data/{uploads,images,logs},data-mongo,data-meili,data-vectordb}
   sudo chown -R 1000:1000 ~/docker/prod/librechat
   ```
6. **A CPU Mongo 8 supports**: AVX on x86, ARMv8.2-A on arm64 (the Pi 5's
   Cortex-A76 qualifies). On older hardware pin `MONGO_TAG` to 4.4.

## First run

```bash
cp .env.example .env
cp .env.librechat.example .env.librechat   # then fill in the secrets
docker compose -f docker-compose.librechat.yml up -d
docker compose -f docker-compose.librechat.yml logs -f librechat
```

On `rpi`, `make decrypt` recreates both files from their `.sops` copies.
On a fresh install, generate `CREDS_KEY`, `CREDS_IV`, `JWT_SECRET`,
`JWT_REFRESH_SECRET` and `MEILI_MASTER_KEY` as `.env.librechat.example`
says. Register the first account at `http://<host>:3080`,
then set `ALLOW_REGISTRATION=false` in `.env.librechat` and restart; there
is no admin invite flow to fall back on.

> Changing `CREDS_KEY`/`CREDS_IV` after users have saved provider keys makes
> those stored keys undecryptable. Treat them as permanent.

## The LiteRT-LM endpoint

`librechat-config/librechat.yaml` declares one custom endpoint. Its `apiKey`
and `baseURL` are `${...}` references that LibreChat resolves from the
container's environment — they come from `.env.librechat`, not from
`librechat.yaml`, so the config file stays free of anything secret.

LiteRT-LM's `serve` command (0.18) listens on `0.0.0.0:9379` and implements
`GET /v1/models`, `POST /v1/chat/completions` (streaming, and OpenAI-style
`tools` / `tool_calls`, streamed too) and `POST /v1/embeddings`. Verified on
`rpi` on 2026-10-06 with `gemma4-e2b` (0.17). 0.18 changed `serve`'s default
host to `127.0.0.1`, which LibreChat's container can't reach; the entrypoint
passes `--host 0.0.0.0`. What the image changes in the server (prefix
cache, speculative decoding, titles, citations, …) is in
[litert-lm](../litert-lm/README.md). Consequences for this side, all of them
already reflected in the config:

* **Model list is fixed** (`models.fetch: false`) to the `default` list,
  `gemma4-e2b`. `/v1/models` returns `litert-lm-1`'s whole registry, which
  also holds `gemma4-e4b` and `embeddinggemma2-text` (imported 2026-10-09,
  before embeddings moved to `litert-lm-2`); a chat request for the
  embedding model makes litert-lm close Gemma's engine to load it.
* **No authentication.** LiteRT-LM ignores the key, but LibreChat drops any
  custom endpoint whose `apiKey` is empty, so `LITERT_LM_API_KEY` carries the
  placeholder `litert-lm`.
* **File search (RAG) embeds on `litert-lm-2`**, not here; see
  [File search](#file-search-rag).
* **Tool calling works**, so agents and MCP tools run against this
  endpoint. A 2B model picks tools reliably for one or two steps; long
  multi-step plans are beyond it.
* **Context is 16384 tokens server-side** (`LITERT_MAX_NUM_TOKENS` in the
  litert-lm stack; halved from the model's maximum, 32768, on 2026-10-08).
  LibreChat maps any `gemma*` model to 128k and would overrun it
  (LiteRT-LM answers `500 Input token ids are too long`), so `tokenConfig`
  and the `pi-chat` spec set a 13312-token window with `max_tokens: 4096`
  (history gets 9216; the rest is headroom for tiktoken vs Gemma's
  tokenizer). Agents built in the UI need the same *Max context tokens* set
  in their parameters. The MongoDB presets differ: **E2B - Thinker** keeps
  13312, **E2B - Fast** has half, 6656 (since 2026-10-13), for shorter
  prefills. They live only in MongoDB; back them up before editing.
  Measured 2026-10-06 on `rpi`: a 31k-token prompt
  peaks at 5.5 GiB (under the 7g `mem_limit`) and takes ~19 min to
  prefill; 12k takes ~4 min.
* `dropParams` strips the four parameters a minimal server is most likely to
  reject. If requests come back `400`, extend that list before touching the
  server.

LibreChat's SSRF validator applies to *user-provided* base URLs only
(`baseURL: 'user_provided'`). This endpoint is admin-configured, so the
private address needs no `allowedAddresses` exemption.

## Web search (SearXNG via MCP)

`settings.yml` on `rpi` (host-only) keeps only DuckDuckGo and Wikipedia
(`use_default_settings.engines.keep_only`), sets `outgoing.request_timeout:
3.0` and `search.default_lang: "en"` — DuckDuckGo answers the
all-languages region (`wt-wt`) with a CAPTCHA. Searches take ~0.6–1.7 s.


LibreChat's built-in `webSearch` cannot run here: v0.8.7 (and v0.8.8) will
not enable it without a scraper — Firecrawl, Serper or Tavily, i.e. a paid
API or a self-hosted Firecrawl (API + Playwright + Redis + Postgres) that
the Pi cannot afford. Instead `librechat.yaml` declares an `mcpServers`
entry that spawns [`mcp-searxng`](https://github.com/ihor-sokoliuk/mcp-searxng)
as its own sandboxed container in [agent-tools](../agent-tools/README.md) (it used to be
an `npx` child of `librechat-1`, which meant fetching untrusted pages in the
container holding LibreChat's secrets, plus ~12 s of npm on every start). It
gives the model `searxng_web_search` (queries `searxng-1:8080`) and
`web_url_read` (fetches a page as Markdown — the scraper role).

Model specs, one per task, each attaching only the servers it needs:

| Spec | MCP servers | Tool-schema cost |
|---|---|---|
| `pi-agent` (default) | `searxng`, `shell` | ~0.5k tokens |
| `pi-browser` | `playwright` | ~4.3k tokens (~2 min extra per call) |
| `pi-files` | `filesystem`, `shell` | ~2.2k tokens |
| `pi-organiser` | `calendar`, `tasks`, `notes` | ~1.2k tokens |
| `pi-feeds` | `freshrss`, `calendar-add` | ~1k tokens |

`pi-organiser` and `pi-feeds` put `Today is {{current_date}}` in the prompt
(LibreChat fills it in, in the user's time zone): the model can't place
"tomorrow" or "Friday" without it, and it changes only once a day, so the
prefix cache survives. `pi-feeds` reads FreshRSS and files dated items into
Nextcloud through the add-only `calendar-add` endpoint, passing each item's
link so nothing is filed twice (see [agent-connectors](../agent-connectors/README.md)).

Since 2026-10-10 only `searxng` (Web Search), `wikipedia`
([Wikipedia](../agent-tools/README.md#wikipedia-mcp-wikipedia)), `calc`
([Calculator](../agent-tools/README.md#calculator-mcp-calc)), `playwright` (Playwright Browser)
and `sandbox` (Debian Sandbox) are active; the others are
commented out in `mcpServers` and `mcpSettings.allowedAddresses`, so the
specs and agents that use them lose those tools until they're uncommented.

LibreChat v0.8.7 cannot filter individual MCP tools in config; for finer
cuts build an agent in the UI and tick only the tools it needs.

## File search (RAG)

`RAG_API_URL` turns on LibreChat's file search: a file uploaded *for file
search* goes to `rag_api`, which splits it into ~1500-character chunks,
embeds each on `litert-lm-2` (EmbeddingGemma 2 Text 270M, 768-d) and stores
the vectors in `vectordb` (pgvector). The model then gets a `file_search`
tool that embeds its query the same way and returns the closest chunks.
"Upload as Text" still pastes the whole file into the prompt; with
`RAG_API_URL` set, LibreChat parses it through `rag_api`'s `/text` first,
falling back to its own parser if `rag_api` is down.

* **Separate LiteRT-LM server.** `serve` handles one request at a time, so
  embeddings run on `litert-lm-2` and never queue behind a chat reply on
  `litert-lm-1`. It is unpublished, CPU-only, capped at 1g
  (`LITERT_EMBED_MEM_LIMIT`), and downloads its 157 MB model from Hugging
  Face on first start.
* **`RAG_CHECK_EMBEDDING_CTX_LENGTH=false` is required.** Left at rag_api's
  default, langchain sends tiktoken token ids instead of text, LiteRT-LM
  embeds each id as a number string, and every vector is noise without an
  error. Checked 2026-10-09 with the same client and model: an aurora
  query scored an unrelated pasta paragraph 0.937 vs the aurora paragraph
  0.933; with `false`, 0.604 vs 0.862.
* **Task prefixes.** EmbeddingGemma 2 is trained on prefixed input
  (`task: search result | query: …` for queries, `title: … | text: …` for
  documents) and rag_api sends plain text. `rag-api/run_with_prefixes.py`,
  mounted read-only and run as the container's `command`, patches
  langchain's `OpenAIEmbeddings` to add `EMBEDDINGS_QUERY_PREFIX` /
  `EMBEDDINGS_DOCUMENT_PREFIX` (compose `environment:`), then runs
  rag_api's `main.py`. Chunks carry no usable title, so documents get
  `title: none`. Checked 2026-10-09 (8 passages, 8 paraphrased questions):
  both forms found every passage, but the gap to the nearest wrong passage
  went from +0.117 to +0.150 on average and from +0.030 to +0.074 at worst.
  The wrapper exits at startup if a prefix variable is unset; recheck it
  when bumping `RAG_API_TAG` (it relies on rag_api building
  `OpenAIEmbeddings` at import and serving `uvicorn.run(app)` in-process).
* **1024 tokens per input.** LiteRT-LM's EmbeddingGemma 2 rejects a longer
  input (`Input sequence length (…) exceeds maximum supported signature
  length (1024)` in its log) and answers 500 for the whole request, which
  used to fail the upload. A 1500-character chunk fits prose,
  but not digits: Gemma's tokenizer splits them, and a list of
  years is ~1 token per character (1509 for 1500). The wrapper retries a
  failed request one input at a time and cuts an input that still fails to
  80% until it embeds, logging `embedded the first N of M characters`. The
  stored chunk stays whole; only its vector misses the cut tail. Checked
  2026-10-09 against `litert-lm-rpi5:local` with the pinned rag_api image:
  the 1500-character year list embedded at 960 characters, every other
  vector identical to a direct call.
* **The model and prefixes are fixed.** Vectors from different models or
  prefixes don't compare. Changing `litert-lm-2`'s model, rag_api's
  `EMBEDDINGS_MODEL` or either prefix means deleting and re-uploading every
  file-search file.
* **Speed.** The model card says ~160 ms per 128-token input on a Pi 5
  CPU; measured, a full 1400-character chunk takes 2.2–4.1 s and a short
  query 0.23–0.7 s, depending on what else the Pi is doing (2026-10-09
  during a download; 2026-10-10 through `rag_api` with Immich using ~1.7
  cores). A long PDF takes several minutes to index.
* `rag_api` reads `JWT_SECRET` from `.env.librechat` to check LibreChat's
  tokens; without it, it serves every request unauthenticated.

## Conventions and deliberate divergences

Against upstream's `docker-compose.yml` (pinned to `v0.8.7`):

| Upstream | Here | Why |
|---|---|---|
| binds `./.env` to `/app/.env` | `env_file:` | matches the repo's `.env` / `.env.<service>` split |
| `rag_api` + `vectordb` | kept, renamed `librechat-*-1`, on `internal` (+ `litert` for `rag_api`) | embeddings come from `litert-lm-2`, not OpenAI; see [File search](#file-search-rag) |
| `vectordb` password `mypassword` | trust auth | same reasoning as Mongo's `--noauth`, below |
| `admin-panel` | omitted | would need a second published port and its own session secret |
| nginx `client` | omitted | the port is published directly |
| one default bridge | `external` / `internal` / `litert` / `searxng` | Mongo and Meilisearch get no route off the host |
| `:latest` everywhere | pinned tags | digests are in comments beside each `image:` |

`mongod --noauth` is upstream's default and is kept. It is only defensible
because `mongodb` sits on `librechat-internal` (`internal: true`) and
publishes no port — do not move it onto `external`. The same holds for
`vectordb`, which runs Postgres with `POSTGRES_HOST_AUTH_METHOD=trust`.

The config schema version is tied to the app version: `v0.8.7` expects
`librechat.yaml` `version: 1.3.13`. Bump `TAG` and that line together.

## The Qwen endpoint (the default model)

New chats default to Qwen3.5-9B on the workstation's GPU, LibreChat's `Qwen`
custom endpoint (`QWEN_BASE_URL`, `QWEN_API_KEY` in `.env.librechat`). The
server, the `qwen-chat` spec and its context settings are in
[workstation/llama-qwen](../../../workstation/llama-qwen/README.md).

## MCP tool servers (agent-tools)

Besides `searxng` (stdio, inside `librechat-1`), `librechat.yaml` registers
the three [agent-tools](../agent-tools/README.md) servers over Streamable HTTP. Their
hostnames are listed in `mcpSettings.allowedAddresses`: LibreChat refuses
MCP URLs that resolve to private addresses otherwise.

Each MCP server's tool definitions are sent with every request and count
against the context, so each [model spec](#web-search-searxng-via-mcp)
attaches only what its task needs.

`interface.remoteAgents` turns on the agents API
(`POST /api/agents/v1/chat/completions`, `model` = agent id, `Authorization:
Bearer <LibreChat API key>`). Create the key under *Settings → API keys*;
that is what [n8n](https://n8n.io), a separate stack, calls.

## Still to do before this is fully production-ready

* **Set `deploy.resources`**. Commented out on all three services, per the
  template. LibreChat's Node process and Meilisearch will both happily grow.
* **`read_only: true`** is commented out on all three; it needs per-service
  testing.
* **Backups.** `data-mongo` holds every conversation and account. Once
  deployed it sits under `${DATA_ROOT}`, which Backrest copies — but live,
  without a `mongodump`, so add a dump sidecar like `pgdump-linkwarden`.
* **No TLS.** The UI is published as plain HTTP on `${APP_PORT}`. If it needs
  to leave the host, put it behind the Tailscale + Caddy variant the template
  ships (see `vaultwarden` for a worked example) rather than exposing 3080.
