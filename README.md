# pi-ai-stack

A self-hosted AI assistant built from Docker Compose stacks.
[LibreChat](https://www.librechat.ai) is the chat UI, with two models
behind it:

- **Gemma 4 E2B** on a Raspberry Pi 5's CPU, served by
  [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM).
- **Qwen3.5-9B** on a desktop GPU, served by llama.cpp (Vulkan).

[MCP](https://modelcontextprotocol.io) tool servers give the models web
search, offline Wikipedia, a calculator, translation, a browser, and a Debian
sandbox per chat.

```
                         browser
                            │
              ┌─────────────▼──────────────┐
              │ LibreChat  (Pi)            │── MongoDB, Meilisearch,
              └──┬──────────┬──────────┬───┘   rag_api → pgvector
                 │          │          │
   litert-lm-1: Gemma 4 E2B │          llama-server: Qwen3.5-9B
   litert-lm-2: EmbeddingGemma        (desktop GPU, over Tailscale)
                 │
     MCP: agent-tools    SearXNG search, Wikipedia (Kiwix ZIM), calculator,
                         translation (Argos), Playwright browser
          agent-sandbox  one Debian container per chat, kept between visits
          agent-connectors  FreshRSS, Nextcloud Calendar/Tasks/Notes (off)
```

## What's here

| Path | What it is |
|---|---|
| [`docker/prod/librechat`](docker/prod/librechat/README.md) | the chat UI: endpoints, model specs, web search, file search (RAG) |
| [`docker/prod/litert-lm`](docker/prod/litert-lm/README.md) | LiteRT-LM on the Pi, and the patches its image applies to the server |
| [`docker/prod/agent-tools`](docker/prod/agent-tools/README.md) | the MCP servers: search, Wikipedia, calculator, translation, browser, shell, files |
| [`docker/prod/agent-sandbox`](docker/prod/agent-sandbox/README.md) | a Debian sandbox per chat, behind a Docker socket proxy |
| [`docker/prod/agent-connectors`](docker/prod/agent-connectors/README.md) | MCP servers over FreshRSS and Nextcloud (CalDAV, Notes) |
| `docker/prod/searxng` | the metasearch engine `mcp-searxng` queries |
| [`workstation/llama-qwen`](workstation/llama-qwen/README.md) | a systemd user unit for Qwen on the desktop GPU |
| [`template-docker`](template-docker/README.md) | the scaffold and conventions every stack follows |

The parts most worth reading:

- **The LiteRT-LM patches** ([litert-lm](docker/prod/litert-lm/README.md)).
  A prefix (KV) cache with system-prompt snapshots, so a chat's next turn
  prefills only the new messages. Speculative decoding (MTP), with a fix
  for an upstream bug that dropped stop tokens from the cache. Sources
  lines written by the server rather than the model. Automatic translation
  for chats in other languages. The `PLAN-*.md` files are the research
  notes behind the bigger changes.
- **Offline Wikipedia for a 2B model**
  ([agent-tools](docker/prod/agent-tools/README.md#wikipedia-mcp-wikipedia)):
  candidate selection, passage ranking and the measurements behind each
  choice.
- **Sandboxes kept per chat** ([agent-sandbox](docker/prod/agent-sandbox/README.md)):
  root and the network inside fixed limits, stopped when idle and resumed
  on the chat's next call, with the threat model spelled out.

## Running it

Each directory under `docker/prod/` is its own Compose stack. Its
`Makefile` includes the root `compose.mk`:

```bash
cd docker/prod/<stack>
cp .env.example .env               # and each .env.<name>.example to .env.<name>
make up                            # docker compose -f docker-compose.<stack>.yml up -d
```

Start them in this order: `searxng`, `agent-tools`, `agent-sandbox` (and
`agent-connectors` if wanted), `litert-lm`, then `librechat`. Each stack
owns the networks the next one joins. The locally built images target
arm64 (the Pi). Read
[`template-docker/README.md`](template-docker/README.md) first for the
`.env` rule, and each stack's README for its prerequisites and first run.

**Placeholders.** Host-specific values are written as placeholders:
`<tailnet>` (the Tailscale tailnet name), `<rpi-lan-ip>`,
`<workstation-tailnet-ip>`, `/home/<user>`. The docs call the two hosts
`rpi` and "the workstation".

**Secrets** are never committed, only the `.env*.example` templates. The
real env files are kept SOPS-encrypted (`scripts/secrets.sh`, `make
encrypt` / `make decrypt`). Those files and the `.sops.yaml` recipient
aren't part of this repository; bring your own age key and `.sops.yaml`.

## About this repository

This repository is exported automatically from a private one that also runs
the rest of a home server. That's why the docs mention stacks that aren't
here (n8n, Backrest, Immich, Gitea). Only the paths above are exported. The
history is rewritten:

- host names, addresses and similar values are replaced by the
  placeholders above, in every commit;
- the encrypted env files are dropped;
- commit dates are in UTC.

So commit hashes don't match the source.

## License

[MIT](LICENSE).
