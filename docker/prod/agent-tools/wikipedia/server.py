"""Offline English Wikipedia for LibreChat agents, from a Kiwix ZIM file.

One tool, wikipedia_search: the best 5 passages for a query from the top 3
articles, then a Sources line of their links for the model to end its
answer with (~2.1k characters, ~550 tokens). Links go to the article on
kiwix-serve when WIKIPEDIA_LINK_BASE points there, else en.wikipedia.org. Measured on rpi 2026-10-09: ~0.2 s a search with a cold page
cache, ~340 MB RSS. No network: the ZIM is a read-only bind mount.

Endpoint: /mcp (Streamable HTTP, stateless, JSON responses); /healthz.
"""

import contextlib
import datetime
import logging
import os
import threading
import time

import anyio
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route

from wiki import Wiki, render


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name) or default
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


RESULTS = int(env("WIKIPEDIA_RESULTS", "5"))
ARTICLES = int(env("WIKIPEDIA_ARTICLES", "3"))
PASSAGE_CHARS = int(env("WIKIPEDIA_PASSAGE_CHARS", "500"))
LINK_BASE = env("WIKIPEDIA_LINK_BASE", "https://en.wikipedia.org/wiki/")
MAX_QUERY_CHARS = 300

log = logging.getLogger("wikipedia")
wiki = Wiki(env("WIKIPEDIA_ZIM"))
# libzim's searchers aren't safe to share across threads, and the Pi gains
# nothing from parallel searches: one at a time.
lock = threading.Lock()
snapshot = datetime.date.fromisoformat(wiki.date).strftime("%B %Y") if wiki.date[:4].isdigit() else wiki.date

mcp = MCPServer("wikipedia", log_level="WARNING")


def _search(query: str) -> str:
    with lock:
        # Names typed in lowercase are capitalised, so they're looked up by name.
        query = wiki.proper(query)
        return render(query, wiki.passages(query, ARTICLES), wiki.date, RESULTS, PASSAGE_CHARS, LINK_BASE)


async def wikipedia_search(query: str) -> str:
    started = time.monotonic()
    query = " ".join(query.split())[:MAX_QUERY_CHARS]
    if not query:
        raise ToolError("Give a search query.")
    outcome = "ok"
    try:
        return await anyio.to_thread.run_sync(_search, query)
    except Exception as exc:
        outcome = type(exc).__name__
        log.exception("search failed")
        raise ToolError("The Wikipedia search failed.") from exc
    finally:
        # Timing only: queries may be personal.
        log.info("wikipedia_search %s %.0fms", outcome, (time.monotonic() - started) * 1000)


# The description is what the model reads on every request: one line.
mcp.add_tool(
    wikipedia_search,
    description=f"Search an offline copy of English Wikipedia ({snapshot}). Returns the {RESULTS} most relevant passages with links.",
    structured_output=False,
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
)
# Hand-written: pydantic's titles cost tokens and tell a small model nothing.
_tool = mcp._tool_manager.get_tool("wikipedia_search")
_tool.parameters = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "What to look up: a name, topic or question."}},
    "required": ["query"],
}


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    log.setLevel(logging.INFO)
    # Requests come only from LibreChat, by container name. The Host allowlist
    # stops DNS rebinding; any Origin (i.e. any browser) is refused.
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[h.strip() for h in env("MCP_ALLOWED_HOSTS").split(",") if h.strip()],
        allowed_origins=[],
    )
    app = mcp.streamable_http_app(stateless_http=True, json_response=True, transport_security=security, host="0.0.0.0")

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette):
        # A mounted app's own lifespan doesn't run; start its session manager here.
        async with mcp.session_manager.run():
            yield

    async def healthz(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    log.info("serving %s (%s, %d articles)", env("WIKIPEDIA_ZIM"), wiki.date, wiki.archive.article_count)
    uvicorn.run(
        Starlette(routes=[Route("/healthz", healthz), Mount("/", app=app)], lifespan=lifespan),
        host="0.0.0.0",
        port=int(env("MCP_PORT", "8000")),
        log_level="warning",
        server_header=False,
    )


if __name__ == "__main__":
    main()
