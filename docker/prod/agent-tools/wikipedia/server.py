"""Offline English Wikipedia for LibreChat agents, from a Kiwix ZIM file.

Two endpoints over the same open ZIM, sized for two models:

  /mcp        gemma4-e2b on the Pi: one tool, wikipedia_search, the best 5
              passages of <=500 characters from the top 3 articles, then a
              Sources line of their links for the model to end its answer
              with (~2.1k characters, ~550 tokens).
  /large/mcp  Qwen on the workstation GPU: wikipedia_search with more and
              longer passages (WIKIPEDIA_LARGE_*, ~12k characters at most),
              plus wikipedia_read for an article's lead or one section.

Links go to the article on kiwix-serve when WIKIPEDIA_LINK_BASE points
there, else en.wikipedia.org. Measured on rpi 2026-10-09: ~0.2 s a search
with a cold page cache, ~340 MB RSS. No network: the ZIM is a read-only bind
mount.

Endpoints: /mcp, /large/mcp (Streamable HTTP, stateless, JSON responses);
/healthz.
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

from wiki import Wiki, article, render, render_article


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name) or default
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


RESULTS = int(env("WIKIPEDIA_RESULTS", "5"))
ARTICLES = int(env("WIKIPEDIA_ARTICLES", "3"))
PASSAGE_CHARS = int(env("WIKIPEDIA_PASSAGE_CHARS", "500"))
LARGE_RESULTS = int(env("WIKIPEDIA_LARGE_RESULTS", "10"))
LARGE_ARTICLES = int(env("WIKIPEDIA_LARGE_ARTICLES", "5"))
LARGE_PASSAGE_CHARS = int(env("WIKIPEDIA_LARGE_PASSAGE_CHARS", "1200"))
READ_CHARS = int(env("WIKIPEDIA_READ_CHARS", "12000"))
LINK_BASE = env("WIKIPEDIA_LINK_BASE", "https://en.wikipedia.org/wiki/")
MAX_QUERY_CHARS = 300

log = logging.getLogger("wikipedia")
wiki = Wiki(env("WIKIPEDIA_ZIM"))
# libzim's searchers aren't safe to share across threads, and the Pi gains
# nothing from parallel searches: one at a time.
lock = threading.Lock()
snapshot = datetime.date.fromisoformat(wiki.date).strftime("%B %Y") if wiki.date[:4].isdigit() else wiki.date

def _timed(name: str, run) -> str:
    started = time.monotonic()
    outcome = "ok"
    try:
        return run()
    except ToolError:
        outcome = "ToolError"
        raise
    except Exception as exc:
        outcome = type(exc).__name__
        log.exception("%s failed", name)
        raise ToolError(f"The Wikipedia {name.split('_')[1]} failed.") from exc
    finally:
        # Timing only: queries may be personal.
        log.info("%s %s %.0fms", name, outcome, (time.monotonic() - started) * 1000)


def build_server(results: int, articles: int, chars: int, read: bool) -> MCPServer:
    """One endpoint's tools. The small profile's wikipedia_search is the
    one gemma4-e2b was measured with; keep its text unchanged."""
    server = MCPServer("wikipedia", log_level="WARNING")

    def _search(query: str) -> str:
        with lock:
            # Names typed in lowercase are capitalised, so they're looked up by name.
            query = wiki.proper(query)
            return render(query, wiki.passages(query, articles), wiki.date, results, chars, LINK_BASE)

    async def wikipedia_search(query: str) -> str:
        query = " ".join(query.split())[:MAX_QUERY_CHARS]
        if not query:
            raise ToolError("Give a search query.")
        return await anyio.to_thread.run_sync(lambda: _timed("wikipedia_search", lambda: _search(query)))

    # The description is what the model reads on every request: one line.
    server.add_tool(
        wikipedia_search,
        description=f"Search an offline copy of English Wikipedia ({snapshot}). Returns the {results} most relevant passages with links.",
        structured_output=False,
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    )
    # Hand-written: pydantic's titles cost tokens and tell a small model nothing.
    server._tool_manager.get_tool("wikipedia_search").parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to look up: a name, topic or question."}},
        "required": ["query"],
    }
    if not read:
        return server

    def _read(title: str, section: str) -> str:
        with lock:
            found = article(wiki, title)
            if found is None:
                raise ToolError(f'No article titled "{title}". Search for it with wikipedia_search.')
            return render_article(found, section, wiki.date, READ_CHARS, LINK_BASE)

    async def wikipedia_read(title: str, section: str = "") -> str:
        title = " ".join(title.split())[:MAX_QUERY_CHARS]
        section = " ".join(section.split())[:MAX_QUERY_CHARS]
        if not title:
            raise ToolError("Give an article title.")
        return await anyio.to_thread.run_sync(lambda: _timed("wikipedia_read", lambda: _read(title, section)))

    server.add_tool(
        wikipedia_read,
        description=(
            f"Read one article of the same Wikipedia copy: its lead, or one section with its subsections"
            f" (up to {READ_CHARS} characters), plus the list of its sections. Use it when a search"
            " passage is cut short or the answer sits in a section the search didn't show."
        ),
        structured_output=False,
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
    )
    server._tool_manager.get_tool("wikipedia_read").parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "The article's title, as in a search result."},
            "section": {"type": "string", "description": "A section heading from the article's Sections list; omit for the lead."},
        },
        "required": ["title"],
    }
    return server


small = build_server(RESULTS, ARTICLES, PASSAGE_CHARS, read=False)
large = build_server(LARGE_RESULTS, LARGE_ARTICLES, LARGE_PASSAGE_CHARS, read=True)


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
    apps = {
        name: server.streamable_http_app(stateless_http=True, json_response=True, transport_security=security, host="0.0.0.0")
        for name, server in (("small", small), ("large", large))
    }

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette):
        # A mounted app's own lifespan doesn't run; start the session managers here.
        async with small.session_manager.run(), large.session_manager.run():
            yield

    async def healthz(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    log.info("serving %s (%s, %d articles)", env("WIKIPEDIA_ZIM"), wiki.date, wiki.archive.article_count)
    uvicorn.run(
        Starlette(routes=[Route("/healthz", healthz), Mount("/large", app=apps["large"]), Mount("/", app=apps["small"])], lifespan=lifespan),
        host="0.0.0.0",
        port=int(env("MCP_PORT", "8000")),
        log_level="warning",
        server_header=False,
    )


if __name__ == "__main__":
    main()
