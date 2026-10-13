"""Translation tool for LibreChat agents, offline (translate.py).

The models work best in English: Wikipedia is English, search does best in
English, and gemma4-e2b reasons worse in other languages. With this tool a
model translates a message into English, works in English, and translates
its answer back. One tool with a one-line description, to keep the prompt
small. No network; models are read from a read-only mount.

Endpoint: /mcp (Streamable HTTP, stateless, JSON responses); /healthz.
"""

import contextlib
import logging
import os

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route

import translate as tr


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name) or default
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


MAX_CHARS = 4000

log = logging.getLogger("translate")
mcp = MCPServer("translate", log_level="WARNING")
engine = tr.Translator(
    env("TRANSLATE_MODELS"),
    keep_loaded=int(env("TRANSLATE_KEEP_LOADED", "2")),
    threads=int(env("TRANSLATE_THREADS", "2")),
)


def translate(text: str, target: str = "en", source: str = "") -> str:
    try:
        src, dst, out = engine.translate(text[:MAX_CHARS], target, source)
    except tr.TranslateError as exc:
        raise ToolError(f"translate: {exc}") from exc
    finally:
        # Language pair only: the text may be personal.
        log.info("translate %s->%s", source or "auto", target)
    return f"{tr.NAMES[src]} -> {tr.NAMES[dst]}:\n{out}"


ANNOTATIONS = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
# The description is what the model reads on every request: one line.
mcp.add_tool(
    translate,
    description="Translate text between English and 30 languages (Greek, Spanish, Chinese, Arabic...); says the source language.",
    structured_output=False,
    annotations=ANNOTATIONS,
)
mcp._tool_manager.get_tool("translate").parameters = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "target": {"type": "string", "description": "language code, default en"},
        "source": {"type": "string", "description": "language code; omit to detect"},
    },
    "required": ["text"],
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

    uvicorn.run(
        Starlette(routes=[Route("/healthz", healthz), Mount("/", app=app)], lifespan=lifespan),
        host="0.0.0.0",
        port=int(env("MCP_PORT", "8000")),
        log_level="warning",
        server_header=False,
    )


if __name__ == "__main__":
    main()
