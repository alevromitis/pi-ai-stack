"""Calculator tools for LibreChat agents: arithmetic, dates, counting letters.

gemma4-e2b miscounts and miscalculates even when it thinks step by step;
these return the exact answer instead (calc.py). Four tools with one-line
descriptions and hand-written schemas, to keep the prompt small. Pure
Python, no files, no network.

Endpoint: /mcp (Streamable HTTP, stateless, JSON responses); /healthz.
"""

import contextlib
import datetime
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

import calc


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name) or default
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


MAX_CHARS = 2000

log = logging.getLogger("calc")
mcp = MCPServer("calc", log_level="WARNING")


def _run(name: str, fn, *args) -> str:
    try:
        return fn(*args)
    except calc.CalcError as exc:
        raise ToolError(f"{name}: {exc}") from exc
    finally:
        # Tool name only: arguments may be personal.
        log.info("%s", name)


def _today() -> datetime.date:
    return datetime.date.today()  # in the container's TZ


def calculate(expression: str) -> str:
    return _run("calculate", calc.calculate, expression[:MAX_CHARS])


def date_diff(start: str, end: str) -> str:
    return _run("date_diff", calc.date_diff, start[:100], end[:100], _today())


def date_add(date: str, days: int = 0, weeks: int = 0, months: int = 0, years: int = 0) -> str:
    return _run("date_add", calc.date_add, date[:100], days, weeks, months, years, _today())


def count_text(text: str, find: str = "") -> str:
    return _run("count_text", calc.count_text, text[:MAX_CHARS], find[:100])


ANNOTATIONS = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
DATE = {"type": "string", "description": "YYYY-MM-DD or today"}
INT = {"type": "integer"}
# The description is what the model reads on every request: one line each.
# Schemas are hand-written: pydantic's titles cost tokens and tell a small
# model nothing.
TOOLS = [
    (calculate, "Exact arithmetic: + - * / ^ %, sqrt, log, sin, factorial, pi.", {
        "properties": {"expression": {"type": "string", "description": "e.g. (17*23)^2 or 15% of 80"}},
        "required": ["expression"],
    }),
    (date_diff, "Days, weeks, months and years between two dates, with weekdays. For \"until <date>\" without a year, use its next occurrence from today.", {
        "properties": {"start": DATE, "end": DATE},
        "required": ["start", "end"],
    }),
    (date_add, "The date and weekday some days, weeks, months or years after a date (negative goes back; none: that date's weekday).", {
        "properties": {"date": DATE, "days": INT, "weeks": INT, "months": INT, "years": INT},
        "required": ["date"],
    }),
    (count_text, "Count characters, letters and words in a text, or how often a letter or string appears.", {
        "properties": {"text": {"type": "string"}, "find": {"type": "string", "description": "letter or string to count"}},
        "required": ["text"],
    }),
]
for fn, description, schema in TOOLS:
    mcp.add_tool(fn, description=description, structured_output=False, annotations=ANNOTATIONS)
    mcp._tool_manager.get_tool(fn.__name__).parameters = {"type": "object", **schema}


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
