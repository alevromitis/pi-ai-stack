"""Plumbing shared by the connector MCP servers.

Config from the environment, the tool decorator (read-only / delete gating,
compact schemas, upstream errors turned into short messages for the model),
date parsing in local time, text trimming, and the HTTP app: Streamable
HTTP, stateless, JSON responses, several servers mounted under one port.
"""

import contextlib
import functools
import logging
import os
import re
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from html.parser import HTMLParser
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route

log = logging.getLogger("connector")


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name) or default
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return default if not value else value in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


READ_ONLY = env_flag("READ_ONLY")
ALLOW_DELETE = env_flag("ALLOW_DELETE")
TZ = ZoneInfo(os.environ.get("TZ") or "UTC")


# --- Tools -------------------------------------------------------------------


def _compact(schema: Any) -> Any:
    """Drop what a small model doesn't need: pydantic titles, null branches."""
    if isinstance(schema, list):
        return [_compact(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: _compact(v) for k, v in schema.items() if k not in ("title", "properties")}
    if "properties" in schema:
        # Keys here are parameter names (one may well be "title"), not keywords.
        out["properties"] = {name: _compact(prop) for name, prop in schema["properties"].items()}
    branches = out.get("anyOf")
    if isinstance(branches, list):
        kept = [b for b in branches if b != {"type": "null"}]
        if len(kept) == 1:
            del out["anyOf"]
            out.update(kept[0])
    if out.get("default", "") is None:
        del out["default"]
    return out


def tool(*servers: MCPServer, kind: str = "read", service: str = "upstream") -> Callable:
    """Register an async tool on each server, unless READ_ONLY / ALLOW_DELETE
    rule it out.

    kind is "read", "write" or "delete". The docstring is the description the
    model sees: keep it to a line or two.
    """

    def decorator(fn: Callable) -> Callable:
        if kind != "read" and READ_ONLY:
            return fn
        if kind == "delete" and not ALLOW_DELETE:
            return fn

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> str:
            started = time.monotonic()
            outcome = "ok"
            try:
                return await fn(*args, **kwargs)
            except ToolError as exc:
                outcome = f"refused: {exc}"
                raise
            except httpx.HTTPStatusError as exc:
                outcome = f"HTTP {exc.response.status_code}"
                raise ToolError(f"{service} answered HTTP {exc.response.status_code}.") from exc
            except httpx.HTTPError as exc:
                outcome = type(exc).__name__
                raise ToolError(f"{service} is unreachable ({type(exc).__name__}).") from exc
            finally:
                # Tool name and timing only: arguments may be personal data.
                log.info("%s %s %.0fms", fn.__name__, outcome, (time.monotonic() - started) * 1000)

        for server in servers:
            server.add_tool(
                wrapper,
                structured_output=False,
                annotations=ToolAnnotations(
                    readOnlyHint=kind == "read",
                    destructiveHint=kind == "delete",
                    idempotentHint=kind != "write",
                    openWorldHint=False,
                ),
            )
            registered = server._tool_manager.get_tool(fn.__name__)
            registered.parameters = _compact(registered.parameters)
        return fn

    return decorator


# --- Dates (local time, TZ) --------------------------------------------------

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_TIME = re.compile(r"^(\d{1,2}):(\d{2})$")
DATE_HELP = "Use YYYY-MM-DD, YYYY-MM-DDTHH:MM, 'today', 'tomorrow' or a weekday, optionally followed by HH:MM."


def now() -> datetime:
    return datetime.now(TZ)


def parse_when(value: str, field: str) -> date | datetime:
    """A date (all-day) or an aware datetime in TZ, from what a model writes."""
    text = value.strip()
    lower = text.lower()
    day_word, _, clock = lower.partition(" ")
    today = now().date()
    day: date | None = None
    if day_word == "today":
        day = today
    elif day_word == "tomorrow":
        day = today + timedelta(days=1)
    elif day_word in _WEEKDAYS:
        # The next one after today: "friday" on a Friday means next week.
        ahead = (_WEEKDAYS.index(day_word) - today.weekday() - 1) % 7 + 1
        day = today + timedelta(days=ahead)
    if day is not None:
        clock = clock.strip()
        if not clock:
            return day
        match = _TIME.match(clock)
        if not match:
            raise ToolError(f"{field}: can't read {value!r}. {DATE_HELP}")
        return datetime(day.year, day.month, day.day, int(match[1]), int(match[2]), tzinfo=TZ)
    try:
        if len(text) == 10:
            return date.fromisoformat(text)
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ToolError(f"{field}: can't read {value!r}. {DATE_HELP}") from None
    return parsed.replace(tzinfo=TZ) if parsed.tzinfo is None else parsed.astimezone(TZ)


def start_of(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=TZ)


def fmt_day(day: date) -> str:
    return day.strftime("%a %Y-%m-%d")


def fmt_when(value: date | datetime) -> str:
    if isinstance(value, datetime):
        return value.astimezone(TZ).strftime("%a %Y-%m-%d %H:%M")
    return fmt_day(value)


# --- Text --------------------------------------------------------------------


class _TextExtractor(HTMLParser):
    _BLOCK = {"p", "div", "br", "ul", "ol", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "section", "article"}
    _SKIP = {"script", "style", "noscript", "template", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skipping = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self.skipping += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self.skipping = max(0, self.skipping - 1)
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skipping:
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    parser.feed(markup)
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}\n[... cut at {limit} of {len(text)} characters]"


def one_line(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def pick(name: str, options: list[str], what: str) -> str:
    """Match a name a model typed: exact, case-insensitive, then unique substring."""
    lowered = name.strip().lower()
    for option in options:
        if option.lower() == lowered:
            return option
    partial = [o for o in options if lowered in o.lower()]
    if len(partial) == 1:
        return partial[0]
    raise ToolError(f"No {what} {name!r}. Available: {', '.join(options) or 'none'}.")


# --- HTTP app ----------------------------------------------------------------


def new_server(name: str) -> MCPServer:
    return MCPServer(name, log_level="WARNING", warn_on_duplicate_tools=False)


def serve(servers: dict[str, MCPServer]) -> None:
    """Serve each server at /<path>/mcp, plus /healthz, on MCP_PORT."""
    # MCPServer already configured the root logger (at WARNING, which keeps
    # the SDK quiet); one INFO line per tool call is ours.
    log.setLevel(logging.INFO)
    # Requests come only from LibreChat, by container name. The Host allowlist
    # stops DNS rebinding; any Origin (i.e. any browser) is refused.
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[h.strip() for h in env("MCP_ALLOWED_HOSTS").split(",") if h.strip()],
        allowed_origins=[],
    )
    apps = {
        path: server.streamable_http_app(
            stateless_http=True, json_response=True, transport_security=security, host="0.0.0.0"
        )
        for path, server in servers.items()
    }

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette):
        # Mounted apps' own lifespans don't run; start each session manager here.
        async with contextlib.AsyncExitStack() as stack:
            for server in servers.values():
                await stack.enter_async_context(server.session_manager.run())
            yield

    async def healthz(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    routes: list[Route | Mount] = [Route("/healthz", healthz)]
    routes += [Mount(f"/{path}", app=app) for path, app in apps.items()]
    uvicorn.run(
        Starlette(routes=routes, lifespan=lifespan),
        host="0.0.0.0",
        port=env_int("MCP_PORT", 8000),
        log_level="warning",
        server_header=False,
    )
