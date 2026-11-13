"""Plumbing shared by the connector MCP servers.

Config from the environment, the tool decorator (read-only / delete gating,
compact schemas, upstream errors turned into short messages for the model),
date parsing in local time, text trimming, and the HTTP app: Streamable
HTTP, stateless, JSON responses, several servers mounted under one port.
With MCP_TRANSPORT=stdio, a process serves all its tools as one server on
stdin/stdout instead, for a client that spawns it (Claude Code on the
workstation).
"""

import contextlib
import functools
import logging
import os
import re
import time
import unicodedata
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
# The user's own words, passed through by the model: Greek day words (accents
# dropped) and month names in English or Greek, matched by their start.
_GREEK_DAYS = {"δευτερα": "monday", "τριτη": "tuesday", "τεταρτη": "wednesday", "πεμπτη": "thursday",
               "παρασκευη": "friday", "σαββατο": "saturday", "κυριακη": "sunday",
               "σημερα": "today", "αυριο": "tomorrow", "χθες": "yesterday", "επομενη": "next", "αυτη": "this"}
_MONTHS = [("jan", 1), ("feb", 2), ("mar", 3), ("apr", 4), ("may", 5), ("jun", 6), ("jul", 7), ("aug", 8),
           ("sep", 9), ("oct", 10), ("nov", 11), ("dec", 12), ("ιαν", 1), ("φεβ", 2), ("μαρ", 3), ("απρ", 4),
           ("μαι", 5), ("μαη", 5), ("ιουν", 6), ("ιουλ", 7), ("αυγ", 8), ("σεπ", 9), ("οκτ", 10), ("νοε", 11),
           ("δεκ", 12)]
_TIME = re.compile(r"^(\d{1,2}):(\d{2})$")
# "9am", "5:30pm", "17h": a clock the model may copy from the user's words.
_CLOCK_WORD = re.compile(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm|h)$")
_OFFSET = re.compile(r"^(?:\+\s*(\d+)\s*([dw])|in\s+(\d+)\s+(day|days|week|weeks))$")
DATE_HELP = (
    "Use YYYY-MM-DD, YYYY-MM-DDTHH:MM, 'today', 'tomorrow', 'yesterday', a weekday ('friday', 'next friday',"
    " 'this friday'), '12 november', 'next week', '+3d', '+2w' or 'in 3 days', optionally followed by a time"
    " (HH:MM, 9am, 5pm)."
)


def now() -> datetime:
    return datetime.now(TZ)


def _plain(text: str) -> str:
    """Lower case, accents dropped, Greek day words in English."""
    folded = "".join(c for c in unicodedata.normalize("NFD", text.lower()) if not unicodedata.combining(c))
    return " ".join(_GREEK_DAYS.get(w, w) for w in folded.replace(",", " ").split())


def _month(word: str) -> int | None:
    for prefix, number in sorted(_MONTHS, key=lambda m: -len(m[0])):
        if word.startswith(prefix):
            return number
    return None


def _named_date(words: str, today: date) -> date | None:
    """'12 november', 'november 12', 'thursday 12 nov 2026': a day and a month
    name, an optional weekday (ignored) and year; without a year, the next
    such date from today."""
    parts = [w.rstrip(".") for w in words.split() if w.rstrip(".") not in _WEEKDAYS]
    days = [w for w in parts if w.isdigit() and 1 <= int(w) <= 31 and len(w) <= 2]
    years = [w for w in parts if w.isdigit() and len(w) == 4]
    months = [m for m in (_month(w) for w in parts if not w.isdigit()) if m]
    if len(days) != 1 or len(months) != 1 or len(parts) != 2 + len(years) or len(years) > 1:
        return None
    try:
        found = date(int(years[0]) if years else today.year, months[0], int(days[0]))
        if not years and found < today:
            found = found.replace(year=today.year + 1)
    except ValueError:
        return None
    return found


def _relative_day(words: str, today: date) -> date | None:
    """The day a relative phrase names, or None. A bare or "next" weekday is
    the next one after today ("friday" on a Friday means next week); "this"
    weekday includes today. "next week" is next week's Monday."""
    if words == "today":
        return today
    if words == "tomorrow":
        return today + timedelta(days=1)
    if words == "yesterday":
        return today - timedelta(days=1)
    if words == "next week":
        return today + timedelta(days=7 - today.weekday())
    first, _, rest = words.partition(" ")
    if first in ("this", "next") and rest in _WEEKDAYS:
        target = _WEEKDAYS.index(rest)
        if first == "this":
            return today + timedelta(days=(target - today.weekday()) % 7)
        words = rest
    if words in _WEEKDAYS:
        return today + timedelta(days=(_WEEKDAYS.index(words) - today.weekday() - 1) % 7 + 1)
    match = _OFFSET.match(words)
    if match:
        count = int(match[1] or match[3])
        unit = (match[2] or match[4])[0]
        return today + timedelta(days=count * (7 if unit == "w" else 1))
    return _named_date(words, today)


def _clock(text: str) -> tuple[int, int] | None:
    match = _TIME.match(text)
    if match:
        hour, minute = int(match[1]), int(match[2])
    else:
        match = _CLOCK_WORD.match(text)
        if not match:
            return None
        hour, minute, suffix = int(match[1]), int(match[2] or 0), match[3]
        if suffix == "pm" and hour < 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0
    return (hour, minute) if hour < 24 and minute < 60 else None


def parse_when(value: str, field: str) -> date | datetime:
    """A date (all-day) or an aware datetime in TZ, from what a model writes."""
    text = " ".join(value.strip().split())
    lower = _plain(text)
    today = now().date()
    # A relative day, then optionally a clock: "next tuesday 10:00", "+2w 9am".
    words, clock = lower, ""
    parts = lower.rsplit(" ", 1)
    if len(parts) == 2 and _clock(parts[1].removeprefix("at ")) is not None:
        words, clock = parts[0].removesuffix(" at"), parts[1]
    day = _relative_day(words, today)
    if day is not None:
        if not clock:
            return day
        hour, minute = _clock(clock)
        return datetime(day.year, day.month, day.day, hour, minute, tzinfo=TZ)
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


def serve(
    servers: dict[str, MCPServer],
    name: str,
    on_start: Callable[[], Any] | None = None,
    routes: list[Route] | None = None,
) -> None:
    """Serve each server at /<path>/mcp, plus /healthz, on MCP_PORT; or, with
    MCP_TRANSPORT=stdio, all their tools as one server called name.

    routes: plain HTTP routes served next to them (not over stdio), for
    callers that are programs, not models.

    on_start: a coroutine function run once at HTTP startup, before requests
    (e.g. to fill tool schemas from the service); a failure is logged, not fatal."""
    # MCPServer already configured the root logger (at WARNING, which keeps
    # the SDK quiet); one INFO line per tool call is ours. It logs to stderr,
    # so stdio's stdout carries only the protocol.
    log.setLevel(logging.INFO)
    transport = os.environ.get("MCP_TRANSPORT", "http").strip().lower()
    if transport == "stdio":
        # One client spawns this process and gets every endpoint's tools. A
        # tool on several servers (calendar-add's) is the same function, so
        # one copy of each name is kept.
        tools = {t.name: t for server in servers.values() for t in server._tool_manager.list_tools()}
        MCPServer(name, log_level="WARNING", tools=list(tools.values())).run("stdio")
        return
    if transport != "http":
        raise SystemExit(f"MCP_TRANSPORT must be http or stdio, not {transport!r}")
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
            if on_start is not None:
                try:
                    await on_start()
                except Exception as exc:  # the service may be down; serve anyway
                    log.warning("startup hook failed: %s: %s", type(exc).__name__, exc)
            yield

    async def healthz(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    served: list[Route | Mount] = [Route("/healthz", healthz), *(routes or [])]
    served += [Mount(f"/{path}", app=app) for path, app in apps.items()]
    uvicorn.run(
        Starlette(routes=served, lifespan=lifespan),
        host="0.0.0.0",
        port=env_int("MCP_PORT", 8000),
        log_level="warning",
        server_header=False,
    )
