"""Calculator tools for LibreChat agents: arithmetic, dates, counting letters.

gemma4-e2b miscounts and miscalculates even when it thinks step by step;
these return the exact answer instead (calc.py). Four tools with one-line
descriptions and hand-written schemas, to keep the prompt small. Pure
Python, no files, no network.

A second server shares the process: academic_calendar, NTUA's official
academic calendar (academic.py, data in ntua_calendar.json, baked into the
image). Its own endpoint, so LibreChat lists it as its own server and a
spec takes it without the calculator.

A third, /math/mcp: symbolic_math (sympy) and physics_units (pint) for
<feed> coursework (mathtools.py). They take untrusted text that can ask for
something huge, so each call runs in a child process forked from a
forkserver that has already imported sympy and pint, and is killed after
MATH_SECONDS.

Endpoints: /mcp (calculator), /academic/mcp (calendar), /math/mcp (maths);
Streamable HTTP, stateless, JSON responses; /healthz.
"""

import contextlib
import datetime
import logging
import multiprocessing
import os
from typing import Annotated, Literal

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

import academic
import calc
import mathchild


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
    (calculate, "Exact arithmetic: + - * / ^ %, sqrt, log, sin, factorial, pi; clock times and durations (h, min, days).", {
        "properties": {"expression": {"type": "string", "description": "e.g. (17*23)^2, 15% of 80 or 06:40 - 2h - 45min"}},
        "required": ["expression"],
    }),
    (date_diff, "Days, weeks, months and years between two dates, with weekdays and how many are weekdays.", {
        "properties": {"start": DATE, "end": DATE},
        "required": ["start", "end"],
    }),
    (date_add, "The date (and weekday) some days, weeks, months or years after a date; negative goes back.", {
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

calendar = MCPServer("academic", log_level="WARNING")


def academic_calendar(query: str = "", start: str = "", end: str = "") -> str:
    try:
        return academic.search(query[:200], start[:100], end[:100], _today())
    except academic.CalendarError as exc:
        raise ToolError(f"academic_calendar: {exc}") from exc
    finally:
        log.info("academic_calendar")


calendar.add_tool(
    academic_calendar,
    description="NTUA's official academic calendar (every school, <feed> too): semesters, exam periods, "
                "registration and thesis deadlines, days with no classes. No arguments: what is on now and next.",
    structured_output=False,
    annotations=ANNOTATIONS,
)
calendar._tool_manager.get_tool("academic_calendar").parameters = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "what to find, e.g. winter exams, Easter break, thesis deadline; add postgraduate for master's"},
        "start": {**DATE, "description": "YYYY-MM-DD or today; alone, that one day"},
        "end": DATE,
    },
}


maths = MCPServer("math", log_level="WARNING")
MATH_SECONDS = int(os.environ.get("MATH_SECONDS", "15"))
_forks = multiprocessing.get_context("forkserver")
_forks.set_forkserver_preload(["mathtools"])
_math_slots = anyio.Semaphore(2)


def _in_child(name: str, args: tuple) -> str:
    receive, send = _forks.Pipe(duplex=False)
    child = _forks.Process(target=mathchild.run, args=(send, name, args, MATH_SECONDS + 5), daemon=True)
    child.start()
    send.close()
    try:
        if not receive.poll(MATH_SECONDS):
            raise ToolError(f"{name}: took longer than {MATH_SECONDS} s; try a simpler or smaller form")
        status, text = receive.recv()
    except EOFError:
        raise ToolError(f"{name}: stopped (out of memory or time); try a simpler or smaller form") from None
    finally:
        if child.is_alive():
            child.kill()
        child.join(2)
        receive.close()
    if status != "ok":
        raise ToolError(f"{name}: {text}")
    return text


async def _isolated(name: str, *args) -> str:
    try:
        async with _math_slots:
            return await anyio.to_thread.run_sync(_in_child, name, args)
    finally:
        log.info("%s", name)


def _warm_forkserver() -> None:
    """Start the forkserver (it imports sympy and pint, ~10 s on the Pi) now,
    not on a student's first question."""
    receive, send = _forks.Pipe(duplex=False)
    child = _forks.Process(target=mathchild.ping, args=(send,), daemon=True)
    child.start()
    send.close()
    receive.poll(120)
    child.join(5)
    receive.close()


Operation = Literal["evaluate", "simplify", "expand", "factor", "derivative", "integral", "limit", "solve",
                    "series", "determinant", "inverse", "eigenvalues", "eigenvectors", "rank", "rref"]


async def symbolic_math(
    operation: Annotated[Operation, "what to do"],
    expression: Annotated[str, "e.g. x^2*sin(x), x^2-5x+6=0, x+y=3; x-y=1, [[1,2],[3,4]]"],
    variable: str = "",
    at: str = "",
    order: int = 0,
) -> str:
    return await _isolated("symbolic_math", operation, expression[:MAX_CHARS], variable[:50], at[:100], order)


async def physics_units(quantity: str, to: str = "") -> str:
    return await _isolated("physics_units", quantity[:MAX_CHARS], to[:100])


MATH_TOOLS = [
    (symbolic_math, "Exact algebra and calculus: simplify, expand, factor, derivative, integral, limit, solve "
                    "equations or systems, series; matrices: determinant, inverse, eigenvalues, eigenvectors, rank, rref.", {
        "properties": {
            "operation": {"type": "string", "enum": list(Operation.__args__)},
            "expression": {"type": "string", "description": "e.g. x^2*sin(x), x^2-5x+6=0, x+y=3; x-y=1 or [[1,2],[3,4]]"},
            "variable": {"type": "string", "description": "e.g. x, or x,y to solve for; default: the only one"},
            "at": {"type": "string", "description": "limit or series point (0, oo, 0+) or integral bounds (0, pi)"},
            "order": {"type": "integer", "description": "derivative order or series terms"},
        },
        "required": ["operation", "expression"],
    }),
    (physics_units, "Physics units and constants: convert a quantity, or compute with constants such as speed_of_light, "
                    "planck_constant, hbar, elementary_charge, electron_mass, boltzmann_constant, avogadro_constant, "
                    "gravitational_constant, standard_gravity, vacuum_permittivity.", {
        "properties": {
            "quantity": {"type": "string", "description": "e.g. 72 km/h, 25 degC or planck_constant*speed_of_light/(500 nm)"},
            "to": {"type": "string", "description": "unit wanted, e.g. m/s, K, eV; empty: SI base units"},
        },
        "required": ["quantity"],
    }),
]
for fn, description, schema in MATH_TOOLS:
    maths.add_tool(fn, description=description, structured_output=False, annotations=ANNOTATIONS)
    maths._tool_manager.get_tool(fn.__name__).parameters = {"type": "object", **schema}


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
        for name, server in (("calc", mcp), ("academic", calendar), ("math", maths))
    }

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette):
        # A mounted app's own lifespan doesn't run; start the session managers here.
        async with mcp.session_manager.run(), calendar.session_manager.run(), maths.session_manager.run():
            await anyio.to_thread.run_sync(_warm_forkserver)
            yield

    async def healthz(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    uvicorn.run(
        Starlette(routes=[Route("/healthz", healthz), Mount("/academic", app=apps["academic"]), Mount("/math", app=apps["math"]), Mount("/", app=apps["calc"])], lifespan=lifespan),
        host="0.0.0.0",
        port=int(env("MCP_PORT", "8000")),
        log_level="warning",
        server_header=False,
    )


if __name__ == "__main__":
    main()
