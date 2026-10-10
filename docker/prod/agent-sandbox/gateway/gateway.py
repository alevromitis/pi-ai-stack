"""MCP gateway to throwaway Debian sandboxes, one per LibreChat conversation.

LibreChat sends X-Conversation-ID with every tool call (librechat.yaml:
'{{LIBRECHAT_BODY_CONVERSATIONID}}', resolved per call). The first call in
a conversation starts a fresh container from SANDBOX_IMAGE; later calls
exec into it. A new chat is a new id, so it always starts clean. A sandbox
is removed when idle (SANDBOX_IDLE_MINUTES), too old (SANDBOX_MAX_HOURS),
over its disk allowance (SANDBOX_DISK_LIMIT), to make room for a newer one
(SANDBOX_MAX_ACTIVE, least recently used first), on `reset`, and - all of
them - when the gateway starts or stops. The next call from that chat gets
a fresh one and is told why.

This is the stack's only Docker API client (through the socket proxy), so
it is the privileged piece: it runs nothing itself, commands go into the
sandbox's exec argv untouched, and every sandbox gets the fixed HostConfig
built below from the environment.
"""

import asyncio
import contextlib
import functools
import json
import logging
import os
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import aiodocker
import uvicorn
from aiodocker.containers import DockerContainer
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route

log = logging.getLogger("sandbox-gateway")


# --- Config ------------------------------------------------------------------


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name) or default
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


_SIZE = re.compile(r"^(\d+(?:\.\d+)?)\s*([kmgt]?)i?b?$", re.IGNORECASE)


def env_size(name: str, default: str) -> int:
    """Bytes from Docker-style sizes: 512m, 1g, 4G, 50MiB (binary units)."""
    value = os.environ.get(name) or default
    match = _SIZE.match(value.strip())
    if not match:
        raise SystemExit(f"{name}: can't read size {value!r}")
    power = " kmgt".index(match[2].lower() or " ")
    return int(float(match[1]) * 1024**power)


IMAGE = env("SANDBOX_IMAGE", "agent-sandbox:local")
NETWORK = env("SANDBOX_NETWORK", "agent-sandbox-egress")
# RAM, and RAM + swap: the difference is what the kernel may page out.
MEMORY = env_size("SANDBOX_MEMORY", "256m")
MEMORY_SWAP = max(env_size("SANDBOX_MEMORY_SWAP", "1g"), MEMORY)
CPUS = float(env("SANDBOX_CPUS", "1"))
CPU_SHARES = env_int("SANDBOX_CPU_SHARES", 256)
PIDS = env_int("SANDBOX_PIDS", 512)
SHM = env_size("SANDBOX_SHM", "64m")
DISK_LIMIT = env_size("SANDBOX_DISK_LIMIT", "4g")
DISK_DEVICE = os.environ.get("SANDBOX_DISK_DEVICE", "").strip()
DISK_WRITE_BPS = env_size("SANDBOX_DISK_WRITE_BPS", "50m")
RUNTIME = os.environ.get("SANDBOX_RUNTIME", "").strip()
MAX_ACTIVE = env_int("SANDBOX_MAX_ACTIVE", 1)
IDLE_SECONDS = env_int("SANDBOX_IDLE_MINUTES", 30) * 60
MAX_AGE_SECONDS = env_int("SANDBOX_MAX_HOURS", 4) * 3600
DEFAULT_TIMEOUT = env_int("SANDBOX_DEFAULT_TIMEOUT", 300)
MAX_TIMEOUT = env_int("SANDBOX_MAX_TIMEOUT", 300)
OUTPUT_LIMIT = env_int("SANDBOX_OUTPUT_LIMIT", 8000)
# write_file goes through exec argv; Linux caps one argument at 128 KiB.
WRITE_LIMIT = 100_000
REAP_INTERVAL = 30

LABEL = "agent-sandbox.managed"
WORKDIR = "/work"


def human(size: float) -> str:
    for unit in ("B", "KiB", "MiB"):
        if size < 1024:
            return f"{round(size, 1):g} {unit}"
        size /= 1024
    return f"{round(size, 1):g} GiB"


# Asked for its RAM or CPUs, the model reported the host's from `free` and
# `lscpu`; the limits come with the first result instead.
NEW_NOTE = (
    f"[New sandbox: Debian 13, root, cwd /work. Limits: {CPUS:g} CPU, {human(MEMORY)} RAM"
    f" + {human(MEMORY_SWAP - MEMORY)} swap, {human(DISK_LIMIT)} disk (free and nproc show the host's).]"
)


def sandbox_config(conversation: str, user: str) -> dict[str, Any]:
    """The one container spec every sandbox gets. Root inside, Docker's
    default capabilities and seccomp profile minus a few, no mounts, no
    socket, no credentials; limits from the environment."""
    host: dict[str, Any] = {
        "Init": True,
        "NetworkMode": NETWORK,
        "Memory": MEMORY,
        # RAM + swap (memory.swap.max = MemorySwap - Memory).
        "MemorySwap": MEMORY_SWAP,
        "NanoCpus": int(CPUS * 1e9),
        "CpuShares": CPU_SHARES,
        "PidsLimit": PIDS,
        "ShmSize": SHM,
        "Ulimits": [{"Name": "nofile", "Soft": 8192, "Hard": 8192}],
        # Raw sockets (packet spoofing), device nodes, audit log.
        # ping still works: Docker opens ICMP datagram sockets to all groups.
        "CapDrop": ["NET_RAW", "MKNOD", "AUDIT_WRITE"],
        "SecurityOpt": ["no-new-privileges:true"],
        "RestartPolicy": {"Name": "no"},
        # Exec output comes back to the gateway; PID 1 is `sleep`.
        "LogConfig": {"Type": "none"},
    }
    if DISK_DEVICE:
        host["BlkioDeviceWriteBps"] = [{"Path": DISK_DEVICE, "Rate": DISK_WRITE_BPS}]
    if RUNTIME:
        host["Runtime"] = RUNTIME
    return {
        "Image": IMAGE,
        "Cmd": ["sleep", "infinity"],
        "Hostname": "sandbox",
        "WorkingDir": WORKDIR,
        "Labels": {
            LABEL: "1",
            "agent-sandbox.conversation": conversation,
            "agent-sandbox.user": user,
        },
        "HostConfig": host,
    }


# --- Sandboxes ---------------------------------------------------------------


@dataclass
class Sandbox:
    conversation: str
    container: DockerContainer
    name: str
    created: float = field(default_factory=time.monotonic)
    used: float = field(default_factory=time.monotonic)


class Sandboxes:
    def __init__(self) -> None:
        self.docker: aiodocker.Docker | None = None
        self.active: dict[str, Sandbox] = {}
        # Why a conversation's sandbox went away, for its next call.
        self.gone: dict[str, str] = {}
        self.purged = False
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        self.docker = aiodocker.Docker()  # DOCKER_HOST
        try:
            async with self.lock:
                await self._purge()
        except aiodocker.DockerError as exc:
            # Not fatal: retried before the first sandbox is created.
            log.warning("startup purge failed, Docker API unreachable? %s", exc)

    async def stop(self) -> None:
        if self.docker is None:
            return
        with contextlib.suppress(aiodocker.DockerError):
            async with self.lock:
                await self._purge()
        await self.docker.close()

    async def _purge(self) -> None:
        """Remove every sandbox, including those of a previous gateway run."""
        assert self.docker is not None
        filters = json.dumps({"label": [f"{LABEL}=1"]})
        containers = await self.docker.containers.list(all="1", filters=filters)
        for container in containers:
            with contextlib.suppress(aiodocker.DockerError):
                await container.delete(force=True)
        if containers:
            log.info("purged %d sandbox(es)", len(containers))
        self.active.clear()
        self.purged = True

    async def get(self, conversation: str, user: str) -> tuple[Sandbox, str]:
        """This conversation's sandbox, created if needed, and a note for the
        model when it is new."""
        async with self.lock:
            if not self.purged:
                await self._purge()
            box = self.active.get(conversation)
            if box is not None:
                problem = await self._problem(box)
                if problem is None:
                    box.used = time.monotonic()
                    return box, ""
                await self._drop(conversation, problem)
            box = await self._create(conversation, user)
        reason = self.gone.pop(conversation, None)
        if reason:
            return box, f"[Your previous sandbox was removed ({reason}). This is a fresh one: earlier files and installs are gone.]"
        return box, NEW_NOTE

    async def reset(self, conversation: str) -> bool:
        async with self.lock:
            existed = conversation in self.active
            await self._drop(conversation, "reset", remember=False)
            return existed

    async def _problem(self, box: Sandbox) -> str | None:
        """None if the container is still running, else what happened to it."""
        try:
            info = await box.container.show()
        except aiodocker.DockerError as exc:
            if exc.status == 404:
                return "it was deleted outside the gateway"
            raise
        state = info["State"]
        if state.get("Running"):
            return None
        if state.get("OOMKilled"):
            return f"it hit the {human(MEMORY_SWAP)} memory limit"
        return f"its container stopped (exit {state.get('ExitCode')})"

    async def _create(self, conversation: str, user: str) -> Sandbox:
        assert self.docker is not None
        while len(self.active) >= MAX_ACTIVE:
            oldest = min(self.active.values(), key=lambda b: b.used)
            await self._drop(oldest.conversation, "another chat needed its slot")
        name = f"agent-sandbox-{conversation[:8]}-{secrets.token_hex(2)}"
        # create + start, not containers.run(): that pulls on a missing image,
        # which the proxy refuses (403) and which isn't wanted anyway.
        try:
            container = await self.docker.containers.create(sandbox_config(conversation, user), name=name)
        except aiodocker.DockerError as exc:
            if exc.status == 404:
                raise ToolError(f"The sandbox image {IMAGE} is missing. (Admin: `make up` in agent-sandbox builds it.)") from exc
            raise
        try:
            await container.start()
        except aiodocker.DockerError:
            with contextlib.suppress(aiodocker.DockerError):
                await container.delete(force=True)
            raise
        box = Sandbox(conversation, container, name)
        self.active[conversation] = box
        log.info("created %s (%d active)", name, len(self.active))
        return box

    async def _drop(self, conversation: str, reason: str, remember: bool = True) -> None:
        box = self.active.pop(conversation, None)
        if box is None:
            return
        if remember:
            self.gone[conversation] = reason
            while len(self.gone) > 256:
                self.gone.pop(next(iter(self.gone)))
        with contextlib.suppress(aiodocker.DockerError):
            await box.container.delete(force=True)
        log.info("removed %s: %s", box.name, reason)

    async def reap(self) -> None:
        while True:
            await asyncio.sleep(REAP_INTERVAL)
            try:
                await self._reap_once()
            except Exception:
                log.exception("reaper")

    async def _reap_once(self) -> None:
        now = time.monotonic()
        verdicts: list[tuple[Sandbox, str]] = []
        # Disk usage is measured outside the lock: `size` walks the
        # container's whole writable layer and would hold up tool calls.
        for box in list(self.active.values()):
            if now - box.used > IDLE_SECONDS:
                verdicts.append((box, f"idle for over {IDLE_SECONDS // 60} minutes"))
            elif now - box.created > MAX_AGE_SECONDS:
                verdicts.append((box, f"older than {MAX_AGE_SECONDS // 3600} hours"))
            else:
                try:
                    info = await box.container.show(size="1")
                except aiodocker.DockerError:
                    continue
                written = info.get("SizeRw") or 0
                if written > DISK_LIMIT:
                    verdicts.append((box, f"it wrote {human(written)}, over the {human(DISK_LIMIT)} disk limit"))
        if not verdicts:
            return
        async with self.lock:
            for box, reason in verdicts:
                # Still the same sandbox (not reset and recreated meanwhile)?
                if self.active.get(box.conversation) is box:
                    await self._drop(box.conversation, reason)


sandboxes = Sandboxes()


# --- Exec --------------------------------------------------------------------

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")


def clean(raw: bytes) -> str:
    text = _ANSI.sub("", raw.decode("utf-8", "replace"))
    # Progress bars redraw with \r: keep what each line ended up as.
    return "\n".join(line.rstrip("\r").rsplit("\r", 1)[-1] for line in text.split("\n"))


# Stderr lines that report a failure; apt notices, git's "Cloning into" and
# download progress go to stderr too.
_ERRORISH = re.compile(
    r"No such file|not found|cannot|can't|[Ee]rror|ERROR|[Ff]ailed|FAILED|[Dd]enied|[Ii]nvalid"
    r"|[Ff]atal|Traceback|unexpected|[Uu]nable to|not permitted|[Ss]yntax|[Uu]sage:|unrecognized|missing"
)


@dataclass
class Result:
    exit_code: int | None
    output: str
    seconds: float
    hung: bool
    # Lines on stderr that report a failure, and the first of them.
    errors: int = 0
    first_error: str = ""


async def execute(box: Sandbox, argv: list[str], timeout: int) -> Result:
    """Run argv in the sandbox; stdout and stderr interleaved as they arrive.
    Keeps the head and tail of long output, within OUTPUT_LIMIT."""
    head_cap = OUTPUT_LIMIT // 4
    tail_cap = OUTPUT_LIMIT - head_cap
    head, tail = bytearray(), bytearray()
    total = 0
    stderr = bytearray()
    started = time.monotonic()
    exe = await box.container.exec(cmd=argv, stdout=True, stderr=True, workdir=WORKDIR)

    async def pump(stream: Any) -> None:
        nonlocal total
        while (message := await stream.read_out()) is not None:
            data = message.data
            total += len(data)
            if message.stream == 2 and len(stderr) < 65536:
                stderr.extend(data)
            room = head_cap - len(head)
            if room > 0:
                head.extend(data[:room])
                data = data[room:]
            if data:
                tail.extend(data)
                del tail[:-tail_cap]

    hung = False
    async with exe.start(detach=False) as stream:
        try:
            # `timeout` inside the sandbox ends the command; this catches
            # output held open past it (a background job without a redirect).
            await asyncio.wait_for(pump(stream), timeout + 15)
        except TimeoutError:
            hung = True
    seconds = time.monotonic() - started
    info = await exe.inspect()
    cut = total - len(head) - len(tail)
    output, end = clean(bytes(head)), clean(bytes(tail))
    if cut > 0:
        # Whole lines either side of the cut.
        output = output[: output.rfind("\n") + 1] or output
        end = end[end.find("\n") + 1 :] if "\n" in end else end
        output += f"[... {human(cut)} of output cut ...]\n"
    output += end
    box.used = time.monotonic()
    lines = [line.strip() for line in clean(bytes(stderr)).splitlines() if _ERRORISH.search(line)]
    return Result(
        None if info.get("Running") else info.get("ExitCode"), output.strip("\n"), seconds, hung,
        len(lines), lines[0][:120] if lines else "",
    )


# apt's "Unable to locate package X" and "Package 'X' has no installation
# candidate". gemma4-e2b gave up after the first one every time (2026-10-10:
# openjdk-17-jdk and neofetch, both missing from Debian 13), with or without
# a line about apt-cache search in this tool's description. It follows what
# a result says, so the result lists what the sandbox has instead.
_NO_PACKAGE = re.compile(r"E: (?:Unable to locate package (\S+)|Package '?([^'\s]+?)'? has no installation candidate)")
# Packages whose name or description holds the missing name, else whose
# names hold all its words but the version numbers (openjdk-17-jdk: openjdk,
# jdk). Prints NOLISTS before `apt-get update` has run.
_APT_SEARCH = (
    'ls /var/lib/apt/lists | grep -q Packages || { echo NOLISTS; exit; }\n'
    'r=$(apt-cache search -- "$1" | sort | head -n 10)\n'
    '[ -n "$r" ] || { shift; r=$(apt-cache search --names-only -- "$@" | sort | head -n 10); }\n'
    'printf %s "$r"'
)


# Asked how much disk it had, gemma4-e2b ran df and reported the host's 39 GB
# free, with the limits stated in its first result long gone from view: it
# follows the newest result, so these commands' results restate them.
_HOST_VIEW = re.compile(r"\b(?:df|free|nproc|lscpu|vmstat|top|htop|duf|swapon)\b|/proc/(?:meminfo|cpuinfo|swaps)")
# gemma4-e2b took "Connection refused" from its own unstarted nginx for
# "network restrictions in the sandbox".
_REFUSED = re.compile(r"(?:localhost|127\.0\.0\.1|::1|0\.0\.0\.0)\S*[^\n]*(?:Connection refused|Could not connect)|(?:Connection refused|Couldn't connect to server)[^\n]*(?:localhost|127\.0\.0\.1)")


async def limits_note(box: Sandbox, command: str) -> str:
    """The sandbox's limits, after a command that shows the host's instead."""
    if not _HOST_VIEW.search(command):
        return ""
    try:
        written = (await box.container.show(size="1")).get("SizeRw") or 0
    except aiodocker.DockerError:
        written = None
    disk = f"{human(DISK_LIMIT)} disk" + (f", {human(written)} written so far" if written is not None else "")
    return (
        f"[These figures are the host's. This sandbox's own limits: {CPUS:g} CPU, {human(MEMORY)} RAM"
        f" + {human(MEMORY_SWAP - MEMORY)} swap, {disk}.]"
    )


async def apt_alternatives(box: Sandbox, output: str) -> str:
    """For each package apt couldn't find, the ones it has like it."""
    notes = []
    for name in dict.fromkeys(m[1] or m[2] for m in _NO_PACKAGE.finditer(output)):
        words = [w for w in re.split(r"[-_.+:]", name) if w and not w.isdigit()] or [name]
        found = (await execute(box, ["bash", "-c", _APT_SEARCH, "apt-search", name, *words], 30)).output
        if found == "NOLISTS":
            notes.append(f"[No package lists yet: run `apt-get update` before installing {name}.]")
        elif found:
            notes.append(f"[Debian 13 has no package {name}. Packages like it (apt-cache search):\n{found}]")
        else:
            notes.append(f"[Debian 13 has no package {name}, and apt-cache search finds none like it.]")
    return "\n".join(notes)


# --- MCP ---------------------------------------------------------------------

_CONVERSATION = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def caller(ctx: Context) -> tuple[str, str]:
    """(conversation id, user id) from the headers LibreChat fills in."""
    headers = ctx.headers or {}
    conversation = (headers.get("x-conversation-id") or "").strip()
    if not _CONVERSATION.match(conversation):
        # Missing, or the placeholder left unresolved: never fall back to a
        # shared sandbox, which would carry state from chat to chat.
        raise ToolError("This call came without a conversation id, so there is no sandbox for it. (Admin: X-Conversation-ID header in librechat.yaml.)")
    return conversation, (headers.get("x-user-id") or "")[:64]


def _compact(schema: Any) -> Any:
    """Drop what a small model doesn't need: pydantic titles, null branches.
    (Same as agent-connectors' common.py.)"""
    if isinstance(schema, list):
        return [_compact(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: _compact(v) for k, v in schema.items() if k not in ("title", "properties")}
    if "properties" in schema:
        out["properties"] = {name: _compact(prop) for name, prop in schema["properties"].items()}
    return out


mcp = MCPServer("sandbox", log_level="WARNING", warn_on_duplicate_tools=False)


def tool(destructive: bool = False) -> Callable:
    """Register a tool: Docker errors become short messages for the model,
    and one log line per call (tool, conversation prefix, outcome, time;
    never the command, which may carry anything). The docstring is the
    description the model sees, {max_timeout} filled in: keep it short."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> str:
            started = time.monotonic()
            outcome = "ok"
            ctx = kwargs.get("ctx")
            who = ((ctx.headers or {}).get("x-conversation-id") or "-")[:8] if ctx else "-"
            try:
                return await fn(*args, **kwargs)
            except ToolError as exc:
                outcome = f"refused: {exc}"
                raise
            except aiodocker.DockerError as exc:
                outcome = f"Docker {exc.status}"
                raise ToolError(f"The sandbox host answered with an error ({exc.status}: {exc.message}).") from exc
            finally:
                log.info("%s %s %s %.0fms", fn.__name__, who, outcome, (time.monotonic() - started) * 1000)

        wrapper.__doc__ = (fn.__doc__ or "").format(max_timeout=MAX_TIMEOUT)
        mcp.add_tool(
            wrapper,
            structured_output=False,
            annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=destructive, openWorldHint=True),
        )
        registered = mcp._tool_manager.get_tool(fn.__name__)
        registered.parameters = _compact(registered.parameters)
        return fn

    return decorator


@tool()
async def run(command: str, ctx: Context, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Run a bash command in this chat's own Debian 13 sandbox: do the user's task here (set up, install, run, check) and test the code you give. You are root in /work, with internet. Install with `apt-get update && apt-get install -y X`, `pip install X`, `npm i X`. Files and installs last for this chat only; read a file with `cat`. Long jobs: `nohup CMD > log 2>&1 &`. timeout: seconds, at most {max_timeout}."""
    conversation, user = caller(ctx)
    timeout = max(1, min(timeout, MAX_TIMEOUT))
    box, note = await sandboxes.get(conversation, user)
    result = await execute(box, ["timeout", "--kill-after=5", str(timeout), "bash", "-c", command], timeout)
    if result.hung:
        status = f"[stopped waiting after {timeout}s: something still holds the output open; redirect background jobs: > log 2>&1 &]"
    elif result.exit_code in (124, 137) and result.seconds >= timeout:
        # Most often a program reading input that never comes (gemma4-e2b ran
        # its own number-guessing game bare, then said it awaited a guess).
        status = f"[timed out after {timeout}s and was stopped. Reads input? There is no keyboard: pipe it in (`printf '5\\n' | CMD`). Long jobs: nohup CMD > log 2>&1 &]"
    elif result.exit_code == 137:
        status = f"[killed (exit 137), most likely by the {human(MEMORY_SWAP)} memory limit, {result.seconds:.1f}s]"
    elif result.exit_code == 0 and result.errors:
        # A script whose last command succeeds exits 0 after any number of
        # failures: gemma4-e2b ran its test of a backup script it had never
        # written ("./backup_etc.sh: No such file or directory", exit 0 from
        # the closing rm) and said the test passed.
        lines = "1 error" if result.errors == 1 else f"{result.errors} errors"
        status = (
            f"[exit 0 from the last command, {result.seconds:.1f}s; but {lines} on stderr, the first:"
            f" {result.first_error!r}. Earlier commands may have failed.]"
        )
    else:
        status = f"[exit {result.exit_code}, {result.seconds:.1f}s]"
    alternatives = "" if result.hung else await apt_alternatives(box, result.output)
    limits = await limits_note(box, command)
    refused = (
        "[Connection refused on the sandbox's own address: nothing listens on that port. Connections"
        " inside the sandbox work; check the server is running and on which port (ss -ltnp, its log).]"
        if _REFUSED.search(result.output) else ""
    )
    return "\n".join(part for part in (note, result.output, status, alternatives, limits, refused) if part)


@tool()
async def write_file(path: str, content: str, ctx: Context) -> str:
    """Write a text file in the sandbox (relative paths are under /work), creating folders and replacing any existing file."""
    conversation, user = caller(ctx)
    size = len(content.encode())
    if size > WRITE_LIMIT:
        raise ToolError(f"That's {human(size)}; write_file takes up to {human(WRITE_LIMIT)}. Split it, or fetch the file with `run`.")
    box, note = await sandboxes.get(conversation, user)
    script = 'mkdir -p -- "$(dirname -- "$1")" && printf %s "$2" > "$1" && realpath -- "$1"'
    result = await execute(box, ["sh", "-c", script, "write_file", path, content], 30)
    if result.exit_code != 0:
        raise ToolError(f"Couldn't write {path}: {result.output or f'exit {result.exit_code}'}")
    return "\n".join(part for part in (note, f"Wrote {human(size)} to {result.output}.") if part)


@tool(destructive=True)
async def reset(ctx: Context) -> str:
    """Throw this chat's sandbox away; the next command starts a fresh one."""
    conversation, _ = caller(ctx)
    if await sandboxes.reset(conversation):
        return "Sandbox removed. The next command starts a fresh one."
    return "There was no sandbox for this chat; the next command starts a fresh one."


# --- HTTP app ----------------------------------------------------------------


def main() -> None:
    # MCPServer already configured the root logger (at WARNING, which keeps
    # the SDK quiet); one INFO line per tool call and lifecycle event is ours.
    log.setLevel(logging.INFO)
    # Requests come only from LibreChat, by container name. The Host
    # allowlist stops DNS rebinding; any Origin (i.e. any browser) is refused.
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[h.strip() for h in env("MCP_ALLOWED_HOSTS").split(",") if h.strip()],
        allowed_origins=[],
    )
    # Stateless: LibreChat opens a connection per call for a server whose
    # headers carry {{LIBRECHAT_BODY_*}} placeholders.
    app = mcp.streamable_http_app(stateless_http=True, json_response=True, transport_security=security, host="0.0.0.0")

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette):
        await sandboxes.start()
        reaper = asyncio.create_task(sandboxes.reap())
        try:
            # The mounted app's own lifespan doesn't run; start its session manager here.
            async with mcp.session_manager.run():
                yield
        finally:
            reaper.cancel()
            await sandboxes.stop()

    async def healthz(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    uvicorn.run(
        Starlette(routes=[Route("/healthz", healthz), Mount("/", app=app)], lifespan=lifespan),
        host="0.0.0.0",
        port=env_int("MCP_PORT", 8000),
        log_level="warning",
        server_header=False,
    )


if __name__ == "__main__":
    main()
