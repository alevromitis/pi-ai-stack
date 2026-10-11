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
import shlex
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
# `lscpu`; the limits come with the first result instead (the image's free
# and nproc now report them too).
NEW_NOTE = (
    f"[New sandbox: Debian 13, root, cwd /work. Limits: {CPUS:g} CPU, {human(MEMORY)} RAM"
    f" + {human(MEMORY_SWAP - MEMORY)} swap, {human(DISK_LIMIT)} disk (df, lscpu and top show the host's).]"
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
    # The last `run` command, if it failed, and its exit code.
    failed: tuple[str, int | None] | None = None


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
# What a package's install says in any container, all well: services are
# started by hand here. Counted as errors, it read "but 2 errors ... Earlier
# commands may have failed" after a redis install, and gemma4-e2b said the
# server hadn't started (it had).
_NOT_ERRORS = re.compile(r"invoke-rc\.d: (?:policy-rc\.d denied execution|could not determine current runlevel)")


@dataclass
class Result:
    exit_code: int | None
    output: str
    seconds: float
    hung: bool
    # Lines on stderr that report a failure, and the first of them.
    errors: int = 0
    first_error: str = ""


async def execute(box: Sandbox, argv: list[str], timeout: int, env: dict[str, str] | None = None) -> Result:
    """Run argv in the sandbox; stdout and stderr interleaved as they arrive.
    Keeps the head and tail of long output, within OUTPUT_LIMIT."""
    head_cap = OUTPUT_LIMIT // 4
    tail_cap = OUTPUT_LIMIT - head_cap
    head, tail = bytearray(), bytearray()
    total = 0
    stderr = bytearray()
    started = time.monotonic()
    exe = await box.container.exec(cmd=argv, stdout=True, stderr=True, workdir=WORKDIR, environment=env)

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
    lines = [
        line.strip() for line in clean(bytes(stderr)).splitlines()
        if _ERRORISH.search(line) and not _NOT_ERRORS.search(line)
    ]
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
        f"[Figures from /proc or the filesystem are the whole machine's. This sandbox may use only"
        f" {CPUS:g} CPU, {human(MEMORY)} RAM + {human(MEMORY_SWAP - MEMORY)} swap and {disk}:"
        " answer with these.]"
    )


# Folders a failed command needed that aren't there: asked for /data/shop.db,
# gemma4-e2b got "unable to open database file" and quietly used /work.
_ABS_PATH = re.compile(r"(?<![\w.~$/-])/(?:[\w.@+-]+/)+[\w.@+-]+")
_MISSING_DIRS = 'for p in "$@"; do d=$(dirname -- "$p"); [ -e "$d" ] || echo "$d"; done | sort -u | head -n 5'


async def missing_dirs(box: Sandbox, command: str) -> str:
    paths = list(dict.fromkeys(_ABS_PATH.findall(command)))[:20]
    if not paths:
        return ""
    found = (await execute(box, ["sh", "-c", _MISSING_DIRS, "missing", *paths], 10)).output.split()
    if not found:
        return ""
    return f"[These folders don't exist: {', '.join(found)}. Create them (mkdir -p) rather than switching to another path.]"


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


# "X: command not found" from bash or dash, and from env (the sudo shim):
# which package has X (the image's index of Debian 13's commands). A
# command no package has in PATH comes from another bin folder by an
# alternative (java: openjdk-21-jre-headless) or is a versioned name (php:
# php8.4 in php8.4-cli, lua: lua5.4).
_NOT_FOUND = re.compile(r"(?m)(?:^|(?<!type)(?<!hash): )([\w.+-]+): (?:command )?not found$|^env: [‘'\x22]?([\w.+-]+)[’'\x22]?: No such file or directory$")
_WHICH_PACKAGE = (
    'tsv=/usr/share/sandbox-commands.tsv; [ -s $tsv ] || exit 0\n'
    'for c in "$@"; do command -v "$c" >/dev/null && continue\n'
    '  p=$(awk -F "\t" -v c="$c" \'$1 == c { print $2 ":" $3 }\' $tsv | head -n 8 | paste -sd " ")\n'
    '  [ -n "$p" ] || p=$(awk -F "\t" -v c="$c" \'$3 != 1 && index($1, c) == 1 && substr($1, length(c) + 1) ~ /^-?[0-9][0-9.]*$/ { print $2 ":0" }\' $tsv'
    ' | sort -rVu | head -n 8 | paste -sd " ")\n'
    '  printf "%s\\t%s\\n" "$c" "${p:--}"; done'
)


# Toolchains Debian doesn't have, or has a different program under the name
# (Debian's csc is Chicken Scheme's compiler), and what runs that language here.
_INSTEAD = {
    "dotnet": "C#: Debian has Mono instead: apt-get update && apt-get install -y mono-mcs, then mcs X.cs && mono X.exe",
    "csc": "C#: Debian has Mono instead (its csc is a Scheme compiler): apt-get update && apt-get install -y mono-mcs, then mcs X.cs && mono X.exe",
    "deno": "JavaScript and TypeScript run here with node X.js, tsx X.ts or ts-node X.ts",
    "bun": "JavaScript and TypeScript run here with node X.js, tsx X.ts or ts-node X.ts",
}


async def missing_commands(box: Sandbox, output: str) -> str:
    names = list(dict.fromkeys(m[1] or m[2] for m in _NOT_FOUND.finditer(output)))[:3]
    names = [n for n in names if not n.endswith((".sh", ".py")) and not n.startswith(".")]
    notes = [f"[{n} isn't installed. {_INSTEAD[n]}]" for n in names if n in _INSTEAD]
    names = [n for n in names if n not in _INSTEAD]
    if not names:
        return "\n".join(notes)
    found = (await execute(box, ["bash", "-c", _WHICH_PACKAGE, "which-package", *names], 15)).output
    for line in found.splitlines():
        name, _, packages = line.partition("\t")
        if packages == "-":
            notes.append(f"[No Debian 13 package has a command called {name}: check the name, or install it another way (pip, npm, its own instructions).]")
        elif packages:
            rank: dict[str, int] = {}
            for entry in packages.split():
                package, _, r = entry.rpartition(":")
                rank[package] = min(rank.get(package, 9), int(r or 0))
            names = list(rank)
            # The package named like the command, else in PATH, else the
            # shortest, else the first (the newest version): cron for crontab
            # (not bcron), docker-cli for docker, swiftlang for swift (not
            # python3-swiftclient's swift), lua5.4 for lua.
            first = min(names, key=lambda p: (not (name.startswith(p) or p.startswith(name)), rank[p], len(p), names.index(p)))
            where = f"package {first}" if len(names) == 1 else f"packages {', '.join(names)}"
            notes.append(f"[{name} isn't installed. Debian 13 has it in {where}: apt-get update && apt-get install -y {first}]")
    return "\n".join(notes)


# Programs that ask for a password or a yes on the terminal fail here, with
# messages that don't say so: gpg's "problem with the agent: Inappropriate
# ioctl for device", passwd's "Authentication token manipulation error".
# gemma4-e2b gave up on gpg, and said passwd had set the password.
_NO_TTY = re.compile(
    r"Inappropriate ioctl for device|Authentication token manipulation error|cannot open /dev/tty|"
    r"[Nn]o tty|not a tty|is not a terminal|a terminal is required|[Nn]ot a terminal|Device not configured"
)
NO_TTY_NOTE = (
    "[No terminal here: nothing can ask for a password or a yes. Give it in the command:"
    " gpg --batch --pinentry-mode loopback --passphrase PW ..., echo 'user:PW' | chpasswd,"
    " psql -c \"ALTER USER x PASSWORD 'PW'\", -y or --yes flags.]"
)


# A run command's stdin is /dev/null: a program that reads its input meets
# end of input at once, and each language says so its own way (Python's
# EOFError, Java's Scanner, Rust's unwrap on "", Kotlin's readLine()!!...).
# Coding tasks give input on stdin, and a test run without it fails.
_NO_INPUT = re.compile(
    r"EOFError|NoSuchElementException|ReadAfterEOFException|readLine\(\)[^\n]*is null|Cannot parse null string"
    r"|kind: Empty|cannot parse integer from empty string|<stdin>: \w+: end of file|Unexpectedly found nil while unwrapping"
    r"|ArgumentNullException|for nil(?::NilClass)?\b|on a nil value"
)
NO_INPUT_NOTE = (
    "[Its input was empty: a command here gets nothing on stdin, so give a program its input with a pipe"
    " or a file: printf '3\\n1 2 3\\n' | python3 sol.py, or python3 sol.py < input.txt]"
)


# Root in a container without kernel privileges: gemma4-e2b took dockerd's
# failure for a PATH problem, and keeps retrying firewall rules.
_NO_PRIVILEGE = re.compile(r"you must be root|Operation not permitted|Read-only file system|permission denied on key")
# It also took an init script's harmless "ulimit: error setting limit
# (Operation not permitted)" for redis not starting: the note says only
# those lines failed, and to check the rest.
NO_PRIVILEGE_NOTE = (
    "[You are root, but this sandbox is a container without kernel privileges: the lines above that say"
    " so failed for that reason (firewalls, Docker, mounts, kernel modules, sysctl and limits can't change"
    " here). Check whether the rest worked; for what can't be done here, say so and give the commands for"
    " a real server.]"
)


# Commands that change modes, owners, users or groups print nothing:
# gemma4-e2b ran `chmod g+w /srv/shared` to give group devs write access,
# left the folder in group root, and said it was done. The result shows what
# they changed, as it is now.
_CHANGES = re.compile(
    r"(?:^|[;&|(]\s*|\bsudo\s+)(chmod|chown|chgrp|setfacl|useradd|adduser|usermod|groupadd|addgroup|gpasswd)\b([^;&|\n]*)", re.M
)
_STATE = (
    'for p; do case "$p" in u:*) id -- "${p#u:}" ;; g:*) getent group -- "${p#g:}" ;;'
    ' *) [ -e "$p" ] && stat -c "%A %U:%G %n" -- "$p" ;; esac; done 2>/dev/null | head -n 10'
)


def changed(command: str) -> list[str]:
    """What the command's mode, owner, user and group changes touched:
    paths, "u:user" and "g:group"."""
    things = []
    for name, rest in _CHANGES.findall(command):
        try:
            words = shlex.split(rest)
        except ValueError:
            words = rest.split()
        args, groups, take = [], [], False
        for w in words:
            if take:
                groups += w.split(",")
                take = False
            elif w in ("-G", "-g", "--groups", "--gid", "--ingroup") or (w.startswith("-") and w.rstrip("G").endswith("a") and w.endswith("G")):
                take = True
            elif not w.startswith("-"):
                args.append(w)
        if name in ("chmod", "chown", "chgrp", "setfacl"):
            things += args[1:]
        elif name in ("groupadd", "addgroup") and args:
            things.append("g:" + args[-1])
        elif name == "gpasswd" and args:
            things += ["g:" + args[-1], *("u:" + a for a in args[:-1])]
        elif args:
            things.append("u:" + args[-1])
        things += ["g:" + g for g in groups if g]
    return list(dict.fromkeys(things))[:8]


async def state_note(box: Sandbox, command: str) -> str:
    things = changed(command)
    if not things:
        return ""
    now = (await execute(box, ["bash", "-c", _STATE, "state", *things], 10)).output
    return f"[Now:\n{now}]" if now else ""


# ShellCheck on the model's shell code: its errors and warnings, and the
# notes that a variable set in a pipeline's subshell is lost (gemma4-e2b's
# duplicate finder reported 0 groups of 53 that way). Not the style and
# quoting advice, nor warnings that don't change what runs.
# $1: a file to check (bash without a shebang), or "-" for the text in $2
# (a run command, as bash).
_SHELLCHECK = (
    'cd /work || exit\n'
    'if [ "$1" = - ]; then f=$(mktemp); printf %s "$2" > "$f"; set -- -s bash "$f"\n'
    'elif [ "$(head -c 2 "$1")" = "#!" ]; then set -- "$1"; else set -- -s bash "$1"; fi\n'
    'shellcheck -f gcc -S info "$@" 2>/dev/null | head -n 40; [ -z "$f" ] || rm -f "$f"'
)
_SC_LINE = re.compile(r"^(?:[^:]*):(\d+):\d+: (error|warning|note): (.*) \[(SC\d+)\]$")
_SC_NOTES = {"SC2030", "SC2031"}
_SC_SKIP = {"SC2034", "SC2148", "SC2155", "SC2164"}


async def shellcheck(box: Sandbox, path: str = "-", text: str = "") -> str:
    found = (await execute(box, ["bash", "-c", _SHELLCHECK, "shellcheck", path, text], 20)).output
    problems = []
    for line in found.splitlines():
        m = _SC_LINE.match(line)
        if m and m[4] not in _SC_SKIP and (m[2] != "note" or m[4] in _SC_NOTES):
            problems.append(f"line {m[1]}: {m[3]} ({m[4]})")
    if not problems:
        return ""
    shown = "\n".join(dict.fromkeys(problems[:6]))
    return f"[ShellCheck finds problems; fix them, then run it:\n{shown}]"


def is_shell(path: str, content: str) -> bool:
    first = content.split("\n", 1)[0]
    return path.endswith((".sh", ".bash")) or bool(re.match(r"#!\s*\S*(?:/|env\s+)(?:ba|da)?sh\b", first))


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


async def still_running(box: Sandbox, timeout: int) -> list[str]:
    """Just before `timeout` stops a command, what it is still running: the
    innermost processes under it."""
    await asyncio.sleep(max(timeout - 2, timeout * 0.8))
    table = (await execute(box, ["ps", "-eo", "pid=,ppid=,args="], 10)).output
    children: dict[str, list[str]] = {}
    args: dict[str, str] = {}
    roots = []
    for line in table.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, cmd = parts
        children.setdefault(ppid, []).append(pid)
        args[pid] = cmd
        if cmd.startswith(f"timeout --kill-after=5 {timeout} bash -c"):
            roots.append(pid)
    leaves, todo = [], list(roots)
    while todo:
        pid = todo.pop()
        if pid in children:
            todo.extend(children[pid])
        elif pid not in roots:
            leaves.append(" ".join(args[pid].split())[:80])
    return list(dict.fromkeys(leaves))[:3]


# "." as a path argument: `find .`, `du -sh .`, `ls ./` (not `. ~/.bashrc`).
_HERE = re.compile(r"(?<=[\w-]\s)\.(?:/|\s|$)", re.M)

# The image's bashenv: each command reads ~/.bashrc, with aliases on.
RUN_ENV = {"BASH_ENV": "/etc/sandbox/bashenv"}


@tool()
async def run(command: str, ctx: Context, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Run a bash command in this chat's own Debian 13 sandbox: do the user's task here (set up, install, run, check) and test the code you give. You are root in /work, with internet. Installed: python3, gcc/g++, java/javac 21, node 20 with tsc/tsx/ts-node, go, rustc/cargo, ruby, php. Install more with `apt-get update && apt-get install -y X`, `pip install X`, `npm i X`. Files and installs last for this chat only; read a file with `cat`. Long jobs: `nohup CMD > log 2>&1 &`. timeout: seconds, at most {max_timeout}."""
    conversation, user = caller(ctx)
    timeout = max(1, min(timeout, MAX_TIMEOUT))
    box, note = await sandboxes.get(conversation, user)
    watch = asyncio.create_task(still_running(box, timeout))
    try:
        result = await execute(box, ["timeout", "--kill-after=5", str(timeout), "bash", "-c", command], timeout, RUN_ENV)
    finally:
        watch.cancel()
    try:
        running = watch.result() if watch.done() else []
    except (asyncio.CancelledError, aiodocker.DockerError):
        running = []
    if result.hung:
        status = f"[stopped waiting after {timeout}s: something still holds the output open; redirect background jobs: > log 2>&1 &]"
    elif result.exit_code in (124, 137) and result.seconds >= timeout:
        # A program reading input that never comes (gemma4-e2b ran its own
        # number-guessing game bare, then said it awaited a guess), or one
        # that never ends by itself: its CSV monitor called `vmstat 1`, which
        # samples until stopped, and it blamed the sandbox. Naming what was
        # still running points at the cause.
        where = f", still running {', '.join(f'`{a}`' for a in running)}" if running else ""
        status = (
            f"[timed out after {timeout}s and was stopped{where}. A program that waits for input or never"
            " ends by itself needs its input piped in (`printf '5\\n' | CMD`) or a count or limit"
            " (`vmstat 1 5`, `ping -c 3`, `top -b -n 1`). Long jobs: nohup CMD > log 2>&1 &]"
        )
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
            f" {result.first_error!r}. Earlier commands may have failed: fix the cause and run it again.]"
        )
    elif result.exit_code:
        # It explained failures instead of fixing them: "the API returned
        # non-JSON data" from its own script's .json() on a text endpoint.
        status = f"[exit {result.exit_code}, {result.seconds:.1f}s: it failed. Fix the cause and run it again, or say why it can't be done.]"
    elif result.output.strip():
        status = f"[exit {result.exit_code}, {result.seconds:.1f}s]"
    else:
        # gemma4-e2b's status check printed nothing (`read` got no newline
        # from curl -w) and it answered with the "UP" it expected.
        status = f"[exit {result.exit_code}, {result.seconds:.1f}s, no output]"
        # Asked for every file over 10MB on the system, it ran `find .` in
        # /work and said there were none.
        if _HERE.search(command):
            status = status[:-1] + f"; . is {WORKDIR}, this chat's work folder, not the whole system]"
    failed = result.exit_code not in (0, None) or result.errors
    # gemma4-e2b ran a failing `git commit` again unchanged ("Author identity
    # unknown" both times), then reported the failure.
    if failed and box.failed == (command, result.exit_code):
        status += "\n[That is the same command that failed the same way just before: running it again won't change it. Do what the error asks first, then run it.]"
    box.failed = (command, result.exit_code) if failed else None
    alternatives = "" if result.hung else await apt_alternatives(box, result.output)
    commands = "" if result.hung else await missing_commands(box, result.output)
    # Its `[[ a =~ x || [[ b =~ y ]]` failed three times on bash's terse
    # "conditional binary operator expected".
    syntax = await shellcheck(box, text=command) if result.exit_code == 2 and "syntax error" in result.output else ""
    privilege = NO_PRIVILEGE_NOTE if _NO_PRIVILEGE.search(result.output) else ""
    no_tty = NO_TTY_NOTE if _NO_TTY.search(result.output) else ""
    state = "" if result.hung else await state_note(box, command)
    limits = await limits_note(box, command)
    folders = await missing_dirs(box, command) if failed and not result.hung else ""
    refused = (
        "[Connection refused on the sandbox's own address: nothing listens on that port. Connections"
        " inside the sandbox work; check the server is running and on which port (ss -ltnp, its log).]"
        if _REFUSED.search(result.output) else ""
    )
    no_input = NO_INPUT_NOTE if failed and _NO_INPUT.search(result.output) and not re.search(r"[|<]", command) else ""
    return "\n".join(part for part in (note, result.output, status, state, syntax, alternatives, commands, folders, no_tty, no_input, privilege, limits, refused) if part)


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
    checked = await shellcheck(box, result.output) if is_shell(path, content) else ""
    return "\n".join(part for part in (note, f"Wrote {human(size)} to {result.output}.", checked) if part)


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
