# agent-sandbox

A Debian container per LibreChat conversation, over MCP: the agent gets a
root shell and the network, inside fixed CPU, memory, disk and time limits.
Idle, it is stopped, not thrown away: reopen the chat days later and its
files and installs are still there. Lives in `docker/prod/agent-sandbox/`. Nothing is published
on the host; LibreChat reaches the gateway on `agent-sandbox-internal`.
Deployed on `rpi` 2026-10-08.

Unlike [agent-tools](../agent-tools/README.md)' shell (allowlisted commands, no
network, a shared `/workspace`), this is a whole machine: interpreters,
`apt`, `pip`, `npm`, background jobs, network. So the boundary is the
container, not the tool.

```
librechat-1 ──agent-sandbox-internal──► agent-sandbox-gateway-1:8000/mcp
                                              │  agent-sandbox-docker (internal)
                                              ▼
                                agent-sandbox-docker-proxy-1:2375 ──► docker.sock (ro)
                                              │  create / exec / delete
                                              ▼
        agent-sandbox-<chat>-<hex>, one per conversation, on agent-sandbox-egress
                                              │  Docker NAT, no extra rules
                                              ▼
                                internet (and LAN, tailnet, the host's ports)
```

| Container | What | Network |
|---|---|---|
| `agent-sandbox-gateway-1` | MCP server (`gateway/gateway.py`, MCP SDK 2.3, stateless Streamable HTTP): tools `run`, `run_code`, `write_file`, `reset`; creates, stops, starts again and removes the sandboxes | `agent-sandbox-internal`, `agent-sandbox-docker` |
| `agent-sandbox-docker-proxy-1` | [linuxserver/socket-proxy](https://github.com/linuxserver/docker-socket-proxy): the Docker API, containers and exec only | `agent-sandbox-docker` |
| `agent-sandbox-image-1` | builds `agent-sandbox:local` (`sandbox/Dockerfile`), then exits | `agent-sandbox-egress` |
| `agent-sandbox-<chat>-<hex>` | the sandbox; made by the gateway, not a Compose service (labels `agent-sandbox.managed=1`, `agent-sandbox.conversation`) | `agent-sandbox-egress` (external: `make up` creates it, `make down` leaves it) |

**A sandbox per chat, kept between visits.** `librechat.yaml` sends `X-Conversation-ID:
'{{LIBRECHAT_BODY_CONVERSATIONID}}'` with the `sandbox` server. LibreChat
v0.8.7 resolves body placeholders on every tool call (`MCPManager.callTool`)
and opens a connection per call for such a server; a new chat gets its
UUID before any tool runs (`agents/request.js`). The gateway keys sandboxes
on it: the first call creates a container, later calls `docker exec` into
it, and a new chat is a new id, so it always starts clean. A call without a
usable id is refused, never handed a shared sandbox.

A sandbox is **stopped** (`docker stop`: its processes end, its writable
layer stays) when:

| When | Setting (default) |
|---|---|
| idle | `SANDBOX_IDLE_MINUTES` (30) |
| running too long since it (re)started | `SANDBOX_MAX_HOURS` (4) |
| another chat needs the slot (least recently used goes first) | `SANDBOX_MAX_ACTIVE` (1) |
| the gateway stops (all of them), or finds them running at its start | — |
| its container stopped by itself (OOM-killed, Docker restarted, host reboot) | — |

The chat's next call starts the same container again, whether that is a
minute or two weeks later: same files, same installed packages, same users
and config, with the model told so and that nothing is running any more
("[Resumed this chat's sandbox, stopped 2 days ago (idle for over 60
minutes): its files and installs are as they were, but nothing it ran is
running now. Start servers and background jobs again.]"). Docker is the only
record: the gateway finds a chat's sandbox by its `agent-sandbox.conversation`
label, so this survives gateway restarts, `make down` and reboots. A forked
chat is a new conversation id, so it starts clean.

A sandbox is **removed** (files gone; the chat's next call gets a fresh one
and is told why) when:

| When | Setting (default) |
|---|---|
| its writable layer is over the limit (checked every 30 s while running) | `SANDBOX_DISK_LIMIT` (4g) |
| the `reset` tool, running or kept | — |
| kept stopped longer than this since last used | `SANDBOX_KEEP_DAYS` (14) |
| past this many kept ones (least recently used go first) | `SANDBOX_KEEP_MAX` (50) |
| kept ones' writable layers past this in all (least recently used first) | `SANDBOX_KEEP_DISK` (10g) |
| it won't start again (e.g. its network was removed and remade) | — |

Kept sandboxes are swept after every stop and hourly. They live in Docker's
own storage, not a volume, so they aren't backed up; and one made before a
rebuild of `agent-sandbox:local` keeps the old image (mostly shared layers)
until it is removed. `SANDBOX_KEEP_DAYS=0` keeps nothing: every stop is a
removal, as before 2026-10-11.

`agent-sandbox-egress` is external, made by `make up` (Makefile target
`egress-network`), because a stopped container holds its network's id, not
its name (`getNetworkID` in moby's `daemon/container_operations.go`): had
`make down` removed the network, every kept sandbox would fail to start
with "network … not found". To throw them all away: `docker rm $(docker ps
-aq --filter label=agent-sandbox.managed=1 --filter status=exited)`.

**Limits**, all in `.env`:

| | Default | How |
|---|---|---|
| Memory | `SANDBOX_MEMORY` 256m of RAM, `SANDBOX_MEMORY_SWAP` 1g in all | cgroup `memory.max` 256 MiB and `memory.swap.max` 768 MiB: past 256 MiB the kernel pages the sandbox out to `/var/swap` (8 GiB file on disk, not zram), so it leaves RAM to LiteRT-LM. `/dev/shm` (64m) counts too. Past 1 GiB an OOM kills the biggest process; `run` reports exit 137 as "most likely the memory limit" |
| CPU | `SANDBOX_CPUS` 1 core, weight `SANDBOX_CPU_SHARES` 256 | `NanoCpus`; the low weight (default 1024) lets LiteRT-LM win when both want the CPU |
| Processes | `SANDBOX_PIDS` 512, `nofile` 8192 | pids cgroup, ulimit |
| Disk | `SANDBOX_DISK_LIMIT` 4g | Docker can't cap overlay2 on ext4, so the gateway measures `SizeRw` every 30 s and removes the sandbox past it. `SANDBOX_DISK_DEVICE` + `SANDBOX_DISK_WRITE_BPS` (50m) can throttle writes to bound the overshoot |
| Count | `SANDBOX_MAX_ACTIVE` 1 | the gateway |
| Per command | `SANDBOX_DEFAULT_TIMEOUT` and `SANDBOX_MAX_TIMEOUT` 300 s, `SANDBOX_OUTPUT_LIMIT` 8000 chars | `timeout --kill-after=5` inside, a read deadline in the gateway; head + tail of long output |
| Network | none | plain Docker bridge + NAT, by choice: no rules on the host (see the threat model) |

The default is the cap because the model rarely passes `timeout`, and at
the old 60 s default `apt-get install default-jre` was killed mid-install
(2026-10-09). LibreChat v0.8.7 waits for an MCP server as long as that
server's `timeout` in `librechat.yaml`: the MCP request and undici's
header and body timeouts all take it (`createFetchFunction`), and the
gateway answers in one JSON response at the end. So the sandbox's `timeout`
(360 s) must stay above the cap + 15 s plus a fresh sandbox's start.

**Near unrestricted inside.** Root, writable root filesystem, Docker's
default seccomp profile and capabilities minus `NET_RAW`, `MKNOD` and
`AUDIT_WRITE` (ping still works through ICMP datagram sockets), and
`no-new-privileges`. Preinstalled: Python 3 with pip and venv, Node and npm,
build-essential, git, curl, wget, an ssh client, jq, ripgrep, sqlite3,
dnsutils and iproute2. `PIP_BREAK_SYSTEM_PACKAGES=1`, so `pip install`
just works, and colours, pagers and progress bars are off. apt skips
Recommends and Suggests, `python` is python3, and `sudo` just runs its
command (the model adds it out of habit). Every fresh
sandbox repeats its installs, so add whatever the agent keeps installing to
`sandbox/Dockerfile`. Long jobs go in the background (`nohup CMD > log 2>&1
&`). A background job that keeps the command's output open makes `run` stop
waiting at its timeout + 15 s, and say so.

**Tools** (~1.2k characters of schema in all):

* `run(command, timeout=300)`: `bash -c` in `/work`. Returns stdout and
  stderr interleaved, with ANSI colours and `\r` redraws stripped, then the
  exit code and duration. gemma4-e2b acts on what a result says, not on
  the tool description (tested 2026-10-10), so results carry the help: a
  new sandbox's first result states its limits (`free` and `nproc` show the
  host's); an apt install of a package Debian 13 doesn't have
  (`openjdk-17-jdk`, `neofetch`) ends with what `apt-cache search` finds
  instead; a timeout says there is no keyboard and how to pipe input in.
  The result starts with `$ command` (cut at 600 characters): LibreChat
  shows a tool's result as plain text but its arguments on one line behind
  a second click, so this is what makes the session readable as a terminal.
  `run_code` results likewise name the compile command and show the input.
  With *Settings → Chat → Auto-expand tool details* on, every result is
  open in the chat without clicking.
* `write_file(path, content)`: creates folders. Up to 100 KB, since the
  content travels in the exec argv.
* `reset()`.

**Threat model.** Assume the model gets prompt-injected (it reads web
pages) and runs whatever a page says, in its sandbox. That's allowed by
design; what matters is what a sandbox can reach.

* **A sandbox** holds nothing: no host mounts, no Docker socket, no
  credentials, no other chat's files, and it is on none of the internal
  networks (LibreChat's, the connectors', agent-tools', the gateway's).
  **Its network is not restricted** (decided 2026-10-08: no rules on the
  host). Like `mcp-playwright-1`, it reaches the internet and, through
  Docker's NAT, the LAN, the tailnet and every port the Pi publishes on
  `0.0.0.0`: among them LiteRT-LM `:9379` (no auth), Gitea, Portainer,
  n8n, Backrest, Homepage. An injected page can use those like any LAN
  client could. If that changes, a `DOCKER-USER` rule dropping
  `agent-sandbox-egress` (10.89.32.0/24) to private ranges closes it.
  Escaping the container would take a kernel or runc bug: runc containers
  share the host's kernel. `SANDBOX_RUNTIME=runsc` ([gVisor](https://gvisor.dev))
  narrows that a lot, once installed. Check first that it runs on the
  Pi 5's 16K-page kernel.
* **The gateway is the privileged piece.** Through the proxy it can create
  containers. The proxy filters by API path and method, not by request
  body, so it can't stop a create asking for `Privileged` or a host bind
  mount. Subverting the gateway means root on the host, as with `portainer`
  and `watchtower`. What keeps it honest: its only input is MCP calls from
  LibreChat on an `internal` network (sandboxes can't reach it); it builds
  one fixed `HostConfig` from its environment; commands go into the exec
  argv as they are, never interpreted on its side; it runs as uid 1000 with
  a read-only root and no capabilities; it logs tool, conversation prefix,
  outcome and duration, never commands.
* **The proxy** is reachable only from the gateway (`agent-sandbox-docker`,
  `internal: true`). It allows `/containers` (create, start, stop, inspect,
  list, delete, exec) and `/exec`. Images (so no pulls), build, volumes,
  networks, system, events and archive stay off.

**Verified on `rpi` 2026-10-08**, calling the gateway from inside
`librechat-1` with LibreChat's headers:

* The cgroup limits are as configured: `memory.max` 256 MiB,
  `memory.swap.max` 768 MiB, `cpu.max` one CPU, `pids.max` 512.
* A process touching 700 MiB ran with 253 MiB in RAM and 514 MiB in swap.
  One touching 1100 MiB was OOM-killed (exit 137).
* HTTPS, DNS, ping and `apt-get install` work, and files persist within a
  chat.
* A second chat evicted the first, which got a fresh sandbox and the note
  on its next call. `reset` works, and so does exec through the socket
  proxy.

Not yet tried from the UI: enabling *Debian Sandbox* from the input bar.
LibreChat's startup inspection skips servers with `{{LIBRECHAT_BODY_*}}`
placeholders (`MCPServerInspector`, so "Tools: undefined" in its log is
expected); it fetches their tools when a user connects. `nproc` reports 4:
the limit is one core's worth of CPU time across the four, not a pinned
core.

## First run

```bash
cd docker/prod/agent-sandbox
make decrypt && make up    # builds agent-sandbox{,-gateway}:local (arm64)
```

Then `make up` in `docker/prod/librechat`, so `librechat-1` joins
`agent-sandbox-internal` and loads the `sandbox` server. This stack must be
up before `librechat` (it owns `agent-sandbox-internal`).

* After editing `sandbox/Dockerfile`: `docker compose -f
  docker-compose.agent-sandbox.yml build sandbox-image`. New sandboxes use
  the new image; a running or kept one keeps its own until removed.
* After editing `gateway.py`: `... up -d --build gateway`. Restarting the
  gateway stops every sandbox; each chat's next call starts its own again.

