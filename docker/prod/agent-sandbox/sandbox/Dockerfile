# syntax=docker/dockerfile:1
#
# The per-chat sandbox: plain Debian with what an agent reaches for first.
# The gateway starts one container from this image per conversation - as
# root, no host mounts, no credentials, limits from .env - and removes it
# afterwards. Anything else the agent installs itself (apt, pip and npm
# all work), which costs time on every fresh sandbox: put what it keeps
# installing here instead.
#
# Built on rpi (arm64) by the stack's sandbox-image service.
FROM debian:trixie-slim

# Non-interactive, plain-text output: no prompts, pagers, colours or
# progress bars eating the model's context.
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    TERM=dumb \
    NO_COLOR=1 \
    PAGER=cat \
    GIT_PAGER=cat \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off \
    npm_config_fund=false \
    npm_config_audit=false \
    npm_config_update_notifier=false \
    npm_config_progress=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        bzip2 \
        ca-certificates \
        curl \
        dnsutils \
        file \
        git \
        gnupg \
        iproute2 \
        iputils-ping \
        jq \
        less \
        netcat-openbsd \
        nodejs \
        npm \
        openssh-client \
        procps \
        psmisc \
        python3 \
        python3-pip \
        python3-venv \
        ripgrep \
        sqlite3 \
        unzip \
        wget \
        xz-utils \
        zip \
        zstd \
    && rm -rf /var/lib/apt/lists/*

# Agents' own installs: no Recommends/Suggests (default-jre otherwise pulls
# GTK and LLVM, past the 300 s command limit on the Pi), no progress output.
RUN printf '%s\n' \
        'APT::Install-Recommends "false";' \
        'APT::Install-Suggests "false";' \
        'Dpkg::Use-Pty "0";' \
        'quiet "1";' \
        > /etc/apt/apt.conf.d/90sandbox

# Commands copied from guides start with sudo, and gemma4-e2b adds it on its
# own even when told it is root: run them as given (already root, no
# password). Leading sudo options are dropped; env runs VAR=value too. But
# `-u USER` runs it as that user (runuser): `sudo -u postgres psql` ran as
# root, "role root does not exist". And `python` is python3, which the
# model also types.
RUN printf '%s\n' \
        '#!/bin/sh' \
        '# Already root: run the command as given; -u USER runs it as USER.' \
        'user=; login=' \
        'while [ $# -gt 0 ]; do' \
        '  case "$1" in' \
        '    -u) user=$2; shift 2 ;;' \
        '    --user=*) user=${1#--user=}; shift ;;' \
        '    -i|--login) login=1; shift ;;' \
        '    -g|-p|-C) shift 2 ;;' \
        '    --) shift; break ;;' \
        '    -*) shift ;;' \
        '    *) break ;;' \
        '  esac' \
        'done' \
        '[ $# -gt 0 ] || set -- bash' \
        'case "$user" in ""|root|"#0") exec env "$@" ;; esac' \
        '[ -z "$login" ] || cd "$(getent passwd "$user" | cut -d: -f6)" 2>/dev/null || true' \
        'exec runuser -u "$user" -- env "$@"' \
        > /usr/local/bin/sudo \
    && chmod 755 /usr/local/bin/sudo \
    && ln -s /usr/bin/python3 /usr/local/bin/python

# No systemd here (PID 1 is sleep), but guides and gemma4-e2b reach for
# systemctl ("systemctl: command not found" after setting up nginx):
# start/stop/restart/reload/status go to the service's init script, and
# enable (nothing boots here) says so, starting it with --now.
RUN printf '%s\n' \
        '#!/bin/sh' \
        '# No systemd in this sandbox: systemctl ACTION UNIT -> service UNIT ACTION.' \
        'action=; units=; now=' \
        'for a in "$@"; do case "$a" in --now) now=1 ;; -*) ;; *) if [ -z "$action" ]; then action=$a; else units="$units ${a%.service}"; fi ;; esac; done' \
        'case "$action" in' \
        '  start|stop|restart|reload|force-reload|status|try-restart) for u in $units; do service "$u" "$action" || exit $?; done ;;' \
        '  is-active) for u in $units; do if service "$u" status >/dev/null 2>&1; then echo active; else echo inactive; exit 3; fi; done ;;' \
        '  enable|disable) echo "systemctl: no systemd here, so nothing starts at boot; service UNIT start runs it now." >&2' \
        '    if [ -n "$now" ]; then for u in $units; do if [ "$action" = enable ]; then service "$u" start; else service "$u" stop; fi; done; fi ;;' \
        '  daemon-reload) ;;' \
        '  *) echo "systemctl: no systemd in this sandbox; use: service UNIT start|stop|restart|status" >&2; exit 1 ;;' \
        'esac' \
        > /usr/local/bin/systemctl \
    && chmod 755 /usr/local/bin/systemctl

# nproc and free report this container's cgroup limits, not the host's.
COPY --chmod=755 bin/nproc bin/free /usr/local/bin/

# Command -> Debian package, for the gateway's note after "command not
# found": gemma4-e2b installed docker.io, which leaves the docker command to
# docker-cli (a Recommends, skipped above), and took the missing command for
# a PATH problem; logrotate it never installed at all. Column 3: 0 for a
# command in PATH, 1 for one in another bin folder, where an alternative
# links it from at install (java, javac, swift: no package has /usr/bin/java).
RUN arch=$(dpkg --print-architecture) \
    && for c in all "$arch"; do curl -fsSL "https://deb.debian.org/debian/dists/trixie/main/Contents-$c.gz" | zcat; done \
    | awk '{ r = $1 ~ /^(usr\/)?(s?bin|games)\/[^\/]+$/ ? 0 : $1 ~ /^(usr|opt)\/.*\/bin\/[^\/]+$/ ? 1 : -1 } r >= 0 { n = split($1, p, "/"); m = split($NF, q, ","); for (i = 1; i <= m; i++) { sub(/.*\//, "", q[i]); print p[n] "\t" q[i] "\t" r } }' \
    | sort -u > /usr/share/sandbox-commands.tsv \
    && test -s /usr/share/sandbox-commands.tsv

# Each `run` command is a fresh non-interactive bash, which skips ~/.bashrc
# and aliases: gemma4-e2b added `alias ll='ls -la'` to /root/.bashrc,
# sourced it, got "ll: command not found" and said the sandbox failed. The
# gateway sets BASH_ENV to this file, so a command starts like a new
# terminal; scripts it runs don't (BASH_ENV is unset for them).
RUN mkdir -p /etc/sandbox && printf '%s\n' \
        '# BASH_ENV of the gateway'"'"'s run commands: start like a new terminal.' \
        'unset BASH_ENV' \
        'shopt -s expand_aliases' \
        '[ -r ~/.bashrc ] && . ~/.bashrc' \
        > /etc/sandbox/bashenv

# For the gateway's check of shell scripts the model writes: gemma4-e2b's
# duplicate finder filled an array inside `find | while read`, a subshell,
# and reported the empty result (0 groups of 53).
RUN apt-get update && apt-get install -y --no-install-recommends shellcheck \
    && rm -rf /var/lib/apt/lists/*

# The languages of coding tasks, ready to run: a JDK, Go and Rust each take
# minutes to install on the Pi (and 256 MiB of RAM), time a coding task
# shouldn't spend before its first test. python3-sortedcontainers: LeetCode's
# Python has it, so solutions import it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cargo \
        default-jdk-headless \
        golang-go \
        php-cli \
        python3-sortedcontainers \
        ruby \
        rustc \
    && rm -rf /var/lib/apt/lists/*

# TypeScript: tsc, and tsx and ts-node to run a .ts file. TypeScript 5:
# ts-node fails on 7 (no JavaScript API), and 6 and 7 leave out @types/node
# (tsc: "Cannot find name 'require'"). /node_modules is npm's global folder,
# so Node and tsc find what `npm i -g` installed from any folder
# (require('lodash') after npm i -g lodash; @types/node for process).
# Without a tsconfig.json, ts-node picks Node 16's module setting, and Node
# 20 then fails on the file: "Unknown file extension .ts". CommonJS runs it.
# Not strict: gemma4-e2b's working script stopped at "implicitly has an
# 'any' type", and it said the sandbox couldn't run TypeScript.
ENV TS_NODE_COMPILER_OPTIONS={\"module\":\"commonjs\",\"strict\":false}
RUN npm install -g typescript@5 tsx ts-node @types/node \
    && ln -s "$(npm root -g)" /node_modules \
    && npm cache clean --force

# Go builds its standard library on first use: a minute for the first
# `go run` on one core. Build the packages solutions import now. trim.txt
# says the cache was just trimmed (in 2100), or the first go command in an
# image older than 5 days would delete it all as unused.
# GOTOOLCHAIN=local: a go.mod asking for a newer Go fails instead of downloading it.
ENV GOTOOLCHAIN=local
RUN mkdir -p /tmp/warm && cd /tmp/warm \
    && printf '%s\n' 'package main' 'import (' '_ "bufio"' '_ "container/heap"' '_ "container/list"' \
        '_ "errors"' '_ "fmt"' '_ "math"' '_ "math/bits"' '_ "math/rand"' '_ "os"' '_ "slices"' '_ "maps"' \
        '_ "sort"' '_ "strconv"' '_ "strings"' '_ "sync"' '_ "time"' '_ "unicode"' '_ "encoding/json"' ')' \
        'func main() {}' > main.go \
    && go build -o /dev/null main.go \
    && cd / && rm -rf /tmp/warm && echo 4102444800 > "$(go env GOCACHE)/trim.txt"

# bits/stdc++.h, which C++ solutions include, takes 13 s to compile on the
# Pi; precompiled for the gateway's run_code flags (-O2 -std=c++20) it takes
# 2 s. Other flags compile it as before.
RUN h=$(find /usr/include -path '*/bits/stdc++.h' | head -n 1) \
    && g++ -O2 -std=c++20 -x c++-header -o "$h.gch" "$h"

WORKDIR /work
# PID 1 only keeps the container up; commands arrive by `docker exec`.
CMD ["sleep", "infinity"]
