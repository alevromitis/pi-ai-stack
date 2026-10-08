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

WORKDIR /work
# PID 1 only keeps the container up; commands arrive by `docker exec`.
CMD ["sleep", "infinity"]
