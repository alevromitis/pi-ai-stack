# syntax=docker/dockerfile:1
#
# One image, two MCP servers for LibreChat agents, each bridged from stdio to
# Streamable HTTP by supergateway so they run in their own sandboxed
# containers instead of as children of librechat-1 (which holds its secrets):
#
#   mcp-shell       mcp-shell-server (allowlisted argv commands, no shell)
#   mcp-filesystem  @modelcontextprotocol/server-filesystem on /workspace
#
# Built on rpi (arm64). Versions are pinned here; bump deliberately.
FROM node:22-bookworm-slim

ARG SUPERGATEWAY_VERSION=4.1.0
ARG MCP_FILESYSTEM_VERSION=2026.8.31
ARG MCP_SHELL_VERSION=1.1.12

# Tools the shell server may run (see ALLOW_COMMANDS in the compose file).
# coreutils, grep, sed, findutils, tar, gzip, diffutils ship with the base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-venv \
        jq \
        file \
        bc \
        gawk \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g --omit=dev \
        "supergateway@${SUPERGATEWAY_VERSION}" \
        "@modelcontextprotocol/server-filesystem@${MCP_FILESYSTEM_VERSION}" \
    && npm cache clean --force

# Own venv, kept off the children's PATH (MCP_SHELL_SAFE_PATH default), so an
# allowed `python3` is the plain system interpreter.
RUN python3 -m venv /opt/mcp-shell \
    && /opt/mcp-shell/bin/pip install --no-cache-dir "mcp-shell-server==${MCP_SHELL_VERSION}" \
    && ln -s /opt/mcp-shell/bin/mcp-shell-server /usr/local/bin/mcp-shell-server

USER node
WORKDIR /workspace
