# syntax=docker/dockerfile:1
#
# One image, two MCP servers for LibreChat agents that reach the user's own
# data, over Streamable HTTP (see docker-compose.agent-connectors.yml):
#
#   mcp-freshrss    python freshrss.py   FreshRSS (Google Reader API)
#   mcp-nextcloud   python nextcloud.py  Nextcloud Calendar, Tasks, Notes
#
# Built on rpi (arm64). Versions are pinned in requirements.txt.
FROM python:3.13-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONTZPATH=

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY app/ /app/
WORKDIR /app
USER 1000:1000
