# syntax=docker/dockerfile:1
#
# The sandbox gateway: an MCP server (tools run / write_file / reset) that
# starts, execs into and removes one Debian sandbox per LibreChat
# conversation, through the stack's Docker socket proxy (see gateway.py).
#
# Built on rpi (arm64). Versions are pinned in requirements.txt.
FROM python:3.13-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY gateway.py /app/
WORKDIR /app
USER 1000:1000
CMD ["python", "gateway.py"]
