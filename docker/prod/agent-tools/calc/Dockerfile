# syntax=docker/dockerfile:1
#
# mcp-calc: arithmetic, date and letter-counting tools over Streamable HTTP
# (see docker-compose.agent-tools.yml). Built on rpi (arm64). Versions are
# pinned in requirements.txt.
FROM python:3.13-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY calc.py server.py /app/
WORKDIR /app
USER 1000:1000
CMD ["python", "server.py"]
