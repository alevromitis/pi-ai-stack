# syntax=docker/dockerfile:1
#
# mcp-translate: offline translation (Argos Translate models via
# CTranslate2) over Streamable HTTP (see docker-compose.agent-tools.yml).
# Built on rpi (arm64). Versions are pinned in requirements.txt; the models
# are mounted read-only from ${TRANSLATE_MODELS_DIR} (fetch_models.py).
FROM python:3.13-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY translate.py server.py /app/
WORKDIR /app
USER 1000:1000
CMD ["python", "server.py"]
