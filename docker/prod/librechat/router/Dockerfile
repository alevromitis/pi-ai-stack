# syntax=docker/dockerfile:1
#
# librechat-router: LibreChat's Qwen endpoint. Forwards to llama-server on the
# workstation, or to litert-lm-1 with the E2B - Thinker preset when that is
# down (see router.py). Built on rpi (arm64). Versions are pinned in
# requirements.txt.
FROM python:3.13-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY router.py /app/
WORKDIR /app
USER 1000:1000
CMD ["python", "router.py"]
