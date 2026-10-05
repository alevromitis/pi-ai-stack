# syntax=docker/dockerfile:1
#
# Runs Google's LiteRT-LM CLI (https://github.com/google-ai-edge/LiteRT-LM)
# on a Raspberry Pi 5, serving its built-in OpenAI-compatible endpoint
# (`litert-lm serve`) so Open WebUI can connect to it like any other
# OpenAI backend.
#
# Base image: Raspberry Pi OS (64-bit) is Debian "trixie"-based as of 2026,
# so we match that here to stay ABI-compatible with the Pi's kernel/DRM
# drivers and Raspberry Pi's own apt packages. Build/run this on the Pi
# itself (arm64) — there is nothing here for x86_64.
FROM debian:trixie-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:${PATH}"

# --- Base tooling -----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        python3 \
        python3-venv \
        python3-pip \
        gettext-base \
    && rm -rf /var/lib/apt/lists/*

# --- GPU userspace (Mesa / V3DV Vulkan) -------------------------------------
# The Pi 5's GPU (VideoCore VII) is exposed to LiteRT-LM's GPU backend via
# Dawn (WebGPU) -> Vulkan -> Mesa's V3DV driver. Debian trixie's own
# mesa-vulkan-drivers package already includes V3DV (it's been upstream in
# Mesa for years), so we install straight from Debian's official archive —
# no third-party apt repo required, which is both simpler and more reliable
# to build than pulling Raspberry Pi's own repo into a generic container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-dri \
        libegl-mesa0 \
        libglx-mesa0 \
        mesa-vulkan-drivers \
        libvulkan1 \
        vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# --- LiteRT-LM CLI ------------------------------------------------------------
# Installs the `litert-lm` binary (run / import / list / serve, incl. its
# native OpenAI-compatible server) inside an isolated venv.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade litert-lm

WORKDIR /app
COPY config/config.json.template /app/config.json.template
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Default port for `litert-lm serve`'s OpenAI-compatible API.
EXPOSE 9379

ENTRYPOINT ["/app/entrypoint.sh"]
