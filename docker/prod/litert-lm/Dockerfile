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
    && /opt/venv/bin/pip install --no-cache-dir litert-lm==0.18.0

# `serve` never passes enable_ynnpack to the Engine (only `run`/`benchmark`
# have --enable-ynnpack, and config.json ignores the key), so patch it in.
# Benchmarked 2026-10-07 (1024 prefill / 256 decode, 4 threads): gemma4-e2b
# prefill 148 vs 103 tok/s, decode 9.1 vs 8.5 over XNNPACK (gemma4-e4b: no
# gain). It is also what fits the context: under XNNPACK gemma4-e2b is
# OOM-killed at max_num_tokens >= 16384 (7 GB cap); YNNPACK serves 32768.
# arm64 only: the x86_64 wheels lack YNNPACK, and since 0.18.0 requesting it
# there fails engine creation instead of being ignored.
# Fails the build if upstream changes the line this anchors on.
RUN f="$(find /opt/venv -path '*litert_lm_cli/commands/serve/util.py')" \
    && sed -i 's/^\( *\)use_ringbuffers_local_attention=True,$/&\n\1enable_ynnpack=True,/' "$f" \
    && grep -q 'enable_ynnpack=True' "$f"

# Prefix (KV) cache: upstream re-prefills the whole chat on every request;
# patches/prefix_cache.py keeps the last conversation and prefills only the
# new messages. Also honours `max_tokens`. patches/apply_reasoning.py streams
# Gemma's thinking as reasoning_content (upstream generates and discards it).
# patches/memory_gate.py answers LibreChat's memory agent without the engine
# unless the user asked to remember or forget something.
# patches/citations.py ends answers built on search results with a Sources
# line the server writes from those results, since the model cites
# unpredictably. patches/presearch.py answers a plain factual question with
# a search tool call, without the engine, so the model never skips the search.
# patches/reminder.py repeats the system prompt's "Before answering:"
# paragraph in every user message: the model follows the newest turn.
# patches/length.py reports a reply cut off at max_tokens as "length", with a
# closing line saying so (upstream says "stop", an empty reply looks done).
# Fails the build on upstream drift.
COPY patches/ /tmp/patches/
RUN d="$(dirname "$(find /opt/venv -path '*litert_lm_cli/commands/serve/openai_handler.py')")" \
    && cp /tmp/patches/prefix_cache.py /tmp/patches/memory_gate.py \
          /tmp/patches/citations.py /tmp/patches/presearch.py \
          /tmp/patches/reminder.py /tmp/patches/length.py "$d/" \
    && python3 /tmp/patches/apply_prefix_cache.py "$d" \
    && python3 /tmp/patches/apply_reasoning.py "$d" \
    && python3 /tmp/patches/apply_memory_gate.py "$d" \
    && python3 /tmp/patches/apply_presearch.py "$d" \
    && python3 /tmp/patches/apply_reminder.py "$d" \
    && python3 /tmp/patches/apply_citations.py "$d" \
    && python3 /tmp/patches/apply_length.py "$d" \
    && rm -rf /tmp/patches

WORKDIR /app
COPY config/config.json.template /app/config.json.template
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Default port for `litert-lm serve`'s OpenAI-compatible API.
EXPOSE 9379

ENTRYPOINT ["/app/entrypoint.sh"]
