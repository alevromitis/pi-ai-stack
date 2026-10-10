"""LibreChat's Qwen endpoint: the workstation's Qwen3.5-9B, else Gemma on the Pi.

POST /v1/chat/completions goes unchanged to llama-server on the workstation
(PRIMARY_BASE_URL). When that is unreachable, still loading, or answers 5xx
before sending anything, the request goes to litert-lm-1 instead, rewritten
into what LibreChat sends with the FALLBACK_PRESET preset ("E2B - Thinker",
read from MongoDB on each fallback, so edits to it apply):

  * model, temperature, top_p, penalties, reasoning_effort and max_tokens
    from the preset, then the fallback endpoint's addParams (top_k) from
    librechat.yaml; the Qwen-only sampling params are not sent.
  * the Qwen prompt that starts the system message (the promptPrefix of a
    spec or preset on PRIMARY_ENDPOINT, its {{vars}} matched as wildcards)
    becomes the preset's prompt, so what LibreChat appends after it
    (memories, MCP server instructions) stays.
  * the oldest turns are dropped until the request fits the preset's
    maxContextTokens.
  * the reply opens with a line of reasoning saying the Pi is answering.

A failure after the reply has started is passed on, not retried.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
import time
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp
import yaml
from aiohttp import web
from pymongo import MongoClient

PRIMARY_BASE_URL = os.environ["PRIMARY_BASE_URL"].rstrip("/")
PRIMARY_API_KEY = os.environ["PRIMARY_API_KEY"]
PRIMARY_MODEL = os.environ.get("PRIMARY_MODEL", "qwen3.5-9b")
PRIMARY_ENDPOINT = os.environ.get("PRIMARY_ENDPOINT", "Qwen")
FALLBACK_BASE_URL = os.environ["FALLBACK_BASE_URL"].rstrip("/")
FALLBACK_API_KEY = os.environ.get("FALLBACK_API_KEY", "litert-lm")
FALLBACK_ENDPOINT = os.environ.get("FALLBACK_ENDPOINT", "LiteRT-LM")
FALLBACK_PRESET = os.environ.get("FALLBACK_PRESET", "E2B - Thinker")
# Used when the preset is missing from MongoDB.
FALLBACK_SPEC = os.environ.get("FALLBACK_SPEC", "pi-chat")
MONGO_URI = os.environ["MONGO_URI"]
LIBRECHAT_YAML = os.environ.get("LIBRECHAT_YAML", "/config/librechat.yaml")
TZ = ZoneInfo(os.environ.get("TZ", "UTC"))
# A failed health check skips the workstation for this long, so each tool
# round of a turn doesn't wait out the connect timeout again.
DOWN_TTL = float(os.environ.get("PRIMARY_DOWN_TTL", "20"))
CONNECT_TIMEOUT = float(os.environ.get("PRIMARY_CONNECT_TIMEOUT", "3"))
# Characters per token when trimming for the Pi (Gemma and tiktoken both
# average ~4 on English; lower is safer).
CHARS_PER_TOKEN = float(os.environ.get("FALLBACK_CHARS_PER_TOKEN", "3.5"))
# Reply tokens kept free when the preset sets no max_tokens (pi-chat's value).
REPLY_RESERVE = int(os.environ.get("FALLBACK_REPLY_RESERVE", "4096"))

# The OpenAI request fields that mean the same to LiteRT-LM; sampling and
# llama.cpp-only fields are rebuilt from the preset instead.
KEEP_FIELDS = (
    "messages", "tools", "tool_choice", "parallel_tool_calls", "stream",
    "stream_options", "stop", "n", "response_format", "user",
)
# Preset fields LibreChat sends as request parameters to a custom endpoint.
PRESET_PARAMS = (
    "temperature", "top_p", "presence_penalty", "frequency_penalty",
    "max_tokens", "reasoning_effort",
)
_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")

log = logging.getLogger("router")
_mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
_down_until = 0.0
_session: aiohttp.ClientSession | None = None


class Unavailable(Exception):
    pass


# --- Config: presets, prompts, endpoint params -------------------------------

def _yaml() -> dict[str, Any]:
    with open(LIBRECHAT_YAML) as f:
        return yaml.safe_load(f) or {}


def _load_config() -> dict[str, Any]:
    """The fallback preset, its endpoint's add/dropParams, the Qwen prompts."""
    conf = _yaml()
    specs = (conf.get("modelSpecs") or {}).get("list") or []
    db = _mongo.get_default_database()
    preset = db.presets.find_one({"title": FALLBACK_PRESET}, sort=[("updatedAt", -1)])
    if preset is None:
        log.warning("preset %r not in MongoDB; using spec %r", FALLBACK_PRESET, FALLBACK_SPEC)
        preset = next(s["preset"] for s in specs if s.get("name") == FALLBACK_SPEC)
    endpoint = next(
        (e for e in (conf.get("endpoints") or {}).get("custom") or []
         if e.get("name") == FALLBACK_ENDPOINT),
        {},
    )
    prompts = {s["preset"].get("promptPrefix") for s in specs
               if (s.get("preset") or {}).get("endpoint") == PRIMARY_ENDPOINT}
    prompts |= {p.get("promptPrefix") for p in db.presets.find(
        {"endpoint": PRIMARY_ENDPOINT}, {"promptPrefix": 1})}
    prompts = sorted((p.strip() for p in prompts if p and p.strip()), key=len, reverse=True)
    return {
        "preset": preset,
        "add": endpoint.get("addParams") or {},
        "drop": endpoint.get("dropParams") or [],
        "prompts": prompts,
    }


def _template_match(template: str, text: str) -> tuple[int, dict[str, str]] | None:
    """Where `template` (with LibreChat's {{vars}} rendered) ends at the start of text."""
    parts = _VAR.split(template)
    names = parts[1::2]
    pattern = "".join(re.escape(p) if i % 2 == 0 else "(.*?)" for i, p in enumerate(parts))
    m = re.compile(pattern, re.DOTALL).match(text)
    if not m:
        return None
    return m.end(), dict(zip(names, m.groups()))


def _render(template: str, values: dict[str, str]) -> str:
    """Fill {{vars}} as LibreChat's replaceSpecialVars does, preferring values
    captured from the Qwen prompt so both say the same date."""
    now = datetime.datetime.now(TZ)
    day = now.strftime("%A")
    z = now.strftime("%z")
    own = {
        "current_date": f"{now:%Y-%m-%d} ({day})",
        "current_datetime": f"{now:%Y-%m-%d %H:%M:%S} {z[:3]}:{z[3:]} ({day})",
        "iso_datetime": now.astimezone(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
    }

    def fill(m: re.Match[str]) -> str:
        name = m.group(1).lower()
        return values.get(name) or own.get(name) or m.group(0)

    return _VAR.sub(fill, template.strip())


# --- Request rewriting for the Pi ----------------------------------------------

def _system_text(msg: dict[str, Any]) -> str | None:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[0], dict) and c[0].get("type") == "text":
        return c[0].get("text", "")
    return None


def _set_system_text(msg: dict[str, Any], text: str) -> None:
    if isinstance(msg.get("content"), str):
        msg["content"] = text
    else:
        msg["content"][0]["text"] = text


def _swap_prompt(messages: list[dict[str, Any]], cfg: dict[str, Any]) -> bool:
    if not messages or messages[0].get("role") != "system":
        return False
    text = _system_text(messages[0])
    target = cfg["preset"].get("promptPrefix") or ""
    if text is None or not target.strip():
        return False
    for prompt in cfg["prompts"]:
        hit = _template_match(prompt, text)
        if hit:
            end, values = hit
            _set_system_text(messages[0], _render(target, values) + text[end:])
            return True
    return False


def _tokens(obj: Any) -> int:
    return int(len(json.dumps(obj, ensure_ascii=False)) / CHARS_PER_TOKEN)


def _trim(body: dict[str, Any], budget: int) -> int:
    """Drop the oldest turns (a user message and what follows it) until the
    request fits; the system message and the current turn always stay."""
    msgs = body["messages"]
    head = 1 if msgs and msgs[0].get("role") == "system" else 0
    fixed = _tokens(msgs[:head]) + _tokens(body.get("tools") or [])
    dropped = 0
    while True:
        users = [i for i in range(head, len(msgs)) if msgs[i].get("role") == "user"]
        if len(users) < 2 or fixed + _tokens(msgs[head:]) <= budget:
            return dropped
        cut = users[1]
        dropped += cut - head
        del msgs[head:cut]


def _fallback_body(body: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    preset = cfg["preset"]
    out = {k: body[k] for k in KEEP_FIELDS if k in body}
    out["messages"] = json.loads(json.dumps(body.get("messages") or []))
    out["model"] = preset.get("model")
    for k in PRESET_PARAMS:
        if preset.get(k) not in (None, ""):
            out[k] = preset[k]
    out.update(cfg["add"])
    for k in cfg["drop"]:
        out.pop(k, None)
    swapped = _swap_prompt(out["messages"], cfg)
    window = int(preset.get("maxContextTokens") or 13312)
    dropped = _trim(out, window - int(preset.get("max_tokens") or REPLY_RESERVE))
    log.info("fallback: preset %r, prompt %s, %d old messages dropped",
             preset.get("title") or FALLBACK_SPEC, "swapped" if swapped else "kept", dropped)
    return out


# --- HTTP -------------------------------------------------------------------------

async def _primary_up() -> bool:
    global _down_until
    if time.monotonic() < _down_until:
        return False
    try:
        async with _session.get(
            PRIMARY_BASE_URL.removesuffix("/v1") + "/health",
            timeout=aiohttp.ClientTimeout(total=CONNECT_TIMEOUT),
        ) as r:
            if r.status == 200:
                return True
            log.info("primary health: HTTP %d", r.status)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        log.info("primary health: %s", type(e).__name__)
    _down_until = time.monotonic() + DOWN_TTL
    return False


async def _relay(request: web.Request, upstream: aiohttp.ClientResponse,
                 note: str | None = None) -> web.StreamResponse:
    resp = web.StreamResponse(status=upstream.status)
    resp.content_type = upstream.content_type
    await resp.prepare(request)
    if note:
        await resp.write(note.encode())
    async for chunk in upstream.content.iter_any():
        await resp.write(chunk)
    await resp.write_eof()
    return resp


async def _post(base: str, key: str, body: dict[str, Any]) -> aiohttp.ClientResponse:
    return await _session.post(
        base + "/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {key}"},
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=CONNECT_TIMEOUT),
    )


def _note(preset: dict[str, Any]) -> str:
    chunk = {
        "id": "chatcmpl-router-note",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": preset.get("model"),
        "choices": [{"index": 0, "finish_reason": None, "delta": {
            "role": "assistant",
            "reasoning_content": "[The workstation is unavailable, so Gemma 4 E2B on the "
                                 f"Pi is answering, with the {preset.get('title') or FALLBACK_SPEC} preset.]\n\n",
        }}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def chat(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    if await _primary_up():
        try:
            upstream = await _post(PRIMARY_BASE_URL, PRIMARY_API_KEY, body)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.info("primary request failed: %s", type(e).__name__)
        else:
            if upstream.status < 500:
                async with upstream:
                    log.info("primary: HTTP %d", upstream.status)
                    return await _relay(request, upstream)
            log.info("primary: HTTP %d, falling back", upstream.status)
            upstream.release()
    cfg = await asyncio.to_thread(_load_config)
    fb = _fallback_body(body, cfg)
    msgs = fb["messages"]
    note = _note(cfg["preset"]) if fb.get("stream") and msgs and msgs[-1].get("role") == "user" else None
    async with await _post(FALLBACK_BASE_URL, FALLBACK_API_KEY, fb) as upstream:
        log.info("fallback: HTTP %d", upstream.status)
        return await _relay(request, upstream, note if upstream.status == 200 else None)


async def models(_: web.Request) -> web.Response:
    return web.json_response({"object": "list", "data": [
        {"id": PRIMARY_MODEL, "object": "model", "owned_by": "router"}]})


async def health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def _start(_: web.Application) -> None:
    global _session
    _session = aiohttp.ClientSession()


async def _stop(_: web.Application) -> None:
    await _session.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.on_startup.append(_start)
    app.on_cleanup.append(_stop)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    web.run_app(app, port=int(os.environ.get("PORT", "8000")), access_log=None)


if __name__ == "__main__":
    main()
