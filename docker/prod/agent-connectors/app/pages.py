"""Read an RSS item's own web page, and the PDFs it links, for feeds that
carry only a title (e.g. announcement feeds, whose dates live in
attached PDFs).

The link comes from feed data, i.e. from the web, so the fetch is bounded:
http(s) only; every hop (redirects included) must resolve to public
addresses only, so nothing on the LAN or the Docker networks is reachable;
PDFs only from the page's own site; size, page and time caps.
"""

import html
import io
import ipaddress
import logging
import re
import socket
from urllib.parse import unquote, urljoin, urlsplit

import anyio
import httpx
import pypdf

from common import env_int, html_to_text, log

# pypdf warns about every font quirk (e.g. missing fontTools); text still comes out.
logging.getLogger("pypdf").setLevel(logging.ERROR)

MAX_BYTES = 4 * 1024 * 1024
MAX_PDFS = env_int("FRESHRSS_MAX_PDFS", 2)
PDF_PAGES = env_int("FRESHRSS_PDF_PAGES", 4)
TIME_LIMIT = 60  # seconds for the page and its PDFs together

http = httpx.AsyncClient(timeout=20, headers={"User-Agent": "agent-connectors (FreshRSS item reader)"})


class FetchError(Exception):
    pass


def _site(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


async def _require_public(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError("not an http(s) link")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        infos = await anyio.to_thread.run_sync(socket.getaddrinfo, parts.hostname, port)
    except OSError:
        raise FetchError(f"cannot resolve {parts.hostname}") from None
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise FetchError(f"{parts.hostname} is not a public address")


async def _get(url: str) -> tuple[str, str, bytes, str]:
    """(final url, content type, body, encoding), following up to 3 redirects."""
    for _ in range(4):
        await _require_public(url)
        async with http.stream("GET", url) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers.get("location", ""))
                continue
            if response.status_code >= 400:
                raise FetchError(f"HTTP {response.status_code}")
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body += chunk
                if len(body) > MAX_BYTES:
                    raise FetchError("larger than 4 MB")
            return url, response.headers.get("content-type", ""), bytes(body), response.encoding or "utf-8"
    raise FetchError("too many redirects")


def _main_part(markup: str) -> str:
    for tag in ("article", "main", "body"):
        match = re.search(rf"<{tag}\b.*?</{tag}>", markup, re.S | re.I)
        if match:
            return match.group(0)
    return markup


def _pdf_links(region: str, page_url: str) -> list[str]:
    links: list[str] = []
    for href in re.findall(r"""href\s*=\s*["']([^"']+)["']""", region, re.I):
        url = urljoin(page_url, html.unescape(href))
        if urlsplit(url).path.lower().endswith(".pdf") and _site(url) == _site(page_url) and url not in links:
            links.append(url)
    return links


def _pdf_text(data: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(data))
    text = "\n".join((page.extract_text() or "") for page in reader.pages[:PDF_PAGES])
    return re.sub(r"[ \t]+", " ", text).strip()


async def _read_pdf(data: bytes) -> str:
    try:
        return await anyio.to_thread.run_sync(_pdf_text, data) or "(no text layer; a scan?)"
    except Exception as exc:  # pypdf raises a zoo of errors on bad files
        return f"(unreadable PDF: {type(exc).__name__})"


async def page_text(link: str, limit: int) -> str:
    """The item page's main text plus its PDFs' text, or why it can't be read."""
    parts: list[str] = []
    try:
        with anyio.fail_after(TIME_LIMIT):
            url, content_type, body, encoding = await _get(link)
            if "pdf" in content_type or urlsplit(url).path.lower().endswith(".pdf"):
                return f"[Linked PDF]\n{await _read_pdf(body)}"
            region = _main_part(body.decode(encoding, errors="replace"))
            parts.append(html_to_text(region))
            for pdf_url in _pdf_links(region, url)[:MAX_PDFS]:
                if sum(len(p) for p in parts) >= limit:
                    break
                name = unquote(urlsplit(pdf_url).path.rsplit("/", 1)[-1])
                try:
                    _, _, data, _ = await _get(pdf_url)
                    parts.append(f"[Attachment: {name}]\n{await _read_pdf(data)}")
                except (FetchError, httpx.HTTPError) as exc:
                    parts.append(f"[Attachment: {name}: not read ({exc or type(exc).__name__})]")
    except TimeoutError:
        parts.append("[stopped: the page took too long]")
    except (FetchError, httpx.HTTPError) as exc:
        log.info("page fetch failed: %s", type(exc).__name__)
        return f"(the feed has no text and the page could not be read: {exc or type(exc).__name__})"
    return "\n\n".join(p for p in parts if p)
