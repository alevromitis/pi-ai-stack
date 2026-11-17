"""Single-request-at-a-time HTTP gateway for LibreChat's LiteRT chat model.

Only one request reaches the model at a time: the next one is forwarded
when the current upstream response has finished streaming. Requests are
accepted and read in parallel, and the free slot goes to

1. the next step of the agent run that just had the model,
2. the next step of any other run already in progress,
3. a new request, oldest first.

An agent reply is several model calls (think, call a tool, read its result,
answer). In arrival order, a run that was mid-way waited behind every
request that came in meanwhile, so its reply took as long as all of them.
Letting a started run finish first sends each reply when its own work is
done, and its steps reuse litert-lm's prefix cache for that conversation.
A run step is a request whose messages already hold a tool call or tool
result; the run is named by its first system and user messages.

A separate lightweight health server stays responsive while a long model
request is in progress.
"""

from __future__ import annotations

import hashlib
import http.client
import itertools
import json
import logging
import os
import select
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

LOG = logging.getLogger("litert_chat_gateway")
MAX_BODY_BYTES = 128 * 1024 * 1024
COPY_BYTES = 64 * 1024
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "expect", "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade",
}


class RequestError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# Lower runs first: the run that last had the model, any started run, the
# rest (new requests, model lists, anything that is not a chat request).
PRIORITY_SAME_RUN, PRIORITY_RUN_STEP, PRIORITY_NEW = 0, 1, 2


def run_shape(body: bytes) -> tuple[str | None, bool]:
    """(run key, is a later step of a run) of a chat completions body.

    The key hashes the system and first user message, which every step of
    one agent run repeats. Unreadable bodies are new requests."""
    try:
        data = json.loads(body)
        messages = data.get("messages")
        if not isinstance(messages, list):
            return None, False
    except (ValueError, AttributeError, UnicodeDecodeError):
        return None, False
    head, step = [], False
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "tool" or (role == "assistant" and message.get("tool_calls")):
            step = True
        if len(head) < 2 and role in ("system", "developer", "user"):
            if role == "user" and any(m.get("role") == "user" for m in head):
                continue
            head.append({"role": role, "content": message.get("content")})
    if not any(m["role"] == "user" for m in head):
        return None, step
    key = hashlib.sha256(json.dumps(head, sort_keys=True, ensure_ascii=False)
                         .encode("utf-8", "surrogatepass")).hexdigest()
    return key, step


class Slot:
    """The single upstream slot, granted by priority, then arrival order."""

    def __init__(self):
        self._lock = threading.Condition()
        self._busy = False
        self._waiting: list[tuple[int, str | None, bool]] = []
        self._order = itertools.count()
        self.last_run: str | None = None

    def acquire(self, run_key: str | None, step: bool) -> None:
        with self._lock:
            entry = (next(self._order), run_key, step)
            self._waiting.append(entry)
            try:
                while self._busy or self._first() is not entry:
                    self._lock.wait()
            finally:
                self._waiting.remove(entry)
            self._busy = True
            if run_key is not None:
                self.last_run = run_key

    def release(self) -> None:
        with self._lock:
            self._busy = False
            self._lock.notify_all()

    def _priority(self, entry) -> int:
        _order, run_key, step = entry
        if step and run_key is not None and run_key == self.last_run:
            return PRIORITY_SAME_RUN
        return PRIORITY_RUN_STEP if step else PRIORITY_NEW

    def _first(self):
        return min(self._waiting, key=lambda e: (self._priority(e), e[0]))

    def waiting(self) -> int:
        with self._lock:
            return len(self._waiting)


def client_gone(connection: socket.socket) -> bool:
    """True when the caller closed its connection while it waited."""
    try:
        readable, _, _ = select.select([connection], [], [], 0)
        if not readable:
            return False
        return connection.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError):
        return True


class SerialProxyHandler(BaseHTTPRequestHandler):
    # HTTP/1.1 lets us acknowledge Expect: 100-continue before reading a body.
    # Every final response below still closes the client connection.
    protocol_version = "HTTP/1.1"
    server_version = "LibreChat-LiteRT-Serial-Gateway/1.0"
    sys_version = ""

    def do_GET(self):
        self._proxy()

    def do_HEAD(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def handle_expect_100(self):
        self.send_response_only(100)
        self.end_headers()
        return True

    def _proxy(self):
        self.close_connection = True
        target = urlsplit(self.path)
        if target.scheme or target.netloc or not target.path.startswith("/v1/"):
            self.send_error(404, "Only LiteRT v1 endpoints are available")
            return

        try:
            # The whole body is read before waiting for the slot: the run
            # it belongs to decides its place, and a slow sender must not
            # hold the model.
            content_length, chunked = self._request_body_shape()
            if chunked:
                body = self._read_chunked_body()
            elif content_length:
                body = self._read_fixed_body(content_length)
            else:
                body = b""
        except RequestError as error:
            self.send_error(error.status, str(error))
            return
        except (OSError, TimeoutError):
            return

        run_key, step = (run_shape(body) if self.command == "POST" else (None, False))
        slot = self.server.slot
        slot.acquire(run_key, step)
        try:
            if client_gone(self.connection):
                LOG.info("caller left while queued; request dropped")
                return
            self._forward(body)
        finally:
            slot.release()

    def _forward(self, body: bytes):
        try:
            upstream = http.client.HTTPConnection(
                self.server.upstream_host,
                self.server.upstream_port,
                timeout=self.server.upstream_timeout,
            )
            try:
                upstream.putrequest(
                    self.command,
                    self.path,
                    skip_host=True,
                    skip_accept_encoding=True,
                )
                upstream.putheader(
                    "Host", f"{self.server.upstream_host}:{self.server.upstream_port}")
                connection_tokens = {
                    part.strip().lower()
                    for value in self.headers.get_all("Connection", [])
                    for part in value.split(",")
                }
                for name, value in self.headers.items():
                    if (name.lower() in HOP_BY_HOP
                            or name.lower() in connection_tokens
                            or name.lower() in ("host", "content-length")):
                        continue
                    upstream.putheader(name, value)
                if body or self.command == "POST":
                    upstream.putheader("Content-Length", str(len(body)))
                upstream.putheader("Connection", "close")
                upstream.endheaders()
                if body:
                    upstream.send(body)

                response = upstream.getresponse()
                self._send_upstream_response(response)
            finally:
                upstream.close()
        except (OSError, http.client.HTTPException, TimeoutError) as error:
            LOG.warning("upstream request failed: %s", type(error).__name__)
            try:
                self.send_error(502, "LiteRT model server unavailable")
            except OSError:
                pass

    def _request_body_shape(self) -> tuple[int, bool]:
        transfer = self.headers.get("Transfer-Encoding", "").strip().lower()
        length_value = self.headers.get("Content-Length")
        if transfer:
            if transfer != "chunked" or length_value is not None:
                raise RequestError(400, "Invalid request body framing")
            return 0, True
        if length_value is None:
            return 0, False
        try:
            length = int(length_value)
        except ValueError as error:
            raise RequestError(400, "Invalid Content-Length") from error
        if length < 0:
            raise RequestError(400, "Invalid Content-Length")
        if length > MAX_BODY_BYTES:
            raise RequestError(413, "Request body too large")
        return length, False

    def _read_fixed_body(self, length: int) -> bytes:
        parts, remaining = [], length
        while remaining:
            data = self.rfile.read(min(COPY_BYTES, remaining))
            if not data:
                raise RequestError(400, "Incomplete request body")
            parts.append(data)
            remaining -= len(data)
        return b"".join(parts)

    def _read_chunked_body(self) -> bytes:
        parts, total = [], 0
        while True:
            line = self.rfile.readline(8192)
            if not line or not line.endswith(b"\n"):
                raise RequestError(400, "Invalid chunked request body")
            try:
                size = int(line.split(b";", 1)[0].strip(), 16)
            except ValueError as error:
                raise RequestError(400, "Invalid chunk size") from error
            if size < 0:
                raise RequestError(400, "Invalid chunk size")
            if size == 0:
                while True:
                    trailer = self.rfile.readline(8192)
                    if not trailer:
                        raise RequestError(400, "Incomplete chunk trailers")
                    if trailer in (b"\r\n", b"\n"):
                        return b"".join(parts)
            total += size
            if total > MAX_BODY_BYTES:
                raise RequestError(413, "Request body too large")
            chunk = self._read_fixed_body(size + 2)
            if chunk[-2:] != b"\r\n":
                raise RequestError(400, "Invalid chunked request body")
            parts.append(chunk[:-2])

    def _send_upstream_response(self, response: http.client.HTTPResponse):
        self.send_response_only(response.status, response.reason)
        connection_tokens = {
            part.strip().lower()
            for value in response.headers.get_all("Connection", [])
            for part in value.split(",")
        }
        excluded = HOP_BY_HOP | connection_tokens
        for name, value in response.getheaders():
            if name.lower() not in excluded:
                self.send_header(name, value)
        self.send_header("Connection", "close")
        self.end_headers()

        if self.command == "HEAD" or response.status in (204, 304):
            return
        client_connected = True
        while True:
            chunk = response.read1(COPY_BYTES)
            if not chunk:
                break
            if client_connected:
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (OSError, TimeoutError):
                    # Keep the upstream slot until generation ends even if the
                    # caller disconnects in the middle of a streamed response.
                    client_connected = False

    def log_message(self, fmt: str, *args):
        LOG.info("%s - %s", self.client_address[0], fmt % args)


class SerialProxyServer(ThreadingHTTPServer):
    """Accepts in parallel; Slot lets one request at a time reach the model."""

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, address, upstream_host: str, upstream_port: int,
                 upstream_timeout: float = 1800, client_timeout: float = 300):
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.upstream_timeout = upstream_timeout
        self.client_timeout = client_timeout
        self.slot = Slot()
        super().__init__(address, SerialProxyHandler)

    def get_request(self):
        request, client_address = super().get_request()
        # A client that opens a connection and stops sending headers or body
        # must not hold a handler thread forever. It no longer holds the
        # upstream slot: bodies are read before the slot is requested.
        request.settimeout(self.client_timeout)
        return request, client_address


class HealthHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        body = b"ok\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _fmt: str, *_args):
        return


def serve():
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    bind = os.getenv("GATEWAY_BIND", "0.0.0.0")
    port = int(os.getenv("GATEWAY_PORT", "9379"))
    health_port = int(os.getenv("GATEWAY_HEALTH_PORT", "9380"))
    upstream_host = os.getenv("UPSTREAM_HOST", "litert-lm-1")
    upstream_port = int(os.getenv("UPSTREAM_PORT", "9379"))
    upstream_timeout = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "1800"))
    client_timeout = float(os.getenv("CLIENT_IO_TIMEOUT_SECONDS", "300"))

    health = ThreadingHTTPServer((bind, health_port), HealthHandler)
    threading.Thread(target=health.serve_forever, name="health", daemon=True).start()
    proxy = SerialProxyServer(
        (bind, port), upstream_host, upstream_port, upstream_timeout, client_timeout)
    LOG.info("serial gateway listening on %s:%d; upstream=%s:%d",
             bind, port, upstream_host, upstream_port)
    try:
        proxy.serve_forever(poll_interval=0.5)
    finally:
        proxy.server_close()
        health.shutdown()
        health.server_close()


if __name__ == "__main__":
    serve()
