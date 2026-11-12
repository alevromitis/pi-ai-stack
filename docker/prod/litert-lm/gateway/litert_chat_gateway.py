"""Single-request-at-a-time HTTP gateway for LibreChat's LiteRT chat model.

The main HTTPServer is deliberately single-threaded. It does not accept the
next LibreChat request for forwarding until the current upstream response has
finished streaming. A separate lightweight health server stays responsive
while a long model request is in progress.
"""

from __future__ import annotations

import http.client
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
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
            content_length, chunked = self._request_body_shape()
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
                            or name.lower() == "host"):
                        continue
                    upstream.putheader(name, value)
                if chunked:
                    upstream.putheader("Transfer-Encoding", "chunked")
                upstream.putheader("Connection", "close")
                upstream.endheaders()

                if chunked:
                    self._forward_chunked_body(upstream)
                elif content_length:
                    self._forward_fixed_body(upstream, content_length)

                response = upstream.getresponse()
                self._send_upstream_response(response)
            finally:
                upstream.close()
        except RequestError as error:
            self.send_error(error.status, str(error))
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

    def _forward_fixed_body(self, upstream: http.client.HTTPConnection, length: int):
        remaining = length
        while remaining:
            data = self.rfile.read(min(COPY_BYTES, remaining))
            if not data:
                raise RequestError(400, "Incomplete request body")
            upstream.send(data)
            remaining -= len(data)

    def _forward_chunked_body(self, upstream: http.client.HTTPConnection):
        total = 0
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
            upstream.send(line)
            if size == 0:
                while True:
                    trailer = self.rfile.readline(8192)
                    if not trailer:
                        raise RequestError(400, "Incomplete chunk trailers")
                    upstream.send(trailer)
                    if trailer in (b"\r\n", b"\n"):
                        return
            total += size
            if total > MAX_BODY_BYTES:
                raise RequestError(413, "Request body too large")
            self._copy_exact(upstream, size + 2)

    def _copy_exact(self, upstream: http.client.HTTPConnection, length: int):
        remaining = length
        while remaining:
            data = self.rfile.read(min(COPY_BYTES, remaining))
            if not data:
                raise RequestError(400, "Incomplete request body")
            upstream.send(data)
            remaining -= len(data)

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


class SerialProxyServer(HTTPServer):
    """An HTTPServer (not ThreadingHTTPServer) to serialize model requests."""

    allow_reuse_address = True
    request_queue_size = 128

    def __init__(self, address, upstream_host: str, upstream_port: int,
                 upstream_timeout: float = 1800, client_timeout: float = 300):
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.upstream_timeout = upstream_timeout
        self.client_timeout = client_timeout
        super().__init__(address, SerialProxyHandler)

    def get_request(self):
        request, client_address = super().get_request()
        # A client that opens a connection and stops sending headers or body
        # must not hold the single upstream slot forever.
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
