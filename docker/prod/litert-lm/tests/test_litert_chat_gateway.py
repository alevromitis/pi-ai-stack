import concurrent.futures
import http.server
import http.client
import json
import socket
import threading
import time
import unittest
import urllib.request
from pathlib import Path
import importlib.util


MODULE = Path(__file__).parents[1] / 'gateway' / 'litert_chat_gateway.py'
SPEC = importlib.util.spec_from_file_location('litert_chat_gateway', MODULE)
gateway = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gateway)


class ModelHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"data":[{"id":"gemma4-e2b"}]}'
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(length)
        with self.server.active_lock:
            self.server.active += 1
            self.server.maximum_active = max(self.server.maximum_active, self.server.active)
        try:
            time.sleep(0.04)
            if self.path == '/v1/stream':
                chunks = [b'data: first\n\n', b'data: second\n\n']
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(f'{len(chunk):X}\r\n'.encode() + chunk + b'\r\n')
                    self.wfile.flush()
                self.wfile.write(b'0\r\n\r\n')
                self.wfile.flush()
            else:
                response = json.dumps({'received': body.decode()}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response)))
                self.end_headers()
                self.wfile.write(response)
        finally:
            with self.server.active_lock:
                self.server.active -= 1

    def log_message(self, *_args):
        return


class SerialGatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = http.server.ThreadingHTTPServer(('127.0.0.1', 0), ModelHandler)
        cls.model.active = 0
        cls.model.maximum_active = 0
        cls.model.active_lock = threading.Lock()
        cls.model_thread = threading.Thread(target=cls.model.serve_forever, daemon=True)
        cls.model_thread.start()
        cls.gateway = gateway.SerialProxyServer(
            ('127.0.0.1', 0), '127.0.0.1', cls.model.server_port, 5)
        cls.gateway_thread = threading.Thread(target=cls.gateway.serve_forever, daemon=True)
        cls.gateway_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.gateway.shutdown()
        cls.gateway.server_close()
        cls.gateway_thread.join(timeout=2)
        cls.model.shutdown()
        cls.model.server_close()
        cls.model_thread.join(timeout=2)

    def post(self, path, body):
        request = urllib.request.Request(
            f'http://127.0.0.1:{self.gateway.server_port}{path}',
            data=body.encode(),
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()

    def test_concurrent_completions_reach_upstream_one_at_a_time(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self.post, '/v1/chat/completions', str(i))
                       for i in range(8)]
            results = [future.result(timeout=10) for future in futures]

        self.assertEqual([status for status, _ in results], [200] * 8)
        self.assertEqual(
            sorted(json.loads(body)['received'] for _, body in results),
            [str(i) for i in range(8)],
        )
        self.assertEqual(self.model.maximum_active, 1)

    def test_streaming_response_body_is_preserved(self):
        status, body = self.post('/v1/stream', '{}')
        self.assertEqual(status, 200)
        self.assertEqual(body, b'data: first\n\ndata: second\n\n')

    def test_get_models_is_forwarded(self):
        request = urllib.request.Request(
            f'http://127.0.0.1:{self.gateway.server_port}/v1/models')
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())['data'][0]['id'], 'gemma4-e2b')

    def test_expect_continue_body_is_forwarded_without_deadlock(self):
        client = socket.create_connection(('127.0.0.1', self.gateway.server_port), timeout=5)
        stream = client.makefile('rb')
        body = b'continued body'
        client.sendall((
            'POST /v1/chat/completions HTTP/1.1\r\n'
            'Host: localhost\r\n'
            'Expect: 100-continue\r\n'
            'Content-Type: application/json\r\n'
            f'Content-Length: {len(body)}\r\n\r\n'
        ).encode())
        interim = stream.readline()
        while stream.readline() not in (b'\r\n', b'\n'):
            pass
        self.assertIn(b' 100 ', interim)
        client.sendall(body)
        status = stream.readline()
        response_headers = {}
        while True:
            line = stream.readline()
            if line in (b'\r\n', b'\n', b''):
                break
            name, value = line.decode().split(':', 1)
            response_headers[name.lower()] = value.strip()
        response_body = stream.read(int(response_headers['content-length']))
        self.assertIn(b' 200 ', status)
        self.assertEqual(json.loads(response_body)['received'], body.decode())
        stream.close()
        client.close()


class GatedModelHandler(http.server.BaseHTTPRequestHandler):
    """Records arrival order; the first request waits until the test opens the gate."""

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
        name = json.loads(body)['name']
        with self.server.lock:
            self.server.order.append(name)
            first = len(self.server.order) == 1
        if first:
            self.server.gate.wait(5)
        response = json.dumps({'name': name}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_args):
        return


def chat(name, user, step=False):
    messages = [{'role': 'system', 'content': 'agent'}, {'role': 'user', 'content': user}]
    if step:
        messages += [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': '1'}]},
                     {'role': 'tool', 'tool_call_id': '1', 'content': 'result'}]
    return json.dumps({'name': name, 'messages': messages})


class RunPriorityTests(unittest.TestCase):
    def setUp(self):
        self.model = http.server.ThreadingHTTPServer(('127.0.0.1', 0), GatedModelHandler)
        self.model.order, self.model.lock = [], threading.Lock()
        self.model.gate = threading.Event()
        threading.Thread(target=self.model.serve_forever, daemon=True).start()
        self.gateway = gateway.SerialProxyServer(
            ('127.0.0.1', 0), '127.0.0.1', self.model.server_port, 5)
        threading.Thread(target=self.gateway.serve_forever, daemon=True).start()

    def tearDown(self):
        self.model.gate.set()
        for server in (self.gateway, self.model):
            server.shutdown()
            server.server_close()

    def post(self, body):
        request = urllib.request.Request(
            f'http://127.0.0.1:{self.gateway.server_port}/v1/chat/completions',
            data=body.encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())['name']

    def queued(self, count):
        deadline = time.monotonic() + 5
        while self.gateway.slot.waiting() < count and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.gateway.slot.waiting(), count)

    def test_started_run_goes_before_new_requests(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(self.post, chat('A1', 'question A'))]
            deadline = time.monotonic() + 5
            while not self.model.order and time.monotonic() < deadline:
                time.sleep(0.01)
            # While A's first step runs: a new request, another run's step,
            # then A's own next step, in that arrival order.
            futures.append(pool.submit(self.post, chat('B1', 'question B')))
            self.queued(1)
            futures.append(pool.submit(self.post, chat('C2', 'question C', step=True)))
            self.queued(2)
            futures.append(pool.submit(self.post, chat('A2', 'question A', step=True)))
            self.queued(3)
            self.model.gate.set()
            names = sorted(f.result(timeout=10) for f in futures)
        self.assertEqual(names, ['A1', 'A2', 'B1', 'C2'])
        self.assertEqual(self.model.order, ['A1', 'A2', 'C2', 'B1'])

    def test_new_requests_keep_arrival_order(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self.post, chat('N0', 'zero'))]
            deadline = time.monotonic() + 5
            while not self.model.order and time.monotonic() < deadline:
                time.sleep(0.01)
            for index, name in enumerate(('N1', 'N2', 'N3'), 1):
                futures.append(pool.submit(self.post, chat(name, name)))
                self.queued(index)
            self.model.gate.set()
            for future in futures:
                future.result(timeout=10)
        self.assertEqual(self.model.order, ['N0', 'N1', 'N2', 'N3'])

    def test_run_shape(self):
        key_a, step = gateway.run_shape(chat('x', 'question A').encode())
        self.assertFalse(step)
        key_a2, step = gateway.run_shape(chat('y', 'question A', step=True).encode())
        self.assertTrue(step)
        self.assertEqual(key_a, key_a2)
        self.assertNotEqual(key_a, gateway.run_shape(chat('z', 'question B').encode())[0])
        self.assertEqual(gateway.run_shape(b'not json'), (None, False))


if __name__ == '__main__':
    unittest.main()
