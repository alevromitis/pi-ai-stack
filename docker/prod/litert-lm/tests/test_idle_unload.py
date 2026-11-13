import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]


def load(env_value):
    spec = importlib.util.spec_from_file_location('idle_unload', ROOT / 'patches/idle_unload.py')
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict('os.environ', {'LITERT_IDLE_UNLOAD_S': env_value}):
        spec.loader.exec_module(module)
        module.idle_seconds()  # read at install
    return module


class FakeServer:
    def __init__(self):
        self.litert_lm_engine = None
        self.litert_lm_embedding_engine = None

    def process_request(self, request, client_address):
        self.handled.append(request)


def fake_util():
    util = types.SimpleNamespace()
    cls = type('LiteRTLMServer', (FakeServer,), {})
    util.LiteRTLMServer = cls

    def get_engine(server, **kwargs):
        if server.litert_lm_engine is None:
            server.litert_lm_engine = mock.MagicMock(name='engine')
        return server.litert_lm_engine

    def get_embedding(server, **kwargs):
        if server.litert_lm_embedding_engine is None:
            server.litert_lm_embedding_engine = mock.MagicMock(name='embedding')
        return server.litert_lm_embedding_engine

    util.get_or_initialize_server_engine = get_engine
    util.get_or_initialize_server_embedding_engine = get_embedding
    return util


class IdleUnloadTests(unittest.TestCase):
    def setUp(self):
        self.idle = load('600')
        self.util = fake_util()
        with mock.patch.dict('os.environ', {'LITERT_IDLE_UNLOAD_S': '600'}):
            self.idle.install(self.util)
        self.server = self.util.LiteRTLMServer()
        self.server.handled = []

    def request(self, uses_engine):
        """One request through the patched process_request.

        A handler that needs the engine looks it up during the request; the
        lookup is made just before the wrapped call, which then does the
        bookkeeping exactly as after a real handler.
        """
        if uses_engine:
            self.util.get_or_initialize_server_engine(self.server, model_id='m')
        self.util.LiteRTLMServer.process_request(self.server, 'req', None)

    def test_disabled_by_default(self):
        idle = load('0')
        util = fake_util()
        before = util.LiteRTLMServer.__dict__.get('service_actions')
        with mock.patch.dict('os.environ', {'LITERT_IDLE_UNLOAD_S': '0'}):
            idle.install(util)
        self.assertIs(util.LiteRTLMServer.__dict__.get('service_actions'), before)

    def test_unloads_after_idle_and_reloads_on_demand(self):
        self.request(uses_engine=True)
        engine = self.server.litert_lm_engine
        self.assertIsNotNone(engine)
        with mock.patch('time.monotonic', return_value=self.server.idle_unload_last + 599):
            self.server.service_actions()
        self.assertIs(self.server.litert_lm_engine, engine)
        with mock.patch('time.monotonic', return_value=self.server.idle_unload_last + 600):
            self.server.service_actions()
        engine.__exit__.assert_called_once_with(None, None, None)
        self.assertIsNone(self.server.litert_lm_engine)
        self.assertIsNone(self.server.idle_unload_last)
        # The next request loads a new engine.
        self.request(uses_engine=True)
        self.assertIsNotNone(self.server.litert_lm_engine)
        self.assertIsNot(self.server.litert_lm_engine, engine)

    def test_requests_without_engine_do_not_count_as_use(self):
        self.request(uses_engine=False)
        self.assertIsNone(getattr(self.server, 'idle_unload_last', None))
        self.request(uses_engine=True)
        last = self.server.idle_unload_last
        with mock.patch('time.monotonic', return_value=last + 300):
            self.request(uses_engine=False)
        self.assertEqual(self.server.idle_unload_last, last)

    def test_embedding_engine_is_closed_too(self):
        self.util.get_or_initialize_server_embedding_engine(self.server, model_id='e')
        self.util.LiteRTLMServer.process_request(self.server, 'req', None)
        embedding = self.server.litert_lm_embedding_engine
        with mock.patch('time.monotonic', return_value=self.server.idle_unload_last + 600):
            self.server.service_actions()
        embedding.close.assert_called_once()
        self.assertIsNone(self.server.litert_lm_embedding_engine)

    def test_prefix_cache_conversations_are_closed_first(self):
        order = []
        entry = types.SimpleNamespace(conv=mock.Mock(close=lambda: order.append('entry')))
        pc = types.SimpleNamespace(_entries=[entry], _snapshots=['s'])
        pc._close_snapshot = lambda why: (pc._snapshots.pop(), order.append('snapshot'))
        self.request(uses_engine=True)
        self.server.litert_lm_engine.__exit__ = lambda *a: order.append('engine')
        with mock.patch.dict(sys.modules, {'litert_lm_cli.commands.serve.prefix_cache': pc}):
            self.assertTrue(self.idle.unload(self.server, 'test'))
        self.assertEqual(order, ['entry', 'snapshot', 'engine'])
        self.assertEqual(pc._entries, [])
        self.assertEqual(pc._snapshots, [])

    def test_nothing_loaded_is_a_no_op(self):
        self.assertFalse(self.idle.unload(self.server, 'test'))


if __name__ == '__main__':
    unittest.main()
