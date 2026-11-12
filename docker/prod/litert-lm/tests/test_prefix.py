import ast
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('translation', ROOT / 'patches/translation.py')
translation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(translation)
# Exercise the actual routing function without allocating a LiteRT engine.
tree = ast.parse((ROOT / 'patches/prefix_cache.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_new_chat')


class NativePrefixTests(unittest.TestCase):
    def test_instagram_releases_snapshots_and_uses_a_fresh_conversation(self):
        snapshots = [object()]
        close = Mock(side_effect=lambda reason: snapshots.pop())
        env = {'_translation': translation, '_snapshots': snapshots,
               '_close_snapshot': close, '_trim': Mock(), '_save_warm': Mock()}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'prefix_cache.py', 'exec'), env)
        engine = Mock()
        messages = [{'role': 'system', 'content': '[instagram-personal:native]'},
                    {'role': 'user', 'content': 'hello'}]
        self.assertIsNone(env['_new_chat'](engine, None, [], messages, [], {}, {}))
        engine.create_conversation.assert_not_called()
        self.assertEqual(snapshots, [])
        env['_save_warm'].assert_called_once()

    def test_other_chats_do_not_release_snapshots(self):
        snapshots = [object()]
        env = {'_translation': translation, '_snapshots': snapshots, '_MAX_SNAPSHOTS': 0,
               '_close_snapshot': Mock()}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'prefix_cache.py', 'exec'), env)
        self.assertIsNone(env['_new_chat'](Mock(), None, [], [{'role': 'user', 'content': 'hello'}], [], {}, {}))
        self.assertEqual(len(snapshots), 1)
        env['_close_snapshot'].assert_not_called()


if __name__ == '__main__':
    unittest.main()
