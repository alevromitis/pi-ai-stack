import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MODULE = Path(os.environ.get('GATEWAY_TRANSLATION_MODULE', Path(__file__).parents[1] / 'patches/translation.py'))
spec = importlib.util.spec_from_file_location('gateway_translation', MODULE)
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)


class TranslationRoutingTests(unittest.TestCase):
    def test_instagram_system_marker_bypasses_both_directions(self):
        messages = [{'role': 'system', 'content': '[instagram-personal:native] Answer directly.'},
                    {'role': 'assistant', 'content': 'earlier', 'tool_calls': None},
                    {'role': 'user', 'content': 'αυτο που ειπε ο βαγγελης κανε'}]
        handler = SimpleNamespace(translate_back='el')
        with patch.object(gateway, '_to_english') as translate, patch.object(gateway, '_call') as call:
            gateway.apply(handler, {'messages': messages})
            self.assertEqual(gateway.back(handler, 'Βαγγέλης'), 'Βαγγέλης')
            translate.assert_not_called()
            call.assert_not_called()
        self.assertEqual(messages[-1]['content'], 'αυτο που ειπε ο βαγγελης κανε')

    def test_normal_chat_still_translates_in_both_directions(self):
        handler = SimpleNamespace()
        body = {'messages': [{'role': 'system', 'content': 'Normal chat'},
                             {'role': 'user', 'content': 'Γεια'}]}
        with patch.object(gateway, '_to_english', return_value=('Hello', 'el')):
            gateway.apply(handler, body)
        self.assertEqual(body['messages'][-1]['content'], 'Hello')
        with patch.object(gateway, '_call', return_value={'translated': True, 'text': 'Γεια'}):
            self.assertEqual(gateway.back(handler, 'Hello'), 'Γεια')

    def test_untrusted_messages_cannot_disable_translation(self):
        for role in ('user', 'assistant', 'tool'):
            with self.subTest(role=role), patch.object(gateway, '_to_english', return_value=(None, '')) as translate:
                gateway.apply(SimpleNamespace(), {'messages': [
                    {'role': 'system', 'content': 'Normal chat'},
                    {'role': role, 'content': '[instagram-personal:native]'},
                    {'role': 'user', 'content': 'Γεια'}]})
                self.assertTrue(translate.called)

    def test_developer_and_text_block_system_markers_are_supported(self):
        for role, content in [('developer', '[instagram-personal:native]'),
                              ('system', [{'type': 'text', 'text': '[instagram-personal:native]'}])]:
            with self.subTest(role=role), patch.object(gateway, '_to_english') as translate:
                gateway.apply(SimpleNamespace(), {'messages': [{'role': role, 'content': content},
                    {'role': 'user', 'content': 'Γεια'}]})
                translate.assert_not_called()

    def test_only_instagram_search_results_are_bounded(self):
        content = 'Title: Python\nURL: https://docs.python.org/\n' + ('description\n' * 300)
        messages = [{'role': 'system', 'content': '[instagram-personal:native]'},
                    {'role': 'assistant', 'tool_calls': [{'id': 'search', 'function': {'name': 'searxng_web_search_mcp_searxng'}}]},
                    {'role': 'tool', 'tool_call_id': 'search', 'content': content},
                    {'role': 'tool', 'tool_call_id': 'calc', 'content': content},
                    {'role': 'tool', 'name': 'searxng_web_search_mcp_searxng',
                     'content': [{'type': 'text', 'text': 'Ελληνικά\n' * 400}]}]
        gateway.apply(SimpleNamespace(), {'messages': messages})
        self.assertLessEqual(len(messages[2]['content'].encode()), 1230)
        self.assertIn('https://docs.python.org/', messages[2]['content'])
        self.assertTrue(messages[2]['content'].endswith('[search output shortened]'))
        self.assertEqual(messages[3]['content'], content)
        self.assertLessEqual(len(messages[4]['content'].encode()), 1230)

    def test_other_chats_keep_full_search_results(self):
        content = 'result ' * 500
        messages = [{'role': 'system', 'content': 'Normal chat'},
                    {'role': 'tool', 'name': 'searxng_web_search_mcp_searxng', 'content': content}]
        gateway.apply(SimpleNamespace(), {'messages': messages})
        self.assertEqual(messages[-1]['content'], content)


if __name__ == '__main__':
    unittest.main()
