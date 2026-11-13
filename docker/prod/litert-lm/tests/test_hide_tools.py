import copy
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE = Path(__file__).parents[1] / 'patches/hide_tools.py'


def load(env=None):
    with patch.dict(os.environ, env or {}, clear=False):
        if env is None:
            os.environ.pop('LITERT_TOOL_DESCRIPTIONS', None)
        spec = importlib.util.spec_from_file_location('hide_tools_under_test', MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def tool(name, description, properties, required=()):
    return {'type': 'function', 'function': {
        'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties, 'required': list(required)}}}


def shape(schema):
    if isinstance(schema, dict):
        return {k: shape(v) for k, v in schema.items() if k != 'description'}
    if isinstance(schema, list):
        return [shape(x) for x in schema]
    return schema


class HideToolsTests(unittest.TestCase):
    def setUp(self):
        self.tools = [
            tool('edit_file', 'A long manual. ' * 40, {
                'path': {'type': 'string', 'description': 'long ' * 30},
                'old_text': {'type': 'string', 'description': 'x'},
                'new_text': {'type': 'string', 'description': 'Replacement text.'},
                'edits': {'type': 'array', 'description': 'batch',
                          'items': {'type': 'object', 'properties': {'old_text': {'type': 'string'}}}},
            }, required=['path']),
            tool('searxng_instance_info_mcp_searxng', 'info', {}),
            tool('calculate_mcp_calc', 'Exact arithmetic.', {'expression': {'type': 'string'}}, ['expression']),
        ]

    def test_shortens_text_but_keeps_every_name_type_and_required_list(self):
        hide = load()
        body = {'tools': copy.deepcopy(self.tools)}
        hide.apply(body)
        names = [t['function']['name'] for t in body['tools']]
        self.assertEqual(names, ['edit_file', 'calculate_mcp_calc'])
        edit = body['tools'][0]['function']
        self.assertEqual(edit['description'], 'Replace text in an existing skill file. Returns a diff.')
        props = edit['parameters']['properties']
        self.assertEqual(props['path']['description'], 'skills/{skill}/{file}')
        self.assertNotIn('description', props['new_text'])
        self.assertEqual(shape(edit['parameters']), shape(self.tools[0]['function']['parameters']))
        self.assertEqual(body['tools'][1], self.tools[2])
        self.assertIn('A long manual', self.tools[0]['function']['description'])  # input not mutated

    def test_env_replaces_the_defaults(self):
        hide = load({'LITERT_TOOL_DESCRIPTIONS': '{"calculate_mcp_calc": {"description": "Maths.", '
                                                 '"params": {"expression": "e.g. 2+2", "gone": "x"}}}'})
        body = {'tools': copy.deepcopy(self.tools)}
        hide.apply(body)
        self.assertIn('A long manual', body['tools'][0]['function']['description'])
        calc = body['tools'][1]['function']
        self.assertEqual(calc['description'], 'Maths.')
        self.assertEqual(calc['parameters']['properties'], {'expression': {'type': 'string', 'description': 'e.g. 2+2'}})

    def test_empty_env_changes_no_text(self):
        hide = load({'LITERT_TOOL_DESCRIPTIONS': ''})
        body = {'tools': copy.deepcopy(self.tools)}
        hide.apply(body)
        self.assertEqual(body['tools'][0], self.tools[0])


if __name__ == '__main__':
    unittest.main()
