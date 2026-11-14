import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

# citations.py logs through click, which only the LiteRT image has.
try:
    import click  # noqa: F401
except ImportError:
    sys.modules['click'] = types.SimpleNamespace(echo=lambda *a, **k: None, style=lambda t, **k: t)

MODULE = Path(os.environ.get('CITATIONS_MODULE', Path(__file__).parents[1] / 'patches/citations.py'))
spec = importlib.util.spec_from_file_location('citations', MODULE)
citations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(citations)


def run(sources, chunks, known=()):
    offered = citations.Offered(sources)
    offered.asked = 'τι είναι η υπόθεση Riemann'
    offered.known = frozenset(known)
    filt = citations.Filter(offered)
    return ''.join(filt.feed(c) for c in chunks) + filt.finish(False)


def stream(text, size=3):
    return [text[i:i + size] for i in range(0, len(text), size)]


RIEMANN = [[('Riemann hypothesis', 'https://en.wikipedia.org/wiki/Riemann_hypothesis',
             'The Riemann hypothesis concerns the zeros of the Riemann zeta function, Bernhard Riemann 1859.')]]


class GreekHeaderTests(unittest.TestCase):
    def test_greek_block_of_bare_links_is_swapped_for_the_server_line(self):
        # The 2026-11-14 13:09 Instagram reply.
        text = ('Η υπόθεση του Riemann (1859) αφορά τις ρίζες της συνάρτησης ζήτα.\n\n'
                '**Πηγές:**\nhttps://github.com/x/Riemann-GPU/blob/main/README.md\n'
                'https://en.wikipedia.org/wiki/Riemann_hypothesis\n')
        for chunks in ([text], stream(text)):
            out = run(RIEMANN, chunks)
            self.assertNotIn('Πηγές', out)
            self.assertNotIn('github', out)
            self.assertTrue(out.endswith('\n\nSources: [Riemann hypothesis]'
                                         '(https://en.wikipedia.org/wiki/Riemann_hypothesis)'), out)
            self.assertEqual(out.count('Sources:'), 1)

    def test_greek_header_variants(self):
        for header in ('Πηγές:', 'Πηγη:', '### Πηγές', '- **Πηγή**:', 'ΠΗΓΕΣ:'):
            self.assertTrue(citations._HEADER.match(header + '\n'), header)
        for prose in ('Πηγή της ενέργειας είναι ο ήλιος.\n', 'Πηγαίνω σπίτι.\n'):
            self.assertFalse(citations._HEADER.match(prose), prose)

    def test_greek_prose_starting_like_a_header_streams(self):
        text = 'Πηγαίνω σπίτι τώρα.\nΑύριο πάλι.'
        self.assertEqual(run([], stream(text, 2)), text)

    def test_made_up_links_without_search_dropped_in_greek(self):
        out = run([], ['Η απάντηση είναι 42.\n\nΠηγές:\nhttps://example.com/made-up\n'])
        self.assertEqual(out.strip(), 'Η απάντηση είναι 42.')

    def test_known_links_without_search_pass(self):
        text = 'Όπως είπαμε.\n\nΠηγές:\nhttps://en.wikipedia.org/wiki/Riemann_hypothesis\n'
        self.assertEqual(run([], [text], known=['https://en.wikipedia.org/wiki/Riemann_hypothesis']), text)


class NamesOnlyTests(unittest.TestCase):
    def test_file_names_without_links_pass(self):
        for text in ('Η εξεταστική ξεκινά 25/1.\n\nΠηγές:\n• Οδηγός Σπουδών <feed>\n• Ακαδημαϊκό Ημερολόγιο\n',
                     'The exam starts on 25/1.\n\nSources: Study guide\n'):
            self.assertEqual(run([], stream(text)), text)

    def test_empty_or_fake_link_lists_still_go(self):
        self.assertEqual(run([], ['Done.\n\nSources: []\n']).strip(), 'Done.')
        self.assertEqual(run([], ['Done.\n\nSources: [Odyssey]([URL not provided])\n']).strip(), 'Done.')
        self.assertEqual(run([], ['Done.\n\nSources: []\n\nHope this helps!']).strip(), 'Done.\n\nHope this helps!')


if __name__ == '__main__':
    unittest.main()
