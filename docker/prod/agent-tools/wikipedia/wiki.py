"""Search a Kiwix Wikipedia ZIM and pick the passages that answer a query.

Three steps, chosen by measurement (2026-10-09, 44 questions, see README
"agent-tools"):

  candidates  libzim's title suggestions first (named things: "Eiffel
              Tower" beats "Eiffel Tower replicas"), then the ZIM's Xapian
              full-text index with every word, topped up with any word
  passages    the top articles split into lead / sections / paragraphs,
              infobox rows kept as one line; references, see-also etc. dropped
  ranking     BM25 over those paragraphs, nudged towards leads and
              higher-ranked articles
"""

from __future__ import annotations

import math
import re
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field

from libzim.reader import Archive
from libzim.search import Query, Searcher
from libzim.suggestion import SuggestionSearcher
from selectolax.lexbor import LexborHTMLParser, LexborNode

SKIP_SECTIONS = {
    "see also", "references", "notes", "external links", "further reading",
    "bibliography", "sources", "citations", "footnotes", "works cited",
    "notes and references", "references and notes", "general and cited sources",
    "explanatory notes", "cited sources", "general references",
}
# Removed before reading an article: citation markers, navigation, images.
STRIP = (
    "style", "script", "sup.reference", ".mw-ref", ".reference", ".navbox",
    ".mw-editsection", ".hatnote", ".metadata", ".shortdescription", "figure",
    ".thumb", ".gallery", ".mw-empty-elt", ".noprint", ".sidebar",
)
STOP = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor
not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself yourselves
called name named whose year years tell give many much
""".split())
TOKEN = re.compile(r"[\wÀ-ɏ]+")
CITATION = re.compile(r"\[\s*(\d+|[a-z]|citation needed|note \d+|update)\s*\]")


def words(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def page_url(base: str, path: str) -> str:
    """An article's link: base is a kiwix-serve /content/<name>/ or en.wikipedia.org/wiki/."""
    return base + urllib.parse.quote(path, safe="/_()',-.")


@dataclass
class Section:
    title: str  # "" for the lead
    level: int
    paras: list[str] = field(default_factory=list)


@dataclass
class Passage:
    title: str
    path: str  # the ZIM entry, e.g. "Mitsotakis_family"
    section: str
    text: str
    article_rank: int
    score: float = 0.0


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", CITATION.sub("", text)).strip()


def _infobox(table: LexborNode) -> str:
    rows = []
    for tr in table.css("tr"):
        th, td = tr.css_first("th"), tr.css_first("td")
        if th is not None and td is not None:
            key, value = _clean(th.text(separator=" ")), _clean(td.text(separator=" "))
            if key and value and len(value) < 200:
                rows.append(f"{key}: {value}")
    return "; ".join(rows)


def sections(html: str) -> list[Section]:
    """An mwoffliner article as lead + sections of plain-text paragraphs."""
    tree = LexborHTMLParser(html)
    for selector in STRIP:
        for node in tree.css(selector):
            node.decompose()
    out = [Section("", 1)]
    skipping = False

    def walk(node: LexborNode) -> None:
        nonlocal skipping
        for child in node.iter():
            tag = child.tag
            if tag in ("h2", "h3", "h4"):
                title = _clean(child.text(separator=" "))
                if tag == "h2":
                    skipping = title.lower() in SKIP_SECTIONS
                if not skipping:
                    out.append(Section(title, int(tag[1])))
            elif skipping:
                # mwoffliner nests each section in <details>; the next h2
                # inside one still has to be seen to stop skipping.
                if tag in ("details", "section", "div", "summary"):
                    walk(child)
            elif tag == "p":
                text = _clean(child.text(separator=""))
                if len(text) > 30:
                    out[-1].paras.append(text)
            elif tag in ("ul", "ol"):
                items = [i for i in (_clean(li.text(separator=" ")) for li in child.css("li")) if i]
                if sum(map(len, items)) > 40:
                    out[-1].paras.append(" • ".join(items)[:1500])
            elif tag == "table":
                if "infobox" in (child.attributes.get("class") or ""):
                    rows = _infobox(child)
                    if rows:
                        out[0].paras.insert(0, "Infobox: " + rows)
            elif tag != "h1":
                walk(child)

    walk(tree.body or tree.root)
    return [s for s in out if s.paras or s.title]


class Wiki:
    """One open ZIM. Not thread-safe: callers serialise access."""

    def __init__(self, path: str):
        self.archive = Archive(path)
        if not self.archive.has_fulltext_index:
            raise SystemExit(f"{path} has no full-text index")
        self.searcher = Searcher(self.archive)
        self.suggester = SuggestionSearcher(self.archive)
        try:
            self.date = bytes(self.archive.get_metadata("Date")).decode()  # e.g. 2026-06-17
        except KeyError:
            self.date = "unknown date"

    def _resolve(self, path: str):
        entry = self.archive.get_entry_by_path(path)
        for _ in range(5):
            if not entry.is_redirect:
                break
            entry = entry.get_redirect_entry()
        return entry

    def _collect(self, paths, out: dict[str, str], limit: int) -> None:
        for path in paths:
            if len(out) >= limit:
                return
            try:
                entry = self._resolve(path)
            except KeyError:
                continue
            out.setdefault(entry.path, entry.title)

    def candidates(self, query: str, limit: int = 20) -> list[tuple[str, str]]:
        """[(path, title)]: title suggestions, then all words, then any word."""
        terms = words(query) or TOKEN.findall(query.lower())
        if not terms:
            return []
        found: dict[str, str] = {}
        self._collect(self.suggester.suggest(" ".join(terms)).getResults(0, 3), found, limit)
        for text in (" ".join(terms), " OR ".join(terms)):
            if len(found) >= limit:
                break
            self._collect(self.searcher.search(Query().set_query(text)).getResults(0, limit), found, limit)
        return list(found.items())

    def passages(self, query: str, articles: int = 3, per_article: int = 60) -> list[Passage]:
        """Paragraphs of the top `articles` candidates, best first (BM25)."""
        out: list[Passage] = []
        for rank, (path, title) in enumerate(self.candidates(query)[:articles]):
            html = bytes(self._resolve(path).get_item().content).decode("utf-8", "replace")
            paras = [(s.title, p) for s in sections(html) for p in s.paras][:per_article]
            out.extend(Passage(title, path, section, text, rank) for section, text in paras)
        return bm25(query, out)


def bm25(query: str, passages: list[Passage], k1: float = 1.2, b: float = 0.75) -> list[Passage]:
    terms = words(query)
    if not passages or not terms:
        return passages
    docs = [Counter(words(f"{p.title} {p.section} {p.text}")) for p in passages]
    n = len(docs)
    avgdl = sum(sum(d.values()) for d in docs) / n or 1.0
    df = Counter(t for d in docs for t in set(d))
    for p, d in zip(passages, docs):
        length = sum(d.values())
        score = 0.0
        for t in terms:
            if t in d:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * d[t] * (k1 + 1) / (d[t] + k1 * (1 - b + b * length / avgdl))
        # Leads hold definitions and headline facts; prefer better articles.
        p.score = score + 0.3 / (1 + p.article_rank) + (0.2 if not p.section else 0.0)
    return sorted(passages, key=lambda p: -p.score)


def trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = cut.rfind(". ")
    return (cut[: stop + 1] if stop > limit * 0.6 else cut.rstrip()) + " …"


def render(query: str, passages: list[Passage], date: str, results: int, chars: int, link_base: str) -> str:
    """Passages, then one ready-made Sources line the model is told to end with.

    Measured 2026-10-10 (gemma4-e2b, 12 questions): a link under each
    passage was cited 0/12; this line plus pi-chat's prompt rule, 11/12.
    """
    lines = [f"Wikipedia (offline copy, {date[:7]}) for: {query}"]
    if not passages:
        lines.append("No results.")
        return "\n".join(lines)
    shown = passages[:results]
    for i, p in enumerate(shown, 1):
        where = p.title + (f" › {p.section}" if p.section else "")
        lines.append(f"\n[{i}] {where}\n{trim(p.text, chars)}")
    articles = dict.fromkeys((p.path, p.title) for p in shown)
    links = ", ".join(f"[{title}]({page_url(link_base, path)})" for path, title in articles)
    lines.append(f"\nEnd your answer with this line:\nSources: {links}")
    return "\n".join(lines)
