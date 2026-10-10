"""Search a Kiwix Wikipedia ZIM and pick the passages that answer a query.

Three steps, chosen by measurement (2026-10-09, 44 questions, see README
"agent-tools"):

  candidates  the article titled with the first name the query gives,
              then libzim's title suggestions (named things: "Eiffel
              Tower" beats "Eiffel Tower replicas"), then the ZIM's Xapian
              full-text index with every word, topped up with any word
  passages    the top articles split into lead / sections / paragraphs,
              infobox rows kept as one line; references, see-also etc. dropped
  ranking     BM25 over those paragraphs, nudged towards leads and
              higher-ranked articles; the named article's opening paragraph
              first (its first prose paragraph, then the infobox)

Names (2026-10-10) took answers in the output from 35/44 to 38/44 for the
questions verbatim and from 36/43 to 38/43 for gemma4-e2b's own queries.
render() also says when no passage names what the query names: gemma4-e2b
otherwise answers "Tao Lung" from a page about Tan Tao-liang.
"""

from __future__ import annotations

import math
import re
import unicodedata
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
QUOTED = re.compile(r'"([^"]{2,80})"|\u201c([^\u201d]{2,80})\u201d')
# Lowercase words allowed inside a name: "Leonardo da Vinci", "Gulf of Mexico".
CONNECTORS = {"of", "the", "de", "da", "del", "der", "di", "du", "la", "le", "van", "von", "bin", "al", "y", "and"}
# Capitalised but not names: they break a run ("Chancellor of Germany May 2025").
NOT_NAMES = set("""
january february march april may june july august september october november
december monday tuesday wednesday thursday friday saturday sunday
""".split())
# A number next to a name is part of it: "2010 FIFA World Cup", "Apollo 11".
NUMBER = re.compile(r"^\d[\d.,-]*$")
# Titles of pages that list things rather than describe one.
COLLECTION = re.compile(r"^(?:Lists? of|Outline of|Index of|Glossary of)\b|^Period \d+ element$")
# Capitalised only because they open the query ("Describe Paris").
OPENERS = set("""
describe explain define summarise summarize list compare find search show
look get history biography info information
""".split())


def words(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def fold(text: str) -> str:
    """Lowercase, accents and punctuation dropped, space-padded: " tan tao liang "."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return " " + " ".join(re.findall(r"[^\W_]+", text)) + " "


def names(query: str) -> list[str]:
    """What the query names: quoted phrases, and runs of capitalised words
    ("Tao Lung", "Gulf of Mexico") with the numbers next to them ("2010 FIFA
    World Cup", "Apollo 11")."""
    found = [a or b for a, b in QUOTED.findall(query)]
    tokens = [re.sub(r"^\W+|\W+$", "", t) for t in QUOTED.sub(" , ", query).split()]
    if tokens and tokens[0].lower() in OPENERS:
        tokens[0] = ""

    def capital(t: str) -> bool:
        return t[:1].isupper() and t.lower() not in STOP | NOT_NAMES

    run: list[str] = []
    for i, token in enumerate(tokens + [""]):
        after = tokens[i + 1] if i + 1 < len(tokens) else ""
        if capital(token):
            run.append(token)
        elif run and token.lower() in CONNECTORS and after[:1].isupper():
            run.append(token)
        elif NUMBER.match(token) and (run or capital(after)):
            run.append(token)
        else:
            if run:
                # A possessive ends a name: "Tolkien's books", not "Rubik's Cube".
                run[-1] = re.sub(r"(?:'s|\u2019s)$", "", run[-1])
                found.append(" ".join(run))
            run = []
    return [n for n in found if fold(n).strip()]


def forms(name: str) -> list[str]:
    """The name, then without its leading and trailing numbers: "2010 FIFA
    World Cup", "FIFA World Cup"."""
    parts = name.split()
    out = [parts, parts[1:] if NUMBER.match(parts[0]) else parts, parts[:-1] if NUMBER.match(parts[-1]) else parts]
    out.append([t for t in parts if not NUMBER.match(t)])
    return list(dict.fromkeys(" ".join(p) for p in out if p))


def _disambiguation(html: str) -> bool:
    """A page that lists meanings ("Mercury most commonly refers to",
    "Fe may refer to") rather than describing one thing. Only its opening
    is checked: a long article may say "may refer to" further down."""
    head = html[:3000].lower()
    return any(s in head for s in ("may refer to", "may also refer to", "commonly refers to"))


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
        """[(path, title)]: articles titled with a name the query gives, title
        suggestions, then all words, then any word."""
        terms = words(query) or TOKEN.findall(query.lower())
        if not terms:
            return []
        found: dict[str, str] = {}
        # "When was the Eiffel Tower completed": suggestions rank its replicas
        # first. Only the first name: "Epirus region in Greece" needs no Greece.
        named = self._named(query)
        if named:
            found[named] = self.archive.get_entry_by_path(named).title
        self._collect(self.suggester.suggest(" ".join(terms)).getResults(0, 3), found, limit)
        for text in (" ".join(terms), " OR ".join(terms)):
            if len(found) >= limit:
                break
            self._collect(self.searcher.search(Query().set_query(text)).getResults(0, limit), found, limit)
        return list(found.items())

    def passages(self, query: str, articles: int = 3, per_article: int = 60) -> list[Passage]:
        """Paragraphs of the top `articles` candidates, best first (BM25)."""
        out: list[Passage] = []
        named = self._named(query)
        rank = 0
        for path, title in self.candidates(query):
            if rank >= articles:
                break
            html = bytes(self._resolve(path).get_item().content).decode("utf-8", "replace")
            # Portals and disambiguation pages describe nothing themselves:
            # "Portal:Solar System" crowded out the article for "largest
            # planet". Kept when the query names one.
            if path != named and (path.startswith("Portal:") or _disambiguation(html)):
                continue
            paras = [(s.title, p) for s in sections(html) for p in s.paras][:per_article]
            out.extend(Passage(title, path, section, text, rank) for section, text in paras)
            rank += 1
        ranked = bm25(query, out)
        # The article the query names comes first with its opening paragraph
        # (the infobox, if any): "Rembrandt birth date" matches no word of
        # "Born: 15 July 1606". After an infobox, the first prose paragraph
        # goes in front of it: it says what the subject is ("HTTP (Hypertext
        # Transfer Protocol) is ..."), which the infobox of HTTP never does.
        lead = [p for p in out if p.path == named and not p.section]
        opening = lead[1::-1] if lead and lead[0].text.startswith("Infobox: ") else lead[:1]
        for p in reversed(opening):
            ranked.remove(p)
            ranked.insert(0, p)
        return ranked

    def _named(self, query: str) -> str | None:
        """The path of the article titled with the query's first name, if any."""
        for name in forms(names(query)[0]) if names(query) else []:
            path = name.replace(" ", "_")
            # "Who painted the Night Watch": The_Night_Watch, not Night_Watch.
            the = re.search(rf"\bthe\s+{re.escape(name)}", query, re.IGNORECASE)
            for candidate in ([f"The_{path}"] if the else []) + [path]:
                try:
                    return self._first_meaning(self._resolve(candidate).path)
                except KeyError:
                    continue
        return None

    def proper(self, query: str) -> str:
        """The query with the names it writes in lowercase capitalised, so
        they are looked up by name ("population of nigeria" -> "population
        of Nigeria", "how tall is mount fuji" -> "how tall is Mount Fuji").
        Real users, and the pre-search that sends their words, write
        lowercase; the model searched "population of nigeria" three times,
        got "Islam in Nigeria" and "Ido, Nigeria", and gave up. Runs of
        words first, longest first, then single words; see _is_name."""
        tokens = query.split()
        bare = [re.sub(r"^\W+|\W+$", "", t) for t in tokens]
        taken: set[int] = set()
        for n in (4, 3, 2, 1):
            for lo in range(len(bare) - n + 1):
                run = bare[lo : lo + n]
                if taken & set(range(lo, lo + n)) or not all(w and w.islower() for w in run):
                    continue
                if run[0] in STOP | CONNECTORS or run[-1] in STOP | CONNECTORS or NUMBER.match(run[0]):
                    continue
                if any(w in STOP and w not in CONNECTORS for w in run) or len(run[0]) < 3:
                    continue
                # "how tall", "how old": a measure, not a name.
                if lo and bare[lo - 1].lower() == "how":
                    continue
                if self._is_name(" ".join(w if w in CONNECTORS else w[:1].upper() + w[1:] for w in run)):
                    taken |= set(range(lo, lo + n))
        for k in sorted(taken):
            if bare[k] not in CONNECTORS:
                tokens[k] = tokens[k].replace(bare[k], bare[k][:1].upper() + bare[k][1:], 1)
        return " ".join(tokens)

    def _is_name(self, title: str) -> bool:
        """Whether a lowercase phrase is a name: its article's lead writes
        it capitalised after another word ("Federal Republic of Nigeria",
        "The Sydney Opera House is") - or, for a run of words, opens with
        it and is titled with it ("Mount Fuji (...) is") - and never in
        lowercase ("the boiling point", "the name tungsten"). A single word
        also needs a lead that defines something, not a list of meanings
        ("Tall, a degree of height")."""
        try:
            entry = self._resolve(title.replace(" ", "_"))
        except KeyError:
            return False
        html = bytes(entry.get_item().content).decode("utf-8", "replace")
        if _disambiguation(html):
            return False
        parts = sections(html) if html else []
        lead = " ".join([p for p in parts[0].paras if not p.startswith("Infobox:")][:2]) if parts else ""
        if re.search(rf"\b{re.escape(title.lower())}\b", lead):
            return False
        after_word = re.search(rf"(?<=[\w,;] ){re.escape(title)}\b", lead) is not None
        if " " in title:
            return after_word or entry.title == title and lead.startswith(title)
        return after_word and re.search(r"\b(?:is|was|are|were)\b", lead) is not None

    def _first_meaning(self, path: str) -> str:
        """For a page that only lists meanings, the first one it lists: asked
        in Italian, the model searched "Cappella Sistina", got that page
        ("may refer to: The Sistine Chapel in the Vatican Palace ...") and a
        painter of another chapel of that name, and named him."""
        html = bytes(self._resolve(path).get_item().content).decode("utf-8", "replace")
        if not _disambiguation(html):
            return path
        link = LexborHTMLParser(html).css_first(".mw-parser-output li a[href]")
        href = urllib.parse.unquote((link.attributes.get("href") or "") if link else "").split("#")[0]
        try:
            return self._resolve(href).path if href else path
        except KeyError:
            return path


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
        # Leads hold definitions and headline facts; prefer better articles,
        # and an article about one thing over a list or outline that mentions
        # it ("Gold" over "Period 6 element", "List of islands of ...").
        p.score = (
            score + 0.3 / (1 + p.article_rank) + (0.2 if not p.section else 0.0)
            - (0.3 if COLLECTION.match(p.title) else 0.0)
        )
    return sorted(passages, key=lambda p: -p.score)


SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def trim(text: str, limit: int, query: str = "") -> str:
    """The passage within `limit` characters: its start, or the run of
    sentences holding more of the query's words. "...water boil at the top
    of everest" got the Boiling point paragraph cut before its sentence on
    Mount Everest (71 °C), and the model answered with 93.4 °C at 1,905 m."""
    if len(text) <= limit:
        return text
    sentences = SENTENCE.split(text)
    terms = list(dict.fromkeys(words(query)))
    if len(sentences) > 1 and terms:
        folded = [fold(x) for x in sentences]

        def covered(i: int, j: int) -> int:
            window = "".join(folded[i:j])
            return sum(1 + (len(t) >= 5) for t in terms if f" {fold(t).strip()}" in window)

        best, start, end = -1, 0, 0
        for i in range(len(sentences)):
            j, size = i, 0
            while j < len(sentences) and size + len(sentences[j]) + 1 <= limit:
                size += len(sentences[j]) + 1
                j += 1
            if j > i and covered(i, j) > best:
                best, start, end = covered(i, j), i, j
        if start > 0:
            return "… " + " ".join(sentences[start:end]) + (" …" if end < len(sentences) else "")
    cut = text[:limit]
    stop = cut.rfind(". ")
    return (cut[: stop + 1] if stop > limit * 0.6 else cut.rstrip()) + " …"


# Words that mark a query as written in another language: articles,
# pronouns, question words and verbs of the languages most often asked in.
# Left out: words that are English too ("son", "war", "dove", "come") and
# lone name particles ("Los Angeles", "da Vinci", "von der Leyen"), which
# is why it takes two. (Same list as litert-lm's patches/presearch.py.)
FOREIGN = frozenset("""
el los las del una unos unas como cual cuál cuáles donde dónde qué quién quiénes cuándo cuánto cuántos
cuántas más por para pero también sobre entre hasta desde está están fue son y
der die das und ist sind wie wo wer wann warum welche welcher welchem welches nicht ein eine einen
einem eines dem zum zur im vom hoch liegt gibt
le les des une est sont dans pour sur qui quel quelle quels quelles où quand comment pourquoi combien
était avec
il gli della delle degli dei nel nella che quale quali quando perché quanto quanti quante sono
uma umas uns qual quais onde quem porque quantos quantas não são tem
het een niet waar hoeveel waarom
""".split())
ENGLISH = frozenset("""
the of and is in what who whom how where when which why are was were does did do to for with on at by
from a an it this that
""".split())


def english(text: str) -> bool:
    """Whether the text reads as English: mostly Latin letters, and not two
    or more of another language's function words outnumbering English ones."""
    letters = [c for c in text if c.isalpha()]
    if sum(unicodedata.name(c, "").startswith("LATIN") for c in letters) * 2 < len(letters):
        return False
    words = re.findall(r"[^\W\d_]+", text.lower())
    foreign = sum(w in FOREIGN for w in words)
    return foreign < 2 or foreign <= sum(w in ENGLISH for w in words)


def render(query: str, passages: list[Passage], date: str, results: int, chars: int, link_base: str) -> str:
    """Passages, then one ready-made Sources line the model is told to end with.

    Measured 2026-10-10 (gemma4-e2b, 12 questions): a link under each
    passage was cited 0/12; this line plus pi-chat's prompt rule, 11/12.

    When no passage names anything the query names, the passages are left
    out: shown with a warning, gemma4-e2b still answered "Tao Lung" from
    Tan Tao-liang's films. 0 of 103 real questions and model queries hit
    this (2026-10-10).
    """
    lines = [f"Wikipedia (offline copy, {date[:7]}) for: {query}"]
    if not english(query):
        # Searches are in English (user, 2026-10-10). In another language
        # they find only stray matches: "富士山の高さ" got "Japanese
        # conjugation", and the model said it didn't know.
        lines.append(
            "No results: this Wikipedia is in English, and the query isn't. Search again"
            " in English, with the English name of what you are asking about first"
            " (\"Oxygen atomic number\"). Still answer in the user's language."
        )
        return "\n".join(lines)
    if not passages:
        lines.append("No results.")
        return "\n".join(lines)
    shown = passages[:results]
    seen = fold(" ".join(f"{p.title} {p.text}" for p in shown))
    wanted = names(query)
    missing = [n for n in wanted if not any(fold(f) in seen for f in forms(n))]
    if not wanted:
        # Lowercase, names look like any word: "...boil at the top of
        # everest" got "Boiling point" passages, and the model answered with
        # the boiling point at 1,905 m. Longer words no passage holds (as a
        # word or its start: "boil" in "boiling") are named the same way.
        missing = [w for w in dict.fromkeys(words(query)) if len(w) >= 5 and not NUMBER.match(w) and f" {fold(w).strip()}" not in seen]
        missing = [w for w in missing if w not in ("whats", "hows", "wheres", "whos")]
    quoted = " or ".join(f'"{m}"' for m in missing)
    if missing and len(missing) == len(wanted):
        others = "; ".join(dict.fromkeys(p.title for p in shown))
        lines.append(
            f"Nothing about {quoted}: no article found mentions it. The closest are"
            f" about other things ({others}), so they don't answer this. Search again"
            " with other words, or say you don't know."
        )
        return "\n".join(lines)
    if missing:
        lines.append(
            f"Note: no passage below mentions {quoted}. Don't answer about it from"
            " passages about something else."
        )
    # A query that names nothing in capitals ("crocodile vs alligator") gets
    # no check above. When no article shown is titled with any of its words,
    # say so: from "Caiman", the model answered with how caimans differ.
    topic = [w for w in words(query) if len(w) >= 4 and not NUMBER.match(w)]
    titles = fold(" ".join(dict.fromkeys(p.title for p in shown)))
    if not wanted and topic and not any(f" {w.rstrip('s')}" in titles for w in topic):
        others = "; ".join(dict.fromkeys(p.title for p in shown))
        lines.append(
            f"Note: these passages are from articles about other things ({others}), not"
            " about what the query asks. Don't answer from them unless they state it;"
            " search again with the name of what you are asking about, capitalised."
        )
    for i, p in enumerate(shown, 1):
        where = p.title + (f" › {p.section}" if p.section else "")
        lines.append(f"\n[{i}] {where}\n{trim(p.text, chars, query)}")
    if not wanted:
        # The query names nothing, so no article was looked up by name: the
        # model searched "atomic number of oxygen" three times, never got
        # the Oxygen article, and said it didn't know. Hints in tool results
        # are followed; ones in tool descriptions weren't.
        lines.append(
            "\nIf these passages don't answer it, search again with the name of"
            " what you are asking about first, capitalised (\"Oxygen atomic"
            " number\", not \"what is the atomic number of oxygen\")."
        )
    articles = dict.fromkeys((p.path, p.title) for p in shown)
    links = ", ".join(f"[{title}]({page_url(link_base, path)})" for path, title in articles)
    lines.append(f"\nEnd your answer with this line:\nSources: {links}")
    return "\n".join(lines)
