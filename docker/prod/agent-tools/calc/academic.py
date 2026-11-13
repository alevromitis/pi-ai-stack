"""NTUA's official academic calendar, searchable by words and dates.

The data (ntua_calendar.json) is the NTUA Senate's two decisions for one
academic year, undergraduate and postgraduate, which <feed> (SAMPS) links
on its own calendar page: semesters, exam periods, deadlines, thesis
rounds and the days with no classes. Transcribed by hand from the
scanned PDFs; nothing is fetched at run time.

The reply is shaped like mcp-wikipedia's: a "[n] Title" line per decision,
its entries, then a "Sources: [Title](URL)" line, so litert-lm's
citations patch ends the answer with the decision's link.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from dataclasses import dataclass

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ntua_calendar.json")
MAX_LINES = 20
# With no words or dates: what is on now and next.
UPCOMING = 10


class CalendarError(ValueError):
    pass


@dataclass(frozen=True)
class Entry:
    programme: str  # ug, pg or all
    start: datetime.date
    end: datetime.date
    what: str
    tags: frozenset[str]


# Query words -> the tags they ask for. English first (litert-lm hands the
# model English), Greek stems too for a query typed by hand. A word that is
# a key's prefix-stem matches: "exams", "examination", "εξεταστική".
SYNONYMS: dict[str, tuple[str, ...]] = {
    "exam": ("exam",), "test": ("exam",), "εξετ": ("exam",), "εξεταστ": ("exam",),
    "resit": ("resit",), "retake": ("resit",), "repeat": ("resit",), "makeup": ("resit",),
    "επαναληπτ": ("resit",),
    "holiday": ("holiday",), "break": ("holiday",), "vacation": ("holiday",), "closed": ("holiday",),
    "off": ("holiday",), "argia": ("holiday",), "αργ": ("holiday",), "διακοπ": ("holiday",),
    "christmas": ("christmas",), "xmas": ("christmas",), "new year": ("christmas",), "χριστουγ": ("christmas",),
    "easter": ("easter",), "πασχα": ("easter",), "πάσχα": ("easter",),
    "thesis": ("thesis",), "diploma": ("thesis",), "dissertation": ("thesis",), "διπλωματ": ("thesis",),
    "defen": ("defence",), "oral": ("defence",), "present": ("defence",), "submi": ("deadline",),
    "hand in": ("deadline",), "παραδ": ("deadline",),
    "regist": ("registration",), "enrol": ("registration",), "sign up": ("registration",),
    "δηλωσ": ("registration",), "δήλωσ": ("registration",), "εγγραφ": ("registration",),
    "withdraw": ("withdraw",), "drop": ("withdraw",), "παραιτ": ("withdraw",),
    "grade": ("grades",), "mark": ("grades",), "result": ("grades", "results"), "βαθμ": ("grades",),
    "class": ("classes",), "lecture": ("classes",), "teaching": ("classes",), "course": ("classes",),
    "lesson": ("classes",), "μαθημ": ("classes",), "μάθημ": ("classes",),
    "semester": ("semester",), "term": ("semester",), "εξαμην": ("semester",), "εξάμην": ("semester",),
    "start": ("start",), "begin": ("start",), "open": ("start",), "first day": ("start",), "εναρξ": ("start",), "έναρξ": ("start",),
    "end": ("end",), "finish": ("end",), "last day": ("end",), "over": ("end",), "ληξ": ("end",), "λήξ": ("end",),
    "deadline": ("deadline",), "due": ("deadline",), "προθεσμ": ("deadline",),
    "winter": ("winter",), "fall": ("winter",), "autumn": ("winter",), "χειμ": ("winter",),
    "spring": ("spring",), "summer": ("spring",), "εαρ": ("spring",),
    "admission": ("admissions",), "apply": ("admissions",), "application": ("admissions",),
    "field": ("exercises",), "exercise": ("exercises",),
    "polytechnic": ("polytechnic",), "πολυτεχν": ("polytechnic",),
    "clean monday": ("clean",), "καθαρ": ("clean",), "independence": ("independence",),
    "may day": ("labour",), "labour": ("labour",), "labor": ("labour",), "πρωτομαγ": ("labour",),
    "holy spirit": ("spirit",), "whit": ("spirit",), "πνευματ": ("spirit",),
    "three hierarchs": ("hierarchs",), "ιεραρχ": ("hierarchs",),
}
# Asks for the postgraduate decision instead of the undergraduate one.
POSTGRAD = ("postgrad", "graduate", "master", "msc", "m.sc", "μεταπτυχ", "δπμσ", "πμσ")
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "ιανουαρ": 1, "φεβρουαρ": 2, "μαρτ": 3, "απριλ": 4, "μαΐ": 5, "μαι": 5, "ιουν": 6, "ιουλ": 7,
    "αυγουστ": 8, "σεπτεμβρ": 9, "οκτωβρ": 10, "νοεμβρ": 11, "δεκεμβρ": 12,
}
_WORD = re.compile(r"[^\W\d_]+(?:[.'][^\W\d_]+)*", re.UNICODE)


def _load(path: str = DATA) -> tuple[list[Entry], dict, tuple[datetime.date, datetime.date]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    entries = []
    for e in raw["entries"]:
        start = datetime.date.fromisoformat(e["start"])
        end = datetime.date.fromisoformat(e.get("end") or e["start"])
        if end < start or e["p"] not in ("ug", "pg", "all"):
            raise ValueError(f"bad entry: {e}")
        entries.append(Entry(e["p"], start, end, e["what"], frozenset(e.get("tags", ()))))
    entries.sort(key=lambda x: (x.start, x.end))
    lo, hi = (datetime.date.fromisoformat(d) for d in raw["covers"])
    return entries, raw["sources"], (lo, hi)


ENTRIES, SOURCES, COVERS = _load()


def _date(text: str, today: datetime.date) -> datetime.date | None:
    text = text.strip().lower()
    if not text:
        return None
    if text == "today":
        return today
    if text == "tomorrow":
        return today + datetime.timedelta(days=1)
    try:
        return datetime.date.fromisoformat(text[:10])
    except ValueError:
        raise CalendarError(f"dates are YYYY-MM-DD or today, not {text!r}") from None


def _matches(word: str, key: str) -> bool:
    # Short keys whole ("off", not "offer"; "end", not "endless"), longer
    # ones as stems ("exam" -> "examination"), and a typed stem of a key.
    if len(key) <= 4 and key.isascii():
        return word == key or word == key + "s"
    return word.startswith(key) or (len(word) >= 5 and key.startswith(word))


def _concepts(query: str) -> tuple[set[str], bool, set[int]]:
    """The tags the query asks for, whether it asks for the postgraduate
    decision, and the months it names."""
    q = query.lower()
    words = _WORD.findall(q)
    tags: set[str] = set()
    for key, wanted in SYNONYMS.items():
        if (" " in key and key in q) or any(_matches(w, key) for w in words):
            tags.update(wanted)
    postgrad = any(w.startswith(p) for w in words for p in POSTGRAD)
    months: set[int] = set()
    for w in _WORD.findall(query):
        lw = w.lower()
        for k, n in MONTHS.items():
            # "May" the month, not "may" the verb.
            if lw == "may" and k == "may" and w != "May":
                continue
            if lw.startswith(k) and (len(k) > 3 or lw == k):
                months.add(n)
    return tags, postgrad, months


_GR_DAYS = ("Δευτέρα", "Τρίτη", "Τετάρτη", "Πέμπτη", "Παρασκευή", "Σάββατο", "Κυριακή")
_GR_MONTHS = ("Ιανουαρίου", "Φεβρουαρίου", "Μαρτίου", "Απριλίου", "Μαΐου", "Ιουνίου", "Ιουλίου",
              "Αυγούστου", "Σεπτεμβρίου", "Οκτωβρίου", "Νοεμβρίου", "Δεκεμβρίου")


def _greek(d: datetime.date) -> str:
    return f"{_GR_DAYS[d.weekday()]} {d.day} {_GR_MONTHS[d.month - 1]} {d.year}"


def _when(e: Entry) -> str:
    # Greek first, written like the NTUA group's source files: the Pi model
    # mistranslated "Mon 18 January 2027" (Κυριακή, Ιανουαρίου for February)
    # but copies a Greek date as given. English follows for other languages.
    fmt = lambda d: f"{d:%a} {d.day} {d:%B %Y}"
    if e.start == e.end:
        return f"{_greek(e.start)} ({fmt(e.start)})"
    return (f"από {_greek(e.start)} έως {_greek(e.end)} "
            f"({fmt(e.start)} to {fmt(e.end)})")


def _status(e: Entry, today: datetime.date) -> str:
    if e.end < today:
        return " (past)"
    if e.start <= today:
        return " (now)" if e.start != e.end else " (today)"
    return ""


def _month_range(months: set[int]) -> list[tuple[datetime.date, datetime.date]]:
    lo, hi = COVERS
    ranges = []
    for m in months:
        for y in range(lo.year, hi.year + 1):
            first = datetime.date(y, m, 1)
            last = (datetime.date(y + (m == 12), m % 12 + 1, 1) - datetime.timedelta(days=1))
            if last >= lo and first <= hi:
                ranges.append((first, last))
    return ranges


def search(query: str = "", start: str = "", end: str = "", today: datetime.date | None = None) -> str:
    today = today or datetime.date.today()
    query = (query or "").strip()
    d_start, d_end = _date(start, today), _date(end, today)
    if d_start and not d_end:
        d_end = d_start
    if d_end and not d_start:
        d_start = d_end
    if d_start and d_end < d_start:
        d_start, d_end = d_end, d_start

    tags, postgrad, months = _concepts(query)
    programme = "pg" if postgrad else "ug"
    pool = [e for e in ENTRIES if e.programme in (programme, "all")]
    notes: list[str] = []

    ranges: list[tuple[datetime.date, datetime.date]] = []
    if d_start:
        ranges = [(d_start, d_end)]
    elif months:
        ranges = _month_range(months)
    if ranges:
        lo, hi = COVERS
        if all(b < lo or a > hi for a, b in ranges):
            return (f"The calendar here covers {lo:%-d %B %Y} to {hi:%-d %B %Y} only "
                    f"(NTUA academic year {lo.year}-{hi.year}); nothing is known outside it.")
        pool = [e for e in pool if any(e.start <= b and e.end >= a for a, b in ranges)]

    picked = pool
    if tags:
        scored = [(len(tags & e.tags), e) for e in pool]
        best = max((s for s, _ in scored), default=0)
        if best:
            picked = [e for s, e in scored if s == best]
        else:
            notes.append(f"Nothing in the calendar matched {query!r}; "
                         + ("everything in those dates:" if ranges else "what is on now and next:"))
            if not ranges:
                picked = [e for e in pool if e.end >= today][:UPCOMING]
    elif not ranges:
        if query:
            notes.append(f"Nothing in the calendar matched {query!r}; what is on now and next:")
        picked = [e for e in pool if e.end >= today][:UPCOMING]
        if not picked:
            picked = pool[-UPCOMING:]

    if d_start and d_start == d_end:
        day = d_start
        off = [e for e in picked if "holiday" in e.tags]
        teaching = [e for e in picked if "semester" in e.tags or "exam" in e.tags]
        if day.weekday() >= 5:
            notes.append(f"{day:%A} {day.day} {day:%B %Y} is a weekend day.")
        elif off:
            notes.append(f"{day:%A} {day.day} {day:%B %Y}: no classes or exams (see below).")
        elif teaching:
            notes.append(f"{day:%A} {day.day} {day:%B %Y}: a normal teaching or exam day, as far as this calendar says.")
        else:
            notes.append(f"{day:%A} {day.day} {day:%B %Y} falls outside the semesters and exam periods.")

    if not picked:
        return "\n".join(notes + ["No calendar entries in that range."])

    extra = len(picked) - MAX_LINES
    picked = picked[:MAX_LINES]
    groups = {"ug": [], "pg": []}
    for e in picked:
        groups[programme if e.programme == "all" else e.programme].append(e)

    if today > COVERS[1]:
        notes.insert(0, f"Warning: this is the {COVERS[0].year}-{COVERS[1].year} calendar; the current one is not loaded.")
    out = [f"Today is {_greek(today)} ({today:%A} {today.day} {today:%B %Y})."] + notes
    cited = []
    for n, key in enumerate(k for k in ("ug", "pg") if groups[k]):
        src = SOURCES[key]
        out.append(f"[{n + 1}] {src['title']}")
        out.extend(f"- {_when(e)}: {e.what}{_status(e, today)}" for e in groups[key])
        cited.append(f"[{src['title']}]({src['url']})")
    if extra > 0:
        out.append(f"... and {extra} more; narrow it with words or dates.")
    if not postgrad:
        out.append("Undergraduate dates; ask with 'postgraduate' for the master's programmes.")
    out.append("Sources: " + ", ".join(cited))
    return "\n".join(out)
