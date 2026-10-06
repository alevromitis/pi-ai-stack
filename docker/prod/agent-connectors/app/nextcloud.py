"""Nextcloud Calendar, Tasks and Notes for LibreChat agents.

Three MCP servers in one process, so a model spec attaches only what it needs:

  /calendar/mcp  events, over CalDAV (VEVENT; recurrences expanded server-side)
  /tasks/mcp     tasks, over CalDAV (VTODO, as the Tasks app stores them)
  /notes/mcp     notes, over the Notes app's REST API v1
  /calendar-add/mcp  events and tasks: list and add only (for feed agents)

Calendar objects are addressed by a short id, the first 8 hex digits of
sha1(UID): a small model copies those far more reliably than a 36-character
UID. Ids are remembered from listings; tasks can always be found again by a
scan, events need a fresh listing after a restart. A full UID works too.
"""

import hashlib
import os
import time
import uuid
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote
from xml.sax.saxutils import escape

import httpx
from icalendar import Calendar, Event, Todo
from mcp.server.mcpserver.exceptions import ToolError

from common import (
    DATE_HELP,
    clip,
    env,
    env_int,
    fmt_day,
    fmt_when,
    log,
    new_server,
    now,
    one_line,
    parse_when,
    pick,
    serve,
    start_of,
    tool,
    TZ,
)

BASE = env("NEXTCLOUD_URL").rstrip("/")
USER = env("NEXTCLOUD_USER")
PASSWORD = env("NEXTCLOUD_APP_PASSWORD")
# Host header to send when NEXTCLOUD_URL is an internal address (AIO's
# apache container): Nextcloud refuses hosts outside trusted_domains.
HOST = os.environ.get("NEXTCLOUD_HOST", "")
DEFAULT_CALENDAR = os.environ.get("NEXTCLOUD_DEFAULT_CALENDAR", "")
DEFAULT_TASK_LIST = os.environ.get("NEXTCLOUD_DEFAULT_TASK_LIST", "")
NOTE_MAX_CHARS = env_int("NEXTCLOUD_NOTE_MAX_CHARS", 8000)
MAX_LISTED = 80

_headers = {"User-Agent": "agent-connectors", "OCS-APIRequest": "true"}
if HOST:
    _headers["Host"] = HOST
http = httpx.AsyncClient(auth=(USER, PASSWORD), timeout=30, headers=_headers)

DAV = "{DAV:}"
CAL = "{urn:ietf:params:xml:ns:caldav}"
NONE_WORDS = {"none", "clear", "-"}
PRIORITIES = {"high": 1, "medium": 5, "low": 9}


def short_id(uid: str) -> str:
    return hashlib.sha1(uid.encode()).hexdigest()[:8]


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


# --- CalDAV ------------------------------------------------------------------


@dataclass(frozen=True)
class Collection:
    href: str
    name: str
    components: frozenset[str]
    writable: bool


@dataclass
class Found:
    href: str
    etag: str
    ical: Calendar
    collection: Collection | None = None


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def calendar_query(
    component: str,
    start: datetime | None = None,
    end: datetime | None = None,
    expand: bool = False,
    match: tuple[str, str] | None = None,
    open_only: bool = False,
) -> str:
    """A calendar-query REPORT body. match=(PROPERTY, text) is a substring,
    case-insensitive match (RFC 4791 text-match); callers recheck exactly."""
    data = "<c:calendar-data/>"
    if expand and start and end:
        data = f'<c:calendar-data><c:expand start="{_stamp(start)}" end="{_stamp(end)}"/></c:calendar-data>'
    filters = ""
    if start and end:
        filters += f'<c:time-range start="{_stamp(start)}" end="{_stamp(end)}"/>'
    if match:
        name, text = match
        filters += f'<c:prop-filter name="{name}"><c:text-match>{escape(text)}</c:text-match></c:prop-filter>'
    if open_only:
        filters += '<c:prop-filter name="COMPLETED"><c:is-not-defined/></c:prop-filter>'
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
        f"<d:prop><d:getetag/>{data}</d:prop>"
        f'<c:filter><c:comp-filter name="VCALENDAR"><c:comp-filter name="{component}">{filters}'
        "</c:comp-filter></c:comp-filter></c:filter></c:calendar-query>"
    )


PROPFIND = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
    "<d:prop><d:displayname/><d:resourcetype/><c:supported-calendar-component-set/>"
    "<d:current-user-privilege-set/></d:prop></d:propfind>"
)


def _ok_props(response: ET.Element) -> ET.Element | None:
    for propstat in response.findall(f"{DAV}propstat"):
        status = propstat.findtext(f"{DAV}status", "")
        if " 200 " in status:
            return propstat.find(f"{DAV}prop")
    return None


class CalDav:
    def __init__(self) -> None:
        self._collections: list[Collection] = []
        self._fetched = 0.0
        # short id -> (UID, object href), from every listing and write.
        self.known: OrderedDict[str, tuple[str, str]] = OrderedDict()

    def url(self, href: str) -> str:
        return str(httpx.URL(BASE).join(href))

    def remember(self, uid: str, href: str) -> str:
        sid = short_id(uid)
        self.known[sid] = (uid, href)
        self.known.move_to_end(sid)
        while len(self.known) > 5000:
            self.known.popitem(last=False)
        return sid

    async def collections(self) -> list[Collection]:
        if self._collections and time.monotonic() - self._fetched < 300:
            return self._collections
        response = await http.request(
            "PROPFIND",
            f"{BASE}/remote.php/dav/calendars/{quote(USER)}/",
            content=PROPFIND,
            headers={"Depth": "1", "Content-Type": "application/xml; charset=utf-8"},
        )
        if response.status_code == 401:
            raise ToolError("Nextcloud rejected the app password (NEXTCLOUD_APP_PASSWORD).")
        response.raise_for_status()
        found = []
        for item in ET.fromstring(response.content).findall(f"{DAV}response"):
            props = _ok_props(item)
            if props is None or props.find(f"{DAV}resourcetype/{CAL}calendar") is None:
                continue
            href = item.findtext(f"{DAV}href", "")
            components = frozenset(
                c.get("name", "") for c in props.findall(f"{CAL}supported-calendar-component-set/{CAL}comp")
            ) or frozenset({"VEVENT", "VTODO"})
            privileges = {p.tag for p in props.iterfind(f"{DAV}current-user-privilege-set/{DAV}privilege/*")}
            writable = bool(privileges & {f"{DAV}write", f"{DAV}all", f"{DAV}write-content"})
            name = props.findtext(f"{DAV}displayname") or href.rstrip("/").rsplit("/", 1)[-1]
            found.append(Collection(href, name, components, writable))
        self._collections, self._fetched = found, time.monotonic()
        return found

    async def of(self, component: str, name: str = "", writable: bool = False) -> list[Collection]:
        candidates = [
            c for c in await self.collections() if component in c.components and (c.writable or not writable)
        ]
        if name:
            what = "calendar" if component == "VEVENT" else "task list"
            chosen = pick(name, [c.name for c in candidates], what)
            candidates = [c for c in candidates if c.name == chosen]
        return candidates

    async def target(self, component: str, name: str, default: str) -> Collection:
        """The writable collection to create in: name, else the configured
        default, else Nextcloud's own default ("personal"), else the first."""
        candidates = await self.of(component, name or default, writable=True)
        if not candidates:
            raise ToolError("No writable calendar or task list found.")
        if not (name or default):
            candidates.sort(key=lambda c: not c.href.rstrip("/").endswith("/personal"))
        return candidates[0]

    async def report(self, collection: Collection, body: str) -> list[Found]:
        response = await http.request(
            "REPORT",
            self.url(collection.href),
            content=body,
            headers={"Depth": "1", "Content-Type": "application/xml; charset=utf-8"},
        )
        response.raise_for_status()
        results = []
        for item in ET.fromstring(response.content).findall(f"{DAV}response"):
            props = _ok_props(item)
            data = props.findtext(f"{CAL}calendar-data") if props is not None else None
            if not data:
                continue
            href = item.findtext(f"{DAV}href", "")
            try:
                ical = Calendar.from_ical(data)
            except ValueError:
                log.warning("unparseable calendar object %s", href)
                continue
            results.append(Found(href, props.findtext(f"{DAV}getetag", ""), ical, collection))
        return results

    async def get(self, href: str) -> Found | None:
        response = await http.get(self.url(href))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return Found(href, response.headers.get("ETag", ""), Calendar.from_ical(response.content))

    async def put(self, href: str, ical: Calendar, etag: str = "") -> None:
        headers = {"Content-Type": "text/calendar; charset=utf-8"}
        headers.update({"If-Match": etag} if etag else {"If-None-Match": "*"})
        response = await http.put(self.url(href), content=ical.to_ical(), headers=headers)
        if response.status_code == 412:
            raise ToolError("It changed in Nextcloud meanwhile; list again and retry.")
        response.raise_for_status()

    async def delete(self, href: str, etag: str) -> None:
        response = await http.delete(self.url(href), headers={"If-Match": etag} if etag else {})
        if response.status_code == 412:
            raise ToolError("It changed in Nextcloud meanwhile; list again and retry.")
        response.raise_for_status()

    async def find(self, ident: str, component: str) -> Found:
        ident = ident.strip().strip("[]")
        noun = "event" if component == "VEVENT" else "task"
        if not ident:
            raise ToolError(f"Give the {noun} id from the listing.")
        if ident.lower() not in self.known and len(ident) <= 8 and component == "VTODO":
            await scan_tasks(show_completed=True)  # repopulates ids after a restart
        found: Found | None = None
        if ident.lower() in self.known:
            _, href = self.known[ident.lower()]
            found = await self.get(href)
        elif len(ident) > 8:  # a full UID
            for collection in await self.of(component):
                matches = await self.report(collection, calendar_query(component, match=("UID", ident)))
                found = next((m for m in matches if _uid_of(m.ical, component) == ident), None)
                if found:
                    self.remember(ident, found.href)
                    break
        if found is None or not list(found.ical.walk(component)):
            hint = "call events to list current ids" if component == "VEVENT" else "call tasks to list them"
            raise ToolError(f"No {noun} with id {ident!r}; {hint}.")
        if found.collection is None:
            parent = found.href.rstrip("/").rsplit("/", 1)[0] + "/"
            found.collection = next((c for c in await self.collections() if c.href == parent), None)
        return found


def _uid_of(ical: Calendar, component: str) -> str:
    first = next(iter(ical.walk(component)), None)
    return str(first.get("UID", "")) if first is not None else ""


def _master(ical: Calendar, component: str):
    items = list(ical.walk(component))
    return next((c for c in items if "RECURRENCE-ID" not in c), items[0])


def _replace(component, name: str, value) -> None:
    component.pop(name, None)
    if value is not None:
        component.add(name, value)


def _touch(component) -> None:
    stamp = utcnow()
    _replace(component, "DTSTAMP", stamp)
    _replace(component, "LAST-MODIFIED", stamp)
    _replace(component, "SEQUENCE", int(component.get("SEQUENCE", 0)) + 1)


def _add_timezones(ical: Calendar, component) -> None:
    """VTIMEZONEs for any TZID the component now uses, covering a few years
    around it (icalendar's default spans 1970-2038: kilobytes per object)."""
    anchor = next((component.decoded(k) for k in ("DTSTART", "DUE") if k in component), now())
    year = anchor.year
    ical.add_missing_timezones(first_date=date(year - 1, 1, 1), last_date=date(year + 3, 1, 1))


def _wrap(component) -> Calendar:
    ical = Calendar()
    ical.add("PRODID", "-//agent-connectors//Nextcloud MCP//EN")
    ical.add("VERSION", "2.0")
    ical.add_component(component)
    _add_timezones(ical, component)
    return ical


def _cleared(value: str) -> bool:
    return value.strip().lower() in NONE_WORDS


dav = CalDav()


# --- Calendar ----------------------------------------------------------------

calendar_server = new_server("nextcloud-calendar")
# Add-only: list and add events and tasks, nothing that changes or removes
# what is there. For agents that read untrusted text (feeds) and file what
# they find: an injected item can at worst add an entry.
add_server = new_server("nextcloud-calendar-add")


def _as_local(value: date | datetime) -> date | datetime:
    return value.astimezone(TZ) if isinstance(value, datetime) else value


def _local_start(value: date | datetime) -> datetime:
    return value if isinstance(value, datetime) else start_of(value)


def _span(event) -> tuple[date | datetime, date | datetime]:
    begin = event.decoded("DTSTART")
    if "DTEND" in event:
        end = event.decoded("DTEND")
    elif "DURATION" in event:
        end = begin + event.decoded("DURATION")
    else:
        end = begin + timedelta(days=1) if not isinstance(begin, datetime) else begin
    return _as_local(begin), _as_local(end)


def _parse_end(end: str, begin: date | datetime) -> date | datetime:
    """DTEND from what the model wrote. All-day ends are inclusive dates."""
    timed = isinstance(begin, datetime)
    text = end.strip()
    if not text:
        return begin + (timedelta(hours=1) if timed else timedelta(days=1))
    if timed and len(text) <= 5 and ":" in text:
        hours, _, minutes = text.partition(":")
        try:
            finish = begin.replace(hour=int(hours), minute=int(minutes))
        except ValueError:
            raise ToolError(f"end: can't read {end!r}. {DATE_HELP}") from None
    else:
        finish = parse_when(text, "end")
        if timed and not isinstance(finish, datetime):
            raise ToolError("end needs a time (HH:MM) because start has one.")
        if not timed:
            if isinstance(finish, datetime):
                raise ToolError("An all-day event ends on a date (YYYY-MM-DD), not a time.")
            finish = finish + timedelta(days=1)
    if finish <= begin:
        raise ToolError("end must be after start.")
    return finish


def _describe(begin: date | datetime, end: date | datetime) -> str:
    if isinstance(begin, datetime):
        same_day = begin.date() == end.date()
        return f"{fmt_when(begin)}–{end.strftime('%H:%M') if same_day else fmt_when(end)}"
    last = end - timedelta(days=1)
    return f"{fmt_day(begin)} (all day)" if last <= begin else f"{fmt_day(begin)} to {fmt_day(last)} (all day)"


def _refuse_if_invited(event, action: str) -> None:
    # Changing or deleting an event with attendees makes Nextcloud email them.
    if "ATTENDEE" in event:
        raise ToolError(f"This event has attendees; {action} it in Nextcloud Calendar so invitations are handled there.")


@tool(calendar_server, add_server, service="Nextcloud")
async def events(start: str = "today", days: int = 7, calendar: str = "", details: bool = False) -> str:
    """List calendar events from start (a date or 'today') for a number of days, with ids. details adds notes."""
    first = parse_when(start, "start")
    first = first.date() if isinstance(first, datetime) else first
    days = max(1, min(days, 62))
    window = (start_of(first), start_of(first + timedelta(days=days)))
    collections = await dav.of("VEVENT", calendar)
    occurrences = []
    for collection in collections:
        for found in await dav.report(collection, calendar_query("VEVENT", *window, expand=True)):
            for event in found.ical.walk("VEVENT"):
                sid = dav.remember(str(event.get("UID", "")), found.href)
                begin, end = _span(event)
                # The server compares all-day (floating) dates as UTC; recheck locally.
                if _local_start(end) <= window[0] or _local_start(begin) >= window[1]:
                    continue
                occurrences.append((begin, end, sid, collection.name, event))
    occurrences.sort(key=lambda o: _local_start(o[0]))
    names = sorted({c.name for c in collections})
    last = first + timedelta(days=days - 1)
    lines = [f"Events {fmt_day(first)} to {fmt_day(last)} (now {fmt_when(now())}; calendars: {', '.join(names)}):"]
    if not occurrences:
        lines.append("None.")
    current_day = None
    for begin, end, sid, cal_name, event in occurrences[:MAX_LISTED]:
        day = begin.date() if isinstance(begin, datetime) else begin
        if day != current_day:
            lines.append(fmt_day(day))
            current_day = day
        if isinstance(begin, datetime):
            span = f"{begin:%H:%M}–{end:%H:%M}" if end.date() == begin.date() else f"{begin:%H:%M} to {fmt_when(end)}"
        else:
            last_day = end - timedelta(days=1)
            span = "all day" if last_day <= begin else f"all day until {fmt_day(last_day)}"
        line = f"- {span} {one_line(str(event.get('SUMMARY', '(no title)')), 120)} [{sid}]"
        if event.get("LOCATION"):
            line += f" @ {one_line(str(event['LOCATION']), 80)}"
        if len(names) > 1:
            line += f" · {cal_name}"
        if "RECURRENCE-ID" in event or "RRULE" in event:
            line += " · repeats"
        lines.append(line)
        if details and event.get("DESCRIPTION"):
            lines.append(f"  {one_line(str(event['DESCRIPTION']), 300)}")
    if len(occurrences) > MAX_LISTED:
        lines.append(f"({len(occurrences) - MAX_LISTED} more not shown; ask for fewer days)")
    return "\n".join(lines)


def _source(link: str) -> str:
    link = link.strip()
    if link and not link.lower().startswith(("http://", "https://")):
        raise ToolError("link must be the item's http(s) URL.")
    return link


async def _already_added(link: str) -> str | None:
    """'Already added' for an event or task (open or done) with this source link."""
    for component in ("VEVENT", "VTODO"):
        for collection in await dav.of(component):
            for found in await dav.report(collection, calendar_query(component, match=("URL", link))):
                for item in found.ical.walk(component):
                    if str(item.get("URL", "")).strip() == link:
                        sid = dav.remember(str(item.get("UID", "")), found.href)
                        noun = "event" if component == "VEVENT" else "task"
                        return f"Already added as {noun} [{sid}] {item.get('SUMMARY', '')} in {collection.name}; nothing created."
    return None


def _with_source(notes: str, link: str) -> str:
    return "\n\n".join(part for part in (notes.strip(), f"Source: {link}" if link else "") if part)


@tool(calendar_server, add_server, kind="write", service="Nextcloud")
async def event_add(
    title: str, start: str, end: str = "", calendar: str = "", location: str = "", notes: str = "", link: str = ""
) -> str:
    """Create an event. A date-only start makes an all-day event. end: HH:MM, a datetime, or a last date; default 1 hour.
    link: the source URL (e.g. the RSS item's); a link already in the calendar or tasks is never added twice."""
    link = _source(link)
    if link and (existing := await _already_added(link)):
        return existing
    begin = parse_when(start, "start")
    finish = _parse_end(end, begin)
    collection = await dav.target("VEVENT", calendar, DEFAULT_CALENDAR)
    uid = str(uuid.uuid4()).upper()
    event = Event()
    stamp = utcnow()
    for name, value in (("UID", uid), ("DTSTAMP", stamp), ("CREATED", stamp), ("LAST-MODIFIED", stamp)):
        event.add(name, value)
    event.add("SUMMARY", title.strip())
    event.add("DTSTART", begin)
    event.add("DTEND", finish)
    if location.strip():
        event.add("LOCATION", location.strip())
    if notes.strip() or link:
        event.add("DESCRIPTION", _with_source(notes, link))
    if link:
        event.add("URL", link)
    href = f"{collection.href}{uid}.ics"
    await dav.put(href, _wrap(event))
    sid = dav.remember(uid, href)
    return f"Created [{sid}] {title.strip()}: {_describe(begin, finish)} in {collection.name}."


@tool(calendar_server, kind="write", service="Nextcloud")
async def event_edit(id: str, title: str = "", start: str = "", end: str = "", location: str = "", notes: str = "") -> str:
    """Change an event by id. Give only what changes; 'none' clears location or notes. Moving start keeps the length."""
    found = await dav.find(id, "VEVENT")
    event = _master(found.ical, "VEVENT")
    if found.collection is not None and not found.collection.writable:
        raise ToolError(f"{found.collection.name} is read-only.")
    _refuse_if_invited(event, "change")
    if start or end:
        if "RRULE" in event or len(list(found.ical.walk("VEVENT"))) > 1:
            raise ToolError("This is a repeating event; change its time in Nextcloud Calendar.")
        old_begin, old_end = _span(event)
        begin = parse_when(start, "start") if start else old_begin
        if end:
            finish = _parse_end(end, begin)
        elif type(begin) is type(old_begin):
            finish = begin + (old_end - old_begin)
        else:
            finish = _parse_end("", begin)
        _replace(event, "DTSTART", begin)
        _replace(event, "DTEND", finish)
        event.pop("DURATION", None)
    if title.strip():
        _replace(event, "SUMMARY", title.strip())
    for field, value in (("LOCATION", location), ("DESCRIPTION", notes)):
        if value.strip():
            _replace(event, field, None if _cleared(value) else value.strip())
    _touch(event)
    _add_timezones(found.ical, event)
    await dav.put(found.href, found.ical, found.etag)
    begin, finish = _span(event)
    series = " (repeating; first occurrence)" if "RRULE" in event else ""
    return f"Updated [{short_id(str(event['UID']))}] {event.get('SUMMARY', '')}: {_describe(begin, finish)}{series}."


@tool(calendar_server, kind="delete", service="Nextcloud")
async def event_delete(id: str) -> str:
    """Delete an event by id. A repeating event is deleted with all its occurrences."""
    found = await dav.find(id, "VEVENT")
    event = _master(found.ical, "VEVENT")
    _refuse_if_invited(event, "delete")
    await dav.delete(found.href, found.etag)
    series = " (the whole series)" if "RRULE" in event else ""
    return f"Deleted {event.get('SUMMARY', '')}{series}. It stays in Nextcloud's calendar trash bin for 30 days."


# --- Tasks -------------------------------------------------------------------

tasks_server = new_server("nextcloud-tasks")


def _is_done(todo) -> bool:
    return "COMPLETED" in todo or str(todo.get("STATUS", "")).upper() in ("COMPLETED", "CANCELLED")


def _parent_uid(todo) -> str:
    related = todo.get("RELATED-TO")
    if related is None:
        return ""
    for value in related if isinstance(related, list) else [related]:
        if str(getattr(value, "params", {}).get("RELTYPE", "PARENT")).upper() == "PARENT":
            return str(value)
    return ""


async def scan_tasks(list_name: str = "", show_completed: bool = False) -> list[tuple[Collection, object]]:
    results = []
    for collection in await dav.of("VTODO", list_name):
        for found in await dav.report(collection, calendar_query("VTODO", open_only=not show_completed)):
            for todo in found.ical.walk("VTODO"):
                dav.remember(str(todo.get("UID", "")), found.href)
                if show_completed or not _is_done(todo):
                    results.append((collection, todo))
    return results


def _due_sort_key(todo) -> tuple:
    due = todo.decoded("DUE") if "DUE" in todo else None
    when = (start_of(due) if not isinstance(due, datetime) else due.astimezone(TZ)) if due else None
    priority = int(todo.get("PRIORITY", 0)) or 10
    return (_is_done(todo), when is None, when or now(), priority, str(todo.get("SUMMARY", "")).lower())


def _task_line(todo, depth: int, details: bool) -> list[str]:
    mark = "[x] " if _is_done(todo) else ""
    line = f"{'  ' * depth}- {mark}{one_line(str(todo.get('SUMMARY', '(no title)')), 120)} [{short_id(str(todo['UID']))}]"
    if "DUE" in todo:
        due = _as_local(todo.decoded("DUE"))
        overdue = not _is_done(todo) and (
            due < now() if isinstance(due, datetime) else due < now().date()
        )
        line += f" · {'OVERDUE ' if overdue else ''}due {fmt_when(due)}"
    priority = int(todo.get("PRIORITY", 0))
    if priority:
        line += " · " + ("high" if priority <= 4 else "medium" if priority == 5 else "low")
    lines = [line]
    if details and todo.get("DESCRIPTION"):
        lines.append(f"{'  ' * (depth + 1)}{one_line(str(todo['DESCRIPTION']), 300)}")
    return lines


@tool(tasks_server, add_server, service="Nextcloud")
async def tasks(list: str = "", show_completed: bool = False, details: bool = False) -> str:
    """List tasks (open ones unless show_completed), soonest due first, with ids. list: a task list name."""
    found = await scan_tasks(list, show_completed)
    lists = [c.name for c in await dav.of("VTODO", list)]
    lines = [f"Now {fmt_when(now())}. Task lists: {', '.join(lists) or 'none'}."]
    if not found:
        lines.append("No open tasks." if not show_completed else "No tasks.")
        return "\n".join(lines)
    shown = 0
    for list_name in lists:
        todos = [t for c, t in found if c.name == list_name]
        if not todos:
            continue
        lines.append(f"{list_name}:")
        uids = {str(t["UID"]) for t in todos}
        children: dict = {}
        for todo in todos:
            parent = _parent_uid(todo)
            children.setdefault(parent if parent in uids else "", []).append(todo)

        def walk(parent_uid: str, depth: int) -> None:
            nonlocal shown
            for todo in sorted(children.get(parent_uid, []), key=_due_sort_key):
                if shown < MAX_LISTED:
                    lines.extend(_task_line(todo, depth, details))
                shown += 1
                walk(str(todo["UID"]), depth + 1)

        walk("", 0)
    if shown > MAX_LISTED:
        lines.append(f"({shown - MAX_LISTED} more not shown; pick a list)")
    return "\n".join(lines)


def _priority(value: str) -> int | None:
    key = value.strip().lower()
    if key in NONE_WORDS:
        return None
    if key not in PRIORITIES:
        raise ToolError("priority must be high, medium, low or none.")
    return PRIORITIES[key]


@tool(tasks_server, add_server, kind="write", service="Nextcloud")
async def task_add(
    title: str, due: str = "", list: str = "", notes: str = "", priority: str = "", parent: str = "", link: str = ""
) -> str:
    """Add a task. due: a date or datetime. priority: high, medium or low. parent: id of a task to nest it under.
    link: the source URL (e.g. the RSS item's); a link already in the calendar or tasks is never added twice."""
    link = _source(link)
    if link and (existing := await _already_added(link)):
        return existing
    todo = Todo()
    uid = str(uuid.uuid4()).upper()
    stamp = utcnow()
    for name, value in (("UID", uid), ("DTSTAMP", stamp), ("CREATED", stamp), ("LAST-MODIFIED", stamp)):
        todo.add(name, value)
    todo.add("SUMMARY", title.strip())
    todo.add("STATUS", "NEEDS-ACTION")
    if due.strip():
        todo.add("DUE", parse_when(due, "due"))
    if notes.strip() or link:
        todo.add("DESCRIPTION", _with_source(notes, link))
    if link:
        todo.add("URL", link)
    if priority.strip() and _priority(priority) is not None:
        todo.add("PRIORITY", _priority(priority))
    if parent.strip():
        parent_found = await dav.find(parent, "VTODO")
        todo.add("RELATED-TO", _uid_of(parent_found.ical, "VTODO"))
        collection = parent_found.collection  # subtasks live in the parent's list
        if collection is None or not collection.writable:
            raise ToolError("The parent task's list is read-only.")
    else:
        collection = await dav.target("VTODO", list, DEFAULT_TASK_LIST)
    href = f"{collection.href}{uid}.ics"
    await dav.put(href, _wrap(todo))
    sid = dav.remember(uid, href)
    due_text = f", due {fmt_when(todo.decoded('DUE'))}" if "DUE" in todo else ""
    return f"Added [{sid}] {title.strip()} to {collection.name}{due_text}."


@tool(tasks_server, kind="write", service="Nextcloud")
async def task_edit(id: str, done: bool | None = None, title: str = "", due: str = "", notes: str = "", priority: str = "") -> str:
    """Change a task by id: done=true completes it, false reopens it. Give only what changes; 'none' clears due, notes or priority."""
    found = await dav.find(id, "VTODO")
    if found.collection is not None and not found.collection.writable:
        raise ToolError(f"{found.collection.name} is read-only.")
    todo = _master(found.ical, "VTODO")
    if done is True:
        _replace(todo, "STATUS", "COMPLETED")
        _replace(todo, "COMPLETED", utcnow())
        _replace(todo, "PERCENT-COMPLETE", 100)
    elif done is False:
        _replace(todo, "STATUS", "NEEDS-ACTION")
        todo.pop("COMPLETED", None)
        _replace(todo, "PERCENT-COMPLETE", 0)
    if title.strip():
        _replace(todo, "SUMMARY", title.strip())
    if due.strip():
        _replace(todo, "DUE", None if _cleared(due) else parse_when(due, "due"))
    if notes.strip():
        _replace(todo, "DESCRIPTION", None if _cleared(notes) else notes.strip())
    if priority.strip():
        _replace(todo, "PRIORITY", _priority(priority))
    _touch(todo)
    _add_timezones(found.ical, todo)
    await dav.put(found.href, found.ical, found.etag)
    state = "done" if _is_done(todo) else "open"
    return f"Updated [{short_id(str(todo['UID']))}] {todo.get('SUMMARY', '')} ({state})."


@tool(tasks_server, kind="delete", service="Nextcloud")
async def task_delete(id: str) -> str:
    """Delete a task by id. To finish a task, use task_edit with done=true instead."""
    found = await dav.find(id, "VTODO")
    todo = _master(found.ical, "VTODO")
    await dav.delete(found.href, found.etag)
    return f"Deleted {todo.get('SUMMARY', '')}. It stays in Nextcloud's calendar trash bin for 30 days."


# --- Notes -------------------------------------------------------------------

notes_server = new_server("nextcloud-notes")
NOTES_API = f"{BASE}/index.php/apps/notes/api/v1/notes"


async def notes_call(method: str, note_id: int | None = None, **kwargs) -> httpx.Response:
    url = NOTES_API if note_id is None else f"{NOTES_API}/{note_id}"
    response = await http.request(method, url, **kwargs)
    if response.status_code == 404 and note_id is not None:
        raise ToolError(f"No note {note_id}; call notes to list them.")
    if response.status_code == 412:
        raise ToolError("The note changed in Nextcloud meanwhile; read it again and retry.")
    if response.status_code == 401:
        raise ToolError("Nextcloud rejected the app password (NEXTCLOUD_APP_PASSWORD).")
    response.raise_for_status()
    return response


def _modified(note: dict) -> str:
    return fmt_day(datetime.fromtimestamp(int(note.get("modified", 0)), TZ).date())


@tool(notes_server, service="Nextcloud")
async def notes(query: str = "", category: str = "") -> str:
    """List notes, newest first, with ids. query: words to find in title or text. category: a category name."""
    params = {} if query.strip() else {"exclude": "content"}
    all_notes = (await notes_call("GET", params=params)).json()
    categories = sorted({n.get("category", "") for n in all_notes if n.get("category")})
    chosen = all_notes
    if category.strip():
        name = pick(category, categories, "category")
        chosen = [n for n in chosen if n.get("category") == name]
    words = query.lower().split()
    if words:
        chosen = [n for n in chosen if all(w in f"{n.get('title', '')}\n{n.get('content', '')}".lower() for w in words)]
    chosen.sort(key=lambda n: (not n.get("favorite"), -int(n.get("modified", 0))))
    lines = [f"{len(chosen)} notes. Categories: {', '.join(categories) or 'none'}."]
    for note in chosen[:MAX_LISTED]:
        star = " ★" if note.get("favorite") else ""
        where = f" · {note['category']}" if note.get("category") else ""
        lines.append(f"[{note['id']}] {one_line(note.get('title', ''), 100)}{star}{where} · {_modified(note)}")
    if len(chosen) > MAX_LISTED:
        lines.append(f"({len(chosen) - MAX_LISTED} more not shown; narrow with query or category)")
    return "\n".join(lines)


@tool(notes_server, service="Nextcloud")
async def note_read(id: int) -> str:
    """Read a note by id."""
    note = (await notes_call("GET", id)).json()
    header = f"Title: {note.get('title', '')}\nCategory: {note.get('category') or 'none'} · modified {_modified(note)}"
    return f"{header}\n\n{clip(note.get('content', ''), NOTE_MAX_CHARS)}"


@tool(notes_server, kind="write", service="Nextcloud")
async def note_add(title: str, content: str, category: str = "") -> str:
    """Create a note (Markdown)."""
    body = {"title": title.strip(), "content": content, "category": category.strip()}
    note = (await notes_call("POST", json=body)).json()
    return f"Created note [{note['id']}] {note.get('title', title)}."


@tool(notes_server, kind="write", service="Nextcloud")
async def note_append(id: int, text: str) -> str:
    """Add text to the end of a note, on a new line. Prefer this to note_edit for additions."""
    response = await notes_call("GET", id)
    content = response.json().get("content", "")
    if content and not content.endswith("\n"):
        content += "\n"
    headers = {"If-Match": response.headers["ETag"]} if "ETag" in response.headers else {}
    note = (await notes_call("PUT", id, json={"content": content + text}, headers=headers)).json()
    return f"Appended to [{id}] {note.get('title', '')}."


@tool(notes_server, kind="write", service="Nextcloud")
async def note_edit(id: int, title: str = "", content: str = "", category: str = "") -> str:
    """Change a note by id. content REPLACES the whole text; give only what changes."""
    body = {k: v for k, v in (("title", title.strip()), ("content", content), ("category", category.strip())) if v}
    if not body:
        raise ToolError("Nothing to change: give title, content or category.")
    note = (await notes_call("PUT", id, json=body)).json()
    return f"Updated note [{id}] {note.get('title', '')}."


@tool(notes_server, kind="delete", service="Nextcloud")
async def note_delete(id: int) -> str:
    """Delete a note by id."""
    await notes_call("DELETE", id)
    return f"Deleted note {id}. It stays in Nextcloud's Files trash bin."


if __name__ == "__main__":
    serve({"calendar": calendar_server, "tasks": tasks_server, "notes": notes_server, "calendar-add": add_server})
