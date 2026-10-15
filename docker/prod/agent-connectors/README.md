# agent-connectors

MCP servers that give LibreChat agents the user's own data: FreshRSS and
Nextcloud Calendar, Tasks and Notes. The main job is **FreshRSS → CalDAV**:
read feed items and file the dated ones (events, deadlines) into the
Nextcloud calendar or tasks, once each (`pi-feeds`). Lives in
`docker/prod/agent-connectors/`.
Nothing is published on the host; LibreChat reaches them on
`agent-connectors-internal`. **Written 2026-10-07; both servers disabled
2026-10-13** (Compose profile `disabled`, unused by LibreChat, to free RAM
on rpi): `docker compose -f docker-compose.agent-connectors.yml --profile
disabled up -d` brings them back. See [First run](#first-run).

| Endpoint | Container | Tools | Talks to |
|---|---|---|---|
| `/freshrss/mcp` | `mcp-freshrss-1` | `rss_feeds`, `rss_items`, `rss_read`, `rss_mark`, `rss_mark_all_read`, `rss_subscribe` | FreshRSS's Google Reader API, `freshrss-1:80` |
| `/calendar/mcp` | `mcp-nextcloud-1` | `events`, `event_add`, `event_edit` (`event_delete`) | CalDAV (VEVENT) |
| `/tasks/mcp` | `mcp-nextcloud-1` | `tasks`, `task_add`, `task_edit` (`task_delete`) | CalDAV (VTODO, as the Tasks app stores them) |
| `/notes/mcp` | `mcp-nextcloud-1` | `notes`, `note_read`, `note_add`, `note_append`, `note_edit` (`note_delete`) | Notes app REST API v1 |
| `/calendar-add/mcp` | `mcp-nextcloud-1` | `events`, `event_add`, `tasks`, `task_add` — no edit or delete, whatever the flags | CalDAV |

Bracketed tools exist only with `ALLOW_DELETE=true`; `READ_ONLY=true` leaves
just the listing and reading tools. Both are per container, in
`.env.mcp-<service>`.

Purpose-built rather than an off-the-shelf Nextcloud MCP server, because
the model is a 2–4B Gemma on the Pi's CPU: every attached tool schema is
prefilled on every call. So each endpoint is ~0.35–0.45k tokens of schema
(few tools, one-line descriptions, pydantic titles stripped), answers are
terse plain text with size caps, and calendar objects get 8-hex-digit ids
(sha1 of the UID) instead of 36-character UIDs. One Python process (MCP SDK
2.3, stateless Streamable HTTP) serves all three Nextcloud endpoints, so a
model spec attaches only the ones its task needs. Code: `app/common.py`,
`app/freshrss.py`, `app/nextcloud.py`.

**Behaviour worth knowing:**

* **Dates** are `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM`, `today`, `tomorrow` or a
  weekday (the next one, never today), optionally followed by `HH:MM`, in
  `TZ`. A date-only start makes an all-day event; an all-day end date is
  inclusive.
* **Ids.** Event ids are remembered from listings, so after a container
  restart the model must call `events` again; task ids are found again by a
  rescan; a full UID always works. Notes and RSS items use Nextcloud's and
  FreshRSS's own ids.
* **Recurring events** are expanded server-side in listings. `event_edit`
  changes a series' title, location or notes but refuses to move it, and a
  delete removes the whole series.
* **Events with attendees** cannot be edited or deleted here: Nextcloud
  would email the attendees.
* **Filed once per link.** `event_add` and `task_add` take `link` (the RSS
  item's URL). It is stored as the entry's `URL` and as "Source: …" in its
  notes, and if any event or task — open or done — already has that `URL`,
  the call answers "Already added" and creates nothing. "Add it if needed"
  is therefore enforced by the server, not left to the model, and a
  scheduled re-run over the same items adds nothing.
* **House style.** New events copy the conventions of the user's own
  calendars (checked 2026-10-07 over two of them): `STATUS:
  CONFIRMED`, a `DISPLAY` reminder 10 minutes before whose text is the title
  (`NEXTCLOUD_EVENT_ALARM`, empty for none), all-day events `TRANSPARENT`,
  and title / location / one-line notes as in those calendars.
* **Title-only feeds.** When an item has (almost) no text in the feed,
  `rss_read` fetches its page (`<article>`/`<main>`) and up to two PDFs it
  links, which is where some announcement feeds keep their dates. Links are
  feed data, so: http(s) only, every hop must resolve to public addresses
  only (no LAN, no Docker networks), PDFs only from the page's own site,
  4 MB / 4 pages / 60 s caps. `FRESHRSS_FETCH_PAGES=false` turns it off.
* **Time window.** `rss_items` takes `hours`, so a scheduled run looks only
  at what arrived since the last one (FreshRSS's `ot`).
* **Writes are conditional** (ETag / `If-Match`): a concurrent change gives
  the model a "list again" error instead of a silent overwrite.
* **New items** go to `NEXTCLOUD_DEFAULT_CALENDAR` / `_TASK_LIST`, else to
  Nextcloud's default `personal` calendar. They carry a `TZID` and a
  `VTIMEZONE` from the pinned `tzdata` wheel (`PYTHONTZPATH` is empty in the
  image, so the host's tz files are never used).
* **Nextcloud is reached inside `nextcloud-aio`** at
  `nextcloud-aio-apache:11000` with `Host: ${NEXTCLOUD_HOST}`: AIO's Caddy
  routes on the Host header, and Nextcloud checks it against
  `trusted_domains`. Verified with `status.php` on 2026-10-07.

**Threat model.** These are the only agent servers that hold credentials
and can change personal data.

* They sit on their own `internal: true` network that only LibreChat
  joins, not on `agent-tools-internal`, so the shell, filesystem and browser
  containers (which handle untrusted input) cannot reach them.
* Each container holds one credential and joins only its service's network.
  FreshRSS gets its API password (not the login password). Nextcloud gets an
  app password, which is full-account (Nextcloud app passwords have no
  scopes); revoke it under *Security → Devices & sessions*.
* Requests must carry the exact `Host` LibreChat sends (`421` otherwise),
  and any `Origin` header is refused (`403`), so no browser can drive them.
* Deletes are off by default. Anything deleted lands in Nextcloud's trash
  bins (30 days for calendar objects).
* Feed items are untrusted text, and the point of `pi-feeds` is to act on
  them. So it pairs FreshRSS with `/calendar-add` only: an injected article
  can at worst add an entry (still marked with its source link), never
  change, move or delete existing ones, or touch notes. Keep it that way in
  any agent built in the UI for this job (n8n's agents API runs UI agents):
  tick the `calendar-add` tools, not `calendar`/`tasks`.
* Logs are one line per call: tool, outcome, milliseconds. Arguments (i.e.
  personal data) are never logged.

## First run

1. In FreshRSS, set an API password for your user (*Settings → Profile →
   API management*; the API is already enabled on `rpi`).
2. In Nextcloud, create an app password named `agent-connectors`
   (*Personal settings → Security → Devices & sessions*).
3. Then, with FreshRSS and Nextcloud AIO running:

   ```bash
   cd docker/prod/agent-connectors
   cp .env.example .env
   cp .env.mcp-freshrss.example .env.mcp-freshrss     # FRESHRSS_API_PASSWORD
   cp .env.mcp-nextcloud.example .env.mcp-nextcloud   # NEXTCLOUD_APP_PASSWORD
   make encrypt && make up    # builds agent-connectors-mcp:local (arm64)
   # after a code change: docker compose -f docker-compose.agent-connectors.yml up -d --build
   ```

4. `make up` in `docker/prod/librechat` so `librechat-1` joins
   `agent-connectors-internal` and reloads `librechat.yaml`.

**Tested so far** (2026-10-07, off the Pi): every tool end to end, through
the TS MCP SDK 1.29 that LibreChat 0.8.7 ships, against Radicale 3.8
(CalDAV, recurrence expansion) and fakes of FreshRSS 1.30's `greader.php`
and the Notes API, written from their source. `librechat.yaml` passes
LibreChat's own `configSchema` on `rpi`. Still unverified against the live
services: how sabre (Nextcloud's CalDAV) expands all-day recurring events
(the listing also filters by local time), and the Notes API's `If-Match`.

