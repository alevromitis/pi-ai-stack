# agent-connectors

MCP servers that give LibreChat agents the user's own data: FreshRSS,
Nextcloud Calendar, Tasks and Notes, and a mailbox over IMAP (read and
save drafts; see [Mail](#mail)). The main job is **FreshRSS → CalDAV**:
read feed items and file the dated ones (events, deadlines) into the
Nextcloud calendar or tasks, once each (now `pi-calendar`). Lives in
`docker/prod/agent-connectors/`.
Nothing is published on the host; LibreChat reaches them on
`agent-connectors-internal`. **Written 2026-10-07; off 2026-10-13 to
free RAM, back on 2026-11-02** for a calendar agent: LibreChat (v0.8.8+)
pauses every write tool for approval (`endpoints.agents.toolApproval` in
`librechat.yaml`), so nothing changes without a click. See
[First run](#first-run).

| Endpoint | Container | Tools | Talks to |
|---|---|---|---|
| `/freshrss/mcp` | `mcp-freshrss-1` | `rss_feeds`, `rss_items`, `rss_read`, `rss_mark`, `rss_mark_all_read`, `rss_subscribe` | FreshRSS's Google Reader API, `freshrss-1:80` |
| `/freshrss-feeds/mcp` | `mcp-freshrss-1` | `samps_announcements`, `ntua_news`, `uniaction_events` (one per FreshRSS category: `FRESHRSS_SAMPS_CATEGORY`, `FRESHRSS_NTUA_CATEGORY`, `FRESHRSS_UNIACTION_CATEGORY`), `read_feed_item`; read-only, marks nothing read | FreshRSS's Google Reader API |
| `GET /items/<category>` (plain JSON, not MCP) | `mcp-freshrss-1` | none: a category's latest items (id, title, url, feed, published, crawled) for instagram-personal's RSS poster (`feeds.py`); only `FRESHRSS_FEED_CATEGORIES` (`SAMPS,NTUA,UniAction`), read-only, marks nothing read | FreshRSS's Google Reader API |
| `/calendar/mcp` | `mcp-nextcloud-1` | `events`, `event_add`, `event_edit` (`event_delete`) | CalDAV (VEVENT) |
| `/tasks/mcp` | `mcp-nextcloud-1` | `tasks`, `task_add`, `task_edit` (`task_delete`) | CalDAV (VTODO, as the Tasks app stores them) |
| `/notes/mcp` | `mcp-nextcloud-1` | `notes`, `note_read`, `note_add`, `note_append`, `note_edit` (`note_delete`) | Notes app REST API v1 |
| `/calendar-add/mcp` | `mcp-nextcloud-1` | `events`, `event_add`, `tasks`, `task_add` — no edit or delete, whatever the flags | CalDAV |
| `/lookup/mcp` | `mcp-nextcloud-1` | `find`: one search over events (a month back to a year ahead), tasks and notes | CalDAV + Notes API |
| `/mail/mcp` | `mcp-mail-1` (Compose profile `mail`) | `mail_folders`, `mail_list`, `mail_read`, `mail_draft` — no send, move, delete or flag | IMAP over TLS (`mail.ntua.gr:993`) |

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
`app/freshrss.py`, `app/nextcloud.py`, `app/mail.py`.

**stdio mode.** With `MCP_TRANSPORT=stdio` (default `http`), a process
serves on stdin/stdout for a client that spawns it, one server per
process: `nextcloud` (the calendar, tasks and notes tools together;
`calendar-add` adds nothing new) or `freshrss`. `MCP_PORT` and
`MCP_ALLOWED_HOSTS` are unused, and logs stay on stderr. Claude Code on the
workstation runs them this way, outside Docker, with
`NEXTCLOUD_URL`/`FRESHRSS_URL` set to the services' tailnet addresses (so
no `NEXTCLOUD_HOST`), its own Nextcloud app password, and `PYTHONTZPATH`
empty as in the image. Nothing in this stack changes for it.

**Behaviour worth knowing:**

* **Dates** are `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM`, `today`, `tomorrow`,
  `yesterday`, a weekday or `next <weekday>` (the next one, never today),
  `this <weekday>` (today counts), `next week` (its Monday), `+3d`, `+2w`,
  `in 3 days`, or a day and month name (`12 November`, `Thursday Nov 12`,
  «12 Νοεμβρίου»; without a year, the next such date), optionally followed
  by a time (`HH:MM`, `9am`, `5pm`, `at 10:00`), in `TZ`. Greek day words
  («Δευτέρα», «αύριο», «επόμενη Τρίτη») work too: the model passes the
  user's words. The model copies the user's words instead of doing
  date arithmetic. A date-only start makes an all-day event; an all-day end
  date is inclusive.
* **One search: `find`** (`/lookup/mcp`, since 2026-11-06). Words are
  matched by stem against event titles, places and notes, task titles,
  notes and list names, and note titles, text and categories, accents and
  case ignored ("lessons" for "lesson", «οδήγηση» for «οδήγησης»). Entries
  holding all the words come first; with none, those holding some, under a
  note. A repeating event is one line at its next occurrence. The header
  names every calendar, task list and note category, and a miss ends with
  "search again with other words … or the language the entry is likely
  written in": the user's entries are mostly Greek, the model's first query
  often English. `events` also takes `query` (all words, up to 366 days).
  Meaning is not matched: "shopping list" doesn't find a note titled
  "Groceries" (the category list shows it).
* **Real names only.** At startup the `calendar` / `list` parameters of
  `events`, `event_add`, `tasks` and `task_add` get the user's calendars
  and task lists as their only allowed values (JSON Schema `enum`, with
  `""` for the default), so a model picks an existing one instead of
  inventing "Shopping List". A calendar or list made later appears after a
  restart (`docker restart mcp-nextcloud-1`); if Nextcloud is down at
  startup, the parameters stay free text and a warning is logged.
* **Ids.** Event ids are remembered from listings, so after a container
  restart the model must call `events` again; task ids are found again by a
  rescan; a full UID always works. Notes and RSS items use Nextcloud's and
  FreshRSS's own ids.
* **Recurring events** are expanded server-side in listings. `event_edit`
  on a repeating event needs either `occurrence` (a date, words allowed:
  "monday") or `all_occurrences=true`; without one it refuses, so "cancel
  Monday's lecture" can't mark every Monday (Qwen did that in 3 of 4 test
  runs on 2026-11-06). `occurrence` writes a standard per-date override
  (`RECURRENCE-ID` in the series' own TZID, the series' fields and
  reminders copied) and may move that one date; `cancel=true` sets
  `STATUS:CANCELLED` (Nextcloud shows it struck through; listings say
  CANCELLED). A whole series can't be moved here, and a delete removes the
  whole series.
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
* **Which calendar.** The model names one only when the user did. Otherwise
  `NEXTCLOUD_CALENDAR_RULES` picks it, deterministically: `Calendar: term
  term | Calendar: …`, first match wins; terms with a dot are domains tried
  first against the link's host (an RSS item's feed site), then every term
  is a substring of the title, location, notes or link (case and accents
  ignored, `_` is a space). The reply names the calendar and the matching
  term. The rules name the user's own doctors, school and employer, so they
  live only in the encrypted `.env.mcp-nextcloud`.
* **New items** with no rule match go to `NEXTCLOUD_DEFAULT_CALENDAR` /
  `_TASK_LIST`, else to Nextcloud's default `personal` calendar. They
  carry a `TZID` and a `VTIMEZONE` from the pinned `tzdata` wheel
  (`PYTHONTZPATH` is empty in the image, so the host's tz files are never
  used).
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
* Feed items are untrusted text, and the point of a feed agent is to act
  on them. `/calendar-add` exists for that: paired with FreshRSS, an
  injected article can at worst add an entry (still marked with its source
  link), never change, move or delete existing ones, or touch notes. Use it
  for any **unattended** feed agent (n8n's agents API runs UI agents): tick
  the `calendar-add` tools, not `calendar`/`tasks`/`notes`. The interactive
  `pi-calendar` spec (since 2026-11-02; the add-only `pi-feeds` spec went
  on 2026-10-07) pairs FreshRSS with the full calendar, tasks and notes
  tools instead, and relies on LibreChat's approval card: every write shows
  its arguments and waits for a click.
* Logs are one line per call: tool, outcome, milliseconds. Arguments (i.e.
  personal data) are never logged.
* Mail is untrusted text too, and `mcp-mail` is the one connector with an
  internet route (`agent-connectors-mail-egress`, to the IMAP server). It
  can't send: the only write is a draft, which waits for approval and then
  for the user to send it from their own mail app. Keep it out of any spec
  with web reading or a sandbox (`qwen-organiser` has neither).

## Mail

`mcp-mail-1` (`app/mail.py`, same image) serves one mailbox over IMAP at
`/mail/mcp`. Written and tested 2026-11-06 against a throwaway GreenMail
server on rpi (Greek subject, HTML body with a stub plain part, an
attachment, a threaded reply draft); **not yet run against the real
mailbox**, which needs the user's password.

* `mail_list(folder, days, unread_only, query, limit)`: newest first, with
  UIDs. Latin-script words are searched by the server (sender, subject,
  text); others, e.g. Greek, are matched here against sender and subject
  (imaplib can't send non-ASCII search keys), over the last 500 messages.
* `mail_read(id, folder)`: headers, the text body (the HTML one when the
  plain part is a stub), attachments named and sized but never opened. It
  uses `BODY.PEEK`, so a read message stays unread.
* `mail_draft(to, subject, body, cc, reply_to_id)`: an RFC 5322 message
  `APPEND`ed to the Drafts folder (`\Draft`, auto-detected or
  `MAIL_DRAFTS_FOLDER`); a reply gets `In-Reply-To`/`References`, `Re:`
  and a quote. Approval card first (`mail_draft*` in `toolApproval.ask`).
* **Logins:** `AUTHENTICATE PLAIN` when offered (NTUA's Cyrus takes PLAIN,
  not DIGEST-MD5), else `LOGIN`. A rejected login is never retried: every
  later call refuses without connecting until the password is fixed and
  the container restarted, so a wrong password can't get the account
  locked. One IMAP session per call.
* **NTUA:** `mail.ntua.gr:993` answers from outside the VPN, so no
  OpenVPN here (Roundcube needs it only for webmail). Its Cyrus offers only
  DHE with a 1024-bit key: `MAIL_TLS_CIPHERS=DEFAULT@SECLEVEL=1`.

To turn it on (the password is the user's to enter; `.sops` files are
edited on rpi, see the repo README):

1. On rpi: `cp .env.mcp-mail.example .env.mcp-mail`, fill in `MAIL_USER`,
   `MAIL_PASSWORD`, `MAIL_ADDRESS`, then `make encrypt` and commit the
   `.sops` copy.
2. `docker compose --profile mail up -d mcp-mail` (or add
   `COMPOSE_PROFILES=mail` to `.env`), and check `docker logs mcp-mail-1`
   after a first `mail_folders` call.
3. In `librechat.yaml`, uncomment `mail` under `mcpServers`, its
   `allowedAddresses` line and `qwen-organiser`'s `- 'mail'`, then restart
   `librechat-1`.

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

