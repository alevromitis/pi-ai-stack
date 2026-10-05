# template-docker

A hardened, reusable Docker Compose template for running an app either
standalone or fronted by Tailscale + Caddy. Every service shares the same
security/logging defaults; you enable extra bits (build, resource limits,
read-only rootfs, internal network) by uncommenting the relevant lines.

This is the scaffold for new stacks in this repo. Stacks run by hand on
`rpi` (Raspberry Pi 5, arm64), so pick images that publish a `linux/arm64`
build.

## Starting a new stack

1. Copy `template-docker/` to `docker/dev/<service>/`. Stacks under
   `docker/dev/` are kept in the repo but not deployed; once the stack runs
   on `rpi`, `git mv` it to `docker/prod/<service>/`.
2. Delete the compose variant you don't use, then rename what's left to this
   repo's naming:
   * `docker-compose.app.yml` → `docker-compose.<service>.yml`, or
     `docker-compose.app-tailscale.yml` → `docker-compose.<service>-tailscale.yml`
   * `.env.app.example` → `.env.<service>.example` (and update `env_file:`)
   * `.env.example` stays `.env.example`; rename its `APP_*` vars to
     `<SERVICE>_*` and the `app/` path segment to `<service>/`
3. Replace the `app` placeholders: project `name:`, `container_name`
   (`<service>-1`, `<service>-tailscale-1`, …), the `<service>-external`
   network, the image, and the Caddyfile upstream.

## Layout

```
.
├── docker-compose.app.yml            # App only, published on APP_PORT
├── docker-compose.app-tailscale.yml  # App + Tailscale sidecar + Caddy reverse proxy
├── Makefile                          # includes the repo-root compose.mk: up/down/pull/config/ps/logs/restart
├── .env.example                      # Shared vars - see "The .env rule" below
├── .env.app.example                  # App-specific vars
├── .env.tailscale.example            # Tailscale sidecar vars
├── caddy-config/
│   └── Caddyfile                     # Reverse proxy → app-1:${APP_CONTAINER_PORT}
├── tailscale-config/
│   └── serve-config.json             # Tailscale Serve configuration
└── init-scripts/                     # Mount point for app/db init scripts (empty by default)
```

## The `.env` rule

This is the single easiest thing to get wrong, so it is worth stating
plainly:

> **`env_file:` sets container environment. It is NOT read for `${...}`
> interpolation.** Compose interpolates `${VAR}` in `ports:`, `volumes:`,
> `image:` and friends from the shell and from one file named *exactly*
> `.env`, auto-loaded from the project directory.

So:

| File | Read for `${...}` in the compose file? | Reaches the container? |
|---|---|---|
| `.env` | **Yes** (auto-loaded) | Yes - it is also listed under `env_file:` |
| `.env.app` | No | Yes |
| `.env.tailscale` | No | Yes |

Practical consequences:

* Host ports, host paths and image tags go in `.env`, because the compose
  file interpolates them.
* **Credentials go in `.env.app` / `.env.tailscale`**, passed straight
  through by `env_file:` and never interpolated. This is deliberate: it
  keeps secrets out of the one file that everything reads.
* A `${VAR}` with a `:-default` that you "set" in `.env.app` will silently
  use the default instead. A `${VAR:?message}` will fail loudly. Prefer
  `:?` for anything whose wrong value would be destructive - a host data
  path, for example.

## Data paths

Persistent state lives on the host, not in named volumes, under one data
root. `.env.example` sets it once and derives every per-service dir from it
(Compose interpolates `${...}` inside `.env` itself):

```bash
DATA_ROOT=/home/<user>/docker/prod          # host data root on rpi
APP_DATA_DIR=${DATA_ROOT}/app/data
APP_TAILSCALE_STATE_DIR=${DATA_ROOT}/app/data-tailscale
```

The compose files reference them as
`${APP_DATA_DIR:?set APP_DATA_DIR in .env}`, so a missing `.env` fails
`docker compose config` instead of silently bind-mounting a wrong (or
empty, root-owned) path. Create the directories on the host before the
first `up`, owned by the UID the container runs as.

Tailscale state is a host directory for the same reason: if it is lost the
sidecar registers a fresh node (`app-1`, `app-2`, …) and the old
`*.ts.net` name stops pointing at it. Only Caddy's `caddy_data` /
`caddy_config` stay named volumes - Serve terminates TLS, so there is
nothing in them worth keeping.

## Compose variants

| Variant | File | Services | Env files |
|---|---|---|---|
| App only | `docker-compose.app.yml` | `app` | `.env`, `.env.app` |
| Tailscale | `docker-compose.app-tailscale.yml` | `app`, `tailscale`, `caddy` | `.env`, `.env.app`, `.env.tailscale` |

```bash
cp .env.example .env
cp .env.app.example .env.app
# cp .env.tailscale.example .env.tailscale   # tailscale variant only; fill in TS_AUTHKEY

docker compose -f docker-compose.app.yml up -d   # or: make up
```

`make` targets come from the repo-root `compose.mk` (the `Makefile` includes
it via `git rev-parse --show-toplevel`, like every stack's) and need exactly
one `docker-compose.*.yml` in the directory: delete the variant you don't use.

> `.env*` files are git-ignored except the `.example` templates — never commit
> real secrets.

## Variant details

### App only (`docker-compose.app.yml`)
`app`, published as `${APP_PORT}:${APP_CONTAINER_PORT}`, on an external
bridge network (`app-external`).

Note the two distinct port variables. `APP_PORT` is where you reach the
service from the host; `APP_CONTAINER_PORT` is where the process actually
binds inside the container. They are equal only by coincidence, and
conflating them is a real failure mode: using `APP_PORT=443` as Caddy's
upstream port for Vaultwarden, which listens on 80, leaves the proxy pointing
at nothing.

### Tailscale (`docker-compose.app-tailscale.yml`)

```
        tailnet
           │  Serve terminates TLS on :443
           ▼
   ┌──────────────────────────────┐
   │ tailscale  (owns the netns)  │───┐ joins app-external
   │   caddy    (network_mode:    │   │
   │             service:tailscale)│  │
   └──────────────────────────────┘   │
           │ 127.0.0.1:${CADDY_PORT}  │
           ▼                          ▼
        caddy ──────────────────► app-1:${APP_CONTAINER_PORT}
```

Three things make this work, and all three are load-bearing:

1. `caddy` uses `network_mode: service:tailscale`, sharing the sidecar's
   network namespace. That is the only reason Serve's
   `http://127.0.0.1:${CADDY_PORT}` reaches Caddy at all.
2. Because `caddy` shares that namespace, it must **not** declare
   `networks:` — the two keys are mutually exclusive in Compose. It
   reaches `app-1` because `tailscale` is on the `app-external` bridge.
3. `CADDY_PORT` is 8080, not 80. Caddy runs as `user: "1000:1000"` with
   `no-new-privileges:true`, which blocks the file-capability elevation
   that binding a port below 1024 would need. If you move Caddy to :80 you
   must add `cap_add: NET_BIND_SERVICE` *and* drop `no-new-privileges`, or
   run it as root. Staying above 1024 is the cheaper answer.

`tailscale` needs `/dev/net/tun` and `NET_ADMIN`/`NET_RAW`, and is
health-checked with `tailscale status`; `caddy` waits on it being healthy.

## Caddyfile syntax

Caddy's environment placeholder is `{$VAR}`, **not** `${VAR}`. `${VAR}` is
not expanded — Caddy parses it as a literal, and the upstream address never
resolves. The shipped Caddyfile uses `{$CADDY_PORT:8080}` and
`{$APP_CONTAINER_PORT:3000}`, both supplied via `env_file: .env`.

## Before this is production-ready

This template ships intentionally minimal — most safety nets are present but
commented out so you opt in deliberately. At minimum, before deploying:

1. **Pin every tag.** `.env.example` defaults `TAG`, `TAILSCALE_TAG` and
   `CADDY_TAG` to floating tags. They are three separate variables on
   purpose: one shared `TAG` cannot pin three different images, which is
   what the template used to do (`app:${TAG}`, `tailscale/tailscale:${TAG}`
   and `caddy:${TAG}` all forced to the same string).
2. **Fill in the `app` healthcheck.** `test: []` in both compose files always
   passes trivially — it gives you no real liveness signal. Uncomment and
   point the `curl -f http://localhost:3000/health` line (or your own check)
   at a real endpoint. If the image has no shell and no curl/wget, leave the
   healthcheck commented out and monitor externally rather than shipping a
   probe that can only fail.
3. **Decide how the image gets built.** Both compose files default to
   `image: app:${TAG:-latest}` with `build:` commented out, i.e. they assume
   an image already exists (e.g. from a CI pipeline). If you want
   `docker compose up --build` to work, uncomment `build:` and add a
   `Dockerfile`.
4. **Set resource limits.** `deploy.resources` is commented out on every
   service — without it a runaway container can exhaust the host.
5. **Add TLS.** The `Caddyfile` serves plain `:${CADDY_PORT}`, which is
   correct when Tailscale Serve terminates TLS in front of it. If Caddy is
   your edge instead, add a real domain and let it handle ACME.
6. **Turn on `read_only: true`** once you've confirmed the app doesn't write
   outside its declared volumes/tmpfs — it's commented out on every service
   for now.
7. **Set `TS_AUTHKEY`** (and rotate it) before using the tailscale variant —
   the example file ships with a placeholder.

## Conventions already applied to every service

- Non-root (`user: "1000:1000"`) and `no-new-privileges:true`, except where
  an image's entrypoint genuinely needs root (document it in a comment when
  you leave `user:` unset — see the migrated services for examples)
- JSON file logging, capped at 10MB × 3 files
- `/etc/localtime` mounted read-only for correct container time (not `/etc/timezone`: Debian 13 dropped it, and a missing source makes Docker create a directory there)
- `tmpfs` for `/tmp`
- Host bind mounts under `DATA_ROOT` for persistent state
  (`APP_DATA_DIR`, `APP_TAILSCALE_STATE_DIR`), `:?`-guarded; named volumes
  only for Caddy's disposable `caddy_data` / `caddy_config`
- `restart: unless-stopped`
- Commented-out `internal` network, ready to uncomment if you add a db/cache
  service that shouldn't be reachable externally

## Customizing

- **Init scripts**: drop scripts into `init-scripts/` and uncomment the
  matching volume mount (`./init-scripts:/docker-entrypoint-initdb.d:ro`) to
  run them on first startup of a service that supports it.
- **Adding a db/cache service**: uncomment the `internal` network block and
  the relevant `depends_on` / `environment` lines already stubbed in as
  comments (`DATABASE_URL`, `REDIS_URL`, `condition: service_healthy`).
- **Resource limits**: uncomment the `deploy.resources` block per service and
  set `cpus`/`memory` to fit your host.

## Suggestions to make this prod-ready

The section above covers fixes local to this repo. Beyond that, a production
deployment typically also needs:

- **CI/CD pipeline**: build, tag, and push the image on merge (GitHub
  Actions/GitLab CI); never build on the prod host. Tag with a commit SHA or
  semver, not `latest`.
- **A `docker compose config` check in CI.** `check-compose.sh` at the repo
  root already runs it over every stack; wiring it into CI keeps that true
  automatically — it is what catches the class of bug this template had (a `network:`/`networks:`
  typo, an unreachable sidecar, an uninterpolated variable).
- **Image scanning**: run `docker scout`, `trivy`, or `grype` against the
  built image in CI and fail on high/critical CVEs before it ships.
- **Secrets management**: move beyond plain `.env` files on the host — use
  Docker/Swarm secrets, Vault, SOPS, or your cloud provider's secret manager,
  especially for `TS_AUTHKEY` and any DB/API credentials.
- **Centralized logging**: `json-file` with size caps prevents disk from
  filling up, but logs still only live on the host. Ship them to somewhere
  queryable (Loki, ELK, CloudWatch) if you need to debug after a container is
  gone.
- **Monitoring & alerting**: container-level metrics (cAdvisor/Prometheus)
  and an uptime check on the health endpoint, with alerts wired to
  Slack/PagerDuty/email rather than someone noticing it's down.
- **Backups**: nothing backs up `DATA_ROOT`. Add a scheduled backup of the
  service's dirs (e.g. `restic`, `borgmatic`) if they hold anything you
  can't regenerate.
- **Automated rollout/rollback**: `docker compose up -d` with a pinned tag
  gives you repeatability, but no automatic rollback on a failed healthcheck.
  Consider a deploy script or orchestrator (Swarm, Nomad, k8s) if zero-downtime
  rollback matters.
- **Keep the host and the repo in sync**: stacks are deployed by hand, so
  nothing stops `rpi` drifting from what is committed. After changing a
  live stack, commit the compose/env-example change in the same go, and
  keep each git-ignored `.env` on the host matching its `.env.example`.
- **Dependency updates**: with `TAG`, `TAILSCALE_TAG` and `CADDY_TAG`
  separated you can now pin each independently — do that, and set up
  Renovate/Dependabot so base image and dependency bumps are proposed
  automatically instead of drifting unnoticed.
