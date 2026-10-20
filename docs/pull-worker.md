# Pull worker architecture

This document describes the implemented pull-worker path. The design rationale
and longer-term roadmap live in `PULL_WORKER.PLAN.md`.

## Overview

The host owns durable state in SQLite and runs the browser/API service plus an
orchestrator. The GPU rig runs an agent that makes only outbound HTTPS calls to
the host:

```text
submit job -> jobs/tasks in SQLite -> orchestrator prepares tasks
                                         |
                 fetch_extract (host) -> tts (worker) -> finalize_audio (host)
                                         ^
                    agent claims / heartbeats / uploads logs + WAV
```

The worker container also exposes an async local runtime API used by the agent:

- `POST /runtime/tasks`
- `GET /runtime/tasks/{id}`
- `GET /runtime/tasks/{id}/logs?since=N`
- `POST /runtime/tasks/{id}/cancel`
- `GET /runtime/tasks/{id}/artifact`

The legacy `/health` and `/generate` endpoints remain available.

## Host configuration

Add to `config.toml` (or an override):

```toml
[paths]
db = "var/reeder.db"
artifacts = "var/artifacts"
logs = "var/logs"

[jobs]
mode = "pull"                 # push | pull | dual
lease_seconds = 180
heartbeat_seconds = 15
claim_wait_seconds = 30
max_attempts = 2

[jobs.worker_api]
token_env = "REEDER_WORKER_TOKEN"
max_artifact_mb = 4096
```

`push` preserves the old inbox/systemd.path flow. `pull` uses the database and
orchestrator. `dual` uses the database; set `jobs.mirror_inbox = true` only if
you explicitly want a legacy inbox copy.

The token is never stored in `config.toml`. Put it in a root-readable
environment file on the host:

```bash
sudo install -d -m 0750 /etc/reeder
sudo sh -c 'umask 077; printf "REEDER_WORKER_TOKEN=%s\n" "$(openssl rand -hex 32)" > /etc/reeder/worker.env'
sudo chown root:reeder /etc/reeder/worker.env
```

Both `reeder-web.service` and `reeder-orchestrator.service` load
`/etc/reeder/worker.env`.

## Host API

Public job/status routes:

```text
GET  /                       form
GET  /add?url=...            bookmarklet-compatible submission
POST /add                    form submission
GET  /jobs                   job list page
GET  /jobs/{id}              live-ish job detail page
GET  /status                 legacy text status
GET  /healthz                health JSON
GET  /api/v1/status          queue/worker summary
GET  /api/v1/workers         worker inventory
POST /api/v1/jobs            create a job
GET  /api/v1/jobs            list jobs
GET  /api/v1/jobs/{id}       job detail (tasks, artifacts, events)
GET  /api/v1/jobs/{id}/events
GET  /api/v1/jobs/{id}/logs?tail=200
POST /api/v1/jobs/{id}/cancel
```

Authenticated worker routes (`Authorization: Bearer $REEDER_WORKER_TOKEN`):

```text
POST /api/v1/workers/register
POST /api/v1/workers/heartbeat
POST /api/v1/tasks/claim
GET  /api/v1/tasks/{id}
POST /api/v1/tasks/{id}/heartbeat
POST /api/v1/tasks/{id}/logs
POST /api/v1/tasks/{id}/artifacts
PUT  /api/v1/artifacts/{id}?offset=N
POST /api/v1/artifacts/{id}/complete
GET  /api/v1/tasks/{id}/input
POST /api/v1/tasks/{id}/complete
POST /api/v1/tasks/{id}/fail
POST /api/v1/tasks/{id}/release
```

Every task mutation carries `worker_id`, `attempt`, and `lease_token`. A stale
attempt or token returns HTTP 409. Log uploads use per-entry `seq` values and
are idempotent. Artifact upload is resumable from an offset and verified by
size and SHA-256 before completion.

## Worker agent

The agent lives in `worker/agent/` and runs on the rig host, outside the GPU
container. Install the example config as `/etc/reeder-worker/agent.toml` and
put `REEDER_WORKER_TOKEN` in a root-readable environment file used by its
systemd unit.

Key behavior:

- waits for host reachability and registers capabilities/inventory;
- reconciles a durable outbox under `work_dir/outbox/`;
- starts the compose container (optional) and waits for `/health`;
- claims at most `max_tasks_per_wake` tasks;
- downloads `input.json`, starts the local runtime, and forwards progress and
  logs concurrently while the task runs;
- uploads `output.wav` + `metrics.json`, uploads final logs, then completes;
- keeps results locally until the host acknowledges completion;
- stops the container and runs the configured `on_idle` hook (`none`, `exit`,
  `suspend`, or a command).

Run it manually:

```bash
export REEDER_WORKER_TOKEN=...
worker/reeder-worker-agent --config /etc/reeder-worker/agent.toml --once
```

The container needs the shared work volume mounted at `/work`. This is already
present in `worker/docker-compose.yml`:

```yaml
volumes:
  - ${WORK_DIR:-/var/lib/reeder-worker}:/work
environment:
  - WORK_DIR=/work
```

## Testing

Host and worker unit tests (no GPU needed):

```bash
uv run pytest -q tests worker/tests
```

Manual host smoke test:

```bash
REEDER_WORKER_TOKEN=... ./deploy/verify-pull-host.sh http://127.0.0.1:8081
```

## Legacy compatibility

- `done/{job_id}.json` and `www/feed.xml` are still written by
  `finalize_audio`.
- Existing audio URLs under `/audio/` are unchanged.
- `bin/process-job`, the inbox, `reeder.path`, and `reeder.timer` remain as a
  development/fallback path. Do not run them against the same jobs as the
  orchestrator.
