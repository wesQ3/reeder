# Pull-worker deployment runbook

This runbook deploys the code completed so far in stages. It is written for the
always-on Reeder host first; the GPU-rig agent is stage 3.

Paths assume the Arch package layout (`/usr/lib/reeder`, `/etc/reeder`,
`/var/lib/reeder`). Adjust for a source install if needed.

## Stage 0 — build and sanity checks

On the development machine:

```bash
uv run pytest -q tests worker/tests
```

If `uv` has not installed the new host dependencies yet:

```bash
uv sync
```

Build/deploy the package with your usual `DEPLOY_HOST`:

```bash
DEPLOY_HOST=reeder-host make deploy
```

The package now includes:

- `reeder-web` (FastAPI + uvicorn)
- `reeder-orchestrator`
- `reeder/store.py`, `reeder/web.py`, `reeder/orchestrator.py`
- `reeder-orchestrator.service`
- `var/artifacts` and `var/logs` directories
- worker-agent sources/examples under `/usr/lib/reeder/worker`

## Stage 1 — deploy the new web/API without changing job mode

This stage preserves the old inbox/systemd.path flow (`jobs.mode = "push"`).
It lets you verify the FastAPI UI, the worker API, and the database before any
worker is involved.

On the host:

```bash
# Create the shared worker token once. It is read by systemd, not config.toml.
sudo install -d -m 0750 /etc/reeder
if [ ! -f /etc/reeder/worker.env ]; then
  sudo sh -c 'umask 077; printf "REEDER_WORKER_TOKEN=%s\n" "$(openssl rand -hex 32)" > /etc/reeder/worker.env'
fi
sudo chown root:reeder /etc/reeder/worker.env
sudo chmod 0640 /etc/reeder/worker.env

# Make sure the active config keeps push mode for this stage.
sudoedit /etc/reeder/config.toml
#   [jobs]
#   mode = "push"
#   (or put the same value in /etc/reeder/config.override.toml)

# The package install should have synced the venv already. If this is a source
# install instead, run:
#   cd /var/lib/reeder && sudo -u reeder uv sync

sudo systemctl daemon-reload
sudo systemctl restart reeder-web
sudo systemctl status --no-pager reeder-web
```

### Verify stage 1

```bash
curl -fsS http://127.0.0.1:8081/healthz
curl -fsS http://127.0.0.1:8081/api/v1/status
curl -fsS http://127.0.0.1:8081/status
```

Open the normal public form and submit a URL or text job. In push mode it should
still create an `inbox/*.json` file and the old processor should handle it as
before.

Check worker authentication (replace the token):

```bash
TOKEN=$(sudo sed -n 's/^REEDER_WORKER_TOKEN=//p' /etc/reeder/worker.env)
curl -fsS -X POST http://127.0.0.1:8081/api/v1/workers/register \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"worker_id":"smoke-test","name":"Smoke test","capabilities":[]}'
```

From the development machine you can run the automated smoke test:

```bash
REEDER_WORKER_TOKEN="$TOKEN" ./deploy/verify-pull-host.sh https://your-host
```

## Stage 2 — enable the pull path and host orchestrator

Do this only after stage 1 passes. The old path and the orchestrator must not
process the same jobs.

On the host:

```bash
# Stop the legacy trigger before switching modes.
sudo systemctl disable --now reeder.path reeder.timer 2>/dev/null || true

# If any inbox jobs are still pending (push mode had not drained yet), import
# them into SQLite before switching. The importer moves successful files to
# /var/lib/reeder/var/imported/ so they cannot be processed twice.
sudo /usr/lib/reeder/bin/reeder-import-inbox --dry-run
sudo /usr/lib/reeder/bin/reeder-import-inbox

# Add/merge this in /etc/reeder/config.override.toml (or edit config.toml):
sudo tee /etc/reeder/config.override.toml >/dev/null <<'EOF'
[jobs]
mode = "pull"
lease_seconds = 180
heartbeat_seconds = 15
claim_wait_seconds = 30
max_attempts = 2
retention_days = 90
local_tts_fallback = false

[jobs.worker_api]
token_env = "REEDER_WORKER_TOKEN"
max_artifact_mb = 4096
EOF

sudo systemctl daemon-reload
sudo systemctl restart reeder-web
sudo systemctl enable --now reeder-orchestrator
sudo systemctl status --no-pager reeder-web reeder-orchestrator
```

The orchestrator creates `var/reeder.db` in WAL mode and processes host tasks.
`fetch_extract` runs immediately. `tts` remains queued until a worker claims it.
`finalize_audio`, `done/*.json`, and `www/feed.xml` run on the host after the
worker uploads a WAV.

### Verify stage 2 without a worker

Submit a text job through the JSON API:

```bash
curl -fsS -X POST http://127.0.0.1:8081/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"type":"text","text":"Hello from the pull queue.","title":"Pull smoke test"}'
```

The response contains an `id`. Watch it:

```bash
curl -fsS http://127.0.0.1:8081/api/v1/jobs/<id> | python3 -m json.tool
curl -fsS http://127.0.0.1:8081/api/v1/status | python3 -m json.tool
```

Expected: the job is `queued`/`running`, and it has a `tts` task with status
`queued` waiting for a compatible worker. No audio is published yet.

For a URL job, `fetch_extract` completes first and then a tokenizer-prepared
`tts` task appears. Extraction failures fail the job on the host before the rig
is involved.

### Verify stage 2 with a worker

If a worker agent is already available, start it (stage 3) and submit a job.
Expected sequence:

```text
fetch_extract completed -> tts claimed/running/completed -> finalize_audio completed
job completed -> done/<id>.json + www/audio/<id>.opus + feed.xml updated
```

Inspect live progress and logs at:

```text
/jobs/<id>
/api/v1/jobs/<id>
/api/v1/jobs/<id>/logs?tail=200
```

## Stage 3 — install the rig agent (separate host)

The agent runs on the rig host, not inside the GPU container. It needs only
Python 3.11+ (stdlib), the worker container, and outbound HTTPS to the Reeder
host.

Example layout:

```text
/opt/reeder-worker/                 # checkout or copied worker/ source
/etc/reeder-worker/agent.toml       # from worker/agent.toml.example
/etc/reeder-worker/worker.env       # REEDER_WORKER_TOKEN, root-readable
/etc/reeder-worker/docker-compose.yml
/var/lib/reeder-worker/             # work_dir + outbox
```

Install/config:

```bash
sudo install -d -m 0755 /opt/reeder-worker /etc/reeder-worker
sudo install -d -m 0750 /var/lib/reeder-worker
sudo cp -a worker/agent worker/reeder-worker-agent worker/docker-compose.yml \
  /opt/reeder-worker/
sudo cp worker/agent.toml.example /etc/reeder-worker/agent.toml
sudo cp worker/docker-compose.yml /etc/reeder-worker/docker-compose.yml

sudo sh -c 'umask 077; printf "REEDER_WORKER_TOKEN=%s\n" "<same token as host>" \
  > /etc/reeder-worker/worker.env'
sudo chmod 0600 /etc/reeder-worker/worker.env
```

Set `host.url`, `paths.work_dir`, `paths.compose_file`, and the container
`WORK_DIR` correctly. Run one task manually before enabling a timer:

```bash
sudo env REEDER_WORKER_TOKEN="$(sed -n 's/^REEDER_WORKER_TOKEN=//p' /etc/reeder-worker/worker.env)" \
  /opt/reeder-worker/reeder-worker-agent \
  --config /etc/reeder-worker/agent.toml --once
```

The agent registers, starts the container if configured, claims one compatible
task, uploads `output.wav`/`metrics.json`, completes it, stops the container,
and exits (`on_idle = "none"` in the example). Change `on_idle` to `"suspend"`
only after suspend permissions and wake scheduling are configured; those are
rig-administration tasks documented in `PULL_WORKER.PLAN.md` §15.

Example systemd units are in `deploy/reeder-worker-agent.service` and
`deploy/reeder-worker-agent.timer`:

```bash
sudo install -m 0644 deploy/reeder-worker-agent.service \
  /etc/systemd/system/reeder-worker-agent.service
sudo install -m 0644 deploy/reeder-worker-agent.timer \
  /etc/systemd/system/reeder-worker-agent.timer
sudo systemctl daemon-reload
sudo systemctl enable --now reeder-worker-agent.timer
```

Use a timer (or the rig's wake mechanism) to start it. Do not add
`Restart=always`: the agent is designed to drain and exit/suspend. RTC
`WakeSystem=true` is left as a commented rig-specific step.

## Rollback

To return to the old push path:

```bash
sudo systemctl disable --now reeder-orchestrator
# Set [jobs] mode = "push" in /etc/reeder/config.override.toml
sudo systemctl restart reeder-web
sudo systemctl enable --now reeder.path reeder.timer
```

Pending SQLite jobs remain in `var/reeder.db`; they are not deleted by rolling
back. To process them later, re-enable pull mode. Existing `inbox/*.json` files
are still handled by `process-job` in push mode, but should never be processed
by both paths at once. Pending inbox files can be moved into the pull queue
with `reeder-import-inbox`. Existing `done/*.json` history is left in place and
continues to feed `www/feed.xml`; no import is required.

## Troubleshooting

- `503 Worker API token is not configured`: `/etc/reeder/worker.env` is missing
  or the service did not load it. Check `systemctl show reeder-web -p EnvironmentFile`
  and restart the unit.
- `401 Invalid worker token`: host and rig tokens differ, or the rig sent no
  bearer token.
- Worker gets `409 Conflict`: its lease expired or the task was reassigned. The
  agent cancels the local runtime and keeps/reconciles its outbox.
- Jobs stay queued: check `/api/v1/workers` for capabilities and last-seen
  time, and `journalctl -u reeder-orchestrator -f`.
- Final audio missing but `done/*.json` exists: inspect the `finalize_audio`
  task error and feed-update logs; the task is retryable and idempotent.
