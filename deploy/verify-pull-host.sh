#!/usr/bin/env bash
# Smoke-test a deployed Reeder pull-worker host.
# Usage: [REEDER_WORKER_TOKEN=...] ./deploy/verify-pull-host.sh https://host
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8081}"
TOKEN="${REEDER_WORKER_TOKEN:-}"
SMOKE_JOB="${SMOKE_JOB:-1}"

say() { printf '\n== %s ==\n' "$*"; }

say "healthz"
curl -fsS "$BASE_URL/healthz"
printf '\n'

say "api status"
curl -fsS "$BASE_URL/api/v1/status"
printf '\n'

say "legacy status"
curl -fsS "$BASE_URL/status"
printf '\n'

if [ -n "$TOKEN" ]; then
  say "worker auth + registration"
  curl -fsS -X POST "$BASE_URL/api/v1/workers/register" \
    -H "Authorization: Bearer $TOKEN" \
    -H 'Content-Type: application/json' \
    -d '{"worker_id":"deploy-smoke-test","name":"Deployment smoke test","capabilities":[]}'
  printf '\n'

  say "claim with no compatible task (expect 204 or a task)"
  curl -sS -o /tmp/reeder-claim.out -w 'HTTP %{http_code}\n' \
    -X POST "$BASE_URL/api/v1/tasks/claim" \
    -H "Authorization: Bearer $TOKEN" \
    -H 'Content-Type: application/json' \
    -d '{"worker_id":"deploy-smoke-test","capabilities":[],"wait_seconds":0}'
  cat /tmp/reeder-claim.out 2>/dev/null || true
  printf '\n'
else
  say "worker auth skipped (REEDER_WORKER_TOKEN not set)"
fi

if [ "$SMOKE_JOB" = "1" ]; then
  say "submit a text smoke job"
  response=$(curl -fsS -X POST "$BASE_URL/api/v1/jobs" \
    -H 'Content-Type: application/json' \
    -d "{\"type\":\"text\",\"text\":\"Deployment smoke test $(date +%s).\",\"title\":\"Deployment smoke test\"}")
  printf '%s\n' "$response"
  job_id=$(printf '%s' "$response" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
  printf 'job_id=%s\n' "$job_id"

  say "job after 5s"
  sleep 5
  curl -fsS "$BASE_URL/api/v1/jobs/$job_id" | python3 -m json.tool
fi

say "done"
