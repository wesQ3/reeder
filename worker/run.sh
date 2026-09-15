#!/usr/bin/env bash
set -euo pipefail

# Configuration
AUDIOCPP_CONFIG="${AUDIOCPP_CONFIG:-/app/server.json}"
AUDIOCPP_HOST="${AUDIOCPP_HOST:-0.0.0.0}"
AUDIOCPP_PORT="${AUDIOCPP_PORT:-8080}"
AUDIOCPP_BIN="${AUDIOCPP_BIN:-/app/audiocpp_server}"
if [ ! -x "$AUDIOCPP_BIN" ] && command -v audiocpp_server >/dev/null 2>&1; then
    AUDIOCPP_BIN="audiocpp_server"
fi
GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
GATEWAY_PORT="${GATEWAY_PORT:-8100}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"

AUDIOCPP_PID=""
GATEWAY_PID=""

cleanup() {
    echo "Shutting down worker processes..."
    if [ -n "$GATEWAY_PID" ] && kill -0 "$GATEWAY_PID" 2>/dev/null; then
        kill -TERM "$GATEWAY_PID" 2>/dev/null || true
    fi
    if [ -n "$AUDIOCPP_PID" ] && kill -0 "$AUDIOCPP_PID" 2>/dev/null; then
        kill -TERM "$AUDIOCPP_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "Worker stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo "================================================="
echo " Starting Reeder audio.cpp Worker"
echo "================================================="

# Start audio.cpp server in background
echo "Starting audio.cpp native server on ${AUDIOCPP_HOST}:${AUDIOCPP_PORT}..."
$AUDIOCPP_BIN \
    --config "$AUDIOCPP_CONFIG" \
    --host "$AUDIOCPP_HOST" \
    --port "$AUDIOCPP_PORT" \
    --ui \
    --ui-management \
    ${AUDIOCPP_EXTRA_ARGS:-} &
AUDIOCPP_PID=$!

# Wait for audio.cpp server to become ready
echo "Waiting up to ${WAIT_TIMEOUT}s for audio.cpp server readiness at http://127.0.0.1:${AUDIOCPP_PORT}..."
start_ts=$(date +%s)
ready=0

while true; do
    if curl -sf "http://127.0.0.1:${AUDIOCPP_PORT}/health" >/dev/null 2>&1 || \
       curl -sf "http://127.0.0.1:${AUDIOCPP_PORT}/v1/models" >/dev/null 2>&1; then
        ready=1
        echo "audio.cpp server is ready!"
        break
    fi

    if ! kill -0 "$AUDIOCPP_PID" 2>/dev/null; then
        echo "ERROR: audio.cpp process exited prematurely."
        exit 1
    fi

    now=$(date +%s)
    if [ $((now - start_ts)) -ge "$WAIT_TIMEOUT" ]; then
        echo "WARNING: Timeout waiting for audio.cpp server. Starting gateway anyway..."
        break
    fi

    sleep 1
done

# Start Reeder compatibility gateway
echo "Starting Reeder compatibility gateway on ${GATEWAY_HOST}:${GATEWAY_PORT}..."
export AUDIOCPP_URL="http://127.0.0.1:${AUDIOCPP_PORT}"
export SERVER_CONFIG_PATH="$AUDIOCPP_CONFIG"

uv run python tts_api.py --host "$GATEWAY_HOST" --port "$GATEWAY_PORT" &
GATEWAY_PID=$!

# Wait for either process to terminate
wait -n "$AUDIOCPP_PID" "$GATEWAY_PID"
