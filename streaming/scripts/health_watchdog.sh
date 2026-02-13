#!/usr/bin/env bash
# SonicForge Stream Health Watchdog
# Monitors the YouTube stream and triggers failover if needed

set -euo pipefail

CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
API_URL="${API_URL:-http://localhost:8000}"
YOUTUBE_API_KEY="${YOUTUBE_API_KEY:-}"
YOUTUBE_CHANNEL_ID="${YOUTUBE_CHANNEL_ID:-}"
GYRE_ENABLED="${GYRE_ENABLED:-false}"
MAX_OFFLINE_SECONDS=60

offline_since=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Watchdog] $1"
}

check_ffmpeg() {
    if pgrep -x "ffmpeg" > /dev/null 2>&1; then
        echo "running"
    else
        echo "stopped"
    fi
}

check_youtube_stream() {
    if [ -z "$YOUTUBE_API_KEY" ] || [ -z "$YOUTUBE_CHANNEL_ID" ]; then
        echo "unknown"
        return
    fi

    local response
    response=$(curl -s "https://www.googleapis.com/youtube/v3/search?part=snippet&channelId=${YOUTUBE_CHANNEL_ID}&type=video&eventType=live&key=${YOUTUBE_API_KEY}" 2>/dev/null)

    if echo "$response" | grep -q '"totalResults": 0'; then
        echo "offline"
    else
        echo "live"
    fi
}

notify_backend() {
    local status="$1"
    curl -s -X POST "${API_URL}/api/v1/stream/health" \
        -H "Content-Type: application/json" \
        -d "{\"status\": \"$status\", \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
        || true
}

trigger_failover() {
    log "FAILOVER: Attempting to restart stream..."
    curl -s -X POST "${API_URL}/api/v1/stream/control" \
        -H "Content-Type: application/json" \
        -d '{"action": "restart"}' \
        || true

    if [ "$GYRE_ENABLED" = "true" ]; then
        log "FAILOVER: Gyre backup would be activated here"
    fi
}

main() {
    log "Watchdog started. Check interval: ${CHECK_INTERVAL}s"

    while true; do
        ffmpeg_status=$(check_ffmpeg)
        yt_status=$(check_youtube_stream)

        log "FFmpeg: $ffmpeg_status | YouTube: $yt_status"

        if [ "$ffmpeg_status" = "stopped" ] || [ "$yt_status" = "offline" ]; then
            if [ $offline_since -eq 0 ]; then
                offline_since=$(date +%s)
            fi

            elapsed=$(( $(date +%s) - offline_since ))
            log "Stream appears offline for ${elapsed}s (threshold: ${MAX_OFFLINE_SECONDS}s)"

            if [ $elapsed -ge $MAX_OFFLINE_SECONDS ]; then
                log "ALERT: Stream offline for > ${MAX_OFFLINE_SECONDS}s!"
                trigger_failover
                offline_since=0
            fi
        else
            offline_since=0
        fi

        notify_backend "$ffmpeg_status"
        sleep "$CHECK_INTERVAL"
    done
}

main "$@"
