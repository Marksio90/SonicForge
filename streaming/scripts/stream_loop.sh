#!/usr/bin/env bash
# SonicForge 24/7 Streaming Loop
# Auto-restarts FFmpeg on crash with health monitoring

set -euo pipefail

STREAM_KEY="${YOUTUBE_STREAM_KEY:-}"
RTMP_URL="${YOUTUBE_RTMP_URL:-rtmps://a.rtmp.youtube.com/live2}"
QUEUE_FILE="${QUEUE_FILE:-/tmp/sonicforge_queue.txt}"
RESOLUTION="${STREAM_RESOLUTION:-1920x1080}"
VIDEO_BITRATE="${STREAM_VIDEO_BITRATE:-4500k}"
AUDIO_BITRATE="${STREAM_AUDIO_BITRATE:-192k}"
FPS="${STREAM_FPS:-30}"
HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://localhost:8000/health}"
CROSSFADE="${CROSSFADE_DURATION:-12}"
MAX_RESTARTS=100
RESTART_DELAY=5

restart_count=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SonicForge Stream] $1"
}

check_dependencies() {
    for cmd in ffmpeg curl; do
        if ! command -v "$cmd" &> /dev/null; then
            log "ERROR: $cmd is not installed"
            exit 1
        fi
    done
}

build_ffmpeg_cmd() {
    local output_url="${RTMP_URL}/${STREAM_KEY}"

    echo ffmpeg \
        -re \
        -f concat \
        -safe 0 \
        -i "$QUEUE_FILE" \
        -c:a aac \
        -b:a "$AUDIO_BITRATE" \
        -ar 44100 \
        -f lavfi \
        -i "color=c=black:s=${RESOLUTION}:r=${FPS}" \
        -c:v libx264 \
        -preset veryfast \
        -b:v "$VIDEO_BITRATE" \
        -maxrate "$VIDEO_BITRATE" \
        -bufsize 9000k \
        -pix_fmt yuv420p \
        -g $((FPS * 2)) \
        -f flv \
        "$output_url"
}

stream() {
    log "Starting FFmpeg stream (attempt $((restart_count + 1)))"
    local cmd
    cmd=$(build_ffmpeg_cmd)

    log "Command: $cmd"

    eval "$cmd" 2>&1 | while IFS= read -r line; do
        echo "[FFmpeg] $line"
    done

    return ${PIPESTATUS[0]}
}

main() {
    check_dependencies

    if [ -z "$STREAM_KEY" ]; then
        log "ERROR: YOUTUBE_STREAM_KEY not set"
        exit 1
    fi

    log "SonicForge Streaming Engine starting..."
    log "Resolution: $RESOLUTION, Video: $VIDEO_BITRATE, Audio: $AUDIO_BITRATE"
    log "Target: $RTMP_URL"

    while [ $restart_count -lt $MAX_RESTARTS ]; do
        stream || true

        restart_count=$((restart_count + 1))
        log "Stream ended. Restart count: $restart_count/$MAX_RESTARTS"
        log "Restarting in ${RESTART_DELAY}s..."
        sleep $RESTART_DELAY

        # Notify backend about restart
        curl -s -X POST "$HEALTH_CHECK_URL" \
            -H "Content-Type: application/json" \
            -d '{"event": "stream_restart", "count": '"$restart_count"'}' \
            || true
    done

    log "Max restarts reached. Exiting."
    exit 1
}

main "$@"
