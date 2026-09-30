#!/bin/sh
# Serves every .mp4 in test_video_source/ as its own independent TCP source, each on
# its own pair of ports: <base> for mirage's own capture process, <base+1> for go2rtc's
# live-view restream. Both loop the same source. Two separate server processes per video
# are needed because this ffmpeg build's `-listen` TCP muxer only accepts ONE client
# connection before exiting -- real cameras (RTSP) accept many concurrent clients, so
# this split only exists for local testing; each camera's --go2rtc-source override (see
# HOW_TO_RUN.md) is what tells go2rtc to use the second port instead of duplicating
# mirage's own capture connection.
#
# test.mp4 is always pinned to ports 19500/19501 specifically (not just "first
# alphabetically") so an existing camera config pointing at those ports -- e.g. this
# repo's own test_cam -- never silently breaks just because another video file got
# added or removed. Every OTHER .mp4 found gets the next port pair in alphabetical
# order, starting at 19502/19503.
#
# If no .mp4 files exist at all, falls back to a single synthetic lavfi test pattern on
# 19500/19501 so this still works with no setup.
#
# Usage: ./scripts/run_test_stream.sh
# Stop with Ctrl+C (or, if that doesn't work: pkill -9 -f run_test_stream; pkill -9 -f ffmpeg).

BASE_PORT=19500
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SOURCE_DIR="${SCRIPT_DIR}/../test_video_source"

_serve() {
  port=$1
  video=$2
  if [ -n "$video" ]; then
    while true; do
      ffmpeg -hide_banner -loglevel warning \
        -re -stream_loop -1 -i "$video" \
        -pix_fmt yuv420p -c:v libx264 -preset ultrafast \
        -g 20 -force_key_frames 'expr:gte(t,n_forced*2)' \
        -an -f mpegts "tcp://127.0.0.1:${port}?listen"
    done
  else
    while true; do
      ffmpeg -hide_banner -loglevel warning \
        -re -f lavfi -i "testsrc=size=640x480:rate=10" \
        -pix_fmt yuv420p -c:v libx264 -preset ultrafast \
        -g 20 -force_key_frames 'expr:gte(t,n_forced*2)' \
        -f mpegts "tcp://127.0.0.1:${port}?listen"
    done
  fi
}

# Build the ordered video list: test.mp4 first (pinned to BASE_PORT) if present, then
# every other *.mp4 alphabetically. `ls` (not a glob directly) so a SOURCE_DIR with zero
# .mp4 files doesn't leave an unexpanded literal "*.mp4" in the list.
VIDEOS=""
if [ -f "${SOURCE_DIR}/test.mp4" ]; then
  VIDEOS="${SOURCE_DIR}/test.mp4"
fi
for f in $(ls "${SOURCE_DIR}"/*.mp4 2>/dev/null | sort); do
  case "$f" in
    */test.mp4) continue ;;  # already placed first, above
  esac
  VIDEOS="${VIDEOS} ${f}"
done

PIDS=""
port=$BASE_PORT
if [ -z "$VIDEOS" ]; then
  echo "no .mp4 files found in ${SOURCE_DIR}, falling back to synthetic test pattern on ports ${port} (capture) and $((port + 1)) (go2rtc)"
  _serve "$port" "" &
  PIDS="$PIDS $!"
  _serve $((port + 1)) "" &
  PIDS="$PIDS $!"
else
  for video in $VIDEOS; do
    echo "serving ${video} on ports ${port} (capture) and $((port + 1)) (go2rtc)"
    _serve "$port" "$video" &
    PIDS="$PIDS $!"
    _serve $((port + 1)) "$video" &
    PIDS="$PIDS $!"
    port=$((port + 2))
  done
fi

trap 'kill -9 $PIDS 2>/dev/null' INT TERM

wait
