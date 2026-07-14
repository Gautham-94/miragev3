#!/bin/sh
# Serves TWO independent copies of a video stream over TCP: one on port 19500 (for
# mirage's own capture process) and one on port 19501 (for go2rtc's live-view restream).
# Both loop the same source. Two separate server processes are needed because this
# ffmpeg build's `-listen` TCP muxer only accepts ONE client connection before exiting
# -- real cameras (RTSP) accept many concurrent clients, so this split only exists for
# local testing; config/mirage.yaml's go2rtc_stream_override in __main__.py's --go2rtc-source
# flag is what tells go2rtc to use the second port instead of duplicating mirage's own
# capture connection.
#
# If test_video_source/test.mp4 exists, it is looped as the source (so you can drop in
# real footage with real objects for the detector/tracker to actually exercise -- the
# synthetic test pattern below has no recognizable COCO-class objects in it). Otherwise
# falls back to a synthetic lavfi test pattern so this still works with no setup.
#
# Usage: ./scripts/run_test_stream.sh
# Stop with Ctrl+C (or, if that doesn't work: pkill -9 -f run_test_stream; pkill -9 -f 19500; pkill -9 -f 19501).

PORT_CAPTURE=19500
PORT_GO2RTC=19501
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SOURCE_VIDEO="${SCRIPT_DIR}/../test_video_source/test.mp4"

_serve() {
  port=$1
  if [ -f "$SOURCE_VIDEO" ]; then
    while true; do
      ffmpeg -hide_banner -loglevel warning \
        -re -stream_loop -1 -i "$SOURCE_VIDEO" \
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

if [ -f "$SOURCE_VIDEO" ]; then
  echo "serving real video source on ports ${PORT_CAPTURE} (capture) and ${PORT_GO2RTC} (go2rtc): ${SOURCE_VIDEO}"
else
  echo "no ${SOURCE_VIDEO} found, falling back to synthetic test pattern on ports ${PORT_CAPTURE} (capture) and ${PORT_GO2RTC} (go2rtc)"
fi

_serve "$PORT_CAPTURE" &
CAPTURE_PID=$!
_serve "$PORT_GO2RTC" &
GO2RTC_PID=$!

trap 'kill -9 "$CAPTURE_PID" "$GO2RTC_PID" 2>/dev/null' INT TERM

wait
