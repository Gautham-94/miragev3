we have a # RTSP → WebRTC Multi-Camera Live Streaming Architecture

## 1. Objective

Design a production-grade live-view system that can ingest RTSP streams from approximately 10–12 cameras and display them simultaneously in a custom Angular dashboard with:

- Very low latency
- Minimal stutter
- Fast recovery from network/camera failures
- Low CPU/GPU usage
- No unnecessary transcoding
- Independent failure/reconnection per camera
- Ability to scale beyond the initial 12-camera deployment

The core engineering principle is:

> Keep compressed video compressed for as long as possible. Decode and encode only when necessary.

---

# 2. Recommended High-Level Architecture

```text
             ┌──────────── Camera 1 (RTSP)
             │
             ├──────────── Camera 2 (RTSP)
             │
             ├──────────── Camera 3 (RTSP)
             │
             │       ...
             │
             └──────────── Camera 12 (RTSP)
                            │
                            ▼
                  ┌───────────────────┐
                  │    Media Gateway  │
                  │                   │
                  │ RTSP ingestion    │
                  │ Stream management │
                  │ WebRTC output     │
                  └─────────┬─────────┘
                            │
                        WebRTC
                            │
              ┌─────────────┴─────────────┐
              │                           │
        Angular Dashboard           Other clients
              │
        ┌─────┴─────┐
        │ WebRTC    │
        │ video     │
        │ elements  │
        └───────────┘
```

The preferred architecture is:

**RTSP cameras → Media Gateway → WebRTC → Browser**

For the media gateway, **MediaMTX** is a strong choice.

---

# 3. Why WebRTC Instead of HLS?

For a CCTV/wildlife dashboard where live latency matters, WebRTC is preferable to traditional HLS.

### HLS

```text
RTSP
 ↓
FFmpeg
 ↓
HLS segments
 ↓
HTTP
 ↓
Browser
```

This is simple but normally introduces more latency because the video is segmented.

### WebRTC

```text
RTSP
 ↓
MediaMTX
 ↓
WebRTC
 ↓
Browser
```

This is much better suited to near-real-time camera viewing.

The goal should be to keep latency in the sub-second-to-low-seconds range rather than accepting several seconds of delay.

---

# 4. Media Gateway: MediaMTX

MediaMTX should sit between the cameras and browsers.

It can:

- Ingest RTSP
- Manage camera streams
- Expose streams through WebRTC
- Bridge protocols
- Handle stream lifecycle
- Avoid transcoding when the source codec is already compatible

Conceptually:

```text
Camera
 H.264
   │
   │ RTSP
   ▼
MediaMTX
   │
   │ WebRTC
   ▼
Chrome / Edge
```

The important part is that the H.264 bitstream can remain compressed rather than being decoded and re-encoded unnecessarily.

---

# 5. Avoid Unnecessary Transcoding

This is one of the most important design decisions.

### Preferred

```text
Camera H.264
      ↓
MediaMTX
      ↓
WebRTC H.264
      ↓
Browser hardware decoder
```

### Avoid unless necessary

```text
Camera H.264
      ↓
Decode
      ↓
Raw frames
      ↓
Encode
      ↓
H.264
      ↓
WebRTC
```

The second approach consumes considerably more CPU/GPU resources.

For 12 simultaneous cameras, avoiding unnecessary transcoding is critical.

---

# 6. Camera Configuration

The camera configuration has a major effect on system performance.

A typical main stream could be:

```text
Resolution: 1920 × 1080
FPS:        15
Codec:      H.264
Bitrate:    ~2–4 Mbps
GOP:        15–30 frames
```

The exact values should be tuned according to the cameras, network and desired image quality.

## Use a secondary/substream

Configure a lower-resolution stream such as:

```text
Resolution: 640 × 360
FPS:        10–15
Codec:      H.264
Bitrate:    ~300–700 Kbps
```

This enables an important optimization:

### Grid view

Use the substream:

```text
┌──────┬──────┬──────┐
│ Cam1 │ Cam2 │ Cam3 │
├──────┼──────┼──────┤
│ Cam4 │ Cam5 │ Cam6 │
├──────┼──────┼──────┤
│ Cam7 │ Cam8 │ Cam9 │
└──────┴──────┴──────┘
```

### Selected camera

When the user opens one camera:

```text
Cam7 → switch to 1080p main stream
```

while the other cameras remain on their lower-resolution streams.

This avoids wasting bandwidth and decoding resources on high-resolution video that is being displayed as a small tile.

---

# 7. Why the Substream/Main-Stream Strategy Matters

Suppose the substream is approximately 400 Kbps.

For 12 cameras:

```text
12 × 400 Kbps
≈ 4.8 Mbps
```

This is much easier to handle than continuously sending twelve 1080p streams.

When a user selects one camera:

```text
Cam7 → 1080p
Cam1–6, Cam8–12 → substream
```

This provides a much better resource/performance balance.

---

# 8. Angular Frontend

The Angular application should not perform video decoding itself.

Use the browser's native WebRTC stack:

```text
Angular
 │
 ├── CameraService
 │
 ├── WebRTCService
 │
 └── <video>
```

The browser handles:

- Video decoding
- Jitter buffering
- Packet loss handling
- Synchronization
- Hardware acceleration where available

The Angular application should primarily manage:

- Camera state
- WebRTC connections
- UI
- Reconnection
- Camera selection
- Main/substream switching
- Detection overlays
- Camera metadata

---

# 9. Avoid Canvas-Based Video Processing

Avoid this architecture for normal viewing:

```text
WebRTC
 ↓
Canvas
 ↓
JavaScript processing
 ↓
Display
```

Instead:

```html
<video autoplay muted playsinline></video>
```

Use canvas or WebGL only when there is a specific requirement, such as custom overlays or computer-vision visualization.

---

# 10. Spring Boot Should Not Carry the Video

Spring Boot should handle application-level functionality, not raw video transport.

### Spring Boot responsibilities

```text
REST APIs
Authentication
Authorization
Camera configuration
Camera metadata
Events
Detection metadata
Recording metadata
PTZ commands
WebSocket/SSE notifications
```

### MediaMTX responsibilities

```text
RTSP ingestion
Media stream lifecycle
WebRTC delivery
Protocol conversion
```

Avoid:

```text
Camera
 ↓
Spring Boot
 ↓
WebSocket
 ↓
Angular
```

for raw video.

Instead:

```text
Camera
 ↓ RTSP
MediaMTX
 ↓ WebRTC
Angular
```

while:

```text
Angular
 ↕
Spring Boot
```

handles application state.

---

# 11. AI Pipeline Should Be Independent of Live Viewing

For a wildlife monitoring system, the live video path should not depend on AI inference performance.

Recommended architecture:

```text
                     ┌───────────────┐
                     │     RTSP      │
                     │    Camera     │
                     └───────┬───────┘
                             │
                      ┌──────┴───────┐
                      │              │
                      ▼              ▼
                 MediaMTX       AI Pipeline
                      │              │
                      │              ├── MegaDetector
                      │              ├── YOLO
                      │              └── Tracker
                      │
                      ▼              │
                   WebRTC            │
                      │              │
                      ▼              ▼
                  Angular       Detection Events
                                     │
                                     ▼
                                Spring Boot
                                     │
                                     ▼
                                  Angular
```

If inference becomes slow or temporarily fails, the live video should continue normally.

This separation is especially important for Mirage.

---

# 12. Detection Overlays

Do not create another encoded video stream just to show bounding boxes.

Instead:

```text
WebRTC video
      │
      ▼
   <video>
      │
      └──── Detection overlay
                    ▲
                    │
             Detection metadata
```

For example:

```json
{
  "camera": "cam-07",
  "timestamp": 1759200000,
  "detections": [
    {
      "class": "animal",
      "confidence": 0.91,
      "bbox": [0.32, 0.21, 0.52, 0.63]
    }
  ]
}
```

The Angular application can render these boxes over the video.

This is much cheaper than re-encoding video frames.

---

# 13. RTX 3050 and Hardware Acceleration

If transcoding is unavoidable, use the RTX 3050's hardware video capabilities rather than doing everything on the CPU.

A typical hardware pipeline is:

```text
Input
 ↓
NVDEC
 ↓
GPU
 ↓
NVENC
 ↓
H.264
 ↓
WebRTC
```

FFmpeg can use NVIDIA hardware acceleration.

Conceptually:

```bash
ffmpeg \
  -hwaccel cuda \
  -c:v hevc_cuvid \
  ... \
  -c:v h264_nvenc \
  ...
```

The exact parameters depend on the input codec and output requirements.

The important principle is:

> Do not make the CPU perform twelve simultaneous decode/encode pipelines if the GPU's hardware video engines can handle the workload.

---

# 14. Prefer H.264 for Browser Live View

For WebRTC browser compatibility, H.264 is generally the safer choice.

If cameras support:

```text
H.264
H.265
```

use H.264 for the live WebRTC path when practical.

H.265 can be useful for recording/storage efficiency but may complicate browser/WebRTC compatibility.

---

# 15. Network Architecture

The physical network matters as much as the software.

Recommended:

```text
Cameras
   │
   │ Gigabit Ethernet
   ▼
PoE Switch
   │
   ├── NVR
   │
   └── Media Server
```

Prefer wired Ethernet between:

```text
Camera
 ↓
Switch
 ↓
Media Server
```

rather than putting the camera-to-server path entirely over Wi-Fi.

For 12 cameras, calculate aggregate bitrate.

For example:

```text
12 × 4 Mbps
≈ 48 Mbps
```

before accounting for protocol overhead, other traffic and additional clients.

A Gigabit Ethernet infrastructure gives substantially more headroom.

---

# 16. Persistent Stream Connections

The media gateway should maintain independent connections to the cameras.

```text
Camera 1 ─────┐
Camera 2 ─────┤
Camera 3 ─────┤
Camera 4 ─────┤
...            ├── MediaMTX
Camera 12 ─────┘
```

The browser should not connect directly to the cameras' RTSP endpoints.

Benefits:

- Centralized access control
- Connection management
- Reconnection
- Monitoring
- Protocol conversion
- Easier scaling
- No camera credentials exposed to clients

---

# 17. Independent Failure Handling

Every camera needs its own lifecycle.

```text
CONNECTED
   │
   ▼
STREAMING
   │
   ├── network failure
   │
   ▼
RECONNECTING
   │
   ├── retry
   ├── retry
   ├── retry
   │
   ▼
STREAMING
```

A failed camera must not block the other 11 cameras.

The frontend should also display state such as:

```text
● LIVE
● CONNECTING
● RECONNECTING
● OFFLINE
```

---

# 18. Avoid Excessive Buffering

Increasing buffers can reduce visible stutter but increases latency.

For live surveillance, aim for a controlled small buffer:

```text
Camera
 ↓
Network jitter
 ↓
Small jitter buffer
 ↓
Decoder
 ↓
Display
```

Avoid:

```text
Camera
 ↓
Large buffer
 ↓
Browser
```

because the stream can gradually become seconds behind real time.

---

# 19. GOP / Keyframe Configuration

Pay attention to:

- GOP size
- I-frame interval
- Keyframe interval

For example:

```text
FPS = 15
GOP = 15–30
```

means a keyframe approximately every 1–2 seconds.

Very long GOP intervals can make stream startup and recovery slower.

Exact settings should be tuned against the camera encoder and network conditions.

---

# 20. Avoid Unnecessary Video Transformations

Every transformation consumes resources:

```text
Decode
 ↓
Resize
 ↓
Colorspace conversion
 ↓
Overlay
 ↓
Encode
```

If possible, keep the pipeline as:

```text
H.264
 ↓
H.264 packets
 ↓
WebRTC
```

The goal is to move compressed packets instead of repeatedly manipulating raw frames.

---

# 21. Browser Hardware Decoding

The client machines should also use hardware video decoding where available.

Chrome/Edge can expose GPU/video acceleration information through:

```text
chrome://gpu
```

Check that video decode is hardware accelerated.

This is particularly important if the client is displaying many simultaneous streams.

---

# 22. Visibility-Based Stream Management

For a dashboard with many cameras, the frontend can use viewport visibility.

For example:

```text
Visible camera
    ↓
PLAY

Partially visible camera
    ↓
LOW QUALITY

Completely invisible camera
    ↓
PAUSE / unsubscribe
```

The browser can use `IntersectionObserver` to determine which camera tiles are currently visible.

This can reduce client CPU/GPU usage when the dashboard grows beyond 12 cameras.

---

# 23. Recommended Technology Stack

## Cameras

```text
RTSP
H.264
15–25 FPS
CBR
Reasonable bitrate
```

## Media server

```text
MediaMTX
```

## Browser transport

```text
WebRTC
```

## Frontend

```text
Angular
HTMLVideoElement
RTCPeerConnection
```

## Backend

```text
Spring Boot
```

Use it for:

```text
REST
Authentication
Authorization
Camera metadata
Configuration
Events
WebSocket/SSE
```

## AI

```text
Python
MegaDetector
YOLO
ONNX Runtime / TensorRT
Tracking
```

## GPU

```text
RTX 3050
```

Use it for AI inference and, when required, hardware video decode/encode.

---

# 24. Recommended Mirage Architecture

For Mirage, the clean separation should look like:

```text
                 ┌───────────────┐
                 │     RTSP      │
                 │    Camera     │
                 └───────┬───────┘
                         │
                  ┌──────┴───────┐
                  │              │
                  ▼              ▼
             MediaMTX        Mirage AI
                  │              │
                  │              ├── MegaDetector
                  │              ├── YOLO
                  │              └── Tracker
                  │
                  ▼
               WebRTC
                  │
                  ▼
              Angular
                  │
          ┌───────┴────────┐
          │                │
       Video           Detection
       stream           overlay
```

Spring Boot or another application backend can sit alongside this for:

```text
Camera configuration
Authentication
Events
Detection metadata
User permissions
Recording metadata
```

---

# 25. Resource Scaling Strategy

The scaling strategy should be:

### 1–4 cameras

```text
Main or substream
```

### 5–12 cameras

```text
Substreams for grid
Main stream for selected camera
```

### 12+ cameras

Introduce:

```text
Visibility-based playback
Substream/main-stream switching
Camera grouping
Lazy WebRTC connection creation
```

The important thing is that increasing the number of cameras should not automatically mean increasing the number of CPU-based transcoding processes.

---

# 26. Architecture to Avoid

Avoid:

```text
RTSP
 ↓
OpenCV
 ↓
Python
 ↓
JPEG frames
 ↓
WebSocket
 ↓
Angular
```

This can work as a prototype but becomes expensive and difficult to scale for 12 simultaneous live streams.

Also avoid:

```text
RTSP
 ↓
FFmpeg
 ↓
HLS
 ↓
Angular
```

if very low latency is a primary requirement.

And avoid:

```text
12 RTSP streams
 ↓
12 CPU transcoders
 ↓
WebRTC
```

unless codec compatibility actually requires transcoding.

---

# 27. Core Engineering Principle

The entire architecture can be summarized as:

> Keep the camera's compressed video compressed for as long as possible.

Ideal path:

```text
Camera H.264
     │
     │ compressed packets
     ▼
  MediaMTX
     │
     │ compressed packets
     ▼
   WebRTC
     │
     ▼
Browser hardware decoder
```

This minimizes:

- CPU usage
- GPU usage
- Memory bandwidth
- Latency
- Copies
- Encoding overhead

and makes a 10–12 camera live dashboard much easier to operate reliably.

---

# 28. Final Recommended Architecture

```text
                         ┌──────────────────────────┐
                         │         Cameras          │
                         │                          │
                         │ Cam1 Cam2 ... Cam12     │
                         └────────────┬─────────────┘
                                      │
                                     RTSP
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │         MediaMTX          │
                         │                          │
                         │ RTSP ingestion           │
                         │ Stream management        │
                         │ WebRTC output            │
                         └───────┬───────────┬──────┘
                                 │           │
                              WebRTC       RTSP
                                 │           │
                                 ▼           ▼
                       ┌──────────────┐   ┌─────────────┐
                       │   Angular    │   │ AI Pipeline │
                       │  Dashboard   │   │             │
                       │              │   │ MegaDetector│
                       │ WebRTC       │   │ YOLO        │
                       │ Video        │   │ Tracker     │
                       │ Overlays     │   └──────┬──────┘
                       └──────┬───────┘          │
                              │                  │
                              │             Detection
                              │               events
                              │                  │
                              ▼                  ▼
                       ┌────────────────────────────┐
                       │        Spring Boot         │
                       │                            │
                       │ Camera API                 │
                       │ Authentication             │
                       │ Events                     │
                       │ Configuration              │
                       │ WebSocket/SSE              │
                       └────────────────────────────┘
```

## Bottom line

For the Mirage-style wildlife monitoring system, the recommended live-video plane is:

**RTSP → MediaMTX → WebRTC → Angular**

with:

- H.264 wherever possible
- No transcoding unless necessary
- Camera substreams for the 12-camera grid
- Main stream for the selected camera
- Hardware decoding in the browser
- Wired Gigabit networking
- Independent reconnection per camera
- AI inference isolated from the live-view path
- Detection metadata rendered as an overlay rather than encoded into video
- RTX 3050 hardware video acceleration only when transcoding is actually required

This gives you a clean separation between the **media plane**, **AI plane**, and **application plane**, which is the architecture I would use as the foundation for a production 10–12 camera system.
