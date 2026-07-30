
"""
streaming/webrtc_server.py

Week 1: WebRTC Foundation, and the module every later week builds on top of.

This module has NO NVIDIA dependency — aiortc, av, and asyncio all run on
plain CPU. This is exactly why the spec puts it in Week 1: you build and
test the entire streaming path with a plain video file first, then swap
the frame source for the GPU pipeline later without touching this file.

Two video track implementations are provided:
  - FileVideoTrack       : streams a video file as-is (Week 1 milestone)
  - PipelineVideoTrack    : streams annotated frames from ZeroCopyPipeline
                            (wired in from Week 3 onward)
"""

import asyncio
import fractions
import logging
import time

import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer
from av import VideoFrame

from core.config import WEBRTC, STREAM

log = logging.getLogger("webrtc_server")

pcs: set[RTCPeerConnection] = set()
class FileVideoTrack(VideoStreamTrack):
 """
    Week 1 milestone: stream a plain video file, unedited, to prove the
    WebRTC plumbing (offer/answer, ICE, encoding) works end to end before
    any GPU code exists.
    """

def __init__(self, video_path: str):
    super().__init__()
    self._player = MediaPlayer(video_path, loop=True)

    async def recv(self):
        frame = await self._player.video.recv()
        return frame


class PipelineVideoTrack(VideoStreamTrack):
    """
    Streams frames produced by ZeroCopyPipeline (Week 3+, NVIDIA-only).
    Frames arrive as GPU (CuPy) arrays from the pipeline; this is the point
    where they cross back to host memory, since aiortc's encoder needs a
    CPU-side av.VideoFrame. That single host copy per output frame is
    unavoidable — it's the price of getting pixels onto the network — but
    everything upstream of this line stayed in VRAM.
    """

    def __init__(self, frame_queue: "asyncio.Queue"):
        super().__init__()
        self._queue = frame_queue
        self._frame_count = 0

    async def recv(self):
        annotated_gpu = await self._queue.get()

        import cupy as cp
        annotated_host = cp.asnumpy(annotated_gpu)  # single unavoidable GPU->CPU copy

        video_frame = VideoFrame.from_ndarray(annotated_host, format="rgb24")
        video_frame.pts = self._frame_count
        video_frame.time_base = fractions.Fraction(1, STREAM.target_fps)
        self._frame_count += 1
        return video_frame


class CPUPipelineVideoTrack(VideoStreamTrack):
    """
    Streams frames produced by PyTorchPipeline (no NVIDIA GPU required).
    Frames already arrive as plain host-memory numpy arrays (BGR, from
    OpenCV), so there's no GPU->CPU copy step here — just a color-space
    fix, since OpenCV uses BGR and av.VideoFrame expects RGB.
    """

    def __init__(self, frame_queue: "asyncio.Queue"):
        super().__init__()
        self._queue = frame_queue
        self._frame_count = 0

    async def recv(self):
        import cv2

        annotated_bgr = await self._queue.get()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        video_frame = VideoFrame.from_ndarray(annotated_rgb, format="rgb24")
        video_frame.pts = self._frame_count
        video_frame.time_base = fractions.Fraction(1, STREAM.target_fps)
        self._frame_count += 1
        return video_frame


async def offer(request: web.Request) -> web.Response:
    """
    WebRTC signaling endpoint. The React frontend POSTs its SDP offer here
    and gets an SDP answer back — standard aiortc signaling pattern.
    """
    params = await request.json()
    stream_id = params.get("stream_id", "default")
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log.info("Connection state for stream=%s: %s", stream_id, pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            pcs.discard(pc)
            await pc.close()

    track = request.app["track_factory"](stream_id, request.app)
    pc.addTrack(track)

    await pc.setRemoteDescription(offer_sdp)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })


async def on_shutdown(app: web.Application):
    await asyncio.gather(*(pc.close() for pc in pcs))
    pcs.clear()


def build_app(track_factory) -> web.Application:
    """
    track_factory(stream_id: str) -> VideoStreamTrack
        Called per incoming connection. Week 1 passes a factory that always
        returns a FileVideoTrack; later weeks pass one that looks up the
        right PipelineVideoTrack for stream_id.
    """
    app = web.Application()
    app["track_factory"] = track_factory
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)

    # Permissive CORS for local dev — tighten before any real deployment.
    @web.middleware
    async def cors_middleware(request, handler):
        response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response

    app.middlewares.append(cors_middleware)
    return app


def run_week1_demo():
    """
    Entry point for the Week 1 milestone: stream STREAM.mock_video_path,
    unedited, to whatever connects. No GPU, no detector, no TensorRT —
    just proving the transport layer.

    Run with: python -m streaming.webrtc_server
    """
    logging.basicConfig(level=logging.INFO)

    def factory(stream_id: str):
        return FileVideoTrack(STREAM.mock_video_path)

    app = build_app(factory)
    log.info("WebRTC signaling server on http://%s:%s/offer", WEBRTC.host, WEBRTC.port)
    web.run_app(app, host=WEBRTC.host, port=WEBRTC.port)


if __name__ == "__main__":
    run_week1_demo()
