"""
main.py

Application entrypoint. Wires together:
  - streaming.webrtc_server  (WebRTC signaling + video tracks)
  - orchestration.stream_manager (multi-stream lifecycle, GPU mode only)
  - benchmark.telemetry      (live metrics for the dashboard)

into one aiohttp app the React frontend talks to.

Run modes
---------
Week 1 milestone only, no detection, no GPU:
    python main.py --mode file-demo

Full pipeline with REAL detection, no NVIDIA GPU required (PyTorch, CPU
or whatever device torch picks up — MPS on Mac, CUDA if present but not
required):
    python main.py --mode cpu

Full pipeline, TensorRT/CuPy zero-copy (requires NVIDIA GPU + built engine):
    python main.py --mode gpu
"""

import argparse
import asyncio
import logging
import sys

from aiohttp import web

from core.config import WEBRTC, STREAM, MODEL
from orchestration.stream_manager import manager as stream_manager
from benchmark.telemetry import hub as telemetry_hub
from streaming.webrtc_server import build_app, FileVideoTrack, PipelineVideoTrack, CPUPipelineVideoTrack

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("main")


async def _windows_ctrl_c_keepalive():
    """
    Windows-only fix for Ctrl+C being silently ignored.

    This is a long-standing, documented Python/asyncio limitation on
    Windows (see bugs.python.org / GitHub python/asyncio#191): the default
    ProactorEventLoop blocks on a low-level OS wait and only gets a chance
    to notice a pending Ctrl+C when something else wakes it up. The
    official workaround, straight from that issue, is exactly this — a
    trivial periodic task that wakes the loop regularly.

    Deliberately NOT switching to SelectorEventLoop as an alternative fix:
    that was tried first and caused a worse problem — SelectorEventLoop on
    Windows has a hard 512-socket limit and weaker handling of the kind of
    UDP churn aiortc's ICE negotiation does across many simultaneous
    connections, which showed up as the server silently freezing after
    about a minute of real use. Waking the *default* loop periodically
    fixes Ctrl+C without giving up ProactorEventLoop's better Windows
    networking support.
    """
    while True:
        await asyncio.sleep(0.5)

CPU_STREAM_ID = "local-cpu"
_cpu_frame_queue: "asyncio.Queue | None" = None

# The dashboard's 10 tiles (App.jsx) are hard-coded to these IDs, built for
# the full GPU multi-stream scenario. --mode cpu runs exactly one real
# pipeline, so we report that pipeline's one real FPS number under every
# display name the dashboard is asking about — otherwise the FPS field
# stays empty even though video is genuinely flowing (telemetry_hub simply
# never heard of "intersection-01" etc., since --mode cpu never registered
# a stream under those names).
DASHBOARD_STREAM_IDS = [f"intersection-{i:02d}" for i in range(1, 11)]


async def handle_start_stream(request: web.Request) -> web.Response:
    body = await request.json()
    stream_id = body["stream_id"]
    source = body.get("source", STREAM.mock_video_path)
    engine_path = body.get("engine_path", MODEL.engine_path)

    await stream_manager.start_stream(stream_id, source, engine_path)
    return web.json_response({"status": "started", "stream_id": stream_id})


async def handle_stop_stream(request: web.Request) -> web.Response:
    stream_id = request.match_info["stream_id"]
    await stream_manager.stop_stream(stream_id)
    return web.json_response({"status": "stopped", "stream_id": stream_id})


async def handle_swap_engine(request: web.Request) -> web.Response:
    """Backs the 'upload a TensorRT engine to swap models on the fly' feature."""
    stream_id = request.match_info["stream_id"]
    reader = await request.multipart()
    field = await reader.next()

    upload_path = f"engines/uploaded_{stream_id}.engine"
    with open(upload_path, "wb") as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            f.write(chunk)

    await stream_manager.swap_engine(stream_id, upload_path)
    return web.json_response({"status": "swapped", "stream_id": stream_id, "engine_path": upload_path})


async def handle_telemetry(request: web.Request) -> web.Response:
    return web.json_response(telemetry_hub.snapshot())


def gpu_track_factory(stream_id: str, app: web.Application):
    """Returns a live WebRTC track backed by a running ZeroCopyPipeline's frame queue."""
    managed = stream_manager.streams.get(stream_id)
    if not managed:
        raise web.HTTPNotFound(text=f"Stream '{stream_id}' is not running. POST /streams/start first.")
    return PipelineVideoTrack(managed.frame_queue)


def file_track_factory(stream_id: str, app: web.Application):
    """Week 1 milestone factory: always streams the mock video file, no GPU needed."""
    return FileVideoTrack(STREAM.mock_video_path)


def cpu_track_factory(stream_id: str, app: web.Application):
    """
    --mode cpu factory: every incoming connection (all 10 dashboard tiles)
    gets a RELAYED copy of the same single underlying PyTorch-pipeline
    track, via aiortc's MediaRelay — not a fresh independent track each
    time. This mode runs one real stream (not the 10-camera scale story;
    it's a no-GPU-required local demo), so every tile ends up showing the
    same live detected feed rather than 10 tiles starving one shared queue.
    """
    base_track = app.get("cpu_base_track")
    relay = app.get("cpu_relay")
    if base_track is None or relay is None:
        raise web.HTTPServiceUnavailable(text="CPU pipeline has not started yet.")
    return relay.subscribe(base_track)


def build_full_app(track_factory) -> web.Application:
    app = build_app(track_factory)
    app.router.add_post("/streams/start", handle_start_stream)
    app.router.add_post("/streams/{stream_id}/stop", handle_stop_stream)
    app.router.add_post("/streams/{stream_id}/swap-engine", handle_swap_engine)
    app.router.add_get("/telemetry", handle_telemetry)

    async def on_startup_keepalive(app: web.Application):
        if sys.platform == "win32":
            app["_win_keepalive_task"] = asyncio.create_task(_windows_ctrl_c_keepalive())

    app.on_startup.append(on_startup_keepalive)
    return app


def build_cpu_app(source) -> web.Application:
    """
    --mode cpu: starts PyTorchPipeline once at app startup (background
    thread), feeding a single shared frame queue. A single CPUPipelineVideoTrack
    reads that queue, and MediaRelay fans it out to every connecting client
    (the dashboard opens up to 10 simultaneous connections for its 10 tiles).

    Without the relay, each of the 10 tiles would create its own independent
    track all draining the SAME queue — 10 consumers competing over one
    producer's frames, starving each other badly enough that connections
    time out. MediaRelay exists in aiortc specifically for this "one
    source, many viewers" case: it subscribes each new peer connection to
    its own copy of the same underlying track's frames.

    Parameters
    ----------
    source : str | int
        A video file path, or an integer webcam device index (0 is
        usually the default/built-in camera). Passed straight to
        cv2.VideoCapture.
    """
    from pipeline.pytorch_pipeline import PyTorchPipeline
    from streaming.webrtc_server import CPUPipelineVideoTrack
    from aiortc.contrib.media import MediaRelay

    app = build_full_app(cpu_track_factory)
    relay = MediaRelay()
    app["cpu_relay"] = relay

    async def on_startup(app: web.Application):
        global _cpu_frame_queue
        _cpu_frame_queue = asyncio.Queue(maxsize=2)
        for stream_id in DASHBOARD_STREAM_IDS:
            telemetry_hub.register_stream(stream_id)

        loop = asyncio.get_running_loop()
        pipeline = PyTorchPipeline(source=source, weights_path=MODEL.pytorch_weights)

        base_track = CPUPipelineVideoTrack(_cpu_frame_queue)
        app["cpu_base_track"] = base_track
        app["cpu_pipeline"] = pipeline

        def _safe_put(queue: "asyncio.Queue", item):
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                pass  # drop oldest-unsent frame under backpressure

        def on_frame(annotated_bgr, results, frame_index):
            # One real pipeline's tick reported under all 10 dashboard
            # names — see DASHBOARD_STREAM_IDS comment above.
            for stream_id in DASHBOARD_STREAM_IDS:
                telemetry_hub.tick(stream_id)
            loop.call_soon_threadsafe(_safe_put, _cpu_frame_queue, annotated_bgr)

        def run_pipeline():
            log.info("PyTorch pipeline starting (device=%s, source=%s)", pipeline.detector.device, source)
            pipeline.run(on_frame=on_frame)

        # CPU-bound tight loop -> dedicated thread, not the event loop.
        loop.run_in_executor(None, run_pipeline)

    async def on_cleanup(app: web.Application):
        # Without this, Ctrl+C hangs: run_pipeline() above loops the video
        # file forever (loop=True default), and Python's ThreadPoolExecutor
        # blocks process exit until that background thread's work actually
        # returns. pipeline.stop() sets an event run()'s loop checks every
        # frame, so it exits promptly instead of leaving the process stuck.
        pipeline = app.get("cpu_pipeline")
        if pipeline:
            log.info("Stopping PyTorch pipeline...")
            pipeline.stop()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main():
    parser = argparse.ArgumentParser(description="VisionEdge backend")
    parser.add_argument("--mode", choices=["file-demo", "cpu", "gpu"], default="file-demo",
                         help="file-demo: Week 1 milestone, no detection. "
                              "cpu: full pipeline with real PyTorch detection, no NVIDIA GPU required. "
                              "gpu: full TensorRT/CuPy zero-copy pipeline, needs NVIDIA hardware.")
    args = parser.parse_args()

    if args.mode == "file-demo":
        log.info("Starting in file-demo mode (Week 1 milestone — no GPU required, no detection)")
        app = build_full_app(file_track_factory)
    elif args.mode == "cpu":
        log.info("Starting in cpu mode (real PyTorch detection, no NVIDIA GPU required)")
        app = build_cpu_app(source=STREAM.mock_video_path)
    else:
        log.info("Starting in gpu mode (requires NVIDIA GPU + TensorRT engine at %s)", MODEL.engine_path)
        app = build_full_app(gpu_track_factory)

    web.run_app(app, host=WEBRTC.host, port=WEBRTC.port)


if __name__ == "__main__":
    main()
