"""
orchestration/stream_manager.py

Week 4: Multi-Stream Orchestration.

Manages N independent camera streams concurrently on the same GPU using
asyncio, plus dynamic swapping of which TensorRT engine a stream uses
(the "upload a different engine file to swap models on the fly" feature).

*** Each managed pipeline is a ZeroCopyPipeline, so this module inherits
the NVIDIA GPU requirement — it's an orchestration layer, not a new
GPU dependency in itself. ***
"""

import asyncio
import logging
from dataclasses import dataclass, field

from backend.core.config import ORCHESTRATOR, MODEL
from backend.benchmark.telemetry import hub as telemetry_hub
from backend.pipeline.zero_copy_pipeline import ZeroCopyPipeline

log = logging.getLogger("stream_manager")


@dataclass
class ManagedStream:
    stream_id: str
    source: str
    engine_path: str
    pipeline: ZeroCopyPipeline
    task: asyncio.Task | None = None
    frame_queue: "asyncio.Queue" = field(default_factory=lambda: asyncio.Queue(maxsize=2))


class StreamManager:
    """
    Owns the lifecycle of every active camera stream: start, stop, and
    hot-swap the detection engine without dropping the stream.

    A bounded frame_queue (maxsize=2) per stream is deliberate backpressure:
    if the WebRTC consumer falls behind, the pipeline drops the oldest
    unsent frame rather than buffering unboundedly and growing VRAM usage
    over time — directly relevant to the Week 2 "no VRAM leaks" audit.
    """

    def __init__(self, max_concurrent: int = ORCHESTRATOR.max_concurrent_streams):
        self.max_concurrent = max_concurrent
        self.streams: dict[str, ManagedStream] = {}
        self._watchdog_task: asyncio.Task | None = None

    async def start_stream(self, stream_id: str, source: str, engine_path: str = MODEL.engine_path):
        if stream_id in self.streams:
            raise ValueError(f"Stream '{stream_id}' is already running")
        if len(self.streams) >= self.max_concurrent:
            raise RuntimeError(
                f"At capacity ({self.max_concurrent} streams). Stop a stream "
                f"or raise ORCHESTRATOR.max_concurrent_streams if VRAM allows it."
            )

        pipeline = ZeroCopyPipeline(source=source, engine_path=engine_path)
        managed = ManagedStream(stream_id=stream_id, source=source, engine_path=engine_path, pipeline=pipeline)
        self.streams[stream_id] = managed

        telemetry_hub.register_stream(stream_id)

        async def _run():
            loop = asyncio.get_running_loop()

            def _safe_put(queue: "asyncio.Queue", item):
                # See main.py's identical fix for why the try/except must be
                # inside the scheduled callback, not around call_soon_threadsafe.
                try:
                    queue.put_nowait(item)
                except asyncio.QueueFull:
                    pass  # deliberate drop — see class docstring

            def on_frame(annotated_gpu, detections, frame_index):
                telemetry_hub.tick(stream_id)
                loop.call_soon_threadsafe(_safe_put, managed.frame_queue, annotated_gpu)

            # ZeroCopyPipeline.run() is CPU/GPU-bound, not I/O-bound, so it
            # runs in a worker thread and communicates back via the queue.
            await loop.run_in_executor(None, pipeline.run, on_frame)

        managed.task = asyncio.create_task(_run())
        log.info("Started stream '%s' (%d/%d active)", stream_id, len(self.streams), self.max_concurrent)

        if self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(self._watchdog())

    async def stop_stream(self, stream_id: str):
        managed = self.streams.pop(stream_id, None)
        if not managed:
            return
        if managed.task:
            managed.task.cancel()
        managed.pipeline.close()
        telemetry_hub.unregister_stream(stream_id)
        log.info("Stopped stream '%s' (%d remaining)", stream_id, len(self.streams))

    async def swap_engine(self, stream_id: str, new_engine_path: str):
        """
        Hot-swaps the TensorRT engine a running stream uses — e.g. the user
        uploads a differently-trained model from the React UI and wants to
        see it applied without restarting the whole stream/connection.

        Implemented as stop-and-restart of the pipeline (not in-place engine
        replacement): TensorRT execution contexts are not designed to swap
        engines mid-flight, so a clean restart is the reliable approach.
        The WebRTC connection itself stays up — only the underlying source
        pipeline restarts, so the viewer sees a brief freeze, not a
        disconnect.
        """
        managed = self.streams.get(stream_id)
        if not managed:
            raise ValueError(f"No active stream '{stream_id}'")

        source = managed.source
        await self.stop_stream(stream_id)
        await self.start_stream(stream_id, source, engine_path=new_engine_path)
        log.info("Swapped engine for stream '%s' -> %s", stream_id, new_engine_path)

    async def _watchdog(self):
        """Periodically checks for VRAM growth across all active streams."""
        while True:
            await asyncio.sleep(ORCHESTRATOR.watchdog_interval_s)
            if not self.streams:
                continue
            telemetry_hub.check_for_leak()

    async def stop_all(self):
        for stream_id in list(self.streams.keys()):
            await self.stop_stream(stream_id)
        if self._watchdog_task:
            self._watchdog_task.cancel()


# Module-level singleton, mirroring telemetry.hub — main.py wires the REST/
# WebRTC layer to this same instance.
manager = StreamManager()
