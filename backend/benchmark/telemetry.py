from backend.core.logger import get_logger
import time
from collections import deque
from dataclasses import dataclass, field

log = get_logger(__name__)

@dataclass
class StreamTelemetry:
    stream_id: str
    fps_window: deque = field(default_factory=lambda: deque(maxlen=60))
    last_frame_ts: float = field(default_factory=time.time)

    def tick(self):
        now = time.time()
        instant_fps = 1.0 / (now - self.last_frame_ts) if now > self.last_frame_ts else 0.0
        self.fps_window.append(instant_fps)
        self.last_frame_ts = now

    @property
    def fps(self) -> float:
        return sum(self.fps_window) / len(self.fps_window) if self.fps_window else 0.0


class TelemetryHub:
    """
    Central collector the orchestrator and every ZeroCopyPipeline instance
    report into. The WebRTC/REST layer reads `snapshot()` to feed the React
    telemetry dashboard (FPS per stream, GPU memory %, decoder utilization).
    """

    def __init__(self):
        self._streams: dict[str, StreamTelemetry] = {}
        self._leak_baseline_mb: float | None = None

    def register_stream(self, stream_id: str):
        self._streams[stream_id] = StreamTelemetry(stream_id=stream_id)
        log.info("Registered stream: %s", stream_id)

    def unregister_stream(self, stream_id: str):
        self._streams.pop(stream_id, None)

    def tick(self, stream_id: str):
        if stream_id in self._streams:
            self._streams[stream_id].tick()

    def gpu_stats(self) -> dict:
        """
        Polls NVML for VRAM usage and decoder/compute utilization.
        Returns zeros with a warning if pynvml or the driver isn't available
        (e.g. running this on a non-NVIDIA dev machine) so the dashboard
        degrades gracefully instead of crashing.
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            try:
                dec_util, _ = pynvml.nvmlDeviceGetDecoderUtilization(handle)
            except pynvml.NVMLError:
                dec_util = None  # not all cards/drivers expose this

            pynvml.nvmlShutdown()

            return {
                "vram_used_mb": mem.used / (1 << 20),
                "vram_total_mb": mem.total / (1 << 20),
                "vram_percent": 100.0 * mem.used / mem.total,
                "gpu_util_percent": util.gpu,
                "decoder_util_percent": dec_util,
            }
        except Exception as e:
            log.warning("GPU stats unavailable (%s) — is an NVIDIA GPU present with pynvml installed?", e)
            return {
                "vram_used_mb": 0, "vram_total_mb": 0, "vram_percent": 0,
                "gpu_util_percent": 0, "decoder_util_percent": None,
            }

    def check_for_leak(self, warn_growth_mb: float = 500.0) -> bool:
        """
        Simple leak heuristic: compare current VRAM usage against the first
        reading taken after warmup. Call this periodically (e.g. every 60s)
        during a soak test. A steadily growing delta across many calls,
        not just one high reading, is the actual leak signal — log the
        returned bool over time rather than trusting a single sample.
        """
        current = self.gpu_stats()["vram_used_mb"]
        if self._leak_baseline_mb is None:
            self._leak_baseline_mb = current
            return False

        growth = current - self._leak_baseline_mb
        if growth > warn_growth_mb:
            log.warning("VRAM usage grew %.0fMB since baseline — possible leak", growth)
            return True
        return False

    def snapshot(self) -> dict:
        """Full state for the dashboard: per-stream FPS + current GPU stats."""
        return {
            "streams": {sid: {"fps": s.fps} for sid, s in self._streams.items()},
            "gpu": self.gpu_stats(),
            "timestamp": time.time(),
        }


# Module-level singleton — every part of the backend imports this same instance.
hub = TelemetryHub()


