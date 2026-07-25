"""
pipeline/pytorch_pipeline.py

A third pipeline mode, alongside file-demo (no detection) and gpu
(TensorRT/CuPy). This one runs REAL detection end-to-end using only
PyTorch + OpenCV — no NVIDIA GPU, no TensorRT, no CuPy, no NVDEC.

Runs on any machine: CPU-only, Mac (Apple Silicon via MPS), AMD GPU host,
whatever. Slower than the TensorRT path, but it's the same DIP-friendly
swap the project's Detector abstraction was designed to support from Week 1 —
same idea discussed earlier: nothing outside this class cares which
inference backend is underneath it.

Everything here is plain numpy/OpenCV (host memory), not CuPy — there is
no "zero-copy" claim for this path, and none is made. This exists purely
so the full stack (decode -> detect -> draw -> stream) is demoable on
hardware that doesn't have an NVIDIA GPU at all.
"""

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from backend.core.config import MODEL, STREAM
from backend.detector.pytorch_baseline import PyTorchDetector

log = logging.getLogger("pytorch_pipeline")

# Same fixed palette as the CUDA-drawing path (pipeline/cuda_draw.py), kept
# in sync so annotated output looks the same regardless of which backend
# actually ran the pipeline.
_PALETTE = [
    (255, 99, 71), (60, 179, 113), (65, 105, 225), (255, 215, 0),
    (218, 112, 214), (0, 206, 209), (255, 140, 0), (154, 205, 50),
]


@dataclass
class PipelineStats:
    frames_processed: int = 0
    total_time_s: float = 0.0

    @property
    def avg_fps(self) -> float:
        return self.frames_processed / self.total_time_s if self.total_time_s > 0 else 0.0


class PyTorchPipeline:
    """
    Software decode (OpenCV) -> PyTorch inference -> OpenCV drawing ->
    output frame (numpy, host memory). Same run()/on_frame() shape as
    ZeroCopyPipeline so streaming/webrtc_server.py and main.py can treat
    both interchangeably.
    """

    def __init__(self, source: str, weights_path: str = MODEL.pytorch_weights):
        self.source = source
        self.detector = PyTorchDetector(weights_path)  # auto-picks cuda if available, else cpu
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        self.stats = PipelineStats()
        self._stop_event = threading.Event()

    def stop(self):
        """
        Signals run()'s loop to exit at the next frame boundary. Required
        because run(loop=True) otherwise loops the video file forever with
        no natural exit point — without calling this, the background
        thread running run() never returns, and Python's ThreadPoolExecutor
        will hang waiting for it to finish even after Ctrl+C, since
        run_in_executor's worker threads block process exit until their
        submitted work completes.
        """
        self._stop_event.set()

    def _draw_boxes(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Ultralytics results -> boxes drawn with OpenCV, host-side.

        Labels use the model's actual class names (e.g. "car", "person"),
        not raw numeric class IDs — Ultralytics exposes this mapping as
        `result.names` (a {id: name} dict) on every results object, so no
        extra lookup table is needed.
        """
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            class_names = r.names  # {0: 'person', 1: 'bicycle', 2: 'car', ...}

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                color = _PALETTE[cls_id % len(_PALETTE)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                class_name = class_names.get(cls_id, str(cls_id))
                label = f"{class_name} {conf * 100:.0f}%"

                # Filled background behind the text so labels stay
                # readable over busy/bright video instead of plain
                # colored text that can wash out against the frame.
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y1 = max(y1 - text_h - baseline - 4, 0)
                cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 6, y1), color, -1)

                # Black or white text, whichever contrasts better against
                # this box's color (simple luminance check).
                luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
                text_color = (0, 0, 0) if luminance > 150 else (255, 255, 255)
                cv2.putText(frame, label, (x1 + 3, y1 - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        return frame

    def run(self, on_frame=None, max_frames: int | None = None, loop: bool = True):
        """
        Same signature/behavior as ZeroCopyPipeline.run(): consumes the
        stream frame by frame, calling on_frame(annotated_frame, results, i)
        for each processed frame.

        loop=True restarts the video file automatically when it ends —
        useful for a demo video file rather than a real RTSP feed, which
        wouldn't need this.
        """
        i = 0
        while not self._stop_event.is_set():
            ret, frame_bgr = self.cap.read()
            if not ret:
                if loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            if max_frames is not None and i >= max_frames:
                break

            t0 = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.detector.predict(frame_rgb)
            annotated = self._draw_boxes(frame_bgr.copy(), results)
            t1 = time.perf_counter()

            self.stats.frames_processed += 1
            self.stats.total_time_s += (t1 - t0)

            if on_frame:
                on_frame(annotated, results, i)

            if i % 100 == 0 and i > 0:
                log.info("[%s] frame=%d avg_fps=%.1f", self.source, i, self.stats.avg_fps)

            i += 1

    def close(self):
        self.cap.release()

