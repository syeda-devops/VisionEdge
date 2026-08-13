import logging
import time
from dataclasses import dataclass

from core.config import MODEL, STREAM
from decoder.frame_provider import HardwareFrameProvider
from detector.detector import Detector
from pipeline.cuda_draw import draw_boxes_gpu

log = logging.getLogger("zero_copy_pipeline")


@dataclass
class PipelineStats:
    frames_processed: int = 0
    total_decode_s: float = 0.0
    total_inference_s: float = 0.0
    total_draw_s: float = 0.0

    @property
    def avg_fps(self) -> float:
        total = self.total_decode_s + self.total_inference_s + self.total_draw_s
        return self.frames_processed / total if total > 0 else 0.0


class ZeroCopyPipeline:
    """
    Orchestrates one camera stream end-to-end, entirely in GPU memory:

        NVDEC decode -> preprocess -> TensorRT inference -> CUDA draw -> output

    The output frame (still a CuPy/GPU array) is handed to the caller
    (normally the WebRTC streaming module), which is the first point where
    the data may need to touch host memory again — for encoding to send
    over the network.
    """

    def __init__(
        self,
        source: str,
        engine_path: str = MODEL.engine_path,
        gpu_id: int = STREAM.gpu_id,
        input_size: tuple = MODEL.input_size,
    ):
        self.source = source
        self.gpu_id = gpu_id
        self.input_size = input_size

        self.decoder = HardwareFrameProvider(source, gpu_id=gpu_id, target_size=None)
        self.detector = Detector(engine_path, input_size=input_size)
        self.stats = PipelineStats()

    def _preprocess_gpu(self, frame_gpu):
        """
        Resize to network input size, normalize to [0,1], and transpose
        HWC -> CHW. Stays in CuPy the entire time — no numpy involved.
        """
        import cupy as cp
        import cupyx.scipy.ndimage as cndi

        h, w = frame_gpu.shape[:2]
        target_h, target_w = self.input_size
        zoom = (target_h / h, target_w / w, 1)
        resized = cndi.zoom(frame_gpu, zoom, order=1)

        normalized = (resized.astype(cp.float32) / 255.0)
        chw = cp.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        return cp.ascontiguousarray(chw)

    def run(self, on_frame=None, max_frames: int | None = None):
        """
        Consumes the stream frame by frame.

        Parameters
        ----------
        on_frame : callable | None
            Called as on_frame(annotated_frame_gpu, detections, frame_index)
            for every processed frame — the WebRTC module hooks in here.
        max_frames : int | None
            Stop after N frames (useful for local testing); None = run
            until the source is exhausted or the process is stopped.
        """
        for i, raw_frame_gpu in enumerate(self.decoder.frames_gpu()):
            if max_frames is not None and i >= max_frames:
                break

            t0 = time.perf_counter()
            preprocessed = self._preprocess_gpu(raw_frame_gpu)
            t1 = time.perf_counter()

            detections = self.detector.predict(preprocessed)
            t2 = time.perf_counter()

            annotated_gpu = draw_boxes_gpu(raw_frame_gpu, detections, self.input_size)
            t3 = time.perf_counter()

            self.stats.frames_processed += 1
            self.stats.total_decode_s += (t1 - t0)   # includes GPU preprocessing;
                                            # decode time is measured separately by FrameProvider
            self.stats.total_inference_s += (t2 - t1)
            self.stats.total_draw_s += (t3 - t2)

            if on_frame:
                on_frame(annotated_gpu, detections, i)

            if i % 300 == 0 and i > 0:
                log.info("[%s] frame=%d avg_fps=%.1f", self.source, i, self.stats.avg_fps)

    def close(self):
        self.decoder.close()

