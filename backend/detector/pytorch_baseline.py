import logging
import time

log = logging.getLogger("pytorch_baseline")


class PyTorchDetector:
    """Same predict() interface as detector.Detector, so benchmarks are apples-to-apples."""

    def __init__(self, weights_path: str, device: str | None = None):
        from ultralytics import YOLO
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading PyTorch model on device=%s", self.device)
        self.model = YOLO(weights_path)
        self.model.to(self.device)

    def predict(self, frame):
        """
        Parameters
        ----------
        frame : numpy.ndarray
            HWC, BGR or RGB uint8 image — standard Ultralytics input.
        """
        return self.model(frame, verbose=False)

    def predict_timed(self, frame) -> tuple[object, float]:
        """Returns (results, latency_seconds) for benchmarking."""
        start = time.perf_counter()
        results = self.predict(frame)
        elapsed = time.perf_counter() - start
        return results, elapsed