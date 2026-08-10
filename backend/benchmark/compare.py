import argparse
from backend.core.logger import get_logger
import statistics
import time

import numpy as np

from core.config import MODEL, STREAM

log = get_logger(__name__)


def _load_sample_frames(video_path: str, num_frames: int) -> list[np.ndarray]:
    """Grabs num_frames evenly spaced frames from a video file for a repeatable benchmark."""
    import av

    frames = []
    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames or num_frames * 10
    step = max(1, total // num_frames)

    for i, frame in enumerate(container.decode(stream)):
        if i % step == 0:
            frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= num_frames:
            break

    container.close()
    return frames


def benchmark_pytorch(weights_path: str, frames: list[np.ndarray]) -> dict:
    from detector.pytorch_baseline import PyTorchDetector

    detector = PyTorchDetector(weights_path)
    latencies = []

    # warmup
    for f in frames[:5]:
        detector.predict(f)

    for f in frames:
        _, elapsed = detector.predict_timed(f)
        latencies.append(elapsed)

    return _summarize("PyTorch (native)", latencies)


def benchmark_tensorrt(engine_path: str, input_size: tuple, frames: list[np.ndarray]) -> dict:
    import cupy as cp
    from detector.detector import Detector

    detector = Detector(engine_path, input_size=input_size)
    latencies = []

    def preprocess(frame_np):
        gpu = cp.asarray(frame_np)
        import cupyx.scipy.ndimage as cndi
        h, w = gpu.shape[:2]
        th, tw = input_size
        resized = cndi.zoom(gpu, (th / h, tw / w, 1), order=1)
        normalized = resized.astype(cp.float32) / 255.0
        chw = cp.ascontiguousarray(cp.transpose(normalized, (2, 0, 1)))
        return chw

    # warmup — first inference includes CUDA context / kernel JIT overhead,
    # excluded from the timed measurement so the benchmark reflects
    # steady-state throughput, not cold-start cost.
    for f in frames[:5]:
        detector.predict(preprocess(f))

    for f in frames:
        gpu_input = preprocess(f)
        start = time.perf_counter()
        detector.predict(gpu_input)
        latencies.append(time.perf_counter() - start)

    return _summarize("TensorRT", latencies)


def _summarize(label: str, latencies: list[float]) -> dict:
    mean_latency = statistics.mean(latencies)
    return {
        "label": label,
        "mean_latency_ms": mean_latency * 1000,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
        "fps": 1.0 / mean_latency if mean_latency > 0 else float("inf"),
        "num_samples": len(latencies),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch vs TensorRT inference speed")
    parser.add_argument("--video", default=STREAM.mock_video_path)
    parser.add_argument("--weights", default=MODEL.pytorch_weights)
    parser.add_argument("--engine", default=MODEL.engine_path)
    parser.add_argument("--num-frames", type=int, default=200)
    args = parser.parse_args()

    log.info("Loading %d sample frames from %s", args.num_frames, args.video)
    frames = _load_sample_frames(args.video, args.num_frames)

    pytorch_result = benchmark_pytorch(args.weights, frames)
    trt_result = benchmark_tensorrt(args.engine, MODEL.input_size, frames)

    speedup = trt_result["fps"] / pytorch_result["fps"]

    log.info("=" * 60)
    for r in (pytorch_result, trt_result):
        log.info("%-16s  fps=%.1f  mean=%.2fms  p95=%.2fms",
                  r["label"], r["fps"], r["mean_latency_ms"], r["p95_latency_ms"])
    log.info("=" * 60)
    log.info("Speedup: %.2fx  (target: >=3.0x)", speedup)
    log.info("Result: %s", "PASS" if speedup >= 3.0 else "FAIL — below 3x target")


if __name__ == "__main__":
    main()

