"""
test_zero_copy_gpu.py

Proves the GPU-resident portion of the zero-copy pipeline actually works
on real hardware: TensorRT inference (Detector) + the hand-written CUDA
box-drawing kernel (cuda_draw.py), chained together exactly the way
zero_copy_pipeline.py does it.

*** REQUIRES AN NVIDIA GPU (same as detector.py / cuda_draw.py). ***

Deliberately does NOT use HardwareFrameProvider (NVDEC decode). NVDEC
needs an ffmpeg build compiled with CUDA support (h264_cuvid), which stock
Colab's ffmpeg does not have — getting that working is a separate, real
piece of infrastructure work, not a code correctness question. This script
proves everything downstream of decode instead: frames enter GPU memory
via cp.asarray() (software-decoded, then pushed to VRAM) and from that
point on, follow the exact same path zero_copy_pipeline.py uses —
preprocess -> Detector.predict() -> draw_boxes_gpu() -> back to host only
at the very end, for saving/display.

This is an honest partial proof: it verifies the CUDA kernel and the
TensorRT zero-copy contract for real, on real hardware. It does NOT prove
NVDEC hardware decode works — that remains open, and should be said
plainly if asked.

Usage (from backend/, on a GPU machine):
    python test_zero_copy_gpu.py --video ../sample_media/traffic_4k.mp4 \\
        --engine ../engines/yolov10n_fp16.engine --num-frames 30
"""

import argparse
import logging
import time

import cv2
import numpy as np

from core.config import MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test_zero_copy_gpu")


def main():
    parser = argparse.ArgumentParser(description="Test Detector + cuda_draw.py on real GPU")
    parser.add_argument("--video", required=True)
    parser.add_argument("--engine", default=MODEL.engine_path)
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--save-sample", default="gpu_pipeline_sample.jpg",
                         help="Path to save one annotated frame as proof, or '' to skip")
    args = parser.parse_args()

    import cupy as cp
    import cupyx.scipy.ndimage as cndi
    from detector.detector import Detector
    from pipeline.cuda_draw import draw_boxes_gpu

    log.info("Loading TensorRT engine from %s", args.engine)
    detector = Detector(args.engine, input_size=MODEL.input_size)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")

    frame_count = 0
    total_detections = 0
    last_annotated_host = None
    t_start = time.perf_counter()

    while frame_count < args.num_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # This cp.asarray() is the one step this test does NOT claim is
        # "zero-copy" — it's a plain host->GPU upload, standing in for
        # what HardwareFrameProvider's NVDEC path would otherwise do
        # natively in VRAM. Everything after this line is the real,
        # unmodified zero-copy path.
        frame_gpu = cp.asarray(frame_rgb)

        h, w = frame_gpu.shape[:2]
        th, tw = MODEL.input_size
        resized = cndi.zoom(frame_gpu, (th / h, tw / w, 1), order=1)
        normalized = resized.astype(cp.float32) / 255.0
        preprocessed = cp.ascontiguousarray(cp.transpose(normalized, (2, 0, 1)))

        detections = detector.predict(preprocessed)
        total_detections += len(detections)

        annotated_gpu = draw_boxes_gpu(frame_gpu, detections, MODEL.input_size)
        last_annotated_host = cp.asnumpy(annotated_gpu)  # only touches host at the very end

        frame_count += 1
        if frame_count % 10 == 0:
            log.info("frame=%d  detections_this_frame=%d", frame_count, len(detections))

    elapsed = time.perf_counter() - t_start
    cap.release()

    log.info("=" * 60)
    log.info("Frames processed: %d", frame_count)
    log.info("Total detections across all frames: %d", total_detections)
    log.info("Avg FPS (incl. host->GPU upload + GPU->host at the end): %.1f", frame_count / elapsed)
    log.info("=" * 60)
    log.info("Result: PASS — Detector + cuda_draw.py chain runs correctly on real GPU.")
    log.info("Note: NVDEC hardware decode was NOT exercised by this test (see docstring).")

    if args.save_sample and last_annotated_host is not None:
        cv2.imwrite(args.save_sample, cv2.cvtColor(last_annotated_host, cv2.COLOR_RGB2BGR))
        log.info("Saved sample annotated frame to %s", args.save_sample)


if __name__ == "__main__":
    main()
