"""
test_nvdec_pipeline.py

The full end-to-end test: HardwareFrameProvider (NVDEC decode) ->
Detector (TensorRT) -> cuda_draw (CUDA kernel), exactly as
zero_copy_pipeline.py orchestrates them. Unlike test_zero_copy_gpu.py,
this one does NOT bypass decode — it's only meaningful once PyAV has
been rebuilt from source against a CUDA-enabled ffmpeg (see
docs/SETUP.md and the NVDEC Colab notebook).

*** REQUIRES: NVIDIA GPU + a from-source PyAV build with NVDEC support.
Run test_zero_copy_gpu.py first if you haven't already — it isolates
TensorRT + the CUDA kernel from the decode question entirely, and is a
faster way to confirm those two are solid before adding decode into
the mix here. ***

Usage (from backend/, on a GPU machine with NVDEC-enabled PyAV):
    python test_nvdec_pipeline.py --video ../sample_media/traffic_4k.mp4 \\
        --engine ../engines/yolov10n_fp16.engine --num-frames 30
"""

import argparse
import logging

import cv2

from core.config import MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test_nvdec_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Test the full NVDEC zero-copy pipeline")
    parser.add_argument("--video", required=True)
    parser.add_argument("--engine", default=MODEL.engine_path)
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--save-sample", default="nvdec_pipeline_sample.jpg",
                         help="Path to save one annotated frame as proof, or '' to skip")
    args = parser.parse_args()

    from pipeline.zero_copy_pipeline import ZeroCopyPipeline

    log.info("Attempting to open %s with hardware (NVDEC) decode...", args.video)
    try:
        pipeline = ZeroCopyPipeline(source=args.video, engine_path=args.engine)
    except RuntimeError as e:
        log.error("NVDEC decode is not active: %s", e)
        log.error("This is expected unless PyAV has been rebuilt from source against a "
                   "CUDA-enabled ffmpeg. Run test_zero_copy_gpu.py instead to test "
                   "TensorRT + the CUDA kernel without needing real hardware decode.")
        raise SystemExit(1)

    total_detections = 0
    last_annotated_gpu = None

    def on_frame(annotated_gpu, detections, frame_index):
        nonlocal total_detections, last_annotated_gpu
        total_detections += len(detections)
        last_annotated_gpu = annotated_gpu
        if (frame_index + 1) % 10 == 0:
            log.info("frame=%d  detections_this_frame=%d", frame_index + 1, len(detections))

    pipeline.run(on_frame=on_frame, max_frames=args.num_frames)
    pipeline.close()

    log.info("=" * 60)
    log.info("Frames processed: %d", pipeline.stats.frames_processed)
    log.info("Total detections across all frames: %d", total_detections)
    log.info("Avg FPS: %.1f", pipeline.stats.avg_fps)
    log.info("  decode+preprocess: %.1fms/frame  inference: %.1fms/frame  draw: %.1fms/frame",
              1000 * pipeline.stats.total_decode_s / max(pipeline.stats.frames_processed, 1),
              1000 * pipeline.stats.total_inference_s / max(pipeline.stats.frames_processed, 1),
              1000 * pipeline.stats.total_draw_s / max(pipeline.stats.frames_processed, 1))
    log.info("=" * 60)
    log.info("Result: PASS — full NVDEC zero-copy pipeline runs end-to-end on real GPU hardware.")

    if args.save_sample and last_annotated_gpu is not None:
        import cupy as cp
        annotated_host = cp.asnumpy(last_annotated_gpu)
        cv2.imwrite(args.save_sample, cv2.cvtColor(annotated_host, cv2.COLOR_RGB2BGR))
        log.info("Saved sample annotated frame to %s", args.save_sample)


if __name__ == "__main__":
    main()
