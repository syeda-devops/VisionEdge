import cv2
import logging

from backend.core.config import STREAM, MODEL
from backend.pipeline.pytorch_pipeline import PyTorchPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

log = logging.getLogger("VisionEdge")


def main():

    log.info("=" * 60)
    log.info("VisionEdge - Week 2 Demo")
    log.info("=" * 60)

    pipeline = PyTorchPipeline(
        source=STREAM.mock_video_path,
        weights_path=MODEL.pytorch_weights,
    )

    def on_frame(frame, detections, frame_index):

        display = cv2.resize(frame, (1280, 720)) 
        cv2.imshow("VisionEdge Detection", display)

        if frame_index % 30 == 0:
            object_count = sum(
                len(r.boxes) if r.boxes is not None else 0
                for r in detections
            )

            log.info(
                "Frame %d | Objects: %d | Average FPS: %.2f",
                frame_index,
                object_count,
                pipeline.stats.avg_fps,
            )

        key = cv2.waitKey(1)

        if key == ord("q"):
            pipeline.stop()

    try:

        pipeline.run(
            on_frame=on_frame,
            loop=False,
        )

    finally:

        pipeline.close()

        cv2.destroyAllWindows()

        log.info("Pipeline stopped.")


if __name__ == "__main__":
    main()