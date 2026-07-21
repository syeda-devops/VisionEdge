from backend.core.config import MODEL_PATH, IMAGE_PATH
from backend.detector.detector import Detector
from backend.streaming.frame_provider import FrameProvider
from backend.benchmark.benchmark import Benchmark

def main():

    provider = FrameProvider(IMAGE_PATH)

    detector = Detector(MODEL_PATH)

    benchmark = Benchmark()

    frame = provider.get_frame()

    detections, elapsed = benchmark.measure(
        detector.predict,
        frame
    )

    benchmark.report()

    print("\nDetection Results")

    for detection in detections:
        print(detection)


if __name__ == "__main__":
    main()