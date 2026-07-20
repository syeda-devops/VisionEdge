from backend.core.config import MODEL_PATH, IMAGE_PATH
from backend.detector.detector import Detector
from backend.streaming.frame_provider import FrameProvider


def main():

    provider = FrameProvider(IMAGE_PATH)

    detector = Detector(MODEL_PATH)

    frame = provider.get_frame()

    detections = detector.predict(frame)


    print(detections)


if __name__ == "__main__":
    main()