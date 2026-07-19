from ultralytics import YOLO


class Detector:
    """
    Handles loading the YOLO model
    and performing object detection.
    """

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image_path):
        results = self.model(image_path)

        for result in results:
            names = result.names

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                print(
                    f"Detected: {names[cls_id]} "
                    f"(Confidence: {conf:.2f})"
                )

        return results


if __name__ == "__main__":
    detector = Detector("yolov8n.pt")
    detector.predict("backend/data/sample.jpg")