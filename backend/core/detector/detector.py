from ultralytics import YOLO


def detect_objects(image_path):
    model = YOLO("yolov8n.pt")

    results = model(image_path)

    for result in results:
        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0])
            print("Detected:", names[cls_id])

    return results


if __name__ == "__main__":
    detect_objects("backend/data/sample.jpg")