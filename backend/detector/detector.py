def predict(self, frame):

    results = self.model(frame)

    detections = []

    for box in results[0].boxes:

        detections.append({
            "bbox": box.xyxy.cpu().numpy().tolist()[0],
            "confidence": float(box.conf),
            "class_id": int(box.cls)
        })

    return detections