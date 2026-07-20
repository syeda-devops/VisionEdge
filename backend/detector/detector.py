from ultralytics import YOLO


class Detector:
    """
    Handles loading the AI model
    and performing object detection.
    """

    def __init__(self, model_path: str):
        """
        Load the YOLO model.

        Parameters
        ----------
        model_path : str
            Path to the trained YOLO model.
        """

        self.model = YOLO(model_path)

    def predict(self, frame):
        """
        Perform inference.

        Parameters
        ----------
        frame : numpy.ndarray

        Returns
        -------
        list
        """

        results = self.model(frame)

        detections = []

        for box in results[0].boxes:
            detections.append({
                "bbox": box.xyxy.cpu().numpy().tolist()[0],
                "confidence": float(box.conf),
                "class_id": int(box.cls)
            })

        return detections