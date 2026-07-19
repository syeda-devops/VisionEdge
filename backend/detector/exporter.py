from ultralytics import YOLO


class ModelExporter:
    """
    Handles exporting the YOLO model
    to ONNX format.
    """

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def export_to_onnx(self):
        """
        Export YOLO model to ONNX format.
        """
        self.model.export(
            format="onnx",
            opset=12,
            imgsz=640,
            simplify=True,
            dynamic=False,
            nms=True


        )

        print("ONNX Export Successful!")

        print("ONNX Export Successful!")