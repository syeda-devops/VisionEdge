import cv2


from core.config import MODEL_PATH
from detector.detector import Detector
from detector.onnx_validator import ONNXValidator


frame = cv2.imread("backend/data/sample.jpg")

detector = Detector(MODEL_PATH)

validator = ONNXValidator(
    "backend/models/yolov8n.onnx"
)

result = validator.validate(
    detector,
    frame
)

print("Validation Passed:", result)