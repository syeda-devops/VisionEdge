import cv2


from backend.core.config import MODEL_PATH
from backend.detector.detector import Detector
from backend.detector.onnx_validator import ONNXValidator


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