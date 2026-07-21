from backend.detector.tensorrt_builder import TensorRTBuilder


builder = TensorRTBuilder()

success = builder.build_engine(
    "backend/models/yolov8n.onnx",
    "backend/models/yolov8n.engine"
)

print(
    "Engine Built:",
    success
)