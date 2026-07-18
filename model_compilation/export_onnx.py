from ultralytics import YOLO

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Export to ONNX format
model.export(format="onnx")

print("ONNX Export Successful!")