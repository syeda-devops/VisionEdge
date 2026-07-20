from detector.exporter import ModelExporter

MODEL_PATH = "backend/models/yolov8n.pt"

exporter = ModelExporter(MODEL_PATH)

exporter.export_to_onnx()