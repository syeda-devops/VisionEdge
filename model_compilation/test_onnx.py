import onnx

model = onnx.load("yolov8n.onnx")

onnx.checker.check_model(model)

print("ONNX model is valid!")