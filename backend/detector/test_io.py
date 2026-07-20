import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("backend/models/yolov8n.onnx")

print("Inputs:")
for i in session.get_inputs():
    print(i.name, i.shape)

print("\nOutputs:")
for o in session.get_outputs():
    print(o.name, o.shape)

dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
outputs = session.run(None, {session.get_inputs()[0].name: dummy})

print("\nOutput shape:", outputs[0].shape)