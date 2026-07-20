import os
import sys

# Add the VisionEdge project root to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
)

from backend.streaming.frame_provider import FrameProvider
from backend.detector.detector import Detector

IMAGE_PATH = r"C:\Users\LOKESH\Desktop\VisionEdge\backend\data\sample.jpg"
MODEL_PATH = r"C:\Users\LOKESH\Desktop\VisionEdge\backend\models\yolov8n.pt"

# -------- FrameProvider Test --------
provider = FrameProvider(IMAGE_PATH)

frame = provider.get_frame()

print("Frame Type:", type(frame))
print("Frame Shape:", frame.shape)

info = provider.get_frame_info()
print("Frame Info:", info)

# -------- Integration Test --------
detector = Detector(MODEL_PATH)

results = detector.predict(frame)

print("\nDetections:")
print(results)