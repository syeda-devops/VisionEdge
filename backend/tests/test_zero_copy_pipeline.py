import sys
from pathlib import Path

# Add the backend directory to Python's search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.zero_copy_pipeline import ZeroCopyPipeline

VIDEO_PATH = "../sample_media/traffic_4k.mp4"

count = 0

def on_frame(frame, detections, frame_no):
    global count
    count += 1
    print(f"Frame {frame_no}")
    print("Shape:", frame.shape)
    print("Detections:", len(detections))

pipeline = ZeroCopyPipeline(VIDEO_PATH)

try:
    pipeline.run(
        on_frame=on_frame,
        max_frames=5
    )
    print("Pipeline Test Passed")
finally:
    pipeline.close()