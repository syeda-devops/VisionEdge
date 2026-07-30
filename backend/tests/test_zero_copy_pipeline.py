from pipeline.zero_copy_pipeline import ZeroCopyPipeline

VIDEO_PATH = "../sample_media/traffic_4k.mp4"   # Change if your video has a different name

def on_frame(frame, detections, frame_no):
    print(f"Frame: {frame_no}")
    print("Detections:", len(detections))

pipeline = ZeroCopyPipeline(VIDEO_PATH)

try:
    pipeline.run(
        on_frame=on_frame,
        max_frames=5
    )
    print("ZeroCopyPipeline test passed.")
finally:
    pipeline.close()