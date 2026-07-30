from pipeline.zero_copy_pipeline import ZeroCopyPipeline

VIDEO_PATH = "sample_media/sample.mp4"   # change if your video is elsewhere

count = 0

def on_frame(frame, detections, frame_no):
    global count
    count += 1
    print(f"Frame {frame_no}")
    print("Shape :", frame.shape)
    print("Detections :", len(detections))

    if frame_no == 4:
        pipeline.close()

pipeline = ZeroCopyPipeline(VIDEO_PATH)

try:
    pipeline.run(
        on_frame=on_frame,
        max_frames=5
    )
    print("Pipeline Test Passed")
finally:
    pipeline.close()