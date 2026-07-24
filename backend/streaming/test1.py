from frame_provider import FrameProvider

VIDEO_PATH = r"C:\Users\LOKESH\Desktop\VisionEdge\backend\data\sample.mp4"

provider = FrameProvider(VIDEO_PATH)

try:
    print("Testing FrameProvider...\n")

    for i, frame in enumerate(provider.frames()):
        print(f"Frame {i + 1}")
        print("Frame Type :", type(frame))
        print("Frame Shape:", frame.shape)
        break

finally:
    provider.close()
    print("FrameProvider closed successfully.")