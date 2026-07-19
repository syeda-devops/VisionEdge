from backend.streaming.frame_provider import FrameProvider

print("Test 1: Valid Image")
provider = FrameProvider("backend/data/sample.jpg")
frame = provider.get_frame()
print(type(frame))
print(frame.shape)

print("\nTest 2: Unsupported Extension")
try:
    provider = FrameProvider("backend/data/sample.mp4")
    provider.get_frame()
except Exception as e:
    print(type(e).__name__, ":", e)