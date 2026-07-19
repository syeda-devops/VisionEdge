from backend.streaming.frame_provider import FrameProvider

provider = FrameProvider("backend/data/sample.jpg")

frame = provider.get_frame()

print(type(frame))
print(frame.shape)