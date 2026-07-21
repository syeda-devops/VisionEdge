from backend.streaming.frame_provider import FrameProvider
from backend.core.config import IMAGE_PATH

provider = FrameProvider(IMAGE_PATH)

frame = provider.get_frame()

print(type(frame))
print(frame.shape)