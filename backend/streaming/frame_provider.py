import os
import cv2


class FrameProvider:

    SUPPORTED_EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    )

    def __init__(self, image_path: str):

        self.image_path = image_path

    def _validate_path(self):

        if not os.path.exists(self.image_path):
            raise FileNotFoundError(
                f"File not found: {self.image_path}"
            )

    def _validate_extension(self):

        _, extension = os.path.splitext(self.image_path)

        if extension.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: {extension}"
            )

    def get_frame(self):

        self._validate_path()

        self._validate_extension()

        frame = cv2.imread(self.image_path)

        if frame is None:
            raise RuntimeError(
                "Failed to decode image."
            )

        return frame

    def get_frame_info(self):

        frame = self.get_frame()

        height, width = frame.shape[:2]

        channels = frame.shape[2]

        return {
            "width": width,
            "height": height,
            "channels": channels
        }
