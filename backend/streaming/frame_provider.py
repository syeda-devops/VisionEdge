import cv2


class FrameProvider:
    def __init__(self, image_path):
        self.image_path = image_path

    def get_frame(self):
        frame = cv2.imread(self.image_path)

        if frame is None:
            raise FileNotFoundError(
                f"Unable to load image: {self.image_path}"
            )

        return frame