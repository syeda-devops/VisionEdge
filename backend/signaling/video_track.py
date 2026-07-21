"""Temporary video source for WebRTC playback verification.

This track intentionally supplies only media frames. A later streaming source
can replace the test pattern without changing signaling or peer management.
"""

from __future__ import annotations

import asyncio
from fractions import Fraction

import av
import numpy as np
from aiortc import MediaStreamTrack


class VideoTrack(MediaStreamTrack):
    """Provide a smooth test-pattern video stream to a WebRTC peer."""

    kind = "video"

    def __init__(self, width: int = 640, height: int = 360, frame_rate: int = 30):
        super().__init__()
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self._frame_index = 0
        self._next_frame_time: float | None = None

    async def recv(self) -> av.VideoFrame:
        """Return the next paced video frame."""
        if self.readyState != "live":
            raise asyncio.CancelledError

        loop = asyncio.get_running_loop()
        if self._next_frame_time is None:
            self._next_frame_time = loop.time()
        self._next_frame_time += 1 / self.frame_rate
        await asyncio.sleep(max(0, self._next_frame_time - loop.time()))

        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        progress = (self._frame_index * 4) % self.width
        image[:, :progress, 1] = 96
        image[:, progress:, 2] = 96
        image[24:72, 24 : self.width - 24] = (24, 24, 24)

        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = self._frame_index
        frame.time_base = Fraction(1, self.frame_rate)
        self._frame_index += 1
        return frame
