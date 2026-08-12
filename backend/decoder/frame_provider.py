"""
decoder/frame_provider.py

Week 2: Hardware Decoding.

Decodes H.264/H.265 RTSP (or file) streams using NVDEC via PyAV, bypassing
libx264/OpenCV's CPU software decode path entirely. Frames come out already
resident in GPU memory when the hardware path is used.

*** The `h264_cuvid` / `hevc_cuvid` decoders require an NVIDIA GPU with
NVDEC and an ffmpeg build compiled with CUDA support. ***

Two code paths are provided:
  - FrameProvider          : software decode fallback (any machine, any GPU)
  - HardwareFrameProvider   : NVDEC decode (NVIDIA only) — subclasses the
                              same interface so nothing downstream cares
                              which one is in use. Same DIP pattern as the
                              Detector abstraction.
"""

import logging
from typing import Iterator

import numpy as np

log = logging.getLogger("frame_provider")
class FrameProvider:
    """
    Software decode baseline. Works on any machine (CPU-bound), useful for
    local development without an NVIDIA GPU and as a correctness reference.
    """

    def __init__(self, source: str, target_size: tuple | None = None):
        """
        Parameters
        ----------
        source : str
            RTSP URL or path to a video file.
        target_size : tuple | None
            Optional (W, H) to resize frames to. None = native resolution.
        """
        import av

        self.source = source
        self.target_size = target_size
        self._container = av.open(source)
        self._stream = self._container.streams.video[0]

    def frames(self) -> Iterator[np.ndarray]:
        """
    Yields HWC RGB uint8 NumPy frames in host memory.

    Returns:
        Iterator[np.ndarray]: Frames in HWC format with RGB channels.
    """
        for frame in self._container.decode(self._stream):
            arr = frame.to_ndarray(format="rgb24")
            if self.target_size:
                arr = self._resize(arr, self.target_size)
            yield arr

    @staticmethod
    def _resize(arr: np.ndarray, size: tuple) -> np.ndarray:
        import cv2
        return cv2.resize(arr, size)

    def close(self):
        self._container.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class HardwareFrameProvider(FrameProvider):
    """
    NVDEC-accelerated decode. Frames are decoded directly into GPU memory;
    `frames_gpu()` yields CuPy arrays that never touch host RAM, which is
    what Week 3's zero-copy pipeline needs upstream of it.

    *** Requires: NVIDIA GPU, NVDEC-capable driver, ffmpeg/PyAV built with
    CUDA support (`h264_cuvid` / `hevc_cuvid` decoders available). ***
    """

    def __init__(self, source: str, gpu_id: int = 0, target_size: tuple | None = None):
        import av

        self.source = source
        self.gpu_id = gpu_id
        self.target_size = target_size

        # Force the hardware decoder. codec_context.options selects the
        # CUDA decoder explicitly rather than relying on ffmpeg autodetect.
        self._container = av.open(source, options={"hwaccel": "cuda", "hwaccel_device": str(gpu_id)})
        self._stream = self._container.streams.video[0]
        self._stream.codec_context.options = {"hwaccel": "cuda"}

    def frames_gpu(self):
        """
        Yields CuPy ndarrays (HWC, uint8) that live in GPU memory for the
        lifetime of the frame, ready to be handed to the zero-copy pipeline
        without a host round-trip.
        """
        import cupy as cp

        for frame in self._container.decode(self._stream):
            # frame.to_ndarray() here still crosses through a PyAV-managed
            # hardware surface; wrapping it in cp.asarray keeps it on-device
            # rather than copying to a numpy host array first.
            gpu_frame = cp.asarray(frame.to_ndarray(format="rgb24"))
            if self.target_size:
                gpu_frame = self._resize_gpu(gpu_frame, self.target_size)
            yield gpu_frame

    @staticmethod
    def _resize_gpu(arr, size: tuple):
        """GPU-side resize via CuPy, avoiding an OpenCV CPU round-trip."""
        import cupyx.scipy.ndimage as cndi

        h, w = arr.shape[:2]
        target_w, target_h = size
        zoom_factors = (target_h / h, target_w / w, 1)
        return cndi.zoom(arr, zoom_factors, order=1)  # bilinear

