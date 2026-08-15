import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("detector")


@dataclass
class Detection:
    """One detected object, already in pixel coordinates of the input frame."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class Detector:
    """
    Loads a TensorRT engine once and performs GPU-resident inference.

    Parameters
    ----------
    engine_path : str
        Path to a .engine file produced by build_engine.py.
    input_size : tuple
        (H, W) the engine was compiled for. Must match exactly — TensorRT
        engines built with static shapes will not accept other sizes.
    confidence_threshold, nms_iou_threshold : float
        Post-processing thresholds applied after the raw network output.
    """

    def __init__(
        self,
        engine_path: str,
        input_size: tuple = (640, 640),
        confidence_threshold: float = 0.45,
        nms_iou_threshold: float = 0.5,
    ):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  (initializes the CUDA context)

        self._trt = trt
        self._cuda = cuda
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold

        log.info("Loading TensorRT engine from %s", engine_path)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine at {engine_path}")

        self.context = self.engine.create_execution_context()
        self._allocate_buffers()
        self.stream = cuda.Stream()

        log.info("Detector ready. Input size=%s", input_size)

    def _allocate_buffers(self):
        """Pre-allocate device buffers for every engine binding, once, at load time."""
        self.bindings = []
        self.device_buffers = {}
        self.output_shapes = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.context.get_tensor_shape(name)
            dtype = self._trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize

            device_mem = self._cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))
            self.device_buffers[name] = device_mem

            if self.engine.get_tensor_mode(name) == self._trt.TensorIOMode.OUTPUT:
                self.output_shapes[name] = (shape, dtype)

    def predict(self, frame_gpu) -> list[Detection]:
        """
        Run inference on a single preprocessed frame that already lives on GPU.

        Parameters
        ----------
        frame_gpu : cupy.ndarray
            Shape (3, H, W), float32, normalized to [0, 1], already resized
            to self.input_size. Preprocessing (resize/normalize) happens in
            the pipeline module, not here — same single-responsibility rule
            as the rest of this project.

        Returns
        -------
        list[Detection]
            Post-NMS detections in pixel coordinates of the *input* frame.
        """
        import cupy as cp  # deferred import — NVIDIA-only dependency

        input_name = self.engine.get_tensor_name(0)
        self.context.set_tensor_address(input_name, int(frame_gpu.data.ptr))

        for name, device_mem in self.device_buffers.items():
            if name != input_name:
                self.context.set_tensor_address(name, int(device_mem))

        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()

        raw_outputs = {}
        for name, (shape, dtype) in self.output_shapes.items():
            host_out = cp.ndarray(shape, dtype=dtype, memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(int(self.device_buffers[name]),
                                       int(np.prod(shape)) * np.dtype(dtype).itemsize, self),
                0,
            ))
            raw_outputs[name] = host_out

        return self._postprocess(raw_outputs)

    def _postprocess(self, raw_outputs: dict) -> list[Detection]:
        """
        Decode raw network output into Detection objects: threshold by
        confidence, then apply NMS. Kept deliberately simple/CPU-side here
        for Week 1 correctness; Week 3 moves this onto CUDA kernels so
        boxes never leave VRAM (see pipeline/cuda_draw.py).
        """
        import cupy as cp

        detections: list[Detection] = []
        output = next(iter(raw_outputs.values()))
        output_host = cp.asnumpy(output)  # deliberate host copy for Week 1 simplicity/debugging

        # YOLOv10 end-to-end head outputs (batch, num_dets, 6) as
        # [x1, y1, x2, y2, confidence, class_id] with NMS already applied
        # inside the graph (that's the "v10" difference vs v8/v9). If your
        # exported model does NOT have the fused NMS head, add an explicit
        # NMS pass here before returning.
        for det in output_host[0]:
            x1, y1, x2, y2, conf, cls = det[:6]
            if conf < self.confidence_threshold:
                continue
            detections.append(Detection(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                confidence=float(conf), class_id=int(cls),
            ))

        return detections

    def __del__(self):
        # Explicit cleanup ordering matters with CUDA — release device
        # buffers before the context they belong to goes out of scope.
        for mem in getattr(self, "device_buffers", {}).values():
            try:
                mem.free()
            except Exception:
                pass
