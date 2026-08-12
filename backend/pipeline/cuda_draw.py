"""
pipeline/cuda_draw.py
# GPU-side annotation module.
# Uses a custom CUDA kernel to draw bounding boxes directly in VRAM,
# avoiding CPU memory transfers and improving pipeline efficiency.
Week 3: draw bounding boxes using a raw CUDA kernel via CuPy's RawKernel,
so annotation happens in VRAM without round-tripping to OpenCV/CPU
(cv2.rectangle would require pulling the frame back to host memory first).
# NOTE:
# This module keeps bounding-box rendering entirely on the GPU.
# Avoiding CPU round-trips reduces latency and supports real-time inference.
*** Requires an NVIDIA GPU (CuPy + a working CUDA toolkit for kernel JIT). ***
"""
# CUDA kernel executes fully on GPU memory to avoid CPU round-trips.
from detector.detector import Detection

# Draws axis-aligned box outlines directly into an HWC uint8 RGB image
# sitting in device memory. One CUDA thread per pixel; each thread checks
# whether it lies on any box's border and, if so, writes the box color.
# Review note:
# GPU rendering is kept separate from inference logic
# to make the pipeline easier to benchmark and maintain.
# This is intentionally simple (outline only, fixed line thickness) —
# the point of the exercise is demonstrating a hand-written CUDA kernel
# in the pipeline, not building a full rendering library.
_KERNEL_SOURCE = r"""
extern "C" __global__
void draw_boxes(
    unsigned char* image,      // HWC uint8, contiguous
    const int height,
    const int width,
    const float* boxes,        // N x 4: x1, y1, x2, y2
    const unsigned char* colors, // N x 3: r, g, b
    const int num_boxes,
    const int thickness
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    for (int b = 0; b < num_boxes; b++) {
        float x1 = boxes[b * 4 + 0];
        float y1 = boxes[b * 4 + 1];
        float x2 = boxes[b * 4 + 2];
        float y2 = boxes[b * 4 + 3];

        bool on_vertical_edge =
            (fabsf(x - x1) <= thickness || fabsf(x - x2) <= thickness) &&
            (y >= y1 && y <= y2);
        bool on_horizontal_edge =
            (fabsf(y - y1) <= thickness || fabsf(y - y2) <= thickness) &&
            (x >= x1 && x <= x2);

        if (on_vertical_edge || on_horizontal_edge) {
            int idx = (y * width + x) * 3;
            image[idx + 0] = colors[b * 3 + 0];
            image[idx + 1] = colors[b * 3 + 1];
            image[idx + 2] = colors[b * 3 + 2];
        }
    }
}
"""

# Lazily initialized on first use — keeps this module importable on
# machines without CuPy/CUDA available (matches the deferred-import
# pattern used throughout the rest of this codebase, e.g. detector.py).
_kernel = None
_palette = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        import cupy as cp
        _kernel = cp.RawKernel(_KERNEL_SOURCE, "draw_boxes")
    return _kernel


def _get_palette():
    global _palette
    if _palette is None:
        import cupy as cp
        # Fixed palette so box color is stable per class across frames,
        # rather than random per call — makes a live dashboard/video far
        # easier to read.
        _palette = cp.array([
            [255, 99, 71], [60, 179, 113], [65, 105, 225], [255, 215, 0],
            [218, 112, 214], [0, 206, 209], [255, 140, 0], [154, 205, 50],
        ], dtype=cp.uint8)
    return _palette


def draw_boxes_gpu(
    frame_gpu,
    detections: list[Detection],
    model_input_size: tuple,
    thickness: int = 2
):
    """
    Draws detection boxes onto frame_gpu in place and returns it.

    Parameters
    ----------
    frame_gpu : cupy.ndarray
        HWC uint8 RGB frame, full resolution (the raw decoded frame, not
        the resized model-input tensor).
    detections : list[Detection]
        Boxes in *model input* coordinate space (model_input_size).
        Rescaled to frame_gpu's actual resolution before drawing.
    model_input_size : tuple
        (H, W) the detector was run at, needed to rescale box coordinates
        back up to the frame's native resolution.
    thickness : int
        Line thickness in pixels.
    """
    if not detections:
        return frame_gpu

    import cupy as cp

    kernel = _get_kernel()
    palette = _get_palette()

    frame_h, frame_w = frame_gpu.shape[:2]
    model_h, model_w = model_input_size

    scale_x = frame_w / model_w
    scale_y = frame_h / model_h

    boxes = cp.array([
        [d.x1 * scale_x, d.y1 * scale_y, d.x2 * scale_x, d.y2 * scale_y]
        for d in detections
    ], dtype=cp.float32)

    colors = cp.array([
        palette[d.class_id % len(palette)].get()
        for d in detections
    ], dtype=cp.uint8)

    frame_gpu = cp.ascontiguousarray(frame_gpu)
    # CUDA grid configuration:
# Each thread processes a single pixel,
# allowing box rendering directly in GPU memory.
    threads_per_block = (16, 16)

    blocks = (
        (frame_w + threads_per_block[0] - 1) // threads_per_block[0],
        (frame_h + threads_per_block[1] - 1) // threads_per_block[1],
    )

    kernel(
        blocks,
        threads_per_block,
        (
            frame_gpu,
            frame_h,
            frame_w,
            boxes,
            colors,
            len(detections),
            thickness,
        ),
    )

    return frame_gpu