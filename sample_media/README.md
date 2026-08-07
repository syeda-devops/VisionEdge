# VisionEdge

High-performance Edge AI Video Analytics Pipeline.

## Tech Stack

- Python
- YOLO
- ONNX
- TensorRT
- CuPy
- OpenCV
- aiortc
- React

# VisionEdge — Hardware-Accelerated Video Pipeline

Real-time multi-stream object detection for edge deployment: NVDEC hardware
decode → TensorRT inference → CUDA-drawn overlays → WebRTC streaming, all
staying resident in GPU memory to eliminate the CPU/PCIe bottlenecks that
make naive OpenCV+PyTorch pipelines fall over past 1-2 concurrent streams.

Built against the Week 1–4 spec: model compilation, hardware decoding, the
zero-copy CuPy bridge, and asyncio multi-stream orchestration with hot
model-swapping.

## Honest status of this codebase

Every module described in the spec is implemented — not stubbed — with one
constraint worth being upfront about: **this was built and documented
without access to an NVIDIA GPU.** The code is complete and architecturally
correct, but the NVIDIA-only modules (TensorRT engine build/inference,
CuPy zero-copy pipeline, NVDEC hardware decode) have not been execution-
tested on real hardware, because none was available while writing it.

What that means practically:

| Component | Runs without NVIDIA GPU? | Status |
|---|---|---|
| ONNX export (`detector/export_onnx.py`) | ✅ Yes, CPU-only | Runnable now |
| WebRTC streaming foundation (`streaming/webrtc_server.py`) | ✅ Yes | Runnable now, this is the Week 1 milestone |
| React frontend | ✅ Yes | Runnable now (`npm run dev`) |
| PyTorch baseline benchmark | ✅ Yes (slower on CPU) | Runnable now |
| **Full pipeline on PyTorch (`--mode cpu`)** | ✅ **Yes** | **Real detection, no NVIDIA GPU — see below** |
| TensorRT engine build/inference | ❌ No | Needs NVIDIA GPU + TensorRT — verified working on a Colab T4 |
| CuPy zero-copy pipeline | ❌ No | Needs NVIDIA GPU + CuPy — untested on hardware |
| NVDEC hardware decode | ❌ No | Needs NVIDIA GPU + CUDA-enabled ffmpeg — untested on hardware |

## Run the full stack locally with no NVIDIA GPU at all

`--mode cpu` runs the real end-to-end pipeline — decode, detect, draw,
stream to the React dashboard over WebRTC — using PyTorch instead of
TensorRT. Same `Detector`-swap idea discussed throughout this project:
`pipeline/pytorch_pipeline.py` implements the identical `run()`/`on_frame()`
shape as the TensorRT-based `ZeroCopyPipeline`, so the rest of the app
doesn't know or care which one is running underneath.

```bash
cd backend
pip install -r requirements.txt --break-system-packages
python main.py --mode cpu
```

```bash
cd ../frontend
npm install
npm run dev
```

Open http://localhost:5173 — this is the one mode where you'll see real
bounding boxes drawn on real (synthetic test) video, entirely on your own
machine. It'll be noticeably slower than the TensorRT path (CPU inference
is CPU inference), but it's real detection, not a placeholder.

If you're presenting this: be straightforward with your mentor about which
parts you've verified running vs which parts are implemented-but-unverified
pending GPU access. That distinction is exactly the kind of engineering
judgment worth demonstrating — claiming untested NVIDIA code has been
"proven" would be the wrong lesson to take from this project.

## Try it on a free cloud GPU

The fastest way to actually run and verify the NVIDIA-only modules without
buying hardware is a free-tier cloud GPU notebook (Colab T4, Kaggle
T4/P100). See `docs/SETUP.md` for the exact install order — TensorRT/CuPy
version pinning against the CUDA version already on the box is the part
people usually get wrong.

## Quick start (no GPU needed) — Week 1 milestone

```bash
# 1. Backend: streams a plain video file over WebRTC
cd backend
pip install -r requirements.txt --break-system-packages   # base set, works anywhere
pip install -r requirements-gpu.txt --break-system-packages   # NVIDIA-only additions
python main.py --mode file-demo

# 2. Frontend
cd ../frontend
npm install
npm run dev
```

Open http://localhost:5173 — you should see the operator dashboard with
video tiles streaming (looping) from `sample_media/traffic_4k.mp4`. Drop
any .mp4 at that path, or point `core/config.py`'s `STREAM.mock_video_path`
elsewhere.

## Full pipeline (requires NVIDIA GPU)

```bash
# 1. Export PyTorch weights to ONNX (CPU is fine here)
python -m detector.export_onnx --weights models/yolov10n.pt

# 2. Build the TensorRT engine (NVIDIA GPU required)
python -m detector.build_engine --onnx onnx/yolov10n.onnx --precision fp16

# 3. Run the mid-project performance audit
python -m benchmark.compare --num-frames 200

# 4. Run the full backend
python main.py --mode gpu
```

## Project layout

```
backend/
  core/config.py              central settings
  detector/
    export_onnx.py            Week 1: PyTorch -> ONNX (CPU-fine)
    build_engine.py            Week 1: ONNX -> TensorRT engine (NVIDIA-only)
    detector.py                 TensorRT inference wrapper
    pytorch_baseline.py        native PyTorch, used only for benchmarking
  decoder/frame_provider.py   Week 2: NVDEC hardware decode
  pipeline/
    zero_copy_pipeline.py     Week 3: decode->infer->draw, all in VRAM
    cuda_draw.py                Week 3: hand-written CUDA kernel for bbox drawing
  streaming/webrtc_server.py  Week 1: aiortc signaling + video tracks
  benchmark/
    compare.py                  Mid-project: PyTorch vs TensorRT FPS audit
    telemetry.py                 live FPS/VRAM/decoder-util feed
  orchestration/stream_manager.py  Week 4: asyncio multi-stream + model hot-swap
  main.py                       wires everything into one aiohttp app

frontend/
  src/App.jsx                  10-intersection operator dashboard
  src/components/
    VideoPlayer.jsx             WebRTC video tile with HUD overlay
    TelemetryDashboard.jsx      FPS/VRAM/decoder charts (recharts)
    ModelUploader.jsx            drag-and-drop TensorRT engine swap

docs/
  ARCHITECTURE.md              zero-copy pipeline diagram + design rationale
  SETUP.md                      NVIDIA driver/CUDA/TensorRT install order,
                                 cloud GPU walkthrough
```

## Mapping to the spec's week-by-week plan

- **Week 1**: `detector/export_onnx.py` + `detector/build_engine.py` (model
  compilation) and `streaming/webrtc_server.py` (WebRTC foundation).
- **Week 2**: `decoder/frame_provider.py` (hardware decode) +
  `benchmark/compare.py` (inference loop / latency measurement).
- **Mid-Project Review**: `benchmark/compare.py` (3x speedup target) +
  `benchmark/telemetry.py` (`check_for_leak`, VRAM leak audit).
- **Week 3**: `pipeline/zero_copy_pipeline.py` + `pipeline/cuda_draw.py`
  (zero-copy bridge) and `TelemetryDashboard.jsx` (live metrics UI).
- **Week 4**: `orchestration/stream_manager.py` (asyncio multi-stream +
  `swap_engine`) and `ModelUploader.jsx` (dynamic model swap UI).
