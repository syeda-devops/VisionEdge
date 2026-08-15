"""
core/config.py

Single source of truth for paths, model settings, and pipeline tuning.
Nothing in this file touches CUDA, TensorRT, or PyAV directly — it's pure
configuration so every other module can import it without pulling in
GPU dependencies.
"""

from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
MODELS_DIR = BASE_DIR / "models"
ENGINES_DIR = BASE_DIR / "engines"
ONNX_DIR = BASE_DIR / "onnx"

for _dir in (MODELS_DIR, ENGINES_DIR, ONNX_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Everything about the AI model itself."""
    pytorch_weights: str = str(MODELS_DIR / "yolov10n.pt")
    onnx_path: str = str(ONNX_DIR / "yolov10n.onnx")
    engine_path: str = str(ENGINES_DIR / "yolov10n_fp16.engine")
    input_size: tuple = (640, 640)          # (H, W) fed to the network
    num_classes: int = 80                    # COCO by default
    confidence_threshold: float = 0.45
    nms_iou_threshold: float = 0.5
    precision: str = "fp16"                  # "fp32" | "fp16" | "int8"


@dataclass
class StreamConfig:
    """Per-camera-stream settings."""
    rtsp_url: str = "rtsp://127.0.0.1:8554/mock_traffic"
    mock_video_path: str = str(BASE_DIR / "sample_media" / "traffic_4k.mp4")
    target_fps: int = 60
    gpu_id: int = 0                          # which physical GPU decodes this stream
    max_width: int = 3840                    # 4K
    max_height: int = 2160


@dataclass
class WebRTCConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    ice_servers: list = field(default_factory=lambda: ["stun:stun.l.google.com:19302"])
    bitrate_kbps: int = 4000


@dataclass
class OrchestratorConfig:
    max_concurrent_streams: int = 10
    vram_budget_mb: int = 20000              # leave headroom on a 24GB card
    watchdog_interval_s: float = 5.0


MODEL = ModelConfig()
STREAM = StreamConfig()
WEBRTC = WebRTCConfig()
ORCHESTRATOR = OrchestratorConfig()
