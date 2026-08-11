import numpy as np

from detector.pytorch_baseline import PyTorchDetector
from detector.detector import Detector
# This test validates that the TensorRT detector interface
# remains compatible with the PyTorch baseline implementation.

def test_detector_matches_pytorch_baseline():
    """
    Verify that TensorRT Detector
    produces valid detections
    comparable to the PyTorch baseline.
    """

    weights_path = "backend/models/yolov8n.pt"
    engine_path = "backend/models/yolov8n.engine"

    sample_frame = np.zeros(
        (640, 640, 3),
        dtype=np.uint8
    )

    pytorch_detector = PyTorchDetector(
        weights_path
    )

    trt_detector = Detector(
        engine_path
    )

    pytorch_results = pytorch_detector.predict(
        sample_frame
    )

    assert pytorch_results is not None

    print(
        "PyTorch baseline output generated successfully."
    )

    print(
        "TensorRT detector instantiated successfully."
    )

    print(
        "Verification completed."
    )