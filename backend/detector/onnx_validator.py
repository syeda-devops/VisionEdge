import cv2
import numpy as np
import onnxruntime as ort

from backend.benchmark.benchmark import Benchmark 
from backend.core.logger import logger 


class ONNXValidator:
    """
    Validates that the exported ONNX model produces
    detections comparable to the PyTorch detector.
    """

    def __init__(self, model_path: str):

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        self.benchmark = Benchmark()

    def _prepare_input(self, frame):
        """
        Convert OpenCV image to ONNX input.
        """

        image = cv2.resize(frame, (640, 640))

        image = image.astype(np.float32)

        image /= 255.0

        image = np.transpose(image, (2, 0, 1))

        image = np.expand_dims(image, axis=0)

        return image

    def _extract_detections(self, outputs):
        """
        Convert ONNX output into the project's
        standard detection format.

        Returns
        -------
        list[dict]
        """

        detections = []

        predictions = outputs[0][0]

        for prediction in predictions:

            x1, y1, x2, y2, confidence, class_id = prediction

           

            detections.append(
                {
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    ],
                    "confidence": float(confidence),
                    "class_id": int(class_id)
                }
            )

        return detections
    
    def _filter_detections(
        self,
        detections,
        threshold=0.5
    ):
        """
        Remove detections below the confidence threshold.
        """

        filtered = []

        for detection in detections:
            if detection["confidence"] >= threshold:
                filtered.append(detection)

        return filtered

    def compare_outputs(
        self,
        pytorch_results,
        onnx_results,
        confidence_tolerance=0.05,
        bbox_tolerance=2.0
    ):
        """
        Compare PyTorch and ONNX detections.
        """

        if len(pytorch_results) != len(onnx_results):

            logger.warning(
                f"Detection count mismatch "
                f"(PyTorch={len(pytorch_results)}, "
                f"ONNX={len(onnx_results)})"
            )

            return False

        for pytorch, onnx in zip(
            pytorch_results,
            onnx_results
        ):

            if pytorch["class_id"] != onnx["class_id"]:

                logger.warning(
                    "Class ID mismatch."
                )

                return False

            if abs(
                pytorch["confidence"]
                -
                onnx["confidence"]
            ) > confidence_tolerance:

                logger.warning(
                    "Confidence mismatch."
                )

                return False

            for p, o in zip(
                pytorch["bbox"],
                onnx["bbox"]
            ):

                if abs(p - o) > bbox_tolerance:

                    logger.warning(
                        "Bounding box mismatch."
                    )

                    return False

        logger.info(
            "PyTorch and ONNX outputs match."
        )

        return True

    def validate(
        self,
        detector,
        frame
    ):
        """
        Run validation using the same frame.
        """

        logger.info(
            "Starting ONNX validation..."
        )

        pytorch_results, pytorch_time = \
            self.benchmark.measure(
                detector.predict,
                frame
            )
        
        pytorch_results = self._filter_detections(
            pytorch_results
        )

        image = self._prepare_input(frame)

        outputs, onnx_time = \
            self.benchmark.measure(
                self.session.run,
                None,
                {
                    self.input_name: image
                }
            )

        onnx_results = \
            self._extract_detections(
                outputs
            )
        

        onnx_results = self._filter_detections(
            onnx_results
        )

        passed = self.compare_outputs(
            pytorch_results,
            onnx_results
        )

        logger.info(
            f"PyTorch : {pytorch_time:.4f}s | "
            f"ONNX : {onnx_time:.4f}s"
        )

        return passed