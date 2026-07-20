import cv2
import numpy as np
import onnxruntime as ort

from ultralytics.data.augment import LetterBox

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

        self.letterbox = LetterBox(
            new_shape=(640, 640),
            auto=False,
            scale_fill=False,
            scaleup=True,
            stride=32
        )

        self.benchmark = Benchmark()

    def _prepare_input(self, frame):
        """
        Convert OpenCV image to ONNX input.
        """
        
        image = self.letterbox(image=frame)

        
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype(np.float32)

        image /= 255.0

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = np.expand_dims(
            image,
            axis=0
        )

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
        threshold=0.25
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
        onnx_results
    ):
        """
        Compare PyTorch and ONNX outputs.
        """

        validation_passed = True

        matches = self._match_by_class(
            pytorch_results,
            onnx_results
        )

        if len(matches) != len(onnx_results):

            logger.warning(
                f"Matched {len(matches)} of "
                f"{len(onnx_results)} ONNX detections."
            )

            validation_passed = False

        for pytorch, onnx, iou in matches:

            if iou < 0.50:

                logger.warning(
                    f"Low IoU: {iou:.3f}"
                )

                validation_passed = False
                continue

            if not self._compare_confidence(
                pytorch,
                onnx
            ):

                logger.warning(
                    "Confidence mismatch."
                )

                validation_passed = False

            if not self._compare_bbox(
                pytorch,
                onnx
            ):

                logger.warning(
                    "Bounding box mismatch."
                )

                validation_passed = False

        return validation_passed
    def _match_by_class(
        self,
        pytorch_results,
        onnx_results
    ):
        """
        Match detections using class IDs and
        highest IoU.
        """

        matches = []

        remaining = pytorch_results.copy()

        for onnx in onnx_results:

            best_match = None
            best_iou = 0.0

            for pytorch in remaining:

                if (
                    pytorch["class_id"]
                    !=
                    onnx["class_id"]
                ):
                    continue

                iou = self._calculate_iou(
                    pytorch["bbox"],
                    onnx["bbox"]
                )



                if iou > best_iou:

                    best_iou = iou
                    best_match = pytorch

            if best_match is not None :
                
                matches.append(
                    (
                        best_match,
                        onnx,
                        best_iou
                    )
                )

                remaining.remove(
                    best_match
                )

        return matches
    def _compare_confidence(
        self,
        pytorch,
        onnx,
        tolerance=0.05
    ):
        """
        Compare confidence scores.
        """

        return (
            abs(
                pytorch["confidence"]
                -
                onnx["confidence"]
            )
            <= tolerance
        )

    def _compare_bbox(
        self,
        pytorch,
        onnx,
        tolerance=2.0
    ):
        """
        Compare bounding boxes.
        """

        for p, o in zip(
            pytorch["bbox"],
            onnx["bbox"]
        ):

            if abs(p - o) > tolerance:
                logger.warning(
                    f"Bounding box mismatch.\n"
                    f"PyTorch: {pytorch['bbox']}\n"
                    f"ONNX: {onnx['bbox']}"
                )
  

    
                return False

        return True 

    def _calculate_iou(
        self,
        box1,
        box2
    ):
        """
        Calculate Intersection over Union (IoU)
        between two bounding boxes.
        """

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (
            (box1[2] - box1[0])
            *
            (box1[3] - box1[1])
        )

        area2 = (
            (box2[2] - box2[0])
            *
            (box2[3] - box2[1])
        )

        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union  

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
        
        
        

        image = self._prepare_input(frame)

        outputs, onnx_time = \
            self.benchmark.measure(
                self.session.run,
                None,
                {
                    self.input_name: image
                }
            )
        
        
        for i, output in enumerate(outputs):
            print(f"Output {i}: shape={output.shape}")

        onnx_results = \
            self._extract_detections(
                outputs
            )
        
        print("PyTorch before filter:", len(pytorch_results))
        print("ONNX before filter:", len(onnx_results))
        pytorch_results = self._filter_detections(pytorch_results)
        onnx_results = self._filter_detections(
            onnx_results
        )

    

        passed = self.compare_outputs(
            pytorch_results,
            onnx_results
        )

        if passed:

            logger.info(
                "ONNX validation PASSED."
            )

        else:

            logger.warning(
                "ONNX validation FAILED."
            )

        print([d["class_id"] for d in pytorch_results])

        print([d["class_id"] for d in onnx_results])

        print("PyTorch after filter:", len(pytorch_results))
        print("ONNX after filter:", len(onnx_results))

        logger.info(
            "Validation Summary"
        )

        logger.info(
            f"PyTorch detections : {len(pytorch_results)}"
        )

        logger.info(
            f"ONNX detections    : {len(onnx_results)}"
        )


        logger.info(
            f"PyTorch : {pytorch_time:.4f}s | "
            f"ONNX : {onnx_time:.4f}s"
        )

        return passed