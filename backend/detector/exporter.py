import argparse
import logging
import shutil
from pathlib import Path

from core.config import MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("export_onnx")


def export_to_onnx(weights_path: str, output_path: str, imgsz: tuple, opset: int = 17) -> str:
    """
    Export a PyTorch YOLO checkpoint to ONNX.

    Parameters
    ----------
    weights_path : str
        Path to the .pt file (PyTorch weights).
    output_path : str
        Where to write the .onnx file.
    imgsz : tuple
        (height, width) the network expects.
    opset : int
        ONNX opset version. 17 is a safe default for current TensorRT parsers.

    Returns
    -------
    str
        Path to the exported ONNX file.
    """
    from ultralytics import YOLO  # deferred import: keeps this module importable
                                    # even in environments without ultralytics installed

    log.info("Loading PyTorch weights from %s", weights_path)
    model = YOLO(weights_path)

    log.info("Exporting to ONNX (imgsz=%s, opset=%s)...", imgsz, opset)
    exported_path = model.export(
        format="onnx",
        imgsz=list(imgsz),
        opset=opset,
        dynamic=False,      # static shapes = TensorRT can optimize harder
        simplify=True,      # runs onnx-simplifier to fold constants / clean the graph
    )

    # ultralytics writes the ONNX file next to the source .pt weights by
    # default — it does NOT respect an arbitrary destination path passed
    # here. Move it to where the caller actually asked for it if different.
    exported_path = Path(exported_path)
    output_path = Path(output_path)
    if exported_path.resolve() != output_path.resolve():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported_path), str(output_path))
        log.info("Moved ONNX from %s -> %s", exported_path, output_path)

    log.info("ONNX model written to %s", output_path)
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Export YOLO PyTorch weights to ONNX")
    parser.add_argument("--weights", default=MODEL.pytorch_weights)
    parser.add_argument("--output", default=MODEL.onnx_path)
    parser.add_argument("--imgsz", type=int, nargs=2, default=list(MODEL.input_size))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_to_onnx(args.weights, args.output, tuple(args.imgsz), args.opset)


if __name__ == "__main__":
    main()
