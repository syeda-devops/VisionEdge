import argparse
import logging

from core.config import MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_engine")


def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    max_workspace_mb: int = 4096,
    calibration_data_dir: str | None = None,
) -> str:
    """
    Compile an ONNX model into a serialized TensorRT engine.

    Parameters
    ----------
    onnx_path : str
        Path to the .onnx file produced by export_onnx.py.
    engine_path : str
        Output path for the serialized .engine file.
    precision : str
        "fp32", "fp16", or "int8". fp16 is the standard tradeoff for
        real-time detection: ~2x throughput vs fp32 with negligible mAP loss.
    max_workspace_mb : int
        Scratch memory TensorRT is allowed to use while searching for the
        fastest kernel implementations during the build (build-time only,
        not needed at inference time).
    calibration_data_dir : str | None
        Directory of representative images, required only if precision="int8".

    Returns
    -------
    str
        Path to the written engine file.
    """
    import tensorrt as trt  # deferred import — this module is NVIDIA-only

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    # TensorRT <10 required explicitly passing the EXPLICIT_BATCH flag for
    # ONNX-imported networks. TensorRT 10+ kept the NetworkDefinitionCreationFlag
    # class around but removed the EXPLICIT_BATCH member specifically (explicit
    # batch is now the only supported mode, so the flag is meaningless) —
    # checking for the class alone isn't enough, check the member itself.
    has_explicit_batch_flag = (
        hasattr(trt, "NetworkDefinitionCreationFlag")
        and hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH")
    )
    if has_explicit_batch_flag:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
    else:
        network = builder.create_network()  # TensorRT 10+: no flags needed/accepted
    parser = trt.OnnxParser(network, logger)

    log.info("Parsing ONNX file: %s", onnx_path)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error("ONNX parse error: %s", parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX model at {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_mb * (1 << 20))

    if precision == "fp16":
        # platform_has_fast_fp16 was a purely informational Builder property
        # in TensorRT <10 and was removed in 10+. Setting the FP16 flag
        # below works regardless of whether this check is available, so
        # treat it as optional diagnostics, not a hard requirement.
        if hasattr(builder, "platform_has_fast_fp16") and not builder.platform_has_fast_fp16:
            log.warning("Platform reports no fast FP16 support — build will proceed, "
                        "but throughput gains may be smaller than expected.")
        config.set_flag(trt.BuilderFlag.FP16)

    elif precision == "int8":
        if hasattr(builder, "platform_has_fast_int8") and not builder.platform_has_fast_int8:
            raise RuntimeError("Platform does not support fast INT8.")
        if not calibration_data_dir:
            raise ValueError("INT8 precision requires calibration_data_dir "
                              "(a folder of representative sample frames).")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _build_int8_calibrator(calibration_data_dir)

    elif precision != "fp32":
        raise ValueError(f"Unknown precision '{precision}'. Use fp32, fp16, or int8.")

    log.info("Building TensorRT engine (precision=%s). This can take several minutes "
              "the first time — TensorRT is profiling kernel implementations for your "
              "specific GPU.", precision)
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed — see TensorRT log output above.")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    log.info("Engine written to %s", engine_path)
    return engine_path


def _build_int8_calibrator(calibration_data_dir: str):
    """
    Minimal INT8 entropy calibrator stub.

    A real implementation loads representative frames, feeds them through
    the network to build an activation histogram, and caches the resulting
    calibration table so future builds are instant. Left as a documented
    stub since Week 1 only targets fp16 per the project spec — wire this up
    if/when you pursue INT8 in a later optimization pass.
    """
    raise NotImplementedError(
        "INT8 calibration is not required for the Week 1 milestone (fp16 is the "
        "target precision). Implement an trt.IInt8EntropyCalibrator2 subclass here "
        "if you extend the project to INT8."
    )


def main():
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from ONNX")
    parser.add_argument("--onnx", default=MODEL.onnx_path)
    parser.add_argument("--engine", default=MODEL.engine_path)
    parser.add_argument("--precision", default=MODEL.precision, choices=["fp32", "fp16", "int8"])
    parser.add_argument("--workspace-mb", type=int, default=4096)
    args = parser.parse_args()

    build_engine(args.onnx, args.engine, args.precision, args.workspace_mb)


if __name__ == "__main__":
    main()