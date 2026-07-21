import tensorrt as trt


class TensorRTBuilder:
    """
    Builds a TensorRT engine from
    an ONNX model.
    """

    def __init__(self):

        self.logger = trt.Logger(
            trt.Logger.INFO
        )

    def build_engine(
        self,
        onnx_path,
        engine_path
    ):
        """
        Convert an ONNX model into
        a TensorRT engine.
        """

        builder = trt.Builder(
            self.logger
        )

        network = builder.create_network(
            1 << int(
                trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
            )
        )

        parser = trt.OnnxParser(
            network,
            self.logger
        )

        with open(
            onnx_path,
            "rb"
        ) as model:

            if not parser.parse(model.read()):

                for error in range(
                    parser.num_errors
                ):

                    print(
                        parser.get_error(error)
                    )

                return False

        config = builder.create_builder_config()

        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            1 << 30
        )

        serialized_engine = builder.build_serialized_network(
            network,
            config
        )

        if serialized_engine is None:

            print(
                "Failed to build TensorRT engine."
            )

            return False

        with open(
            engine_path,
            "wb"
        ) as engine:

            engine.write(
                serialized_engine
            )

        print(
            "TensorRT Engine Built Successfully!"
        )

        return True