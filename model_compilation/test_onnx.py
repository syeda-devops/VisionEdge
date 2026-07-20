import onnx

def load_onnx_model(model_path):
    """
    Load ONNX model from file.
    """
    return onnx.load(model_path)

def validate_onnx_model(model):
    """
    Validate ONNX model.
    """
    onnx.checker.check_model(model)
    print("ONNX model is valid!")

def main():
    model_path = "yolov8n.onnx"

    model = load_onnx_model(model_path)
    validate_onnx_model(model)

if __name__ == "__main__":
    main()