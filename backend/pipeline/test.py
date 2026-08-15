from backend.pipeline.pytorch_pipeline import PyTorchPipeline

VIDEO_PATH = r"C:\Users\LOKESH\Desktop\VisionEdge\sample_media\sample.mp4"

try:
    pipeline = PyTorchPipeline(VIDEO_PATH)
    print("PyTorchPipeline object created successfully.")
    pipeline.close()
    print("TEST PASSED")
except Exception as e:
    print("TEST FAILED")
    print(e)