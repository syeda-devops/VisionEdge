from backend.core.timer import Timer
from backend.core.logger import logger


class Benchmark:
    def __init__(self):
        self.timer = Timer()
        self.results = {
    "PyTorch": [],
    "TensorRT": []
}

    def measure(self, backend, func, *args, **kwargs):
        self.timer.start()

        result = func(*args, **kwargs)

        elapsed = self.timer.stop()

        if backend not in self.results:
            raise ValueError(f"Unknown backend: {backend}")

        self.results[backend].append(elapsed)

        return result, elapsed

    def calculate_fps(self, elapsed):
        if elapsed <= 0:
            return 0

        return 1 / elapsed

    def statistics(self, backend):
        """
        Return benchmark statistics.
        """
        data = self.results.get(backend)

        if not data:
            return None

        average = sum(data) / len(data)
        minimum = min(data)
        maximum = max(data)
        fps = self.calculate_fps(average)

        return {
            "average": average,
            "minimum": minimum,
            "maximum": maximum,
            "fps": fps,
        }

    def report(self, backend):
        stats = self.statistics(backend)

        if stats is None:
            return

        logger.info(
            "Average: %.4fs | "
            "Min: %.4fs | "
            "Max: %.4fs | "
            "FPS: %.2f",
            stats["average"],
            stats["minimum"],
            stats["maximum"],
            stats["fps"],
        )
    def compare(self):
        return {
        "PyTorch": self.statistics("PyTorch"),
        "TensorRT": self.statistics("TensorRT"),
    }