from backend.core.timer import Timer
from backend.core.logger import logger


class Benchmark:
    def __init__(self):
        self.timer = Timer()

    def measure(self, func, *args, **kwargs):
        self.timer.start()

        result = func(*args, **kwargs)

        elapsed = self.timer.stop()

        return result, elapsed

    def calculate_fps(self, elapsed):
        if elapsed <= 0:
            return 0

        return 1 / elapsed

    def report(self, elapsed):
        fps = self.calculate_fps(elapsed)

        logger.info(
            f"Elapsed: {elapsed:.4f}s | FPS: {fps:.2f}"
        )