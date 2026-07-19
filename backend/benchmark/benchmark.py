from backend.core.timer import Timer
from backend.core.logger import logger



class Benchmark:
    def __init__(self):
        self.timer = Timer()
    def run(self, func, *args, **kwargs):
        self.timer.start()

        result = func(*args, **kwargs)

        elapsed = self.timer.stop()

        fps = 1 / elapsed if elapsed > 0 else 0

        logger.info(f"Execution Time: {elapsed:.6f} seconds")
        logger.info(f"FPS: {fps:.2f}")

        return result, elapsed, fps