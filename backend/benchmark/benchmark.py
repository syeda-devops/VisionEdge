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
def sample_function():
    total = 0
    for i in range(1000000):
        total += i
    return total


if __name__ == "__main__":
    benchmark = Benchmark()

    result, elapsed, fps = benchmark.run(sample_function)

    print(f"Result: {result}")
    print(f"Elapsed: {elapsed:.6f} seconds")
    print(f"FPS: {fps:.2f}")