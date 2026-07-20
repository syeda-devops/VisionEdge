from backend.core.timer import Timer
from backend.core.logger import logger


class Benchmark:
    def __init__(self):
        self.timer = Timer()
        self.results = []

    def measure(self, func, *args, **kwargs):
        self.timer.start()

        result = func(*args, **kwargs)

        elapsed = self.timer.stop()

        self.results.append(elapsed)

        return result, elapsed

    def calculate_fps(self, elapsed):
        if elapsed <= 0:
            return 0

        return 1 / elapsed

    def statistics(self):
        """
        Return benchmark statistics.
        """
        if not self.results:
            return None

        average = sum(self.results) / len(self.results)
        minimum = min(self.results)
        maximum = max(self.results)
        fps = self.calculate_fps(average)

        return {
            "average": average,
            "minimum": minimum,
            "maximum": maximum,
            "fps": fps,
        }

    def report(self):
        stats = self.statistics()

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