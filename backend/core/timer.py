# core/timer.py
import time
from contextlib import contextmanager

@contextmanager
def timed(label: str, logger=None):
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    msg = f"{label}: {elapsed:.2f}ms"
    logger.info(msg) if logger else print(msg)
