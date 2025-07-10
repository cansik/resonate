from contextlib import contextmanager
from time import perf_counter
from typing import Any, Dict, List


class ResonateTiming:
    """
    Utility class to measure durations of code segments.
    Supports manual start/stop timing and context manager pattern.
    """

    def __init__(self) -> None:
        self._start_times: Dict[str, float] = {}
        self.timings: Dict[str, List[float]] = {}

    def start(self, key: str) -> None:
        """
        Start a timer for the given key.
        """
        self._start_times[key] = perf_counter()

    def stop(self, key: str) -> float:
        """
        Stop the timer for the given key, record the elapsed time, and return it.
        """
        if key not in self._start_times:
            raise KeyError(f"No active timer for key '{key}'.")
        elapsed = perf_counter() - self._start_times.pop(key)
        self.timings.setdefault(key, []).append(elapsed)
        return elapsed

    @contextmanager
    def time_block(self, key: str):
        """
        Context manager to time a block of code.
        Usage:
            with rt.time_block("my_step"):
                # code to measure
        """
        self.start(key)
        try:
            yield
        finally:
            self.stop(key)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a dict of recorded timings.
        Each key maps to a list of elapsed times.
        """
        # Return a copy to prevent external modifications
        return {k: v.copy() for k, v in self.timings.items()}
