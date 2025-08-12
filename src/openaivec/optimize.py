import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PerformanceMetric:
    duration: float
    exception: BaseException | None = None


@dataclass
class BatchSizeSuggester:
    current_batch_size: int = 10
    min_batch_size: int = 10
    min_duration: float = 30.0
    max_duration: float = 60.0
    step_ratio: float = 0.1
    sample_size: int = 10
    _history: List[PerformanceMetric] = field(default_factory=list)

    _MAX_HISTORY_SIZE: int = 1024

    def __post_init__(self) -> None:
        if self.min_batch_size <= 0:
            raise ValueError("min_batch_size must be > 0")
        if self.current_batch_size < self.min_batch_size:
            raise ValueError("current_batch_size must be >= min_batch_size")
        if self.sample_size <= 0:
            raise ValueError("sample_size must be > 0")
        if self.step_ratio <= 0:
            raise ValueError("step_ratio must be > 0")
        if self.min_duration <= 0 or self.max_duration <= 0:
            raise ValueError("min_duration and max_duration must be > 0")
        if self.min_duration >= self.max_duration:
            raise ValueError("min_duration must be < max_duration")
        if self._MAX_HISTORY_SIZE <= 0:
            raise ValueError("_MAX_HISTORY_SIZE must be > 0")

    @contextmanager
    def record(self):
        start_time = time.perf_counter()
        caught_exception: BaseException | None = None
        try:
            yield
        except BaseException as e:
            caught_exception = e
            raise
        finally:
            duration = time.perf_counter() - start_time
            self._history.append(PerformanceMetric(duration, exception=caught_exception))
            self._history = self._history[-self._MAX_HISTORY_SIZE :]

    @property
    def samples(self) -> List[PerformanceMetric]:
        selected: List[PerformanceMetric] = []
        for metric in reversed(self._history):
            if metric.exception is None:
                selected.append(metric)
                if len(selected) >= self.sample_size:
                    break
        return list(reversed(selected))

    def clear_history(self):
        self._history.clear()

    def suggest_batch_size(self) -> int:
        samples = self.samples
        if len(samples) < self.sample_size:
            return self.current_batch_size

        average_duration = sum(metric.duration for metric in samples) / len(samples)

        if average_duration < self.min_duration:
            new_batch_size = int(self.current_batch_size * (1 + self.step_ratio))

        elif average_duration > self.max_duration:
            new_batch_size = int(self.current_batch_size * (1 - self.step_ratio))

        else:
            new_batch_size = self.current_batch_size

        new_batch_size = max(new_batch_size, self.min_batch_size)
        if new_batch_size != self.current_batch_size:
            self.clear_history()

        self.current_batch_size = new_batch_size

        return self.current_batch_size
