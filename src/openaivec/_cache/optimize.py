import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

__all__ = []


@dataclass(frozen=True)
class PerformanceMetric:
    duration: float
    batch_size: int
    executed_at: datetime
    exception: BaseException | None = None


@dataclass
class BatchSizeSuggester:
    current_batch_size: int = 10
    min_batch_size: int = 10
    min_duration: float = 30.0
    max_duration: float = 60.0
    step_ratio: float = 0.2
    sample_size: int = 4
    _history: list[PerformanceMetric] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _batch_size_changed_at: datetime | None = field(default=None, init=False)

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

    @contextmanager
    def record(self, batch_size: int):
        start_time = time.perf_counter()
        executed_at = datetime.now(timezone.utc)
        caught_exception: BaseException | None = None
        try:
            yield
        except BaseException as e:
            caught_exception = e
            raise
        finally:
            duration = time.perf_counter() - start_time
            with self._lock:
                self._history.append(
                    PerformanceMetric(
                        duration=duration,
                        batch_size=batch_size,
                        executed_at=executed_at,
                        exception=caught_exception,
                    )
                )

    @property
    def samples(self) -> list[PerformanceMetric]:
        with self._lock:
            selected: list[PerformanceMetric] = []
            for metric in reversed(self._history):
                if metric.exception is not None:
                    continue
                if self._batch_size_changed_at and metric.executed_at < self._batch_size_changed_at:
                    continue
                selected.append(metric)
                if len(selected) >= self.sample_size:
                    break
            return list(reversed(selected))

    def clear_history(self):
        with self._lock:
            self._history.clear()

    def suggest_batch_size(self) -> int:
        selected = self.samples

        if len(selected) < self.sample_size:
            with self._lock:
                return self.current_batch_size

        average_duration = sum(m.duration for m in selected) / len(selected)

        with self._lock:
            current_size = self.current_batch_size

            if average_duration < self.min_duration:
                new_batch_size = int(current_size * (1 + self.step_ratio))
            elif average_duration > self.max_duration:
                new_batch_size = int(current_size * (1 - self.step_ratio))
            else:
                new_batch_size = current_size

            new_batch_size = max(new_batch_size, self.min_batch_size)

            if new_batch_size != self.current_batch_size:
                self._batch_size_changed_at = datetime.now(timezone.utc)
                self.current_batch_size = new_batch_size

            return self.current_batch_size
